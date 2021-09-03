from collections import Counter
import os, sys, csv, time, pickle
import functools
import operator
from tokenize import group

from numpy.core.numeric import False_
from numpy.lib.npyio import save
from pandas.core.indexes import multi
from utility import interface_decorator
import math, random
import pandas as pd
import numpy as np
import scipy, scipy.stats
from scipy.stats import norm, kurtosis, kurtosistest, binned_statistic
import statistics
import matplotlib.pyplot as plt
import statsmodels as sm
import statsmodels.formula.api as smf
from statsmodels.multivariate.pca import PCA
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import preprocess
import rich


data_folder = 'dev_data_selected'
data_folder = 'dev_data_all'
patent_data = 'csv_data_compiled/1990-2021'
country_codes_path = 'references/country_codes_full.csv'    # grabbed from wikiepdia and UN statistics
regression_dataset_folder = 'regression_datasets'
optimized_dataset_folder = 'optimized_country_data'



def main():
    """ This is the user interface, the main driver of all features
            Step 1 builds the dataframe from raw data
            Step 2 reads in the dataframe if already exists
            Each step after Step 2 can be ran independently
            Try it out!
    """
    
    # -- Step 1: Build the LARGE singleIndex dataframe from raw JSON Data --
    # df = build_LARGE_multiIndex_dataset(start=2000, end=2020)     # if a different data range is needed
    
    
    # -- Step 2: read in the the build dataframe --
    multi_df_path = f'{regression_dataset_folder}/MultiIndexed_(1960-2020)_(N=1444).csv'
    multi_df = pd.read_csv(multi_df_path, index_col=['country_code', 'year'])
    # multi_df = multi_df.sort_values(by=['total number of patents'], ascending=False)


    # -- Step 2.5 quick visualizations to check data spread (if necessary) --
    checkForNormality(num_bins=50)
    # quickPlot(multi_df, country='ARG', indicator='total number of patents')        # quick plot of one indicator of one country
    
    
    # -- Step 3. Generate simple variable regressions for ALL countries by default --
    simple_plot(multi_df)   # outputs graphs to simple_plots subdirectory
    
    
    # -- Step 4. Analyze using OLS --
    # choose your countries by their unique alpha-3 code
    runOLS(multi_df, country=['CHN', 'USA', 'JPN', 'KOR', 'DEU', 'GBR', 'JOR'], num_sets=5) 
    
    
    # -- Step 5. Analyze using PCA --
    run_PCA_scikit(multi_df, groupby='country_code')     # currently only supports grouping data by country before running PCA



def flattenAndClean(df):
    flat = df.to_numpy().flatten()
    clean = [entry for entry in flat 
             if type(entry) == float 
             and not math.isnan(entry) ]
    return clean
    
    # row = df.iloc[0, 1:].values
    # clean_row = np.array( [e for e in row if not math.isnan(e)] )
    # print(kurtosis(clean_row, fisher=True))
    # print( scipy.stats.kurtosistest(clean_row, axis=0, nan_policy='propagate', alternative='two-sided'))

def transcribe_patent_data_for_theWorldBank():
    # 1. compile patent data for matching year range
    data_as_dict = preprocess.compile_json_data_by_year(1960, 2020, lang='EN', suppressPrint=True)
    
    # 2. get all country for which we have patent data
    patent_df = pd.DataFrame.from_dict(data_as_dict, orient='index')
    
    # translate from alpha-2 to alpha-3 (ISO 3166)
    codes_df = pd.read_csv(country_codes_path, index_col='name')
    alpha_2_to_3 = { two:three for two, three in zip( codes_df['alpha-2'].values, codes_df['alpha-3'].values ) }
    
    new_indicies = {}
    for name in patent_df.index.values:
        alpha_2_code = name.split('(')[-1][0:2]
        if alpha_2_code not in alpha_2_to_3:
            patent_df.drop(name, inplace=True)        # meaning code is not a country
            # print(f'dropped {name}')
        else:
            new_indicies[name] = alpha_2_to_3[alpha_2_code]
    
    # 2.3 Rename dataframe indicies, as well as dataframe variable name
    patent_df = patent_df.rename(index=new_indicies)
    
    return patent_df


def build_LARGE_singleIndex_dataframe(track_extra_data=False):
    # 1. Get transcribed patent dataset
    LARGE_df = transcribe_patent_data_for_theWorldBank()
    
    # make each entry into a dictionary of its own
    for index, row in LARGE_df.iterrows():
        for col in LARGE_df.columns:
            row[col] = {'patent count': row[col]}
    
    
    # 3. Open all indicator data files (csv format)
    # array of tuples (filehandler, filename)
    file_handles = []
    
    for f in os.listdir(data_folder):
        filename = os.fsdecode(f)
        if filename.endswith('.csv'):
            filepath = os.path.join(data_folder,filename)
            filestem = filename.rsplit('.', 1)[0].strip()
            file_handles.append( (open(filepath), filestem) )

    
    # 4. Only extend data for known countries (those with patent data)
    known_countries = set(LARGE_df.index)

    # go through each indicator file
    codes_not_found = set()
    for data_file in file_handles:
        f_handler, f_name = data_file
        indicator_df = pd.read_csv(f_handler, index_col='economy')  # 'economy' is the country code alpha-3
    
        # first country in first file
        for country_code, data in indicator_df.iterrows():
            if country_code not in known_countries:
                codes_not_found.add(country_code)
                continue
            
            country_data = LARGE_df.loc[country_code, :].to_numpy()
            for year_data, indicator_data in zip(country_data, data.values[1:]):     # removing the first data (which is just country name in english)
                year_data[f_name] = indicator_data
    
    LARGE_df.index.name = 'country_code'
    LARGE_df.to_csv(f'{regression_dataset_folder}/LARGE_DATA.csv')
    
    if track_extra_data:
        print(f'{codes_not_found=}')    
    
    for f in file_handles:
        f[0].close()
    

def build_LARGE_multiIndex_dataset(start=1960, end=2020, track_extra_data=False):
    patent_df = transcribe_patent_data_for_theWorldBank()
    
    # 1. Build MultiIndexed DataFrame
    known_countries = sorted(patent_df.index)
    # note: year is indexed in INTEGER, not string
    years = [ int(i) for i in reversed(range(start, end+1)) ]       
    index = pd.MultiIndex.from_product([ known_countries, years ])
    
    multi_df = pd.DataFrame(index=index)
    multi_df.index.names = ['country_code', 'year']
    
    # 2. Flatten and insert patent data into first data column
    patent_df.sort_index(0, inplace=True)
    patent_df = patent_df.loc[:, end:start]
    patent_flat = patent_df.to_numpy().flatten()
    
    multi_df.insert(0, 'total number of patents', patent_flat, allow_duplicates=False)
    multi_df.drop('TWN', inplace=True)      # no developement data for "TWN"
    known_countries.remove('TWN')
    
    # '''
    # 3. Loop through all indicator dataset, flatten, and insert
    # array of tuples (filehandler, filename)
    files = []
    
    for f in os.listdir(data_folder):
        filename = os.fsdecode(f)
        if filename.endswith('.csv'):
            filepath = os.path.join(data_folder,filename)
            filestem = filename.rsplit('.', 1)[0].strip()
            files.append( (filepath, filestem) )
    
    codes_not_found = set()
    for i, data_file in enumerate(files):
        f_path, f_name = data_file
        print(f'Handling {i+1}/{len(files)}: {f_name}')
        
        with open(file=f_path, mode='rt') as f_handler:
            indicator_df = pd.read_csv(f_handler, index_col='economy')  # 'economy' is the country code alpha-3
            
        indicator_df.sort_index(0, inplace=True)
        indicator_df = indicator_df.loc[:, f'YR{end}':f'YR{start}']

        flat_data = []
        # first country in first file
        for country_code, data in indicator_df.iterrows():
            if country_code not in known_countries:
                codes_not_found.add(country_code)
                continue
            
            flat_data.extend(data.to_numpy())
        
        last_col_index = multi_df.shape[1]
        multi_df.insert(last_col_index, f_name, flat_data, allow_duplicates=True)
    
    if track_extra_data:
        print(f'{codes_not_found=}')
    
    outfile_name = f'MultiIndexed_({start}-{end})_(N={len(files)+1})'
    multi_df.to_csv(f'{regression_dataset_folder}/{outfile_name}.csv')
    
    print(f'\nDONE. Compiled {outfile_name}.\n')
    return multi_df
    
    

def multipleRegression():
    
    filepath = f'{data_folder}/Mortality rate, neonatal (per 1,000 live births).csv'

    # data processing
    indicator_df = pd.read_csv(filepath, index_col = 'economy')
    patent_df = pd.read_csv(filepath, index_col = 'economy')
    
    # get alpha-2 and alpha-3 codes
    codes_df = pd.read_csv(country_codes_path, index_col='name')
    alpha_2_to_3 = { two:three for two, three in zip( codes_df['alpha-2'].values, codes_df['alpha-3'].values ) }
    
    # organize by country
    country_data = {}
    
    
    # perform ols
    x = [0.2, 0.3, 0.9]
    y = np.arctanh(x)  # arctanh is fisher's z transformation
    print(np.tanh(y))   # and tanh is the inverse of fisher's z transformation
    # model = sm.ols("z ~ x + y", data).fit()
    
@interface_decorator
def checkForNormality(num_bins=50):
    # import test data
    filepath = f'{data_folder}/GDP (current US$).csv'
    filepath = f'{data_folder}/GNI per capita, Atlas method (current US$).csv'
    filepath = f'{data_folder}/Urban population.csv'
    filepath = f'{data_folder}/Adjusted net savings, excluding particulate emission damage (current US$).csv'
    filepath = f'{data_folder}/Literacy rate, adult total (% of people ages 15 and above).csv'
    filepath = f'{data_folder}/Mortality rate, neonatal (per 1,000 live births).csv'
    
    # with open(filepath, newline='') as csvfile:
    #     reader = csv.reader(csvfile, delimiter=',', quotechar='\"')
    #     for row in reader:
    #         for entry in row:
    #             print(entry)
    
    df = pd.read_csv(filepath, index_col='economy')
    years = [int(year[2:]) for year in df.columns.values[1:]]
    
    data = flattenAndClean(df)
    data, alpha = scipy.stats.boxcox(data)
    
    print( scipy.stats.kurtosistest(data, axis=0, nan_policy='propagate', alternative='two-sided') )
    print( scipy.stats.bartlett(data, years))       # test for equal variance between random variables
    # print( scipy.stats.skewtest(data, axis=0, nan_policy='propagate', alternative='two-sided') )
    print( f"Skewness: {scipy.stats.skew(data, axis=0, nan_policy='propagate')}")
    print( f'{statistics.stdev(data)=}\n' )
    
    # plot distribution
    # binned = scipy.stats.binned_statistic(df_clean, df_clean, statistic='mean', bins=10, range=None)
    # print(type(binned[0]))
    
    frequency, bins = np.histogram(data, bins=num_bins, range=None)
    print(f'{bins=}, {frequency=}')
    plt.hist(data, bins=num_bins)
    plt.show()

@interface_decorator
def quickPlot(multi_df, country, indicator='total number of patents'):
    y = multi_df.loc[[country], [indicator]]
    y = y.dropna().values
    x = [i+1 for i in range(0, len(y))]
    
    plt.plot(x, y)
    plt.title(f'{country} - {indicator}')
    plt.show()


def playground():
    # check for skewness
    data = norm.rvs(size=30, random_state=3)
    print(type(data))
    
    data = np.random.rand(100)
    bin_means = binned_statistic(data, data, bins=5, range=None)[0]
    print(bin_means)
    
    
    data = [ random.normalvariate(mu=100, sigma=5)  for _ in range(1000) ]
    print( f"Skewness: {scipy.stats.skew(data, axis=0, nan_policy='propagate')}")
    print( scipy.stats.skewtest(data, axis=0, nan_policy='propagate', alternative='two-sided'))
    
    # df_clean = df.iloc[:, 1:10].dropna(how='any')
    # df_clean = functools.reduce(operator.iconcat, np.array(df_clean), [])
   
    
    # check for kurtosis (with threshold)
    # print(kurtosis(data, fisher=True))
    # print(kurtosistest(data))


# given a multiindexed dataframe, return a model
def buildModel(multi_df, dependent_variable='GDP (current US$)', recursionDepthMultiplier=5):
        import sys
        sys.setrecursionlimit(sys.getrecursionlimit()*recursionDepthMultiplier)
        
        predictors = list(filter(lambda e: e != dependent_variable, 
                                list(multi_df.columns)))
        
        # build separate dataframe with compatible naming for OLS fit
        ols_df = pd.DataFrame()
        ols_df.insert(0, 'dependent', multi_df[dependent_variable])
        predictor_names = {}
        for i, old_name in enumerate(predictors):
            new_name = f'predictor_{i}'
            predictor_names[new_name] = old_name        # so we can map them back
            ols_df[new_name] = multi_df[old_name]
        
        formula = 'dependent ~ ' + ' + '.join(list(predictor_names.keys()))
        model = sm.formula.api.ols(formula, ols_df).fit()           # requires separate import of sm.formula (prob just init file construction)
        
        
        sys.setrecursionlimit(1000)     # resetting recursion limit
        
        return model


@interface_decorator
def runOLS(multi_df, start_year = 2000, end_year = 2020, country = ['GBR'], 
              num_best_indicators = 6, num_sets = 6, MUST_HAVE_INDICATOR = 'total number of patents',
              results_folder = 'results'):
    
    # 0. construct naming
    country_names = ' + '.join(country)
    
    # 1. Extract country + year data from large dataset
    rich.print("[green] >>> 1. Extracing data by Country and Year ... [/green]\n")
    multi_df = country_year_extractor(multi_df, MUST_HAVE_INDICATOR, start_year, end_year, country)

    # 2. Filter indicators which are some linear shift or scaling of one another
    rich.print("[green] >>> 2. Removing redundant indicators via continuous elimination ... [/green]\n")
    clean_X = filter_redundancy(multi_df, MUST_HAVE_INDICATOR, start_year, end_year, country_names)
    
    # 3. choose top 'num_best' (or some other) indicator groups with lowest VIFs
    rich.print("[green] >>> 3. Choosing Best Sets of Indicators via stochastic process ... [/green]\n")
    best_indicators = findBestSets(multi_df[list(clean_X.columns) + [MUST_HAVE_INDICATOR]], MUST_HAVE=[MUST_HAVE_INDICATOR], 
                                   group_size=num_best_indicators, top_choices=num_sets, method='avg', random_sampling=1000)    
    
    # 4. Using those top groups of indicators, run regression for each group
    
    rich.print("[green] >>> 4. Running OLS on best indicator sets ... [/green]\n")
    built_models = []
    output_file = f'{results_folder}/{country_names} ({start_year}-{end_year}).txt'
    with open(file=output_file, mode='wt') as results:
        results.write(f">>> Country: {country_names}, Years: {start_year}-{end_year}\n\n")
        
        i = 1
        for vif, indicators in best_indicators:
            try:
                model = performOLS(multi_df, X=indicators, Y='Services, value added (annual % growth)')
            except Exception as e:
                rich.print('[#e88718]OLS internal error. Dropping a set.[/#e88718]')
                continue
            
            built_models.append(model)
            full_vifs = checkVIFs(multi_df[indicators]).drop(labels=['const'])
            
            # record results
            results.write(f'({i})\n')
            results.write(f'-> Indicators: \n{indicators}\n\n')
            results.write(f'-> Full VIF: \n{ full_vifs }\n\n')
            results.write(f'-> Summary: \n{model.get_robustcov_results().summary()}\n\n\n\n')
            
            i += 1
    
    rich.print(f"[#1da5ee] >>> DONE. Results recorded ---> {output_file} [/#1da5ee]\n")
    
    # 5. Display results
    # print(model.get_robustcov_results().summary())
    # print(model.params)
    # print(f'r2 = {model.get_robustcov_results().rsquared_adj}')
    
    return


def performOLS(multi_df, X, Y):
    # X = ['Population ages 50-54, female (% of female population)', 'GDP per person employed (constant 2017 PPP $)','Travel services (% of commercial service exports)', MUST_HAVE_INDICATOR]
    # Y = 'Services, value added (annual % growth)'
    
    model_df = multi_df.loc[ :, X + [Y]]
    with open('temporary.csv', mode='wt') as f:
        model_df.to_csv(f)


    # alternatively, you can implement VIFs this way, but we're sticking with the library in this case
    # vifs = pd.Series(np.linalg.inv(X.corr().to_numpy()).diagonal(), 
    #              index=X.columns, 
    #              name='VIF')


    # 2. Model using OLS
    model = buildModel(model_df, dependent_variable=Y)
    return model



def filter_redundancy(multi_df, MUST_HAVE_INDICATOR, start_year, end_year, country_names):
    optimized_data_path = f'{optimized_dataset_folder}/{country_names} ({start_year}-{end_year}).pickle'
    
    if not os.path.exists(optimized_data_path):
        X = findMostIndependent(multi_df.drop(MUST_HAVE_INDICATOR, axis=1),
                                savepath=optimized_data_path)      # so we can check VIF without taking into account the must-have
    else:
        with open(file=optimized_data_path, mode='rb') as f:
            X = pickle.load(f)
    return X


def country_year_extractor(multi_df, MUST_HAVE_INDICATOR, start_year, end_year, country):
    # 1. dataframe selection
    if not isinstance(multi_df.index, pd.MultiIndex):
        multi_df.set_index(['country_code', 'year'], inplace=True)
    # multi_df.fillna(0, inplace=True)      # only useful for testing (cannot be justified statistically)
    # multi_df.dropna(inplace=True)     # performed later
    
    
    # 1.1 Choose country
    multi_df = multi_df.loc[country]
    multi_df = multi_df.groupby(by=["year"]).mean()
    
    # 1.2 Choose year range
    multi_df = multi_df.loc[ start_year: end_year , :]

    # 1.3 Check if patents column has any NaN (and since we're definitely using patents, we must have complete data)
    if not all(multi_df[MUST_HAVE_INDICATOR].values): 
        print(f'Do not have complete patent data for {country} ({start_year}-{end_year})')
    
    # 1.4 Drop any columns with missing data
    multi_df.dropna(subset=[MUST_HAVE_INDICATOR], inplace=True)
    multi_df.dropna(axis=1, inplace=True)

    # print(multi_df.columns.values)
    return multi_df



def checkVIFs(df, sorted_=True):
    """Takes pandas dataframe and returns pandas series with VIF for each column

    Args:
        df: input pandas dataframe
        sorted_ (bool, optional): sorts by VIFs in ascending order. Defaults to True.

    Returns:
        VIFs: pandas Series
    """
    
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    if 'const' not in df.columns.values:
        df = df.assign(const=1)   # statsmodels VIF function expects a column of constants (not sure why, but okay)
        
    VIFs = pd.Series([ variance_inflation_factor(df.values, i)
                      for i in range(df.shape[1])], name='VIF',
                     index=pd.Index(df.columns,
                                    name='X'))
    if sorted_:
        VIFs.sort_values(ascending=True, inplace=True)
    return VIFs


def findMostIndependent(df, savepath=None):
    """Takes DF, and removes columns (indicators) that are scaled or shifted version of others
    Does so by assuming those columns have VIF = inf

    Args:
        df (pandas dataframe): contains the data in rows, and attributes/indicators in columns
        savepath (str, optional): Path to save filtered dataframe. Defaults to None.

    Returns:
        df (pandas df): filtered dataframe
    """
    
    s = time.perf_counter()
    for _ in range(df.shape[1]-1):      # since one cannot be evaluated anymore
        sys.stdout.write('\r')
        rich.print(f'[i]Num Indicators:[/i] {df.shape[1]}', end='   ')      # trailing spaces are for removal formatting
        VIFs = checkVIFs(df, sorted_=False)
        
        # check if VIFs returns any NaN, inf, or 0 (which we are considering absolutely impossible)
        VIFs.drop('const', inplace=True)      # also drop the artifical 'const' column
        NaNs = np.argwhere( np.isnan(VIFs.values) )
        Infs = np.argwhere( np.isinf(VIFs.values) )
        Zeros = np.argwhere( VIFs.values == 0 )
        
        if len(NaNs) > 0:
            df = df.drop(VIFs.index[NaNs[0][0]], axis=1)
        elif len(Infs) > 0:
            df = df.drop(VIFs.index[Infs[0][0]], axis=1)
        elif len(Zeros) > 0:
            df = df.drop(VIFs.index[Zeros[0][0]], axis=1)
        else:
            break
        
    
    rich.print(f'\nTime: {time.perf_counter()-s}')
    
    if savepath is not None:
        with open(file=savepath, mode='wb') as f:
            pickle.dump(df, f)
    
    return df


def findBestSets(df, group_size, top_choices=5, method='avg', random_sampling=None, MUST_HAVE=[], sorted_=True):
    """Finds best sets of indicators (i.e. those with lowest sum VIFs)

    Args:
        df (pandas dataframe): self explanatory
        group_size ([type]): the number of indicators you want in your predictor group
        top_choices (int, optional): number of top scoring groups to be returned. Defaults to 5.
        method (str, optional): scoring method for each group ('avg', 'max', 'min')    
        random_sampling (int, optional): defines a subset of all possible sample to be randomly chosen to be evaluated
            effectively decreasing number of possibilities to go through
        MUST_HAVE (list): defines indicators that must be included 
            (if set, the returned best set is guarenteed to return a set that includes this indicator)
        sorted_ (bool, optional): output is sorted. Defaults to True.

    Returns:
        [type]: [description]
    """
    
    from itertools import combinations
    
    if df.shape[1] < group_size + len(MUST_HAVE):
        return []
    if df.shape[1] == group_size + len(MUST_HAVE):
        return df.columns
    
    s = time.perf_counter()
    
    scores = []
    samples = list(combinations(df.columns.drop(MUST_HAVE), group_size))

    if random_sampling != None and random_sampling < len(samples):
        samples = random.sample(samples, random_sampling)
    
    for i, group in enumerate(samples):
        sys.stdout.write('\r')
        rich.print(f'running {i+1}/{len(samples)}', end=' ')
        
        group = list(group) + MUST_HAVE
        vifs = checkVIFs(df[group])
        
        if method == 'avg':
            score = sum(vifs.drop('const'))/len(group)     # simple scoring with avg
        elif method == 'max':
            score = max(vifs.drop('const'))      # simple scoring with max
        elif method == 'min':
            score = min(vifs.drop('const'))      # simple scoring with min
        
        scores.append( (score, group) )
    
    print(f"\nDone: {time.perf_counter()-s:2f}s\n")
    scores.sort(key=lambda e: e[0])
    return scores[:top_choices]
        
    
@interface_decorator
def simple_plot(multi_df, country_codes=None):
    multi_df = multi_df.reset_index(level=[0, 1])       # flatten row multiindexing

    if country_codes == None:
        country_codes = multi_df['country_code'].unique()
            
    # get country names
    codes_df = pd.read_csv(country_codes_path)
    alpha3_to_name = {code:name for code, name in zip(codes_df['alpha-3'], codes_df['name'])}
    
    
    for i, country_code in enumerate(country_codes):
        print(f'{i+1}/{len(country_codes)}: fitting {country_code} data') 
        
        df = multi_df[multi_df['country_code'] == country_code].loc[:, ['year', 'GDP (constant LCU)', 'total number of patents']]
        # df['ratio'] = df['GDP (constant LCU)'] / df['total number of patents']
        # df = df[df['total number of patents'] > 10]
        df = df.dropna()
        df.sort_values(by='year', ascending=True, inplace=True)
        # print(df.head)
        
        x = 'total number of patents'
        y = 'GDP (constant LCU)'
        x_vals, y_vals = df[x].values, df[y].values
        
        fig, ax = plt.subplots(figsize=[20, 15])
        ax.plot(x_vals, y_vals, markersize=3, marker='o', linestyle='dashed')
        for i, year in enumerate(df['year']):
            ax.annotate(year, (x_vals[i], y_vals[i]))
        
        
        # plot line of best fit
        # model = performOLS(df, X=['total number of patents'], Y='GDP (constant LCU)')
        # print(model.get_robustcov_results().summary())ow
        # print(model.params)
        # print(f'r2 = {model.get_robustcov_results().rsquared_adj}')

        m, b, r_value, p_value, std_err = scipy.stats.linregress(x_vals, y_vals)
        # m, b = np.polyfit(x_vals, y_vals, 1)
        plt.plot(x_vals, m*x_vals + b, linewidth=1.2, label=f'{m:.2f}*{x} + {b:.2f}')
        
        name = alpha3_to_name[country_code]
        plt.title(f'{name}: Total patents to GDP (n={len(df[x])}) (r2 = {r_value**2:.5f}) (p = {p_value:.8f})')
        plt.grid(alpha=.4,linestyle='--')
        plt.yscale('linear')
        plt.legend()
        plt.savefig(f'simple_plots/{country_code}-{name}.png')
        plt.close()
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(f'\nView Simple Plots in: \n{dir_path}/simple_plots\n')
    

def clean_data_for_PCA(multi_df, groupby):
    
    df = multi_df.reset_index(level=[0, 1])       # flatten row multiindexing`
    
    # -- singular country --
    # df = df[ (df['country_code'] == 'CHN') & (df['year'] >= 1990) ]     # choosing country and year
    # df.drop(['country_code'], axis=1, inplace=True)
    # df.set_index('year', inplace=True)

    # -- all countries as one --
    df = df[ df['year'] >= 1980 ]     # choosing country and year
    df = df.groupby('country_code').mean()     # choosing country and year
    
    
    # -- try no grouping, so each entry is Country+Year, e.g. USA2012--
    # df['new_index'] = df['country_code'] + df['year'].map(str)
    # df.set_index('new_index', inplace=True)
    # df.drop(['country_code', 'year'], axis=1, inplace=True)
    
    # remove indicators with NaN or Inf
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any')
    
    # for ind in df.columns:
    #     if ind == 'total number of patents':
    #         continue
    #     if np.isnan(df[ind].values).any():
    #         del df[ind]

            
    # best_indicators = ['total number of patents', 'Urban population', 'Population, total', 'Primary education, pupils', 'Secondary education, general pupils', 'Secondary education, pupils', 'Capture fisheries production (metric tons)', 'Rural population', 'Total fisheries production (metric tons)', 'Adjusted savings: mineral depletion (current US$)', 'Fish species, threatened', 'Mammal species, threatened', 'Fixed broadband subscriptions', 'Fixed telephone subscriptions', 'Bird species, threatened', 'Patent applications, residents', 'Patent applications, nonresidents', 'Secondary education, general pupils (% female)', 'Land area (sq. km)', 'Surface area (sq. km)']
    best_indicators = set(list(df.columns.values) + ['total number of patents'])     # just use all + 'patent count'
    df = df.loc[:, best_indicators]
    print(df.shape)
    return df

@interface_decorator
def run_PCA_scikit(multi_df, groupby='country_code'):
    # first, data cleaning
    df = clean_data_for_PCA(multi_df, groupby=groupby)
    print(df)

    scaled_data = preprocessing.scale(df)   # now mean = 0, and sd = 1 (standard normal distribution)

    pca = PCA(n_components=10)
    pca.fit(scaled_data)        # the calculations
    pca_data = pca.transform(scaled_data)   # projecting points onto new principle components

    
    # Generate Scree Plot
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=2)

    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

    fig, ax = plt.subplots(figsize=[12, 9])
    ax.bar(x = labels, height=per_var)
    for bar in ax.patches:
        ax.annotate(f"{bar.get_height():.2f}%",
                    (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                    size=12, xytext=(0, 8),
                    textcoords='offset points')

    ax.set_ylabel('Percentage of Explained Variance')
    ax.set_ylabel('Principal Component')
    ax.set_title('Scree Plot')
    plt.show()
    plt.close()



    # # Generate PCA Plot with first 2 principle components
    pca_df = pd.DataFrame(pca_data, index=[*df.index.values], columns=labels)
    print(pca_df)
    plt.figure(figsize=[12, 9])
    plt.scatter(pca_df.PC1, pca_df.PC2, c=df['GDP (constant 2010 US$)'], cmap='Spectral')
    plt.title('PCA Graph')
    plt.xlabel(f'PC1 - {per_var[0]}%')
    plt.ylabel(f'PC2 - {per_var[1]}%')

    for sample in pca_df.index:
        plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
        
        

    plt.show()

    # # Check loading scores (on PC1) to see which indicators have most influence on variance
    loading_scores = pd.Series(pca.components_[0], index=df.columns)
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    top_indicators = sorted_loading_scores[0:20].index
    
    df = pd.DataFrame( [top_indicators, loading_scores[top_indicators].values], 
                      index=['Indicator', 'Score'], 
                      columns=[list(range(1, len(top_indicators)+1))] )     # note this dataframe has row and columns flipped
    print(df.T)
    print()



def run_PCA_Statsmodel(multi_df):
    df = clean_data_for_PCA(multi_df)
    
    pc = sm.multivariate.pca.PCA(df, ncomp=10, missing='drop-row', standardize=True)      # could potentially use EM's algorithms to fill in missing values
    
    print(pc.coeff)
    print(pc.projection)
    pc.plot_scree(log_scale=False)
    pc.plot_rsquare()
    # plt.show()

    # print(pc.project(ncomp=2))
    

if __name__ == '__main__':
    main()
