from collections import Counter
import os, csv, time, pickle
import functools
import operator

from numpy.core.numeric import False_
from numpy.lib.npyio import save
import utility
import math, random
import pandas as pd
import numpy as np
import scipy, scipy.stats
from scipy.stats import norm, kurtosis, kurtosistest, binned_statistic
import statistics
import matplotlib.pyplot as plt
import statsmodels as sm
import statsmodels.formula.api as smf
import preprocess
import rich


data_folder = 'dev_data_selected'
data_folder = 'dev_data_all'
patent_data = 'csv_data_compiled/1990-2021'
country_codes_path = 'references/country_codes_full.csv'    # grabbed from wikiepdia and UN statistics
regression_dataset_folder = 'regression_datasets'
optimized_dataset_folder = 'optimized_country_data'

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
    

def checkForNormality():
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
    
    num_bins = 50
    frequency, bins = np.histogram(data, bins=num_bins, range=None)
    print(f'{bins=}, {frequency=}')
    plt.hist(data, bins=num_bins)
    plt.show()
    

def playground():
    # check for skewness
    # data = norm.rvs(size=30, random_state=3)
    # print(type(data))
    
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
    s = time.perf_counter()
    for _ in range(df.shape[1]-1):      # since one cannot be evaluated anymore
        rich.print(f'[i]Num Indicators:[/i] {df.shape[1]}\n')
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
        
    
    rich.print(f'Time: {time.perf_counter()-s}')
    
    if savepath is not None:
        with open(file=savepath, mode='wb') as f:
            pickle.dump(df, f)
    
    return df
    

def analyzeDF(multi_df_path):
    MUST_HAVE_INDICATOR = 'total number of patents'
    start_year = 2000
    end_year = 2020
    country = 'GBR'
    
    
    # 1. dataframe selection
    
    multi_df = pd.read_csv(multi_df_path)
    multi_df.set_index(['country_code', 'year'], inplace=True)
    
    # multi_df.fillna(0, inplace=True)      # only useful for testing (cannot be justified statistically)
    # multi_df.dropna(inplace=True)
    
    # 1.1 Choose country
    multi_df = multi_df.loc[(country)]
    
    # 1.2 Choose year range
    multi_df = multi_df.loc[ end_year: start_year, :]

    # 1.3 Check if patents column has any NaN (and since we're definitely using patents, we must have complete data)
    if not all(multi_df[MUST_HAVE_INDICATOR].values): 
        print(f'Do not have complete patent data for {country} ({start_year}-{end_year})')
    
    
    # 1.4 Drop any columns with missing data
    multi_df.dropna(subset=[MUST_HAVE_INDICATOR], inplace=True)
    multi_df.dropna(axis=1, inplace=True)
    # print(multi_df.columns.values)
    

    # 1.5 choose indicators using those with lowest VIFs
    optimized_data_path = f'{optimized_dataset_folder}/{country} ({start_year}-{end_year}).pickle'
    # X = findMostIndependent(multi_df.drop(MUST_HAVE_INDICATOR, axis=1),
    #                         savepath=optimized_data_path)      # so we can check VIF without taking into account the must-have

    with open(file=optimized_data_path, mode='rb') as f:
        X = pickle.load(f)
    
    # choose top 5 (or some other) and check their VIFs
    best_indicators = list(checkVIFs(X).drop('const')[0:8].index)
    X = multi_df[[MUST_HAVE_INDICATOR] + best_indicators]
    print(checkVIFs(X))
    X.to_csv('temporary.csv')
    
    return

    # NOTE: Try all combinations of 5 or some indicators to see which one gets best VIFs
    
    
    
    # temporary bestIndicators (which is just my own chosen list)
    bestIndicators = ['Population ages 50-54, female (% of female population)', 'GDP per person employed (constant 2017 PPP $)','Travel services (% of commercial service exports)', MUST_HAVE_INDICATOR]
    dependentVariable = 'Services, value added (annual % growth)'
    
    multi_df = multi_df.loc[ :, bestIndicators+[dependentVariable]]
    with open('temporary.csv', mode='wt') as f:
        multi_df.to_csv(f)
    
    # X is independent data set
    X = multi_df[bestIndicators]
    VIFs = checkVIFs(X, sorted_=False)
    print(f'\n --- VIF of Indicators ---')
    print(VIFs.reset_index())       # genius move, just reset index to force Series into Dataframe so we have both enumeration and name indexing

    # alternatively, you can implement VIFs this way, but we're sticking with the library in this case
    # vifs = pd.Series(np.linalg.inv(X.corr().to_numpy()).diagonal(), 
    #              index=X.columns, 
    #              name='VIF')
        

    print(f'Remaining data: {multi_df.shape}')
    
    
    # 2. Model using OLS
    model = buildModel(multi_df, dependent_variable=dependentVariable)
    
    
    # 3. Display results
    print(model.get_robustcov_results().summary())
    print(model.params)
    print(model.rsquared)
    
    
    # by_country_df = multi_df.groupby(by=["country_code"]).sum()
    # print(by_country_df)
    
    
    
    
    return model


def main():
    # checkForNormality()
    # playground()
    # multipleRegression()
    # build_LARGE_singleIndex_dataframe()
    # df = build_LARGE_multiIndex_dataset(start=2000, end=2020)
    
    analyzeDF(f'{regression_dataset_folder}/MultiIndexed_(1960-2020)_(N=1444).csv')

    # test dataset
    # data = pd.read_csv('~/Downloads/BMI.csv')
    # X = data[['Height', 'Weight']]
    # print(np.corrcoef(X['Height'].values, X['Weight'].values)[0, 1])
    # print(checkVIFs(X))
    
    


if __name__ == '__main__':
    main()
