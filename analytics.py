from collections import Counter
import os, csv, time
import functools
import operator
import utility
import math, random
import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm, kurtosis, kurtosistest, binned_statistic
import statistics
import matplotlib.pyplot as plt
import statsmodels as sm
import preprocess


data_folder = 'dev_data_selected'
patent_data = 'csv_data_compiled/1990-2021'
country_codes_path = 'references/country_codes_full.csv'    # grabbed from wikiepdia and UN statistics
regression_dataset_folder = 'regression_datasets'

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
    years = [ str(i) for i in reversed(range(start, end+1)) ]
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
    file_handles = []
    
    for f in os.listdir(data_folder):
        filename = os.fsdecode(f)
        if filename.endswith('.csv'):
            filepath = os.path.join(data_folder,filename)
            filestem = filename.rsplit('.', 1)[0].strip()
            file_handles.append( (open(filepath), filestem) )
    
    codes_not_found = set()
    for data_file in file_handles:
        f_handler, f_name = data_file
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
        
    multi_df.to_csv(f'{regression_dataset_folder}/MULTI_DATA_{start}-{end}.csv')
    
    for f in file_handles:
        f[0].close()
    
    print(f'DONE. Compiled MultiIndexed data ({start}-{end}).\n')
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


def build_country_dataframe():
    large_dataset_path = f'{regression_dataset_folder}/LARGE_DATA.csv'
    if not os.path.exists(large_dataset_path):
        build_LARGE_singleIndex_dataframe()
    
    with open(file=large_dataset_path, mode='rt') as f:
        large_df = pd.read_csv(f, index_col='country_code')
        # print(large_df.columns.values)
        print(large_df)
        

def analyzeDF(multi_df):
    # usa_df = multi_df.loc[('USA', '2020')]
    # usa_df = multi_df.loc[('USA')]
    # print(usa_df.head)
    multi_df.dropna(inplace=True)
    print(f'Remaining data: {multi_df.shape}')
    
    from statsmodels.formula.api import ols
    dependent_variable = 'GDP (current US$)'
    predictors = list(filter(lambda e: e != dependent_variable, 
                             list(multi_df.columns)))
    
    # build separate dataframe with compatible naming for OLS fit
    ols_df = pd.DataFrame()
    ols_df.insert(0, 'dependent', multi_df[dependent_variable])
    predictor_names = []
    for i, p in enumerate(predictors):
        new_name = f'predictor_{i}'
        predictor_names.append(new_name)
        ols_df[new_name] = multi_df[p]
    print(ols_df)
    formula = 'dependent ~ ' + ' + '.join(predictor_names)
    print(formula)
    
    model = sm.formula.api.ols(formula, ols_df).fit()
    print(model.get_robustcov_results().summary())
    # print(model.summary())
    
    
    # by_country_df = multi_df.groupby(by=["country_code"]).sum()
    # print(by_country_df)
    
        

def main():
    # checkForNormality()
    # playground()
    # multipleRegression()
    # build_LARGE_singleIndex_dataframe()
    df = build_LARGE_multiIndex_dataset(start=2010, end=2020)
    analyzeDF(df)
    
    # build_country_dataframe()
    # test()
    


if __name__ == '__main__':
    main()