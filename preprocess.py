import math, statistics
from patenthub_api import PatentHub_API
from generic_api import creat_dir
from utility import eprint, next_available_path

import os, json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
    

country_to_code = {}
code_to_country = {}
DATA_PATH = 'json_data'

def init_country_codes():
    global country_codes
    df = pd.read_csv('references/country_codes.csv')
    country_codes = df.set_index('Name').T.to_dict('list')
    for name, code in country_codes.items():
        country_to_code[name] = code[0]
        code_to_country[code[0]] = name
    
    # 2 special cases
    special_cases = {
        'WO': 'WIPO',
        'EP': 'European Union',
        'EA': 'Eurasian Patent Organization',
        'AP': 'African Regional Intellectual Property Organization',
        'OA': 'African Intellectual Property Organization',
        'GC': 'The Patent Office of the Gulf Cooperation Council',
        'YU': 'Yugoslavia',
        'DD': 'German Democratic Republic',
        'SU': 'Soviet Union',
        'CS': 'Czechoslovakia',
        'EM': 'Office for Harmonization in the Internal Market',
        
    }
    for key, value in special_cases.items():
        code_to_country[key] = value
        country_to_code[value] = key
    
        


def obtain_data():
    # for pg in range(1, 10):
        
        # PatentHub_API.s_port(
        #     query='电器',
        #     datascope='CN',
        #     page=pg,
        #     pagesize=maxEntryPerPage,
        #     sorting='!applicationDate'
        # )
    
    
    # can classify using IPC in query???    

    index = PatentHub_API.query_by_year_and_country(
        query='electronic',
        start_year=2010, 
        end_year=2020, 
        countryCode=country_codes['United Kingdom']
    )
    print(index)
    



def json_to_csv():
    df = pd.read_json (r'clean_json/s_port-11-clean.json')
    df.to_csv (r'TRIAL.csv', index = None)
    

# compiles responses from start to end, inclusive
def compile_responses_to_csv(response_name, start=None, end=None, response_directory='response_json'):
    # filepath = os.path.dirname(os.path.realpath(__file__))
    if start == None or end == None:
        raise Exception("Start AND End indicies required")
    if end < start:
        raise Exception("End must be greater than or equal to Start index")


    patents = []
    for i in range(start, end+1):
        try:
            infile = open(f'{response_directory}/{response_name}-{i}.json', mode='rt')
        except Exception:
            eprint('Error: No Such File or Directory')
            break
        
        patents += json.load(infile)['patents']
        
    df = pd.DataFrame(patents)
    
    csv_path = creat_dir('csv_from_api')
    df.to_csv(f'{csv_path}/{response_name}-{start}-{end}.csv', index = True)



def plot_generic(x: list, curves: list, labels: list, filename: str, 
                 title: str = '', x_label:str ='x', y_label:str = 'y', custom_width:int = 20, 
                 overlay=False, x_overlay=None, curve_overlay=None, label_overlay=None) -> None: 
    """plots a list of curves on the same graph

    Args:
        x (list): x axis values
        curves (list): list of y axis values for one or more curves
        labels (list): labels corresponding to each curves
        filename (str): filename to save the output graph under (will be within subdirectory 'graphs')
        title (str, optional): title of graph. Defaults to ''.
        x_label (str, optional): x_axis label. Defaults to 'x'.
        y_label (str, optional): y_axis label. Defaults to 'y'.
        custom_width (str, optional): for custom width (usually used for large timeseries data). Defaults to '20'.
    """
    
    if len(curves) != len(labels): 
        raise Exception("Error: curves and labels count mismatch")
    
    
    plt.figure(figsize=(custom_width, 15), dpi=80, facecolor='w', edgecolor='k', linewidth=1)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    for curve, label in zip(curves, labels):
        plt.plot(x, curve, '-o', markersize=8, alpha=0.6, label=label)
        
    if overlay:
        plt.plot(x_overlay, curve_overlay, '-^', markersize=8, alpha=0.6, label=label_overlay)
        
    
    plt.grid()
    plt.legend()
    graphs_path = creat_dir(folder_name='graphs')      # initialize directory
    plt.savefig(f'{graphs_path}/{filename}.png', format="PNG")
    plt.close()

      

def graph_csv(filename, directory='csv', mode=1):
    with open(f'{directory}/{filename}.csv', mode='rt') as f:
        
        if mode == 1:
            df = pd.read_csv(f, index_col='applicationDate')
            print(df.head())
            # print(df.dtypes)
            # print(df.index)
            # print(df.columns)
            # print(df.values)
            
            # print(df.describe())
            # dates = pd.to_datetime(col, format='%Y/%m/%d')
            # df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
            # print(df.index)
            # print(col)
            
            # print(np.array(df.index))
            
            # print(df.loc[:, ['applicationNumber', 'applicationDate']])
            
            
            col = df.index
            col = reversed(list(col))
            col = ['-'.join(e.split('-')[1:]) for e in col]
            
            # count number of same days
            days = {}
            for date in col:
                days[date] = days.get(date, 0) + 1
            
            
            # # NOTE: experimental: month average overlay plot
            # month_tot = {}
            # for date in col:
            #     m = date.split('-')[0]
            #     month_tot[m] = month_tot.get(m, 0) + 1
                
            # print((month_tot))
            # index_by_month = list(month_tot.keys())
            # print((index_by_month))
            # # return
            
            
            plot_generic (
                x = list(days.keys()),
                curves = [ list(days.values()) ],
                labels = [ 'electronics' ],
                filename = 'patent_by_days',
                title = 'patent group for days',
                custom_width = 30,
            )
    
        elif mode == 2:
            df = pd.read_csv(f, index_col='Country Name')
            china = df.loc['China', :]
            # print(china.head())

            time = list(reversed(china.columns.values[3:]))
            data = [ list(reversed(china.iloc[i, 3:])) for i in range(0, 3) ]
            labels = list(china.loc[:, 'Series Name'])
            
            plot_generic (
                x = time,
                curves = data,
                labels = labels,
                x_label = 'year',
                y_label = 'USD (variable)',
                filename = 'China_GDP',
                title = 'Economic Development Indicators of China',
                custom_width = 30,
            )
            
        
        else:
            raise Exception(f'Mode \"{mode}\" Not Implemented')
        
        
        
def compile_json_data_by_year(start, end, lang='CN', suppressPrint=False):
    if len(code_to_country) == 0:
        init_country_codes()
    organized_data = {}
    lang = lang.upper()
    
    for year in reversed(range(start, end+1)):
        path = '{folder}/{year}.json'.format(folder=DATA_PATH, year=year)
        with open(file=path, mode='rt') as infile:
            data = json.load(infile)
        
        for entry in data['countryCode']:
            name_cn, code, cnt = entry[0], entry[1].upper(), entry[2]
            country_fullname = f'{name_cn} ({code})'
            
            if lang == 'EN':
                try:
                    name_en = code_to_country[code]
                except:
                    eprint(f'cannot find English name for: {code}')
                    continue
                country_fullname = f'{name_en} ({code})'
            
            entry = {year:cnt}
            country_data = organized_data.get(country_fullname, {})
            country_data.update(entry)
            organized_data[country_fullname] = country_data
            
    df = pd.DataFrame(organized_data).transpose()
    csv_path = creat_dir('csv_data_compiled')
    
    if lang == 'CN': df.to_csv(f'{csv_path}/{start}-{end}-CN.csv', index = True)
    if lang == 'EN': df.to_csv(f'{csv_path}/{start}-{end}-EN.csv', index = True)
    
    if not suppressPrint:
        print(f'Done. {start}-{end} Compiled.')
    
    return organized_data


def plot_by_country(data):
    years = [i for i in range(1980, 2021+1)]
    country_names = [ name.split('(')[-1][0:2] for name in data.keys() ]
    country_data = []
    
    for value in data.values():
        cnts = []
        for y in years:
            cnts.append(value.get(y, 0))
        
        country_data.append(cnts)
    
    
    plot_generic (
                x = years,
                curves = list(country_data[0:6]),
                labels = list(country_names[0:6]),
                filename = 'patent_by_years',
                title = 'patent count per country per year',
                custom_width = 30,)


def get_known_country_codes(csv_name:str):
    with open(file=csv_name, mode='rt') as f:
        df = pd.read_csv(f)
    country_data = list(df.iloc[:, 0])
    
    known_codes = []
    for name in country_data:
        known_codes.append( name.split('(')[-1][0:2] )
        
    return known_codes
    


def compile_patenthub_dataframe(start=2015, end=2021):
    compile_json_data_by_year(start, end, lang='EN')
    with open(file=f'csv_data_compiled/{start}-{end}-EN.csv', mode='rt') as f:
        df = pd.read_csv(f)
    fullnames = list(df.iloc[:, 0])
    codes = [ name.split('(')[-1][0:2] for name in fullnames ]
    
    df.rename(columns = {'Unnamed: 0':'Country'}, inplace = True)
    df = df.set_index('Country')     # set country names as index column
    df = df.iloc[:, ::-1]   # order list into ascending chronological order
    
    return df



def main():
    init_country_codes()
    df = compile_patenthub_dataframe(start=2010, end=2021)
    
    
    sub_df = df.iloc[0:5, :]
    stacked_df = sub_df.apply(lambda x: x*100/sum(x), axis=1)
    
    # sub_df.plot.bar(stacked=True, 
    # y = list(sub_df.iloc[:, ::-1]) )
    sub_df.transpose().plot.line()
    stacked_df.plot.bar(stacked=True, )
    
    plt.xticks(rotation=30, horizontalalignment='center')
    plt.show()
    

    # get_known_country_codes('csv_data_compiled/1900-2021.csv')
    
    

    
    # graph_csv(filename='s_port-13-21')
    # graph_csv(filename='china_dev', directory='csv_
    # ternal', mode=2)

    
    
    

if __name__ == '__main__':
    main()