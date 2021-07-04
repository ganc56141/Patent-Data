import math, statistics
from patenthub_api import PatentHub_API
from generic_api import creat_dir
from utility import eprint, next_available_path

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

minPageNum, maxPageNum = 1, 100
minEntryPerPage, maxEntryPerPage = 10, 50


def obtain_data():
    for pg in range(1, 10):
        
        PatentHub_API.s_port(
            query='电器',
            datascope='CN',
            page=pg,
            pagesize=maxEntryPerPage,
            sorting='!applicationDate'
        )
    
    # PatentHub_API.base_port(
    #     uniqueID='JP3558752B2',
    # )

    
    # PatentHub_API.ration_port(
    #     query='2010',
    #     category='applicationYear'
    # )
    
    
    # PatentHub_API.used_port(
    #     apiURL = '/api/patent/base'
    # )
    
    
    
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
        
    df.to_csv(f'csv/{response_name}-{start}-{end}.csv', index = True)



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
    
    filepath = os.path.dirname(os.path.realpath(__file__))          # NOT os.getcwd() <——> this incantation is faulty
    
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
    creat_dir(folder_name='graphs', path=filepath)      # initialize directory
    plt.savefig(f'{filepath}/graphs/{filename}.png', format="PNG")
    plt.close()

      

def graph_csv(filename, directory='csv'):
    with open(f'{directory}/{filename}.csv', mode='rt') as f:
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
    
    
    



def main():
    # PatentHub_API.test_connectivity(saveToDisk=False)
    # obtain_data()
    # compile_responses_to_csv(response_name='s_port', start=13, end=21)
    graph_csv(filename='s_port-13-21')

    
    
    

if __name__ == '__main__':
    main()