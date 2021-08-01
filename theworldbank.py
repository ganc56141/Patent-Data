import numpy as np
import pandas as pd
import wbgapi as wb
import requests_cache
import asyncio
import concurrent.futures
import datetime
import os, time
from rich.console import Console
from utility import creat_dir, eprint, touch

console = Console()
requests_cache.install_cache('wb_cache')
reference_folder = 'references'
indicator_filename = f'{reference_folder}/indicators.txt'
worldbankcodes_path = f'{reference_folder}/worldbankcodes.csv'
wb.db = 2       # the index for "development indicators database" (every indicator is pulled fomr this database)


def pullIndicators():
    touch(indicator_filename)
    with open(indicator_filename, mode='wt') as f:
        info = wb.series.info()     # this shows all the possible indicators in this database
        f.write(repr(info))

def findIndicator(name='education', forceUpdate=False):
    if not os.path.exists(indicator_filename) or forceUpdate:
        pullIndicators()

    try:
        f = open(indicator_filename, mode='rt')
        lines = f.readlines()
        f.close()
    except:
        eprint("Error: cannot open indicators file")
        return
    
    with open(f'Indicators-{name}.csv', mode='wt') as f:    
        id_, value = lines[0].split('  ')[0], lines[0].split('  ')[-1]
        f.write( '{},{}'.format(id_, value) )
        cnt = 0
        for line in lines:
            if line.lower().find(name.lower()) != -1:
                id_, value = line.split('  ')[0], line.split('  ')[-1]
                f.write( '{},{}'.format(id_, value) )
                cnt+= 1
        print(f'Done. {cnt} entries found.') 
    
    
        
    
def getCodeToCountryName_worldbank(path=worldbankcodes_path):
    table = {}
    
    df = pd.read_csv(worldbankcodes_path)
    codes, names = list(df.iloc[:,0]), list(df.iloc[:,1])
    for code, name in zip(codes, names):
        table[code] = name
    
    return table



def wbdataWorkflow():
    import wbdata
    
    # translate country name to code (matching with other database)
    wbdata.search_countries('united')
    
    # EITHER find right database to use
    wbdata.get_source()

    # then find right indicators
    wbdata.get_indicator(source=2)  
    
    # OR seach for indicator directly (apparently each indicator will have internally correlated database)
    wbdata.search_indicators("gdp per capita")
    wbdata.search_indicators("education")
    
    # could also divide into income levels
    wbdata.get_incomelevel()
    
    # get right timeframe
    data_date = datetime.datetime(2010, 1, 1), datetime.datetime(2011, 1, 1)                                                                                          
    
    # query data
    wbdata.get_data("IC.BUS.EASE.XQ", country="USA")
    wbdata.get_data("NY.GDP.MKTP.CD", country="USA")
    
    # format into nice pandas dataframe
    countries = [i['id'] for i in wbdata.get_country(incomelevel='HIC')]                                                                                                 
    indicators = {"IC.BUS.EASE.XQ": "doing_business", "NY.GDP.PCAP.PP.KD": "gdppc"}         
    df = wbdata.get_dataframe(indicators, country=countries, convert_date=True)   
    df.describe()


def calCorrelation():
    pass


# give you translation table of indicators to their respective names
def getIndicatorTable(filename=indicator_filename):
    if not os.path.exists(indicator_filename):
        pullIndicators()
    
    indict_lookup = {}
    with open(file=indicator_filename, mode='rt') as f:
        lines = f.readlines()
        for line in lines:
            id_, value = line.split('  ')[0], line.split('  ')[-1].strip()
            indict_lookup[id_] = value
    return indict_lookup
    

# only works with theWorldBank formatted CSV
def addCountryNamesColumn(csv_path):
    df = pd.read_csv(csv_path, index_col='economy')
    
    if 'Country Name' in df.columns:
        return
    
    table = getCodeToCountryName_worldbank()
    codes = list(df.index.values)
    names = []
    for code in codes:
        name = table.get(code, 'Unknown')
        names.append(name)
    df.insert(0, "Country Name", names, True)
    
    df.to_csv(csv_path)


# async version (unfortunately wb did not implement asyncio sleep) (also broken right now)
async def buildDataSet_async(indicators, folderpath='dev_data_all', forceUpdate=False):
    table = getIndicatorTable()
    
    async def request_data(indi, job_num):
        print(f'Retrieving ({job_num+1}/{len(indicators)}): {table[indi]}')
        start = time.perf_counter()
        
        df = wb.data.DataFrame(indi)
        df = df.iloc[:, ::-1]   # order list into ascending chronological order
        
        touch(csv_path)
        with open(file=csv_path, mode='wt') as f:
            df.to_csv(path_or_buf=f)
        
        console.print(f'[i]({i+1}/{len(indicators)})[/i]: [bold cyan]{time.perf_counter() - start}s[/bold cyan]\n')
            
    tasks = []
    for i, indi in enumerate(indicators):
        csv_path = f'{folderpath}/{table[indi]}.csv'
        
        if os.path.exists(csv_path) and forceUpdate==False:
            addCountryNamesColumn(csv_path)
            console.print(f'({i+1}/{len(indicators)}): [bold cyan]Instant[/bold cyan]\n')
            continue
        
        job = asyncio.create_task(request_data(indi, job_num=i))
        tasks.append(job)
    
    for job in tasks:
        await job
    
    print('Done. ALL indicator data retrieved.')

        
def buildDataSet_sequential(indicators, folderpath='dev_data_all', forceUpdate=False):
    table = getIndicatorTable()
    
    # data fetch method (depreciated)
    # data = []
    # for elem in wb.data.fetch('NY.GDP.MKTP.CD',labels=True):
    #     observation = '{} {} {}'.format(elem['economy']['value'], elem['time']['value'], elem['value'])
    #     data.append(observation)
    
    for i, indi in enumerate(indicators):
        print(f'Retrieving ({i+1}/{len(indicators)}): {table[indi]}')
        csv_path = f'{folderpath}/{table[indi]}.csv'
        
        if os.path.exists(csv_path) and forceUpdate==False:
            addCountryNamesColumn(csv_path)
            console.print(f'Response Time: [bold cyan]Instant[/bold cyan]\n')
            continue
        
        start = time.perf_counter()
        
        df = wb.data.DataFrame(indi)
        df = df.iloc[:, ::-1]   # order list into ascending chronological order
        
        touch(csv_path)
        with open(file=csv_path, mode='wt') as f:
            df.to_csv(path_or_buf=f)
        
        console.print(f'Response Time: [bold cyan]{time.perf_counter() - start}s[/bold cyan]\n')
        
        

# threaded version
def buildDataSet_threaded(indicators, folderpath='dev_data_all', forceUpdate=False):
    table = getIndicatorTable()
    
    tot_time, tot_jobs = 0, 0
    
    def request_data(indi, csv_path, job_num):
        nonlocal tot_time, tot_jobs
        
        console.print(f'Retrieving ({job_num+1}/{len(indicators)}): {table[indi]}\n')
        start = time.perf_counter()
        
        df = wb.data.DataFrame(indi)
        df = df.iloc[:, ::-1]   # order list into ascending chronological order
        
        touch(csv_path)       # dashes in the filename are causing additional directory to be created (issue known)
        with open(file=csv_path, mode='wt') as f:
            df.to_csv(path_or_buf=f)
        
        # useful stat printouts
        time_elapsed = time.perf_counter() - start
        console.print(f'[i]({job_num+1}/{len(indicators)})[/i]: [bold cyan]{time_elapsed}s[/bold cyan]\n')
        
        tot_time += time_elapsed
        tot_jobs += 1
        console.print(f'   [u]Average Reponse Time[/u]  -- [#e642aa]{tot_time / tot_jobs:.2f}s [/#e642aa] \n')
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        for i, indi in enumerate(indicators):
            csv_path = f'{folderpath}/{table[indi]}.csv'
            
            if os.path.exists(csv_path) and forceUpdate==False:
                addCountryNamesColumn(csv_path)
                console.print(f'({i+1}/{len(indicators)}): [bold cyan]Instant[/bold cyan]\n')
                continue
            
            executor.submit(request_data, indi=indi, csv_path=csv_path, job_num=i)
        

def main():
    # db_info = wb.source.info()
    # print(db_info)
    
    # 1. get select indicators
    # GDP at PPP, GNI per capita, Urban population, Literacy rate (adult total), mortality rate (neonatal)
    indicators = ['NY.GDP.MKTP.CD', "NY.GNP.PCAP.CD", "SP.URB.TOTL", "SE.ADT.LITR.ZS", "SH.DYN.NMRT", "NY.ADJ.SVNX.CD"]
    buildDataSet_sequential(indicators, folderpath='dev_data_selected')
    print('Done. SELECTED indicator data retrieved.\n')
    
    # 2. get all indicators
    table = getIndicatorTable()
    indicators = list(table.keys())[2:-1]       # some headers and footers to remove
    # buildDataSet_threaded(indicators, folderpath='dev_data_all')
    # print('Done. ALL indicator data retrieved.\n')
    
    
    
    

    
    # import rich
    # table = getIndicatorTable()
    # rich.print(table)



if __name__ == '__main__':
    main()
