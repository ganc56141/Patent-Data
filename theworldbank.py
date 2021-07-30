import numpy as np
import pandas as pd
import wbgapi as wb
import requests_cache
import datetime
import os
from utility import eprint, touch

requests_cache.install_cache('wb_cache')
indicator_filename = 'indicators.txt'
wb.db = 2       # the index for "development indicators database"


def pullIndicators():
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

def getIndicatorTable(filename=indicator_filename):
    pullIndicators()
    
    indict_lookup = {}
    with open(file=indicator_filename, mode='rt') as f:
        lines = f.readlines()
        for line in lines:
            id_, value = line.split('  ')[0], line.split('  ')[-1].strip()
            indict_lookup[id_] = value
    return indict_lookup
    

def buildDataSet(indicators, forceUpdate=False):
    table = getIndicatorTable()
    
    data = []
    # data fetch method (depreciated)
    # for elem in wb.data.fetch('NY.GDP.MKTP.CD',labels=True):
    #     observation = '{} {} {}'.format(elem['economy']['value'], elem['time']['value'], elem['value'])
    #     data.append(observation)
    for indi in indicators:
        csv_path = f'dev_data/{table[indi]}.csv'
        if os.path.exists(csv_path) and forceUpdate==False:
            continue
        
        df = wb.data.DataFrame(indi)
        df = df.iloc[:, ::-1]   # order list into ascending chronological order
        
        touch(csv_path)
        with open(file=csv_path, mode='wt') as f:
            df.to_csv(path_or_buf=f)
    
    
    

def main():
    # db_info = wb.source.info()
    # print(db_info)
    
    # GDP at PPP, GNI per capita, Urban population, Literacy rate (adult total), mortality rate (neonatal)
    indicators = ['NY.GDP.MKTP.CD', "NY.GNP.PCAP.CD", "SP.URB.TOTL", "SE.ADT.LITR.ZS", "SH.DYN.NMRT", "NY.ADJ.SVNX.CD"]
    buildDataSet(indicators)
    
    # import rich
    # table = getIndicatorTable()
    # rich.print(table)



if __name__ == '__main__':
    main()
