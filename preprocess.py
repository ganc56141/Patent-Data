import math, statistics
from patenthub_api import PatentHub_API
from generic_api import creat_dir
from utility import next_available_path

minPageNum, maxPageNum = 1, 100
minEntryPerPage, maxEntryPerPage = 10, 50

def obtain_data():
    PatentHub_API.s_port(
        query='电器',
        datascope='CN',
        page=1,
        pagesize=maxEntryPerPage,
        sorting='!documentDate'
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
    



def main():
    PatentHub_API.test_connectivity(saveToDisk=False)
    # obtain_data()
    json
    
    

    
    
    

if __name__ == '__main__':
    main()