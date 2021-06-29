import math, statistics
from patenthub_api import PatentHub_API

minPageNum, maxPageNum = 1, 100
minEntryPerPage, maxEntryPerPage = 10, 50


def main():
    # PatentHub_API.s_port(
    #     query='electronics',
    #     datascope='all',
    #     page=1,
    #     pagesize=maxEntryPerPage,
    # )
    
    # PatentHub_API.base_port(
    #     uniqueID='JP3558752B2',
    # )
    
    PatentHub_API.test_connectivity(saveToDisk=True)
    PatentHub_API.test_connectivity(saveToDisk=True)
    
    

if __name__ == '__main__':
    main()