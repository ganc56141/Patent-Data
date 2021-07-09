from typing import *
import os, requests, json
import http.client
from dotenv import load_dotenv

from utility import *
from generic_api import Generic_API

load_dotenv()
API_USERNAME = os.environ['USER']
API_KEY = os.environ['API_KEY']

minPageNum, maxPageNum = 1, 100
minEntryPerPage, maxEntryPerPage = 10, 50


class PatentHub_API(Generic_API):
    @staticmethod
    @request_decorator
    def make_query(api, parameters, custom_json_file=None, enumerated=True):
        
        # params = [f'{p}={value}' for p, value in parameters.items()]
        # url = f'{api}?' + '&'.join(params)
        
        r = Generic_API.make_query(url=api, payload=parameters)
        print(f'Payload Status: {r.json()["code"]}')
        if json_path := Generic_API.cache_response_in_json(r, custom_json_file, enumerated=enumerated):
            print(f'JSON exported successfully --> {json_path}')

        json_index = json_path.rstrip('.json').rpartition('-')[-1]
        return json_index
        
    
    @classmethod
    def s_port(cls, query, datascope='cn', sorting='relation', page=1, pagesize=50, version=1):
        api = 'https://www.patenthub.cn/api/s'
        parameters = {
            'ds' : datascope,     # cn / all (meaning global)
            't' : API_KEY,
            'q' : query,
            'p' : page,       # the page number to return 1 to 100
            'ps': pagesize,       # 10-50 entries per page
            'v' : version,
            's' : sorting       # relation, applicationDate, documentDate, rank     (prefix with ! for descending order)
        }
        
        
        return cls.make_query(api, parameters, custom_json_file='s_port')
        
        
    # single patent detailed search
    @classmethod
    def base_port(cls, uniqueID, version=1, enumerated=False):
        api = 'https://www.patenthub.cn/api/patent/base'
        parameters = {
            't' : API_KEY,
            'id' : uniqueID,        # the unique ID of that patent
            'v' : version,
        }
        
        return cls.make_query(api, parameters, custom_json_file='base_port')
    
    
    
    
    @classmethod
    def ration_port(cls, query, version=1, datascope='all', category='countryCode'):
        api = 'https://www.patenthub.cn/api/ration'
        parameters = {
            't' : API_KEY,
            'ds' : datascope,        # the unique ID of that patent
            'v' : version,
            'q' : query,
            'c' : category,
        }
        
        return cls.make_query(api, parameters, custom_json_file='ration_port')
    
    
    # check remaining queries of a certain api 
    @classmethod
    def used_port(cls, apiURL, version=1):
        api = 'https://www.patenthub.cn/api/used'
        parameters = {
            't' : API_KEY,
            'v' : version,
            'apiUrl' : apiURL.replace('/', '%2F'),        # the unique ID of that patent
        }
        
        return cls.make_query(api, parameters, custom_json_file='used_port', enumerated=False)
        
    @classmethod
    def query_by_year_and_country(cls, query, start_year:int, end_year:int = None, countryCode='US', sorting='!applicationDate'):
        end_year = end_year or start_year
        year_range = f'applicationYear:[{start_year} TO {end_year}]'
        country = f'countryCode:{countryCode}'
        
        logical_query = ' AND '.join([query, year_range, country])
        
        
        json_index = cls.s_port(
            query=logical_query,
            datascope='all',
            page=1,
            pagesize=maxEntryPerPage,
            sorting='!applicationDate'
        )
        
        return json_index
        
        
        
        
