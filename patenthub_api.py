from typing import *
import os, requests, json
import http.client

from utility import *
from generic_api import Generic_API


token = '1eeb30aba878a2502fb14fb7747f84ff197cf4de'
    
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

        return r
        
    
    @classmethod
    def s_port(cls, query, datascope='cn', sorting='relation', page=1, pagesize=50, version=1):
        api = 'https://www.patenthub.cn/api/s'
        parameters = {
            'ds' : datascope,     # cn / all (meaning global)
            't' : token,
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
            't' : token,
            'id' : uniqueID,        # the unique ID of that patent
            'v' : version,
        }
        
        return cls.make_query(api, parameters, custom_json_file='base_port')
    
    
    
    
    @classmethod
    def ration_port(cls, query, version=1, datascope='all', category='countryCode'):
        api = 'https://www.patenthub.cn/api/ration'
        parameters = {
            't' : token,
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
            't' : token,
            'v' : version,
            'apiUrl' : apiURL.replace('/', '%2F'),        # the unique ID of that patent
        }
        
        return cls.make_query(api, parameters, custom_json_file='used_port', enumerated=False)
        
        