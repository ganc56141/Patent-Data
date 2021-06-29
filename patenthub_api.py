from typing import *
import os, requests, json
import http.client

from generic_api import Generic_API, creat_dir, eprint, request_decorator


filepath = os.path.dirname(os.path.realpath(__file__))          # NOT os.getcwd() <——> this incantation is faulty
token = '1eeb30aba878a2502fb14fb7747f84ff197cf4de'
    
class PatentHub_API(Generic_API):
    @staticmethod
    @request_decorator
    def make_query(api, parameters, custom_json_cache=None):
        
        params = [f'{p}={value}' for p, value in parameters.items()]
        url = f'{api}?' + '&'.join(params)
        
        r = Generic_API.make_query(url=url)
        if json_path := Generic_API.cache_response_in_json(r, custom_json_cache):
            print(f'JSON exported successfully --> {json_path}')

        return r
        
    
    @classmethod
    def s_port(cls, query, datascope='cn', page=1, pagesize=50, version=1):
        api = 'https://www.patenthub.cn/api/s'
        parameters = {
            'ds' : datascope,     # cn / all (meaning global)
            't' : token,
            'q' : query,
            'p' : page,       # the page number to return 1 to 100
            'ps': pagesize,       # 10-50 entries per page
            'v' : version,
        }
        
        return cls.make_query(api, parameters, custom_json_cache='s_port.json')
        
        
    # single patent detailed search
    @classmethod
    def base_port(cls, uniqueID, version=1):
        api = 'https://www.patenthub.cn/api/patent/base'
        parameters = {
            't' : token,
            'id' : uniqueID,        # the unique ID of that patent
            'v' : version,
        }
        
        return cls.make_query(api, parameters, custom_json_cache='base_port.json')
    