from typing import *
from utility import *
import requests, json
import http.client
import pickle


class Generic_API:
    cache = 'cache/saved.pickle'
    default_json_cache = 'cache/response.json'
    
    @classmethod
    def get_fox(cls, forceUpdate: bool = False) -> None:
        api = 'https://randomfox.ca/floof/'
        
        # getting response
        
        r = cls.pickle_response(op="load")
        if forceUpdate or not r:
            r = Generic_API.make_query(url=api)
        
        return r
        # print('Response = ' + json.dumps(data) +'\n')


    @classmethod
    @request_decorator
    def test_connectivity(cls, method='fox', saveToDisk=False) -> int:
        # will implement others methods in the future
        print("Running Test API...")
        
        r = cls.get_fox(forceUpdate=True)
        data = r.json()
        
        try:
            if data['image'].find('randomfox.ca') and data['link'].find('randomfox.ca'):
                print('Response: Valid')
                
                if saveToDisk:
                    json_path = cls.cache_response_in_json(r, custom_json_file="cache/test_api")
                    json_index = json_path.rstrip('.json').rpartition('-')[-1]
                    print(json_index)
                    return json_index
                
                return True
            else:
                print('Response: Invalid')
                return False
        except:
            print('Response: Invalid')
            return False
    

            
            
    # makes request and caches entire response in binary format
    @classmethod
    def make_query(cls, url, payload=None, encodeParam=False):
        if encodeParam and payload:
            for p in payload.keys():
                payload[p] = requests.utils.quote(str(payload[p]))
        
        r = requests.get(url, params=payload)
            
        printout = f'Status: {r.status_code} {http.client.responses[r.status_code]}'
            
        if r.status_code != 200:
            raise Exception(printout)
        print(printout)
        
        cls.pickle_response(op="dump", data=r)
        return r


    @classmethod
    def pickle_response(cls, op: str = "load", data = "", custom_cache:str = None):
        """allows simple caching of data

            Args:
                op (str, optional): specify operation, supports load and dump. Defaults to "load".
                data (str, optional): data to be cached. Defaults to "".

            Raises:
                Exception: operation not supported

            Returns:
                varies
        """
        cache = custom_cache if custom_cache != None else cls.cache
        touch(cache)
        
        if op == "dump":
            with open(cache, 'wb') as outfile:
                pickle.dump(data, outfile)
                
        elif op == "load":
            try:
                with open(cache, 'rb') as infile:
                    return pickle.load(infile)
            except Exception:
                return False
        
        else:
            raise Exception(f'{op} not implemented')
    
    
    @classmethod
    def pickle_bytestream(cls, *args, **kwargs):
        ''' Alternate interface for 'pickle_response' function '''
        return cls.pickle_response(*args, **kwargs)
            
    

    @classmethod
    def cache_response_in_json(cls, r, custom_json_file=None, enumerated=True):
        json_path = custom_json_file if custom_json_file != None else cls.default_json_cache
        json_path = format_path_for_enumeration(json_path)
        
        if enumerated: json_path = next_available_path(json_path)
        else: json_path = json_path % 'volatile'
        touch(json_path)
        
        try:
            f = open(file=json_path, mode='wt')
            f.write(json.dumps(r.json()))
            f.close()
        except:
            eprint("Error: Cannot convert response to JSON")
            return False
        return json_path