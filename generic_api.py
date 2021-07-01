from typing import *
from utility import *
import requests, json
import http.client
import pickle


class Generic_API:
    cache = 'saved.pickle'
    default_json_cache = 'response.json'
    
    @classmethod
    def get_fox(cls, save: bool = False, forceUpdate: bool = False) -> None:
        api = 'https://randomfox.ca/floof'
        
        # getting response
        r = cls.pickle_response(op="load")
        if forceUpdate or not r:
            Generic_API.make_query(url=api)
            
        if save: 
            cls.cache_response_in_json(r, custom_json_file="test_api")
        
        data = r.json()
        # print('Response = ' + json.dumps(data) +'\n')
        return data

    @classmethod
    @request_decorator
    def test_connectivity(cls, method='fox', saveToDisk=False) -> bool:
        # will implement others methods in the future
        print("Running Test API...")
        
        data = cls.get_fox(save=True, forceUpdate=True) if saveToDisk else cls.get_fox(save=False, forceUpdate=True)
        
        try:
            if data['image'].find('randomfox.ca') and data['link'].find('randomfox.ca'):
                print('Response: Valid')
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
    def pickle_response(cls, op: str = "load", data = ""):
        """allows simple caching of data

            Args:
                op (str, optional): specify operation, supports load and dump. Defaults to "load".
                data (str, optional): data to be cached. Defaults to "".

            Raises:
                Exception: operation not supported

            Returns:
                varies
        """
        
        if op == "dump":
            with open(cls.cache, 'wb') as outfile:
                pickle.dump(data, outfile)
                
        elif op == "load":
            if os.path.isfile(cls.cache):
                with open(cls.cache, 'rb') as infile:
                    return pickle.load(infile)
            else:
                return False
        
        else:
            raise Exception(f'{op} not implemented')
        

    @classmethod
    def cache_response_in_json(cls, r, custom_json_file=None, custom_json_folder='response_json', enumerated=True):
        folder_path = creat_dir(folder_name=custom_json_folder)
        
        json_filename = cls.default_json_cache if custom_json_file == None else custom_json_file
        json_path = '{path}/{filename}-%s.json'.format(path=folder_path, filename=json_filename)
        if enumerated: json_path = next_available_path(json_path)
        else: json_path = json_path % 'volatile'
        
        try:
            f = open(file=json_path, mode='wt')
            f.write(json.dumps(r.json()))
            f.close()
        except:
            eprint("Error: Cannot convert response to JSON")
            return False
        return json_path



