from typing import *
import os, sys, requests, json
import http.client
import pickle

filepath = os.path.dirname(os.path.realpath(__file__))          # NOT os.getcwd() <——> this incantation is faulty


def request_decorator(interface):
    def inner(*args, **kwargs):     # must have inner function to take and transfer the proper arguments
        print("\n----------------------------------")
        print("         Start of Request\n")
        
        interface(*args, **kwargs)

        print("\n         End Of Request")
        print("----------------------------------\n")
    return inner


class Generic_API:
    cache = 'saved.pickle'
    default_json_cache = 'response.json'
    
    @classmethod
    def get_fox(cls, save: bool = False, forceUpdate: bool = False) -> None:
        api = 'https://randomfox.ca/floof'
        
        # getting response
        r = cls.pickle_response(op="load")
        if forceUpdate or not r:
            r = requests.get(api)
            
            printout = f'Status: {r.status_code} {http.client.responses[r.status_code]}'
            
            if r.status_code != 200:
                raise Exception(printout)
            print(printout)
            
        if save: cls.pickle_response(op="dump", data=r)
        
        # using response
        if save:
            with open(file='response.json', mode='wt') as f:
                f.write(repr(r.json()))
        
        data = r.json()
        
        # print('Response = ' + json.dumps(data) +'\n')      # converts json dictionary to string (not necessary, but safe to have)
        return data

    @classmethod
    @request_decorator
    def test_connectivity(cls, method='fox', saveToDisk=False) -> bool:
        # will implement others methods in the future
        print("Running Test API...")
        
        data = cls.get_fox(save=True, forceUpdate=True) if saveToDisk else cls.get_fox(save=False, forceUpdate=True)
        
        if data['image'].find('randomfox.ca') and data['link'].find('randomfox.ca'):
            print('Response: Valid')
            return True
        else:
            print('Response: Invalid')
            return False
            
        
            
    # makes request and caches entire response in binary format
    @classmethod
    def make_query(cls, url):
        r = requests.get(url)
            
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
    def cache_response_in_json(cls, r, custom_json_cache=None):
        json_cache = cls.default_json_cache if custom_json_cache == None else custom_json_cache
        try:
            f = open(file=json_cache, mode='wt')
            f.write(json.dumps(r.json()))
            f.close()
        except:
            eprint("Error: Cannot convert response to JSON")
            return False
        return json_cache        



# ====================================
#           Utility Methods
# ====================================

def creat_dir(folder_name: str) -> str:
    """helper function that creates directory if not yet exists

        Args:
            folder_name (str): name of new subdirectory

        Returns:
            str: complete path to subdirectory
            
    """    
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = "{directory}/{subdirectory}/".format(directory = dir_path, subdirectory=folder_name)
    mode = 0o755
    try:  
        os.makedirs(path, mode)
    except OSError:
        pass
    return path


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

