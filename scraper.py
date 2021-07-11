import requests, time, datetime
from typing import Generic
from bs4 import BeautifulSoup
from decouple import config
from fake_useragent import UserAgent
import http.cookiejar
from selenium import webdriver as webdriver_basic   # for webdriver
from selenium.webdriver.support.ui import WebDriverWait  # for implicit and explict waits
from selenium.webdriver.chrome.options import Options  # for suppressing the browser
from seleniumwire import webdriver
import pandas as pd
import json

from generic_api import Generic_API
from utility import eprint, touch, HiddenPrints, Dots


API_USERNAME = config('USER')
API_KEY = config('API_KEY')
COOKIE_PATH = config('COOKIE_PATH')
USER_AGENT = config('USER_AGENT')
try:
    DRIVER_PATH_CHROME = config('DRIVER_PATH_CHROME')
    DRIVER_PATH_FIREFOX = config('DRIVER_PATH_FIREFOX')
except Exception:
    DRIVER_PATH_CHROME = 'drivers/chromedriver-91'
    DRIVER_PATH_FIREFOX = 'drivers/geckodriver'

CACHE = 'cache/html.pickle'
JSON_DATA_PATH = 'json_data'
CSV_DATA_PATH = 'csv_data'
JAVASCRIPT_DELAY = 8

# requires admin privileges (not used in current version)
def detect_key():
    import keyboard  # using module keyboard
    while True:  # making a loop
        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('q'):  # if key 'q' is pressed 
                print('Terminating Page...')
                break  # finishing the loop
        except:
            break  # if user pressed a key other than the given key the loop will break


def scrape_patenthub_html_static():
    ua = UserAgent()

    url = 'https://www.patenthub.cn/s?ds=all&dm=mix&p=&ps=10&s=score!&q2=&m=none&fc=[{%22type%22%3A%22applicationYear%22%2C%22op%22%3A%22include%22%2C%22values%22%3A[%222014%22]}]&q=5g'
    # url = 'https://www.patenthub.cn/s?ds=all&q=5g'
    headers = {'User-Agent': ua.chrome}

    cjar = http.cookiejar.MozillaCookieJar(COOKIE_PATH)
    cjar.load()

    r = requests.get(url, headers=headers, cookies=cjar)
    f = open('webpage.html', mode='wb')
    f.write(r.content)
    print(r.status_code)
    
    
    # test run
    page = requests.get('https://www.tmea.org/programs/all-state/history')

    soup = BeautifulSoup(page.text, 'html.parser')

    body = soup.find(id = 'organization')
    options = body.find_all('option')

    for name in options:
        child = name.contents[0]
        print(child)




def scrape_patenthub_html_dynamic(target_url:str, debug:str=None, suppressError=True) -> str:
    dummy_url = 'https://www.patenthub.cn/this404page'
    url = target_url

    cjar = http.cookiejar.MozillaCookieJar(COOKIE_PATH)
    cjar.load()

    # json_cookies = requests.utils.dict_from_cookiejar(cjar)       # slightly slower due to additional function call
    json_cookies = [ {'name':cookie.name, 
                    'value':cookie.value, 
                    'secure':cookie.secure, 
                    'expires':cookie.expires,
                    'discard': cookie.discard,
                    'rfc2109':cookie.rfc2109} for cookie in cjar if cookie.name != 'pref' ]

    # to enable headless
    options = webdriver.ChromeOptions()
    options.add_experimental_option('detach', True)
    options.add_argument('window-size=1200x600') # optional
    options.add_argument('headless')
    driver = webdriver.Chrome(executable_path=DRIVER_PATH_CHROME, options=options)
    
    
    # Create a request interceptor
    def interceptor(request):
        try: 
            del request.headers['user-agent']  # Delete the header first
        except Exception: 
            pass
        request.headers['user-agent'] = USER_AGENT

    # Set the interceptor on the driver
    driver.request_interceptor = interceptor

    # hides some weird library errors
    try:
        print('Phase 1: Requesting page... ', end='')
        driver.get(dummy_url)

        # driver.delete_all_cookies()
        for cookie in json_cookies:
            driver.delete_cookie(cookie['name'])
            driver.add_cookie(cookie)

        if suppressError:
            # suppresses some VPN related printouts
            with HiddenPrints():
                driver.get(url)
        else:
            driver.get(url)
            # driver.refresh()
        
        print("Done")

        
    except Exception as e:
        pass
        
        
    # reveal all hidden items
    def expand_hidden_items(driver) -> int:        
        print('Phase 2: script injections... ', end='')
        base_path = '/html/body/div[3]/div[2]/div[1]/div/div[%i]/a'
        try:
            driver.find_element_by_xpath(xpath= (base_path % 2) )
        except: 
            eprint('Error: expandable menu does not exists/is not loaded')
            return False
        
        i = 2
        cnt = 0
        js_script_action = 'arguments[0].click()'
        while 1:
            try:
                clickable = driver.find_element_by_xpath(xpath= (base_path % i) )
                driver.execute_script(js_script_action, clickable)
                cnt += 1
            except:
                break
            i += 1
        print('Done')
        return cnt
            
    hidden_item_cnt = expand_hidden_items(driver)
    
    # additional wait feature to make sure hidden content has been loaded
    print("Phase 2.1: script executing...", flush=True)
    while 1:
        soup = BeautifulSoup(markup=driver.page_source, features='html.parser')
        activated = soup.find_all(name='div', class_='content active')
        if len(activated) < hidden_item_cnt + 1:        # +1 since one content was active by default (thus not hidden)
            continue
        else:
            break
    
    # spin wait (appaently time.sleep stops EVERYTHING, not applicable here)
    start = time.monotonic()
    dots = Dots()
    dots.start()
    while time.monotonic() - start < JAVASCRIPT_DELAY:
        pass
    dots.stop()
        

    print("Phase 3: saving page source... ", end='')
    realtime_html = driver.page_source
    # realtime_html = driver.execute_script('return document.querySelector("html").innerHTML')
    
    path = touch('cache/human_readable.html')
    with open(file=path, mode='wt') as tmp:
        tmp.write(realtime_html)
    Generic_API.pickle_bytestream(op='dump', data=realtime_html, custom_cache=CACHE)
    
    print("Done")
    
    
    if debug == 'console':
        print("Engaging Interactive Console... \n")
        while 1:
            try:
                exec(input(">>> "))
            except Exception as e:
                print(e)
        
    driver.close()
    return realtime_html


def parse_patenthub_html(savefile:str =f'{JSON_DATA_PATH}/NEWEST_DATA.json', category='N/A') -> dict:
    page_source = Generic_API.pickle_response(op='load', custom_cache=CACHE)
    soup = BeautifulSoup(markup=page_source, features='html.parser')
    
    # -- initialize data dictionary ---
    stats = {}
    stats.update( {'category': category,
                'countryCode': None} )
    
    # --- get patent by country ---
    countries = soup.find_all(name='i', class_='flag')
    
    country_data = []
    for elem in countries:
        code = elem['class'][-1]
        name = elem.parent.find(name='span', class_='key').text
        cnt = elem.parent.find(name='span', class_='doc-count').text
        country_data.append( [name, code, cnt] )
    
    stats['countryCode'] = country_data

    
    # --- get patent by remaining statistics ---
    
    filter_ = soup.find(name='div', attrs={'id': 'filter'})
    dropdowns = filter_.find_all(name='a', class_='title')[1:]
    
    for dd in dropdowns:
        # item_name_cn = dd.text.strip()
        item_name_eng = dd['data-type'].strip()
        data = []
        content = dd.parent.find(name='div', class_='content active')
        content_entries = content.find_all(name='li')
        for entry in content_entries:
            if entry.text.find('无数据') != -1:
                data.append( [None, '0'] )
                break
            entry_key = entry.find(name='span', class_='key').text
            entry_val = entry.find(name='span', class_='doc-count').text
            data.append( [entry_key, entry_val] )
        stats[item_name_eng] = data
    
    touch(savefile)
    with open(file=savefile, mode='wt') as f:
        json.dump(stats, f)
    
    return stats
    
    
def check_exists(html:str, name:str, attrs:dict) -> bool:
    """Check whether a certain DOM element exists in given html page

    Args:
        html (str): the html source page
        name (str): tag name 
        attrs (dict): tag attributes

    Returns:
        bool: exists=True, nonexistant=False
    """
    
    soup = BeautifulSoup(markup=html, features='html.parser')
    exists = soup.find(name=name, attrs=attrs)

    return exists
    
    


def pull_data_by_year(start=None, end=None, delay=1):
    start = start if start != None else 1421       # first patent ever was granted in 1421 in Florence
    
    # string = 'https://www.patenthub.cn/s?ds=all&dm=mix&p=&ps=10&s=score%21&q2=&m=none&fc=%5B%5D&q=5g'
    query_by_year = 'https://www.patenthub.cn/s?ds=all&dm=mix&s=score%21&q=applicationYear%3A'
    
    for year in reversed(range(start, end+1)):
        print(f'Retriving {year} data...')
        start = time.perf_counter()
        query = query_by_year + str(year)
        
        try:
            html = scrape_patenthub_html_dynamic(target_url=query)
        except Exception:
            eprint(f'Internal Error: Data Retrieved from {year+1} to {end}.")')
            return
        
        if not check_exists(html, name='div', attrs={'id': 'countryCode'}):
            print(f"DONE. Data Retrieved from {year+1} to {end}.")
            return
        else:
            print('POSTPROCESSING: Parsing + Saving Data...')
            parse_patenthub_html(savefile=f'{JSON_DATA_PATH}/{year}.json', category=str(year))
            
            print(f'Complete! -- {time.perf_counter() - start:.2f}s\n')
        
        time.sleep(delay)
        
    print(f"DONE. Data Retrieved from {start} to {end}.")
    return


# deprecated
def interface(mode:str, categories:list = None):
    if mode == 'scrape':
        current_year = datetime.datetime.today().year
        pull_data_by_year(end=current_year)
    
    if mode == 'parse':
        clean_data = parse_patenthub_html()
        print(clean_data)
        
        

# upcoming feature
def test_limit(num=10):
    from random_word import RandomWords
    print(RandomWords().get_random_word())
    ...
    
# upcoming feature
def export_as_xml():    
    with open('NEWEST_DATA.json', mode='rt') as f:
        x = json.load(f)
        from dict2xml import dict2xml
        xml = dict2xml(x)
        f2 = open(file='NEWEST_DATA.xml', mode='wt')
        f2.write(xml)


    
def main():
    pull_data_by_year(start=2021, end=2021, delay=1)
    # parse_patenthub_html(savefile=f'{JSON_DATA_PATH}/{1983}.json', category=str(1983))

    
    
    

if __name__ == '__main__':
    main()    
