import urllib.request
import re
url = 'https://www.jestdili.az/az/lugat'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8')

js_files = re.findall(r'src=\"([^\"]+\.js)\"', html)
for js in js_files:
    if not js.startswith('http'):
        js = 'https://www.jestdili.az' + js
    try:
        js_code = urllib.request.urlopen(urllib.request.Request(js, headers={'User-Agent': 'Mozilla/5.0'})).read().decode('utf-8')
        if 'apiGetRaw' in js_code:
            idx = js_code.find('apiGetRaw')
            print(f"--- Found apiGetRaw in {js} ---")
            print(js_code[max(0, idx-500):min(len(js_code), idx+500)])
            
            # also let's search for "http" to find the backend url in that file
            urls = re.findall(r'https?://[^\s\"\'\\]+', js_code)
            for u in set(urls):
                print("Found URL:", u)
    except:
        pass
