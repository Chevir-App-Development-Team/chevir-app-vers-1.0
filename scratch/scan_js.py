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
        apis = re.findall(r'(https?://[^\s\"\'\\]+)', js_code)
        for a in apis:
            if 'jest' in a or 'api' in a or 'backend' in a or 'herokuapp' in a:
                print(a)
        rels = re.findall(r'([\'\"]/api/[^\'\"]+[\'\"])', js_code)
        for r in rels:
            print('Relative API:', r)
    except:
        pass
print('Done scanning JS files.')
