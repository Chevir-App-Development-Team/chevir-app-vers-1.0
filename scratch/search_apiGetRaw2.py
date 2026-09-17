import urllib.request
import re
url = 'https://www.jestdili.az/_next/static/chunks/76c4226447962f24.js'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    js_code = urllib.request.urlopen(req).read().decode('utf-8')
    idx = js_code.find('apiGetRaw')
    print(js_code[max(0, idx-1000):min(len(js_code), idx+1000)])
except Exception as e:
    print(e)
