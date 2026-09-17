import urllib.request
import re

url = 'https://www.jestdili.az/az/lugat/word/salam'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'RSC': '1'})
try:
    data = urllib.request.urlopen(req).read().decode('utf-8')
    matches2 = re.findall(r'\"([^\"]*mp4[^\"]*)\"', data)
    for m in set(matches2):
        print('Found mp4:', m)
except Exception as e:
    print('Error:', e)
