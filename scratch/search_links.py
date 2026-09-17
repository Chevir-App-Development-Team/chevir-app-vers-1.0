import urllib.request
import re
import urllib.parse
url = 'https://www.jestdili.az/az/lugat/word/salam'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8')
print('salam is in HTML:', 'salam' in html.lower())
links = re.findall(r'http[^\s\"\'\\]+', html)
for l in set(links):
    if '115' in l or '9000' in l or 'mp4' in l or 'video' in l or 'api' in l:
        print(l)
