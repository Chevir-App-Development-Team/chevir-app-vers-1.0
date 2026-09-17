import urllib.request
import re
import urllib.parse
html = urllib.request.urlopen('https://www.youtube.com/results?search_query=' + urllib.parse.quote('salam işarət dili')).read().decode('utf-8')
vids = re.findall(r'\"videoId\":\"([^\"]+)\"', html)
print('Videos:', set(vids))
