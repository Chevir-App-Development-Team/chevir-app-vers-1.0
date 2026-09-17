import urllib.request
import urllib.parse
import re
url = 'https://www.jestdili.az/az/lugat/word/salam'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'RSC': '1'})
data = urllib.request.urlopen(req).read().decode('utf-8')
data_decoded = urllib.parse.unquote(data)
matches = re.findall(r'(https?://[^\s\"\'\\]+\.mp4)', data_decoded)
matches2 = re.findall(r'\"videoUrl\":\"([^\"]+)\"', data_decoded)
print('MP4s:', set(matches))
print('videoUrls:', set(matches2))
