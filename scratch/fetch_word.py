import urllib.request
import re
import sys

word = sys.argv[1] if len(sys.argv) > 1 else 'salam'
url = f'https://www.jestdili.az/az/lugat/word/{urllib.parse.quote(word)}'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    html = urllib.request.urlopen(req).read().decode('utf-8')
    mp4s = re.findall(r'(https?://[^\s\"\'\\]+\.mp4)', html)
    print(f'MP4s for {word}:', set(mp4s))
    vids = re.findall(r'videoUrl\":\"([^\"]+)\"', html)
    print(f'videoUrls for {word}:', set(vids))
except Exception as e:
    print(f'Error fetching {word}:', e)
