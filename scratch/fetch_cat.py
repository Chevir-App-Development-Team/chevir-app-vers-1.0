import urllib.request
import re
import urllib.parse
url = 'https://www.jestdili.az/az/lugat/10000000-0000-0000-0000-000000000024'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'RSC': '1'})
try:
    data = urllib.request.urlopen(req).read().decode('utf-8')
    data = urllib.parse.unquote(data)
    
    # Try to find words and their video urls in the RSC payload
    # The payload is usually JSON fragments. We can search for \"wordAz\":\"salam\" or similar
    
    word_matches = re.findall(r'\"wordAz\":\"([^\"]+)\".*?\"videoUrl\":\"([^\"]+)\"', data, re.IGNORECASE)
    for w, v in set(word_matches):
        print(f"Word: {w}, Video: {v}")
        
    word_matches2 = re.findall(r'\"word\":\"([^\"]+)\".*?\"videoUrl\":\"([^\"]+)\"', data, re.IGNORECASE)
    for w, v in set(word_matches2):
        print(f"Word: {w}, Video: {v}")
        
    print("Found videoUrl count:", data.count('videoUrl'))
    
except Exception as e:
    print(e)
