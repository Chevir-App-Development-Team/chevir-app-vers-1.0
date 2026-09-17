import re
with open('category.txt', 'r', encoding='utf-8') as f:
    text = f.read()

import urllib.parse
text = urllib.parse.unquote(text)

words = re.findall(r'\"wordAz\":\"([^\"]+)\".*?\"videoUrl\":\"([^\"]+)\"', text, re.IGNORECASE)
for w, v in set(words):
    print(f"{w} -> {v}")
    
words2 = re.findall(r'\"word\":\"([^\"]+)\".*?\"videoUrl\":\"([^\"]+)\"', text, re.IGNORECASE)
for w, v in set(words2):
    print(f"{w} -> {v}")

print("Total mp4 links:", len(re.findall(r'mp4', text)))
