import urllib.request
import re
import json

url = 'https://www.jestdili.az/az/lugat/10000000-0000-0000-0000-000000000024'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8')

# Try to find words and their video urls in the raw HTML string
word_matches = re.findall(r'\"wordAz\":\"([^\"]+)\".*?\"videoUrl\":\"([^\"]+)\"', html)
for w, v in word_matches:
    print(f"Found match: {w} -> {v}")

if not word_matches:
    print("No matches with wordAz, trying word_az or word...")
    matches2 = re.findall(r'\"word\":\"([^\"]+)\".*?\"videoUrl\":\"([^\"]+)\"', html)
    for w, v in matches2:
        print(f"Found match: {w} -> {v}")

    # Also extract all JSON-like strings to see what we are dealing with
    matches3 = re.findall(r'\"([^\"]+\.mp4)\"', html)
    print("MP4 links found:", len(matches3))
    for m in matches3:
        print(m)
