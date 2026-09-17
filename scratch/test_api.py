import urllib.request

endpoints = [
    '/api/words?search=salam',
    '/api/words',
    '/api/search?q=salam',
    '/api/media-proxy',
    '/api/v1/words?search=salam'
]

for ep in endpoints:
    url = f"https://www.jestdili.az{ep}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        res = urllib.request.urlopen(req)
        print(f"Success {ep}: {res.status}")
        print(res.read().decode('utf-8')[:200])
    except Exception as e:
        print(f"Failed {ep}: {e}")
