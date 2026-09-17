import re
with open(r'C:\Users\ELNUR\.gemini\antigravity\brain\11d9ac2b-1dfd-4e2c-b4ae-b8352979ecc3\.system_generated\steps\584\content.md', 'r', encoding='utf-8') as f:
    text = f.read()

urls = re.findall(r'https?://[^\s\"\'\\]+', text)
for u in set(urls):
    print(u)
