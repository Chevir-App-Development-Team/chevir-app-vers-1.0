"""Turn the DTW merge groups (data/azsld/same_sign/groups.json) into ml/same_sign.csv rows, named like sign_merge():
a group that meets a confirmed family or an existing same_sign group takes its name; otherwise the dictionary
form (infinitive or the bare word), otherwise the most frequent gloss.
.venv/Scripts/python.exe -I tools/same_sign/write_same_sign.py <repo> <groups.json>   (prints the rows)"""
import csv, json, sys
from collections import Counter
from pathlib import Path
root, gpath = Path(sys.argv[1]), Path(sys.argv[2])
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(root / 'tools')); sys.path.insert(0, str(root / 'ml'))
import build_avatar_words as B  # noqa: E402
load_groups = lambda p: [(r["gloss"], r["group"]) for r in csv.DictReader(open(p, encoding="utf8")) if r["confirmed"] == "yes"]

ents = B.entries(root / 'ml/splits', 3, str(root / 'data/azsld/AzSLD_Words_200.zip'))
labels = set()
for m in ('cont_model', 'word_model'):
    labels |= set(json.loads((root / f'public/tercume/assets/{m}/manifest.json').read_text(encoding='utf8'))['labels'])
freq = Counter(r['gloss'] for r in csv.DictReader(open(root / 'ml/splits/glosses_split.csv', encoding='utf8')))
fam = dict(load_groups(str(root / 'ml/lemma_groups.csv')))
for r in csv.DictReader(open(root / 'ml/same_sign.csv', encoding='utf8')):
    fam[r['gloss']] = r['group']
rows = []
for g in json.loads(gpath.read_text(encoding='utf8')):
    gl = {t: sorted(x for x in ents[t]['labels'] if x in labels) for t in g['texts']}
    members = sorted({x for v in gl.values() for x in v})
    known = sorted({fam[x] for x in members if x in fam})
    base = [x for x in members if x.endswith(('MAQ', 'MƏK'))] or [x for x in members if B.az_lower(x) == g['key']]
    name = known[0] if known else base[0] if base else max(members, key=lambda x: freq[x])
    for x in members:
        if x != name and fam.get(x) != name:
            rows.append((name, x, g['bal_acc'], g['ratio'], ' = '.join(members)))
    if len(known) > 1:
        print('WARNING: group meets two families', known, members)
for r in rows:
    print(','.join(map(str, r)))
