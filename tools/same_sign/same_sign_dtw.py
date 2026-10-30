"""Are two glosses the same sign? Hand trajectories of up to 8 signers per gloss (the avatar lexicon's
candidate clips, data/azsld/lexicon_cache, lite pose, VIDEO mode), compared with DTW.

For a pair A / B: 1-NN leave-one-out balanced accuracy over the clips of both (0.5: the clips of A are as
close to B's as to their own = one sign; 1.0: always nearest to their own gloss), and the ratio of the mean
A-B distance to the mean within-gloss distance. Calibrated on known pairs: confirmed families
(ml/lemma_groups.csv) and random pairs of different words.

.venv/Scripts/python.exe -I tools/same_sign/same_sign_dtw.py <repo> <families.json> <out.json>
"""
import csv, itertools, json, random, sys
from pathlib import Path
import numpy as np

root, fam_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(root / 'tools'))
import hashlib  # noqa: E402
from azsld_zip import AzsldZip, cam2  # noqa: E402
import build_avatar_words as B  # noqa: E402

z = AzsldZip(str(root / 'data/azsld/AzSLD_Sentences.zip'))
w200 = str(root / 'data/azsld/AzSLD_Words_200.zip')
ents = B.entries(root / 'ml/splits', 3, w200)
tag = 'pose_landmarker_lite-face-r3-m3-w8-s2'
cache = root / 'data/azsld/lexicon_cache'

feats = {}
def clips(text):
    """Feature sequences of the gloss's candidate clips found in the cache (quality ≥ 0.45)."""
    if text in feats:
        return feats[text]
    e = ents.get(text)
    out = []
    if e:
        for c in B.pick_candidates(text, e, 8, z.available):
            src = c['rec'] or c['member']
            h = hashlib.md5(f"{src}|{c['a']}|{c['b']}|{tag}".encode()).hexdigest()[:20]
            f = cache / f'{h}.json'
            if not f.exists():
                continue
            r = json.loads(f.read_text(encoding='utf8'))
            if r.get('rec') and r['score'] >= 0.45:
                out.append((c['signer'], B.features(r['rec'])))
    feats[text] = out
    return out

def compare(a, b):
    A, Bc = clips(a), clips(b)
    if len(A) < 3 or len(Bc) < 3:
        return None
    items = [(0, s, x) for s, x in A] + [(1, s, x) for s, x in Bc]
    n = len(items)
    D = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        D[i, j] = D[j, i] = B.dtw(items[i][2], items[j][2])
    np.fill_diagonal(D, np.inf)
    lab = np.array([it[0] for it in items])
    nn = lab[np.argmin(D, axis=1)]
    acc = np.mean([np.mean(nn[lab == k] == k) for k in (0, 1)])
    fin = lambda m: m[np.isfinite(m)].mean()
    wa, wb = fin(D[np.ix_(lab == 0, lab == 0)]), fin(D[np.ix_(lab == 1, lab == 1)])
    cross = D[np.ix_(lab == 0, lab == 1)].mean()
    return {'a': a, 'b': b, 'n': [len(A), len(Bc)], 'bal_acc': round(float(acc), 3),
            'ratio': round(float(cross / ((wa + wb) / 2)), 3)}

low = B.az_lower
text_of = {}
for t, e in ents.items():
    for g in e['labels']:
        text_of[g] = t

families = json.loads(fam_path.read_text(encoding='utf8'))
fam_pairs = []
for f in families:
    ts = sorted({text_of.get(g) for g in f['labels']} - {None})
    fam_pairs += [(f['key'], a, b) for a, b in itertools.combinations(ts, 2)]

confirmed = []
with open(root / 'ml/lemma_groups.csv', encoding='utf8') as fh:
    rows = [r for r in csv.DictReader(fh) if r['confirmed'] == 'yes']
for g in {r['group'] for r in rows}:
    ts = sorted({text_of.get(r['gloss']) for r in rows if r['group'] == g} - {None})
    confirmed += [(g, a, b) for a, b in itertools.combinations(ts, 2)]

rng = random.Random(0)
words = sorted(t for t in ents if len(clips(t)) >= 3 and ' ' not in t)
negatives = set()
while len(negatives) < 300:
    a, b = rng.sample(words, 2)
    if a[:3] != b[:3]:
        negatives.add(tuple(sorted((a, b))))

res = {'confirmed': [], 'random_different': [], 'families': []}
for name, pairs in (('confirmed', confirmed), ('random_different', [('', a, b) for a, b in negatives]),
                    ('families', fam_pairs)):
    for key, a, b in pairs:
        r = compare(a, b)
        if r:
            r['key'] = key
            res[name].append(r)
    accs = [r['bal_acc'] for r in res[name]]
    rats = [r['ratio'] for r in res[name]]
    if accs:
        print(f'{name:17s} pairs {len(accs):3d} | 1-NN bal. acc median {np.median(accs):.2f} '
              f'(p10 {np.percentile(accs, 10):.2f}, p90 {np.percentile(accs, 90):.2f}) | '
              f'distance ratio median {np.median(rats):.2f} (p10 {np.percentile(rats, 10):.2f}, p90 {np.percentile(rats, 90):.2f})')
out_path.write_text(json.dumps(res, ensure_ascii=False, indent=0), encoding='utf8')
