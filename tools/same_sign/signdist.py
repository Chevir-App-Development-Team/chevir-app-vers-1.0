"""Hand-path distance between glosses from the avatar builder's cached clips (lite pose, VIDEO, 8 signers).
compare(a, b) works on gloss texts (build_avatar_words.entries keys) and returns 1-NN leave-one-out balanced
accuracy + mean cross / within DTW distance ratio, or None when a gloss has fewer than 3 usable clips."""
import hashlib, itertools, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from azsld_zip import AzsldZip  # noqa: E402
import build_avatar_words as B  # noqa: E402

TAG = 'pose_landmarker_lite-face-r3-m3-w8-s2'
CACHE = ROOT / 'data/azsld/lexicon_cache'
_z = AzsldZip(str(ROOT / 'data/azsld/AzSLD_Sentences.zip'))
ents = B.entries(ROOT / 'ml/splits', 1, str(ROOT / 'data/azsld/AzSLD_Words_200.zip'))
text_of = {g: t for t, e in ents.items() for g in e['labels']}
_feats = {}


def clips(text):
    if text not in _feats:
        out, e = [], ents.get(text)
        if e:
            for c in B.pick_candidates(text, e, 8, _z.available):
                src = c['rec'] or c['member']
                h = hashlib.md5(f"{src}|{c['a']}|{c['b']}|{TAG}".encode()).hexdigest()[:20]
                f = CACHE / f'{h}.json'
                if f.exists():
                    r = json.loads(f.read_text(encoding='utf8'))
                    if r.get('rec') and r['score'] >= 0.45:
                        out.append(B.features(r['rec']))
        _feats[text] = out
    return _feats[text]


def compare(a, b):
    A, Bc = clips(a), clips(b)
    if len(A) < 3 or len(Bc) < 3:
        return None
    items = [(0, x) for x in A] + [(1, x) for x in Bc]
    n = len(items)
    D = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        D[i, j] = D[j, i] = B.dtw(items[i][1], items[j][1])
    np.fill_diagonal(D, np.inf)
    lab = np.array([k for k, _ in items])
    nn = lab[np.argmin(D, axis=1)]
    acc = np.mean([np.mean(nn[lab == k] == k) for k in (0, 1)])
    fin = lambda m: m[np.isfinite(m)].mean()
    wa, wb = fin(D[np.ix_(lab == 0, lab == 0)]), fin(D[np.ix_(lab == 1, lab == 1)])
    cross = D[np.ix_(lab == 0, lab == 1)].mean()
    return {'n': [len(A), len(Bc)], 'bal_acc': round(float(acc), 3), 'ratio': round(float(cross / ((wa + wb) / 2)), 3)}
