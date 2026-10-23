"""How often the browser's sentence suggestion is right: test detections of a notebook 06 run → nearest AzSLD
sentence, as src/tercume/continuous.js nearestSentence does it (edit distance / longer length ≤ max_dist).

    python ml/sentence_match.py ml/results/cont2/seq_splice

The sentence list is built from train signers' annotations only (install_continuous.sentences), so a test
signer's own annotation never decides the answer. One test recording is one sentence. A suggestion is
right when it is the recording's sentence, or one with the same sign sequence (AzSLD has different texts
signed alike, which no model can tell apart). Unseen = sentences left out of the model's training.
"""
import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from install_continuous import sentences  # noqa: E402


def levenshtein(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[len(b)]


def nearest(words, sents):
    """(distance / longer length, sentence) of the nearest sentence, ties → the first in the list (as in JS)."""
    best = (2.0, None)
    for s in sents:
        d = levenshtein(words, s['glosses']) / max(len(words), len(s['glosses']))
        if d < best[0]:
            best = (d, s)
    return best


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', help='a notebook 06 run folder with manifest.json and detections_test.csv')
    ap.add_argument('--max-dist', type=float, nargs='+', default=[0.2, 0.34, 0.5, 1.0])
    a = ap.parse_args()
    with open(os.path.join(a.run_dir, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    sents = sentences(manifest, splits=('train',))
    seq_of = {s['folder']: tuple(s['glosses']) for s in sents}
    det = pd.read_csv(os.path.join(a.run_dir, 'detections_test.csv'), dtype={'rec_id': str})
    det = det[~det.rec_id.str.endswith('|side')].sort_values(['rec_id', 'start0'])
    words = {r: list(t.word) for r, t in det.groupby('rec_id')}
    s = pd.read_csv(os.path.join(HERE, 'splits', 'sentences_split.csv'), dtype={'rec_id': str, 'folder': str})
    g = pd.read_csv(os.path.join(HERE, 'splits', 'glosses_split.csv'), dtype={'rec_id': str})
    test = s[(s.split == 'test') & s.rec_id.isin(set(g[g.start.notna()].rec_id)) & s.folder.isin(seq_of)]

    rows = []
    for r in test.itertuples():
        w = words.get(r.rec_id, [])
        d, m = nearest(w, sents) if w else (2.0, None)
        rows.append({'rec_id': r.rec_id, 'unseen': bool(r.heldout_sentence), 'dist': d,
                     'exact': m is not None and m['folder'] == r.folder,
                     'right': m is not None and tuple(m['glosses']) == seq_of[r.folder]})
    df = pd.DataFrame(rows)
    alike = len(sents) - len(set(seq_of.values()))
    print(f'{manifest["run"]}: {len(df)} test recordings ({df.unseen.sum()} of unseen sentences), '
          f'{len(sents)} sentences in the list ({alike} share their sign sequence with another), '
          f'{(df.dist > 1).sum()} recordings without any word')
    print('max_dist | part | suggested | right of suggested | right of all (exact text)')
    for md in a.max_dist:
        for name, part in (('all', df), ('seen', df[~df.unseen]), ('unseen', df[df.unseen])):
            sug = part[part.dist <= md]
            print(f'{md:8.2f} | {name:6} | {len(sug) / len(part):9.3f} | {sug.right.mean():18.3f} | '
                  f'{(part.right & (part.dist <= md)).mean():.3f} ({(part.exact & (part.dist <= md)).mean():.3f})')


if __name__ == '__main__':
    main()
