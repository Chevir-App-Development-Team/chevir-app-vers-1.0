"""Test scores of sentence models whose labels differ (E15), from their saved detections_test.csv, scored as
notebook 06 scores them (segment_eval.Tally over all test recordings, frontal camera):
  own     each model in its own labels (the annotations merged the way that model was trained);
  common  every label of every model and annotation mapped into one label space where all the given merge
          lists hold at once, so models trained on different merges are compared on equal terms.

python tools/same_sign/compare_cont.py NAME=DETECTIONS_CSV:MANIFEST_JSON:GIT_REV ...
  GIT_REV: the commit whose ml/lemma_groups.csv + ml/same_sign.csv the model was trained with ('-' = working tree)
"""
import csv
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ml'))
sys.stdout.reconfigure(encoding='utf-8')
import groups  # noqa: E402
from azsl import segment_eval as SE  # noqa: E402

BLANK, UNK = '<blank>', '<unk>'


def merges_at(rev):
    if rev == '-':
        return groups.sign_merge()
    tmp = tempfile.mkdtemp(prefix='merge_')
    paths = []
    for name in ('lemma_groups.csv', 'same_sign.csv'):
        p = os.path.join(tmp, name)
        with open(p, 'wb') as f:
            f.write(subprocess.run(['git', 'show', f'{rev}:ml/{name}'], cwd=ROOT, capture_output=True, check=True).stdout)
        paths.append(p)
    return groups.sign_merge(*paths)


def read(p):
    with open(p, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


models = []
for arg in sys.argv[1:]:
    name, rest = arg.split('=', 1)
    det, man, rev = rest.split(':') if rest.count(':') == 2 else (*rest.rsplit(':', 2)[:2], '-')
    import json
    labels = json.load(open(man, encoding='utf-8'))['labels']
    models.append(dict(name=name, det=det, labels=labels, merge=merges_at(rev)))

sp = groups.spelling()
# common label space: every merge list at once (union-find over gloss → group of each list)
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
for m in models:
    for g, grp in m['merge'].items():
        parent[find(g)] = find(grp)
common = lambda g: find(sp.get(g, g))      # noqa: E731

frames = {r['rec_id']: int(float(r['frames'])) for r in read(ROOT / 'ml/splits/sentences_split.csv') if r['frames']}
tags = [r for r in read(ROOT / 'ml/splits/glosses_split.csv')
        if r['split'] == 'test' and r['has_landmarks'] == 'True' and r['start'] and r['end']]
raw_anns = defaultdict(list)
for r in tags:
    raw_anns[r['rec_id']].append((float(r['start']), float(r['end']), r['gloss']))

print(f'{len(raw_anns)} test recordings with annotations')
for m in models:
    dets = defaultdict(list)
    for r in read(m['det']):
        dets[r['rec_id']].append((int(r['start0']), int(r['end0']), r['word']))
    own = lambda g, mg=m['merge']: mg.get(sp.get(g, g), sp.get(g, g))      # noqa: E731
    known_own = set(m['labels']) - {BLANK, UNK}
    known_common = {common(g) for g in known_own}
    res = {}
    for space, f, known in (('own', own, known_own), ('common', common, known_common)):
        tally = SE.Tally()
        for rid in dets.keys() | raw_anns.keys():
            if rid not in frames:
                continue
            T = frames[rid]
            a_ = [(max(0, int(s) - 1), min(int(e), T), f(g), f(g) in known) for s, e, g in raw_anns.get(rid, [])]
            d_ = [(a, b, w if space == 'own' else common(w)) for a, b, w in dets.get(rid, [])]
            tally.add(sorted(d_), a_, {})
        res[space] = tally.metrics()
    print(f"{m['name']:14s} {len(known_own)} words | " + ' | '.join(
        f"{s}: P {v['precision']:.3f} R {v['recall_all']:.3f} F1 {v['f1_all']:.4f} WER {v['wer_all']:.4f}"
        for s, v in res.items()))
print(f'common label space: {len({common(g) for m in models for g in m["labels"]})} labels over the models')
