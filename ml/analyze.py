"""Offline analyses of downloaded training results — no GPU, no retraining.

    python ml/analyze.py ml/results/run4 [--ref ml/results/run3_split_v6] [--groups ml/lemma_groups.csv]

1. the runs of a batch (summary.csv) as a table
2. mean ± std over the seed runs (+ the reference run, seed 0)
3. the saved test predictions re-scored with inflected word forms merged. Only families marked
   `confirmed=yes` in lemma_groups.csv count — whether two forms are one sign is for AzSL
   signers to decide. "All candidates" is shown as an upper bound, labelled as unconfirmed.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from azsl.train import scores  # noqa: E402

KEYS = ('top1', 'top5', 'macro_f1', 'balanced_acc')


def run_metrics(run_dir):
    with open(os.path.join(run_dir, 'metrics.json'), encoding='utf-8') as f:
        return json.load(f)


def merged_scores(run_dir, groups, which='test_words'):
    """Re-score saved probabilities with each family collapsed to one class (probabilities summed)."""
    pz = np.load(os.path.join(run_dir, 'predictions.npz'))
    with open(os.path.join(run_dir, 'vocab.json'), encoding='utf-8') as f:
        vocab = json.load(f)
    to = {g: g for g in vocab}
    for gloss, fam in groups.items():
        if gloss in to:
            to[gloss] = fam
    fams = sorted(set(to.values()))
    M = np.zeros((len(vocab), len(fams)), np.float32)
    for i, g in enumerate(vocab):
        M[i, fams.index(to[g])] = 1
    p = pz[f'{which}_probs'].astype(np.float32) @ M
    y = np.array([fams.index(to[vocab[c]]) for c in pz[f'{which}_labels']])
    return scores(p, y), len(fams)


YES = ('yes', 'y', 'bəli', 'hə', 'he')
NO = ('no', 'n', 'xeyr', 'yox')


def load_groups(path, confirmed_only=True):
    """gloss → family. A form marked yes joins its family; the family's base form (gloss == group)
    joins automatically unless it is marked no. Empty or no = the form stays its own class."""
    with open(path, newline='', encoding='utf-8') as f:
        rows = [{k: (v or '').strip() for k, v in r.items()} for r in csv.DictReader(f)]
    out = {}
    for r in rows:
        if not confirmed_only or r['confirmed'].lower() in YES:
            out[r['gloss']] = r['group']
    if confirmed_only:
        for r in rows:
            if r['gloss'] == r['group'] and r['group'] in out.values() and r['confirmed'].lower() not in NO:
                out[r['gloss']] = r['group']
    # a family needs at least two forms to merge anything
    sizes = {}
    for fam in out.values():
        sizes[fam] = sizes.get(fam, 0) + 1
    return {g: fam for g, fam in out.items() if sizes[fam] > 1}


def fmt(d):
    return ' / '.join(f'{d[k]:.3f}' for k in KEYS)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('batch', help='folder from `kaggle_run.py results --to …` (has summary.csv)')
    ap.add_argument('--ref', help='reference single run (e.g. run 3, seed 0) to include in the seed statistics')
    ap.add_argument('--groups', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lemma_groups.csv'))
    a = ap.parse_args()

    runs = {os.path.basename(os.path.dirname(p)): os.path.dirname(p)
            for p in glob.glob(os.path.join(a.batch, '*', 'metrics.json'))}
    if a.ref:
        runs = {'ref (seed 0)': a.ref, **runs}
    print(f'{"run":16s} {"feat":>4s} {"epoch":>5s}   test W: top-1 / top-5 / macro-F1 / balanced   '
          f'test S: top-1 / macro-F1   majority W')
    allm = {}
    for name, d in runs.items():
        m = run_metrics(d)
        allm[name] = m
        tw, ts = m.get('test_words', {}), m.get('test_signs', {})
        print(f'{name:16s} {m.get("n_features", 656):>4d} {m["best_epoch"]:>5d}   {fmt(tw)}          '
              f'{ts.get("top1", float("nan")):.3f} / {ts.get("macro_f1", float("nan")):.3f}     '
              f'{tw.get("majority_acc", float("nan")):.3f}')

    # seed runs: 'seed1', 'no_face_seed2', … plus the reference run (seed 0 of the same setup)
    seeds = [n for n in allm if n.startswith(('seed', 'ref')) or '_seed' in n]
    if len(seeds) > 1:
        print(f'\nseed runs ({", ".join(seeds)}): mean ± std')
        for which in ('test_words', 'test_signs'):
            vals = {k: np.array([allm[n][which][k] for n in seeds]) for k in KEYS}
            print(f'  {which:10s} ' + '   '.join(f'{k} {v.mean():.3f} ± {v.std(ddof=1):.3f}' for k, v in vals.items()))

    have_pred = {n: d for n, d in runs.items() if os.path.exists(os.path.join(d, 'predictions.npz'))}
    if have_pred and os.path.exists(a.groups):
        conf, cand = load_groups(a.groups, True), load_groups(a.groups, False)
        print(f'\nword forms merged — confirmed families: {len(set(conf.values()))}, '
              f'candidates: {len(set(cand.values()))} (lemma_groups.csv)')
        for name, d in have_pred.items():
            base = allm[name]['test_words']
            line = f'  {name:16s} plain {fmt(base)}'
            if conf:
                s, n = merged_scores(d, conf)
                line += f' | confirmed merged ({n} classes) {fmt(s)}'
            s, n = merged_scores(d, cand)
            line += f' | ALL CANDIDATES, unconfirmed ({n} classes) {fmt(s)}'
            print(line)
        if not conf:
            print('  (no family confirmed yet — fill the `confirmed` column with yes/no)')


if __name__ == '__main__':
    main()
