"""
Slovo xəritəsinin (ml/slovo_azsld.csv) işarə-işarə yoxlanması (E27): yalnız Slovo ilə öyrənmiş model (rus işarə dili,
ml/pretrain_slovo.py) AzSLD iştirakçılarının Words_200 kliplərini (cümlələrdən kəsilmiş işarələr) tanıyırmı? Hər
AzSLD etiketi üçün top-1 / top-5 və ən çox verilən cavab: tanınmayan işarə xəritədən çıxır, başqa rus işarəsi kimi
tanınan işarəyə o da əlavə olunur. Qərar üçün yalnız train bölgüsü (test iştirakçıları və JestDili.az kənarda qalır).

  D:/chevir/venv/Scripts/python ml/slovo_map_check.py --model D:/chevir/runs/slovo_pre/train/slovo_pre/best.pt
      --words200 <01 çıxışı>/landmarks/words200 --splits <03 çıxışı>/splits [--split train] [--out map_check.json]
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from groups import spelling  # noqa: E402
from azsl import features as FT  # noqa: E402
from azsl.store import LandmarkStore  # noqa: E402
from eval_jestdili import load_word_model  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--words200', required=True)
    ap.add_argument('--splits', required=True)
    ap.add_argument('--map', default=os.path.join(HERE, 'slovo_azsld.csv'))
    ap.add_argument('--split', default='train')
    ap.add_argument('--out')
    a = ap.parse_args()
    sp = spelling()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net, cfg, vocab = load_word_model(a.model, dev)
    vi = {g: i for i, g in enumerate(vocab)}
    mp = list(csv.DictReader(open(a.map, encoding='utf-8')))
    mapped = {r['azsld'] for r in mp} & set(vi)
    w = pd.read_csv(os.path.join(a.splits, 'words200_split.csv'), dtype={'key': str})
    w['g'] = w.gloss.map(lambda g: sp.get(g, g))
    store = LandmarkStore(a.words200)
    w = w[w.g.isin(mapped) & (w.split == a.split) & w.key.isin(store.index.index)]
    res = collections.defaultdict(lambda: [0, 0, 0, collections.Counter()])
    for r in w.itertuples():
        raw = FT.reduce(store.get(r.key))
        if raw['hand'].shape[0] < 2:
            continue
        aspect = float(store.index.at[r.key, 'width']) / float(store.index.at[r.key, 'height'])
        x = torch.from_numpy(FT.preprocess(raw, aspect, cfg['groups'], face=False)).float()
        with torch.no_grad():
            top = np.argsort(-net(x[None].to(dev))[0].float().cpu().numpy())[:5]
        s = res[r.g]
        s[0] += 1
        s[1] += int(top[0] == vi[r.g])
        s[2] += int(vi[r.g] in top)
        s[3][vocab[top[0]]] += 1
    n = sum(v[0] for v in res.values())
    print(f'{len(res)} mapped signs, {n} {a.split} clips: top-1 {sum(v[1] for v in res.values()) / n:.3f}, '
          f'top-5 {sum(v[2] for v in res.values()) / n:.3f}')
    out = []
    for g, (k, t1, t5, c) in sorted(res.items(), key=lambda kv: kv[1][2] / kv[1][0]):
        ru = '/'.join(r['slovo'] for r in mp if r['azsld'] == g)
        print(f'{g:14s} {ru:28s} n={k:4d} top1 {t1 / k:.2f} top5 {t5 / k:.2f}  most: {c.most_common(2)}')
        out.append({'azsld': g, 'slovo': ru, 'n': k, 'top1': round(t1 / k, 3), 'top5': round(t5 / k, 3),
                    'most': c.most_common(3)})
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
