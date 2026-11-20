"""
Ön öyrətmə (roadmap R2): söz modelinin şəbəkəsi (train.Net, AzSLD modellərinin xüsusiyyətləri ilə) Slovo-nun rus
işarələrini tanımağı öyrənir; sonra AzSLD söz və cümlə modelləri bu çəkilərdən başlayır (train.fit / continuous.fit
cfg['init'], notebook 06 INIT_RUN). AzSL işarələrinin çoxu RSL ilə ortaqdır; 194 adam AzSLD-nin 27-sindən çoxdur.

  D:/chevir/venv/Scripts/python ml/pretrain_slovo.py --landmarks D:/chevir/landmarks --out D:/chevir/runs/slovo_pre
      [--epochs 40] [--workers 6] [--map ml/slovo_azsld.csv]
  → <out>/train/slovo_pre/best.pt (+ manifest.json: notebook 06-nın INIT_RUN axtarışı üçün)
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from azsl import extra as EX, train as TR  # noqa: E402

NO_FACE = ['hands', 'hand_shape', 'hand_vel', 'flags', 'pose']     # AzSLD modellərinin xüsusiyyətləri (face_drop 1)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--landmarks', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--name', default='slovo_pre')
    ap.add_argument('--map', type=Path, default=None,
                    help='CSV slovo,azsld: ortaq işarələr AzSLD etiketi ilə öyrənilir (söz və cümlə modelinin başının '
                         'həmin sətirləri də köçür); qaralama, doğma istifadəçi təsdiqləməyib')
    args = ap.parse_args()
    rename = None
    if args.map:
        with open(args.map, encoding='utf-8') as f:
            rename = {r['slovo']: r['azsld'] for r in csv.DictReader(f)}
    banks, names = EX.load_slovo(str(args.landmarks / 'slovo'), rename=rename)
    cfg = dict(epochs=args.epochs, batch=64, groups=NO_FACE, face_drop=1.0, side=False, unknown=0.0, seed=1,
               workers=args.workers)
    out = args.out / 'train' / args.name
    res = TR.fit(cfg, {'train': banks['train'], 'val': banks['val']}, {}, names, str(out))
    man = {'run': args.name, 'kind': 'pretrain', 'source': 'Slovo (CC BY-SA 4.0)', 'labels': len(names),
           'map': str(args.map) if args.map else None,
           'best_epoch': res.get('best_epoch') if isinstance(res, dict) else None,
           'cfg': {k: v for k, v in cfg.items()}}
    with open(out / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    print('done:', out)


if __name__ == '__main__':
    main()
