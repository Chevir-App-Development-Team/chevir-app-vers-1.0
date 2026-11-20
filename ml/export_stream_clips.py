"""Landmark yazıları → brauzerin canlı yolunu Node-da təkrarlamaq üçün (tools/sim_danis.mjs) bir .bin + .json.

Hər yazı üçün features.reduce-un sahələri (hand, hand_world, pose, face, blend) float32, ardıcıl; json-da
hər sahənin offset-i (float32 sayı ilə), T, aspect, mətn. Çıxış repodan kənarda qalır (video nöqtələri).

  D:/chevir/venv/Scripts/python ml/export_stream_clips.py --landmarks D:/chevir/landmarks/jestdili_sentences \
      --out D:/chevir/eval/danis_sim/jestdili [--limit 0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from azsl import features as FT  # noqa: E402
from azsl.store import LandmarkStore  # noqa: E402

FIELDS = ['hand', 'hand_world', 'pose', 'face', 'blend']


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--landmarks', required=True)
    ap.add_argument('--out', type=Path, required=True, help='çıxış adı, .json və .bin əlavə olunur')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    store = LandmarkStore(args.landmarks)
    clips, offset = [], 0
    with open(args.out.with_suffix('.bin'), 'wb') as f:
        for n, (item_id, r) in enumerate(store.index.iterrows()):
            if args.limit and n >= args.limit:
                break
            raw = FT.reduce(store.get(item_id))
            T = int(raw['hand'].shape[0])
            c = {'id': item_id, 'text': str(r.get('text', '')), 'T': T,
                 'aspect': float(r['width']) / float(r['height'])}
            for k in FIELDS:
                a = np.ascontiguousarray(raw[k], dtype=np.float32).reshape(T, -1)
                c[k] = {'offset': offset, 'per': int(a.shape[1])}
                f.write(a.tobytes())
                offset += a.size
            clips.append(c)
    args.out.with_suffix('.json').write_text(json.dumps({'source': str(args.landmarks), 'clips': clips},
                                                        ensure_ascii=False), encoding='utf-8')
    print(f'{len(clips)} yazı, {offset * 4 / 1e6:.0f} MB → {args.out}.bin')


if __name__ == '__main__':
    main()
