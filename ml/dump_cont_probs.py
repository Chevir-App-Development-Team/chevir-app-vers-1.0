"""
Cümlə modelinin kadr ehtimalları, brauzerdəki kimi axınla (pəncərə 96, addım 8, gecikmə 24), diskə: deşifrənin
(decode, dil modeli ilə axtarış, E32) təcrübələri modeli hər dəfə işlətmədən aparılır.

  # AzSLD: notebook 06-nın bölgüsü (val / test yazıları, annotasiya ilə)
  D:/chevir/venv/Scripts/python ml/dump_cont_probs.py --ckpt <run>/best.pt --input D:/chevir/kaggle \
      --out D:/chevir/eval/e32/probs_<ad> --splits val,test
  # JestDili.az cümlələri (yalnız "holdout" yarısı, --sentences ilə)
  D:/chevir/venv/Scripts/python ml/dump_cont_probs.py --ckpt <run>/best.pt --jestdili D:/chevir/landmarks \
      --sentences D:/chevir/kaggle/jestdili_ctc_r4/jestdili_sentence_glosses_r4.json --out D:/chevir/eval/e32/probs_<ad>

Çıxış: <out>/<split>.bin (float16, yazılar ardıcıl, hər biri T × C) və <out>/<split>.json: etiketlər, deşifrə
(manifest), hər yazının id-si, T-si, ofseti, annotasiyası (start, end, gloss; 1-dən, daxil) və ya mətni.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from azsl import continuous as CT, features as FT, train as TR  # noqa: E402
from azsl.store import LandmarkStore  # noqa: E402


def find(base, pattern):
    return sorted({os.path.dirname(p) for p in glob.glob(os.path.join(base, '**', pattern), recursive=True)})


def write(out, name, items, labels, decode_p, extra):
    """items: [(meta dict, T×C probs)] → <name>.bin + <name>.json"""
    meta, off = [], 0
    with open(out / f'{name}.bin', 'wb') as f:
        for m, P in items:
            a = np.ascontiguousarray(P.astype(np.float16))
            f.write(a.tobytes())
            meta.append({**m, 'T': int(P.shape[0]), 'offset': off})
            off += a.size
    with open(out / f'{name}.json', 'w', encoding='utf-8') as f:
        json.dump({'labels': labels, 'decode': decode_p, 'dtype': 'float16', **extra, 'items': meta}, f,
                  ensure_ascii=False)
    print(f'{name}: {len(meta)} recordings, {off // len(labels):,} frames → {out / name}.bin')


def main():
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--input', default='', help='notebook 06-nın girişi (D:/chevir/kaggle): AzSLD yazıları')
    ap.add_argument('--splits', default='val,test')
    ap.add_argument('--jestdili', default='', help='landmarks qovluğu (jestdili_sentences/ içində)')
    ap.add_argument('--sentences', default='', help='JestDili.az: yalnız bu faylın "holdout" cümlələri')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, cfg, vocab, epoch = CT.load_checkpoint(args.ckpt, dev)
    man_path = Path(args.ckpt).parent / 'manifest.json'
    man = json.loads(man_path.read_text(encoding='utf8')) if man_path.exists() else {}
    decode_p = man.get('decode') or cfg.get('decode') or CT.DECODE
    face = cfg.get('face_drop', 1.0) < 1
    print('model:', args.ckpt, 'epoch', epoch, len(vocab), 'labels, decode', decode_p)

    if args.input:
        import pandas as pd
        sys.path.insert(0, str(ROOT))
        from groups import sign_merge, spelling
        MERGE, SPELLING = sign_merge(), spelling()
        merge = lambda g: MERGE.get(SPELLING.get(g, g), SPELLING.get(g, g))      # noqa: E731
        splits_dir = find(args.input, 'splits/sentences_split.csv')[0]
        sentence_dirs = find(args.input, 'landmarks/sentences_cam2/shard*/manifest.csv')
        tags = pd.read_csv(os.path.join(splits_dir, 'glosses_split.csv'), dtype={'key': str, 'rec_id': str})
        sents = pd.read_csv(os.path.join(splits_dir, 'sentences_split.csv'), dtype={'rec_id': str})
        tags = tags[tags.has_landmarks & tags.split.notna() & tags.start.notna() & tags.end.notna()].copy()
        tags['gloss'] = tags.gloss.map(merge)
        words = [g for g in vocab if g not in (CT.BLANK, TR.UNKNOWN)]
        index = {g: i for i, g in enumerate(vocab)}
        exclude = TR.ambiguous_oov(tags.gloss.unique(), set(words))
        anns = {rid: list(zip(t.start, t.end, t.gloss)) for rid, t in tags.groupby('rec_id')}
        want = args.splits.split(',')
        recs = sents[sents.split.isin(want) & sents.rec_id.isin(anns)][['rec_id', 'split', 'heldout_sentence']].copy()
        held = dict(zip(recs.rec_id, recs.heldout_sentence))
        banks = CT.load_recordings(sentence_dirs, recs, anns, index, unknown=TR.UNKNOWN, exclude=exclude)
        for split in want:
            bank, items = banks[split], []
            for i in range(len(bank)):
                raw, _, aspect, rid = bank.get(i)
                P = CT.stream_probs(net, raw, aspect, cfg['groups'], face, dev, **CT.STREAM)
                items.append(({'id': rid, 'heldout': bool(held.get(rid, False)),
                               'anns': [[int(s), int(e), g] for s, e, g in anns.get(rid, [])]}, P))
                if i % 300 == 0:
                    print(f'  {split} {i}/{len(bank)}')
            write(args.out, split, items, vocab, decode_p, {'exclude': sorted(exclude), 'ckpt': args.ckpt})

    if args.jestdili:
        store = LandmarkStore(str(Path(args.jestdili) / 'jestdili_sentences'))
        keep = set(json.loads(Path(args.sentences).read_text(encoding='utf8'))['holdout']) if args.sentences else None
        items = []
        for item_id, r in store.index.iterrows():
            if keep is not None and item_id not in keep:
                continue
            raw = FT.reduce(store.get(item_id))
            aspect = float(r['width']) / float(r['height'])
            P = CT.stream_probs(net, raw, aspect, cfg['groups'], face, dev, **CT.STREAM)
            items.append(({'id': item_id, 'text': r['text']}, P))
        write(args.out, 'jestdili', items, vocab, decode_p, {'ckpt': args.ckpt})


if __name__ == '__main__':
    main()
