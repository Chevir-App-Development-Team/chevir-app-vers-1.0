"""
Modellər AzSLD-də olmayan bir adamda və kamerada (JestDili.az, roadmap V4): söz modeli lüğətin sözlərində, cümlə
modeli 500 cümlədə (işarə dili daşıyıcılarının təbii göstərişi, mətni ilə; glos yoxdur).

  D:/chevir/venv/Scripts/python ml/eval_jestdili.py --landmarks D:/chevir/landmarks --out D:/chevir/eval/<ad> \
      [--word ml/results/run11_model/final_side_noface/best.pt] [--cont ml/results/cont_v15/final_merged/best.pt]

Söz: JestDili.az-ın sözü modelin sözüdürsə (etiketin yazılışı = sözün yazılışı, ml/label_text.csv ilə), klip bütöv
modelə verilir: top-1, top-5. Cümlə: brauzerdəki kimi axınla (pəncərə 96, addım 8, gecikmə 24) və modelin öz
deşifrəsi ilə tanınan işarələr → <out>/sentences.json; mətnlə müqayisə tools/score_jestdili.mjs-də (kök üzrə).
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'ml'))
from azsl import continuous as CT, features as FT, train as TR  # noqa: E402
from azsl.store import LandmarkStore  # noqa: E402


def az_lower(s):
    return s.replace('I', 'ı').replace('İ', 'i').lower()


def label_texts(vocab):
    """Model etiketi → kiçik hərfli yazılış (annotasiyanın yazı xətaları düzəldilmiş)."""
    fix = {}
    with open(ROOT / 'ml' / 'label_text.csv', encoding='utf8') as f:
        for r in csv.DictReader(f):
            fix[r['label']] = r['text']
    return {i: az_lower(fix.get(g, g)) for i, g in enumerate(vocab)}


def load_word_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = {**TR.DEFAULTS, **ck['cfg']} if hasattr(TR, 'DEFAULTS') else ck['cfg']
    net = TR.Net(len(FT.columns(cfg['groups'])), len(ck['vocab']), cfg.get('dim', 192), cfg.get('stages', 2),
                 cfg.get('convs', 3), cfg.get('kernel', 17), cfg.get('heads', 4), cfg.get('drop', 0.2),
                 cfg.get('head_drop', 0.4)).to(device)
    net.load_state_dict(ck['model'])
    return net.eval(), cfg, list(ck['vocab'])


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--landmarks', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--word', default=str(ROOT / 'ml/results/run11_model/final_side_noface/best.pt'))
    ap.add_argument('--cont', default=str(ROOT / 'ml/results/cont_v15/final_merged/best.pt'))
    ap.add_argument('--decode', default='', help='JSON, manifestdəki deşifrə əvəzinə (müqayisə üçün)')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    report = {}

    # 1. söz modeli
    if args.word and (args.landmarks / 'jestdili_words').exists():
        net, cfg, vocab = load_word_model(args.word, dev)
        texts = label_texts(vocab)
        by_text = {}
        for i, t in texts.items():
            by_text.setdefault(t, i)
        store = LandmarkStore(str(args.landmarks / 'jestdili_words'))
        rows = []
        for item_id, r in store.index.iterrows():
            name = az_lower(str(r['name'])).split('(')[0].strip()
            if name not in by_text:
                continue
            raw = FT.reduce(store.get(item_id))
            aspect = float(r['width']) / float(r['height'])
            x = torch.from_numpy(FT.preprocess(raw, aspect, cfg['groups'], face=cfg.get('face_drop', 0) < 1)).float()
            with torch.no_grad():
                logits = net(x[None].to(dev))[0].float().cpu().numpy()
            top = np.argsort(-logits)[:5]
            rows.append({'id': item_id, 'word': name, 'label': vocab[by_text[name]], 'pred': [vocab[k] for k in top],
                         'top1': bool(top[0] == by_text[name]), 'top5': bool(by_text[name] in top),
                         'hand_rate': float(r['hand_rate'])})
        n = len(rows)
        report['words'] = {'model': args.word, 'clips': n, 'top1': round(sum(r['top1'] for r in rows) / max(1, n), 4),
                           'top5': round(sum(r['top5'] for r in rows) / max(1, n), 4)}
        with open(args.out / 'words.json', 'w', encoding='utf8') as f:
            json.dump(rows, f, ensure_ascii=False, indent=0)
        print('words:', report['words'])

    # 2. cümlə modeli, brauzerdəki kimi
    if args.cont and (args.landmarks / 'jestdili_sentences').exists():
        net, cfg, vocab, _ = CT.load_checkpoint(args.cont, dev)
        man = json.loads((Path(args.cont).parent / 'manifest.json').read_text(encoding='utf8'))
        decode_p = json.loads(args.decode) if args.decode else (man.get('decode') or cfg.get('decode') or CT.DECODE)
        texts = label_texts(vocab)
        unk = vocab.index('<unk>') if '<unk>' in vocab else None
        short = set(CT.short_classes(vocab))
        store = LandmarkStore(str(args.landmarks / 'jestdili_sentences'))
        out = []
        for item_id, r in store.index.iterrows():
            raw = FT.reduce(store.get(item_id))
            aspect = float(r['width']) / float(r['height'])
            P = CT.stream_probs(net, raw, aspect, cfg['groups'], cfg.get('face_drop', 1.0) < 1, dev, **CT.STREAM)
            dets = CT.decode(P, decode_p, unknown=unk, short=short)
            out.append({'id': item_id, 'text': r['text'], 'frames': int(r['n_frames']),
                        'signs': [vocab[c] for _, _, c, _ in dets], 'words': [texts[c] for _, _, c, _ in dets]})
        report['sentences'] = {'model': args.cont, 'sentences': len(out), 'decode': decode_p,
                               'signs_per_sentence': round(sum(len(o['signs']) for o in out) / max(1, len(out)), 2)}
        with open(args.out / 'sentences.json', 'w', encoding='utf8') as f:
            json.dump(out, f, ensure_ascii=False, indent=0)
        print('sentences:', report['sentences'])
    with open(args.out / 'report.json', 'w', encoding='utf8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
