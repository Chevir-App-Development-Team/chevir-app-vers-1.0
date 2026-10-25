"""Install a continuous-recognition model (notebook 06) for the browser's sentence mode.

    python ml/install_continuous.py ml/results/cont2/seq_splice            # reference + sentences.json
    python ml/install_continuous.py ml/results/cont2/seq_splice --install  # + copy to the browser assets
    python ml/install_continuous.py ml/results/cont_final/final_splice --eval ml/results/cont_v7/final_splice --install

- --eval: an evaluation-only run of the same model (notebook 06, ckpt=...) re-chose its decoding: its
  decode and test scores replace the manifest's (the run folder's manifest.json is updated).

- Reference: best.pt (torch, the training forward pass) on the windows of the golden fixture
  (ml/fixtures/continuous_v<N>) → ml/fixtures/cont_model_ref.json. tools/verify_cont_model.mjs feeds the
  same windows to features.js + onnxruntime-web and must reproduce them.
- sentences.json: the 492 AzSLD sentences, each with its text and its sign sequence (the most common
  annotated gloss sequence of the sentence's recordings, with the model's spelling and same-sign merges).
  The browser shows the nearest one under the recognized words.
"""
import argparse
import collections
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from azsl import continuous as CT  # noqa: E402
from azsl import features as FT  # noqa: E402
from azsl import train as TR  # noqa: E402
import build_notebooks as B  # noqa: E402

ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, 'public', 'tercume', 'assets', 'cont_model')
FIXTURE = os.path.join(HERE, 'fixtures', f'continuous_v{CT.CONTINUOUS_VERSION}')
REF = os.path.join(HERE, 'fixtures', 'cont_model_ref.json')
KEYS = ('hand', 'hand_world', 'pose', 'face', 'blend')


def load_net(run_dir, manifest):
    ck = torch.load(os.path.join(run_dir, 'best.pt'), map_location='cpu', weights_only=True)
    cfg = {**CT.DEFAULTS, **ck['cfg']}
    assert ck.get('feature_version') == FT.FEATURE_VERSION, 'checkpoint from another feature version'
    net = CT.SeqNet(manifest['input']['n_features'], len(manifest['labels']), cfg['dim'], cfg['stages'],
                    cfg['convs'], cfg['kernel'], cfg['heads'], cfg['drop'], cfg['head_drop'])
    net.load_state_dict(ck['model'])
    return net.eval()


def reference(net, manifest, every=8):
    """Per-frame logits of the fixture's windows (every `every`-th frame kept in full, argmax of all)."""
    with open(FIXTURE + '.json', encoding='utf-8') as f:
        meta = json.load(f)
    blob = np.fromfile(FIXTURE + '.bin', dtype='<f4')
    face = not manifest['works_without_face']
    out = []
    for i, c in enumerate(meta['features']):
        raw = {k: blob[c[k]['offset']:c[k]['offset'] + int(np.prod(c[k]['shape']))].reshape(c[k]['shape']) for k in KEYS}
        x = torch.from_numpy(FT.preprocess(raw, c['aspect'], manifest['input']['groups'], face, max_len=None))[None]
        with torch.no_grad():
            lg = net(x, torch.ones(x.shape[:2], dtype=torch.bool))[0].numpy()
        out.append({'case': i, 'T': int(x.shape[1]), 'face': face, 'argmax': [int(v) for v in lg.argmax(1)],
                    'frames': list(range(0, len(lg), every)),
                    'logits': [[float(v) for v in lg[t]] for t in range(0, len(lg), every)]})
    return out


def sentences(manifest, splits=None):
    """Each AzSLD sentence: its text and most common annotated sign sequence, in the model's labels (the
    glosses it cannot output are left out). splits: only these signers' recordings (e.g. ('train',))."""
    sp, mg = B.spelling(), B.sign_merge()
    merge = lambda g: mg.get(sp.get(g, g), sp.get(g, g))      # noqa: E731
    words = set(manifest['labels']) - {manifest.get('blank_label'), manifest.get('unknown_label')}
    s = pd.read_csv(os.path.join(HERE, 'splits', 'sentences_split.csv'), dtype={'rec_id': str})
    if splits:
        s = s[s.split.isin(splits)]
    g = pd.read_csv(os.path.join(HERE, 'splits', 'glosses_split.csv'), dtype={'rec_id': str})
    g = g[g.start.notna()].sort_values(['rec_id', 'start'])
    seq = {rid: tuple(w for w in map(merge, t.gloss) if w in words) for rid, t in g.groupby('rec_id')}
    out = []
    for folder, t in s.groupby('folder'):
        seqs = collections.Counter(seq[r] for r in t.rec_id if r in seq and seq[r])
        if seqs:
            out.append({'folder': str(folder), 'text': str(t.text.iloc[0]), 'glosses': list(seqs.most_common(1)[0][0])})
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', help='a notebook 06 run folder: manifest.json, best.pt, model.onnx')
    ap.add_argument('--install', action='store_true', help=f'copy model + manifest + sentences.json to {ASSETS}')
    ap.add_argument('--eval', help='an evaluation-only notebook 06 run of this model: its decode and test scores')
    a = ap.parse_args()
    with open(os.path.join(a.run_dir, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    assert manifest.get('kind') == 'continuous', 'not a continuous model'
    if a.eval:
        with open(os.path.join(a.eval, 'metrics.json'), encoding='utf-8') as f:
            ev = json.load(f)
        assert ev['cfg'].get('ckpt') == manifest['run'], f'{a.eval} evaluated {ev["cfg"].get("ckpt")}, not {manifest["run"]}'
        manifest['decode'] = ev['decode']
        manifest['metrics'].update({k: v for k, v in ev.items() if k.startswith('test')})
        manifest['decode_source'] = {'eval': os.path.basename(os.path.normpath(a.eval)),
                                     'decode_from': ev['cfg'].get('decode_from')}
        manifest['continuous_version'] = CT.CONTINUOUS_VERSION
        with open(os.path.join(a.run_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
        print(f'decode {ev["decode"]} (chosen by {ev["cfg"].get("decode_from")}), test f1_all {ev["test"]["f1_all"]:.4f}')
    net = load_net(a.run_dir, manifest)
    onnx = os.path.join(a.run_dir, manifest['model_file'])
    with open(onnx, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    ref = {'run': manifest['run'], 'model_md5': md5, 'continuous_version': CT.CONTINUOUS_VERSION,
           'tolerance': 1e-3, 'cases': reference(net, manifest)}
    with open(REF, 'w', encoding='utf-8') as f:
        json.dump(ref, f)
    print(f'wrote {REF}: {len(ref["cases"])} windows')
    sent = sentences(manifest)
    with open(os.path.join(a.run_dir, 'sentences.json'), 'w', encoding='utf-8') as f:
        json.dump(sent, f, ensure_ascii=False, separators=(',', ':'))
    print(f'sentences.json: {len(sent)} sentences')
    if a.install:
        os.makedirs(ASSETS, exist_ok=True)
        for f in ('manifest.json', manifest['model_file'], 'sentences.json'):
            shutil.copy2(os.path.join(a.run_dir, f), os.path.join(ASSETS, f))
        print(f'installed {manifest["run"]} → {ASSETS} ({os.path.getsize(onnx) / 1e6:.1f} MB, '
              f'{len(manifest["labels"])} labels, decode {manifest["decode"]}, stream {manifest["stream"]})')


if __name__ == '__main__':
    main()
