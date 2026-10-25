"""Install a letter model (notebook 07) for the browser's letter mode.

    python ml/install_letters.py ml/results/letters1/hand              # reference only
    python ml/install_letters.py ml/results/letters1/hand --install    # + copy to the browser assets

- Reference: best.pt (torch) on the cases of the golden fixture (ml/fixtures/letters_v<N>) →
  ml/fixtures/letters_model_ref.json. tools/verify_letters.mjs runs letters.js + onnxruntime-web on the same
  cases and must reproduce the probabilities.
- The old letter model's scores on the same test samples (letters/old_model/metrics.json next to the run, if
  there) go into the manifest as `old_model`, for the app's model page.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from azsl import features as FT  # noqa: E402
from azsl import letters as L  # noqa: E402
from azsl import train as TR  # noqa: E402

ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, 'public', 'tercume', 'assets', 'letter_model')
FIXTURE = os.path.join(HERE, 'fixtures', f'letters_v{L.LETTERS_VERSION}')
REF = os.path.join(HERE, 'fixtures', 'letters_model_ref.json')
KEYS = ('hand', 'hand_world', 'pose', 'face', 'blend')


def load_net(run_dir):
    ck = torch.load(os.path.join(run_dir, 'best.pt'), map_location='cpu', weights_only=False)
    cfg = {**L.DEFAULTS, **ck['cfg']}
    assert ck.get('feature_version') == FT.FEATURE_VERSION, 'checkpoint from another feature version'
    assert ck.get('letters_version') == L.LETTERS_VERSION, 'checkpoint from another letters version'
    net = TR.Net(len(FT.columns(cfg['groups'])), len(ck['labels']), cfg['dim'], cfg['stages'], cfg['convs'],
                 cfg['kernel'], cfg['heads'], cfg['drop'], cfg['head_drop'])
    net.load_state_dict(ck['model'])
    return net.eval(), cfg


def reference(net, groups):
    with open(FIXTURE + '.json', encoding='utf-8') as f:
        meta = json.load(f)
    blob = np.fromfile(FIXTURE + '.bin', dtype='<f4')
    out = []
    for c in meta['cases']:
        raw = {k: blob[c[k]['offset']:c[k]['offset'] + int(np.prod(c[k]['shape']))].reshape(c[k]['shape']) for k in KEYS}
        x = torch.from_numpy(L.preprocess(raw, c['aspect'], groups))[None]
        with torch.no_grad():
            p = net(x, torch.ones(x.shape[:2], dtype=torch.bool)).softmax(1)[0].numpy()
        out.append({'case': c['name'], 'probs': [float(v) for v in p]})
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', help='a notebook 07 run folder: manifest.json, best.pt, model.onnx')
    ap.add_argument('--install', action='store_true', help=f'copy model + manifest to {ASSETS}')
    a = ap.parse_args()
    with open(os.path.join(a.run_dir, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    assert manifest.get('kind') == 'letters', 'not a letter model'
    old = os.path.join(os.path.dirname(os.path.normpath(a.run_dir)), 'old_model', 'metrics.json')
    if os.path.exists(old):
        with open(old, encoding='utf-8') as f:
            manifest['old_model'] = json.load(f)
    pred = os.path.join(a.run_dir, 'predictions_test.npz')
    if os.path.exists(pred):                       # per-letter test recall (the argmax among the letters)
        d = np.load(pred)
        p, y = d['probs'].copy(), d['labels']
        none = L.LABELS.index(L.NONE)
        p[:, none] = -1
        manifest['test_per_letter'] = {L.LABELS[c]: {'recall': float((p[y == c].argmax(1) == c).mean()),
                                                     'n': int((y == c).sum())}
                                       for c in np.unique(y) if c != none}
    with open(os.path.join(a.run_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    net, cfg = load_net(a.run_dir)
    with open(os.path.join(a.run_dir, manifest['model_file']), 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    ref = {'run': manifest['run'], 'model_md5': md5, 'letters_version': L.LETTERS_VERSION, 'tolerance': 1e-4,
           'cases': reference(net, cfg['groups'])}
    with open(REF, 'w', encoding='utf-8') as f:
        json.dump(ref, f)
    print(f'wrote {REF}: {len(ref["cases"])} cases')
    if a.install:
        os.makedirs(ASSETS, exist_ok=True)
        for f in ('manifest.json', manifest['model_file']):
            shutil.copy2(os.path.join(a.run_dir, f), os.path.join(ASSETS, f))
        size = os.path.getsize(os.path.join(ASSETS, manifest['model_file'])) / 1e6
        print(f'installed {manifest["run"]} → {ASSETS} ({size:.1f} MB, {len(manifest["labels"])} labels, '
              f'groups {manifest["input"]["groups"]})')


if __name__ == '__main__':
    main()
