"""Install a trained word model for the browser: manifest.json + model.onnx + a torch reference.

    python ml/make_manifest.py ml/results/run4d_model/no_face            # manifest.json in the run dir
    python ml/make_manifest.py ml/results/run4d_model/no_face --install  # + copy to the browser assets
    python ml/make_manifest.py <run_dir> --export --install                # re-export model.onnx first

Runs 1–4 predate the deployment contract and have no manifest.json; it is rebuilt here with the same
write_manifest() the trainer uses (cfg and metrics from metrics.json, labels from vocab.json).
From run 5 on, fit() writes it itself and only the reference step is new.

Runs 1–5 were exported while the attention layer fixed the clip length to the 32 traced frames (any
other length fails in onnxruntime); --export redoes model.onnx from best.pt with the current export.

Reference: best.pt (torch) on the synthetic cases of the golden fixture, run through
features.preprocess with the model's own groups → ml/fixtures/word_model_ref.json. The browser side
(tools/verify_recognizer.mjs) feeds the same cases to the JS preprocess + onnxruntime-web and must
reproduce the logits, so torch → ONNX → WASM is checked end to end, without anyone's video.
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
from azsl import train as TR  # noqa: E402
from make_fixture import KEYS  # noqa: E402

ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, 'public', 'tercume', 'assets', 'word_model')
FIXTURE = os.path.join(HERE, 'fixtures', f'features_v{FT.FEATURE_VERSION}')
REF = os.path.join(HERE, 'fixtures', 'word_model_ref.json')


def md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def split_id():
    """As notebook 04 records it: sha1 of words200_split.csv, first 10 hex digits."""
    with open(os.path.join(HERE, 'splits', 'words200_split.csv'), 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()[:10]


def fixture_cases():
    with open(FIXTURE + '.json', encoding='utf-8') as f:
        meta = json.load(f)
    blob = np.fromfile(FIXTURE + '.bin', dtype='<f4')
    for c in meta['cases']:
        raw = {k: blob[c[k]['offset']:c[k]['offset'] + int(np.prod(c[k]['shape']))].reshape(c[k]['shape'])
               for k in KEYS}
        yield c, raw


def load_net(run_dir, manifest):
    ck = torch.load(os.path.join(run_dir, 'best.pt'), map_location='cpu', weights_only=True)
    cfg = {**TR.DEFAULTS, **ck['cfg']}
    # runs 1–4 have no stamp: their reduce/to_square/features are the code that became version 1
    assert ck.get('feature_version', 1) == FT.FEATURE_VERSION, 'checkpoint from another feature version'
    net = TR.Net(manifest['input']['n_features'], len(manifest['labels']), cfg['dim'], cfg['stages'],
                 cfg['convs'], cfg['kernel'], cfg['heads'], cfg['drop'], cfg['head_drop'])
    net.load_state_dict(ck['model'])
    return net.eval()


def reference(net, manifest):
    """Logits of the training/eval path (masked attention), which the ONNX export has to reproduce."""
    groups = manifest['input']['groups']
    cases = []
    for c, raw in fixture_cases():
        if c['groups'] is not None:          # the fixture's own group cases test the port, not the model
            continue
        for face in (True, False):
            x = torch.from_numpy(FT.preprocess(raw, c['aspect'], groups, face).astype(np.float32))[None]
            with torch.no_grad():
                logits = net(x, torch.ones(x.shape[:2], dtype=torch.bool))[0].numpy()
            cases.append({'name': c['name'], 'face': face, 'T': int(x.shape[1]),
                          'top5': [int(i) for i in np.argsort(-logits)[:5]],
                          'logits': [float(v) for v in logits]})
    return cases


def add_segment(path, manifest, summary_path):
    """Word-mode settings depend on the model (with an unknown class a lower threshold wins), so the
    val choice of notebook 05 travels in the manifest; the browser applies it if the version matches."""
    from azsl import segment as SG
    with open(summary_path, encoding='utf-8') as f:
        s = json.load(f)
    chosen = s.get('chosen_model', s.get('model'))
    if chosen != manifest['run']:
        sys.exit(f'{summary_path} chose {chosen}, not {manifest["run"]}')
    c = s['configs']['chosen on val (F1)']
    keep = ('precision', 'recall', 'f1', 'wer', 'f1_merged')
    manifest['segment'] = {'version': s['segment_version'], 'params': c['params'],
                           'chosen_on': 'val sentence recordings (notebook 05)',
                           'val': {k: round(c['val'][k], 4) for k in keep},
                           'test': {k: round(c['test'][k], 4) for k in keep}}
    assert s['segment_version'] == SG.SEGMENT_VERSION, 'settings from another segmentation version'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f'segment settings from {summary_path}: {c["params"]["lens"]}, min_p {c["params"]["min_p"]}, '
          f'max_overlap {c["params"]["max_overlap"]} (test F1 {c["test"]["f1"]:.3f})')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', help='a run folder with metrics.json, vocab.json, best.pt, model.onnx')
    ap.add_argument('--export', action='store_true',
                    help='re-export model.onnx from best.pt (runs 1–5: exported before T was dynamic)')
    ap.add_argument('--install', action='store_true', help=f'copy model + manifest to {ASSETS}')
    ap.add_argument('--segment', help="notebook 05 summary.json: the word-mode settings chosen on val for "
                                      "this model go into the manifest (the browser uses them)")
    a = ap.parse_args()

    path = os.path.join(a.run_dir, 'manifest.json')
    if not os.path.exists(path):
        with open(os.path.join(a.run_dir, 'metrics.json'), encoding='utf-8') as f:
            result = json.load(f)
        with open(os.path.join(a.run_dir, 'vocab.json'), encoding='utf-8') as f:
            vocab = json.load(f)
        cfg = {**TR.DEFAULTS, **result['cfg'], 'split_id': result['cfg'].get('split_id') or split_id()}
        n_in = len(FT.columns(cfg['groups']))
        assert n_in == result['n_features'], (n_in, result['n_features'])
        TR.write_manifest(a.run_dir, cfg, vocab, result, n_in,
                          onnx_ok=os.path.exists(os.path.join(a.run_dir, 'model.onnx')))
        print('wrote', path)
    with open(path, encoding='utf-8') as f:
        manifest = json.load(f)
    net = load_net(a.run_dir, manifest)

    onnx = os.path.join(a.run_dir, manifest['model_file'] or 'model.onnx')
    if a.export:
        if not TR.export_onnx(net, onnx, n_in=manifest['input']['n_features']):
            sys.exit('export failed — nothing installed')
        net = load_net(a.run_dir, manifest)       # export_onnx() moved it to float/cpu/eval; reload clean
        if manifest['model_file'] is None:
            manifest['model_file'] = 'model.onnx'
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, ensure_ascii=False, indent=1)
    if a.segment:
        add_segment(path, manifest, a.segment)
    ref = {'run': manifest['run'], 'model_md5': md5(onnx), 'feature_version': FT.FEATURE_VERSION,
           'tolerance': 1e-3, 'cases': reference(net, manifest)}
    with open(REF, 'w', encoding='utf-8') as f:
        json.dump(ref, f, indent=1)
    print(f'wrote {REF}: {len(ref["cases"])} cases, top-1 ' +
          ', '.join(f'{c["name"]}{"" if c["face"] else " (no face)"} → {manifest["labels"][c["top5"][0]]}'
                    for c in ref['cases']))

    if a.install:
        os.makedirs(ASSETS, exist_ok=True)
        for f in ('manifest.json', manifest['model_file']):
            shutil.copy2(os.path.join(a.run_dir, f), os.path.join(ASSETS, f))
        print(f'installed {manifest["run"]} → {ASSETS} '
              f'({os.path.getsize(onnx) / 1e6:.1f} MB, {len(manifest["labels"])} labels, '
              f'{manifest["input"]["n_features"]} features, works without face: {manifest["works_without_face"]})')


if __name__ == '__main__':
    main()
