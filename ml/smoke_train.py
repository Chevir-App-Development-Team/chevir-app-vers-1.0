"""Training smoke test on real AzSLD landmarks (~25 MB over HTTP): extracts a few sentence recordings
and Words_200 clips of six glosses, makes a toy split, trains 2 epochs on CPU, exports ONNX.

    python ml/smoke_train.py [out_dir]
Needs everything smoke_test.py needs, plus torch.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from azsl import pipeline, sources, store, zenodo  # noqa: E402

GLOSSES = ['SİZ', 'AD', 'NƏDİR', 'MƏN', 'BAKI', 'YAŞAMAQ']   # the signs of sentences 1 and 2


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix='azsl_smoke_train_')
    print('output:', out)
    paths = pipeline.setup_models(os.path.join(tempfile.gettempdir(), 'azsl_models'), sface=True)
    tags, recs, _ = pipeline.run_index(out, folders=['1', '2'])

    members = zenodo.list_members(zenodo.archive_url('sentences'))
    cam2 = sources.sentence_recordings(members)
    pick = [m for f in ('1', '2') for m in [x for x in cam2 if x.name.split('/')[1] == f][:5]]
    sdir = os.path.join(out, 'landmarks', 'sentences_cam2', 'shard00')
    pipeline.run_pool([(sources.rec_id(m.name), m, {'rec_id': sources.rec_id(m.name), 'camera': 'Cam2', 'bytes': m.size})
                       for m in pick], pipeline._video_job, store.PartWriter(sdir, embeddings=True),
                      (paths, True, True, None), workers=4, label='sentences')

    wmembers = zenodo.list_members(zenodo.archive_url('words200'))
    wl = sources.words_listing([m.name for m in wmembers])
    wl = wl[wl.gloss.isin(GLOSSES)].groupby('gloss').head(4)
    by_name = {m.name: m for m in wmembers}
    wdir = os.path.join(out, 'landmarks', 'words200')
    pipeline.run_pool([(r.key, by_name[r.member], {'key': r.key, 'gloss': r.gloss}) for r in wl.itertuples()],
                      pipeline._video_job, store.PartWriter(wdir), (paths, True, False, None), workers=4, label='words')

    # toy split: recordings round-robin train/train/train/val/test; Words_200 clips the same way
    done = set(store.LandmarkStore(sdir).manifest.id)
    order = ['train', 'train', 'train', 'val', 'test']
    rsplit = {r: order[i % 5] for i, r in enumerate(sorted(done))}
    t = tags[~tags.dup & tags.rec_id.isin(done)].copy()
    t['split'], t['in_w200'], t['has_landmarks'] = t.rec_id.map(rsplit), t.gloss.isin(GLOSSES), True
    w = store.LandmarkStore(wdir).manifest[['id', 'gloss']].rename(columns={'id': 'key'})
    w['split'] = [order[i % 5] for i in range(len(w))]

    from azsl import train as T
    vocab_list = sorted(GLOSSES)
    vocab = {g: i for i, g in enumerate(vocab_list)}
    signs = T.load_sentence_signs([sdir], t[t.in_w200], vocab, margin=8)
    words = T.load_words200([wdir], w, vocab)
    base = dict(epochs=2, batch=8, dim=64, workers=0)
    hands = ['hands', 'hand_shape', 'hand_vel', 'flags']
    rows = T.fit_many({'full': base, 'hands_only': {**base, 'groups': hands},
                       'face_drop': {**base, 'face_drop': 0.5}}, signs, words, vocab_list,
                      os.path.join(out, 'train'))
    print(pd.DataFrame(rows).to_string())
    assert len(rows) == 3 and all('error' not in r for r in rows), rows
    man = json.load(open(os.path.join(out, 'train', 'face_drop', 'manifest.json'), encoding='utf-8'))
    assert man['feature_version'] == T.FT.FEATURE_VERSION and man['works_without_face']
    assert 'test_words_noface' in man['metrics'] and man['input']['n_features'] == T.FT.N_FEAT
    assert 'test_words_noface' not in json.load(open(os.path.join(out, 'train', 'hands_only', 'metrics.json'),
                                                     encoding='utf-8'))
    assert rows[1]['n_features'] == len(T.FT.columns(hands)) < T.FT.N_FEAT
    pz = np.load(os.path.join(out, 'train', 'hands_only', 'predictions.npz'))
    assert pz['test_words_probs'].shape == (len(words['test']), len(vocab_list)), pz['test_words_probs'].shape
    res = json.load(open(os.path.join(out, 'train', 'full', 'metrics.json'), encoding='utf-8'))

    onnx_path = os.path.join(out, 'train', 'full', 'model.onnx')
    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            import torch
            ck = torch.load(os.path.join(out, 'train', 'full', 'best.pt'), map_location='cpu')
            cfg = ck['cfg']
            net = T.Net(T.FT.N_FEAT, len(vocab_list), cfg['dim'], cfg['stages'], cfg['convs'], cfg['kernel'],
                        cfg['heads'], cfg['drop'], cfg['head_drop'])
            net.load_state_dict(ck['model'])
            net.eval()
            x = torch.from_numpy(T.SignDataset(words['test'])[0][0][None].numpy())
            ref = T.Exportable(net)(x).detach().numpy()
            got = ort.InferenceSession(onnx_path).run(None, {'x': x.numpy()})[0]
            print('ONNX vs torch max abs diff', float(np.abs(ref - got).max()), 'for T =', x.shape[1])
        except ImportError:
            print('onnxruntime not installed — ONNX file written but not compared')
    print(json.dumps({k: res[k] for k in ('best_epoch', 'test_words', 'test_signs') if k in res}, indent=1))
    print('training smoke test passed')


if __name__ == '__main__':
    main()
