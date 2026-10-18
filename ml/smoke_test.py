"""End-to-end check of the pipeline on a handful of real AzSLD files, read over HTTP (~30 MB).

Run before spending Kaggle hours:
    python ml/smoke_test.py [out_dir]
Needs: mediapipe, opencv-python, numpy, pandas, scikit-learn.
"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from azsl import pipeline, store  # noqa: E402
from azsl.landmarks import FIELDS  # noqa: E402


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix='azsl_smoke_')
    models = os.path.join(tempfile.gettempdir(), 'azsl_models')
    print('output:', out)
    t0 = time.time()
    paths = pipeline.setup_models(models, sface=True)
    print('models ok', round(time.time() - t0, 1), 's')

    pipeline.run_index(out, folders=['1', '2'])
    w = pipeline.run_words200(out, paths, workers=2, limit=3)
    f = pipeline.run_fingerspelling(out, paths, workers=2, limit=4)
    s = pipeline.run_sentences(out, paths, shard=0, num_shards=200, workers=2, limit=3)
    assert w['done'] == 3 and f['done'] == 4 and s['done'] == 3, (w, f, s)

    lm = os.path.join(out, 'landmarks')
    for sub in ('words200', 'fingerspelling', os.path.join('sentences_cam2', 'shard00')):
        st = store.LandmarkStore(os.path.join(lm, sub))
        item = st.get(st.manifest.id.iloc[0])
        for k, shape in FIELDS.items():
            assert item[k].shape[1:] == shape, (sub, k, item[k].shape)
        cols = ['id', 'n_frames', 'hand_rate', 'two_hand_rate', 'pose_rate', 'face_rate', 'frontal']
        print(sub, '\n', st.manifest[cols].to_string(index=False))
    ids, emb, thumbs = store.LandmarkStore(os.path.join(lm, 'sentences_cam2', 'shard00')).embeddings()
    print('embeddings', emb.shape, 'thumb bytes', [len(t) for t in thumbs])

    bench, signers, _ = pipeline.run_split(
        os.path.join(out, 'index'), [os.path.join(lm, 'words200')], [os.path.join(lm, 'fingerspelling')],
        [os.path.join(lm, 'sentences_cam2', 'shard00')], os.path.join(out, 'splits'), trials=50)
    print(signers.head().to_string())
    print(json.dumps(bench['signers'], default=str))
    print(f'smoke test passed in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
