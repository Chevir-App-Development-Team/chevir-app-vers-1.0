"""Letter models on another person's alphabet photos (JestDili.az, roadmap session 2, E30).

JestDili.az's alphabet: one photo per letter (a hand on white, moving letters drawn as two hands and an arrow),
another person than AzSLD's contributors. A photo becomes a still clip of PHOTO_LEN frames, as AzSLD's photos
in training (`letters.load_photos`), and goes through `letters.preprocess`.

Two steps, two environments (the landmarker and torch live in different venvs here):

  .venv/Scripts/python -I ml/eval_letters_photos.py extract --photos D:/chevir/jestdili_alphabet \\
      --models D:/chevir/landmarks/models --out D:/chevir/eval/letters_jestdili
  D:/chevir/venv/Scripts/python -I ml/eval_letters_photos.py score --out D:/chevir/eval/letters_jestdili \\
      --run final_hand=ml/results/letters2/final_hand --run final_hand_all=D:/chevir/runs/letters_all/letters/final_hand_all

  D:/chevir/venv/Scripts/python -I ml/eval_letters_photos.py export-sentences \
      --store D:/chevir/landmarks/jestdili_sentences --out D:/chevir/eval/letters_jestdili
  LETTERS_SPEEDS=1 node tools/letters_tune.mjs <letter model folder> D:/chevir/eval/letters_jestdili/sent '{}'

`extract` writes landmarks.npz (+ photos.csv: file, letter, hands found); `score` prints and writes scores.json:
top-1 / top-3 among the letters (a letter photo is never answered `<none>`), the full argmax (with `<none>`),
static and moving letters apart, the photos mirrored too (the same letter with the other hand), and the pairs
of runs (photos one gets right and the other not; exact sign test). A finer, paired measure on so few photos:
the probability given to the right letter, and `jittered` = the share right over JITTER copies of each photo
through the training augmentation (mirror, rotation, scale, shear, point noise, a hand missing for a stretch;
the same copies for every run), both compared with a paired bootstrap over photos. Photos are downloaded data:
run with -I.
"""
JITTER = 30
BOOT = 10000
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def extract(args):
    from azsl import landmarks as LM
    from azsl import features as FT
    paths = LM.ensure_models(args.models)
    ex = LM.Extractor(paths, face=False)
    with open(os.path.join(args.photos, 'index.csv'), encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if os.path.exists(os.path.join(args.photos, r['file']))]
    os.makedirs(args.out, exist_ok=True)
    blobs, meta = {}, []
    for k, r in enumerate(rows):
        with open(os.path.join(args.photos, r['file']), 'rb') as f:
            arrays, m = ex.image(f.read())
        raw = FT.reduce(arrays)
        hands = int(np.isfinite(raw['hand'][0, :, 0, 0]).sum())
        for key, v in raw.items():
            blobs[f'{k}/{key}'] = v
        meta.append({'i': k, 'file': r['file'], 'character': r['character'], 'field': r['field'],
                     'hands': hands, 'aspect': m['width'] / m['height']})
        print(r['character'], r['field'], 'hands', hands, flush=True)
    np.savez_compressed(os.path.join(args.out, 'landmarks.npz'), **blobs)
    with open(os.path.join(args.out, 'photos.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f'{len(meta)} photos, {sum(m["hands"] > 0 for m in meta)} with a hand, '
          f'{sum(m["hands"] == 2 for m in meta)} with two')


def mirror(raw):
    """The same photo flipped left-right: image x → 1 − x, world x → −x."""
    out = dict(raw)
    h = raw['hand'].astype(np.float32).copy()
    h[..., 0] = 1 - h[..., 0]
    w = raw['hand_world'].astype(np.float32).copy()
    w[..., 0] = -w[..., 0]
    out['hand'], out['hand_world'] = h.astype(raw['hand'].dtype), w.astype(raw['hand_world'].dtype)
    return out


def sign_test(b, c):
    """Exact two-sided sign test on the discordant pairs (b: only A right, c: only B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def boot(d, rng, n=BOOT):
    """Paired bootstrap over photos of a per-photo difference → (mean, 2.5 %, 97.5 %)."""
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def score(args):
    import torch
    from azsl import letters as L
    from install_letters import load_net
    with open(os.path.join(args.out, 'photos.json'), encoding='utf-8') as f:
        meta = json.load(f)
    z = np.load(os.path.join(args.out, 'landmarks.npz'))
    raws = [{k: z[f'{m["i"]}/{k}'] for k in ('hand', 'hand_world', 'pose', 'face', 'blend')} for m in meta]
    keep = [i for i, m in enumerate(meta) if m['hands'] > 0 and L.letter_of(m['character']) is not None]
    y = np.array([L.LABELS.index(L.letter_of(meta[i]['character'])) for i in keep])
    moving = np.array([L.LETTERS[t] in L.DYNAMIC for t in y])
    none = L.LABELS.index(L.NONE)
    runs = dict(r.split('=', 1) for r in args.run)
    from azsl import features as FT
    res, right, ptrue, jit = {}, {}, {}, {}
    for name, run_dir in runs.items():
        net, cfg = load_net(run_dir)
        res[name] = {}
        cols = FT.columns(cfg['groups'])
        hits = np.zeros(len(keep))
        for j, i in enumerate(keep):                 # the same copies for every run (seed per photo)
            rng = np.random.default_rng(1000 + i)
            still = {k: np.repeat(v, L.PHOTO_LEN, 0) for k, v in L.strip(raws[i]).items()}
            sq = FT.to_square(still, meta[i]['aspect'])
            xs = [FT.features(L.canonical(FT.augment(sq, rng, meta[i]['aspect'], (0, L.PHOTO_LEN))))[:, cols]
                  for _ in range(JITTER)]
            n = max(len(x) for x in xs)
            x = torch.zeros(JITTER, n, len(cols))
            m = torch.zeros(JITTER, n, dtype=torch.bool)
            for k, a in enumerate(xs):
                x[k, :len(a)], m[k, :len(a)] = torch.from_numpy(a), True
            with torch.no_grad():
                p = net(x, m).softmax(1).numpy()
            p[:, none] = -1
            hits[j] = (p.argmax(1) == y[j]).mean()
        jit[name] = hits
        res[name]['jittered'] = {'top1': float(hits.mean()), 'static_top1': float(hits[~moving].mean()),
                                 'copies': JITTER}
        print(f'{name:16s} jittered  top-1 {hits.mean():.3f}  static {hits[~moving].mean():.3f}  ({JITTER} copies each)')
        for view in ('as is', 'mirrored'):
            xs = []
            for i in keep:
                raw = raws[i] if view == 'as is' else mirror(raws[i])
                still = {k: np.repeat(v, L.PHOTO_LEN, 0) for k, v in L.strip(raw).items()}
                xs.append(L.preprocess(still, meta[i]['aspect'], cfg['groups']))
            x = torch.from_numpy(np.stack(xs))
            with torch.no_grad():
                p = net(x, torch.ones(x.shape[:2], dtype=torch.bool)).softmax(1).numpy()
            q = p.copy()
            q[:, none] = -1
            top = np.argsort(-q, 1)
            ok1 = top[:, 0] == y
            ok3 = (top[:, :3] == y[:, None]).any(1)
            right[(name, view)] = ok1
            ptrue[(name, view)] = p[np.arange(len(y)), y]
            r = {'n': int(len(y)), 'top1': float(ok1.mean()), 'top3': float(ok3.mean()),
                 'p_true': float(ptrue[(name, view)].mean()),
                 'top1_with_none': float((p.argmax(1) == y).mean()), 'as_none': float((p.argmax(1) == none).mean()),
                 'static_top1': float(ok1[~moving].mean()), 'static_n': int((~moving).sum()),
                 'moving_top1': float(ok1[moving].mean()) if moving.any() else None, 'moving_n': int(moving.sum()),
                 'per_letter': {L.LETTERS[t]: [L.LABELS[top[j, 0]], round(float(p[j, t]), 3)] for j, t in enumerate(y)}}
            res[name][view] = r
            print(f'{name:16s} {view:9s} top-1 {r["top1"]:.3f} ({ok1.sum()}/{len(y)})  top-3 {r["top3"]:.3f}  '
                  f'static {r["static_top1"]:.3f} / {r["static_n"]}  moving {r["moving_top1"]} / {r["moving_n"]}  '
                  f'with <none> {r["top1_with_none"]:.3f}  as <none> {r["as_none"]:.3f}  p(right) {r["p_true"]:.3f}')
    names = list(runs)
    pairs = {}
    rng = np.random.default_rng(0)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            key = f'{names[b]} - {names[a]}'
            pairs[key + ' (jittered top-1)'] = boot(jit[names[b]] - jit[names[a]], rng)
            pairs[key + ' (static, jittered top-1)'] = boot((jit[names[b]] - jit[names[a]])[~moving], rng)
            for view in ('as is', 'mirrored'):
                pairs[key + f' (p(right), {view})'] = boot(ptrue[(names[b], view)] - ptrue[(names[a], view)], rng)
    for k, v in list(pairs.items()):
        print(f'{k}: {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]')
    for view in ('as is', 'mirrored'):
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                A, B = right[(names[a], view)], right[(names[b], view)]
                only_a, only_b = int((A & ~B).sum()), int((B & ~A).sum())
                key = f'{names[a]} vs {names[b]} ({view})'
                pairs[key] = {'only_first': only_a, 'only_second': only_b, 'sign_test_p': sign_test(only_a, only_b)}
                print(f'{key}: only the first right {only_a}, only the second {only_b}, '
                      f'sign test p = {pairs[key]["sign_test_p"]:.3f}')
    skipped = [meta[i]['character'] for i in range(len(meta)) if i not in keep]
    with open(os.path.join(args.out, 'scores.json'), 'w', encoding='utf-8') as f:
        json.dump({'runs': runs, 'scores': res, 'pairs': pairs, 'no_hand': skipped}, f, ensure_ascii=False, indent=1)
    if skipped:
        print('no hand found:', ' '.join(skipped))


def export_sentences(args):
    """Another person's signing with no spelling (JestDili.az's sentence videos) → tools/letters_tune.mjs input:
    <out>/sent/<k>_hand.npy, <k>_world.npy (T × 126) + <out>/sent.json ({T, aspect, fps, letters: []}), for
    the false letters the letter mode types per minute."""
    from azsl.store import LandmarkStore
    st = LandmarkStore(args.store)
    d = os.path.join(args.out, 'sent')
    os.makedirs(d, exist_ok=True)
    rows = []
    for r in st.manifest.itertuples():
        raw = st.get(r.id)
        k = len(rows)
        np.save(os.path.join(d, f'{k}_hand.npy'), raw['hand'].reshape(len(raw['hand']), -1).astype(np.float32))
        np.save(os.path.join(d, f'{k}_world.npy'), raw['hand_world'].reshape(len(raw['hand']), -1).astype(np.float32))
        rows.append({'id': r.id, 'T': int(len(raw['hand'])), 'aspect': float(r.width) / float(r.height),
                     'fps': float(r.fps), 'letters': []})
    with open(d + '.json', 'w', encoding='utf-8') as f:
        json.dump(rows, f)
    print(f'{len(rows)} videos, {sum(r["T"] / r["fps"] for r in rows) / 60:.1f} min')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract')
    e.add_argument('--photos', required=True)
    e.add_argument('--models', required=True, help='MediaPipe .task models (downloaded and md5-checked if missing)')
    e.add_argument('--out', required=True)
    s = sub.add_parser('score')
    s.add_argument('--out', required=True)
    s.add_argument('--run', action='append', required=True, help='name=run dir with best.pt')
    x = sub.add_parser('export-sentences')
    x.add_argument('--store', required=True, help='landmark store, e.g. D:/chevir/landmarks/jestdili_sentences')
    x.add_argument('--out', required=True)
    args = ap.parse_args()
    {'extract': extract, 'score': score, 'export-sentences': export_sentences}[args.cmd](args)


if __name__ == '__main__':
    main()
