"""The steps the Kaggle notebooks run. Every step is resumable and takes `limit` for smoke tests
(with `limit`, archives are read member-by-member over HTTP instead of being downloaded)."""
import json
import multiprocessing as mp
import os
import shutil
import tempfile
import time
import zipfile
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

from . import landmarks, sources, store, zenodo
from . import split as S

_EX = None
_ZIP = None


# ---------------------------------------------------------------- workers

def _init_worker(paths, face, embed, zip_path):
    global _EX, _ZIP
    # MediaPipe's C++ side logs ~20 warnings per landmarker; thousands of videos would flood the Kaggle log
    os.environ.setdefault('GLOG_minloglevel', '2')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import cv2
    cv2.setNumThreads(1)                     # one process per core already
    _EX = landmarks.Extractor(paths, face=face, embed=embed)
    _ZIP = zipfile.ZipFile(zip_path) if zip_path else None


def _read(src):
    return _ZIP.read(src) if isinstance(src, str) else zenodo.read_member(src)


def _video_job(task):
    item_id, src, extra = task
    t0 = time.time()
    data = _read(src)
    fd, tmp = tempfile.mkstemp(suffix='.mp4')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        arrays, meta, face = _EX.video(tmp)
    finally:
        os.remove(tmp)
    meta.update(extra)
    meta['sec'] = round(time.time() - t0, 2)
    return item_id, arrays, meta, face


def _image_job(task):
    item_id, src, extra = task
    arrays, meta = _EX.image(_read(src))
    meta.update(extra)
    return item_id, arrays, meta, None


def run_pool(tasks, job, writer, init_args, workers=None, time_budget_h=11.0, log_every=250, label=''):
    """Runs `job` over tasks [(id, source, extra meta)] in spawned processes, writing as results come.
    Stops taking new tasks after `time_budget_h` so a Kaggle run can finish and save its output.
    A crashed worker (MediaPipe segfault) restarts the pool; its in-flight tasks get one retry."""
    workers = workers or os.cpu_count() or 2
    done = writer.done_ids()
    queue = deque(t for t in tasks if t[0] not in done)
    total = len(queue)
    workers = max(1, min(workers, total))
    print(f'{label}: {len(tasks)} items, {len(done)} already done, {total} to go, {workers} workers', flush=True)
    t0, deadline = time.time(), time.time() + time_budget_h * 3600
    errors, retries, n, frames = [], {}, 0, 0
    ctx = mp.get_context('spawn')
    fruitless = 0                            # pool restarts in a row without a finished item
    while queue and time.time() < deadline:
        pending, n_start = {}, n
        try:
            with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker, initargs=init_args) as ex:
                while queue or pending:
                    while queue and len(pending) < workers * 3 and time.time() < deadline:
                        t = queue.popleft()
                        try:
                            pending[ex.submit(job, t)] = t
                        except BrokenProcessPool:
                            queue.appendleft(t)      # not in `pending` yet — would be lost
                            raise
                    if not pending:
                        break
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for f in finished:
                        t = pending.pop(f)
                        try:
                            item_id, arrays, meta, face = f.result()
                        except BrokenProcessPool:
                            raise
                        except Exception as e:  # noqa: BLE001 — one bad file must not stop the run
                            errors.append({'id': t[0], 'error': repr(e)[:300]})
                            continue
                        writer.add(item_id, arrays, meta, face)
                        n, frames = n + 1, frames + meta['n_frames']
                        if n % log_every == 0:
                            el = time.time() - t0
                            eta = el / n * (total - n) / 3600
                            print(f'  {n}/{total}  {frames / el:.0f} frames/s  {el / 3600:.2f} h  ETA {eta:.2f} h',
                                  flush=True)
        except BrokenProcessPool:
            fruitless = fruitless + 1 if n == n_start else 0
            if fruitless >= 3:
                writer.flush()
                raise RuntimeError('workers keep crashing before finishing anything — mediapipe or a model '
                                   'is broken in this environment (run the self-test cell)')
            print('  worker crashed — restarting pool', flush=True)
            for t in pending.values():
                retries[t[0]] = retries.get(t[0], 0) + 1
                if retries[t[0]] <= 1:
                    queue.appendleft(t)
                else:
                    errors.append({'id': t[0], 'error': 'worker crashed twice'})
    writer.flush()
    if errors:
        import csv
        path = os.path.join(writer.dir, 'errors.csv')
        new = not os.path.exists(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['id', 'error'])
            if new:
                w.writeheader()
            w.writerows(errors)
    el = time.time() - t0
    left = total - n - len(errors)
    summary = {'done': n, 'errors': len(errors), 'remaining': left, 'hours': round(el / 3600, 2),
               'frames_per_s': round(frames / el, 1) if el else 0.0, 'complete': left == 0}
    print(f'{label}: {summary}', flush=True)
    if left:
        print(f'  {left} items left (time budget) — rerun with this output attached to resume', flush=True)
    return summary


# ---------------------------------------------------------------- steps

def setup_models(model_dir='/tmp/azsl_models', sface=False):
    return landmarks.ensure_models(model_dir, sface=sface)


def mediapipe_ok(paths):
    """Creates a hand landmarker in a fresh process (a broken mediapipe/numpy pairing can crash
    the interpreter). → mediapipe version, or None."""
    import subprocess
    import sys
    code = ('import mediapipe as mp, numpy as np\n'
            'from mediapipe.tasks.python import vision as V, BaseOptions as B\n'
            f'h = V.HandLandmarker.create_from_options(V.HandLandmarkerOptions(base_options=B(model_asset_path={paths["hand"]!r})))\n'
            'h.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((64, 64, 3), np.uint8)))\n'
            'print(mp.__version__)')
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-1500:])
        return None
    return r.stdout.strip().splitlines()[-1]


def run_index(out_dir, folders=None):
    """Sentences annotations → index/tags.csv, recordings.csv, sentences.csv (a few MB, ~minutes)."""
    members = zenodo.list_members(zenodo.archive_url('sentences'))
    if folders is not None:
        keep = {str(f) for f in folders}
        members = [m for m in members if m.name.split('/')[1] in keep or m.name.endswith('sentence_list.csv')]
    tags, recs, sents = sources.sentences_index(members)
    d = os.path.join(out_dir, 'index')
    os.makedirs(d, exist_ok=True)
    tags.to_csv(os.path.join(d, 'tags.csv'), index=False)
    recs.to_csv(os.path.join(d, 'recordings.csv'), index=False)
    sents.to_csv(os.path.join(d, 'sentences.csv'), index=False)
    print(f'index: {len(recs)} recordings ({int(recs.cam2.sum())} with Cam2), {len(tags)} tags '
          f'({int(tags.dup.sum())} duplicates), {tags.gloss.nunique()} glosses, {len(sents)} sentences')
    return tags, recs, sents


def _archive(key, tmp_dir, limit):
    """Full runs download the archive (one big request is kinder to Zenodo than thousands);
    smoke tests read the needed members over HTTP."""
    url = zenodo.archive_url(key)
    if limit:
        return None, zenodo.list_members(url)
    path = os.path.join(tmp_dir, zenodo.ARCHIVES[key])
    t0 = time.time()
    zenodo.download(url, path)
    print(f'{zenodo.ARCHIVES[key]}: {os.path.getsize(path) / 1e9:.2f} GB in {time.time() - t0:.0f} s', flush=True)
    with zipfile.ZipFile(path) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
    return path, names


def _by_name(members):
    """name → task source: the name itself (read from the local zip) or a zenodo.Member (HTTP)."""
    return {m if isinstance(m, str) else m.name: m for m in members}


def run_words200(out_dir, paths, workers=None, limit=None, tmp_dir='/tmp', time_budget_h=11.0):
    zpath, members = _archive('words200', tmp_dir, limit)
    by_name = _by_name(members)
    listing = sources.words_listing(list(by_name))
    if limit:
        listing = listing.groupby('gloss').head(1).head(limit)
    tasks = [(r.key, by_name[r.member], {'key': r.key, 'gloss': r.gloss}) for r in listing.itertuples()]
    writer = store.PartWriter(os.path.join(out_dir, 'landmarks', 'words200'), items_per_part=1000)
    return run_pool(tasks, _video_job, writer, (paths, True, False, zpath), workers, time_budget_h,
                    label='Words_200')


def run_fingerspelling(out_dir, paths, workers=None, limit=None, tmp_dir='/tmp', time_budget_h=11.0):
    zpath, members = _archive('fingerspelling', tmp_dir, limit)
    by_name = _by_name(members)
    listing = sources.fingerspelling_listing(list(by_name))
    if limit:
        listing = listing.groupby('letter').head(1).head(limit)
    tasks = [(r.member.split('/', 1)[1], by_name[r.member],
              {'letter': r.letter, 'kind': r.kind, 'group': r.group, 'run': r.run, 'frame_no': r.frame_no})
             for r in listing.itertuples()]
    writer = store.PartWriter(os.path.join(out_dir, 'landmarks', 'fingerspelling'), items_per_part=4000)
    return run_pool(tasks, _image_job, writer, (paths, True, False, zpath), workers, time_budget_h,
                    log_every=1000, label='Fingerspelling')


def run_sentences(out_dir, paths, shard=0, num_shards=1, camera='Cam2', workers=None, limit=None,
                  time_budget_h=11.0, resume_from=()):
    """One shard of the Sentences recordings, read one by one from the 43.9 GB zip.
    Cam2 (frontal) also stores a face embedding per recording for the signer split."""
    members = zenodo.list_members(zenodo.archive_url('sentences'))
    recs = sources.sentence_recordings(members, camera)
    mine = [m for k, m in enumerate(recs) if k % num_shards == shard]
    if limit:
        mine = mine[:limit]
    sub = os.path.join(out_dir, 'landmarks', f'sentences_{camera.lower()}', f'shard{shard:02d}')
    for prev in resume_from:                 # earlier run of this shard attached as input
        if os.path.exists(os.path.join(prev, 'manifest.csv')) and not os.path.exists(os.path.join(sub, 'manifest.csv')):
            shutil.copytree(prev, sub, dirs_exist_ok=True)
            print(f'resumed from {prev}', flush=True)
    embed = camera == 'Cam2'
    writer = store.PartWriter(sub, items_per_part=200, embeddings=embed)
    tasks = [(sources.rec_id(m.name), m, {'rec_id': sources.rec_id(m.name), 'camera': camera, 'bytes': m.size})
             for m in mine]
    return run_pool(tasks, _video_job, writer, (paths, True, embed, None), workers, time_budget_h,
                    log_every=100, label=f'Sentences {camera} shard {shard}/{num_shards}')


# ---------------------------------------------------------------- split

def run_split(index_dir, words_dirs, fs_dirs, sentence_dirs, out_dir, sim=0.45, link_sim=0.40, gap_min=15.0,
              seed=0, override_csv=None, heldout_sentence_frac=0.1, min_clips=20, k_folds=5, trials=4000):
    """Pseudo-signers → one signer-level train/val/test split → per-task split files + benchmark.json."""
    import numpy as np
    import pandas as pd
    tags = pd.read_csv(os.path.join(index_dir, 'tags.csv'), dtype={'key': str, 'rec_id': str})
    recs = pd.read_csv(os.path.join(index_dir, 'recordings.csv'), dtype={'rec_id': str, 'folder': str},
                       parse_dates=['stamp'])
    sents = pd.read_csv(os.path.join(index_dir, 'sentences.csv'), dtype={'folder': str})

    sen = store.LandmarkStore(sentence_dirs)
    emb_ids, emb, thumbs = sen.embeddings()
    ok = ~np.isnan(emb).any(1) if len(emb) else np.zeros(0, bool)
    emb_ids, emb = [i for i, k in zip(emb_ids, ok) if k], emb[ok]
    # compact copy for tuning the clustering offline (anonymous vectors, no images; never committed)
    priv = os.path.join(os.path.dirname(os.path.abspath(out_dir)), 'private')
    os.makedirs(priv, exist_ok=True)
    np.savez_compressed(os.path.join(priv, 'face_embeddings.npz'), ids=np.array(emb_ids), emb=emb.astype(np.float16))
    override = pd.read_csv(override_csv, dtype=str) if override_csv else None
    signers, rep = S.assign_signers(recs, emb_ids, emb, sim=sim, gap_min=gap_min, override=override)
    group_of = S.split_groups(signers, emb_ids, emb, link_sim)
    signers['split_group'] = signers.signer.map(group_of)
    m = sen.manifest.set_index('rec_id') if 'rec_id' in sen.manifest else sen.manifest.set_index('id')
    signers['has_landmarks'] = signers.rec_id.isin(m.index)
    signers['frontal'] = signers.rec_id.map(m['frontal'])
    signer_of = dict(zip(signers.rec_id, signers.signer))

    words = store.LandmarkStore(words_dirs).manifest[['id', 'gloss', 'hand_rate', 'n_frames']].rename(columns={'id': 'key'})
    vocab = set(words.gloss)
    t = tags[~tags.dup].copy()
    t['signer'] = t.rec_id.map(signer_of)
    t = t.dropna(subset=['signer'])
    t['split_group'] = t.signer.map(group_of)

    # one split over split groups (pseudo-signers + the ones ambiguously alike), optimised for
    # covering the 200-word vocabulary in val and test, then shared by every task
    t200 = t[t.gloss.isin(vocab)]
    unknown = sorted({group_of[g] for g in t.signer.unique() if g.startswith('unk-')})
    assign, cover = S.pick_split(t200, 'split_group', 'gloss', seed=seed, trials=trials, forced_train=unknown)
    for g in set(group_of.values()):
        assign.setdefault(g, 'train')
    fold_of = S.folds(t200[t200.split_group.map(assign) == 'train'], 'split_group', k_folds)
    signers['split'] = signers.split_group.map(assign).fillna('train')
    signers['fold'] = signers.split_group.map(fold_of)
    split_of, fold_by_signer = dict(zip(signers.signer, signers.split)), dict(zip(signers.signer, signers.fold))

    # task 1 — Words_200 clips (link via tag key → recording → signer)
    w = words.merge(tags[['key', 'rec_id', 'start', 'end']], on='key', how='left')
    w['signer'] = w.rec_id.map(signer_of).fillna('unknown')
    w['split'] = w.signer.map(split_of).fillna('train')
    w['fold'] = w.signer.map(fold_by_signer)
    # task 2 — every annotated gloss cut from the sentences (slice the recording's landmarks)
    n_clips, n_sig = t.groupby('gloss').size(), t.groupby('gloss').signer.nunique()
    t['in_w200'] = t.gloss.isin(vocab)
    t['min_clips'] = t.gloss.map((n_clips >= min_clips) & (n_sig >= 3))
    t['split'], t['fold'] = t.signer.map(split_of), t.signer.map(fold_by_signer)
    t['has_landmarks'] = t.rec_id.isin(m.index)
    # task 3 — whole sentences; a held-out set of sentence folders measures generalisation to unseen sentences
    rng = np.random.default_rng(seed)
    folders = sorted(recs.folder.unique(), key=lambda x: (len(x), x))
    held = set(rng.choice(folders, max(1, int(len(folders) * heldout_sentence_frac)), replace=False))
    s = signers.merge(recs[['rec_id', 'folder', 'frames']], on='rec_id', how='left')
    s['heldout_sentence'] = s.folder.isin(held)
    s = s.merge(sents, on='folder', how='left')      # sentence text + gloss sequence, for Phase 4
    # task 4 — fingerspelling (contributor id or clip run as the group)
    fs = store.LandmarkStore(fs_dirs).manifest
    fs_assign, fs_cover = {}, {}
    for kind, g in fs.groupby('kind'):
        a, c = S.pick_split(g, 'group', 'letter', seed=seed, trials=trials)
        fs_assign.update(a)
        fs_cover[kind] = c
    fs['split'] = fs.group.map(fs_assign)

    os.makedirs(out_dir, exist_ok=True)
    signers.to_csv(os.path.join(out_dir, 'signers.csv'), index=False)
    w.to_csv(os.path.join(out_dir, 'words200_split.csv'), index=False)
    t.to_csv(os.path.join(out_dir, 'glosses_split.csv'), index=False)
    s.to_csv(os.path.join(out_dir, 'sentences_split.csv'), index=False)
    fs[['id', 'letter', 'kind', 'group', 'run', 'frame_no', 'hand_rate', 'split']].to_csv(
        os.path.join(out_dir, 'fingerspelling_split.csv'), index=False)

    split_signers = signers.drop_duplicates('signer').groupby('split').size().to_dict()
    rep['closest_pairs'] = S.closest_pairs(signers, emb_ids, emb)
    cross = [p['sim'] for p in S.closest_pairs(signers, emb_ids, emb, top=10_000) if len(set(p['splits'].split('/'))) > 1]
    gsize = signers.groupby('split_group').size()
    rep['split_groups'] = {'count': int(len(gsize)), 'largest_share': round(float(gsize.max() / gsize.sum()), 3),
                           'per_split': signers.drop_duplicates('split_group').groupby('split').size().to_dict(),
                           'max_cross_split_sim': max(cross, default=None), 'link_sim': link_sim}
    bench = {
        'protocol': 'signer-independent; one split-group→split map shared by all tasks; seed %d' % seed,
        'signers': {**rep, 'per_split': split_signers, 'sim_threshold': sim, 'session_gap_min': gap_min},
        'frontal_outliers': int((signers.frontal > 0.35).sum()),
        'words200': {'coverage': cover, 'unlinked_clips': int((w.signer == 'unknown').sum()),
                     **S.baselines(w, 'gloss')},
        'glosses_min%d' % min_clips: S.baselines(t[t.min_clips], 'gloss'),
        'glosses_all': S.baselines(t, 'gloss'),
        'sentences': {'heldout_folders': len(held),
                      **{k: int(v) for k, v in s.groupby('split').size().items()}},
        'fingerspelling': {kind: {'coverage': fs_cover[kind], **S.baselines(g, 'letter')}
                           for kind, g in fs.groupby('kind')},
    }
    with open(os.path.join(out_dir, 'benchmark.json'), 'w', encoding='utf-8') as f:
        json.dump(bench, f, ensure_ascii=False, indent=1, default=str)
    return bench, signers, dict(zip(emb_ids, thumbs))


def signer_sheet(signers, thumbs, per_signer=6, size=72):
    """Contact sheet: one row per pseudo-signer, a few face crops each — to eyeball the clustering.
    Shown in the notebook only; not written to the output."""
    import cv2
    import numpy as np
    rows = []
    for sig, g in signers[signers.rec_id.isin(thumbs)].groupby('signer'):
        tiles = []
        for rid in g.rec_id.sample(min(per_signer, len(g)), random_state=0):
            im = cv2.imdecode(np.frombuffer(thumbs[rid], np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                tiles.append(cv2.resize(im, (size, size)))
        tiles += [np.zeros((size, size, 3), np.uint8)] * (per_signer - len(tiles))
        label = np.full((size, 90, 3), 255, np.uint8)
        cv2.putText(label, f'{sig} {len(g)}', (2, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        rows.append(np.hstack([label] + tiles))
    return cv2.cvtColor(np.vstack(rows), cv2.COLOR_BGR2RGB) if rows else None
