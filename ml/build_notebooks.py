"""Generates the Kaggle notebooks in ml/notebooks/ from the azsl package.

Each notebook is self-contained: the package is written out with %%writefile cells, so
nothing has to be uploaded to Kaggle besides the .ipynb itself. Edit azsl/*.py, then:
    python ml/build_notebooks.py
"""
import json
import os
import sys

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = ['__init__', 'zenodo', 'landmarks', 'store', 'sources', 'split', 'pipeline', 'views']
TRAIN_MODULES = MODULES + ['features', 'train']
NUM_SHARDS = 4

SETUP = '''import glob, json, os, subprocess, sys, tempfile
ON_KAGGLE = os.path.isdir('/kaggle/working')
OUT = '/kaggle/working' if ON_KAGGLE else os.path.abspath('azsl_out')
PKG = os.path.join(tempfile.gettempdir(), 'azsl_pkg')
MODELS = os.path.join(tempfile.gettempdir(), 'azsl_models')
os.makedirs(os.path.join(PKG, 'azsl'), exist_ok=True)
os.makedirs(OUT, exist_ok=True)
if PKG not in sys.path:
    sys.path.insert(0, PKG)'''

INSTALL = '''# MediaPipe 0.10.14 = the version the browser runs (@mediapipe/tasks-vision 0.10.14).
# If it does not work with this Kaggle image, a newer 0.10.x is used — the .task models
# (what actually decides the landmarks) are pinned by md5 either way.
def pip(*args):
    return subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *args]).returncode == 0

if ON_KAGGLE and not pip('mediapipe==0.10.14'):
    pip('mediapipe>=0.10.14,<0.11')'''

MODELS_CELL = '''from azsl import pipeline

paths = pipeline.setup_models(MODELS, sface={sface})
mp_version = pipeline.mediapipe_ok(paths)
if mp_version is None and ON_KAGGLE:
    print('pinned mediapipe failed its self-test — upgrading')
    pip('-U', 'mediapipe')
    mp_version = pipeline.mediapipe_ok(paths)
assert mp_version, 'mediapipe does not run — see the error above'
versions = {{'mediapipe': mp_version, 'python': sys.version.split()[0],
            'models': {{k: os.path.basename(p) for k, p in paths.items()}}}}
with open(os.path.join(OUT, 'versions_{name}.json'), 'w') as f:
    json.dump(versions, f, indent=1)
versions'''


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def package_cells(modules=MODULES):
    cells = [md('### The `azsl` package\nGenerated from `ml/azsl/` — edit it there, not here.')]
    for m in modules:
        with open(os.path.join(HERE, 'azsl', f'{m}.py'), encoding='utf-8') as f:
            cells.append(code(f'%%writefile {{PKG}}/azsl/{m}.py\n' + f.read()))
    return cells


def notebook(cells):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'kaggle': {'accelerator': 'none', 'isInternetEnabled': True, 'isGpuEnabled': False},
    }
    nb.cells = cells
    return nb


def nb01():
    return notebook([
        md('''
# AzSLD 01 — index, Words_200, Fingerspelling → landmarks

Phase 1 of the model plan (`ml/README.md` in the chevir repo). Reads AzSLD (Zenodo 14222948,
CC BY 4.0) and stores MediaPipe landmarks made with **the same models the browser uses**.

**Kaggle settings:** Accelerator **None** (CPU only, no GPU quota), Internet **On**.
Run with **Save Version → Save & Run All (Commit)** so it keeps going when the tab is closed.

**Takes** about 1–1.5 h. **Writes** to `/kaggle/working`:
- `index/` — every annotated sign in the Sentences recordings (tags, recordings, sentences)
- `landmarks/words200/` — 8,557 word clips
- `landmarks/fingerspelling/` — 7,359 letter images

Set `SMOKE = True` for a 3-minute check of the setup first.
'''),
        code('''
SMOKE = False          # True: a few files of each kind, to check the setup
WORKERS = None         # None = every CPU (4 on Kaggle)
TIME_BUDGET_H = 11.0   # stop taking new items after this, so the run saves its output
'''),
        code(SETUP),
        *package_cells(),
        code(INSTALL),
        code(MODELS_CELL.format(sface=False, name='01')),
        md('### Sentences annotation index\nReads the 10,104 annotation JSONs by byte range — a few MB.'),
        code('''
tags, recs, sents = pipeline.run_index(OUT, folders=['1', '2'] if SMOKE else None)
tags.head()
'''),
        md('### Words_200\nDownloads the 1.36 GB zip, then landmarks per clip (VIDEO mode, like the live camera).'),
        code('''
words_summary = pipeline.run_words200(OUT, paths, WORKERS, limit=5 if SMOKE else None,
                                      tmp_dir=tempfile.gettempdir(), time_budget_h=TIME_BUDGET_H)
'''),
        md('### Fingerspelling\n0.55 GB of images (IMAGE mode).'),
        code('''
fs_summary = pipeline.run_fingerspelling(OUT, paths, WORKERS, limit=8 if SMOKE else None,
                                         tmp_dir=tempfile.gettempdir(), time_budget_h=TIME_BUDGET_H)
'''),
        md('### Check'),
        code('''
from azsl.store import LandmarkStore
w = LandmarkStore(os.path.join(OUT, 'landmarks', 'words200')).manifest
f = LandmarkStore(os.path.join(OUT, 'landmarks', 'fingerspelling')).manifest
print(f'words200: {len(w)} clips, {w.gloss.nunique()} glosses, median {w.n_frames.median():.0f} frames')
print(w[['hand_rate', 'two_hand_rate', 'pose_rate', 'face_rate']].describe().round(3))
print(f'\\nfingerspelling: {len(f)} images; share with a hand found, per letter:')
print(f.groupby('letter').hand_rate.mean().round(2).to_dict())
for z in glob.glob(os.path.join(tempfile.gettempdir(), 'AzSLD_*.zip')):
    os.remove(z)   # not part of the output anyway; frees disk
'''),
    ])


def nb02(shard, camera='Cam2'):
    side = camera == 'Cam1'
    view = 'side camera' if side else 'frontal camera'
    about = (f"""Reads the side-view (`Cam1`) recordings one at a time from the 43.9 GB zip by HTTP byte range and
keeps hand and body landmarks (no face: the word models do not use it, and it saves a third of the
time). This shard takes every {NUM_SHARDS}th recording starting at #{shard}. The annotations belong to the
frontal camera, so the last cell finds each recording's frame offset to it (`azsl.views`) and writes
`align.csv`.

**Inputs:** the four Cam2 shard notebooks (02, for the alignment).""" if side else f"""Reads the frontal (`Cam2`) recordings one at a time from the 43.9 GB zip by HTTP byte range —
nothing big is downloaded — and keeps only landmarks. This shard takes every {NUM_SHARDS}th
recording starting at #{shard}. Each recording also gets one face embedding, used **only** by
notebook 03 to group recordings by signer (AzSLD has no signer ids); it is not a training feature.""")
    cells = [
        md(f'''
# AzSLD 02 — Sentences ({view}) → landmarks · shard {shard} of {NUM_SHARDS}

{about}

**Kaggle settings:** Accelerator **None**, Internet **On**. Run with **Save Version → Save & Run All**.
The four shard notebooks can run at the same time.

**Takes** {'4–6' if side else '6–7'} h. **Writes** `landmarks/sentences_{camera.lower()}/shard{shard:02d}/`.

If it stops at the time budget (`complete: False` at the end): add this notebook's own output
as an input and run again — finished recordings are skipped.
'''),
        code(f'''
SHARD = {shard}
NUM_SHARDS = {NUM_SHARDS}
CAMERA = '{camera}'        # Cam2 = frontal (annotated), Cam1 = side view
SMOKE = False          # True: {12 if side else 3} recordings, to check the setup
WORKERS = None
TIME_BUDGET_H = 11.0
'''),
        code(SETUP),
        *package_cells(),
        code(INSTALL),
        code(MODELS_CELL.format(sface=not side, name=f'02_{camera.lower()}_shard{shard}' if side else f'02_shard{shard}')),
        code(f'''
# an earlier run of this same shard, attached as input → resume from it
prev = glob.glob(f'/kaggle/input/**/landmarks/sentences_{{CAMERA.lower()}}/shard{{SHARD:02d}}', recursive=True)
print('resume from:', prev or 'nothing')
summary = pipeline.run_sentences(OUT, paths, SHARD, NUM_SHARDS, CAMERA, WORKERS,
                                 limit={12 if side else 3} if SMOKE else None, time_budget_h=TIME_BUDGET_H, resume_from=prev)
summary
'''),
        code('''
import pandas as pd
from azsl.store import LandmarkStore
m = LandmarkStore(os.path.join(OUT, 'landmarks', f'sentences_{CAMERA.lower()}', f'shard{SHARD:02d}')).manifest
print(f'{len(m)} recordings, {m.n_frames.sum():,} frames, {int((m.n_frames == 0).sum())} unreadable (0 frames)')
m['frontal'] = pd.to_numeric(m.frontal, errors='coerce')        # NaN (written as text) without any pose
print(m[['n_frames', 'hand_rate', 'two_hand_rate', 'pose_rate', 'face_rate', 'frontal']].describe().round(3))
print('not frontal (> 0.35):', int((m.frontal > 0.35).sum()))
'''),
    ]
    if side:
        cells += [md('''
### Frame offset to the frontal camera
`lag`: Cam1 frame i shows Cam2 frame i + lag. `corr` = agreement of the arms' vertical movement at that lag,
`margin` = how far it is ahead of the best lag more than 5 frames away. Training uses the recordings
whose alignment is clear (thresholds set in notebook 04 from this table, not from model scores).
'''), code('''
from azsl import views
side_dir = os.path.join(OUT, 'landmarks', f'sentences_{CAMERA.lower()}', f'shard{SHARD:02d}')
front_dirs = sorted({os.path.dirname(p) for p in glob.glob('/kaggle/input/**/landmarks/sentences_cam2/shard*/manifest.csv',
                                                           recursive=True)})
print('frontal shards:', len(front_dirs))
try:                                   # the landmarks above are saved either way
    al = views.align_store(LandmarkStore(side_dir), LandmarkStore(front_dirs))
    al.to_csv(os.path.join(side_dir, 'align.csv'), index=False)
    print(al[['n_side', 'n_front', 'lag', 'corr', 'second', 'margin']].describe().round(3))
    print('lag counts:', al.lag.value_counts().sort_index().to_dict())
except Exception as e:  # noqa: BLE001
    print('alignment failed:', repr(e))
''')]
    return notebook(cells)


def nb03():
    return notebook([
        md('''
# AzSLD 03 — signer-independent split + benchmark (Phase 0)

**Inputs:** add the outputs of notebook 01 and of all four 02 shards
(*Add Input → Your Work → Notebooks*). **Kaggle settings:** Accelerator **None**; Internet not needed.

AzSLD has no signer ids, so recordings are grouped into pseudo-signers: sessions from the
recording faces clustered per recording, then look-alikes tied into split groups (see `azsl/split.py`). One signer → train/val/test
assignment is shared by every task, so a test signer is never seen in training.

**Writes** `splits/`: `signers.csv`, `words200_split.csv`, `glosses_split.csv`,
`sentences_split.csv`, `fingerspelling_split.csv`, `benchmark.json`.
Download `splits/` and commit it to `ml/splits/` — that is the frozen benchmark every model is scored on.
'''),
        code('''
SIM = 0.55             # recordings at least this alike (average linkage) form one pseudo-signer
LINK_SIM = 0.50        # pseudo-signers with ANY two sittings this alike always share a split
GAP_MIN = 15.0         # minutes between recordings that start a new session
SEED = 0
OVERRIDE_CSV = None    # optional CSV rec_id,signer with real signer ids, if the dataset authors share them
'''),
        code(SETUP),
        *package_cells(),
        code('''
from azsl import pipeline
INPUT = '/kaggle/input' if ON_KAGGLE else OUT

def find(pattern):
    return sorted({os.path.dirname(p) for p in glob.glob(os.path.join(INPUT, '**', pattern), recursive=True)})

index_dir = find('index/tags.csv')
words_dirs = find('landmarks/words200/manifest.csv')
fs_dirs = find('landmarks/fingerspelling/manifest.csv')
sentence_dirs = find('landmarks/sentences_cam2/shard*/manifest.csv')
print('index:', index_dir, '\\nwords:', words_dirs, '\\nfingerspelling:', fs_dirs, '\\nsentence shards:', sentence_dirs)
assert index_dir and words_dirs and fs_dirs and sentence_dirs, 'attach the outputs of notebooks 01 and 02'
if len(sentence_dirs) < 4:
    print(f'WARNING: only {len(sentence_dirs)} of 4 sentence shards attached — signers in the missing ones '
          'fall back to session ids')
'''),
        code('''
bench, signers, thumbs = pipeline.run_split(index_dir[0], words_dirs, fs_dirs, sentence_dirs,
                                            os.path.join(OUT, 'splits'), sim=SIM, link_sim=LINK_SIM, gap_min=GAP_MIN,
                                            seed=SEED, override_csv=OVERRIDE_CSV)
print(json.dumps({k: v for k, v in bench['signers'].items() if k != 'closest_pairs'}, indent=1, default=str))
import pandas as pd
pd.DataFrame(bench['signers']['closest_pairs'])   # a close pair across splits would be one person leaking
'''),
        md('''
### Check the pseudo-signers
Each row should be **one person**. One person spread over several rows is only harmful if the
rows land in different splits — pseudo-signers with any two sittings at least `LINK_SIM` alike
are tied to one split (`split_group`); `closest_pairs` / `max_cross_split_sim` compare sittings
and show what is left across splits. Two people in one row → raise `SIM`. `nearest_other_signer_sim` above should stay well below `SIM`, and
`signer_sentence_repeats` small (a signer rarely records the same sentence twice).
'''),
        code('''
import matplotlib.pyplot as plt
sheet = pipeline.signer_sheet(signers, thumbs)
if sheet is not None:
    plt.figure(figsize=(12, max(3, sheet.shape[0] / 60)))
    plt.imshow(sheet); plt.axis('off'); plt.show()
signers.groupby('signer').agg(recordings=('rec_id', 'size'), sessions=('session', 'nunique'),
                              split=('split', 'first')).sort_values('recordings', ascending=False).head(60)
'''),
        md('### Benchmark'),
        code('''
for task in ('words200', 'glosses_min20', 'glosses_all'):
    b = bench[task]
    print(f"{task}: majority class {b['majority_class']!r}")
    for s in ('train', 'val', 'test'):
        print(f"   {s:5s} {b[s]}")
print('words200 coverage:', bench['words200']['coverage'], '| unlinked clips (train only):', bench['words200']['unlinked_clips'])
print('sentences:', bench['sentences'])
for kind, b in bench['fingerspelling'].items():
    print(f"fingerspelling {kind}: {b['coverage']} | test {b['test']}")
print('frontal-camera outliers:', bench['frontal_outliers'])
'''),
    ])


def sign_merge():
    """One sign, several glosses → {gloss: group}: the confirmed word-form families (ml/lemma_groups.csv)
    and the forms the models cannot tell apart on val signers (ml/same_sign.csv)."""
    import csv
    sys.path.insert(0, HERE)
    from analyze import load_groups
    out = dict(load_groups(os.path.join(HERE, 'lemma_groups.csv')))
    with open(os.path.join(HERE, 'same_sign.csv'), encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        out[r['gloss']] = r['group']
    for r in rows:                         # a group that joins a confirmed family takes its name
        out[r['group']] = out.get(r['group'], r['group'])
    return {g: grp for g, grp in out.items() if g != grp}


def spelling():
    """ml/gloss_spelling.csv → {variant: gloss}: the same word typed two ways in the annotations."""
    import csv
    with open(os.path.join(HERE, 'gloss_spelling.csv'), encoding='utf-8') as f:
        return {r['variant']: r['gloss'] for r in csv.DictReader(f)}


def nb04():
    return notebook([
        md("""
# AzSLD 04 — train the word model (Phase 2)

Isolated-sign recognition on landmarks, **signer-independent**: train on the sentence-cut signs
of train signers, report every epoch on the val signers, score the test signers once at the end
with the best-on-val checkpoint. Model: Conv1D + Transformer, ~1.7 M parameters, exported to ONNX
for the browser. Vocabulary: the 200 words, plus (from run 8) every sentence gloss with enough signs
of train signers. With `SIDE`, the side camera's view of the same signs is training data too.

**Inputs:** outputs of 01, the four 02 shards (and the four Cam1 shards for `SIDE`) and 03
(*Add Input → Your Work → Notebooks*).
**Kaggle settings:** Accelerator **GPU T4**, Internet **On** (only to install `onnx` if missing).
**Writes** `train/`: `history.csv` (per epoch), `metrics.json` (test), `per_class_test_words.csv`,
`best.pt`, `model.onnx`, `vocab.json`.

Columns per epoch: `W_*` = Words_200 val clips (the benchmark), `S_*` = sentence-cut val signs,
`S200` = those of the 200 words, `side` = val signers seen by the side camera; `mF1` = macro-F1.
Always compare with the majority-class accuracy printed at the start.
"""),
        code("""
SMOKE = False          # True: 2 epochs on a sample, to check the setup
CFG = dict(epochs=60, batch=64, lr=1e-3, dim=192, drop=0.2, margin=8, workers=4, seed=0)
# Each entry = one experiment: CFG with these changes. All share the loaded data; results go to
# train/<name>/ and train/summary.csv. Run 3 (split v6, seed 0, all features) is the reference.
# Run 4 (seed1, seed2, hands_only, no_face, epochs100): no_face = hands + pose was best, one seed.
# Run 5: does no_face hold over seeds, and without the face tracker (the "face detected" flag)?
# Run 6: an "unknown sign" class from the out-of-vocabulary sentence signs, for word-mode precision
# (5a setup = no face, seed 1; the class's sampling weight is the variable).
# Run 7: run 6 with the checkpoint chosen on val open-set accuracy (rejection kept improving after
# the Words_200 macro-F1 peak, so run 6 saved early checkpoints).
# Run 8: every learnable word (7b setup, two seeds). Run 9: spelling variants merged (9a), and the side
# camera added (9b, 9c: two seeds).
# Run 10: run 9's side-camera runs without any face (face_drop 1.0). The side views have no face, so in
# run 9 the "face detected" flag told the model which camera it saw, and without face it lost ~0.05.
# Run 11: the final model. Run 10a's recipe (chosen in notebook 05 v7, E6) on train + val signers, saved at
# 10a's best epoch; test signers stay unseen.
# Run 12: 10a without the train signers' signs of the 49 sentences held out as unseen, so word mode and the
# continuous model (notebook 06) are compared on sentences neither has seen (10a had them).
VOCAB_MIN_TRAIN = 10   # a sentence gloss joins the vocabulary with >= 10 signs of train signers (None: 200 words)
SPELLING = {spelling}   # ml/gloss_spelling.csv: one word typed two ways
SIDE = True            # load the side camera (Cam1), shifted by each recording's frame offset (per run: side=)
SIDE_MIN_CORR, SIDE_MIN_MARGIN = 0.3, 0.1     # alignment quality a recording needs (set from 02c1's align.csv)
TRAIN_ON_VAL = False   # final model: val signers' signs become training data (runs then need save_epoch)
EXCLUDE_HELDOUT = True  # leave out the train signs of the held-out sentences (heldout_sentence); vocabulary unchanged
HANDS = ['hands', 'hand_shape', 'hand_vel', 'flags']
NO_FACE = HANDS + ['pose']
BASE = dict(groups=NO_FACE, unknown=4.0, select='open')
RUNS = {
    'side_noface_s1_noheld': dict(BASE, seed=1, side=True, face_drop=1.0),
}
""".replace('{spelling}', json.dumps(spelling(), ensure_ascii=False))),
        code(SETUP),
        code("""
# ONNX export (for the browser) needs onnx; onnxruntime checks the export against PyTorch.
# Installed only if this image lacks them.
import importlib.util
missing = [p for p in ('onnx', 'onnxruntime') if importlib.util.find_spec(p) is None]
if ON_KAGGLE and missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing])
"""),
        *package_cells(TRAIN_MODULES),
        code("""
import pandas as pd, torch
from azsl import train as T
INPUT = '/kaggle/input' if ON_KAGGLE else OUT

def find(pattern):
    return sorted({os.path.dirname(p) for p in glob.glob(os.path.join(INPUT, '**', pattern), recursive=True)})

splits_dir = find('splits/words200_split.csv')[0]
words_dirs = find('landmarks/words200/manifest.csv')
sentence_dirs = find('landmarks/sentences_cam2/shard*/manifest.csv')
print('splits:', splits_dir, '\\nwords:', words_dirs, '\\nsentence shards:', len(sentence_dirs))
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none — this will be slow')
with open(os.path.join(splits_dir, 'benchmark.json'), encoding='utf-8') as f:
    print('split:', json.load(f)['signers'].get('split_groups'))
"""),
        code("""
wsplit = pd.read_csv(os.path.join(splits_dir, 'words200_split.csv'), dtype={'key': str, 'rec_id': str})
tags = pd.read_csv(os.path.join(splits_dir, 'glosses_split.csv'), dtype={'key': str, 'rec_id': str})
import hashlib
with open(os.path.join(splits_dir, 'words200_split.csv'), 'rb') as f:
    CFG['split_id'] = hashlib.sha1(f.read()).hexdigest()[:10]     # recorded in every model manifest
UNKNOWN = any(r.get('unknown') for r in RUNS.values())   # out-of-vocabulary signs as one extra class
tags = tags[tags.has_landmarks & tags.split.notna() & ~tags.dup.astype(bool)]
tags['gloss'] = tags.gloss.replace(SPELLING)
words_list = (T.learnable_vocab(tags, wsplit.gloss.unique(), VOCAB_MIN_TRAIN) if VOCAB_MIN_TRAIN
              else sorted(wsplit.gloss.unique()))
vocab_list = words_list + ([T.UNKNOWN] if UNKNOWN else [])
vocab = {g: i for i, g in enumerate(vocab_list)}
print(f'vocabulary: {len(words_list)} words ({len(set(words_list) - set(wsplit.gloss))} beyond the 200), '
      f'covering {tags.gloss.isin(words_list).mean():.3f} of the annotated signs')
tags = tags[tags.gloss.isin(words_list) | UNKNOWN]
if EXCLUDE_HELDOUT:
    sents = pd.read_csv(os.path.join(splits_dir, 'sentences_split.csv'), dtype={'rec_id': str})
    held = set(sents[sents.heldout_sentence].rec_id)
    drop = (tags.split == 'train') & tags.rec_id.isin(held)
    tags = tags[~drop]
    print(f'held-out sentences: {int(drop.sum())} train signs left out')
if TRAIN_ON_VAL:       # the vocabulary above is still the train signers' one, as in the runs chosen on val
    tags['split'] = tags.split.replace({'val': 'train'})
    wsplit = wsplit[wsplit.split != 'val']
    assert all(r.get('save_epoch') for r in RUNS.values()), 'without val, every run needs a fixed save_epoch'
    print('training on train + val signers:', int((tags.split == 'train').sum()), 'signs')
if SMOKE:
    tags = tags.groupby('split', group_keys=False).apply(lambda d: d.sample(min(len(d), 1500), random_state=0))
    wsplit = wsplit.groupby('split', group_keys=False).apply(lambda d: d.sample(min(len(d), 300), random_state=0))
    CFG.update(epochs=2)
    RUNS = dict(list(RUNS.items())[:2])
AMBIGUOUS = T.ambiguous_oov(tags.gloss.unique(), set(words_list)) if UNKNOWN else []
print(len(AMBIGUOUS), 'out-of-vocabulary glosses look like forms of known words, left out:', AMBIGUOUS[:12])
side = None
if SIDE:
    side_dirs = find('landmarks/sentences_cam1/shard*/manifest.csv')
    al = pd.concat([pd.read_csv(os.path.join(d, 'align.csv'), dtype={'id': str}) for d in side_dirs])
    good = al[(al['corr'] >= SIDE_MIN_CORR) & (al.margin >= SIDE_MIN_MARGIN)]
    print(f'side camera: {len(side_dirs)} shards, {len(al)} recordings, {len(good)} clearly aligned '
          f'(lag {good.lag.min()}…{good.lag.max()}, median corr {good["corr"].median():.3f})')
    side = (side_dirs, dict(zip(good.id, good.lag)))
signs = T.load_sentence_signs(sentence_dirs, tags, vocab, margin=CFG['margin'],
                              unknown=T.UNKNOWN if UNKNOWN else None, exclude=AMBIGUOUS, side=side)
words = T.load_words200(words_dirs, wsplit, vocab)
"""),
        md('### Training — one line per epoch per run (`*` = new best on val macro-F1)'),
        code("""
summary = T.fit_many({name: {**CFG, **change} for name, change in RUNS.items()},
                     signs, words, vocab_list, os.path.join(OUT, 'train'))
"""),
        md('### Test (scored once per run, best-on-val checkpoint)'),
        code("""
import matplotlib.pyplot as plt
summ = pd.read_csv(os.path.join(OUT, 'train', 'summary.csv'))
cols = [c for c in ('run', 'best_epoch', 'n_features', 'params', 'minutes', 'val_words_top1', 'test_words_top1',
                    'test_words_top5', 'test_words_macro_f1', 'test_words_balanced_acc', 'test_signs_top1',
                    'test_signs_macro_f1', 'test_signs200_top1', 'test_signs200_macro_f1', 'test_side_top1',
                    'test_side_macro_f1', 'test_words_majority', 'val_open_acc', 'test_open_acc',
                    'test_unk_rejected', 'test_words_unk_rate', 'test_signs_unk_rate', 'error') if c in summ]
display(summ[cols])
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for name in RUNS:
    path = os.path.join(OUT, 'train', name, 'history.csv')
    if os.path.exists(path) and 'val_words_top1' in pd.read_csv(path, nrows=1):   # a train + val run has no val
        h = pd.read_csv(path)
        ax[0].plot(h.epoch, h.val_words_top1, label=name)
        ax[1].plot(h.epoch, h.val_words_macro_f1, label=name)
ax[0].set_title('val Words_200 top-1'); ax[1].set_title('val Words_200 macro-F1')
ax[0].legend(); plt.show()
"""),
    ])

def nb05():
    sys.path.insert(0, HERE)
    from analyze import load_groups                       # the confirmed word-form families
    families = load_groups(os.path.join(HERE, 'lemma_groups.csv'))
    return notebook([
        md("""
# AzSLD 05 — word mode on held-out sentences

How well does the browser's word mode (`segment.py`, twin of `src/tercume/segment.js`) find the words
of whole sentence recordings by signers it has never seen? Every val / test recording runs through
word mode as in the browser (activity → spans → windows → the word model → non-maximum suppression)
and the detected words are matched with the annotated glosses.

**Model and parameters are chosen on val**: every model in `MODEL_RUNS` × every configuration of
`GRID`, best F1 over all annotated signs (`f1_all`; models with different vocabularies are compared
on the same reference). **Test is scored once per model**, with that model's val-best configuration
(and the chosen model also with the segmentation defaults of `segment.py`).

Version 1 of this notebook (run 5a, the full `segment_eval.GRID`) chose the defaults of segmentation
v2. Versions 3 and 4 compare the models with an "unknown sign" class (runs 6 and 7) on a narrower
grid: the window set and rest line that won in version 1, with a wider range of thresholds. Version 5
adds the models that know every learnable word (runs 8 and 9) and chooses on `f1_all`. From version 5
the annotations' spelling variants of one word (`ml/gloss_spelling.csv`) count as that word, for the
references and for every model's labels.

**Inputs:** the four 02 shards, 03 (splits), 04 (trained models: `train/<run>/best.pt`) and this
notebook's own previous output (models kept from earlier runs of 04: `eval/model/<run>/`).
**Kaggle settings:** CPU (scoring is ~2 ms per window; the GPU queue is not worth it), Internet off.
**Writes** `eval/`: `grid_val.csv`, `summary.json`, `detections_test.csv`, and `model/<run>/` (each
compared model's `best.pt`, `model.onnx`, `manifest.json`).
"""),
        code(f"""
SMOKE = False                  # True: 30 recordings per split, to check the setup
MODEL_RUNS = ['side_noface_s1_noheld']         # run 12: 10a without the held-out sentences (E7)
# (v8: 10a and run 11 = 10a's recipe on train + val signers, run 11 with FIXED = 10a's val choice below)
# (v5 compared 7b / 8a / 8b over all signs; best val f1_all 0.779, 8a, p >= 0.7, overlap <= 25 %)
# (v6 compared 8a / 9a; 9a, spellings merged: val f1_all 0.773)
# (v7 compared 8a / 10a / 10b; val f1_all 0.779 / 0.767 / 0.759; 10a's recipe chosen for angle robustness)
# (v3 compared run 6: unk_w1 / unk_w4 / unk_w8; best val F1 0.736, unk_w8, p >= 0.5, overlap <= 25 %)
# (v4 compared run 7: unk_w1_open / unk_w4_open; best val F1 0.782, unk_w4_open, p >= 0.7, overlap <= 50 %)
FIXED = {{}}   # v8: {{'final_side_noface': dict(rest_line=1.0, lens=[16, 24, 32], min_p=0.7, max_overlap=0.5)}}
# FIXED: run → word-mode settings fixed before; test only (a model trained on val signers has no fair val)
FACE = False                   # the browser runs these models without the face tracker (s2w.js needsFace)
GRID = dict(rest_line=(1.0,), lens=((16, 24, 32),),
            min_p=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), max_overlap=(0.0, 0.25, 0.5))
FAMILIES = {json.dumps(families, ensure_ascii=False)}   # ml/lemma_groups.csv, confirmed
SPELLING = {json.dumps(spelling(), ensure_ascii=False)}   # ml/gloss_spelling.csv: one word typed two ways
"""),
        code(SETUP),
        *package_cells(TRAIN_MODULES + ['segment', 'segment_eval']),
        code("""
import numpy as np, pandas as pd, torch
from azsl import features as FT, segment as SG, segment_eval as E
from azsl.store import LandmarkStore
INPUT = '/kaggle/input' if ON_KAGGLE else OUT

def find(pattern):
    return sorted({os.path.dirname(p) for p in glob.glob(os.path.join(INPUT, '**', pattern), recursive=True)})

splits_dir = find('splits/sentences_split.csv')[0]
sentence_dirs = find('landmarks/sentences_cam2/shard*/manifest.csv')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def find_ckpt(run):
    # 04's output first; a model 04 no longer has is kept in 05's own output; a model from a 04 version that
    # ended in an error comes as a dataset (kaggle_run.py model-dataset), found by its manifest's run name
    hits = glob.glob(os.path.join(INPUT, '**', run, 'best.pt'), recursive=True)
    for mp in [] if hits else glob.glob(os.path.join(INPUT, '**', 'manifest.json'), recursive=True):
        with open(mp, encoding='utf-8') as f:
            if json.load(f).get('run') == run and os.path.exists(os.path.join(os.path.dirname(mp), 'best.pt')):
                hits.append(os.path.join(os.path.dirname(mp), 'best.pt'))
    return sorted(hits, key=lambda p: '/train/' not in p)[0]

models, ckpts = {}, {}
for run in MODEL_RUNS:
    ckpt = find_ckpt(run)
    ckpts[run] = ckpt
    net, vocab, groups = E.load_net(ckpt, device)
    vocab = [SPELLING.get(v, v) for v in vocab]      # a model trained before the merge: both spellings = one word
    models[run] = (E.Scorer(net, groups, FACE, device), vocab)
    print(run, ckpt, '| groups:', groups, '| classes:', len(vocab), '| unknown class:', E.TR.UNKNOWN in vocab)
store = LandmarkStore(sentence_dirs)
idx = store.index
print('device:', device, '| sentence shards:', len(sentence_dirs))
"""),
        code("""
sents = pd.read_csv(os.path.join(splits_dir, 'sentences_split.csv'), dtype={'rec_id': str})
tags = pd.read_csv(os.path.join(splits_dir, 'glosses_split.csv'), dtype={'key': str, 'rec_id': str})
tags = tags[tags.start.notna() & tags.end.notna()]
tags['gloss'] = tags.gloss.replace(SPELLING)
anns = {rid: list(zip(t.start, t.end, t.gloss)) for rid, t in tags.groupby('rec_id')}   # in vocabulary: per model

def recs(split):
    r = sents[(sents.split == split) & sents.rec_id.isin(idx.index)].rec_id.tolist()
    r = r[:30] if SMOKE else r
    return [(rid, float(idx.at[rid, 'width']) / float(idx.at[rid, 'height'])) for rid in r]

def load(rid):
    return FT.reduce(store.get(rid, dtype=np.float32))

counts = {s: {'recordings': len(recs(s)), 'annotated': int(tags[tags.split == s].shape[0]),
              **{f'in_vocab {run}': int(tags[(tags.split == s) & tags.gloss.isin(vocab)].shape[0])
                 for run, (_, vocab) in models.items()}}
          for s in ('val', 'test')}
counts
"""),
        md('### Val: every model × every configuration'),
        code("""
rows, val_t, val_oracle = [], {}, {}
for run, (scorer, vocab) in models.items():
    if run in FIXED:                           # its val signers were training data
        continue
    val_t[run], val_oracle[run], _ = E.evaluate(load, recs('val'), anns, scorer, vocab, FAMILIES,
                                                cfgs=list(E.configs(GRID)))
    o = val_oracle[run]
    print(run, f'oracle boundaries, val: top-1 {o["top1"] / o["n"]:.4f} of its words, '
               f'{o["top1"] / o["n_all"]:.4f} of all signs')
    rows += [{'model': run, 'rest_line': k[0], 'lens': ' '.join(map(str, k[1])) or 'whole span', 'min_p': k[2],
              'max_overlap': k[3], **t.metrics()} for k, t in val_t[run].items()]
grid = pd.DataFrame(rows).sort_values('f1_all', ascending=False)
os.makedirs(os.path.join(OUT, 'eval'), exist_ok=True)
grid.to_csv(os.path.join(OUT, 'eval', 'grid_val.csv'), index=False)
grid.groupby('model').head(3)
"""),
        code("""
def as_cfg(row):
    lens = () if row['lens'] == 'whole span' else tuple(int(x) for x in str(row['lens']).split())
    return {**SG.SEGMENT, 'rest_line': float(row['rest_line']), 'lens': lens,
            'min_p': float(row['min_p']), 'max_overlap': float(row['max_overlap'])}

best_model = grid.iloc[0]['model']
best_cfg = {run: as_cfg(g.iloc[0]) for run, g in grid.groupby('model')}      # each model's val-best configuration
best_cfg.update({run: {**SG.SEGMENT, **p, 'lens': tuple(p['lens'])} for run, p in FIXED.items()})
chosen = {'chosen on val (F1)': best_cfg[best_model],
          f'segment.py v{SG.SEGMENT_VERSION} defaults': dict(SG.SEGMENT)}
best_model, {r: {k: c[k] for k in ('rest_line', 'lens', 'min_p', 'max_overlap')} for r, c in best_cfg.items()}
"""),
        md('### Test: scored once, with the model and configurations fixed on val'),
        code("""
test_of, kept = {}, None
for run, (scorer, vocab) in models.items():       # once per model, at its own val choice
    cfgs = list(chosen.values()) if run == best_model else [best_cfg[run]]
    t, o, k = E.evaluate(load, recs('test'), anns, scorer, vocab, FAMILIES, cfgs=cfgs, keep=best_cfg[run])
    test_of[run] = (t, o)
    kept = k if run == best_model else kept
test_t, test_oracle = test_of[best_model]
val_of = lambda p: val_t[best_model][E.config_key(p)].metrics() if E.config_key(p) in val_t[best_model] else None
plain = lambda p: {k: list(v) if isinstance(v, tuple) else v for k, v in p.items()}   # noqa: E731
summary = {'models': MODEL_RUNS, 'chosen_model': best_model, 'chosen_by': 'val f1_all', 'face': FACE,
           'segment_version': SG.SEGMENT_VERSION, 'checkpoints': ckpts,
           'grid': {k: [list(x) if isinstance(x, tuple) else x for x in v] for k, v in GRID.items()}, 'counts': counts,
           'oracle_top1': {**{f'val {r}': o['top1'] / o['n'] for r, o in val_oracle.items()},
                           **{f'val_all {r}': o['top1'] / o['n_all'] for r, o in val_oracle.items()},
                           **{f'test {r}': o['top1'] / o['n'] for r, (_, o) in test_of.items()},
                           **{f'test_all {r}': o['top1'] / o['n_all'] for r, (_, o) in test_of.items()},
                           'test_merged': test_oracle['top1_m'] / test_oracle['n']},
           'fixed': {r: plain(p) for r, p in FIXED.items()},
           'per_model': {r: {'params': plain(best_cfg[r]),
                             'val': val_t[r][E.config_key(best_cfg[r])].metrics() if r in val_t else None,
                             'test': test_of[r][0][E.config_key(best_cfg[r])].metrics()} for r in models},
           'best_val_per_model': {r: g.iloc[0][['min_p', 'max_overlap', 'precision', 'recall', 'f1', 'recall_all',
                                                'f1_all', 'wer', 'wer_all']].to_dict()
                                  for r, g in grid.groupby('model')},
           'configs': {name: {'params': plain(p), 'val': val_of(p), 'test': test_t[E.config_key(p)].metrics()}
                       for name, p in chosen.items()}}
with open(os.path.join(OUT, 'eval', 'summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
pd.DataFrame(kept, columns=['rec_id', 'start0', 'end0', 'word']).to_csv(
    os.path.join(OUT, 'eval', 'detections_test.csv'), index=False)
# the compared models travel with their evaluation (04's output is replaced by its next version)
import shutil
for run, ck in ckpts.items():
    dst = os.path.join(OUT, 'eval', 'model', run)
    os.makedirs(dst, exist_ok=True)
    for f in ('best.pt', 'model.onnx', 'manifest.json', 'metrics.json', 'vocab.json'):
        src = os.path.join(os.path.dirname(ck), f)
        if os.path.exists(src) and os.path.abspath(os.path.dirname(src)) != os.path.abspath(dst):
            shutil.copy2(src, dst)
cols = ['precision', 'recall_all', 'f1_all', 'wer_all', 'recall', 'f1', 'wer', 'f1_merged',
        'start_err_frames', 'end_err_frames', 'detections']
pd.DataFrame({**{f'{r} (val-best) | test': {c: round(summary['per_model'][r]['test'][c], 4) for c in cols}
                 for r in models},
              **{f'{best_model}, {name} | test': {c: round(summary['configs'][name]['test'][c], 4) for c in cols}
                 for name in chosen}}).T
"""),
    ])


def nb06():
    return notebook([
        md("""
# AzSLD 06 — continuous recognition: a sentence in one run

Word mode (05) cuts the landmark stream into windows and asks the isolated-sign model about each one. This
notebook trains a model that labels **every frame** of a sentence recording: which word, or no sign
(`<blank>`: rests and transitions). It learns from whole sentences, so it uses all frames (the half
between signs that the isolated-sign model never saw), the sentence context, and both cameras.

- **Labels:** the vocabulary of run 10 (every gloss with ≥ 10 train signs, spellings merged) with the forms
  of one sign merged (`ml/lemma_groups.csv` confirmed, `ml/same_sign.csv`); rare glosses → `<unk>`.
- **Train:** whole recordings of train signers, frontal + side camera, random 48–160-frame windows. The
  sentences reserved as unseen (`heldout_sentence`, 49 of 492) are left out of training, so the test
  can say how it does on sentences it never saw.
- **Checkpoint:** best val word F1 (default decoding) by each run's `select`: over all annotated signs, or
  on the unseen sentences' recordings, all signs (`f1_unseen`) or only words with training signs outside
  them (`f1_unseen_learnable`; detections on the other signs are then neither right nor wrong).
- **Evaluation as in the browser:** the model runs on the last 96 frames every 8 frames; a frame is fixed
  24 frames (0.8 s) after it. Decoding settings chosen on val; test scored once (all test recordings,
  the unseen sentences, the side camera).

**Inputs:** 02 shards, 02c1 shards, 03, 04 (run 10a, to start from). **Kaggle:** GPU T4, Internet on.
**Writes** `cont/<run>/`: history, metrics (val grid, test), best.pt, model.onnx, manifest.json.
"""),
        code("""
SMOKE = False          # True: a few recordings, 2 epochs
HANDS = ['hands', 'hand_shape', 'hand_vel', 'flags']
NO_FACE = HANDS + ['pose']
BASE = dict(groups=NO_FACE, seed=1, face_drop=1.0, epochs=60, repeats=2, batch=32, workers=4)
INIT_RUN = 'side_noface_s1'                    # run 10a: isolated signs, train signers only
# v2: seq_scratch / seq_init (every layer from 10a): test F1 0.909 / 0.908, but 0.584 / 0.609 on the test
# recordings of sentences left out of training (word mode: 0.776).
# v3: seq_splice / seq_short from 10a, chosen on val unseen sentences: both chose epoch 1. But 10a was
# trained on the held-out sentences' signs, and the score counted words with no training sign outside them.
# v5: from scratch (nothing from 10a), epoch and decoding chosen on val unseen sentences, learnable words only;
# word mode is compared with run 12 (10a without the held-out sentences).
# v5 runs: seq_plain (v2's recipe), seq_splice (splice=0.5), seq_short (crop (24, 64), 48-frame stream), all with
# select='f1_unseen_learnable'. Val unseen learnable F1 0.747 / 0.811 / 0.758 → seq_splice, epoch 59.
# v6: the final model. seq_splice's recipe on train + val signers and every sentence, saved at epoch 59, with
# its val decoding; test signers stay unseen (their sentences are no longer unseen):
#   'final_splice': dict(BASE, splice=0.5, save_epoch=59, decode=dict(smooth=3, min_p=0.3, min_len=10, gap=2))
# v7 (CPU, no training): fingerspelled letters and digits are short (median 13 frames, 28 % under 10) and
# min_len 10 drops them. min_len_short (single-character labels) was chosen on all val recordings with
# seq_splice (v5: never saw the val signers), v5's other decode settings kept (smooth 3, p >= 0.3, min_len 10,
# gap 2): min_len_short 6. final_splice (the browser model) was then scored on test with it:
#   'seq_splice': dict(ckpt='seq_splice', select='f1_all'),
#   'final_splice': dict(ckpt='final_splice', decode_from='seq_splice', unseen=False),
# v8 (GPU; its first push was refused, the weekly quota was used up): more new sentences, compared with
# seq_splice (v5, decoded as in v7) under v5's protocol. MƏN is a third of all signs, so uniformly drawn spliced signs repeat a
# few words; splice_power 0.5 draws them like the loss weights. Val unseen learnable F1 0.797 / 0.797 against
# seq_splice's 0.811, val F1 0.860 / 0.882 against 0.890: seq_splice's recipe stays (E7):
#   'splice_75': dict(BASE, splice=0.75, select='f1_unseen_learnable'),
#   'splice_bal': dict(BASE, splice=0.5, splice_power=0.5, select='f1_unseen_learnable'),
# v9 (CPU, no training): the browser model at lower camera frame rates. Hand and body trackers may run at
# 10-20 fps on weaker devices; the browser fills the frames back to 30 fps by linear interpolation. Every test
# recording as a camera at that rate would give it (continuous.camera_resample), with the model's decoding.
#   {f'final_fps{f}': dict(ckpt='final_splice', fixed_decode=FINAL_DECODE, fps=f, unseen=False) for f in (30, 20, 15, 10)}
# (v9 failed on Kaggle before its first cell; v10 includes its question)
# v10 = v12 (GPU; versions 9 and 10 were killed by Kaggle without output, 11 was a smoke run, 12 ran through):
# training windows that come as a slower camera gives them (fps_aug: half of them, at 8-30 fps,
# refilled to 30 fps as the browser does), under v5's protocol. seq_splice (v5's model, loaded) is the
# baseline; both are also scored at 15 and 10 fps (FPS_CHECKS) with the decoding chosen at 30.
# final_splice (the browser model) is scored on test at the same rates, with its decoding. Result: from 30 to
# 10 fps final_splice loses 0.008 test F1; splice_fps is level with seq_splice at every rate on val (E7):
#   'splice_fps': dict(BASE, splice=0.5, fps_aug=0.5, select='f1_unseen_learnable'),
# v14 (GPU; v13 was its CPU smoke run): letters. In sentences they are rarely taken for another letter (0.5 % of test letters) but lost in
# fast spelled runs (A-D-A, M-E-S-S-İ): 5.4 % absorbed by a neighbour, 2.6 % missed. AzSLD's fingerspelling photos
# only teach hand shapes, so instead spell = P(a sample is 3-8 letter signs of other recordings in a row), taken
# before the splice draw. Scored also on 'letters' (recordings with letters, letter signs only). Result: val
# letters F1 0.832 / 0.831 against seq_splice's 0.853, val F1 0.884 / 0.882 against 0.890: no (E7).
FPS_CHECKS = ()
RUNS = {
    'seq_splice': dict(ckpt='seq_splice'),
    'splice_spell15': dict(BASE, splice=0.5, spell=0.15, select='f1_unseen_learnable'),
    'splice_spell30': dict(BASE, splice=0.5, spell=0.3, select='f1_unseen_learnable'),
}
VOCAB_MIN_TRAIN = 10
SPELLING = __SPELLING__
MERGE = __MERGE__
SIDE_MIN_CORR, SIDE_MIN_MARGIN = 0.3, 0.1
EXCLUDE_HELDOUT = True                         # unseen sentences stay out of training
TRAIN_ON_VAL = False                           # final model: val signers too (runs need save_epoch and decode)
DECODE_GRID = dict(smooth=(1, 3, 5, 9), min_p=(0.3, 0.4, 0.5, 0.6, 0.7), min_len=(4, 6, 8, 10), gap=(2, 4))
DECODE_GRID['min_len_short'] = (6,)            # v7's choice for letters and digits
LOOKAHEADS = (12, 40)                          # val only: shorter / longer delay than the browser's
                                               # (must stay below window - stride of every run)
""".replace('__SPELLING__', json.dumps(spelling(), ensure_ascii=False)).replace(
            '__MERGE__', json.dumps(sign_merge(), ensure_ascii=False))),
        code(SETUP),
        code("""
import importlib.util
missing = [p for p in ('onnx', 'onnxruntime') if importlib.util.find_spec(p) is None]
if ON_KAGGLE and missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing])
"""),
        *package_cells(TRAIN_MODULES + ['segment', 'segment_eval', 'continuous']),
        code("""
import pandas as pd, numpy as np, torch
from azsl import train as T, continuous as CT, features as FT
INPUT = '/kaggle/input' if ON_KAGGLE else OUT

def find(pattern):
    return sorted({os.path.dirname(p) for p in glob.glob(os.path.join(INPUT, '**', pattern), recursive=True)})

splits_dir = find('splits/sentences_split.csv')[0]
sentence_dirs = find('landmarks/sentences_cam2/shard*/manifest.csv')
side_dirs = find('landmarks/sentences_cam1/shard*/manifest.csv')
init = sorted(glob.glob(os.path.join(INPUT, '**', INIT_RUN, 'best.pt'), recursive=True))
for r in RUNS.values():
    if r.get('init'):
        r['init'] = init[0]
print('splits:', splits_dir, '| frontal shards:', len(sentence_dirs), '| side shards:', len(side_dirs), '| init:', init[:1])
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
"""),
        code("""
wsplit = pd.read_csv(os.path.join(splits_dir, 'words200_split.csv'), dtype={'key': str, 'rec_id': str})
tags = pd.read_csv(os.path.join(splits_dir, 'glosses_split.csv'), dtype={'key': str, 'rec_id': str})
sents = pd.read_csv(os.path.join(splits_dir, 'sentences_split.csv'), dtype={'rec_id': str})
import hashlib
with open(os.path.join(splits_dir, 'words200_split.csv'), 'rb') as f:
    split_id = hashlib.sha1(f.read()).hexdigest()[:10]
merge = lambda g: MERGE.get(SPELLING.get(g, g), SPELLING.get(g, g))      # noqa: E731
tags = tags[tags.has_landmarks & tags.split.notna() & tags.start.notna() & tags.end.notna()].copy()
tags['gloss'] = tags.gloss.map(merge)
core = {merge(g) for g in wsplit.gloss.unique()}
words = T.learnable_vocab(tags, core, VOCAB_MIN_TRAIN)
vocab_list = [CT.BLANK] + words + [T.UNKNOWN]
index = {g: i for i, g in enumerate(vocab_list)}
exclude = T.ambiguous_oov(tags.gloss.unique(), set(words))
anns = {rid: list(zip(t.start, t.end, t.gloss)) for rid, t in tags.groupby('rec_id')}
recs = sents[sents.split.notna() & sents.rec_id.isin(anns)][['rec_id', 'split', 'heldout_sentence']].copy()
if TRAIN_ON_VAL:
    recs['split'] = recs.split.replace({'val': 'train'})
    assert all(r.get('save_epoch') and r.get('decode') for r in RUNS.values()), 'final runs need save_epoch and decode'
if EXCLUDE_HELDOUT:
    recs = recs[~((recs.split == 'train') & recs.heldout_sentence)]
if all(r.get('ckpt') for r in RUNS.values()):   # evaluation only: no training recordings needed
    recs = recs[recs.split != 'train']
unseen = set(recs[recs.split.isin(['val', 'test']) & recs.heldout_sentence].rec_id)
# words a model can know in an unseen sentence: >= 10 train signs outside the held-out sentences
outside = tags[(tags.split == 'train') & ~tags.rec_id.isin(set(sents[sents.heldout_sentence].rec_id))]
learnable = set(outside.gloss.value_counts()[lambda c: c >= 10].index)
if SMOKE:
    recs = recs.groupby('split', group_keys=False).apply(lambda d: d.sample(min(len(d), 60), random_state=0))
    for r in RUNS.values():
        r.update(epochs=2)
    DECODE_GRID = dict(smooth=(1, 5), min_p=(0.5,), min_len=(6,), gap=(3,))
print(f'{len(words)} words ({len(MERGE)} glosses merged into their sign), {len(exclude)} ambiguous glosses not scored;',
      'recordings:', recs.split.value_counts().to_dict(), '| unseen-sentence val + test recordings:', len(unseen),
      f'| {len(learnable)} glosses learnable outside them')
al = pd.concat([pd.read_csv(os.path.join(d, 'align.csv'), dtype={'id': str}) for d in side_dirs])
good = al[(al['corr'] >= SIDE_MIN_CORR) & (al.margin >= SIDE_MIN_MARGIN)]
banks = CT.load_recordings(sentence_dirs, recs, anns, index, unknown=T.UNKNOWN, exclude=exclude,
                           side=(side_dirs, dict(zip(good.id, good.lag))))
banks['val_unseen'] = CT.subset(banks['val'], unseen)          # per-epoch score on sentences never trained on
SUBSETS = {'unseen_sentences': (unseen, None), 'unseen_learnable': (unseen, learnable)}
letters = {g for g in words if len(g) == 1}                 # fingerspelled letters and digits
SUBSETS['letters'] = (set(tags[tags.gloss.isin(letters)].rec_id), letters)   # their recordings, letters only
"""),
        md('### Training and evaluation, one run after another'),
        code("""
import time
rows, decided = [], {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def find_model(run):
    # a run folder (best.pt + manifest.json) among the inputs, by the manifest's run name
    for mp in glob.glob(os.path.join(INPUT, '**', 'manifest.json'), recursive=True):
        with open(mp, encoding='utf-8') as f:
            if json.load(f).get('run') == run and os.path.exists(os.path.join(os.path.dirname(mp), 'best.pt')):
                return os.path.join(os.path.dirname(mp), 'best.pt')
    raise FileNotFoundError(f'no saved model {run} among the inputs')

for name, change in RUNS.items():
    out = os.path.join(OUT, 'cont', name)
    os.makedirs(out, exist_ok=True)
    cfg = {**change, 'split_id': split_id}
    print(f'\\n===== {name}: {cfg}')
    t0 = time.time()
    if cfg.get('ckpt'):        # evaluation only: a model saved before (a dataset from kaggle_run.py model-dataset)
        net, saved, labels, epoch = CT.load_checkpoint(find_model(cfg['ckpt']), device)
        assert labels == vocab_list, 'the saved model has other labels'
        cfg = {**saved, **cfg}
        res = {'best_epoch': epoch, 'params': sum(p.numel() for p in net.parameters()),
               'n_features': len(FT.columns(cfg['groups'])), 'cfg': cfg}
    else:
        net, res = CT.fit(cfg, banks, vocab_list, anns, out, unseen_keep=learnable)
    # decoding fixed before: a final model (TRAIN_ON_VAL, cfg['decode']) or chosen by an earlier run here
    fixed = (decided[cfg['decode_from']] if cfg.get('decode_from') else cfg.get('fixed_decode')
             or (cfg['decode'] if TRAIN_ON_VAL else None))
    choose = {'f1_unseen': 'f1_unseen_sentences'}.get(cfg.get('select'), cfg.get('select', 'f1_all'))
    ev = CT.evaluate(net, banks, anns, vocab_list, cfg['groups'], cfg['face_drop'] < 1, net.head.weight.device,
                     DECODE_GRID, stream=cfg.get('stream') or CT.STREAM,
                     subsets=SUBSETS if EXCLUDE_HELDOUT and cfg.get('unseen', True) else {},
                     lookaheads=() if fixed or cfg.get('ckpt') else LOOKAHEADS, choose=choose, fixed=fixed,
                     fps=cfg.get('fps'), fps_checks=FPS_CHECKS)
    decode_p = decided[name] = ev['decode']
    pd.DataFrame(ev.pop('test_detections'), columns=['rec_id', 'start0', 'end0', 'word', 'p']).to_csv(
        os.path.join(out, 'detections_test.csv'), index=False)
    res.update({k: v for k, v in ev.items() if k != 'val_grid'})
    if ev['val_grid']:
        pd.DataFrame(ev['val_grid']).to_csv(os.path.join(out, 'grid_val.csv'), index=False)
    res['minutes'] = round((time.time() - t0) / 60, 1)
    if not cfg.get('ckpt'):    # a loaded model keeps its own ONNX file and manifest
        ok = T.export_onnx(net, os.path.join(out, 'model.onnx'), n_in=res['n_features'])
        CT.write_manifest(out, {**CT.DEFAULTS, **cfg}, vocab_list, res, res['n_features'], ok, decode_p,
                          stream=cfg.get('stream') or CT.STREAM)
    with open(os.path.join(out, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    row = {'run': name, 'best_epoch': res['best_epoch'], 'minutes': res['minutes']}
    for k, v in res.items():
        if isinstance(v, dict) and 'f1_all' in v:
            row.update({f'{k}_{m}': round(v[m], 4) for m in ('precision', 'recall_all', 'f1_all', 'wer_all', 'frame_acc')})
    rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'cont', 'summary.csv'), index=False)
pd.DataFrame(rows).set_index('run').T
"""),
    ])


def nb07():
    return notebook([
        md("""
# AzSLD 07 — fingerspelled letters for the letter mode

The app's letter mode types a word letter by letter with the hand tracker alone. Its model so far (a dense
network on one hand's image coordinates) types static letters only while the hand is still, and its own
evaluation on AzSLD's photos is 78 % on static letters and 2 % on the 8 moving ones. This notebook trains a
letter model on the landmark pipeline (`azsl/letters.py`) from every AzSLD source of letters:

- **sentences:** the letter signs annotated in the sentence recordings, frontal and side camera, the signer
  split (an "O" counts as a letter only next to another spelled letter: alone it is the pronoun "o");
- **photos:** AzSLD_Fingerspelling's static letters (24 letters × 40 photos), split by contributor;
- **clips:** its moving letters (8 letters, frames sampled from clips, filled back in between);
- **`<none>`:** gaps between annotated signs, so the model can say that no letter is shown.

The model reads only the hand (no body, no face: most photos have none, and the browser runs no body
tracker here). The epoch is chosen on val macro-F1 over the letters; test is scored once, per source, for
the new models and for the old one on the same samples.

**Inputs:** 01 (fingerspelling landmarks), 02 shards (both cameras), 03 (splits), the old model as a dataset.
**Kaggle:** GPU T4. **Writes** `letters/<run>/`: history, metrics, predictions, best.pt, model.onnx, manifest.json.
"""),
        code("""
SMOKE = False          # True: a few hundred samples, 2 epochs
SPELLING = __SPELLING__
SIDE_MIN_CORR, SIDE_MIN_MARGIN = 0.3, 0.1
NONE_PER_LETTER = 1.0   # <none> samples per letter sign of the sentences, per split
TRAIN_ON_VAL = False    # final model: val signers / contributors too (runs then need save_epoch)
# v1: hand shape + movement only, or with the hand's position in the image and the presence flags
RUNS = {
    'hand': dict(seed=1, groups=['hand_shape', 'hand_vel']),
    'hand_pos': dict(seed=1, groups=['hands', 'hand_shape', 'hand_vel', 'flags']),
}
""".replace('__SPELLING__', json.dumps(spelling(), ensure_ascii=False))),
        code(SETUP),
        code("""
import importlib.util
missing = [p for p in ('onnx', 'onnxruntime') if importlib.util.find_spec(p) is None]
if ON_KAGGLE and missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing])
"""),
        *package_cells(TRAIN_MODULES + ['letters']),
        code("""
import numpy as np, pandas as pd, torch
from azsl import train as T, letters as L
INPUT = '/kaggle/input' if ON_KAGGLE else OUT

def find(pattern):
    return sorted({os.path.dirname(p) for p in glob.glob(os.path.join(INPUT, '**', pattern), recursive=True)})

splits_dir = find('splits/glosses_split.csv')[0]
sentence_dirs = find('landmarks/sentences_cam2/shard*/manifest.csv')
side_dirs = find('landmarks/sentences_cam1/shard*/manifest.csv')
fs_dirs = find('landmarks/fingerspelling/manifest.csv')
old_meta = [p for p in glob.glob(os.path.join(INPUT, '**', 'model.json'), recursive=True)
            if os.path.exists(os.path.join(os.path.dirname(p), 'model.bin'))]
import hashlib
with open(os.path.join(splits_dir, 'words200_split.csv'), 'rb') as f:
    split_id = hashlib.sha1(f.read()).hexdigest()[:10]
print('splits:', splits_dir, '| frontal shards:', len(sentence_dirs), '| side shards:', len(side_dirs),
      '| fingerspelling:', fs_dirs, '| old model:', old_meta[:1])
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
"""),
        code("""
rng = np.random.default_rng(0)
tags = pd.read_csv(os.path.join(splits_dir, 'glosses_split.csv'), dtype={'key': str, 'rec_id': str})
tags = tags[tags.has_landmarks & tags.split.notna() & tags.start.notna() & tags.end.notna()
            & ~tags.dup.astype(bool)].copy()
tags['gloss'] = tags.gloss.replace(SPELLING)
fsplit = pd.read_csv(os.path.join(splits_dir, 'fingerspelling_split.csv'))
if TRAIN_ON_VAL:
    tags['split'] = tags.split.replace({'val': 'train'})
    fsplit['split'] = fsplit.split.replace({'val': 'train'})
    assert all(r.get('save_epoch') for r in RUNS.values()), 'without val, every run needs a fixed save_epoch'
lt = L.letter_tags(tags)
if SMOKE:
    lt = lt.groupby('split', group_keys=False).apply(lambda d: d.sample(min(len(d), 150), random_state=0))
    fsplit = fsplit.groupby(['kind', 'split'], group_keys=False).apply(lambda d: d.head(80))
    for r in RUNS.values():
        r.update(epochs=2)
n_letters = lt.split.value_counts()
nt = L.none_tags(tags, {s: int(n * NONE_PER_LETTER) for s, n in n_letters.items()}, rng)
print('letter signs in the sentences:', n_letters.to_dict(), '| <none> gaps:', nt.split.value_counts().to_dict())
index = {g: i for i, g in enumerate(L.LABELS)}
al = pd.concat([pd.read_csv(os.path.join(d, 'align.csv'), dtype={'id': str}) for d in side_dirs])
good = al[(al['corr'] >= SIDE_MIN_CORR) & (al.margin >= SIDE_MIN_MARGIN)]
sb = L.load_sentences(sentence_dirs, pd.concat([lt, nt], ignore_index=True), index, L.DEFAULTS['margin'],
                      side=(side_dirs, dict(zip(good.id, good.lag))))
pb = L.load_photos(fs_dirs, fsplit, index)
cb = L.load_clips(fs_dirs, fsplit, index)
banks = {}
for s in ('train', 'val', 'test'):
    b = T.Bank.merge(T.Bank.merge(sb[s], sb.get(f'{s}_side', T.Bank().done())), T.Bank.merge(pb[s], cb[s]))
    banks[s] = b
    src = pd.Series([L.source_of(i) for i in b.ids])
    print(s, len(b), src.value_counts().to_dict(), '| letters with samples:',
          len(set(b.labels.tolist()) - {index[L.NONE]}))
"""),
        md('### Training, then val and test per source; the old model on the same samples'),
        code("""
import time
rows = []
for name, change in RUNS.items():
    out = os.path.join(OUT, 'letters', name)
    cfg = {**change, 'split_id': split_id}
    print(f'\\n===== {name}: {cfg}')
    t0 = time.time()
    net, res = L.fit(cfg, banks, out)
    device = next(net.parameters()).device
    for part in ('val', 'test'):
        if len(banks[part]):
            p = L.predict(net, banks[part], device, res['cfg']['groups'])
            res[part] = L.letter_scores(p, banks[part].labels, banks[part].ids)
            np.savez_compressed(os.path.join(out, f'predictions_{part}.npz'), probs=p, labels=banks[part].labels,
                                ids=np.array(banks[part].ids))
            print(part, {k: (round(v['top1'], 4), round(v['macro_f1'], 4)) if isinstance(v, dict) else round(v, 4)
                         for k, v in res[part].items()})
    res['minutes'] = round((time.time() - t0) / 60, 1)
    ok = T.export_onnx(net, os.path.join(out, 'model.onnx'), n_in=res['n_features'])
    L.write_manifest(out, {**L.DEFAULTS, **res['cfg']}, res, ok)
    with open(os.path.join(out, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    row = {'run': name, 'best_epoch': res['best_epoch'], 'minutes': res['minutes']}
    for part in ('val', 'test'):
        for k, v in res.get(part, {}).items():
            row.update({f'{part}_{k}_{m}': round(v[m], 4) for m in ('top1', 'macro_f1')} if isinstance(v, dict)
                       else {f'{part}_{k}': round(v, 4)})
    rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'letters', 'summary.csv'), index=False)
if old_meta:
    old = L.OldModel(old_meta[0], os.path.join(os.path.dirname(old_meta[0]), 'model.bin'))
    res, row = {}, {'run': 'old_model'}
    for part in ('val', 'test'):
        if len(banks[part]):
            res[part] = L.letter_scores(old.predict_bank(banks[part]), banks[part].labels, banks[part].ids)
            row.update({f'{part}_{k}_{m}': round(v[m], 4) for k, v in res[part].items() if isinstance(v, dict)
                        for m in ('top1', 'macro_f1')})
            print('old model', part, {k: (round(v['top1'], 4), round(v['macro_f1'], 4))
                                      for k, v in res[part].items() if isinstance(v, dict)})
    os.makedirs(os.path.join(OUT, 'letters', 'old_model'), exist_ok=True)
    with open(os.path.join(OUT, 'letters', 'old_model', 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'letters', 'summary.csv'), index=False)
pd.DataFrame(rows).set_index('run').T
"""),
    ])


def main():
    out = os.path.join(HERE, 'notebooks')
    os.makedirs(out, exist_ok=True)
    books = {'01_index_words200_fingerspelling.ipynb': nb01(), '03_split_benchmark.ipynb': nb03(),
             '04_train_words.ipynb': nb04(), '05_eval_word_mode.ipynb': nb05(),
             '06_train_continuous.ipynb': nb06(), '07_train_letters.ipynb': nb07()}
    for k in range(NUM_SHARDS):
        books[f'02_sentences_shard{k}.ipynb'] = nb02(k)
        books[f'02_sentences_cam1_shard{k}.ipynb'] = nb02(k, 'Cam1')
    for name, nb in books.items():
        for i, c in enumerate(nb.cells):          # stable ids: a rebuild without changes is a no-op in git
            c['id'] = f'{os.path.splitext(name)[0][:20]}-{i:02d}'
        nbf.validate(nb)
        with open(os.path.join(out, name), 'w', encoding='utf-8', newline='\n') as f:
            nbf.write(nb, f)
        print('wrote', name)


if __name__ == '__main__':
    main()
