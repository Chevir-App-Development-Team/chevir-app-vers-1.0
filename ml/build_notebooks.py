"""Generates the Kaggle notebooks in ml/notebooks/ from the azsl package.

Each notebook is self-contained: the package is written out with %%writefile cells, so
nothing has to be uploaded to Kaggle besides the .ipynb itself. Edit azsl/*.py, then:
    python ml/build_notebooks.py
"""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = ['__init__', 'zenodo', 'landmarks', 'store', 'sources', 'split', 'pipeline']
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


def package_cells():
    cells = [md('### The `azsl` package\nGenerated from `ml/azsl/` — edit it there, not here.')]
    for m in MODULES:
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


def nb02(shard):
    return notebook([
        md(f'''
# AzSLD 02 — Sentences (frontal camera) → landmarks · shard {shard} of {NUM_SHARDS}

Reads the frontal (`Cam2`) recordings one at a time from the 43.9 GB zip by HTTP byte range —
nothing big is downloaded — and keeps only landmarks. This shard takes every {NUM_SHARDS}th
recording starting at #{shard}. Each recording also gets one face embedding, used **only** by
notebook 03 to group recordings by signer (AzSLD has no signer ids); it is not a training feature.

**Kaggle settings:** Accelerator **None**, Internet **On**. Run with **Save Version → Save & Run All**.
The four shard notebooks can run at the same time.

**Takes** roughly 2–3 h. **Writes** `landmarks/sentences_cam2/shard{shard:02d}/`.

If it stops at the time budget (`complete: False` at the end): add this notebook's own output
as an input and run again — finished recordings are skipped.
'''),
        code(f'''
SHARD = {shard}
NUM_SHARDS = {NUM_SHARDS}
CAMERA = 'Cam2'        # Cam2 = frontal. 'Cam1' (side view) only if you want it as extra data later.
SMOKE = False          # True: 3 recordings, to check the setup
WORKERS = None
TIME_BUDGET_H = 11.0
'''),
        code(SETUP),
        *package_cells(),
        code(INSTALL),
        code(MODELS_CELL.format(sface=True, name=f'02_shard{shard}')),
        code('''
# an earlier run of this same shard, attached as input → resume from it
prev = glob.glob(f'/kaggle/input/**/landmarks/sentences_{CAMERA.lower()}/shard{SHARD:02d}', recursive=True)
print('resume from:', prev or 'nothing')
summary = pipeline.run_sentences(OUT, paths, SHARD, NUM_SHARDS, CAMERA, WORKERS,
                                 limit=3 if SMOKE else None, time_budget_h=TIME_BUDGET_H, resume_from=prev)
summary
'''),
        code('''
from azsl.store import LandmarkStore
m = LandmarkStore(os.path.join(OUT, 'landmarks', f'sentences_{CAMERA.lower()}', f'shard{SHARD:02d}')).manifest
print(f'{len(m)} recordings, {m.n_frames.sum():,} frames')
print(m[['n_frames', 'hand_rate', 'two_hand_rate', 'pose_rate', 'face_rate', 'frontal']].describe().round(3))
print('not frontal (> 0.35):', int((m.frontal > 0.35).sum()))
'''),
    ])


def nb03():
    return notebook([
        md('''
# AzSLD 03 — signer-independent split + benchmark (Phase 0)

**Inputs:** add the outputs of notebook 01 and of all four 02 shards
(*Add Input → Your Work → Notebooks*). **Kaggle settings:** Accelerator **None**; Internet not needed.

AzSLD has no signer ids, so recordings are grouped into pseudo-signers: sessions from the
recording timestamps, then face embeddings (see `azsl/split.py`). One signer → train/val/test
assignment is shared by every task, so a test signer is never seen in training.

**Writes** `splits/`: `signers.csv`, `words200_split.csv`, `glosses_split.csv`,
`sentences_split.csv`, `fingerspelling_split.csv`, `benchmark.json`.
Download `splits/` and commit it to `ml/splits/` — that is the frozen benchmark every model is scored on.
'''),
        code('''
SIM = 0.45             # same-signer cosine similarity threshold (SFace); see the checks below
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
                                            os.path.join(OUT, 'splits'), sim=SIM, gap_min=GAP_MIN,
                                            seed=SEED, override_csv=OVERRIDE_CSV)
print(json.dumps(bench['signers'], indent=1, default=str))
'''),
        md('''
### Check the pseudo-signers
Each row should be **one person**. Two people in one row → raise `SIM`; one person spread over
several rows → lower it. `nearest_other_signer_sim` above should stay well below `SIM`, and
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


def main():
    out = os.path.join(HERE, 'notebooks')
    os.makedirs(out, exist_ok=True)
    books = {'01_index_words200_fingerspelling.ipynb': nb01(), '03_split_benchmark.ipynb': nb03()}
    for k in range(NUM_SHARDS):
        books[f'02_sentences_shard{k}.ipynb'] = nb02(k)
    for name, nb in books.items():
        nbf.validate(nb)
        with open(os.path.join(out, name), 'w', encoding='utf-8', newline='\n') as f:
            nbf.write(nb, f)
        print('wrote', name)


if __name__ == '__main__':
    main()
