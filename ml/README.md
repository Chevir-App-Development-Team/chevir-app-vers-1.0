# ml/ — AzSLD → landmarks → signer-independent benchmark

Phases 0–1 of replacing the camera model. The current recognizer (an MLP on raw
coordinates) and the old word model (`Nigar21/WL-AzSL-I3DCNN`, an RGB I3D on 10 words
whose ~0.9 accuracy came from a test set that was 57.5% one class plus a split that
leaked signers) are replaced by small models trained on **MediaPipe landmarks**. Those
models run in the browser next to the existing pipeline. This folder produces the
training data and the frozen benchmark; training (Phase 2) builds on it.

Everything heavy runs on **Kaggle** — nothing large is downloaded locally. The 43.9 GB
Sentences archive is never downloaded at all: Zenodo serves byte ranges, so each
recording is read out of the zip on its own and only its landmarks are kept.

## Running on Kaggle

One-time: a Kaggle account with a **verified phone number** (notebooks need it for
Internet access). Import each `.ipynb` from `notebooks/` (*Create → New Notebook →
File → Import Notebook*), then set **Accelerator: None** and **Internet: On** in the
notebook settings. Start every notebook with **Save Version → Save & Run All (Commit)**
so it keeps running after you close the tab.

| # | Notebook | Takes | Writes |
|---|---|---|---|
| 1 | `01_index_words200_fingerspelling` | ~1–1.5 h | `index/`, `landmarks/words200/`, `landmarks/fingerspelling/` |
| 2 | `02_sentences_shard0` … `shard3` | ~2–3 h each | `landmarks/sentences_cam2/shardNN/` |
| 3 | `03_split_benchmark` | minutes | `splits/` |

- 1 and the four 2s are independent — start them together if Kaggle lets you
  (CPU sessions don't use the weekly GPU quota). If a shard log shows many `429`
  retries, Zenodo is throttling: run the shards one after another instead.
- 3 needs the outputs of 1 and all four 2s: *Add Input → Your Work → Notebooks*.
  No internet needed.
- **Try `SMOKE = True` first** (top cell) — a ~3-minute run on a few files that checks
  the install, models and Zenodo access before you commit hours.
- A run that hits `TIME_BUDGET_H` (11 h) stops cleanly with `complete: False`. Add that
  notebook's own output as an input and run it again; finished items are skipped.
- **Keep the outputs private.** Notebook 02 stores one face embedding and one small face
  crop per recording, used only to group recordings by signer (see below).

When 03 is done, download `splits/` (a few MB) and commit it here as `ml/splits/` —
that is the benchmark every later model is scored on.

## What the extraction produces

Same `.task` models as the browser, verified by md5: `hand_landmarker.task` is
byte-identical to `public/tercume/assets/hand_landmarker.task`. Pose and face are the
`float16/1` releases the browser will load. The hand options match `src/tercume/hands.js`.
Videos run in VIDEO mode (tracking), as the live camera does. Python MediaPipe is pinned to
0.10.14 like `@mediapipe/tasks-vision`; if that build doesn't work on the Kaggle image, a newer
0.10.x is used and recorded in `versions_*.json`.

Per frame (float16, NaN = not detected):

| Field | Shape | |
|---|---|---|
| `hand` | 2×21×3 | image x, y ∈ [0, 1], relative z. **Slot 0 = signer's right hand**, slot 1 = left, decided by the pose wrists (same rule as `tools/build_words.py`), so crossing hands keep their slot |
| `hand_world` | 2×21×3 | metric 3D hand |
| `hand_score` | 2 | handedness score of the detection in each slot |
| `pose` | 33×4 | image coords + visibility |
| `pose_world` | 33×3 | metric 3D body |
| `face` | 102×3 | lips, eyes, brows, nose, face-oval extremes, irises (`landmarks.FACE_IDX`) |
| `blend` | 52 | face blendshapes (brows, eyes, mouth — non-manual signals) |

Items are packed into `part-NNNNN.npz` files, with one `manifest.csv` per subset. Each
manifest row gives the item's id, part, offset, length, size/fps, detection rates and source
fields such as gloss, key, letter or rec_id. Coordinates are normalised separately by width
and height, so multiply x by `width / height` before using geometry.

```python
from azsl.store import LandmarkStore
words = LandmarkStore('/kaggle/input/<01-output>/landmarks/words200')
clip = words.get(words.manifest.id[0])          # dict of float32 arrays, (T, ...)
```

Sentence recordings are stored whole. Isolated glosses are slices of them, taken from
`index/tags.csv` (`start`, `end` are 1-based inclusive frame numbers), so a model can be given
context margins instead of the black padding the old I3D loader used.

## The split (notebook 03)

AzSLD has no signer ids, so every frontal recording gets a pseudo-signer:

1. **Sessions:** recordings sorted by the timestamp in their file name. A gap over 15 min
   starts a new session (one sitting ≈ one signer). Different signers do record on the same day.
2. **Faces:** a mean [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
   embedding per recording. Recordings far from their session's centroid are split off, then
   everything is clustered at cosine similarity 0.45 → 35 pseudo-signers. A session is one
   person (recordings within it agree at a median 0.95). Between sessions, different people
   often reach 0.35–0.40, while one person's sittings reach 0.8+.
3. **Split groups:** pseudo-signers whose centroids are at least 0.40 alike (`LINK_SIM`) are tied
   together → 21 groups. Nobody can tell from faces whether such pairs are one person, so they
   never straddle two splits (max cross-split similarity: 0.398). Merging them outright chained
   distinct people into one 30% cluster, which is why the ties only constrain the split.
4. **One group → train/val/test map (≈70/15/15 of samples)** is shared by every task, chosen so
   the 200-word vocabulary is covered in val and test. A test signer is never seen in training
   by any model. Train groups also get 5 CV folds.

Current split (`ml/splits/`): 25 / 5 / 5 pseudo-signers. Words_200: 5,995 / 1,274 / 1,288 clips,
covering 200 / 187 / 191 glosses. Majority class (MƏN) ≈ 29% of val and test.

Check the contact sheet in 03 (one row per pseudo-signer) before trusting the split. If the
dataset authors can share real signer ids, pass them as `OVERRIDE_CSV` (`rec_id,signer`).

| File | One row per | Notes |
|---|---|---|
| `signers.csv` | recording | session, signer, split_group, split, fold, frontal-camera check |
| `words200_split.csv` | Words_200 clip | linked to its recording via the clip's tag key; unlinked clips → train only |
| `glosses_split.csv` | annotated sign in the sentences | `in_w200`, `min_clips` (≥20 clips and ≥3 signers) flags; duplicate annotations removed |
| `sentences_split.csv` | recording | `heldout_sentence`: 10% of sentence folders reserved to test unseen sentences (Phase 4) |
| `fingerspelling_split.csv` | image | group = contributor (hashed) or clip run |
| `benchmark.json` | — | sizes per split, majority-class and chance baselines, coverage, clustering diagnostics |

Always report a model next to `majority_acc`. Report macro-F1 / balanced accuracy too,
because the classes are very uneven: in the old repo's label map MƏN has 3,818 clips,
while the median gloss has 21.

## AzSLD layout (checked Oct 2026, Zenodo 14222948)

- `AzSLD_Sentences/<folder>/{Cam1,Cam2}/<timestamp>.mp4` + `ann/<timestamp>.mp4.json`, 1280×960 at 30 fps.
  **Cam2 is the frontal camera** and the annotations belong to it (frame counts match). Cam1 is the
  side view, a few frames shorter, and missing for ~600 recordings. 10,189 Cam2 recordings,
  ~2.2 M frames (estimated from file sizes).
- Annotation tags: `{name, frameRange: [start, end], key}`. Ranges are 1-based inclusive
  (a Words_200 clip equals Cam2 frames `start-1 … end-1`). Some signs are annotated twice → `dup`.
- `AzSLD_Words_200/<GLOSS>/<key>.mp4`: 8,557 clips cut from Cam2 with no margin. `key` is the tag key,
  which is how clips are linked back to recordings and signers.
- `AzSLD_Fingerspelling`: on Zenodo only 40 images per static letter and ~800 frames per dynamic
  letter, far fewer than the 13k+ samples in the paper. Several letters come from a single
  contributor or clip, so they cannot be tested signer-independently; `benchmark.json` says
  which. The full JestDiliBot set would have to come from the authors.

## Training (Phase 2, notebook 04)

`04_train_words` (Kaggle **GPU T4**) trains the word model on landmarks:

- **Train:** every annotated sign of the 200-word vocabulary cut from train signers' recordings
  (≈21.7k), with ±8 frames of real context that augmentation crops inside.
- **Val, every epoch:** Words_200 clips of val signers (`W_*`, the benchmark) and sentence-cut
  signs of val signers (`S_*`).
- **Test, once:** the best-on-val (macro-F1) checkpoint, plain and with mirror test-time augmentation.

The features (`azsl/features.py`, 656 per frame) are body-relative: hand positions, metric hand
shape, hand velocity, upper-body pose, lips/eyes/brows and the 52 face blendshapes. They are
computed in numpy so the browser can mirror them in JS. Augmentation: boundary jitter, speed
0.8–1.25×, frame drop, mirror (hands, pose, face points and blendshapes swap sides), rotation,
scale, shear, hand dropout.

The model (`azsl/train.py`) is Conv1D (depthwise, ECA) + Transformer blocks in the style of the
Kaggle isolated-sign winners, ~1.7 M parameters. Batches are drawn with weights n_c^-0.5 against
the class imbalance, with label smoothing 0.1, AdamW and cosine decay. The model is exported to
ONNX (`train/model.onnx`, checked against PyTorch with onnxruntime).

```bash
python ml/kaggle_run.py push 04      # start a run (CLI setup: see kaggle_run.py)
python ml/kaggle_run.py log 04       # per-epoch lines so far
python ml/kaggle_run.py results      # history.csv, metrics.json, model.onnx, best.pt → ml/results/
python ml/smoke_train.py             # local check: a few real recordings, 2 epochs on CPU
```

## Developing

```bash
python ml/smoke_test.py          # real files over HTTP (~30 MB), all steps incl. the split
python ml/build_notebooks.py     # regenerate notebooks/ after editing azsl/
```

`azsl/` is the source of truth. The notebooks embed it with `%%writefile` cells, so the
`.ipynb` is the only thing that has to go to Kaggle.
