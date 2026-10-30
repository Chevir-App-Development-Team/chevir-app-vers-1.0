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
| 2 | `02_sentences_shard0` … `shard3` | 6–7 h each | `landmarks/sentences_cam2/shardNN/` |
| 2c1 | `02_sentences_cam1_shard0` … `shard3` | ~6 h each | `landmarks/sentences_cam1/shardNN/` + `align.csv` |
| 3 | `03_split_benchmark` | minutes | `splits/` |

- 1 and the four 2s are independent — start them together if Kaggle lets you
  (CPU sessions don't use the weekly GPU quota). If a shard log shows many `429`
  retries, Zenodo is throttling: run the shards one after another instead.
- 3 needs the outputs of 1 and all four 2s: *Add Input → Your Work → Notebooks*.
  No internet needed.
- 2c1 (the side camera, extra training data) needs the four 2s as inputs, for the frame alignment.
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

## The side camera (notebooks 02c1)

Cam1 records the same signing from another angle, so each annotated sign is a second training sample.
Only hands and body are extracted (the word models do not use the face). Annotations belong to Cam2, so
`azsl/views.py` finds each recording's frame offset: the lag (±45 frames) at which the arms' vertical
movement agrees best between the views (mean wrist height, the higher wrist, mean elbow height, each
minus its 1 s moving average). `align.csv` keeps the lag, its agreement `corr` and the `margin` over the
best lag more than 5 frames away. Notebook 04 uses a recording when both are clear (`SIDE_MIN_CORR`,
`SIDE_MIN_MARGIN`). The split is by recording, so a side view is always in the same split as its
frontal view. Val/test signers' side views are scored separately (`*_side`) and never choose a checkpoint.

## The split (notebook 03)

AzSLD has no signer ids, so every frontal recording gets a pseudo-signer:

1. **Sessions:** recordings sorted by the timestamp in their file name. A gap over 15 min
   starts a new session (one sitting ≈ one signer). Different signers do record on the same day.
2. **Faces:** a mean [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
   embedding per recording, clustered **per recording** (average linkage, cosine 0.55, `SIM`)
   → 39 pseudo-signers. The clusters are identical from 0.50 to 0.60, and no cluster has face pairs
   below 0.4. Clustering per sitting failed: 29 of 179 sittings have two people taking turns,
   and a sitting's average face resembles neither.
3. **Split groups:** two pseudo-signers are tied if *any* of their (sitting, signer) units are at least
   0.50 alike (`LINK_SIM`) → 35 groups. One person's sittings score 0.6+, different people mostly
   < 0.5. Tied pseudo-signers always share a split; the closest pair across splits scores 0.49.
4. **One group → train/val/test map (≈70/15/15 of samples)** is shared by every task, chosen so
   the 200-word vocabulary is covered in val and test. A test signer is never seen in training
   by any model. Train groups also get 5 CV folds.

Current split (`ml/splits/`, notebook 03 v6): 27 / 7 / 5 pseudo-signers. Words_200: 6,009 / 1,254 /
1,294 clips, covering 200 / 186 / 180 glosses. Majority class (MƏN): 28.6% of val, 31.1% of test.
The history of the split, with what each version got wrong, is in `EXPERIMENTS.md` (E2).

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
  **Cam2 is the frontal camera** and the annotations belong to it (frame counts match). 10,189 Cam2
  recordings, 2.18 M frames. Cam1 is an oblique view from one side and above, 9,571
  recordings. It starts and stops a few frames apart from Cam2.
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

- **Vocabulary:** up to run 7 the 200 words of Words_200. From run 8 every sentence gloss with at least
  10 signs of train signers joins them (`VOCAB_MIN_TRAIN`, `train.learnable_vocab`): 768 words, which
  cover 99.0% of the annotated signs (the 200 cover 63%). The remaining rare glosses form the unknown class.
- **Train:** every annotated sign of the vocabulary cut from train signers' recordings, with ±8 frames
  of real context that augmentation crops inside. With `SIDE`, the same signs from the side camera too.
- **Val, every epoch:** Words_200 clips of val signers (`W_*`, the benchmark) and sentence-cut
  signs of val signers (`S_*`; `S200` = only those of the 200 words, comparable with runs before 8).
- **Test, once:** the best-on-val checkpoint, plain and with mirror test-time augmentation.
- **Final model (`TRAIN_ON_VAL`):** once the recipe, the checkpoint epoch and the word-mode settings have
  been chosen on val, the same recipe is trained on train + val signers and saved at that epoch
  (`save_epoch`), without looking at any score. Test signers stay unseen. Its word-mode test uses the
  settings fixed before (`FIXED` in notebook 05). The browser model (run 11) was made this way.
- **Unseen sentences (`EXCLUDE_HELDOUT`):** leaves out the train signers' signs of the 49 sentences marked
  `heldout_sentence` (vocabulary unchanged), so word mode can be tested on sentences its model has never
  seen (run 12, E6 / E7). Runs up to 11 trained on them.
- **Several experiments per session:** `RUNS` in the notebook (e.g. seeds, feature-group ablations via
  `groups`, longer schedules) share the loaded data; each writes `train/<name>/` plus a line in
  `train/summary.csv`, and keeps its per-sample `predictions.npz` for later analyses.

The features (`azsl/features.py`, 656 per frame) are body-relative: hand positions, metric hand
shape, hand velocity, upper-body pose, lips/eyes/brows and the 52 face blendshapes. They are
computed in numpy so the browser can mirror them in JS. Augmentation: boundary jitter, speed
0.8–1.25×, frame drop, mirror (hands, pose, face points and blendshapes swap sides), rotation,
scale, shear, hand dropout.

The model (`azsl/train.py`) is Conv1D (depthwise, ECA) + Transformer blocks in the style of the
Kaggle isolated-sign winners, ~1.7 M parameters. Batches are drawn with weights n_c^-0.5 against
the class imbalance, with label smoothing 0.1, AdamW and cosine decay. The model is exported to
ONNX (`train/model.onnx`, checked against PyTorch with onnxruntime).

Kaggle attaches only notebook versions that completed as inputs to other notebooks. If a 04 version
ends in an error after writing its models, `model-dataset` uploads a run's `best.pt` and manifest as a
private dataset, and notebook 05 finds it by the manifest's run name.

```bash
python ml/kaggle_run.py push 04      # start a run (CLI setup: see kaggle_run.py)
python ml/kaggle_run.py log 04       # per-epoch lines so far
python ml/kaggle_run.py results --to run4   # summary, history, metrics, predictions → ml/results/run4/ (--models: + ONNX, best.pt)
python ml/kaggle_run.py model-dataset ml/results/run11_model/final_side_noface   # see below
python ml/smoke_train.py             # local check: a few real recordings, 2 epochs on CPU
```

## Sentence mode (notebook 06)

`06_train_continuous` (Kaggle **GPU T4**) trains a model that labels every frame of a sentence recording
with a word or `<blank>` (`azsl/continuous.py`). It learns from whole recordings, so the frames between
signs and the side camera's views (shifted by their alignment lag) are used too.

- **Labels:** `<blank>`, the word model's vocabulary with the same-sign merges (`ml/lemma_groups.csv`
  confirmed families + `ml/same_sign.csv`), and `<unk>` for rarer glosses. Frames within 1 of an annotated
  boundary are not scored.
- **Network:** the word model's Conv1D + Transformer with a per-frame head (`SeqNet`); `init` starts it
  from a word-model run. `splice` builds new sentences from signs of different recordings, so the word
  order of AzSLD's 492 sentences cannot be learned; `crop` sets the training window lengths.
- **Unseen sentences:** with `EXCLUDE_HELDOUT` the 49 sentences marked `heldout_sentence` stay out of
  training. Val and test report them separately (`unseen_sentences`, and `unseen_learnable` = only words
  with training signs outside them). `select='f1_unseen_learnable'` chooses the epoch on val's unseen
  sentences, learnable words only.
- **As in the browser:** the model runs on the last `window` frames every `stride` frames, and a frame is
  final `lookahead` frames after it (`STREAM`). Decoding (`DECODE`: smoothing, minimum probability,
  minimum length, gap joining; from `CONTINUOUS_VERSION` 2 an optional shorter `min_len_short` for
  single-character labels, i.e. fingerspelled letters and digits) is chosen on val from `DECODE_GRID`;
  test is scored once.
- **Final model:** `TRAIN_ON_VAL` + `save_epoch` + its `decode`, as for the word model.
- **Evaluation only:** a run with `ckpt='<run name>'` loads a saved model (uploaded with
  `kaggle_run.py model-dataset`) instead of training; `decode_from='<run>'` scores test with the decoding
  an earlier run of the same session chose. Push such a version with `--cpu`: no GPU quota is used.

```bash
python ml/kaggle_run.py push 06 [--smoke] [--cpu]
python ml/kaggle_run.py cont --to cont2 [--models]      # cont/ → ml/results/cont2/ (--models: + best.pt, ONNX)
python ml/install_continuous.py ml/results/cont2/<run> --install
node tools/verify_continuous.mjs     # streaming + decoding: JS against continuous_v<N> fixture
node tools/verify_cont_model.mjs     # installed model: features.js + onnxruntime-web against PyTorch
node tools/bench_cont_model.mjs      # its time per window (96 frames: ~11 ms; the budget at 30 fps is 267 ms)
```

`install_continuous.py` records PyTorch's per-frame logits on the fixture windows
(`ml/fixtures/cont_model_ref.json`), writes `sentences.json` (the AzSLD sentences with their sign
sequences, in the model's labels) and with `--install` copies the model to `public/tercume/assets/cont_model/`.
`python ml/sentence_match.py <run_dir>` scores the browser's sentence suggestion on a run's test detections,
with the sentence list built from train signers only.

The browser's sentence model is `final_splice` (notebook 06 v6): spliced training, train + val signers, every
sentence. Test, signers never seen: word F1 0.927, word error rate 0.105, the right AzSLD sentence suggested
for 91 % of the recordings. On sentences outside AzSLD expect about F1 0.6 (0.77 on the words it knows),
measured on 49 sentences left out of training. Results: `EXPERIMENTS.md` E7.

## Letter mode (notebook 07)

`07_train_letters` (Kaggle **GPU T4**, ~10 min a run) trains the letter mode's model on the landmark pipeline
(`azsl/letters.py`) from every AzSLD source of letters: the letter signs annotated in the sentences (both
cameras, signer split; an "O" counts only next to another spelled letter), the fingerspelling photos (static
letters, split by contributor), the moving-letter clips (frames filled back in) and `<none>` from the gaps
between signs. The model reads only the hand (`hand_shape` + `hand_vel`): body, face and blendshapes are removed
(`strip`), and the hand present in more frames is slot 0 (`canonical`). The old letter model is scored on the
same samples (`letters.OldModel`, a numpy port of `src/tercume/model.js`).

```bash
python ml/kaggle_run.py files-dataset chevir-letter-model-old public/tercume/assets/model.json public/tercume/assets/model.bin
python ml/kaggle_run.py push 07 [--smoke] [--cpu]
python ml/kaggle_run.py letters --to letters2 [--models]   # letters/ → ml/results/letters2/ (+ spelled words)
python ml/install_letters.py ml/results/letters2/final_hand --install
node tools/verify_letters.mjs        # letters.js against Python (fixture letters_v<N>) and the model against PyTorch
node tools/letters_tune.mjs <model dir> <spelled dir> grid   # the typing rules on spelled words (val), or one set
```

The browser's letter model is `final_hand` (train + val, epoch 51): test top-1 0.933 (letters in the test
signers' sentences 0.963, moving letters 0.931; the old model 0.214 on the same samples). The typing rules
(`LETTER_GATE` in `s2t.js`) were chosen on val signers' spelled words. Results: `EXPERIMENTS.md` E8.

## Deployment contract: browser and landmark server

The recognizer is meant to run in two places: **in the browser** (the default and the fallback) and
on a small **landmark server** that receives only landmark numbers, never video, and averages several
models. The server is not built yet. These pieces keep both paths identical:

- **One preprocessing:** `features.preprocess(raw, aspect, groups, face)` is the only way from landmarks
  to model input. Training evaluation calls it, and the server will import it. The browser port is
  `src/tercume/features.js`. It reproduces the golden cases in `ml/fixtures/features_v<N>.{json,bin}`
  bit for bit (the no-pose fallback within 3e-6):
  `node tools/verify_features.mjs`. `python ml/make_fixture.py --check` fails if the Python features
  change without a version bump. After a bump, update features.js to match and regenerate the fixture.
- **Browser landmarks:** `src/tercume/landmarks.js` runs the same three `.task` models with the same
  settings (`public/tercume/assets/`, md5 as in `azsl/landmarks.py`). It assigns hand slots with a port
  of `assign_hands` (300 / 300 golden cases equal) and buffers frames for `features.js`. On a real AzSLD
  recording it agrees with the Python pipeline, see `EXPERIMENTS.md` (E4). Frames also carry
  `pose_world` (the 9 body points in metres, for the avatar). It is outside the recognizer contract:
  `toRaw()` leaves it out unless `{ extra: true }` is passed, and the model input is identical either way.
- **`FEATURE_VERSION`** (in `features.py`) is written into every checkpoint, manifest and fixture. The
  browser and the server must refuse a model whose version differs from their preprocessing.
- **Input:** raw compact landmarks per frame (`features.spec()['raw']`: hands, hand world, 9 pose points,
  102 face points, 52 blendshapes; NaN = missing) plus the frame's width/height → logits over the same
  vocabulary. Without face points (privacy mode), send NaN or pass `face=False`.
- **`manifest.json` next to every `model.onnx`:** feature_version, labels, input layout (1 × T × F,
  max_len, groups), the full `spec()`, seed / fold / split_id, val and test metrics (including `*_noface`).
  Every run keeps its own model, so seeds and folds can be ensembled: average the softmax probabilities.
- **Privacy mode:** train with `face_drop` (probability that a clip loses its face and blendshapes) so one
  model serves both modes. Every run whose input the face can change reports scores with and without
  face. That includes hands-only and hands + pose models: one of the 4 presence flags is "face detected",
  so they are not face-free unless trained with `face_drop` (`train.reads_face`).
- **Browser model:** `src/tercume/recognizer.js` (`LocalRecognizer`) loads `manifest.json` +
  `model.onnx` from `public/tercume/assets/word_model/`, refuses another `feature_version`, and runs
  onnxruntime-web 1.30.0 (WASM, one thread, so no special HTTP headers; Vite bundles its `.wasm`
  next to the JS). Install a trained run with
  `python ml/make_manifest.py <run_dir> [--export] --install`. It writes the manifest for runs that
  predate it, re-exports `model.onnx` from `best.pt` with `--export`, and records PyTorch logits for the
  fixture cases in `ml/fixtures/word_model_ref.json`. `node tools/verify_recognizer.mjs` then runs the
  browser code (features.js → onnxruntime-web) on the same cases and must match them.
- **Word mode (continuous signing):** `azsl/segment.py` is the reference for cutting a landmark
  stream into signs (activity from the hand height, spans between rests, then the model picks
  non-overlapping windows inside a span). The browser twin is `src/tercume/segment.js`. Both are checked
  on `ml/fixtures/segment_v<N>.json` (`node tools/verify_segment.mjs`). `make_fixture.py` writes it, and
  `--check` fails if the rules change without a `SEGMENT_VERSION` bump. Notebook 05
  (`azsl/segment_eval.py`) scores word mode on the held-out sentence recordings and chooses its
  parameters on val; version 2 uses that choice. Fetch its output with
  `python ml/kaggle_run.py eval --to NAME`. Results: `EXPERIMENTS.md` E6.
- **Sentence mode in the browser:** `src/tercume/s2s.js` resamples the camera to 30 fps and streams
  windows into the continuous model (`continuous.js`: `WindowStream`, `decode`, `SentenceStream`, ports of
  `continuous.py`, checked on `ml/fixtures/continuous_v<N>`). Words are shown when final; 60 blank frames
  end a sentence, and the nearest AzSLD sentence (`sentences.json`) is suggested under it; with none near,
  the signs become Azerbaijani text (`gloss2text.js`: root + case / person / tense learned from the AzSLD
  sentences, `assets/gloss_text.json` from `tools/build_gloss_text.mjs`; label spellings from
  `ml/label_text.csv`; WER and chrF `tools/eval_gloss_text.mjs`, `EXPERIMENTS.md` E16, E17).
- **Letter mode in the browser:** `src/tercume/letters.js` (`LetterModel`, `strip`, `canonical`) runs the
  letter model on the hand tracker's last 24 frames (`landmarks.js frameFromResults` without body); `s2t.js`
  types a letter when the rules hold. `loadLetterModel` falls back to the old model (`model.js`) if
  `public/tercume/assets/letter_model/` is missing; `LETTERS_VERSION` guards the preprocessing.
- **Face tracker in the browser:** word mode loads it only if the model's validation macro-F1
  without face (`val_words_noface` in the manifest) is more than 0.01 below the score with face.
- **ONNX export:** the exported graph takes any clip length (1 × T × F). Models exported before this
  fix (runs 2–5) accept only T = 32 frames: `nn.MultiheadAttention` fixed the traced length into a
  reshape. Re-export them with `make_manifest.py --export`. `export_onnx` now checks lengths 7, 32 and 64
  against the training forward pass and leaves `model_file` empty in the manifest if they differ.

## Analysing results (no GPU)

```bash
python ml/analyze.py ml/results/run4 --ref ml/results/run3_split_v6
```

This prints the runs of a batch, mean ± std over the seed runs, and test scores with inflected word
forms merged, using `ml/lemma_groups.csv`. Whether two forms are one sign is decided by AzSL signers,
in the `confirmed` column:

- `yes` (also `bəli`, `hə`): the form is merged with its family. Predicting any form of the family
  counts as correct. The family's base form (where `gloss` = `group`) joins automatically unless it
  is marked `no`.
- `no` or empty: the form stays its own class, and confusing it with the base form is an error.

Merged scores are an extra evaluation with fewer classes; they never replace the plain scores.
Scores using all candidate families are printed only as an upper bound, labelled as unconfirmed.

## Developing

```bash
python ml/smoke_test.py          # real files over HTTP (~30 MB), all steps incl. the split
python ml/build_notebooks.py     # regenerate notebooks/ after editing azsl/
```

`azsl/` is the source of truth. The notebooks embed it with `%%writefile` cells, so the
`.ipynb` is the only thing that has to go to Kaggle.
