# Experiment log

Every number here is tied to the exact run that produced it and the split it was measured on.
**Split vN** = version N of Kaggle notebook `03-split-benchmark`; **run N** = version N of
`04-train-words`. A number measured on one split version is not comparable with another.
"Majority" = accuracy of always answering the most frequent training class (MƏN). It is the
floor every model has to beat.

Kaggle account `mammadovelnur`. All runs are on Kaggle, the extraction on CPU, training on a T4.

---

## E0 — Audit of the previous word model (`Nigar21/WL-AzSL-I3DCNN`), not re-run

Numbers are read from the repo's own training plots and data files.

| | |
|---|---|
| Model | Keras two-stream I3D (Inception, Kinetics-pretrained): RGB + TV-L1 flow |
| Data | 10 words, 4,383 clips cut from AzSLD sentence videos, 1280×960 at 30 fps, median 0.75 s |
| Split | random per clip, stratified by label; validation set = test set (880 clips) |
| Reported val accuracy | RGB ≈ 0.87–0.93 · flow ≈ 0.60–0.64 |
| Majority on that test set | **0.575** (506 / 880 clips are MƏN) |

Why those numbers overstate the model:
1. The same signers and recording sessions appear in train and test, because the clips are UUID-named and carry no signer id.
2. The best epoch was picked on the test set.
3. Clips shorter than 20 frames were padded with black frames. Clip length differs by class (MƏN median 17 frames, ƏLİL 32), so the padding leaks the label.
4. The flow stream is roughly at the majority baseline.

---

## E1 — Landmark extraction (Phase 1)

MediaPipe 0.10.35 (Python 3.13 on Kaggle), using the same md5-pinned `.task` models as the browser.
Hands run in VIDEO mode (tracking) for videos and IMAGE mode for the fingerspelling images.

| Subset | Items | Frames | Median length | Hand found | Both hands | Pose | Face | Errors | Runtime (Kaggle CPU) |
|---|---|---|---|---|---|---|---|---|---|
| Words_200 clips | 8,557 | 186,934 | 20 | 0.971 | 0.524 | 1.000 | 0.998 | 0 | part of notebook 01 |
| Sentences, frontal camera | 10,189 | 2,177,018 | 197 | 0.805 | 0.495 | 1.000 | 1.000 | 0 | 4 shards, 5.6–7.0 h each, 22–27 frames/s |
| Fingerspelling images | 7,359 | — | — | 0.935 (static 0.931, dynamic 0.935) | — | — | — | 0 | 0.1 h |

- "Hand found" for words and sentences is the frame-weighted share of frames with at least one hand.
  Sentences are lower because signers rest their hands between signs.
- Frontal-camera check (nose offset from the shoulder centre ÷ shoulder width): median 0.025, max 0.125.
  All 10,189 recordings are frontal.
- The annotation index has 10,104 annotated recordings and 49,287 sign annotations; 7 lack a frame range.

---

## E2 — Signer-independent split (Phase 0): iterations

AzSLD has no signer ids, so pseudo-signers come from SFace face embeddings (one mean vector per
recording). The table is the history of how that grouping was made leak-free.

| Split | Method | Pseudo-signers | Split groups | Signers train / val / test | Closest pair across splits | Mixed rows | Words_200 clips train / val / test | Majority val / test | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| v1 | sittings (15-min gap) → session centroids → average linkage 0.45 | 35 | — | 23 / 7 / 5 | not measured | yes (later check) | 6,008 / 1,313 / 1,236 | 0.314 / 0.279 | superseded |
| v2–v3 | v1 + merging clusters whose centroids are ≥ 0.40 alike | 20 | — | 14 / 3 / 3 | 0.493 (cluster centroids) | yes: one cluster of 3,015 recordings | 5,955 / 1,266 / 1,336 | 0.294 / 0.308 | rejected: chained different people |
| v4 | v1 + tying clusters with centroids ≥ 0.40 alike into split groups | 35 | 21 | 25 / 5 / 5 | 0.398 by centroids, **0.67 by sittings** | yes | 5,995 / 1,274 / 1,288 | 0.287 / 0.289 | rejected: sitting-level leaks (test p030 ~ train p011 at 0.58–0.67) |
| v5 | sittings clustered at 0.60; tie if any two sittings are ≥ 0.50 alike | 38 | 28 | 24 / 5 / 9 | 0.492 (sittings) | **yes**: 29 of 179 sittings hold two people; 5 clusters mixed | 5,957 / 1,302 / 1,298 | 0.312 / 0.294 | rejected: two people per row (seen on the contact sheet) |
| v6 | **recordings** clustered at 0.55; tie if any two (sitting, signer) units are ≥ 0.50 alike | 39 (6 with < 20 recordings) | 35 (largest 9.7%) | 27 / 7 / 5 | 0.492 (units) | none: 0% of within-cluster face pairs < 0.40 | 6,009 / 1,254 / 1,294 (glosses 200 / 186 / 180) | 0.286 / 0.311 | **current** (committed in `ml/splits/`) |

What the data showed:
- One person's recordings agree at a median similarity of 0.95. Different people are mostly below 0.4.
- Recordings of the same person in different sittings score 0.6 or more.
- Clustering per recording gives the same 39 clusters at thresholds 0.50, 0.55 and 0.60, so the gap is real.
- Clustering per sitting fails because some sittings have two people taking turns. Their average face resembles neither.
- v6 sentence-cut signs with ≥ 20 clips and ≥ 3 signers (645 glosses): 32,946 / 6,576 / 7,156.
  Signer–sentence repeats fell from 965 (v1) to 415, now mostly genuine retakes.
- Every version is optimised for 70/15/15 of samples, with all testable glosses of the 200-word vocabulary
  in both val and test (198 testable). Unlinked Words_200 clips (78) go to train only.

---

## E3 — Word recognition from landmarks (Phase 2)

Model: Conv1D (depthwise, ECA) + Transformer, 656 features per frame, ~1.7 M parameters.
Training data: sentence-cut signs of train signers, 200-word vocabulary, ±8 frames of context.
Defaults: 60 epochs, batch 64, AdamW lr 1e-3, weight decay 0.05, cosine decay, label smoothing 0.1,
batches drawn with weights n_c^-0.5. The checkpoint is chosen on val Words_200 macro-F1, and test is
scored once with it. `W` = Words_200 clips (the benchmark); `S` = sentence-cut signs.

| Run | Split | Status | Best epoch | Val W top-1 / macro-F1 | Test W top-1 / top-5 / macro-F1 / balanced | Test S top-1 / macro-F1 | Majority (test W) | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | v4 | crashed while loading data | — | — | — | — | — | 7 annotations without a frame range (fixed) |
| 2 | v5 | done, 60 epochs × 26 s on T4 | 60 | 0.834 / 0.855 (top-1 peaked at 0.875, epoch 35) | 0.853 / 0.988 / 0.859 / 0.871 · mirror-TTA 0.859 / 0.988 / 0.869 / 0.882 | 0.872 / 0.874 (majority 0.179) | 0.294 | v5 later rejected for mixed rows, so this checks the setup; the number may be slightly optimistic |
| **3** | **v6** | done, 60 epochs × 26.6 s on T4 | 58 | 0.885 / 0.857 (top-1 peaked at 0.893, epoch 32) | **0.853 / 0.981 / 0.835 / 0.850** · mirror-TTA 0.851 / 0.981 / 0.832 / 0.848 | 0.858 / 0.853 (majority 0.162) | 0.309 | **headline**: clean split, same config as run 2 |
| 4a–4e | v6 | *running*: notebook 04 v4, one session | | | | | | seed 1, seed 2, hands only (382 features), no face (400 features), 100 epochs |

### Run 2: per epoch (split v5)

| Epoch | Train loss | Train acc | Val W top-1 | Val W top-5 | Val W macro-F1 | Val W balanced | Val S top-1 | Val S macro-F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | 4.563 | 0.140 | 0.582 | 0.818 | 0.231 | 0.279 | 0.558 | 0.276 |
| 2 | 2.671 | 0.509 | 0.572 | 0.842 | 0.476 | 0.531 | 0.589 | 0.536 |
| 3 | 1.968 | 0.670 | 0.782 | 0.960 | 0.635 | 0.669 | 0.756 | 0.682 |
| 5 | 1.607 | 0.772 | 0.736 | 0.971 | 0.683 | 0.727 | 0.747 | 0.727 |
| 10 | 1.342 | 0.852 | 0.860 | 0.982 | 0.790 | 0.816 | 0.845 | 0.821 |
| 15 | 1.245 | 0.880 | 0.857 | 0.986 | 0.803 | 0.826 | 0.851 | 0.833 |
| 20 | 1.177 | 0.900 | 0.857 | 0.983 | 0.814 | 0.823 | 0.843 | 0.823 |
| 30 | 1.088 | 0.927 | 0.865 | 0.986 | 0.814 | 0.843 | 0.854 | 0.836 |
| 40 | 1.020 | 0.948 | 0.865 | 0.981 | 0.838 | 0.853 | 0.878 | 0.861 |
| 50 | 0.974 | 0.966 | 0.853 | 0.982 | 0.846 | 0.863 | 0.866 | 0.867 |
| 60 | 0.961 | 0.971 | 0.834 | 0.982 | 0.855 | 0.869 | 0.857 | 0.869 |

Observations (run 2):
- Top-1 peaks mid-training while macro-F1 keeps rising. The n_c^-0.5 sampling trades some accuracy
  on MƏN for the rare glosses, and the best macro-F1 is at the last epoch, so longer training is
  worth its own run.
- Mirror test-time augmentation (average of the clip and its mirror) adds +0.6 top-1 and +1.0 macro-F1
  in run 2. Run 3 does not replicate this (−0.2 top-1), so it is not counted as a gain.
- Most test errors are between word forms that are likely the same sign: İSTƏYİRƏM → İSTƏMƏK (0/5),
  İŞSİZ → İŞLƏMƏK (0/5), MƏNİM → MƏN (2/21), GEDƏ → GETMƏK, OLMUSUZ → OLMAQ. AzSLD labels
  Azerbaijani word forms, not sign glosses, so a lemma-merged evaluation is a candidate experiment.
- ONNX export works (6.8 MB). The onnxruntime check was skipped because Kaggle lacks onnxruntime.

### Run 3: per epoch (split v6, headline)

| Epoch | Train loss | Train acc | Val W top-1 | Val W top-5 | Val W macro-F1 | Val W balanced | Val S top-1 | Val S macro-F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | 4.561 | 0.140 | 0.553 | 0.807 | 0.221 | 0.279 | 0.544 | 0.226 |
| 2 | 2.668 | 0.514 | 0.742 | 0.941 | 0.540 | 0.590 | 0.746 | 0.587 |
| 3 | 1.982 | 0.672 | 0.743 | 0.946 | 0.613 | 0.690 | 0.725 | 0.622 |
| 5 | 1.622 | 0.768 | 0.796 | 0.970 | 0.716 | 0.769 | 0.797 | 0.734 |
| 10 | 1.354 | 0.846 | 0.845 | 0.979 | 0.782 | 0.819 | 0.835 | 0.790 |
| 15 | 1.247 | 0.879 | 0.858 | 0.986 | 0.791 | 0.824 | 0.864 | 0.832 |
| 20 | 1.184 | 0.894 | 0.819 | 0.985 | 0.813 | 0.850 | 0.843 | 0.821 |
| 30 | 1.097 | 0.921 | 0.830 | 0.988 | 0.847 | 0.867 | 0.852 | 0.860 |
| 40 | 1.024 | 0.948 | 0.878 | 0.983 | 0.836 | 0.857 | 0.888 | 0.864 |
| 50 | 0.975 | 0.967 | 0.878 | 0.980 | 0.852 | 0.871 | 0.898 | 0.881 |
| 60 | 0.966 | 0.969 | 0.884 | 0.982 | 0.851 | 0.871 | 0.897 | 0.872 |

The full 60-epoch history is in `history.csv` in the Kaggle output of `04-train-words` version 3.

Test, best-on-val checkpoint (epoch 58), split v6:

| Test set | Samples | Top-1 | Top-5 | Macro-F1 | Balanced acc | Majority |
|---|---|---|---|---|---|---|
| Words_200 clips | 1,290 | 0.853 | 0.981 | 0.835 | 0.850 | 0.309 |
| Sentence-cut signs | 4,646 | 0.858 | 0.979 | 0.853 | 0.856 | 0.162 |

Observations (run 3):
- 129 of the 180 glosses in the test set reach recall ≥ 0.9. The 11 at recall 0 mostly have only 1–2
  test clips (HARDA, HƏKİM, ORA, QARABAĞ, SONRA, TAKSİ, YATMAQ, YOXDUR, ZƏNG, ÇANTA, İŞSİZ).
- MƏNİM → MƏN is still the main systematic confusion (recall 3/33): a possessive "my" vs "I",
  probably the same or a near-identical sign.
- Val top-1 swings by ±3 points between epochs (0.819 at epoch 20, 0.893 at 32). Val has 7 signers,
  so one signer's clips move the number. Macro-F1 is steadier and keeps improving to the end.
- Runs 2 and 3 both score 0.853 test top-1, but on different test sets (v5 vs v6); this is a coincidence.
- Test is below val (0.853 vs 0.885 top-1), as expected when val is used to pick the checkpoint and the
  test signers are new.
- Local copies of run 2's full outputs were overwritten by a download bug (now fixed). Run 2's
  numbers above were copied before that.

Candidate next experiments (one change at a time, all on split v6):
- longer training (100 epochs; best macro-F1 was at the end in both runs)
- lemma-merged evaluation (İSTƏYİRƏM/İSTƏMƏK, MƏNİM/MƏN, GEDƏ/GETMƏK, …)
- feature ablations: no face/blendshapes, hands only (what the browser can run most cheaply)
- several seeds, to put error bars on the headline number

### Word-form families (evaluation only)

AzSLD labels Azerbaijani word forms, so some classes are inflections of one sign. The 13 candidate
families in `ml/lemma_groups.csv` (30 forms → 13 classes) were **all confirmed as the same sign by
Nigar Alishzade on 2026-10-18**. Merged scores (`ml/analyze.py`) count a prediction of any form in a
family as correct. They are reported next to the plain 200-class scores, not instead of them.

### Deployment contract (from run 5 onward)

From the next batch on, every run also writes `manifest.json` (feature_version 1, labels, input layout,
seed / split_id, metrics) and reports test scores **without face points** (`*_noface`, privacy mode).
`face_drop` (training with the face randomly removed) is a new experiment variable. Runs 1–4 predate
this, so they have no manifests and no no-face scores.

---

## E4 — Browser landmarks vs the training pipeline

Same recording (`AzSLD_Sentences/1/Cam2/2022-05-31 15-58-30.mp4`, 148 frames, 1280×960). Reference:
the Python pipeline (`azsl.landmarks.Extractor`, MediaPipe 1.0.1, CPU). Browser: `src/tercume/landmarks.js`
(@mediapipe/tasks-vision 0.10.14, GPU, in-app Chromium), with the same `.task` files, settings and
frame timestamps.

| | Browser vs Python |
|---|---|
| Hand present in the same slot and frame | 288 / 296 (97.3%); 3 frames only in Python, 5 only in the browser |
| Right/left slot swaps | 0 |
| Hand image x, y, mean / max abs difference | 0.0027 / 0.173 (fraction of frame; ≈ 3.4 px mean at 1280 px) |
| Hand world (metric) | 0.0019 / 0.091 |
| Pose x, y | 0.0037 / 0.031 |
| Face x, y | 0.0002 / 0.005 |
| Blendshapes | 0.0022 / 0.158 |
| Model input after `preprocess` (656 features), mean abs | 0.0107 (groups 0.002–0.013; max 1 = presence flags in the 8 frames above) |
| Speed, three models | 88 ms per frame (≈ 11 fps), load 0.43 s. Measured in the app's browser pane, so live-camera speed needs its own check |

The differences are far below the training augmentation (±10° rotation, ±15% scale, 0.003 jitter), and
the slot rule never disagrees. The reference used MediaPipe 1.0.1, while the training data was extracted
with 0.10.35; both use the same models.
