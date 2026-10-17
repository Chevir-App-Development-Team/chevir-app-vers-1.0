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
| 2 | v5 | *running* | | | | | | v5 later rejected (mixed rows), so this run only checks the training setup |
| 3 | v6 | planned | | | | | | |
