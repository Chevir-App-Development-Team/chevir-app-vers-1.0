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
| Sentences, side camera (Cam1) | 9,571 | 2,011,680 | 195 | 0.785 | 0.347 | 1.000 | not extracted | 0 (21 videos empty) | 4 shards, 3.1–5.1 h each, ~34 frames/s |
| Fingerspelling images | 7,359 | — | — | 0.935 (static 0.931, dynamic 0.935) | — | — | — | 0 | 0.1 h |

- "Hand found" for words and sentences is the frame-weighted share of frames with at least one hand.
  Sentences are lower because signers rest their hands between signs.
- Frontal-camera check (nose offset from the shoulder centre ÷ shoulder width): median 0.025, max 0.125.
  All 10,189 recordings are frontal.
- The annotation index has 10,104 annotated recordings and 49,287 sign annotations; 7 lack a frame range.

### The side camera (Cam1, notebooks 02c1, added for run 9)

AzSLD films every sentence from a second camera, to one side and above the signer. These 9,571
recordings were not used before. They hold the same annotated signs from another angle, so they can be
a second training sample of each sign. Only hands and body were extracted (the word models use no face),
which made it faster than the frontal pass. The angle measure (nose offset ÷ shoulder width) has median
0.37 (frontal camera: 0.025). Hands are found about as often as from the front (0.785 vs 0.805), but
both hands much less often (0.347 vs 0.495), as one hand hides the other.

The annotations belong to the frontal camera, and the two cameras start a few frames apart. Each
recording gets the frame lag at which the arms' vertical movement agrees best between the views
(`azsl/views.py`, searched over ±45 frames):

| | |
|---|---|
| Recordings aligned | 9,550 (21 empty) |
| Agreement at the best lag (Pearson r, mean of 3 signals) | median 0.693, IQR 0.594–0.786 |
| Best lag minus the best lag more than 5 frames away (margin) | median 0.434 |
| Lag | −3 … +4 frames for 99 % (0: 1,778, +1: 1,939, +2: 1,869, −1: 1,606, +3: 1,333) |
| Used (agreement ≥ 0.3 and margin ≥ 0.1, fixed before any training) | **9,463 (98.9 %)**, of which 99.9 % have \|lag\| ≤ 5 |

The thresholds were set from this table only. A recording is in the same split as its frontal view,
since the split is by recording.

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
| 4a | v6 | done, 60 epochs × 27.4 s | 39 | 0.865 / 0.882 | 0.775 / 0.985 / 0.825 / 0.845 | 0.829 / 0.852 | 0.309 | seed 1 (otherwise run 3) |
| 4b | v6 | done, 60 epochs × 27.4 s | 41 | 0.877 / 0.856 | 0.822 / 0.981 / 0.830 / 0.847 | 0.848 / 0.848 | 0.309 | seed 2 |
| 4c | v6 | done, 60 epochs × 25.9 s | 35 | 0.889 / 0.861 | 0.820 / 0.985 / 0.833 / 0.849 | 0.845 / 0.837 | 0.309 | hands only: 382 features (hands, hand shape, hand velocity, flags) |
| **4d** | **v6** | done, 60 epochs × 26.0 s | 45 | 0.902 / 0.861 | **0.878 / 0.986 / 0.863 / 0.885** | 0.868 / 0.852 | 0.309 | **no face**: 400 features (hands + 9 body points); best run so far, one seed |
| 4e | v6 | done, 100 epochs × 27.6 s | 84 | 0.890 / 0.861 | 0.843 / 0.977 / 0.825 / 0.844 | 0.861 / 0.849 | 0.309 | 100 epochs |

Runs 4a–4e are one Kaggle session (notebook 04 v4, 2 h 35 min of training on a T4). Everything not named in
Notes is the run 3 setup (656 features, 60 epochs, seed 0).

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

Candidate next experiments after run 3 (one change at a time, all on split v6), all run in run 4:
- longer training (100 epochs; best macro-F1 was at the end in both runs)
- lemma-merged evaluation (İSTƏYİRƏM/İSTƏMƏK, MƏNİM/MƏN, GEDƏ/GETMƏK, …)
- feature ablations: no face/blendshapes, hands only (what the browser can run most cheaply)
- several seeds, to put error bars on the headline number

### Run 4: seeds, feature ablations, 100 epochs (split v6)

Validation per epoch, Words_200 clips, **top-1 / macro-F1**. The full histories (every epoch, all
columns of the run 3 table) are in `history.csv` of each run in the Kaggle output of
`04-train-words` version 4.

| Epoch | 4a seed 1 | 4b seed 2 | 4c hands only | 4d no face | 4e 100 epochs |
|---|---|---|---|---|---|
| 1 | 0.581 / 0.230 | 0.576 / 0.226 | 0.584 / 0.270 | 0.620 / 0.265 | 0.567 / 0.238 |
| 2 | 0.634 / 0.507 | 0.698 / 0.551 | 0.727 / 0.575 | 0.733 / 0.595 | 0.668 / 0.571 |
| 3 | 0.580 / 0.536 | 0.742 / 0.640 | 0.795 / 0.679 | 0.748 / 0.677 | 0.763 / 0.631 |
| 5 | 0.820 / 0.723 | 0.807 / 0.725 | 0.851 / 0.762 | 0.828 / 0.750 | 0.812 / 0.741 |
| 10 | 0.807 / 0.801 | 0.871 / 0.806 | 0.854 / 0.811 | 0.861 / 0.806 | 0.854 / 0.788 |
| 15 | 0.813 / 0.822 | 0.855 / 0.799 | 0.847 / 0.816 | 0.859 / 0.794 | 0.802 / 0.786 |
| 20 | 0.824 / 0.839 | 0.875 / 0.817 | 0.874 / 0.824 | 0.872 / 0.816 | 0.731 / 0.814 |
| 30 | 0.836 / 0.847 | 0.883 / 0.854 | 0.884 / 0.846 | 0.865 / 0.832 | 0.851 / 0.821 |
| 40 | 0.867 / 0.865 | 0.883 / 0.830 | 0.851 / 0.846 | 0.890 / 0.850 | 0.859 / 0.830 |
| 50 | 0.865 / 0.862 | 0.881 / 0.843 | 0.886 / 0.856 | 0.897 / 0.851 | 0.881 / 0.840 |
| 60 | 0.897 / 0.867 | 0.872 / 0.838 | 0.884 / 0.845 | 0.899 / 0.854 | 0.866 / 0.847 |
| 70 | — | — | — | — | 0.880 / 0.836 |
| 80 | — | — | — | — | 0.854 / 0.851 |
| 90 | — | — | — | — | 0.867 / 0.858 |
| 100 | — | — | — | — | 0.877 / 0.856 |

Test, best-on-val checkpoint of each run. "Merged" = the 13 word-form families confirmed by Nigar
Alishzade counted as one class (183 classes, `ml/analyze.py`). Run 3 saved no predictions, so it
has no merged score.

| Run | Features | Test W top-1 / top-5 / macro-F1 / balanced | Test W merged (183): top-1 / top-5 / macro-F1 / balanced | Test S top-1 / top-5 / macro-F1 / balanced | Mirror-TTA test W top-1 / macro-F1 |
|---|---|---|---|---|---|
| 3 (seed 0) | 656 | 0.853 / 0.981 / 0.835 / 0.850 | — | 0.858 / 0.979 / 0.853 / 0.856 | 0.851 / 0.832 |
| 4a seed 1 | 656 | 0.775 / 0.985 / 0.825 / 0.845 | 0.937 / 0.987 / 0.871 / 0.889 | 0.829 / 0.984 / 0.852 / 0.863 | 0.761 / 0.828 |
| 4b seed 2 | 656 | 0.822 / 0.981 / 0.830 / 0.847 | 0.940 / 0.983 / 0.880 / 0.891 | 0.848 / 0.981 / 0.848 / 0.857 | 0.824 / 0.835 |
| 4c hands only | 382 | 0.820 / 0.985 / 0.833 / 0.849 | 0.936 / 0.985 / 0.881 / 0.893 | 0.845 / 0.981 / 0.837 / 0.851 | 0.833 / 0.833 |
| **4d no face** | **400** | **0.878 / 0.986 / 0.863 / 0.885** | **0.950 / 0.988 / 0.906 / 0.925** | 0.868 / 0.984 / 0.852 / 0.863 | 0.872 / 0.854 |
| 4e 100 epochs | 656 | 0.843 / 0.977 / 0.825 / 0.844 | 0.942 / 0.981 / 0.875 / 0.891 | 0.861 / 0.978 / 0.849 / 0.859 | 0.848 / 0.842 |

Full model (656 features), 3 seeds (runs 3, 4a, 4b), mean ± sample std:

| Test set | Top-1 | Top-5 | Macro-F1 | Balanced acc |
|---|---|---|---|---|
| Words_200 clips | 0.817 ± 0.039 | 0.983 ± 0.002 | 0.830 ± 0.005 | 0.847 ± 0.002 |
| Sentence-cut signs | 0.845 ± 0.015 | 0.981 ± 0.002 | 0.851 ± 0.003 | 0.859 ± 0.004 |

Where the top-1 spread comes from (test Words_200, most frequent wrong pairs):

| Run | MƏN recall (399 clips, 31 % of test) | MƏN → MƏNİM | Other top confusions |
|---|---|---|---|
| 4a seed 1 | 0.699 | 115 | O → ONUN 26, MƏNİM → MƏN 20 |
| 4b seed 2 | 0.852 | 53 | MƏNİM → MƏN 24, SİZ → SİZİN 16 |
| 4c hands only | 0.895 | 31 | O → ONUN 33, MƏNİM → MƏN 27 |
| 4d no face | 0.987 | 1 | MƏNİM → MƏN 33, O → ONUN 16 |
| 4e 100 epochs | 0.925 | 15 | O → ONUN 30, MƏNİM → MƏN 28 |

Observations (run 4):
- **Top-1 is not a stable headline metric on this test set.** With the same setup, top-1 ranges
  0.775–0.853 over three seeds, while macro-F1 (±0.005) and balanced accuracy (±0.002) barely move.
  Almost all of the difference is one boundary: MƏN is 31 % of the test clips, and a seed that puts
  that boundary differently sends up to 115 of them to MƏNİM (same sign, confirmed family). Merged
  top-1 is 0.936–0.950 in every run. Report macro-F1 / balanced accuracy as the main numbers, and
  top-1 next to merged top-1.
- **No face (hands + body, 400 features) is the best run on every Words_200 metric**: macro-F1 0.863
  vs 0.830 ± 0.005 for the full model (+0.033, about 6 seed standard deviations), balanced 0.885 vs
  0.847 ± 0.002, merged macro-F1 0.906. On sentence-cut signs it equals the full model in macro-F1
  (0.852 vs 0.851 ± 0.003) and is higher in top-1 (0.868). It is one seed, so it needs seeds 1 and 2
  before it is a headline. A plausible reason (not tested): the 204 face coordinates and 52
  blendshapes describe the signer's face as much as the sign, which does not transfer to new signers.
- **The body points matter, the face does not (so far)**: hands only 0.833 macro-F1 W (inside the
  full-model spread) and 0.837 on S (−0.014); adding the 9 body points (no face) gives +0.030 W and
  +0.015 S over hands only.
- **100 epochs does not help**: best epoch 84, test macro-F1 0.825, inside the 60-epoch seed spread.
  60 epochs stays the default.
- **Mirror TTA is not a consistent gain** (−0.014 to +0.013 top-1, −0.009 to +0.017 macro-F1 over five
  runs). It stays off.
- For the browser, a model without face points could save the face tracker. It is **not face-free
  yet**: its 4 presence flags include "face detected", which was ≈ 1 in all training data. Switching
  the face off changes its logits by up to 0.34 on the synthetic fixture cases. On the 3 real signs in
  E5 the change is ≤ 0.002 in probability. Run 5 measures it (`*_noface` scores, `face_drop`).

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

| Run | Split | Status | Runs (each = run 3 setup with this change) |
|---|---|---|---|
| 5a–5d | v6 | done: notebook 04 v5, 4 × 28 min on a T4 | no face seed 1, no face seed 2, no face + `face_drop` 0.5, all features + `face_drop` 0.5 |

Run 5 was pushed before the ONNX export fix (E5), so its `model.onnx` files accept only 32 frames
and their export check fails. The checkpoints are fine; models for the browser are re-exported from
`best.pt` (`ml/make_manifest.py --export`).

### Run 5: no face over seeds, face dropout (split v6)

Test, best-on-val checkpoint. "Face off" = the same model scored with the face removed (privacy mode,
or no face tracker in the browser). Run 4d is the seed-0 run of the no-face setup.

| Run | Features | Best epoch | Val W macro-F1 | Test W top-1 / top-5 / macro-F1 / balanced | Test W, face off: top-1 / macro-F1 | Test W merged (183): top-1 / macro-F1 | Test S top-1 / macro-F1 |
|---|---|---|---|---|---|---|---|
| 4d no face, seed 0 | 400 | 45 | 0.861 | 0.878 / 0.986 / 0.863 / 0.885 | not scored | 0.950 / 0.906 | 0.868 / 0.852 |
| **5a no face, seed 1** | 400 | 55 | **0.873** | 0.854 / 0.985 / 0.849 / 0.867 | 0.854 / 0.846 | 0.948 / 0.892 | 0.870 / 0.856 |
| 5b no face, seed 2 | 400 | 44 | 0.861 | 0.852 / 0.986 / 0.830 / 0.856 | 0.852 / 0.829 | 0.940 / 0.882 | 0.863 / 0.846 |
| 5c no face + `face_drop` 0.5 | 400 | 38 | 0.861 | 0.788 / 0.982 / 0.802 / 0.825 | 0.789 / 0.804 | 0.933 / 0.854 | 0.839 / 0.833 |
| 5d all features + `face_drop` 0.5 | 656 | 48 | 0.861 | 0.851 / 0.983 / 0.848 / 0.863 | 0.848 / 0.847 | 0.940 / 0.892 | 0.863 / 0.850 |

Mean ± sample std over three seeds:

| Setup | Test W top-1 | Test W macro-F1 | Test W balanced | Test S top-1 | Test S macro-F1 |
|---|---|---|---|---|---|
| All features (runs 3, 4a, 4b) | 0.817 ± 0.039 | 0.830 ± 0.005 | 0.847 ± 0.002 | 0.845 ± 0.015 | 0.851 ± 0.003 |
| No face (runs 4d, 5a, 5b) | 0.861 ± 0.015 | 0.847 ± 0.016 | 0.869 ± 0.014 | 0.867 ± 0.003 | 0.851 ± 0.005 |

Ensembles (average of the softmax probabilities, from the saved test predictions):

| Ensemble | Test W top-1 / top-5 / macro-F1 / balanced | Test S top-1 / top-5 / macro-F1 / balanced |
|---|---|---|
| No face, 3 seeds (4d, 5a, 5b) | 0.880 / 0.987 / 0.860 / 0.881 | 0.882 / 0.988 / 0.862 / 0.872 |
| All features, 2 seeds (4a, 4b) | 0.800 / 0.986 / 0.853 / 0.869 | 0.846 / 0.985 / 0.857 / 0.865 |
| All 7 runs with saved predictions (4a, 4b, 4c, 4d, 5a, 5b, 5d) | 0.874 / 0.987 / 0.862 / 0.877 | 0.882 / 0.988 / 0.875 / 0.882 |

Observations (run 5):
- **Run 4d was the lucky seed.** Over three seeds, no face gives Words_200 macro-F1 0.847 ± 0.016, against
  0.830 ± 0.005 for all features. On sentence-cut signs both are 0.851. Without face is at least as good
  and needs one tracker less, but "the face hurts" is not supported: the difference is within the
  run-to-run spread.
- **The run-to-run spread is larger than the seed spread suggests.** 5c differs from 4d only in that the
  "face detected" flag is randomly zeroed in training, and its test macro-F1 is 0.061 lower, with the
  same val macro-F1 (0.861). With 5 test signers, single-run test differences below ~0.03 should not be
  read as real. Report means over seeds.
- **The face flag barely matters in practice.** Scored with the face removed, the no-face models lose
  0.003 / 0.001 test macro-F1 (val 0.0007 for 5a), so the browser can drop the face tracker (E6).
- **Face dropout gives a full model that works without the face**: 5d loses 0.001 without face (0.848 →
  0.847) and is as good as the full-feature seeds. Runs 3/4 have no face-off scores to compare with.
- **Ensembles help**: three no-face seeds reach 0.860 Words_200 macro-F1 (seed mean 0.847) and 0.862 on
  sentence-cut signs. All seven runs together reach 0.875 on sentence-cut signs. This is what the
  planned landmark server would run; the browser keeps one model.
- **Browser model = 5a, picked on validation** (highest val macro-F1, 0.873), not on test. 4d has the
  best test score, but choosing it by that would make the test number optimistic.


### Run 6: an "unknown sign" class (split v6)

Word mode on held-out sentences (E6) mostly fails on signs outside the 200 words: the model names them
as a known word. Run 6 adds one class, `<unk>`, made of the out-of-vocabulary sentence signs of the
train signers: 12,031 signs of 584 glosses. Another 59 glosses (1,030 signs) are left out because they
share their first 4 letters with a vocabulary word (ALMALIYAM, DEYİLƏM, QARABAQ, …), so they are
probably the same sign. The setup is otherwise run 5a (no face, seed 1). The variable is the
`<unk>` class's sampling weight: the usual n^-0.5, times 1, 4 or 8. The checkpoint is still chosen
on val Words_200 macro-F1, as in every run.

Out-of-vocabulary sentence signs of the val / test signers (2,067 / 2,652) are kept apart, so the
word and sign scores stay comparable with run 5a:
- **Unknown rejected:** the share of them predicted `<unk>`.
- **Known rejected:** the share of in-vocabulary sentence signs wrongly predicted `<unk>`.
- **Open-set accuracy:** over all sentence signs, known signs named right plus unknown ones rejected.
- **Known-only macro-F1:** the `<unk>` output is ignored.

| Run | Best epoch | Val W macro-F1 | Test W top-1 / macro-F1 | Test W macro-F1, known only | Test S top-1 | Unknown rejected, val / test | Known rejected, test S | Open-set accuracy, val / test |
|---|---|---|---|---|---|---|---|---|
| 5a (no `<unk>`) | 55 | 0.873 | 0.854 / 0.849 | | 0.870 | | | |
| 6a, weight 1 | 22 | 0.862 | 0.854 / 0.815 | 0.817 | 0.850 | 0.672 / 0.685 | 2.1 % | 0.809 / 0.790 |
| 6b, weight 4 | 24 | 0.850 | 0.793 / 0.812 | 0.820 | 0.822 | 0.820 / 0.830 | 3.9 % | 0.840 / 0.825 |
| 6c, weight 8 | 32 | 0.838 | 0.819 / 0.796 | 0.819 | 0.815 | 0.872 / 0.887 | 5.1 % | 0.855 / 0.841 |

Validation per epoch (Words_200 macro-F1 / unknown rejected / open-set accuracy):

| Epoch | 6a weight 1 | 6b weight 4 | 6c weight 8 | 5a (W macro-F1) |
|---|---|---|---|---|
| 10 | 0.823 / 0.610 / 0.777 | 0.798 / 0.725 / 0.803 | 0.768 / 0.835 / 0.804 | 0.803 |
| 20 | 0.842 / 0.678 / 0.802 | 0.805 / 0.830 / 0.843 | 0.803 / 0.861 / 0.841 | 0.825 |
| 30 | 0.850 / 0.715 / 0.826 | 0.839 / 0.870 / 0.863 | 0.807 / 0.867 / 0.849 | 0.846 |
| 40 | 0.850 / 0.818 / 0.840 | 0.825 / 0.878 / 0.864 | 0.817 / 0.886 / 0.867 | 0.869 |
| 50 | 0.844 / 0.849 / 0.868 | 0.819 / 0.896 / 0.878 | 0.816 / 0.913 / 0.864 | 0.863 |
| 60 | 0.842 / 0.855 / 0.859 | 0.824 / 0.902 / 0.875 | 0.809 / 0.914 / 0.872 | 0.861 |

Observations (run 6):
- **The `<unk>` class works:** 69–89 % of unknown signs on test are rejected, while 2–5 % of known
  signs are wrongly rejected. The weight moves along that trade-off.
- **It costs known-word accuracy:** test Words_200 macro-F1 drops from 0.849 to 0.796–0.815, also when
  the `<unk>` output is ignored (0.817–0.820). So it is not only false rejections: the shared layers
  fit the known words a little less well.
- **The checkpoint rule now works against the new class.** Rejection keeps improving until the end
  (weight 1: 0.672 at the chosen epoch 22, 0.855 at epoch 60), while Words_200 macro-F1 peaks early.
  So the saved models are early ones with weaker rejection. Next run: choose the checkpoint on val
  open-set accuracy.
- Which model is better for word mode is decided on the val sentences (E6, notebook 05 v3).

### Run 7: unknown class, checkpoint on open-set accuracy (split v6)

Run 6 with one change: the checkpoint is the epoch with the best val open-set accuracy
(`select='open'`), not the best val Words_200 macro-F1. Weights 1 and 4. Notebook 04 v7, 2 × ~42 min.

| Run | Checkpoint rule | Best epoch | Val W macro-F1 | Val unknown rejected | Val open-set acc. | Test W macro-F1 | Test W macro-F1, known only | Test unknown rejected | Known rejected, test S | Test open-set acc. |
|---|---|---|---|---|---|---|---|---|---|---|
| 6a, weight 1 | W macro-F1 | 22 | 0.862 | 0.672 | 0.809 | 0.815 | 0.817 | 0.685 | 2.1 % | 0.790 |
| 6b, weight 4 | W macro-F1 | 24 | 0.850 | 0.820 | 0.840 | 0.812 | 0.820 | 0.830 | 3.9 % | 0.825 |
| 6c, weight 8 | W macro-F1 | 32 | 0.838 | 0.872 | 0.855 | 0.796 | 0.819 | 0.887 | 5.1 % | 0.841 |
| **7a, weight 1** | open-set acc. | 54 | 0.853 | 0.848 | 0.869 | 0.827 | **0.847** | 0.863 | 3.1 % | 0.854 |
| **7b, weight 4** | open-set acc. | 53 | 0.812 | 0.896 | **0.875** | 0.813 | 0.846 | **0.911** | 5.0 % | **0.864** |

Observations (run 7):
- **The later checkpoints reject unknown signs much better at no cost to the known words.** At
  weight 1, rejection rises from 0.685 to 0.863 on test, and the known-only Words_200 macro-F1 goes
  back up to 0.847, against 0.849 for 5a, which has no unknown class.
- Run 6's checkpoints cannot be fetched any more: the Kaggle CLI serves only the latest version of a
  notebook's output and ignores a version suffix. Notebook 05 now saves the model it picks.

### Run 8: every learnable word (split v6)

Up to run 7 the vocabulary was the 200 words of Words_200, which cover 63 % of the signs annotated in
the sentences. AzSLD's sentences contain 841 glosses. Run 8 learns every gloss with at least 10 signs of
train signers: **768 words** (the 200 plus 568), covering **99.0 %** of the annotated signs. The median
gloss has 16 train signs from 15 different signers, and 382 of the 768 occur in only one sentence. The
remaining 53 rare glosses (231 train signs) form the unknown class, so its job changes: in AzSLD it is
now rare. Setup otherwise 7b (no face, unknown weight 4, checkpoint on val open-set accuracy), seeds 1
and 2. Notebook 04 v8, 2 × 44 min on a T4.

Test, best-on-val checkpoint. `S` = sentence-cut signs of the test signers: for 7b only the 200 words,
for run 8 all 768. "S, 200 words" = run 8 scored on the signs of the 200 words only (argmax over all
769 outputs), the number comparable with 7b. Merged = the 13 confirmed word-form families.

| Run | Classes | Best epoch | Val W macro-F1 | Test W top-1 / top-5 / macro-F1 / balanced | Test W merged top-1 / macro-F1 | Test S (all words) top-1 / top-5 / macro-F1 | Test S, 200 words: top-1 / macro-F1 | Test unknown rejected |
|---|---|---|---|---|---|---|---|---|
| 7b (reference) | 201 | 53 | 0.812 | 0.812 / — / 0.813 / — | — | (200 words only) | 0.838 / 0.827 | 0.911 of 2,652 |
| **8a, seed 1** | 769 | 50 | 0.851 | 0.851 / 0.985 / 0.835 / 0.835 | 0.928 / 0.873 | **0.833 / 0.970 / 0.786** | 0.830 / 0.827 | 0.353 of 51 |
| **8b, seed 2** | 769 | 58 | 0.840 | 0.823 / 0.984 / 0.831 / 0.829 | 0.916 / 0.874 | **0.831 / 0.975 / 0.790** | 0.824 / 0.830 | 0.490 of 51 |

Per word (8a, test sentence-cut signs): of the 597 words with at least 3 test signs, 410 reach recall
≥ 0.8 (8b: 417) and 6 have recall 0 (8b: 8); the median recall is 1.0.

Observations (run 8):
- **The model learns 3.8 × as many words at almost no cost to the 200.** On the 200 words' sentence
  signs it scores 0.827 / 0.830 macro-F1, the same as 7b (0.827). On Words_200 it is 0.833 on average,
  higher than 7b (0.813) and inside the spread of the 200-word models without an unknown class (no face,
  three seeds: 0.847 ± 0.016).
- **On all 768 words: test top-1 0.832, macro-F1 0.788** (mean of two seeds), on signers never seen in
  training. Top-5 is 0.97.
- **The remaining errors are mostly not real confusions.** The most-missed words are word forms
  (MƏNİM → MƏN, BİLƏR → BİLMƏK, İSTƏYİRƏM → İSTƏMƏK, GEDƏ → GETMƏK), synonyms that are probably one
  sign (YARDİM, YARDİMİ → KOMƏK), and one word annotated in two spellings (BİR DƏFƏ / BİRDƏFƏ).
- **The unknown class is weak now** (35–49 % of 51 test unknown signs rejected, against 91 % for 7b),
  because it has 231 training signs instead of 12,000. Almost every AzSLD sign now has its own class, so
  word mode shows whether this matters (E6).

### Spelling variants in the annotations

Some words are annotated in two spellings, typed with and without Azerbaijani letters: ÇOX / COX, KÖMƏK
/ KOMƏK, İL / IL, SAĞLAM / SAGLAM, DÖVLƏT / DOVLƏT, … and BİR DƏFƏ / BİRDƏFƏ (14 pairs,
`ml/gloss_spelling.csv`). Up to run 8 each pair was two classes of the same sign. Words_200 contains only
one spelling of each pair, so the benchmark does not change; the merged word takes the Words_200
spelling where there is one. Pairs that are different words stay apart (NECƏ "how" / NEÇƏ "how many",
SƏHƏR / ŞƏHƏR, fingerspelled letters such as S / Ş).

Run 8's test predictions with the pairs merged afterwards (no retraining): sentence-cut top-1 / macro-F1
0.833 / 0.786 → 0.839 / 0.792 (8a) and 0.831 / 0.790 → 0.837 / 0.796 (8b); with the word-form families
merged too, 0.870 / 0.802 and 0.873 / 0.804. From run 9 the pairs are merged in training (756 words), and
notebook 05 merges them in the references and in every model's labels.

### Run 9: spelling variants merged, and the side camera (split v6)

Run 8 setup with the 14 spelling pairs merged in training (756 words + unknown). 9a is the frontal camera
only. 9b and 9c add the side camera's view of the train signers' signs (9,463 aligned recordings, E1),
which doubles the training set; 60 epochs as before, so twice the steps. The checkpoint is still chosen
on the frontal val signers only. `Side` = the test signers' signs seen by the side camera (7,126),
never used for any choice. Notebook 04 v9: 47 + 89 + 91 min on a T4.

| Run | Side camera in training | Best epoch | Val W macro-F1 | Test W top-1 / macro-F1 / balanced | Test W merged top-1 / macro-F1 | Test S (756 words) top-1 / macro-F1 | Test S, 200 words: top-1 / macro-F1 | **Test side view** top-1 / macro-F1 | Test W macro-F1 without face |
|---|---|---|---|---|---|---|---|---|---|
| 9a, seed 1 | no | 53 | 0.852 | 0.830 / 0.827 / 0.831 | 0.921 / 0.863 | 0.837 / 0.800 | 0.835 / 0.845 | 0.451 / 0.405 | 0.838 |
| **9b, seed 1** | yes | 51 | 0.858 | 0.847 / **0.857** / 0.863 | 0.934 / 0.894 | **0.844 / 0.807** | 0.833 / 0.844 | **0.779 / 0.749** | 0.801 |
| **9c, seed 2** | yes | 60 | 0.856 | 0.852 / **0.862** / 0.872 | 0.935 / 0.897 | **0.846 / 0.808** | 0.840 / 0.853 | **0.788 / 0.750** | 0.799 |

Words with at least 3 test signs that reach recall ≥ 0.8: 395 of 587 (9a), 430 (9b), 416 (9c).

Observations (run 9):
- **The side camera improves recognition from the front too:** Words_200 macro-F1 0.827 → 0.857 / 0.862
  (+0.03, both seeds), merged 0.863 → 0.894 / 0.897, and all-word sentence signs 0.837 → 0.844 / 0.846
  top-1. The same signs from a second angle act as real (not synthetic) viewpoint augmentation.
- **It makes the model robust to the camera angle:** on the test signers seen from the side, top-1 rises
  from 0.451 to 0.779 / 0.788. A webcam is rarely exactly in front of the user.
- **But without face the side models lose 0.05–0.06** (Words_200 macro-F1 0.857 → 0.801; val 0.858 →
  0.753), while 9a loses nothing (0.827 → 0.838). The side views have no face points, so the "face
  detected" flag told the model which camera it was looking at; with the face tracker off (the browser's
  mode) every input looks like a side view to it. Run 10 removes the flag's information: the same side
  runs with `face_drop` 1.0 (no face in training or scoring).
- Merging the spelling variants (9a vs 8a): test S 0.837 / 0.800 on 756 words against 0.833 / 0.786 on
  768, about what merging run 8's predictions afterwards gave (0.839 / 0.792).

### Run 10: side camera without any face (split v6)

Runs 9b / 9c with `face_drop` 1.0: no face points in training, so the "face detected" flag is always 0,
as in the browser without the face tracker, and the flag can no longer tell the cameras apart. Such a
model is also scored without face (`train.fit`), and its manifest says `works_without_face`, so the
browser never loads the face tracker for it. Notebook 04 v10, 2 × 83 min on a T4.

All numbers without face (the browser's mode). For 9a this is its `*_noface` score; 8a is shown with
face, which it barely uses (E3 run 5).

| Run | Side camera | Best epoch | Val W macro-F1 | Test W top-1 / macro-F1 / balanced | Test W merged top-1 / macro-F1 | Test S (all words) top-1 / macro-F1 | Test S, 200 words: top-1 / macro-F1 | Test side view top-1 / macro-F1 |
|---|---|---|---|---|---|---|---|---|
| 8a (768 words) | no | 50 | 0.851 | 0.851 / 0.835 / 0.835 | 0.928 / 0.873 | 0.833 / 0.786 | 0.830 / 0.827 | — |
| 9a (756 words) | no | 53 | 0.850 | 0.831 / 0.838 / — | — | 0.836 / 0.797 | — | 0.451 / 0.405 (with face) |
| 9b, with face flag | yes | 51 | 0.753 | 0.816 / 0.801 / — | — | 0.801 / 0.753 | — | — |
| **10a, seed 1** | yes | 59 | **0.864** | 0.839 / **0.850** / 0.856 | 0.916 / 0.888 | **0.850 / 0.815** | 0.840 / 0.850 | **0.797 / 0.756** |
| **10b, seed 2** | yes | 51 | 0.851 | 0.858 / **0.846** / 0.851 | 0.929 / 0.878 | **0.851 / 0.812** | 0.844 / 0.855 | **0.789 / 0.755** |

Words with at least 3 test signs that reach recall ≥ 0.8: 428 of 587 (10a), 425 (10b).

Observations (run 10):
- **Without any face, the side camera's gain stays.** Against the frontal-only models in the browser's
  mode, Words_200 macro-F1 0.835–0.838 → 0.846–0.850 and all-word sentence signs 0.833–0.836 → 0.850 /
  0.851 top-1 (macro-F1 0.786–0.797 → 0.812–0.815). Side-view top-1 0.797 / 0.789.
- With face, 9b / 9c were 0.857 / 0.862 on Words_200: a little higher, but only with the face tracker,
  which the browser does not run. 10a / 10b are the best models for the browser so far.
- The two seeds agree within 0.004 on every test metric above except Words_200 top-1 (MƏN / MƏNİM, E3).

### Run 11: the final model, on train + val signers (split v6)

Everything up to here was chosen on the 7 val signers: the vocabulary rule, the side camera, no face,
seed 1, the checkpoint epoch (10a's best, 59), the word-mode settings (E6). With those fixed, run 11
trains 10a's recipe on the train **and** val signers, 34 instead of 27 (about 41,700 annotated frontal
signs instead of 34,800, plus their side views), and keeps epoch 59 without looking at any score. The 5 test
signers are still never seen, so its test scores stay signer-independent. It has no val scores, and its
word-mode settings are 10a's. Notebook 04 v11, 94 min on a T4 (the run ended in an error in the last,
plotting cell, which assumed val curves; the model and every result had been written).

| Run | Training signers | Test W top-1 / macro-F1 / balanced | Test W merged top-1 / macro-F1 | Test S (756 words) top-1 / top-5 / macro-F1 | Test S, 200 words: top-1 / macro-F1 | Test side view top-1 / macro-F1 |
|---|---|---|---|---|---|---|
| 10a | 27 | 0.839 / 0.850 / 0.856 | 0.916 / 0.888 | 0.850 / 0.977 / 0.815 | 0.840 / 0.850 | 0.797 / 0.756 |
| **11 (final)** | 34 | 0.840 / **0.862** / 0.865 | **0.938 / 0.903** | 0.850 / 0.979 / **0.826** | 0.837 / **0.865** | **0.810 / 0.778** |

- Seven more signers raise every macro-F1 by about 0.01–0.02 (Words_200 0.850 → 0.862, all words
  0.815 → 0.826, side view 0.756 → 0.778) at the same top-1. One run each, so the size of the gain is
  uncertain (seed differences are 0.004–0.02), but it is consistent across all test sets.
- **Run 11 is the browser model.** Test Words_200 macro-F1 0.862 (merged 0.903): the 200-word benchmark
  is as high as the best 200-word models while the model knows 756 words.

### Run 12: 10a without the held-out sentences (for E7)

The 49 sentences marked `heldout_sentence` are the unseen-sentence test of word mode (E6) and of the
continuous model (E7). But every run up to 11 trained on all signs of the train signers, including their
recordings of those sentences. Run 12 is 10a's recipe without them (notebook 04 v12, `EXCLUDE_HELDOUT`):
3,446 of the 34,812 frontal train signs (9.9 %) and their side views are left out; the vocabulary (756
words) stays. 86 min on a T4, best epoch 49.

| Run | Test W top-1 / macro-F1 / balanced | Test S top-1 / top-5 / macro-F1 | Test S, 200 words: top-1 / macro-F1 | Test side view top-1 / macro-F1 | Test open-set accuracy |
|---|---|---|---|---|---|
| 10a | 0.839 / 0.850 / 0.856 | 0.850 / 0.977 / 0.815 | 0.840 / 0.850 | 0.797 / 0.756 | 0.847 |
| 12 (no held-out sentences) | 0.846 / 0.811 / 0.815 | 0.826 / 0.954 / 0.746 | 0.832 / 0.806 | 0.761 / 0.688 | 0.822 |

- Top-1 changes little (S 0.850 → 0.826), macro-F1 drops by 0.04–0.07: the words that occur mostly in the
  held-out sentences lose most of their training signs. Run 12 is for the comparison in E7 only.

### Forms of one sign, found on val (after run 11)

AzSLD labels Azerbaijani word forms, so some classes are one sign. Nigar Alishzade confirmed 13 families
(30 forms). To find more, run 10a's predictions on the **val** signers were used (train signers never
took part in the decision, test signers were not looked at). For every pair of glosses that the model
confuses (≥ 3 confusions and ≥ 20 % of one gloss's val signs, ≥ 3 val signs each), the two-way balanced
accuracy says whether the model can tell them apart when only the two are considered (0.5 = not at all):

| Pair | Val signs | Two-way balanced accuracy | Merged? |
|---|---|---|---|
| HARA / HARDA | 8 / 37 | 0.552 | yes (new) |
| İSTƏMƏK / İSTƏYİRƏM | 145 / 44 | 0.607 | confirmed family |
| YOX / YOXDUR | 34 / 3 | 0.637 | confirmed family |
| KÖMƏK / YARDİM | 17 / 3 | 0.637 | no: different words (synonyms), to be confirmed |
| GEDƏ / GETMƏK | 14 / 57 | 0.671 | confirmed family |
| ORA / ORDA | 18 / 19 | 0.674 | confirmed family |
| TƏHSİL / ÖYRƏNMƏK | 15 / 17 | 0.690 | no: different words, to be confirmed |
| O / ONUN | 53 / 15 | 0.701 | confirmed family |
| YAZI / YAZMAQ | 13 / 6 | 0.712 | yes (new) |
| BİLMƏK / BİLİR | 27 / 14 | 0.714 | yes (new, joins BİLMƏK) |
| BURA / BURDA | 17 / 64 | 0.718 | confirmed family |
| BİZ / MƏN | 39 / 997 | 0.730 | **no**: different signs ("we" / "I"), confused because MƏN is a third of all signs |
| BİLİR / BİLƏR | 14 / 69 | 0.736 | yes (new) |
| YAZA / YAZMAQ | 14 / 6 | 0.738 | yes (new) |
| ALMALIYAM / ALMAQ | 7 / 66 | 0.748 | yes (new) |
| MƏN / MƏNİM | 997 / 9 | 0.758 | confirmed family |
| İŞ / İŞLƏMƏK | 43 / 16 | 0.770 | yes (new, joins İŞLƏMƏK) |
| ALMAQ / ALMIŞAM | 66 / 10 | 0.770 | yes (new) |
| KƏND / RAYON | 10 / 7 | 0.786 | no: different words, to be confirmed |
| SƏHƏR / ŞƏHƏR | 18 / 4 | 0.792 | **no**: different words ("morning" / "city") |
| VAXT / YAŞ | 25 / 34 | 0.802 | no |

Inseparability alone is not enough: the confirmed families score 0.61–0.77, but so do different words
that the model confuses for other reasons (BİZ / MƏN, SƏHƏR / ŞƏHƏR). **Rule:** a pair is merged when the
model cannot separate it on val (≤ 0.80) **and** the two are forms of one word (same first three letters,
or one is the start of the other). The 13 confirmed families stay. This adds HARA, YAZMAQ and ALMAQ, and
BİLİR and İŞ to their confirmed families: 24 glosses are now counted as their sign
(`ml/same_sign.csv` + `ml/lemma_groups.csv`). Synonym pairs (KÖMƏK / YARDİM, TƏHSİL / ÖYRƏNMƏK, KƏND / RAYON)
are listed for Nigar to confirm and are not merged.

From the continuous model on (E7), labels and evaluation use the merged signs; the word-mode numbers
it is compared with are re-scored the same way.

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
| Body world coordinates, 9 points (avatar field `pose_world`, not a model input) | 0.005 m / 0.053 m; all 148 frames; median shoulder width 0.33 m |
| Model input after `preprocess` (656 features), mean abs | 0.0107 (groups 0.002–0.013; max 1 = presence flags in the 8 frames above) |
| Speed, three models | 88 ms per frame (≈ 11 fps), load 0.43 s. Measured in the app's browser pane, so live-camera speed needs its own check |

The differences are far below the training augmentation (±10° rotation, ±15% scale, 0.003 jitter), and
the slot rule never disagrees. The reference used MediaPipe 1.0.1, while the training data was extracted
with 0.10.35; both use the same models.

---

## E5 — Word model in the browser (local recognizer)

Model: run 4d (no face, 400 features, test Words_200 macro-F1 0.863), re-exported from `best.pt`
(6.6 MB ONNX, opset 17), in `public/tercume/assets/word_model/` with its `manifest.json`.
Runtime: `src/tercume/recognizer.js`, onnxruntime-web 1.30.0, WASM, one thread.
Installed models since then:
- **5a** after run 5 (no face, seed 1, picked on validation; test Words_200 macro-F1 0.849, merged
  0.892). Check: max logit difference 4.5e-6, same top-5 on all 8 cases.
- **7b** after run 7: unknown class, weight 4, checkpoint on open-set accuracy. It was picked for word
  mode on the val sentences (E6), and its word-mode settings are in its manifest. Check: max logit
  difference 2.9e-6, same top-5 on all 8 cases. All 8 synthetic cases come out as `<unk>`.
- **Run 11** (final_side_noface): 756 words + unknown, side camera, no face (E3). The browser loads no
  face tracker for it (`works_without_face`). Check: max logit difference 3.3e-6 on all 8 cases.

The tables below are for 4d; E6 has 5a and 7b on the same recording.

**Same input → same output as PyTorch** (`node tools/verify_recognizer.mjs`): the synthetic fixture
cases go through features.js and onnxruntime-web, against `best.pt` on the training forward pass
(`ml/fixtures/word_model_ref.json`):

| Case | Frames | Max abs logit difference | Top-5 order |
|---|---|---|---|
| short, one hand lost for 3 frames (with / without face) | 12 | 1.9e-6 / 1.9e-6 | same |
| no body found (centre from the hands) | 20 | 5.5e-6 / 6.7e-6 | same |
| 70 frames resampled to 64, no face detected | 64 | 1.4e-6 / 1.4e-6 | same |
| privacy mode (face off) | 12 | 1.9e-6 / 1.9e-6 | same |

**A real recording, full chain in the browser.** This is the E4 recording (sentence "Sizin adınız
nədir?", signer p005, *train* split). It checks correctness, not accuracy. Browser = MediaPipe in the
page → `LandmarkClip.toRaw(start − 1, end)` → recognizer. Python = the reference landmarks → features.py
→ PyTorch.

| Sign (frames) | Python + PyTorch | Browser, face on | Browser, face off |
|---|---|---|---|
| SİZ (23–42) | SİZ 0.928 | SİZ 0.927 | SİZ 0.929 |
| AD (46–57) | AD 0.926 | AD 0.926 | AD 0.926 |
| NƏDİR (61–85) | NƏDİR 0.913 | NƏDİR 0.925 | NƏDİR 0.923 |

| Speed (in-app browser pane) | |
|---|---|
| Recognizer load (manifest + 6.6 MB model + 14.2 MB WASM, dev server) | 0.47 s |
| One sign, 12–25 frames | 5–8 ms (first call 26 ms) |
| Landmarks, three models | 60 ms per frame (88 ms in E4; same pane, so the live-camera check is still open) |

Two export bugs, found and fixed on the way:
- **The ONNX graph only took 32 frames.** `nn.MultiheadAttention` fixed the traced length (32) into a
  reshape, so any other clip length fails in onnxruntime (`{20,1,192}` → `{32,4,48}`). This affects every
  `model.onnx` of runs 2–5. Export now writes the same attention out with the same weights. Its outputs
  match the training forward pass to 5e-7 in PyTorch and 2e-6 in onnxruntime at T = 7, 32 and 64.
- **The export check compared against a model in training mode.** `torch.onnx.export` restores the
  wrapper module's mode afterwards, and a new wrapper is in training mode, so dropout and batch
  statistics were on in the comparison. The wrapper is now put in eval mode first.

---

## E6 — Word mode: cutting continuous signing into words (browser)

`src/tercume/s2w.js` (`SignToWords`) + `src/tercume/segment.js`, with the Python reference
`ml/azsl/segment.py`. In the app: Jest → Mətn, "Nə tanınsın: sözlər".

Method (no training; version 2 since the held-out evaluation below, which chose the numbers in
step 3 on val; version 1 used windows of 12, 18 and 26 frames, p ≥ 0.5, overlap ≤ 25 %):
1. **Activity.** A frame is active when a hand is seen with its wrist above the rest line: shoulder
   line + 1.0 shoulder width. In AzSLD the hands rest in the lap, 1.25–1.3 shoulder widths below the
   shoulders, and are mostly not detected there.
2. **Spans.** A span opens after 3 active frames and closes after 8 resting ones, with 8 frames of
   context on both sides (the training margin). Words appear when the hands go down.
3. **Words inside a span.** Continuous signing has no rest between signs, so the model chooses the cuts.
   Candidates are the whole span plus windows of 16, 24 and 32 frames every 3 frames, centred where the
   hands are up. They are scored by top-1 probability, then greedy non-maximum suppression runs (no
   overlap between kept windows, p ≥ 0.8), and neighbours with the same word merge. Isolated words
   (rest → sign → rest, like Words_200) usually win as the whole span. If no window reaches p ≥ 0.8, the
   app shows the most confident one as an unsure guess ("SİZ?", dashed). This is display only and is
   not counted in the evaluation.

**Python and browser give the same cuts** (`node tools/verify_segment.mjs` against
`ml/fixtures/segment_v<N>.json`, written by `ml/make_fixture.py`; equal for v1 and v2):

| Step | Equal |
|---|---|
| frame activity | 8 / 8 random cases (320 frames) |
| spans (online in JS, offline in Python) | 60 / 60 random sequences (135 spans) |
| candidate windows | 30 / 30 |
| choice + merge | 30 / 30 |

**On the E4 recording** ("Sizin adınız nədir?", *train* split: this checks the mechanism, not
accuracy), in the app, video file processed frame by frame:

| | SİZ (annotated 23–42) | AD (46–57) | NƏDİR (61–85) | Span split |
|---|---|---|---|---|
| Python reference, run 4d | SİZ 23–40, 0.932 | AD 44–61, 0.928 | NƏDİR 68–85, 0.925 | 53 model calls |
| Browser, run 4d, 3 trackers | SİZ 22–47, 0.937 | AD 46–57, 0.926 | NƏDİR 58–83, 0.930 | 155 ms |
| Browser, run 5a, no face tracker | SİZ 25–45, 0.923 | AD 46–57, 0.915 | NƏDİR 61–86, 0.919 | 164 ms |
| Browser, run 5a, v2 defaults | SİZ 22–37, 0.924 | AD 46–61, 0.908 | NƏDİR 67–90, 0.918 | 0.56–0.61 s (56 windows) |

The whole span as one sign gives SİZ 0.244, so cutting is needed for sentences. The first prototype,
without the "centre where the hands are up" rule and without merging, found the three words plus a
duplicate SİZ and a spurious AYAQ while the hands went down.

| Speed (in-app browser pane, this recording) | |
|---|---|
| Landmarks, hands + body (5a: no face tracker, see run 5) | 39.9 ms per frame (25 fps) |
| Landmarks, hands + body + face | 52.5 ms per frame (19 fps) |
| One window, 12–32 frames | 3–8 ms earlier in the session, 7–12 ms later (machine load) |
| Splitting one 70-frame span (~50 windows) | 0.16–0.6 s after the hands go down |

The face tracker is loaded only if the model needs it: when the manifest's validation macro-F1
without face is more than 0.01 below with face (`needsFace` in s2w.js). For 5a the gap is 0.0007.

A bug found on the way: MediaPipe's video mode needs increasing timestamps, and a second video (or a
video after the camera) restarted them at 1 ms and failed. Word mode now keeps one increasing clock.

### Held-out sentences (notebook 05, signers never seen in training)

Every val / test sentence recording runs through word mode as in the browser: run 5a, face off,
activity → spans → windows → non-maximum suppression (`ml/azsl/segment_eval.py`). The detected words
are matched with the annotated glosses: same word, frames overlap, one to one by largest overlap.

| | Val | Test |
|---|---|---|
| Recordings | 1,545 | 1,484 |
| Annotated signs | 6,891 | 7,577 |
| … in the 200-word vocabulary | 4,668 (68 %) | 4,663 (62 %) |

- **Recall:** the share of in-vocabulary annotated signs that were found as the right word.
- **Precision:** the share of all detections that are right. A word shown on an out-of-vocabulary sign
  still counts as wrong.
- **Precision, in-vocabulary:** the same, leaving out detections that only overlap out-of-vocabulary
  signs.
- **WER:** word error rate of the detected sequence against the in-vocabulary glosses.
- **Merged:** the confirmed word-form families count as one word.

All 270 configurations (`segment_eval.GRID`: rest line, window lengths, p threshold, overlap) were
compared **on val only**. Test was scored once, only with three configurations fixed beforehand:

| Configuration | Val P / R / F1 | **Test P / R / F1** | Test P, in-vocab. | Test WER (in-vocab.) | Test F1 merged | Boundary error, start / end |
|---|---|---|---|---|---|---|
| **Chosen on val = v2**: windows 16/24/32, p ≥ 0.8, no overlap, rest line 1.0 | 0.642 / 0.775 / 0.703 | **0.582 / 0.724 / 0.645** | 0.812 | 0.64 (0.33) | 0.677 | 4.4 / 5.0 frames |
| v1 defaults: windows 12/18/26, p ≥ 0.5, overlap ≤ 25 % | 0.457 / 0.903 / 0.607 | 0.413 / 0.859 / 0.558 | 0.649 | 1.23 (0.48) | 0.593 | 3.7 / 4.6 frames |
| Baseline: each span is one word (best on val) | 0.752 / 0.147 / 0.246 | 0.691 / 0.131 / 0.220 | 0.712 | 0.87 (0.87) | 0.227 | 44 / 38 frames |
| Oracle: annotated boundaries given (top-1) | recall 0.892 | recall 0.870 | | | merged 0.927 | |

On val, the window lengths matter most: best F1 is 0.703 with 16/24/32, against 0.682–0.686 with the
other sets. The p threshold comes next: best F1 rises from 0.663 at p ≥ 0.3 to 0.703 at p ≥ 0.8. The
rest line barely matters (best F1 0.698–0.703 for 0.8 / 1.0 / 1.2 shoulder widths).

Observations (held-out):
- **v1 found about two words per real sign** on unseen signers: 9,709 detections for 4,663
  in-vocabulary signs. v2 outputs 5,805 and raises test F1 from 0.558 to 0.645.
- **Most remaining errors are signs outside the vocabulary.** 32–38 % of the annotated signs in these
  sentences are not among the 200 words, and the model gives them a known word with high confidence.
  Leaving those detections out, precision is 0.81, and WER drops from 0.64 to 0.33. Next experiment:
  an "unknown sign" class, trained on the out-of-vocabulary sentence signs of the train signers.
- **Cutting costs about 15 points of recall:** 0.72 found against 0.87 when the boundaries are given.
  Boundaries of the found words are within 4–5 frames (0.15 s) of the annotation.
- **Per word (test, ≥ 10 signs):** 65 of 105 words have recall ≥ 0.8, none has 0. The lowest are
  word forms of a family: MƏNİM 0.02 (found as MƏN), ONUN 0.26 (O), BİLƏR 0.39 (BİLMƏK), ORDA 0.31
  (ORA), İSTƏYİRƏM 0.47 (İSTƏMƏK). Merged scoring recovers part of this (F1 0.645 → 0.677). Also
  ÜÇÜN 0.48 and "2" 0.47.
- Treating each span as one word is precise (0.69) but finds only 13 % of the signs: in AzSLD
  sentences the hands rarely rest between words.

### With the unknown-sign class (notebook 05 v3, run 6 models)

The three run 6 models (E3) went through the same held-out evaluation. To keep the run short, the grid
was narrowed to the window set and rest line that won before (16/24/32 frames, 1.0), with a wider
threshold range (p ≥ 0.2 … 0.9) and overlap 0 / 25 / 50 %. The model and the configuration were
chosen on val; test was scored once.

Best val F1 per model (over overlap), by threshold:

| min p | 6a weight 1 | 6b weight 4 | 6c weight 8 |
|---|---|---|---|
| 0.3 | 0.704 | 0.710 | 0.729 |
| 0.4 | 0.707 | 0.715 | 0.733 |
| 0.5 | 0.713 | 0.720 | **0.736** |
| 0.6 | 0.711 | 0.725 | 0.734 |
| 0.7 | 0.693 | 0.726 | 0.729 |
| 0.8 | 0.627 | 0.722 | 0.706 |
| 0.9 | 0.540 | 0.542 | 0.562 |

Chosen on val: **6c (weight 8), p ≥ 0.5, overlap ≤ 25 %**. Test, against the browser model so far:

| Model + settings | Val F1 | Test P / R / F1 | Test WER | Test F1 merged | Test P, in-vocab. |
|---|---|---|---|---|---|
| 5a + v2 (p ≥ 0.8, no overlap), browser so far | 0.703 | 0.582 / 0.724 / 0.645 | 0.640 | 0.677 | 0.812 |
| **6c + p ≥ 0.5, overlap ≤ 25 %** | **0.736** | **0.671 / 0.724 / 0.696** | **0.487** | **0.749** | 0.768 |
| 6c + v2 (p ≥ 0.8, no overlap) | 0.671 | 0.762 / 0.520 / 0.618 | 0.540 | 0.638 | 0.870 |

The oracle (annotated boundaries given) is lower for 6c than for 5a: test top-1 0.814 vs 0.870. That is
the known-word cost seen in E3, and 2–5 % of known signs are now rejected.

Observations:
- **The unknown class improves word mode on unseen signers**: test F1 0.645 → 0.696 and WER 0.640 →
  0.487 at the same recall (0.724). Precision rises from 0.58 to 0.67 because fewer
  out-of-vocabulary signs are shown as words.
- **The best threshold depends on the model.** With an unknown class, windows over unknown signs are
  already absorbed by `<unk>`, so a lower threshold (0.5) keeps more real words. The v2 threshold (0.8)
  costs 6c a fifth of its recall. So the word-mode settings now travel with the model: `manifest.json`
  → `segment` (written by `make_manifest.py --segment`), and the browser applies them.
- 6c was trained with run 6's checkpoint rule (val Words_200 macro-F1), which saved an early
  checkpoint. Run 7 chooses on open-set accuracy instead.

### Run 7 models in word mode (notebook 05 v4)

The same narrowed grid as v3, so val F1 can be compared with the run 6 models directly.

| min p | 7a weight 1 | 7b weight 4 |
|---|---|---|
| 0.3 | 0.755 | 0.766 |
| 0.5 | 0.762 | 0.775 |
| 0.6 | 0.764 | 0.779 |
| 0.7 | 0.767 | **0.782** |
| 0.8 | 0.765 | 0.772 |
| 0.9 | 0.698 | 0.670 |

Chosen on val: **7b, p ≥ 0.7, overlap ≤ 50 %** (val F1 0.782; best run 6 model 0.736, 5a 0.703). Test,
scored once:

| Model + settings | Val F1 | Test P / R / F1 | Test WER (in-vocab.) | Test F1 merged | Test P, in-vocab. | Oracle top-1, test |
|---|---|---|---|---|---|---|
| 5a + v2, browser until now | 0.703 | 0.582 / 0.724 / 0.645 | 0.640 (0.33) | 0.677 | 0.812 | 0.870 |
| 6c + p ≥ 0.5, overlap ≤ 25 % | 0.736 | 0.671 / 0.724 / 0.696 | 0.487 | 0.749 | 0.768 | 0.814 |
| **7b + p ≥ 0.7, overlap ≤ 50 %** | **0.782** | **0.709 / 0.763 / 0.735** | **0.445 (0.33)** | **0.765** | 0.807 | 0.838 |
| 7b + v2 (p ≥ 0.8, no overlap) | 0.723 | 0.768 / 0.611 / 0.680 | 0.465 | 0.710 | 0.860 | |

- **On unseen signers, word mode improves from F1 0.645 to 0.735** (WER 0.640 → 0.445). Precision
  and recall both rise: precision because unknown signs are rejected, recall because the later
  checkpoint names the known words better (oracle 0.814 → 0.838).
- **7b is the browser model now**, with its val settings in `manifest.json` → `segment`. In the app
  it reads the E4 recording as SİZ 0.93, AD 0.91, NƏDİR 0.93 (train split, mechanism check). A span
  whose best window is `<unk>` shows a "naməlum" chip.

### Every learnable word in word mode (notebook 05 v5, run 8 models)

Up to v4, recall and WER counted only the annotated signs of the 200 words, which is what a 200-word
model can know. A model with 768 words has to be compared on the same reference, so v5 adds metrics over
**all annotated signs**: `recall_all` = matched / all annotated signs, `f1_all` with the same precision,
`wer_all` against every annotated gloss. The model and its configuration are chosen on val `f1_all`;
test is scored once per model at its own val-best configuration. From v5 the spelling variants (E3)
count as one word in the references and in every model's labels, so 7b's numbers differ slightly from v4.

Best val `f1_all` per model (over overlap), by threshold:

| min p | 7b (200 words) | 8a (768, seed 1) | 8b (768, seed 2) |
|---|---|---|---|
| 0.3 | 0.632 | 0.754 | 0.742 |
| 0.5 | 0.639 | 0.770 | 0.758 |
| 0.6 | **0.641** | 0.775 | **0.765** |
| 0.7 | 0.641 | **0.779** | 0.764 |
| 0.8 | 0.628 | 0.765 | 0.755 |
| 0.9 | 0.527 | 0.637 | 0.627 |

Test (1,484 recordings of the 5 test signers, 7,577 annotated signs), each model at its val-best setting:

| Model + settings | Val f1_all | Test precision | Test recall_all | **Test f1_all** | **Test wer_all** | Matched signs | Test f1 / WER, own vocabulary | Test f1 merged |
|---|---|---|---|---|---|---|---|---|
| 7b, p ≥ 0.6, overlap ≤ 50 % | 0.641 | 0.699 | 0.488 | 0.575 | 0.567 | 3,697 | 0.737 / 0.451 | 0.769 |
| **8a, p ≥ 0.7, overlap ≤ 25 %** (chosen on val) | **0.779** | **0.785** | 0.745 | **0.765** | **0.319** | 5,646 | 0.768 / 0.317 | 0.789 |
| 8b, p ≥ 0.6, overlap ≤ 25 % | 0.765 | 0.752 | 0.767 | 0.759 | 0.331 | 5,813 | 0.763 / 0.330 | 0.790 |
| 8a + v2 defaults (p ≥ 0.8, no overlap) | 0.736 | 0.824 | 0.629 | 0.714 | 0.389 | | | |

Oracle (annotated boundaries given), test top-1 of all annotated signs: 7b 0.519, 8a 0.829, 8b 0.829.

Observations:
- **Learning every word changes word mode most:** on whole sentences of unseen signers, 8a finds
  5,646 of the 7,577 signs instead of 3,697 (recall 0.488 → 0.745). F1 over all signs goes from 0.575
  to 0.765, and the word error rate from 0.567 to 0.319.
- **Precision rises too** (0.699 → 0.785), although the unknown class is now weak (run 8). A sign that
  used to be "unknown" is now mostly a word the model knows, so it does not have to be rejected.
- Even counted on its own vocabulary, 8a is at least as good as 7b (F1 0.768 vs 0.737, WER 0.317 vs
  0.451), so the extra words do not cost the 200.
- The two seeds differ by 0.006 in test `f1_all`; val chose 8a.

### Spelling variants merged in training (notebook 05 v6, run 9a)

Same evaluation as v5. 8a was scored again as the reference and gave identical numbers.

| Model + settings | Val f1_all | Test precision | Test recall_all | Test f1_all | Test wer_all | Test f1 merged |
|---|---|---|---|---|---|---|
| 8a (768 words), p ≥ 0.7, overlap ≤ 25 % | **0.779** | 0.785 | 0.745 | **0.765** | **0.319** | 0.789 |
| 9a (756 words, spellings merged), p ≥ 0.6, overlap ≤ 25 % | 0.773 | 0.753 | 0.768 | 0.760 | 0.330 | 0.791 |

Merging the spelling variants makes no difference in word mode (−0.006 val, −0.005 test, one seed each,
inside the 8a / 8b seed difference). It stays, as a correction of the labels, not as a gain.

### The side camera in word mode (notebook 05 v7, run 10 models)

Same evaluation, no face (the browser's mode). Best val `f1_all` by threshold:

| min p | 8a (front only) | 10a (+ side, seed 1) | 10b (+ side, seed 2) |
|---|---|---|---|
| 0.5 | 0.770 | 0.757 | 0.753 |
| 0.6 | 0.775 | 0.763 | 0.759 |
| 0.7 | **0.779** | **0.767** | **0.759** |
| 0.8 | 0.765 | 0.753 | 0.735 |

| Model + settings | Val f1_all | Test precision | Test recall_all | Test f1_all | Test wer_all | Test f1 merged | Oracle top-1 of all signs, val / test |
|---|---|---|---|---|---|---|---|
| 8a, p ≥ 0.7, overlap ≤ 25 % | **0.779** | 0.785 | 0.745 | 0.765 | 0.319 | 0.789 | 0.850 / 0.829 |
| 10a, p ≥ 0.7, overlap ≤ 50 % | 0.767 | 0.750 | 0.795 | **0.772** | 0.345 | 0.793 | 0.847 / 0.842 |
| 10b, p ≥ 0.7, overlap ≤ 25 % | 0.759 | 0.776 | 0.763 | 0.769 | 0.320 | 0.797 | 0.847 / 0.843 |

- **On frontal sentence recordings the side camera neither helps nor hurts word mode.** Averaged over
  the two seeds of each recipe, it is −0.009 on val (0.772 → 0.763) and +0.008 on test (0.762 → 0.770);
  the seed-to-seed difference is 0.006–0.014. With the true boundaries (oracle) the picture is the same:
  val 0.850 vs 0.847, test 0.829 vs 0.842.
- Val alone, on this criterion, picks 8a. What frontal word mode cannot see is the camera angle: on the
  val signers seen from the side, single-sign top-1 is 0.841 for 10a against 0.550 for the frontal-only
  9a, and a webcam is rarely exactly in front of the user.
- **Choice for the browser: run 10a's recipe** (side camera, no face, seed 1, p ≥ 0.7, overlap ≤ 50 %),
  for equal frontal word mode and much better robustness to the angle. Both reasons come from val
  numbers; the test numbers were not used for the choice. Run 11 trains this recipe on train + val
  signers (E3).

### The final model in word mode (notebook 05 v9, run 11)

Run 11 was trained on the val signers too, so it cannot be tuned on them: its settings are 10a's val
choice (p ≥ 0.7, overlap ≤ 50 %), fixed before its test (`FIXED`). 10a was scored again in the same run
and gave the same numbers as v7. (v8 stopped at the start: run 11's notebook version ended in an error in
its plotting cell, and Kaggle attaches only complete versions as inputs, so the model went in as a private
dataset, `kaggle_run.py model-dataset`.)

| Model | Training signers | Test precision | Test recall_all | **Test f1_all** | **Test wer_all** | Test f1 merged | Oracle top-1 of all signs, test |
|---|---|---|---|---|---|---|---|
| 7b (browser until now, 200 words) | 27 | 0.699 | 0.488 | 0.575 | 0.567 | 0.769 | 0.519 |
| 10a | 27 | 0.750 | 0.795 | 0.772 | 0.345 | 0.793 | 0.842 |
| **11 (final, browser)** | 34 | 0.743 | 0.795 | **0.768** | **0.350** | 0.793 | 0.842 |

- In word mode the seven extra signers change nothing (−0.004 f1_all), even with the true boundaries
  (oracle 0.842 for both); their gain shows only in single-sign macro-F1 (E3, run 11).
- **Against the browser model before this work (7b):** on whole sentences of unseen signers, recall over
  all signs 0.488 → 0.795, F1 0.575 → 0.768, word error rate 0.567 → 0.350. The model now knows 756
  words, without the face tracker, and is robust to the camera angle.

**In the app** (dev server, the browser pane, run 11 installed with its word-mode settings): the E4
recording ("Sizin adınız nədir?", train split, a mechanism check) is read as SİZ 0.72 (frames 23–39),
AD 0.93 (41–57), NƏDİR 0.93 (65–81); annotated 23–42, 46–57, 61–85. No face tracker is loaded, 756 words
are listed, 148 frames took 13.9 s (94 ms per frame in the pane).

### Unseen sentences, fairly: run 12 in word mode (notebook 05 v10)

The "unseen sentences" rows above used models trained on the train signers' recordings of those
sentences. Run 12 (E3) is 10a without them. Notebook 05 v10 chose its settings on val as for 10a
(p ≥ 0.7, overlap ≤ 25 %, the same windows): val f1_all 0.750, test 0.740, WER 0.346 (spellings merged).
Re-scored with the same-sign merge (E7), on the test recordings:

| Word model | Seen sentences F1 / WER | Unseen, all signs F1 / WER | Unseen, learnable words only F1 / WER |
|---|---|---|---|
| 10a (trained on the held-out sentences) | 0.795 / 0.322 | 0.776 / 0.337 | 0.814 / 0.307 |
| 12 (not) | 0.795 / 0.290 | **0.572 / 0.526** | **0.702 / 0.423** |

- **The unseen-sentence numbers of word mode above were too high.** On sentences no training recording
  contains, F1 falls from 0.795 to 0.572, and to 0.702 even when only words with training signs count.
  A sign in a new sentence context looks different, and other signers' recordings of the same sentence
  were a large part of what made those sentences easy.
- On the seen sentences run 12 equals 10a (F1 0.795), so the loss is specific to the left-out sentences.

**Open:** a live-webcam test; live-camera speed (the pane is not a real camera).

---

## E7 — Continuous recognition: a sentence in one run (notebook 06)

Word mode (E6) cuts the stream into windows and asks the isolated-sign model about each; the words of a
sentence appear when the hands go down. The continuous model labels **every frame** instead: a word, or
`<blank>` (no sign). It is trained on whole sentence recordings, so it uses all frames, including the
49 % between signs that no model had used (1.05 M frontal frames + the side camera's), and both cameras.
In the browser the words come out while the sentence is being signed.

| | |
|---|---|
| Model | run 10a's network (Conv1D + Transformer, 1.75 M parameters, 400 features, no face) with a label per frame (`SeqNet`) |
| Labels | 734: `<blank>`, 732 words (run 10's vocabulary with the 24 same-sign forms merged, E3), `<unk>` |
| Boundaries | frames within 1 of an annotated boundary are not scored (the annotations are ±1–2 frames uncertain) |
| Train | 6,358 frontal recordings of train signers (1.39 M frames) + 5,775 aligned side views, random 48–160-frame windows, the isolated-sign augmentation; frames weighted by label frequency^−0.5 |
| Left out of training | the 49 sentences reserved as unseen (`heldout_sentence`): 717 train recordings |
| Val / test | 1,545 / 1,484 frontal recordings (166 / 165 of unseen sentences) + their side views |
| Checkpoint | best val F1 over all annotated signs (default decoding) |
| As in the browser | the model runs on the last 96 frames every 8 frames; a frame is fixed 24 frames (0.8 s) after it; decode settings chosen on val, test scored once |
| Word mode, same terms | 10a's test detections (E6 v9) re-scored with the same-sign merge: F1 0.793, WER 0.324; on the unseen sentences F1 0.776, WER 0.337 |

### Notebook 06 v2: from scratch vs from run 10a

Per epoch (seq_init, default decoding, whole recordings): val frame accuracy 0.810 → 0.905, val F1
0.755 → 0.843 (epoch 1 → 58). From scratch reaches the same by epoch 50 (F1 0.839). 60 epochs, 67–69 min.

| Run | Best epoch | Decoding (val) | Val F1 / WER | Test precision / recall | **Test F1 / WER** | Test frame accuracy | Test side view F1 | **Unseen sentences (test, 165)** F1 / WER |
|---|---|---|---|---|---|---|---|---|
| seq_scratch | 50 | smooth 9, p ≥ 0.3, ≥ 4 frames, gap 4 | 0.907 / 0.136 | 0.919 / 0.900 | **0.909 / 0.130** | 0.904 | 0.887 | 0.584 / 0.513 |
| seq_init (from 10a) | 58 | smooth 5, p ≥ 0.4, ≥ 8 frames, gap 4 | 0.909 / 0.132 | 0.925 / 0.892 | **0.908 / 0.130** | 0.906 | 0.892 | 0.609 / 0.482 |
| word mode (10a), for comparison | — | — | — | 0.770 / 0.816 | 0.793 / 0.324 | — | — | 0.776 / 0.337 |

Latency on val (seq_init): fixing a frame 12 frames after it gives F1 0.894, 24 frames 0.909, 40 frames
0.913; the whole recording at once 0.924.

Observations (v2):
- **On AzSLD's sentences the continuous model is far better than word mode:** test F1 0.908 against
  0.793, word error rate 0.130 against 0.324, on signers it never saw. The words come out 0.8 s after
  they are signed, with no pause between them.
- **On sentences it never trained on it is much worse** (F1 0.61 against 0.78). Two reasons, both
  measured: 19 % of the signs in those sentences are words with no training sign outside them (word mode
  had them, the continuous model could not), and that alone explains only part of the drop (frame accuracy
  would fall to about 0.82; it falls to 0.726). The rest is the model leaning on the word order of the 492
  training sentences: every test signer signs the same sentences as the train signers.
- So v2 is a sentence recognizer for AzSLD's sentences, not yet a general one. Real users sign new
  sentences, so the number that matters is the unseen-sentence one. v3 tries to fix that.
- Starting from 10a changes little in the end (test F1 0.908 vs 0.909; unseen 0.609 vs 0.584).
- **Corrected in v5:** the word-mode numbers here come from 10a, which had trained on the held-out
  sentences. A word model that has not (run 12) gets 0.572 on them, so seq_scratch (0.584) was not behind.

### Notebook 06 v3: spliced sentences and short context, chosen on unseen sentences

Two ways to stop the model leaning on the word order of the 492 sentences, both started from 10a and both
choosing the checkpoint on the val recordings of the unseen sentences (`select='f1_unseen'`, all signs):

- **seq_splice:** half the training samples are new sentences: 2–6 signs from different recordings of the
  same camera in random order, moved to the first sign's shoulder position, joined by a 3-frame cross-fade
  that is not scored.
- **seq_short:** 24–64-frame training windows (one or two signs of context), streamed with 48-frame
  windows and a 16-frame (0.53 s) delay.

| Run | Best epoch | Val F1 | Val unseen F1 | Test F1 / WER | Test side view F1 | **Test unseen F1 / WER** | Test unseen, learnable only (notebook) |
|---|---|---|---|---|---|---|---|
| seq_splice | 1 | 0.783 | 0.695 | 0.795 / 0.286 | 0.737 | 0.629 / 0.472 | 0.689 |
| seq_short | 1 | 0.789 | 0.695 | 0.791 / 0.302 | 0.722 | 0.635 / 0.471 | 0.705 |

Per epoch (default decoding), the val unseen F1 is highest at epoch 1 (0.583 / 0.589), falls to
0.46 / 0.42 by epochs 3–8 and climbs back to 0.53 / 0.52 by epoch 60, while the val F1 over all
recordings rises to 0.804 / 0.819 (v2: 0.843). So both runs kept epoch 1, which is 10a with one epoch of
training.

**Why, and what was wrong with the measurement.**
- **10a has seen the held-out sentences.** Notebook 04 trains on every sign of the train signers,
  including their recordings of the 49 held-out sentences. A sentence model started from 10a knows those
  sentences' words at epoch 1, also the 19 % of signs whose word occurs only there, and forgets them while
  training on sentences that never contain them. Word mode with 10a is not an unseen-sentence test either.
- **The selection counted unlearnable signs.** With 19 % of the unseen sentences' signs out of reach, the
  per-epoch score mostly measured that forgetting, not word order.
- **The "learnable only" filter was too strict for word mode:** it removed the other signs' annotations
  but kept the detections on them as errors. From v5 on, detections that overlap only removed signs are
  dropped (`continuous.score(keep=...)`).

Re-scored the same way (test, 165 recordings of unseen sentences, words with ≥ 10 train signs outside the
held-out sentences; all systems with the same-sign merge):

| System | Seen sentences F1 / WER | Unseen, all signs F1 / WER | **Unseen, learnable only F1 / WER** |
|---|---|---|---|
| word mode, 10a (has seen the held-out sentences) | 0.795 / 0.322 | 0.776 / 0.337 | 0.814 / 0.307 |
| seq_splice (epoch 1, from 10a) | 0.815 / 0.264 | 0.629 / 0.472 | 0.770 / 0.324 |
| seq_short (epoch 1, from 10a) | 0.809 / 0.281 | 0.635 / 0.471 | 0.773 / 0.330 |

**Sentence suggestion** (`ml/sentence_match.py`: nearest of the 492 sentences, sign sequences from train
signers' annotations; one test recording = one sentence). With the annotated signs as input (a perfect
recognizer) it is right for 97.4 % of the test recordings at the browser's threshold (distance ≤ 0.34),
97.0 % for unseen sentences. With seq_splice's detections: a suggestion for 72.8 % of the recordings,
95.7 % of them right (69.7 % of all; seen sentences 73.8 %, unseen 37.6 %). The threshold was set before
and is not tuned on test.

### Notebook 06 v5: a fair test on unseen sentences

Nothing in this comparison has seen the 49 held-out sentences: word mode uses run 12 (E3, scored in E6 by
notebook 05 v10), and the three continuous runs are trained from scratch. Each continuous run chooses its
checkpoint and its decoding on the val recordings of the unseen sentences, counting only words with
≥ 10 train signs outside them (`select='f1_unseen_learnable'`; detections on the other signs are then
neither right nor wrong). The three runs: v2's recipe (`seq_plain`), half the samples spliced
(`seq_splice`), 24–64-frame windows with a 48-frame stream (`seq_short`).

| Run | Best epoch | Minutes | Val F1 / WER | Val unseen F1 / WER | **Val unseen, learnable** F1 / WER | Test F1 / WER | Test side view F1 / WER |
|---|---|---|---|---|---|---|---|
| seq_plain | 48 | 88 | 0.901 / 0.140 | 0.618 / 0.477 | 0.747 / 0.340 | 0.905 / 0.132 | 0.886 / 0.157 |
| **seq_splice** | 59 | 97 | 0.886 / 0.158 | 0.668 / 0.408 | **0.811 / 0.247** | 0.893 / 0.143 | 0.859 / 0.189 |
| seq_short | 57 | 67 | 0.881 / 0.169 | 0.623 / 0.470 | 0.758 / 0.305 | 0.883 / 0.165 | 0.854 / 0.205 |

All three chose decoding with ≥ 10 frames per word, p ≥ 0.3, gap 2 (smoothing 3 / 3 / 1). Per epoch
(default decoding) every curve now rises to the end (seq_splice val unseen learnable 0.494 at epoch 10,
0.609 at 39, 0.625 at 59): with nothing seen before, there is nothing to forget.

Test, every system scored the same way (`same_sign` merge; seen = the 1,319 test recordings of the other
443 sentences, unseen = the 165 of the held-out ones):

| System | Seen sentences F1 / WER | Unseen, all signs F1 / WER | **Unseen, learnable words** F1 / WER |
|---|---|---|---|
| word mode, run 12 | 0.795 / 0.290 | 0.572 / 0.526 | 0.702 / 0.423 |
| seq_plain | **0.941 / 0.087** | 0.581 / 0.505 | 0.718 / 0.369 |
| **seq_splice** | 0.924 / 0.107 | **0.621 / 0.444** | **0.774 / 0.285** |
| seq_short | 0.916 / 0.126 | 0.595 / 0.490 | 0.747 / 0.340 |

Sentence suggestion on test (`sentence_match.py`, browser threshold 0.34):

| System | Suggested | Right of suggested | Right of all: seen / unseen sentences |
|---|---|---|---|
| seq_plain | 0.887 | 0.960 | 0.925 / 0.267 |
| seq_splice | 0.879 | 0.957 | 0.899 / 0.382 |
| seq_short | 0.867 | 0.962 | 0.895 / 0.339 |

Observations (v5):
- **The continuous model is better than word mode on every part of the test.** On sentences no model has
  seen, seq_splice gets F1 0.621 against 0.572 (all signs) and 0.774 against 0.702 (learnable words), with a
  word error rate of 0.285 against 0.423. On the other sentences it is far ahead (0.924 against 0.795).
- **Splicing helps on new sentences** (val unseen learnable 0.811 against 0.747 without it, test 0.774
  against 0.718), at a small cost on the known sentences (0.924 against 0.941): the model leans less on
  the 492 sentences' word order. Short context helps less (0.758 / 0.747) and is the weakest on the
  known sentences.
- **Every model is much weaker on new sentences**, word mode included (E6 v10). The v2 and v3 numbers that
  said otherwise compared with a word model that had trained on those sentences.
- The suggestion is right for 90 % of known-sentence recordings and rarely wrong when it is shown (96 %).
  The held-out sentences are in the list (only the model has not trained on them), and there it is right
  for 38 %, limited by the weaker recognition. A sentence that is not in AzSLD cannot be suggested.
- **Choice (on val): seq_splice's recipe.** The model in the browser is trained with it on all data (v6).

In the app (dev server, the browser pane, seq_splice installed): the E4 recording (train split, a mechanism
check) gives SİZ 0.58 (frames 22–42), AD 0.81 (42–59), NƏDİR 0.82 (59–88) in one run, annotated 23–42,
46–57, 61–85, and the suggestion "Sizin adınız nədir ?" (distance 0). `verify_cont_model.mjs`: logits
within 4.3e-6 of PyTorch, every frame's argmax equal. 148 frames took 12.9 s in the (hidden) pane.

### Notebook 06 v6: the final model, on all sentences and train + val signers

seq_splice's recipe with everything fixed on val before: 60 epochs, saved at epoch 59, decoding smooth 3,
p ≥ 0.3, ≥ 10 frames, gap 2. Trained on the train **and** val signers (34) and **every** sentence, the 49
held-out ones included: about 8,600 frontal recordings and their side views. The 5 test signers stay unseen, so
its test is still signer-independent, but no test sentence is new to it any more. 94 min on a T4.

| Model | Test precision / recall | **Test F1 / WER** | Test frame accuracy | Test side view F1 / WER | Suggestion right (of all / when shown) |
|---|---|---|---|---|---|
| seq_splice (v5: 27 signers, without the held-out sentences) | 0.909 / 0.877 | 0.893 / 0.143 | 0.894 | 0.859 / 0.189 | 0.842 / 0.957 |
| **final_splice** | 0.939 / 0.915 | **0.927 / 0.105** | 0.916 | **0.897 / 0.144** | **0.912 / 0.977** |
| word mode, run 11 (the browser's word model; E6 v9, spellings merged, not the same-sign forms) | | 0.768 / 0.350 | | | |

- **final_splice is the browser's sentence model** (`public/tercume/assets/cont_model/`). On AzSLD sentences
  signed by people it has never seen it finds the words with F1 0.927 and a word error rate of 0.105
  (word mode: 0.768 / 0.350), and names the right AzSLD sentence for 91 % of the recordings.
- For sentences outside AzSLD the honest estimate is v5's: F1 about 0.62 over all signs and 0.77 over the
  words it knows (word mode 0.57 / 0.70). The app says so under the sentence mode.
- `verify_cont_model.mjs`: logits within 4.3e-6 of PyTorch, argmax equal on every frame. In the app the E4
  recording gives SİZ 0.56 (22–42), AD 0.82 (42–59), NƏDİR 0.85 (59–88) and "Sizin adınız nədir ?".

### What the final model still gets wrong (test, frontal camera)

Words_200 clips are cut from the sentence recordings (same recording, same frames, 99 % match an annotation
within 2 frames), so the continuous model already uses all of AzSLD's sentence data of train and val
signers; only the 5 test signers are left out, on purpose. Of final_splice's 7,577 annotated test signs:

| Outcome | Share |
|---|---|
| right | 0.915 |
| missed (nothing detected there) | 0.041 |
| another word detected | 0.034 |
| not in the vocabulary | 0.010 |
| detections with no sign there (of the signs) | 0.005 |

- **Short signs are missed:** recall 0.336 for signs under 8 frames (211), 0.89–0.95 for longer ones.
  The weakest words are fingerspelled letters (E 0.53, N 0.66, İ 0.66, A 0.67, L 0.67; median recall of the
  97 words with ≥ 15 test signs 0.94). Letters and digits are 9.2 % of all signs, median 13 frames long
  (words 22), 28 % shorter than the 10 frames that decoding requires (`min_len` 10).
- **Confusions** are few and mostly between signs that look alike or are forms of one word: MƏN → BİZ (7),
  SİZ → SƏN, HARA → HANSI, İSTƏMƏK → İSTƏYİR, DOĞMAQ → DOĞULMAQ (3–4 each).

### Notebook 06 v7: a shorter minimum for letters (CPU, no training)

Decoding v2 (`CONTINUOUS_VERSION` 2, `continuous.js` and fixture `continuous_v2` with it) takes an optional
`min_len_short` for single-character labels: the 33 letters and digits in the vocabulary. It was chosen on
**all** val recordings with seq_splice (v5, never saw the val signers), v5's other settings kept; then
final_splice, the browser model, was scored on test with it. Both models came from datasets
(`kaggle_run.py model-dataset`), evaluated on Kaggle CPU (no GPU quota).

| min_len_short | 3 | 4 | 5 | **6** | 8 | 10 (= v5) |
|---|---|---|---|---|---|---|
| seq_splice val F1 | 0.8896 | 0.8899 | 0.8901 | **0.8901** | 0.8895 | 0.8862 |

| Model, test | F1 / WER before | **F1 / WER with min_len_short 6** | Side view F1 / WER |
|---|---|---|---|
| seq_splice (v5), all | 0.893 / 0.143 | 0.897 / 0.136 | 0.863 / 0.182 |
| seq_splice, unseen sentences / learnable words | 0.621 / 0.774 | 0.625 / 0.780 | |
| **final_splice (browser)** | 0.927 / 0.105 | **0.931 / 0.097** | 0.902 / 0.136 |

- final_splice: signs under 8 frames are found 0.559 of the time instead of 0.336, right 0.924 instead of
  0.915, missed 0.032 instead of 0.041, detections with no sign there 0.006 instead of 0.005. Letters: L
  0.667 → 0.857, Ə 0.714 → 0.837, A 0.667 → 0.772, Y 0.750 → 0.821; E, N, İ, R and 2 left the 15 weakest.
- Installed with `install_continuous.py --eval`: the manifest takes the new decoding and test scores and
  says `continuous_version` 2 (the browser refuses a model that needs a newer decoding). In the app the E4
  recording reads as before (SİZ AD NƏDİR); 33 single-character labels get the shorter minimum.

### Notebook 06 v8: more splicing, rarer words in the spliced sentences

Its first push was refused (the weekly GPU quota of 30 h was used up); it ran after the reset. v5's protocol (held-out sentences out of
training, from scratch, epoch chosen on val unseen sentences, learnable words), decoding chosen on val from
v5's grid with v7's `min_len_short` 6. Two changes to seq_splice's recipe:

- **splice_75:** three quarters of the samples spliced instead of half.
- **splice_bal:** half spliced, but the spliced signs drawn with weight (signs of their word)^−0.5
  (`splice_power`), as the loss weighs frames: MƏN is a third of all signs, and uniformly drawn signs
  repeat a few frequent words.

| Run | Best epoch | Minutes | Val F1 / WER | **Val unseen, learnable** F1 / WER | Test F1 / WER | Test unseen F1 / learnable F1 | Test side view F1 |
|---|---|---|---|---|---|---|---|
| seq_splice (v5, decoded as in v7) | 59 | 97 | **0.890** / 0.152 | **0.811** / 0.248 | **0.897** / 0.136 | 0.625 / 0.780 | 0.863 |
| splice_75 | 52 | 105 | 0.860 / 0.184 | 0.797 / 0.264 | 0.861 / 0.179 | 0.599 / 0.755 | 0.818 |
| splice_bal | 52 | 96 | 0.882 / 0.159 | 0.797 / 0.260 | 0.887 / 0.149 | 0.617 / 0.779 | 0.848 |

- **Neither helps.** On val unseen sentences both are 0.014 below seq_splice (one run each, so within
  what a seed could change, but not better), and both lose on the known sentences: three quarters spliced
  costs 0.03 F1 there, the balanced draw 0.008. Test agrees. Half of the samples spliced, signs drawn
  uniformly, stays the recipe; the browser model (final_splice) is unchanged.
- Per epoch (default decoding) the three runs reach the same val unseen learnable F1 by epoch 55 (0.62),
  while val F1 over all recordings ends at 0.769 / 0.804 / 0.812: more splicing mostly costs real-sentence
  context, without gaining on new sentences.

### Speed, and slower cameras (notebook 06 v12)

**Model speed:** `tools/bench_cont_model.mjs` runs the browser model as the app does (features.js →
onnxruntime-web WASM, one thread, this laptop): 10.7 ms per 96-frame window (p90 11.8 ms). At 30 fps the
model runs every 8 frames, which leaves 267 ms: the model uses 4 % of it. The landmark trackers (hands,
body) are the slow part; on a weak device they may deliver 10–20 frames a second, and the browser fills
the stream back to 30 fps by linear interpolation (`Resampler`).

**Simulated:** `continuous.camera_resample` gives a recording as a camera at a lower frame rate would
(frames at k/fps, the nearest recorded frame), filled back to 30 fps exactly as the browser does (checked
against `continuous.js` on the same frames: within 4.8e-7, missing points alike). Notebook 06 v12 scores
the models at 15 and 10 fps with the decoding chosen at 30, and trains one more run under v5's protocol:
`splice_fps`, seq_splice's recipe with half the training windows given at a random 8–30 fps
(`fps_aug` 0.5). (Kaggle versions 9 and 10 of this notebook were killed before writing any output, with no
message; a smoke run, version 11, completed, and version 12, the same as 10, ran through in 143 min.)

| Model | 30 fps F1 / WER | 15 fps | 10 fps |
|---|---|---|---|
| **final_splice (browser), test** | 0.931 / 0.097 | 0.927 / 0.102 | 0.923 / 0.109 |
| seq_splice (v5), val | 0.890 / 0.153 | 0.887 / 0.155 | 0.886 / 0.156 |
| splice_fps, val | 0.891 / 0.155 | 0.888 / 0.159 | 0.886 / 0.159 |
| seq_splice, test | 0.897 / 0.136 | 0.894 / 0.139 | 0.889 / 0.146 |
| splice_fps, test | 0.897 / 0.137 | 0.897 / 0.138 | 0.892 / 0.147 |

| Model | Best epoch | Val unseen learnable F1 / WER | Test unseen F1 / learnable F1 | Test side view F1 |
|---|---|---|---|---|
| seq_splice | 59 | 0.811 / 0.248 | 0.625 / 0.779 | 0.863 |
| splice_fps | 51 | 0.818 / 0.236 | 0.637 / 0.795 | 0.862 |

- **A slower camera costs little:** from 30 to 10 fps the browser model loses 0.008 F1 on test, the v5
  model 0.004 on val. The interpolated stream is close enough to what the model was trained on.
- **Training for it is not needed:** splice_fps is level with seq_splice at every frame rate on val. Its
  0.007 on the val unseen sentences is within what one seed changes (v8: 0.014 between runs of similar
  recipes). The browser model is unchanged.

### Fingerspelled letters in sentences (notebook 06 v14)

**AzSLD's fingerspelling set** (`fingerspelling_split.csv`, landmarks from notebook 01) is 959 phone photos
of 24 static letters (40 each, from 21–32 contributors per letter except A and B, one each) and 6,400 frames
of 8 moving letters sampled from clips (a median of 6 frames per clip over 34 frames, too sparse to rebuild
the movement). The photos are portrait close-ups: a hand in 94 %, a body in 30 % of the static ones, a face
almost never. They can teach hand shapes, from another camera and framing than the sentences.

**How the browser model loses letters** (final_splice, decoding v7, test; 759 letter and digit signs, 90.3 %
right): taken for another letter 4 (0.5 %), absorbed by a neighbouring detection 41 (5.4 %: 22 by another
letter in fast spelled runs such as A-D-A, O-N-L-A-Y-N, M-E-S-S-İ-N-İ-N, 10 by the same letter doubled, 9 by a
word), nothing detected 20 (2.6 %), taken for a word 9 (1.2 %, mostly names annotated as one gloss, e.g.
HƏLİMƏ). Hand shapes are not the problem, so the photos would not help here.

**Spelled sequences in training** (`spell`: a share of the samples are 3–8 letter signs of other recordings
in a row, letters drawn with weight n^−0.5, taken before the splice draw), v5's protocol. The `letters` score
counts only letter signs, on the recordings that have them.

| Run | Best epoch | Val F1 | Val unseen learnable F1 | **Val letters** F1 (P / R) | Test F1 | Test letters F1 (P / R) |
|---|---|---|---|---|---|---|
| seq_splice (v5) | 59 | **0.890** | 0.811 | **0.853** (0.871 / 0.836) | **0.897** | **0.875** (0.902 / 0.850) |
| splice_spell15 | 47 | 0.884 | 0.793 | 0.832 (0.853 / 0.812) | 0.889 | 0.857 (0.878 / 0.837) |
| splice_spell30 | 44 | 0.882 | 0.815 | 0.831 (0.863 / 0.801) | 0.892 | 0.867 (0.887 / 0.848) |

- **Spliced letter runs make letters worse,** on val by 0.02 F1, and the known sentences a little too.
  Letters joined from different recordings do not move like real fast fingerspelling, where the hand
  flows from shape to shape in one place.
- **Decoding is not the limit either:** on seq_splice's val grid, letters F1 is 0.855 without smoothing,
  0.853 with the chosen smoothing 3, 0.846 / 0.839 with 5 / 9.
- What is left is fast spelled runs; more real fingerspelled sentences would be the remedy.
- **Isolated letters are another matter:** run 10a's per-class test recall on letter signs cut from the
  sentences is 0.9–1.0 for most letters, the moving ones included (D 0.98, Y 0.96, K 1.00, Z 0.80, Ü 0.80).
  The app's letter mode uses an older model (`public/tercume/assets/model.json`: one hand, 20 frames,
  a dense network, 35 labels with backspace / enter / space), whose own evaluation on the AzSLD photos
  (`eval.json`) is 77.7 % on static letters and 1.9 % on the 8 moving letters.

**Open:** a live-webcam test of the sentence mode; sentences from outside AzSLD signed by a real user.


---

## E8 — Letter mode: fingerspelling on the landmark pipeline (notebook 07)

The app's default mode types a word letter by letter with the hand tracker alone. Its model until now
(`public/tercume/assets/model.json` + `model.bin`, `src/tercume/model.js`) is a dense network on 20 frames ×
the first hand's 21 raw image points, with 32 letters and three control gestures (backspace, enter, space;
the page also has buttons and keys for them). It types a letter only while the hand is still.

**Data** (`azsl/letters.py`): every AzSLD source of letters, the hand alone (body, face and blendshapes
removed, as the letter mode runs no body tracker and most photos show none):

| Source | What | Split | Train / val / test letters |
|---|---|---|---|
| sentences | letter signs annotated in the sentence recordings, frontal + side camera; "O" only next to another spelled letter (alone it is the pronoun "o": 279 of 672 kept) | signers | 2,408 / 415 / 588 frontal, side views alike |
| photos | AzSLD_Fingerspelling static letters, one image repeated 24 frames | contributor | 671 / 144 / 144 photos (143 / 135 val / test with a hand) |
| clips | its moving letters (8), frames put back on their frame numbers and filled in, cut where > 12 frames apart | clip run | from 380 / 52 / 48 runs: 110 / 102 val / test clips |
| `<none>` | gaps of ≥ 12 frames between annotated signs, as many as letter signs | signers | |

Each letter has training data (g, ğ, ö and x only from photos or clips). Slots: the hand present in more
frames of a window is slot 0 (`canonical`), so a lone hand, a right-handed and a left-handed letter look
alike to the model. Model: run 10a's Conv1D + Transformer with dim 128 (0.73 M parameters), 60 epochs,
classes balanced n^−0.5; checkpoint on val macro-F1 over the letters. The old model is scored on the same
samples through a numpy port (`letters.OldModel`, equal to `model.js` within 1.8e-7), its argmax among the
32 letters. Scores: top-1 / macro-F1 over the letters (a letter sample is never answered `<none>`).

### Notebook 07 v2: hand shape + movement, with or without the hand's place in the image

| Run | Features | Best epoch | Val top-1 / macro-F1 | Val photos | Test all | Test sentences | Test side view | Test photos | Test clips (moving letters) |
|---|---|---|---|---|---|---|---|---|---|
| **hand** | hand_shape + hand_vel (252) | 51 | 0.922 / 0.907 | **0.902** / 0.897 | **0.918** / 0.872 | 0.951 / 0.887 | 0.902 / 0.807 | 0.837 / 0.799 | 0.922 / 0.887 |
| hand_pos | + hands, flags (382) | 49 | 0.925 / 0.913 | 0.888 / 0.881 | 0.920 / 0.872 | 0.951 / 0.889 | 0.902 / 0.809 | 0.837 / 0.799 | 0.951 / 0.930 |
| old model | 20 × the first hand's image points | — | 0.309 / 0.294 | 0.832 / 0.764 | 0.214 / 0.278 | 0.167 / 0.177 | 0.121 / 0.139 | **0.830** / 0.809 | 0.196 / 0.166 |

- **The new model reads letters the old one could not:** 0.951 of the letters in the test signers' sentences
  against 0.167, the moving letters 0.922 against 0.196. On the photos both are near 0.84: the old model is
  as good as the new one only there, and the photos may have been its training data.
- The two feature sets are level on val (0.006); `hand` is better on the val photos, which are taken at
  another camera distance, and needs no image position (distance-independent): `hand` is chosen.
- Gaps between signs: 11 % of `<none>` samples get a letter as argmax, 2 % of letters get `<none>`.

### Notebook 07 v3: the final model, and the letter mode in the browser

`hand` on train + val (signers, contributors, clip runs), saved at epoch 51 (`final_hand`, 7 min on a T4).
Test, everything still unseen: **top-1 0.933, macro-F1 0.893** (sentences 0.963 / 0.923, side view 0.921 /
0.858, photos 0.852 / 0.810, clips 0.931 / 0.923). The weakest test letters: p 0.50 (4 signs), c 0.69 (13),
l 0.71 (49), j 0.79 (14), m 0.82 (38).

In the browser (`src/tercume/letters.js`, `SignToText` with `LetterModel`): the hand tracker's frames,
`strip` → `canonical` → `features.preprocess`, the last 24 frames, every `stride` frames, asynchronous.
Moving letters (ç, d, g, k, ö, ü, y, z) skip the "hand still" rule; a typed letter can be typed again when the
prediction changes, `<none>` comes, or the hand leaves. `verify_letters.mjs`: letters.js equals Python on
fixture `letters_v1` (within 1.1e-6), and the installed model equals PyTorch (within 1.8e-7). In the app the
E4 recording (no fingerspelling in it) types nothing: `<none>` keeps every letter's probability near 0.
`loadLetterModel` falls back to the old model if the new one is missing (checked); the NSosyal page uses it too.

### Typing rules, on spelled words (notebook 07 v4, `tools/letters_tune.mjs`)

Every spelled word (≥ 3 letters, ≤ 20 frames apart) of val (67 words, 415 letters) and test signers (104
words, 574 letters), run frame by frame through `SignToText` itself: at the sentences' speed (letters
~13 frames) and slowed 2.5× (linear in between, close to holding each letter ~1 s). Letter error = edit
distance between the typed and the annotated letters / annotated letters. The rules (window 12 / 16 / 24,
stride 2 / 4, dwell 2 / 3, still-hand limit 0.05 / 0.1 / 0.2 / none, confidence 0.5 / 0.6 / 0.7) were chosen
on val with v2's `hand` (it never saw the val signers): window 24, **stride 2, dwell 2, confidence ≥ 0.7**,
still-hand limit 0.05 (val mean 0.581 against 0.622 with the old rules). Test, once:

| Letter mode | At the sentences' speed | Slowed 2.5× |
|---|---|---|
| old model, its rules | 0.997 | 0.932 |
| final_hand, the old rules (stride 4, dwell 3, ≥ 0.5) | 0.730 | 0.389 |
| **final_hand, the rules chosen on val** | **0.643** | **0.368** |

- With deliberate spelling about two thirds of the letters are typed right, against almost none before.
- **Fast, natural spelling is beyond a letter-by-letter typing mode** whatever the rules (the best val
  setting still types 0.2 letters per letter): one window holds two or three letters, so short letters are
  covered by their neighbours. Inside fluent signing, the sentence model finds 90 % of letters (E7).
- Relaxing the still-hand limit does not help at either speed: transitions then become letters.

**Open:** a live-webcam test of the letter mode; the control gestures (backspace, enter, space) exist
only as buttons and keys with the new model.


---

## E9 — Text → sign: Azerbaijani word forms to the avatar's signs (no training)

The text-to-sign mode (`src/tercume/t2s.js`) shows a word as a whole sign only if its spelling is in the
avatar lexicon (`public/tercume/words/`, 292 AzSLD signs); every other word is fingerspelled. Azerbaijani is
agglutinative, so most words in a text are inflected: on AzSLD's own 492 sentence texts **47 % of the words
became signs and the rest were spelled letter by letter**, although 74 % of the signs annotated in those
sentences are in the lexicon ("evdə", "anam", "yaşayıram", "sənədləri", "bilərəm" were all spelled).

AzSLD signs a word in its dictionary form (evdə → EV, yaşayıram → YAŞAMAQ), but a negated verb is a sign
of its own (bilmirəm → BİLMİR, düşmürəm → DÜŞMƏMƏK, sevmirəm → SEVMİRƏM). `src/tercume/morph.js` therefore
maps a word to a key = root + negation:

- the word is cut from the right in suffix order (stem, negation, tense / infinitive / participle, plural,
  possessive, case, copula, person, question), and candidate roots are tried longest first against roots
  taken from the lexicon itself: infinitives give verb roots (yaşamaq → yaşa), an inflected verb entry gives
  its verb's key (bilmir → bil-, istəyir → istə), an entry that is a longer form of another (mənim ← mən)
  joins it;
- sound changes are undone: uşağ → uşaq, köynəy → köynək, istəy → istə, verbs ged → get, nouns oğl → oğul;
  pronouns by table (onun, ona → o; burda → bura);
- guards found on dev sentences: function words are not cut (ilə ≠ il, kimi ≠ kim), a bare short verb stem
  is not its verb ("it" is the dog, not itmək), d → t only in verbs (yadımda ≠ yatmaq), and a capitalised
  acronym in lower-case text is spelled (DOST, the agency, is not "dost", friend);
- among entries with one key: the one the word continues (istəyirsən → istəyir), else the dictionary form
  (olan → olmaq, not olar "may"), else the most similar. Phrases ("ad günü") match word by word.

`WordLexicon.segment` uses it after the exact spelling (`via: 'text' | 'lemma'`). Cases in
`tools/verify_words.mjs`.

**Evaluation** (`tools/eval_text_gloss.mjs`): for each of the 492 sentences, the reference is the sign
sequence most signers used (median agreement of the signers with it 0.90); fingerspelled letters are left
out of it. The predicted signs are compared with it in order (LCS): precision, recall, F1. "Same gloss" uses
the gloss merges of the word model (`gloss_spelling`, `same_sign`, confirmed `lemma_groups`); "same word"
also counts forms of one word as one sign (EDİB = ETMƏK, BALACADIR = BALACA; negated forms stay apart).
The rules were written looking only at the 443 dev sentences; the 49 held-out sentences (the word model's
unseen-sentence test) were scored at the end. "Before" is the code before this change.

| Lexicon | Text → signs | Sentences | Words as signs | Letters spelled | Same gloss P / R / F1 | Same word P / R / F1 |
|---|---|---|---|---|---|---|
| avatar (292 signs) | before | dev | 46.7 % | 8,126 | 0.910 / 0.438 / 0.591 | 0.933 / 0.449 / 0.606 |
| | **word forms** | dev | **74.6 %** | 3,844 | 0.883 / 0.675 / 0.765 | 0.919 / 0.702 / 0.796 |
| | before | test | 47.6 % | 849 | 0.916 / 0.464 / 0.616 | 0.925 / 0.469 / 0.623 |
| | **word forms** | test | **77.1 %** | 363 | 0.855 / 0.697 / 0.768 | 0.895 / **0.730** / **0.804** |
| every AzSLD gloss (740) | before | test | 55.5 % | 715 | 0.886 / 0.517 / 0.653 | 0.902 / 0.526 / 0.665 |
| | **word forms** | test | **89.0 %** | 150 | 0.862 / 0.796 / 0.828 | 0.897 / **0.829** / **0.862** |

- Recall goes from 0.47 to 0.73 on unseen sentences with the same 292 clips; letters spelled drop 57 %.
  Precision falls 0.03: some forms now reach a wrong sign (below).
- The last two rows assume a clip for every AzSLD gloss: the lexicon is the larger lever once word forms
  work (recall 0.83).
- What is left (dev): idioms (anadan olmaq → DOĞMAQ, rəhmətə getmək → ÖLMƏK, yerinə yetirmək), the
  pronoun "o" that signers point instead of signing, light verbs signed or not (müraciət etmək vs
  istirahət etmək), homonyms that only the full lexicon separates (tələbə → TƏLƏB "demand", Adası → AD).


---

## E10 — The avatar lexicon from every AzSLD sign (local, no training)

The full AzSLD is now downloaded locally (`tools/download_azsld.py data/azsld`, the four Zenodo zips,
46.9 GB, md5 equal to Zenodo's). Words_100 is a subset of Words_200 (all 7,248 clips), and Words_200 clips are cut from the
Sentences recordings (Cam2), so the Sentences zip is the only source; only UĞUR's 19 Words_200 clips have no
sentence annotation.

**Builder** (`tools/build_avatar_words.py`, reads single videos out of the zip with `tools/azsld_zip.py`):

- every gloss with ≥ 3 annotated signs; single letters are fingerspelling, "O" counts as the pronoun when
  no other letter is within 15 frames (395 signs); the annotators' spelling is fixed against the sentence
  texts when a label is not a known word form (İ for ı, g for ğ, c for ç…: 53 labels, e.g. YARDİM →
  yardım, QİRMİZİ → qırmızı, ALİM → alım "let me receive", İT / IT → it); signs that share a spelling merge;
- up to 8 candidates from different signers, facing the camera and of typical length; MediaPipe pose +
  hands + face in VIDEO mode (tracking), annotated range ± 3 frames, 15 fps;
- choice: the candidates with a quality (`build_words.trim_and_score`) ≥ 85 % of the best, and among them
  the medoid: the smallest mean DTW distance to all candidates (hand place relative to the shoulders +
  hand shape). An idiosyncratic or left-handed performance is not picked.

### Which pose model (40 signs, 37 signers, `tools/compare_pose_models.py`)

Wrist error: pixel distance between the pose wrist and the hand landmarker's wrist; jitter: mean |second
difference| of the world wrist and elbow.

| Pose model | Mode | Wrist error px, median / p90 | Jitter mm/frame² | ms/frame (8 in parallel) |
|---|---|---|---|---|
| lite | IMAGE (the old lexicon) | 18.2 / 40.0 | 36.3 | 41 |
| lite | VIDEO | 20.5 / 41.9 | **15.7** | 21 |
| full | IMAGE | 15.8 / 33.2 | 37.7 | 51 |
| full | VIDEO | 17.3 / 35.9 | 15.9 | 31 |
| heavy | IMAGE | **13.6 / 29.3** | 41.8 | 134 |
| heavy | VIDEO | 15.5 / 30.9 | 18.6 | 108 |

VIDEO mode halves the jitter. Heavy finds the wrist best in the image, so the first build used heavy VIDEO;
in the avatar it was worse (table below). The cause is depth: heavy puts the wrist **0.68 shoulder widths in
front of the shoulders (median; 10th percentile 0.21)**, lite 0.88 (0.64), and lower. The avatar's arm solver
(`vrm.js`) was tuned on lite, and with heavy the hands go into the body. The lexicon uses **lite in VIDEO
mode**. The side camera later showed that heavy's depth is the right one (E11).

### The avatar on the new lexicon (`node tools/measure_hands.mjs`)

| | Old (292, lite IMAGE, best of 6 clips) | Lite VIDEO, same 292 | Heavy VIDEO, same 292 | **New, all 786** |
|---|---|---|---|---|
| Palm from its target, median / p90 | 1.1 / 3.9 cm | 1.2 / **3.6** cm | 1.4 / 4.9 cm | 1.2 / 3.7 cm |
| IK reach error p90 | 2.0 cm | **1.8** cm | 3.9 cm | 1.9 cm |
| Words with the palm in the body (> 5 mm) | 110 (38 %) | **106 (36 %)** | 189 | 310 (39 %) |
| Words with the arm in the body | 107 | 119 | 240 | 385 (49 %) |
| Words with a hand out of the frame | 4 | 5 | 2 | 14 (1.8 %) |
| Words with the wrist > 5° past its limit | 119 | 144 | 199 | 413 |
| Words past the wrist's hard ceiling | 9 | **4** | 14 | 21 (2.7 %) |
| Hands through each other in playback | 4 | **2** | 3 | 12 |
| Palm jump per update in playback, p90 | 10.1 cm | **8.7** cm | 8.7 cm | 8.7 cm |

On the same words the new lexicon is as good or better than the old one except the mild wrist limit
(144 against 119 words), and the motion is smoother. The 786 signs behave like the 292.

**Text → sign** (E9's evaluation, the same word-form code, 786 signs instead of 292):

| Sentences | Words as signs | Letters spelled | Same gloss P / R / F1 | Same word P / R / F1 |
|---|---|---|---|---|
| dev (443) | 92.4 % | 1,168 | 0.866 / 0.825 / 0.845 | 0.893 / 0.851 / 0.872 |
| test (49 held out) | **92.6 %** | 103 | 0.873 / 0.844 / 0.858 | 0.902 / **0.872** / **0.887** |

Against the morning (exact spelling, 292 signs): words as signs 47.6 % → 92.6 %, letters spelled 849 → 103,
same-word F1 0.623 → 0.887 on the held-out sentences.

**Face.** Each frame also stores the 51 face blendshapes (0–100) and the head rotation (`f`). MediaPipe's
values are compressed (jawOpen reaches 16 in the top 1 % of frames), so the avatar scales each one between
its lexicon median (`faceNeutral`, the signers' resting face) and 99th percentile (`faceFull`) into the VRM
expressions: mouth shapes aa / ih / ou / ee / oh, blink, brows up (surprised) and down (angry), and adds the
sign's own head movement (rotation minus the word's median). The female avatar shows it; **the default
avatar (`avatar.vrm`) has no face morph targets at all** (its expressions are bound to nothing, it cannot
even blink), so it only gets the head movement.

Size: 786 word files, 23 MB (the 292 were 7.6 MB); coordinates are stored at 1 mm / 0.001.

**Open:** the avatar on heavy's depth (E11); the default avatar's face; sentence-level non-manuals (question
brows over the whole question, negation head shake); transitions and timing between signs.


---

## E11 — Which single-camera depth is right? The side camera decides (no training)

E10 found that pose heavy puts the wrists much closer to the shoulders' plane than pose lite, and that the
avatar looks better with lite. Which depth is true cannot be told from one camera. AzSLD films every
sentence a second time from the signer's right, above (Cam1), so `tools/stereo_depth.py` takes 12 random
sessions × 10 recordings (120 recordings, both cameras, pose lite and heavy + hands, VIDEO mode; frames
paired with `azsl/views.py`'s lag).

**Triangulation was tried first and does not work on this data:**

| Test | Result |
|---|---|
| Points fitting one fundamental matrix per recording (RANSAC, 4 px) | slow body points (nose, ears, shoulders) 56-84 %; hand landmarks 19-57 %; hands still in both views (< 3-6 px per frame) 57-100 % |
| Sub-frame shift of the hand tracks (−1.5 … +1.5 frames) under the slow points' geometry | hand points stay 17-79 px from their epipolar lines at every shift |
| Focal length per session (rigid hand bones as the check) | estimates 600-3,000 px, camera angle 16-110°; bone variation 0.07-0.10 with no minimum over 700-1,400 px |
| Audio sync / camera model | audio tracks are silent; the files name no camera |

The cameras are not frame-locked (a moving hand is 10-30 px off between the views), their focal lengths
are unknown and barely observable when both cameras aim at one spot, and the slow body points lie too close
together to fix the geometry where the hands are.

**Cross-view check instead (no calibration):** the same moment must give the same wrist position in a frame
built from the body itself (shoulder axis, "up" from the shoulders toward the ears, forward = axis × up). From
the side, the hands' distance in front of the body lies in the image plane, so a model that perceives depth
gives the same answer from both views.

| Forward depth, shoulder widths | Front view (Cam2) | Side view (Cam1) | \|front − side\| | Correlation front ↔ side |
|---|---|---|---|---|
| pose lite (45,484 wrist moments) | 0.81 [0.60, 1.03] | 0.87 [0.66, 1.05] | 0.14 | **0.21** |
| pose heavy (35,569) | 1.00 [0.74, 1.37] | 1.10 [0.80, 1.38] | 0.12 | **0.79** |

On the 34,983 moments all four estimates have (correlation of the forward depth):

| | lite front | lite side | heavy front | heavy side |
|---|---|---|---|---|
| lite front | 1 | 0.06 | −0.26 | −0.44 |
| lite side | 0.06 | 1 | −0.10 | 0.07 |
| heavy front | −0.26 | −0.10 | 1 | **0.79** |
| heavy side | −0.44 | 0.07 | **0.79** | 1 |

Height and the sideways position agree for both models (r 0.89-0.94). Per session heavy's front and side
medians are 0.00-0.19 apart.

- **Heavy perceives how far the hands are in front of the body; lite does not.** Lite's forward depth is
  close to a fixed guess plus noise: it does not follow its own side view (0.06) and runs against heavy's.
- Lite looked better in the avatar (E10) because the arm solver (`vrm.js`) was tuned on it: a near-constant
  depth keeps the hands about one shoulder width in front of the chest, so they never meet the body. With
  heavy, hands touch the chest as the signer's do, and the avatar's torso, with other proportions, takes
  them in.
- Next: the avatar maps chest contacts onto its own torso (as it does for the face), and the lexicon is
  rebuilt with heavy (E12).


---

## E12 — The avatar on heavy's depth: chest contacts (no training)

E11 showed that pose heavy sees how far the hands are in front of the body. With heavy, signers' hands come
close to their chest, and the avatar's arm solver (`vrm.js`), tuned on lite's near-constant depth, put hands
and forearms into the avatar's torso. Three changes:

- **The signers' chest line**, measured from the lexicon itself (`build_avatar_words.py` → `index.json`
  `chest`): hands stop at the body, so the 3rd percentile of the palm's distance in front of the shoulders
  at each height (palm in front of the torso) traces the chest; a line through it. Heavy: forward =
  0.616 + 0.424 × height (shoulder widths): it slants because seated signers lean toward the camera.
  (Lite: 1.036 + 0.636 × height, about a third of a shoulder width further forward.)
- **Depth from the chest** (`#signerToAvatar`): in front of the torso the palm's distance from the signer's
  chest line is placed the same (scaled) distance in front of the avatar's measured torso surface, at that
  height and sideways position; above the shoulders and beside the torso the old shoulder-relative placement
  stays. When a fingertip or palm edge is within 4-10 cm of the chest (MƏN: the finger on the chest), that
  point goes onto the avatar's chest (plus the finger's thickness) and the palm follows it.
- **Elbow targets out of the torso** (`#elbowOut`): seated signers keep their elbows near the body, and
  29 % of heavy's elbow targets fell inside the avatar's torso (lite: 3 %); they are moved out by the upper
  arm's thickness.

**Where it fails** (every third word, per hand and frame; lite against heavy with the chest mapping):

| | lite | heavy + chest |
|---|---|---|
| Hand in the body (> 5 mm) | 3.2 % | 7.6 % |
| Arm in the body | 6.4 % | 13.4 % |
| … with the hand at upper-chest height (0-0.4 shoulder widths below the shoulders) | 2.9 % | 29.5 % |
| Wrist > 5° past its limit | 7.0 % | 12.5 % |
| IK misses its target by > 3 cm | 4.2 % | 12.4 % |

**All 786 signs** (`node tools/measure_hands.mjs`):

| | lite | lite + chest | heavy | **heavy + chest + elbows** |
|---|---|---|---|---|
| Palm from its target, median / p90 | 1.2 / 3.7 cm | 1.4 / 4.5 cm | 1.5 / 4.9 cm | 1.3 / 4.6 cm |
| IK reach error p90 | 1.9 cm | 2.4 cm | 3.9 cm | 3.5 cm |
| Words with the palm in the body (> 5 mm) / deepest | 310 / 5.8 cm | 474 / 4.3 cm | 522 / 7.6 cm | 442 / 6.8 cm |
| Words with the arm in the body / deepest | 385 / 5.6 cm | 640 / 4.8 cm | 651 / 8.0 cm | 501 / 5.5 cm |
| Words with a hand out of the frame | 14 | 19 | 9 | 13 |
| Words with the wrist > 5° past its limit / past the hard ceiling | 413 / 21 | 368 / 19 | 540 / 36 | 579 / 39 |
| Hands through each other in playback | 12 | 19 | 8 | 9 |
| Palm jump per update in playback, p90 | 8.7 cm | 9.1 cm | 8.7 cm | 8.8 cm |

- The chest mapping and the elbow fix recover part of heavy's loss (palm in the body 522 → 442 words, arm
  651 → 501, deepest arm 8.0 → 5.5 cm), but heavy stays behind lite on every count of the body and the wrist.
- Lite with the chest mapping is also worse than plain lite (474 against 310 words): anything that brings
  the hands to the body, the true depth or a chest-relative placement, makes the current solver push hands
  and forearms into the avatar more often, though less deeply. Lite's clean numbers come from hands that
  never reach the body: MƏN's finger stays in front of the chest, and no measure counts a missed contact.
- Side by side (MƏN, EV, the middle frame) both look plausible; heavy puts the hands nearer the body. The
  lexicons pick different signers for some words, so frames are not one-to-one.

**The lexicon goes back to lite** (`build_avatar_words.py` default, no chest line; the word files are
byte-identical to E10's). The elbow fix stays, as it helps lite a little: palm in the body 304 against 310
words, arm 380 against 385, wrist past its limit 409 against 413, hands through each other 10 against 12
(a hand out of the frame 16 against 14). Heavy with the chest line stays one command away:
`--pose-model data/models/pose_landmarker_heavy.task --chest`. **Next:** the arm solver for real depth: the forearm at upper-chest height (most of the arm
penetrations), and the avatar's arm proportions against the signers' (IK misses tripled).


---

## E13 — Hands that touch each other: contact signs (no training)

In some signs the two hands' fingers pass between each other or one hand lies on the other. The avatar
treated every overlap of the two hands' capsules as an error: the two-hand fix (`#fixTwoHanded`) pushed the
hands apart until 2 mm separated them (with up to 8 cm of movement), the playback guard (`#guardHands`)
did the same on every update, and the step that restores the signer's palm-to-palm vector was skipped when
a hand had to move more than 4 cm (7 cm if the data shows the hands further apart).

**Data.** For each of the 556 two-handed words and each of its candidate signers: the share of frames in
which a finger of one hand is within 1.6 cm (bone axes) of a finger of the other. Most signers agree on
MƏSƏLƏ (8 of 8 signers), FİLM (7/8), BİRLİKDƏ (4/5), BİTKİLƏRİ, GÖSTƏRMƏK, TƏZYİQ, … The signer's video
frame next to the avatar's at the moment the signer's hands are closest: BİRLİKDƏ clasped vs side by side,
TƏZYİQ one palm on the other hand vs ~10 cm apart, KİTAB palms together vs a roof, GÖSTƏRMƏK fingers on
the other hand vs apart; MƏSƏLƏ, FİLM and BALIQ (crossed, overlapping hands) roughly right.

**Changes.**
- `build_avatar_words.py`: `contact` on a two-handed word when at least half of its signers have the
  fingers closer than 1.6 cm in more than a quarter of the frames (38 words); a single signer's depth error
  does not count.
- `vrm.js`, contact words only: in a frame where the signer's own hands overlap, the two-hand fix and the
  playback guard allow the same overlap (at most 2.5 cm) instead of pushing the hands apart; the palm-to-palm
  vector is restored even when a hand has to move up to 10 cm.
- `measure_hands.js`: `handInHandBeyond`, words whose hands overlap more than 5 mm deeper than the
  signer's own (overlap in a contact sign is the sign, not an error).

**Result** (all 786 words; the other 748 are unchanged by construction):

| | Before | After |
|---|---|---|
| Hands overlapping > 5 mm (words) | 1 | 23 |
| … deeper than the signer's own hands by > 5 mm | 1 | 3 |
| … during playback | 10 | 29 (19 contact words) |
| A hand in the other forearm | 3 | 12 |
| Palm in the body / arm in the body | 304 / 380 | 302 / 381 |
| Wrist past its hard ceiling | 21 | 20 |
| Palm from its target median / p90 | 1.2 / 3.7 cm | 1.2 / 3.7 cm |

The closest the two hands come, avatar before → after (signer): BİRLİKDƏ 0.3 → −0.9 cm (−1.1), TƏZYİQ 0.3 →
−2.1 (−2.1), MƏSƏLƏ 0.1 → −1.5 (−1.6), GÖSTƏRMƏK 0.9 → −0.4 (−0.6). In the frames: TƏZYİQ now has one hand
on the other and GÖSTƏRMƏK the fingers on the other hand. **Open:** BİRLİKDƏ's hands touch but are not
clasped, and KİTAB's palms face down in a roof instead of facing each other: the hand orientation, not the
distance.

---

## E14 — New default avatar with a face rig (no training)

The team's new `avatar.vrm` is the same model on the same skeleton (bone positions equal, torso slices
within 3 mm) with a face: a Rigify face rig (87 bones under the head: cheeks, lips, brows, lids, eyes),
29 morph targets bound to the VRM expressions (aa / ih / ou / ee / oh, blink, happy, angry, sad, relaxed,
surprised) and eye bones. The feet meshes are gone (never in frame) and the textures are WebP. Two versions
came the same day: the first (8.2 MB) and an optimized one with about half the vertices in the hair,
lashes, shirt and torso (6.1 MB, shipped; the old model was 17.5 MB).

**Fix.** `vrm.js #measureSolid` measured the head from vertices skinned to the head and neck bones only.
On the new model the face is skinned to the face-rig bones, so the face was missing: the "nose" became a
brow (6 cm higher, 1.7 cm back) and the head centre moved 6 cm back. Every bone under the head now counts.
Nose: old model 1.679 m / 16.0 cm forward, new 1.682 / 15.9. Head centre and radius (vertex centroid, 85th
percentile): old 5.6 cm forward / 13.9 cm, first version 2.9 / 12.4, shipped 4.1 / 13.1; the old model's
dense eyelashes pulled its centroid forward.

**Face.** The word's face now shows on the default avatar (headless render: SALAM brows down and eyes
narrowed, NƏ mouth shape), so both avatars have it (E10).

**Hands** (all 786 words, `node tools/measure_hands.mjs`; the old model re-measured on the same code gives
the E13 numbers exactly):

| | Old model | Face, first | Face, optimized (shipped) |
|---|---|---|---|
| Palm in the body > 5 mm (words) | 302 | 312 | 321 |
| Arm in the body (words) | 381 | 375 | 400 |
| Hands overlapping > 5 mm / deeper than the signer's | 23 / 3 | 26 / 4 | 27 / 4 |
| Hands through each other in playback | 29 | 29 | 35 |
| Wrist past its hard ceiling | 20 | 20 | 19 |
| Palm from its target median / p90 | 1.2 / 3.7 cm | 1.2 / 3.7 cm | 1.2 / 3.7 cm |
| Out of frame | 16 | 15 | 16 |

Shipped vs old, per word: 83 words have the palm more than 1 cm further from the body (mostly signs at the
face: GÖZ, BAŞ, HƏKİM, FİKİR, ANASI; the old head sphere was too large and pushed hands near the face down
to the chest), 25 more than 1 cm deeper (8 +3.9 cm, ƏRZAQ +3.3, HÖRMƏT +2.4, KÖNÜLLÜ +2.4). The optimized
mesh is coarser: its lower torso measures 5 mm deeper and the forearm 1 mm thicker than the old model's, so
the arm touches the body in 25 more words than on the first version. **Open:** if that shows, the solver's body margin can grow
by the same few millimetres.
