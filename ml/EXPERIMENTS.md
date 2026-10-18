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
