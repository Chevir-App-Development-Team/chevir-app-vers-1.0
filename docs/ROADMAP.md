# Chevir roadmap: data, recognition, language, avatar, product (2026-11, status 2026-11-20)

The plan for the next months across the whole system. The language layer has its own detailed plan,
[NLP_PLAN.md](NLP_PLAN.md); experiments are logged in [`ml/EXPERIMENTS.md`](../ml/EXPERIMENTS.md) (E0–E37). Every item
below has an exit criterion; items are numbered so issues and commits can refer to them (D3, R2, N6, …). Where each
item stands: §4.

## 1. Where we are

On 2026-11-20 (the plan's starting point, 2026-11-09, in brackets):

| Part | Today | Measured | Main weakness |
|---|---|---|---|
| Sign → glosses, sentences (browser) | landmark model `final_all_ctc500_slovo`: 804 labels (802 words), all AzSLD signers, JestDili.az's sentences and Slovo signs spliced in (E27, E29); decoding with a sign language model, 2–3 alternatives (E32) | AzSLD test, the same recipe on train signers: F1 0.899 (5 unseen people); new sentences F1 0.66 all signs, 0.82 words it knows (0.62 / 0.78). JestDili.az's 246 held-out sentences (the same recipe without them; a new person and camera): F1 0.703 | **new people**: 0.70 against 0.90; the 101 Slovo signs recognized 1 of 19 times in a new person's sentences |
| Sign → glosses, words (browser) | `final_all_words_r4`: 858 classes, 802 words + unknown, with Slovo clips and 101 Slovo signs (E27, E28) | JestDili.az: words AzSLD has top-1 0.426 / top-5 0.603, the new words 0.553 / 0.796 | a new person: top-1 under 0.5 |
| Fingerspelling | `final_hand_all`: 32 letters, all AzSLD signers (E30); names spelled inside a sentence come out as one word (R6, E30) | JestDili.az's alphabet photos: 19 of 24 static letters; AzSLD test sentences with spelled words: WER 50.0 → 38.5 % | moving letters not tested on a new person |
| Glosses → text | gloss2text: roots, form rules, counts, native markers (E16–E27), spelled names (E30), a learned word-form model under the rules (E33) | WER test 39.3 %, recognized new sentence 42.5 % (49.3 / 51.8 %); JestDili.az held out 57.5 % (66.7 %, E26) | a new person's recognition errors; questions from the face not used (E31) |
| Text → glosses | morphology, derivation, numbers in spoken parts, native rules (E9–E25, E36) | words as signs 96.4 % test, 97.8 % dev, 92.4 % on set C's draft text; same-gloss F1 0.890 AzSLD style, 0.853 native | no native user has judged it yet; numbers without a sign of their own |
| Avatar | VRM with face rig, 3,173 signs (786 AzSLD, 2,387 JestDili.az), grammar on the face (E25), fitted timing and movement between signs (E34), hand orientation (E37) | V3 back-translation sign F1 0.624 (the signers' own clips through the same path: 0.699) | two-handed and later signs; no native rating yet |
| Live signing | `/danis/`: captions while signing, a word grey then white, a conversation list (E28, E35) | ~30 fps on this laptop: MediaPipe ~31 ms per frame (the limit), model 8–14 ms per window | not tried on a real phone |
| Data | AzSLD: 492 sentences, ~8,600 frontal recordings + side, 49k sign annotations; JestDili.az: 2,766 words, 500 sentences (permission); Slovo: landmarks of 20,400 clips; HPLT: 148k pseudo-gloss pairs in two styles | | no natural recordings of our own (D4) |
| Evaluation | AzSLD splits (signer-independent); JestDili.az as a new person (E26–E33); V3 (E34); 14 `verify_*` scripts and the gates in CI; a review package for native users (E36) | | **no review by native users yet; set C not recorded** |

What changed the picture (E20, E21):
- **AzSLD is partly "Azerbaijani in signs"**: its sentences were given as written text and signed word by word.
  Native grammar differs (tense as OLMAQ / OLACAQ, subject pronouns, BİRLİKDƏ, face for questions). From now on a
  native user's judgement is the reference; AzSLD scores are regression checks.
- **AzSL shares 95–97 % of its signs with Russian Sign Language** (the shared sign language of the former Soviet
  countries). RSL data and dictionaries are the largest source of new signs and pretraining we can use.
- **The recognizer generalizes poorly to new sentences** (0.62 vs 0.935), and real users only sign new sentences.
- **There is no written grammar of AzSL** and very few experts: our rules and test sets are part of the product.

## 2. Principles

1. **Native signing is the target.** Rules come from native users and are checked with them (set D); dataset
   agreement is kept as a second measure, never the goal.
2. **Glosses stay the backbone, learned models fill in under them.** 492 sentences cannot train end-to-end
   translation; the gloss layer keeps every step visible, testable and correctable. Learned parts go where data
   allows: recognition, word forms (S3), motion between signs (A1).
3. **On the device by default.** No video or text leaves the device unless the user opts in.
4. **Measured and reversible.** Each change has an exit criterion, a regression script, an entry in the log; rule
   sets are switchable (`WordLexicon({ native })`).
5. **Licences and consent first.** Every dataset has a recorded licence; every new recording has written consent;
   the repository is public, so personal data never enters it (`data/` is outside git).

## 3. Workstreams

Effort: S ≤ 1 week, M 2–4 weeks, L > 1 month (one person).

### W1 Data

| # | Work | Effort | Exit criterion |
|---|---|---|---|
| D1 | **Licences and requests**: `docs/DATA.md` with every source and its terms: AzSLD (CC BY 4.0), HPLT 2.0 (CC0), Slovo (RSL, 1,000 signs, 20k videos, CC BY-SA 4.0: share-alike), Logos (RSL, 2,863 glosses, 200k videos: academic research only, request through a university), FluentSigners-50 (K-RSL, 173 sentences × 50 people: licence unclear, ask the authors), JestDili.az (~3,000 words, videos of ~2,100: written permission from the owner) | S | table complete; requests sent |
| D2 | **JestDili.az signs**: with permission, run the avatar builder on its videos (one person per word: quality gate instead of "most typical of 8"); words are Azerbaijani already | M | avatar lexicon 786 → ≥ 2,000 signs; words as signs on set C ≥ 97 % |
| D3 | **Russian Sign Language**: extract our landmarks from Slovo (360p, 13 GB, on Kaggle); translate its 1,000 glosses into Azerbaijani; a native user marks which signs AzSL shares. Uses: pretraining (R2), new recognizer classes (R4), avatar signs flagged "RSL" until confirmed | M | list of confirmed shared signs; landmarks on Kaggle |
| D4 | **Our own recordings** (most valuable data): a recording page in the app (video + landmarks kept locally, consent form), webcam and phone, natural signing. (a) missing signs: numbers (0–20, tens, hundreds, thousands, years, months, time words), the 49 everyday words without a sign, grammar signs (BACARMAQ, MÜTLƏQ, DOĞRU), country names; (b) set C (V1) | M | ≥ 300 new sign clips from ≥ 3 native users; set C recorded |
| D5 | **AzSLD clean-up**: review round with native users (`ml/review/word_forms_review_w500.csv`, 22 pairs), the İŞSİZ → İŞLƏMƏK merge (meaning flip), literal translations flagged (ANADAN OLMAQ), weak question labels from the text's "?" (for R3) | S | answers applied to `ml/lemma_groups.csv`, `sign_merge.json`; flags in `ml/` |
| D6 | **Text corpus**: more HPLT shards; domain texts (DOST / ASAN pages, medical and bank leaflets) for S3 and for choosing set C sentences | S | domain corpus ≥ 20k sentences |

### W2 Evaluation

| # | Work | Effort | Exit criterion |
|---|---|---|---|
| V1 | **Set C**: ~150 fresh sentences from the product's domains (DOST, medicine, bank, daily life), each signed naturally by 2–3 native users (D4b), glossed in AzSLD's conventions plus face labels (question, negation). Split: half dev, half test. The main test for both directions | M | recorded, glossed, frozen |
| V2 | **Set D, review by native users**: a review page in the app. Text → sign: "what did the avatar say?" + naturalness 1–5; sign → text: adequacy and fluency 1–5. 50 items per release, results exported as CSV | S | first review of the current release (E22 output) |
| V3 | **Back-translation proxy** (T4): avatar landmarks → our recognizer → sign F1; automatic on every avatar or text → sign change | S | number tracks V2 on 20 items |
| V4 | **New cameras**: phone and webcam recordings of people not in AzSLD (from D4) to measure robustness | S | set ready; baseline measured |
| V5 | **Continuous integration**: the 12 `verify_*` scripts and the eval gates run on every pull request (GitHub Actions) | S | green on main |

### W3 Recognition (sign → glosses)

| # | Work | Effort | Exit criterion |
|---|---|---|---|
| R1 | **New sentences**: synthetic continuous training data spliced from isolated signs along gloss sequences of the pseudo-gloss corpus (E18, 62k sequences; AzSLD style) and of set C's dev half; with N3's decoding | M | unseen-sentence F1 0.62 → 0.72 (all signs), 0.78 → 0.85 (known words) |
| R2 | **Pretraining**: our landmark encoder pretrained on Slovo (D3; Logos only if an academic licence allows it) and self-supervised (masked frames / motion tokens) on all AzSLD landmarks, then the word and sentence models fine-tuned | M | words macro-F1 +0.03, unseen F1 +0.05, V4 drop smaller |
| R3 | **Face and head channel** (non-manual grammar): face expression relative to the person's own neutral face + head pose; sentence-level question (yes/no, wh) and negation detection; weak labels from AzSLD's "?" + set C labels | M | question detection ≥ 0.85 on test people; used by N1/N2 for "-mı" and "?" |
| R4 | **Vocabulary**: 701 → 1,500 classes with shared RSL signs, JestDili.az and D4; unknown / improvised signs stay an explicit class | M | 1,500 classes at ≤ 0.02 F1 loss on the old ones |
| R5 | **Robustness and speed**: V4 cameras, light, phones; ONNX timing on a mid-range phone | S | F1 on V4 within 0.10 of AzSLD test; ≥ 15 fps on the phone |
| R6 | **Fingerspelling inside sentences**: names spelled within a sentence in sentence mode | S | names right on set C ≥ 70 % |

### W4 Language (NLP; details in NLP_PLAN.md)

Sign → text:

| # | Work | Effort | Exit criterion |
|---|---|---|---|
| N1 | **Native markers in gloss2text** (inverse of E22): OLMAQ / OLACAQ after a verb → past / future, LAZIM → -malı, BİRLİKDƏ after a person → -la, a pronoun → the subject; so both directions follow the same grammar | S | verify cases; set C WER measured |
| N2 | **Form tagger** (S3): small ONNX model predicting each sign's word form, trained on pseudo-gloss pairs in both styles + AzSLD + set C dev | M | test WER ≤ 40 %, recognized new sentence ≤ 45 % |
| N3 | **Decoding with the recognizer** (S4, S5): beam over frame probabilities + gloss LM from the pseudo-gloss corpus; N-best signs into N2, word LM rerank | M | recognized new sentence −5 points |
| N4 | **Alternatives and correction** (S6): 2–3 sentences with confidence, one-tap correction kept on the device, exportable (opt-in) | S | V2 adequacy ≥ 4 / 5 |
| N5 | **Server post-edit** (S7): opt-in only, constrained to keep every content root; reference on set C | S | decision needed (§6) |

Text → sign:

| # | Work | Effort | Exit criterion |
|---|---|---|---|
| N6 | **Grammar on the face** (T2): questions (yes/no brows up, wh brows down, head tilt), negation head shake with YOX, held at the end of a question; from the sentence type and negation found in the text; on the E14 face rig | M | V2 comprehension ≥ 4 / 5 on set C |
| N7 | **Numbers** once recorded (D4a): whole-number signs, years (number + İL), time (12:30 = 12 YARIM), phone numbers in spoken groups | S | numbers right on set C |
| N8 | **Sign choice**: synonyms and idioms reviewed with native users (`ml/text_alt.csv`); polysemy (ALMAQ, BİLMƏK; EŞİTMƏK has three signs) by context | M | set C F1 +0.02; review approved |
| N9 | **Open native rules**: order verb + YOX + tense, BİRLİKDƏ with reciprocal verbs (razıyam, görüşdüm), third-person subjects; confirmed with a second native user | S | rules confirmed or changed |
| N10 | **Space** (research): directional verbs (VERMƏK, ZƏNG ETMƏK, KÖMƏK ETMƏK: the path from subject to object), plural by repetition in space | L | prototype rated by native users |

### W5 Avatar and motion

| # | Work | Effort | Exit criterion |
|---|---|---|---|
| A1 | **Movement between signs** (co-articulation): a small model trained on AzSLD sentence landmarks that generates the transition from the end of one sign to the start of the next (motion tokens or direct poses); later a full gloss → motion generator | L | V3 up; V2 naturalness up |
| A2 | **Clip quality**: candidate clips re-picked with Words 500's grouping, JestDili.az clips (D2), `measure_hands` gates | S | no hand-check regressions |
| A3 | **Face from the language layer**: N6's markers on the face rig, blended with each sign's own face | S | with N6 |
| A4 | **Timing**: per-sign speed, holds at sentence and question ends | S | V2 naturalness |
| A5 | **Performance**: phone frame rate, lazy lexicon loading, avatar variants | S | ≥ 30 fps on a mid-range phone |

### W6 Product

| # | Work | Effort | Exit criterion |
|---|---|---|---|
| P1 | **Translation page**: both directions with alternatives (N4), face (N6), confidence, a correction button | S | released |
| P2 | **NSosyal**: posts in sign with captions (existing), alternatives in captions | S | released |
| P3 | **DOST pilot**: domain vocabulary and sentences (D4, D6), consent, staff mode; measured in place with set D | M | pilot report |
| P4 | **Privacy**: on-device default, opt-in exports, a plain-language privacy note | S | note published |

### W7 Research and documentation

- **Paper 1**: signer-independent landmark recognition on AzSLD (E1–E8, E15), with the unseen-sentence result.
- **Paper 2**: a language layer for a low-resource sign language: morphology, native rules vs dataset grammar
  (E16–E22), evaluation with native users (sets C, D).
- `ml/EXPERIMENTS.md` stays the source; systematic tables for the papers at the end.

## 4. Order

| Phase | When | Items |
|---|---|---|
| **0. Start** | weeks 1–2 | commit E18–E22; D1 requests out; D5 review round; N1; V5; V2 review page; D4 recording page; first native review of the current output (V2) |
| **1. Data and quick wins** | weeks 3–8 | D4 recordings (missing signs, set C); D2 JestDili.az lexicon; D3 Slovo; N6 face on the avatar; N7 numbers; R3 questions from the face; N2 form tagger; V3 back-translation; V4 |
| **2. Models** | weeks 9–16 | R2 pretraining → R1 new sentences → R4 vocabulary; N3 decoding; A1 transitions; N4 UX; first full evaluation on sets C and D |
| **3. Pilot and research** | week 17 on | P3 DOST pilot; N10 space; papers |

Parallel tracks: data and evaluation (team + native users), recognition (ML), language (NLP), avatar (graphics),
product. Phase 0 unblocks everything: without set C and the review page nothing later can be judged.

**Phase 0 status (2026-11-09):** D1 ✓ (`docs/DATA.md`; Logos request drafted), D2 ✓ (JestDili.az: 2,766 words and
500 sentences fetched, 2,387 avatar signs in the app, false-match guard, E24), D5 ✓ (review sheet ready,
`ml/review/word_forms_review_w500.csv`, answers pending), N1 ✓ (E23), V5 ✓ (CI and gates). Open: V2 review page, D4
recording page, the native review itself.

**Status (2026-11-20).** ✓ done, ◐ partly, ✗ not started. Everything that needs a native user or new recordings waits:
the review package (E36) makes the review take minutes, the recording page and set C's draft are ready (E25).

| Phase | Item | Status |
|---|---|---|
| 0 | V2 review page | ✓ `/rey/`, 70 items per release, CSV export (E25) |
| 0 | D4 recording page | ✓ `/qeyd/` with a consent text, 201 prompts (E25) |
| 0 | First native review (V2), D5 answers | ◐ package ready (`tools/build_native_review.py`, answers applied by `tools/apply_native_review.py`, E36); not held yet |
| 1 | D4 recordings, V1 set C | ◐ set C's 150 sentences drafted (E25); nothing recorded (decision 1 open) |
| 1 | D3 Slovo | ◐ landmarks of all 20,400 clips, map of shared signs v2 (168 glosses), 101 new classes in both models, flagged RSL until a native user confirms them (E27–E29) |
| 1 | N6 face on the avatar | ✓ question brows, negation head shake (E25) |
| 1 | N7 numbers | ◐ numbers in spoken parts from JestDili.az's number signs (E25); own number signs wait on D4 |
| 1 | R3 questions from the face | ◐ built, not turned on: F1 0.769 with question signs on AzSLD test, 0.52 on JestDili.az; target 0.80 (E31) |
| 1 | N2 form tagger | ✓ turned on: test WER 39.3 %, recognized new sentence 42.5 % (E33) |
| 1 | V3 back-translation | ✓ `tools/backtranslate.mjs`, sign F1 0.624 (E34, E37); does it track V2? waits on the review |
| 1 | V4 new cameras | ◐ JestDili.az as a new person and camera (E26–E33); phone and webcam recordings wait on D4 |
| 2 | R2 pretraining | ◐ Slovo initialization gave nothing for words (E27); Slovo clips as training data did (E27); self-supervised not tried |
| 2 | R1 new sentences | ◐ spliced sentences with Slovo signs (E29): unseen F1 0.66 / 0.82, target 0.72 / 0.85 |
| 2 | R4 vocabulary | ◐ 701 → 802 words (E28, E29); phase 1 target 1,000 |
| 2 | R6 spelled names | ✓ in sentence mode (E30); set C not recorded |
| 2 | N3 decoding with a sign LM | ◐ JestDili.az F1 0.680 → 0.703, WER 60.3 → 57.5 % (E32); AzSLD recognized new sentence only 46.3 → 46.2 %, target −5 not reached (perfect recognition gives 41.3 %; N2 took it to 42.5 %) |
| 2 | N4 alternatives and correction | ✓ 3 alternatives, one-tap correction kept on the device (E32); V2 adequacy not measured |
| 2 | A1 movement between signs | ◐ fitted path between signs, on by default (E34); no learned model |
| 2 | A4 timing | ✓ gaps and final hold fitted on AzSLD (E34) |
| 2 | R5, A5 speed | ◐ laptop measured (E35), `npm run dev:https` for phones; no real phone yet |
| 2 | First full evaluation on sets C and D | ✗ waits on recordings and the review |
| 3 | P3 DOST pilot, N10 space, papers | ✗ |

## 5. Targets

| Measure | At the start (2026-11-09) | Now (2026-11-20) | After phase 1 | After phase 2 |
|---|---|---|---|---|
| Recognized new sentences: sign F1 (all / known words) | 0.62 / 0.78 | 0.66 / 0.82 (E29, E32) | 0.66 / 0.80 | 0.72 / 0.85 |
| Recognizer vocabulary | 701 | 802 (E28, E29) | 1,000 | 1,500 |
| Avatar signs | 3,173 (was 786) | 3,173 | 3,500 | 4,000 |
| Sign → text WER, recognized new sentence (AzSLD) | 51.8 % | 42.5 % (E33) | 45 % | 38 % |
| Set C: words as signs / native comprehension (V2) | – | 92.4 % on the draft text (E27) / – | 97 % / 3.5 | 98 % / 4.0 |
| Set D: sign → text adequacy | – | – (package ready, E36) | 3.0 | 4.0 |
| Question detection from the face | – | F1 0.77 with question signs, AzSLD test (E31, off) | 0.80 | 0.85 |
| New cameras: F1 drop vs AzSLD test | – | 0.20 on JestDili.az (0.70 against 0.90; new person and new sentences at once, E32) | measured | ≤ 0.10 |

Measured outside the targets: sign → text WER on JestDili.az's held-out sentences 66.7 % (E26) → 57.5 % (E32, E33); the
avatar's V3 sign F1 0.539 (E34's defaults) → 0.624 (E37).

## 6. Decisions needed

1. **Recordings (D4, V1)**: who signs (≥ 3 native users), when, consent form, budget. Still open (2026-11-20): the
   recording page with its consent text and set C's draft are ready (E25); people, dates and budget not chosen.
2. **JestDili.az**: written permission for videos and word list (D2). ✓ Granted (2026-11).
3. **Logos**: request through a university for research only, or skip (D1, R2). ✓ Skipped (2026-11-20).
4. **Server post-edit** (N5): allowed as an opt-in, or never. ✓ An opt-in later, never the default (2026-11-20);
   measured on set C once it exists.
5. **Licence of the repository.** ✓ None for now (2026-11-20): the code is public, all rights reserved. Whether
   weights trained on Slovo are adapted material under its licence stays open (`docs/DATA.md`).
6. **Standard avatar** (`avatar.vrm`): its file allows personal non-profit use only. ✓ Kept as it is (2026-11-20);
   only the file's author can change it.

## 7. Risks

- **Few native experts**: one main language expert. Mitigation: a second native user (N9), recorded decisions,
  the review page (V2) so reviews take minutes, not meetings.
- **Licences**: CC BY-SA (Slovo) makes derived data share-alike; academic-only data (Logos) cannot ship in the
  product. D1 decides before data enters a model. The word and sentence models are trained with Slovo signs and the
  repository has no licence yet (§6, 5): a licence for them has to respect Slovo's terms.
- **RSL differences**: 3–4 % of signs differ and young people improvise. Shared signs only after native review.
- **Synthetic data drift**: spliced sentences (R1) and pseudo-glosses (N2) are not real signing; set C decides.
- **Model size on phones**: every new model has a size and speed budget (R5, A5).
