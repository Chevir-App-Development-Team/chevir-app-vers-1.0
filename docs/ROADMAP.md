# Chevir roadmap: data, recognition, language, avatar, product (2026-11)

The plan for the next months across the whole system. The language layer has its own detailed plan,
[NLP_PLAN.md](NLP_PLAN.md); experiments are logged in [`ml/EXPERIMENTS.md`](../ml/EXPERIMENTS.md) (E0–E22). Every item
below has an exit criterion; items are numbered so issues and commits can refer to them (D3, R2, N6, …).

## 1. Where we are

| Part | Today | Measured | Main weakness |
|---|---|---|---|
| Sign → glosses, sentences (browser) | landmark model, 701 words, continuous (E7, E15) | test F1 0.935 / WER 0.091 (5 unseen people) | **new sentences: F1 0.62 all signs, 0.78 words it knows** |
| Sign → glosses, words (browser) | 756 words + unknown (E3, E5) | test Words_200 macro-F1 0.862 | vocabulary limited to AzSLD |
| Fingerspelling | 32 letters (E8) | test top-1 0.933 | not inside sentence mode |
| Glosses → text | gloss2text: roots + form rules + counts (E16–E18) | WER test 49.3 %, recognized new sentence 51.8 % (50.7 % with the corpus, E18) | choosing the word form; no face (questions) |
| Text → glosses | morphology, derivation, numbers, native rules (E9–E22, E24) | words as signs 95.6 %; F1 0.890 AzSLD style, 0.853 native | 3,173 signs (786 AzSLD, 2,387 JestDili.az, E24): web text 59 % as signs; no face; numbers digit by digit |
| Avatar | VRM with face rig, clip lexicon of 3,173 signs (E10–E14, E19, E24) | hand checks on all signs | clips joined without co-articulation; no grammar on the face |
| Data | AzSLD: 492 sentences, ~8,600 frontal recordings + side, 49k sign annotations; HPLT corpus 62k pseudo pairs | | 492 distinct sentences; signed word by word from written text |
| Evaluation | AzSLD splits (signer-independent), regression scripts | 12 `verify_*` scripts | **no test outside AzSLD, no review by native users** |

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

## 5. Targets

| Measure | Now | After phase 1 | After phase 2 |
|---|---|---|---|
| Recognized new sentences: sign F1 (all / known words) | 0.62 / 0.78 | 0.66 / 0.80 | 0.72 / 0.85 |
| Recognizer vocabulary | 701 | 1,000 | 1,500 |
| Avatar signs | 3,173 (was 786) | 3,500 | 4,000 |
| Sign → text WER, recognized new sentence (AzSLD) | 51.8 % | 45 % | 38 % |
| Set C: words as signs / native comprehension (V2) | – | 97 % / 3.5 | 98 % / 4.0 |
| Set D: sign → text adequacy | – | 3.0 | 4.0 |
| Question detection from the face | – | 0.80 | 0.85 |
| New cameras: F1 drop vs AzSLD test | – | measured | ≤ 0.10 |

## 6. Decisions needed

1. **Recordings (D4, V1)**: who signs (≥ 3 native users), when, consent form, budget.
2. **JestDili.az**: written permission for videos and word list (D2). ✓ Granted (2026-11).
3. **Logos**: request through a university for research only, or skip (D1, R2).
4. **Server post-edit** (N5): allowed as an opt-in, or never.

## 7. Risks

- **Few native experts**: one main language expert. Mitigation: a second native user (N9), recorded decisions,
  the review page (V2) so reviews take minutes, not meetings.
- **Licences**: CC BY-SA (Slovo) makes derived data share-alike; academic-only data (Logos) cannot ship in the
  product. D1 decides before data enters a model.
- **RSL differences**: 3–4 % of signs differ and young people improvise. Shared signs only after native review.
- **Synthetic data drift**: spliced sentences (R1) and pseudo-glosses (N2) are not real signing; set C decides.
- **Model size on phones**: every new model has a size and speed budget (R5, A5).
