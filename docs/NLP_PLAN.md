# Chevir NLP plan: text ↔ sign (2026-10-29, updated 2026-10-30)

The language layer between Azerbaijani text and AzSL signs, in both directions. Numbers are from
`ml/EXPERIMENTS.md` E16 and E17 (AzSLD, fixed splits). Every step below has a measurable exit criterion.
The next steps wait on an interview with a fluent AzSL signer: questions and material in
[SIGNER_INTERVIEW.md](SIGNER_INTERVIEW.md); its last table maps each answer to the steps it changes.

## 1. Where we are

| Direction | What it does today | Measured |
|---|---|---|
| Text → sign (t2s) | words → morph.js roots → 786-sign lexicon; phrases; negated verb without a sign → verb + YOX; hyphen suffixes dropped; "ilə/idi/ki" dropped; rest spelled | same-gloss F1 **0.885** test / 0.887 dev; **93.8 %** of words as signs; negation meaning flips **0** of 906 forms (were 313) |
| Sign → text (s2t) | recognizer labels → nearest AzSLD sentence, else gloss2text (root + case/person/tense, agreement) | WER / chrF, unseen sentences: **49.3 %** / 60.2 from annotated signs (gloss string 68.6 % / 52.1), **51.8 %** / 58.3 from the recognizer (66.1 %); AzSLD's own sentences **4.8 %** |

What the remaining s2t errors are (dev, 2,150 words, E16): **28 %** right root in the wrong form (mən/mənim/mənə,
question -mı, person), **13 %** another word (signers' synonyms, label typos, number suffixes), **8 %** missing
(words with no sign: ilə, implied possessors). So the biggest lever is choosing the form, and that needs more
data than AzSLD's 492 sentences.

## 2. Architecture

```
text ─► normalize ─► tokens ─► morph.js analysis ─► lexicon mapping ─────────────► sign plan ─► avatar (+ face: NMM)
                                    │               (sign | phrase | verb+YOX |
                                    │                drop | stem spelled)
                    shared core ────┤  morph.js (analysis)  inflect.js (synthesis)  sign_merge.json  labels
                                    │
camera ─► landmarks ─► recognizer ─► gloss decoder ─► gloss2text ─────────────────► text + alternatives
                       (frame probs)  (N-best, gloss LM)  (form features, agreement,     (user correction)
                                                           inserted words, LM rerank)
```

- **One morphology core.** `morph.js` splits words into root + suffixes (t2s, alignment); `inflect.js` builds
  words from root + features (s2t). Feature bundles (`{case, poss, pl, cop, q}` / `{tam, neg, pers, q}`) are the
  shared interface: s2t predicts them, inflect.js realises them, so output words are always well-formed.
- **On the device.** Models are JSON / ONNX loaded by the page (gloss_text.json 51 KB gzip). No text or video
  leaves the device by default; any server step (§4 S7) is opt-in.
- **Rules first, learned where data says so.** Rules for what is certain (agreement, harmony, negation);
  counts and small models for choices (which case, which tense); every choice is visible in `words[].via`.
- **Measured.** Each component has an eval on fixed splits and a regression check (`tools/verify_*.mjs`).

## 3. Evaluation (the base for everything else)

| Set | What | Status |
|---|---|---|
| A. AzSLD pairs | 492 (text, signs): dev 443 (leave-one-out), test 49 held-out sentences; both directions | in place (`eval_text_gloss.mjs`, `eval_gloss_text.mjs`) |
| B. Recognized signs | sentence model on 1,484 test recordings; "new sentence" removes it from the model and the list | in place (`eval_gloss_text.mjs --detections`) |
| C. Fresh text | ~150 sentences from the product's domain (NSosyal posts, ASAN/DOST services, daily life), signed by 2–3 Deaf signers, glossed by the AzSLD conventions | **to build**: the only test of generalisation beyond AzSLD's 492 sentences |
| D. People | Deaf reviewers: t2s "what did the avatar say?" + 1–5; s2t adequacy / fluency 1–5; 50 items per release | **to build** |

Metrics: t2s same-gloss F1, words as signs, letters spelled, **meaning flips = 0** (negation, numbers) as a
hard gate; s2t WER plus chrF (partial credit for a right root, fairer for agglutinative words; in
`eval_gloss_text.mjs` since E17) and the error classes above. Gates: test WER may not rise by > 1 point, F1 may not fall by > 0.005, flips stay 0.

## 4. Sign → text roadmap

| Step | Work | Effort | Exit criterion |
|---|---|---|---|
| S0 ✓ | gloss2text v1: alignment, synthesis-based features, context counts, agreement rules, letters, numbers (E16) | done | test 50.2 %, recognized-new 53.0 % |
| S1 | **Label hygiene** ✓ (E17): one spelling table `ml/label_text.csv` (55 labels) for the avatar builder, both evals and gloss2text; morphology fixes; test 50.2 → 49.3 %. **Question rule** waits for the interview (C 10–11): AzSLD yes/no questions carry no question sign, so -mı needs the face | 2 days | test ≤ 48 % |
| S2 | **Pseudo-gloss corpus**: run the t2s pipeline (calibrated to signers, F1 0.88) over an open Azerbaijani corpus (HPLT, CC0; allmalab/aze-books, Apache-2.0; not AzTC: NC-ND), keep 3–12-word sentences whose words reach ≥ 80 % recognizer labels, train gloss2text's counts on pseudo + AzSLD (AzSLD weighted up, weight chosen on dev) | 1 week | dev and test ≤ 42 % |
| S3 | **Form tagger**: small sequence model (BiGRU or 2-layer transformer, 1–2 M params, ONNX ≤ 5 MB) predicting per sign the feature bundle (~300 classes) and an "insert after" word from a closed list (~40: ilə, idi, var, də, üçün, light verbs); inflect.js realises; trained on S2 data, chosen on dev; JS ↔ Python parity fixture like the other models | 2 weeks | test ≤ 35 %, recognized-new ≤ 40 % |
| S4 | **N-best + LM rerank**: beam over the tagger's top bundles and inserts, word 3-gram LM from the corpus (pruned, ≤ 5 MB) | 1 week | −3 points |
| S5 | **Joint with the recognizer**: beam over frame probabilities instead of greedy decoding + a gloss LM from the pseudo-gloss sequences; N-best signs into S3/S4 | 1–2 weeks | recognized-new −3–5 points; sign F1 up |
| S6 | **UX loop**: 2–3 alternatives, confidence, one-tap correction; corrections stay on the device unless the user exports them (future training data); AzSLD sentence used only when its distance is small *and* the recognizer is confident | 1 week | D: adequacy ≥ 4/5 |
| S7 | Optional **server post-edit** (LLM), opt-in only, constrained to keep every content root; measured on A/C as a reference, never the default | later | — |

## 5. Text → sign roadmap

| Step | Work | Effort | Exit criterion |
|---|---|---|---|
| T0 ✓ | negation without a sign → verb + YOX; hyphen suffixes; numbers; no one-letter roots; verb-only sound change; function words the signers drop (E16) | done | flips 0, F1 0.880 |
| T1 | **Coverage**: synonym fallback (aligned AzSLD pairs where the signers' sign has another root: məlumat → İNFORMASİYA, problem → MƏSƏLƏ; reviewed by signers); stem-only spelling for names (Messinin → M E S S İ); numbers composed from number signs (record the missing 0, 7, 9); phrase / idiom table (həyat yoldaşı, rəhmətə getmək → ÖLMƏK) | 1 week | words as signs ≥ 96 %, letters −30 %, F1 ≥ 0.89 |
| T2 | **Grammar on the face**: negation head shake with YOX, question brows (wh: lowered, yes/no: raised) from "?" and question words, on the E14 face rig; no inserted pronouns (signers add MƏN 0 of 25 times) | 2 weeks | D: comprehension ≥ 4/5 on set C |
| T3 | **Sign choice in context**: polysemous words (ALMAQ, BİLMƏK; reviewers' notes) and homographs (alma / al-ma) with the same context counts as gloss2text | 1 week | set C F1 +0.02 |
| T4 | **Back-translation check**: avatar landmarks → sentence recognizer → sign F1, an automatic proxy for "is the avatar readable" | 1 week | proxy tracks D |

## 6. Shared work

- **Round-trip test** morph.js ↔ inflect.js ✓ (`verify_gloss_text.mjs`, E17): 99.4 % of 171,950 forms built for
  the lexicon's roots analyse back to their root (gate 99 %; the rest are homographs).
- **Label and spelling hygiene** in one place: `ml/gloss_spelling.csv` and `ml/groups.py` → `sign_merge.json`
  (which labels are one sign; recognizer, both evals, gloss2text), `ml/label_text.csv` (how a label is spelled
  as Azerbaijani text; avatar builders, both evals, gloss2text).
- **Signer review loop** (as for the word forms): synonyms, idioms, the negation strategy, set C.
- **Licences** recorded for every corpus; AzSLD (CC BY 4.0) attribution kept.

## 7. Decisions for the team

0. **Signer interview** ([SIGNER_INTERVIEW.md](SIGNER_INTERVIEW.md)) before S2, T1–T3: who, when, a second signer
   for the key items (B, D, E17, G28).
1. **Negation fallback**: implemented as verb + YOX (reversible in `WordLexicon.segment`); AzSLD has no example
   either way, so confirm with signers (T2 adds the head shake).
2. **Corpus** for S2: HPLT (CC0) + aze-books (Apache-2.0)?
3. **Set C**: who signs and glosses ~150 fresh sentences (2–3 Deaf signers, the review-sheet process).
4. **Server post-edit** (S7): allowed as an opt-in, or never?

## 8. Risks

- AzSLD is small and domain-specific (government services): counts overfit. S2 and set C address it.
- Pseudo-glosses may differ from real signing: weight AzSLD up, select on dev, check on set C.
- Recognition errors dominate on new signers and cameras: S5 and the UX loop (S6) limit the damage.
- Morphology edge cases (irregular verbs, loanwords): inflect.js tests grow with every error class found.
