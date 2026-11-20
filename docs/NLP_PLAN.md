# Chevir NLP plan: text ↔ sign (2026-10-29, updated 2026-11-20)

The whole system's plan (data, recognition, avatar, product) is [ROADMAP.md](ROADMAP.md); this file is its language part
(items N1–N10 there map to S1–S7 and T1–T4 here; the "Roadmap" columns below give the pairs).

The language layer between Azerbaijani text and AzSL signs, in both directions. Numbers are from
`ml/EXPERIMENTS.md` (E16–E37; AzSLD, fixed splits; JestDili.az as a new person). Every step below has a measurable exit
criterion. The interview with a native AzSL user was held (E21; questions and material in
[SIGNER_INTERVIEW.md](SIGNER_INTERVIEW.md)) and its decisions are in both directions (E22, E23). What is still open waits
on two things: the native review package (E36) and set C's recordings (§3).

## 1. Where we are

| Direction | What it does today | Measured |
|---|---|---|
| Text → sign (t2s) | words → morph.js roots → 3,173-sign lexicon (AzSLD 786 + JestDili.az 2,387); derived words → their root (əlillik → ƏLİL, verilir → VERMƏK); phrases and idioms (`ml/text_alt.csv`); native rules (E22): a dropped subject signed, tense as verb + OLMAQ / OLACAQ, "-malı" as verb + LAZIM, "ilə" with a person as BİRLİKDƏ (AzSLD style kept as `{ native: false }`); negated verb without a sign → verb + YOX; numbers in spoken parts when every part has a sign, else digit by digit (E25); grammar on the avatar's face: question brows, negation head shake (E25); rest spelled | same-gloss F1 test **0.890** AzSLD style / **0.853** native; words as signs **96.4 %** test, 97.8 % dev, **92.4 %** on set C's draft text; the avatar read back by our recognizer (V3): sign F1 **0.624** (E37) |
| Sign → text (s2t) | recognizer labels, decoded with a sign language model, 3 alternatives with their shares (E32) → AzSLD's own sentence when the signs are within distance 0.2 of it (E28), else gloss2text: roots, word forms from AzSLD counts mixed with a learned form model (E33), rules on top (agreement, native markers E23, have-sentences and verb government E26, E27), spelled names as one word (E30) | WER (chrF), unseen sentences: **37.9 %** (69.3) dev, **39.3 %** (70.0) test from annotated signs; **42.5 %** (65.5) from the recognizer; AzSLD's own sentences 4.9 %; JestDili.az's held-out sentences (a new person) **57.5 %** |

October's numbers for comparison (E16–E18): t2s F1 0.893, words as signs 95.2 %; s2t test 49.3 %, recognized new
sentence 51.8 %.

Where the s2t errors are now (E32, E33): with perfect word forms (the aligned word's own features, rules kept) WER would
be dev 22.0 % / test 23.6 %; the form model picks the right features for 73.2 % of dev signs (the counts alone 64.6 %).
From the recognizer, perfect recognition of the same recordings gives 41.3 % against 46.3 % (rules only): the rest is
recognition, not language. On JestDili.az's person the recognizer is the limit (sign F1 0.703).

Waiting:
- **Native review package** (E36, `tools/build_native_review.py`, answers applied by `tools/apply_native_review.py`):
  D5 word-form pairs (`ml/review/word_forms_review_w500.csv`), N8 synonyms (`ml/text_alt.csv`), N9 open native rules,
  the Slovo map and the 101 Slovo signs, V2's first 50 items.
- **Set C** (V1): 150 sentences drafted (E25), not recorded. It is the main test for both directions, for spelled names
  (R6) and numbers (N7).

## 2. Architecture

```
text ─► normalize ─► tokens ─► morph.js analysis ─► lexicon mapping ─────────────► sign plan ─► avatar (+ face: NMM)
                                    │               (sign | phrase | verb+YOX |
                                    │                drop | stem spelled)
                    shared core ────┤  morph.js (analysis)  inflect.js (synthesis)  sign_merge.json  labels
                                    │
camera ─► landmarks ─► recognizer ─► sign decoder ─► gloss2text ──────────────────► text + alternatives
                       (frame probs)  (sign LM,       (form model + counts, rules,     (one-tap correction,
                                       N-best)         agreement, spelled names)        kept on the device)
                       face channel (built, off) ─┘
```

- **One morphology core.** `morph.js` splits words into root + suffixes (t2s, alignment); `inflect.js` builds
  words from root + features (s2t). Feature bundles (`{case, poss, pl, cop, q}` / `{tam, neg, pers, q}`) are the
  shared interface: s2t predicts them, inflect.js realises them, so output words are always well-formed.
- **On the device.** Models are JSON / ONNX loaded by the page: `gloss_text.json` 1,398 KB (829 KB gzip) with the form
  model embedded, `sign_lm.json` 347 KB (140 KB gzip). No text or video leaves the device by default; any server step
  (§4 S7) is opt-in.
- **Rules first, learned where data says so.** Rules for what is certain (agreement, harmony, negation; negation never
  comes from the form model); counts and a small model for choices (which case, which tense); every choice is visible
  in `words[].via`, and the model is behind a switch (`new GlossToText(model, { tagger: false })`).
- **Measured.** Each component has an eval on fixed splits and a regression check (`tools/verify_*.mjs`).

## 3. Evaluation (the base for everything else)

| Set | What | Status |
|---|---|---|
| A. AzSLD pairs | 492 (text, signs): dev 443 (leave-one-out), test 49 held-out sentences; both directions | in place (`eval_text_gloss.mjs`, `eval_gloss_text.mjs`) |
| B. Recognized signs | sentence model on 1,484 test recordings; "new sentence" removes it from the model and the list | in place (`eval_gloss_text.mjs --detections`, `eval_nbest_text.mjs` for alternatives) |
| J. A new person | JestDili.az: 500 natively signed sentences (text only, no glosses); 246 held out by the installed recipe's twin | in place (`score_jestdili.mjs`, `compare_jestdili.mjs`, E26–E33) |
| C. Fresh text | ~150 sentences from the product's domains (social services, health, bank, daily life), signed by 2–3 native users, glossed by the AzSLD conventions | drafted (`ml/setc/`, E25); words as signs 92.4 % on its text; **not recorded** |
| D. People | native users: t2s "what did the avatar say?" + 1–5; s2t meaning and language 1–5; 50 items per release | review page `/rey/` (E25) and the offline package (E36) ready; **not held** |

Metrics: t2s same-gloss F1, words as signs, letters spelled, **meaning flips = 0** (negation, numbers) as a
hard gate; s2t WER plus chrF (partial credit for a right root, fairer for agglutinative words; in
`eval_gloss_text.mjs` since E17) and the error classes above. Gates (`ml/gates.json`, `tools/ci_gates.mjs`, in CI):
t2s F1 ≥ 0.888 AzSLD style / 0.851 native, words as signs ≥ 95.5 %; s2t rules path dev ≤ 42.9 %, test ≤ 43.8 %; with
the form model (`--tagger <fold models>`) dev ≤ 38.8 %, test ≤ 40.3 %, recognized new sentence ≤ 43.5 %.
`verify_gloss_text.mjs` checks that the form model never changes a negation.

## 4. Sign → text roadmap

| Step | Roadmap | Work | Effort | Exit criterion |
|---|---|---|---|---|
| S0 ✓ | | gloss2text v1: alignment, synthesis-based features, context counts, agreement rules, letters, numbers (E16) | done | test 50.2 %, recognized-new 53.0 % |
| S1 ✓ | N1, R3 | **Label hygiene** ✓ (E17): one spelling table `ml/label_text.csv` for the avatar builder, both evals and gloss2text; morphology fixes; test 50.2 → 49.3 %. **Native markers** ✓ (N1, E23): OLMAQ / OLACAQ, LAZIM, BİRLİKDƏ, subject pronouns read as in E22. **Question from the face** (E31): yes/no vs statement AUC 0.91 on AzSLD test, but the detector (a recognized question sign or the face) reaches F1 0.769 on AzSLD, 0.52 on JestDili.az, and "-mı" / "?" did not lower WER: built (`nonmanual.js`, `generate(labels, { nonmanual })`), not turned on | done | test ≤ 48 % (met) |
| S2 ✓ | N2 | **Pseudo-gloss corpus** (E18, rebuilt in E33): t2s over HPLT 2.0 (CC0, 1 of 54 shards, 927k sentences) in both styles, AzSLD and native: 148k pairs. In the browser through S3's model and S5's sign LM, not as counts | done | dev and test ≤ 42 % (met with S3) |
| S3 ✓ | N2 | **Form model** (E33): BiGRU, 362k parameters, float16, own JS forward verified against PyTorch; per sign one of 168 feature bundles, mixed with the counts (λ 0.6), rules on top; trained on the corpus pairs + AzSLD; on by default. Dev 41.9 → 37.9 %, test 42.8 → 39.3 %, recognized new sentence 46.3 → 42.5 %, JestDili.az level. Not built: an "insert after" word (ilə, idi, var, …) | done | roadmap N2: test ≤ 40 %, recognized-new ≤ 45 % (met); this plan's test ≤ 35 %, recognized-new ≤ 40 % (not met) |
| S4 ✗ | N3 | **N-best + word LM rerank**: tried in E32 (HPLT word bigram over the N-best list): no gain, not shipped | 1 week | −3 points |
| S5 ✓ | N3 | **Joint with the recognizer** (E32): left-to-right decoding with a bigram sign LM from the pseudo-glosses (`sign_lm.json`), identical in Python and the browser. JestDili.az F1 0.680 → 0.703, WER 60.3 → 57.5 %; AzSLD recognized new sentence 46.3 → 46.2 % | done | recognized-new −3–5 points (not met: perfect recognition gives 41.3 %); sign F1 up (met) |
| S6 ✓ | N4 | **UX loop** (E32): 3 alternatives (the shown sentence first) with their shares in `/danis/` and the translator's sentence mode, tap or "Düzəlt" to correct, kept in localStorage, exported only by button; AzSLD's own sentence only within distance 0.2 (E28) | done | D: adequacy ≥ 4/5 (waits on the review) |
| S7 | N5 | Optional **server post-edit** (LLM), opt-in only, constrained to keep every content root; measured on A/C as a reference, never the default. Decided 2026-11-20: an opt-in later, once set C exists | later | — |

Also in sign → text: names spelled inside a sentence become one word, capitalized, with a case suffix from the verb
(roadmap R6, E30; `spelled.js`): AzSLD test sentences with spelled words WER 50.0 → 38.5 %.

## 5. Text → sign roadmap

| Step | Roadmap | Work | Effort | Exit criterion |
|---|---|---|---|---|
| T0 ✓ | | negation without a sign → verb + YOX; hyphen suffixes; numbers; no one-letter roots; verb-only sound change; function words the signers drop (E16) | done | flips 0, F1 0.880 |
| T1 ✓ | N7, N8 | **Coverage**: derived words to their root (E18), JestDili.az's 2,387 signs with a guard against false matches (E24), numbers in spoken parts from JestDili.az's number signs (E25), four fixes from the review package (E36 follow-up). Test 95.2 → 96.4 %, dev 97.8 %, F1 0.890. **Left**: synonym fallback (məlumat → İNFORMASİYA, maşın → AVTOMOBİL) in the review package; number signs of their own wait on recordings | done | words as signs ≥ 96 %, F1 ≥ 0.89 (met) |
| T1b ✓ | N1, N9 | **Native rules** (E22): what the native user decided, AzSLD style kept as an option. **Left** (N9, in the review package): the order verb + YOX + tense, BİRLİKDƏ with reciprocal verbs, third-person subjects, confirmed by a second native user | done | rules confirmed or changed |
| T2 ✓ | N6, A3 | **Grammar on the face** (E25): a question word → brows down and the head tilted, a yes/no question → brows up and the head forward, held 0.45 s at the end; negation shakes the head from the negated verb on; layered on each sign's own face; no inserted pronouns | done | D: comprehension ≥ 4/5 on set C (waits on the review) |
| T3 ◐ | N8 | **Sign choice in context**: synonyms and idioms in the review package (E36); polysemous words (ALMAQ, BİLMƏK; EŞİTMƏK has three signs) and homographs (alma / al-ma) not started | 1 week | set C F1 +0.02 |
| T4 ✓ | V3 | **Back-translation check** (E34, E37): avatar landmarks → sentence recognizer → sign F1 on 49 AzSLD test + 150 set C sentences: 0.539 → 0.624 (the signers' own clips through the same path: 0.699); `tools/backtranslate.mjs`, `tools/compare_v3.mjs` | done | proxy tracks D (waits on the review) |

## 6. Shared work

- **Round-trip test** morph.js ↔ inflect.js ✓ (`verify_gloss_text.mjs`, E17, E18): 99.5 % of 172,955 forms built for
  the lexicon's roots analyse back to their root (gate 99 %; the rest are homographs).
- **Label and spelling hygiene** in one place: `ml/gloss_spelling.csv` and `ml/groups.py` → `sign_merge.json`
  (which labels are one sign; recognizer, both evals, gloss2text), `ml/label_text.csv` (how a label is spelled
  as Azerbaijani text; avatar builders, both evals, gloss2text), `ml/text_alt.csv` (other texts for one sign:
  colloquial forms, idioms, compounds; avatar lexicon via `build_avatar_words.py --alt-only`).
- **Native review loop** ✓ (E36): one offline package with every open item; an answers CSV updates the maps and flags.
- **Licences** recorded for every corpus (`docs/DATA.md`); AzSLD (CC BY 4.0) attribution kept; HPLT 2.0 (CC0) in
  `tools/corpus_sentences.py`; Slovo credited in the models' manifests. The repository itself has no licence (team
  decision 2026-11-20).

## 7. Decisions for the team

0. **Interview** ([SIGNER_INTERVIEW.md](SIGNER_INTERVIEW.md)) ✓ held (E21): five sessions with a native AzSL user
   (recordings and notes stay in `data/interview/`, outside git); the key items (B, D, E17, G28) still to repeat with a
   second native user (N9, in the review package).
1. **Negation fallback** ✓ confirmed: the verb + YOX is how negation is signed (E21). T2 adds the head shake (E25).
1a. **Native rules that AzSLD does not follow** ✓ decided: follow the native user (E22), and sign → text reads them
   (E23). "ilə" with a person as BİRLİKDƏ, a dropped 1st/2nd-person subject signed, tense as verb + OLMAQ / OLACAQ,
   "-malı" as verb + LAZIM; AzSLD style stays as `{ native: false }` for the evaluation (F1 0.890 vs 0.853 native on
   test). Still to confirm: the order verb + YOX + tense.
1b. **More signs** ✓: JestDili.az (2,387 signs in the app with permission, E24) and Russian Sign Language (Slovo: 101
   new recognizer classes, E28, E29, flagged RSL until a native user confirms them). Logos skipped (2026-11-20).
2. **Corpus** for S2: HPLT 2.0 (CC0), one shard used (E18, E33); aze-books (Apache-2.0) needs an account to accept
   its terms. More shards only if a model needs them.
3. **Set C**: who signs and glosses the ~150 drafted sentences (2–3 native users): still open (roadmap §6, 1).
4. **Server post-edit** (S7) ✓ decided (2026-11-20): an opt-in later, never the default.

## 8. Risks

- AzSLD is small and domain-specific (government services): counts overfit. S2, S3 and set C address it.
- Pseudo-glosses may differ from real signing: weight AzSLD up, select on dev, check on set C.
- Recognition errors dominate on new signers and cameras (JestDili.az: 57.5 % against 42.5 % on AzSLD): S5 and the UX
  loop (S6) limit the damage; new recordings (D4) are the fix.
- Morphology edge cases (irregular verbs, loanwords): inflect.js tests grow with every error class found.
