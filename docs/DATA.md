# Data sources, licences and use (roadmap D1)

Every dataset that enters a model, the avatar lexicon or a test set is listed here with its terms. Raw data stays in
`data/` (outside git). The repository is public: no personal data and no recordings of people enter it.

| Source | What | Licence / terms | Status | Used for |
|---|---|---|---|---|
| [AzSLD](https://zenodo.org/records/14222948) (Zenodo 14222948) | Azerbaijani SL: fingerspelling, Words 100 / 200, 492 sentences (~8,600 front + side recordings, 49k sign annotations) | CC BY 4.0 (attribution) | in use | all recognizer models, avatar lexicon, both NLP evaluations |
| AzSLD Words 500 (team Drive, not on Zenodo) | the Sentences recordings cut at 44,316 annotations, 500 grouped labels | as AzSLD (same recordings) | in `data/azsld/` | label grouping review (E20) |
| [JestDili.az](https://jestdili.az) | AzSL dictionary: 2,766 words with one studio video each, 500 sentences with video | permission granted to the team (2026-11) | fetched to `data/jestdili/`; 2,387 avatar signs in the app (E24) | avatar lexicon (D2), recognizer vocabulary (R4) |
| [HPLT 2.0 cleaned](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned), `azj_Latn` | Azerbaijani web text | CC0 | 1 of 54 shards in `data/corpus/` | pseudo-gloss corpus (E18), word LMs |
| [Slovo](https://github.com/hukenovs/slovo) | Russian SL: 1,000 signs, 20,400 videos, 194 people | CC BY-SA 4.0: attribution, derived datasets share-alike | not fetched | pretraining (R2), shared signs (D3) after native review |
| [Logos](https://github.com/ai-forever/Logos) | Russian SL: 2,863 glosses (2,004 classes), 199,668 videos, 381 people | academic research by universities and research institutes only, on request | to request | research pretraining only (R2); not in the shipped product |
| [FluentSigners-50](https://krslproject.github.io/fluentsigners-50/) | Kazakh-Russian SL: 173 sentences × 50 people | unclear (catalogue says CC BY 4.0, Zenodo copy restricted) | ask the authors | continuous pretraining (R1, R2) if allowed |
| Own recordings (D4, set C) | missing signs, fresh test sentences | written consent per person; kept in `data/`, never in git | to record | lexicon, recognizer, sets C and D |
| Interviews with native users | recordings, transcripts, notes | private, consent of the interviewee | `data/interview/` only | language facts logged without personal data (E21) |

Rules:
- A source enters a model only after its row here says what it may be used for.
- Share-alike (Slovo): a dataset we derive from it (landmarks, clips) is shared under CC BY-SA 4.0 with attribution.
- Academic-only data (Logos) never enters a model that ships in the app.
- Attribution for AzSLD (CC BY 4.0) is kept in the app's about page and in `public/tercume/words/index.json`'s `source`;
  JestDili.az is credited on the lexicon sheet, in the index's `sources` and in every `jd-` sign file's `source`.
