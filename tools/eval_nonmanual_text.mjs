// Üz kanalı (E31) işarələr → mətn-də: annotasiyanın işarələri (tools/eval_gloss_text.mjs-in 1-ci bölməsi: dev —
// leave-one-out, test — 49 heldout cümlə) + sual / inkar. WER durğu işarəsiz sayılır: "?" onu dəyişmir, -mı və inkar dəyişir.
//
//   node tools/eval_nonmanual_text.mjs [--face D:/chevir/eval/nonmanual/final/face_azsld.csv]
//
// 1. oracle: etiketlər mətndən (ml/nonmanual_labels.csv) — üz səhvsiz olsa ən çox nə qazanılar.
// 2. --face: üz modelinin hər yazı üçün ehtimalları (ml/nonmanual.py fit, val / test iştirakçıları — model onları
//    görməyib); sual sözü işarəsi annotasiyadan, bəli/xeyr və inkar üzdən (nonmanual.js marks()).
import fs from 'node:fs';
import { prepare, fit, GlossToText, textWords } from '../src/tercume/gloss2text.js';
import { levenshtein } from '../src/tercume/continuous.js';
import { marks } from '../src/tercume/nonmanual.js';

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const FACE = opt('--face', 'D:/chevir/eval/nonmanual/final/face_azsld.csv');

const S = JSON.parse(fs.readFileSync('public/tercume/assets/cont_model/sentences.json', 'utf8'));
const merge = JSON.parse(fs.readFileSync('public/tercume/assets/sign_merge.json', 'utf8'));
const text = Object.fromEntries(fs.readFileSync('ml/label_text.csv', 'utf8').trim().split(/\r?\n/).slice(1)
  .map((r) => r.split(',').slice(0, 2)));
const rows = fs.readFileSync('ml/splits/sentences_split.csv', 'utf8').trim().split(/\r?\n/).map((r) => r.split(','));
const fi = rows[0].indexOf('folder'), hi = rows[0].indexOf('heldout_sentence');
const held = new Set(rows.slice(1).filter((r) => r[hi] === 'True').map((r) => r[fi]));
const labels = JSON.parse(fs.readFileSync('public/tercume/assets/cont_model/manifest.json', 'utf8')).labels
  .filter((l) => !l.startsWith('<'));
const lemmas = new Set(fs.readFileSync('ml/slovo_new.csv', 'utf8').trim().split(/\r?\n/).slice(1).map((r) => r.split(',')[1]));
const model = JSON.parse(fs.readFileSync('public/tercume/assets/nonmanual.json', 'utf8'));
const lab = new Map(fs.readFileSync('ml/nonmanual_labels.csv', 'utf8').trim().split(/\r?\n/).slice(1)
  .map((r) => r.split(',')).map(([folder, q, qtype, neg]) => [folder, { q, qtype, neg: neg === '1' }]));

const prep = prepare(S, merge, labels, text, lemmas);
const isDev = (i) => !held.has(S[i].folder);
const g2t = (keep) => new GlossToText(fit(prep, keep));
const devIdx = S.map((_, i) => i).filter(isDev), testIdx = S.map((_, i) => i).filter((i) => !isDev(i));
const models = new Map(devIdx.map((i) => [i, null]));
const modelFor = (i) => {
  if (!isDev(i)) return (modelFor.test ??= g2t(isDev));
  if (!models.get(i)) models.set(i, g2t((j) => isDev(j) && j !== i));
  return models.get(i);
};

const wer = (items) => {
  let e = 0, n = 0;
  for (const { hyp, ref } of items) { const r = textWords(ref); e += levenshtein(textWords(hyp), r); n += r.length; }
  return (100 * e / n).toFixed(2).padStart(6);
};
const marked = (items) => {   // "?" düz qoyulubmu (mətndə "?" var / yox)
  let tp = 0, fp = 0, fn = 0;
  for (const { hyp, ref } of items) { const h = hyp.endsWith('?'), r = ref.includes('?'); tp += h && r; fp += h && !r; fn += !h && r; }
  return `"?" P ${(tp / Math.max(1, tp + fp)).toFixed(2)} R ${(tp / Math.max(1, tp + fn)).toFixed(2)}`;
};

// 1. oracle
const oracle = (i, { mi, neg }) => {
  const l = lab.get(S[i].folder);
  const q = l.q === '1' ? l.qtype : null;
  return { question: q, neg: neg && l.neg, mi };
};
const variants = [
  ['no face channel', null],
  ['"?" only', { mi: false, neg: false }],
  ['"?" + -mı (yes/no)', { mi: true, neg: false }],
  ['negation', { mi: false, neg: true }],
  ['"?" + -mı + negation', { mi: true, neg: true }],
];
console.log('1. annotated glosses + oracle face labels (from the text) — WER %, lower is better');
for (const [name, v] of variants) {
  const out = {};
  for (const [split, idx] of [['dev', devIdx], ['test', testIdx]]) {
    out[split] = idx.map((i) => ({ hyp: modelFor(i).generate(S[i].glosses, v ? { nonmanual: oracle(i, v) } : {}).text, ref: S[i].text }));
  }
  console.log(`  ${name.padEnd(22)} dev ${wer(out.dev)}  test ${wer(out.test)}   ${marked([...out.dev, ...out.test])}`);
}

// 2. üz modeli, hər yazı ayrıca (val və test iştirakçıları)
if (fs.existsSync(FACE)) {
  const byIdx = new Map(S.map((s, i) => [s.folder, i]));
  const recs = fs.readFileSync(FACE, 'utf8').trim().split(/\r?\n/).slice(1).map((r) => r.split(','))
    .filter(([, folder, split]) => split !== 'train' && byIdx.has(folder) && lab.get(folder).q !== 'amb')
    .map(([rec, folder, split, yn, neg]) => ({ i: byIdx.get(folder), yn: yn === '' ? null : Number(yn), neg: neg === '' ? null : Number(neg) }));
  console.log(`\n2. annotated glosses + the face model's per-recording decisions (${recs.length} val / test-signer recordings)`);
  for (const [name, v] of variants) {
    const out = { dev: [], test: [] };
    for (const r of recs) {
      const m = v && marks(model, S[r.i].glosses, { yn: r.yn, neg: r.neg });
      const nm = v && { question: m.question, neg: v.neg && m.neg, mi: v.mi };
      out[isDev(r.i) ? 'dev' : 'test'].push({ hyp: modelFor(r.i).generate(S[r.i].glosses, nm ? { nonmanual: nm } : {}).text, ref: S[r.i].text });
    }
    console.log(`  ${name.padEnd(22)} dev ${wer(out.dev)}  test ${wer(out.test)}   ${marked([...out.dev, ...out.test])}`);
  }
}
