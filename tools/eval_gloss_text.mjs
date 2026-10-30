// İşarələr → mətn (src/tercume/gloss2text.js), AzSLD-nin 492 cümləsi üzərində. Ölçü: WER — mətnin
// sözləri (kiçik hərf, durğu işarəsiz) ilə səhv sözlərin nisbəti; chrF — simvol n-qramları üzrə oxşarlıq.
//
//   node tools/eval_gloss_text.mjs [--detections ml/results/cont_v15/final_merged/detections_test.csv] [--show N]
//
// 1. Annotasiyanın işarələri (tanıma səhvsiz olsa): dev — 443 cümlə, hər biri özü çıxarılmış modellə
//    (leave-one-out), test — 49 heldout cümlə, dev-də öyrənmiş modellə.
// 2. Cümlə modelinin tanıdığı işarələr (--detections, test yazılarının detections_test.csv-si):
//    "yeni cümlə" — cümlə həm modeldən, həm AzSLD siyahısından çıxarılır (görünməmiş cümlə);
//    "məlum cümlə" — brauzerdəki kimi: siyahıda ən yaxın cümlə (nearestSentence), yoxsa işarələrdən mətn.
import fs from 'node:fs';
import { prepare, fit, GlossToText, textWords } from '../src/tercume/gloss2text.js';
import { levenshtein, nearestSentence } from '../src/tercume/continuous.js';
import { azLower } from '../src/tercume/words.js';

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const DET = opt('--detections', 'ml/results/cont_v15/final_merged/detections_test.csv');
const SHOW = Number(opt('--show', 0));

const S = JSON.parse(fs.readFileSync('public/tercume/assets/cont_model/sentences.json', 'utf8'));
const merge = JSON.parse(fs.readFileSync('public/tercume/assets/sign_merge.json', 'utf8'));
// etiketin düzgün yazılışı (annotasiyadakı yazı xətaları)
const text = Object.fromEntries(fs.readFileSync('ml/label_text.csv', 'utf8').trim().split(/\r?\n/).slice(1)
  .map((r) => r.split(',').slice(0, 2)));
const rows = fs.readFileSync('ml/splits/sentences_split.csv', 'utf8').trim().split(/\r?\n/).map((r) => r.split(','));
const fi = rows[0].indexOf('folder'), hi = rows[0].indexOf('heldout_sentence');
const held = new Set(rows.slice(1).filter((r) => r[hi] === 'True').map((r) => r[fi]));

const labels = JSON.parse(fs.readFileSync('public/tercume/assets/cont_model/manifest.json', 'utf8')).labels
  .filter((l) => !l.startsWith('<'));
const prep = prepare(S, merge, labels, text);
const idx = S.map((s, i) => i);
const devIdx = idx.filter((i) => !held.has(S[i].folder)), testIdx = idx.filter((i) => held.has(S[i].folder));
const isDev = (i) => !held.has(S[i].folder);
let weights;
const fitWithout = (skip) => new GlossToText(fit(prep, (i) => isDev(i) && i !== skip), { weights });
const devModel = () => new GlossToText(fit(prep, isDev), { weights });
let all = null;
const g2tAll = () => (all ??= new GlossToText(fit(prep), { weights }));   // bütün 492 cümlə (brauzerdəki model)
const raw = (labels) => labels.map((l) => azLower(l.split('/')[0])).join(' ');

function wer(items) {
  let e = 0, n = 0;
  for (const { hyp, ref } of items) { const r = textWords(ref); e += levenshtein(textWords(hyp), r); n += r.length; }
  return `${(100 * e / n).toFixed(1).padStart(5)} %`;
}

// chrF (sacrebleu kimi): simvol 1–6-qramlarının F-ölçüsü (β = 2, boşluqsuz), sıralar üzrə orta; böyük
// yaxşıdır. Kökü düz, şəkilçisi səhv söz (mənim → mən) qismən sayılır
function chrf(items, N = 6, beta = 2) {
  const st = Array.from({ length: N }, () => ({ h: 0, r: 0, m: 0 }));
  const grams = (s, n) => { const c = new Map(); for (let i = 0; i + n <= s.length; i++) c.set(s.slice(i, i + n), (c.get(s.slice(i, i + n)) ?? 0) + 1); return c; };
  for (const { hyp, ref } of items) {
    const a = textWords(hyp).join(''), b = textWords(ref).join('');
    for (let n = 1; n <= N; n++) {
      const ga = grams(a, n), gb = grams(b, n), x = st[n - 1];
      for (const [g, c] of ga) { x.h += c; x.m += Math.min(c, gb.get(g) ?? 0); }
      for (const c of gb.values()) x.r += c;
    }
  }
  const used = st.filter((x) => x.h && x.r);
  const f = used.reduce((sum, { h, r, m }) => { const p = m / h, q = m / r; return sum + (p + q ? (1 + beta ** 2) * p * q / (beta ** 2 * p + q) : 0); }, 0);
  return (100 * f / used.length).toFixed(1).padStart(4);
}
const score = (items) => `${wer(items)} / ${chrf(items)}`;

// --tune: kontekstlərin çəkiləri dev cümlələrində (leave-one-out) seçilir, test cümlələrinə baxılmır
if (args.includes('--tune')) {
  const { WEIGHTS } = await import('../src/tercume/gloss2text.js');
  const devWer = (w) => { weights = w; let e = 0, n = 0;
    for (const i of devIdx) { const r = textWords(S[i].text); e += levenshtein(textWords(fitWithout(i).generate(S[i].glosses).text), r); n += r.length; }
    return e / n; };
  let best = { ...WEIGHTS }, bestWer = devWer(best);
  console.log(`tune: start ${(100 * bestWer).toFixed(2)} %`, JSON.stringify(best));
  for (let round = 0; round < 2; round++) for (const k of Object.keys(best)) for (const v of [0, 0.3, 1, 2, 4, 8]) {
    const w = { ...best, [k]: v }, e = devWer(w);
    if (e < bestWer - 1e-9) { best = w; bestWer = e; console.log(`  ${k}=${v}: ${(100 * e).toFixed(2)} %`); }
  }
  console.log('tuned weights', JSON.stringify(best));
  weights = best;
}

console.log(`1. annotated glosses → text (WER, lower is better / chrF, higher is better)`);
const gold = { dev: [], test: [] };
for (const i of devIdx) gold.dev.push({ s: S[i], hyp: fitWithout(i).generate(S[i].glosses).text, raw: raw(S[i].glosses), ref: S[i].text });
const dm = devModel();
for (const i of testIdx) gold.test.push({ s: S[i], hyp: dm.generate(S[i].glosses).text, raw: raw(S[i].glosses), ref: S[i].text });
for (const name of ['dev', 'test']) {
  const set = gold[name];
  console.log(`  ${name.padEnd(4)} ${String(set.length).padStart(3)} sentences | gloss string ${score(set.map((x) => ({ hyp: x.raw, ref: x.ref })))}`
    + ` | gloss2text ${score(set)}`);
}
if (SHOW) {
  console.log(`\n  test examples:`);
  for (const x of gold.test.slice(0, SHOW)) console.log(`    ${x.s.glosses.join(' ')}\n      → ${x.hyp}\n      = ${x.ref}`);
}

if (fs.existsSync(DET)) {
  const seqs = new Map();
  for (const line of fs.readFileSync(DET, 'utf8').trim().split(/\r?\n/).slice(1)) {
    const [rec, start] = line.split(',');
    const word = line.split(',').at(-2);
    if (!seqs.has(rec)) seqs.set(rec, []);
    seqs.get(rec).push([Number(start), word]);
  }
  const byFolder = new Map(S.map((s, i) => [s.folder, i]));
  const unseen = [], known = [];
  const models = new Map();
  for (const [rec, list] of seqs) {
    const i = byFolder.get(rec.split('/')[0]);
    if (i === undefined) continue;
    const labels = list.sort((a, b) => a[0] - b[0]).map((x) => x[1]);
    const ref = S[i].text;
    // yeni cümlə: model və siyahı bu cümləsiz (bütün 492-dən, test cümlələri də öyrənməyə girir)
    if (!models.has(i)) models.set(i, new GlossToText(fit(prep, (j) => j !== i), { weights }));
    const g2t = models.get(i).generate(labels).text;
    const others = S.filter((_, j) => j !== i);
    const m0 = nearestSentence(labels, others);
    unseen.push({ ref, raw: raw(labels), g2t, retr: m0 ? m0.text : raw(labels), both: m0 ? m0.text : g2t, labels, others, i });
    const m1 = nearestSentence(labels, S);
    known.push({ ref, raw: raw(labels), retr: m1 ? m1.text : raw(labels), both: m1 ? m1.text : g2tAll().generate(labels).text, labels });
  }
  console.log(`\n2. recognized glosses (${DET}, ${unseen.length} test recordings) → text`);
  const line = (name, set, cols) => console.log(`  ${name.padEnd(13)} ` + cols.map(([c, k]) => `${c} ${score(set.map((x) => ({ hyp: x[k], ref: x.ref })))}`).join(' | '));
  line('new sentence', unseen, [['gloss string', 'raw'], ['nearest AzSLD sentence', 'retr'], ['gloss2text', 'g2t'], ['nearest, else gloss2text', 'both']]);
  line('known sentence', known, [['gloss string', 'raw'], ['nearest AzSLD sentence', 'retr'], ['nearest, else gloss2text', 'both']]);
  // --sweep: AzSLD cümləsinin götürülmə həddi (nearestSentence maxDist) — yeni və məlum cümlə arasında seçim
  if (args.includes('--sweep')) {
    for (const t of [0, 0.1, 0.15, 0.2, 0.25, 0.34]) {
      const u = unseen.map((x) => { const m = nearestSentence(x.labels, x.others, t); return { ref: x.ref, hyp: m ? m.text : x.g2t }; });
      const k = known.map((x) => { const m = nearestSentence(x.labels, S, t); return { ref: x.ref, hyp: m ? m.text : g2tAll().generate(x.labels).text }; });
      console.log(`  maxDist ${String(t).padEnd(4)} new sentence ${score(u)} | known sentence ${score(k)}`);
    }
  }
}

