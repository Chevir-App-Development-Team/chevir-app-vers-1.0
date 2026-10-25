// Mətn → işarə ardıcıllığı, AzSLD cümlələri üzərində (mətn + annotasiya olunmuş işarələr).
//
// Hər cümlə qovluğu üçün istinad: siqnalçıların ən çox işlətdiyi işarə ardıcıllığı
// (ml/splits/glosses_split.csv). Proqnoz: WordLexicon.segment — lüğətdə tapılan söz işarədir,
// qalanı hərf-hərf göstərilir. İşarələr eyni sayılır: eyni glos (ml/gloss_spelling.csv,
// ml/same_sign.csv, ml/lemma_groups.csv birləşmələri ilə).
//
//   node tools/eval_text_gloss.mjs [--words public/tercume/words] [--all-glosses] [--show N]
//
// --all-glosses: lüğət əvəzinə AzSLD-nin bütün gloslarını götürür (hər glosun kliplə
// lüğətdə olduğunu fərz edir) — mətnin işarələrə bölünməsini lüğətin ölçüsündən ayrı ölçür.
// Qaydalar dev cümlələrində (heldout_sentence=False) qurulur, test — 49 heldout cümlə.
import fs from 'node:fs';
import path from 'node:path';
import { WordLexicon, azLower } from '../src/tercume/words.js';
import { Lemmatizer } from '../src/tercume/morph.js';

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const WORDS = opt('--words', 'public/tercume/words');
const ALL = args.includes('--all-glosses');
const SHOW = Number(opt('--show', 0));

function readCsv(file) {
  const text = fs.readFileSync(file, 'utf8');
  const rows = [];
  let row = [], field = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; } else if (c === '"') q = false; else field += c;
    } else if (c === '"') q = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n') { row.push(field.replace(/\r$/, '')); rows.push(row); row = []; field = ''; }
    else field += c;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  const [head, ...body] = rows;
  return body.filter((r) => r.length === head.length).map((r) => Object.fromEntries(head.map((h, k) => [h, r[k]])));
}

// Glosların birləşdirilməsi: yazılış variantı → glos → qrup
const canon = new Map();
for (const r of readCsv('ml/gloss_spelling.csv')) canon.set(r.variant, r.gloss);
const group = new Map();
for (const r of readCsv('ml/same_sign.csv')) group.set(r.gloss, r.group);
for (const r of readCsv('ml/lemma_groups.csv')) if (r.confirmed === 'yes') group.set(r.gloss, r.group);
const sign = (g) => { const c = canon.get(g) ?? g; return group.get(c) ?? c; };
// Datasetin etiketlərindəki yazı xətaları (tools/build_words.py ALIAS)
const ALIAS = new Map([...fs.readFileSync('tools/build_words.py', 'utf8').matchAll(/'([^']+)': '([^']+)'/g)]
  .map((m) => [m[1], m[2]]));
const glossText = (g) => { const c = canon.get(g) ?? g; return azLower(ALIAS.get(c) ?? ALIAS.get(g) ?? c); };
const isLetter = (g) => [...g].length === 1 && /\p{L}/u.test(g);

// İstinad: qovluq → ən çox rast gələn işarə ardıcıllığı
const recs = new Map(readCsv('ml/splits/sentences_split.csv').map((r) => [r.rec_id, r]));
const seqs = new Map();
for (const r of readCsv('ml/splits/glosses_split.csv')) {
  if (r.dup === 'True' || !r.start) continue;
  if (!seqs.has(r.rec_id)) seqs.set(r.rec_id, []);
  seqs.get(r.rec_id).push([Number(r.start), r.gloss]);
}
const folders = new Map();
for (const [rec, list] of seqs) {
  const s = recs.get(rec);
  if (!s) continue;
  const key = list.sort((a, b) => a[0] - b[0]).map((x) => x[1]).join('|');
  if (!folders.has(s.folder)) folders.set(s.folder, { text: s.text, heldout: s.heldout_sentence === 'True', counts: new Map() });
  const f = folders.get(s.folder);
  f.counts.set(key, (f.counts.get(key) ?? 0) + 1);
}
const sentences = [...folders.values()].map((f) => {
  const [best] = [...f.counts].sort((a, b) => b[1] - a[1]);
  return { text: f.text, heldout: f.heldout, tags: best[0].split('|') };
});

// Lüğət: real avatar lüğəti, ya da AzSLD-nin bütün glosları
let index;
if (ALL) {
  const glosses = new Set(sentences.flatMap((s) => s.tags).filter((g) => !isLetter(g) && !/^\d+$/.test(g)));
  index = { words: [...glosses].map((g) => ({ id: g, text: azLower(canon.get(g) ?? g), label: g })) };
} else {
  index = JSON.parse(fs.readFileSync(path.join(WORDS, 'index.json'), 'utf8'));
  for (const w of index.words) w.label = JSON.parse(fs.readFileSync(path.join(WORDS, `${w.id}.json`), 'utf8')).label;
}

// Yumşaq müqayisə: eyni sözün formaları bir işarə sayılır (EDİB = ETMƏK, BALACADIR = BALACA),
// inkarlı forma ayrıca qalır. Açarlar bütün AzSLD gloslarının kökləri ilə hesablanır.
const allGlosses = [...new Set(sentences.flatMap((s) => s.tags))].filter((g) => !isLetter(g));
const refLemma = new Lemmatizer([...allGlosses, ...index.words.map((w) => w.label)].map(glossText));
const loose = (g) => { const t = glossText(sign(g)); return refLemma.key(t) ?? t; };

function lcs(a, b) {
  const d = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(0));
  for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++) {
    d[i][j] = a[i - 1] === b[j - 1] ? d[i - 1][j - 1] + 1 : Math.max(d[i - 1][j], d[i][j - 1]);
  }
  return d[a.length][b.length];
}

function run(lex, lemma) {
  const res = { dev: [], test: [] };
  for (const s of sentences) {
    const pred = [];
    let words = 0, signed = 0, letters = 0;
    for (const seg of lex.segment(s.text)) {
      if (seg.kind === 'word' && (lemma || seg.via !== 'lemma')) {
        pred.push(sign(seg.entry.label));
        const n = seg.text.split(' ').length;
        words += n; signed += n;
      } else if (seg.kind === 'word' || (seg.kind === 'text' && /\p{L}/u.test(seg.text))) {
        const shown = s.text.slice(seg.from, seg.to);       // mətndəki yazılış, lüğət sözü yox
        words += shown.split(/\s+/).length;
        letters += [...shown].filter((c) => /\p{L}/u.test(c)).length;
      }
    }
    const ref = s.tags.filter((g) => !isLetter(g)).map(sign);
    const m = lcs(pred, ref);
    const ml = lcs(pred.map(loose), ref.map(loose));
    (s.heldout ? res.test : res.dev).push({ s, pred, ref, m, ml, words, signed, letters });
  }
  return res;
}

function report(name, rows) {
  const sum = (k) => rows.reduce((a, r) => a + (typeof k === 'function' ? k(r) : r[k]), 0);
  const prf = (k) => {
    const P = sum(k) / Math.max(1, sum((r) => r.pred.length));
    const R = sum(k) / sum((r) => r.ref.length);
    return `P ${P.toFixed(3)} R ${R.toFixed(3)} F1 ${(2 * P * R / (P + R)).toFixed(3)}`;
  };
  return `${name.padEnd(5)} ${String(rows.length).padStart(3)} sentences | words as signs ${(sum('signed') / sum('words') * 100).toFixed(1).padStart(5)}%`
    + ` | letters ${String(sum('letters')).padStart(5)} | same gloss: ${prf('m')} | same word: ${prf('ml')}`;
}

const lex = new WordLexicon(index);
console.log(`lexicon: ${ALL ? 'all AzSLD glosses' : WORDS} (${index.words.length} entries)`);
for (const [name, lemma] of [['exact word form (before)', false], ['word forms → lexicon (morph.js)', true]]) {
  const r = run(lex, lemma);
  console.log(`\n${name}`);
  console.log('  ' + report('dev', r.dev));
  console.log('  ' + report('test', r.test));
  if (lemma && SHOW) {
    const bad = r.dev.filter((x) => x.pred.length > x.ml).slice(0, SHOW);
    console.log(`\n  dev sentences with a sign not in the reference (${bad.length} shown):`);
    for (const x of bad) {
      console.log(`    ${x.s.text}\n      pred: ${x.pred.join(' ')}  (${x.pred.map(loose).join(' ')})`
        + `\n      ref:  ${x.ref.join(' ')}  (${x.ref.map(loose).join(' ')})`);
    }
  }
}
