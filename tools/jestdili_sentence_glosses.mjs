// JestDili.az cümlələrinin işarə ardıcıllığı mətndən (cümlə modelinin CTC ilə öyrənməsi üçün, E27): mətn → işarə yolu
// AzSLD üslubunda (tools/build_pseudo_gloss.mjs kimi), cümlə modelinin etiketlərinə (sign_merge.json birləşmələri ilə).
// Modelin bilmədiyi işarə və işarəsi olmayan söz <unk> olur (AzSLD-nin naməlum işarəsi kimi).
// Cümlənin yarısı (id-nin md5-i ilə) test üçün kənarda qalır: "holdout" siyahısı.
//
//   node tools/jestdili_sentence_glosses.mjs D:/chevir/landmarks/jestdili_sentences/manifest.csv <out.json> [--holdout 0.5]
import fs from 'node:fs';
import crypto from 'node:crypto';
import { WordLexicon } from '../src/tercume/words.js';

const [src, dst] = process.argv.slice(2);
const opt = (name, def) => { const i = process.argv.indexOf(name); return i >= 0 ? process.argv[i + 1] : def; };
const HOLD = Number(opt('--holdout', 0.5));

const WORDS = 'public/tercume/words';
const index = JSON.parse(fs.readFileSync(`${WORDS}/index.json`, 'utf8'));
const labelOf = new Map(index.words.map((w) => [w.id, JSON.parse(fs.readFileSync(`${WORDS}/${w.id}.json`, 'utf8')).label]));
const merge = JSON.parse(fs.readFileSync('public/tercume/assets/sign_merge.json', 'utf8'));
const known = new Set(JSON.parse(fs.readFileSync('public/tercume/assets/cont_model/manifest.json', 'utf8')).labels);
const canon = (g) => { const s = merge.spelling?.[g] ?? g; return merge.merge?.[s] ?? s; };
const lex = new WordLexicon(index, { native: false });

// manifest.csv: id, ..., text, ... (mətndə vergül ola bilər: sadə CSV oxuyucusu)
function parseCsv(t) {
  const rows = [];
  let row = [], cell = '', q = false;
  for (let i = 0; i < t.length; i++) {
    const c = t[i];
    if (q) { if (c === '"' && t[i + 1] === '"') { cell += '"'; i++; } else if (c === '"') q = false; else cell += c; }
    else if (c === '"') q = true;
    else if (c === ',') { row.push(cell); cell = ''; }
    else if (c === '\n' || c === '\r') { if (c === '\r' && t[i + 1] === '\n') i++; row.push(cell); rows.push(row); row = []; cell = ''; }
    else cell += c;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  return rows;
}
const rows = parseCsv(fs.readFileSync(src, 'utf8')).filter((r) => r.length > 1);
const head = rows[0], ID = head.indexOf('id'), TEXT = head.indexOf('text');
const out = { glosses: {}, holdout: [] };
let total = 0, signs = 0;
for (const r of rows.slice(1)) {
  const id = r[ID], text = r[TEXT];
  const seq = [];
  for (const seg of lex.segment(text)) {
    if (seg.kind === 'word') {
      const l = canon(labelOf.get(seg.entry.id));
      seq.push(known.has(l) ? l : '<unk>');            // işarəsi var, model bilmir: naməlum işarə
    } else if (seg.kind === 'text' && /\p{L}/u.test(seg.text)) {
      seq.push('<unk>');                               // işarəsi yoxdur: hərf-hərf və ya başqa işarə
    }
  }
  const h = parseInt(crypto.createHash('md5').update(id).digest('hex').slice(0, 8), 16) / 0xffffffff;
  if (h < HOLD) out.holdout.push(id);
  else if (seq.length) { out.glosses[id] = seq; total++; signs += seq.length; }
}
fs.writeFileSync(dst, JSON.stringify(out, null, 0));
console.log(`${rows.length - 1} cümlə: ${total} təlim üçün (${(signs / Math.max(1, total)).toFixed(2)} işarə), ${out.holdout.length} kənarda → ${dst}`);
