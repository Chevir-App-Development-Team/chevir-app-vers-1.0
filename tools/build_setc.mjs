// Set C (roadmap V1): ml/setc/sentences.txt → ml/setc/setc.csv. Hər mövzunun cümlələri növbə ilə dev / test
// yarısına düşür; xüsusiyyətlər (sual, sual sözü, rəqəm, inkar, zaman, mübtəda, BİRLİKDƏ) təhlil üçündür.
// AzSLD cümlələri ilə eyni və ya çox yaxın cümlə xəbərdarlıqla göstərilir (set C təzə olmalıdır).
//
//   node tools/build_setc.mjs
import fs from 'node:fs';
import { WordLexicon, azLower } from '../src/tercume/words.js';
import { csvLine } from '../src/tercume/tool.js';

const SRC = 'ml/setc/sentences.txt', OUT = 'ml/setc/setc.csv';
const WH = new Set(['nə', 'niyə', 'hara', 'harada', 'haraya', 'haradan', 'harası', 'necə', 'neçə', 'neçədə', 'neçəyə',
  'neçəyədir', 'neçədir', 'neçənci', 'kim', 'kimə', 'kimdən', 'hansı', 'nəyə', 'nədir', 'nə vaxt']);
const NUMBER = /\d|^(bir|iki|üç|dörd|beş|altı|yeddi|səkkiz|doqquz|on|iyirmi|otuz|qırx|əlli|yüz|min)$|^(beşində|üçdür)$/;

const words = (t) => azLower(t).replace(/[^\p{L}\d -]/gu, ' ').split(/\s+/).filter(Boolean);
function lev(a, b) {
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...new Array(b.length).fill(0)]);
  for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++) {
    d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
  }
  return d[a.length][b.length];
}

const index = JSON.parse(fs.readFileSync('public/tercume/words/index.json', 'utf8'));
const lex = new WordLexicon(index);
const azsld = JSON.parse(fs.readFileSync('public/tercume/assets/cont_model/sentences.json', 'utf8')).map((s) => s.text);
const azWords = azsld.map(words);

let domain = null;
const counts = new Map();
const rows = [], warn = [];
for (const line of fs.readFileSync(SRC, 'utf8').split(/\r?\n/)) {
  const m = /^#\s*mövzu:\s*(.+)$/.exec(line);
  if (m) { domain = m[1].trim(); continue; }
  const text = line.trim();
  if (!text || text.startsWith('#')) continue;
  const k = counts.get(domain) ?? 0;
  counts.set(domain, k + 1);
  const w = words(text);
  const tags = new Set();
  if (text.endsWith('?')) tags.add(w.some((x, i) => WH.has(x) || WH.has(`${x} ${w[i + 1]}`)) ? 'sual sözü' : 'hə/yox sualı');
  if (w.some((x) => NUMBER.test(x))) tags.add('rəqəm');
  // inkar: işarəsi olmayan inkarlı fel (YOX əlavə olunur), inkarlı açar (bilmirəm → "bil-"), yox / deyil
  if (w.some((x) => /^(yox|yoxdur|deyil|deyiləm|deyilsən|deyilik)$/.test(x) || lex.lemma.key(x)?.endsWith('-'))) tags.add('inkar');
  for (const s of lex.segment(text)) {
    if (s.via === 'neg') tags.add('inkar');
    if (s.via === 'tense') tags.add(`zaman ${azLower(s.entry.text)}`);
    if (s.via === 'subject') tags.add('buraxılmış mübtəda');
    if (s.via === 'with') tags.add('birlikdə');
  }
  // AzSLD-də eyni və ya çox yaxın cümlə: söz səviyyəsində fərq 25 %-dən az
  const near = azWords.map((a, i) => [i, lev(w, a) / Math.max(w.length, a.length)]).sort((a, b) => a[1] - b[1])[0];
  if (near[1] < 0.25) warn.push(`${text}  ~  ${azsld[near[0]]} (${near[1].toFixed(2)})`);
  rows.push([`c${String(rows.length + 1).padStart(3, '0')}`, domain, k % 2 === 0 ? 'dev' : 'test', text, [...tags].join('; ')]);
}
fs.writeFileSync(OUT, [csvLine(['id', 'domain', 'split', 'text', 'tags']), ...rows.map(csvLine)].join('\n') + '\n');
const by = (f) => [...rows.reduce((m, r) => m.set(f(r), (m.get(f(r)) ?? 0) + 1), new Map())].map(([k, v]) => `${k} ${v}`).join(', ');
console.log(`${OUT}: ${rows.length} cümlə | ${by((r) => r[1])} | ${by((r) => r[2])}`);
const tagCount = new Map();
for (const r of rows) for (const t of r[4].split('; ').filter(Boolean)) tagCount.set(t, (tagCount.get(t) ?? 0) + 1);
console.log('xüsusiyyətlər:', [...tagCount].sort((a, b) => b[1] - a[1]).map(([t, n]) => `${t} ${n}`).join(', '));
if (warn.length) console.log(`AzSLD-yə çox yaxın (${warn.length}):\n  ${warn.join('\n  ')}`);
