// JestDili.az cümlələrində cümlə modelinin tanıdıqları mətnlə müqayisədə (roadmap V4; ml/eval_jestdili.py-nin çıxışı).
// Glos yoxdur, ona görə müqayisə kök üzrədir və sırasızdır: mətnin hər sözü və tanınan hər işarə morph.js açarına
// gətirilir (evdə → ev, gəlmirəm → gəl-), sonra çoxluqlar tutuşdurulur. Dəqiqlik: tanınan işarələrin neçəsi mətndə
// var; tutma: mətnin modelin bildiyi sözlərindən neçəsi tanınıb (modelin lüğətində olmayan söz sayılmır).
// Buraxılmış mübtəda (doğma işarədə MƏN əlavə olunur) mətndə olmadığı üçün səhv sayılır: dəqiqlik bir az aşağıdır.
//
//   node tools/score_jestdili.mjs D:/chevir/eval/<ad>/sentences.json [--labels public/tercume/assets/cont_model/manifest.json] [--show 10]
import fs from 'node:fs';
import { Lemmatizer } from '../src/tercume/morph.js';
import { azLower } from '../src/tercume/words.js';

const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : d; };
const file = args.find((a) => a.endsWith('.json') && !a.includes('manifest'));
const SHOW = Number(opt('--show', 0));
const labels = JSON.parse(fs.readFileSync(opt('--labels', 'public/tercume/assets/cont_model/manifest.json'), 'utf8')).labels
  .filter((l) => !l.startsWith('<'));
const fix = Object.fromEntries(fs.readFileSync('ml/label_text.csv', 'utf8').trim().split(/\r?\n/).slice(1).map((r) => r.split(',').slice(0, 2)));
const labelText = (l) => azLower(fix[l] ?? l);
const lem = new Lemmatizer(labels.map(labelText).filter((t) => !t.includes(' ')));
const key = (w) => lem.key(w) ?? w;
const known = new Set(labels.map((l) => labelText(l).split(' ').map(key).join(' ')));
const words = (t) => azLower(t).replace(/[^\p{L}\d\s-]/gu, ' ').split(/\s+/).filter(Boolean);

const items = JSON.parse(fs.readFileSync(file, 'utf8'));
let tp = 0, det = 0, ref = 0, refKnown = 0, hitKnown = 0, empty = 0;
const rows = [];
for (const it of items) {
  const textKeys = words(it.text).map(key);
  const bag = new Map();
  for (const k of textKeys) bag.set(k, (bag.get(k) ?? 0) + 1);
  const detKeys = it.signs.map((s) => labelText(s).split(' ').map(key).join(' '));
  let m = 0;
  const left = new Map(bag);
  for (const k of detKeys) { if ((left.get(k) ?? 0) > 0) { m++; left.set(k, left.get(k) - 1); } }
  const kn = textKeys.filter((k) => known.has(k));
  const leftK = new Map();
  for (const k of detKeys) leftK.set(k, (leftK.get(k) ?? 0) + 1);
  let hk = 0;
  for (const k of kn) if ((leftK.get(k) ?? 0) > 0) { hk++; leftK.set(k, leftK.get(k) - 1); }
  tp += m; det += detKeys.length; ref += textKeys.length; refKnown += kn.length; hitKnown += hk;
  if (!detKeys.length) empty++;
  rows.push({ text: it.text, signs: it.signs.join(' '), p: detKeys.length ? m / detKeys.length : 0, r: kn.length ? hk / kn.length : 1 });
}
const P = tp / Math.max(1, det), R = hitKnown / Math.max(1, refKnown);
console.log(`${items.length} cümlə, ${det} tanınan işarə (${(det / items.length).toFixed(2)} hər cümlədə), ${empty} cümlədə heç nə`);
console.log(`dəqiqlik ${P.toFixed(3)} | tutma (modelin bildiyi sözlər, ${(100 * refKnown / ref).toFixed(0)} % sözlər) ${R.toFixed(3)} | F1 ${(2 * P * R / Math.max(1e-9, P + R)).toFixed(3)}`);
if (SHOW) {
  for (const r of rows.sort((a, b) => a.r - b.r).slice(0, SHOW)) console.log(`  ${r.text}\n    → ${r.signs || '-'}`);
}
