// İki modelin JestDili.az cümlələrində (ml/eval_jestdili.py çıxışları, eyni cümlələr) kök üzrə F1-i: cütlənmiş bootstrap
// (2000 dəfə cümlələr yenidən seçilir), fərqin 95 % aralığı və neçə cümlənin yaxşılaşıb pisləşdiyi (E27).
// Sayma tools/score_jestdili.mjs ilə eynidir.
//
// --labels: modelin bildiyi sözlər (tutma bunlar üzrə), susmaya görə quraşdırılmış cümlə modeli; lüğəti böyüyən model
// köhnə ilə müqayisə olunanda yeninin manifest.json / vocab.json-u (köhnə model yeni sözləri bilmədiyi üçün itirir).
//
//   node tools/compare_jestdili.mjs D:/chevir/eval/<A>/sentences.json D:/chevir/eval/<B>/sentences.json [--labels vocab.json]
import fs from 'node:fs';
import { Lemmatizer } from '../src/tercume/morph.js';
import { azLower } from '../src/tercume/words.js';
const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : d; };
const [fa, fb] = args.filter((a, i) => !a.startsWith('--') && !args[i - 1]?.startsWith('--'));
const lab = JSON.parse(fs.readFileSync(opt('--labels', 'public/tercume/assets/cont_model/manifest.json'), 'utf8'));
const labels = (Array.isArray(lab) ? lab : lab.labels).filter((l) => !l.startsWith('<'));
const fix = Object.fromEntries(fs.readFileSync('ml/label_text.csv', 'utf8').trim().split(/\r?\n/).slice(1).map((r) => r.split(',').slice(0, 2)));
const labelText = (l) => azLower(fix[l] ?? l);
const lem = new Lemmatizer(labels.map(labelText).filter((t) => !t.includes(' ')));
const key = (w) => lem.key(w) ?? w;
const known = new Set(labels.map((l) => labelText(l).split(' ').map(key).join(' ')));
const words = (t) => azLower(t).replace(/[^\p{L}\d\s-]/gu, ' ').split(/\s+/).filter(Boolean);
function counts(file) {
  const out = new Map();
  for (const it of JSON.parse(fs.readFileSync(file, 'utf8'))) {
    const textKeys = words(it.text).map(key);
    const left = new Map(); for (const k of textKeys) left.set(k, (left.get(k) ?? 0) + 1);
    const detKeys = it.signs.map((s) => labelText(s).split(' ').map(key).join(' '));
    let m = 0; for (const k of detKeys) if ((left.get(k) ?? 0) > 0) { m++; left.set(k, left.get(k) - 1); }
    const kn = textKeys.filter((k) => known.has(k));
    const lk = new Map(); for (const k of detKeys) lk.set(k, (lk.get(k) ?? 0) + 1);
    let hk = 0; for (const k of kn) if ((lk.get(k) ?? 0) > 0) { hk++; lk.set(k, lk.get(k) - 1); }
    out.set(it.id, [m, detKeys.length, hk, kn.length]);
  }
  return out;
}
const A = counts(fa), B = counts(fb), ids = [...A.keys()].filter((i) => B.has(i));
const f1 = (C, s) => { let m = 0, d = 0, h = 0, k = 0; for (const i of s) { const c = C.get(i); m += c[0]; d += c[1]; h += c[2]; k += c[3]; } const P = m / d, R = h / k; return 2 * P * R / (P + R); };
let win = 0; const diffs = [];
let seed = 1; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
for (let b = 0; b < 2000; b++) { const s = ids.map(() => ids[Math.floor(rnd() * ids.length)]); const d = f1(B, s) - f1(A, s); diffs.push(d); if (d > 0) win++; }
diffs.sort((x, y) => x - y);
let better = 0, worse = 0; for (const i of ids) { const a = A.get(i), b = B.get(i); const sa = a[0] + a[2] - (a[1] - a[0]), sb = b[0] + b[2] - (b[1] - b[0]); if (sb > sa) better++; else if (sb < sa) worse++; }
console.log(`${ids.length} sentences: F1 ${f1(A, ids).toFixed(3)} -> ${f1(B, ids).toFixed(3)}; bootstrap P(B>A) ${(win / 2000).toFixed(3)}, 95% CI of diff [${diffs[50].toFixed(3)}, ${diffs[1949].toFixed(3)}]; sentences better ${better} / worse ${worse}`);
