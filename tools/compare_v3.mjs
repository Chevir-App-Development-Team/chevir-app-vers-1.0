// İki V3 hesabatının (tools/backtranslate.mjs --out) cümlə-cümlə müqayisəsi: işarələrin F1-i, WER, fərqin
// cütləşmiş bootstrap etibar aralığı (cümlələr təkrar seçilir), hansı cümlələrdə dəyişdi.
//   node tools/compare_v3.mjs before.json after.json [--mode avatar] [--n 2000] [--show 10]
import { readFileSync } from 'node:fs';

const args = process.argv.slice(2);
const opt = (n, d = null) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : d; };
const [fa, fb] = args.filter((a, i) => !a.startsWith('--') && !(i && args[i - 1].startsWith('--')));
const mode = opt('--mode', 'avatar');
const modeB = opt('--mode-b', mode);
const N = Number(opt('--n', 2000));
const SHOW = Number(opt('--show', 0));
const A = JSON.parse(readFileSync(fa, 'utf8')), B = JSON.parse(readFileSync(fb, 'utf8'));
const byId = new Map(B.rows.map((r) => [r.id, r]));
const pairs = A.rows.filter((r) => byId.has(r.id) && r[mode] && byId.get(r.id)[modeB]).map((r) => [r, byId.get(r.id)]);

const stats = (r, m) => {
  const x = r[m];
  // --phases ilə: sətir bütün fazaların cəmini daşıyır (ann, det); köhnə hesabatda birinci fazadan sayılır
  const ann = x.ann ?? x.intended.filter((i) => i.kind === 'word' && i.c != null).length;
  const det = x.det ?? x.detected.filter((d) => !([...d.label].length === 1 && /\p{L}/u.test(d.label))).length;
  return { tp: x.tp, det, ann, e: x.e, T: x.T };
};
const S = pairs.map(([a, b]) => [stats(a, mode), stats(b, modeB)]);
const f1 = (rows, k) => {
  let tp = 0, det = 0, ann = 0, e = 0;
  for (const r of rows) { tp += r[k].tp; det += r[k].det; ann += r[k].ann; e += r[k].e; }
  const P = tp / Math.max(1, det), R = tp / Math.max(1, ann);
  return { f1: P + R ? 2 * P * R / (P + R) : 0, wer: e / Math.max(1, ann), P, R };
};
const a = f1(S, 0), b = f1(S, 1);
// cütləşmiş bootstrap (sabit toxum)
let seed = 12345;
const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const diffs = [], wdiffs = [];
for (let i = 0; i < N; i++) {
  const s = Array.from({ length: S.length }, () => S[Math.floor(rnd() * S.length)]);
  diffs.push(f1(s, 1).f1 - f1(s, 0).f1);
  wdiffs.push(f1(s, 1).wer - f1(s, 0).wer);
}
const ci = (d) => { const x = [...d].sort((p, q) => p - q); return [x[Math.floor(0.025 * N)], x[Math.floor(0.975 * N)]]; };
const f = (v) => v.toFixed(3);
const [lo, hi] = ci(diffs), [wl, wh] = ci(wdiffs);
console.log(`${pairs.length} sentences`);
console.log(`A ${fa} (${mode}): F1 ${f(a.f1)} (P ${f(a.P)} R ${f(a.R)}), WER ${(a.wer * 100).toFixed(1)} %`);
console.log(`B ${fb} (${modeB}): F1 ${f(b.f1)} (P ${f(b.P)} R ${f(b.R)}), WER ${(b.wer * 100).toFixed(1)} %`);
console.log(`B − A: F1 ${f(b.f1 - a.f1)} [${f(lo)}, ${f(hi)}], WER ${((b.wer - a.wer) * 100).toFixed(1)} [${(wl * 100).toFixed(1)}, ${(wh * 100).toFixed(1)}] points`);
const better = S.filter(([x, y]) => y.e < x.e).length, worse = S.filter(([x, y]) => y.e > x.e).length;
console.log(`sentences with fewer / more errors in B: ${better} / ${worse}`);
const secs = (k) => S.reduce((s, r) => s + r[k].T, 0) / 30 / S.length;
console.log(`seconds per sentence: A ${secs(0).toFixed(2)}, B ${secs(1).toFixed(2)}`);
if (SHOW) {
  const ch = pairs.map(([x, y], i) => ({ x, y, d: S[i][1].e - S[i][0].e })).filter((r) => r.d).sort((p, q) => p.d - q.d);
  for (const r of [...ch.slice(0, SHOW), ...ch.slice(-SHOW)]) {
    const want = r.x[mode].intended.filter((i) => i.kind === 'word').map((i) => i.c ?? `(${i.text})`).join(' ');
    console.log(`\n${r.d > 0 ? '+' : ''}${r.d} ${r.x.id} ${r.x.text}\n  want: ${want}\n  A:    ${r.x[mode].detected.map((d) => d.label).join(' ')}`
      + `\n  B:    ${r.y[modeB].detected.map((d) => d.label).join(' ')}`);
  }
}
