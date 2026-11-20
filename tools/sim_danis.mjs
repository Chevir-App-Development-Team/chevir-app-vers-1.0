// /danis/-in canlı yolu Node-da: yazıların landmark-ları (ml/export_stream_clips.py) kadr-kadr brauzerin kodu ilə
// (WindowStream, recognizer.frameProbs, SentenceStream.add/peek) — boz (hələ yekun olmayan) sözlərin nə qədər tez
// göründüyü, neçəsinin yalançı olduğu (boz görünüb yekun olmayan), göstərmə qaydalarının müqayisəsi və cümlə
// fasiləsinin (sentenceGap) cümləni ortadan bölüb-bölmədiyi. Kadr sayı 30 fps-dədir.
//   node tools/sim_danis.mjs D:/chevir/eval/danis_sim/jestdili [--limit N] [--out nəticə.json] [--lm]
// --lm: işarə dil modeli ilə (sign_lm.json, E32), brauzerdəki kimi
import { readFileSync, writeFileSync } from 'node:fs';

import { LocalRecognizer } from '../src/tercume/recognizer.js';
import * as CT from '../src/tercume/continuous.js';
import { MIN_HANDS, PENDING_RULE, showPending } from '../src/tercume/live.js';

const args = process.argv.slice(2);
const opt = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const base = args[0];
const limit = Number(opt('--limit', 0));
const outPath = opt('--out', null);
const root = new URL('../', import.meta.url);
const fetchImpl = async (url) => {
  const data = readFileSync(new URL(url));
  return { ok: true, status: 200, json: async () => JSON.parse(data.toString('utf8')), arrayBuffer: async () => data };
};
const rec = await LocalRecognizer.load(new URL('public/tercume/assets/cont_model/', root).href,
  { fetchImpl, merge: new URL('public/tercume/assets/sign_merge.json', root).href });
const m = rec.manifest, C = rec.labels.length;
const sp = { ...CT.STREAM, ...m.stream }, dp = { ...CT.DECODE, ...m.decode }, face = !m.works_without_face;
const unk = m.unknown_label ? rec.labels.indexOf(m.unknown_label) : -1;
const unknown = unk >= 0 ? unk : null, short = CT.shortClasses(rec.labels);
const lmJson = args.includes('--lm') ? JSON.parse(readFileSync(new URL('public/tercume/assets/sign_lm.json', root), 'utf8')) : null;
const lm = lmJson ? new CT.SignLM(lmJson, rec.labels) : null, lp = { ...CT.LM_DECODE, ...lmJson?.decode };

const meta = JSON.parse(readFileSync(`${base}.json`, 'utf8'));
const buf = readFileSync(`${base}.bin`);
const all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const FIELDS = ['hand', 'hand_world', 'pose', 'face', 'blend'];
const GAPS = [24, 30, 36, 42, 48, 60];

const clips = limit ? meta.clips.slice(0, limit) : meta.clips;
const results = [];
const t0 = performance.now();
for (const [ci, clip] of clips.entries()) {
  const slice = (a, b) => {
    const r = { T: b - a };
    for (const k of FIELDS) r[k] = all.subarray(clip[k].offset + a * clip[k].per, clip[k].offset + b * clip[k].per);
    return r;
  };
  const mk = (sentenceGap) => new CT.SentenceStream({ C, decode: dp, unknown, short, sentenceGap, lm, lp });
  const ss = mk(1e9);
  const gs = GAPS.map((G) => ({ G, ss: mk(G), words: 0, ends: [] }));
  const finals = [], windows = [];
  const hand = all.subarray(clip.hand.offset, clip.hand.offset + clip.T * clip.hand.per);
  const handsIn = (a, b) => {          // [a, b) kadrlarının neçə hissəsində ən azı bir əl var (s2s.js #handsIn)
    let h = 0;
    for (let t = a; t < b; t++) h += Number.isFinite(hand[t * clip.hand.per]) || Number.isFinite(hand[t * clip.hand.per + 63]);
    return h / Math.max(1, b - a);
  };
  const run = async (w, flush) => {
    const { probs } = await rec.frameProbs(slice(w.a, w.b), clip.aspect, { face });
    const k = w.to - w.from, fixed = probs.subarray((w.from - w.a) * C, (w.to - w.a) * C);
    let b0 = w.from - ss.rows.length;
    const add = (res) => res.added.forEach((s) => finals.push({ i: finals.length, c: s.c, p: s.p, a: s.a + b0, b: s.b + b0, n: w.b }));
    add(ss.add(fixed, k));
    if (flush) { b0 = w.to - ss.rows.length; add(ss.flush()); }
    const pb = w.to - ss.rows.length;
    const pending = flush ? [] : ss.peek(probs.subarray((w.to - w.a) * C, (w.b - w.a) * C), w.b - w.to)
      .map((s, i) => ({ i: finals.length + i, c: s.c, p: s.p, a: s.a + pb, b: s.b + pb, len: s.b - s.a,
        hands: handsIn(s.a + pb, s.b + pb) }));
    windows.push({ n: w.b, pending });
    for (const g of gs) {
      const r = g.ss.add(fixed, k);
      g.words += r.added.length;
      if (r.sentenceEnded) g.ends.push(g.words);
      if (flush) g.words += g.ss.flush().added.length;
    }
  };
  const ws = new CT.WindowStream(sp);
  for (let t = 0; t < clip.T; t++) { const w = ws.push(); if (w) await run(w, false); }
  const last = ws.flush();
  if (last) await run(last, true);
  // sözün kadrlarında ən azı bir əl olan kadrların payı (canlıda kadrın kənarında əllə çıxan sözlərlə müqayisə)
  for (const f of finals) f.hands = handsIn(f.a, f.b);
  // cümlə yalnız sonda bitməlidir: sonrakı sözlər varsa bölünüb
  const splits = Object.fromEntries(gs.map((g) => [g.G, g.ends.filter((e) => e < g.words).length]));
  results.push({ id: clip.id, text: clip.text, T: clip.T, finals, windows, splits,
    signs: finals.map((f) => rec.labels[f.c]) });
  if ((ci + 1) % 50 === 0) console.log(`${ci + 1}/${clips.length} · ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}

// Boz göstərişin "qaçışları": eyni yerdə (i) eyni söz ardıcıl pəncərələrdə. Yekun olub-olmadığı, qaydalarla göstərilməsi.
function analyse(rule) {
  let shownFalse = 0, falseRuns = 0, trueRuns = 0, greyed = 0, noGrey = 0;
  const greyLead = [], greyFromStart = [], whiteFromEnd = [], delay = [];
  let falseFrames = 0, signFrames = 0;
  for (const r of results) {
    signFrames += r.T;
    const fin = new Map(r.finals.map((f) => [f.i, f]));
    const seen = new Map();          // `${i}:${c}` → { first, firstShown, count, done }
    let prev = new Set();
    for (const w of r.windows) {
      const now = new Set();
      for (const p of w.pending) {
        const key = `${p.i}:${p.c}`;
        now.add(key);
        let s = seen.get(key);
        if (!s || !prev.has(key)) {
          if (s) s.count = 0;          // yenidən başladı
          s ??= { first: w.n, firstShown: null, count: 0, final: fin.get(p.i)?.c === p.c ? fin.get(p.i) : null };
          seen.set(key, s);
        }
        s.count++;
        if (s.firstShown == null && p.hands >= (rule.minHands ?? 0) && showPending(p, s.count, rule)) s.firstShown = w.n;
        if (s.firstShown != null && !s.final) falseFrames += 8;
      }
      prev = now;
    }
    for (const [, s] of seen) {
      if (s.final) {
        trueRuns++;
        if (s.firstShown != null && s.firstShown <= s.final.n) {
          greyed++;
          greyFromStart.push(s.firstShown - s.final.a);
          greyLead.push(s.final.n - s.firstShown);
          delay.push(s.firstShown - s.first);
        }
      } else {
        falseRuns++;
        if (s.firstShown != null) shownFalse++;
      }
    }
    for (const f of r.finals) whiteFromEnd.push(f.n - f.b);
    noGrey += r.finals.length - [...seen.values()].filter((s) => s.final && s.firstShown != null).length;
  }
  const med = (a) => (a.length ? [...a].sort((x, y) => x - y)[a.length >> 1] : NaN);
  const q = (a, p) => (a.length ? [...a].sort((x, y) => x - y)[Math.floor(p * (a.length - 1))] : NaN);
  return { rule, shownFalse, falseRuns, trueRuns, greyed, noGrey, falsePerMin: shownFalse / (signFrames / 1800),
    greyFromStartMed: med(greyFromStart), greyFromStartP90: q(greyFromStart, 0.9), whiteFromEndMed: med(whiteFromEnd),
    whiteFromEndP90: q(whiteFromEnd, 0.9), addedDelayMed: med(delay), addedDelayMean: delay.reduce((a, b) => a + b, 0) / (delay.length || 1) };
}

const RULES = [
  { name: 'hamısı (E28)', seen: 1, p: 0, len: 0 },
  { name: '2 pəncərə', seen: 2, p: 0, len: 0 },
  { name: 'p ≥ 0.4', seen: 1, p: 0.4, len: 0 },
  { name: 'p ≥ 0.5', seen: 1, p: 0.5, len: 0 },
  { name: 'uzunluq ≥ 14', seen: 1, p: 0, len: 14 },
  { name: '2 pəncərə və ya p ≥ 0.6', seen: 2, p: 0, len: 0, fastP: 0.6 },
  { name: '2 pəncərə və ya p ≥ 0.7', seen: 2, p: 0, len: 0, fastP: 0.7 },
  { name: '2 pəncərə və ya p ≥ 0.8', seen: 2, p: 0, len: 0, fastP: 0.8 },
  { name: `brauzer (${JSON.stringify(PENDING_RULE)}, əl ≥ ${MIN_HANDS})`, ...PENDING_RULE, minHands: MIN_HANDS },
];
const rows = RULES.map(analyse);
const words = results.reduce((a, r) => a + r.finals.length, 0);
const minutes = results.reduce((a, r) => a + r.T, 0) / 1800;
console.log(`\n${results.length} yazı, ${minutes.toFixed(1)} dəq, ${words} yekun söz; kadrlar 30 fps (1 kadr = 33 ms)`);
console.log('qayda | yalançı boz göstərilən (dəqiqədə) | boz görünən doğru söz | boz yox (yalnız ağ) | boz: işarənin başlanğıcından, kadr (median/p90) | qaydanın əlavə gecikməsi (median/orta)');
for (const r of rows) {
  console.log(`${r.rule.name} | ${r.shownFalse} (${r.falsePerMin.toFixed(1)}) | ${r.greyed} | ${r.noGrey} | ${r.greyFromStartMed} / ${r.greyFromStartP90} | ${r.addedDelayMed} / ${r.addedDelayMean.toFixed(1)}`);
}
console.log(`ağ: işarənin sonundan median ${rows[0].whiteFromEndMed} kadr, p90 ${rows[0].whiteFromEndP90}`);

// cümlə daxilində sözlər arası fasilə (yekun sözlər, kadr) və sentenceGap-in cümləni bölməsi
const gaps = results.flatMap((r) => r.finals.slice(1).map((f, j) => f.a - r.finals[j].b));
gaps.sort((a, b) => a - b);
const qg = (p) => gaps[Math.floor(p * (gaps.length - 1))];
console.log(`\nsözlər arası fasilə, kadr: median ${qg(0.5)}, p90 ${qg(0.9)}, p99 ${qg(0.99)}, maks ${gaps[gaps.length - 1]}`);
for (const G of GAPS) {
  const n = results.filter((r) => r.splits[G] > 0).length;
  console.log(`sentenceGap ${G} kadr (son işarədən ~${((G + sp.lookahead) / 30).toFixed(1)} s sonra bağlanır): ${n} cümlə bölünür (${(100 * n / results.length).toFixed(1)} %)`);
}
const hs = results.flatMap((r) => r.finals.map((f) => f.hands)).sort((a, b) => a - b);
console.log(`
sözün kadrlarında əl olan pay: p5 ${hs[Math.floor(0.05 * hs.length)].toFixed(2)}, p10 ${hs[Math.floor(0.1 * hs.length)].toFixed(2)}, `
  + `< 0.5: ${(100 * hs.filter((h) => h < 0.5).length / hs.length).toFixed(1)} %, < 0.8: ${(100 * hs.filter((h) => h < 0.8).length / hs.length).toFixed(1)} %; `
  + `MIN_HANDS ${MIN_HANDS} ilə süzülən yekun söz: ${hs.filter((h) => h < MIN_HANDS).length}`);
if (outPath) writeFileSync(outPath, JSON.stringify({ model: m.run, rules: rows, gaps, results }, null, 0));

