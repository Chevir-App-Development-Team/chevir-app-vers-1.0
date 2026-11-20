// Cümlə rejimini (src/tercume/continuous.js) Python ilə tutuşdurur: ml/fixtures/continuous_v3.{json,bin}
// — axın cədvəli, decode() və 64 kadr həddi olmayan pəncərələrin ön emalı. Sonra onlayn hissələr:
// WindowStream = streamWindows, SentenceStream (hissə-hissə) = decode (bütöv). v3 (E32): işarə dil modeli (SignLM),
// decodeLM, nbest, alternatives, confidences; dil modelli SentenceStream = decodeLM, bağlananda alternativlər = Python.
import { readFileSync } from 'node:fs';
import * as CT from '../src/tercume/continuous.js';
import { preprocess } from '../src/tercume/features.js';

const meta = JSON.parse(readFileSync(new URL('../ml/fixtures/continuous_v3.json', import.meta.url), 'utf8'));
const buf = readFileSync(new URL('../ml/fixtures/continuous_v3.bin', import.meta.url));
const all = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const view = ({ offset, shape }) => all.subarray(offset, offset + shape.reduce((a, b) => a * b, 1));
let ok = true;
const fail = (m) => { ok = false; console.log('❌', m); };
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// 1. axın cədvəli: hər kadr bir dəfə, Python ilə eyni pəncərələr; onlayn variant da eyni
for (const s of meta.stream) {
  const st = { window: s.window, stride: s.stride, lookahead: s.lookahead };
  const got = CT.streamWindows(s.T, st).map((w) => [w.a, w.b, w.from, w.to]);
  if (!same(got, s.windows)) fail(`streamWindows T=${s.T} ${JSON.stringify(st)}`);
  const ws = new CT.WindowStream(st), online = [];
  for (let t = 0; t < s.T; t++) { const w = ws.push(); if (w) online.push([w.a, w.b, w.from, w.to]); }
  const last = ws.flush();
  if (last) online.push([last.a, last.b, last.from, last.to]);
  // onlayn axın sonu bilmir: sonda eyni pəncərəni bir də işlədə bilər. Vacib olan hər kadrın
  // ehtimalını hansı pəncərənin verdiyidir
  const source = (wins) => {
    const src = new Array(s.T);
    for (const [a, b, f0, f1] of wins) for (let t = f0; t < f1; t++) src[t] = `${a}-${b}`;
    return src;
  };
  if (!same(source(online), source(s.windows))) fail(`WindowStream T=${s.T} ${JSON.stringify(st)}: ${JSON.stringify(online)}`);
}
console.log(`${ok ? '✅' : '❌'} axın cədvəli: ${meta.stream.length} hal`);

// 2. decode: eyni sözlər, eyni kadrlar, ehtimal 1e-9 dəqiqliklə; hissə-hissə SentenceStream də eyni
for (const [i, d] of meta.decode.entries()) {
  const P = view(d.probs);
  const short = new Set(d.short);
  const got = CT.decode(P, d.T, d.C, d.params, d.unknown, short);
  const g = got.map((s) => [s.a, s.b, s.c]), e = d.expected.map((s) => s.slice(0, 3));
  const dp = Math.max(0, ...got.map((s, j) => Math.abs(s.p - (d.expected[j]?.[3] ?? NaN))));
  if (!same(g, e) || !(dp < 1e-9)) fail(`decode ${i}: ${JSON.stringify(g)} vs ${JSON.stringify(e)} (Δp ${dp})`);
  const ss = new CT.SentenceStream({ C: d.C, decode: d.params, unknown: d.unknown, short, sentenceGap: 1e9 });
  const words = [];
  for (let t = 0; t < d.T; t += 8) {
    const k = Math.min(8, d.T - t);
    words.push(...ss.add(P.subarray(t * d.C, (t + k) * d.C), k).added);
    // canlı göstəriş (/danis/): peek vəziyyəti dəyişmir və qalan kadrlarla bütöv decode-un qalan sözlərini verir
    const rest = d.T - t - k;
    const pk = ss.peek(P.subarray((t + k) * d.C, d.T * d.C), rest).map((s) => [s.a, s.b, s.c]);
    if (!same(pk, e.slice(words.length))) fail(`SentenceStream.peek ${i} t=${t}: ${JSON.stringify(pk)} vs ${JSON.stringify(e.slice(words.length))}`);
  }
  words.push(...ss.flush().added);
  if (!same(words.map((s) => [s.a, s.b, s.c]), e)) fail(`SentenceStream ${i}: ${JSON.stringify(words.map((s) => [s.a, s.b, s.c]))}`);
  console.log(`${same(g, e) ? '✅' : '❌'} decode ${i} ${JSON.stringify(d.params)}: ${g.length} söz`);
}

// 3. 64 kadr həddi olmayan ön emal (davamlı modelin pəncərələri)
for (const f of meta.features) {
  const raw = { T: f.T };
  for (const k of ['hand', 'hand_world', 'pose', 'face', 'blend']) raw[k] = view(f[k]);
  const exp = view(f.expected);
  const got = preprocess(raw, f.aspect, { groups: f.groups, face: false, maxLen: null });
  let md = 0;
  for (let i = 0; i < exp.length; i++) md = Math.max(md, Math.abs(got.data[i] - exp[i]));
  const good = got.T === f.expected.shape[0] && got.F === f.expected.shape[1] && md <= meta.tolerance;
  if (!good) fail(`features T=${f.T}: ${got.T}×${got.F}, maks fərq ${md}`);
  console.log(`${good ? '✅' : '❌'} ön emal, ${f.T} kadr, həddsiz: maks fərq ${md.toExponential(2)}`);
}

// 4. işarə dil modeli ilə deşifrə (v3, E32): eyni seqmentlər, sözlər, alternativlər və ballar (1e-9)
{
  const L = meta.lm;
  const close = (a, b) => Math.abs(a - b) < 1e-9;
  const sameWords = (got, exp) => got.length === exp.length && got.every((w, j) => w.a === exp[j][0] && w.b === exp[j][1]
    && w.c === exp[j][2] && close(w.p, exp[j][3]));
  const sameAlts = (got, exp) => got.length === exp.length && got.every((x, j) => close(x.score, exp[j][0]) && sameWords(x.words, exp[j][1]));
  const m3 = new CT.SignLM(L.models['3'], L.cases[0].labels);
  let bad = L.logp3.filter(([h, w, e]) => !close(m3.logp(h, w), e)).length;
  if (bad) fail(`SignLM.logp: ${bad} / ${L.logp3.length} fərqli`);
  if (!same(m3.tok, L.tokens_of_labels)) fail(`SignLM.tok ${JSON.stringify(m3.tok)}`);
  console.log(`${bad ? '❌' : '✅'} dil modeli: ${L.logp3.length} ln P(w | h), etiketlərin tokenləri`);
  for (const [i, d] of L.cases.entries()) {
    const P = view(d.probs), lp = d.lm_params, short = new Set(d.short);
    const lm = d.lm ? new CT.SignLM(L.models[String(d.lm)], d.labels) : null;
    const segs = CT.lmSegments(P, d.T, d.C, d.params, d.unknown, short, lp.top, lp.floor);
    const okSeg = segs.length === d.segments.length && segs.every((s, j) => s.a === d.segments[j][0] && s.b === d.segments[j][1]
      && s.cands.length === d.segments[j][2].length && s.cands.every((x, k) => x.c === d.segments[j][2][k][0] && close(x.q, d.segments[j][2][k][1])));
    const words = CT.decodeLM(P, d.T, d.C, d.params, lm, lp, d.unknown, short);
    const nb = CT.nbest(P, d.T, d.C, d.params, lm, lp, d.unknown, short);
    const alts = CT.alternatives(P, d.T, d.C, d.params, lm, lp, d.unknown, short);
    const conf = CT.confidences(alts.map((x) => x.score), lp.temp);
    const good = okSeg && sameWords(words, d.decode_lm) && sameAlts(nb, d.nbest) && sameAlts(alts, d.alternatives)
      && conf.length === d.confidences.length && conf.every((x, j) => close(x, d.confidences[j]));
    if (!good) fail(`dil modeli ${i}: seqmentlər ${okSeg}, decodeLM ${sameWords(words, d.decode_lm)}, nbest ${sameAlts(nb, d.nbest)}, `
      + `alternatives ${sameAlts(alts, d.alternatives)}`);
    // axın: dil modelli SentenceStream hissə-hissə = decodeLM bütöv; peek qalanı verir; bağlananda alternativlər eyni
    let okStream = true;
    if (lm) {
      const ss = new CT.SentenceStream({ C: d.C, decode: d.params, unknown: d.unknown, short, sentenceGap: 1e9, lm, lp });
      const got = [];
      const exp = d.decode_lm.map((w) => w.slice(0, 3));
      for (let t = 0; t < d.T; t += 8) {
        const k = Math.min(8, d.T - t);
        got.push(...ss.add(P.subarray(t * d.C, (t + k) * d.C), k).added);
        const pk = ss.peek(P.subarray((t + k) * d.C, d.T * d.C), d.T - t - k).map((s) => [s.a, s.b, s.c]);
        if (!same(pk, exp.slice(got.length))) { okStream = false; fail(`dil modelli peek ${i} t=${t}: ${JSON.stringify(pk)}`); break; }
      }
      const end = ss.flush();
      got.push(...end.added);
      if (!same(got.map((s) => [s.a, s.b, s.c]), exp)) { okStream = false; fail(`dil modelli SentenceStream ${i}: ${JSON.stringify(got.map((s) => [s.a, s.b, s.c]))}`); }
      if (!end.alternatives || !sameAlts(end.alternatives, d.alternatives)
        || !end.alternatives.every((x, j) => close(x.conf, d.confidences[j]))) { okStream = false; fail(`SentenceStream.alternatives ${i}`); }
    }
    console.log(`${good && okStream ? '✅' : '❌'} dil modeli ilə deşifrə ${i} (${d.lm ? `${d.lm}-qram` : 'dil modelsiz'}, `
      + `alpha ${lp.alpha} beta ${lp.beta} norm ${lp.norm}): ${segs.length} seqment, ${words.length} söz, ${alts.length} alternativ`
      + `${lm ? ', axın da' : ''}`);
  }
}

// 5. 30 fps-ə gətirmə: 30 fps-dən gələn kadrlar dəyişmir, 15 fps-də aradakı kadr ortadadır
{
  const fr = (v) => ({ hand: new Float32Array([v, NaN]) });
  const r = new CT.Resampler(30);
  const out = [];
  for (let i = 0; i < 4; i++) out.push(...r.push(fr(i), i * 1000 / 30));
  const r2 = new CT.Resampler(30), out2 = [];
  for (let i = 0; i < 3; i++) out2.push(...r2.push(fr(i * 2), i * 1000 / 15));
  const good = same(out.map((f) => f.hand[0]), [0, 1, 2, 3]) && same(out2.map((f) => Math.round(f.hand[0] * 1000) / 1000), [0, 1, 2, 3, 4])
    && out2.every((f) => Number.isNaN(f.hand[1]));
  if (!good) fail(`Resampler: ${JSON.stringify(out.map((f) => f.hand[0]))} ${JSON.stringify(out2.map((f) => f.hand[0]))}`);
  console.log(`${good ? '✅' : '❌'} 30 fps-ə gətirmə`);
}

console.log(ok ? `\n✅ UYĞUNDUR — cümlə rejimi Python continuous.py ilə eynidir (continuous_version ${meta.continuous_version})`
  : '\n❌ FƏRQ VAR');
process.exit(ok ? 0 : 1);
