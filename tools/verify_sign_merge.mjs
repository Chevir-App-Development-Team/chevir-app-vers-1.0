// Eyni işarənin etiketlərinin birləşməsi (recognizer.js signGroups, sign_merge.json): hər iki brauzer modeli
// birləşmə ilə və onsuz yüklənir, fixture-in halları ikisindən də keçir. Yoxlanılır: birləşmiş sinfin
// ehtimalı onun etiketlərinin ehtimallarının cəmidir, hər paylanma 1-ə bərabərdir, <blank> 0-cı yerdə qalır,
// birləşmənin özü (sinif sayı, qrup adları) sign_merge.json ilə eynidir.
//   node tools/verify_sign_merge.mjs
import { readFileSync } from 'node:fs';
import * as FT from '../src/tercume/features.js';
import { LocalRecognizer, signGroups } from '../src/tercume/recognizer.js';

const root = new URL('../', import.meta.url);
const merge = JSON.parse(readFileSync(new URL('public/tercume/assets/sign_merge.json', root), 'utf8'));
const fetchImpl = async (url) => {
  const data = readFileSync(new URL(url));
  return { ok: true, status: 200, json: async () => JSON.parse(data.toString('utf8')), arrayBuffer: async () => data };
};
let ok = true;
const check = (pass, msg) => { console.log(`${pass ? '✅' : '❌'} ${msg}`); if (!pass) ok = false; };

// 1. signGroups özü
{
  const g = signGroups(['<blank>', 'ACAM', 'EV', 'ACMİŞAM', 'COX', 'ÇOX'], merge);
  check(JSON.stringify(g.labels) === JSON.stringify(['<blank>', 'ACMİŞAM', 'EV', 'ÇOX'])
    && JSON.stringify([...g.classOf]) === JSON.stringify([0, 1, 2, 1, 3, 3]),
  `signGroups: ACAM + ACMİŞAM → ${g.labels[1]}, COX (yazılış) + ÇOX → ${g.labels[3]}, <blank> yerində`);
  check(signGroups(['A', 'B'], merge).classOf === null, 'birləşəcək etiket yoxdursa classOf = null (köhnə yol)');
  check(g.group('MƏNİM') === 'MƏN' && g.group('GEDƏCƏM') === 'GETMƏK', 'group(): MƏNİM → MƏN, GEDƏCƏM → GETMƏK');
}

// 2. hər model: birləşmiş ehtimal = cəm
const maxDiff = (rec, plain, pm, pp) => {
  let d = 0, s = 0;
  const sum = new Float64Array(rec.labels.length);
  for (let i = 0; i < pp.length; i++) sum[rec.classOf ? rec.classOf[i] : i] += pp[i];
  for (let c = 0; c < sum.length; c++) { d = Math.max(d, Math.abs(sum[c] - pm[c])); s += pm[c]; }
  return { d, s };
};
const F = new URL(`ml/fixtures/features_v${FT.FEATURE_VERSION}`, root);
const meta = JSON.parse(readFileSync(new URL(`${F.href}.json`), 'utf8'));
const buf = readFileSync(new URL(`${F.href}.bin`));
const all = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const view = ({ offset, shape }) => all.subarray(offset, offset + shape.reduce((a, b) => a * b, 1));

{
  const base = new URL('public/tercume/assets/word_model/', root).href;
  const plain = await LocalRecognizer.load(base, { fetchImpl });
  const rec = await LocalRecognizer.load(base, { fetchImpl, merge: new URL('public/tercume/assets/sign_merge.json', root).href });
  const want = new Set(plain.labels.map(rec.group)).size;
  check(rec.labels.length === want && rec.labels.length < plain.labels.length,
    `söz modeli ${rec.manifest.run}: ${plain.labels.length} etiket → ${rec.labels.length} işarə`);
  let worst = 0, sums = 1, n = 0;
  for (const cs of meta.cases) {
    const raw = { T: cs.T };
    for (const k of ['hand', 'hand_world', 'pose', 'face', 'blend']) raw[k] = view(cs[k]);
    const [a, b] = await Promise.all([rec.recognize(raw, cs.aspect), plain.recognize(raw, cs.aspect)]);
    const { d, s } = maxDiff(rec, plain, a.probs, b.probs);
    worst = Math.max(worst, d); sums = Math.max(sums, Math.abs(s - 1) + 1); n++;
    const pl = Float32Array.from(a.logits, Math.exp);
    worst = Math.max(worst, ...pl.map((v, i) => Math.abs(v - a.probs[i])));
  }
  check(worst < 1e-5 && sums - 1 < 1e-4, `söz modeli, ${n} hal: birləşmiş ehtimal = cəm (maks fərq ${worst.toExponential(1)}), logit-lər onlarla uyğun`);
}

{
  const base = new URL('public/tercume/assets/cont_model/', root).href;
  const plain = await LocalRecognizer.load(base, { fetchImpl });
  const rec = await LocalRecognizer.load(base, { fetchImpl, merge });
  // Birləşmələrlə öyrədilmiş modeldə etiketlər artıq işarələrdir: birləşmə heç nəyi dəyişməməlidir
  const trained = plain.labels.every((l) => rec.group(l) === l);
  check(rec.labels[0] === '<blank>' && (trained
    ? rec.classOf === null && rec.labels.length === plain.labels.length
    : rec.labels.length === new Set(plain.labels.map(rec.group)).size && rec.labels.length < plain.labels.length),
  `cümlə modeli ${rec.manifest.run}: ${plain.labels.length} etiket → ${rec.labels.length} işarə`
    + `${trained ? ' (birləşmə öyrədilərkən edilib, brauzer dəyişmir)' : ''}, <blank> 0-cı yerdə`);
  const cm = JSON.parse(readFileSync(new URL('ml/fixtures/continuous_v2.json', root), 'utf8'));
  const cb = readFileSync(new URL('ml/fixtures/continuous_v2.bin', root));
  const call = new Float32Array(cb.buffer.slice(cb.byteOffset, cb.byteOffset + cb.byteLength));
  let worst = 0, sums = 0, frames = 0;
  for (const f of cm.features) {
    const raw = { T: f.T };
    for (const k of ['hand', 'hand_world', 'pose', 'face', 'blend']) {
      const per = f[k].shape.slice(1).reduce((x, y) => x * y, 1);
      raw[k] = call.subarray(f[k].offset, f[k].offset + f.T * per);
    }
    const [a, b] = await Promise.all([rec.frameProbs(raw, f.aspect, { face: false }), plain.frameProbs(raw, f.aspect, { face: false })]);
    for (let t = 0; t < a.T; t++) {
      const { d, s } = maxDiff(rec, plain, a.probs.subarray(t * a.C, (t + 1) * a.C), b.probs.subarray(t * b.C, (t + 1) * b.C));
      worst = Math.max(worst, d); sums = Math.max(sums, Math.abs(s - 1)); frames++;
    }
  }
  check(worst < 1e-5 && sums < 1e-4, `cümlə modeli, ${frames} kadr: birləşmiş ehtimal = cəm (maks fərq ${worst.toExponential(1)}), hər kadr 1-ə bərabər`);
}

console.log(ok ? '\nHamısı keçdi.' : '\nUYĞUNSUZLUQ VAR.');
process.exit(ok ? 0 : 1);
