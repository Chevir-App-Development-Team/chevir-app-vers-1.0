// Üz kanalının JS portunu (src/tercume/nonmanual.js) Python ilə tutuşdurur (ml/azsl/nonmanual.py, ml/nonmanual.py):
// ml/fixtures/nonmanual_v<N>.{json,bin} — sintetik kliplər, Python-un kadr əlamətləri, qalxmış əllər, aktiv aralıq,
// neytral üz, cümlə vektoru və quraşdırılmış modelin (public/tercume/assets/nonmanual.json) ehtimalları.
//   node tools/verify_nonmanual.mjs
import { readFileSync } from 'node:fs';
import * as NM from '../src/tercume/nonmanual.js';
import { GlossToText } from '../src/tercume/gloss2text.js';

const F = new URL(`../ml/fixtures/nonmanual_v${NM.NM_VERSION}`, import.meta.url);
const meta = JSON.parse(readFileSync(new URL(`${F.href}.json`), 'utf8'));
const buf = readFileSync(new URL(`${F.href}.bin`));
const all = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const view = ({ offset, shape }) => all.subarray(offset, offset + shape.reduce((a, b) => a * b, 1));
const model = JSON.parse(readFileSync(new URL('../public/tercume/assets/nonmanual.json', import.meta.url), 'utf8'));

let ok = true;
const fail = (msg) => { ok = false; console.log('❌', msg); };

if (meta.nm_version !== NM.NM_VERSION || model.nm_version !== NM.NM_VERSION) {
  fail(`nm_version: fixture ${meta.nm_version}, model ${model.nm_version}, JS ${NM.NM_VERSION}`);
}
if (meta.frame_names.length !== NM.N_FRAME || model.frame_names.length !== NM.N_FRAME) fail('kadr əlamətlərinin sayı fərqlidir');
if (JSON.stringify(meta.stats) !== JSON.stringify(NM.STATS)) fail('STATS fərqlidir');
if (model.n_clip !== NM.N_CLIP) fail(`cümlə vektoru: model ${model.n_clip}, JS ${NM.N_CLIP}`);
for (const [k, h] of Object.entries(model.heads)) if (h.coef.length !== NM.N_CLIP) fail(`${k}: ${h.coef.length} çəki`);

/** maks fərq; NaN-lar eyni yerdə olmalıdır */
function diff(got, exp) {
  let m = 0;
  for (let i = 0; i < exp.length; i++) {
    const a = got[i], b = exp[i];
    if (Number.isNaN(b) || Number.isNaN(a)) { if (Number.isNaN(a) !== Number.isNaN(b)) return Infinity; continue; }
    m = Math.max(m, Math.abs(a - b) / Math.max(1, Math.abs(b)));
  }
  return m;
}

for (const cs of meta.cases) {
  const raw = { T: cs.T };
  for (const k of ['hand', 'pose', 'face', 'blend']) raw[k] = view(cs[k]);
  const r = NM.toFps(raw, cs.fps);
  const x = NM.frameFeatures(r, cs.aspect);
  const up = NM.handsUp(r, cs.aspect);
  const act = NM.activeRange(up);
  const base = NM.neutral(x, up, model.neutral, model.min_rest);
  const clip = NM.clipFeatures(x, up, base);
  const p = NM.predict(model, raw, cs.aspect, cs.fps);
  const d = { frames: diff(x, view(cs.frames)), neutral: diff(base, view(cs.neutral)), clip: diff(clip, view(cs.clip)) };
  const upSame = up.length === cs.up.length && cs.up.every((v, i) => v === up[i]);
  const actSame = act[0] === cs.active[0] && act[1] === cs.active[1];
  let pd = Math.abs(p.face_rate - cs.p.face_rate);
  for (const k of Object.keys(model.heads)) {
    if ((p[k] === null) !== (cs.p[k] === null)) pd = Infinity;
    else if (p[k] !== null) pd = Math.max(pd, Math.abs(p[k] - cs.p[k]));
  }
  const pass = upSame && actSame && d.frames <= meta.tolerance && d.neutral <= meta.tolerance && d.clip <= meta.tolerance
    && pd <= meta.p_tolerance;
  console.log(`${pass ? '✅' : '❌'} ${cs.name.padEnd(22)} ${String(r.T).padStart(3)} kadr  əllər ${upSame ? 'eyni' : 'FƏRQLİ'}, `
    + `aralıq ${actSame ? 'eyni' : `FƏRQLİ ${act} / ${cs.active}`}  maks fərq: kadr ${d.frames.toExponential(1)}, `
    + `neytral ${d.neutral.toExponential(1)}, cümlə ${d.clip.toExponential(1)}, ehtimal ${pd.toExponential(1)}`
    + `  (yn ${p.yn === null ? '—' : p.yn.toFixed(4)}, neg ${p.neg === null ? '—' : p.neg.toFixed(4)})`);
  if (!pass) ok = false;
}

// marks(): sual sözü işarəsi → wh; üz ≥ hədd → yn; hər ikisi yoxdursa null
const t = model.heads.yn.t, tn = model.heads.neg.t;
const cases = [
  [['SİZ', 'AD', 'NƏDİR'], { yn: 0, neg: 0 }, 'wh', false],
  [['SİZ', 'QƏLƏM', 'VAR'], { yn: t + 0.01, neg: 0 }, 'yn', false],
  [['SİZ', 'QƏLƏM', 'VAR'], { yn: t - 0.01, neg: tn + 0.01 }, null, true],
  [['MƏN', 'GETMƏK'], { yn: null, neg: null }, null, false],
];
for (const [labels, pr, q, n] of cases) {
  const m = NM.marks(model, labels, pr);
  if (m.question !== q || m.neg !== n) fail(`marks(${labels.join(' ')}): ${JSON.stringify(m)}, gözlənilən ${q}, ${n}`);
}
console.log(`${ok ? '✅' : '❌'} marks(): ${cases.length} hal`);

// gloss2text-in nonmanual seçimi: verilməyibsə mətn dəyişmir; "?", -mı, inkar (əl ilə inkar varsa üz toxunmur)
const g2t = new GlossToText(JSON.parse(readFileSync(new URL('../public/tercume/assets/gloss_text.json', import.meta.url), 'utf8')));
const gen = (ls, nm) => g2t.generate(ls, nm ? { nonmanual: nm } : {}).text;
const textCases = [
  [['SİZ', 'QƏLƏM', 'VAR'], { question: 'yn', mi: true }, (t, base) => t === `${base.slice(0, -3)}varmı?`],
  [['SİZ', 'QƏLƏM', 'VAR'], { question: 'yn' }, (t, base) => t === `${base}?`],
  [['SİZ', 'AD', 'NƏDİR'], { question: 'wh', mi: true }, (t, base) => t === `${base}?`],
  [['MƏN', 'ÇAY', 'İÇMƏK', 'İSTƏMƏK'], { neg: true }, (t) => t.endsWith('istəmirəm')],
  [['MƏN', 'ÇAY', 'İÇMƏK', 'İSTƏMİR'], { neg: true }, (t, base) => t === base],      // əl ilə inkar: dəyişmir
  [['MƏN', 'VAXT', 'YOX'], { neg: true }, (t, base) => t === base],
  [['MƏN', 'ÇAY', 'İÇMƏK'], { question: null, neg: false }, (t, base) => t === base],
];
let textFails = 0;
for (const [ls, nm, check] of textCases) {
  const base = gen(ls, null), t = gen(ls, nm);
  if (!check(t, base)) { textFails++; console.log('❌', ls.join(' '), JSON.stringify(nm), '→', t, `(üzsüz: ${base})`); }
}
console.log(`${textFails ? '❌' : '✅'} gloss2text nonmanual: ${textCases.length - textFails}/${textCases.length}`);
if (textFails) ok = false;

console.log(ok ? '\nhamısı keçdi' : '\nFƏRQ VAR');
process.exit(ok ? 0 : 1);
