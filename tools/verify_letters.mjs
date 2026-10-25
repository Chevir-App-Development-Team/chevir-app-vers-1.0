// Hərf rejimini (src/tercume/letters.js) Python ilə tutuşdurur: ml/fixtures/letters_v<N>.{json,bin} —
// strip → canonical → əl xüsusiyyətləri. Hərf modeli quraşdırılıbsa (public/tercume/assets/letter_model/),
// onun ehtimallarını da ml/fixtures/letters_model_ref.json ilə (ml/install_letters.py yazır).
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { LETTERS_VERSION, LetterModel, preprocessLetters } from '../src/tercume/letters.js';

const root = new URL('../', import.meta.url);
const meta = JSON.parse(readFileSync(new URL(`ml/fixtures/letters_v${LETTERS_VERSION}.json`, root), 'utf8'));
const buf = readFileSync(new URL(`ml/fixtures/letters_v${LETTERS_VERSION}.bin`, root));
const all = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const view = ({ offset, shape }) => all.subarray(offset, offset + shape.reduce((a, b) => a * b, 1));
let ok = true;
const fail = (m) => { ok = false; console.log('❌', m); };
const rawOf = (c) => {
  const raw = { T: c.T };
  for (const k of ['hand', 'hand_world', 'pose', 'face', 'blend']) raw[k] = Float32Array.from(view(c[k]));
  return raw;
};

for (const c of meta.cases) {
  for (const groups of meta.groups) {
    const exp = view(c['expected_' + groups.join('+')]);
    const x = preprocessLetters(rawOf(c), c.aspect, groups);
    let md = 0;
    for (let i = 0; i < exp.length; i++) md = Math.max(md, Math.abs(x.data[i] - exp[i]));
    const good = x.data.length === exp.length && md <= meta.tolerance;
    if (!good) fail(`${c.name} ${groups.join('+')}: ${x.T}×${x.F}, maks fərq ${md}`);
    console.log(`${good ? '✅' : '❌'} ${c.name}, ${groups.join('+')}: ${x.T}×${x.F}, maks fərq ${md.toExponential(2)}`);
  }
}

const MODEL = new URL('public/tercume/assets/letter_model/', root);
const REF = new URL('ml/fixtures/letters_model_ref.json', root);
if (existsSync(new URL('manifest.json', MODEL)) && existsSync(REF)) {
  const ref = JSON.parse(readFileSync(REF, 'utf8'));
  const fetchImpl = async (url) => {
    const data = readFileSync(new URL(url));
    return { ok: true, status: 200, json: async () => JSON.parse(data.toString('utf8')), arrayBuffer: async () => data };
  };
  const model = await LetterModel.load(MODEL.href, { fetchImpl });
  const md5 = createHash('md5').update(readFileSync(new URL(model.manifest.model_file, MODEL))).digest('hex');
  if (md5 !== ref.model_md5) fail('model.onnx referansın modeli deyil: install_letters.py-ni yenidən işə salın');
  for (const r of ref.cases) {
    const c = meta.cases.find((x) => x.name === r.case);
    const p = await model.predictRaw(rawOf(c), c.aspect);
    let md = 0;
    p.forEach((v, j) => { md = Math.max(md, Math.abs(v - r.probs[j])); });
    const good = md <= ref.tolerance;
    if (!good) fail(`model ${r.case}: maks ehtimal fərqi ${md}`);
    console.log(`${good ? '✅' : '❌'} model (${model.manifest.run}) ${r.case}: maks ehtimal fərqi ${md.toExponential(2)}, `
      + `ən yaxşı ${model.topK(p, 1)[0].label}`);
  }
} else {
  console.log('(hərf modeli quraşdırılmayıb: yalnız ön emal yoxlanıldı)');
}
console.log(ok ? `\n✅ UYĞUNDUR — hərf rejimi Python letters.py ilə eynidir (letters_version ${meta.letters_version})` : '\n❌ FƏRQ VAR');
process.exit(ok ? 0 : 1);
