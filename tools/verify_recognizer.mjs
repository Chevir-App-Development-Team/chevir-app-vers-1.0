// Brauzerdəki söz modelini (public/tercume/assets/word_model/) PyTorch ilə tutuşdurur:
// fixture-in sintetik halları → features.js preprocess → onnxruntime-web (WASM) → logit-lər, və
// ml/fixtures/word_model_ref.json — eyni hallar üçün best.pt-nin (təlimdəki yol) logit-ləri.
// Referansı ml/make_manifest.py yazır; model dəyişəndə onu yenidən işə salın.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import * as FT from '../src/tercume/features.js';
import { LocalRecognizer } from '../src/tercume/recognizer.js';

const root = new URL('../', import.meta.url);
const MODEL = new URL('public/tercume/assets/word_model/', root);
const md5 = (url) => createHash('md5').update(readFileSync(url)).digest('hex');

let ok = true;
const fail = (msg) => { ok = false; console.log('❌', msg); };

const ref = JSON.parse(readFileSync(new URL('ml/fixtures/word_model_ref.json', root), 'utf8'));
const F = new URL(`ml/fixtures/features_v${FT.FEATURE_VERSION}`, root);
const meta = JSON.parse(readFileSync(new URL(`${F.href}.json`), 'utf8'));
const buf = readFileSync(new URL(`${F.href}.bin`));
const all = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const view = ({ offset, shape }) => all.subarray(offset, offset + shape.reduce((a, b) => a * b, 1));

// 1. quraşdırılmış model referansın modelidir
const manifest = JSON.parse(readFileSync(new URL('manifest.json', MODEL), 'utf8'));
const modelMd5 = md5(new URL(manifest.model_file, MODEL));
if (modelMd5 !== ref.model_md5) fail(`model.onnx referansın modeli deyil (${modelMd5} ≠ ${ref.model_md5}): make_manifest.py-ni yenidən işə salın`);
const pkg = JSON.parse(readFileSync(new URL('node_modules/onnxruntime-web/package.json', root), 'utf8'));

// 2. hər hal: JS preprocess + onnxruntime-web ↔ PyTorch
const fetchImpl = async (url) => {
  const data = readFileSync(new URL(url));
  return { ok: true, status: 200, json: async () => JSON.parse(data.toString('utf8')), arrayBuffer: async () => data };
};
const rec = await LocalRecognizer.load(MODEL.href, { fetchImpl });
console.log(`model ${manifest.run}: ${manifest.labels.length} söz, ${manifest.input.n_features} sütun, ` +
  `qruplar ${manifest.input.groups.join(', ')}; onnxruntime-web ${pkg.version}`);
const cases = new Map(meta.cases.map((c) => [c.name, c]));
for (const r of ref.cases) {
  const cs = cases.get(r.name);
  const raw = { T: cs.T };
  for (const k of ['hand', 'hand_world', 'pose', 'face', 'blend']) raw[k] = view(cs[k]);
  const got = await rec.recognize(raw, cs.aspect, { face: r.face, topK: 5 });
  let maxDiff = 0;
  for (let i = 0; i < r.logits.length; i++) maxDiff = Math.max(maxDiff, Math.abs(got.logits[i] - r.logits[i]));
  const sameTop = JSON.stringify(got.top.map((t) => t.idx)) === JSON.stringify(r.top5);
  const pass = got.frames === r.T && maxDiff <= ref.tolerance && sameTop;
  console.log(`${pass ? '✅' : '❌'} ${(r.name + (r.face ? '' : ' (üzsüz)')).padEnd(44)} T=${String(got.frames).padStart(2)}` +
    `  maks logit fərqi ${maxDiff.toExponential(2)}  top-1 ${got.top[0].label} (${got.top[0].p.toFixed(3)})` +
    `  ${got.ms.toFixed(0)} ms${sameTop ? '' : `  top-5 fərqli: JS ${got.top.map((t) => t.idx)}, PyTorch ${r.top5}`}`);
  if (!pass) ok = false;
}

console.log(ok ? '\nHamısı keçdi.' : '\nUYĞUNSUZLUQ VAR.');
process.exit(ok ? 0 : 1);
