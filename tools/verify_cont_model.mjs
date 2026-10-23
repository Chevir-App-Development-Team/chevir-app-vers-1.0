// Brauzerdəki cümlə modelini (public/tercume/assets/cont_model/) PyTorch ilə tutuşdurur: continuous_v<N>
// fixture-inin pəncərələri → features.js (64 kadr həddi yoxdur) → onnxruntime-web → hər kadrın logit-ləri,
// ml/fixtures/cont_model_ref.json ilə (ml/install_continuous.py yazır; model dəyişəndə yenidən işə salın).
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { LocalRecognizer } from '../src/tercume/recognizer.js';
import { preprocess } from '../src/tercume/features.js';

const root = new URL('../', import.meta.url);
const MODEL = new URL('public/tercume/assets/cont_model/', root);
let ok = true;
const fail = (m) => { ok = false; console.log('❌', m); };

const ref = JSON.parse(readFileSync(new URL('ml/fixtures/cont_model_ref.json', root), 'utf8'));
const meta = JSON.parse(readFileSync(new URL(`ml/fixtures/continuous_v${ref.continuous_version}.json`, root), 'utf8'));
const buf = readFileSync(new URL(`ml/fixtures/continuous_v${ref.continuous_version}.bin`, root));
const all = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const view = ({ offset, shape }) => all.subarray(offset, offset + shape.reduce((a, b) => a * b, 1));

const manifest = JSON.parse(readFileSync(new URL('manifest.json', MODEL), 'utf8'));
const md5 = createHash('md5').update(readFileSync(new URL(manifest.model_file, MODEL))).digest('hex');
if (md5 !== ref.model_md5) fail(`model.onnx referansın modeli deyil: install_continuous.py-ni yenidən işə salın`);
const fetchImpl = async (url) => {
  const data = readFileSync(new URL(url));
  return { ok: true, status: 200, json: async () => JSON.parse(data.toString('utf8')), arrayBuffer: async () => data };
};
const rec = await LocalRecognizer.load(MODEL.href, { fetchImpl });
console.log(`model ${manifest.run}: ${manifest.labels.length} etiket (${manifest.kind}), ${manifest.input.n_features} sütun`);

for (const r of ref.cases) {
  const f = meta.features[r.case];
  const raw = { T: f.T };
  for (const k of ['hand', 'hand_world', 'pose', 'face', 'blend']) raw[k] = view(f[k]);
  const x = preprocess(raw, f.aspect, { groups: manifest.input.groups, face: r.face, maxLen: null });
  const out = await rec.session.run({ x: new rec.ort.Tensor('float32', x.data, [1, x.T, x.F]) });
  const lg = out[manifest.output.name].data, C = manifest.labels.length;
  let md = 0, argOk = 0;
  for (let t = 0; t < x.T; t++) {
    let best = 0;
    for (let j = 1; j < C; j++) if (lg[t * C + j] > lg[t * C + best]) best = j;
    argOk += best === r.argmax[t];
  }
  r.frames.forEach((t, i) => { for (let j = 0; j < C; j++) md = Math.max(md, Math.abs(lg[t * C + j] - r.logits[i][j])); });
  const good = x.T === r.T && md <= ref.tolerance && argOk === r.T;
  if (!good) fail(`pəncərə ${r.case}: maks logit fərqi ${md}, argmax ${argOk}/${r.T}`);
  console.log(`${good ? '✅' : '❌'} pəncərə ${r.case} (${r.T} kadr): maks logit fərqi ${md.toExponential(2)}, argmax ${argOk}/${r.T} eyni`);
  const fp = await rec.frameProbs(raw, f.aspect, { face: r.face });
  if (fp.T !== x.T || fp.C !== C) fail('frameProbs ölçüsü');
}
console.log(ok ? '\nHamısı keçdi.' : '\n❌ FƏRQ VAR');
process.exit(ok ? 0 : 1);
