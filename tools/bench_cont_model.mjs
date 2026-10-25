// Cümlə modelinin sürəti: brauzerin işlətdiyi kimi (features.js → onnxruntime-web WASM, bir axın), hər
// pəncərə üçün vaxt. 30 fps-də model hər 8 kadrda (267 ms) bir pəncərəni bitirməlidir.
//   node tools/bench_cont_model.mjs [pəncərə uzunluğu=96] [təkrar=30]
import { readFileSync } from 'node:fs';
import { LocalRecognizer } from '../src/tercume/recognizer.js';

const root = new URL('../', import.meta.url);
const MODEL = new URL('public/tercume/assets/cont_model/', root);
const T = Number(process.argv[2] ?? 96), N = Number(process.argv[3] ?? 30);
const fetchImpl = async (url) => {
  const data = readFileSync(new URL(url));
  return { ok: true, status: 200, json: async () => JSON.parse(data.toString('utf8')), arrayBuffer: async () => data };
};
const rec = await LocalRecognizer.load(MODEL.href, { fetchImpl });
const meta = JSON.parse(readFileSync(new URL('ml/fixtures/continuous_v2.json', root), 'utf8'));
const buf = readFileSync(new URL('ml/fixtures/continuous_v2.bin', root));
const all = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const f = meta.features.find((x) => x.T >= T);
const raw = { T };
for (const k of ['hand', 'hand_world', 'pose', 'face', 'blend']) {
  const per = f[k].shape.slice(1).reduce((a, b) => a * b, 1);
  raw[k] = all.subarray(f[k].offset, f[k].offset + T * per);
}
const times = [];
for (let i = 0; i < N + 3; i++) {
  const t0 = performance.now();
  await rec.frameProbs(raw, f.aspect, { face: false });
  if (i >= 3) times.push(performance.now() - t0);        // ilk 3: istiləşmə
}
times.sort((a, b) => a - b);
const q = (p) => times[Math.min(times.length - 1, Math.floor(p * times.length))].toFixed(1);
console.log(`${rec.manifest.run}: ${T} kadrlıq pəncərə, ${N} dəfə — median ${q(0.5)} ms, p90 ${q(0.9)} ms, maks ${q(1)} ms`);
console.log(`30 fps-də büdcə: 8 kadr = 267 ms; median büdcənin ${(100 * times[times.length >> 1] / 266.7).toFixed(0)} %-i`);
