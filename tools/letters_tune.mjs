// Hərf rejiminin yazma şərtləri (SignToText gate + pəncərə uzunluğu), hərf-hərf yazılmış sözlər üzərində:
// cümlə videolarındakı sürətlə (1×) və yavaşladılmış (2.5×, saxlanılan hərfə yaxın). Modelin ehtimalları hər
// kadr üçün bir dəfə hesablanır, şərtlər isə SignToText-in özü ilə (saxta model həmin ehtimalları qaytarır).
//   python -c "..."  → spelled_<split>/<k>_hand.npy, <k>_world.npy (ml/results/<run>/spelled_<split>.npz-dən)
//   node tools/letters_tune.mjs <model qovluğu> <spelled qovluğu> grid          → bütün şəbəkə, ən yaxşıları
//   node tools/letters_tune.mjs <model qovluğu> <spelled qovluğu> '{"window":16,...}'  → bir şərt dəsti
//   node tools/letters_tune.mjs old <spelled qovluğu> '{}'   → köhnə model (model.json + model.bin), öz şərtləri ilə
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { LetterModel } from '../src/tercume/letters.js';
import { FingerspellModel } from '../src/tercume/model.js';
import { DEFAULT_GATE, LETTER_GATE, SignToText } from '../src/tercume/s2t.js';

const [modelDir, spelledDir, mode = 'grid'] = process.argv.slice(2);
const SPEEDS = [1, 2.5];

function npy(path) {
  const b = readFileSync(path);
  const hl = b.readUInt16LE(8);
  const header = b.subarray(10, 10 + hl).toString('latin1');
  const n = header.match(/\(([^)]*)\)/)[1].split(',').filter((s) => s.trim()).map(Number);
  const body = b.subarray(10 + hl);
  if (header.includes("'<f2'")) {                       // float16 → float32
    const out = new Float32Array(body.length / 2);
    for (let i = 0; i < out.length; i++) {
      const h = body.readUInt16LE(i * 2), e = (h >> 10) & 31, m = h & 1023, s = h >> 15 ? -1 : 1;
      out[i] = e === 0 ? s * 2 ** -14 * (m / 1024) : e === 31 ? (m ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + m / 1024);
    }
    return { shape: n, data: out };
  }
  return { shape: n, data: new Float32Array(body.buffer.slice(body.byteOffset, body.byteOffset + body.byteLength)) };
}

/** Kadrlar (T × 126) → `speed` dəfə yavaş (xətti; NaN qonşuya keçir). */
function slow(a, T, speed) {
  if (speed === 1) return { a, T };
  const T2 = Math.round(T * speed), out = new Float32Array(T2 * 126);
  for (let i = 0; i < T2; i++) {
    const x = Math.min(T - 1, i / speed), lo = Math.floor(x), hi = Math.min(T - 1, lo + 1), w = x - lo;
    for (let j = 0; j < 126; j++) out[i * 126 + j] = a[lo * 126 + j] * (1 - w) + a[hi * 126 + j] * w;
  }
  return { a: out, T: T2 };
}

const fetchImpl = async (url) => {
  const data = readFileSync(new URL(url));
  return { ok: true, status: 200, json: async () => JSON.parse(data.toString('utf8')), arrayBuffer: async () => data };
};
const OLD = modelDir === 'old';
const assets = new URL('../public/tercume/assets/', import.meta.url);
const model = OLD
  ? new FingerspellModel(JSON.parse(readFileSync(new URL('model.json', assets), 'utf8')),
    (() => { const b = readFileSync(new URL('model.bin', assets)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); })())
  : await LetterModel.load(pathToFileURL(`${modelDir}/`).href, { fetchImpl });
if (OLD) model.manifest = { run: 'old model (model.json)' };
const words = JSON.parse(readFileSync(`${spelledDir}.json`, 'utf8'));
const lev = (a, b) => {
  const d = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    let prev = d[0]; d[0] = i;
    for (let j = 1; j <= b.length; j++) { const c = d[j]; d[j] = Math.min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1)); prev = c; }
  }
  return d[b.length];
};

const nan = (n) => new Float32Array(n).fill(NaN);
const cache = new Map();          // `${k}|${speed}|${window}` → { frames, probs[t] }
async function prepared(k, speed, window) {
  const key = `${k}|${speed}|${window}`;
  if (cache.has(key)) return cache.get(key);
  const w = words[k];
  const h = slow(npy(`${spelledDir}/${k}_hand.npy`).data, w.T, speed), wd = slow(npy(`${spelledDir}/${k}_world.npy`).data, w.T, speed);
  const frames = [];
  for (let t = 0; t < h.T; t++) {
    frames.push({ hand: h.a.slice(t * 126, (t + 1) * 126), hand_world: wd.a.slice(t * 126, (t + 1) * 126),
      pose: nan(27), face: nan(204), blend: nan(52) });
  }
  const probs = [];
  for (let t = 0; !OLD && t < frames.length; t++) {
    const fr = frames.slice(Math.max(0, t + 1 - window), t + 1);
    const raw = { T: fr.length };
    for (const [f, n] of [['hand', 126], ['hand_world', 126], ['pose', 27], ['face', 204], ['blend', 52]]) {
      const arr = new Float32Array(fr.length * n);
      fr.forEach((x, i) => arr.set(x[f], i * n));
      raw[f] = arr;
    }
    probs.push(await model.predictRaw(raw, w.aspect));
  }
  const out = { frames, probs };
  cache.set(key, out);
  return out;
}

/** Bir şərt dəsti, bütün sözlər və sürətlər: hərf səhvi = redaktə / annotasiya hərfi. */
async function run(gate) {
  const res = {};
  for (const speed of SPEEDS) {
    let edits = 0, total = 0, typedN = 0;
    for (let k = 0; k < words.length; k++) {
      const { frames, probs } = await prepared(k, speed, gate.window);
      let cur = 0;
      const fake = OLD ? model : { kind: 'letters', labels: model.labels, none: model.none, dynamic: model.dynamic,
        window: gate.window, topK: (p, n) => model.topK(p, n), predictRaw: async () => probs[cur] };
      const typed = [];
      const s2t = new SignToText({ model: fake, decoder: { decode: () => ({}) }, video: null,
        onDecode: ({ letters }) => { if (letters) typed.splice(0, typed.length, ...letters); } });
      s2t.gate = { ...(OLD ? DEFAULT_GATE : LETTER_GATE), ...gate };
      for (cur = 0; cur < frames.length; cur++) {
        const f = frames[cur];
        const slot = Number.isFinite(f.hand[0]) ? 0 : Number.isFinite(f.hand[63]) ? 63 : -1;
        s2t.pushFrame(slot < 0 ? null : f.hand.slice(slot, slot + 63), f, words[k].aspect);
        if (s2t.pending) await s2t.pending;
      }
      const want = words[k].letters.map((x) => x[2]);
      edits += lev(typed, want); total += want.length; typedN += typed.length;
    }
    res[speed] = { ler: edits / total, typed: typedN / total };
  }
  res.mean = SPEEDS.reduce((s, sp) => s + res[sp].ler, 0) / SPEEDS.length;
  return res;
}

const fmt = (g, r) => `${JSON.stringify(g)} → ${SPEEDS.map((s) => `${s}×: səhv ${r[s].ler.toFixed(3)} (yazılan/hərf ${r[s].typed.toFixed(2)})`).join(' | ')} | orta ${r.mean.toFixed(3)}`;
console.log(`${model.manifest.run}: ${words.length} söz, ${words.reduce((s, w) => s + w.letters.length, 0)} hərf`);
if (mode === 'grid') {
  const rows = [];
  for (const window of [12, 16, 24]) {
    for (const stride of [2, 4]) for (const dwell of [2, 3]) for (const maxMotion of [0.05, 0.1, 0.2, 99]) {
      for (const minConfidence of [0.5, 0.6, 0.7]) {
        const g = { window, stride, dwell, maxMotion, minConfidence };
        rows.push([g, await run(g)]);
      }
    }
  }
  rows.sort((a, b) => a[1].mean - b[1].mean);
  console.log('ən yaxşı 10:'); rows.slice(0, 10).forEach(([g, r]) => console.log(' ', fmt(g, r)));
  const now = rows.find(([g]) => g.window === 24 && g.stride === 4 && g.dwell === 3 && g.maxMotion === 0.05 && g.minConfidence === 0.5);
  console.log('indiki (LETTER_GATE, pəncərə 24):', fmt(...now));
} else {
  const g = { window: model.window ?? 20, ...JSON.parse(mode) };
  console.log(fmt(g, await run(g)));
}
