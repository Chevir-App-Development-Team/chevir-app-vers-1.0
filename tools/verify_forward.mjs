// JS irəli keçidini numpy referansı ilə tutuşdurur.
import { readFileSync } from 'node:fs';
import { FingerspellModel } from '../src/tercume/model.js';

const A = new URL('../public/tercume/assets/', import.meta.url);
const F = new URL('./fixture/', import.meta.url);

const meta = JSON.parse(readFileSync(new URL('model.json', A), 'utf8'));
const bin = readFileSync(new URL('model.bin', A));
const model = new FingerspellModel(meta, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));

const inBuf = readFileSync(new URL('input.bin', F));
const x = new Float32Array(inBuf.buffer.slice(inBuf.byteOffset, inBuf.byteOffset + inBuf.byteLength));
const expected = JSON.parse(readFileSync(new URL('expected.json', F), 'utf8'));

const probs = model.predict(x);
let maxDiff = 0;
for (let i = 0; i < probs.length; i++) {
  maxDiff = Math.max(maxDiff, Math.abs(probs[i] - expected.probs[i]));
}
const top5 = model.topK(probs, 5);

console.log('JS (ilk 8)      :', Array.from(probs.slice(0, 8)).map(v => v.toFixed(6)).join(' '));
console.log('cəm             :', Array.from(probs).reduce((a, b) => a + b, 0));
console.log('top5            :', top5.map(t => `${t.idx}:${t.label}=${t.p.toFixed(5)}`).join('  '));
console.log('gözlənən top5   :', expected.top5.join(', '));
console.log('maks fərq       :', maxDiff.toExponential(3));
const idxOk = JSON.stringify(top5.map(t => t.idx)) === JSON.stringify(expected.top5);
console.log(maxDiff < 1e-5 && idxOk ? '\n✅ UYĞUNDUR — JS portu Keras çəkiləri ilə eyni nəticə verir'
                                   : '\n❌ UYĞUN DEYİL');
process.exit(maxDiff < 1e-5 && idxOk ? 0 : 1);
