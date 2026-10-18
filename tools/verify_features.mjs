// JS ön emalını (src/tercume/features.js) və əl slotlarını (landmarks.js) Python ilə tutuşdurur:
// ml/fixtures/features_v<N>.{json,bin} — sintetik hallar və Python-un gözlənilən çıxışı.
import { readFileSync } from 'node:fs';
import * as FT from '../src/tercume/features.js';
import { assignHands } from '../src/tercume/landmarks.js';

const F = new URL(`../ml/fixtures/features_v${FT.FEATURE_VERSION}`, import.meta.url);
const meta = JSON.parse(readFileSync(new URL(`${F.href}.json`), 'utf8'));
const buf = readFileSync(new URL(`${F.href}.bin`));
const all = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const view = ({ offset, shape }) => all.subarray(offset, offset + shape.reduce((a, b) => a * b, 1));

let ok = true;
const fail = (msg) => { ok = false; console.log('❌', msg); };

// 1. sabitlər müqavilə ilə eynidir
const c = meta.contract;
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
if (meta.feature_version !== FT.FEATURE_VERSION) fail(`feature_version: fixture ${meta.feature_version}, JS ${FT.FEATURE_VERSION}`);
if (!same(c.pose_idx, FT.POSE_IDX)) fail('POSE_IDX fərqlidir');
if (!same(c.face_idx, FT.FACE_IDX)) fail('FACE_IDX fərqlidir');
if (!same(c.blend_names, FT.BLEND_NAMES)) fail('BLEND_NAMES fərqlidir');
if (!same(c.groups, FT.GROUPS)) fail('GROUPS fərqlidir');
if (c.max_len !== FT.MAX_LEN || c.n_features !== FT.N_FEAT) fail('MAX_LEN / N_FEAT fərqlidir');

// 2. hər hal üçün çıxış
for (const cs of meta.cases) {
  const raw = { T: cs.T };
  for (const k of ['hand', 'hand_world', 'pose', 'face', 'blend']) raw[k] = view(cs[k]);
  const exp = view(cs.expected);
  const [eT, eF] = cs.expected.shape;
  const got = FT.preprocess(raw, cs.aspect, { groups: cs.groups, face: cs.use_face });
  if (got.T !== eT || got.F !== eF) {
    fail(`${cs.name}: ölçü ${got.T}×${got.F}, gözlənilən ${eT}×${eF}`);
    continue;
  }
  let maxDiff = 0, worst = -1;
  for (let i = 0; i < exp.length; i++) {
    const d = Math.abs(got.data[i] - exp[i]);
    if (!(d <= maxDiff)) { maxDiff = d; worst = i; }
  }
  const pass = maxDiff <= meta.tolerance;
  console.log(`${pass ? '✅' : '❌'} ${cs.name.padEnd(34)} ${eT}×${eF}  maks fərq ${maxDiff.toExponential(2)}` +
    (pass ? '' : `  (sətir ${Math.floor(worst / eF)}, sütun ${worst % eF}: JS ${got.data[worst]}, Python ${exp[worst]})`));
  if (!pass) ok = false;
}

// 3. sağ/sol əl slotları (landmarks.js assignHands ↔ Python assign_hands)
let slotFails = 0;
for (const h of meta.hand_slots) {
  const got = assignHands(h.wrists, h.pose_wrists, h.aspect);
  if (JSON.stringify(got) !== JSON.stringify(h.slots)) {
    if (!slotFails) console.log('❌ ilk fərq:', JSON.stringify(h), '→ JS', JSON.stringify(got));
    slotFails++;
  }
}
console.log(`${slotFails ? '❌' : '✅'} əl slotları (assignHands)              ` +
  `${meta.hand_slots.length - slotFails}/${meta.hand_slots.length} eyni`);
if (slotFails) ok = false;

console.log(ok ? `\n✅ UYĞUNDUR — JS ön emalı Python preprocess() ilə eynidir (feature_version ${FT.FEATURE_VERSION})`
               : '\n❌ UYĞUN DEYİL');
process.exit(ok ? 0 : 1);
