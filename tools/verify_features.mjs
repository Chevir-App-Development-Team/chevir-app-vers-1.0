// JS ön emalını (src/tercume/features.js) və əl slotlarını (landmarks.js) Python ilə tutuşdurur:
// ml/fixtures/features_v<N>.{json,bin} — sintetik hallar və Python-un gözlənilən çıxışı.
import { readFileSync } from 'node:fs';
import * as FT from '../src/tercume/features.js';
import { assignHands, frameFromResults, LandmarkClip, FRAME_SIZES, EXTRA_SIZES } from '../src/tercume/landmarks.js';

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

// 4. kadr quruluşu: saxta MediaPipe nəticəsi → frameFromResults → toRaw
{
  const pt = (x, y, z) => ({ x, y, z });
  const pose = Array.from({ length: 33 }, (_, j) => pt(0.4 + j / 100, 0.3 + j / 200, -j / 1000));
  const world = Array.from({ length: 33 }, (_, j) => pt(j / 10, -j / 20, j / 30));
  const hand = (cx) => Array.from({ length: 21 }, (_, j) => pt(cx + j / 1000, 0.6, j / 500));
  const hr = { landmarks: [hand(0.56), hand(0.30)], worldLandmarks: [hand(0.01), hand(0.02)] };
  const pr = { landmarks: [pose], worldLandmarks: [world] };
  const f = frameFromResults(hr, pr, null, 4 / 3);
  const clip = new LandmarkClip();
  clip.push(f);
  clip.push(f);
  const plain = clip.toRaw(), full = clip.toRaw(0, 2, { extra: true });
  const checks = [
    ['pose_world = worldLandmarks[POSE_IDX]', FT.POSE_IDX.every((j, k) => f.pose_world[k * 3] === Math.fround(j / 10)
      && f.pose_world[k * 3 + 1] === Math.fround(-j / 20) && f.pose_world[k * 3 + 2] === Math.fround(j / 30))],
    ['pose = landmarks[POSE_IDX] (x, y, z)', FT.POSE_IDX.every((j, k) => f.pose[k * 3] === Math.fround(0.4 + j / 100)
      && f.pose[k * 3 + 2] === Math.fround(-j / 1000))],
    ['üz yoxdursa face/blend NaN', f.face.every(Number.isNaN) && f.blend.every(Number.isNaN)],
    ['toRaw() — yalnız müqavilə sahələri', same(Object.keys(plain).filter((k) => k !== 'T'), Object.keys(FRAME_SIZES))],
    ['toRaw({ extra }) — pose_world da', full.pose_world?.length === 2 * EXTRA_SIZES.pose_world],
    ['preprocess əlavə sahədən asılı deyil',
      same(Array.from(FT.preprocess(plain, 4 / 3).data), Array.from(FT.preprocess(full, 4 / 3).data))],
  ];
  for (const [name, pass] of checks) {
    console.log(`${pass ? '✅' : '❌'} kadr: ${name}`);
    if (!pass) ok = false;
  }
}

console.log(ok ? `\n✅ UYĞUNDUR — JS ön emalı Python preprocess() ilə eynidir (feature_version ${FT.FEATURE_VERSION})`
               : '\n❌ UYĞUN DEYİL');
process.exit(ok ? 0 : 1);
