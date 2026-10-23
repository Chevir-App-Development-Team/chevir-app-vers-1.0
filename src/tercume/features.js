/**
 * Landmark-lar → söz modelinin girişi. ml/azsl/features.py-dakı `preprocess()`-in dəqiq JS portu.
 *
 * Model, brauzer və landmark serveri eyni ön emalı işlətməlidir: FEATURE_VERSION modelin
 * manifest.json-undakı ilə eyni deyilsə, model işə salınmır (assertCompatible). Portun
 * düzgünlüyü ml/fixtures/features_v<N>.{json,bin} ilə yoxlanılır:
 *
 *   node tools/verify_features.mjs
 *
 * Python float32 ilə hesablayır; burada hər addım Math.fround ilə float32-yə yuvarlanır, median
 * və 64 kadra endirmə (numpy round — cütə yuvarlaqlaşdırma) eyni qaydada aparılır.
 *
 * Giriş (hər sahə T kadrlıq, sətir-sətir düz Float32Array, NaN = aşkarlanmayıb):
 *   hand        T×2×21×3   şəkil koordinatları x, y ∈ [0,1] (x en/hündürlüyə vurulMAMIŞ), z;
 *                          slot 0 = siqnalçının SAĞ əli (pozanın biləkləri qərar verir)
 *   hand_world  T×2×21×3   əlin metrik 3D forması (MediaPipe world landmarks)
 *   pose        T×9×3      POSE_IDX nöqtələri: x, y, z (yalnız x, y işlədilir)
 *   face        T×102×2    FACE_IDX nöqtələri: x, y
 *   blend       T×52       üz blendshape-ləri, BLEND_NAMES sırası ilə
 * Çıxış: { T, F, data } — T ≤ MAX_LEN kadr, F = sütun sayı, data = T×F Float32Array.
 */

export const FEATURE_VERSION = 1;
export const MAX_LEN = 64;
export const N_FEAT = 656;
export const TRAIN_FPS = 30;

export const POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24];
export const FACE_IDX = [
  61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
  78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
  263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398,
  33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173,
  276, 283, 282, 295, 285, 300, 293, 334, 296, 336,
  46, 53, 52, 65, 55, 70, 63, 105, 66, 107,
  1, 2, 98, 327, 10, 152, 234, 454, 468, 473,
];
export const BLEND_NAMES = [
  '_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
  'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft',
  'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft',
  'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft',
  'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft',
  'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft',
  'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
  'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
  'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight',
];

/** Xüsusiyyət sütunlarının qrupları (ablasiya modelləri yalnız bəzilərini oxuyur). */
export const GROUPS = {
  hands: [0, 126], hand_shape: [126, 252], hand_vel: [252, 378], pose: [378, 396],
  face: [396, 600], blend: [600, 652], flags: [652, 656],
};

const SIZES = { hand: 126, hand_world: 126, pose: 27, face: 204, blend: 52 };
const f32 = Math.fround;

/** Qrupların sütun indeksləri (boş = hamısı). */
export function columns(groups) {
  if (!groups || !groups.length) return Array.from({ length: N_FEAT }, (_, i) => i);
  const out = [];
  for (const g of groups) {
    const r = GROUPS[g];
    if (!r) throw new Error(`naməlum xüsusiyyət qrupu: ${g}`);
    for (let i = r[0]; i < r[1]; i++) out.push(i);
  }
  return out;
}

/** Modelin manifest.json-u bu ön emalla uyğun deyilsə xəta atır. */
export function assertCompatible(manifest) {
  if (manifest?.feature_version !== FEATURE_VERSION) {
    throw new Error(`model feature_version ${manifest?.feature_version} gözləyir, brauzer ${FEATURE_VERSION} hesablayır`);
  }
}

/**
 * Landmark-lardan model girişi.
 * @param {{T:number, hand:Float32Array, hand_world:Float32Array, pose:Float32Array, face:Float32Array, blend:Float32Array}} raw
 * @param {number} aspect kadrın eni / hündürlüyü
 * @param {{groups?: string[], face?: boolean, maxLen?: number | null}} [opts] face=false — məxfilik rejimi
 *   (üz nöqtələri yoxdur); maxLen=null — kadrlar azaldılmır (davamlı model, features.py max_len=None)
 */
export function preprocess(raw, aspect, { groups = null, face = true, maxLen = MAX_LEN } = {}) {
  let T = raw.T;
  for (const [k, n] of Object.entries(SIZES)) {
    if (raw[k].length !== T * n) throw new Error(`${k}: ${raw[k].length} ədəd, ${T}×${n} gözlənilirdi`);
  }
  // 1. kadr seçimi: maxLen-dən uzun klip bərabər addımla maxLen kadra endirilir
  let sel = Array.from({ length: T }, (_, i) => i);
  if (maxLen && T > maxLen) {
    const step = (T - 1) / (maxLen - 1);
    sel = Array.from({ length: maxLen }, (_, i) => roundHalfEven(i === maxLen - 1 ? T - 1 : i * step));
    T = maxLen;
  }
  // 2. kvadrat fəza: x *= en/hündürlük (float32)
  const a = f32(aspect);
  const pick = (k, n, scaleX, stride) => {
    const out = new Float32Array(T * n);
    sel.forEach((s, t) => {
      for (let j = 0; j < n; j++) {
        const v = raw[k][s * n + j];
        out[t * n + j] = scaleX && j % stride === 0 ? f32(v * a) : v;
      }
    });
    return out;
  };
  const hand = pick('hand', 126, true, 3);
  const handWorld = pick('hand_world', 126, false);
  const pose = pick('pose', 27, true, 3);
  const faceXY = face ? pick('face', 204, true, 2) : new Float32Array(T * 204).fill(NaN);
  const blend = face ? pick('blend', 52, false) : new Float32Array(T * 52).fill(NaN);

  // 3. mərkəz = çiyinlərin ortası (median), vahid = çiyin eni (median)
  const [cx, cy] = centre(T, pose, hand);
  const s = f32(scale(T, pose));

  // 4. sütunlar: əl mövqeyi · əl forması ×10 · əl sürəti · poza xy · üz xy · blendshape · bayraqlar
  const x = new Float32Array(T * N_FEAT);
  const handN = new Float32Array(T * 126);
  for (let i = 0; i < T * 42; i++) {
    handN[i * 3] = f32(f32(hand[i * 3] - cx) / s);
    handN[i * 3 + 1] = f32(f32(hand[i * 3 + 1] - cy) / s);
    handN[i * 3 + 2] = f32(hand[i * 3 + 2] / s);
  }
  for (let t = 0; t < T; t++) {
    const row = t * N_FEAT;
    for (let j = 0; j < 126; j++) {
      x[row + j] = handN[t * 126 + j];
      x[row + 126 + j] = f32(handWorld[t * 126 + j] * 10);
      x[row + 252 + j] = t === 0 ? 0 : f32(handN[t * 126 + j] - handN[(t - 1) * 126 + j]);
    }
    for (let p = 0; p < 9; p++) {
      x[row + 378 + p * 2] = f32(f32(pose[t * 27 + p * 3] - cx) / s);
      x[row + 378 + p * 2 + 1] = f32(f32(pose[t * 27 + p * 3 + 1] - cy) / s);
    }
    for (let p = 0; p < 102; p++) {
      x[row + 396 + p * 2] = f32(f32(faceXY[t * 204 + p * 2] - cx) / s);
      x[row + 396 + p * 2 + 1] = f32(f32(faceXY[t * 204 + p * 2 + 1] - cy) / s);
    }
    for (let j = 0; j < 52; j++) x[row + 600 + j] = blend[t * 52 + j];
    x[row + 652] = Number.isFinite(hand[t * 126]) ? 1 : 0;             // sağ əl
    x[row + 653] = Number.isFinite(hand[t * 126 + 63]) ? 1 : 0;        // sol əl
    x[row + 654] = Number.isFinite(faceXY[t * 204]) ? 1 : 0;           // üz
    x[row + 655] = Number.isFinite(pose[t * 27 + 3]) ? 1 : 0;          // poza (sol çiyin)
  }
  // 5. NaN → 0, ±10 ilə kəsilir
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    x[i] = Number.isFinite(v) ? Math.max(-10, Math.min(10, v)) : 0;
  }
  if (!groups || !groups.length) return { T, F: N_FEAT, data: x };
  const cols = columns(groups);
  const out = new Float32Array(T * cols.length);
  for (let t = 0; t < T; t++) {
    for (let c = 0; c < cols.length; c++) out[t * cols.length + c] = x[t * N_FEAT + cols[c]];
  }
  return { T, F: cols.length, data: out };
}

/** numpy.round kimi: tam yarımlar cüt ədədə yuvarlanır. */
function roundHalfEven(v) {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** NaN-ları atıb median, numpy kimi float32-də: cüt sayda ikisinin ortası (a+b)/2. */
function nanMedian(values) {
  const v = values.filter(Number.isFinite).sort((p, q) => p - q);
  if (!v.length) return NaN;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : f32(f32(v[m - 1] + v[m]) / 2);
}

function centre(T, pose, hand) {
  // poza nöqtələri 1 və 2 = sol və sağ çiyin (POSE_IDX-də 11, 12)
  const xs = [], ys = [];
  for (let t = 0; t < T; t++) {
    for (const p of [1, 2]) {
      xs.push(pose[t * 27 + p * 3]);
      ys.push(pose[t * 27 + p * 3 + 1]);
    }
  }
  let c = [nanMedian(xs), nanMedian(ys)];
  if (!Number.isFinite(c[0]) || !Number.isFinite(c[1])) {
    // poza yoxdursa — əl nöqtələrinin ortası, o da yoxdursa kadrın ortası
    let sx = 0, sy = 0, nx = 0, ny = 0;
    for (let i = 0; i < T * 42; i++) {
      const hx = hand[i * 3], hy = hand[i * 3 + 1];
      if (Number.isFinite(hx)) { sx += hx; nx++; }
      if (Number.isFinite(hy)) { sy += hy; ny++; }
    }
    c = nx && ny ? [f32(sx / nx), f32(sy / ny)] : [0.5, 0.5];
  }
  return [f32(c[0]), f32(c[1])];
}

function scale(T, pose) {
  const w = [];
  for (let t = 0; t < T; t++) {
    const dx = f32(pose[t * 27 + 3] - pose[t * 27 + 6]);
    const dy = f32(pose[t * 27 + 4] - pose[t * 27 + 7]);
    w.push(f32(Math.sqrt(f32(f32(dx * dx) + f32(dy * dy)))));
  }
  const m = nanMedian(w);
  return Number.isFinite(m) && m > 1e-3 ? m : 0.3;
}
