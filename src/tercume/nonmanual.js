/**
 * Üz kanalı (roadmap R3, ml/EXPERIMENTS.md E31): cümlə bəli/xeyr sualıdırmı, inkardırmı — üzdən və başdan.
 * ml/azsl/nonmanual.py-ın dəqiq JS portu (tools/verify_nonmanual.mjs, ml/fixtures/nonmanual_v<N>).
 *
 * Giriş tanıma müqaviləsinin kadrlarıdır (landmarks.js LandmarkClip.toRaw: hand T×126, pose T×27, face T×204,
 * blend T×52; NaN = yoxdur), 30 fps. Model: public/tercume/assets/nonmanual.json (ml/nonmanual.py fit).
 *
 * 1. Kadr (frameFeatures): baş pozası, qaşların gözdən hündürlüyü, gözün, ağzın açıqlığı (başın öz oxlarında,
 *    göz arası məsafə vahidi ilə) və 51 blendshape.
 * 2. Neytral üz (neutral): adamın öz üzü, əllər aşağıdaykən; kadrlardan çıxılır.
 * 3. Cümlə (clipFeatures): əllərin qalxdığı aralıqda orta, std, maks, min, son və ilk üçdə bir, başın hərəkəti.
 * 4. Başlıqlar: logistik reqressiya — yn (bəli/xeyr sualı), neg (inkar). Sual sözü olan sual (NƏ, HARA…) əllə
 *    göstərilir, onu tanınan işarələr verir (marks()).
 */
import { FACE_IDX, POSE_IDX } from './features.js';

export const NM_VERSION = 1;
export const FPS = 30;
const F = new Map(FACE_IDX.map((m, k) => [m, k]));
const EYE_R = [33, 133], EYE_L = [263, 362];
const LID_R = [159, 145], LID_L = [386, 374];
const BROW_R = [107, 105, 70], BROW_L = [336, 334, 300];
const NOSE = 1, CHIN = 152, CHEEK_R = 234, CHEEK_L = 454;
const LIP_IN = [13, 14], LIP_OUT = [0, 17], MOUTH_C = [61, 291];
const SH_L = POSE_IDX.indexOf(11), SH_R = POSE_IDX.indexOf(12);
export const N_GEOM = 19;
export const N_FRAME = N_GEOM + 51;
export const STATS = ['mean', 'std', 'max', 'min', 'last', 'first'];
const MOTION = [1, 3, 0];                        // yaw, pitch, roll (FRAME_NAMES-dəki yerləri)
export const N_CLIP = N_FRAME * STATS.length + 7;

/** numpy.round kimi: yarımlar cütə. */
const roundEven = (x) => {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
};

/** Kadrları 30 fps-ə gətirir (ən yaxın kadr). raw: {T, hand, pose, face, blend}. */
export function toFps(raw, fps, target = FPS) {
  const T = raw.T;
  if (Math.abs(fps - target) < 0.5 || T === 0) return raw;
  const n = Math.max(1, roundEven((T * target) / fps));
  const out = { T: n };
  for (const k of ['hand', 'pose', 'face', 'blend']) {
    const w = raw[k].length / T, a = new Float32Array(n * w);
    for (let i = 0; i < n; i++) {
      const j = Math.min(T - 1, Math.max(0, roundEven((i * fps) / target)));
      a.set(raw[k].subarray(j * w, (j + 1) * w), i * w);
    }
    out[k] = a;
  }
  return out;
}

/** Bir kadrın əlamətləri (N_FRAME, üz yoxdursa NaN) — frame_features-in bir sətri. */
export function frameRow(face, pose, blend, aspect, out = new Float32Array(N_FRAME), fo = 0, po = 0, bo = 0) {
  const P = (m) => { const k = fo + F.get(m) * 2; return [face[k] * aspect, face[k + 1]]; };
  const sh = (k) => [pose[po + k * 3] * aspect, pose[po + k * 3 + 1]];
  const avg = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const er = avg(P(EYE_R[0]), P(EYE_R[1])), el = avg(P(EYE_L[0]), P(EYE_L[1]));
  const d = [el[0] - er[0], el[1] - er[1]];
  let io = Math.hypot(d[0], d[1]);
  if (!(io > 1e-6)) io = NaN;
  const ex = [d[0] / io, d[1] / io], ey = [-ex[1], ex[0]];
  const mid = avg(er, el);
  const u = (p, o = mid) => ((p[0] - o[0]) * ex[0] + (p[1] - o[1]) * ex[1]) / io;
  const v = (p, o = mid) => ((p[0] - o[0]) * ey[0] + (p[1] - o[1]) * ey[1]) / io;
  const nose = P(NOSE);
  const cr = P(CHEEK_R), cl = P(CHEEK_L);
  const dr = Math.hypot(nose[0] - cr[0], nose[1] - cr[1]), dl = Math.hypot(nose[0] - cl[0], nose[1] - cl[1]);
  const sl = sh(SH_L), sr = sh(SH_R);
  const shMid = avg(sl, sr);
  let sw = Math.hypot(sl[0] - sr[0], sl[1] - sr[1]);
  if (!(sw > 1e-6)) sw = NaN;
  const g = [
    Math.atan2(ex[1], ex[0]), u(nose), (dr - dl) / (dr + dl), v(nose), v(P(CHIN)),
    (nose[0] - shMid[0]) / sw, (nose[1] - shMid[1]) / sw, io / sw,
  ];
  for (let k = 0; k < 3; k++) g.push(-v(P(BROW_R[k]), er), -v(P(BROW_L[k]), el));
  g.push(v(P(LID_R[1]), P(LID_R[0])), v(P(LID_L[1]), P(LID_L[0])));
  const mc0 = P(MOUTH_C[0]), mc1 = P(MOUTH_C[1]);
  g.push(v(P(LIP_IN[1]), P(LIP_IN[0])), Math.hypot(mc0[0] - mc1[0], mc0[1] - mc1[1]) / io,
    v(avg(mc0, mc1), avg(P(LIP_OUT[0]), P(LIP_OUT[1]))));
  let bad = !Number.isFinite(face[fo]);
  for (let i = 0; i < N_FRAME; i++) {
    const x = i < N_GEOM ? g[i] : blend[bo + i - N_GEOM + 1];
    if (!Number.isFinite(x)) bad = true;
    out[i] = x;
  }
  if (bad) out.fill(NaN);
  return out;
}

/** raw → T×N_FRAME Float32Array (frame_features). */
export function frameFeatures(raw, aspect) {
  const x = new Float32Array(raw.T * N_FRAME), row = new Float32Array(N_FRAME);
  for (let t = 0; t < raw.T; t++) {
    frameRow(raw.face, raw.pose, raw.blend, aspect, row, t * 204, t * 27, t * 52);
    x.set(row, t * N_FRAME);
  }
  return x;
}

/** Bir kadrda əl qalxıbmı: görünür və biləyi çiyin xəttindən bir çiyin enindən az aşağıdadır (hands_up). */
export function handUp(hand, pose, aspect, ho = 0, po = 0) {
  const sy = (pose[po + SH_L * 3 + 1] + pose[po + SH_R * 3 + 1]) / 2;
  const sw = Math.hypot((pose[po + SH_L * 3] - pose[po + SH_R * 3]) * aspect, pose[po + SH_L * 3 + 1] - pose[po + SH_R * 3 + 1]);
  const ok = Number.isFinite(sy) && Number.isFinite(sw) && sw > 1e-6;
  for (let s = 0; s < 2; s++) {
    const wx = hand[ho + s * 63], wy = hand[ho + s * 63 + 1];
    if (!Number.isFinite(wx)) continue;
    if (!ok || wy - sy < sw) return true;
  }
  return false;
}

export function handsUp(raw, aspect) {
  const up = new Uint8Array(raw.T);
  for (let t = 0; t < raw.T; t++) up[t] = handUp(raw.hand, raw.pose, aspect, t * 126, t * 27) ? 1 : 0;
  return up;
}

/** Əllərin ilk qalxdığı və son endiyi kadr: [a, b) (active_range). */
export function activeRange(up, pad = 3) {
  let first = -1, last = -1;
  for (let t = 0; t < up.length; t++) if (up[t]) { if (first < 0) first = t; last = t; }
  if (first < 0) return [0, up.length];
  return [Math.max(0, first - pad), Math.min(up.length, last + 1 + pad)];
}

function median(vals) {
  vals.sort((a, b) => a - b);
  const n = vals.length, h = n >> 1;
  return n % 2 ? vals[h] : (vals[h - 1] + vals[h]) / 2;
}

/** Neytral üz (N_FRAME): 'rest' — əllər aşağıdakı kadrların medianı (min_rest-dən azdırsa bütün klip), 'clip', 'none'. */
export function neutral(x, up, mode = 'rest', minRest = 5) {
  const T = up.length, out = new Float32Array(N_FRAME);
  if (mode === 'none') return out;
  const ok = [], rest = [];
  for (let t = 0; t < T; t++) if (Number.isFinite(x[t * N_FRAME])) { ok.push(t); if (!up[t]) rest.push(t); }
  const sel = mode === 'rest' && rest.length >= minRest ? rest : ok;
  if (!sel.length) return out;
  for (let i = 0; i < N_FRAME; i++) out[i] = median(sel.map((t) => x[t * N_FRAME + i]));
  return out;
}

/** T×N_FRAME + neytral → cümlənin vektoru (N_CLIP), clip_features ilə eyni. */
export function clipFeatures(x, up, base) {
  const [a, b] = activeRange(up);
  const out = new Float32Array(N_CLIP);
  const rows = [];
  for (let t = a; t < b; t++) if (Number.isFinite(x[t * N_FRAME])) rows.push(t);
  const L = rows.length;
  if (L < 3) return out;
  const fr = (v) => Math.fround(v);
  const k = Math.max(1, Math.floor(L / 3));
  for (let i = 0; i < N_FRAME; i++) {
    const s = rows.map((t) => fr(x[t * N_FRAME + i] - base[i]));
    let sum = 0, mx = -Infinity, mn = Infinity, last = 0, first = 0;
    for (let j = 0; j < L; j++) { sum += s[j]; if (s[j] > mx) mx = s[j]; if (s[j] < mn) mn = s[j]; }
    const mean = sum / L;
    let ss = 0;
    for (let j = 0; j < L; j++) ss += (s[j] - mean) ** 2;
    for (let j = L - k; j < L; j++) last += s[j];
    for (let j = 0; j < k; j++) first += s[j];
    const st = [mean, Math.sqrt(ss / L), mx, mn, last / k, first / k];
    st.forEach((v, q) => { out[q * N_FRAME + i] = v; });
  }
  // başın hərəkəti: hər iki kadrı görünən ardıcıl kadr cütləri arası fərq (aktiv aralıqda)
  const dv = [];
  for (let t = a + 1; t < b; t++) {
    const row = MOTION.map((c) => fr(x[t * N_FRAME + c] - base[c]) - fr(x[(t - 1) * N_FRAME + c] - base[c]));
    if (row.every(Number.isFinite)) dv.push(row);
  }
  const o = N_FRAME * STATS.length;
  if (dv.length >= 2) {
    for (let c = 0; c < 3; c++) {
      let sp = 0, flips = 0;
      for (let j = 0; j < dv.length; j++) sp += Math.abs(dv[j][c]);
      for (let j = 1; j < dv.length; j++) flips += Math.sign(dv[j][c]) * Math.sign(dv[j - 1][c]) < 0;
      out[o + c] = (sp / dv.length) * FPS;
      out[o + 3 + c] = flips / (dv.length - 1);
    }
  }
  out[o + 6] = L / (b - a);
  return out;
}

/** Bir başlığın ehtimalı (standartlaşdırma çəkilərə qatlanıb). */
export function head(h, f) {
  let z = h.bias;
  for (let i = 0; i < f.length; i++) z += h.coef[i] * f[i];
  return 1 / (1 + Math.exp(-z));
}

/**
 * Bir klip → { yn, neg, face_rate } (üz min_face-dən az görünürsə yn, neg null). fps — klipin kadr tezliyi.
 * @param {object} model nonmanual.json
 */
export function predict(model, raw, aspect, fps = FPS) {
  const r = toFps(raw, fps);
  const x = frameFeatures(r, aspect), up = handsUp(r, aspect);
  const f = clipFeatures(x, up, neutral(x, up, model.neutral, model.min_rest));
  const faceRate = f[N_CLIP - 1], seen = faceRate >= model.min_face;
  const out = { face_rate: faceRate };
  for (const [k, h] of Object.entries(model.heads)) out[k] = seen ? head(h, f) : null;
  return out;
}

/**
 * Cümlənin işarələri + üzün ehtimalları → { question: 'wh' | 'yn' | null, neg: bool } — gloss2text-in nonmanual
 * seçimi. Sual sözü işarəsi (model.wh_labels) varsa 'wh'; yoxsa üz yn ≥ hədd → 'yn'. İnkar: üz neg ≥ hədd.
 */
export function marks(model, labels, p) {
  const wh = new Set(model.wh_labels.map((l) => l.replace(/\?$/, '')));
  const isWh = labels.some((l) => wh.has(String(l).split('/')[0].toLocaleUpperCase('az').replace(/\?$/, '')));
  const yn = p?.yn != null && p.yn >= model.heads.yn.t;
  return { question: isWh ? 'wh' : yn ? 'yn' : null, neg: p?.neg != null && p.neg >= model.heads.neg.t };
}

/**
 * Canlı axın üçün: 30 fps kadrlar push() ilə gəlir (landmarks.js kadrları), cümlə bağlananda sentence() son
 * cümlənin (əvvəlki sentence()-dən bəri) ehtimallarını verir. Bufer maxFrames-dən uzundursa köhnə kadrlar düşür.
 */
export class NonManualTracker {
  constructor(model, { maxFrames = 900 } = {}) {
    this.model = model;
    this.max = maxFrames;
    this.reset();
  }

  reset() { this.frames = []; }

  push(frame, aspect) {
    this.aspect = aspect;
    this.frames.push({ face: frame.face, pose: frame.pose, blend: frame.blend, hand: frame.hand });
    if (this.frames.length > this.max) this.frames.shift();
  }

  /** → { yn, neg, face_rate } və bufer boşalır. */
  sentence() {
    const fr = this.frames, T = fr.length;
    const raw = { T };
    for (const [k, n] of Object.entries({ hand: 126, pose: 27, face: 204, blend: 52 })) {
      const a = new Float32Array(T * n);
      fr.forEach((f, t) => a.set(f[k], t * n));
      raw[k] = a;
    }
    this.reset();
    return predict(this.model, raw, this.aspect ?? 4 / 3);
  }
}
