/**
 * Söz modeli üçün landmark-lar: iki əl + poza + üz — ml/azsl/landmarks.py-ın brauzer əkizi.
 *
 * Modellər təlimdəkilərlə eynidir (public/tercume/assets/*.task, md5 ml/azsl/landmarks.py-da):
 * hand_landmarker.task (barmaq əlifbası da bunu işlədir), pose_landmarker_lite.task,
 * face_landmarker.task. Ayarlar da eynidir: VIDEO rejimi, 2 əl, 0.5 hədlər, 1 poza, 1 üz + blendshape.
 *
 * Hər kadr features.js-in gözlədiyi formadadır (NaN = aşkarlanmayıb):
 *   hand 2×21×3 (slot 0 = siqnalçının SAĞ əli) · hand_world 2×21×3 · pose 9×3 (x, y, z) ·
 *   face 102×2 · blend 52 (BLEND_NAMES sırası)
 * Sağ/sol əli pozanın biləkləri təyin edir (assignHands — Python-dakı qayda ilə eyni), ona görə
 * əllər çarpazlaşanda da slot dəyişmir. Üzü söndürmək (face: false) məxfilik rejimidir: üz
 * modeli yüklənmir, üz sahələri NaN qalır.
 */
import { ASSET_BASE, WASM_BASE } from './paths.js';
import { POSE_IDX, FACE_IDX, BLEND_NAMES } from './features.js';

export const MODEL_FILES = {
  hand: 'hand_landmarker.task',
  pose: 'pose_landmarker_lite.task',
  face: 'face_landmarker.task',
};
export const FRAME_SIZES = { hand: 126, hand_world: 126, pose: 27, face: 204, blend: 52 };

const P_L_WR = 15, P_R_WR = 16;            // MediaPipe Pose: sol və sağ bilək
const BLEND_INDEX = new Map(BLEND_NAMES.map((n, i) => [n, i]));

/**
 * Üç landmarker (üz istəyə bağlı). GPU alınmasa CPU ilə davam edir — hands.js-dəki kimi.
 * @returns {Promise<{hand, pose, face, close: () => void}>}
 */
export async function createLandmarkers({ face = true, wasmPath = WASM_BASE, assetBase = ASSET_BASE } = {}) {
  const vision = await import('@mediapipe/tasks-vision');
  const fileset = await vision.FilesetResolver.forVisionTasks(wasmPath);
  const make = async (Task, file, extra) => {
    const options = (delegate) => ({
      baseOptions: { modelAssetPath: `${assetBase}${file}`, delegate },
      runningMode: 'VIDEO',
      ...extra,
    });
    try {
      return await Task.createFromOptions(fileset, options('GPU'));
    } catch (err) {
      console.warn(`MediaPipe GPU alınmadı (${file}), CPU-ya keçilir:`, err);
      return Task.createFromOptions(fileset, options('CPU'));
    }
  };
  const lms = {
    hand: await make(vision.HandLandmarker, MODEL_FILES.hand, {
      numHands: 2, minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    }),
    pose: await make(vision.PoseLandmarker, MODEL_FILES.pose, { numPoses: 1 }),
    face: face ? await make(vision.FaceLandmarker, MODEL_FILES.face, { numFaces: 1, outputFaceBlendshapes: true }) : null,
  };
  lms.close = () => { for (const k of ['hand', 'pose', 'face']) lms[k]?.close(); };
  return lms;
}

/** Bir kadr: üç model + kadrın forması. `ts` — artan zaman damğası (ms). */
export function detectFrame(lms, image, ts) {
  const W = image.videoWidth || image.width, H = image.videoHeight || image.height;
  const hr = lms.hand.detectForVideo(image, ts);
  const pr = lms.pose.detectForVideo(image, ts);
  const fr = lms.face ? lms.face.detectForVideo(image, ts) : null;
  return frameFromResults(hr, pr, fr, W / H);
}

export function emptyFrame() {
  const f = {};
  for (const [k, n] of Object.entries(FRAME_SIZES)) f[k] = new Float32Array(n).fill(NaN);
  return f;
}

/** MediaPipe nəticələri → təlimdəki formada bir kadr. */
export function frameFromResults(hr, pr, fr, aspect) {
  const f = emptyFrame();
  let wrists = null;
  const pl = pr?.landmarks?.[0];
  if (pl) {
    POSE_IDX.forEach((j, k) => {
      f.pose[k * 3] = pl[j].x; f.pose[k * 3 + 1] = pl[j].y; f.pose[k * 3 + 2] = pl[j].z;
    });
    wrists = [[pl[P_L_WR].x, pl[P_L_WR].y], [pl[P_R_WR].x, pl[P_R_WR].y]];
  }
  const hands = hr?.landmarks ?? [];
  const slots = assignHands(hands.map((h) => [h[0].x, h[0].y]), wrists, aspect);
  slots.forEach((k, s) => {
    if (k === null) return;
    const img = hands[k], world = hr.worldLandmarks[k];
    for (let j = 0; j < 21; j++) {
      const o = s * 63 + j * 3;
      f.hand[o] = img[j].x; f.hand[o + 1] = img[j].y; f.hand[o + 2] = img[j].z;
      f.hand_world[o] = world[j].x; f.hand_world[o + 1] = world[j].y; f.hand_world[o + 2] = world[j].z;
    }
  });
  const fl = fr?.faceLandmarks?.[0];
  if (fl) {
    FACE_IDX.forEach((j, k) => { f.face[k * 2] = fl[j].x; f.face[k * 2 + 1] = fl[j].y; });
    for (const c of fr.faceBlendshapes?.[0]?.categories ?? []) {
      const i = BLEND_INDEX.get(c.categoryName);
      if (i !== undefined) f.blend[i] = c.score;
    }
  }
  return f;
}

/**
 * Aşkarlanan əllər → [sağ slot, sol slot] (əllərin indeksi və ya null). ml/azsl/landmarks.py
 * assign_hands ilə eyni: biləyə yaxınlıq poza ilə; poza yoxdursa siqnalçının sağ əli kadrın
 * solundadır (kamera güzgülənməyib).
 * @param {number[][]} wrists hər aşkarlanan əlin bilək nöqtəsi [x, y]
 * @param {number[][]|null} poseWrists [[solX, solY], [sağX, sağY]]
 */
export function assignHands(wrists, poseWrists, aspect) {
  const slots = [null, null];
  if (!wrists.length) return slots;
  if (!poseWrists) {
    const order = wrists.map((_, k) => k).sort((a, b) => wrists[a][0] - wrists[b][0]).slice(0, 2);
    order.forEach((k, s) => { slots[s] = k; });
    return slots;
  }
  const [[lx, ly], [rx, ry]] = poseWrists;
  const d = (k, px, py) => Math.hypot((wrists[k][0] - px) * aspect, wrists[k][1] - py);
  if (wrists.length >= 2) {
    return d(0, rx, ry) + d(1, lx, ly) <= d(0, lx, ly) + d(1, rx, ry) ? [0, 1] : [1, 0];
  }
  slots[d(0, rx, ry) <= d(0, lx, ly) ? 0 : 1] = 0;
  return slots;
}

/** Son kadrların buferi; istənilən aralığı features.preprocess()-in girişi kimi verir. */
export class LandmarkClip {
  constructor(maxFrames = 300) {
    this.max = maxFrames;
    this.frames = [];
  }

  get length() { return this.frames.length; }

  push(frame) {
    this.frames.push(frame);
    if (this.frames.length > this.max) this.frames.shift();
  }

  reset() { this.frames = []; }

  /** [from, to) kadrları → { T, hand, hand_world, pose, face, blend } */
  toRaw(from = 0, to = this.frames.length) {
    const fr = this.frames.slice(from, to);
    const raw = { T: fr.length };
    for (const [k, n] of Object.entries(FRAME_SIZES)) {
      const a = new Float32Array(fr.length * n);
      fr.forEach((f, t) => a.set(f[k], t * n));
      raw[k] = a;
    }
    return raw;
  }
}
