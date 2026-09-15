/**
 * MediaPipe əl aşkarlanması + müəllimin boru xətti ilə eyni xüsusiyyət çıxarma.
 *
 * Vacib uyğunluq detalları (Beam Search notebook-undan):
 *   - YALNIZ birinci aşkarlanan əl işlədilir (multi_hand_landmarks[0])
 *   - normallaşdırılmamış xam landmark-lar: x, y ∈ [0,1], z nisbi → 21×3 = 63
 *   - əl görünməyəndə 63 sıfır
 *   - 30 kadr yığılır, SON 20-si modelə verilir, sonra bufer sıfırlanır
 *
 * Güzgü qeydi: təlim kadrları güzgülənməmişdir. Video elementi CSS ilə
 * güzgülənir (yalnız görüntü), aşkarlama isə xam kadr üzərində işləyir —
 * ona görə koordinatlar təlimlə uyğun qalır.
 */

const HAND_LM = 21;
export const FEAT_PER_FRAME = HAND_LM * 3;   // 63

/** MediaPipe əl skeleti — barmaq zəncirləri (çəkmək üçün) */
export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],            // baş barmaq
  [0, 5], [5, 6], [6, 7], [7, 8],            // şəhadət
  [5, 9], [9, 10], [10, 11], [11, 12],       // orta
  [9, 13], [13, 14], [14, 15], [15, 16],     // üzük
  [13, 17], [17, 18], [18, 19], [19, 20],    // çeçələ
  [0, 17],                                    // avuc bağlantısı
];

export const FINGERS = {
  thumb: [1, 2, 3, 4],
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  little: [17, 18, 19, 20],
};

/**
 * Yollar bu faylın öz ünvanına görə açılır. `import()` nisbi yolu modulun
 * yerinə (/js/), `fetch` isə səhifənin yerinə görə açır — ikisini qarışdırmamaq
 * üçün hamısı mütləq URL-ə çevrilir. Sayt alt qovluqda yerləşsə də işləyir.
 */
const fromHere = (p) => new URL(p, import.meta.url).href;

export async function createHandLandmarker({
  visionPath = fromHere('../vendor/vision_bundle.mjs'),
  wasmPath = fromHere('../vendor/wasm'),
  modelPath = fromHere('../assets/hand_landmarker.task'),
  numHands = 2,
  minDetection = 0.5,
  minTracking = 0.5,
} = {}) {
  const vision = await import(visionPath);
  const fileset = await vision.FilesetResolver.forVisionTasks(wasmPath);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: modelPath, delegate },
    runningMode: 'VIDEO',
    numHands,
    minHandDetectionConfidence: minDetection,
    minTrackingConfidence: minTracking,
  });
  try {
    return await vision.HandLandmarker.createFromOptions(fileset, options('GPU'));
  } catch (err) {
    // WebGL2 olmayan və ya GPU-su bloklanmış brauzerlərdə CPU ilə davam et
    console.warn('MediaPipe GPU alınmadı, CPU-ya keçilir:', err);
    return vision.HandLandmarker.createFromOptions(fileset, options('CPU'));
  }
}

/** MediaPipe nəticəsindən 63-ölçülü xüsusiyyət vektoru (yoxsa null). */
export function extractKeypoints(result) {
  const lms = result?.landmarks;
  if (!lms || !lms.length) return null;
  const hand = lms[0];                       // notebook ilə eyni: birinci əl
  const out = new Float32Array(FEAT_PER_FRAME);
  for (let i = 0; i < HAND_LM; i++) {
    out[i * 3] = hand[i].x;
    out[i * 3 + 1] = hand[i].y;
    out[i * 3 + 2] = hand[i].z;
  }
  return out;
}

/**
 * Kadr buferi: `fill` kadr yığır, sonuncu `take`-ni düz vektor kimi verir.
 * Müəllimin qurğusu: fill=30, take=20.
 */
export class FrameBuffer {
  constructor({ fill = 30, take = 20 } = {}) {
    this.fill = fill;
    this.take = take;
    this.frames = [];
    this.handFrames = 0;
  }

  get progress() { return Math.min(this.frames.length / this.fill, 1); }
  /** Bufer nə qədər "canlıdır" — əl görünən kadrların payı */
  get quality() { return this.frames.length ? this.handFrames / this.frames.length : 0; }

  /** @returns {Float32Array|null} bufer dolubsa 20×63 düz vektor */
  push(keypoints) {
    this.frames.push(keypoints ?? new Float32Array(FEAT_PER_FRAME));
    if (keypoints) this.handFrames++;
    if (this.frames.length > this.fill) this.frames.shift();
    if (this.frames.length < this.fill) return null;

    const window = this.frames.slice(-this.take);
    const flat = new Float32Array(this.take * FEAT_PER_FRAME);
    window.forEach((f, i) => flat.set(f, i * FEAT_PER_FRAME));
    this.reset();
    return flat;
  }

  reset() { this.frames = []; this.handFrames = 0; }
}
