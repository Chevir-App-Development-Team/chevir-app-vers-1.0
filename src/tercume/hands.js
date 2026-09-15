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

import { ASSET_BASE, WASM_BASE } from './paths.js';

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
 * MediaPipe kitabxanası kamera açılanda dinamik yüklənir (ilk açılış yüngül qalsın).
 * Paket versiyası 0.10.14-ə bağlıdır: public/tercume/wasm faylları həmin versiyanın
 * wasm-ı ilə eyni olmalıdır.
 */
export async function createHandLandmarker({
  wasmPath = WASM_BASE,
  modelPath = `${ASSET_BASE}hand_landmarker.task`,
  numHands = 2,
  minDetection = 0.5,
  minTracking = 0.5,
} = {}) {
  const vision = await import('@mediapipe/tasks-vision');
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
 * Sürüşən kadr pəncərəsi: son `take` kadrı saxlayır və modelə verir.
 *
 * Əvvəlki qurğu 30 kadr yığıb ŞƏRTSİZ proqnoz verirdi, ona görə əlin kadra
 * girdiyi və ya bir pozadan digərinə keçdiyi anlar da hərf kimi yazılırdı
 * (ölçülüb: 5 boş kadr + 15 kadr "s" → model "ö" deyir, özü də 0.89 ilə).
 * İndi pəncərə sürüşür, `presence` və `motion` isə hərfin nə vaxt yazıla
 * biləcəyini müəyyən edir (bax s2t.js).
 */
export class FrameWindow {
  constructor({ take = 20 } = {}) {
    this.take = take;
    this.frames = [];
    this.hands = [];
  }

  get full() { return this.frames.length >= this.take; }
  get fill() { return Math.min(this.frames.length / this.take, 1); }
  /** Pəncərədə əl görünən kadrların payı (1 = bütün kadrlarda əl var). */
  get presence() {
    if (!this.hands.length) return 0;
    return this.hands.reduce((n, h) => n + (h ? 1 : 0), 0) / this.hands.length;
  }

  /**
   * Hərəkət ölçüsü: ardıcıl kadrlar arası orta yerdəyişmə, əl ölçüsünə bölünüb.
   * Ölçülüb: sabit poza 0.00–0.04, pozadan pozaya keçid 0.06–0.11.
   */
  get motion() {
    if (this.frames.length < 2) return 0;
    let sum = 0;
    for (let i = 1; i < this.frames.length; i++) {
      const a = this.frames[i - 1], b = this.frames[i];
      const size = Math.hypot(b[27] - b[0], b[28] - b[1]) || 1;   // bilək → orta barmaq dibi
      let d = 0;
      for (let j = 0; j < HAND_LM; j++) {
        d += Math.hypot(b[j * 3] - a[j * 3], b[j * 3 + 1] - a[j * 3 + 1], b[j * 3 + 2] - a[j * 3 + 2]);
      }
      sum += d / HAND_LM / size;
    }
    return sum / (this.frames.length - 1);
  }

  push(keypoints) {
    this.frames.push(keypoints ?? new Float32Array(FEAT_PER_FRAME));
    this.hands.push(!!keypoints);
    if (this.frames.length > this.take) { this.frames.shift(); this.hands.shift(); }
  }

  /** Modelə veriləcək düz vektor (take × 63). */
  flat() {
    const out = new Float32Array(this.take * FEAT_PER_FRAME);
    this.frames.forEach((f, i) => out.set(f, i * FEAT_PER_FRAME));
    return out;
  }

  reset() { this.frames = []; this.hands = []; }
}
