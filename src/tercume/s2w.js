/**
 * Jest → söz (söz rejimi): kamera və ya video → landmark-lar (landmarks.js: iki əl, bədən, üz) →
 * aktivlik və aralıqlar (segment.js) → aralıq bağlananda pəncərələr söz modelinə (recognizer.js).
 *
 * Söz əllər istirahətə enəndən sonra çıxır: aralığın hamısı lazımdır, çünki fasiləsiz jestdə
 * işarələrin sərhədini model özü seçir. Barmaq əlifbası (s2t.js) ayrıca qalır.
 * Video və landmark-lar cihazdan çıxmır — model brauzerdə işləyir.
 */
import { createLandmarkers, detectFrame, LandmarkClip, FRAME_SIZES } from './landmarks.js';
import { LocalRecognizer } from './recognizer.js';
import { SEGMENT, SignSegmenter, frameActive, segmentFromManifest, segmentSpan } from './segment.js';

/**
 * Üz izləyicisi lazımdırmı? Model üzsüz işləməyə öyrədilibsə yox; əks halda manifest-dəki
 * ölçməyə baxılır: validasiyada üzlə və üzsüz macro-F1 fərqi `maxLoss`-dan kiçikdirsə, izləyici
 * yüklənmir (hər kadrda üç modeldən biri az işləyir). Ölçmə yoxdursa — ehtiyat üçün lazımdır.
 */
export function needsFace(manifest, maxLoss = 0.01) {
  if (manifest.works_without_face) return false;
  const withFace = manifest.metrics?.val_words?.macro_f1;
  const without = manifest.metrics?.val_words_noface?.macro_f1;
  return !(Number.isFinite(withFace) && Number.isFinite(without) && withFace - without < maxLoss);
}

export class SignToWords {
  /**
   * @param {object} o
   * @param {HTMLVideoElement} o.video
   * @param {(e: {frame, active, state, aspect, fileProgress?}) => void} [o.onFrame] hər kadr
   * @param {(e: {signs, added, span, ms}) => void} [o.onSigns] aralıq tanınanda (unsure: true —
   *   heç bir söz əminlik həddini keçmədi, ən yaxşı təxmin; unknown: true — model bunu lüğətdən
   *   kənar işarə sayır)
   * @param {(e: {stage, message}) => void} [o.onStatus]
   * @param {object} [o.segment] segment.js SEGMENT-in dəyişdirilən sahələri
   */
  constructor({ video, onFrame, onSigns, onStatus, segment = {} } = {}) {
    this.video = video;
    this.onFrame = onFrame;
    this.onSigns = onSigns;
    this.onStatus = onStatus;
    this.userSegment = segment;
    this.p = { ...SEGMENT, ...segment };
    this.rec = null;
    this.lms = null;
    this.running = false;
    this.signs = [];
    this.lastTs = 0;                             // MediaPipe VIDEO rejimi: zaman damğası həmişə artmalıdır
    this.#reset();
  }

  async init() {
    this.onStatus?.({ stage: 'landmarker', message: 'Söz modeli yüklənir…' });
    this.rec = await LocalRecognizer.load();
    // modelin öz söz rejimi parametrləri (manifest.segment), sonra əl ilə verilənlər
    this.p = { ...SEGMENT, ...segmentFromManifest(this.rec.manifest.segment), ...this.userSegment };
    this.#reset();
    this.face = needsFace(this.rec.manifest);
    this.onStatus?.({ stage: 'landmarker', message: `MediaPipe yüklənir (əllər, bədən${this.face ? ', üz' : ''})…` });
    this.lms = await createLandmarkers({ face: this.face });
    this.onStatus?.({ stage: 'ready', message: `Hazır · ${this.rec.labels.length} söz` });
  }

  async start() {
    if (!this.lms) await this.init();
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    this.video.srcObject = stream;
    await this.video.play();
    this.#reset();
    this.running = true;
    this.onStatus?.({ stage: 'running', message: 'Kamera işləyir · sözü göstər, sonra əlləri endir' });
    this.#tick();
  }

  /** Yüklənmiş video kadr-kadr (s2t.js startFile kimi): zəif cihazda da kadr buraxılmır. */
  async startFile(url, { fps = 30 } = {}) {
    if (!this.lms) await this.init();
    const v = this.video;
    v.srcObject = null;
    v.muted = true;
    v.src = url;
    await new Promise((resolve, reject) => {
      v.onloadeddata = resolve;
      v.onerror = () => reject(new Error('video oxunmadı (format dəstəklənmir?)'));
    });
    const token = {};
    this._file = token;
    this.#reset();
    this.running = true;
    this.onStatus?.({ stage: 'running', message: 'Video emal olunur' });
    const n = Math.round(v.duration * fps);
    const t0 = this.lastTs + 1;                  // növbəti video da əvvəlkinin davamı kimi
    for (let i = 0; i < n; i++) {
      v.currentTime = (i + 0.5) / fps;
      await new Promise((resolve) => v.addEventListener('seeked', resolve, { once: true }));
      if (this._file !== token) return;
      await this.#process(v, t0 + Math.round((i * 1000) / fps), { fileProgress: (i + 1) / n });
    }
    const last = this.seg.flush();
    if (last) await this.#recognize(last);
    await this.busy;
    this._file = null;
    this.running = false;
    this.onStatus?.({ stage: 'ended', message: 'Video bitdi' });
  }

  stop() {
    this.running = false;
    this._file = null;
    cancelAnimationFrame(this._raf);
    const s = this.video.srcObject;
    if (s) s.getTracks().forEach((t) => t.stop());
    this.video.srcObject = null;
    if (this.video.src) {
      this.video.pause();
      this.video.removeAttribute('src');
      this.video.load();
    }
    this.#reset();
    this.onStatus?.({ stage: 'stopped', message: 'Dayandırıldı' });
  }

  clear() {
    this.signs = [];
    this.onSigns?.({ signs: [], added: [], span: null, ms: 0 });
  }

  #reset() {
    this.clip = new LandmarkClip(600);           // ən uzun aralıq + kənarlar (240 + 16) sığır
    this.seg = new SignSegmenter(this.p);
    this.busy = Promise.resolve();
    this.pending = 0;
    this._lastVideoTime = -1;
  }

  #tick = () => {
    if (!this.running || this._file) return;
    this._raf = requestAnimationFrame(this.#tick);
    const v = this.video;
    if (v.readyState < 2 || v.currentTime === this._lastVideoTime) return;
    this._lastVideoTime = v.currentTime;
    this.#process(v, performance.now());
  };

  /** Bir kadr: landmark-lar → bufer → aktivlik → (aralıq bağlandısa) tanıma. */
  #process(v, ts, extra = {}) {
    this.aspect = v.videoWidth / v.videoHeight;
    ts = Math.max(Math.round(ts), this.lastTs + 1);   // kameradan videoya, videodan videoya keçəndə də
    this.lastTs = ts;
    const frame = detectFrame(this.lms, v, ts);
    this.clip.push(frame);
    const active = frameActive(frame, this.aspect, this.p.restLine);
    const span = this.seg.push(active);
    const state = this.pending ? 'recognizing' : this.seg.open ? 'signing' : 'rest';
    this.onFrame?.({ frame, active, state, aspect: this.aspect, ...extra });
    return span ? this.#recognize(span) : undefined;
  }

  /** Aralığın kadrları indi köçürülür (bufer dəyişməyə davam edir), model növbə ilə işləyir. */
  #recognize(span) {
    const first = this.seg.n - this.clip.length;          // bufer-in ilk kadrının mütləq nömrəsi
    const raw = this.clip.toRaw(span.lo - first, span.hi - first);
    const aspect = this.aspect;
    this.pending++;
    const job = this.busy.then(async () => {
      const t0 = performance.now();
      const score = async (r) => {
        const x = await this.rec.recognize(r, aspect, { topK: 3 });
        return { label: x.top[0].label, p: x.top[0].p, top: x.top };
      };
      const { signs, best } = await segmentSpan(span, raw, score, FRAME_SIZES, this.p);
      // "naməlum" sinfi (manifest.unknown_label): seçimdə iştirak edir, amma söz kimi çıxmır
      const unk = this.rec.manifest.unknown_label;
      const words = signs.filter((s) => s.label !== unk);
      // heç bir söz kifayət qədər əmin deyilsə, ən yaxşı təxmin işarələnib göstərilir
      const added = words.length ? words
        : best ? [{ ...best, unsure: true, unknown: best.label === unk }] : [];
      this.signs.push(...added);
      this.onSigns?.({ signs: this.signs, added, span, ms: performance.now() - t0 });
    }).catch((err) => {
      console.error(err);
      this.onStatus?.({ stage: 'error', message: 'Tanıma xətası: ' + err.message });
    }).finally(() => { this.pending--; });
    this.busy = job;
    return job;
  }
}
