/**
 * İşarə → mətn: veb-kamera axını, barmaq əlifbası proqnozu, beam dekoder.
 *
 * Müəllimin notebook-undaki idarə məntiqi saxlanılır:
 *   - hər proqnozda ən yaxşı `width` hərf beam-ə əlavə olunur
 *   - `backspace` sonuncu hərfi silir
 *   - `enter` yığılmış addımları dekod edib sözü verir
 *   - boşluq sinfi söz sərhədi kimi işlənir
 *
 * Əlavə: aşkarlanan landmark-lar canlı olaraq cyber skeletə də verilir,
 * beləcə modelin gördüyü şey ekranda görünür.
 */
import { createHandLandmarker, extractKeypoints, FrameBuffer } from './hands.js';

export class SignToText {
  constructor({ model, decoder, video, onFrame, onPrediction, onDecode, onStatus } = {}) {
    this.model = model;
    this.decoder = decoder;
    this.video = video;
    this.onFrame = onFrame;
    this.onPrediction = onPrediction;
    this.onDecode = onDecode;
    this.onStatus = onStatus;

    this.buffer = new FrameBuffer({ fill: 30, take: 20 });
    this.steps = [];          // [{label,p}[]] — beam üçün toplanan proqnozlar
    this.letters = [];        // qəbul edilmiş hərflər
    this.beamWidth = 3;
    this.running = false;
    this.landmarker = null;
    this._lastVideoTime = -1;
    this.decodeOpts = { mode: 'soft', width: 8, alpha: 1.0, beta: 2.0 };
  }

  async init() {
    this.onStatus?.({ stage: 'landmarker', message: 'MediaPipe yüklənir…' });
    this.landmarker = await createHandLandmarker();
    this.onStatus?.({ stage: 'ready', message: 'Hazır' });
  }

  async start() {
    if (!this.landmarker) await this.init();
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    this.video.srcObject = stream;
    await this.video.play();
    this.running = true;
    this.onStatus?.({ stage: 'running', message: 'Kamera işləyir' });
    this.#tick();
  }

  stop() {
    this.running = false;
    const s = this.video.srcObject;
    if (s) s.getTracks().forEach((t) => t.stop());
    this.video.srcObject = null;
    this.buffer.reset();
    this.onStatus?.({ stage: 'stopped', message: 'Dayandırıldı' });
  }

  clear() {
    this.steps = [];
    this.letters = [];
    this.buffer.reset();
    this.onDecode?.({ candidates: [], best: null, verified: null, suggestions: [], letters: [] });
  }

  #tick = () => {
    if (!this.running) return;
    this._raf = requestAnimationFrame(this.#tick);
    const v = this.video;
    if (v.readyState < 2 || v.currentTime === this._lastVideoTime) return;
    this._lastVideoTime = v.currentTime;

    const result = this.landmarker.detectForVideo(v, performance.now());
    const landmarks = result?.landmarks?.[0] ?? null;
    this.onFrame?.({ landmarks, progress: this.buffer.progress, quality: this.buffer.quality });

    const kp = extractKeypoints(result);
    const window = this.buffer.push(kp);
    if (!window) return;

    const probs = this.model.predict(window);
    const top = this.model.topK(probs, Math.max(this.beamWidth, 3));
    this.onPrediction?.({ top, quality: this.buffer.quality });
    this.#handle(top);
  };

  /** Proqnozu idarə siniflərinə görə tətbiq edir. */
  #handle(top) {
    const best = top[0];
    if (!best) return;

    if (best.label === 'backspace') {
      this.steps.pop();
      this.letters.pop();
      this.#decode();
      return;
    }
    if (best.label === 'enter') {
      this.#decode(true);
      return;
    }
    if (best.label === ' ') {
      if (this.steps.length) this.#decode(true);
      return;
    }
    // Adi hərf: beam addımı kimi saxlanılır
    this.steps.push(top.slice(0, this.beamWidth).map(({ label, p }) => ({ label, p })));
    this.letters.push(best.label);
    this.#decode();
  }

  #decode(final = false) {
    if (!this.steps.length) {
      this.onDecode?.({ candidates: [], best: null, verified: null, suggestions: [], letters: [], final });
      return;
    }
    const r = this.decoder.decode(this.steps, this.decodeOpts);
    this.onDecode?.({ ...r, letters: [...this.letters], final });
    if (final) { this.steps = []; this.letters = []; }
  }

  /** Klaviatura ilə əl ilə hərf əlavə etmə (demo/sınaq üçün). */
  pushLetter(ch, p = 0.9) {
    const others = this.decoder.letters.filter((c) => c !== ch).slice(0, 2);
    const rest = (1 - p) / Math.max(others.length, 1);
    this.steps.push([{ label: ch, p }, ...others.map((c) => ({ label: c, p: rest }))]);
    this.letters.push(ch);
    this.#decode();
  }

  /** Sonuncu hərfi geri alır. */
  undo() {
    this.steps.pop();
    this.letters.pop();
    this.#decode();
  }

  finish() { this.#decode(true); }
}
