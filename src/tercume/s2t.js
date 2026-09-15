/**
 * Jest → mətn: veb-kamera axını, barmaq əlifbası proqnozu, beam dekoder.
 *
 * Dekod məntiqi dəyişməyib (top-3 → beam → hərf-bigram → leksikon), model və
 * xüsusiyyət çıxarma da eynidir. Dəyişən odur ki, hərfin NƏ VAXT yazılacağına
 * şərt qoyulur:
 *
 *   1. pəncərənin bütün kadrlarında əl görünməlidir — əl kadra yeni girəndə
 *      pəncərəyə boş kadrlar düşür və model onları dinamik hərf kimi oxuyur
 *      (ölçülüb: 5 boş + 15 "s" → "ö" 0.89 ilə — əminlik həddi bunu tutmur);
 *   2. əl sabit olmalıdır — pozadan pozaya keçid də yanlış hərf verir
 *      (sabit poza 0.00–0.04, keçid 0.06–0.11);
 *   3. model kifayət qədər əmin olmalıdır;
 *   4. eyni hərf ardıcıl bir neçə proqnozda təsdiqlənməlidir (titrəməyə qarşı);
 *   5. hərf yazıldıqdan sonra əl pozadan çıxana qədər təkrar yazılmır.
 *
 * Əlavə: aşkarlanan landmark-lar canlı olaraq cyber skeletə də verilir,
 * beləcə modelin gördüyü şey ekranda görünür.
 */
import { createHandLandmarker, extractKeypoints, FrameWindow } from './hands.js';

/** Hədlər ölçülərək seçilib (bax yuxarıdaki şərh); runtime-da dəyişdirilə bilər. */
export const DEFAULT_GATE = {
  stride: 4,            // hər neçə kadrdan bir proqnoz verilir
  minPresence: 1,       // pəncərədə əl görünən kadrların minimal payı
  maxMotion: 0.05,      // bundan yuxarı — əl hərəkətdədir, hərf yazılmır
  // top-1 ehtimalı: modelin özü bəzi hərflərdə alçaq qalır (ölçülüb: sabit "m" → 0.56),
  // ona görə hədd aşağıdır — keçidləri hərəkət və əl-yoxluğu şərtləri kəsir,
  // aşağı əminlikli hərfləri isə beam + leksikon düzəldir
  minConfidence: 0.45,
  dwell: 3,             // eyni hərf neçə ardıcıl proqnozda təsdiqlənməlidir
  releaseMotion: 0.06,  // eyni hərfin təkrarı üçün əl pozadan bu qədər çıxmalıdır
};

export class SignToText {
  constructor({ model, decoder, video, onFrame, onPrediction, onDecode, onStatus } = {}) {
    this.model = model;
    this.decoder = decoder;
    this.video = video;
    this.onFrame = onFrame;
    this.onPrediction = onPrediction;
    this.onDecode = onDecode;
    this.onStatus = onStatus;

    this.gate = { ...DEFAULT_GATE };
    this.win = new FrameWindow({ take: 20 });
    this.steps = [];          // [{label,p}[]] — beam üçün toplanan proqnozlar
    this.letters = [];        // qəbul edilmiş hərflər
    this.beamWidth = 3;
    this.running = false;
    this.landmarker = null;
    this._lastVideoTime = -1;
    this.decodeOpts = { mode: 'soft', width: 8, alpha: 1.0, beta: 2.0 };
    this.#resetGate();
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
    this.onStatus?.({ stage: 'running', message: 'Kamera işləyir · hərfi 1 saniyə sabit saxla' });
    this.#tick();
  }

  stop() {
    this.running = false;
    const s = this.video.srcObject;
    if (s) s.getTracks().forEach((t) => t.stop());
    this.video.srcObject = null;
    this.win.reset();
    this.#resetGate();
    this.onStatus?.({ stage: 'stopped', message: 'Dayandırıldı' });
  }

  clear() {
    this.steps = [];
    this.letters = [];
    this.win.reset();
    this.#resetGate();
    this.onDecode?.({ candidates: [], best: null, verified: null, suggestions: [], letters: [] });
  }

  #resetGate() {
    this.frameNo = 0;
    this.dwellLabel = null;
    this.dwellCount = 0;
    this.locked = null;      // yazılmış hərf: əl pozadan çıxana qədər təkrarlanmır
    this.state = 'idle';     // idle | wait-hand | moving | unsure | hold | typed
  }

  #status() {
    return {
      state: this.state,
      progress: this.state === 'hold' || this.state === 'typed'
        ? Math.min(this.dwellCount / this.gate.dwell, 1)
        : this.win.fill * 0,                       // saxlama yoxdursa halqa boşdur
      presence: this.win.presence,
      motion: this.win.motion,
    };
  }

  /**
   * Bir kadr əlavə edir və şərtlər ödənəndə hərfi yazır.
   * Kameradan asılı deyil — sınaqda süni ardıcıllıqla da çağırıla bilər.
   * @param {Float32Array|null} keypoints 63 ölçülü vektor (əl yoxdursa null)
   */
  pushFrame(keypoints) {
    this.win.push(keypoints);
    this.frameNo++;

    const { gate } = this;
    const presence = this.win.presence;
    const motion = this.win.motion;

    // Əl pozadan çıxdı (tərpəndi və ya kadrdan getdi) → eyni hərf yenidən yazıla bilər
    if (presence < 1 || motion > gate.releaseMotion) this.locked = null;

    if (!this.win.full || this.frameNo % gate.stride) return this.#status();

    if (presence < gate.minPresence) {
      this.state = 'wait-hand';
      this.dwellLabel = null; this.dwellCount = 0;
      return this.#status();
    }
    if (motion > gate.maxMotion) {
      this.state = 'moving';
      this.dwellLabel = null; this.dwellCount = 0;
      return this.#status();
    }

    const top = this.model.topK(this.model.predict(this.win.flat()), Math.max(this.beamWidth, 3));
    this.onPrediction?.({ top, quality: presence, motion });
    const best = top[0];
    if (!best || best.p < gate.minConfidence) {
      this.state = 'unsure';
      this.dwellLabel = null; this.dwellCount = 0;
      return this.#status();
    }

    if (best.label === this.dwellLabel) this.dwellCount++;
    else { this.dwellLabel = best.label; this.dwellCount = 1; }
    this.state = 'hold';

    if (this.locked !== best.label && this.dwellCount >= gate.dwell) {
      this.locked = best.label;
      this.state = 'typed';
      this.#handle(top);
    }
    return this.#status();
  }

  #tick = () => {
    if (!this.running) return;
    this._raf = requestAnimationFrame(this.#tick);
    const v = this.video;
    if (v.readyState < 2 || v.currentTime === this._lastVideoTime) return;
    this._lastVideoTime = v.currentTime;

    const result = this.landmarker.detectForVideo(v, performance.now());
    const landmarks = result?.landmarks?.[0] ?? null;
    const status = this.pushFrame(extractKeypoints(result));
    this.onFrame?.({ landmarks, ...status });
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
