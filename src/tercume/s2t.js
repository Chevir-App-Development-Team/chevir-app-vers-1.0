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
 *
 * Model iki cürdür. Köhnə model (model.js, FingerspellModel) 20 kadrın xam əl koordinatlarını
 * oxuyur. Landmark modeli (letters.js, LetterModel, notebook 07) əlin formasını və hərəkətini oxuyur,
 * '<none>' (hərf yoxdur) deyə bilir və hərəkətli hərfləri (ç, d, g, k, ö, ü, y, z) də tanıyır: onlar
 * üçün "əl sabit olmalıdır" şərti tətbiq olunmur, eyni hərfin təkrarı isə proqnoz dəyişəndə açılır.
 */
import { createHandLandmarker, extractKeypoints, FrameWindow } from './hands.js';
import { frameFromResults, LandmarkClip } from './landmarks.js';

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

/** Landmark modeli üçün: '<none>' sinfi keçidləri özü tutur, ona görə əminlik həddi bir az yuxarıdır. */
export const LETTER_GATE = { ...DEFAULT_GATE, minConfidence: 0.5 };

export class SignToText {
  constructor({ model, decoder, video, onFrame, onPrediction, onDecode, onStatus } = {}) {
    this.model = model;
    this.decoder = decoder;
    this.video = video;
    this.onFrame = onFrame;
    this.onPrediction = onPrediction;
    this.onDecode = onDecode;
    this.onStatus = onStatus;

    this.landmarkModel = model?.kind === 'letters';
    this.gate = { ...(this.landmarkModel ? LETTER_GATE : DEFAULT_GATE) };
    this.win = new FrameWindow({ take: 20 });
    this.clip = this.landmarkModel ? new LandmarkClip(model.window + 8) : null;
    this.pending = null;      // landmark modelinin gözlənilən proqnozu (onnxruntime asinxrondur)
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

  /**
   * Yüklənmiş videonu kadr-kadr emal edir: real vaxtda oynatmaq əvəzinə hər 1/fps
   * saniyəyə keçilir, ona görə zəif cihazda da heç bir kadr buraxılmır və nəticə
   * kameradakı ilə eyni kadr tezliyindədir. Video bitəndə yarımçıq söz tamamlanır.
   * @param {string} url video faylının ünvanı (məs. URL.createObjectURL(file))
   */
  async startFile(url, { fps = 30 } = {}) {
    if (!this.landmarker) await this.init();
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
    this.running = true;
    this.onStatus?.({ stage: 'running', message: 'Video emal olunur' });
    for (let t = 0; t < v.duration; t += 1 / fps) {
      v.currentTime = t;
      await new Promise((resolve) => v.addEventListener('seeked', resolve, { once: true }));
      if (this._file !== token) return;               // dayandırıldı
      this.#process(v, { fileProgress: Math.min(t / v.duration, 1) });
    }
    this._file = null;
    this.running = false;
    this.finish();
    this.onStatus?.({ stage: 'ended', message: 'Video bitdi' });
  }

  stop() {
    this.running = false;
    this._file = null;
    const s = this.video.srcObject;
    if (s) s.getTracks().forEach((t) => t.stop());
    this.video.srcObject = null;
    if (this.video.src) {
      this.video.pause();
      this.video.removeAttribute('src');
      this.video.load();
    }
    this.win.reset();
    this.clip?.reset();
    this.#resetGate();
    this.onStatus?.({ stage: 'stopped', message: 'Dayandırıldı' });
  }

  clear() {
    this.steps = [];
    this.letters = [];
    this.win.reset();
    this.clip?.reset();
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
   * @param {object|null} frame landmark modeli üçün kadr (landmarks.js frameFromResults; bədənsiz)
   * @param {number} aspect kadrın eni / hündürlüyü
   */
  pushFrame(keypoints, frame = null, aspect = 4 / 3) {
    this.win.push(keypoints);
    if (this.clip && frame) this.clip.push(frame);
    this.frameNo++;

    const { gate } = this;
    const presence = this.win.presence;
    const motion = this.win.motion;

    // Əl pozadan çıxdı (tərpəndi və ya kadrdan getdi) → eyni hərf yenidən yazıla bilər.
    // Hərəkətli hərf özü tərpənir: onun təkrarını yalnız əlin getməsi və ya başqa proqnoz açır.
    if (presence < 1 || (motion > gate.releaseMotion && !this.#dynamic(this.locked))) this.locked = null;

    if (!this.win.full || this.frameNo % gate.stride) return this.#status();

    if (presence < gate.minPresence) {
      this.state = 'wait-hand';
      this.dwellLabel = null; this.dwellCount = 0;
      return this.#status();
    }
    if (this.landmarkModel) {
      if (!this.pending && this.clip.length) {
        const raw = this.clip.toRaw(Math.max(0, this.clip.length - this.model.window));
        this.pending = this.model.predictRaw(raw, aspect)
          .then((probs) => this.#decide(this.model.topK(probs, Math.max(this.beamWidth, 3) + 1), presence, motion))
          .catch((err) => console.warn('hərf proqnozu alınmadı:', err))
          .finally(() => { this.pending = null; });
      }
      return this.#status();
    }
    if (motion > gate.maxMotion) {
      this.state = 'moving';
      this.dwellLabel = null; this.dwellCount = 0;
      return this.#status();
    }
    this.#decide(this.model.topK(this.model.predict(this.win.flat()), Math.max(this.beamWidth, 3)), presence, motion);
    return this.#status();
  }

  #dynamic(label) {
    return !!label && this.landmarkModel && this.model.dynamic.has(label);
  }

  /** Proqnoz → şərtlər (əminlik, sabitlik, təsdiq) → hərf. */
  #decide(top, presence, motion) {
    const { gate } = this;
    const none = this.landmarkModel ? this.model.none : null;
    const letters = top.filter((t) => t.label !== none);
    this.onPrediction?.({ top: letters, quality: presence, motion });
    const best = top[0];
    if (!best || best.label === none || best.p < gate.minConfidence) {
      if (best?.label === none) this.locked = null;      // hərf bitdi
      this.state = 'unsure';
      this.dwellLabel = null; this.dwellCount = 0;
      return;
    }
    if (this.landmarkModel) {
      if (!this.#dynamic(best.label) && motion > gate.maxMotion) {
        this.state = 'moving';
        this.dwellLabel = null; this.dwellCount = 0;
        return;
      }
      if (this.locked && best.label !== this.locked) this.locked = null;   // başqa hərf: əvvəlki bitib
    }

    if (best.label === this.dwellLabel) this.dwellCount++;
    else { this.dwellLabel = best.label; this.dwellCount = 1; }
    this.state = 'hold';

    if (this.locked !== best.label && this.dwellCount >= gate.dwell) {
      this.locked = best.label;
      this.state = 'typed';
      this.#handle(letters);
    }
  }

  #tick = () => {
    if (!this.running) return;
    this._raf = requestAnimationFrame(this.#tick);
    const v = this.video;
    if (v.readyState < 2 || v.currentTime === this._lastVideoTime) return;
    this._lastVideoTime = v.currentTime;
    this.#process(v);
  };

  /** Bir video kadrı: əl landmark-ları → pəncərə → (şərtlər ödənsə) hərf. */
  #process(v, extra = {}) {
    const result = this.landmarker.detectForVideo(v, performance.now());
    const landmarks = result?.landmarks?.[0] ?? null;
    const aspect = (v.videoWidth || 4) / (v.videoHeight || 3);
    const frame = this.landmarkModel ? frameFromResults(result, null, null, aspect) : null;
    const status = this.pushFrame(extractKeypoints(result), frame, aspect);
    this.onFrame?.({ landmarks, ...status, ...extra });
  }

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
