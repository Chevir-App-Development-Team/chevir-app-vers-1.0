/**
 * Jest → cümlə: davamlı model (manifest.kind = 'continuous') landmark axınının hər kadrına etiket verir,
 * sözlər işarələnərkən, fasiləsiz çıxır. Uzun fasilə cümləni bağlayır; bir sessiyada istənilən sayda cümlə.
 *
 * Axın: kamera kadrı → landmark-lar → 30 fps-ə gətirmə (model 30 fps-də öyrədilib) → son `window` kadr
 * hər `stride` yeni kadrda modelə → bərkimiş kadrların ehtimalları → SentenceStream → yeni sözlər.
 * Qaydalar ml/azsl/continuous.py ilə eynidir (tools/verify_continuous.mjs). Video cihazdan çıxmır.
 */
import { createLandmarkers, detectFrame, LandmarkClip } from './landmarks.js';
import { LocalRecognizer } from './recognizer.js';
import { CONT_MODEL_BASE, GLOSS_TEXT_URL, SIGN_MERGE_URL } from './paths.js';
import { GlossToText } from './gloss2text.js';
import {
  CONTINUOUS_VERSION, DECODE, nearestSentence, Resampler, SentenceStream, shortClasses, STREAM, WindowStream,
} from './continuous.js';

export class SignToSentence {
  /**
   * @param {object} o
   * @param {HTMLVideoElement} o.video
   * @param {(e: {frame, aspect, fileProgress?}) => void} [o.onFrame] hər emal olunmuş kamera kadrı
   * @param {(e: {words, added, sentence, text, ms}) => void} [o.onWords] yeni yekun sözlər (cari cümlə)
   * @param {(e: {words, match, text}) => void} [o.onSentence] cümlə bağlandı; match — ən yaxın AzSLD cümləsi,
   *   text — cümlənin mətni: match-in mətni, yoxsa işarələrdən qurulmuş mətn (gloss2text.js)
   * @param {(e: {stage, message}) => void} [o.onStatus]
   */
  constructor({ video, onFrame, onWords, onSentence, onStatus, base = CONT_MODEL_BASE } = {}) {
    Object.assign(this, { video, onFrame, onWords, onSentence, onStatus, base });
    this.words = [];
    this.sentences = [];
    this.known = [];
    this.lastTs = 0;
  }

  async init() {
    this.onStatus?.({ stage: 'loading', message: 'Cümlə modeli yüklənir…' });
    this.rec = await LocalRecognizer.load(this.base, { merge: SIGN_MERGE_URL });   // eyni işarənin formaları bir söz
    const m = this.rec.manifest;
    if (m.kind !== 'continuous') throw new Error('bu model davamlı tanıma üçün deyil');
    if ((m.continuous_version ?? 1) > CONTINUOUS_VERSION) {
      throw new Error(`model continuous_version ${m.continuous_version} istəyir, brauzer ${CONTINUOUS_VERSION}-dir`);
    }
    this.p = { ...STREAM, ...m.stream };
    this.dp = { ...DECODE, ...m.decode };
    this.face = !m.works_without_face;
    const unk = m.unknown_label ? this.rec.labels.indexOf(m.unknown_label) : -1;
    this.unk = unk >= 0 ? unk : null;
    try {                                     // AzSLD cümlələri: tanınan ardıcıllığa ən yaxın mətn
      const r = await fetch(`${this.base}sentences.json`);
      // cümlələrin glosları da modelin birləşmiş siniflərinə gətirilir
      if (r.ok) this.known = (await r.json()).map((x) => ({ ...x, glosses: x.glosses.map(this.rec.group) }));
    } catch { /* cümlə siyahısı olmasa da işləyir */ }
    try {                                     // siyahıda olmayan cümlə: işarələrdən Azərbaycan mətni
      const r = await fetch(GLOSS_TEXT_URL);
      if (r.ok) this.g2t = new GlossToText(await r.json());
    } catch { /* model olmasa işarələrin adları göstərilir */ }
    this.onStatus?.({ stage: 'landmarker', message: `MediaPipe yüklənir (əllər, bədən${this.face ? ', üz' : ''})…` });
    this.lms = await createLandmarkers({ face: this.face });
    this.#reset();
    const n = this.rec.labels.length - 1 - (this.unk === null ? 0 : 1);
    this.onStatus?.({ stage: 'ready', message: `Hazır · ${n} söz, davamlı` });
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
    this.onStatus?.({ stage: 'running', message: 'Kamera işləyir · cümləni fasiləsiz göstər, sonda əlləri endir' });
    this.#tick();
  }

  /** Yüklənmiş video kadr-kadr, 30 fps-də (kadr buraxılmır, 30 fps-ə gətirmə lazım deyil). */
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
    const t0 = this.lastTs + 1;
    for (let i = 0; i < n; i++) {
      v.currentTime = (i + 0.5) / fps;
      await new Promise((resolve) => v.addEventListener('seeked', resolve, { once: true }));
      if (this._file !== token) return;
      this.#process(v, t0 + Math.round((i * 1000) / fps), { fileProgress: (i + 1) / n, resample: false });
    }
    await this.#end();
    this._file = null;
    this.running = false;
    this.onStatus?.({ stage: 'ended', message: 'Video bitdi' });
  }

  stop() {
    const wasRunning = this.running;
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
    if (wasRunning) this.#end();
    this.onStatus?.({ stage: 'stopped', message: 'Dayandırıldı' });
  }

  clear() {
    this.words = [];
    this.sentences = [];
    this.#reset();
    this.onWords?.({ words: [], added: [], sentence: null, ms: 0 });
  }

  #reset() {
    this.clip = new LandmarkClip(this.p.window + 32);
    this.ws = new WindowStream(this.p);
    this.ss = new SentenceStream({ C: this.rec.labels.length, decode: this.dp, unknown: this.unk,
      short: shortClasses(this.rec.labels) });
    this.resampler = new Resampler(30);
    this.busy = Promise.resolve();
    this._lastVideoTime = -1;
  }

  #tick = () => {
    if (!this.running || this._file) return;
    this._raf = requestAnimationFrame(this.#tick);
    const v = this.video;
    if (v.readyState < 2 || v.currentTime === this._lastVideoTime) return;
    this._lastVideoTime = v.currentTime;
    this.#process(v, performance.now(), { resample: true });
  };

  #process(v, ts, { resample, ...extra }) {
    this.aspect = v.videoWidth / v.videoHeight;
    ts = Math.max(Math.round(ts), this.lastTs + 1);
    this.lastTs = ts;
    const frame = detectFrame(this.lms, v, ts);
    for (const f of resample ? this.resampler.push(frame, ts) : [frame]) {
      this.clip.push(f);
      const w = this.ws.push();
      if (w) this.#run(w);
    }
    this.onFrame?.({ frame, aspect: this.aspect, ...extra });
  }

  /** Pəncərənin kadrları indi köçürülür (bufer dəyişməyə davam edir), model növbə ilə işləyir. */
  #run(w, flush = false) {
    const first = this.ws.n - this.clip.length;         // buferin ilk kadrının mütləq nömrəsi
    const raw = this.clip.toRaw(w.a - first, w.b - first);
    const aspect = this.aspect, C = this.rec.labels.length;
    this.busy = this.busy.then(async () => {
      const t0 = performance.now();
      const { probs } = await this.rec.frameProbs(raw, aspect, { face: this.face });
      const k = w.to - w.from;
      const res = this.ss.add(probs.subarray((w.from - w.a) * C, (w.to - w.a) * C), k);
      this.#emit(res, performance.now() - t0);
      if (flush) this.#emit(this.ss.flush(), 0);
    }).catch((err) => {
      console.error(err);
      this.onStatus?.({ stage: 'error', message: 'Tanıma xətası: ' + err.message });
    });
    return this.busy;
  }

  #emit({ added, sentenceEnded }, ms) {
    const labels = this.rec.labels;
    const fresh = added.map((s) => ({ label: labels[s.c], p: s.p, a: s.a, b: s.b }));
    this.words.push(...fresh);
    const seq = this.words.map((x) => x.label);
    const match = nearestSentence(seq, this.known);
    const text = match ? match.text : seq.length ? this.g2t?.generate(seq).text ?? null : null;
    if (fresh.length) this.onWords?.({ words: this.words, added: fresh, sentence: match, text, ms });
    if (sentenceEnded && this.words.length) {
      this.sentences.push({ words: this.words, match, text });
      this.onSentence?.({ words: this.words, match, text });
      this.words = [];
    }
  }

  async #end() {
    const w = this.ws.flush();
    if (w) await this.#run(w, true);
    else this.#emit(this.ss.flush(), 0);
    await this.busy;
  }
}
