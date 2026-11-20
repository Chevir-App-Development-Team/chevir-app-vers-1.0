/**
 * Jest → cümlə: davamlı model (manifest.kind = 'continuous') landmark axınının hər kadrına etiket verir,
 * sözlər işarələnərkən, fasiləsiz çıxır. Uzun fasilə cümləni bağlayır; bir sessiyada istənilən sayda cümlə.
 *
 * Axın: kamera kadrı → landmark-lar → 30 fps-ə gətirmə (model 30 fps-də öyrədilib) → son `window` kadr
 * hər `stride` yeni kadrda modelə → bərkimiş kadrların ehtimalları → SentenceStream → yeni sözlər.
 * İşarə dil modeli (sign_lm.json, E32) hər sözü əvvəlki sözlərə görə seçir; cümlə bağlananda 2–3 alternativ mətn
 * payları ilə gəlir (onSentence alternatives, correct.js onları göstərir və düzəlişi cihazda saxlayır).
 * Qaydalar ml/azsl/continuous.py ilə eynidir (tools/verify_continuous.mjs). Video cihazdan çıxmır.
 */
import { createLandmarkers, detectFrame, LandmarkClip } from './landmarks.js';
import { LocalRecognizer } from './recognizer.js';
import { CONT_MODEL_BASE, GLOSS_TEXT_URL, SIGN_LM_URL, SIGN_MERGE_URL } from './paths.js';
import { GlossToText } from './gloss2text.js';
import { FRAME_GAP_MS, MIN_HANDS, PENDING_RULE, SENTENCE_GAP, showPending } from './live.js';
import {
  CONTINUOUS_VERSION, DECODE, LM_DECODE, nearestSentence, Resampler, SentenceStream, shortClasses, SignLM, STREAM,
  WindowStream,
} from './continuous.js';

export class SignToSentence {
  /**
   * @param {object} o
   * @param {HTMLVideoElement} o.video
   * @param {(e: {frame, aspect, fileProgress?}) => void} [o.onFrame] hər emal olunmuş kamera kadrı
   * @param {(e: {words, added, sentence, text, ms}) => void} [o.onWords] yeni yekun sözlər (cari cümlə)
   * @param {(e: {words, match, text, alternatives}) => void} [o.onSentence] cümlə bağlandı; match — işarələri çox yaxın
   *   olan AzSLD cümləsi (textOf), text — cümlənin mətni: match-in mətni, yoxsa işarələrdən qurulmuş mətn (gloss2text.js);
   *   alternatives (dil modeli ilə) — [{labels, text, match, conf}], birincisi cümlənin özüdür, conf — variantlar
   *   arasında payı (eyni mətnli variantlar birləşir)
   * @param {(e: {words, pending, text, match}) => void} [o.onLive] hər pəncərədə (~0.27 s), dəyişəndə: yekun sözlər,
   *   hələ yekun olmayan sözlər (işarə göstərilən anda, sonra dəyişə bilər; live.js PENDING_RULE-a görə) və
   *   hamısından qurulan mətn
   * @param {(e: {progress}) => void} [o.onPause] hər pəncərədə: cümlədə yekun söz var və yeni işarə yoxdursa, cümlənin
   *   bağlanmasına qədər keçən hissə (0…1), yoxsa 0
   * @param {(e: object) => void} [o.onTrace] ölçmə (/danis/?debug=1): hər sözün boz/ağ anı, yalançı boz sözlər
   * @param {(e: {stage, message}) => void} [o.onStatus]
   * @param {boolean} [o.useLM] işarə dil modeli (E32); false — əvvəlki deşifrə (decode), alternativ yoxdur
   */
  constructor({ video, onFrame, onWords, onSentence, onLive, onPause, onStatus, onTrace, base = CONT_MODEL_BASE,
    pendingRule = PENDING_RULE, sentenceGap = SENTENCE_GAP, minHands = MIN_HANDS, useLM = true } = {}) {
    Object.assign(this, { video, onFrame, onWords, onSentence, onLive, onPause, onStatus, onTrace, base, pendingRule,
      sentenceGap, minHands, useLM });
    this.words = [];
    this.sentences = [];
    this.known = [];
    this.lastTs = 0;
    // ölçmə (onTrace verilibsə): kamera kadrları, MediaPipe və model vaxtı, hər sözün boz/ağ anı
    this.stats = { camFrames: 0, lmMs: 0, modelMs: 0, lagMs: 0 };
    this.frameTs = new Float64Array(4096);   // 30 fps kadrının (mütləq nömrə % 4096) gəldiyi an
    this.frameHands = new Uint8Array(4096);  // o kadrda neçə əl var
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
    this.lm = null;
    if (this.useLM) {
      try {                                   // işarə dil modeli: sözlər qonşularına görə, cümlənin alternativləri (E32)
        const r = await fetch(SIGN_LM_URL);
        if (r.ok) {
          const d = await r.json();
          this.lm = new SignLM(d, this.rec.labels);
          this.lp = { ...LM_DECODE, ...d.decode };
        }
      } catch { /* dil modeli olmasa decode() */ }
    }
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
    if (!this._visBound) {                   // səhifə gizlənəndə (başqa pəncərə, nişan) kadr gəlmir: açıq cümlə bağlanır
      document.addEventListener('visibilitychange', () => { if (document.hidden) this.#closeSentence(); });
      this._visBound = true;
    }
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
    this._liveKey = null;
    this._grey = [];
    this._seen = null;
    this.onWords?.({ words: [], added: [], sentence: null, ms: 0 });
    this.onLive?.({ words: [], pending: [], text: '', match: null });
  }

  #reset() {
    this.clip = new LandmarkClip(this.p.window + 32);
    this.ws = new WindowStream(this.p);
    this.ss = new SentenceStream({ C: this.rec.labels.length, decode: this.dp, unknown: this.unk,
      short: shortClasses(this.rec.labels), sentenceGap: this.sentenceGap, lm: this.lm, lp: this.lp });
    this.resampler = new Resampler(30);
    this.busy = Promise.resolve();
    this._lastVideoTime = -1;
    this._gen = (this._gen ?? 0) + 1;         // Təmizlə-dən əvvəl növbəyə düşmüş pəncərə yeni axına qarışmasın
  }

  #tick = () => {
    if (!this.running || this._file) return;
    this._raf = requestAnimationFrame(this.#tick);
    const v = this.video;
    if (this._closing || v.readyState < 2 || v.currentTime === this._lastVideoTime) return;
    this._lastVideoTime = v.currentTime;
    const now = performance.now();
    // FRAME_GAP_MS-dən uzun kadr gəlmədi (pəncərə örtülüb, kamera dayanıb): əvvəlki cümlə yenisinə qarışmasın
    if (this._lastFrameAt && now - this._lastFrameAt > FRAME_GAP_MS) {
      this.#closeSentence();
      return;
    }
    this._lastFrameAt = now;
    this.#process(v, now, { resample: true });
  };

  /** Açıq cümlə indi bağlanır (axının sonu kimi: gözləyən sözlər yekun olur), axın təzədən başlayır. */
  #closeSentence() {
    if (!this.running || this._file || this._closing) return;
    this._closing = true;
    this.#end().then(() => {
      this.#reset();
      this._lastFrameAt = 0;
    }).finally(() => { this._closing = false; });
  }

  #process(v, ts, { resample, ...extra }) {
    this.aspect = v.videoWidth / v.videoHeight;
    ts = Math.max(Math.round(ts), this.lastTs + 1);
    this.lastTs = ts;
    const t0 = performance.now();
    const frame = detectFrame(this.lms, v, ts);
    const st = this.stats;
    st.camFrames++;
    st.lmMs += ((performance.now() - t0) - st.lmMs) * 0.1;          // hərəkətli orta
    for (const f of resample ? this.resampler.push(frame, ts) : [frame]) {
      this.clip.push(f);
      const w = this.ws.push();
      this.frameTs[(this.ws.n - 1) & 4095] = t0;
      this.frameHands[(this.ws.n - 1) & 4095] = Number.isFinite(f.hand[0]) + Number.isFinite(f.hand[63]);
      if (w) this.#run(w);
    }
    this.onFrame?.({ frame, aspect: this.aspect, ...extra });
  }

  /** Pəncərənin kadrları indi köçürülür (bufer dəyişməyə davam edir), model növbə ilə işləyir. */
  #run(w, flush = false) {
    const first = this.ws.n - this.clip.length;         // buferin ilk kadrının mütləq nömrəsi
    const raw = this.clip.toRaw(w.a - first, w.b - first);
    const aspect = this.aspect, C = this.rec.labels.length, gen = this._gen;
    this.busy = this.busy.then(async () => {
      if (gen !== this._gen) return;
      const t0 = performance.now();
      const { probs } = await this.rec.frameProbs(raw, aspect, { face: this.face });
      if (gen !== this._gen) return;
      const st = this.stats, now = performance.now();
      st.modelMs += ((now - t0) - st.modelMs) * 0.1;
      st.lagMs += ((now - this.frameTs[(w.b - 1) & 4095]) - st.lagMs) * 0.1;   // pəncərənin son kadrından nəticəyə
      const k = w.to - w.from;
      const base = w.from - this.ss.rows.length;            // seqment kadrları (cümlənin sətirləri) → mütləq nömrə
      const res = this.ss.add(probs.subarray((w.from - w.a) * C, (w.to - w.a) * C), k);
      this.#emit(res, now - t0, base);
      if (flush) {
        const fb = w.to - this.ss.rows.length;
        this.#emit(this.ss.flush(), 0, fb);
      }
      if (this.onLive || this.onPause) {       // hələ bərkiməmiş kadrlardakı sözlər: dərhal, boz
        const pending = flush ? [] : this.ss.peek(probs.subarray((w.to - w.a) * C, (w.b - w.a) * C), w.b - w.to);
        const pb = w.to - this.ss.rows.length, n = this.words.length;
        if (this.onLive) {
          this.#live(this.#shown(pending.map((s, i) => ({ label: this.rec.labels[s.c], p: s.p, pending: true,
            pos: n + i, fa: s.a + pb, fb: s.b + pb, hands: this.#handsIn(s.a + pb, s.b + pb) }))));
        }
        // cümlə bağlanmağa nə qədər qalıb: son yekun sözdən (quietFrom) bəri bərkimiş kadrlar / sentenceGap (SentenceStream ilə eyni)
        const quiet = n && !pending.length ? (this.ss.rows.length - this.ss.quietFrom) / this.sentenceGap : 0;
        this.onPause?.({ progress: Math.min(1, Math.max(0, quiet)) });
      }
    }).catch((err) => {
      console.error(err);
      this.onStatus?.({ stage: 'error', message: 'Tanıma xətası: ' + err.message });
    });
    return this.busy;
  }

  /**
   * Boz göstərilən sözlər (live.js showPending): eyni yerdə (pos, cümlədəki sıra) eyni söz neçə ardıcıl pəncərədə
   * gəlib. Bir pəncərəlik yalançı sözlər (~0.27 s görünüb yox olan) belə süzülür (E35, tools/sim_danis.mjs).
   */
  #shown(pending) {
    const prev = this._seen ?? new Map(), seen = new Map();
    const out = pending.filter((w) => {
      const key = `${w.pos}:${w.label}`, count = (prev.get(key) ?? 0) + 1;
      seen.set(key, count);
      return w.hands >= this.minHands && showPending({ p: w.p, len: w.fb - w.fa }, count, this.pendingRule);
    });
    this._seen = seen;
    return out;
  }

  /** base: cümlə sətirlərinin mütləq kadr nömrəsi (fa/fb — kadrın mütləq nömrəsi, ölçmə üçün) */
  #emit({ added, sentenceEnded, alternatives }, ms, base = 0) {
    const labels = this.rec.labels;
    const all = added.map((s) => ({ label: labels[s.c], p: s.p, a: s.a, b: s.b, fa: s.a + base, fb: s.b + base,
      hands: this.#handsIn(s.a + base, s.b + base) }));
    // kadrlarında demək olar əl olmayan söz (əllər kadrdan kənarda, yalnız qollar tərpənir): göstərilmir (live.js)
    const fresh = all.filter((w) => w.hands >= this.minHands);
    if (this.onTrace) {
      for (const w of all) if (w.hands < this.minHands) this.onTrace({ kind: 'nohands', label: w.label, p: w.p, hands: w.hands });
      this.#traceFinal(fresh);
    }
    this.words.push(...fresh);
    const { text, match } = this.textOf(this.words.map((x) => x.label));
    if (fresh.length) this.onWords?.({ words: this.words, added: fresh, sentence: match, text, ms });
    if (sentenceEnded && this.words.length) {
      const alts = alternatives ? this.#alternatives(alternatives, base) : null;
      this.sentences.push({ words: this.words, match, text, alternatives: alts });
      this.onSentence?.({ words: this.words, match, text, alternatives: alts });
      if (this.onTrace) this.#traceSentence(text);
      this.words = [];
    }
  }

  /**
   * SentenceStream-in alternativləri → mətnlər: hər variantın sözləri yekun sözlər kimi süzülür (əl az olan söz
   * göstərilmir), mətn textOf ilə; eyni mətnli variantların payı birləşir. Birincisi cümlənin özüdür.
   */
  #alternatives(list, base) {
    const out = [];
    for (const x of list) {
      const labels = x.words.filter((w) => this.#handsIn(w.a + base, w.b + base) >= this.minHands)
        .map((w) => this.rec.labels[w.c]);
      if (!labels.length) continue;
      const { text, match } = this.textOf(labels);
      const same = out.find((o) => o.text === text);
      if (same) same.conf += x.conf;
      else out.push({ labels, text, match, conf: x.conf });
    }
    return out;
  }

  /** [fa, fb) kadrlarının neçə hissəsində ən azı bir əl var (frameHands, son 4096 kadr). */
  #handsIn(fa, fb) {
    let h = 0;
    for (let f = fa; f < fb; f++) h += this.frameHands[f & 4095] > 0;
    return h / Math.max(1, fb - fa);
  }

  /**
   * Ölçmə: hər söz üçün işarənin kadrlarının gəldiyi an (fa, fb), boz göründüyü və ağ olduğu an; boz görünüb
   * yekun olmayan sözlər (dropped). Vaxtlar performance.now() ilə, ms.
   */
  #traceFinal(fresh) {
    const now = performance.now(), t = (f) => this.frameTs[f & 4095];
    fresh.forEach((w, i) => {
      const idx = this.words.length + i, g = this._grey?.[idx];
      const grey = g && g.label === w.label ? g.t : null;
      this.onTrace({ kind: 'word', label: w.label, p: w.p, fa: w.fa, fb: w.fb, len: w.fb - w.fa,
        hands: w.hands, tStart: t(w.fa), tEnd: t(w.fb - 1), grey, white: now,
        sentence: this.sentences.length });
    });
  }

  #traceGrey(pending) {
    const now = performance.now(), n = this.words.length;
    this._grey ??= [];
    // əvvəl boz görünüb indi nə yekun, nə boz olan sözlər: yalançı göstəriş
    for (const [idx, g] of this._grey.entries()) {
      if (!g || g.done) continue;
      const fin = idx < n ? this.words[idx] : null, pen = pending.find((p) => p.pos === idx);
      if (fin?.label === g.label) { g.done = true; continue; }     // yekun oldu
      if (pen?.label === g.label) continue;                        // hələ boz
      g.done = true;
      this.onTrace({ kind: 'dropped', label: g.label, grey: g.t, gone: now, fa: g.fa, fb: g.fb, sentence: this.sentences.length });
    }
    for (const p of pending) {
      const g = this._grey[p.pos];
      if (!g || g.label !== p.label || g.done) this._grey[p.pos] = { label: p.label, t: now, fa: p.fa, fb: p.fb };
    }
  }

  #traceSentence(text) {
    this.onTrace({ kind: 'sentence', text, words: this.words.map((w) => w.label), t: performance.now(),
      sentence: this.sentences.length - 1 });
    this._grey = [];
  }

  /**
   * Mətn: tanınan ardıcıllıq AzSLD cümləsinə çox yaxındırsa (normallaşdırılmış məsafə ≤ 0.2) onun mətni, yoxsa
   * işarələrdən qurulmuş mətn. 0.34 yeni cümlələrdə başqa cümləni göstərirdi; tanınan test yazılarında (E28,
   * eval_gloss_text --sweep) 0.2-də yeni cümlənin WER-i 0-dakı kimidir (48.8 %), məlum cümlənin 14.5 % (0-da 21.5 %).
   */
  textOf(seq) {
    if (!seq.length) return { text: '', match: null };
    const match = nearestSentence(seq, this.known, 0.2);
    return { text: match ? match.text : this.g2t?.generate(seq).text ?? seq.join(' '), match };
  }

  #live(pending) {
    const all = [...this.words, ...pending];
    const key = all.map((w) => w.label + (w.pending ? '?' : '')).join(' ');
    if (key === this._liveKey) return;       // dəyişməyibsə yenidən çəkmə
    this._liveKey = key;
    if (this.onTrace) this.#traceGrey(pending);
    this.onLive?.({ words: this.words, pending, ...this.textOf(all.map((w) => w.label)) });
  }

  async #end() {
    const w = this.ws.flush();
    if (w) await this.#run(w, true);
    else {
      const base = this.ws.n - this.ss.rows.length;      // bütün kadrlar bərkiyib
      this.#emit(this.ss.flush(), 0, base);
    }
    await this.busy;
  }
}
