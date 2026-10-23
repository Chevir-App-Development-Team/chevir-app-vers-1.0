/**
 * Davamlı tanıma (cümlə rejimi): model pəncərənin hər kadrına etiket verir (söz və ya <blank> — işarə
 * yoxdur), sözlər işarələnərkən, fasiləsiz çıxır. ml/azsl/continuous.py ilə eyni qaydalar:
 *
 *   streamWindows / WindowStream  model son `window` kadr üzərində hər `stride` yeni kadrda işləyir; kadrın
 *                                 ehtimalları pəncərənin sağ ucundan `lookahead` kadr geridə qalanda bərkiyir
 *   decode                        kadr ehtimalları → sözlər (hamarlama, eyni sözün qaçışları, boşluqların
 *                                 birləşdirilməsi, min uzunluq və orta ehtimal)
 *   SentenceStream                bərkimiş kadrlar → yeni yekun sözlər; uzun fasilə cümləni bağlayır
 *   Resampler                     kamera kadrları (dəyişkən sürət) → 30 fps (model 30 fps-də öyrədilib)
 *
 * tools/verify_continuous.mjs bunları ml/fixtures/continuous_v1.json üzərində Python ilə tutuşdurur.
 */
import { FRAME_SIZES } from './landmarks.js';

export const STREAM = { window: 96, stride: 8, lookahead: 24 };
export const DECODE = { smooth: 5, min_p: 0.5, min_len: 6, gap: 3 };

/** continuous.py stream_windows: [{a, b, from, to}] — model [a, b) kadrlarında işləyir, [from, to) bərkiyir. */
export function streamWindows(T, { window = 96, stride = 8, lookahead = 24 } = {}) {
  if (lookahead + stride > window) throw new Error('lookahead + stride > window');
  const out = [];
  let fixed = 0, n = 0;
  while (fixed < T) {
    n = Math.min(T, n + stride);
    const a = Math.max(0, n - window);
    const upto = n === T ? T : Math.max(fixed, n - lookahead);
    if (upto > fixed) {
      out.push({ a, b: n, from: fixed, to: upto });
      fixed = upto;
    }
  }
  return out;
}

/** Kadrlar bir-bir gəlir: run() lazım olan pəncərələri verir (sonda flush). streamWindows ilə eynidir. */
export class WindowStream {
  constructor({ window = 96, stride = 8, lookahead = 24 } = {}) {
    Object.assign(this, { window, stride, lookahead });
    this.n = 0;
    this.fixed = 0;
  }

  /** Yeni kadr sayıldı → indi işlədiləcək pəncərə (və ya null). */
  push() {
    this.n++;
    if (this.n % this.stride) return null;
    const upto = Math.max(this.fixed, this.n - this.lookahead);
    if (upto <= this.fixed) return null;
    const w = { a: Math.max(0, this.n - this.window), b: this.n, from: this.fixed, to: upto };
    this.fixed = upto;
    return w;
  }

  /** Axın bitdi: qalan kadrları son pəncərə bərkidir. */
  flush() {
    if (this.fixed >= this.n) return null;
    const w = { a: Math.max(0, this.n - this.window), b: this.n, from: this.fixed, to: this.n };
    this.fixed = this.n;
    return w;
  }
}

/** Mərkəzləşmiş hərəkətli orta (k kadr, kənarlarda qısa) — continuous.py smooth (float64). */
export function smooth(P, T, C, k) {
  if (k <= 1) return Float64Array.from(P);
  const c = new Float64Array((T + 1) * C);
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < C; j++) c[(t + 1) * C + j] = c[t * C + j] + P[t * C + j];
  }
  const out = new Float64Array(T * C);
  const h = Math.floor(k / 2);
  for (let t = 0; t < T; t++) {
    const lo = Math.min(Math.max(t - h, 0), T), hi = Math.min(Math.max(t + k - h, 0), T);
    for (let j = 0; j < C; j++) out[t * C + j] = (c[hi * C + j] - c[lo * C + j]) / (hi - lo);
  }
  return out;
}

/**
 * Kadr ehtimalları (T×C, sütun 0 = <blank>) → [{a, b, c, p}], kadrlar [a, b).
 * continuous.py decode ilə eyni.
 */
export function decode(P, T, C, p = DECODE, unknown = null) {
  const Ps = smooth(P, T, C, p.smooth | 0);
  const lab = new Int32Array(T);
  for (let t = 0; t < T; t++) {
    let best = 0, bv = -Infinity;
    for (let j = 0; j < C; j++) if (Ps[t * C + j] > bv) { bv = Ps[t * C + j]; best = j; }
    lab[t] = best;
  }
  const runs = [];
  for (let t = 0; t < T;) {
    let u = t;
    while (u < T && lab[u] === lab[t]) u++;
    if (lab[t] !== 0) runs.push([t, u, lab[t]]);
    t = u;
  }
  const joined = [];
  for (const r of runs) {
    const last = joined[joined.length - 1];
    if (last && last[2] === r[2] && r[0] - last[1] <= p.gap) last[1] = r[1];
    else joined.push([...r]);
  }
  const out = [];
  for (const [a, b, c] of joined) {
    let s = 0;
    for (let t = a; t < b; t++) s += Ps[t * C + c];
    const pm = s / (b - a);
    if (b - a >= p.min_len && pm >= p.min_p && c !== unknown) out.push({ a, b, c, p: pm });
  }
  return out;
}

/**
 * Bərkimiş kadrlar → sözlər, işarələnərkən. Söz yekundur, nə vaxt ki sonrakı kadrlar onu artıq dəyişə
 * bilməz (b + gap + hamarlama ≤ bərkimiş kadr sayı); yekun sözlər heç vaxt dəyişmir. Son sözdən sonra
 * `sentenceGap` kadr yeni söz yoxdursa cümlə bağlanır. Söz olmayan uzun hissələr atılır (son `keep`
 * kadr qalır: təzə başlayan işarənin əvvəli itməsin).
 */
export class SentenceStream {
  constructor({ C, decode: dp = DECODE, unknown = null, sentenceGap = 60, maxFrames = 3000, keep = 30 }) {
    Object.assign(this, { C, dp, unknown, sentenceGap, maxFrames, keep });
    this.rows = [];             // hər bərkimiş kadr üçün Float32Array(C)
    this.emitted = 0;           // bu cümlədə çıxmış söz sayı
    this.lastEnd = 0;           // sonuncu sözün sonu (cümlənin kadrlarında)
  }

  /** k kadrın ehtimalları (k×C, sətir-sətir) → { added: yeni yekun sözlər, sentenceEnded } */
  add(probs, k) {
    for (let t = 0; t < k; t++) this.rows.push(probs.slice(t * this.C, (t + 1) * this.C));
    return this.#update(false);
  }

  /** Axın bitdi: gözləyən sözlər də yekundur, cümlə bağlanır. */
  flush() { return this.#update(true); }

  #update(final) {
    const T = this.rows.length, C = this.C;
    const P = new Float32Array(T * C);
    this.rows.forEach((r, t) => P.set(r, t * C));
    const segs = decode(P, T, C, this.dp, this.unknown);
    const k = this.dp.smooth | 0;
    const lag = this.dp.gap + Math.max(1, k - Math.floor(k / 2));
    const force = final || T >= this.maxFrames;
    const ready = force ? segs : segs.filter((s) => s.b + lag <= T);
    const added = ready.slice(this.emitted);
    this.emitted = ready.length;
    if (added.length) this.lastEnd = added[added.length - 1].b;
    const quiet = segs.length === ready.length && T - this.lastEnd >= this.sentenceGap;
    const ended = this.emitted > 0 && (force || quiet);
    if (ended || (this.emitted === 0 && !segs.length && T > this.sentenceGap)) {
      this.rows = final ? [] : this.rows.slice(Math.max(ended ? this.lastEnd : 0, T - this.keep));
      this.emitted = 0;
      this.lastEnd = 0;
    }
    return { added, sentenceEnded: ended };
  }
}

/** Kamera kadrları dəyişkən sürətlə gəlir; model 30 fps-də öyrədilib → kadrlar vaxta görə xətti interpolyasiya. */
export class Resampler {
  constructor(fps = 30, maxGapMs = 500) {
    this.dt = 1000 / fps;
    this.maxGap = maxGapMs;
    this.prev = null;
    this.next = null;            // növbəti 30 fps vaxtı
  }

  /** (kadr, ms) → 30 fps şəbəkəsinə düşən kadrlar */
  push(frame, ts) {
    if (!this.prev || ts - this.prev.ts > this.maxGap) {   // ilk kadr və ya uzun fasilə: şəbəkə yenidən
      this.prev = { frame, ts };
      this.next = ts + this.dt;
      return [frame];
    }
    const out = [];
    const { frame: fa, ts: ta } = this.prev;
    while (this.next <= ts) {
      const w = ts === ta ? 1 : (this.next - ta) / (ts - ta);
      out.push(interpolate(fa, frame, w));
      this.next += this.dt;
    }
    this.prev = { frame, ts };
    return out;
  }
}

/** İki kadr arasında; hər hansı tərəfdə nöqtə yoxdursa (NaN), nəticə də NaN. */
export function interpolate(fa, fb, w) {
  const f = {};
  for (const k of Object.keys(fa)) {
    const a = fa[k], b = fb[k];
    const o = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) o[i] = a[i] + (b[i] - a[i]) * w;
    f[k] = o;
  }
  return f;
}

/** Tanınmış söz ardıcıllığına ən yaxın AzSLD cümləsi (normallaşdırılmış Levenshtein ≤ maxDist). */
export function nearestSentence(words, sentences, maxDist = 0.34) {
  if (!words.length) return null;
  let best = null;
  for (const s of sentences) {
    const d = levenshtein(words, s.glosses) / Math.max(words.length, s.glosses.length);
    if (d <= maxDist && (!best || d < best.dist)) best = { ...s, dist: d };
  }
  return best;
}

export function levenshtein(a, b) {
  const d = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    let prev = d[0];
    d[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const cur = d[j];
      d[j] = Math.min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1));
      prev = cur;
    }
  }
  return d[b.length];
}

export { FRAME_SIZES };
