/**
 * Davamlı tanıma (cümlə rejimi): model pəncərənin hər kadrına etiket verir (söz və ya <blank> — işarə
 * yoxdur), sözlər işarələnərkən, fasiləsiz çıxır. ml/azsl/continuous.py ilə eyni qaydalar:
 *
 *   streamWindows / WindowStream  model son `window` kadr üzərində hər `stride` yeni kadrda işləyir; kadrın
 *                                 ehtimalları pəncərənin sağ ucundan `lookahead` kadr geridə qalanda bərkiyir
 *   decode                        kadr ehtimalları → sözlər (hamarlama, eyni sözün qaçışları, boşluqların
 *                                 birləşdirilməsi, min uzunluq və orta ehtimal)
 *   decodeLM / alternatives       decode + işarə dil modeli (SignLM, sign_lm.json, E32): hər söz əvvəlki sözlərə görə
 *                                 seçilir (soldan sağa, axın kimi); cümlənin 2–3 alternativi (şüa axtarışı)
 *   SentenceStream                bərkimiş kadrlar → yeni yekun sözlər; uzun fasilə cümləni bağlayır (və alternativlər)
 *   Resampler                     kamera kadrları (dəyişkən sürət) → 30 fps (model 30 fps-də öyrədilib)
 *
 * tools/verify_continuous.mjs bunları ml/fixtures/continuous_v3.json üzərində Python ilə tutuşdurur
 * (v2: barmaq əlifbası hərfləri və rəqəmlər, yəni bir simvollu etiketlər üçün ayrıca min_len_short; v3: dil modeli).
 */
import { FRAME_SIZES } from './landmarks.js';

export const CONTINUOUS_VERSION = 3;   // continuous.py CONTINUOUS_VERSION: bu fayl onun qaydalarını izləyir
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

/**
 * İşarə dil modelinin tokeni (E32, tools/build_sign_lm.mjs): barmaqla yığılan hərf (tək hərfli etiket, əvəzlik "O"
 * deyilsə) və nöqtə bir sinifdir, '<L>' — hansı hərf olduğunu kadrlar deyir; qalan etiket özüdür. continuous.py lm_token.
 */
export const lmToken = (label) => (([...label].length === 1 && /\p{L}/u.test(label) && label !== 'O') || label === '.' ? '<L>' : label);

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
/** continuous.py short_classes: bir simvollu etiketlər (hərflər, rəqəmlər) sözlərdən qısadır. */
export function shortClasses(labels) {
  return new Set(labels.flatMap((g, i) => ([...g].length === 1 ? [i] : [])));
}

export function decode(P, T, C, p = DECODE, unknown = null, short = null) {
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
    const need = short?.has(c) && p.min_len_short != null ? p.min_len_short : p.min_len;
    if (b - a >= need && pm >= p.min_p && c !== unknown) out.push({ a, b, c, p: pm });
  }
  return out;
}

// ---------------------------------------------------------------- işarə dil modeli ilə deşifrə (v3, E32)

/**
 * alpha — dil modelinin ln-ehtimalının çəkisi, beta — hər sözə bonus (dil modeli hər sözü "bahalaşdırır"), norm — sözün
 * öz unigramının nə qədəri geri alınır (1: nadir söz nadir olduğuna görə yox, qonşularına uymadığına görə cəzalanır),
 * top / floor — seqmentin namizədləri, beam / nbest — cümlənin alternativləri, temp — onların paylarının temperaturu
 * (confidences). alpha = beta = 0 → decode(). continuous.py LM_DECODE.
 */
export const LM_DECODE = { alpha: 0, beta: 0, norm: 0, top: 4, floor: 0.05, beam: 8, nbest: 3, temp: 1 };

/** İşarə n-qram modeli (public/tercume/assets/sign_lm.json): continuous.py SignLM ilə eyni ədədlər, eyni sırada toplanır. */
export class SignLM {
  constructor(d, labels) {
    this.order = d.order;
    const tid = new Map(d.tokens.map((t, i) => [t, i]));
    this.bos = tid.get('<s>');
    this.eos = tid.get('</s>');
    this.V = d.tokens.length - 1;
    this.p1 = d.p1;
    this.p = {};
    this.b = {};
    for (let n = 2; n <= this.order; n++) {
      const m = new Map();
      for (const [h, v] of Object.entries(d[`p${n}`] ?? {})) {
        const e = new Map();
        for (let i = 0; i < v.length; i += 2) e.set(v[i], v[i + 1]);
        m.set(h, e);
      }
      this.p[n] = m;
    }
    for (let n = 1; n < this.order; n++) this.b[n] = new Map(Object.entries(d[`b${n}`] ?? {}));
    this.tok = labels.map((g) => (g.startsWith('<') ? null : tid.get(lmToken(g)) ?? null));   // etiket → token
  }

  start() { return new Array(this.order - 1).fill(this.bos); }

  /** ln P(w | h), h — son order − 1 token: saxlanmış n-qram, yoxsa kontekstin geri çəkilməsi + aşağı dərəcə. */
  logp(h, w) {
    if (!h.length) return this.p1[w];
    const key = h.join(' ');
    const e = this.p[h.length + 1].get(key);
    if (e !== undefined && e.has(w)) return e.get(w);
    return (this.b[h.length].get(key) ?? 0) + this.logp(h.slice(1), w);
  }

  /** Sözün dil modeli balı: ln P(w | h) − norm · ln P(w); modelin bilmədiyi etiket: ln(1 / V). */
  score(h, w, norm) {
    if (w === null) return -Math.log(this.V) * (1 - norm);
    return this.logp(h, w) - norm * this.p1[w];
  }

  /** w-dən sonrakı tarixçə (modelin bilmədiyi etiket onu dəyişmir). */
  push(h, w) { return w === null || this.order === 1 ? h : [...h, w].slice(1); }
}

/**
 * decode()-un qaçışları, ehtimal həddindən əvvəl → [{a, b, cands: [{c, q}]}]: hamarlanmış etiket qaçışları, `gap` üzərindən
 * birləşmiş, kifayət qədər uzun (min_len / min_len_short), <unk> deyil; namizədlər — qaçışın kadrlarında orta ehtimalı ən
 * yüksək `top` etiket (<blank> və <unk> deyil, ≥ floor; qaçışın öz etiketi həmişə), ehtimalla azalan, bərabərdə indekslə.
 * continuous.py lm_segments.
 */
export function lmSegments(P, T, C, p = DECODE, unknown = null, short = null, top = 4, floor = 0.05) {
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
    const need = short?.has(c) && p.min_len_short != null ? p.min_len_short : p.min_len;
    if (b - a < need || c === unknown) continue;
    const q = new Float64Array(C);
    for (let t = a; t < b; t++) for (let j = 0; j < C; j++) q[j] += Ps[t * C + j];
    for (let j = 0; j < C; j++) q[j] /= b - a;
    const js = [];
    for (let j = 1; j < C; j++) if (j !== unknown && (q[j] >= floor || j === c)) js.push(j);
    js.sort((x, y) => q[y] - q[x] || x - y);
    let order = js.slice(0, top);
    if (!order.includes(c)) order = [...order.slice(0, top - 1), c];
    out.push({ a, b, cands: order.map((j) => ({ c: j, q: q[j] })) });
  }
  return out;
}

/** Namizədin balı: ln(orta ehtimal) + alpha · dil modeli + beta (continuous.py _word_score, eyni toplama sırası). */
const wordScore = (lm, lp, h, c, q) => Math.log(q) + (lm ? lp.alpha * lm.score(h, lm.tok[c], lp.norm) : 0) + lp.beta;
const endScore = (lm, lp, h) => (lm ? lp.alpha * lm.score(h, lm.eos, lp.norm) : 0);

/** Hər seqmentin sözü əvvəl seçilənlərdən sonra → {words, score} (yolun balı cümlənin sonu ilə). */
function leftToRight(segs, p, lm, lp) {
  const skip = Math.log(p.min_p);
  let h = lm ? lm.start() : [];
  const words = [];
  let total = 0;
  for (const { a, b, cands } of segs) {
    let best = null, bestS = skip;
    for (const { c, q } of cands) {
      const s = wordScore(lm, lp, h, c, q);
      if (s > bestS || (best === null && s === bestS)) { best = { c, q }; bestS = s; }
    }
    total += bestS;
    if (best) {
      words.push({ a, b, c: best.c, p: best.q });
      if (lm) h = lm.push(h, lm.tok[best.c]);
    }
  }
  return { words, score: total + endScore(lm, lp, h) };
}

// alternativlərin sırası: bal azalan, bərabərdə etiket ardıcıllığı, sonra başlanğıclar (continuous.py _hyp_key)
const cmpSeq = (x, y) => {
  for (let i = 0; i < Math.min(x.length, y.length); i++) if (x[i] !== y[i]) return x[i] - y[i];
  return x.length - y.length;
};
const cmpHyp = (x, y) => (y.score - x.score) || cmpSeq(x.words.map((w) => w.c), y.words.map((w) => w.c))
  || cmpSeq(x.words.map((w) => w.a), y.words.map((w) => w.a));

/** Seqmentlər üzərində şüa axtarışı → fərqli söz ardıcıllıqları [{score, words}], ən yaxşısı əvvəl. */
function beamSearch(segs, p, lm, lp) {
  const skip = Math.log(p.min_p);
  let beams = [{ score: 0, words: [], h: lm ? lm.start() : [] }];
  for (const { a, b, cands } of segs) {
    const grown = [];
    for (const { score: s0, words, h } of beams) {
      grown.push({ score: s0 + skip, words, h });
      for (const { c, q } of cands) {
        grown.push({ score: s0 + wordScore(lm, lp, h, c, q), words: [...words, { a, b, c, p: q }], h: lm ? lm.push(h, lm.tok[c]) : h });
      }
    }
    grown.sort(cmpHyp);
    beams = grown.slice(0, lp.beam);
  }
  const final = beams.map(({ score, words, h }) => ({ score: score + endScore(lm, lp, h), words })).sort(cmpHyp);
  const out = [], seen = new Set();
  for (const x of final) {
    const k = x.words.map((w) => w.c).join(' ');
    if (!seen.has(k)) { seen.add(k); out.push(x); }
  }
  return out;
}

/**
 * decode() dil modeli ilə, soldan sağa (brauzer sözləri cümlə göstərilərkən bərkidir): hər seqmentin sözü — əvvəl seçilən
 * sözlərdən sonra ln(orta ehtimal) + alpha · dil modeli + beta-sı ən yaxşı namizəd, heç biri ln(min_p)-ni keçmirsə söz
 * yoxdur. → [{a, b, c, p}], decode() kimi. Dil modeli olmadan (və ya alpha = beta = 0) decode()-dur (birləşmiş qaçışın
 * ortası başqa etiketə üstünlük verən nadir hal istisna). continuous.py decode_lm.
 */
export function decodeLM(P, T, C, p = DECODE, lm = null, lp = LM_DECODE, unknown = null, short = null) {
  return leftToRight(lmSegments(P, T, C, p, unknown, short, lp.top, lp.floor), p, lm, lp).words;
}

/**
 * Eyni seqmentlər və ballar üzərində şüa axtarışı, cümlənin sonu da dil modeli ilə balanır → ən çoxu lp.nbest fərqli söz
 * ardıcıllığı, ən yaxşısı əvvəl: [{score, words: [{a, b, c, p}]}]. continuous.py nbest.
 */
export function nbest(P, T, C, p = DECODE, lm = null, lp = LM_DECODE, unknown = null, short = null) {
  return beamSearch(lmSegments(P, T, C, p, unknown, short, lp.top, lp.floor), p, lm, lp).slice(0, lp.nbest);
}

/**
 * Cümlənin alternativləri (N4): əvvəl decodeLM-in sözləri (axın cümlə göstərilərkən bunları bərkidib), sonra şüanın ən
 * yaxşı başqa söz ardıcıllıqları → ən çoxu lp.nbest [{score, words}], nbest kimi balanır. continuous.py alternatives.
 */
export function alternatives(P, T, C, p = DECODE, lm = null, lp = LM_DECODE, unknown = null, short = null) {
  const segs = lmSegments(P, T, C, p, unknown, short, lp.top, lp.floor);
  const main = leftToRight(segs, p, lm, lp);
  const key = main.words.map((w) => w.c).join(' ');
  return [main, ...beamSearch(segs, p, lm, lp).filter((x) => x.words.map((w) => w.c).join(' ') !== key)].slice(0, lp.nbest);
}

/** Alternativlərin balları → cəmi 1 olan paylar (göstərilən alternativlər üzərində softmax). continuous.py confidences. */
export function confidences(scores, temp = 1) {
  if (!scores.length) return [];
  const m = Math.max(...scores);
  const e = scores.map((s) => Math.exp((s - m) / temp));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / z);
}

/**
 * Bərkimiş kadrlar → sözlər, işarələnərkən. Söz yekundur, nə vaxt ki sonrakı kadrlar onu artıq dəyişə
 * bilməz (b + gap + hamarlama ≤ bərkimiş kadr sayı); yekun sözlər heç vaxt dəyişmir. Son sözdən sonra
 * `sentenceGap` kadr yeni söz yoxdursa cümlə bağlanır. Söz olmayan uzun hissələr atılır (son `keep`
 * kadr qalır: təzə başlayan işarənin əvvəli itməsin).
 */
export class SentenceStream {
  /** lm (SignLM) verilibsə sözlər decodeLM ilə (lp — onun parametrləri) seçilir və cümlə bağlananda alternativlər də gəlir. */
  constructor({ C, decode: dp = DECODE, unknown = null, short = null, sentenceGap = 60, maxFrames = 3000, keep = 30,
    lm = null, lp = LM_DECODE }) {
    Object.assign(this, { C, dp, unknown, short, sentenceGap, maxFrames, keep, lm, lp });
    this.rows = [];             // hər bərkimiş kadr üçün Float32Array(C)
    this.emitted = 0;           // bu cümlədə çıxmış söz sayı
    this.lastEnd = 0;           // sonuncu sözün sonu (cümlənin kadrlarında)
    this.quietFrom = 0;         // fasilə bu kadrdan sayılır (sonuncu söz və ya dil modelinin atdığı göstəriləcək seqment)
  }

  /**
   * k kadrın ehtimalları (k×C, sətir-sətir) → { added: yeni yekun sözlər, sentenceEnded, alternatives? }. alternatives
   * (dil modeli ilə, cümlə bağlananda): alternatives() — birincisi cümlənin bərkimiş sözləridir, conf — payı (confidences).
   */
  add(probs, k) {
    for (let t = 0; t < k; t++) this.rows.push(probs.slice(t * this.C, (t + 1) * this.C));
    return this.#update(false);
  }

  /** Axın bitdi: gözləyən sözlər də yekundur, cümlə bağlanır. */
  flush() { return this.#update(true); }

  /**
   * Canlı göstəriş: bərkimiş kadrlar + pəncərənin hələ bərkiməmiş kadrları (tail, k×C) → hələ yekun olmayan
   * sözlər. Vəziyyəti dəyişmir; yekun sözlər yenə yalnız add()/flush()-dan çıxır (Python ilə eyni qaydalar),
   * bu isə işarəni göstərilən anda ekrana gətirir (bərkimə ~1 s gecikir).
   */
  peek(tail, k) {
    const T0 = this.rows.length, T = T0 + k, C = this.C;
    if (!T) return [];
    const P = new Float32Array(T * C);
    this.rows.forEach((r, t) => P.set(r, t * C));
    if (k) P.set(tail.subarray(0, k * C), T0 * C);
    return this.#words(P, T).slice(this.emitted);
  }

  #words(P, T) {
    return this.lm ? decodeLM(P, T, this.C, this.dp, this.lm, this.lp, this.unknown, this.short)
      : decode(P, T, this.C, this.dp, this.unknown, this.short);
  }

  #update(final) {
    const T = this.rows.length, C = this.C;
    const P = new Float32Array(T * C);
    this.rows.forEach((r, t) => P.set(r, t * C));
    // dil modeli ilə: sözlər decodeLM-dəndir; cümlənin bağlanma vaxtı isə decode()-un göstərəcəyi seqmentlərə görədir (ən
    // ehtimallı namizədi ≥ min_p), dil modelinin atdığı söz də "danışılır" sayılır: yoxsa sözlər arası fasilə uzanır və
    // cümlə ortadan bölünür (E32: tools/sim_danis.mjs --lm, 60 kadrda 3.0 % cümlə, bununla E35-dəki kimi)
    let segs, active = [];
    if (this.lm) {
      const cand = lmSegments(P, T, C, this.dp, this.unknown, this.short, this.lp.top, this.lp.floor);
      segs = leftToRight(cand, this.dp, this.lm, this.lp).words;
      active = cand.filter((x) => x.cands[0].q >= this.dp.min_p);
    } else segs = this.#words(P, T);
    const k = this.dp.smooth | 0;
    const lag = this.dp.gap + Math.max(1, k - Math.floor(k / 2));
    const force = final || T >= this.maxFrames;
    const ready = force ? segs : segs.filter((s) => s.b + lag <= T);
    const readyActive = force ? active : active.filter((s) => s.b + lag <= T);
    const added = ready.slice(this.emitted);
    this.emitted = ready.length;
    if (added.length) this.lastEnd = added[added.length - 1].b;
    // son söz (və ya dil modelinin atdığı, decode()-un göstərəcəyi seqment) bu kadrda bitib: fasilə buradan sayılır
    this.quietFrom = Math.max(this.lastEnd, readyActive.length ? readyActive[readyActive.length - 1].b : 0);
    const quiet = segs.length === ready.length && active.length === readyActive.length
      && T - this.quietFrom >= this.sentenceGap;
    const ended = this.emitted > 0 && (force || quiet);
    let alts = null;
    if (ended && this.lm) {
      alts = alternatives(P, T, C, this.dp, this.lm, this.lp, this.unknown, this.short);
      const conf = confidences(alts.map((x) => x.score), this.lp.temp);
      alts.forEach((x, i) => { x.conf = conf[i]; });
    }
    const idle = this.emitted === 0 && !segs.length && T > this.sentenceGap && (force || quiet);
    if (ended || idle) {
      this.rows = final ? [] : this.rows.slice(Math.max(ended ? this.lastEnd : 0, T - this.keep));
      this.emitted = 0;
      this.lastEnd = 0;
      this.quietFrom = 0;
    }
    return alts ? { added, sentenceEnded: ended, alternatives: alts } : { added, sentenceEnded: ended };
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
