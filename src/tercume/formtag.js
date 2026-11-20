/**
 * Sözün formasının modeli (roadmap N2, docs/NLP_PLAN.md S3, ml/EXPERIMENTS.md E33): işarələrin ardıcıllığından hər
 * işarənin əlamətləri (hal, mənsubiyyət, cəm, xəbərlik, zaman, inkar, sual) — gloss2text-in kontekst saylarından
 * seçdiyi əlamətlərin (#feats) yanında öyrənilmiş ikinci rəy. Kökü və növü (ad / fel) yenə gloss2text verir, sözü
 * inflect.js qurur, qaydalar (#native, #have, #time...) sonra yenə işləyir.
 *
 * Model: işarə (adı, növü, adının son hərfləri) → iki istiqamətli GRU → əlamətlər dəstəsinin ehtimalları. Öyrənmə
 * ml/formtag.py (AzSLD-nin cümlələri və korpusun süni cütləri, hər iki üslubda); bu fayl onun dəqiq JS portudur
 * (tools/verify_formtag.mjs, ml/fixtures/formtag_v<N>). Çəkilər float16, base64.
 */
import { azLower } from './words.js';
import { PRONOUNS } from './inflect.js';

export const FT_VERSION = 1;

const isLetter = (l) => [...l].length === 1 && /\p{L}/u.test(l);
const labelText = (l) => azLower(l.split('/')[0]);
// gloss2text-dəki kimi: tək hərf barmaqla yığılan hərfdir, əvəzlik "O" deyilsə və ya yanında başqa hərf varsa
const spelled = (labels, i) => isLetter(labels[i])
  && (!PRONOUNS.includes(labelText(labels[i])) || isLetter(labels[i - 1] ?? '') || isLetter(labels[i + 1] ?? ''));

/**
 * Hər işarənin girişi: tok — işarənin adı (rəqəm <NUM>, il <YEAR>, hərf <LET>), pos — adından çıxan növü
 * (gloss2text-in `guess`-i: noun / verb, yoxsa -), suf — adının son sözünün son 3 hərfi (görünməmiş işarə üçün).
 */
export function signTokens(labels, guess = {}) {
  return labels.map((l, i) => {
    if (/^\d+$/.test(l)) {
      const y = /^\d{4}$/.test(l) && Number(l) >= 1900 && Number(l) < 2100;
      return { tok: y ? '<YEAR>' : '<NUM>', pos: '-', suf: '-' };
    }
    if (spelled(labels, i) || (l === '.' && isLetter(labels[i - 1] ?? '') && isLetter(labels[i + 1] ?? ''))) return { tok: '<LET>', pos: '-', suf: '-' };
    const last = labelText(l).split(' ').at(-1);
    return { tok: l, pos: guess[l]?.pos ?? '-', suf: last.slice(-3) };
  });
}

/** float16 (base64, little endian) → Float32Array. */
function half(b64, n) {
  const bytes = typeof atob === 'function' ? Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)) : new Uint8Array(Buffer.from(b64, 'base64'));
  const u = new Uint16Array(bytes.buffer, bytes.byteOffset, n);
  const out = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    const h = u[k], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[k] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return out;
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export class FormTagger {
  /** json: ml/formtag.py export — {version, dims, vocab: {tok, suf, pos}, classes, weights: {ad: {shape, data}}}. */
  constructor(json) {
    if ((json.version ?? 1) > FT_VERSION) throw new Error(`formtag version ${json.version}, brauzer ${FT_VERSION}`);
    this.dims = json.dims;
    this.classes = json.classes;
    this.ix = Object.fromEntries(Object.entries(json.vocab).map(([k, list]) => [k, new Map(list.map((t, i) => [t, i]))]));
    this.W = {};
    for (const [k, { shape, data }] of Object.entries(json.weights)) this.W[k] = { shape, a: half(data, shape.reduce((x, y) => x * y, 1)) };
    // dəstələrin növü: felinkində zaman (tam) var
    this.isVerb = this.classes.map((c) => 'tam' in JSON.parse(c));
  }

  #emb(name, id) { const { shape, a } = this.W[name]; return a.subarray(id * shape[1], (id + 1) * shape[1]); }

  /** Bir istiqamət, bir qat: PyTorch GRU (r, z, n). xs — T×in, qaytarır T×H. */
  #gru(xs, pre, reverse) {
    const H = this.dims.H;
    const Wi = this.W[`${pre}.w_ih`].a, Wh = this.W[`${pre}.w_hh`].a, bi = this.W[`${pre}.b_ih`].a, bh = this.W[`${pre}.b_hh`].a;
    const D = xs[0]?.length ?? 0;
    const out = new Array(xs.length);
    let h = new Float64Array(H);
    const gi = new Float64Array(3 * H), gh = new Float64Array(3 * H);
    for (let s = 0; s < xs.length; s++) {
      const t = reverse ? xs.length - 1 - s : s, x = xs[t];
      for (let r = 0; r < 3 * H; r++) {
        let a = bi[r], b = bh[r];
        const oi = r * D, oh = r * H;
        for (let c = 0; c < D; c++) a += Wi[oi + c] * x[c];
        for (let c = 0; c < H; c++) b += Wh[oh + c] * h[c];
        gi[r] = a; gh[r] = b;
      }
      const nh = new Float64Array(H);
      for (let k = 0; k < H; k++) {
        const r = sigmoid(gi[k] + gh[k]), z = sigmoid(gi[H + k] + gh[H + k]);
        const n = Math.tanh(gi[2 * H + k] + r * gh[2 * H + k]);
        nh[k] = (1 - z) * n + z * h[k];
      }
      h = nh;
      out[t] = nh;
    }
    return out;
  }

  /** İşarələr → hər işarə üçün dəstələrin ehtimalları (Float64Array, `classes` sırası ilə). */
  probs(labels, guess = {}) { return this.probsOf(signTokens(labels, guess)); }

  /** signTokens-in çıxışı → ehtimallar. */
  probsOf(tokens) {
    if (!tokens.length) return [];
    const { tok, suf, pos } = this.ix;
    const id = (m, t) => m.get(t) ?? m.get('<UNK>') ?? 0;
    let xs = tokens.map((x) => {
      const parts = [this.#emb('emb_tok', id(tok, x.tok)), this.#emb('emb_suf', id(suf, x.suf)), this.#emb('emb_pos', id(pos, x.pos))];
      const v = new Float64Array(parts.reduce((n, p) => n + p.length, 0));
      let o = 0;
      for (const p of parts) { v.set(p, o); o += p.length; }
      return v;
    });
    for (let L = 0; L < this.dims.layers; L++) {
      const f = this.#gru(xs, `gru${L}.f`, false), b = this.#gru(xs, `gru${L}.b`, true);
      xs = f.map((v, t) => { const c = new Float64Array(2 * this.dims.H); c.set(v); c.set(b[t], this.dims.H); return c; });
    }
    const { shape, a: Wo } = this.W['out.w'], bo = this.W['out.b'].a;
    const [C, D] = shape;
    return xs.map((x) => {
      const z = new Float64Array(C);
      let mx = -Infinity;
      for (let c = 0; c < C; c++) {
        let s = bo[c];
        for (let d = 0; d < D; d++) s += Wo[c * D + d] * x[d];
        z[c] = s; if (s > mx) mx = s;
      }
      let sum = 0;
      for (let c = 0; c < C; c++) { z[c] = Math.exp(z[c] - mx); sum += z[c]; }
      for (let c = 0; c < C; c++) z[c] /= sum;
      return z;
    });
  }
}
