/**
 * Beam search + leksikon yoxlaması.
 *
 * Müəllimin notebook-undaki məntiq: model hər proqnoz hadisəsində ən yaxşı k
 * hərfi verir, beam bunları birləşdirir və hərf-bigram dil modeli ilə çəkilir.
 * Buradaki fərqlər:
 *   - artımlı (incremental) beam: yaddaş cartesian product kimi partlamır
 *   - loq fəzasında toplama (ehtimal hasili çox kiçik ədədlərə düşmür)
 *   - leksikon prefiks kəsməsi: yalnız real Azərbaycan sözünün başlanğıcı
 *     ola bilən ardıcıllıqlar saxlanılır (rejim: strict / soft / off)
 *   - sonda lüğət yoxlaması: tam uyğunluq bonus alır, uyğunluq yoxdursa
 *     redaktə məsafəsi ilə ən yaxın real sözlər təklif olunur
 */

const NEG = -1e9;

export class LexiconDecoder {
  constructor(lm, vocab, prefixes) {
    this.symbols = lm.symbols;
    this.bow = lm.bow;
    this.logp = lm.logp;
    this.symIdx = new Map(lm.symbols.map((s, i) => [s, i]));
    this.letters = lm.symbols.filter((s) => s !== lm.bow);
    this.words = vocab.words;
    this.freq = vocab.freq;
    this.wordSet = new Set(vocab.words);
    this.prefixes = new Set(prefixes);
    // Redaktə məsafəsi axtarışını sürətləndirmək üçün uzunluğa görə indeks
    this.byLen = new Map();
    for (const w of this.words) {
      if (!this.byLen.has(w.length)) this.byLen.set(w.length, []);
      this.byLen.get(w.length).push(w);
    }
  }

  static async load(lmUrl, vocabUrl, prefUrl) {
    const [lm, vocab, pref] = await Promise.all([
      fetch(lmUrl).then((r) => r.json()),
      fetch(vocabUrl).then((r) => r.json()),
      fetch(prefUrl).then((r) => r.json()),
    ]);
    return new LexiconDecoder(lm, vocab, pref);
  }

  trans(prev, next) {
    const a = this.symIdx.get(prev), b = this.symIdx.get(next);
    if (a === undefined || b === undefined) return NEG;
    return this.logp[a][b];
  }

  /**
   * @param {Array<Array<{label:string,p:number}>>} steps hər hadisə üçün top-k
   * @param {object} opts {width, alpha, beta, mode: 'strict'|'soft'|'off', softPenalty}
   */
  decode(steps, opts = {}) {
    const {
      width = 8, alpha = 1.0, beta = 2.0,
      mode = 'soft', softPenalty = -4.0,
    } = opts;

    let beams = [{ text: '', am: 0, lm: 0, offPath: 0 }];

    for (const step of steps) {
      const next = [];
      for (const beam of beams) {
        const prev = beam.text.length ? beam.text[beam.text.length - 1] : this.bow;
        for (const cand of step) {
          const ch = cand.label;
          if (!this.symIdx.has(ch) || ch === this.bow) continue;  // idarə siniflərini at
          const text = beam.text + ch;
          const isPrefix = this.prefixes.has(text);
          if (mode === 'strict' && !isPrefix) continue;
          next.push({
            text,
            am: beam.am + Math.log(Math.max(cand.p, 1e-12)),
            lm: beam.lm + this.trans(prev, ch),
            offPath: beam.offPath + (isPrefix || mode === 'off' ? 0 : softPenalty),
          });
        }
      }
      if (!next.length) break;                       // strict rejimdə çıxılmaz vəziyyət
      next.sort((a, b) => this.score(b, alpha, beta) - this.score(a, alpha, beta));
      beams = next.slice(0, width);
    }

    const out = beams.map((b) => {
      const last = b.text.length ? b.text[b.text.length - 1] : this.bow;
      const endLm = b.lm + this.trans(last, this.bow);
      const inVocab = this.wordSet.has(b.text);
      return {
        text: b.text,
        am: b.am, lm: endLm, offPath: b.offPath, inVocab,
        freq: this.freq[b.text] || 0,
        score: b.am + alpha * endLm + b.offPath + (inVocab ? beta : 0),
      };
    });
    out.sort((a, b) => b.score - a.score);

    // Softmax ilə namizədlər arasında nisbi güvən
    const max = out.length ? out[0].score : 0;
    let z = 0;
    for (const o of out) { o._e = Math.exp(o.score - max); z += o._e; }
    for (const o of out) { o.conf = o._e / z; delete o._e; }

    const best = out[0];
    return {
      candidates: out,
      best,
      verified: best && best.inVocab ? best.text : null,
      suggestions: best ? this.nearestWords(best.text, 5) : [],
    };
  }

  score(b, alpha, beta) {
    return b.am + alpha * b.lm + b.offPath +
           (this.wordSet.has(b.text) ? beta : 0);
  }

  /** Redaktə məsafəsinə görə ən yaxın real sözlər (leksikon yoxlaması). */
  nearestWords(text, k = 5) {
    if (!text) return [];
    if (this.wordSet.has(text)) return [{ word: text, dist: 0, freq: this.freq[text] }];
    const out = [];
    const maxDist = text.length <= 3 ? 1 : text.length <= 6 ? 2 : 3;
    for (let L = text.length - maxDist; L <= text.length + maxDist; L++) {
      for (const w of this.byLen.get(L) || []) {
        const d = editDistance(text, w, maxDist);
        if (d <= maxDist) out.push({ word: w, dist: d, freq: this.freq[w] || 0 });
      }
    }
    out.sort((a, b) => a.dist - b.dist || b.freq - a.freq);
    return out.slice(0, k);
  }
}

/** Levenshtein, `cutoff`-dan böyük olan kimi dayanır. */
export function editDistance(a, b, cutoff = Infinity) {
  if (Math.abs(a.length - b.length) > cutoff) return cutoff + 1;
  const n = b.length;
  let prev = new Uint16Array(n + 1), cur = new Uint16Array(n + 1);
  for (let j = 0; j <= n; j++) prev[j] = j;
  for (let i = 1; i <= a.length; i++) {
    cur[0] = i;
    let rowMin = cur[0];
    const ca = a[i - 1];
    for (let j = 1; j <= n; j++) {
      const cost = ca === b[j - 1] ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
      if (cur[j] < rowMin) rowMin = cur[j];
    }
    if (rowMin > cutoff) return cutoff + 1;
    [prev, cur] = [cur, prev];
  }
  return prev[n];
}
