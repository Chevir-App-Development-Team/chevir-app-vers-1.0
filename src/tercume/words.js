/**
 * Söz işarələri lüğəti (tools/build_words.py → public/tercume/words/).
 *
 * İndeks bir dəfə yüklənir; sözün kadrları yalnız oynadılanda gətirilir.
 * Mətn sözlərə bölünür və ən uzun uyğunluq axtarılır — "zəhmət olmasa",
 * "bu gün" kimi ifadələr bütöv işarədir.
 */
import { WORDS_BASE } from './paths.js';

/** Azərbaycan qaydası ilə kiçik hərf: I → ı, İ → i (toLowerCase "İ"-ni "i̇" edir). */
export const azLower = (s) => s.replace(/I/g, 'ı').replace(/İ/g, 'i').toLocaleLowerCase('az');

const EDGE_PUNCT = /^[.,!?;:"'“”‘’«»()[\]…–—-]+|[.,!?;:"'“”‘’«»()[\]…–—-]+$/g;

export class WordLexicon {
  constructor(index) {
    this.entries = index?.words ?? [];
    this.byText = new Map(this.entries.map((w) => [w.text, w]));
    this.maxTokens = Math.max(1, ...this.entries.map((w) => w.text.split(' ').length));
    this.cache = new Map();
  }

  /** İndeksi yükləyir; tapılmasa boş lüğət qaytarır (sistem hərflərlə işləməyə davam edir). */
  static async load(url = `${WORDS_BASE}index.json`) {
    try {
      const res = await fetch(url);
      return new WordLexicon(res.ok ? await res.json() : null);
    } catch {
      return new WordLexicon(null);
    }
  }

  /**
   * Mətni hissələrə bölür: {kind:'word', text, entry} | {kind:'text', text} | {kind:'space'}.
   * `text` hissəsi kiçik hərflə və kənar durğu işarələri olmadan qaytarılır.
   */
  segment(text) {
    const parts = text.split(/(\s+)/).filter(Boolean);
    const toks = parts.map((p) => (/^\s+$/.test(p) ? null : azLower(p).replace(EDGE_PUNCT, '')));
    const out = [];
    for (let i = 0; i < parts.length;) {
      if (toks[i] === null) { out.push({ kind: 'space' }); i++; continue; }
      if (!toks[i]) { i++; continue; }                       // yalnız durğu işarəsi
      let match = null;
      for (let n = this.maxTokens; n >= 1 && !match; n--) {
        const idx = [];
        for (let j = i; j < parts.length && idx.length < n; j++) if (toks[j] !== null) idx.push(j);
        if (idx.length < n) continue;
        const entry = this.byText.get(idx.map((j) => toks[j]).join(' '));
        if (entry) match = { entry, last: idx[idx.length - 1] };
      }
      if (match) { out.push({ kind: 'word', text: match.entry.text, entry: match.entry }); i = match.last + 1; }
      else { out.push({ kind: 'text', text: toks[i] }); i++; }
    }
    return out;
  }

  /** Sözün kadrlarını gətirir (keşlə). */
  get(id) {
    if (!this.cache.has(id)) {
      this.cache.set(id, fetch(`${WORDS_BASE}${id}.json`).then((r) => {
        if (!r.ok) throw new Error(`${id}.json yüklənmədi (${r.status})`);
        return r.json();
      }));
    }
    return this.cache.get(id);
  }
}
