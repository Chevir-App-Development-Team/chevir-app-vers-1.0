/**
 * Söz işarələri lüğəti (tools/build_words.py → public/tercume/words/).
 *
 * İndeks bir dəfə yüklənir; sözün kadrları yalnız oynadılanda gətirilir.
 * Mətn sözlərə bölünür və ən uzun uyğunluq axtarılır — "zəhmət olmasa",
 * "bu gün" kimi ifadələr bütöv işarədir.
 */
import { WORDS_BASE } from './paths.js';
import { Lemmatizer } from './morph.js';

/** Azərbaycan qaydası ilə kiçik hərf: I → ı, İ → i (toLowerCase "İ"-ni "i̇" edir). */
export const azLower = (s) => s.replace(/I/g, 'ı').replace(/İ/g, 'i').toLocaleLowerCase('az');

const EDGE_PUNCT = /^[.,!?;:"'“”‘’«»()[\]…–—-]+|[.,!?;:"'“”‘’«»()[\]…–—-]+$/g;

export class WordLexicon {
  constructor(index) {
    this.entries = index?.words ?? [];
    // Lüğət üzrə ölçülər hər sözə qoşulur: üz blendshape-lərinin adları və diapazonu (avatarın
    // üz ifadəsi), siqnalçıların sinə xətti (sinəyə toxunan əl avatarın sinəsinə köçürülür)
    this.shared = { faceKeys: index?.face ?? null, faceNeutral: index?.faceNeutral ?? null,
      faceFull: index?.faceFull ?? null, chest: index?.chest ?? null };
    // Açar kiçik hərflə saxlanır: xüsusi adlar lüğətdə böyük hərflə yazılır (Bakı,
    // Azərbaycan), mətndən gələn token isə həmişə kiçik hərfdədir. `alt` — eyni işarənin
    // digər yazılışları (danışıq forması: burada/burda).
    this.byText = new Map();
    for (const w of this.entries) {
      for (const t of [w.text, ...(w.alt ?? [])]) this.byText.set(azLower(t), w);
    }
    this.maxTokens = Math.max(1, ...this.entries.flatMap(
      (w) => [w.text, ...(w.alt ?? [])].map((t) => t.split(' ').length)));
    // Söz formaları (evdə → ev, bilmirəm → bilmir): açar kök + inkardır (morph.js).
    // Köklər tək sözlü yazılışlardan çıxarılır; ifadənin sözləri ("ad günü") kök sayılmır.
    this.lemma = new Lemmatizer([...this.byText.keys()].filter((t) => !t.includes(' ')));
    this.byKey = new Map();
    for (const [t, w] of this.byText) {
      const k = this.#keys(t.split(' '));
      if (k) this.byKey.set(k, [...(this.byKey.get(k) ?? []), { form: t, entry: w }]);
    }
    this.cache = new Map();
  }

  #keys(tokens) {
    if (tokens.length === 1) return this.lemma.key(tokens[0]);
    return tokens.map((t) => this.lemma.phraseKey(t)).join(' ');
  }

  /**
   * Söz formasına uyğun gələn lüğət sözü: əvvəl yazılışın özü, sonra kökü. Eyni kökdə bir
   * neçə söz varsa: mətndəki söz o sözün davamıdırsa, onu (istəyirsən → istəyir; bilmirəm →
   * bilmir), yoxsa lüğət formasını (olan → olmaq, olar yox), o da yoxdursa ən oxşarını.
   */
  lookup(tokens) {
    const text = tokens.join(' ');
    const exact = this.byText.get(text);
    if (exact) return { entry: exact, via: 'text' };
    const k = this.#keys(tokens);
    const found = k && this.byKey.get(k);
    if (!found) return null;
    const common = (a) => { let n = 0; while (n < a.length && a[n] === text[n]) n++; return n; };
    const longer = found.filter((f) => text.startsWith(f.form) && !this.lemma.isBase(f.form)
      && f.form.length >= (this.lemma.key(f.form) ?? '').replace(/-$/, '').length + 2);
    const base = found.filter((f) => f.form.split(' ').every((t) => this.lemma.isBase(t)));
    const pool = longer.length ? longer : base.length ? base : found;
    const best = pool.reduce((a, b) => (common(b.form) > common(a.form)
      || (common(b.form) === common(a.form) && b.form.length < a.form.length) ? b : a));
    return { entry: best.entry, via: 'lemma' };
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
   * Mətni hissələrə bölür: {kind:'word', text, entry, via} | {kind:'text', text} | {kind:'space'}.
   * `via`: 'text' — yazılış lüğətdədir, 'lemma' — sözün kökü (evdə → ev).
   * `text` hissəsi kiçik hərflə və kənar durğu işarələri olmadan qaytarılır; `from`/`to`
   * onun ilkin mətndəki yeridir (oxunan hissəni vurğulamaq üçün).
   */
  segment(text) {
    const parts = text.split(/(\s+)/).filter(Boolean);
    const toks = parts.map((p) => (/^\s+$/.test(p) ? null : azLower(p).replace(EDGE_PUNCT, '')));
    // Hər tokenin ilkin mətndə yeri: hissənin başı + başdakı durğu işarələrinin uzunluğu
    const at = [];
    let pos = 0;
    parts.forEach((p, i) => {
      const lead = toks[i] ? azLower(p).indexOf(toks[i]) : 0;
      at.push({ from: pos + lead, to: pos + lead + (toks[i]?.length ?? p.length) });
      pos += p.length;
    });
    // Kiçik hərfli mətndə böyük hərflərlə yazılmış qısaltma (DOST, ASAN) hərf-hərf göstərilir:
    // "dost" sözünün işarəsi deyil. Lüğətdə böyük hərflə yazılan ad (Covid) işarə ilə qalır;
    // bütün mətn böyük hərflədirsə, sözlər adi qaydada tanınır.
    const upper = (p) => p === p.toLocaleUpperCase('az') && p !== p.toLocaleLowerCase('az');
    const mixed = /\p{Ll}/u.test(text);
    const acronym = parts.map((p, i) => mixed && Boolean(toks[i]) && upper(p)
      && [...p].filter((c) => /\p{L}/u.test(c)).length >= 2);
    const out = [];
    for (let i = 0; i < parts.length;) {
      if (toks[i] === null) { out.push({ kind: 'space', ...at[i] }); i++; continue; }
      if (!toks[i]) { i++; continue; }                       // yalnız durğu işarəsi
      let match = null;
      for (let n = this.maxTokens; n >= 1 && !match; n--) {
        const idx = [];
        for (let j = i; j < parts.length && idx.length < n; j++) if (toks[j]) idx.push(j);
        if (idx.length < n) continue;
        const hit = this.lookup(idx.map((j) => toks[j]));
        if (hit && idx.some((j) => acronym[j]) && hit.entry.text === azLower(hit.entry.text)) continue;
        if (hit) match = { ...hit, last: idx[idx.length - 1] };
      }
      if (match) {
        out.push({ kind: 'word', text: match.entry.text, entry: match.entry, via: match.via,
          from: at[i].from, to: at[match.last].to });
        i = match.last + 1;
      } else {
        out.push({ kind: 'text', text: toks[i], ...at[i] });
        i++;
      }
    }
    return out;
  }

  /** Sözün kadrlarını gətirir (keşlə). */
  get(id) {
    if (!this.cache.has(id)) {
      this.cache.set(id, fetch(`${WORDS_BASE}${id}.json`).then((r) => {
        if (!r.ok) throw new Error(`${id}.json yüklənmədi (${r.status})`);
        return r.json().then((w) => ({ ...w, ...this.shared }));
      }));
    }
    return this.cache.get(id);
  }
}
