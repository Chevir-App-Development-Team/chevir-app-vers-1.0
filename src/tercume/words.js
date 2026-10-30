/**
 * Söz işarələri lüğəti (tools/build_words.py → public/tercume/words/).
 *
 * İndeks bir dəfə yüklənir; sözün kadrları yalnız oynadılanda gətirilir.
 * Mətn sözlərə bölünür və ən uzun uyğunluq axtarılır — "zəhmət olmasa",
 * "bu gün" kimi ifadələr bütöv işarədir.
 */
import { WORDS_BASE } from './paths.js';
import { Lemmatizer, suffixOnly } from './morph.js';

/** Azərbaycan qaydası ilə kiçik hərf: I → ı, İ → i (toLowerCase "İ"-ni "i̇" edir). */
export const azLower = (s) => s.replace(/I/g, 'ı').replace(/İ/g, 'i').toLocaleLowerCase('az');

const EDGE_PUNCT = /^[.,!?;:"'“”‘’«»()[\]…–—-]+|[.,!?;:"'“”‘’«»()[\]…–—-]+$/g;

// Lüğətdə işarəsi olmayan köməkçi sözlər: AzSLD siqnalçıları onları göstərmir (ilə 0/6, idi 0/4,
// ki 0/2 cümlədə, ml/EXPERIMENTS.md E16), hərf-hərf göstərmək cümləni uzadır
const DROP = new Set(['ilə', 'idi', 'ki']);

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
    // Xəbər sözü (ucuzdur) öz sifətinə də gedir: ucuz → UCUZDUR
    this.lemma = new Lemmatizer([...this.byText.keys()].filter((t) => !t.includes(' ')), { predicates: true });
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
    let found = k && this.byKey.get(k);
    // İnkarlı felin öz işarəsi yoxdursa: felin işarəsi + inkar (neg), müsbət işarə tək qalmasın
    let neg = false;
    if (!found && tokens.length === 1 && k?.endsWith('-')) {
      found = this.byKey.get(k.slice(0, -1));
      neg = Boolean(found);
    }
    if (!found) return null;
    const common = (a) => { let n = 0; while (n < a.length && a[n] === text[n]) n++; return n; };
    const longer = found.filter((f) => text.startsWith(f.form) && !this.lemma.isBase(f.form)
      && f.form.length >= (this.lemma.key(f.form) ?? '').replace(/-$/, '').length + 2);
    // İfadədə son söz (fel) lüğət formasındadırsa: başa düşməmək, başa düşmədim yox
    const base = found.filter((f) => {
      const ts = f.form.split(' ');
      return ts.length > 1 ? this.lemma.isBase(ts.at(-1), true) : this.lemma.isBase(ts[0]);
    });
    const pool = longer.length ? longer : base.length ? base : found;
    const best = pool.reduce((a, b) => (common(b.form) > common(a.form)
      || (common(b.form) === common(a.form) && b.form.length < a.form.length) ? b : a));
    return neg ? { entry: best.entry, via: 'lemma', neg: true } : { entry: best.entry, via: 'lemma' };
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
   * Mətni hissələrə bölür: {kind:'word', text, entry, via} | {kind:'text', text} | {kind:'space'}
   * | {kind:'drop', text} — siqnalçıların göstərmədiyi köməkçi söz (oynadılmır).
   * `via`: 'text' — yazılış lüğətdədir, 'lemma' — sözün kökü (evdə → ev), 'neg' — inkarlı felin
   * öz işarəsi olmadığından felin işarəsindən sonra gələn YOX (oxumadım → OXUMAQ YOX).
   * `text` hissəsi kiçik hərflə və kənar durğu işarələri olmadan qaytarılır; `from`/`to`
   * onun ilkin mətndəki yeridir (oxunan hissəni vurğulamaq üçün).
   */
  segment(text) {
    const parts = text.split(/(\s+)/).filter(Boolean);
    // Kiçik hərfli mətndə böyük hərflərlə yazılmış qısaltma (DOST, ASAN) hərf-hərf göstərilir:
    // "dost" sözünün işarəsi deyil. Lüğətdə böyük hərflə yazılan ad (Covid) işarə ilə qalır;
    // bütün mətn böyük hərflədirsə, sözlər adi qaydada tanınır.
    const upper = (p) => p === p.toLocaleUpperCase('az') && p !== p.toLocaleLowerCase('az');
    const mixed = /\p{Ll}/u.test(text);
    const isAcronym = (raw) => mixed && upper(raw) && [...raw].filter((c) => /\p{L}/u.test(c)).length >= 2;
    // Tokenlər: {tok, from, to, acronym}; tok = null — boşluq, '' — yalnız durğu işarəsi.
    // `from`/`to` — tokenin ilkin mətndə yeri (başdakı durğu işarələri sayılmır).
    const items = [];
    let pos = 0;
    for (const p of parts) {
      if (/^\s+$/.test(p)) { items.push({ tok: null, from: pos, to: pos + p.length }); pos += p.length; continue; }
      const tok = azLower(p).replace(EDGE_PUNCT, '');
      const from = pos + (tok ? azLower(p).indexOf(tok) : 0);
      pos += p.length;
      // Defisli söz (6-dan, TSEK-ə, 2-ci, yazıb-oxuya): lüğətdə bütöv yoxdursa, defisdən sonrakı
      // şəkilçi atılır (işarə — rəqəm və ya ad; AzSLD siqnalçıları da TSEK-ə → TSEK göstərir),
      // şəkilçi deyilsə, hissələr ayrı sözlərdir
      const pieces = tok.split('-');
      if (pieces.length > 1 && !this.byText.has(tok) && pieces.every(Boolean)) {
        const raw = text.slice(from, from + tok.length);
        if (pieces.length === 2 && suffixOnly(pieces[1])) {
          items.push({ tok: pieces[0], from, to: from + tok.length, acronym: isAcronym(raw.slice(0, pieces[0].length)) });
        } else {
          let o = 0;
          for (const piece of pieces) {
            items.push({ tok: piece, from: from + o, to: from + o + piece.length, acronym: isAcronym(raw.slice(o, o + piece.length)) });
            o += piece.length + 1;
          }
        }
        continue;
      }
      items.push({ tok, from, to: from + tok.length, acronym: Boolean(tok) && isAcronym(p) });
    }
    const out = [];
    for (let i = 0; i < items.length;) {
      const it = items[i];
      if (it.tok === null) { out.push({ kind: 'space', from: it.from, to: it.to }); i++; continue; }
      if (!it.tok) { i++; continue; }                         // yalnız durğu işarəsi
      let match = null;
      for (let n = this.maxTokens; n >= 1 && !match; n--) {
        const idx = [];
        for (let j = i; j < items.length && idx.length < n; j++) if (items[j].tok) idx.push(j);
        if (idx.length < n) continue;
        const hit = this.lookup(idx.map((j) => items[j].tok));
        if (hit && idx.some((j) => items[j].acronym) && hit.entry.text === azLower(hit.entry.text)) continue;
        if (hit) match = { ...hit, last: idx[idx.length - 1] };
      }
      if (match) {
        const at = { from: it.from, to: items[match.last].to };
        out.push({ kind: 'word', text: match.entry.text, entry: match.entry, via: match.via, ...at });
        const no = match.neg && this.byText.get('yox');
        if (no) out.push({ kind: 'word', text: no.text, entry: no, via: 'neg', ...at });
        i = match.last + 1;
      } else {
        out.push({ kind: DROP.has(it.tok) ? 'drop' : 'text', text: it.tok, from: it.from, to: it.to });
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
