/**
 * Söz işarələri lüğəti (tools/build_words.py → public/tercume/words/).
 *
 * İndeks bir dəfə yüklənir; sözün kadrları yalnız oynadılanda gətirilir.
 * Mətn sözlərə bölünür və ən uzun uyğunluq axtarılır — "zəhmət olmasa",
 * "bu gün" kimi ifadələr bütöv işarədir.
 */
import { WORDS_BASE } from './paths.js';
import { Lemmatizer, suffixOnly } from './morph.js';
import { paradigm } from './inflect.js';

/** Azərbaycan qaydası ilə kiçik hərf: I → ı, İ → i (toLowerCase "İ"-ni "i̇" edir). */
export const azLower = (s) => s.replace(/I/g, 'ı').replace(/İ/g, 'i').toLocaleLowerCase('az');

const EDGE_PUNCT = /^[.,!?;:"'“”‘’«»()[\]…–—-]+|[.,!?;:"'“”‘’«»()[\]…–—-]+$/g;

// Lüğətdə işarəsi olmayan köməkçi sözlər: AzSLD iştirakçıları onları göstərmir (ilə 0/6, idi 0/4,
// ki 0/2 cümlədə, ml/EXPERIMENTS.md E16), hərf-hərf göstərmək cümləni uzadır
const DROP = new Set(['ilə', 'idi', 'ki']);

// Rəqəmlər: lüğətdə öz işarəsi olmayan ədəd rəqəm-rəqəm göstərilir (25 → 2 5); işarəsi olmayan
// rəqəm (0, 7, 9) adı ilə hərf-hərf. Sözlə yazılan say və sıra sayı (beş, üçüncü, III) ədədin
// işarəsinə gedir, işarəsi varsa (yoxdursa söz özü kimi). "altı" yoxdur: masanın altı
const DIGIT_NAMES = ['sıfır', 'bir', 'iki', 'üç', 'dörd', 'beş', 'altı', 'yeddi', 'səkkiz', 'doqquz'];
const NUMBER_WORDS = new Map([
  ['üç', 3], ['dörd', 4], ['beş', 5], ['yeddi', 7], ['səkkiz', 8], ['doqquz', 9], ['on', 10], ['iyirmi', 20],
  ['otuz', 30], ['qırx', 40], ['əlli', 50], ['altmış', 60], ['yetmiş', 70], ['səksən', 80], ['doxsan', 90],
  ['yüz', 100], ['min', 1000], ['birinci', 1], ['ikinci', 2], ['üçüncü', 3], ['dördüncü', 4], ['beşinci', 5],
  ['altıncı', 6], ['yeddinci', 7], ['səkkizinci', 8], ['doqquzuncu', 9], ['onuncu', 10],
]);
const ROMAN = { I: 1, V: 5, X: 10 };

// Yiyəlik əvəzliyi işarədə əvəzliyin özüdür: AzSLD-də "mənim" olan 83 cümlənin hamısında MƏN (onun 11/11, sizin 11/11,
// heç birində MƏNİM, ONUN, SİZİN); işarə dili daşıyıcısı da belə deyir: MƏNİM işarəsi var, az işlənir (E21)
const POSSESSIVE = new Map([['mənim', 'mən'], ['sənin', 'sən'], ['onun', 'o'], ['bizim', 'biz'], ['sizin', 'siz'], ['onların', 'onlar']]);
const roman = (s) => [...s].reduce((sum, c, i) => sum + (ROMAN[c] < (ROMAN[s[i + 1]] ?? 0) ? -ROMAN[c] : ROMAN[c]), 0);

// Doğma işarə dilinin qaydaları (işarə dili daşıyıcısı ilə müsahibə, ml/EXPERIMENTS.md E21, E22). AzSLD bunları az
// izləyir, çünki cümlələr yazılı mətndən söz-söz göstərilib; `native: false` AzSLD üslubudur (ölçmə üçün).
//  - zaman ayrıca işarədir: keçmiş fel + OLMAQ, gələcək fel + OLACAQ (gəldim = GƏLMƏK OLMAQ); vaciblik fel + LAZIM
//    (daşıyıcının MÜTLƏQ-i lüğətdə yoxdur);
//  - buraxılmış I və II şəxs mübtədası əvəzliklə deyilir: gəldim = MƏN GƏLMƏK OLMAQ;
//  - "ilə" adamla BİRLİKDƏ-dir: anamla bazara getdim = ANA BİRLİKDƏ BAZAR GETMƏK (alətlə yox: qatarla ≠ QATAR BİRLİKDƏ;
//    "ilə bağlı" BAĞLI-dır).
const TENSE_SIGN = { past: 'olmaq', perf: 'olmaq', ev: 'olmaq', fut: 'olacaq', futc: 'olacaq', nec: 'lazım' };
const SUBJECT_OF = { 1: 'mən', 2: 'sən', 4: 'biz', 5: 'siz', 7: 'siz' };
const FINITE = new Set(['pres', 'past', 'perf', 'ev', 'fut', 'futc', 'aor', 'nec', 'cond']);
export const PERSON = new Set(['mən', 'sən', 'o', 'biz', 'siz', 'onlar', 'ana', 'ata', 'bacı', 'qardaş', 'dost', 'uşaq', 'oğul',
  'qız', 'ər', 'arvad', 'nənə', 'baba', 'əmi', 'dayı', 'xala', 'bibi', 'nəvə', 'övlad', 'valideyn', 'ailə', 'yoldaş',
  'həyat yoldaşı', 'adam', 'insan', 'şəxs', 'kişi', 'qadın', 'oğlan', 'körpə', 'həkim', 'müəllim', 'müəllimə', 'tələbə',
  'şagird', 'qonşu', 'qonaq', 'həmkar', 'müdir', 'rəhbər', 'tərcüməçi']);
const COMITATIVE = ['ylə', 'yla', 'lə', 'la'];
const SENTENCE_END = /[.!?…]["'»”)]*$/;

/** Əlavə işarə (YOX, zaman, mübtəda, BİRLİKDƏ): mətndə öz sözü yoxdur, from/to aid olduğu sözündür. */
export const isAdded = (seg) => seg.kind === 'word' && ['neg', 'tense', 'subject', 'with'].includes(seg.via);

export class WordLexicon {
  /** native: doğma işarə dilinin qaydaları (zaman, mübtəda, BİRLİKDƏ); false — AzSLD üslubu. */
  constructor(index, { native = true } = {}) {
    this.native = native;
    this.forms = new Map();
    this.entries = index?.words ?? [];
    // Lüğət üzrə ölçülər hər sözə qoşulur: üz blendshape-lərinin adları və diapazonu (avatarın
    // üz ifadəsi), iştirakçıların sinə xətti (sinəyə toxunan əl avatarın sinəsinə köçürülür)
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

  /** Felin forması {tam, pers, neg, root} (inflect.js paradiqması); fel deyilsə və ya forma tanınmırsa null. */
  #verb(tok) {
    const root = this.lemma.key(tok)?.replace(/-$/, '');
    if (!root || !this.lemma.verbs.has(root)) return null;
    if (!this.forms.has(root)) {
      const m = new Map();
      for (const p of paradigm(root, 'verb')) if (!m.has(p.form)) m.set(p.form, p.feats);
      this.forms.set(root, m);
    }
    const f = this.forms.get(root).get(tok);
    return f ? { ...f, root } : null;
  }

  #isPerson(entry) {
    const t = azLower(entry.text);
    return PERSON.has(t) || PERSON.has(this.lemma.key(t) ?? t);
  }

  /** Söz adamla "ilə"dir: anamla, dostlarımla, onunla, mənimlə (qatarla, ailə yox). */
  #withPerson(tok, entry) {
    const suf = COMITATIVE.find((s) => tok.endsWith(s));
    const base = suf && tok.slice(0, -suf.length);
    if (!base || base.length < 2 || !this.#isPerson(entry)) return false;
    const own = azLower(entry.text);
    return (POSSESSIVE.get(base) ?? this.lemma.key(base) ?? base) === (this.lemma.key(own) ?? own);
  }

  /** Əlavə işarə (YOX, zaman, BİRLİKDƏ, mübtəda): mətndə öz sözü yoxdur, `at` — aid olduğu söz. */
  #sign(text, via, at) {
    const e = this.byText.get(text);
    return e && { kind: 'word', text: e.text, entry: e, via, from: at.from, to: at.to };
  }

  /**
   * Söz formasına uyğun gələn lüğət sözü: əvvəl yazılışın özü, sonra kökü. Eyni kökdə bir
   * neçə söz varsa: mətndəki söz o sözün davamıdırsa, onu (istəyirsən → istəyir; bilmirəm →
   * bilmir), yoxsa lüğət formasını (olan → olmaq, olar yox), o da yoxdursa ən oxşarını.
   */
  lookup(tokens) {
    const text = tokens.join(' ');
    const pron = tokens.length === 1 && this.byText.get(POSSESSIVE.get(text));
    if (pron) return { entry: pron, via: 'lemma' };
    const exact = this.byText.get(text);
    if (exact) return { entry: exact, via: 'text' };
    let k = this.#keys(tokens);
    let via = 'lemma';
    // Törəmə söz (əlillik → əlil, verilir → ver): yalnız başqa yolla tapılmayanda
    const has = (key) => key && (this.byKey.has(key) || (key.endsWith('-') && this.byKey.has(key.slice(0, -1))));
    if (!has(k) && tokens.length === 1) {
      const d = this.lemma.derived(tokens[0]).find(has);
      if (d) { k = d; via = 'derived'; }
    }
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
    return neg ? { entry: best.entry, via, neg: true } : { entry: best.entry, via };
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
   * | {kind:'drop', text} — iştirakçıların göstərmədiyi köməkçi söz (oynadılmır).
   * `via`: 'text' — yazılış lüğətdədir, 'lemma' — sözün kökü (evdə → ev), 'derived' — törəmə sözün
   * kökü (əlillik → əlil, verilir → vermək), 'number' — ədəd rəqəmlərin işarələri ilə (25 → 2 5), 'neg' —
   * inkarlı felin öz işarəsi olmadığından felin işarəsindən sonra gələn YOX (oxumadım → OXUMAQ YOX).
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
      const start = pos;
      pos += p.length;
      // Söz içində vergül, iki nöqtə, nöqtəli vergül, çəp xətt: boşluqsuz yazılmış ayrı sözlər
      // (saatı,qrafiki) və vaxt (11:15 → 11 15, AzSLD iştirakçıları da belə göstərir)
      let off = 0;
      for (const s of p.split(/(?<=[\p{L}\d])[,;:/](?=[\p{L}\d])/u)) {
        const at = start + off;
        off += s.length + 1;
        const tok = azLower(s).replace(EDGE_PUNCT, '');
        const from = at + (tok ? azLower(s).indexOf(tok) : 0);
        const raw = text.slice(from, from + tok.length);
        // Defisli söz (6-dan, TSEK-ə, 2-ci, yazıb-oxuya): lüğətdə bütöv yoxdursa, defisdən sonrakı
        // şəkilçi atılır (işarə — rəqəm və ya ad; AzSLD iştirakçıları da TSEK-ə → TSEK göstərir),
        // şəkilçi deyilsə, hissələr ayrı sözlərdir
        const pieces = tok.split('-');
        if (pieces.length > 1 && !this.byText.has(tok) && pieces.every(Boolean)) {
          if (pieces.length === 2 && suffixOnly(pieces[1])) {
            const r = raw.slice(0, pieces[0].length);
            items.push({ tok: pieces[0], from, to: from + tok.length, acronym: isAcronym(r), raw: r });
          } else {
            let o = 0;
            for (const piece of pieces) {
              const r = raw.slice(o, o + piece.length);
              items.push({ tok: piece, from: from + o, to: from + o + piece.length, acronym: isAcronym(r), raw: r });
              o += piece.length + 1;
            }
          }
          continue;
        }
        items.push({ tok, from, to: from + tok.length, acronym: Boolean(tok) && isAcronym(raw), raw });
      }
      if (SENTENCE_END.test(p) && items.length) items.at(-1).end = true;
    }
    const out = [];
    // Cümlə bitəndə: felin I–II şəxsi var, əvəzliyi yoxdursa, əvəzlik cümlənin əvvəlinə (doğma qayda)
    let start = 0, subject = null;
    const close = () => {
      const pron = subject && SUBJECT_OF[subject.pers];
      const has = pron && out.slice(start).some((s) => s.kind === 'word' && azLower(s.entry.text) === pron);
      const sign = pron && !has && this.#sign(pron, 'subject', subject.at);
      if (sign) {
        let k = start;
        while (k < out.length && out[k].kind === 'space') k++;
        out.splice(k, 0, sign);
      }
      start = out.length;
      subject = null;
    };
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
        if (this.native) {
          const toks = items.slice(i, match.last + 1).filter((x) => x.tok).map((x) => x.tok);
          const v = this.#verb(toks.at(-1));
          const t = v && TENSE_SIGN[v.tam];
          const mark = t && !(v.root === 'ol' && t !== 'lazım') && this.#sign(t, 'tense', at);
          if (mark) out.push(mark);
          if (v && FINITE.has(v.tam) && SUBJECT_OF[v.pers] && !subject) subject = { pers: v.pers, at };
          const w = toks.length === 1 && this.#withPerson(toks[0], match.entry) && this.#sign('birlikdə', 'with', at);
          if (w) out.push(w);
        }
        const ends = items.slice(i, match.last + 1).some((x) => x.end);
        i = match.last + 1;
        if (ends) close();
      } else {
        // "ilə" adamdan sonra BİRLİKDƏ-dir ("ilə bağlı" deyilsə); adı hərf-hərf yığılan adamla da (Leylayla)
        const next = items.slice(i + 1).find((x) => x.tok)?.tok;
        const prev = [...out].reverse().find((s) => s.kind !== 'space');
        const pi = items.slice(0, i).reverse().find((x) => x.tok);
        const isName = (x) => x && /^\p{Lu}/u.test(x.raw ?? '') && !x.acronym;
        const afterPerson = prev && (prev.kind === 'word' ? this.#isPerson(prev.entry) : prev.kind === 'text' && isName(pi));
        const first = !out.slice(start).some((s) => s.kind !== 'space');      // cümlənin ilk sözü ad olmaya da bilər
        const w = this.native && it.tok === 'ilə' && next !== 'bağlı' && afterPerson && this.#sign('birlikdə', 'with', it);
        if (w) out.push(w);
        else out.push(...(this.#number(it) ?? [{ kind: DROP.has(it.tok) ? 'drop' : 'text', text: it.tok, from: it.from, to: it.to }]));
        const name = this.native && !first && isName(it) && it.tok !== 'ilə' && COMITATIVE.some((c) => it.tok.endsWith(c));
        const wn = name && this.#sign('birlikdə', 'with', it);
        if (wn) out.push(wn);
        if (it.end) close();
        i++;
      }
    }
    close();
    return out;
  }

  /**
   * Lüğətdə olmayan ədəd: rəqəm-rəqəm işarələr (via 'number'), işarəsi olmayan rəqəm adı ilə
   * hərf-hərf ({kind:'text', text:'yeddi'}). Sözlə yazılan say yalnız ədədin işarəsi varsa. Ədəd deyilsə null.
   */
  #number(it) {
    const at = { from: it.from, to: it.to };
    const sign = (n) => { const e = this.byText.get(n); return e && { kind: 'word', text: e.text, entry: e, via: 'number', ...at }; };
    let n = /^\d+$/.test(it.tok) ? it.tok : null;
    if (!n && /^[IVX]{2,}$/.test(it.raw ?? '')) n = String(roman(it.raw));
    if (!n && NUMBER_WORDS.has(it.tok)) {
      const w = sign(String(NUMBER_WORDS.get(it.tok)));
      return w ? [w] : null;
    }
    if (!n) return null;
    const whole = sign(n);
    if (whole) return [whole];
    return [...n].map((d) => sign(d) ?? { kind: 'text', text: DIGIT_NAMES[d], ...at });
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
