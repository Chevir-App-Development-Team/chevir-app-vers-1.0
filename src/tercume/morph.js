/**
 * Azərbaycan söz formalarını lüğətin işarələrinə gətirmək (mətn → jest).
 *
 * AzSLD cümlələrində işarə sözün lüğət formasıdır: "evdə" → EV, "anam" → ANA,
 * "yaşayıram" → YAŞAMAQ. İnkarlı fel isə ayrıca işarədir: "bilmirəm" → BİLMİR,
 * "düşmürəm" → DÜŞMƏMƏK. Ona görə açar kök + inkardır ("bil" və "bil-").
 *
 * Söz sağdan şəkilçilərə bölünür, şəkilçilərin sırası gözlənilir (kök + inkar + zaman /
 * məsdər + cəm + mənsubiyyət + hal + xəbərlik / şəxs + sual). Namizəd köklər uzundan
 * qısaya yoxlanır, lüğətin özündən çıxan köklərdən ilki götürülür. Səs dəyişmələri
 * bərpa olunur: uşağ → uşaq, köynəy → köynək, ged → get, oğl → oğul, istəy → istə.
 */

const VOWELS = 'aıoueəiöü';
const BACK = 'aıou';

const v4 = (s) => ['ı', 'i', 'u', 'ü'].map((v) => s.replaceAll('I', v));   // dördvariantlı
const v2 = (s) => ['a', 'ə'].map((v) => s.replaceAll('A', v));             // ikivariantlı
const v2k = (a, b) => [a, b];
const v3 = (s) => s.length >= 3;

// Şəkilçinin yeri (kökdən uzaqlaşdıqca böyüyür). Bir şəkilçi bir neçə yerdə ola bilər.
// Sual şəkilçisi şəxsdən sonra da (bilərsinizmi), əvvəl də (bilirmisən) gəlir.
const TAM = 2, PL = 3, POSS = 4, CASE = 5, COP = 6, Q1 = 6.5, PERS = 7, Q = 8;
const SLOTS = new Map();
const put = (slot, ...forms) => forms.flat().forEach((f) => SLOTS.set(f, [...(SLOTS.get(f) ?? []), slot]));

// zaman, feli sifət, məsdər, bacarıq (verə bilər)
put(TAM, v4('Ir'), v4('yIr'), v4('Ib'), v4('yIb'), v4('mIş'), v4('dI'), ['dıq', 'dik', 'duq', 'dük'], v2k('dığ', 'diy'),
  v2k('duğ', 'düy'), v2k('acaq', 'əcək'), v2k('yacaq', 'yəcək'), v2k('acağ', 'əcəy'), v2k('yacağ', 'yəcəy'),
  v2k('acam', 'əcəm'), v2k('yacam', 'yəcəm'), v2('Ar'), v2('yAr'), v2k('malı', 'məli'), v2('sA'), v2('ysA'),
  v2k('araq', 'ərək'), v2k('yaraq', 'yərək'), v2('An'), v2('yAn'), v2k('maz', 'məz'), v2k('mar', 'mər'),
  v2k('maq', 'mək'), v2k('mağ', 'məy'), v2('A'), v2('yA'));
put(PL, v2k('lar', 'lər'));
put(POSS, v4('Im'), 'm', v4('In'), 'n', v4('I'), v4('sI'), v4('ImIz'), v4('mIz'), v4('InIz'), v4('nIz'),
  v2k('ları', 'ləri'));
put(CASE, v4('In'), v4('nIn'), v2('A'), v2('yA'), v2('nA'), v4('I'), v4('nI'), v4('yI'), v2('dA'), v2('ndA'),
  v2('dAn'), v2('ndAn'), v2('lA'), v2('ylA'), v2k('dakı', 'dəki'), 'ki');
put(COP, v4('dIr'), v4('dI'), v4('ydI'), v4('ymIş'));
put(PERS, v2('Am'), v2('yAm'), ['ıq', 'ik', 'uq', 'ük', 'yıq', 'yik', 'yuq', 'yük'], v2('sAn'),
  v4('sInIz'), v4('sIz'), v2k('lar', 'lər'), 'm', 'n', 'q', 'k', v4('nIz'));
put(Q1, v4('mI'));
put(Q, v4('mI'));

const BY_LEN = [...SLOTS.keys()].sort((a, b) => b.length - a.length);
const PRESENT = new Set(v4('Ir').concat(v4('yIr')));
// İnkarlı qeyri-qəti gələcək: gəl-məz, 1-ci şəxsdə gəl-mər-əm, gəl-mər-ik
const NEG_AORIST = new Set(['maz', 'məz', 'mar', 'mər']);
// Adla da üst-üstə düşən zaman şəkilçiləri (bazar, hava, insan): felin kökü yalnız
// bunlardan başqası ilə tanınır (sev-ir-əm, al-mış-am)
const WEAK_TAM = new Set([...v2('A'), ...v2('yA'), ...v2('Ar'), ...v2('yAr'), ...v2('An'), ...v2('yAn'),
  ...v2('sA'), ...v2('ysA')]);
// Köməkçi sözlər kəsilmir: ilə ≠ il, kimi ≠ kim
const FIXED = new Set(['ilə', 'kimi', 'ki', 'da', 'də', 'isə', 'belə', 'elə', 'hələ', 'yenə', 'deyə', 'nə']);

// Şəxs əvəzlikləri qaydaya sığmır (onun, ona; mənim, məni)
const PRONOUNS = {
  mən: ['mənim', 'məni', 'mənə', 'məndə', 'məndən', 'mənimlə', 'məncə'],
  sən: ['sənin', 'səni', 'sənə', 'səndə', 'səndən', 'səninlə'],
  o: ['onun', 'onu', 'ona', 'onda', 'ondan', 'onunla'],
  biz: ['bizim', 'bizi', 'bizə', 'bizdə', 'bizdən', 'bizimlə'],
  siz: ['sizin', 'sizi', 'sizə', 'sizdə', 'sizdən', 'sizinlə'],
  onlar: ['onların', 'onları', 'onlara', 'onlarda', 'onlardan', 'onlarla'],
  bu: ['bunun', 'bunu', 'buna', 'bunda', 'bundan', 'bununla'],
  bura: ['burda', 'burdan', 'burada', 'buradan', 'buraya'],
  ora: ['orda', 'ordan', 'orada', 'oradan', 'oraya'],
  hara: ['harda', 'hardan', 'harada', 'haradan', 'haraya'],
};
const PRONOUN_OF = new Map(Object.entries(PRONOUNS).flatMap(([p, forms]) => forms.map((f) => [f, p])));

const isVowel = (c) => VOWELS.includes(c);
const hasVowel = (s) => [...s].some(isVowel);
const lastVowel = (s) => [...s].reverse().find(isVowel);

// Xəbər (ucuzdur, yoxdur, məndədir) fel deyil: -dIr-dan əvvəlki söz
const COPULA = new Set(v4('dIr'));
const predicate = (w) => {
  const s = w.slice(0, -3);
  return COPULA.has(w.slice(-3)) && s.length >= 2 && hasVowel(s) ? s : null;
};
// Felin şəkilçisi kökə ahənglə qoşulur, saitlə bitən kökə samitlə (oxu-yur, şe-ir yox, təşəkk-ür yox)
function fits(stem, rest) {
  if (isVowel(stem.at(-1)) && isVowel(rest[0])) return false;
  const v = lastVowel(stem), s = [...rest].find(isVowel);
  if (!v || !s) return true;
  if ('ıiuü'.includes(s)) return s === ('aı'.includes(v) ? 'ı' : 'ou'.includes(v) ? 'u' : 'öü'.includes(v) ? 'ü' : 'i');
  if ('aə'.includes(s)) return (s === 'a') === BACK.includes(v);
  return true;
}

/** Məsdər şəkilçisi ahəngə görə: yaşa → yaşamaq, get → getmək. */
export const infinitive = (stem) => stem + (BACK.includes(lastVowel(stem) ?? 'a') ? 'maq' : 'mək');

/** İnkarlı açar "bil-" kimi yazılır. */
export const keyOf = (stem, neg) => (neg ? `${stem}-` : stem);

/**
 * Sözün mümkün kökləri, uzundan qısaya: [{stem, neg, verb, strong, chain}]. `neg` — kökdən
 * sonra inkar gəlir (bil-mə-di, bil-m-ir, bil-məz); `verb` — zəncirdə zaman/məsdər şəkilçisi
 * var, `strong` — o şəkilçi adla qarışmır; `chain` — soyulmuş hərflərin sayı.
 */
export function candidates(word) {
  const out = new Map();
  const add = (stem, neg, verb, strong = false) => {
    const k = keyOf(stem, neg);
    const prev = out.get(k);
    if (!prev || (verb && !prev.verb) || (strong && !prev.strong)) {
      out.set(k, { stem, neg, verb, strong, chain: word.length - stem.length });
    }
  };
  const pron = PRONOUN_OF.get(word);
  if (pron) add(pron, false, false);
  add(word, false, false);
  if (FIXED.has(word)) return [...out.values()];
  const walk = (form, below, verb) => {
    for (const suf of BY_LEN) {
      if (!form.endsWith(suf) || form.length - suf.length < 2) continue;
      const rest = form.slice(0, -suf.length);
      if (!hasVowel(rest)) continue;
      for (const slot of SLOTS.get(suf)) {
        if (slot >= below) continue;
        if (slot === TAM) {
          // indiki zamandan sonra mənsubiyyət və hal gəlmir: işt-ir-ak, end-ir-im, tədb-ir-inə fel deyil
          if (PRESENT.has(suf) && (below === POSS || below === CASE)) continue;
          const strong = !WEAK_TAM.has(suf);
          if (NEG_AORIST.has(suf)) { add(rest, true, true, true); continue; }
          add(rest, false, true, strong);
          // İnkar zaman şəkilçisindən əvvəldir: bil+mə+di; ır-dan əvvəl sait düşür: bil+m+ir
          for (const n of PRESENT.has(suf) ? ['m'] : ['ma', 'mə']) {
            const stem = rest.slice(0, -n.length);
            if (rest.endsWith(n) && stem.length >= 2 && hasVowel(stem)) add(stem, true, true, true);
          }
        } else {
          add(rest, false, verb);
          walk(rest, slot, verb);
        }
      }
    }
  };
  walk(word, Q + 1, false);
  return [...out.values()].sort((a, b) => b.stem.length - a.stem.length || b.neg - a.neg);
}

// Sıra sayının şəkilçisi rəqəmdən sonra defislə yazılır: 2-ci, 6-cı, 2018-ci
const ORDINAL = new Set([...v4('IncI'), ...v4('ncI'), ...v4('cI')]);

/**
 * Defisdən sonrakı hissə yalnız şəkilçidirmi (6-dan, TSEK-ə, 2-ci, 12-də)? Onda işarə defisdən
 * əvvəlki hissədir; yoxsa iki ayrı sözdür (yazıb-oxuya).
 */
export function suffixOnly(s) {
  if (!s || !hasVowel(s)) return false;
  if (ORDINAL.has(s)) return true;
  // istənilən kök: şəkilçilər soyulub "ev"ə enməlidir (zəncir fel kimi də oxuna bilər: ev-ə)
  return candidates(`ev${s}`).some((c) => c.stem === 'ev' && !c.neg);
}

/**
 * Kökün səs dəyişməsindən əvvəlki formaları: uşağ → uşaq, istəy → istə, felin ged → get,
 * adın oğl → oğul. `verb` — kök fel şəkilçisindən ayrılıb (yadımda ≠ yat-).
 */
export function variants(stem, verb = false) {
  const out = [stem];
  const last = stem.at(-1), prev = stem.at(-2) ?? '';
  if (last === 'ğ') out.push(stem.slice(0, -1) + 'q');
  if (last === 'y' && isVowel(prev)) out.push(stem.slice(0, -1) + 'k', stem.slice(0, -1));
  if (last === 'd' && verb) out.push(stem.slice(0, -1) + 't');
  if (!verb && stem.length >= 3 && !isVowel(last) && !isVowel(prev)) {
    const v = lastVowel(stem);
    const ins = 'aı'.includes(v) ? 'ı' : 'ou'.includes(v) ? 'u' : 'öü'.includes(v) ? 'ü' : 'i';
    out.push(stem.slice(0, -1) + ins + last);
  }
  return out;
}

/**
 * Lüğətin sözlərindən köklər və hər sözün açarı.
 *  - məsdər: gövdəsi açardır (yaşamaq → "yaşa", düşməmək → "düş-");
 *  - felin təsrifli forması (istəyirəm, bilmir, almışam): felin açarı ("istə", "bil-"); fel
 *    lüğətdə məsdərlə yoxdursa, kökü belə formadan götürülür (sevirəm → "sev");
 *  - başqa sözün uzun şəkilçili forması (mənim ← mən): o sözün açarı; şəkilçi zənciri
 *    ən azı 2 hərf, kök ən azı 3 hərf olmalıdır, yoxsa "ata" "at"-a enərdi;
 *  - qalan sözlər özləri açardır.
 */
export class Lemmatizer {
  /** predicates: xəbərin açarı -dIr-dan əvvəlki sözdür (ucuzdur → ucuz; mətn → işarə üçün). */
  constructor(words, { predicates = false } = {}) {
    this.verbs = new Set();
    const plain = [], infinitives = new Map();
    for (const w of new Set(words)) {
      if (w.includes(' ')) continue;
      const m = /^(.+?)(maq|mək)$/.exec(w);
      if (m && m[1].length >= 2 && hasVowel(m[1])) {
        const n = /^(.+?)(ma|mə)$/.exec(m[1]);
        const key = n && n[1].length >= 2 && hasVowel(n[1]) ? keyOf(n[1], true) : m[1];
        this.verbs.add(key.replace(/-$/, ''));
        infinitives.set(w, key);
      } else {
        plain.push(w);
      }
    }
    // Məsdəri lüğətdə olmayan fellər: sevirəm, sevmirəm → "sev". Məlum felə uyğun gələn
    // forma (işləmir → işlə) yeni kök yaratmır; yoxsa ən qısa etibarlı gövdə götürülür.
    for (const w of plain) {
      if (predicate(w)) continue;
      const strong = candidates(w).filter((x) => x.strong && x.stem !== w);
      if (!strong.length || strong.some((x) => variants(x.stem, true).some((v) => this.verbs.has(v)))) continue;
      // yeni kök: ən azı üç hərf, şəkilçi ona uyğundur (fikir, uğur, təşəkkür fel deyil)
      const ok = strong.filter((x) => x.stem.length >= 3 && fits(x.stem, w.slice(x.stem.length)));
      if (ok.length) this.verbs.add(ok.at(-1).stem);
    }
    this.roots = new Set([...this.verbs, ...infinitives.values()]);
    this.keyOfWord = new Map(infinitives);       // lüğətin sözü → açar
    const nouns = new Set(plain);
    for (const w of plain) {
      let key = w;
      const p = predicate(w);
      for (const c of candidates(w)) {
        if (c.stem === w) continue;
        const hit = c.verb && variants(c.stem, true).find((v) => this.verbs.has(v));
        if (hit) { key = keyOf(hit, c.neg); break; }
        if (p) continue;
        const noun = variants(c.stem).find((v) => nouns.has(v) && v !== w);
        if (noun && c.chain >= 2 && v3(noun)) { key = noun; break; }
      }
      // xəbərin açarı -dIr-dan əvvəlki sözdür: ucuzdur → ucuz, məndədir → mən (ged-ir isə feldir)
      if (key === w && p && predicates) key = PRONOUN_OF.get(p) ?? p;
      this.keyOfWord.set(w, key);
    }
    // Açar özü lüğətin başqa açarlı sözüdürsə, onun açarı: bilərsinizmi → bilər → bil
    for (const [w, key] of this.keyOfWord) {
      let k = key;
      for (let n = 0; n < 3 && nouns.has(k) && this.keyOfWord.get(k) !== k; n++) k = this.keyOfWord.get(k);
      this.keyOfWord.set(w, k);
      this.roots.add(k);
    }
  }

  /**
   * Söz lüğət formasıdırmı (kök ad, məsdər)? Eyni açarlı sözlərdən hansını göstərmək üçün.
   * `analyse` — söz lüğətin tək sözü deyilsə (ifadənin sözü), açarı təhlillə tapılır.
   */
  isBase(word, analyse = false) {
    const key = this.keyOfWord.get(word) ?? (analyse ? this.key(word) ?? undefined : undefined);
    if (key === undefined) return false;
    if (key === word) return true;
    const stem = key.replace(/-$/, '');
    const neg = key.endsWith('-') ? (BACK.includes(lastVowel(stem)) ? 'ma' : 'mə') : '';
    return word === infinitive(stem + neg);
  }

  /**
   * İfadənin sözü üçün açar ("xoşum gəlir" ~ "xoşu gəlmək"): məlum kök yoxdursa, şəkilçisiz
   * ən qısa gövdə — ifadə bütöv uyğun gəlməlidir, ona görə təklikdə olduğundan sərbəstdir.
   */
  phraseKey(word) {
    return this.key(word) ?? candidates(word).filter((c) => !c.neg).at(-1).stem;
  }

  /**
   * Sözün açarı: məlum köklərdən ən uzunu, yoxsa null. Şəkilçisiz qısa fel kökü fel sayılmır
   * ("it" heyvandır, "itmək" deyil; "at" — "atmaq" deyil).
   *
   * İnkarlı fel (kökü məlum feldir) inkarlı açar alır, lüğətdə inkarlı işarəsi olmasa da:
   * "oxumadım" → "oxu-". Yoxsa mənasız bölünmə (oxu + m + a + dı + m) müsbət felə, OXUMAQ-a çıxırdı;
   * işarəsi olmayan inkarı lüğət özü həll edir (WordLexicon.lookup: fel + YOX).
   */
  key(word) {
    if (this.keyOfWord.has(word)) return this.keyOfWord.get(word);
    const cands = candidates(word);
    const neg = cands.find((c) => c.neg && variants(c.stem, true).some((v) => this.verbs.has(v)));
    if (neg) {
      const verbs = variants(neg.stem, true).filter((v) => this.verbs.has(v));
      for (const v of verbs) {
        const k = keyOf(v, true);
        if (this.roots.has(k)) return k;
        if (this.keyOfWord.has(k)) return this.keyOfWord.get(k);
      }
      return keyOf(verbs[0], true);
    }
    for (const { stem, neg: n, verb, strong, chain } of cands) {
      for (const v of variants(stem, verb)) {
        // Bir hərfli kök yoxdur (iyun ≠ i), əvəzliyin özündən başqa (onun → o)
        if (v.length < 2 && chain > 0 && PRONOUN_OF.get(word) !== v) continue;
        // Felin səs dəyişməsi (ged → get) fel köküne aparmalıdır: kar-dır ≠ kart (kard + ır)
        if (v !== stem && strong && !this.verbs.has(v)) continue;
        const k = keyOf(v, n);
        if (chain === 0 && v === word && word.length < 4 && this.verbs.has(k)) continue;
        if (this.roots.has(k)) return k;
        if (this.keyOfWord.has(k)) return this.keyOfWord.get(k);
      }
    }
    return null;
  }
}
