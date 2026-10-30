/**
 * Azərbaycan sözünün formalarını qurmaq (jest → mətn): kök + qrammatik əlamətlər → söz.
 *
 * morph.js-in tərsidir: orada söz şəkilçilərə bölünür (evdə → ev), burada kökə şəkilçi qoşulur
 * (ev + yerlik → evdə). Ahəng saitləri: I — ı/i/u/ü, A — a/ə; K — q/k, G — ğ/y (qalın/incə).
 * Bitişdirici samitlər (y, n, s) və kökün səs dəyişməsi (kömək → köməyə, get → gedir) qaydadır.
 * Yalnız gündəlik danışığın formaları: AzSLD cümlələrinin sözlərinin çoxu bunlarla qurulur
 * (tools/eval_gloss_text.mjs ölçür).
 */

const VOWELS = 'aıoueəiöü';
const BACK = 'aıou';
const isVowel = (c) => VOWELS.includes(c);
const lastVowel = (s) => [...s].reverse().find(isVowel) ?? 'a';
const syllables = (s) => [...s].filter(isVowel).length;

/** Ahəng: I, A, K, G hərfləri özlərindən əvvəlki son saitə görə açılır. */
export function harmonize(stem, suffix) {
  let out = stem;
  for (const ch of suffix) {
    const v = lastVowel(out);
    const back = BACK.includes(v);
    if (ch === 'I') out += 'aı'.includes(v) ? 'ı' : 'ou'.includes(v) ? 'u' : 'öü'.includes(v) ? 'ü' : 'i';
    else if (ch === 'A') out += back ? 'a' : 'ə';
    else if (ch === 'K') out += back ? 'q' : 'k';
    else if (ch === 'G') out += back ? 'ğ' : 'y';
    else out += ch;
  }
  return out;
}

// Sait ilə başlayan şəkilçidən əvvəl kökün son samiti dəyişir: çoxhecalı adda k → y, q → ğ
// (kömək → köməyə, uşaq → uşağa); fellərdən get, et → ged, ed (gedir, edəcək)
const VERB_ALT = { get: 'ged', et: 'ed' };
function joinVowel(stem, verb) {
  if (verb) return VERB_ALT[stem] ?? stem;
  if (syllables(stem) > 1 && stem.endsWith('k')) return stem.slice(0, -1) + 'y';
  if (syllables(stem) > 1 && stem.endsWith('q')) return stem.slice(0, -1) + 'ğ';
  return stem;
}

/** Kökə şəkilçi: sait ilə başlayan şəkilçi saitlə bitən kökə `buffer` ilə (y, n, s) qoşulur. */
function attach(stem, suffix, { verb = false, buffer = 'y' } = {}) {
  if (!suffix) return stem;
  const startsVowel = 'IA'.includes(suffix[0]) || isVowel(suffix[0]);
  if (!startsVowel) return harmonize(stem, suffix);
  if (isVowel(stem.at(-1))) return buffer === null ? harmonize(stem, suffix.slice(1)) : harmonize(stem, buffer + suffix);
  return harmonize(joinVowel(stem, verb), suffix);
}

// ── adlar ──
// Şəxs əvəzlikləri qaydaya sığmır (mən → mənim, mənə)
const PRONOUN = {
  mən: { gen: 'mənim', acc: 'məni', dat: 'mənə', loc: 'məndə', abl: 'məndən', ins: 'mənimlə' },
  sən: { gen: 'sənin', acc: 'səni', dat: 'sənə', loc: 'səndə', abl: 'səndən', ins: 'səninlə' },
  o: { gen: 'onun', acc: 'onu', dat: 'ona', loc: 'onda', abl: 'ondan', ins: 'onunla' },
  biz: { gen: 'bizim', acc: 'bizi', dat: 'bizə', loc: 'bizdə', abl: 'bizdən', ins: 'bizimlə' },
  siz: { gen: 'sizin', acc: 'sizi', dat: 'sizə', loc: 'sizdə', abl: 'sizdən', ins: 'sizinlə' },
  onlar: { gen: 'onların', acc: 'onları', dat: 'onlara', loc: 'onlarda', abl: 'onlardan', ins: 'onlarla' },
  bu: { gen: 'bunun', acc: 'bunu', dat: 'buna', loc: 'bunda', abl: 'bundan', ins: 'bununla' },
  bura: { dat: 'bura', loc: 'burda', abl: 'burdan' },
  ora: { dat: 'ora', loc: 'orda', abl: 'ordan' },
  hara: { dat: 'hara', loc: 'harda', abl: 'hardan' },
};
export const PRONOUNS = Object.keys(PRONOUN);

const POSS = { 1: 'Im', 2: 'In', 3: 'I', 4: 'ImIz', 5: 'InIz' };     // 1sg 2sg 3sg 1pl 2pl
const CASE = { gen: 'In', acc: 'I', dat: 'A', loc: 'dA', abl: 'dAn', ins: 'lA' };
const COP = { 1: 'Am', 3: 'dIr', 4: 'IK', 5: 'sInIz', 2: 'sAn', 6: 'dIrlAr' };

/**
 * Ad: {pl, poss (1-5), case, cop (1-6, xəbərlik), q}. Nümunə: noun('ata', {poss: 1}) → atam,
 * noun('ev', {case: 'loc'}) → evdə, noun('ata', {poss: 3, case: 'gen'}) → atasının.
 */
export function noun(stem, f = {}) {
  if (PRONOUN[stem] && !f.pl && !f.poss) {
    let w = f.case ? PRONOUN[stem][f.case] : stem;
    if (!w) return null;
    if (f.cop) w = attach(w, COP[f.cop]);
    return f.q ? harmonize(w, 'mI') : w;
  }
  let w = stem;
  if (f.pl) w = harmonize(w, 'lAr');
  let pron = false;                                      // 3-cü şəxs mənsubiyyətdən sonra hal "n" ilə
  if (f.poss) {
    const p = POSS[f.poss];
    if (isVowel(w.at(-1))) w = f.poss === 3 ? harmonize(w, 'sI') : harmonize(w, p.slice(1));
    else w = attach(w, p);
    pron = f.poss === 3;
  }
  if (f.case) {
    const c = CASE[f.case];
    if (pron) w = harmonize(w, f.case === 'ins' ? 'lA' : 'n' + c);
    else if (isVowel(w.at(-1)) && f.case === 'gen') w = harmonize(w, 'nIn');
    else if (isVowel(w.at(-1)) && f.case === 'acc') w = harmonize(w, 'nI');
    else if (f.case === 'ins') w = harmonize(w, isVowel(w.at(-1)) ? 'ylA' : 'lA');
    else w = attach(w, c);
  }
  if (f.cop) w = f.cop === 1 || f.cop === 4 ? attach(w, COP[f.cop]) : harmonize(w, COP[f.cop]);
  if (f.q) w = harmonize(w, 'mI');
  return w;
}

// ── fellər ──
// Şəxs sonluqları: z-tipi (indiki, nəqli, gələcək, lazımlıq, qeyri-qəti) və k-tipi (keçmiş, şərt)
const PERS_Z = { 1: 'Am', 2: 'sAn', 3: '', 4: 'IK', 5: 'sInIz', 6: 'lAr', 7: 'sIz' };   // 7: danışıq 2pl (işləyirsiz)
const PERS_K = { 1: 'm', 2: 'n', 3: '', 4: 'K', 5: 'nIz', 6: 'lAr', 7: 'nIz' };

/**
 * Fel: {neg, tam, pers (1-6; 7 — danışıqdakı -sIz), q}. tam: pres (gəlir), past (gəldi), perf (gəlmiş),
 * ev (gəlib), fut (gələcək), futc (danışıq: gələcəm), aor (gələr), opt (gələ — "gələ bilər"), nec (gəlməli),
 * inf (gəlmək), cond (gəlsə), imp (gəl / gəlin), part (gələn). Nümunə: verb('yaşa', {tam: 'pres', pers: 1})
 * → yaşayıram, verb('get', {tam: 'fut', pers: 1}) → gedəcəyəm, verb('oxu', {neg: true, tam: 'past', pers: 1})
 * → oxumadım.
 */
export function verb(stem, f = {}) {
  const { neg = false, tam = 'pres', q = false } = f;
  const pers = f.pers ?? 3;
  let w = neg ? harmonize(stem, 'mA') : stem;
  const v = !neg;                                          // inkardan sonra kök artıq "-mA"dır
  let personal = PERS_Z;
  switch (tam) {
    case 'pres':
      w = neg ? harmonize(stem, 'mIr') : isVowel(w.at(-1)) ? harmonize(w, 'yIr') : attach(w, 'Ir', { verb: v });
      break;
    case 'past': w = harmonize(w, 'dI'); personal = PERS_K; break;
    case 'perf': w = harmonize(w, 'mIş'); break;
    case 'ev': w = attach(w, 'Ib', { verb: v }); break;
    case 'fut':
      w = attach(w, 'AcAK', { verb: v });
      if (pers === 1 || pers === 4) w = w.slice(0, -1) + harmonize(w.slice(0, -1), 'G').at(-1);
      break;
    case 'futc':                                           // gələcəm, alacam (danışıq)
      if (pers === 1) return finish(attach(w, 'AcAm', { verb: v }), q);
      w = attach(w, 'AcAK', { verb: v });
      break;
    case 'aor':
      if (neg) { w = harmonize(stem, pers === 1 || pers === 4 ? 'mAr' : 'mAz'); break; }   // gəlmərəm, gəlməz
      w = isVowel(w.at(-1)) ? harmonize(w, 'yAr') : attach(w, 'Ar', { verb: v });
      break;
    case 'opt': return finish(attach(w, 'A', { verb: v }), q);
    case 'nec': w = harmonize(w, 'mAlI'); if (pers === 3) return finish(harmonize(w, 'dIr'), q); break;
    case 'inf': return finish(harmonize(w, 'mAK'), q);
    case 'cond': w = harmonize(w, 'sA'); personal = PERS_K; break;
    case 'imp': return finish(pers === 5 || pers === 7 ? attach(w, 'In', { verb: v }) : w, q);
    case 'part': return finish(attach(w, 'An', { verb: v }), q);
    default: return null;
  }
  const end = personal[pers];
  if (end) w = isVowel(w.at(-1)) && 'IA'.includes(end[0]) ? harmonize(w, 'y' + end) : harmonize(w, end);
  return finish(w, q);
}
const finish = (w, q) => (q ? harmonize(w, 'mI') : w);

// Analiz üçün formaların siyahısı (sintezlə analiz: söz bu formalardan biridirsə, əlamətləri bunlardır)
const NOUN_FEATS = [];
for (const pl of [false, true]) for (const poss of [0, 1, 2, 3, 4, 5]) {
  for (const c of [null, 'gen', 'acc', 'dat', 'loc', 'abl', 'ins']) for (const cop of [0, 1, 3, 5]) {
    for (const q of [false, true]) NOUN_FEATS.push({ ...(pl && { pl }), ...(poss && { poss }), ...(c && { case: c }), ...(cop && { cop }), ...(q && { q }) });
  }
}
const VERB_FEATS = [];
for (const neg of [false, true]) for (const tam of ['pres', 'past', 'perf', 'ev', 'fut', 'futc', 'aor', 'nec', 'cond']) {
  for (const pers of [1, 2, 3, 4, 5, 6, 7]) for (const q of [false, true]) VERB_FEATS.push({ neg, tam, pers, q });
}
for (const neg of [false, true]) for (const tam of ['opt', 'inf', 'part']) VERB_FEATS.push({ neg, tam, pers: 3, q: false });
for (const pers of [3, 5, 7]) VERB_FEATS.push({ neg: false, tam: 'imp', pers, q: false });

/** Kökün bütün formaları: [{form, feats, pos}]. */
export function paradigm(stem, pos) {
  const out = [];
  if (pos !== 'verb') for (const f of NOUN_FEATS) { const w = noun(stem, f); if (w) out.push({ form: w, feats: f, pos: 'noun' }); }
  if (pos !== 'noun') for (const f of VERB_FEATS) { const w = verb(stem, f); if (w) out.push({ form: w, feats: f, pos: 'verb' }); }
  return out;
}

/** Söz bu kökün hansı formasıdır? Ən sadə (ən az əlamətli) uyğunluq, yoxsa null. */
export function analyze(word, stem, pos) {
  const hits = paradigm(stem, pos).filter((p) => p.form === word);
  if (!hits.length) return null;
  return hits.sort((a, b) => Object.keys(a.feats).length - Object.keys(b.feats).length)[0];
}

/** Əlamətlərə görə söz. */
export const realize = (stem, pos, feats) => (pos === 'verb' ? verb(stem, feats) : noun(stem, feats));
