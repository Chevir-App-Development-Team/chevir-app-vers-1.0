/**
 * Barmaqla yığılan söz cümlədə (R6, E30): hərf ardıcıllığı (D O S T, M E S S İ N İ N) → mətndə bir söz.
 *
 * Cümlə modeli hərfləri tanıyır (E7); gloss2text onları bir sözə yığır, bu modul o sözün yazılışını seçir.
 * İşarə dilində hərflə əsasən adlar (Messi, Qəbələ, Sahil metrosu), qısaltmalar (DOST, TSEK, DSMF) və işarəsi
 * olmayan sözlər (manat, ədəbiyyat) göstərilir:
 *   1. AzSLD-nin mətnində bu hərflər necə yazılıbsa (öyrənilmiş cədvəl: DOST, Messinin, manat), elə.
 *   2. Ardıcıllıq iki tanış hissədir (A B Ş E R O N D O S T → Abşeron DOST): iki söz. Əvvəldəki O əvəzlik ola
 *      bilər (O N Ə V Ə S İ N İ N → o + nəvəsinin): qalanı tanışdırsa, bütövü isə yox.
 *   3. Tanınmayan ardıcıllıq bir hərf fərqi ilə öyrənilmiş sözə düzəlir (D S T → DOST), o söz ən azı iki
 *      cümlədə görünübsə və yeganədirsə.
 *   4. Yoxsa: xüsusi ad (yer adları, AzSLD-nin böyük hərfli sözləri: Bakı, Quba) — ad; qısa (≤ 5) və oxunmayan
 *      (saitsiz, üç samit yanaşı, iki samitlə başlayan: DSMF, ÜDSY, TSEK) — qısaltma, böyük hərflərlə; tanış söz
 *      (modelin sözləri və korpusun ümumi sözləri, ml/common_words.txt: manat, ədəbiyyat, onlayn) kiçik hərflə;
 *      qalanı ad, böyük hərflə. Ardınca adı olan söz gəlirsə (METRO, FİLM, UNİVERSİTET, KÜÇƏ), tanış söz də
 *      addır (Sahil metrosu, "Dolu" filmi).
 *   5. Hal (withCase): ad / qısaltmadan sonra halı idarə edən fel gəlirsə (GETMƏK → -a, YAŞAMAQ → -da, MÜRACİƏT
 *      ETMƏK → -a, SEVMƏK → -ı) və hərflərdə o şəkilçi yoxdursa: Messini, TSEK-ə. Qısaltmaya defislə, oxunuşuna
 *      görə: oxunursa sözün özü (TSEK-ə, DOST-a), yoxsa son hərfin adı (DSMF-ə: "ef", BMT-yə: "te").
 */
import { harmonize } from './inflect.js';
import { candidates } from './morph.js';
import { azLower } from './words.js';

const VOWELS = 'aıoueəiöü';
const isVowel = (c) => VOWELS.includes(c);
const hasVowel = (s) => [...s].some(isVowel);
const up = (w) => w.toLocaleUpperCase('az');
const cap = (w) => (w ? up(w[0]) + w.slice(1) : w);
// Annotasiyada İ çox vaxt ı yerinə yazılıb: tutuşdurmada ı = i (gloss2text.js norm)
export const norm = (w) => w.replace(/ı/g, 'i');

// Hərflərin adları (qısaltmanın oxunuşu: BMT — be-em-te, DSMF — de-es-em-ef)
const LETTER_NAME = { a: 'a', b: 'be', c: 'ce', ç: 'çe', d: 'de', e: 'e', ə: 'ə', f: 'ef', g: 'ge', ğ: 'ğe', h: 'he',
  x: 'xe', ı: 'ı', i: 'i', j: 'je', k: 'ka', q: 'qe', l: 'el', m: 'em', n: 'en', o: 'o', ö: 'ö', p: 'pe', r: 'er', s: 'es',
  ş: 'şe', t: 'te', u: 'u', ü: 'ü', v: 've', y: 'ye', z: 'ze' };

// Ardınca gələndə hərflə yığılan sözü ada çevirən sözlər (Sahil metrosu, ADA universiteti, "Dolu" filmi)
export const NAME_HEAD = new Set(['metro', 'film', 'kitab', 'universitet', 'mərkəz', 'küçə', 'prospekt', 'şəhər', 'rayon',
  'kənd', 'qəsəbə', 'məktəb', 'park', 'stansiya', 'mağaza', 'kafe', 'restoran', 'teatr', 'xəstəxana', 'poliklinika',
  'şirkət', 'bank', 'qəzet', 'jurnal', 'mahnı', 'serial', 'kino', 'muzey', 'otel', 'klinika', 'fond', 'akademiya',
  'institut', 'kollec', 'sayt', 'proqram', 'komanda', 'klub', 'stadion', 'liseyi', 'lisey', 'bağça', 'nazirlik']);

const clusters = (w) => /[^aıoueəiöü.]{3}/.test(w);
/** Qısaltmadır: qısa və söz kimi oxunmur (DSMF, ÜDSY, TSEK, BMT). */
export const isAcronym = (w) => [...w].length <= 5 && (!hasVowel(w) || clusters(w) || /^[^aıoueəiöü]{2}/.test(w));
/** Söz kimi oxunur (hal şəkilçisi sözün saitinə görə): TSEK, DOST; DSMF, ÜDSY yox. */
const readable = (w) => hasVowel(w) && !clusters(w);

const CASE_SUFFIX = { dat: ['yA', 'A'], acc: ['nI', 'I'], gen: ['nIn', 'In'], loc: ['dA', 'dA'], abl: ['dAn', 'dAn'] };
/** Sözə hal şəkilçisi (kök dəyişmir: adlarda k / q yumşalmır — İraqa). */
export function suffixFor(spoken, c) {
  const s = CASE_SUFFIX[c];
  if (!s) return '';
  return harmonize(spoken, isVowel(spoken.at(-1)) ? s[0] : s[1]).slice(spoken.length);
}

/**
 * Hərflərdə bu hal artıq varmı (Qəbələyə, Messini, Bakıda)? Hərflər = kök + o şəkilçi, şəkilçi ən azı iki hərf:
 * tək sait (Messi, Leyla) adın öz hərfi ola bilər.
 */
function hasCase(w, c) {
  return candidates(w).some((x) => x.stem.length >= 2 && w.length - x.stem.length >= 2 && x.stem + suffixFor(x.stem, c) === w);
}

/** Ada / qısaltmaya hal: Messini, Qəbələyə, TSEK-ə, DSMF-ə. Hərflərdə şəkilçi varsa dəyişmir. */
export function withCase(word, kind, c) {
  if (!c || kind === 'word') return word;
  const low = azLower(word);
  if (kind === 'acronym') {
    const spoken = readable(low) ? low : LETTER_NAME[low.at(-1)] ?? low;
    return `${word}-${suffixFor(spoken, c)}`;
  }
  if (hasCase(low, c)) return word;
  return word + suffixFor(low, c);
}

const lev1 = (a, b) => {                         // redaktə məsafəsi ≤ 1-dirmi
  if (a === b) return true;
  if (Math.abs(a.length - b.length) > 1) return false;
  let i = 0;
  while (i < a.length && a[i] === b[i]) i++;
  return a.slice(i + 1) === b.slice(i + 1) || a.slice(i) === b.slice(i + 1) || a.slice(i + 1) === b.slice(i);
};

/**
 * Öyrənmə (gloss2text prepare): cümlənin hərf ardıcıllıqları ↔ mətnin sözləri. Mətnin bir-dörd ardıcıl sözü
 * (durğusuz, defisdən sonrakı hal şəkilçisi atılmış: TSEK-ə → TSEK) hərflərlə eynidirsə, onların yazılışı.
 * Cümlənin ilk sözünün böyük hərfi heç nə demir: yalnız hamısı böyük hərfdirsə (DOST) yazılış sayılır.
 * runs: [hərflər (kiçik)] → [{run, surface | null}] (null: tutuşmadı və ya yazılışı bilinmir).
 */
export function alignRuns(runs, text) {
  const toks = text.split(/\s+/).map((t) => t.replace(/^[^\p{L}\d]+|[^\p{L}\d]+$/gu, '')).filter(Boolean)
    .map((t) => t.replace(/-[\p{Ll}]+$/u, ''));
  const keys = toks.map((t) => norm(azLower(t)).replace(/[^\p{L}]/gu, ''));
  return runs.map((run) => {
    const want = norm(run).replace(/[^\p{L}]/gu, '');
    for (let j = 0; j < toks.length; j++) {
      let acc = '';
      for (let k = j; k < Math.min(toks.length, j + 4) && keys[k]; k++) {      // hərfsiz söz (10) ardıcıllığı kəsir
        acc += keys[k];
        if (acc === want) {
          const ws = toks.slice(j, k + 1);
          const informative = j > 0 || (ws[0].length > 1 && ws[0] === up(ws[0]));
          return { run, surface: informative ? ws.join(' ') : null };
        }
        if (!want.startsWith(acc)) break;
      }
    }
    return { run, surface: null };
  });
}

/** Öyrənilmiş yazılışın növü: DOST — qısaltma, Messinin — ad, manat — söz. */
const kindOf = (s) => (s === up(s) && [...s].length <= 5 ? 'acronym' : s[0] !== azLower(s[0]) ? 'name' : 'word');

/**
 * Hərf ardıcıllığının yazılışı. table — öyrənilmiş {run: {n, forms: {yazılış: say}}}; known(w) — söz modelin
 * lüğətindədir (və ya şəkilçili formasıdır), proper(w) — xüsusi addır (Bakı, Abşeron); restore(w) — sözün
 * lüğətdəki yazılışı (cümlə modelində I hərfi yoxdur: B A K İ → Bakı); headNext — ardınca adı olan söz gəlir.
 * → [{word, kind: 'acronym' | 'name' | 'word' | 'pronoun', len}] (bir və ya iki söz; len — neçə hərf).
 */
export function spell(run, { table = {}, known = () => false, proper = () => false, restore = (w) => w,
  headNext = false } = {}) {
  const r = norm(run);
  const seen = (w) => table[norm(w)];
  const learned = (w) => {
    const e = seen(w);
    const best = e && Object.entries(e.forms ?? {}).sort((a, b) => b[1] - a[1])[0];
    return best ? { word: best[0], kind: kindOf(best[0]) } : null;
  };
  const rule = (w, head) => {
    const lw = azLower(w);
    if (proper(lw)) return { word: cap(restore(lw)), kind: 'name' };
    if (isAcronym(lw) && !known(lw)) return { word: up(lw), kind: 'acronym' };
    if (known(lw) && !head) return { word: restore(lw), kind: 'word' };
    return { word: cap(restore(lw)), kind: 'name' };
  };
  const one = (w, head) => ({ ...(learned(w) ?? rule(w, head)), len: [...w].length });
  // 1. öyrənilmiş
  if (learned(r)) return [one(r, headNext)];
  const familiar = (w) => w.length >= 2 && (seen(w) || known(w) || proper(w));
  if (!known(r) && !proper(r)) {
    // 2. iki tanış hissə; əvvəldəki "o" əvəzlikdir
    if (r.length > 2 && r[0] === 'o' && familiar(r.slice(1))) {
      return [{ word: 'o', kind: 'pronoun', len: 1 }, one(r.slice(1), headNext)];
    }
    for (let i = r.length - 3; i >= 3; i--) {
      const a = r.slice(0, i), b = r.slice(i);
      if ((seen(a) || seen(b)) && familiar(a) && familiar(b)) return [one(a, false), one(b, headNext)];
    }
    // 3. bir hərf fərqi ilə öyrənilmiş söz (bu hərflər özü görünməyibsə)
    if (!seen(r) && r.length >= 3) {
      const near = Object.keys(table).filter((k) => k.length >= 4 && (table[k].n ?? 0) >= 2 && lev1(r, k));
      if (near.length === 1) return [{ ...one(near[0], headNext), len: [...r].length }];
    }
  }
  // 4. qayda
  return [one(r, headNext)];
}
