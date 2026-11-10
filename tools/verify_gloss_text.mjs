// İşarələr → mətn sınağı: söz formalarının qurulması (inflect.js), kiçik süni cümlələrdə öyrənmə və
// uzlaşma (gloss2text.js), brauzerin modeli (public/tercume/assets/gloss_text.json) hər cümlədə işləyir,
// inflect.js-in qurduğu formalar morph.js-də öz kökünə qayıdır.
//   node tools/verify_gloss_text.mjs
import fs from 'node:fs';
import { noun, verb, analyze, paradigm } from '../src/tercume/inflect.js';
import { train, prepare, fit, GlossToText } from '../src/tercume/gloss2text.js';
import { WordLexicon } from '../src/tercume/words.js';
import { keyOf } from '../src/tercume/morph.js';

let fails = 0;
const check = (ok, msg) => { console.log(`${ok ? '✓' : '✗'} ${msg}`); if (!ok) fails++; };

// 1. Formalar: ahəng, bitişdirici samitlər, kökün səs dəyişməsi, şəxs, inkar
const forms = [
  [verb('yaşa', { tam: 'pres', pers: 1 }), 'yaşayıram'], [verb('gəl', { tam: 'pres', pers: 1 }), 'gəlirəm'],
  [verb('get', { tam: 'pres', pers: 1 }), 'gedirəm'], [verb('oxu', { tam: 'pres', pers: 3 }), 'oxuyur'],
  [verb('işlə', { tam: 'pres', pers: 7 }), 'işləyirsiz'], [verb('yaşa', { tam: 'pres', pers: 4 }), 'yaşayırıq'],
  [verb('oxu', { neg: true, tam: 'past', pers: 1 }), 'oxumadım'], [verb('oxu', { neg: true, tam: 'pres', pers: 1 }), 'oxumuram'],
  [verb('gəl', { neg: true, tam: 'fut', pers: 1 }), 'gəlməyəcəyəm'], [verb('al', { tam: 'fut', pers: 1 }), 'alacağam'],
  [verb('get', { tam: 'futc', pers: 1 }), 'gedəcəm'], [verb('bil', { tam: 'aor', pers: 5, q: true }), 'bilərsinizmi'],
  [verb('gəl', { tam: 'opt' }), 'gələ'], [verb('oxu', { tam: 'opt' }), 'oxuya'], [verb('et', { tam: 'opt' }), 'edə'],
  [verb('gəl', { tam: 'nec', pers: 1 }), 'gəlməliyəm'], [verb('ol', { tam: 'ev' }), 'olub'], [verb('al', { tam: 'perf', pers: 1 }), 'almışam'],
  [verb('ver', { tam: 'imp', pers: 5 }), 'verin'], [verb('ol', { tam: 'part' }), 'olan'], [verb('al', { tam: 'past', pers: 4 }), 'aldıq'],
  [verb('düş', { neg: true, tam: 'inf' }), 'düşməmək'],
  [verb('gəl', { neg: true, tam: 'aor', pers: 1 }), 'gəlmərəm'], [verb('bil', { neg: true, tam: 'aor', pers: 4 }), 'bilmərik'],
  [verb('al', { neg: true, tam: 'aor', pers: 5 }), 'almazsınız'],
  [noun('ata', { poss: 1 }), 'atam'], [noun('ev', { case: 'loc' }), 'evdə'], [noun('ata', { poss: 3, case: 'gen' }), 'atasının'],
  [noun('kömək', { case: 'dat' }), 'köməyə'], [noun('uşaq', { case: 'gen' }), 'uşağın'], [noun('mən', { case: 'gen' }), 'mənim'],
  [noun('paytaxt', { poss: 3, cop: 3 }), 'paytaxtıdır'], [noun('subay', { cop: 1 }), 'subayam'], [noun('ad', { poss: 5 }), 'adınız'],
  [noun('sənəd', { pl: true, case: 'acc' }), 'sənədləri'], [noun('ata', { case: 'ins' }), 'atayla'], [noun('kitab', { poss: 1, case: 'dat' }), 'kitabıma'],
];
const bad = forms.filter(([got, want]) => got !== want);
check(!bad.length, `${forms.length - bad.length}/${forms.length} söz forması düzgündür${bad.length ? ': ' + bad.map(([g, w]) => `${g} (${w})`).join(', ') : ''}`);
check(JSON.stringify(analyze('gedəcəyəm', 'get', 'verb')?.feats) === JSON.stringify({ neg: false, tam: 'fut', pers: 1, q: false }),
  'analiz sintezlə: gedəcəyəm = get + gələcək zaman + 1-ci şəxs');

// 2. Öyrənmə və uzlaşma: süni cümlələr
const toy = train([
  { glosses: ['MƏN', 'BAKI', 'YAŞAMAQ'], text: 'Mən Bakıda yaşayıram' },
  { glosses: ['O', 'GƏNCƏ', 'YAŞAMAQ'], text: 'O Gəncədə yaşayır' },
  { glosses: ['MƏN', 'ATA', 'İŞLƏMƏK'], text: 'Mənim atam işləyir' },
  { glosses: ['SİZ', 'ANA', 'İŞLƏMƏK'], text: 'Sizin ananız işləyir' },
  { glosses: ['MƏN', 'OXUMAQ', 'BİLMƏK'], text: 'Mən oxuya bilərəm' },
  { glosses: ['O', 'YAZMAQ', 'BİLMƏK'], text: 'O yaza bilər' },
  { glosses: ['MƏN', 'EV', 'GETMƏK'], text: 'Mən evimə gedirəm' },
], undefined, undefined, { BAKI: 'Bakı' });
const g = new GlossToText(toy);
const cases = [
  ['SİZ ŞUŞA YAŞAMAQ', 'Siz Şuşada yaşayırsınız'],    // görünməmiş ad: felin istədiyi hal, yer adı böyük hərflə (E27); fel mübtədanın şəxsində
  ['MƏN ANA İŞLƏMƏK', 'Mənim anam işləyir'],          // yiyə: mənim + mənsubiyyət, fel 3-cü şəxsdə (mübtəda anadır)
  ['MƏN YAZMAQ BİLMƏK', 'Mən yaza bilərəm'],          // BİLMƏK-dən əvvəl -a forması
  ['B A K I', 'Bakı'],                                // barmaqla yığılan hərflər bir söz
  ['MƏN BAKI YAŞAMAQ', 'Mən Bakıda yaşayıram'],       // xüsusi ad böyük hərflə (label_text.csv)
  ['MƏN OTAQ GETMƏK', 'Mən otağıma gedirəm'],         // başqa haldakı ad mübtəda deyil: əvəzlik mübtədadır
];
for (const [labels, want] of cases) {
  const got = g.generate(labels.split(' ')).text;
  check(got === want, `${labels} → ${got}${got === want ? '' : `  (gözlənilən: ${want})`}`);
}

// 3. Brauzerin modeli: AzSLD-nin hər cümləsində mətn qurulur
const model = JSON.parse(fs.readFileSync('public/tercume/assets/gloss_text.json', 'utf8'));
const S = JSON.parse(fs.readFileSync('public/tercume/assets/cont_model/sentences.json', 'utf8'));
const G = new GlossToText(model);
const empty = S.filter((s) => !G.generate(s.glosses).text);
check(!empty.length, `gloss_text.json: ${S.length} cümlənin hamısında mətn var${empty.length ? ` (boş: ${empty.length})` : ''}`);

// Doğma işarə dilinin qrammatikası (E23): fel + YOX inkar, fel + OLMAQ / OLACAQ zaman, şəxsli cümlədə fel + LAZIM -malı,
// adam + BİRLİKDƏ -la; ad + OLMAQ və şəxssiz LAZIM dəyişmir
const native = [
  ['MƏN GƏLMƏK OLMAQ', 'Mən gəldim'], ['MƏN SABAH GƏLMƏK OLACAQ', 'Mən sabah gələcəyəm'], ['MƏN GƏLMƏK LAZIM', 'Mən gəlməliyəm'],
  ['MƏN GƏLMƏK YOX', 'Mən gəlmirəm'], ['SİZ İŞLƏMƏK OLACAQ', 'Siz işləyəcəksiniz'], ['O XƏSTƏ OLMAQ', 'O xəstə olub'],
  ['HƏR GÜN 8 STƏKAN SU İÇMƏK LAZIM', 'Hər gün 8 stəkan su içmək lazımdır'],
];
for (const [labels, want] of native) {
  const got = G.generate(labels.split(' ')).text;
  check(got === want, `doğma: ${labels} → ${got}${got === want ? '' : `  (gözlənilən: ${want})`}`);
}
// Yiyəlik və tamamlıq (E26): var/yox cümləsi, ad + LAZIM, ikinci əvəzliyin halı felin idarəsi ilə, ismi fel + modal,
// qohumluq sözü əvəzlikdən sonra, son hecanın saiti düşən ad
for (const [labels, want] of [['MƏN 2 BACI VAR', 'Mənim 2 bacım var'], ['SİZ EV VAR', 'Sizin eviniz var'],
  ['MƏN DƏRMAN LAZIM', 'Mənə dərman lazımdır'], ['MƏN SƏN SEVMƏK', 'Mən səni sevirəm'], ['O MƏN VERMƏK', 'O mənə verməlidir'],
  ['MƏN SƏN KÖMƏK ETMƏK BİLMƏK', 'Mən sənə kömək edə bilərəm'], ['MƏN SU VERMƏK BİLMƏK', 'Mənə su verə bilərsinizmi'],
  ['MƏN OĞUL EŞİTMƏ MƏHDÜDİYYƏTLİ', 'Mənim oğlum eşitmə məhdüdiyyətli']]) {
  const got = G.generate(labels.split(' ')).text;
  check(got === want, `yiyəlik/tamamlıq: ${labels} → ${got}${got === want ? '' : `  (gözlənilən: ${want})`}`);
}
// Zaman, xəbər, yer, say (E27): keçmiş zaman sözündən sonrakı fel (İNDİ-dən sonra indiki), il + İL, xəbər ad, iş adı +
// et-, yer adının halı felə görə, qohumun yaşı, sayılan sözdən əvvəl rəqəm, SONRA-dan əvvəl çıxışlıq, XOŞU GƏLMƏK,
// ad + VƏ + ad, dalbadal tanınmış eyni işarə
for (const [labels, want] of [['O ƏVVƏL XƏSTƏXANA İŞLƏMƏK', 'O əvvəl xəstəxanada işləyib'],
  ['MƏN ƏVVƏL GƏNCƏ YAŞAMAQ AMMA İNDİ BAKI YAŞAMAQ', 'Mən əvvəl Gəncədə yaşamışam amma indi Bakıda yaşayıram'],
  ['MƏN 2010 İL BAKI GƏLMƏK', 'Mən 2010-cu ildə Bakıya gəldim'], ['MƏN XƏSTƏ', 'Mən xəstəyəm'],
  ['BU MƏN QARDAŞ', 'Bu mənim qardaşımdır'], ['MƏN HƏR GÜN İDMAN', 'Mən hər gün idman edirəm'],
  ['MƏN QIZ 5 YAŞ VAR', 'Mənim qızımın 5 yaşı var'], ['MƏN BİR SUAL VAR', 'Mənim bir sualım var'],
  ['SİZ SAAT 6 SONRA İŞLƏMƏK', 'Siz saat 6-dan sonra işləyirsiniz'],
  ['MƏN 3 ƏDƏD QƏLƏM ALMAQ İSTƏMƏK', 'Mən 3 ədəd qələm almaq istəyirəm'], ['MƏN SİYAHI DƏRMAN LAZIM', 'Mənə siyahı dərman lazımdır'],
  ['MƏN XOŞU+GƏLMƏK', 'Mənim xoşum gəlir'], ['BU ƏTİR MƏN XOŞU+GƏLMƏK', 'Bu ətir mənim xoşuma gəlir'],
  ['MƏN XOŞU+GƏLMƏK YOX', 'Mənim xoşum gəlmir'], ['MƏN BACI VƏ QARDAŞ YOX', 'Mənim bacım və qardaşım yoxdur'],
  ['MƏN MÜAYİNƏ MÜAYİNƏ LAZIM', 'Mənə müayinə lazımdır'], ['MƏN İNDİ YATMAQ', 'Mən indi yatıram'],
  ['MƏN 2 OĞUL VƏ 1 QIZ VAR', 'Mənim 2 oğlum və 1 qızım var'], ['SƏN AD NƏDİR', 'Sənin adın nədir'],
  ['MƏN SƏN BAŞA+DÜŞMƏMƏK', 'Mən səni başa düşmürəm'], ['MƏN AYAQ AĞRIMAQ', 'Mənim ayağım ağrıyır'],
  ['MƏN MOBİL TELEFON İŞLƏMİR', 'Mənim mobil telefonum işləmir'], ['O SƏNƏD İTİRMƏK', 'O sənədi itirib'],
  ['MƏN MÜƏLLİM İŞLƏMƏK', 'Mən müəllim işləyirəm'], ['HARA GÖZLƏMƏK BİLMƏK', 'Harda gözləyə bilərəm'],
  ['O GƏLMƏK BİLMƏK', 'O gələ bilər'], ['MƏN KÖMƏK/YARDİM EDƏ BİLMƏK', 'Mənə kömək edə bilərsinizmi'],
  ['MƏN ZƏNG BİLMƏK', 'Mən zəng edə bilərəmmi'], ['MƏN QARDAŞ AD NƏDİR', 'Mənim qardaşımın adı nədir'],
  ['SİZ ATA AD NƏDİR', 'Sizin atanızın adı nədir']]) {
  const got = G.generate(labels.split(' ').map((l) => l.replaceAll('+', ' '))).text;   // + : çoxsözlü işarə
  check(got === want, `zaman/xəbər/yer/say: ${labels} → ${got}${got === want ? '' : `  (gözlənilən: ${want})`}`);
}
const { ordinal } = await import('../src/tercume/gloss2text.js');
const ords = [1979, 2018, 1990, 2000, 1940, 1800, 30].map(ordinal).join(' ');
check(ords === '1979-cu 2018-ci 1990-cı 2000-ci 1940-cı 1800-cü 30-cu', `sıra sayı: ${ords}`);
const birlikde = G.generate('MƏN ANA BİRLİKDƏ BAZAR GETMƏK OLMAQ'.split(' ')).text;
check(/anamla birlikdə/.test(birlikde) && /getdim$/.test(birlikde), `doğma: MƏN ANA BİRLİKDƏ BAZAR GETMƏK OLMAQ → ${birlikde}`);

// 4. morph.js ↔ inflect.js: lüğətin köklərindən qurulan formalar (sualsız) təhlildə öz kökünə qayıdır.
// Qalanı əsasən omoqraflardır (adam: ad-ım / adam, bağla: bağ-a / bağla); kökün özü (əmr: al, bax) lüğət
// sözü deyil, təhlil olunmur
const lem = new WordLexicon(JSON.parse(fs.readFileSync('public/tercume/words/index.json', 'utf8'))).lemma;
const nouns = [...lem.keyOfWord].filter(([w, k]) => w === k && !lem.verbs.has(k) && !/\d/.test(w) && w.length >= 2)
  .map(([w]) => w);
let n = 0, back = 0;
for (const [pos, roots] of [['verb', [...lem.verbs]], ['noun', nouns]]) for (const r of roots) {
  for (const p of paradigm(r, pos)) {
    if (p.feats.q || p.form === r) continue;
    n++;
    if (lem.key(p.form) === keyOf(r, pos === 'verb' && p.feats.neg)) back++;
  }
}
check(back / n >= 0.99, `morph.js ↔ inflect.js: ${back}/${n} forma öz kökünə qayıdır (${(100 * back / n).toFixed(1)} %, hədd 99 %)`);

// 5. Korpusun sayları (prior, docs/NLP_PLAN.md S2): α ilə vurulub cümlələrin saylarına əlavə olunur; cütün
// çəkisi (w) saylarını vurur
{
  const pairs = [
    { glosses: ['MƏN', 'BAKI', 'YAŞAMAQ'], text: 'Mən Bakıda yaşayıram' },
    { glosses: ['O', 'GƏNCƏ', 'YAŞAMAQ'], text: 'O Gəncədə yaşayır', w: 2 },
    { glosses: ['MƏN', 'BAKI', 'YAŞAMAQ'], text: 'Mən Bakıda yaşayıram' },
  ];
  const prep = prepare(pairs);
  const prior = fit(prep, (i) => i < 2);
  const m = fit(prep, (i) => i === 2, prior, 0.5);
  const f = m.forms.YAŞAMAQ ?? {};
  check(f.yaşayıram === 1.5 && f.yaşayır === 1, `prior: yaşayıram 0.5·1 + 1 = ${f.yaşayıram}, yaşayır 0.5·2 = ${f.yaşayır}`);
}

console.log(fails ? `\n${fails} yoxlama uğursuz` : '\nhamısı keçdi');
process.exit(fails ? 1 : 0);
