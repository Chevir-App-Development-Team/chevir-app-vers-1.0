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

// 3. Brauzerin modeli: AzSLD-nin hər cümləsində mətn qurulur. G — qaydalar və saylar (formanın modeli söndürülüb:
// aşağıdakı hallar qaydaların yoxlamasıdır); GT — brauzerdəki kimi, gloss_text.json-un formTagger-i ilə (E33). Hər hal GT-də
// də yoxlanır: gözlənilən mətn eynidir, MODEL_FORMS-dakılardan başqa (model başqa formanı seçir, mənası eynidir)
const model = JSON.parse(fs.readFileSync('public/tercume/assets/gloss_text.json', 'utf8'));
const S = JSON.parse(fs.readFileSync('public/tercume/assets/cont_model/sentences.json', 'utf8'));
const G = new GlossToText(model, { tagger: false });
const GT = model.formTagger?.on ? new GlossToText(model) : null;
for (const [name, g2] of [['qaydalar', G], ['formanın modeli', GT]]) {
  if (!g2) continue;
  const empty = S.filter((s) => !g2.generate(s.glosses).text);
  check(!empty.length, `gloss_text.json (${name}): ${S.length} cümlənin hamısında mətn var${empty.length ? ` (boş: ${empty.length})` : ''}`);
}
// formanın modeli ilə fərqli, ancaq düzgün forma (zaman işarəsi yoxdursa indiki / keçmiş, -dır, izafət)
const MODEL_FORMS = {
  'O XƏSTƏ OLMAQ': 'O xəstə oldu', 'O MƏN VERMƏK': 'O mənə verir',
  'MƏN OĞUL EŞİTMƏ MƏHDÜDİYYƏTLİ': 'Mənim oğlum eşitmə məhdüdiyyətlidir',
  'MƏN POLİKLİNİKA T S E K GETMƏK': 'Mən poliklinikaya TSEK-ə gedirəm', 'MƏN B A K İ GETMƏK': 'Mən Bakıya gedirəm',
  'MƏN B M T MÜRACİƏT ETMƏK': 'Mən BMT-yə müraciət etdim', 'MƏN D S M F MÜRACİƏT ETMƏK': 'Mən DSMF-ə müraciət etdim',
  'MƏN Ü D S Y MÜRACİƏT ETMƏK': 'Mən ÜDSY-yə müraciət etdim',
  'A B Ş E R O N D O S T MƏRKƏZ MÜRACİƏT ETMƏK': 'Abşeron DOST mərkəzinə müraciət etdi', 'D S T MƏRKƏZ VAR': 'DOST mərkəzi var',
  'MƏN A D A UNİVERSİTET OXUMAQ': 'Mən ADA universitetində oxuyuram', 'MƏN Q Ə B Ə L Ə Y Ə GETMƏK': 'Mən Qəbələyə gedirəm',
};
const both = (name, labels, want, ls = labels.split(' ')) => {
  const got = G.generate(ls).text;
  check(got === want, `${name}: ${labels} → ${got}${got === want ? '' : `  (gözlənilən: ${want})`}`);
  if (!GT) return;
  const w2 = MODEL_FORMS[labels] ?? want, got2 = GT.generate(ls).text;
  check(got2 === w2, `${name} (formanın modeli): ${labels} → ${got2}${got2 === w2 ? '' : `  (gözlənilən: ${w2})`}`);
};

// Doğma işarə dilinin qrammatikası (E23): fel + YOX inkar, fel + OLMAQ / OLACAQ zaman, şəxsli cümlədə fel + LAZIM -malı,
// adam + BİRLİKDƏ -la; ad + OLMAQ və şəxssiz LAZIM dəyişmir
const native = [
  ['MƏN GƏLMƏK OLMAQ', 'Mən gəldim'], ['MƏN SABAH GƏLMƏK OLACAQ', 'Mən sabah gələcəyəm'], ['MƏN GƏLMƏK LAZIM', 'Mən gəlməliyəm'],
  ['MƏN GƏLMƏK YOX', 'Mən gəlmirəm'], ['SİZ İŞLƏMƏK OLACAQ', 'Siz işləyəcəksiniz'], ['O XƏSTƏ OLMAQ', 'O xəstə olub'],
  ['HƏR GÜN 8 STƏKAN SU İÇMƏK LAZIM', 'Hər gün 8 stəkan su içmək lazımdır'],
];
for (const [labels, want] of native) both('doğma', labels, want);
// Yiyəlik və tamamlıq (E26): var/yox cümləsi, ad + LAZIM, ikinci əvəzliyin halı felin idarəsi ilə, ismi fel + modal,
// qohumluq sözü əvəzlikdən sonra, son hecanın saiti düşən ad
for (const [labels, want] of [['MƏN 2 BACI VAR', 'Mənim 2 bacım var'], ['SİZ EV VAR', 'Sizin eviniz var'],
  ['MƏN DƏRMAN LAZIM', 'Mənə dərman lazımdır'], ['MƏN SƏN SEVMƏK', 'Mən səni sevirəm'], ['O MƏN VERMƏK', 'O mənə verməlidir'],
  ['MƏN SƏN KÖMƏK ETMƏK BİLMƏK', 'Mən sənə kömək edə bilərəm'], ['MƏN SU VERMƏK BİLMƏK', 'Mənə su verə bilərsinizmi'],
  ['MƏN OĞUL EŞİTMƏ MƏHDÜDİYYƏTLİ', 'Mənim oğlum eşitmə məhdüdiyyətli']]) both('yiyəlik/tamamlıq', labels, want);
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
  ['SİZ ATA AD NƏDİR', 'Sizin atanızın adı nədir'], ['MƏN AD S A M İ R', 'Mənim adım Samirdir'],
  ['MƏN BACI AD L E Y L A', 'Mənim bacımın adı Leyladır']]) {
  both('zaman/xəbər/yer/say', labels, want, labels.split(' ').map((l) => l.replaceAll('+', ' ')));   // + : çoxsözlü işarə
}
// Hərflə yığılan sözlər (R6, E30, spelled.js): öyrənilmiş yazılış (DOST, TSEK, ADA), qısaltma və ona defislə hal
// (oxunuşa görə: BMT-yə, DSMF-ə), ad və ona hal (Messini, Qubada, I hərfi bərpa: Bakıya), ümumi söz kiçik hərflə
// (manat, kitab), ardınca METRO / FİLM gələndə ümumi söz də ad (Sahil, Dolu), iki tanış hissə (Abşeron DOST),
// O + mənsubiyyət (onun nəvəsinin), bir hərf səhvi öyrənilmiş sözə (D S T → DOST), hərflər arasında nöqtə, tək hərf
// dəyişmir, cümlənin əvvəlindəki ad mübtədadır (hal almır)
if (model.spelled && model.common) {
  for (const [labels, want] of [['MƏN POLİKLİNİKA T S E K GETMƏK', 'Mən poliklinikaya TSEK-ə getmişəm'],
    ['MƏN M E S S İ SEVMƏK', 'Mən Messini sevirəm'], ['MƏN Q U B A YAŞAMAQ', 'Mən Qubada yaşayıram'],
    ['MƏN B A K İ GETMƏK', 'Mən Bakıya getmişəm'], ['MƏN B M T MÜRACİƏT ETMƏK', 'Mən BMT-yə müraciət etmişəm'],
    ['MƏN D S M F MÜRACİƏT ETMƏK', 'Mən DSMF-ə müraciət etmişəm'], ['MƏN Ü D S Y MÜRACİƏT ETMƏK', 'Mən ÜDSY-yə müraciət etmişəm'],
    ['A B Ş E R O N D O S T MƏRKƏZ MÜRACİƏT ETMƏK', 'Abşeron DOST mərkəzinə müraciət edib'],
    ['D S T MƏRKƏZ VAR', 'DOST mərkəzləri var'], ['J E S T D İ L İ . A Z ÇOX FAYDALIDIR', 'Jestdili.az çox faydalıdır'],
    ['BU QİYMƏT 10 M A N A T', 'Bu qiyməti 10 manat'], ['MƏN K İ T A B ALMAQ', 'Mən kitab almışam'],
    ['MƏN S A H İ L METRO GETMƏK', 'Mən Sahil metroya gedirəm'],
    ['MƏN D O L U FİLM BİLET ALMAQ İSTƏMƏK', 'Mən Dolu filminə bilet almaq istəyirəm'],
    ['MƏN A D A UNİVERSİTET OXUMAQ', 'Mən ADA universitetdə oxuyuram'],
    ['O N Ə V Ə S İ N İ N 6 YAŞ VAR', 'Onun nəvəsinin 6 yaşı var'], ['MƏN L E Y L A SEVMƏK', 'Mən Leylanı sevirəm'],
    ['MƏN Q Ə B Ə L Ə Y Ə GETMƏK', 'Mən Qəbələyə getmişəm'], ['S A M İ R GƏLMƏK', 'Samir gəldi'],
    ['MƏN İNDİ A', 'Mən indi a']]) both('hərflə', labels, want);
  const off = new GlossToText(model, { names: false, tagger: false }).generate('D O S T MƏRKƏZ VAR'.split(' ')).text;
  check(off === 'Dost mərkəzləri var', `hərflə, names: false (R6-dan əvvəlki kimi): D O S T MƏRKƏZ VAR → ${off}`);
  const { alignRuns, withCase } = await import('../src/tercume/spelled.js');
  const al = alignRuns(['tsek', 'messinin', 'abşerondost', 'manat', 'sahil'],
    'Sahil yanında (TSEK-ə) getdim, Messinin fanatıyam, Abşeron DOST Mərkəzi 10 manat').map((x) => x.surface);
  check(JSON.stringify(al) === JSON.stringify(['TSEK', 'Messinin', 'Abşeron DOST', 'manat', null]),
    `hərflə, öyrənmə (mətndəki yazılış; cümlənin ilk sözünün böyük hərfi sayılmır): ${JSON.stringify(al)}`);
  const cs = [withCase('DOST', 'acronym', 'dat'), withCase('ÜDSY', 'acronym', 'loc'), withCase('İraq', 'name', 'dat'),
    withCase('Messini', 'name', 'acc'), withCase('manat', 'word', 'dat')].join(' ');
  check(cs === 'DOST-a ÜDSY-də İraqa Messini manat', `hərflə, hal: ${cs}`);
}
const { ordinal } = await import('../src/tercume/gloss2text.js');
const ords = [1979, 2018, 1990, 2000, 1940, 1800, 30].map(ordinal).join(' ');
check(ords === '1979-cu 2018-ci 1990-cı 2000-ci 1940-cı 1800-cü 30-cu', `sıra sayı: ${ords}`);
const birlikde = G.generate('MƏN ANA BİRLİKDƏ BAZAR GETMƏK OLMAQ'.split(' ')).text;
check(/anamla birlikdə/.test(birlikde) && /getdim$/.test(birlikde), `doğma: MƏN ANA BİRLİKDƏ BAZAR GETMƏK OLMAQ → ${birlikde}`);
// Slovo-dan yeni işarələr (E29, ml/slovo_new.csv): lüğət forması təsrifli fel kimi oxunmur (YEDDİ ≠ ye-di), sözləri
// köhnə işarələrin təhlilini dəyişmir (AYI ≠ bu ay, YAZ ≠ yazda)
if (model.guess.YEDDİ) {
  check(model.guess.YEDDİ.pos === 'noun' && model.guess.AYI?.root === 'ayı' && model.guess.DİNLƏMƏK?.pos === 'verb',
    `yeni işarələr: YEDDİ ${model.guess.YEDDİ.pos}, AYI ${model.guess.AYI?.root}, DİNLƏMƏK ${model.guess.DİNLƏMƏK?.pos}`);
  for (const [labels, want] of [['SƏN NİYƏ GƏLMƏK YOX', 'Sən niyə gəlmirsən'], ['MƏN EŞİTMƏK YOX', 'Mən eşitmirəm'],
    ['TƏCİLİ HƏKİM LAZIM', 'Təcili həkim lazımdır'], ['MƏN ƏSƏBİ', 'Mən əsəbiyəm'],
    ['MƏN BU AY PENSİYA ALMAMIŞAM', 'Mən bu ay pensiya almamışam'], ['YAZDA HAVA GÖZƏL OLUR', 'Yazda hava gözəl olur']]) {
    both('yeni işarələr', labels, want);
  }
}

// Formanın modeli (E33) mənanı dəyişmir: AzSLD-nin 492 cümləsində və adında inkar olan hər fel işarəsi ilə (MƏN / O + işarə)
// hər sözün inkarı qaydalardakı kimidir (inkarı model seçmir), hərflə yığılan sözlər (R6) eynidir
if (GT) {
  const { candidates } = await import('../src/tercume/morph.js');
  const { azLower } = await import('../src/tercume/words.js');
  const negOf = (w) => w.split(' ').some((t) => candidates(azLower(t)).some((c) => c.neg));
  const probes = [...S.map((s) => s.glosses)];
  for (const [l, gs] of Object.entries(model.guess)) {
    if (gs.inherent?.neg || /(m[aə]m[aə][kq]|m[aə]y[ıi]b|m[ıiuü]r|m[aə]m[ıiuü]ş)/.test(azLower(l))) probes.push(['MƏN', l], ['O', l]);
  }
  let flips = 0, letters = 0;
  const ex = [];
  for (const ls of probes) {
    const a = G.generate(ls).words, b = GT.generate(ls).words;
    a.forEach((x, k) => {
      if (negOf(x.word ?? '') !== negOf(b[k]?.word ?? '')) { flips++; if (ex.length < 3) ex.push(`${ls.join(' ')}: ${x.word} / ${b[k]?.word}`); }
      if (x.via === 'letters' && x.word !== b[k]?.word) letters++;
    });
  }
  check(!flips, `formanın modeli: ${probes.length} ardıcıllıqda inkar qaydalardakı kimidir${flips ? ` (${flips} fərq: ${ex.join('; ')})` : ''}`);
  check(!letters, `formanın modeli: hərflə yığılan sözlər qaydalardakı kimidir${letters ? ` (${letters} fərq)` : ''}`);
  const keep = GT.generate('MƏN SƏN GÖZLƏMƏK'.split(' ')).text;
  check(keep === 'Mən səni gözləyirəm', `formanın modeli: MƏN SƏN GÖZLƏMƏK → ${keep} (qaydalar: ${G.generate('MƏN SƏN GÖZLƏMƏK'.split(' ')).text})`);
  const off = new GlossToText(model, { tagger: false }).generate(S[0].glosses).text === G.generate(S[0].glosses).text;
  check(off && GT.tagger && GT.lambda === (model.formTagger.lambda ?? GT.lambda), `formanın modeli: açar (tagger: false — yalnız qaydalar), lambda ${GT.lambda}`);
}

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
