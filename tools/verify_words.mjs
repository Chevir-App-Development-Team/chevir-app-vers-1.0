/**
 * Söz lüğətinin sınağı: mətnin söz/ifadə/hərflərə bölünməsi və
 * public/tercume/words/ fayllarının formatı (tools/build_words.py çıxışı).
 */
import { readFileSync, existsSync } from 'node:fs';
import { WordLexicon, azLower } from '../src/tercume/words.js';
import { TextToSign } from '../src/tercume/t2s.js';

const W = new URL('../public/tercume/words/', import.meta.url);
const A = new URL('../public/tercume/assets/', import.meta.url);
const rd = (url) => JSON.parse(readFileSync(url, 'utf8'));

let fails = 0;
const check = (ok, msg) => {
  console.log(`${ok ? '✓' : '✗'} ${msg}`);
  if (!ok) fails++;
};
const show = (segs) => segs.map((s) => (s.kind === 'space' ? '_' : `${s.kind[0]}:${s.text}`)).join(' ');

// 1. Bölmə — kiçik süni lüğətlə (ifadələr, durğu işarələri, böyük İ/I)
const toy = new WordLexicon({ words: [
  { id: 'salam', text: 'salam' }, { id: 'zehmet-olmasa', text: 'zəhmət olmasa' },
  { id: 'komek', text: 'kömək' }, { id: 'bu', text: 'bu' }, { id: 'bu-gun', text: 'bu gün' },
  { id: 'isiq', text: 'işıq' },
] });
const cases = [
  ['Salam, zəhmət olmasa kömək!', 'w:salam _ w:zəhmət olmasa _ w:kömək'],
  ['bu gün bu', 'w:bu gün _ w:bu'],
  ['İŞIQ', 'w:işıq'],
  ['salam ana', 'w:salam _ t:ana'],
  ['zəhmət  yox', 't:zəhmət _ t:yox'],
  ['— salam —', '_ w:salam _'],
];
for (const [text, want] of cases) {
  const got = show(toy.segment(text));
  check(got === want, `segment(${JSON.stringify(text)}) → ${got}${got === want ? '' : `  (gözlənilən: ${want})`}`);
}
check(azLower('İLHAM IŞIQ') === 'ilham ışıq', 'azLower: İ → i, I → ı');

// Söz formaları (morph.js): şəkilçili söz lüğət sözünə gəlir, inkar ayrıca qalır
const forms = new WordLexicon({ words: [
  'ev', 'ana', 'bilmək', 'bilmir', 'olmaq', 'olar', 'itib', 'dost', 'oğul', 'uşaq', 'köynək', 'getmək',
  'yaşamaq', 'istəmək', 'istəyir', 'düşməmək', 'gün', 'ad günü', 'il', 'yatmaq', 'Bakı',
  'oxumaq', 'gəlmək', 'yox', 'kart', 'i', '6', 'yazmaq', 'ay',
  'ucuzdur', 'gedir', 'iş', 'iştirak', 'fikir', 'xatırladım', 'bilər', 'bilərsinizmi',
].map((text) => ({ id: text, text })) });
const formCases = [
  ['evdə', 'w:ev'], ['anam', 'w:ana'], ['Bakıda', 'w:Bakı'], ['oğlum', 'w:oğul'], ['uşağın', 'w:uşaq'],
  ['köynəyini', 'w:köynək'], ['gedəcəm', 'w:getmək'], ['yaşayıram', 'w:yaşamaq'], ['istəyirsən', 'w:istəyir'],
  ['bilirəm', 'w:bilmək'], ['bilmirəm', 'w:bilmir'], ['bilmədim', 'w:bilmir'], ['düşmürəm', 'w:düşməmək'],
  ['olan', 'w:olmaq'], ['itib', 'w:itib'], ['it', 't:it'], ['ilə', 'd:ilə'], ['yadımda', 't:yadımda'],
  ['şənbə günü', 't:şənbə _ w:gün'], ['ad günündə', 'w:ad günü'],
  ['DOST mərkəzi', 't:dost _ t:mərkəzi'], ['DOST', 'w:dost'],
  // inkarlı felin öz işarəsi yoxdursa: fel + YOX (müsbət işarə tək qalmır), hər zamanda
  ['oxumadım', 'w:oxumaq w:yox'], ['oxumuram', 'w:oxumaq w:yox'], ['gəlməyəcəyəm', 'w:gəlmək w:yox'],
  ['gəlmərəm', 'w:gəlmək w:yox'], ['gəlmərik', 'w:gəlmək w:yox'], ['gəlməz', 'w:gəlmək w:yox'],
  // xəbər sözü öz sifətinə gedir (ged-ir isə feldir); isimlər fel kökü yaratmır (işt-ir-ak, fik-ir)
  ['ucuz', 'w:ucuzdur'], ['gedirsən', 'w:gedir'], ['işdir', 'w:iş'], ['fikirlər', 'w:fikir'],
  ['xatırlayıram', 'w:xatırladım'],
  // eyni kökün sözlərindən daha az şəkilçilisi: bilərəm → bilər (bilərsinizmi yox)
  ['bilərəm', 'w:bilər'],
  ['oxuyuram', 'w:oxumaq'], ['oxumaq', 'w:oxumaq'],
  // felin səs dəyişməsi ad kökünə aparmır, bir hərfli kök yoxdur
  ['kardır', 't:kardır'], ['kartı', 'w:kart'], ['iyun', 't:iyun'], ['ayın', 'w:ay'],
  // defis: şəkilçi atılır (rəqəm, ad), söz deyilsə hissələr ayrıdır
  ['6-dan', 'w:6'], ['6-cı', 'w:6'], ['TSEK-ə', 't:tsek'], ['yazıb-oxuya', 'w:yazmaq w:oxumaq'],
];
for (const [text, want] of formCases) {
  const got = show(forms.segment(text));
  check(got === want, `forma: ${JSON.stringify(text)} → ${got}${got === want ? '' : `  (gözlənilən: ${want})`}`);
}

// Mövqelər: hər hissə ilkin mətndə öz yerini göstərir (postda vurğulama üçün)
{
  const text = '«Salam», zəhmət   olmasa İŞIQ!';
  const got = toy.segment(text).filter((s) => s.kind !== 'space')
    .map((s) => text.slice(s.from, s.to)).join('|');
  const want = 'Salam|zəhmət   olmasa|İŞIQ';
  check(got === want, `mövqelər: ${got}${got === want ? '' : `  (gözlənilən: ${want})`}`);
}

// 2. Plan: lüğət sözü bütöv addım, qalan hərf-hərf
const poses = rd(new URL('poses.json', A));
check(!('words' in poses), 'poses.json-da köhnə "words" bölməsi yoxdur');
const t2s = new TextToSign({ poses, lexicon: toy });
const plan = t2s.plan('Salam ana').map((s) => `${s.kind}:${s.char}`).join(' ');
check(plan === 'word:salam space:  letter:a letter:n letter:a', `plan("Salam ana") → ${plan}`);
const pos = t2s.plan('Salam ana').map((s) => `${s.from}-${s.to}`).join(' ');
check(pos === '0-5 5-6 6-7 7-8 8-9', `plan mövqeləri → ${pos}`);

// 3. Real lüğətin faylları
if (!existsSync(new URL('index.json', W))) {
  check(false, 'public/tercume/words/index.json yoxdur — tools/build_words.py işlədilməyib');
} else {
  const index = rd(new URL('index.json', W));
  const lex = new WordLexicon(index);
  check(index.count === index.words.length && index.count > 100, `indeksdə ${index.count} söz`);
  check(new Set(index.words.map((w) => w.id)).size === index.count, 'söz id-ləri təkrarlanmır');
  check(show(lex.segment('Salam, ana!')) === 'w:salam _ w:ana', 'real lüğət: "Salam, ana!" → salam | ana');

  const bad = [];
  for (const e of index.words) {
    const url = new URL(`${e.id}.json`, W);
    if (!existsSync(url)) { bad.push(`${e.id}: fayl yoxdur`); continue; }
    const w = rd(url);
    const problems = [];
    if (w.text !== e.text || w.hands !== e.hands || w.frames.length !== e.frames) problems.push('indekslə uyğun deyil');
    if (!(w.fps > 0) || w.size?.length !== 2) problems.push('fps/size');
    for (const [k, fr] of w.frames.entries()) {
      if (!fr.b || fr.b.length !== 9) { problems.push(`kadr ${k}: bədən`); break; }
      for (const s of ['L', 'R']) {
        const h = fr[s];
        if (h && (h.n.length !== 21 || h.w.length !== 21)) { problems.push(`kadr ${k}: ${s} əli 21 nöqtə deyil`); break; }
        if (h && !e.hands.includes(s)) { problems.push(`kadr ${k}: ${s} əli işarədə yoxdur`); break; }
      }
    }
    for (const s of e.hands) {
      if (!w.frames.some((fr) => fr[s])) problems.push(`${s} əli heç bir kadrda yoxdur`);
    }
    if (problems.length) bad.push(`${e.id}: ${problems.join(', ')}`);
  }
  check(!bad.length, `bütün söz faylları formatda${bad.length ? `:\n    ${bad.slice(0, 10).join('\n    ')}` : ''}`);
}

console.log(fails ? `\n${fails} yoxlama uğursuz` : '\nhamısı keçdi');
process.exit(fails ? 1 : 0);
