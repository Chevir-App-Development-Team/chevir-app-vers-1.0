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

// 2. Plan: lüğət sözü bütöv addım, qalan hərf-hərf
const poses = rd(new URL('poses.json', A));
check(!('words' in poses), 'poses.json-da köhnə "words" bölməsi yoxdur');
const t2s = new TextToSign({ poses, lexicon: toy });
const plan = t2s.plan('Salam ana').map((s) => `${s.kind}:${s.char}`).join(' ');
check(plan === 'word:salam space:  letter:a letter:n letter:a', `plan("Salam ana") → ${plan}`);

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
