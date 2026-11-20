// Yazılış səhifəsinin (/qeyd/, roadmap D4) tapşırıqları: public/qeyd/prompts.json.
//
// (a) çatışmayan işarələr: rəqəmlər (işarə dili daşıyıcısına görə hər ədədin öz işarəsi var, rəqəm-rəqəm göstərmə ara
// həldir), müsahibənin gündəlik sözlərindən (docs/SIGNER_INTERVIEW.md, M3) lüğətdə hələ işarəsi olmayanlar, qrammatika
// işarələri, ölkələr; (b) set C cümlələri (ml/setc/setc.csv). Lüğətdə olan söz siyahıdan çıxır: lüğət
// (public/tercume/words/index.json, AzSLD + JestDili.az) bu yazılış və ya onun forması ilə tapırsa.
//
//   node tools/build_record_prompts.mjs
import fs from 'node:fs';
import { WordLexicon, azLower } from '../src/tercume/words.js';

const index = JSON.parse(fs.readFileSync('public/tercume/words/index.json', 'utf8'));
const lex = new WordLexicon(index, { native: false });
// söz və ya ifadə lüğətdə bütöv işarədir (hərf-hərf yığılmır)
const hasSign = (t) => {
  const segs = lex.segment(t).filter((s) => s.kind !== 'space' && s.kind !== 'drop');
  return segs.length === 1 && segs[0].kind === 'word';
};

const prompts = [];
const add = (group, prefix, list, note = '') => list.forEach((text, k) => prompts.push({
  id: `${prefix}${String(k + 1).padStart(2, '0')}`, group, text, kind: group.startsWith('Cümlə') ? 'sentence' : 'sign', ...(note ? { note } : {}),
}));

// Rəqəmlər: 0–20, onluqlar, yüzlər, minlər, illər, saat, telefon, pul, sıra sayı (müsahibə: hamısının öz işarəsi var)
// (öz işarəsi olan ədəd çıxır; saat, telefon, pul, sıra sayı həmişə qalır: onların quruluşu lazımdır)
add('Rəqəmlər', 'n', [
  ...[...Array.from({ length: 21 }, (_, i) => String(i)), '25', '30', '40', '50', '60', '70', '80', '90', '100', '200', '500',
    '1000', '2000', '1990', '2018', '2025'].filter((t) => !hasSign(t)),
  'saat 11:15', 'saat 12:30', '050 123 45 67', '15 manat 50 qəpik', '2-ci', '3-cü',
], 'ədədi gündəlik danışıqdakı kimi göstərin');

// Gündəlik sözlər: M3 cədvəlinin sözlərindən lüğətdə hələ işarəsi olmayanlar
const doc = fs.readFileSync('docs/SIGNER_INTERVIEW.md', 'utf8');
const m3 = doc.slice(doc.indexOf('### M3.'), doc.indexOf('### M4.'));
const everyday = [];
for (const line of m3.split('\n')) {
  const m = /^\| ([^|*]+?) \| ([^|]+) \|$/.exec(line.trim());
  if (!m || m[1] === 'Mövzu' || m[1].startsWith('---') || m[1] === 'icma') continue;
  for (const part of m[2].replace(/\(.*?\)/g, '').split(/[,;:]/)) {
    const t = azLower(part.replace(/yarısı yığılır/, '').trim());
    if (t && /^[\p{L} ]+$/u.test(t) && !everyday.includes(t)) everyday.push(t);
  }
}
const missingEveryday = everyday.filter((t) => !hasSign(t));
add('Gündəlik sözlər', 'w', missingEveryday);

// Qrammatika işarələri (müsahibə: DOĞRU "evə gedirəm" = EV DOĞRU GETMƏK; bacarıq, vaciblik)
// DOĞRU lüğətdə "düz, həqiqi" mənasındadır (JestDili.az); istiqamət işarəsi ayrıca soruşulur
prompts.push({ id: 'g01', group: 'Qrammatika işarələri', text: 'evə doğru', kind: 'sign',
  note: 'istiqamət: "evə gedirəm" = MƏN EV DOĞRU GETMƏK; DOĞRU işarəsini göstərin' });

// Ölkələr və şəhərlər (adlar: işarəsi yoxdursa avatar hərf-hərf yığır)
const places = ['Azərbaycan', 'Türkiyə', 'Rusiya', 'Gürcüstan', 'İran', 'Ukrayna', 'Qazaxıstan', 'Özbəkistan', 'Almaniya', 'Fransa',
  'İngiltərə', 'Amerika', 'Çin', 'Yaponiya', 'İtaliya', 'İspaniya', 'Misir', 'Ərəbistan', 'Bakı', 'Gəncə', 'Sumqayıt', 'Şəki',
  'Lənkəran', 'Naxçıvan', 'Qəbələ', 'Quba', 'Şamaxı', 'Mingəçevir', 'Qarabağ', 'Şuşa', 'Moskva', 'İstanbul', 'Tbilisi'];
add('Ölkələr və şəhərlər', 'p', places.filter((t) => !hasSign(t)));

// Set C cümlələri: hər iki yarısı yazılır (dev və test), təbii göstərişlə
const setc = fs.existsSync('ml/setc/setc.csv') ? fs.readFileSync('ml/setc/setc.csv', 'utf8') : '';
for (const line of setc.trim().split(/\r?\n/).slice(1)) {
  const m = /^(c\d+),([^,]+),(dev|test),("(?:[^"]|"")*"|[^,]*),/.exec(line);
  if (!m) continue;
  prompts.push({ id: m[1], group: `Cümlələr: ${m[2]}`, text: m[4].replace(/^"|"$/g, '').replace(/""/g, '"'), kind: 'sentence', split: m[3] });
}

fs.mkdirSync('public/qeyd', { recursive: true });
fs.writeFileSync('public/qeyd/prompts.json', JSON.stringify({
  note: 'tools/build_record_prompts.mjs: çatışmayan işarələr (roadmap D4a) və set C cümlələri (V1, D4b)',
  prompts,
}, null, 1));
const groups = [...prompts.reduce((m, p) => m.set(p.group, (m.get(p.group) ?? 0) + 1), new Map())];
console.log(`public/qeyd/prompts.json: ${prompts.length} tapşırıq | ${groups.map(([g, n]) => `${g} ${n}`).join(', ')}`);
console.log(`gündəlik sözlər: ${everyday.length}-dən ${missingEveryday.length}-nin işarəsi yoxdur: ${missingEveryday.join(', ')}`);
console.log(`ölkə və şəhərlər işarəsiz: ${places.filter((t) => !hasSign(t)).join(', ')}`);
