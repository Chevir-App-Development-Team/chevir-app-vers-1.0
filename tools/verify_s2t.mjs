/**
 * Kamera axınını süni landmark ardıcıllığı ilə təqlid edir və hərf yazma
 * şərtlərini yoxlayır: keçid anları, əlin kadra girməsi və titrəyiş hərf
 * yazmamalı, sabit poza isə düzgün hərfi bir dəfə yazmalıdır.
 *
 * Bu sınaq "salam" yazanda əvvəl "ö" çıxması probleminin qayıtmasının qarşısını alır.
 */
import { readFileSync } from 'node:fs';
import { FingerspellModel } from '../src/tercume/model.js';
import { LexiconDecoder } from '../src/tercume/decoder.js';
import { SignToText } from '../src/tercume/s2t.js';

const A = new URL('../public/tercume/assets/', import.meta.url);
const rd = (f) => JSON.parse(readFileSync(new URL(f, A), 'utf8'));
const bin = readFileSync(new URL('model.bin', A));
const model = new FingerspellModel(rd('model.json'), bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const decoder = new LexiconDecoder(rd('lm.json'), rd('vocab.json'), rd('prefixes.json'));
const poses = rd('poses.json');

const pose = (ch) => poses.letters[ch].frames[0].map((p) => [p.x, p.y, p.z]);
const toKp = (frame) => {
  const kp = new Float32Array(63);
  frame.forEach((p, j) => { kp[j * 3] = p[0]; kp[j * 3 + 1] = p[1]; kp[j * 3 + 2] = p[2]; });
  return kp;
};
const lerp = (a, b, t) => a.map((p, j) => p.map((v, k) => v + (b[j][k] - v) * t));
const jitter = (frame, amp) => frame.map((p) => p.map((v) => v + (Math.random() - 0.5) * amp));

/** Əl ilə yazma ritmi: keçid (hərəkət) + saxlama (kiçik titrəyişlə). */
function spell(word, { move = 8, hold = 34, shake = 0.006, startEmpty = 20 } = {}) {
  const seq = Array.from({ length: startEmpty }, () => null);   // əl hələ kadrda yoxdur
  let prev = null;
  for (const ch of word) {
    const target = pose(ch);
    const from = prev ?? target.map((p) => [p[0], Math.min(p[1] + 0.4, 1), p[2]]);   // əl aşağıdan qalxır
    for (let i = 0; i < move; i++) seq.push(toKp(lerp(from, target, (i + 1) / move)));
    for (let i = 0; i < hold; i++) seq.push(toKp(jitter(target, shake)));
    prev = target;
  }
  return seq;
}

function run(name, seq, expected) {
  const s2t = new SignToText({ model, decoder });
  const typed = [];
  s2t.onDecode = ({ letters }) => { typed.length = 0; typed.push(...letters); };
  for (const kp of seq) s2t.pushFrame(kp);
  const got = typed.join('');
  const ok = got === expected;
  console.log(`  ${ok ? '✅' : '❌'} ${name}: "${got}"${ok ? '' : ` (gözlənilən "${expected}")`}`);
  return ok;
}

console.log('Kamera ardıcıllığı — hərf yazma şərtləri\n');
let ok = true;
ok = run('"salam" hərf-hərf (keçidlər daxil)', spell('salam'), 'salam') && ok;
ok = run('əl kadra "s" pozasında birbaşa girir', [
  ...Array.from({ length: 15 }, () => null),
  ...Array.from({ length: 40 }, () => toKp(jitter(pose('s'), 0.006))),
], 's') && ok;
ok = run('yalnız hərəkət (heç bir sabit poza yoxdur)', (() => {
  const a = pose('a'), s = pose('s'), seq = [];
  for (let i = 0; i < 120; i++) seq.push(toKp(lerp(a, s, Math.abs(Math.sin(i / 8)))));
  return seq;
})(), '') && ok;
ok = run('uzun saxlama bir hərf yazır (təkrar yoxdur)', [
  ...Array.from({ length: 150 }, () => toKp(jitter(pose('a'), 0.006))),
], 'a') && ok;
ok = run('əl yoxdur — heç nə yazılmır', Array.from({ length: 90 }, () => null), '') && ok;

console.log(ok ? '\n✅ UYĞUNDUR — keçid və boş kadrlar hərf yazmır' : '\n❌ UYĞUN DEYİL');
process.exit(ok ? 0 : 1);
