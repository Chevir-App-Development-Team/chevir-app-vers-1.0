// Beam search + leksikon yoxlamasını süni proqnozlarla sınayır.
import { readFileSync } from 'node:fs';
import { LexiconDecoder } from '../src/tercume/decoder.js';

const A = new URL('../public/tercume/assets/', import.meta.url);
const rd = (f) => JSON.parse(readFileSync(new URL(f, A), 'utf8'));
const dec = new LexiconDecoder(rd('lm.json'), rd('vocab.json'), rd('prefixes.json'));
console.log(`lüğət: ${dec.words.length} söz | prefiks: ${dec.prefixes.size}\n`);

const ALPHA = dec.letters;
// Verilən hərfi `pTrue` ehtimalla düzgün, qalanını təsadüfi rəqib kimi qaytarır
function step(ch, pTrue, seed) {
  const others = ALPHA.filter((c) => c !== ch);
  const a = others[(seed * 7) % others.length];
  const b = others[(seed * 13 + 3) % others.length];
  const rest = (1 - pTrue) / 2;
  return [{ label: ch, p: pTrue }, { label: a, p: rest }, { label: b, p: rest }]
    .sort((x, y) => y.p - x.p);
}

function run(name, word, mkSteps) {
  const steps = mkSteps(word);
  for (const mode of ['off', 'soft', 'strict']) {
    const r = dec.decode(steps, { mode, width: 8 });
    const top3 = r.candidates.slice(0, 3)
      .map((c) => `${c.text}${c.inVocab ? '✓' : ''}(${(c.conf * 100).toFixed(0)}%)`).join('  ');
    console.log(`  ${mode.padEnd(7)} → ${top3}`);
  }
  const r = dec.decode(mkSteps(word), { mode: 'soft', width: 8 });
  console.log(`  təsdiq: ${r.verified ?? '—'} | təklif: ${r.suggestions.map(s=>`${s.word}(d${s.dist})`).join(', ')}`);
}

console.log('TEST 1 — təmiz siqnal, hər hərf p=0.75 ("sonra")');
run('t1', 'sonra', (w) => [...w].map((c, i) => step(c, 0.75, i + 1)));

console.log('\nTEST 2 — zəif siqnal, hər hərf p=0.40 ("kimi")');
run('t2', 'kimi', (w) => [...w].map((c, i) => step(c, 0.40, i + 1)));

console.log('\nTEST 3 — 3-cü hərf SƏHV oxunub ("insan" -> 3-cü hərfə model yanılır)');
run('t3', 'insan', (w) => [...w].map((c, i) =>
  i === 2 ? step(c, 0.25, i + 9) : step(c, 0.70, i + 1)));

console.log('\nTEST 4 — uzun söz, orta siqnal ("azərbaycan")');
run('t4', 'azərbaycan', (w) => [...w].map((c, i) => step(c, 0.55, i + 1)));
