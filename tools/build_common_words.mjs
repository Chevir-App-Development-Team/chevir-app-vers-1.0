// Ümumi sözlərin kökləri (R6, E30): hərflə yığılan söz ümumi sözdürsə (manat, ədəbiyyat, onlayn) kiçik hərflə,
// yoxsa ad kimi böyük hərflə yazılır (src/tercume/spelled.js). Mənbə: HPLT 2.0 korpusunun cümlələri (CC0,
// tools/corpus_sentences.py → data/corpus/hplt_sentences.txt). Cümlənin ilk sözündən sonra ən azı MIN dəfə kiçik
// hərflə (və böyük hərflə olduğundan az olmayaraq) yazılan sözlər; hər söz öz ən qısa kökünə (ən azı 3 hərf, özü
// də belə söz) enir, siyahıda köklər qalır.
//
//   node tools/build_common_words.mjs [data/corpus/hplt_sentences.txt] [--min 10]  → ml/common_words.txt
import fs from 'node:fs';
import readline from 'node:readline';
import { candidates } from '../src/tercume/morph.js';
import { azLower } from '../src/tercume/words.js';

const args = process.argv.slice(2);
const MIN = Number(args.includes('--min') ? args[args.indexOf('--min') + 1] : 10);
const src = args.find((a, i) => !a.startsWith('--') && args[i - 1] !== '--min') ?? 'data/corpus/hplt_sentences.txt';
const OUT = 'ml/common_words.txt';

const low = new Map(), cap = new Map();
const inc = (m, k) => m.set(k, (m.get(k) ?? 0) + 1);
const word = /^[a-zəöüğıişçA-ZƏÖÜĞIİŞÇ]+$/u, vowel = /[aıoueəiöüAIOUEƏİÖÜ]/u;      // saitsiz (dst, mm) söz deyil
let n = 0;
for await (const line of readline.createInterface({ input: fs.createReadStream(src, 'utf8') })) {
  n++;
  for (let w of line.split(/\s+/).slice(1)) {
    w = w.replace(/^[.,!?;:"«»()–—'’]+|[.,!?;:"«»()–—'’]+$/g, '');
    if (!w || !word.test(w) || !vowel.test(w)) continue;
    if (w[0] === azLower(w[0])) inc(low, w);
    else if (w.length === 1 || w !== w.toLocaleUpperCase('az')) inc(cap, azLower(w));     // DOST kimi qısaltmalar yox
  }
}
const common = new Set([...low].filter(([w, c]) => c >= MIN && c >= (cap.get(w) ?? 0)).map(([w]) => w));
const stems = new Set();
for (const w of common) {
  const c = candidates(w).filter((x) => !x.neg && x.stem.length >= 3 && common.has(x.stem))
    .sort((a, b) => a.stem.length - b.stem.length)[0];
  stems.add(c ? c.stem : w);
}
const list = [...stems].sort((a, b) => a.localeCompare(b, 'az'));
fs.writeFileSync(OUT, `${list.join('\n')}\n`);
console.log(`${src}: ${n} sentences, ${common.size} lowercase words (≥ ${MIN}) → ${list.length} stems, `
  + `${(fs.statSync(OUT).size / 1024).toFixed(0)} KB → ${OUT}`);
