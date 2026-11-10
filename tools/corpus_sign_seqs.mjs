// Mətn korpusunun cümlələrinin işarə ardıcıllıqları, cümlə modelinin öyrənməsi üçün (roadmap R1, notebook 06
// CORPUS): tools/build_pseudo_gloss.mjs-in süni cütlərindən yalnız işarələr, bir sətir bir cümlə, tab ilə.
// Mətn yazılmır. Hərf-hərf göstərilən sözlər (ən azı iki tək simvol dalbadal: D O S T, 5 2) çıxarılır: hərflərin
// öyrənməsi ayrıcadır (spell), burada sözlərin sırası lazımdır. Eyni ardıcıllıq bir dəfə.
//
//   node tools/corpus_sign_seqs.mjs data/corpus/pseudo_gloss.jsonl [data/corpus/pseudo_dialog.jsonl ...] --out <fayl>
import fs from 'node:fs';

const args = process.argv.slice(2);
const oi = args.indexOf('--out');
if (oi < 0 || oi + 1 >= args.length) throw new Error('--out <fayl> lazımdır');
const out = args[oi + 1];
const inputs = args.filter((_, i) => i !== oi && i !== oi + 1);
const seen = new Set();
const lines = [];
for (const f of inputs) {
  for (const line of fs.readFileSync(f, 'utf8').split(/\r?\n/)) {
    if (!line.trim()) continue;
    const all = JSON.parse(line).glosses ?? [];
    const one = (k) => [...(all[k] ?? '')].length === 1;
    const g = all.filter((x, k) => !(one(k) && (one(k - 1) || one(k + 1))));
    if (g.length < 2 || g.some((x) => x.includes('\t'))) continue;
    const key = g.join('\t');
    if (seen.has(key)) continue;
    seen.add(key);
    lines.push(key);
  }
}
fs.writeFileSync(out, lines.join('\n') + '\n', 'utf8');
console.log(`${lines.length} ardıcıllıq → ${out}`);
