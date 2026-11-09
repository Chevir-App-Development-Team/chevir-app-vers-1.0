// Süni (mətn, işarələr) cütləri: mətn → işarə yolu (WordLexicon.segment, iştirakçılarla uyğunluğu F1 ≈ 0.9,
// ml/EXPERIMENTS.md E18) korpusun cümlələrini işarələrə çevirir. İşarələr → mətn modeli (gloss2text) formanı
// seçməyi AzSLD-nin 492 cümləsindən çox nümunədə öyrənsin (docs/NLP_PLAN.md S2).
//
//   node tools/build_pseudo_gloss.mjs data/corpus/hplt_sentences.txt data/corpus/pseudo_gloss.jsonl [--min 0.8]
//
// Saxlanır: sözlərin ən azı 80 %-i tanıyıcının bildiyi işarədir (cont_model/manifest.json, sign_merge.json
// birləşmələri ilə), ən azı 2 işarə. İştirakçıların göstərmədiyi söz (ilə, idi, ki) sayılmır; hərf-hərf
// göstərilən söz hərflərdir (AzSLD-dəki kimi: D O S T).
import fs from 'node:fs';
import readline from 'node:readline';
import { WordLexicon } from '../src/tercume/words.js';

const [src, dst] = process.argv.slice(2);
const opt = (name, def) => { const i = process.argv.indexOf(name); return i >= 0 ? process.argv[i + 1] : def; };
const MIN = Number(opt('--min', 0.8));

const WORDS = 'public/tercume/words';
const index = JSON.parse(fs.readFileSync(`${WORDS}/index.json`, 'utf8'));
const labelOf = new Map(index.words.map((w) => [w.id, JSON.parse(fs.readFileSync(`${WORDS}/${w.id}.json`, 'utf8')).label]));
const merge = JSON.parse(fs.readFileSync('public/tercume/assets/sign_merge.json', 'utf8'));
const known = new Set(JSON.parse(fs.readFileSync('public/tercume/assets/cont_model/manifest.json', 'utf8')).labels);
const canon = (g) => { const s = merge.spelling?.[g] ?? g; return merge.merge?.[s] ?? s; };
const lex = new WordLexicon(index, { native: false });   // AzSLD üslubu: tanıyıcının görəcəyi işarələr kimi
// AzSLD-nin öz cümlələri (dövlət xidmətlərinin saytlarında da ola bilər) korpusdan çıxarılır: ölçmə təmiz qalsın
const norm = (t) => t.toLocaleLowerCase('az').replace(/[^\p{L}\d]+/gu, ' ').trim();
const azsld = new Set(JSON.parse(fs.readFileSync('public/tercume/assets/cont_model/sentences.json', 'utf8')).map((s) => norm(s.text)));

const out = fs.createWriteStream(dst, 'utf8');
let n = 0, kept = 0;
const rl = readline.createInterface({ input: fs.createReadStream(src, 'utf8'), crlfDelay: Infinity });
let leak = 0;
for await (const text of rl) {
  n++;
  if (azsld.has(norm(text))) { leak++; continue; }
  const glosses = [];
  let words = 0, signs = 0;
  for (const seg of lex.segment(text)) {
    if (seg.kind === 'space' || seg.kind === 'drop') continue;
    const shown = text.slice(seg.from, seg.to).split(/\s+/).length;
    if (seg.kind === 'word') {
      const l = canon(labelOf.get(seg.entry.id));
      glosses.push(l);
      if (seg.via !== 'neg') { words += shown; if (known.has(l)) signs += shown; }
    } else if (/\p{L}/u.test(seg.text)) {
      glosses.push(...[...seg.text.toLocaleUpperCase('az')].filter((c) => /\p{L}/u.test(c)));
      words += shown;
    }
  }
  if (words && signs / words >= MIN && glosses.filter((g) => known.has(g)).length >= 2) {
    out.write(JSON.stringify({ text, glosses }) + '\n');
    kept++;
  }
  if (n % 100000 === 0) console.log(`  ${n} cümlə, ${kept} saxlanıb`);
}
out.end();
console.log(`${n} cümlə → ${kept} süni cüt (${(100 * kept / n).toFixed(1)} %), AzSLD-nin ${leak} cümləsi çıxarıldı: ${dst}`);
