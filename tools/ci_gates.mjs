// Ölçmə hədləri (roadmap V5, docs/NLP_PLAN.md §3): mətn → işarə və işarələr → mətn ölçmələri işə salınır, rəqəmlər
// ml/gates.json-dakı hədlərlə müqayisə olunur. Hədd: test WER ən çoxu 1 bənd artır, F1 ən çoxu 0.005 düşür.
//   node tools/ci_gates.mjs
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';

const gates = JSON.parse(fs.readFileSync('ml/gates.json', 'utf8'));
const run = (script) => execFileSync(process.execPath, [script], { encoding: 'utf8', maxBuffer: 1 << 26 });
let fails = 0;
const check = (name, got, op, limit) => {
  const ok = got != null && (op === '>=' ? got >= limit : got <= limit);
  console.log(`${ok ? '✓' : '✗'} ${name}: ${got} (hədd ${op} ${limit})`);
  if (!ok) fails++;
};

// Mətn → işarə: hər blokun "test" sətri (AzSLD üslubu və doğma qaydalar)
const t2s = run('tools/eval_text_gloss.mjs');
const block = (title) => t2s.split(/\n\n/).find((b) => b.includes(title)) ?? '';
const testLine = (b) => b.split('\n').find((l) => l.trim().startsWith('test')) ?? '';
for (const [key, title] of [['t2s_azsld', 'AzSLD style'], ['t2s_native', 'native rules']]) {
  const l = testLine(block(title));
  const f1 = Number(/same gloss: P [\d.]+ R [\d.]+ F1 ([\d.]+)/.exec(l)?.[1]);
  const signs = Number(/words as signs\s+([\d.]+)%/.exec(l)?.[1]);
  check(`${key} test same-gloss F1`, f1, '>=', gates[key].f1);
  check(`${key} test words as signs %`, signs, '>=', gates[key].signs);
}

// İşarələr → mətn: annotasiyanın işarələri, dev (LOO) və test WER
const s2t = run('tools/eval_gloss_text.mjs');
for (const split of ['dev', 'test']) {
  const l = s2t.split('\n').find((x) => x.trim().startsWith(split)) ?? '';
  const wer = Number(/gloss2text\s+([\d.]+) %/.exec(l)?.[1]);
  check(`s2t ${split} WER %`, wer, '<=', gates.s2t[split]);
}

// Sözün formasının modeli ilə (E33): qatların modelləri repoda deyil (12 × 1 MB), ona görə yalnız qovluq verilibsə —
// FORM_TAGGER_FOLDS=<qovluq> və ya --tagger <qovluq> (ml/formtag.py train-in çıxışı: f0–f9, dev, all)
const ti = process.argv.indexOf('--tagger');
const folds = ti >= 0 ? process.argv[ti + 1] : process.env.FORM_TAGGER_FOLDS;
if (folds && gates.s2t_tagger) {
  const out = execFileSync(process.execPath, ['tools/eval_gloss_text.mjs', '--tagger', folds], { encoding: 'utf8', maxBuffer: 1 << 26 });
  for (const split of ['dev', 'test']) {
    const l = out.split('\n').find((x) => x.trim().startsWith(split)) ?? '';
    check(`s2t + form tagger ${split} WER %`, Number(/gloss2text\s+([\d.]+) %/.exec(l)?.[1]), '<=', gates.s2t_tagger[split]);
  }
  const l = out.split('\n').find((x) => x.trim().startsWith('new sentence')) ?? '';
  check('s2t + form tagger recognized new sentence WER %', Number(/\| gloss2text\s+([\d.]+) %/.exec(l)?.[1]), '<=', gates.s2t_tagger.new);
} else if (gates.s2t_tagger) console.log('· s2t + form tagger: qatların modelləri verilməyib (FORM_TAGGER_FOLDS / --tagger), yoxlanmadı');
console.log(fails ? `\n${fails} hədd keçilmədi` : '\nbütün hədlər keçdi');
process.exit(fails ? 1 : 0);
