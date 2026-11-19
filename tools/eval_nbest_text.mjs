// Cümlə modelinin alternativləri (N-best, continuous.py nbest / continuous.js nbest) → mətn, AzSLD-nin test yazılarında
// "yeni cümlə" kimi (tools/eval_gloss_text.mjs-in 2-ci bölməsi: gloss2text cümlənin özü olmadan öyrənilir). E32.
//
//   node tools/eval_nbest_text.mjs --nbest <fayl.jsonl> [--k 3] [--wordlm data/corpus/hplt_sentences.txt --gamma 0,0.05 --mu 0,6]
//
// Fayl: hər sətir {id: "<qovluq>/<yazı>", ref: [annotasiyanın işarələri], alts: [{score, labels}]}, ən yaxşısı əvvəl;
// heç nə tanınmayan yazı da var (alts boş və ya boş labels: mətn boşdur). Ölçülər:
//   top-1         birinci alternativin mətni (deşifrənin özü)
//   oracle@k      ilk k alternativdən mətni istinada ən yaxın olan (alternativ göstərmənin tavanı)
//   gold          annotasiyanın işarələri (tanıma səhvsiz olsaydı)
//   işarələr      top-1 ardıcıllığı annotasiya ilə eynidir; annotasiya ilk k alternativdədir
//   --wordlm      söz səviyyəsində yenidən sıralama: alternativin balı + gamma · (mətnin söz modelindəki ln-ehtimalı + mu · söz sayı)
//                 (korpusdan söz bigramı, AzSLD-nin cümlələri korpusdan çıxarılır; yalnız alternativlərin sözləri sayılır)
import fs from 'node:fs';
import readline from 'node:readline';
import { prepare, fit, GlossToText, textWords } from '../src/tercume/gloss2text.js';
import { levenshtein } from '../src/tercume/continuous.js';

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const NB = opt('--nbest', null);
const K = Number(opt('--k', 3));
const A = 'public/tercume/assets/';
const S = JSON.parse(fs.readFileSync(`${A}cont_model/sentences.json`, 'utf8'));
const merge = JSON.parse(fs.readFileSync(`${A}sign_merge.json`, 'utf8'));
const text = Object.fromEntries(fs.readFileSync('ml/label_text.csv', 'utf8').trim().split(/\r?\n/).slice(1)
  .map((r) => r.split(',').slice(0, 2)));
const labels = JSON.parse(fs.readFileSync(`${A}cont_model/manifest.json`, 'utf8')).labels.filter((l) => !l.startsWith('<'));
const lemmas = new Set(fs.readFileSync('ml/slovo_new.csv', 'utf8').trim().split(/\r?\n/).slice(1).map((r) => r.split(',')[1]));
const COMMON = fs.readFileSync('ml/common_words.txt', 'utf8').trim().split(/\r?\n/).join('\n');
const prep = prepare(S, merge, labels, text, lemmas);
const byFolder = new Map(S.map((s, i) => [s.folder, i]));

const recs = fs.readFileSync(NB, 'utf8').trim().split(/\r?\n/).map((l) => JSON.parse(l))
  .map((r) => ({ ...r, i: byFolder.get(r.id.split('/')[0]) })).filter((r) => r.i !== undefined).sort((a, b) => a.i - b.i);

// hər cümlə üçün bir model (cümlə özü olmadan), yazıları bitəndə atılır
let cur = -1, model = null;
for (const r of recs) {
  if (r.i !== cur) { cur = r.i; model = new GlossToText({ ...fit(prep, (j) => j !== r.i), common: COMMON }); }
  r.texts = (r.alts.length ? r.alts : [{ score: 0, labels: [] }]).map((a) => (a.labels.length ? model.generate(a.labels).text : ''));
  r.gold = model.generate(S[r.i].glosses).text;
  r.refText = S[r.i].text;
}

const werOf = (hyp, ref) => levenshtein(textWords(hyp), textWords(ref));
function score(pick) {
  let e = 0, n = 0;
  for (const r of recs) { const ref = textWords(r.refText); e += levenshtein(textWords(pick(r)), ref); n += ref.length; }
  return `${(100 * e / n).toFixed(1)} %`;
}
const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
console.log(`${recs.length} test recordings, ${NB}`);
console.log(`  text WER, new sentence: top-1 ${score((r) => r.texts[0])} | oracle@2 ${score((r) => r.texts.slice(0, 2).reduce((b, t) => (werOf(t, r.refText) < werOf(b, r.refText) ? t : b)))}`
  + ` | oracle@${K} ${score((r) => r.texts.slice(0, K).reduce((b, t) => (werOf(t, r.refText) < werOf(b, r.refText) ? t : b)))}`
  + ` | gold glosses ${score((r) => r.gold)}`);
if (recs[0].ref) {
  const exact = recs.filter((r) => same(r.alts[0]?.labels ?? [], r.ref)).length;
  const inK = recs.filter((r) => r.alts.slice(0, K).some((a) => same(a.labels, r.ref))).length;
  console.log(`  signs: top-1 = annotation ${(100 * exact / recs.length).toFixed(1)} % | annotation among the first ${K} ${(100 * inK / recs.length).toFixed(1)} %`);
}

// söz səviyyəsində yenidən sıralama: alternativlərin sözlərinin bigram modeli (interpolyasiyalı mütləq endirim)
const WLM = opt('--wordlm', null);
if (WLM) {
  const vocab = new Set();
  for (const r of recs) for (const t of r.texts) for (const w of textWords(t)) vocab.add(w);
  const norm = (t) => t.toLocaleLowerCase('az').replace(/[^\p{L}\d]+/gu, ' ').trim();
  const azsld = new Set(S.map((s) => norm(s.text)));
  const uni = new Map(), bi = new Map();
  let tokens = 0, skipped = 0;
  const rl = readline.createInterface({ input: fs.createReadStream(WLM, 'utf8'), crlfDelay: Infinity });
  for await (const line of rl) {
    if (azsld.has(norm(line))) { skipped++; continue; }
    const ws = ['<s>', ...textWords(line).map((w) => (vocab.has(w) ? w : '<u>')), '</s>'];
    for (let i = 1; i < ws.length; i++) {
      uni.set(ws[i], (uni.get(ws[i]) ?? 0) + 1);
      const k = `${ws[i - 1]} ${ws[i]}`;
      bi.set(k, (bi.get(k) ?? 0) + 1);
      tokens++;
    }
  }
  const ctx = new Map(), types = new Map();
  for (const [k, c] of bi) { const h = k.split(' ')[0]; ctx.set(h, (ctx.get(h) ?? 0) + c); types.set(h, (types.get(h) ?? 0) + 1); }
  const D = 0.75, V = vocab.size + 2;
  const pu = (w) => 0.9 * ((uni.get(w) ?? 0) / tokens) + 0.1 / V;
  const pb = (h, w) => { const c = ctx.get(h); if (!c) return pu(w); return Math.max((bi.get(`${h} ${w}`) ?? 0) - D, 0) / c + (D * types.get(h) / c) * pu(w); };
  const lmText = (t) => { const ws = ['<s>', ...textWords(t).map((w) => (vocab.has(w) ? w : '<u>')), '</s>']; let s = 0; for (let i = 1; i < ws.length; i++) s += Math.log(pb(ws[i - 1], ws[i])); return s; };
  for (const r of recs) r.lm = r.texts.map(lmText);
  console.log(`  word LM: ${tokens} tokens of ${WLM} (${skipped} AzSLD sentences left out), ${vocab.size} words of the alternatives`);
  // gamma — söz modelinin çəkisi, mu — hər mətn sözünə bonus (söz modeli qısa mətni üstün tutur)
  const MUS = opt('--mu', '0').split(',').map(Number);
  for (const g of opt('--gamma', '0,0.02,0.05,0.1,0.2').split(',').map(Number)) {
    for (const mu of MUS) {
      const val = (r, j) => (r.alts[j]?.score ?? 0) + g * (r.lm[j] + mu * textWords(r.texts[j]).length);
      const pick = (r) => { let b = 0; r.texts.forEach((_, j) => { if (val(r, j) > val(r, b)) b = j; }); return r.texts[b]; };
      console.log(`  rerank gamma ${g} mu ${mu}: text WER ${score(pick)}`);
    }
  }
}
