// Söz formasının modeli üçün öyrənmə məlumatı (roadmap N2, ml/EXPERIMENTS.md E33; öyrənmə: ml/formtag.py).
// Hər cümlə: işarələrin girişi (src/tercume/formtag.js signTokens — brauzerdəki ilə eyni) və hər işarənin hədəfi:
// gloss2text-in tutuşdurmasından (prepare) sözün əlamətləri, fit()-in saydığı fəzada (felin şəxsi yoxdur — o
// mübtədadan gəlir; xəbərlik true; əvvəlki əvəzliyə uyğun mənsubiyyət "P"). Sözü tapılmayan işarənin hədəfi null.
//
//   node tools/build_form_data.mjs --out D:/chevir/e33/form_data.jsonl [--pseudo a.jsonl,b.jsonl] [--names hplt,native] [--part k/n]
//
// AzSLD: dev cümlələri 10 qata bölünür (fold 0–9, dev-in sırası ilə k % 10), test cümlələri "test"; korpus cütləri
// fold "pseudo" və öz adları ilə (set). eval_gloss_text.mjs --tagger qatların modellərini eyni qayda ilə seçir.
import fs from 'node:fs';
import { prepare, fit } from '../src/tercume/gloss2text.js';
import { signTokens } from '../src/tercume/formtag.js';
import { azLower } from '../src/tercume/words.js';

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const OUT = opt('--out', null);
if (!OUT) throw new Error('--out lazımdır');
const PSEUDO = (opt('--pseudo', '') || '').split(',').filter(Boolean);
const NAMES = (opt('--names', '') || '').split(',');

const A = 'public/tercume/assets/';
const S = JSON.parse(fs.readFileSync(`${A}cont_model/sentences.json`, 'utf8'));
const merge = JSON.parse(fs.readFileSync(`${A}sign_merge.json`, 'utf8'));
const text = Object.fromEntries(fs.readFileSync('ml/label_text.csv', 'utf8').trim().split(/\r?\n/).slice(1).map((r) => r.split(',').slice(0, 2)));
const labels = JSON.parse(fs.readFileSync(`${A}cont_model/manifest.json`, 'utf8')).labels.filter((l) => !l.startsWith('<'));
const lemmas = new Set(fs.readFileSync('ml/slovo_new.csv', 'utf8').trim().split(/\r?\n/).slice(1).map((r) => r.split(',')[1]));
const rows = fs.readFileSync('ml/splits/sentences_split.csv', 'utf8').trim().split(/\r?\n/).map((r) => r.split(','));
const fi = rows[0].indexOf('folder'), hi = rows[0].indexOf('heldout_sentence');
const held = new Set(rows.slice(1).filter((r) => r[hi] === 'True').map((r) => r[fi]));

export const FOLDS = 10;
/** AzSLD cümləsinin qatı: test cümləsi "test", dev cümləsi dev-in sırası ilə k % 10. */
export function foldsOf(sentences, heldout) {
  let k = 0;
  return sentences.map((s) => (heldout.has(s.folder) ? 'test' : String(k++ % FOLDS)));
}
const fold = foldsOf(S, held);
// AzSLD öz başına tutuşdurulur (eval_gloss_text.mjs-dəki kimi); girişin növü (pos) brauzerdəki gloss_text.json-un
// guess-i kimi yalnız AzSLD + tanıyıcının etiketlərindən (korpusun tanıyıcıda olmayan işarələri lemmatizer-i dəyişməsin)
const azsld = prepare(S, merge, labels, text, lemmas);
const { guess } = azsld;
// --part k/n: korpus cütlərinin yalnız j % n == k olanları (paralel işə salmaq üçün; AzSLD yalnız 0-cı hissədə). Bütün
// hissələrdə eyni etiket çoxluğu (bütün faylların işarələri + tanıyıcınınkılar): lemmatizer, deməli tutuşdurma eynidir
const [PART, PARTS] = (opt('--part', '0/1')).split('/').map(Number);
const items = PART === 0 ? azsld.sentences.map((s, i) => ({ s, set: 'azsld', fold: fold[i], i })) : [];
const files = PSEUDO.map((file) => fs.readFileSync(file, 'utf8').split(/\r?\n/).filter(Boolean).map((l) => JSON.parse(l)));
const all = [...new Set([...labels, ...files.flat().flatMap((p) => p.glosses)])];
files.forEach((pairs, j) => {
  const mine = pairs.filter((_, q) => q % PARTS === PART);
  console.log(`${PSEUDO[j]}: ${mine.length} of ${pairs.length} cümlə, tutuşdurulur…`);
  for (const s of prepare(mine, merge, all, text, lemmas).sentences) items.push({ s, set: NAMES[j] || `pseudo${j}`, fold: 'pseudo' });
});

// Hədəfin əlamətləri öz kökünə görədir: gloss2text-in bu işarə üçün işlədəcəyi kök deyilsə (korpusda HAQQINDA ←
// haqqınızda: kök haqq + yerlik + mənsubiyyət, brauzerdə isə kök "haqqında"dır), hədəf atılır. İşlənən köklər: AzSLD-nin
// dev cümlələrindəki kökləri (fit-in lex-i, növlərlə) və adından çıxan kök (guess); yer adlarında adın özü
const devLex = fit(azsld, (i) => fold[i] !== 'test').lex;
const nrm = (r) => r.replace(/ı/g, 'i');
const rootsOf = new Map();
const usable = (l, pos, root) => {
  if (!rootsOf.has(l)) {
    const r = new Set();
    const lx = devLex[l];
    for (const o of lx ? (lx.options ?? [lx]) : []) r.add(`${o.pos}:${nrm(o.root)}`);
    if (guess[l]) r.add(`${guess[l].pos}:${nrm(guess[l].root)}`);
    r.add(`noun:${nrm(azLower(l.split('/')[0]))}`);
    rootsOf.set(l, r);
  }
  return rootsOf.get(l).has(`${pos}:${nrm(root)}`);
};
const SUBJECT = { mən: 1, sən: 2, o: 3, biz: 4, siz: 5, onlar: 6 };
const fkey = (f) => JSON.stringify(f, Object.keys(f).sort());
const out = fs.createWriteStream(OUT, 'utf8');
let n = 0, t = 0, dropped = 0;
for (const { s, set, fold: f, i } of items) {
  const y = s.glosses.map((l, j) => {
    const a = s.info[j];
    if (!a.feats) return null;
    if (!usable(l, a.pos, a.root)) { dropped++; return null; }
    const ft = { ...a.feats };
    if (a.pos === 'verb') delete ft.pers;
    if (ft.cop) ft.cop = true;
    if (ft.poss && ft.poss === SUBJECT[azLower((s.glosses[j - 1] ?? '').split('/')[0])]) ft.poss = 'P';
    return fkey(ft);
  });
  t += y.filter(Boolean).length;
  n++;
  out.write(JSON.stringify({ set, fold: f, i, glosses: s.glosses, x: signTokens(s.glosses, guess), y }) + '\n');
}
out.end();
fs.writeFileSync(OUT.replace(/\.jsonl$/, '') + '_guess.json', JSON.stringify(guess));
console.log(`${OUT}: ${n} cümlə, ${t} hədəfli işarə, ${dropped} hədəf başqa kökə görə olduğu üçün atıldı`);
