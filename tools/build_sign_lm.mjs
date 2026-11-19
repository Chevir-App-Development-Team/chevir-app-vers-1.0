// İşarə ardıcıllıqlarının dil modeli (N3, E32): cümlə modelinin deşifrəsi (continuous.js decodeLM / nbest) bir
// işarənin ardınca hansının gələ biləcəyini bilsin. Mənbə: mətn korpusunun süni glosları (E18,
// tools/build_pseudo_gloss.mjs: data/corpus/pseudo_gloss.jsonl + pseudo_dialog.jsonl, git-də deyil); işarələr
// cümlə modelinin birləşmiş etiketlərinə gətirilir (sign_merge.json), modelin bilmədiyi işarə ardıcıllıqdan
// çıxarılır (tanıyıcı da <unk>-u atır), hərflər bir sinifdir (<L>: hansı hərf olduğunu kadrlar deyir).
//
// Model: interpolyasiyalı Kneser-Ney (sabit endirim, hər dərəcə üçün n1 / (n1 + 2 n2)), ən aşağı dərəcə bərabər
// paylanma ilə qarışdırılır (korpusda olmayan işarə də ehtimalsız qalmır). Saxlanma ARPA kimi: görünmüş
// n-qramın ln-ehtimalı və kontekstin geri çəkilmə çəkisi; Python (continuous.py SignLM) və brauzer (continuous.js
// SignLM) eyni ədədləri oxuyur. AzSLD cümlələri susmaya görə girmir: AzSLD-nin görünməmiş cümlələrində kömək etmədi (E32),
// ölçmədə test cümlələri modeldə olmasın. Bigram: trigram AzSLD val-da ondan yaxşı deyildi, 4 dəfə böyükdür.
// `decode`: deşifrənin E32-də seçilmiş parametrləri (continuous.js LM_DECODE-un açarları), brauzer bunları oxuyur.
//
//   node tools/build_sign_lm.mjs [--order 2] [--min 2] [--uniform 0.05] [--azsld N [--skip-heldout]] [--out public/tercume/assets/sign_lm.json]
import fs from 'node:fs';
import { lmToken } from '../src/tercume/continuous.js';

const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : d; };
const ORDER = Number(opt('--order', 2));
const MIN = Number(opt('--min', 2));            // ən yüksək dərəcənin bu saydan az görünən n-qramları saxlanmır
const UNIFORM = Number(opt('--uniform', 0.05));  // ən aşağı dərəcədə bərabər paylanmanın payı
const OUT = opt('--out', 'public/tercume/assets/sign_lm.json');
const INPUTS = (opt('--in', 'data/corpus/pseudo_gloss.jsonl,data/corpus/pseudo_dialog.jsonl')).split(',');
const A = 'public/tercume/assets/';
// AzSLD val-da seçilib (E32, v5 kimi: görünməmiş cümlələrin öyrənilə bilən sözlərinin F1-i, bütün val-ın F1-i azalmadan);
// temp — alternativlərin payları birincinin doğru olma tezliyinə ən yaxın (val, ECE 0.10)
const DECODE_LM = { alpha: 0.1, beta: 0.5, norm: 0, top: 4, floor: 0.05, beam: 8, nbest: 3, temp: 0.4 };
const labels = JSON.parse(fs.readFileSync(`${A}cont_model/manifest.json`, 'utf8')).labels;
const merge = JSON.parse(fs.readFileSync(`${A}sign_merge.json`, 'utf8'));
const group = (g) => { const c = merge.spelling?.[g] ?? g; return merge.merge?.[c] ?? c; };
const known = new Set(labels.filter((l) => !l.startsWith('<')));

// lüğət: modelin etiketlərinin tokenləri + cümlənin başı və sonu
const tokens = ['<s>', '</s>', ...new Set(labels.filter((l) => !l.startsWith('<')).map(lmToken))];
const id = new Map(tokens.map((t, i) => [t, i]));
const V = tokens.length - 1;                    // <s> heç vaxt proqnozlaşdırılmır

// --azsld N: AzSLD cümlələrinin işarələri də, N dəfə (işarə dilinin öz sırası); --skip-heldout: 49 heldout cümləsi
// girmir (ölçmə üçün: o cümlələrin test yazıları modeldə olmayan cümlələrdir)
const AZSLD = Number(opt('--azsld', 0));
const extra = [];
if (AZSLD) {
  const rows = fs.readFileSync('ml/splits/sentences_split.csv', 'utf8').trim().split(/\r?\n/).map((r) => r.split(','));
  const fi = rows[0].indexOf('folder'), hi = rows[0].indexOf('heldout_sentence');
  const held = new Set(rows.slice(1).filter((r) => r[hi] === 'True').map((r) => r[fi]));
  for (const s of JSON.parse(fs.readFileSync(`${A}cont_model/sentences.json`, 'utf8'))) {
    if (args.includes('--skip-heldout') && held.has(s.folder)) continue;
    for (let k = 0; k < AZSLD; k++) extra.push(JSON.stringify({ glosses: s.glosses }));
  }
}

// ardıcıllıqlar (eyni ardıcıllıq bir dəfə: veb mətnində təkrarlar çoxdur; AzSLD-ninkilər N dəfə)
const seen = new Set(), seqs = [];
let dropped = 0, total = 0;
for (const f of [...INPUTS, null]) {
  for (const line of f ? fs.readFileSync(f, 'utf8').split(/\r?\n/) : extra) {
    if (!line.trim()) continue;
    const g = (JSON.parse(line).glosses ?? []).map(group);
    total += g.length;
    // modelin bilmədiyi hərf (I, X, Ğ ...) yenə hərfdir; bilmədiyi söz atılır
    const t = g.map((x) => (known.has(x) ? lmToken(x) : lmToken(x) === '<L>' ? '<L>' : null));
    dropped += t.filter((x) => x === null).length;
    const s = t.filter(Boolean);
    if (s.length < 2) continue;
    const key = s.join('\t');
    if (f && seen.has(key)) continue;
    seen.add(key);
    seqs.push(s.map((x) => id.get(x)));
  }
}

// sayılar: ən yüksək dərəcə adi say, aşağılar davam sayı (neçə fərqli sol qonşu ilə görünüb)
const BOS = id.get('<s>'), EOS = id.get('</s>');
const counts = Array.from({ length: ORDER + 1 }, () => new Map());   // counts[n]: "h1 … hn-1 w" → say
const lefts = Array.from({ length: ORDER + 1 }, () => new Map());    // davam sayı üçün: n-qram → sol qonşular
for (const s of seqs) {
  const x = [...Array(ORDER - 1).fill(BOS), ...s, EOS];
  for (let i = ORDER - 1; i < x.length; i++) {
    const g = x.slice(i - ORDER + 1, i + 1).join(' ');
    counts[ORDER].set(g, (counts[ORDER].get(g) ?? 0) + 1);
  }
}
for (let n = ORDER; n > 1; n--) {               // n-qramdan (n−1)-qramın davam sayı: fərqli sol qonşular
  for (const g of counts[n].keys()) {
    const parts = g.split(' '), lower = parts.slice(1).join(' ');
    // <s> <s> w-dakı kimi cümlə başı: sol qonşusu yoxdur, adi say kimi sayılır
    lefts[n - 1].set(lower, (lefts[n - 1].get(lower) ?? 0) + 1);
  }
  counts[n - 1] = lefts[n - 1];
}

// endirim: hər dərəcə üçün D = n1 / (n1 + 2 n2)
const D = [0];
for (let n = 1; n <= ORDER; n++) {
  let n1 = 0, n2 = 0;
  for (const c of counts[n].values()) { if (c === 1) n1++; else if (c === 2) n2++; }
  D[n] = n1 / (n1 + 2 * n2) || 0.5;
}

// kontekst cəmləri: c(h) = Σ_w c(h w), N1+(h •) = fərqli w sayı
const ctx = Array.from({ length: ORDER + 1 }, () => new Map());
for (let n = 1; n <= ORDER; n++) {
  for (const [g, c] of counts[n]) {
    const h = g.split(' ').slice(0, -1).join(' ');
    const e = ctx[n].get(h) ?? { sum: 0, types: 0 };
    e.sum += c; e.types++;
    ctx[n].set(h, e);
  }
}

// ehtimallar: P_n(w | h) = max(c − D, 0) / c(h) + D·N1+(h •)/c(h) · P_{n−1}(w | h'); P_0 = 1 / V
const prob = (n, h, w) => {
  if (n === 0) return 1 / V;
  const e = ctx[n].get(h.slice(h.length - (n - 1)).join(' '));
  const lower = prob(n - 1, h, w);
  if (n === 1) {
    const c = counts[1].get(String(w)) ?? 0, tot = ctx[1].get('').sum;
    return (1 - UNIFORM) * (c / tot) + UNIFORM * lower;
  }
  if (!e) return lower;
  const c = counts[n].get([...h.slice(h.length - (n - 1)), w].join(' ')) ?? 0;
  return Math.max(c - D[n], 0) / e.sum + (D[n] * e.types / e.sum) * lower;
};
const r4 = (x) => Math.round(x * 1e4) / 1e4;

// saxlanma: p1 hər token üçün; p2, p3 görünmüş (və ən yüksək dərəcədə ≥ MIN dəfə) n-qramlar; b — geri çəkilmə
const out = { version: 1, order: ORDER, decode: DECODE_LM, tokens, letter: '<L>', discount: D.slice(1).map(r4), uniform: UNIFORM,
  source: `${INPUTS.map((f) => f.split('/').pop()).join(' + ')}${AZSLD ? ` + AzSLD sentences x${AZSLD}` : ''}: ${seqs.length} distinct sign sequences of the text corpus (E18 pseudo-glosses); `
    + 'labels of cont_model/manifest.json via sign_merge.json; tools/build_sign_lm.mjs',
  p1: tokens.map((t, w) => (w === BOS ? -99 : r4(Math.log(prob(1, [], w))))) };
for (let n = 2; n <= ORDER; n++) {
  const p = {}, b = {};
  for (const [g, c] of counts[n]) {
    if (n === ORDER && c < MIN) continue;
    const parts = g.split(' ').map(Number), h = parts.slice(0, -1), w = parts.at(-1);
    (p[h.join(' ')] ??= []).push(w, r4(Math.log(prob(n, h, w))));
  }
  for (const [h, e] of ctx[n]) if (p[h]) b[h] = r4(Math.log(D[n] * e.types / e.sum));
  out[`p${n}`] = p;
  out[`b${n - 1}`] = b;
}
// p1-in konteksti olaraq b1 (bigram konteksti = bir token), b2 (iki token) — lm_logp bunları oxuyur
fs.writeFileSync(OUT, JSON.stringify(out));
const n2 = Object.values(out.p2 ?? {}).reduce((a, x) => a + x.length / 2, 0), n3 = Object.values(out.p3 ?? {}).reduce((a, x) => a + x.length / 2, 0);
console.log(`${OUT}: ${(fs.statSync(OUT).size / 1024).toFixed(0)} KB · ${seqs.length} sequences (${total} signs, ${dropped} unknown to the model dropped)`
  + ` · ${tokens.length} tokens · ${n2} bigrams · ${n3} trigrams (count ≥ ${MIN}) · D ${out.discount.join(' ')}`);
