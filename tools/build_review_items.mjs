// İşarə dili daşıyıcılarının rəyi (roadmap V2, set D) üçün bəndlər: public/rey/items.json.
//
// Mətn → işarə (t2s): avatar cümləni göstərir, rəy verən "avatar nə dedi?" yazır, sonra mətni görüb anlaşıqlığı və
// təbiiliyi 1–5 qiymətləndirir. Cümlələr: müsahibənin avatar cümlələri (docs/SIGNER_INTERVIEW.md, M4), AzSLD-nin
// test cümlələri (heldout), set C-nin dev yarısından məhsulun mövzularında cümlələr (ml/setc/setc.csv).
// İşarə → mətn (s2t): avatar AzSLD iştirakçılarının test cümləsində göstərdiyi işarələri oynadır, altında cümlə
// modelinin tanıdığı işarələrdən qurulan mətn (gloss2text, bu cümlə öyrənməyə girmədən) görünür; məna düzgünlüyü
// və dil 1–5. Tanınan işarələr: ml/results/cont_v15/final_merged/detections_test.csv (yerli, git-də yoxdur).
// Videolar səhifəyə girmir (repo publikdir): iştirakçının işarələri avatarla göstərilir.
//
//   node tools/build_review_items.mjs [--t2s-azsld 20] [--t2s-setc 17] [--s2t 20]
import fs from 'node:fs';
import { execSync } from 'node:child_process';
import { WordLexicon, azLower } from '../src/tercume/words.js';
import { GlossToText, prepare, fit } from '../src/tercume/gloss2text.js';

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const N_AZSLD = Number(opt('--t2s-azsld', 20)), N_SETC = Number(opt('--t2s-setc', 17)), N_S2T = Number(opt('--s2t', 20));
const DET = opt('--detections', 'ml/results/cont_v15/final_merged/detections_test.csv');
const A = 'public/tercume/assets/', W = 'public/tercume/words/';

const csv = (file) => fs.readFileSync(file, 'utf8').trim().split(/\r?\n/).map((r) => r.split(','));
// dırnaqlı sahələri olan CSV (set C-nin cümlələrində vergül var)
function csvQuoted(text) {
  const out = [];
  let row = [], f = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"' && text[i + 1] === '"') { f += '"'; i++; } else if (c === '"') q = false; else f += c; }
    else if (c === '"') q = true;
    else if (c === ',') { row.push(f); f = ''; }
    else if (c === '\n') { row.push(f.replace(/\r$/, '')); out.push(row); row = []; f = ''; }
    else f += c;
  }
  if (f || row.length) { row.push(f); out.push(row); }
  const [head, ...body] = out;
  return body.filter((r) => r.length === head.length).map((r) => Object.fromEntries(head.map((h, k) => [h, r[k]])));
}
const S = JSON.parse(fs.readFileSync(`${A}cont_model/sentences.json`, 'utf8'));
const rows = csv('ml/splits/sentences_split.csv');
const fi = rows[0].indexOf('folder'), hi = rows[0].indexOf('heldout_sentence');
const held = new Set(rows.slice(1).filter((r) => r[hi] === 'True').map((r) => r[fi]));
const testIdx = S.map((_, i) => i).filter((i) => held.has(S[i].folder));

// Bərabər aralıqla seçmə (təsadüf yox: hər dəfə eyni bəndlər)
const spread = (list, n) => (list.length <= n ? list : Array.from({ length: n }, (_, k) => list[Math.floor((k * list.length) / n)]));

// 1. Mətn → işarə
const t2s = [];
const doc = fs.readFileSync('docs/SIGNER_INTERVIEW.md', 'utf8');
const m4 = doc.slice(doc.indexOf('### M4.'));
for (const line of m4.split('\n')) {
  const m = /^\| ([^|]+?[.?!]) \| [^|]+ \| ([^|]+) \|$/.exec(line.trim());
  if (m) t2s.push({ source: 'müsahibə M4', text: m[1].trim(), checks: m[2].trim() });
}
for (const i of spread(testIdx, N_AZSLD)) t2s.push({ source: `AzSLD test ${S[i].folder}`, text: S[i].text.replace(/\s+([?.!,])/g, '$1') });
// set C-nin dev yarısından, mövzulardan bərabər (test yarısı rəyə də girmir)
const setc = 'ml/setc/setc.csv';
if (fs.existsSync(setc)) {
  const dev = csvQuoted(fs.readFileSync(setc, 'utf8')).filter((r) => r.split === 'dev');
  for (const r of spread(dev, N_SETC)) t2s.push({ source: `set C dev ${r.id} (${r.domain})`, text: r.text });
}

// 2. İşarə → mətn
const index = JSON.parse(fs.readFileSync(`${W}index.json`, 'utf8'));
const azsld = index.words.filter((w) => !w.id.startsWith('jd-'));
const byLabel = new Map();
for (const w of azsld) {
  const { label } = JSON.parse(fs.readFileSync(`${W}${w.id}.json`, 'utf8'));
  if (label && !byLabel.has(label)) byLabel.set(label, w.id);
}
const text = Object.fromEntries(csv('ml/label_text.csv').slice(1).map((r) => r.slice(0, 2)));
const spelling = Object.fromEntries(csv('ml/gloss_spelling.csv').slice(1).map((r) => r.slice(0, 2)));
const lex = new WordLexicon({ ...index, sources: [], words: azsld }, { native: false });
// AzSLD-nin işarəsi → avatarın sözü: etiketin özü, yoxsa mətninin lüğətdəki işarəsi, yoxsa hərf-hərf
const signOf = (g) => {
  const label = spelling[g] ?? g;
  if (byLabel.has(label)) return [{ id: byLabel.get(label), gloss: g }];
  const t = azLower(text[label] ?? label);
  return lex.segment(t).filter((s) => s.kind === 'word' || s.kind === 'text')
    .map((s) => (s.kind === 'word' ? { id: s.entry.id, gloss: g } : { spell: s.text, gloss: g }));
};

const s2t = [];
if (fs.existsSync(DET)) {
  const merge = JSON.parse(fs.readFileSync(`${A}sign_merge.json`, 'utf8'));
  const labels = JSON.parse(fs.readFileSync(`${A}cont_model/manifest.json`, 'utf8')).labels.filter((l) => !l.startsWith('<'));
  const prep = prepare(S, merge, labels, text);
  const seqs = new Map();
  for (const line of fs.readFileSync(DET, 'utf8').trim().split(/\r?\n/).slice(1)) {
    const f = line.split(',');
    if (!seqs.has(f[0])) seqs.set(f[0], []);
    seqs.get(f[0]).push([Number(f[1]), f.at(-2)]);
  }
  // hər test cümləsindən bir yazı (ilki): tanınan işarələr → mətn, model bu cümləsiz
  const firstRec = new Map();
  for (const rec of [...seqs.keys()].sort()) {
    const folder = rec.split('/')[0];
    if (!firstRec.has(folder)) firstRec.set(folder, rec);
  }
  for (const i of spread(testIdx.filter((i) => firstRec.has(S[i].folder)), N_S2T)) {
    const recognized = seqs.get(firstRec.get(S[i].folder)).sort((a, b) => a[0] - b[0]).map((x) => x[1]);
    const g2t = new GlossToText(fit(prep, (j) => j !== i));
    s2t.push({
      source: `AzSLD test ${S[i].folder}`,
      signs: S[i].glosses.flatMap(signOf),
      recognized,
      text: g2t.generate(recognized).text,
      reference: S[i].text.replace(/\s+([?.!,])/g, '$1'),
    });
  }
} else {
  console.warn(`${DET} yoxdur: işarə → mətn bəndləri yazılmır`);
}

const commit = execSync('git rev-parse --short HEAD').toString().trim();
const items = [
  ...t2s.map((x, k) => ({ id: `t${String(k + 1).padStart(2, '0')}`, dir: 't2s', ...x })),
  ...s2t.map((x, k) => ({ id: `s${String(k + 1).padStart(2, '0')}`, dir: 's2t', ...x })),
];
fs.mkdirSync('public/rey', { recursive: true });
fs.writeFileSync('public/rey/items.json', JSON.stringify({
  release: `${new Date().toISOString().slice(0, 10)} ${commit}`,
  note: 'tools/build_review_items.mjs; mətn → işarə doğma qaydalarla (E22), işarə → mətn gloss2text (E23)',
  items,
}, null, 1));
console.log(`public/rey/items.json: ${t2s.length} mətn → işarə, ${s2t.length} işarə → mətn (buraxılış ${commit})`);
