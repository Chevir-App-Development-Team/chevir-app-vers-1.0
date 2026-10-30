// İşarələr → mətn modeli (src/tercume/gloss2text.js) brauzer üçün: AzSLD-nin 492 cümləsinin hamısı
// üzərində öyrənilir (ölçmə: tools/eval_gloss_text.mjs, orada test cümlələri öyrənməyə girmir).
// İşarələr cümlə modelinin birləşmiş adlarındadır (cont_model/sentences.json, sign_merge.json);
// cümlələrdə olmayan, ancaq tanınan işarələrin kökü öz adından çıxarılır.
//
//   node tools/build_gloss_text.mjs   → public/tercume/assets/gloss_text.json
import fs from 'node:fs';
import { train } from '../src/tercume/gloss2text.js';

const A = 'public/tercume/assets/';
const S = JSON.parse(fs.readFileSync(`${A}cont_model/sentences.json`, 'utf8'));
const merge = JSON.parse(fs.readFileSync(`${A}sign_merge.json`, 'utf8'));
// etiketin düzgün yazılışı (annotasiyadakı yazı xətaları)
const text = Object.fromEntries(fs.readFileSync('ml/label_text.csv', 'utf8').trim().split(/\r?\n/).slice(1)
  .map((r) => r.split(',').slice(0, 2)));
const labels = JSON.parse(fs.readFileSync(`${A}cont_model/manifest.json`, 'utf8')).labels.filter((l) => !l.startsWith('<'));
const model = train(S, merge, labels, text);
model.source = `AzSLD sentences (${S.length}) of cont_model/sentences.json; labels of cont_model/manifest.json; ml/label_text.csv`;
const out = `${A}gloss_text.json`;
fs.writeFileSync(out, JSON.stringify(model));
console.log(`${out}: ${(fs.statSync(out).size / 1024).toFixed(0)} KB, ${Object.keys(model.lex).length} signs seen in sentences, `
  + `${Object.keys(model.guess).length} with a guessed root, ${Object.keys(model.feats).length} context tables`);
