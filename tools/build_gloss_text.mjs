// İşarələr → mətn modeli (src/tercume/gloss2text.js) brauzer üçün: AzSLD-nin 492 cümləsinin hamısı
// üzərində öyrənilir (ölçmə: tools/eval_gloss_text.mjs, orada test cümlələri öyrənməyə girmir).
// İşarələr cümlə modelinin birləşmiş adlarındadır (cont_model/sentences.json, sign_merge.json);
// cümlələrdə olmayan, ancaq tanınan işarələrin kökü öz adından çıxarılır.
//
//   node tools/build_gloss_text.mjs [--tagger ml/form_tagger.json | none] [--lambda 0.6] [--off]
//     → public/tercume/assets/gloss_text.json
//
// Sözün formasının modeli (E33, ml/formtag.py install → ml/form_tagger.json) modelin içinə `formTagger` kimi yazılır (susmaya
// görə, fayl varsa); gloss2text onu `on` true-dursa işlədir. --off: yazılır, işlənmir (brauzerdə GlossToText({tagger}) ilə
// açmaq olar); --tagger none: yazılmır (yalnız qaydalar və saylar). --lambda: modelin payı (susmaya görə gloss2text LAMBDA).
import fs from 'node:fs';
import { train, LAMBDA } from '../src/tercume/gloss2text.js';

const A = 'public/tercume/assets/';
const S = JSON.parse(fs.readFileSync(`${A}cont_model/sentences.json`, 'utf8'));
const merge = JSON.parse(fs.readFileSync(`${A}sign_merge.json`, 'utf8'));
// etiketin düzgün yazılışı (annotasiyadakı yazı xətaları)
const text = Object.fromEntries(fs.readFileSync('ml/label_text.csv', 'utf8').trim().split(/\r?\n/).slice(1)
  .map((r) => r.split(',').slice(0, 2)));
const labels = JSON.parse(fs.readFileSync(`${A}cont_model/manifest.json`, 'utf8')).labels.filter((l) => !l.startsWith('<'));
// Slovo-dan yeni işarələr (ml/slovo_new.csv) lüğət formasındadır, təsrifli fel deyil (E29)
const lemmas = new Set(fs.readFileSync('ml/slovo_new.csv', 'utf8').trim().split(/\r?\n/).slice(1).map((r) => r.split(',')[1]));
const model = train(S, merge, labels, text, lemmas);
delete model.opts;                                  // yalnız öyrənmə üçün
// hərflə yığılan söz ümumi sözdürsə ad deyil (R6, E30): korpusun kökləri (tools/build_common_words.mjs)
model.common = fs.readFileSync('ml/common_words.txt', 'utf8').trim().split(/\r?\n/).join('\n');
const opt = (n) => { const i = process.argv.indexOf(n); return i >= 0 ? process.argv[i + 1] : null; };
const TAGGER = opt('--tagger') ?? 'ml/form_tagger.json';
if (TAGGER !== 'none' && fs.existsSync(TAGGER)) {
  const t = JSON.parse(fs.readFileSync(TAGGER, 'utf8'));
  model.formTagger = { ...t, on: !process.argv.includes('--off'), lambda: Number(opt('--lambda') ?? LAMBDA) };
}
model.source = `AzSLD sentences (${S.length}) of cont_model/sentences.json; labels of cont_model/manifest.json; ml/label_text.csv`;
const out = `${A}gloss_text.json`;
fs.writeFileSync(out, JSON.stringify(model));
console.log(`${out}: ${(fs.statSync(out).size / 1024).toFixed(0)} KB, ${Object.keys(model.lex).length} signs seen in sentences, `
  + `${Object.keys(model.guess).length} with a guessed root, ${Object.keys(model.feats).length} context tables`
  + (model.formTagger ? `, form tagger ${model.formTagger.on ? 'on' : 'off'} (lambda ${model.formTagger.lambda ?? 'default'})` : ''));
