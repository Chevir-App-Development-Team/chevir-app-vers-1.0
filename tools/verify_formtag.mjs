// Sözün formasının modelinin JS portunu (src/tercume/formtag.js) Python ilə tutuşdurur (ml/formtag.py fixture):
// ml/fixtures/formtag_v<N>.json — AzSLD-nin və korpusun bir neçə cümləsi, görünməmiş işarə, Python-un ehtimalları
// (float16 çəkilərlə). Model fixture-də yazılmış fayldır (quraşdırılmış ml/form_tagger.json); girişlər (signTokens)
// gloss_text.json-un guess-i ilə yoxlanır; gloss_text.json-un içindəki formTagger (brauzerin işlətdiyi) həmin modeldir.
//   node tools/verify_formtag.mjs [--model <json>]
import { readFileSync, existsSync } from 'node:fs';
import { FormTagger, signTokens, FT_VERSION } from '../src/tercume/formtag.js';

const args = process.argv.slice(2);
const fx = JSON.parse(readFileSync(new URL(`../ml/fixtures/formtag_v${FT_VERSION}.json`, import.meta.url), 'utf8'));
const i = args.indexOf('--model');
const path = i >= 0 ? args[i + 1] : fx.model;
const file = existsSync(path) ? path : new URL(`../${path}`, import.meta.url);
const json = JSON.parse(readFileSync(file, 'utf8'));
const tagger = new FormTagger(json);
const g2t = JSON.parse(readFileSync(new URL('../public/tercume/assets/gloss_text.json', import.meta.url), 'utf8'));

let ok = true, worst = 0, toks = 0;
const fail = (msg) => { ok = false; console.log('❌', msg); };
for (const cs of fx.cases) {
  const got = tagger.probsOf(cs.x);
  if (got.length !== cs.probs.length) { fail(`${cs.glosses.join(' ')}: ${got.length} ≠ ${cs.probs.length}`); continue; }
  got.forEach((p, t) => {
    const e = cs.probs[t];
    if (p.length !== e.length) fail(`${cs.glosses.join(' ')}: sinif sayı ${p.length} ≠ ${e.length}`);
    for (let c = 0; c < e.length; c++) worst = Math.max(worst, Math.abs(p[c] - e[c]));
  });
  // girişlər: AzSLD cümlələri brauzerin guess-i ilə eyni tokenlərə çevrilir
  if (cs.azsld) {
    const x = signTokens(cs.glosses, g2t.guess);
    if (JSON.stringify(x) !== JSON.stringify(cs.x)) fail(`signTokens ${cs.glosses.join(' ')}: ${JSON.stringify(x)} ≠ ${JSON.stringify(cs.x)}`);
    else toks++;
  }
}
if (worst > 1e-4) fail(`ehtimalların maks fərqi ${worst.toExponential(2)} > 1e-4`);
// brauzerin modeli: gloss_text.json-un içindəki formTagger bu fayldır (lüğət, siniflər, çəkilər)
const emb = g2t.formTagger;
if (emb) {
  const same = JSON.stringify(emb.vocab) === JSON.stringify(json.vocab) && JSON.stringify(emb.classes) === JSON.stringify(json.classes)
    && Object.keys(json.weights).every((k) => emb.weights[k]?.data === json.weights[k].data);
  if (!same) fail(`gloss_text.json-un formTagger-i ${path} deyil (tools/build_gloss_text.mjs --tagger ${path})`);
}
console.log(`${ok ? '✓' : '✗'} formtag: ${fx.cases.length} cümlə, ehtimalların maks fərqi ${worst.toExponential(2)}, `
  + `${toks} AzSLD cümləsinin girişi eyni (${path}); gloss_text.json: ${emb ? `formTagger ${emb.on ? 'on' : 'off'}, `
  + `lambda ${emb.lambda ?? 'default'}` : 'formTagger yoxdur'}`);
process.exit(ok ? 0 : 1);
