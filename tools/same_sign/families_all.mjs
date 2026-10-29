// Word-form families over the labels of both browser models, with negated forms tagged (morph.js key "x-").
//   node tools/same_sign/families_all.mjs <out.json> <manifest.json>...
import fs from 'node:fs';
import { Lemmatizer } from '../../src/tercume/morph.js';
import { azLower } from '../../src/tercume/words.js';
const [out, ...mans] = process.argv.slice(2);
const labels = [...new Set(mans.flatMap((p) => JSON.parse(fs.readFileSync(p, 'utf8')).labels))]
  .filter((l) => /\p{L}{2}/u.test(l) && !l.startsWith('<') && !l.includes(' '));
const lem = new Lemmatizer(labels.map(azLower));
const fam = new Map();
for (const l of labels) {
  const k = lem.key(azLower(l)) ?? azLower(l);
  const root = k.replace(/-$/, '');
  if (!fam.has(root)) fam.set(root, []);
  fam.get(root).push({ label: l, neg: k.endsWith('-'), base: lem.isBase(azLower(l)) });
}
const res = [...fam].filter(([, v]) => v.length > 1).map(([key, forms]) => ({ key, forms }));
fs.writeFileSync(out, JSON.stringify(res));
console.log(`${labels.length} word labels, ${res.length} families, ${res.flatMap((f) => f.forms).length} forms, `
  + `${res.filter((f) => f.forms.some((x) => x.neg)).length} with a negated form`);
