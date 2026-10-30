// Word-form families among the sentence model's labels (morph.js keys; negated forms get their own key).
import fs from 'node:fs';
import { Lemmatizer } from '../../src/tercume/morph.js';
import { azLower } from '../../src/tercume/words.js';
const m = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const labels = m.labels.filter((l) => /\p{L}{2}/u.test(l) && !l.startsWith('<'));
const single = labels.filter((l) => !l.includes(' '));
const lem = new Lemmatizer(single.map(azLower));
const fam = new Map();
for (const l of single) { const k = lem.key(azLower(l)) ?? azLower(l); fam.set(k, [...(fam.get(k) ?? []), l]); }
const out = [...fam].filter(([, v]) => v.length > 1).map(([key, v]) => ({ key, labels: v }));
fs.writeFileSync(process.argv[3], JSON.stringify(out, null, 0));
console.log(`${out.length} families, ${out.flatMap((f) => f.labels).length} labels`);
