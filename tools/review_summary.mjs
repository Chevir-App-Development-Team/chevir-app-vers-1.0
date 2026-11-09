// /rey/ səhifəsindən endirilən CSV-lərin xülasəsi (roadmap V2, set D): istiqamət və mənbə üzrə orta qiymətlər,
// rəy verənlərin razılığı, ən aşağı qiymət alan bəndlər. Fayllar yerli saxlanır (data/review/, git-də yoxdur).
//
//   node tools/review_summary.mjs data/review/*.csv [--worst 8]
import fs from 'node:fs';

const args = process.argv.slice(2);
const wi = args.indexOf('--worst');
const WORST = wi >= 0 ? Number(args[wi + 1]) : 8;
const files = args.filter((a, i) => a.endsWith('.csv') && (wi < 0 || i !== wi + 1));
if (!files.length) { console.log('istifadə: node tools/review_summary.mjs <fayl.csv> …'); process.exit(1); }

function parse(text) {
  const rows = [];
  let row = [], f = '', q = false;
  text = text.replace(/^﻿/, '');
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"' && text[i + 1] === '"') { f += '"'; i++; } else if (c === '"') q = false; else f += c; }
    else if (c === '"') q = true;
    else if (c === ',') { row.push(f); f = ''; }
    else if (c === '\n') { row.push(f.replace(/\r$/, '')); rows.push(row); row = []; f = ''; }
    else f += c;
  }
  if (f || row.length) { row.push(f); rows.push(row); }
  const [head, ...body] = rows;
  return body.filter((r) => r.length === head.length).map((r) => Object.fromEntries(head.map((h, k) => [h, r[k]])));
}

const rows = files.flatMap((f) => parse(fs.readFileSync(f, 'utf8')));
const num = (v) => (v === '' || v === undefined ? null : Number(v));
const mean = (xs) => { const v = xs.filter((x) => x !== null); return v.length ? (v.reduce((a, b) => a + b, 0) / v.length).toFixed(2) : '-'; };
const releases = [...new Set(rows.map((r) => r.release))];
const reviewers = [...new Set(rows.map((r) => r.reviewer))];
console.log(`${rows.length} cavab, ${reviewers.length} rəy verən (${reviewers.join(', ')}), buraxılış: ${releases.join('; ')}\n`);

const SCALES = { t2s: ['comprehension', 'naturalness'], s2t: ['adequacy', 'fluency'] };
const NAMES = { comprehension: 'anlaşıqlıq', naturalness: 'təbiilik', adequacy: 'məna', fluency: 'dil' };
for (const [dir, keys] of Object.entries(SCALES)) {
  const rs = rows.filter((r) => r.direction === dir);
  if (!rs.length) continue;
  console.log(`${dir === 't2s' ? 'mətn → işarə' : 'işarə → mətn'}: ${rs.length} cavab | ` + keys.map((k) => `${NAMES[k]} ${mean(rs.map((r) => num(r[k])))}`).join(' | '));
  const bySource = new Map();
  for (const r of rs) { const s = r.source.replace(/ \d+$/, ''); bySource.set(s, [...(bySource.get(s) ?? []), r]); }
  for (const [s, list] of bySource) console.log(`  ${s.padEnd(22)} ${String(list.length).padStart(3)} | ` + keys.map((k) => `${NAMES[k]} ${mean(list.map((r) => num(r[k])))}`).join(' | '));
  // eyni bəndə iki rəy verən: qiymətlərin fərqi (razılıq)
  const byItem = new Map();
  for (const r of rs) byItem.set(r.item, [...(byItem.get(r.item) ?? []), r]);
  const pairs = [...byItem.values()].filter((l) => l.length > 1);
  if (pairs.length) {
    for (const k of keys) {
      const d = pairs.map((l) => Math.max(...l.map((r) => num(r[k]) ?? NaN)) - Math.min(...l.map((r) => num(r[k]) ?? NaN))).filter(Number.isFinite);
      console.log(`  razılıq (${NAMES[k]}): ${pairs.length} bənd, fərq ≤ 1: ${(100 * d.filter((x) => x <= 1).length / Math.max(1, d.length)).toFixed(0)} %`);
    }
  }
  const worst = [...byItem].map(([id, l]) => [id, l, Number(mean(l.map((r) => num(r[keys[0]]))))]).filter((x) => Number.isFinite(x[2]))
    .sort((a, b) => a[2] - b[2]).slice(0, WORST);
  console.log(`  ən aşağı ${NAMES[keys[0]]}:`);
  for (const [id, l, m] of worst) {
    const r = l[0];
    const said = l.map((x) => x.answer || x.correction).filter(Boolean).join(' / ');
    console.log(`    ${id} ${m.toFixed(1)}  ${r.text}${said ? `  →  ${said}` : ''}${l.some((x) => x.comment) ? `  [${l.map((x) => x.comment).filter(Boolean).join(' / ')}]` : ''}`);
  }
  console.log();
}
