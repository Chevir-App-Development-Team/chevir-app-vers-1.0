// Söz rejiminin bölücüsünü (src/tercume/segment.js) Python ilə tutuşdurur:
// ml/fixtures/segment_v<N>.json — təsadüfi girişlər və ml/azsl/segment.py-ın cavabları.
import { readFileSync } from 'node:fs';
import * as SG from '../src/tercume/segment.js';

const fx = JSON.parse(readFileSync(new URL(`../ml/fixtures/segment_v${SG.SEGMENT_VERSION}.json`, import.meta.url), 'utf8'));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const camel = (p) => ({
  restLine: p.rest_line, minOn: p.min_on, minOff: p.min_off, margin: p.margin, minSpan: p.min_span,
  maxSpan: p.max_span, lens: p.lens, stride: p.stride, minP: p.min_p, maxOverlap: p.max_overlap,
});
const f32 = (list) => Float32Array.from(list, (v) => (v === null ? NaN : v));

let ok = true;
const report = (name, fails, n, first) => {
  console.log(`${fails ? '❌' : '✅'} ${name.padEnd(34)} ${n - fails}/${n} eyni${first ? `  — ilk fərq: ${first}` : ''}`);
  if (fails) ok = false;
};

if (!same(camel(fx.params), SG.SEGMENT)) {
  ok = false;
  console.log('❌ standart parametrlər fərqlidir:', JSON.stringify(camel(fx.params)), '≠', JSON.stringify(SG.SEGMENT));
}

// 1. kadr aktivliyi (əl istirahət xəttindən yuxarıda?)
{
  let fails = 0, first = '';
  for (const [i, c] of fx.activity.entries()) {
    const got = SG.activity({ T: c.T, hand: f32(c.hand), pose: f32(c.pose) }, c.aspect, c.rest_line);
    if (!same(got, c.active)) { fails++; first ||= `hal ${i}`; }
  }
  report('aktivlik (frameActive)', fails, fx.activity.length, first);
}

// 2. aralıqlar — onlayn SignSegmenter ↔ Python spans()
{
  let fails = 0, first = '';
  for (const [i, c] of fx.spans.entries()) {
    const seg = new SG.SignSegmenter(camel(c.params));
    const got = [];
    for (const a of c.active) { const s = seg.push(a); if (s) got.push([s.lo, s.hi, ...s.core]); }
    const last = seg.flush();
    if (last) got.push([last.lo, last.hi, ...last.core]);
    if (!same(got, c.spans)) { fails++; first ||= `hal ${i}: JS ${JSON.stringify(got)}, Python ${JSON.stringify(c.spans)}`; }
  }
  report('aralıqlar (SignSegmenter)', fails, fx.spans.length, first);
}

// 3. namizəd pəncərələr
{
  let fails = 0, first = '';
  for (const [i, c] of fx.windows.entries()) {
    const [lo, hi, c0, c1] = c.span;
    const got = SG.windows({ lo, hi, core: [c0, c1] }, camel(c.params));
    if (!same(got, c.windows)) { fails++; first ||= `hal ${i}`; }
  }
  report('pəncərələr (windows)', fails, fx.windows.length, first);
}

// 4. seçim: maksimum olmayanların atılması + eyni sözlü qonşuların birləşməsi
{
  let fails = 0, first = '';
  for (const [i, c] of fx.pick.entries()) {
    const scored = c.scored.map(([a, b, label, p]) => ({ a, b, label, p }));
    const got = SG.pick(scored, camel(c.params)).map((w) => [w.a, w.b, w.label, w.p]);
    if (!same(got, c.kept)) { fails++; first ||= `hal ${i}: JS ${JSON.stringify(got)}, Python ${JSON.stringify(c.kept)}`; }
  }
  report('seçim (pick)', fails, fx.pick.length, first);
}

console.log(ok ? `\n✅ UYĞUNDUR — söz bölücüsü Python segment.py ilə eynidir (segment_version ${SG.SEGMENT_VERSION})`
  : '\nUYĞUNSUZLUQ VAR.');
process.exit(ok ? 0 : 1);
