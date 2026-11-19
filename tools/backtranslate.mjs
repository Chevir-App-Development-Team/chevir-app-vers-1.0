// V3 geri tərcümə (roadmap V3): mətn → TextToSign → avatar (başsız Chrome, tətbiqin oynatması virtual saatla) →
// avatarın sümüklərindən MediaPipe formasında landmark-lar (tools/backtranslate.js) → quraşdırılmış cümlə modeli
// (public/tercume/assets/cont_model, tətbiqin özündəki kimi pəncərələr və decode) → tanınan işarələr niyyət olunan
// işarələrlə tutuşdurulur (ml/azsl/segment_eval.py Tally: eyni sinif və kadr aralığı kəsişir).
//
// Sabit dəst: AzSLD-nin test cümlələri (heldout_sentence, 49) + set C qaralaması (ml/setc/setc.csv, 150).
// Əsas rəqəm — işarələrin F1-i (sözlər; modelin lüğətində olan niyyət olunan işarələr üzrə). Hərflər ayrıca.
//
//   node tools/backtranslate.mjs                         # hər iki dəst, avatar
//   node tools/backtranslate.mjs --mode source           # istinad: iştirakçının öz klipləri eyni kamera ilə
//   node tools/backtranslate.mjs --set setc --limit 20 --out D:/chevir/eval/v3/x.json
//   node tools/backtranslate.mjs --opts '{"wordHoldMs":200}'   # TextToSign.opts dəyişikliyi (A4 sınaqları)
//   node tools/backtranslate.mjs --avatar '{"guard":false}'     # VrmAvatar sahələri
import { readFileSync, writeFileSync } from 'node:fs';
import { createServer } from 'vite';
import puppeteer from 'puppeteer';

import { LocalRecognizer } from '../src/tercume/recognizer.js';
import * as CT from '../src/tercume/continuous.js';
import { azLower } from '../src/tercume/words.js';

const args = process.argv.slice(2);
const opt = (n, d = null) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : d; };
const SET = opt('--set', 'all');
const LIMIT = Number(opt('--limit', 0));
const MODE = opt('--mode', 'avatar');
const OUT = opt('--out');
const T2S_OPTS = JSON.parse(opt('--opts', '{}'));
const AV_OPTS = JSON.parse(opt('--avatar', '{}'));
const SHOW = Number(opt('--show', 0));
// Hər cümlə K dəfə, saatın K fazasında (1/60 s-nin payları): kadrların təsadüfi düzülüşünə həssaslığı azaldır
const PHASES = Number(opt('--phases', 1));
const root = new URL('../', import.meta.url);

// ---------------------------------------------------------------- cümlələr
function readCsv(file) {
  const text = readFileSync(new URL(file, root), 'utf8');
  const rows = [];
  let row = [], field = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; } else if (c === '"') q = false; else field += c;
    } else if (c === '"') q = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n') { row.push(field.replace(/\r$/, '')); rows.push(row); row = []; field = ''; }
    else field += c;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  const [head, ...body] = rows;
  return body.filter((r) => r.length === head.length).map((r) => Object.fromEntries(head.map((h, k) => [h, r[k]])));
}

function sentences(set = 'all') {
  const out = [];
  if (set === 'all' || set === 'azsld') {
    const seen = new Set();
    for (const r of readCsv('ml/splits/sentences_split.csv')) {
      if (r.heldout_sentence !== 'True' || seen.has(r.folder)) continue;
      seen.add(r.folder);
      out.push({ id: `azsld-${r.folder}`, set: 'azsld', text: r.text.replace(/\s+([?.!,])/g, '$1').trim() });
    }
  }
  if (set === 'all' || set === 'setc') {
    for (const r of readCsv('ml/setc/setc.csv')) out.push({ id: r.id, set: 'setc', text: r.text });
  }
  return out;
}

// ---------------------------------------------------------------- tanıyıcı
const fetchImpl = async (url) => {
  const data = readFileSync(new URL(url));
  return { ok: true, status: 200, json: async () => JSON.parse(data.toString('utf8')), arrayBuffer: async () => data };
};
const rec = await LocalRecognizer.load(new URL('public/tercume/assets/cont_model/', root).href,
  { fetchImpl, merge: new URL('public/tercume/assets/sign_merge.json', root).href });
const M = rec.manifest, C = rec.labels.length;
const sp = { ...CT.STREAM, ...M.stream }, dp = { ...CT.DECODE, ...M.decode };
const unk = M.unknown_label ? rec.labels.indexOf(M.unknown_label) : -1;
const unknown = unk >= 0 ? unk : null, short = CT.shortClasses(rec.labels);
const isLetter = (l) => [...l].length === 1 && /\p{L}/u.test(l);
const SKIP = new Set(['<blank>', '<unk>', '.']);
// Lüğətin işarəsi → modelin (birləşmiş) sinfi: etiketin və ya sözün kiçik hərfli yazılışı ilə
const classByKey = new Map();
rec.labels.forEach((l, i) => { if (!SKIP.has(l)) classByKey.set(azLower(l), i); });
M.labels.forEach((l) => { const g = rec.group(l); const i = rec.labels.indexOf(g); if (i > 0 && !SKIP.has(l)) classByKey.set(azLower(l), i); });
function classOf(it) {
  if (it.kind === 'letter') return classByKey.get(azLower(it.text)) ?? null;
  const keys = it.id?.startsWith('jd-') ? [it.text, it.label] : [it.label, it.text];
  for (const k of keys) {
    if (!k) continue;
    const g = rec.group(k);
    const c = classByKey.get(azLower(g)) ?? classByKey.get(azLower(k));
    if (c != null) return c;
  }
  return null;
}

async function recognize(r) {
  const nan = (n) => new Float32Array(r.T * n).fill(NaN);
  const arr = (a) => Float32Array.from(a, (v) => (v == null ? NaN : v));
  const raw = { T: r.T, hand: arr(r.hand), hand_world: arr(r.hand_world), pose: arr(r.pose), face: nan(204), blend: nan(52) };
  const slice = (a, b) => {
    const o = { T: b - a };
    for (const [k, n] of Object.entries({ hand: 126, hand_world: 126, pose: 27, face: 204, blend: 52 })) o[k] = raw[k].subarray(a * n, b * n);
    return o;
  };
  const P = new Float32Array(r.T * C);
  for (const w of CT.streamWindows(r.T, sp)) {
    const { probs } = await rec.frameProbs(slice(w.a, w.b), r.aspect, { face: false });
    P.set(probs.subarray((w.from - w.a) * C, (w.to - w.a) * C), w.from * C);
  }
  return CT.decode(P, r.T, C, dp, unknown, short).filter((d) => !SKIP.has(rec.labels[d.c]));
}

// ---------------------------------------------------------------- tutuşdurma (segment_eval.Tally)
function match(dets, anns) {
  const pairs = [];
  dets.forEach((d, i) => anns.forEach((x, j) => {
    const ov = Math.min(d.b, x.b) - Math.max(d.a, x.a);
    if (ov > 0 && x.c != null && d.c === x.c) pairs.push([ov / (Math.max(d.b, x.b) - Math.min(d.a, x.a)), i, j]);
  }));
  pairs.sort((p, q) => q[0] - p[0] || p[1] - q[1] || p[2] - q[2]);
  const ud = new Set(), ua = new Set(), out = [];
  for (const [, i, j] of pairs) {
    if (ud.has(i) || ua.has(j)) continue;
    ud.add(i); ua.add(j); out.push([i, j]);
  }
  return out;
}
function edits(hyp, ref) {
  const d = Array.from({ length: ref.length + 1 }, (_, j) => j);
  hyp.forEach((h, i) => {
    let prev = d[0];
    d[0] = i + 1;
    ref.forEach((r, j) => { const cur = d[j + 1]; d[j + 1] = Math.min(d[j + 1] + 1, d[j] + 1, prev + (h !== r)); prev = cur; });
  });
  return d[ref.length];
}
const tally = () => ({ sent: 0, det: 0, tp: 0, ann: 0, all: 0, edits: 0, exact: 0, ldet: 0, ltp: 0, lann: 0,
  frames: 0, up: 0, overlap: 0, body: 0, steps: [], right: 0, elbow: 0 });
function score(t, intended, dets) {
  const words = intended.filter((x) => x.kind === 'word'), letters = intended.filter((x) => x.kind === 'letter');
  const wd = dets.filter((d) => !isLetter(rec.labels[d.c])), ld = dets.filter((d) => isLetter(rec.labels[d.c]));
  const inv = words.filter((x) => x.c != null);
  const tp = match(wd, inv).length;
  const e = edits(wd.map((d) => d.c), inv.map((x) => x.c));
  Object.assign(t, {
    sent: t.sent + 1 / PHASES, det: t.det + wd.length, tp: t.tp + tp, ann: t.ann + inv.length, all: t.all + words.length,
    edits: t.edits + e, exact: t.exact + (e === 0 && inv.length > 0 ? 1 / PHASES : 0),
    ldet: t.ldet + ld.length, ltp: t.ltp + match(ld, letters.filter((x) => x.c != null)).length,
    lann: t.lann + letters.filter((x) => x.c != null).length,
  });
  return { tp, e };
}
function metrics(t) {
  const P = t.tp / Math.max(1, t.det), R = t.tp / Math.max(1, t.ann);
  const LP = t.ltp / Math.max(1, t.ldet), LR = t.ltp / Math.max(1, t.lann);
  const r3 = (v) => Math.round(v * 1000) / 1000;
  const st = [...t.steps].sort((a, b) => a - b);
  return {
    seconds: r3(t.frames / 30 / Math.max(1, t.sent)),
    hands: t.up ? { overlap: r3(t.overlap / t.up), body: r3(t.body / t.up), elbow: r3(t.elbow / Math.max(1, t.right)),
      stepMedianCm: r3(st[st.length >> 1] * 100), stepMaxCm: r3(st.at(-1) * 100) } : null,
    sentences: Math.round(t.sent), signs: t.ann, signs_all: t.all, detections: t.det, matched: t.tp,
    precision: r3(P), recall: r3(R), f1: r3(P + R ? 2 * P * R / (P + R) : 0), wer: r3(t.edits / Math.max(1, t.ann)),
    exact: r3(t.exact / Math.max(1, t.sent)), coverage: r3(t.ann / Math.max(1, t.all)),
    letters: { signs: t.lann, detections: t.ldet, f1: r3(LP + LR ? 2 * LP * LR / (LP + LR) : 0) },
  };
}

// ---------------------------------------------------------------- avatar (başsız Chrome)
const list = sentences(SET).slice(0, LIMIT || undefined);
const server = await createServer({ root: new URL('..', import.meta.url).pathname.replace(/^\/(\w:)/, '$1'),
  server: { port: 0, hmr: false, watch: null }, logLevel: 'error' });   // başqa sessiyanın dəyişikliyi səhifəni yeniləməsin
await server.listen();
const url = server.resolvedUrls.local[0];
const browser = await puppeteer.launch({ headless: true, protocolTimeout: 60 * 60 * 1000,
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--window-size=520,560'],
  defaultViewport: { width: 520, height: 560 } });
const modes = MODE === 'both' ? ['avatar', 'source'] : [MODE];
const tallies = Object.fromEntries(modes.map((m) => [m, { all: tally(), azsld: tally(), setc: tally() }]));
const rows = [];
const t0 = Date.now();
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.error('səhifə:', e.message));
  page.on('console', (m) => { if (m.type() === 'error') console.error('konsol:', m.text()); });
  // IDM .bin endirmələrini tutur: başsız Chrome-a .bin faylları birbaşa diskdən verilir
  await page.setRequestInterception(true);
  page.on('request', (req) => {
    const path = new URL(req.url()).pathname;
    if (!path.endsWith('.bin')) { req.continue(); return; }
    req.respond({ status: 200, contentType: 'application/octet-stream', body: readFileSync(new URL(`../public${path}`, import.meta.url)) });
  });
  await page.goto(`${url}rey/`, { waitUntil: 'networkidle0', timeout: 120000 });
  await page.evaluate(async (t2sOpts, avOpts) => {
    const { VrmAvatar } = await import('/src/tercume/vrm.js');
    const { TextToSign, loadPoses } = await import('/src/tercume/t2s.js');
    const { WordLexicon } = await import('/src/tercume/words.js');
    const { ASSET_BASE, WORDS_BASE } = await import('/src/tercume/paths.js');
    const BT = await import('/tools/backtranslate.js');
    document.body.innerHTML = '<div id="av" style="width:480px;height:540px"></div>';
    const [poses, lexicon] = await Promise.all([loadPoses(`${ASSET_BASE}poses.json`), WordLexicon.load(`${WORDS_BASE}index.json`)]);
    const avatar = new VrmAvatar(document.querySelector('#av'));
    await avatar.load(`${ASSET_BASE}avatar.vrm`);
    avatar.setHand('right');
    const t2s = new TextToSign({ poses, lexicon, targets: [avatar] });
    Object.assign(t2s.opts, t2sOpts);
    window.__bt = await BT.setup({ avatar, t2s, lexicon });
    Object.assign(avatar, avOpts);
  }, T2S_OPTS, AV_OPTS);

  for (const [k, s] of list.entries()) {
    const row = { id: s.id, set: s.set, text: s.text };
    for (const mode of modes) {
      const runs = [];
      for (let k = 0; k < PHASES; k++) {
        const phase = (1000 / 60) * k / PHASES;
        const r = await page.evaluate((text, mode, phase) => window.__bt.run(text, { mode, phase }), s.text, mode, phase);
        for (const x of r.intended) x.c = classOf(x);
        const dets = await recognize(r);
        const sc = score(tallies[mode].all, r.intended, dets);
        score(tallies[mode][s.set], r.intended, dets);
        for (const t of [tallies[mode].all, tallies[mode][s.set]]) {
          t.frames += r.T / PHASES;
          if (mode === 'avatar') {
            t.up += r.check.up; t.overlap += r.check.overlap; t.body += r.check.body; t.steps.push(r.check.maxStep);
            t.right += r.check.right; t.elbow += r.check.elbow;
          }
        }
        const words = r.intended.filter((x) => x.kind === 'word');
        runs.push({
          T: r.T, ...sc, check: r.check,
          ann: words.filter((x) => x.c != null).length, det: dets.filter((d) => !isLetter(rec.labels[d.c])).length,
          intended: r.intended.map((x) => ({ kind: x.kind, id: x.id, text: x.text, label: x.label, c: x.c == null ? null : rec.labels[x.c], a: x.a, b: x.b, via: x.via })),
          detected: dets.map((d) => ({ label: rec.labels[d.c], a: d.a, b: d.b, p: Math.round(d.p * 1000) / 1000 })),
        });
      }
      // Cümlənin sətri: birinci fazanın təfsilatı + bütün fazaların cəmi (tools/compare_v3.mjs cəmləri tutuşdurur)
      const sum = (k) => runs.reduce((a, r) => a + r[k], 0);
      row[mode] = { ...runs[0], tp: sum('tp'), e: sum('e'), ann: sum('ann'), det: sum('det'), T: sum('T') / PHASES, phases: PHASES };
    }
    rows.push(row);
    if ((k + 1) % 10 === 0 || k === list.length - 1) {
      const m = metrics(tallies[modes[0]].all);
      console.log(`${k + 1}/${list.length} ${((Date.now() - t0) / 1000).toFixed(0)} s · ${modes[0]} F1 ${m.f1} (P ${m.precision} R ${m.recall})`);
    }
  }
} finally {
  await browser.close();
  await server.close();
}

const report = { opts: T2S_OPTS, avatar: AV_OPTS, model: M.run, set: SET, phases: PHASES, modes: {} };
for (const mode of modes) {
  report.modes[mode] = Object.fromEntries(Object.entries(tallies[mode]).filter(([, t]) => t.sent).map(([k, t]) => [k, metrics(t)]));
  console.log(`\n${mode}:`);
  for (const [k, m] of Object.entries(report.modes[mode])) {
    console.log(`  ${k.padEnd(6)} ${String(m.sentences).padStart(3)} sentences, ${m.signs} signs the model knows (${(m.coverage * 100).toFixed(0)} %)`
      + ` | sign F1 ${m.f1.toFixed(3)} P ${m.precision.toFixed(3)} R ${m.recall.toFixed(3)} | WER ${(m.wer * 100).toFixed(1)} %`
      + ` | exact ${(m.exact * 100).toFixed(0)} % | letters F1 ${m.letters.f1.toFixed(3)} (${m.letters.signs})`
      + ` | ${m.seconds.toFixed(1)} s/sentence`
      + (m.hands ? ` | hands: overlap ${(m.hands.overlap * 100).toFixed(1)} %, in body ${(m.hands.body * 100).toFixed(1)} %`
        + ` of raised frames, right elbow raised ${(m.hands.elbow * 100).toFixed(1)} %`
        + `, palm step median/max ${m.hands.stepMedianCm.toFixed(1)}/${m.hands.stepMaxCm.toFixed(1)} cm` : ''));
  }
}
if (SHOW) {
  const mode = modes[0];
  for (const r of rows.filter((x) => x[mode].e > 0).slice(0, SHOW)) {
    console.log(`\n${r.id} ${r.text}\n  want: ${r[mode].intended.filter((x) => x.kind === 'word').map((x) => x.c ?? `(${x.text})`).join(' ')}`
      + `\n  got:  ${r[mode].detected.map((d) => d.label).join(' ')}`);
  }
}
report.rows = rows;
if (OUT) writeFileSync(OUT, JSON.stringify(report, null, 1));
console.log(`\n${((Date.now() - t0) / 1000).toFixed(0)} s`);
