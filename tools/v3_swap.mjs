// V3-ün diaqnozu (E37): avatarın itkisi hansı hissədən gəlir? Hər cümlə bir dəfə oynanır, hər kadrda həm avatarın,
// həm iştirakçının öz klipinin kadrı (tools/backtranslate.js, dual) götürülür; sonra kadrlar qarışdırılır — məsələn,
// avatarın kadrı, amma sol əl klipdən — və hər qarışıq cümlə modelindən keçirilir (tools/backtranslate.mjs kimi).
// Qarışığın tanıması avatardan nə qədər yaxşıdırsa, itkinin o qədəri həmin hissədədir.
//
//   node tools/v3_swap.mjs [--set azsld|setc] [--limit N] [--mixes avatar,source,left,...] [--out x.json]
//   node tools/v3_swap.mjs --text "Mən Bakıda yaşayıram" --frames --out x.json   # bir cümlə, kadrlar da yazılır
//
// Qarışıqlar (slot 0 — sağ əl, 1 — sol əl):
//   avatar, source         təmiz
//   pose                   avatarın əlləri, klipin pozası (çiyin, dirsək, bilək nöqtələri)
//   right, left, hands     avatar, amma sağ / sol / hər iki əl klipdən (yer + forma + aşkarlanma)
//   detect                 avatar, amma əlin aşkarlanması klipdəki kimi (avatarda yoxdursa klipin əli, klipdə yoxdursa yox)
//   shape                  avatarın əl yerləri (bilək), klipin əl forması və oriyentasiyası (hər ikisində əl varsa)
//   place                  klipin əl yerləri, avatarın əl forması
//   depth                  avatar, amma əlin şəkil z-i klipdən (yalnız dərinlik)
//   world, image           avatar, amma əlin world landmark-ları / şəkil forması klipdən (hər ikisində əl varsa)
//   orient, fingers        avatarın barmaqları klipin oriyentasiyasında / klipin barmaqları avatarın oriyentasiyasında
//   worient, wfingers, wsize   eyni, yalnız world landmark-larında; wsize — avatarın world əli klipin ölçüsündə
// Üstəlik hər söz üçün həndəsə: əllərin aşkarlanması, ovucun yeri (çiyin eni vahidində), iki əl arası, forma fərqi.
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
const OUT = opt('--out');
const T2S_OPTS = JSON.parse(opt('--opts', '{}'));
const AV_OPTS = JSON.parse(opt('--avatar', '{}'));     // VrmAvatar sahələri, məs. {"wordDepth":1}
const TEXT = opt('--text');
const FRAMES = args.includes('--frames');
const MIXES = opt('--mixes', 'avatar,source,pose,right,left,hands,detect,shape,place,depth').split(',');
const root = new URL('../', import.meta.url);

// ---------------------------------------------------------------- cümlələr (backtranslate.mjs ilə eyni dəst)
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

// ---------------------------------------------------------------- tanıyıcı (backtranslate.mjs ilə eyni)
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
async function recognize(r, aspect) {
  const nan = (n) => new Float32Array(r.T * n).fill(NaN);
  const raw = { T: r.T, hand: r.hand, hand_world: r.hand_world, pose: r.pose, face: nan(204), blend: nan(52) };
  const slice = (a, b) => {
    const o = { T: b - a };
    for (const [k, n] of Object.entries({ hand: 126, hand_world: 126, pose: 27, face: 204, blend: 52 })) o[k] = raw[k].subarray(a * n, b * n);
    return o;
  };
  const P = new Float32Array(r.T * C);
  for (const w of CT.streamWindows(r.T, sp)) {
    const { probs } = await rec.frameProbs(slice(w.a, w.b), aspect, { face: false });
    P.set(probs.subarray((w.from - w.a) * C, (w.to - w.a) * C), w.from * C);
  }
  return CT.decode(P, r.T, C, dp, unknown, short).filter((d) => !SKIP.has(rec.labels[d.c]));
}
const hit = (x, dets) => dets.some((d) => d.c === x.c && Math.min(d.b, x.b) - Math.max(d.a, x.a) > 0);

// ---------------------------------------------------------------- qarışıqlar
const ASPECT = 4 / 3, Z_SCALE = 0.46;
const PALM = [0, 5, 9, 13, 17];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const centroid = (P) => [0, 1, 2].map((q) => P.reduce((s, p) => s + p[q], 0) / P.length);
const rot = (R, v) => [0, 1, 2].map((i) => R[i][0] * v[0] + R[i][1] * v[1] + R[i][2] * v[2]);
/** P-ni Q-ya ən yaxşı fırladan matris (Horn-un kvaternion üsulu, güc iterasiyası). */
function kabsch(P, Q) {
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  P.forEach((p, n) => { for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) S[i][j] += p[i] * Q[n][j]; });
  const [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = S;
  const N = [[xx + yy + zz, yz - zy, zx - xz, xy - yx], [yz - zy, xx - yy - zz, xy + yx, zx + xz],
    [zx - xz, xy + yx, -xx + yy - zz, yz + zy], [xy - yx, zx + xz, yz + zy, -xx - yy + zz]];
  const shift = N.reduce((s, r) => s + r.reduce((a, v) => a + Math.abs(v), 0), 0);
  let q = [1, 0.1, 0.1, 0.1];
  for (let it = 0; it < 200; it++) {
    const nq = [0, 1, 2, 3].map((i) => N[i].reduce((a, v, j) => a + v * q[j], 0) + shift * q[i]);
    const l = Math.hypot(...nq) || 1;
    q = nq.map((v) => v / l);
  }
  const [w, x, y, z] = q;
  return [[w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z]];
}
const arr = (a) => Float32Array.from(a, (v) => (v == null ? NaN : v));
const has = (h, t, s) => Number.isFinite(h[t * 126 + s * 63]);
function mix(name, A, S, T) {
  if (name === 'avatar') return A;
  if (name === 'source') return S;
  const o = { T, hand: A.hand.slice(), hand_world: A.hand_world.slice(), pose: A.pose.slice() };
  const copySlot = (t, s, from) => {
    const k = t * 126 + s * 63;
    o.hand.set(from.hand.subarray(k, k + 63), k);
    o.hand_world.set(from.hand_world.subarray(k, k + 63), k);
  };
  for (let t = 0; t < T; t++) {
    if (name === 'pose') o.pose.set(S.pose.subarray(t * 27, t * 27 + 27), t * 27);
    for (let s = 0; s < 2; s++) {
      const a = has(A.hand, t, s), b = has(S.hand, t, s), k = t * 126 + s * 63;
      if (name === 'hands' || (name === 'right' && s === 0) || (name === 'left' && s === 1)) copySlot(t, s, S);
      else if (name === 'detect') {
        if (!a && b) copySlot(t, s, S);
        else if (a && !b) { o.hand.fill(NaN, k, k + 63); o.hand_world.fill(NaN, k, k + 63); }
      } else if ((name === 'shape' || name === 'place') && a && b) {
        // shape: klipin əli avatarın biləyinə köçürülür; place: avatarın əli klipin biləyinə
        const [from, to] = name === 'shape' ? [S, A] : [A, S];
        const dx = to.hand[k] - from.hand[k], dy = to.hand[k + 1] - from.hand[k + 1];
        for (let j = 0; j < 21; j++) {
          o.hand[k + j * 3] = from.hand[k + j * 3] + dx;
          o.hand[k + j * 3 + 1] = from.hand[k + j * 3 + 1] + dy;
          o.hand[k + j * 3 + 2] = from.hand[k + j * 3 + 2];
        }
        o.hand_world.set(from.hand_world.subarray(k, k + 63), k);
      } else if (name === 'depth' && a && b) {
        for (let j = 0; j < 21; j++) o.hand[k + j * 3 + 2] = S.hand[k + j * 3 + 2];
      } else if (name === 'world' && a && b) {
        o.hand_world.set(S.hand_world.subarray(k, k + 63), k);
      } else if (name === 'image' && a && b) {
        const dx = A.hand[k] - S.hand[k], dy = A.hand[k + 1] - S.hand[k + 1];
        for (let j = 0; j < 21; j++) {
          o.hand[k + j * 3] = S.hand[k + j * 3] + dx;
          o.hand[k + j * 3 + 1] = S.hand[k + j * 3 + 1] + dy;
          o.hand[k + j * 3 + 2] = S.hand[k + j * 3 + 2];
        }
      } else if ((name === 'worient' || name === 'wfingers' || name === 'wsize') && a && b) {
        // yalnız world: worient — avatarın world əli klipin ovuc oriyentasiyasına fırladılır; wfingers — klipin world
        // əli avatarın oriyentasiyasında; wsize — avatarın world əli klipin ölçüsünə (bilək → orta barmaq kökü)
        const wld = (H) => Array.from({ length: 21 }, (_, j) => [H[k + j * 3], H[k + j * 3 + 1], H[k + j * 3 + 2]]);
        const P = wld(name === 'wfingers' ? S.hand_world : A.hand_world), Q = wld(name === 'wfingers' ? A.hand_world : S.hand_world);
        let out;
        if (name === 'wsize') {
          const size = (W) => Math.hypot(...sub(W[9], W[0]));
          const r = size(Q) / (size(P) || 1);
          out = P.map((p) => p.map((v) => v * r));
        } else {
          const cP = centroid(PALM.map((j) => P[j])), cQ = centroid(PALM.map((j) => Q[j]));
          const Rm = kabsch(PALM.map((j) => sub(P[j], cP)), PALM.map((j) => sub(Q[j], cQ)));
          const c = centroid(P);
          out = P.map((p) => rot(Rm, sub(p, c)));
        }
        for (let j = 0; j < 21; j++) for (let q = 0; q < 3; q++) o.hand_world[k + j * 3 + q] = out[j][q];
      } else if ((name === 'orient' || name === 'fingers') && a && b) {
        // orient: avatarın barmaqları, klipin oriyentasiyası (ovucun nöqtələri üst-üstə düşsün deyə fırladılır);
        // fingers: klipin barmaqları avatarın oriyentasiyasında. Şəkil nöqtələri psevdo-metrik 3D-də fırladılır
        const [from, to] = name === 'orient' ? [A, S] : [S, A];
        const img = (H) => Array.from({ length: 21 }, (_, j) => [H[k + j * 3] * ASPECT, H[k + j * 3 + 1], H[k + j * 3 + 2] * ASPECT / Z_SCALE]);
        const wld = (H) => Array.from({ length: 21 }, (_, j) => [H[k + j * 3], H[k + j * 3 + 1], H[k + j * 3 + 2]]);
        for (const [get, field] of [[img, 'hand'], [wld, 'hand_world']]) {
          const P = get(from[field]), Q = get(to[field]);
          const cP = centroid(PALM.map((j) => P[j])), cQ = centroid(PALM.map((j) => Q[j]));
          const Rm = kabsch(PALM.map((j) => sub(P[j], cP)), PALM.map((j) => sub(Q[j], cQ)));
          // şəkil: fırlanma biləyin ətrafında, bilək avatarın yerində; world: mərkəz 21 nöqtənin ortası
          const c = field === 'hand' ? P[0] : centroid(P);
          const at = field === 'hand' ? (name === 'orient' ? P[0] : Q[0]) : [0, 0, 0];
          for (let j = 0; j < 21; j++) {
            const r = rot(Rm, sub(P[j], c));
            const v = [r[0] + at[0], r[1] + at[1], r[2] + at[2]];
            if (field === 'hand') { v[0] /= ASPECT; v[2] *= Z_SCALE / ASPECT; }
            o[field][k + j * 3] = v[0]; o[field][k + j * 3 + 1] = v[1]; o[field][k + j * 3 + 2] = v[2];
          }
        }
      }
    }
  }
  return o;
}

// ---------------------------------------------------------------- həndəsə (hər niyyət olunan söz üçün)
function geom(A, S, x) {
  const g = { n: 0, detA: [0, 0], detS: [0, 0], both: [0, 0], dx: [0, 0], dy: [0, 0], dist: [0, 0], shape: [0, 0],
    zA: [0, 0], zS: [0, 0], gapA: 0, gapS: 0, gapN: 0, rot: [0, 0], fing: [0, 0] };
  for (let t = x.a; t < x.b; t++) {
    // çiyin eni kadrda (klipin pozasından; avatarınkı ilə eynidir)
    const sw = Math.hypot((S.pose[t * 27 + 3] - S.pose[t * 27 + 6]) * (4 / 3), S.pose[t * 27 + 4] - S.pose[t * 27 + 7]) || 0.316;
    g.n++;
    const palm = (H, k) => [0, 5, 9, 17].reduce((acc, j) => [acc[0] + H[k + j * 3] / 4, acc[1] + H[k + j * 3 + 1] / 4], [0, 0]);
    const pp = [];
    for (let s = 0; s < 2; s++) {
      const k = t * 126 + s * 63;
      const a = has(A.hand, t, s), b = has(S.hand, t, s);
      g.detA[s] += a; g.detS[s] += b;
      if (!(a && b)) { pp.push(null); continue; }
      g.both[s]++;
      const pa = palm(A.hand, k), ps = palm(S.hand, k);
      g.dx[s] += (pa[0] - ps[0]) * (4 / 3) / sw; g.dy[s] += (pa[1] - ps[1]) / sw;
      g.dist[s] += Math.hypot((pa[0] - ps[0]) * (4 / 3), pa[1] - ps[1]) / sw;
      // forma: world nöqtələri (mərkəzlənmiş, eyni ölçü), orta nöqtə fərqi sm-lə — oriyentasiya daxil
      let d = 0;
      for (let j = 0; j < 21; j++) d += Math.hypot(...[0, 1, 2].map((q) => A.hand_world[k + j * 3 + q] - S.hand_world[k + j * 3 + q]));
      g.shape[s] += d / 21 * 100;
      // oriyentasiya: ovucun nöqtələrini üst-üstə salan fırlanmanın bucağı (world); barmaqlar: fırlanmadan sonra qalan fərq
      const wa = Array.from({ length: 21 }, (_, j) => [0, 1, 2].map((q) => A.hand_world[k + j * 3 + q]));
      const ws = Array.from({ length: 21 }, (_, j) => [0, 1, 2].map((q) => S.hand_world[k + j * 3 + q]));
      const ca = centroid(PALM.map((j) => wa[j])), cs = centroid(PALM.map((j) => ws[j]));
      const Rm = kabsch(PALM.map((j) => sub(wa[j], ca)), PALM.map((j) => sub(ws[j], cs)));
      g.rot[s] += Math.acos(Math.max(-1, Math.min(1, (Rm[0][0] + Rm[1][1] + Rm[2][2] - 1) / 2))) * 180 / Math.PI;
      let fd = 0;
      for (let j = 0; j < 21; j++) {
        const r = rot(Rm, sub(wa[j], ca)), q = sub(ws[j], cs);
        fd += Math.hypot(r[0] - q[0], r[1] - q[1], r[2] - q[2]);
      }
      g.fing[s] += fd / 21 * 100;
      // şəkil z-in yayılması (barmaq ucları biləyə nisbətən)
      g.zA[s] += Math.abs(A.hand[k + 8 * 3 + 2]); g.zS[s] += Math.abs(S.hand[k + 8 * 3 + 2]);
      pp.push([pa, ps]);
    }
    if (pp[0] && pp[1]) {
      g.gapN++;
      g.gapA += Math.hypot((pp[0][0][0] - pp[1][0][0]) * (4 / 3), pp[0][0][1] - pp[1][0][1]) / sw;
      g.gapS += Math.hypot((pp[0][1][0] - pp[1][1][0]) * (4 / 3), pp[0][1][1] - pp[1][1][1]) / sw;
    }
  }
  return g;
}

// ---------------------------------------------------------------- oynatma
const idx = JSON.parse(readFileSync(new URL('public/tercume/words/index.json', root), 'utf8'));
const handsOf = new Map(idx.words.map((w) => [w.id, w.hands]));
const list = TEXT ? [{ id: 'text', set: 'text', text: TEXT }] : sentences(SET).slice(0, LIMIT || undefined);
const server = await createServer({ root: new URL('..', import.meta.url).pathname.replace(/^\/(\w:)/, '$1'),
  server: { port: 0, hmr: false, watch: null }, logLevel: 'error' });
await server.listen();
const url = server.resolvedUrls.local[0];
const browser = await puppeteer.launch({ headless: true, protocolTimeout: 60 * 60 * 1000,
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--window-size=520,560'],
  defaultViewport: { width: 520, height: 560 } });
const tal = {};     // qarışıq → qrup → [n, hit]
const add = (m, k, v) => { tal[m] ??= {}; tal[m][k] ??= [0, 0]; tal[m][k][0]++; tal[m][k][1] += v; };
const words = {};   // söz id → həndəsə cəmi + qarışıqların tapması
const rows = [];
const t0 = Date.now();
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.error('səhifə:', e.message));
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

  for (const [n, s] of list.entries()) {
    const r = await page.evaluate((text) => window.__bt.run(text, { mode: 'avatar', dual: true }), s.text);
    const A = { T: r.T, hand: arr(r.hand), hand_world: arr(r.hand_world), pose: arr(r.pose) };
    const S = { T: r.T, hand: arr(r.src.hand), hand_world: arr(r.src.hand_world), pose: arr(r.src.pose) };
    for (const x of r.intended) x.c = classOf(x);
    const ws = r.intended.filter((x) => x.kind === 'word' && x.c != null);
    const row = { id: s.id, text: s.text, words: ws.map((x) => ({ id: x.id, c: rec.labels[x.c], a: x.a, b: x.b })), hits: {} };
    for (const m of MIXES) {
      const dets = await recognize(mix(m, A, S, r.T), r.aspect);
      row.hits[m] = ws.map((x) => (hit(x, dets) ? 1 : 0));
      ws.forEach((x, i) => {
        const h = row.hits[m][i];
        add(m, 'all', h);
        add(m, `hands ${handsOf.get(x.id) ?? '?'}`, h);
        add(m, i ? 'later' : 'first', h);
      });
      if (m === 'avatar' || m === 'source') row[`det_${m}`] = dets.filter((d) => !isLetter(rec.labels[d.c])).map((d) => [rec.labels[d.c], d.a, d.b]);
    }
    ws.forEach((x, i) => {
      const g = geom(A, S, x);
      const w = (words[x.id] ??= { id: x.id, c: rec.labels[x.c], hands: handsOf.get(x.id), n: 0, hits: {}, g: null });
      w.n++;
      for (const m of MIXES) w.hits[m] = (w.hits[m] ?? 0) + row.hits[m][i];
      if (!w.g) w.g = g;
      else for (const [k, v] of Object.entries(g)) w.g[k] = Array.isArray(v) ? w.g[k].map((u, q) => u + v[q]) : w.g[k] + v;
    });
    if (FRAMES) row.frames = { avatar: { hand: r.hand, hand_world: r.hand_world, pose: r.pose }, source: r.src };
    rows.push(row);
    if ((n + 1) % 10 === 0 || n === list.length - 1) {
      console.log(`${n + 1}/${list.length} ${((Date.now() - t0) / 1000).toFixed(0)} s · ` + MIXES.map((m) => `${m} ${(tal[m].all[1] / tal[m].all[0]).toFixed(3)}`).join(' '));
    }
  }
} finally {
  await browser.close();
  await server.close();
}

// ---------------------------------------------------------------- hesabat
const r3 = (v) => Math.round(v * 1000) / 1000;
console.log('\nrecall of the intended signs (matched by class and overlapping frames):');
const groups = [...new Set(Object.values(tal).flatMap((g) => Object.keys(g)))].sort();
console.log('mix'.padEnd(8) + groups.map((g) => g.padStart(16)).join(''));
for (const m of MIXES) console.log(m.padEnd(8) + groups.map((g) => (tal[m][g] ? `${r3(tal[m][g][1] / tal[m][g][0]).toFixed(3)} (${tal[m][g][0]})` : '-').padStart(16)).join(''));
// Həndəsə qrup üzrə (söz kadrları): aşkarlanma, ovucun yer fərqi, iki əl arası
const G = {};
for (const w of Object.values(words)) {
  const k = `hands ${w.hands}`;
  const o = (G[k] ??= { n: 0, detA: [0, 0], detS: [0, 0], both: [0, 0], dx: [0, 0], dy: [0, 0], dist: [0, 0], shape: [0, 0], zA: [0, 0], zS: [0, 0], gapA: 0, gapS: 0, gapN: 0, rot: [0, 0], fing: [0, 0] });
  for (const [q, v] of Object.entries(w.g)) o[q] = Array.isArray(v) ? o[q].map((u, i) => u + v[i]) : o[q] + v;
}
console.log('\ngeometry over the intended signs\' frames (slot 0 = right hand, 1 = left; distances in shoulder widths, shape in cm):');
for (const [k, o] of Object.entries(G)) {
  const per = (v, s, d) => r3(v[s] / Math.max(1, d[s]));
  console.log(`  ${k}: frames ${o.n} | detected avatar R ${r3(o.detA[0] / o.n)} L ${r3(o.detA[1] / o.n)}, clip R ${r3(o.detS[0] / o.n)} L ${r3(o.detS[1] / o.n)}`
    + ` | palm avatar−clip dx R ${per(o.dx, 0, o.both)} L ${per(o.dx, 1, o.both)}, dy R ${per(o.dy, 0, o.both)} L ${per(o.dy, 1, o.both)}, |d| R ${per(o.dist, 0, o.both)} L ${per(o.dist, 1, o.both)}`
    + ` | shape R ${per(o.shape, 0, o.both)} L ${per(o.shape, 1, o.both)} cm, turn R ${per(o.rot, 0, o.both)} L ${per(o.rot, 1, o.both)}°, fingers after turn R ${per(o.fing, 0, o.both)} L ${per(o.fing, 1, o.both)} cm | tip z avatar ${per(o.zA, 0, o.both)} clip ${per(o.zS, 0, o.both)}`
    + ` | hands apart avatar ${r3(o.gapA / Math.max(1, o.gapN))} clip ${r3(o.gapS / Math.max(1, o.gapN))}`);
}
if (OUT) writeFileSync(OUT, JSON.stringify({ opts: T2S_OPTS, avatar: AV_OPTS, mixes: MIXES, tally: tal, words, rows }, null, 1));
console.log(`\n${((Date.now() - t0) / 1000).toFixed(0)} s`);
