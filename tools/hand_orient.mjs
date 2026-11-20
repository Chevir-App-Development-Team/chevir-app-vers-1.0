// Söz kadrlarında əlin oriyentasiyası (E37): hədəf (şəkil landmark-larından), iştirakçının world landmark-ları və
// avatarın əldə etdiyi (sümüklərdən) — barmaq istiqaməti (d: bilək → orta barmağın kökü) və ovuc normalı (n) arasında
// bucaqlar, biləyin həddi (hard). Başsız Chrome-da (tools/measure_hands.mjs kimi).
//   node tools/hand_orient.mjs --words isteyirem,sen [--avatar '{"wordDepth":1,"wordKnuckles":false}'] [--out x.json]
import { readFileSync, writeFileSync } from 'node:fs';
import { createServer } from 'vite';
import puppeteer from 'puppeteer';
const args = process.argv.slice(2);
const opt = (n, d = null) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : d; };
const ids = opt('--words', '').split(',').filter(Boolean);
const AV = JSON.parse(opt('--avatar', '{}'));
const server = await createServer({ root: new URL('..', import.meta.url).pathname.replace(/^\/(\w:)/, '$1'), server: { port: 0, hmr: false, watch: null }, logLevel: 'error' });
await server.listen();
const url = server.resolvedUrls.local[0];
const browser = await puppeteer.launch({ headless: true, protocolTimeout: 3600000,
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'], defaultViewport: { width: 520, height: 560 } });
let res;
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.error('page:', e.message));
  page.on('console', (m) => { if (m.type() === 'log') console.log(m.text()); });
  await page.setRequestInterception(true);
  page.on('request', (req) => {
    const path = new URL(req.url()).pathname;
    if (!path.endsWith('.bin')) { req.continue(); return; }
    req.respond({ status: 200, contentType: 'application/octet-stream', body: readFileSync(new URL(`../public${path}`, import.meta.url)) });
  });
  await page.goto(`${url}rey/`, { waitUntil: 'networkidle0', timeout: 120000 });
  res = await page.evaluate(async (ids, av) => {
    const { VrmAvatar } = await import('/src/tercume/vrm.js');
    const { WordLexicon } = await import('/src/tercume/words.js');
    const { ASSET_BASE, WORDS_BASE } = await import('/src/tercume/paths.js');
    const PR = await import('/tools/hand_orient.js');
    document.body.innerHTML = '<div id="av" style="width:480px;height:540px"></div>';
    const lexicon = await WordLexicon.load(`${WORDS_BASE}index.json`);
    const avatar = new VrmAvatar(document.querySelector('#av'));
    await avatar.load(`${ASSET_BASE}avatar.vrm`);
    Object.assign(avatar, av);
    const out = await PR.probe(avatar, lexicon, ids);
    return out;
  }, ids, AV);
} finally {
  await browser.close();
  await server.close();
}
const out = opt('--out');
if (out) writeFileSync(out, JSON.stringify(res, null, 1));
for (const w of res) {
  if (w.missing) { console.log(w.id, 'missing'); continue; }
  for (const side of ['right', 'left']) {
    const fs = w.frames.map((r) => r[side]).filter((x) => x.up);
    if (!fs.length) { console.log(`${w.id} ${side}: down (${w.frames.filter((r) => r[side].data).length} data frames)`); continue; }
    const m = (k, i) => (fs.reduce((a, x) => a + x[k][i], 0) / fs.length).toFixed(0);
    console.log(`${w.id.padEnd(12)} ${side.padEnd(5)} up ${fs.length}/${w.frames.length} hard>0.03 ${fs.filter((x) => x.hard > 0.03).length}`
      + ` | ach-view d ${m('achView', 0)}° n ${m('achView', 1)}° | ach-world d ${m('achWorld', 0)}° n ${m('achWorld', 1)}° | view-world d ${m('viewWorld', 0)}° n ${m('viewWorld', 1)}°`);
  }
}
