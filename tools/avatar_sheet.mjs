// Avatarın söz kadrlarının vərəqi (E28): hər sətir bir söz, sütunlar sözün kadrları (--fracs: uzunluğun payı).
// Başsız Chrome-da, tools/measure_hands.mjs kimi; JPEG faylı gözlə yoxlamaq üçündür (qolun, dirsəyin görünüşü).
//   node tools/avatar_sheet.mjs --words men,sabah,sen --out sheet.jpg [--fracs 0.15,0.4,0.65,0.9]
import { readFileSync, writeFileSync } from 'node:fs';
import { createServer } from 'vite';
import puppeteer from 'puppeteer';
const args = process.argv.slice(2);
const opt = (n, d = null) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : d; };
const ids = opt('--words', '').split(',').filter(Boolean);
const fracs = opt('--fracs', '0.15,0.4,0.65,0.9').split(',').map(Number);
const out = opt('--out', 'D:/chevir/eval/sheet.jpg');
const server = await createServer({ root: new URL('..', import.meta.url).pathname.replace(/^\/(\w:)/, '$1'), server: { port: 0 }, logLevel: 'error' });
await server.listen();
const url = server.resolvedUrls.local[0];
const browser = await puppeteer.launch({ headless: true, protocolTimeout: 600000,
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--window-size=1280,900'], defaultViewport: { width: 1280, height: 900 } });
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.error('page:', e.message));
  await page.setRequestInterception(true);
  page.on('request', (req) => {
    const path = new URL(req.url()).pathname;
    if (!path.endsWith('.bin')) { req.continue(); return; }
    req.respond({ status: 200, contentType: 'application/octet-stream', body: readFileSync(new URL(`../public${path}`, import.meta.url)) });
  });
  await page.goto(`${url}tercume/`, { waitUntil: 'networkidle0', timeout: 120000 });
  await page.waitForFunction(() => window.__chevir?.lexicon, { timeout: 120000 });
  await page.click('.tab[data-tab="t2s"]');
  await page.waitForFunction(() => window.__chevir?.avatar?.vrm, { timeout: 180000 });
  const data = await page.evaluate(async (ids, fracs) => {
    const st = window.__chevir; const av = st.avatar; const lex = st.lexicon;
    av.paused = true;
    const src = av.renderer.domElement; const W = src.width, H = src.height;
    const sx = 0, sw = W, sy = 0, sh = H; const tw = 230, th = Math.round(tw * sh / sw);
    const outc = document.createElement('canvas'); outc.width = tw * fracs.length + 80; outc.height = th * ids.length;
    const g = outc.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, outc.width, outc.height);
    g.font = '14px sans-serif'; g.fillStyle = '#fff';
    for (let i = 0; i < ids.length; i++) {
      const word = await lex.get(ids[i]);
      av.prepareWord?.(word);
      g.fillText(ids[i], 4, i * th + 20); g.fillText(word.frames.length + 'f', 4, i * th + 40);
      for (let j = 0; j < fracs.length; j++) {
        const f = Math.min(word.frames.length - 1, Math.round(fracs[j] * (word.frames.length - 1)));
        av.setWordPose(word, f); av.snap(); av.update(0.001); av.snap();
        av.renderer.render(av.scene, av.camera);
        g.drawImage(src, sx, sy, sw, sh, 80 + j * tw, i * th, tw, th);
        g.fillText('f' + f, 80 + j * tw + 4, i * th + th - 6);
      }
    }
    return outc.toDataURL('image/jpeg', 0.88);
  }, ids, fracs);
  writeFileSync(out, Buffer.from(data.split(',')[1], 'base64'));
  console.log('wrote', out);
} finally { await browser.close(); await server.close(); }
