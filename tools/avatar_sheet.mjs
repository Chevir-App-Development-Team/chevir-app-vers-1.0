// Avatarın söz kadrlarının vərəqi (E28): hər sətir bir söz, sütunlar sözün kadrları (--fracs: uzunluğun payı).
// Başsız Chrome-da, tools/measure_hands.mjs kimi; JPEG faylı gözlə yoxlamaq üçündür (qolun, dirsəyin görünüşü).
//   node tools/avatar_sheet.mjs --words men,sabah,sen --out sheet.jpg [--fracs 0.15,0.4,0.65,0.9]
// Cümlə (E34): hər sətir bir oynatma variantı (TextToSign.opts), sütunlar hər --every kadrdan (30 fps) bir şəkil —
// işarələr arası keçidlərə baxmaq üçün (tools/backtranslate.js-in virtual saatı ilə, tətbiqin oynatması):
//   node tools/avatar_sheet.mjs --text "Mən Bakıda yaşayıram" --variants '[{},{"timing":"natural"}]' --every 3 --out s.jpg
// --avatar yeni.vrm: başqa VRM faylı, quraşdırmadan (avatar.vrm sorğusu ona gedir)
// --set '{"wordDepth":1}': söz vərəqində VrmAvatar sahələri (E37-nin müqayisəsi üçün)
import { readFileSync, writeFileSync } from 'node:fs';
import { createServer } from 'vite';
import puppeteer from 'puppeteer';
const args = process.argv.slice(2);
const opt = (n, d = null) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : d; };
const ids = opt('--words', '').split(',').filter(Boolean);
const fracs = opt('--fracs', '0.15,0.4,0.65,0.9').split(',').map(Number);
const out = opt('--out', 'D:/chevir/eval/sheet.jpg');
const avatarFile = opt('--avatar');
const server = await createServer({ root: new URL('..', import.meta.url).pathname.replace(/^\/(\w:)/, '$1'), server: { port: 0, hmr: false, watch: null }, logLevel: 'error' });
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
    if (avatarFile && path.endsWith('/tercume/assets/avatar.vrm')) {
      req.respond({ status: 200, contentType: 'application/octet-stream', body: readFileSync(avatarFile) }); return;
    }
    if (!path.endsWith('.bin')) { req.continue(); return; }
    req.respond({ status: 200, contentType: 'application/octet-stream', body: readFileSync(new URL(`../public${path}`, import.meta.url)) });
  });
  if (opt('--text')) {
    await page.goto(`${url}rey/`, { waitUntil: 'networkidle0', timeout: 120000 });
    const data = await page.evaluate(async (text, variants, every, cols) => {
      const { VrmAvatar } = await import('/src/tercume/vrm.js');
      const { TextToSign, loadPoses } = await import('/src/tercume/t2s.js');
      const { WordLexicon } = await import('/src/tercume/words.js');
      const { ASSET_BASE, WORDS_BASE } = await import('/src/tercume/paths.js');
      const BT = await import('/tools/backtranslate.js');
      document.body.innerHTML = '<div id="av" style="width:360px;height:405px"></div>';
      const [poses, lexicon] = await Promise.all([loadPoses(`${ASSET_BASE}poses.json`), WordLexicon.load(`${WORDS_BASE}index.json`)]);
      const avatar = new VrmAvatar(document.querySelector('#av'));
      await avatar.load(`${ASSET_BASE}avatar.vrm`);
      avatar.setHand('right');
      const t2s = new TextToSign({ poses, lexicon, targets: [avatar] });
      const bt = await BT.setup({ avatar, t2s, lexicon });
      const runs = [];
      for (const v of variants) runs.push(await bt.run(text, { opts: v, render: every, pre: 0.2, tail: 0.3 }));
      // Şəkil avatarın ətrafına kəsilir (kadrın ortası), uzun oynatma bir neçə sətrə bölünür (sətirdə `cols` şəkil)
      const crop = { x: 0.2, y: 0.08, w: 0.6, h: 0.84 };
      const tw = 120, th = Math.round(tw * (crop.h * 405) / (crop.w * 360));
      const lines = runs.map((r) => Math.ceil(r.shots.length / cols));
      const outc = document.createElement('canvas');
      outc.width = 150 + tw * cols; outc.height = th * lines.reduce((a, b) => a + b, 0) + 24 + 6 * runs.length;
      const g = outc.getContext('2d');
      g.fillStyle = '#000'; g.fillRect(0, 0, outc.width, outc.height);
      g.font = '13px sans-serif'; g.fillStyle = '#fff';
      g.fillText(text, 6, 16);
      const load = (u) => new Promise((r) => { const im = new Image(); im.onload = () => r(im); im.src = u; });
      let y = 24;
      for (let i = 0; i < runs.length; i++) {
        const r = runs[i];
        const label = JSON.stringify(variants[i]).replace(/[{}"]/g, '') || 'fixed';
        label.match(/.{1,18}/g).forEach((l, k) => g.fillText(l, 4, y + 18 + 16 * k));
        g.fillText(`${(r.T / 30).toFixed(1)} s`, 4, y + th - 10);
        for (let j = 0; j < r.shots.length; j++) {
          const im = await load(r.shots[j].url);
          const x0 = 150 + (j % cols) * tw, y0 = y + Math.floor(j / cols) * th;
          g.drawImage(im, crop.x * im.width, crop.y * im.height, crop.w * im.width, crop.h * im.height, x0, y0, tw, th);
          g.fillText(`${(r.shots[j].t / 30).toFixed(2)}`, x0 + 4, y0 + th - 6);
        }
        y += lines[i] * th + 6;
        g.fillStyle = '#334'; g.fillRect(0, y - 4, outc.width, 2); g.fillStyle = '#fff';
      }
      return outc.toDataURL('image/jpeg', 0.88);
    }, opt('--text'), JSON.parse(opt('--variants', '[{}]')), Number(opt('--every', '3')), Number(opt('--cols', '16')));
    writeFileSync(out, Buffer.from(data.split(',')[1], 'base64'));
    console.log('wrote', out);
  } else {
  await page.goto(`${url}tercume/`, { waitUntil: 'networkidle0', timeout: 120000 });
  await page.waitForFunction(() => window.__chevir?.lexicon, { timeout: 120000 });
  await page.click('.tab[data-tab="t2s"]');
  await page.waitForFunction(() => window.__chevir?.avatar?.vrm, { timeout: 180000 });
  const data = await page.evaluate(async (ids, fracs, set) => {
    const st = window.__chevir; const av = st.avatar; const lex = st.lexicon;
    av.paused = true;
    Object.assign(av, set);
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
  }, ids, fracs, JSON.parse(opt('--set', '{}')));
  writeFileSync(out, Buffer.from(data.split(',')[1], 'base64'));
  console.log('wrote', out);
  }
} finally { await browser.close(); await server.close(); }
