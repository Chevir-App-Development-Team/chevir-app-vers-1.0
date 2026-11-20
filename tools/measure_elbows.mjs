// Söz işarələrində dirsəyin yeri (E28): lüğətdən --n söz (və ya --words), hər kadrda hər qolun dirsəyi çiyinə nisbətən
// (ey, ex — çölə, ez — önə; metr), iştirakçının avatara köçürülmüş dirsəyi (ty, tx, tz) və ovucun yeri (w*). JSON-a yazır;
// "qalxmış dirsək" ölçüsü: ey > -0.15 və ex > 0.12. Başsız Chrome-da, tools/measure_hands.mjs kimi.
//   node tools/measure_elbows.mjs --n 300 --json elbows.json [--words men,sabah]
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
  await page.evaluate((o) => { window.__only = o; }, (opt('--words', '') || '').split(',').filter(Boolean));
  // --set '{"wordDepth":1}': VrmAvatar sahələri (E37-nin müqayisəsi üçün)
  await page.evaluate((s) => { window.__set = s; }, JSON.parse(opt('--set', '{}')));
  const rep = await page.evaluate(async (n) => {
    const st = window.__chevir; const av = st.avatar; const lex = st.lexicon;
    Object.assign(av, window.__set ?? {});
    av.paused = true;
    const only = (window.__only || []); const ents = only.length ? lex.entries.filter((e) => only.includes(e.id)) : lex.entries.filter((e, k) => k % Math.max(1, Math.floor(lex.entries.length / n)) === 0).slice(0, n);
    const rows = [];
    for (const e of ents) {
      let word; try { word = await lex.get(e.id); } catch { continue; }
      av.prepareWord?.(word);
      for (let f = 0; f < word.frames.length; f++) {
        av.setWordPose(word, f); av.snap(); av.update(0.001); av.snap();
        for (const side of ['right', 'left']) {
          const s = av.last?.[side]; if (!s) continue;
          const rig = av.rigs[side], S = rig.S, o = rig.out;
          const E = s.arm.info.E, T = s.elbow, W = s.anchor;
          const rb = av.bone(side + 'LowerArm').getWorldPosition(new av.vrm.scene.position.constructor());
          const sb = av.bone(side + 'UpperArm').getWorldPosition(new av.vrm.scene.position.constructor());
          rows.push({ id: e.id, src: e.id.startsWith('jd-') ? 'jd' : 'az', side, f,
            ey: E.y - S.y, ex: (E.x - S.x) * o, ez: E.z - S.z,
            ty: T ? T.y - S.y : null, tx: T ? (T.x - S.x) * o : null, tz: T ? T.z - S.z : null,
            ry: rb.y - sb.y, rx: (rb.x - sb.x) * o, rz: rb.z - sb.z, sy: sb.y - S.y, wy: W.y - S.y, wx: (W.x - S.x) * o, wz: W.z - S.z, L1: rig.L1, cost: s.arm.info.cost });
        }
      }
    }
    return rows;
  }, Number(opt('--n', '200')));
  writeFileSync(opt('--json', 'D:/chevir/eval/elbow_rows.json'), JSON.stringify(rep));
  console.log('rows', rep.length);
} finally { await browser.close(); await server.close(); }
