/**
 * Söz işarələrində iki əlin bir-birinə girməsini bütün lüğət üzrə ölçür (bax: measure_hands.js).
 * Vite-i özü qaldırır, səhifəni başsız Chrome-da (puppeteer) açır, avatar yüklənəndən sonra
 * ölçməni səhifənin içində işlədir.
 *
 * İstifadə:
 *   node tools/measure_hands.mjs                      # bütün sözlər, xülasə konsola
 *   node tools/measure_hands.mjs --out hands.json     # tam hesabat faylda
 *   node tools/measure_hands.mjs --words ad-gunu,agirdir --frames
 *   node tools/measure_hands.mjs --no-guard           # oynatma qoruyucusu olmadan (yalnız həll)
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { createServer } from 'vite';
import puppeteer from 'puppeteer';

const args = process.argv.slice(2);
const opt = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : null;
};
const ids = opt('--words')?.split(',') ?? null;
const out = opt('--out');
const top = Number(opt('--top') ?? 25);

const server = await createServer({ root: new URL('..', import.meta.url).pathname.replace(/^\/(\w:)/, '$1'),
                                    server: { port: 0 }, logLevel: 'error' });
await server.listen();
const url = server.resolvedUrls.local[0];
const browser = await puppeteer.launch({
  headless: true,
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--window-size=1280,900'],
  defaultViewport: { width: 1280, height: 900 },
});
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.error('səhifə:', e.message));
  // Bəzi maşınlarda başsız Chrome .bin faylına boş cavab (204) alır, curl isə tam faylı —
  // yerli süzgəc brauzerin binar yükləməsini kəsir. Bu fayllar birbaşa diskdən verilir.
  await page.setRequestInterception(true);
  page.on('request', (req) => {
    const path = new URL(req.url()).pathname;
    if (!path.endsWith('.bin')) { req.continue(); return; }
    req.respond({ status: 200, contentType: 'application/octet-stream',
                  body: readFileSync(new URL(`../public${path}`, import.meta.url)) });
  });
  await page.goto(`${url}tercume/`, { waitUntil: 'networkidle0', timeout: 120000 });
  await page.waitForFunction(() => window.__chevir?.lexicon, { timeout: 120000 });
  await page.click('.tab[data-tab="t2s"]');
  await page.waitForFunction(() => window.__chevir?.avatar?.vrm, { timeout: 180000 });
  const rep = await page.evaluate(async (ids, keep, guard) => {
    const m = await import('/tools/measure_hands.js');
    return m.measureWords(window.__chevir, { ids, keepFrames: keep, guard });
  }, ids, args.includes('--frames') ? (ids ?? []) : [], args.includes('--no-guard') ? false : null);

  if (out) writeFileSync(out, JSON.stringify(rep, null, 1));
  const cm = (v) => (v == null ? '   -  ' : (v * 100).toFixed(1).padStart(6));
  console.log('Ümumi:', rep.totals);
  console.log(`\nƏn pis ${top} söz (gap — avatar, signer — data, play — oynatmada; sm, mənfi = girmə):`);
  console.log('söz                    kadr   gap  signer   skel   play  hissələr');
  for (const r of [...rep.rows].filter((r) => r.gap != null).sort((a, b) => a.gap - b.gap).slice(0, top)) {
    console.log(`${r.text.padEnd(22)} ${String(r.gapFrame).padStart(4)} ${cm(r.gap)} ${cm(r.signer)} ${cm(r.skel)} ${cm(r.playGap)}  ${r.pair}`);
  }
  console.log(`\n${rep.rows.length} söz, ${(rep.ms / 1000).toFixed(0)} s`);
} finally {
  await browser.close();
  await server.close();
}
