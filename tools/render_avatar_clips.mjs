// Avatarın videoları, yerli rəy paketi üçün (E36, tools/build_native_review.py): hər iş ya mətndir (t2s.play,
// tətbiqdəki kimi doğma qaydalarla), ya da işarələrin siyahısı (planSigns: iştirakçının göstərdiyi işarələr, /rey/
// səhifəsinin işarə → mətn bəndləri kimi). Başsız Chrome-da, virtual saatla: performance.now və setTimeout saxta
// zamanla gedir, ona görə video tətbiqdəki oynatmanın eyni vaxtlarındadır, kompüterin sürətindən asılı deyil.
// Kadrlar ffmpeg-ə gedir (H.264 mp4, istənilən brauzer file:// ilə oynadır). Videolar repoya girmir.
//
//   node tools/render_avatar_clips.mjs --jobs jobs.json --out <qovluq> [--fps 25] [--size 480x540] [--only id1,id2]
//
// jobs.json: [{id, text} | {id, signs: [{id, gloss} | {spell, gloss}]}]. Çıxış: <out>/<id>.mp4 və
// <out>/plans.json ({id: {glosses, seconds}}: avatarın göstərdiyi işarələr, hərf-hərf olanlar defislə).
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { createServer } from 'vite';
import puppeteer from 'puppeteer';

const args = process.argv.slice(2);
const opt = (n, d = null) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : d; };
const jobsPath = opt('--jobs');
const outDir = opt('--out');
if (!jobsPath || !outDir) { console.log('istifadə: node tools/render_avatar_clips.mjs --jobs jobs.json --out <qovluq>'); process.exit(1); }
const FPS = Number(opt('--fps', 25));
const [W, H] = opt('--size', '480x540').split('x').map(Number);
const only = opt('--only') ? new Set(opt('--only').split(',')) : null;
const jobs = JSON.parse(readFileSync(jobsPath, 'utf8')).filter((j) => !only || only.has(j.id));
mkdirSync(outDir, { recursive: true });
const plansPath = `${outDir}/plans.json`;
const plans = existsSync(plansPath) ? JSON.parse(readFileSync(plansPath, 'utf8')) : {};

function encode(frames, file) {
  return new Promise((resolve, reject) => {
    const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-i', '-',
      '-c:v', 'libx264', '-preset', 'medium', '-crf', '27', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', file]);
    ff.on('error', reject);
    ff.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`ffmpeg ${code}`))));
    for (const f of frames) ff.stdin.write(Buffer.from(f, 'base64'));
    ff.stdin.end();
  });
}

const root = new URL('..', import.meta.url).pathname.replace(/^\/(\w:)/, '$1');
const server = await createServer({ root, server: { port: 0 }, logLevel: 'error' });
await server.listen();
const url = server.resolvedUrls.local[0];
const browser = await puppeteer.launch({ headless: true, protocolTimeout: 1800000,
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', `--window-size=${W + 40},${H + 40}`],
  defaultViewport: { width: W + 40, height: H + 40 } });
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.error('page:', e.message));
  // IDM .bin endirmələrini tutur: başsız Chrome-a .bin faylları birbaşa verilir (tools/avatar_sheet.mjs kimi)
  await page.setRequestInterception(true);
  page.on('request', (req) => {
    const path = new URL(req.url()).pathname;
    if (!path.endsWith('.bin')) { req.continue(); return; }
    req.respond({ status: 200, contentType: 'application/octet-stream', body: readFileSync(new URL(`../public${path}`, import.meta.url)) });
  });
  await page.goto(`${url}rey/`, { waitUntil: 'networkidle0', timeout: 120000 });
  await page.evaluate(async (W, H) => {
    const { VrmAvatar } = await import('/src/tercume/vrm.js');
    const { TextToSign, loadPoses } = await import('/src/tercume/t2s.js');
    const { WordLexicon } = await import('/src/tercume/words.js');
    const { ASSET_BASE, WORDS_BASE } = await import('/src/tercume/paths.js');
    document.body.innerHTML = `<div id="av" style="width:${W}px;height:${H}px"></div>`;
    document.body.style.margin = '0';
    const [poses, lexicon] = await Promise.all([loadPoses(`${ASSET_BASE}poses.json`), WordLexicon.load(`${WORDS_BASE}index.json`)]);
    const av = new VrmAvatar(document.querySelector('#av'));
    await av.load(`${ASSET_BASE}avatar.vrm`);
    av.setHand('right');
    av.paused = true;                       // zaman addımını aşağıdakı dövr verir
    const t2s = new TextToSign({ poses, lexicon, targets: [av] });

    // Virtual saat: t2s-in gözləmələri (performance.now + setTimeout) yalnız kadr irəlilədikcə bitir
    const vt = { now: performance.now() };
    const q = [];
    let seq = 0;
    performance.now = () => vt.now;
    window.setTimeout = (fn, ms = 0, ...a) => { const id = ++seq; q.push({ t: vt.now + Math.max(0, Number(ms) || 0), id, fn, a }); return id; };
    window.clearTimeout = (id) => { const i = q.findIndex((x) => x.id === id); if (i >= 0) q.splice(i, 1); };
    const flush = () => new Promise((r) => { const ch = new MessageChannel(); ch.port1.onmessage = () => r(); ch.port2.postMessage(0); });
    async function advance(to) {
      for (;;) {
        let k = -1;
        for (let i = 0; i < q.length; i++) {
          if (q[i].t <= to && (k < 0 || q[i].t < q[k].t || (q[i].t === q[k].t && q[i].id < q[k].id))) k = i;
        }
        if (k < 0) break;
        const x = q.splice(k, 1)[0];
        vt.now = Math.max(vt.now, x.t);
        if (typeof x.fn === 'function') x.fn(...x.a);
        await flush();
        await flush();
      }
      vt.now = to;
      await flush();
    }
    const up = (s) => s.replace(/i/g, 'İ').replace(/ı/g, 'I').toLocaleUpperCase('az');
    function glossLine(steps) {
      const out = [];
      let spell = null;
      for (const s of steps) {
        if (s.kind === 'word') { spell = null; out.push(up(s.entry.text)); } else if (s.kind === 'letter' || s.kind === 'unknown') {
          if (spell && spell.seg === s.seg?.from) spell.chars.push(up(s.char));
          else { spell = { seg: s.seg?.from, chars: [up(s.char)] }; out.push(spell); }
        } else spell = null;
      }
      return out.map((x) => (typeof x === 'string' ? x : x.chars.join('-'))).join(' ');
    }
    window.__render = async (job, fps, maxSeconds) => {
      const steps = job.signs ? t2s.planSigns(job.signs) : t2s.plan(job.text);
      await Promise.all(steps.filter((s) => s.kind === 'word').map((s) => lexicon.get(s.entry.id).catch(() => null)));
      // istirahət vəziyyətindən başla
      t2s.stop();
      av.clearHandPose();
      for (let i = 0; i < 2 * fps; i++) { await advance(vt.now + 1000 / fps); av.update(1 / fps); }
      av.snap();
      const frames = [];
      t2s.opts.speed = 1;
      t2s.playSteps(steps);
      await flush();
      let tail = Math.round(0.6 * fps);
      while (frames.length < maxSeconds * fps) {
        await advance(vt.now + 1000 / fps);
        av.update(1 / fps);
        av.renderer.render(av.scene, av.camera);
        frames.push(av.renderer.domElement.toDataURL('image/jpeg', 0.88).slice('data:image/jpeg;base64,'.length));
        if (!t2s.playing && --tail <= 0) break;
      }
      t2s.stop();
      return { frames, glosses: glossLine(steps) };
    };
  }, W, H);

  let k = 0;
  for (const job of jobs) {
    const t0 = Date.now();
    const { frames, glosses } = await page.evaluate((job, fps) => window.__render(job, fps, 40), job, FPS);
    await encode(frames, `${outDir}/${job.id}.mp4`);
    plans[job.id] = { glosses, seconds: +(frames.length / FPS).toFixed(2), text: job.text ?? null };
    writeFileSync(plansPath, JSON.stringify(plans, null, 1));
    console.log(`${++k}/${jobs.length} ${job.id}: ${frames.length} kadr, ${((Date.now() - t0) / 1000).toFixed(1)} s · ${glosses}`);
  }
} finally {
  await browser.close();
  await server.close();
}
