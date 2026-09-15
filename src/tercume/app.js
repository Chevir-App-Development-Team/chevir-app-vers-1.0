/**
 * Chevir tərcümə sistemi — bağlayıcı qat.
 *
 * Hər şey brauzerdə işləyir: server yoxdur, video cihazdan çıxmır.
 * Bu, əlçatanlıq məhsulu üçün həm məxfilik, həm də xərc baxımından vacibdir
 * (statik fayl kimi hər yerdə host oluna bilər).
 */
import { ASSET_BASE } from './paths.js';
import { FingerspellModel } from './model.js';
import { LexiconDecoder } from './decoder.js';
import { CyberHand } from './skeleton.js';
import { SignToText } from './s2t.js';
import { TextToSign, loadPoses } from './t2s.js';
import { HAND_CONNECTIONS } from './hands.js';

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

// Kamera halqasının altındakı vəziyyət yazısı (bax s2t.js — hərf yazma şərtləri)
const GATE_LABEL = {
  idle: '—',
  'wait-hand': 'əl yoxdur',
  moving: 'hərəkət',
  unsure: 'qeyri-müəyyən',
  hold: 'saxla',
  typed: 'yazıldı',
};

const state = {
  model: null, decoder: null, poses: null, eval: null,
  s2t: null, t2s: null,
  skelS2T: null, skelT2S: null, avatar: null,
  booted: { s2t: false, t2s: false, about: false },
};

function setStatus(text, stateName = '') {
  $('#status-text').textContent = text;
  $('#status').dataset.state = stateName;
}

/* ══════════════════ yükləmə ══════════════════ */
async function boot() {
  try {
    setStatus('Model yüklənir…');
    const [model, decoder, poses, evalData] = await Promise.all([
      FingerspellModel.load(`${ASSET_BASE}model.json`, `${ASSET_BASE}model.bin`),
      LexiconDecoder.load(`${ASSET_BASE}lm.json`, `${ASSET_BASE}vocab.json`, `${ASSET_BASE}prefixes.json`),
      loadPoses(`${ASSET_BASE}poses.json`),
      fetch(`${ASSET_BASE}eval.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
    ]);
    Object.assign(state, { model, decoder, poses, eval: evalData });
    setStatus(`${model.labels.length} sinif · ${decoder.words.length.toLocaleString('az')} söz`, 'ready');

    window.__chevir = state;   // sazlama qarmağı (brauzer konsolundan tənzimləmə)
    initTabs();
    initS2T();
    initT2S();
    showTab('s2t');
  } catch (err) {
    console.error(err);
    setStatus('Yükləmə xətası: ' + err.message, 'error');
  }
}

/* ══════════════════ tablar ══════════════════ */
function initTabs() {
  $$('.tab').forEach((btn) => {
    btn.addEventListener('click', () => showTab(btn.dataset.tab));
  });
}

function showTab(name) {
  $$('.tab').forEach((b) => b.classList.toggle('is-active', b.dataset.tab === name));
  $$('.panel').forEach((p) => p.classList.toggle('is-active', p.id === `tab-${name}`));

  // WebGL kontekstləri baha olduğu üçün ilk açılışda qurulur
  if (name === 's2t' && !state.booted.s2t) {
    state.skelS2T = new CyberHand($('#s2t-skeleton'));
    state.booted.s2t = true;
  }
  if (name === 't2s' && !state.booted.t2s) {
    state.booted.t2s = true;
    bootT2SViews();
  }
  if (name === 'about' && !state.booted.about) {
    state.booted.about = true;
    renderAbout();
  }
  requestAnimationFrame(() => {
    state.skelS2T?.resize(); state.skelT2S?.resize(); state.avatar?.resize();
  });
}

/* ══════════════════ İŞARƏ → MƏTN ══════════════════ */
function initS2T() {
  const video = $('#cam');
  const canvas = $('#overlay');
  const ctx = canvas.getContext('2d');

  const s2t = new SignToText({
    model: state.model, decoder: state.decoder, video,
    onStatus: ({ message, stage }) => setStatus(message, stage),
    onFrame: ({ landmarks, progress, state: gate }) => {
      drawOverlay(ctx, canvas, video, landmarks);
      state.skelS2T?.setPose(landmarks ?? null);
      const ring = $('#ring');
      if (ring) ring.style.strokeDashoffset = String(106.8 * (1 - progress));
      $('#ring-label').textContent = GATE_LABEL[gate] ?? '';
    },
    onPrediction: ({ top, quality }) => {
      $('#pred-quality').textContent = `əl ${Math.round(quality * 100)}%`;
      renderPredictions(top);
    },
    onDecode: renderDecode,
  });
  state.s2t = s2t;

  $('#btn-start').addEventListener('click', async () => {
    $('#btn-start').disabled = true;
    try {
      await s2t.start();
      $('#btn-stop').disabled = false;
    } catch (err) {
      setStatus('Kamera açılmadı: ' + err.message, 'error');
      $('#btn-start').disabled = false;
    }
  });
  $('#btn-stop').addEventListener('click', () => {
    s2t.stop();
    $('#btn-start').disabled = false;
    $('#btn-stop').disabled = true;
    state.skelS2T?.clearPose();
  });
  $('#btn-finish').addEventListener('click', () => s2t.finish());
  $('#btn-clear').addEventListener('click', () => {
    s2t.clear();
    renderPredictions([]);
  });
  $('#mode').addEventListener('change', (e) => { s2t.decodeOpts.mode = e.target.value; });

  // Kamerasız sınaq üçün klaviatura girişi
  addEventListener('keydown', (e) => {
    if (!$('#tab-s2t').classList.contains('is-active')) return;
    if (e.target.matches('input,textarea,select')) return;
    if (e.key === 'Enter') { e.preventDefault(); s2t.finish(); return; }
    if (e.key === 'Backspace') { e.preventDefault(); s2t.undo(); return; }
    const ch = e.key.toLowerCase();
    if (ch.length === 1 && state.decoder.letters.includes(ch)) {
      e.preventDefault();
      s2t.pushLetter(ch, 0.86);
    }
  });
}

function drawOverlay(ctx, canvas, video, landmarks) {
  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h) return;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  ctx.clearRect(0, 0, w, h);
  if (!landmarks) return;

  ctx.lineWidth = Math.max(2, w / 420);
  ctx.strokeStyle = 'rgba(63,193,232,.9)';
  ctx.shadowColor = 'rgba(63,193,232,.8)';
  ctx.shadowBlur = 10;
  ctx.beginPath();
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
    ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
  }
  ctx.stroke();

  const tips = new Set([4, 8, 12, 16, 20]);
  landmarks.forEach((p, i) => {
    ctx.beginPath();
    ctx.fillStyle = i === 0 ? '#f5b94f' : tips.has(i) ? '#ff4f97' : '#f1ede9';
    ctx.arc(p.x * w, p.y * h, Math.max(3, w / 260), 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.shadowBlur = 0;
}

function renderPredictions(top) {
  const box = $('#pred-grid');
  if (!top.length) { box.innerHTML = '<div class="pred-empty">Hələ proqnoz yoxdur</div>'; return; }
  box.innerHTML = top.slice(0, 3).map((t) => {
    const label = t.label === ' ' ? '␣' : t.label;
    return `<div class="pred">
      <div class="pred-ch">${escapeHtml(label)}</div>
      <div class="pred-bar"><i style="width:${(t.p * 100).toFixed(1)}%"></i></div>
      <div class="pred-p">${(t.p * 100).toFixed(1)}%</div>
    </div>`;
  }).join('');
}

function renderDecode({ candidates, best, verified, suggestions, letters }) {
  $('#letters').innerHTML = letters?.length
    ? letters.map((l) => `<span class="l">${escapeHtml(l === ' ' ? '␣' : l)}</span>`).join('')
    : '<span class="muted">boş</span>';

  const dec = $('#decoded');
  if (!best) { dec.innerHTML = '<span class="muted">—</span>'; }
  else {
    dec.innerHTML = `<span class="${verified ? 'ok' : 'no'}">${escapeHtml(best.text)}</span>` +
      (verified ? '<span class="tick">lüğətdə var</span>' : '');
  }

  $('#candidates').innerHTML = (candidates ?? []).slice(0, 5).map((c, i) =>
    `<li class="${i === 0 ? 'top' : ''}">
       <span class="w">${escapeHtml(c.text)}</span>
       <span class="v">${c.inVocab ? '✓' : ''}</span>
       <span class="c">${(c.conf * 100).toFixed(0)}%</span>
     </li>`).join('');

  const sg = (suggestions ?? []).filter((s) => s.dist > 0);
  $('#suggest').innerHTML = sg.length
    ? 'Ən yaxın real sözlər: ' + sg.map((s) => `<b>${escapeHtml(s.word)}</b>`).join(', ')
    : '';
}

/* ══════════════════ MƏTN → İŞARƏ ══════════════════ */
async function bootT2SViews() {
  state.skelT2S = new CyberHand($('#t2s-skeleton'));
  const { VrmAvatar } = await import('./vrm.js');
  state.avatar = new VrmAvatar($('#t2s-avatar'));
  setStatus('Avatar yüklənir…');
  try {
    const vrmFile = $('#avatar-select')?.value || 'avatar.vrm';
    await state.avatar.load(`${ASSET_BASE}${vrmFile}`);
    state.avatar.setHand($('#hand')?.value || 'right');
    setStatus('Avatar hazır', 'ready');
  } catch (err) {
    console.error(err);
    setStatus('Avatar yüklənmədi: ' + err.message, 'error');
    $('#t2s-avatar').insertAdjacentHTML('beforeend',
      `<div class="warn" style="position:absolute;inset:auto 12px 12px 12px">
         VRM avatarı yüklənmədi — cyber skeleton işləməyə davam edir.</div>`);
  }
  state.t2s.targets = [state.avatar, state.skelT2S].filter(Boolean);
}

function initT2S() {
  const t2s = new TextToSign({
    poses: state.poses,
    targets: [],
    onLetter: ({ char, index }) => {
      $('#current-letter').textContent = char === ' ' ? '␣' : char;
      $$('#spelled i').forEach((el, i) => {
        el.classList.toggle('now', i === index);
        el.classList.toggle('done', i < index);
      });
      $$('.abc button').forEach((b) => b.classList.toggle('is-on', b.dataset.ch === char));
    },
    onState: ({ playing, index, total }) => {
      $('#play-progress').textContent = playing ? `${index + 1}/${total}` : 'hazır';
      if (!playing) {
        $('#current-letter').textContent = '·';
        $$('.abc button').forEach((b) => b.classList.remove('is-on'));
      }
    },
  });
  state.t2s = t2s;

  const input = $('#t2s-input');
  const refresh = () => {
    const text = input.value;
    $('#spelled').innerHTML = [...text].length
      ? [...text].map((c) => `<i>${escapeHtml(c === ' ' ? '␣' : c)}</i>`).join('')
      : '<span class="muted">—</span>';
    const miss = t2s.missing(text);
    const warn = $('#t2s-warn');
    warn.hidden = !miss.length;
    if (miss.length) {
      warn.textContent = `Bu simvolların pozası yoxdur, buraxılacaq: ${miss.join(' ')}`;
    }
  };
  input.addEventListener('input', refresh);

  $('#btn-play').addEventListener('click', () => t2s.play(input.value));
  $('#btn-stop-play').addEventListener('click', () => {
    t2s.stop();
    state.avatar?.clearHandPose();
    state.skelT2S?.clearPose();
    $('#play-progress').textContent = 'hazır';
  });
  $('#hold').addEventListener('input', (e) => {
    t2s.opts.holdMs = +e.target.value;
    $('#hold-val').textContent = e.target.value + 'ms';
  });
  $('#hand').addEventListener('change', (e) => {
    state.avatar?.setHand(e.target.value);
  });
  const avatarSelect = $('#avatar-select');
  if (avatarSelect) {
    avatarSelect.addEventListener('change', async (e) => {
      setStatus('Avatar yüklənir…');
      try {
        await state.avatar?.load(`${ASSET_BASE}${e.target.value}`);
        state.avatar?.setHand($('#hand').value);
        setStatus('Avatar hazır', 'ready');
      } catch (err) {
        setStatus('Avatar yüklənmədi: ' + err.message, 'error');
      }
    });
  }

  // Əlifba vərəqi
  const letters = Object.keys(state.poses.letters);
  $('#abc-count').textContent = `${letters.length} hərf`;
  $('#abc').innerHTML = letters.map((ch) => {
    const dyn = state.poses.letters[ch].type === 'dynamic';
    return `<button data-ch="${escapeHtml(ch)}" class="${dyn ? 'dyn' : ''}"
      title="${escapeHtml(ch)}${dyn ? ' · dinamik' : ''}">${escapeHtml(ch)}</button>`;
  }).join('');
  $('#abc').addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-ch]');
    if (!btn) return;
    t2s.showLetter(btn.dataset.ch);
    $$('.abc button').forEach((b) => b.classList.toggle('is-on', b === btn));
  });

  refresh();
}

/* ══════════════════ MODEL sənədi ══════════════════ */
function renderAbout() {
  const e = state.eval;
  const m = state.model.meta;
  const arch = m.layers.map((l) => `${l.in}→${l.out}`).join(' · ');

  const bars = (rows) => `<div class="bars">${rows.map(([ch, acc]) =>
    `<div><span>${escapeHtml(ch)}</span><u><i style="width:${acc.toFixed(1)}%"></i></u><em>${acc.toFixed(0)}%</em></div>`
  ).join('')}</div>`;

  $('#doc').innerHTML = `
    <h2>Model və ölçmələr</h2>
    <p>Sistem barmaq əlifbası modelini (<code>fingerspelling_33.h5</code>) olduğu kimi işlədir. Keras çəkiləri
    xam <code>float32</code> kimi çıxarılıb, irəli keçid JS-də 4 matmul ilə hesablanır —
    TensorFlow.js lazım deyil. Port numpy referansı ilə yoxlanılıb, maksimal fərq
    <code>4.8e-7</code> (float32 həddi).</p>

    <h3>Memarlıq</h3>
    <table>
      <tr><th>Giriş</th><td class="n">${m.input.frames} kadr × ${m.input.features} = ${m.input.flat}</td></tr>
      <tr><th>Qatlar</th><td class="n">${arch}</td></tr>
      <tr><th>Parametr</th><td class="n">${m.params.toLocaleString('az')}</td></tr>
      <tr><th>Sinif</th><td class="n">${m.labels.length} (32 hərf + boşluq/enter/backspace)</td></tr>
      <tr><th>Xüsusiyyət</th><td class="n">MediaPipe Hands, birinci əl, xam x/y/z</td></tr>
    </table>

    <h3>Dekoder</h3>
    <p>Hər proqnozdan ən yaxşı 3 hərf beam-ə düşür. Namizədlər hərf-bigram dil modeli ilə
    çəkilir (leksikondan: ${state.decoder.words.length.toLocaleString('az')} unikal söz,
    ${state.decoder.prefixes.size.toLocaleString('az')} prefiks). Sonda lüğət yoxlaması:
    tam uyğunluq bonus alır, uyğunluq yoxdursa redaktə məsafəsi ilə ən yaxın real sözlər təklif olunur.</p>

    <h3>Hərf nə vaxt yazılır</h3>
    <p>Kameradan hərf yalnız əl sabit olanda yazılır: pəncərənin hər 20 kadrında əl görünməli,
    hərəkət ölçüsü kiçik olmalı (sabit poza 0.00–0.04, keçid 0.06–0.11), model ən azı
    <b>45%</b> əmin olmalı və eyni hərf ardıcıl 3 proqnozda təsdiqlənməlidir. Bu şərtlər olmadan
    əlin kadra girdiyi an boş kadrlarla birlikdə dinamik hərf kimi oxunurdu — məsələn
    5 boş kadr + 15 kadr “s” model üçün <code>ö</code> (0.89) deməkdir.</p>

    ${e ? `
    <h3>Real AzSLD datasında ölçmə</h3>
    <p>${e.images.toLocaleString('az')} şəkil (AzSLD Fingerspelling, Zenodo 14222948),
    ${(e.detectRate * 100).toFixed(1)}%-də əl aşkarlandı.</p>
    <table>
      <tr><th>Qrup</th><th>Hərf</th><th>Orta top-1</th><th>Median</th><th>≥90%</th></tr>
      <tr><td>Statik</td><td class="n">${e.static.n}</td><td class="n">${e.static.mean.toFixed(1)}%</td>
          <td class="n">${e.static.median.toFixed(1)}%</td><td class="n">${e.static.ge90}/${e.static.n}</td></tr>
      <tr><td>Dinamik</td><td class="n">${e.dynamic.n}</td><td class="n">${e.dynamic.mean.toFixed(1)}%</td>
          <td class="n">${e.dynamic.median.toFixed(1)}%</td><td class="n">${e.dynamic.ge90}/${e.dynamic.n}</td></tr>
    </table>
    <div class="note"><b>İki vacib qeyd.</b>
    (1) Dinamik hərflərin sıfıra yaxın nəticəsi modelin uğursuzluğu deyil, ölçmə üsulunun
    nəticəsidir: bir kadr 20 dəfə təkrarlanaraq verilir, hərəkət isə belə təmsil oluna bilmir.
    Real ardıcıllıqla verildikdə <b>ç</b> və <b>ö</b> düzgün tanınır.
    (2) Bu şəkillər modelin təlim datası ola bilər — yəni rəqəmlər <b>optimistdir</b>,
    kənarlaşdırılmış test dəsti deyil.</div>
    <h3>Statik hərflər üzrə dəqiqlik</h3>
    ${bars(e.perLetter.filter((r) => r.type === 'static').map((r) => [r.letter, r.acc]))}
    ` : ''}

    <h3>Niyə hər şey brauzerdə</h3>
    <p>Server yoxdur: video cihazdan çıxmır, gecikmə minimaldır və sistem statik fayl kimi
    hər yerdə host oluna bilər. Əlçatanlıq məhsulu üçün məxfilik təsadüfi seçim deyil —
    jest dili videosu istifadəçinin üzünü və kimliyini daşıyır.</p>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

boot();
