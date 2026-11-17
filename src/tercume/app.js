/**
 * Chevir tərcümə sistemi — bağlayıcı qat.
 *
 * Hər şey brauzerdə işləyir: server yoxdur, video cihazdan çıxmır.
 * Bu, əlçatanlıq məhsulu üçün həm məxfilik, həm də xərc baxımından vacibdir
 * (statik fayl kimi hər yerdə host oluna bilər).
 */
import { ASSET_BASE, WORDS_BASE } from './paths.js';
import { WordLexicon, azLower } from './words.js';
import { loadLetterModel } from './letters.js';
import { LexiconDecoder } from './decoder.js';
import { CyberHand } from './skeleton.js';
import { SignToText } from './s2t.js';
import { TextToSign, loadPoses } from './t2s.js';
import { HAND_CONNECTIONS } from './hands.js';
import { t, currentLang } from '../lib/i18n.js';
import { translateStatusMessage } from './ui-i18n.js';
import './layout.js';

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

// Model sənədindəki böyük ədədlər (söz/parametr sayı və s.) bu lokalla formatlaşdırılır -
// yalnız minlik ayırıcı dəyişir, hesablanan dəyərlərin özü toxunulmaz qalır.
const locale = currentLang === 'en' ? 'en-US' : 'az';

// Kamera overlay-nin (#overlay canvas) rəngləri CSS dəyişənlərindən - hər kadrda
// deyil, yalnız yükləmədə və 'chevir:theme' hadisəsində yenilənir (performans).
const camColors = { line: '', glow: '', wrist: '', pink: '', dot: '' };
function readCamColors() {
  const cs = getComputedStyle(document.body);
  const get = (name, fallback) => cs.getPropertyValue(name).trim() || fallback;
  camColors.line = get('--t-cam-line', 'rgba(63,193,232,.9)');
  camColors.glow = get('--t-cam-glow', 'rgba(63,193,232,.8)');
  camColors.wrist = get('--t-cam-wrist', '#f5b94f');
  camColors.pink = get('--t-cam-pink', '#ff4f97');
  camColors.dot = get('--t-cam-dot', '#f1ede9');
}
readCamColors();
document.addEventListener('chevir:theme', readCamColors);

// Söz rejimində halqanın yazısı (bax s2w.js)
const WORD_STATE_LABEL = {
  rest: t('tercume.status.wordState.rest'),
  signing: t('tercume.status.wordState.signing'),
  recognizing: t('tercume.status.wordState.recognizing'),
};

// Kamera halqasının altındakı vəziyyət yazısı (bax s2t.js — hərf yazma şərtləri)
const GATE_LABEL = {
  idle: '-',
  'wait-hand': t('tercume.status.gate.waitHand'),
  moving: t('tercume.status.gate.moving'),
  unsure: t('tercume.status.gate.unsure'),
  hold: t('tercume.status.gate.hold'),
  typed: t('tercume.status.gate.typed'),
};

const state = {
  model: null, decoder: null, poses: null, eval: null, lexicon: null,
  s2t: null, s2w: null, kind: 'letters', t2s: null,
  skelS2T: null, skelT2S: null, avatar: null,
  booted: { s2t: false, t2s: false, about: false },
  activeTab: 's2t',
  camStatus: { text: '', stateName: '' },
  t2sStatus: { text: '', stateName: '' },
};
let t2sIdleStatus = { text: '', stateName: '' };   // son bilinən avatar hazır/xəta statusu — oynatma bitəndə buraya qayıdılır

// Status zolağı tab-a görə hədəflənir: 's2t' kamera, 't2s' avatar/oynatma, 'global' hər zaman
// (yükləmə mərhələsi, hələ heç bir tab aktiv deyilkən). showTab() cari tab üçün uyğun olanı yenidən çəkir.
function setStatus(text, stateName = '', scope = 'global') {
  const translated = translateStatusMessage(text);
  if (scope === 's2t') state.camStatus = { text: translated, stateName };
  else if (scope === 't2s') state.t2sStatus = { text: translated, stateName };
  if (scope !== 'global' && scope !== state.activeTab) return;
  $('#status-text').textContent = translated;
  $('#status').dataset.state = stateName;
}

function syncStatusForTab(name) {
  const el = $('#status');
  el.hidden = name === 'about';
  if (el.hidden) return;
  const s = name === 't2s' ? state.t2sStatus : state.camStatus;
  $('#status-text').textContent = s.text;
  el.dataset.state = s.stateName;
}

/* ══════════════════ yükləmə ══════════════════ */
async function boot() {
  try {
    setStatus(t('tercume.status.modelLoading'), '', 's2t');
    const [model, decoder, poses, evalData, lexicon] = await Promise.all([
      loadLetterModel({ assetBase: ASSET_BASE }),
      LexiconDecoder.load(`${ASSET_BASE}lm.json`, `${ASSET_BASE}vocab.json`, `${ASSET_BASE}prefixes.json`),
      loadPoses(`${ASSET_BASE}poses.json`),
      fetch(`${ASSET_BASE}eval.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
      WordLexicon.load(`${WORDS_BASE}index.json`),
    ]);
    Object.assign(state, { model, decoder, poses, eval: evalData, lexicon });
    setStatus(t('tercume.status.modelReady', { classes: model.labels.length, words: decoder.words.length.toLocaleString(locale) }), 'ready', 's2t');

    window.__chevir = state;   // sazlama qarmağı (brauzer konsolundan tənzimləmə)
    initTabs();
    initS2T();
    initT2S();
    showTab('s2t');
  } catch (err) {
    console.error(err);
    setStatus(`${t('tercume.status.loadError')} (${err.message})`, 'error');
  }
}

/* ══════════════════ tablar ══════════════════ */
function initTabs() {
  $$('.tab').forEach((btn) => {
    btn.addEventListener('click', () => showTab(btn.dataset.tab));
  });
}

function showTab(name) {
  state.activeTab = name;
  $$('.tab').forEach((b) => b.classList.toggle('is-active', b.dataset.tab === name));
  $$('.panel').forEach((p) => p.classList.toggle('is-active', p.id === `tab-${name}`));
  syncStatusForTab(name);

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
  const camViewport = video.closest('.viewport');

  const s2t = new SignToText({
    model: state.model, decoder: state.decoder, video,
    onStatus: ({ message, stage }) => setStatus(message, stage, 's2t'),
    onFrame: ({ landmarks, progress, state: gate }) => {
      drawOverlay(ctx, canvas, video, landmarks ? [landmarks] : []);
      state.skelS2T?.setPose(landmarks ?? null);
      const ring = $('#ring');
      if (ring) ring.style.strokeDashoffset = String(106.8 * (1 - progress));
      $('#ring-label').textContent = GATE_LABEL[gate] ?? '';
    },
    onPrediction: ({ top, quality }) => {
      $('#pred-quality').textContent = t('tercume.status.handConfidence', { q: Math.round(quality * 100) });
      renderPredictions(top);
    },
    onDecode: renderDecode,
  });
  state.s2t = s2t;

  // Söz rejimi: model, onnxruntime və üç izləyici yalnız ilk dəfə seçiləndə yüklənir
  let s2wLoading = null;
  const ensureS2W = () => {
    s2wLoading ??= (async () => {
      const { SignToWords } = await import('./s2w.js');
      const s2w = new SignToWords({
        video,
        onStatus: ({ message, stage }) => setStatus(message, stage, 's2t'),
        onFrame: ({ frame, state: st }) => {
          const hands = [0, 1].map((s) => slotLandmarks(frame.hand, s)).filter(Boolean);
          drawOverlay(ctx, canvas, video, hands);
          state.skelS2T?.setPose(hands[0] ?? null);
          const ring = $('#ring');
          if (ring) ring.style.strokeDashoffset = String(106.8 * (st === 'rest' ? 1 : 0));
          $('#ring-label').textContent = WORD_STATE_LABEL[st] ?? '';
        },
        onSigns: ({ signs, added, ms }) => {
          renderSigns(signs, added.length);
          const last = added[added.length - 1];
          const unk = s2w.rec.manifest.unknown_label;
          if (last) renderPredictions(last.top.map((p) => (p.label === unk ? { ...p, label: t('tercume.status.unknown') } : p)));
          $('#signs-info').textContent = added.length
            ? t('tercume.status.signsCountNew', { n: added.length, ms: Math.round(ms) })
            : signs.length ? t('tercume.status.wordCount', { n: signs.length }) : '-';
          $('#pred-quality').textContent = !last ? '-'
            : last.unsure ? t('tercume.status.unsureWithPercent', { p: Math.round(last.p * 100) })
            : t('tercume.status.confidentWithPercent', { p: Math.round(last.p * 100) });
        },
      });
      await s2w.init();
      const f1 = s2w.rec.manifest.metrics?.test_words?.macro_f1;
      if (f1) $('#words-f1').textContent = f1.toFixed(2);
      state.s2w = s2w;
      return s2w;
    })();
    s2wLoading.catch(() => { s2wLoading = null; });     // xətadan sonra yenidən cəhd etmək olar
    return s2wLoading;
  };
  // Cümlə rejimi: davamlı model (hər kadra etiket), sözlər fasiləsiz çıxır
  let s2sLoading = null;
  const ensureS2S = () => {
    s2sLoading ??= (async () => {
      const { SignToSentence } = await import('./s2s.js');
      const s2s = new SignToSentence({
        video,
        onStatus: ({ message, stage }) => setStatus(message, stage, 's2t'),
        onFrame: ({ frame }) => {
          const hands = [0, 1].map((s) => slotLandmarks(frame.hand, s)).filter(Boolean);
          drawOverlay(ctx, canvas, video, hands);
          state.skelS2T?.setPose(hands[0] ?? null);
          $('#ring-label').textContent = hands.length ? WORD_STATE_LABEL.signing : WORD_STATE_LABEL.rest;
        },
        onWords: ({ words, added, sentence, text, ms }) => {
          renderSentence(words, added.length, sentence, text);
          const last = added[added.length - 1];
          if (last) renderPredictions([{ label: last.label, p: last.p }]);
          $('#sentence-info').textContent = t('tercume.status.signsCountNew', { n: added.length, ms: Math.round(ms) });
        },
        onSentence: ({ words, match, text }) => {
          const li = document.createElement('li');
          li.textContent = words.map((w) => w.label).join(' ') + (text ? `  ${match ? '≈' : '→'}  ${text}` : '');
          $('#sentence-history').prepend(li);
          renderSentence([], 0, null);
        },
      });
      await s2s.init();
      const f1 = s2s.rec.manifest.metrics?.test?.f1_all;
      if (f1) $('#sentence-f1').textContent = f1.toFixed(2);
      state.s2s = s2s;
      return s2s;
    })();
    s2sLoading.catch(() => { s2sLoading = null; });
    return s2sLoading;
  };
  const current = () => (state.kind === 'words' ? state.s2w : state.kind === 'sentence' ? state.s2s : s2t);

  $('#s2t-kind').addEventListener('change', (e) => {
    if (!$('#btn-stop').disabled) $('#btn-stop').click();     // rejim dəyişəndə kamera dayanır
    state.kind = e.target.value;
    $$('#tab-s2t [data-kind]').forEach((el) => { el.hidden = !el.dataset.kind.split(' ').includes(state.kind); });
    $('#pred-grid').classList.toggle('is-words', state.kind !== 'letters');
    $('#ring-label').textContent = state.kind !== 'letters' ? WORD_STATE_LABEL.rest : GATE_LABEL.idle;
    $('#pred-quality').textContent = '-';
    renderPredictions([]);
    if (state.kind === 'words') ensureS2W().catch((err) => setStatus(`${t('tercume.status.wordModelFailed')} (${err.message})`, 'error', 's2t'));
    if (state.kind === 'sentence') ensureS2S().catch((err) => setStatus(`${t('tercume.status.sentenceModelFailed')} (${err.message})`, 'error', 's2t'));
  });

  $('#btn-start').addEventListener('click', async () => {
    $('#btn-start').disabled = true;
    try {
      const runner = state.kind === 'words' ? await ensureS2W() : state.kind === 'sentence' ? await ensureS2S() : s2t;
      await runner.start();
      $('#btn-stop').disabled = false;
      camViewport.classList.add('is-live');
      $('#pred-grid').closest('.card').classList.remove('is-stale');
    } catch (err) {
      setStatus(`${t('tercume.status.cameraFailed')} (${err.message})`, 'error', 's2t');
      $('#btn-start').disabled = false;
    }
  });
  $('#btn-stop').addEventListener('click', () => {
    current()?.stop();
    $('#btn-start').disabled = false;
    $('#btn-stop').disabled = true;
    camViewport.classList.remove('is-live');
    state.skelS2T?.clearPose();
    $('#pred-grid').closest('.card').classList.add('is-stale');
  });
  $('#btn-finish').addEventListener('click', () => s2t.finish());
  $('#btn-clear').addEventListener('click', () => {
    current()?.clear();
    renderPredictions([]);
    if (state.kind === 'sentence') { renderSentence([], 0, null); $('#sentence-history').innerHTML = ''; }
  });
  $('#mode').addEventListener('change', (e) => { s2t.decodeOpts.mode = e.target.value; });

  // Kamerasız sınaq üçün klaviatura girişi
  addEventListener('keydown', (e) => {
    if (!$('#tab-s2t').classList.contains('is-active') || state.kind !== 'letters') return;
    // #theme-toggle öz Enter/Space aktivasiyasını almalıdır (native button
    // davranışı) - bu qlobal qısayol onu əvvəl tutub preventDefault edirdi.
    if (e.target.matches('input,textarea,select') || e.target.id === 'theme-toggle') return;
    if (e.key === 'Enter') { e.preventDefault(); s2t.finish(); return; }
    if (e.key === 'Backspace') { e.preventDefault(); s2t.undo(); return; }
    const ch = e.key.toLowerCase();
    if (ch.length === 1 && state.decoder.letters.includes(ch)) {
      e.preventDefault();
      s2t.pushLetter(ch, 0.86);
    }
  });
}

/** landmarks.js kadrının əl slotu (126 ədəd, slot başına 63) → 21 nöqtə, əl yoxdursa null. */
function slotLandmarks(hand, s) {
  if (!Number.isFinite(hand[s * 63])) return null;
  return Array.from({ length: 21 }, (_, j) => {
    const o = s * 63 + j * 3;
    return { x: hand[o], y: hand[o + 1], z: hand[o + 2] };
  });
}

/** Kadrın üstünə əl skeletləri (bir və ya iki əl). */
function drawOverlay(ctx, canvas, video, hands) {
  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h) return;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  ctx.clearRect(0, 0, w, h);
  for (const landmarks of hands) drawHand(ctx, w, h, landmarks);
}

function drawHand(ctx, w, h, landmarks) {
  ctx.lineWidth = Math.max(2, w / 420);
  ctx.strokeStyle = camColors.line;
  ctx.shadowColor = camColors.glow;
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
    ctx.fillStyle = i === 0 ? camColors.wrist : tips.has(i) ? camColors.pink : camColors.dot;
    ctx.arc(p.x * w, p.y * h, Math.max(3, w / 260), 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.shadowBlur = 0;
}

function renderPredictions(top) {
  const box = $('#pred-grid');
  if (!top.length) { box.innerHTML = `<div class="pred-empty">${escapeHtml(t('tercume.s2t.predEmpty'))}</div>`; return; }
  box.innerHTML = top.slice(0, 3).map((p) => {
    const label = p.label === ' ' ? '␣' : p.label;
    return `<div class="pred">
      <div class="pred-ch">${escapeHtml(label)}</div>
      <div class="pred-bar"><i style="width:${(p.p * 100).toFixed(1)}%"></i></div>
      <div class="pred-p">${(p.p * 100).toFixed(1)}%</div>
    </div>`;
  }).join('');
}

function renderSigns(signs, nNew = 0) {
  $('#signs').innerHTML = signs.length
    ? signs.map((s, i) => {
      const unk = state.s2w?.rec.manifest.unknown_label;
      const unknownLabel = t('tercume.status.unknown');
      const alt = s.top.map((p) => `${p.label === unk ? unknownLabel : p.label} ${(p.p * 100).toFixed(0)}%`).join(' · ');
      const cls = ['s', i >= signs.length - nNew && 'new', s.unsure && 'unsure'].filter(Boolean).join(' ');
      const label = s.unknown ? unknownLabel : s.unsure ? `${s.label}?` : s.label;
      return `<span class="${cls}" title="${escapeHtml(alt)}">${escapeHtml(label)}<em>${Math.round(s.p * 100)}%</em></span>`;
    }).join('')
    : `<span class="muted">${escapeHtml(t('tercume.s2t.emptyMuted'))}</span>`;
}

function renderSentence(words, nNew, match, text = null) {
  $('#sentence-words').innerHTML = words.length
    ? words.map((w, i) => `<span class="s${i >= words.length - nNew ? ' new' : ''}">${escapeHtml(w.label)}<em>${Math.round(w.p * 100)}%</em></span>`).join('')
    : `<span class="muted">${escapeHtml(t('tercume.s2t.emptyMuted'))}</span>`;
  // Mətn: AzSLD-də yaxın cümlə varsa onun mətni, yoxsa işarələrdən qurulmuş mətn (gloss2text.js)
  const m = $('#sentence-match');
  m.hidden = !text;
  if (text) m.textContent = match ? t('tercume.status.sentenceMatchKnown', { text }) : t('tercume.status.sentenceMatchGenerated', { text });
}

function renderDecode({ candidates, best, verified, suggestions, letters }) {
  $('#letters').innerHTML = letters?.length
    ? letters.map((l) => `<span class="l">${escapeHtml(l === ' ' ? '␣' : l)}</span>`).join('')
    : `<span class="muted">${escapeHtml(t('tercume.s2t.lettersEmpty'))}</span>`;

  const dec = $('#decoded');
  if (!best) { dec.innerHTML = '<span class="muted">-</span>'; }
  else {
    dec.innerHTML = `<span class="${verified ? 'ok' : 'no'}">${escapeHtml(best.text)}</span>` +
      (verified ? `<span class="tick">${escapeHtml(t('tercume.status.inVocab'))}</span>` : '');
  }

  $('#candidates').innerHTML = (candidates ?? []).slice(0, 5).map((c, i) =>
    `<li class="${i === 0 ? 'top' : ''}">
       <span class="w">${escapeHtml(c.text)}</span>
       <span class="v">${c.inVocab ? '✓' : ''}</span>
       <span class="c">${(c.conf * 100).toFixed(0)}%</span>
     </li>`).join('');

  const sg = (suggestions ?? []).filter((s) => s.dist > 0);
  $('#suggest').innerHTML = sg.length
    ? `${escapeHtml(t('tercume.status.suggestPrefix'))} ` + sg.map((s) => `<b>${escapeHtml(s.word)}</b>`).join('')
    : '';
}

/* ══════════════════ MƏTN → İŞARƏ ══════════════════ */
async function bootT2SViews() {
  state.skelT2S = new CyberHand($('#t2s-skeleton'));
  const { VrmAvatar } = await import('./vrm.js');
  state.avatar = new VrmAvatar($('#t2s-avatar'));
  setStatus(t('tercume.status.avatarLoading'), '', 't2s');
  try {
    const vrmFile = $('#avatar-select')?.value || 'avatar.vrm';
    await state.avatar.load(`${ASSET_BASE}${vrmFile}`);
    state.avatar.setHand($('#hand')?.value || 'right');
    t2sIdleStatus = { text: t('tercume.status.avatarReady'), stateName: 'ready' };
    setStatus(t2sIdleStatus.text, t2sIdleStatus.stateName, 't2s');
  } catch (err) {
    console.error(err);
    t2sIdleStatus = { text: `${t('tercume.status.avatarFailed')} (${err.message})`, stateName: 'error' };
    setStatus(t2sIdleStatus.text, t2sIdleStatus.stateName, 't2s');
    $('#t2s-avatar').insertAdjacentHTML('beforeend',
      `<div class="warn" style="position:absolute;inset:auto 12px 12px 12px">
         ${escapeHtml(t('tercume.status.vrmFailWarning'))}</div>`);
  }
  state.t2s.targets = [state.avatar, state.skelT2S].filter(Boolean);
}

function initT2S() {
  const t2s = new TextToSign({
    poses: state.poses,
    lexicon: state.lexicon,
    targets: [],
    onLetter: ({ kind, char, index }) => {
      $('#current-letter').textContent = char === ' ' ? '' : char;
      $('#current-letter').classList.toggle('is-word', kind === 'word');
      $$('#spelled i').forEach((el, i) => {
        el.classList.toggle('now', i === index);
        el.classList.toggle('done', i < index);
      });
      $$('.abc button').forEach((b) => b.classList.toggle('is-on', b.dataset.ch === char));
    },
    onState: ({ playing, index, total }) => {
      $('#play-progress').textContent = playing ? `${index + 1}/${total}` : t('tercume.status.playReady');
      if (playing) {
        setStatus(t('tercume.status.signPlaying'), 'running', 't2s');
      } else {
        $('#current-letter').textContent = '';
        $$('.abc button').forEach((b) => b.classList.remove('is-on'));
        setStatus(t2sIdleStatus.text, t2sIdleStatus.stateName, 't2s');
      }
    },
  });
  state.t2s = t2s;

  const input = $('#t2s-input');
  const refresh = () => {
    const text = input.value;
    const steps = t2s.plan(text);
    $('#spelled').innerHTML = steps.length
      ? steps.map((s) => `<i${s.kind === 'word' ? ' class="w"' : ''}>${escapeHtml(s.kind === 'space' ? '␣' : s.char)}</i>`).join('')
      : '<span class="muted">-</span>';
    const miss = t2s.missing(text);
    const warn = $('#t2s-warn');
    warn.hidden = !miss.length;
    if (miss.length) {
      warn.textContent = t('tercume.status.missingChars', { chars: miss.join(' ') });
    }
  };
  input.addEventListener('input', refresh);

  $('#btn-play').addEventListener('click', () => t2s.play(input.value));
  $('#btn-stop-play').addEventListener('click', () => {
    t2s.stop();
    state.avatar?.clearHandPose();
    state.skelT2S?.clearPose();
    $('#play-progress').textContent = t('tercume.status.playReady');
  });
  $('#hold').addEventListener('input', (e) => {
    t2s.opts.holdMs = +e.target.value;
    $('#hold-val').textContent = e.target.value + 'ms';
  });
  $('#speed').addEventListener('change', (e) => {
    t2s.opts.speed = +e.target.value;    // 0.25× — dörd dəfə yavaş, 2× — iki dəfə sürətli
  });
  $('#hand').addEventListener('change', (e) => {
    state.avatar?.setHand(e.target.value);
  });
  const avatarSelect = $('#avatar-select');
  if (avatarSelect) {
    avatarSelect.addEventListener('change', async (e) => {
      setStatus(t('tercume.status.avatarLoading'), '', 't2s');
      try {
        await state.avatar?.load(`${ASSET_BASE}${e.target.value}`);
        state.avatar?.setHand($('#hand').value);
        t2sIdleStatus = { text: t('tercume.status.avatarReady'), stateName: 'ready' };
        setStatus(t2sIdleStatus.text, t2sIdleStatus.stateName, 't2s');
      } catch (err) {
        t2sIdleStatus = { text: `${t('tercume.status.avatarFailed')} (${err.message})`, stateName: 'error' };
        setStatus(t2sIdleStatus.text, t2sIdleStatus.stateName, 't2s');
      }
    });
  }

  // Lüğət vərəqi (AzSLD və JestDili.az söz işarələri); axtarışla süzülür. Eyni yazılışın bir neçə işarəsi
  // (bel: alət / bədən) məna qeydi ilə göstərilir və toxunanda məhz o işarə oynayır
  const collator = new Intl.Collator('az');
  const entries = [...state.lexicon.entries].sort((a, b) => collator.compare(a.text, b.text));
  const byId = new Map(entries.map((w) => [w.id, w]));
  const same = new Map();
  for (const w of entries) same.set(w.text, [...(same.get(w.text) ?? []), w.id]);
  const label = (w) => {
    const ids = same.get(w.text);
    if (ids.length < 2) return w.text;
    return w.sense ? `${w.text} (${w.sense})` : `${w.text} ${ids.indexOf(w.id) + 1}`;
  };
  $('#word-count').textContent = t('tercume.status.wordCount', { n: entries.length });
  const renderWords = (q = '') => {
    const needle = azLower(q.trim());
    const list = entries.filter((w) => !needle || w.text.includes(needle));
    $('#words').innerHTML = list.length
      ? list.map((w) => `<button data-w="${escapeHtml(w.text)}" data-id="${escapeHtml(w.id)}" class="${w.hands === 'LR' ? 'two' : ''}"
          title="${escapeHtml(w.text)}${w.sense ? ` · ${escapeHtml(w.sense)}` : ''}${w.hands === 'LR' ? ` · ${escapeHtml(t('tercume.status.twoHanded'))}` : ''}">${escapeHtml(label(w))}</button>`).join('')
      : `<span class="muted">${escapeHtml(t('tercume.status.notFound'))}</span>`;
  };
  renderWords();
  $('#word-search').addEventListener('input', (e) => renderWords(e.target.value));
  $('#words').addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-w]');
    if (!btn) return;
    t2s.play(btn.dataset.w, byId.get(btn.dataset.id));
    $$('#words button').forEach((b) => b.classList.toggle('is-on', b === btn));
  });

  // Əlifba vərəqi
  const letters = Object.keys(state.poses.letters);
  $('#abc-count').textContent = t('tercume.status.letterCount', { n: letters.length });
  $('#abc').innerHTML = letters.map((ch) => {
    const dyn = state.poses.letters[ch].type === 'dynamic';
    return `<button data-ch="${escapeHtml(ch)}" class="${dyn ? 'dyn' : ''}"
      title="${escapeHtml(ch)}${dyn ? ` - ${escapeHtml(t('tercume.status.dynamicLetter'))}` : ''}">${escapeHtml(ch)}</button>`;
  }).join('');
  $('#abc').addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-ch]');
    if (!btn) return;
    t2s.showLetter(btn.dataset.ch);
    $$('.abc button').forEach((b) => b.classList.toggle('is-on', b === btn));
    $$('#words button').forEach((b) => b.classList.remove('is-on')); // clear words
  });

  refresh();
}

/* ══════════════════ MODEL sənədi ══════════════════ */
function renderAbout() {
  if (state.model.kind === 'letters') return renderAboutLetters();
  const e = state.eval;
  const m = state.model.meta;
  const arch = m.layers.map((l) => `${l.in}→${l.out}`).join(' · ');

  const bars = (rows) => `<div class="bars">${rows.map(([ch, acc]) =>
    `<div><span>${escapeHtml(ch)}</span><u><i style="width:${acc.toFixed(1)}%"></i></u><em>${acc.toFixed(0)}%</em></div>`
  ).join('')}</div>`;

  $('#doc').innerHTML = `
    <h2>${t('tercume.about.heading')}</h2>
    <p>${t('tercume.about.nonLetters.intro')}</p>

    <h3>${t('tercume.about.architectureHeading')}</h3>
    <table>
      <tr><th>${t('tercume.about.th.input')}</th><td class="n">${t('tercume.about.nonLetters.inputValue', { frames: m.input.frames, features: m.input.features, flat: m.input.flat })}</td></tr>
      <tr><th>${t('tercume.about.nonLetters.th.layers')}</th><td class="n">${arch}</td></tr>
      <tr><th>${t('tercume.about.nonLetters.th.params')}</th><td class="n">${m.params.toLocaleString(locale)}</td></tr>
      <tr><th>${t('tercume.about.th.classes')}</th><td class="n">${t('tercume.about.nonLetters.classesValue', { n: m.labels.length })}</td></tr>
      <tr><th>${t('tercume.about.nonLetters.th.features')}</th><td class="n">${t('tercume.about.nonLetters.featuresValue')}</td></tr>
    </table>

    <h3>${t('tercume.about.decoderHeading')}</h3>
    <p>${t('tercume.about.decoderBody', { words: state.decoder.words.length.toLocaleString(locale), prefixes: state.decoder.prefixes.size.toLocaleString(locale) })}</p>

    <h3>${t('tercume.about.whenWrittenHeading')}</h3>
    <p>${t('tercume.about.nonLetters.whenWrittenBody')}</p>

    ${e ? `
    <h3>${t('tercume.about.azsldHeading')}</h3>
    <p>${t('tercume.about.azsldIntro', { images: e.images.toLocaleString(locale), detectRate: (e.detectRate * 100).toFixed(1) })}</p>
    <table>
      <tr><th>${t('tercume.about.azsldTable.group')}</th><th>${t('tercume.about.azsldTable.letter')}</th><th>${t('tercume.about.azsldTable.meanTop1')}</th><th>${t('tercume.about.azsldTable.median')}</th><th>${t('tercume.about.azsldTable.ge90')}</th></tr>
      <tr><td>${t('tercume.about.azsldTable.static')}</td><td class="n">${e.static.n}</td><td class="n">${e.static.mean.toFixed(1)}%</td>
          <td class="n">${e.static.median.toFixed(1)}%</td><td class="n">${e.static.ge90}/${e.static.n}</td></tr>
      <tr><td>${t('tercume.about.azsldTable.dynamic')}</td><td class="n">${e.dynamic.n}</td><td class="n">${e.dynamic.mean.toFixed(1)}%</td>
          <td class="n">${e.dynamic.median.toFixed(1)}%</td><td class="n">${e.dynamic.ge90}/${e.dynamic.n}</td></tr>
    </table>
    <div class="note"><b>${t('tercume.about.azsldNoteHeading')}</b>
    ${t('tercume.about.azsldNote1')}
    ${t('tercume.about.azsldNote2')}</div>
    <h3>${t('tercume.about.staticAccuracyHeading')}</h3>
    ${bars(e.perLetter.filter((r) => r.type === 'static').map((r) => [r.letter, r.acc]))}
    ` : ''}

    <h3>${t('tercume.about.whyBrowserHeading')}</h3>
    <p>${t('tercume.about.whyBrowserBody')}</p>`;
}

/** Landmark hərf modeli (notebook 07): manifest-dəki ölçmələr, köhnə modelin eyni nümunələrdəki nəticəsi ilə. */
function renderAboutLetters() {
  const m = state.model.manifest;
  const test = m.metrics?.test ?? {};
  const old = m.old_model?.test ?? {};
  const pct = (v) => (v === undefined ? '-' : `${(v * 100).toFixed(1)}%`);
  const SOURCES = [['sentence', t('tercume.about.letters.sources.sentence')], ['side', t('tercume.about.letters.sources.side')],
    ['photo', t('tercume.about.letters.sources.photo')], ['clip', t('tercume.about.letters.sources.clip')],
    ['all', t('tercume.about.letters.sources.all')]];
  const rows = SOURCES.filter(([k]) => test[k]).map(([k, name]) =>
    `<tr><td>${name}</td><td class="n">${test[k].n}</td><td class="n">${pct(test[k].top1)}</td>`
    + `<td class="n">${pct(test[k].macro_f1)}</td><td class="n">${pct(old[k]?.top1)}</td></tr>`).join('');
  const per = Object.entries(m.test_per_letter ?? {}).sort((a, b) => a[0].localeCompare(b[0], 'az'));
  const bars = per.length ? `<h3>${t('tercume.about.letters.perLetterHeading')}</h3><div class="bars">${per.map(([ch, r]) =>
    `<div><span>${escapeHtml(ch)}</span><u><i style="width:${(r.recall * 100).toFixed(1)}%"></i></u>`
    + `<em>${(r.recall * 100).toFixed(0)}%</em></div>`).join('')}</div>` : '';
  $('#doc').innerHTML = `
    <h2>${t('tercume.about.heading')}</h2>
    <p>${t('tercume.about.letters.intro', { window: m.input.window, seconds: (m.input.window / 30).toFixed(1), dynamicList: m.dynamic.join(', '), noneLabel: escapeHtml(m.none_label) })}</p>

    <h3>${t('tercume.about.architectureHeading')}</h3>
    <table>
      <tr><th>${t('tercume.about.letters.th.model')}</th><td class="n">${t('tercume.about.letters.modelValue', { params: (m.params / 1e6).toFixed(2) })}</td></tr>
      <tr><th>${t('tercume.about.th.input')}</th><td class="n">${t('tercume.about.letters.inputValue', { nFeatures: m.input.n_features, groups: m.input.groups.join(', ') })}</td></tr>
      <tr><th>${t('tercume.about.th.classes')}</th><td class="n">${t('tercume.about.letters.classesValue', { n: m.labels.length })}</td></tr>
      <tr><th>${t('tercume.about.letters.th.training')}</th><td class="n">${t('tercume.about.letters.trainingValue')}</td></tr>
    </table>

    <h3>${t('tercume.about.letters.testHeading')}</h3>
    <table>
      <tr><th>${t('tercume.about.letters.testTable.source')}</th><th>${t('tercume.about.letters.testTable.sample')}</th><th>${t('tercume.about.letters.testTable.top1')}</th><th>${t('tercume.about.letters.testTable.macroF1')}</th><th>${t('tercume.about.letters.testTable.oldModel')}</th></tr>
      ${rows}
    </table>
    <div class="note">${t('tercume.about.letters.testNote')}</div>
    ${bars}

    <h3>${t('tercume.about.whenWrittenHeading')}</h3>
    <p>${t('tercume.about.letters.whenWrittenBody', { stride: state.s2t?.gate?.stride ?? 2, confidence: Math.round(state.s2t?.gate?.minConfidence * 100 || 50), dwell: state.s2t?.gate?.dwell ?? 3 })}</p>

    <h3>${t('tercume.about.decoderHeading')}</h3>
    <p>${t('tercume.about.decoderBody', { words: state.decoder.words.length.toLocaleString(locale), prefixes: state.decoder.prefixes.size.toLocaleString(locale) })}</p>

    <h3>${t('tercume.about.whyBrowserHeading')}</h3>
    <p>${t('tercume.about.whyBrowserBody')}</p>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

boot();
