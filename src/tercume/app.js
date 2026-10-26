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
import './layout.js';

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

// Söz rejimində halqanın yazısı (bax s2w.js)
const WORD_STATE_LABEL = { rest: 'istirahət', signing: 'jest', recognizing: 'tanınır' };

// Kamera halqasının altındakı vəziyyət yazısı (bax s2t.js — hərf yazma şərtləri)
const GATE_LABEL = {
  idle: '-',
  'wait-hand': 'əl yoxdur',
  moving: 'hərəkət',
  unsure: 'qeyri-müəyyən',
  hold: 'saxla',
  typed: 'yazıldı',
};

const state = {
  model: null, decoder: null, poses: null, eval: null, lexicon: null,
  s2t: null, s2w: null, kind: 'letters', t2s: null,
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
    const [model, decoder, poses, evalData, lexicon] = await Promise.all([
      loadLetterModel({ assetBase: ASSET_BASE }),
      LexiconDecoder.load(`${ASSET_BASE}lm.json`, `${ASSET_BASE}vocab.json`, `${ASSET_BASE}prefixes.json`),
      loadPoses(`${ASSET_BASE}poses.json`),
      fetch(`${ASSET_BASE}eval.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
      WordLexicon.load(`${WORDS_BASE}index.json`),
    ]);
    Object.assign(state, { model, decoder, poses, eval: evalData, lexicon });
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
      drawOverlay(ctx, canvas, video, landmarks ? [landmarks] : []);
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

  // Söz rejimi: model, onnxruntime və üç izləyici yalnız ilk dəfə seçiləndə yüklənir
  let s2wLoading = null;
  const ensureS2W = () => {
    s2wLoading ??= (async () => {
      const { SignToWords } = await import('./s2w.js');
      const s2w = new SignToWords({
        video,
        onStatus: ({ message, stage }) => setStatus(message, stage),
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
          if (last) renderPredictions(last.top.map((t) => (t.label === unk ? { ...t, label: 'naməlum' } : t)));
          $('#signs-info').textContent = added.length
            ? `+${added.length} · ${Math.round(ms)} ms` : signs.length ? `${signs.length} söz` : '-';
          $('#pred-quality').textContent = !last ? '-'
            : last.unsure ? `əmin deyil · ${Math.round(last.p * 100)}%` : `əmin ${Math.round(last.p * 100)}%`;
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
        onStatus: ({ message, stage }) => setStatus(message, stage),
        onFrame: ({ frame }) => {
          const hands = [0, 1].map((s) => slotLandmarks(frame.hand, s)).filter(Boolean);
          drawOverlay(ctx, canvas, video, hands);
          state.skelS2T?.setPose(hands[0] ?? null);
          $('#ring-label').textContent = hands.length ? 'jest' : 'istirahət';
        },
        onWords: ({ words, added, sentence, ms }) => {
          renderSentence(words, added.length, sentence);
          const last = added[added.length - 1];
          if (last) renderPredictions([{ label: last.label, p: last.p }]);
          $('#sentence-info').textContent = `+${added.length} · ${Math.round(ms)} ms`;
        },
        onSentence: ({ words, match }) => {
          const li = document.createElement('li');
          li.textContent = words.map((w) => w.label).join(' ') + (match ? `  ≈  ${match.text}` : '');
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
    if (state.kind === 'words') ensureS2W().catch((err) => setStatus('Söz modeli yüklənmədi: ' + err.message, 'error'));
    if (state.kind === 'sentence') ensureS2S().catch((err) => setStatus('Cümlə modeli yüklənmədi: ' + err.message, 'error'));
  });

  $('#btn-start').addEventListener('click', async () => {
    $('#btn-start').disabled = true;
    try {
      const runner = state.kind === 'words' ? await ensureS2W() : state.kind === 'sentence' ? await ensureS2S() : s2t;
      await runner.start();
      $('#btn-stop').disabled = false;
    } catch (err) {
      setStatus('Kamera açılmadı: ' + err.message, 'error');
      $('#btn-start').disabled = false;
    }
  });
  $('#btn-stop').addEventListener('click', () => {
    current()?.stop();
    $('#btn-start').disabled = false;
    $('#btn-stop').disabled = true;
    state.skelS2T?.clearPose();
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

function renderSigns(signs, nNew = 0) {
  $('#signs').innerHTML = signs.length
    ? signs.map((s, i) => {
      const unk = state.s2w?.rec.manifest.unknown_label;
      const alt = s.top.map((t) => `${t.label === unk ? 'naməlum' : t.label} ${(t.p * 100).toFixed(0)}%`).join(' · ');
      const cls = ['s', i >= signs.length - nNew && 'new', s.unsure && 'unsure'].filter(Boolean).join(' ');
      const label = s.unknown ? 'naməlum' : s.unsure ? `${s.label}?` : s.label;
      return `<span class="${cls}" title="${escapeHtml(alt)}">${escapeHtml(label)}<em>${Math.round(s.p * 100)}%</em></span>`;
    }).join('')
    : '<span class="muted">boş</span>';
}

function renderSentence(words, nNew, match) {
  $('#sentence-words').innerHTML = words.length
    ? words.map((w, i) => `<span class="s${i >= words.length - nNew ? ' new' : ''}">${escapeHtml(w.label)}<em>${Math.round(w.p * 100)}%</em></span>`).join('')
    : '<span class="muted">boş</span>';
  const m = $('#sentence-match');
  m.hidden = !match;
  if (match) m.textContent = `AzSLD-də ən yaxın cümlə: ${match.text}`;
}

function renderDecode({ candidates, best, verified, suggestions, letters }) {
  $('#letters').innerHTML = letters?.length
    ? letters.map((l) => `<span class="l">${escapeHtml(l === ' ' ? '␣' : l)}</span>`).join('')
    : '<span class="muted">boş</span>';

  const dec = $('#decoded');
  if (!best) { dec.innerHTML = '<span class="muted">-</span>'; }
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
         VRM avatarı yüklənmədi - cyber skeleton işləməyə davam edir.</div>`);
  }
  state.t2s.targets = [state.avatar, state.skelT2S].filter(Boolean);
}

function initT2S() {
  const t2s = new TextToSign({
    poses: state.poses,
    lexicon: state.lexicon,
    targets: [],
    onLetter: ({ kind, char, index }) => {
      $('#current-letter').textContent = char === ' ' ? '␣' : char;
      $('#current-letter').classList.toggle('is-word', kind === 'word');
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
    const steps = t2s.plan(text);
    $('#spelled').innerHTML = steps.length
      ? steps.map((s) => `<i${s.kind === 'word' ? ' class="w"' : ''}>${escapeHtml(s.kind === 'space' ? '␣' : s.char)}</i>`).join('')
      : '<span class="muted">-</span>';
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
  $('#speed').addEventListener('change', (e) => {
    t2s.opts.speed = +e.target.value;    // 0.25× — dörd dəfə yavaş, 2× — iki dəfə sürətli
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

  // Lüğət vərəqi (AzSLD söz işarələri); axtarışla süzülür
  const collator = new Intl.Collator('az');
  const entries = [...state.lexicon.entries].sort((a, b) => collator.compare(a.text, b.text));
  $('#word-count').textContent = `${entries.length} söz`;
  const renderWords = (q = '') => {
    const needle = azLower(q.trim());
    const list = entries.filter((w) => !needle || w.text.includes(needle));
    $('#words').innerHTML = list.length
      ? list.map((w) => `<button data-w="${escapeHtml(w.text)}" class="${w.hands === 'LR' ? 'two' : ''}"
          title="${escapeHtml(w.text)}${w.hands === 'LR' ? ' · iki əlli' : ''}">${escapeHtml(w.text)}</button>`).join('')
      : '<span class="muted">tapılmadı</span>';
  };
  renderWords();
  $('#word-search').addEventListener('input', (e) => renderWords(e.target.value));
  $('#words').addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-w]');
    if (!btn) return;
    t2s.play(btn.dataset.w);
    $$('#words button').forEach((b) => b.classList.toggle('is-on', b === btn));
  });

  // Əlifba vərəqi
  const letters = Object.keys(state.poses.letters);
  $('#abc-count').textContent = `${letters.length} hərf`;
  $('#abc').innerHTML = letters.map((ch) => {
    const dyn = state.poses.letters[ch].type === 'dynamic';
    return `<button data-ch="${escapeHtml(ch)}" class="${dyn ? 'dyn' : ''}"
      title="${escapeHtml(ch)}${dyn ? ' - dinamik' : ''}">${escapeHtml(ch)}</button>`;
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
    <h2>Model və ölçmələr</h2>
    <p>Sistem barmaq əlifbası modelini (<code>fingerspelling_33.h5</code>) olduğu kimi işlədir. Keras çəkiləri
    xam <code>float32</code> kimi çıxarılıb, irəli keçid JS-də 4 matmul ilə hesablanır -
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
    hərəkət ölçüsü kiçik olmalı (sabit poza 0.00-0.04, keçid 0.06-0.11), model ən azı
    <b>45%</b> əmin olmalı və eyni hərf ardıcıl 3 proqnozda təsdiqlənməlidir. Bu şərtlər olmadan
    əlin kadra girdiyi an boş kadrlarla birlikdə dinamik hərf kimi oxunurdu - məsələn
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
    (2) Bu şəkillər modelin təlim datası ola bilər - yəni rəqəmlər <b>optimistdir</b>,
    kənarlaşdırılmış test dəsti deyil.</div>
    <h3>Statik hərflər üzrə dəqiqlik</h3>
    ${bars(e.perLetter.filter((r) => r.type === 'static').map((r) => [r.letter, r.acc]))}
    ` : ''}

    <h3>Niyə hər şey brauzerdə</h3>
    <p>Server yoxdur: video cihazdan çıxmır, gecikmə minimaldır və sistem statik fayl kimi
    hər yerdə host oluna bilər. Əlçatanlıq məhsulu üçün məxfilik təsadüfi seçim deyil -
    jest dili videosu istifadəçinin üzünü və kimliyini daşıyır.</p>`;
}

/** Landmark hərf modeli (notebook 07): manifest-dəki ölçmələr, köhnə modelin eyni nümunələrdəki nəticəsi ilə. */
function renderAboutLetters() {
  const m = state.model.manifest;
  const test = m.metrics?.test ?? {};
  const old = m.old_model?.test ?? {};
  const pct = (v) => (v === undefined ? '-' : `${(v * 100).toFixed(1)}%`);
  const SOURCES = [['sentence', 'AzSLD cümlələrindəki hərflər, ön kamera'], ['side', 'eyni hərflər, yan kamera'],
    ['photo', 'Fingerspelling şəkilləri (statik hərflər)'], ['clip', 'Fingerspelling klipləri (hərəkətli hərflər)'],
    ['all', 'hamısı']];
  const rows = SOURCES.filter(([k]) => test[k]).map(([k, name]) =>
    `<tr><td>${name}</td><td class="n">${test[k].n}</td><td class="n">${pct(test[k].top1)}</td>`
    + `<td class="n">${pct(test[k].macro_f1)}</td><td class="n">${pct(old[k]?.top1)}</td></tr>`).join('');
  const per = Object.entries(m.test_per_letter ?? {}).sort((a, b) => a[0].localeCompare(b[0], 'az'));
  const bars = per.length ? `<h3>Hərf üzrə test (görmədiyi insanlar)</h3><div class="bars">${per.map(([ch, r]) =>
    `<div><span>${escapeHtml(ch)}</span><u><i style="width:${(r.recall * 100).toFixed(1)}%"></i></u>`
    + `<em>${(r.recall * 100).toFixed(0)}%</em></div>`).join('')}</div>` : '';
  $('#doc').innerHTML = `
    <h2>Model və ölçmələr</h2>
    <p>Barmaq əlifbası modeli MediaPipe əl nöqtələrindən <b>əlin formasını</b> (metrlə, kameradan və əlin
    kadrdakı yerindən asılı deyil) və <b>hərəkətini</b> oxuyur, son ${m.input.window} kadra baxır (30 fps-də
    ${(m.input.window / 30).toFixed(1)} s). Bədən və üz istifadə olunmur. 32 hərfin hamısını, hərəkətli hərfləri
    (${m.dynamic.join(', ')}) də tanıyır və “hərf yoxdur” (<code>${escapeHtml(m.none_label)}</code>) deyə bilir.</p>

    <h3>Memarlıq</h3>
    <table>
      <tr><th>Model</th><td class="n">Conv1D + Transformer, ${(m.params / 1e6).toFixed(2)} M parametr</td></tr>
      <tr><th>Giriş</th><td class="n">${m.input.n_features} xüsusiyyət × kadr (${m.input.groups.join(', ')})</td></tr>
      <tr><th>Sinif</th><td class="n">${m.labels.length} (32 hərf + hərf yoxdur)</td></tr>
      <tr><th>Təlim</th><td class="n">AzSLD: cümlələrdəki hərflər (iki kamera), Fingerspelling şəkilləri və klipləri</td></tr>
    </table>

    <h3>Test: modelin görmədiyi insanlar</h3>
    <table>
      <tr><th>Mənbə</th><th>Nümunə</th><th>Top-1</th><th>Makro-F1</th><th>Köhnə model, top-1</th></tr>
      ${rows}
    </table>
    <div class="note">Köhnə model (20 kadrın xam əl koordinatları, 4 qatlı şəbəkə) eyni test nümunələrində
    ölçülüb. Fingerspelling şəkilləri ona təlimdə tanış ola bilər - həmin sətirdə onun nəticəsi optimistdir.</div>
    ${bars}

    <h3>Hərf nə vaxt yazılır</h3>
    <p>Model hər ${state.s2t?.gate?.stride ?? 2} kadrdan bir proqnoz verir. Hərf əl pəncərənin hər kadrında görünəndə, model ən azı
    <b>${Math.round(state.s2t?.gate?.minConfidence * 100 || 50)}%</b> əmin olanda və eyni hərf ardıcıl
    ${state.s2t?.gate?.dwell ?? 3} proqnozda təsdiqlənəndə yazılır. Statik hərflərdə əl sabit olmalıdır;
    hərəkətli hərflərdə bu şərt yoxdur. Eyni hərfi təkrar yazmaq üçün əli pozadan çıxar və ya başqa hərf göstər.</p>

    <h3>Dekoder</h3>
    <p>Hər proqnozdan ən yaxşı 3 hərf beam-ə düşür. Namizədlər hərf-bigram dil modeli ilə
    çəkilir (leksikondan: ${state.decoder.words.length.toLocaleString('az')} unikal söz,
    ${state.decoder.prefixes.size.toLocaleString('az')} prefiks). Sonda lüğət yoxlaması:
    tam uyğunluq bonus alır, uyğunluq yoxdursa redaktə məsafəsi ilə ən yaxın real sözlər təklif olunur.</p>

    <h3>Niyə hər şey brauzerdə</h3>
    <p>Server yoxdur: video cihazdan çıxmır, gecikmə minimaldır və sistem statik fayl kimi
    hər yerdə host oluna bilər. Əlçatanlıq məhsulu üçün məxfilik təsadüfi seçim deyil -
    jest dili videosu istifadəçinin üzünü və kimliyini daşıyır.</p>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

boot();
