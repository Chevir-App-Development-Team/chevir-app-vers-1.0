/**
 * /danis/ — işarə dilində danışmaq: səhifə açılır, kamera işə düşür, insan danışır, mətn yazılır.
 *
 * Cümlə modeli (s2s.js, davamlı tanıma) hər ~0.27 s-dən bir işləyir. İşarə göstərilən anda söz boz (hələ yekun
 * deyil, SentenceStream.peek) görünür, ~1 s sonra təsdiqlənib ağ olur; mətn hər dəfə bütün sözlərdən qurulur
 * (gloss2text.js). Bir pəncərəlik yalançı boz sözlər göstərilmir (live.js PENDING_RULE). Sözlər arasında əlləri
 * endirmək lazım deyil; son işarədən ~3 s sonra cümlə söhbətə keçir (live.js SENTENCE_GAP), altyazının altındakı
 * xətt bunu göstərir. Söhbətdəki hər cümlənin altında 2–3 başqa variant (işarə dil modelinin alternativləri, E32): bir
 * toxunuşla seçilir və ya "Düzəlt" ilə yazılır; düzəliş yalnız bu cihazda saxlanılır, faylı insan özü yükləyir
 * (correct.js). Video və nöqtələr cihazda qalır. ?debug=1 — ölçmə (E35); ?nolm=1 — dil modelsiz deşifrə (müqayisə).
 */
import '../styles/base.css'
import '../styles/sections/navbar.css'
import '../tercume/tercume.css'
import '../tercume/tool.css'
import './danis.css'
import { navbarMarkup, initScrollShadow } from '../sections/navbar.js'
import { SignToSentence } from '../tercume/s2s.js'
import { HAND_CONNECTIONS } from '../tercume/hands.js'
import { mountCorrectionsBar, renderSentence, sentenceEntry } from '../tercume/correct.js'

document.querySelector('#site-nav').outerHTML = navbarMarkup({ page: 'tercume' })
initScrollShadow(document.querySelector('[data-navbar]'))

const $ = (s) => document.querySelector(s)
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c])
const video = $('#cam'), canvas = $('#overlay'), ctx = canvas.getContext('2d')
const history = []
let live = { words: [], text: '' }
let noBody = 0, noHands = 0
const params = new URLSearchParams(location.search)
// ?debug=1: ölçmə — konsolda hər sözün boz/ağ anı ([danis] sətirləri), kadrda fps, MediaPipe və model vaxtı
const debug = params.has('debug')
const trace = []
/** ?debug=1: konsola, dev serverdə həm də serverin konsoluna (telefonla sınaqda oxumaq üçün) */
function log(line) {
  console.log(`[danis] ${line}`)
  if (import.meta.env.DEV) fetch('/__danis', { method: 'POST', body: JSON.stringify({ line }), keepalive: true }).catch(() => {})
}

function status(text, stage = '') {
  $('#status-text').textContent = text
  $('#status').dataset.state = stage
}

let lastHint = '', handsUp = false
function hint(text) {
  if (debug && text !== lastHint) log(`göstəriş: ${text || '(yox)'}`)
  lastHint = text
  const h = $('#hint')
  h.hidden = !text
  if (text) h.textContent = text
}

const s2s = new SignToSentence({
  video,
  onStatus: ({ message, stage }) => status(stage === 'running' ? 'Danışın' : message, stage),
  onFrame: ({ frame }) => {
    const hands = [0, 1].map((s) => slot(frame.hand, s)).filter(Boolean)
    draw(hands)
    noBody = Number.isFinite(frame.pose[0]) ? 0 : noBody + 1
    noHands = hands.length ? 0 : noHands + 1
    if (debug && !!hands.length !== handsUp) {
      handsUp = !!hands.length
      log(handsUp ? `əllər kadrda (${hands.length})` : 'əl yoxdur')
    }
    hint(noBody > 45 ? 'Bədəniniz görünmür: bir az geri çəkilin, belinizdən yuxarı kadrda olsun'
      : noHands > 300 ? 'Danışmaq üçün əllərinizi qaldırın' : '')
  },
  onLive: ({ words, pending, text }) => renderLive(words, pending, text),
  onPause: ({ progress }) => { $('#closing').style.transform = `scaleX(${progress})` },
  onTrace: debug ? onTrace : undefined,
  useLM: !params.has('nolm'),
  onSentence: ({ words, text, alternatives }) => {
    history.push(sentenceEntry({ words, text, alternatives }))
    if (debug && alternatives) log(`alternativlər: ${alternatives.map((a) => `"${a.text}" ${Math.round(100 * a.conf)} %`).join(' · ')}`)
    renderHistory()
    renderLive([], [], '')
    $('#closing').style.transform = 'scaleX(0)'
  },
})

/** ?debug=1: sözün kadrları (başlanğıc/son) gələndən boz və ağ olana qədər keçən vaxt, saniyə. */
function onTrace(e) {
  trace.push(e)
  const s = (ms) => (ms == null || !Number.isFinite(ms) ? '-' : (ms / 1000).toFixed(2))
  if (e.kind === 'word') {
    const grey = e.grey == null ? 'boz göstərilmədi' : `boz: başlanğıcdan +${s(e.grey - e.tStart)}`
    log(`${e.label} p=${e.p.toFixed(2)} ${e.len} kadr, əl ${Math.round(100 * e.hands)} % · ${grey}`
      + ` · ağ: sondan +${s(e.white - e.tEnd)}${e.grey == null ? '' : ` · bozdan ağa ${s(e.white - e.grey)}`}`)
  } else if (e.kind === 'nohands') {
    log(`süzüldü (əl ${Math.round(100 * e.hands)} %): ${e.label} p=${e.p.toFixed(2)}`)
  } else if (e.kind === 'dropped') {
    log(`yalançı boz: ${e.label} (${s(e.gone - e.grey)} s göründü)`)
  } else if (e.kind === 'sentence') {
    const last = trace.filter((x) => x.kind === 'word' && x.sentence === e.sentence).pop()
    log(`cümlə: "${e.text}" · son sözün sonundan +${s(last ? e.t - last.tEnd : null)}`)
  }
}
window.danisTrace = trace

function statsLoop() {
  const el = $('#debug')
  let frames = s2s.stats.camFrames, t = performance.now(), ticks = 0
  setInterval(() => {
    const st = s2s.stats, now = performance.now()
    const fps = ((st.camFrames - frames) * 1000) / (now - t)
    frames = st.camFrames
    t = now
    window.danisStats = { fps, lmMs: st.lmMs, modelMs: st.modelMs, lagMs: st.lagMs }
    el.textContent = `kamera ${fps.toFixed(0)} fps · MediaPipe ${st.lmMs.toFixed(0)} ms · model ${st.modelMs.toFixed(0)} ms`
      + ` · pəncərə gecikməsi ${st.lagMs.toFixed(0)} ms`
    if (s2s.running && ++ticks % 5 === 0) log(el.textContent + ` · ${video.videoWidth}×${video.videoHeight}`)
  }, 1000)
}

function renderLive(words, pending, text) {
  const all = [...words, ...pending]
  live = { words, text }
  $('#btn-copy').disabled = !history.length && !words.length
  $('#live-text').innerHTML = all.length
    ? `${esc(text)}${pending.length ? '<span class="danis-cursor"></span>' : ''}`
    : '<span class="muted">İşarə ilə danışın - mətn burada görünəcək</span>'
  $('#live-signs').innerHTML = all.map((w) => `<span class="${w.pending ? 'pending' : 'final'}">${esc(w.label)}</span>`).join(' ')
}

function renderHistory() {
  $('#count').textContent = String(history.length)
  $('#btn-copy').disabled = !history.length && !live.words.length
  const list = $('#history')
  if (!history.length) {
    list.innerHTML = '<li class="muted">Hələ cümlə yoxdur</li>'
    return
  }
  // hər cümlə: mətn, payı, işarələr və başqa variantlar (toxunanda seçilir, düzəliş cihazda saxlanılır)
  list.replaceChildren(...history.slice().reverse().map((h) => renderSentence(document.createElement('li'), h, {
    page: 'danis',
    onChange: (e) => { if (debug) log(`düzəliş: "${e.text}"`) },
  })))
}

/** landmarks.js kadrının əl slotu → 21 nöqtə, əl yoxdursa null. */
function slot(hand, s) {
  if (!Number.isFinite(hand[s * 63])) return null
  return Array.from({ length: 21 }, (_, j) => ({ x: hand[s * 63 + j * 3], y: hand[s * 63 + j * 3 + 1] }))
}

function draw(hands) {
  const w = video.videoWidth, h = video.videoHeight
  if (!w || !h) return
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h }
  ctx.clearRect(0, 0, w, h)
  ctx.lineWidth = Math.max(2, w / 420)
  ctx.strokeStyle = 'rgba(63,193,232,.85)'
  for (const p of hands) {
    ctx.beginPath()
    for (const [a, b] of HAND_CONNECTIONS) {
      ctx.moveTo(p[a].x * w, p[a].y * h)
      ctx.lineTo(p[b].x * w, p[b].y * h)
    }
    ctx.stroke()
  }
}

async function start() {
  $('#btn-start').hidden = true
  try {
    await s2s.start()
    $('#btn-stop').disabled = false
  } catch (err) {
    const denied = err?.name === 'NotAllowedError' || err?.name === 'SecurityError'
    status(denied ? 'Kameraya icazə verilmədi' : 'Kamera açılmadı: ' + err.message, 'error')
    $('#btn-start').hidden = false
  }
}

$('#btn-start').addEventListener('click', start)
$('#btn-stop').addEventListener('click', () => {
  s2s.stop()
  $('#btn-stop').disabled = true
  $('#btn-start').hidden = false
  $('#btn-start').textContent = 'Davam et'
  hint('')
})
$('#btn-clear').addEventListener('click', () => {
  s2s.clear()
  history.length = 0
  renderHistory()
  $('#closing').style.transform = 'scaleX(0)'
  if (debug) log('təmizləndi')
})
// söhbət və hələ söhbətə keçməmiş cari cümlə (yekun sözləri); status bir az sonra əvvəlki halına qayıdır
$('#btn-copy').addEventListener('click', () => {
  const lines = history.map((h) => h.text)
  if (live.words.length) lines.push(live.text)
  const back = [$('#status-text').textContent, $('#status').dataset.state]
  const say = (text, stage) => {
    status(text, stage)
    setTimeout(() => status(...back), 1500)
  }
  if (debug) log(`köçürülür: ${lines.length} sətir`)
  if (!navigator.clipboard) return say('Köçürmək olmadı', 'error')
  navigator.clipboard.writeText(lines.join('\n')).then(() => say('Köçürüldü', back[1]), () => say('Köçürmək olmadı', 'error'))
})
mountCorrectionsBar($('#corrections'))
$('#show-signs').addEventListener('change', (e) => document.body.classList.toggle('hide-signs', !e.target.checked))

// ?video=<ünvan>: kamerasız sınaq və nümayiş, video kadr-kadr eyni yoldan keçir
const file = params.get('video')
if (debug) {
  document.body.classList.add('debug')
  $('#debug').hidden = false
  statsLoop()
}
async function startFile() {
  await s2s.startFile(file)
  status('Video bitdi', 'ended')
}

// Açılan kimi: model və izləyicilər yüklənir, sonra kamera (icazə soruşula bilər)
s2s.init()
  .then(file ? startFile : start)
  .catch((err) => {
    console.error(err)
    status('Model yüklənmədi: ' + err.message, 'error')
  })
