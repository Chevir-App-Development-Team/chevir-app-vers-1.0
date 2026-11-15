/**
 * /danis/ — işarə dilində danışmaq: səhifə açılır, kamera işə düşür, insan danışır, mətn yazılır.
 *
 * Cümlə modeli (s2s.js, davamlı tanıma) hər ~0.27 s-dən bir işləyir. İşarə göstərilən anda söz boz (hələ yekun
 * deyil, SentenceStream.peek) görünür, ~1 s sonra təsdiqlənib ağ olur; mətn hər dəfə bütün sözlərdən qurulur
 * (gloss2text.js). Sözlər arasında əlləri endirmək lazım deyil; iki saniyə yeni söz gəlməyəndə cümlə söhbətə
 * keçir. Video və nöqtələr cihazda qalır.
 */
import '../styles/base.css'
import '../styles/sections/navbar.css'
import '../tercume/tercume.css'
import '../tercume/tool.css'
import './danis.css'
import { navbarMarkup, initScrollShadow } from '../sections/navbar.js'
import { SignToSentence } from '../tercume/s2s.js'
import { HAND_CONNECTIONS } from '../tercume/hands.js'

document.querySelector('#site-nav').outerHTML = navbarMarkup({ page: 'tercume' })
initScrollShadow(document.querySelector('[data-navbar]'))

const $ = (s) => document.querySelector(s)
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c])
const video = $('#cam'), canvas = $('#overlay'), ctx = canvas.getContext('2d')
const history = []
let noBody = 0, noHands = 0

function status(text, stage = '') {
  $('#status-text').textContent = text
  $('#status').dataset.state = stage
}

function hint(text) {
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
    hint(noBody > 45 ? 'Bədəniniz görünmür: bir az geri çəkilin, belinizdən yuxarı kadrda olsun'
      : noHands > 300 ? 'Danışmaq üçün əllərinizi qaldırın' : '')
  },
  onLive: ({ words, pending, text }) => renderLive(words, pending, text),
  onSentence: ({ words, text }) => {
    history.push({ text: text || words.map((w) => w.label).join(' '), signs: words.map((w) => w.label) })
    renderHistory()
    renderLive([], [], '')
  },
})

function renderLive(words, pending, text) {
  const all = [...words, ...pending]
  $('#live-text').innerHTML = all.length
    ? `${esc(text)}${pending.length ? '<span class="danis-cursor"></span>' : ''}`
    : '<span class="muted">İşarə ilə danışın - mətn burada görünəcək</span>'
  $('#live-signs').innerHTML = all.map((w) => `<span class="${w.pending ? 'pending' : 'final'}">${esc(w.label)}</span>`).join(' ')
}

function renderHistory() {
  $('#count').textContent = String(history.length)
  $('#btn-copy').disabled = !history.length
  $('#history').innerHTML = history.length
    ? history.slice().reverse().map((h) => `<li><span class="t">${esc(h.text)}</span>`
      + `<span class="g">${esc(h.signs.join(' '))}</span></li>`).join('')
    : '<li class="muted">Hələ cümlə yoxdur</li>'
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
})
$('#btn-copy').addEventListener('click', () => {
  navigator.clipboard?.writeText(history.map((h) => h.text).join('\n'))
    .then(() => status('Köçürüldü', 'running'), () => status('Köçürmək olmadı', 'error'))
})
$('#show-signs').addEventListener('change', (e) => document.body.classList.toggle('hide-signs', !e.target.checked))

// ?video=<ünvan>: kamerasız sınaq və nümayiş, video kadr-kadr eyni yoldan keçir
const file = new URLSearchParams(location.search).get('video')
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
