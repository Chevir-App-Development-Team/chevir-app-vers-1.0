/**
 * /nsosyal/ — Chevir-in NSosyal axınına inteqrasiya prototipi (texniki hesabat, bölmə 3.3).
 *
 * Akış 1 (paylaşan): "Jest dili ilə paylaş" ayrıca paneldə açılır (başlıq + bağlama).
 *   Kamera və ya yüklənmiş video → SignToText (barmaq əlifbası) → altyazı. Aşağı
 *   əminlikli sözlər işarələnir; istifadəçi altyazını düzəldib təsdiqləyir, sonra post
 *   video ilə birlikdə axına düşür və altyazısı ilə axtarılır.
 * Akış 2 (oxuyan): hər postun əməliyyat sırasında, bəyən/şərh/paylaş ilə eyni ölçüdə
 *   "Jest dili" düyməsi. Postun altında avatar paneli açılır, mətn yerində qalır və oxunan
 *   hissə vurğulanır; sürət, təkrar və güvən göstəricisi var, hamısı klaviatura ilə işləyir.
 *
 * Ağır hissələr lazım olanda yüklənir: avatar ilk "Jest dili" düyməsində, model və
 * MediaPipe paylaşım paneli açılanda.
 */
import '../styles/base.css'
import '../styles/sections/navbar.css'
import './nsosyal.css'
import { navbarMarkup } from '../sections/navbar.js'
import { ASSET_BASE, WORDS_BASE } from '../tercume/paths.js'
import { WordLexicon, azLower } from '../tercume/words.js'
import { TextToSign, loadPoses } from '../tercume/t2s.js'
import { HAND_CONNECTIONS } from '../tercume/hands.js'

document.querySelector('#site-nav').outerHTML = navbarMarkup({ page: 'nsosyal', light: true })

const $ = (s, root = document) => root.querySelector(s)
const $$ = (s, root = document) => [...root.querySelectorAll(s)]
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c])
const store = {
  get(k) { try { return localStorage.getItem(k) } catch { return null } },
  set(k, v) { try { localStorage.setItem(k, v) } catch { /* gizli pəncərə */ } },
}

/** Nümunə postlar: mətnlər əsasən lüğətdəki sözlərdəndir, bir neçəsi hərf-hərf göstərilir. */
const POSTS = [
  {
    id: 'p-aysel', author: 'Aysel Məmmədova', meta: '12 dəq · Bakı', hue: 332, likes: 48,
    text: 'Sabah səhər hava soyuq olacaq, uşaqlar üçün isti köynək lazım.',
  },
  {
    id: 'p-kamran', author: 'Kamran Hüseynov', meta: '1 saat', hue: 198, likes: 17,
    text: 'Telefon itmiş. Zəhmət olmasa kömək edin!',
  },
  {
    id: 'p-chevir', author: 'Chevir komandası', meta: '3 saat', hue: 24, likes: 112, badge: 'Rəsmi',
    text: 'Salam! Bu gün sərgi var, hamı gəlsin.',
  },
  {
    id: 'p-nermin', author: 'Nərmin Əliyeva', meta: 'dünən', hue: 150, likes: 9,
    text: 'Bu həftə futbol oynamaq istəyirəm. Kim gəlir?',
  },
]

const state = {
  posts: POSTS.map((p) => ({ ...p })),
  lexicon: new WordLexicon(null),
  poses: null,
  query: '',
  nextId: 1,
}

/* ══════════════════ postlar ══════════════════ */

/** Mətnin hissələri və hər birinin necə göstəriləcəyi (güvən göstəricisi üçün). */
function analyse(text) {
  const letters = state.poses?.letters ?? {}
  const segs = state.lexicon.segment(text).filter((s) => s.kind !== 'space').map((s) => {
    if (s.kind === 'word') return { ...s, cls: 'word' }
    const shown = [...s.text].filter((ch) => letters[azLower(ch)]).length
    return { ...s, cls: shown ? 'spell' : 'skip' }
  })
  const count = (c) => segs.filter((s) => s.cls === c).length
  return { segs, word: count('word'), spell: count('spell'), skip: count('skip') }
}

function textMarkup(text) {
  const { segs } = analyse(text)
  let html = ''
  let pos = 0
  for (const s of segs) {
    html += esc(text.slice(pos, s.from))
    html += `<span class="seg seg--${s.cls}" data-from="${s.from}">${esc(text.slice(s.from, s.to))}</span>`
    pos = s.to
  }
  return html + esc(text.slice(pos))
}

const initials = (name) => name.split(/\s+/).map((w) => w[0]).join('').slice(0, 2).toUpperCase()

function postMarkup(p) {
  const signLabel = `Jest dili — ${p.author} postunu avatarla göstər`
  return `
<article class="card post${p.fresh ? ' is-fresh' : ''}" id="${p.id}" tabindex="-1" aria-labelledby="${p.id}-author">
  <header class="post__head">
    <div class="ava" style="--h:${p.hue}" aria-hidden="true">${esc(initials(p.author))}</div>
    <div class="post__who">
      <b id="${p.id}-author">${esc(p.author)}</b>
      <span>${esc(p.meta)}</span>
    </div>
    ${p.badge ? `<span class="post__badge">${esc(p.badge)}</span>` : ''}
    ${p.video ? `<span class="post__badge post__badge--sign">
      <svg viewBox="0 0 24 24" aria-hidden="true"><use href="#i-check" /></svg>Jest dili · altyazı təsdiqlənib</span>` : ''}
  </header>
  ${p.video ? `<video class="post__video${p.mirror ? ' is-mirrored' : ''}" src="${p.video}" controls playsinline preload="metadata" aria-label="Jest dili videosu"></video>` : ''}
  <p class="post__text" data-text="${esc(p.text)}">${textMarkup(p.text)}</p>
  <footer class="post__actions" role="group" aria-label="Post ilə əməliyyatlar">
    <button class="act act--like" type="button" aria-pressed="${p.liked ? 'true' : 'false'}" aria-label="Bəyən, ${p.likes} bəyənmə">
      <svg viewBox="0 0 24 24" aria-hidden="true"><use href="#i-heart" /></svg><span>${p.likes}</span>
    </button>
    <button class="act act--mock" type="button" aria-disabled="true">
      <svg viewBox="0 0 24 24" aria-hidden="true"><use href="#i-comment" /></svg><span>Şərh</span>
    </button>
    <button class="act act--mock" type="button" aria-disabled="true">
      <svg viewBox="0 0 24 24" aria-hidden="true"><use href="#i-share" /></svg><span>Paylaş</span>
    </button>
    <button class="act act--sign" type="button" aria-expanded="false" aria-controls="${p.id}-sign" aria-label="${esc(signLabel)}">
      <svg viewBox="0 0 24 24" aria-hidden="true"><use href="#i-hand" /></svg><span>Jest dili</span>
    </button>
  </footer>
  <div class="post__sign" id="${p.id}-sign"></div>
</article>`
}

function renderPosts() {
  const q = azLower(state.query.trim())
  const list = state.posts.filter((p) => !q || azLower(p.text).includes(q) || azLower(p.author).includes(q))
  // Açıq avatar paneli yenidən qurulan DOM-dan əvvəl çıxarılır, sonra yerinə qaytarılır
  const open = sign.post
  sign.panel?.remove()
  $('#posts').innerHTML = list.length
    ? list.map(postMarkup).join('')
    : `<div class="card empty">“${esc(state.query)}” üzrə post tapılmadı.</div>`
  if (open && list.some((p) => p.id === open.id)) attachPanel(open.id)
  else if (open) closeSign({ focus: false })
  state.posts.forEach((p) => { p.fresh = false })
}

$('#posts').addEventListener('click', (e) => {
  const btn = e.target.closest('.act')
  if (!btn) return
  const article = btn.closest('.post')
  const post = state.posts.find((p) => p.id === article.id)
  if (btn.classList.contains('act--like')) {
    post.liked = !post.liked
    post.likes += post.liked ? 1 : -1
    btn.setAttribute('aria-pressed', String(post.liked))
    btn.setAttribute('aria-label', `Bəyən, ${post.likes} bəyənmə`)
    btn.querySelector('span').textContent = post.likes
  } else if (btn.classList.contains('act--sign')) {
    if (sign.post?.id === post.id) closeSign()
    else openSign(post)
  } else {
    toast('Şərh və paylaşım bu prototipdə aktiv deyil — burada yalnız Chevir qatı işləyir.')
  }
})

$('#search').addEventListener('input', (e) => {
  state.query = e.target.value
  renderPosts()
})

/* Mətnli post */
const composeText = $('#compose-text')
composeText.addEventListener('input', () => { $('#btn-post').disabled = !composeText.value.trim() })
$('#btn-post').addEventListener('click', () => {
  const text = composeText.value.trim()
  if (!text) return
  addPost({ text })
  composeText.value = ''
  $('#btn-post').disabled = true
})

function addPost({ text, video = null, mirror = false }) {
  const post = {
    id: `p-me-${state.nextId++}`, author: 'Siz', meta: 'indicə', hue: 262, likes: 0,
    text, video, mirror, fresh: true,
  }
  state.posts.unshift(post)
  state.query = ''
  $('#search').value = ''
  renderPosts()
  const el = document.getElementById(post.id)
  el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  el.focus({ preventScroll: true })
  return post
}

let toastTimer = 0
function toast(msg) {
  let el = $('.toast')
  if (!el) {
    el = document.createElement('div')
    el.className = 'toast'
    el.setAttribute('role', 'status')
    document.body.append(el)
  }
  el.textContent = msg
  el.classList.add('is-on')
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => el.classList.remove('is-on'), 3200)
}

/* ══════════════════ Akış 2: avatar paneli ══════════════════ */

const sign = {
  panel: null,        // bir nüsxə; açılan postun altına köçürülür
  avatar: null,
  loading: null,
  t2s: null,
  post: null,
  speed: Number(store.get('ns-speed')) || 1,
}

function buildPanel() {
  const panel = $('#sign-panel-tpl').content.firstElementChild.cloneNode(true)
  const play = $('.sign__play', panel)
  play.addEventListener('click', () => {
    const t = sign.t2s
    if (!t) return
    if (!t.playing) playPost()
    else if (t.paused) { t.resume(); setPlayIcon('pause') }
    else { t.pause(); setPlayIcon('play') }
  })
  $('.sign__replay', panel).addEventListener('click', () => playPost())
  $('.sign__close', panel).addEventListener('click', () => closeSign())
  $('.speed', panel).addEventListener('click', (e) => {
    const b = e.target.closest('[data-speed]')
    if (b) setSpeed(Number(b.dataset.speed))
  })
  // Radio qrupu: ox düymələri ilə seçim
  $('.speed', panel).addEventListener('keydown', (e) => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) return
    e.preventDefault()
    const opts = $$('[data-speed]', panel)
    const i = opts.findIndex((b) => Number(b.dataset.speed) === sign.speed)
    const next = opts[(i + (e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 1) + opts.length) % opts.length]
    setSpeed(Number(next.dataset.speed))
    next.focus()
  })
  panel.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeSign() })
  return panel
}

function setSpeed(v) {
  sign.speed = v
  store.set('ns-speed', String(v))
  if (sign.t2s) sign.t2s.opts.speed = v
  $$('[data-speed]', sign.panel).forEach((b) => {
    const on = Number(b.dataset.speed) === v
    b.setAttribute('aria-checked', String(on))
    b.tabIndex = on ? 0 : -1
  })
}

function setPlayIcon(kind) {
  const btn = $('.sign__play', sign.panel)
  $('use', btn).setAttribute('href', kind === 'pause' ? '#i-pause' : '#i-play')
  btn.setAttribute('aria-label', kind === 'pause' ? 'Dayandır' : 'Davam et')
}

function attachPanel(postId) {
  const article = document.getElementById(postId)
  $('.post__sign', article).append(sign.panel)
  $('.act--sign', article).setAttribute('aria-expanded', 'true')
  $('.post__text', article).classList.add('is-signing')
  sign.avatar?.resize()
}

function renderConfidence(post) {
  const a = analyse(post.text)
  const total = a.word + a.spell + a.skip || 1
  const ratio = a.word / total
  const conf = $('.conf', sign.panel)
  conf.dataset.level = ratio >= 0.7 ? 'high' : ratio >= 0.4 ? 'mid' : 'low'
  $('.conf__meter i', conf).style.width = `${Math.round(ratio * 100)}%`
  const parts = [`${a.word}/${total} söz işarəsi ilə`]
  if (a.spell) parts.push(`${a.spell} hərf-hərf`)
  if (a.skip) parts.push(`${a.skip} göstərilmir`)
  $('.conf__text', conf).textContent = parts.join(' · ')
  conf.setAttribute('aria-label', `Güvən: ${a.word} söz real siqnalçının işarəsi ilə, ${a.spell} söz hərf-hərf` +
    (a.skip ? `, ${a.skip} hissə göstərilmir` : ''))
}

async function openSign(post) {
  if (sign.post) closeSign({ focus: false })
  sign.panel ??= buildPanel()
  sign.post = post
  attachPanel(post.id)
  setSpeed(sign.speed)
  renderConfidence(post)
  $('.sign__word', sign.panel).textContent = ''
  $('.sign__kind', sign.panel).textContent = ''
  sign.panel.classList.toggle('is-loading', !sign.avatar)
  const article = document.getElementById(post.id)
  article.scrollIntoView({ block: 'start', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' })
  $('.sign__play', sign.panel).focus({ preventScroll: true })
  try {
    await ensureAvatar()
  } catch (err) {
    console.error(err)
    $('.sign__loading-text', sign.panel).textContent = 'Avatar yüklənmədi: ' + err.message
    return
  }
  if (sign.post !== post) return                  // gözləyərkən bağlandı və ya dəyişdi
  sign.panel.classList.remove('is-loading')
  sign.avatar.paused = false
  sign.avatar.resize()
  playPost()
}

function ensureAvatar() {
  sign.loading ??= (async () => {
    const { VrmAvatar } = await import('../tercume/vrm.js')
    const stage = $('.sign__stage', sign.panel)
    const avatar = new VrmAvatar(stage)
    await avatar.load(`${ASSET_BASE}avatar.vrm`)
    avatar.setHand('right')
    sign.avatar = avatar
    sign.t2s = new TextToSign({
      poses: state.poses, lexicon: state.lexicon, targets: [avatar],
      onLetter: showStep,
      onState: ({ playing }) => { if (!playing) finishPlayback() },
    })
    sign.t2s.opts.speed = sign.speed
  })()
  return sign.loading
}

function playPost() {
  if (!sign.post || !sign.t2s) return
  const text = $('.post__text', document.getElementById(sign.post.id))
  $$('.seg', text).forEach((s) => s.classList.remove('is-now', 'is-done'))
  setPlayIcon('pause')
  sign.t2s.play(sign.post.text)
}

/** Oxunan hissə mətndə vurğulanır, panelin yuxarısında isə nə göstərildiyi yazılır. */
function showStep(step) {
  if (!sign.post) return
  const text = $('.post__text', document.getElementById(sign.post.id))
  if (!text || step.kind === 'space') return
  const segs = $$('.seg', text)
  const from = step.seg?.from ?? step.from
  segs.forEach((s) => {
    const f = Number(s.dataset.from)
    s.classList.toggle('is-now', f === from)
    s.classList.toggle('is-done', f < from)
  })
  const word = $('.sign__word', sign.panel)
  const kind = $('.sign__kind', sign.panel)
  if (step.kind === 'word') {
    word.textContent = step.char
    kind.textContent = 'söz işarəsi'
  } else {
    // Hərf-hərf: bütöv söz, indiki hərf seçilmiş
    const src = sign.post.text.slice(step.seg.from, step.seg.to)
    const k = step.from - step.seg.from
    word.innerHTML = `${esc(src.slice(0, k))}<u>${esc(src.slice(k, k + 1))}</u>${esc(src.slice(k + 1))}`
    kind.textContent = step.kind === 'unknown' ? 'işarəsi yoxdur — buraxılır' : 'hərf-hərf'
  }
}

function finishPlayback() {
  if (!sign.post) return
  const text = $('.post__text', document.getElementById(sign.post.id))
  $$('.seg', text ?? document.createElement('p')).forEach((s) => { s.classList.remove('is-now'); s.classList.add('is-done') })
  setPlayIcon('play')
  $('.sign__play', sign.panel).setAttribute('aria-label', 'Yenidən oynat')
}

function closeSign({ focus = true } = {}) {
  const post = sign.post
  sign.post = null
  sign.t2s?.stop()
  sign.avatar?.clearHandPose()
  if (sign.avatar) sign.avatar.paused = true   // görünməyəndə render etmir
  sign.panel?.remove()
  const article = post && document.getElementById(post.id)
  if (!article) return
  const btn = $('.act--sign', article)
  btn.setAttribute('aria-expanded', 'false')
  const text = $('.post__text', article)
  text.classList.remove('is-signing')
  $$('.seg', text).forEach((s) => s.classList.remove('is-now', 'is-done'))
  if (focus) btn.focus()
}

/* ══════════════════ Akış 1: jest dili ilə paylaşım ══════════════════ */

const share = {
  panel: $('#share-panel'),
  video: $('#share-video'),
  overlay: $('#share-overlay'),
  caption: $('#caption'),
  s2t: null,
  loading: null,
  source: null,        // 'camera' | 'file'
  recorder: null,
  chunks: [],
  fileUrl: null,
  confirmed: false,
}

const GATE_TEXT = {
  'wait-hand': 'Əlini kadra gətir',
  moving: 'Əl hərəkətdədir',
  unsure: 'Aydın deyil — hərfi saxla',
  hold: 'Saxla…',
  typed: 'Yazıldı',
}

$('#btn-sign-share').addEventListener('click', () => (share.panel.hidden ? openShare() : closeShare()))
$('#share-close').addEventListener('click', () => closeShare())
share.panel.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !$('#confirm').hidden) { hideConfirm(); return }
  if (e.key === 'Escape') closeShare()
})

function openShare() {
  share.panel.hidden = false
  $('#btn-sign-share').setAttribute('aria-expanded', 'true')
  $('#btn-sign-share').classList.add('is-on')
  $('#cam-start').focus()
  ensureS2T().catch((err) => setCamState('Model yüklənmədi: ' + err.message, 'error'))
}

function closeShare() {
  stopCapture()
  share.panel.hidden = true
  $('#btn-sign-share').setAttribute('aria-expanded', 'false')
  $('#btn-sign-share').classList.remove('is-on')
  $('#btn-sign-share').focus()
}

function ensureS2T() {
  share.loading ??= (async () => {
    const [{ FingerspellModel }, { LexiconDecoder }, { SignToText }] = await Promise.all([
      import('../tercume/model.js'), import('../tercume/decoder.js'), import('../tercume/s2t.js'),
    ])
    const [model, decoder] = await Promise.all([
      FingerspellModel.load(`${ASSET_BASE}model.json`, `${ASSET_BASE}model.bin`),
      LexiconDecoder.load(`${ASSET_BASE}lm.json`, `${ASSET_BASE}vocab.json`, `${ASSET_BASE}prefixes.json`),
    ])
    share.s2t = new SignToText({
      model, decoder, video: share.video,
      onStatus: ({ stage, message }) => {
        if (stage === 'landmarker') setCamState('MediaPipe yüklənir…')
        if (stage === 'ended') { setCamState('Video bitdi — altyazını yoxla'); showIdle(true) }
      },
      onFrame: ({ landmarks, state: gate, progress, fileProgress }) => {
        drawHand(landmarks)
        const text = GATE_TEXT[gate] ?? ''
        if (fileProgress === undefined) setCamState(text, gate, progress)
        else setCamState(`Video emal olunur · ${Math.round(fileProgress * 100)}%${text ? ` · ${text}` : ''}`, gate, progress)
      },
      onDecode: onDecode,
    })
    return share.s2t
  })()
  return share.loading
}

function setCamState(text, kind = '', progress = 0) {
  const el = $('#cam-state')
  el.hidden = !text
  el.textContent = text
  el.dataset.kind = kind
  el.style.setProperty('--p', String(progress))
}

function showIdle(on) {
  $('#cam-idle').hidden = !on
  $('#cam-stop').hidden = on
}

$('#cam-start').addEventListener('click', async () => {
  $('#cam-start').disabled = true
  try {
    const s2t = await ensureS2T()
    stopCapture()
    share.video.classList.add('is-mirrored')
    share.overlay.classList.add('is-mirrored')
    showIdle(false)
    setCamState('Kamera açılır…')
    await s2t.start()
    share.source = 'camera'
    startRecording(share.video.srcObject)
    setCamState('Əlini kadra gətir', 'wait-hand')
    $('#cam').focus()
  } catch (err) {
    showIdle(true)
    setCamState('Kamera açılmadı: ' + err.message, 'error')
  } finally {
    $('#cam-start').disabled = false
  }
})

$('#file-input').addEventListener('change', async (e) => {
  const file = e.target.files?.[0]
  e.target.value = ''
  if (!file) return
  try {
    const s2t = await ensureS2T()
    stopCapture()
    share.fileUrl = URL.createObjectURL(file)
    share.video.classList.remove('is-mirrored')
    share.overlay.classList.remove('is-mirrored')
    showIdle(false)
    share.source = 'file'
    await s2t.startFile(share.fileUrl)
  } catch (err) {
    showIdle(true)
    setCamState('Video oxunmadı: ' + err.message, 'error')
  }
})

$('#cam-stop').addEventListener('click', () => {
  share.s2t?.finish()
  stopCapture({ keepVideo: true })
  showIdle(true)
  setCamState('Dayandırıldı — altyazını yoxla')
  $('#cam-start').focus()
})

/** Kameranın videosu yazılır ki, paylaşılan postda altyazı ilə birgə görünsün (cihazda qalır). */
function startRecording(stream) {
  share.chunks = []
  share.recorder = null
  if (!stream || typeof MediaRecorder === 'undefined') return
  try {
    const rec = new MediaRecorder(stream)
    rec.ondataavailable = (e) => { if (e.data.size) share.chunks.push(e.data) }
    rec.start(1000)
    share.recorder = rec
  } catch { /* yazma dəstəklənmirsə post videosuz paylaşılır */ }
}

/** Kameranı/videonu dayandırır; keepVideo — yazılmış video paylaşım üçün saxlanır. */
function stopCapture({ keepVideo = false } = {}) {
  if (share.recorder?.state === 'recording') share.recorder.stop()
  share.s2t?.stop()
  drawHand(null)
  if (!keepVideo) {
    share.chunks = []
    if (share.fileUrl) URL.revokeObjectURL(share.fileUrl)
    share.fileUrl = null
    share.source = null
  }
}

function drawHand(landmarks) {
  const c = share.overlay
  const w = c.clientWidth, h = c.clientHeight
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h }
  const ctx = c.getContext('2d')
  ctx.clearRect(0, 0, w, h)
  if (!landmarks) return
  // Video "cover" ilə kəsilir: nöqtələr eyni kəsimlə yerləşdirilir
  const vw = share.video.videoWidth || w, vh = share.video.videoHeight || h
  const s = Math.max(w / vw, h / vh)
  const ox = (w - vw * s) / 2, oy = (h - vh * s) / 2
  const P = landmarks.map((p) => [ox + p.x * vw * s, oy + p.y * vh * s])
  ctx.lineWidth = 2.5
  ctx.strokeStyle = 'rgba(255, 79, 151, 0.9)'
  ctx.beginPath()
  for (const [a, b] of HAND_CONNECTIONS) { ctx.moveTo(...P[a]); ctx.lineTo(...P[b]) }
  ctx.stroke()
  ctx.fillStyle = '#ffffff'
  for (const [x, y] of P) { ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill() }
}

/* Canlı hərflər və altyazı */
const LOW_CONF = 0.55

function onDecode({ best, verified, letters, final }) {
  if (final) {
    if (best?.text) appendToken(best.text, best.conf ?? 0, !!verified, (letters ?? []).join(''))
    renderLive([], null)
  } else {
    renderLive(letters ?? [], best, verified)
  }
}

function renderLive(letters, best, verified) {
  $('#live-letters').innerHTML = letters.length
    ? letters.map((l) => `<span>${esc(l)}</span>`).join('')
    : '<span class="muted">—</span>'
  $('#live-guess').innerHTML = best?.text
    ? `söz: <b>${esc(best.text)}</b> <span class="${verified ? 'ok' : 'warn'}">${verified ? 'lüğətdə var' : 'lüğətdə yoxdur'}</span>`
    : ''
  $('#word-done').disabled = !letters.length
  $('#word-undo').disabled = !letters.length
}

/**
 * Söz yoxlanmalıdır, əgər: lüğətdə yoxdursa, namizədlər arasında əminlik aşağıdırsa və ya
 * dekoder kameranın oxuduğu hərfləri dəyişibsə — dekoderin öz əminliyi yalnız namizədlər
 * arasında nisbidir, hərflərin düzəldildiyini göstərmir.
 */
function appendToken(text, conf, verified, seen) {
  const corrected = !!seen && seen !== text
  const low = !verified || conf < LOW_CONF || corrected
  const cap = share.caption
  const tok = document.createElement('span')
  tok.className = `tok${low ? ' is-low' : ''}`
  tok.textContent = text
  tok.dataset.conf = String(Math.round(conf * 100))
  if (low) {
    tok.title = corrected
      ? `Kameradan “${seen}” oxundu, lüğətlə “${text}” kimi düzəldildi — yoxlayın`
      : `Əminlik ${Math.round(conf * 100)}%${verified ? '' : ', lüğətdə yoxdur'} — yoxlayın`
  }
  if (cap.textContent.trim()) cap.append(' ')
  cap.append(tok)
  updateCaption()
}

share.caption.addEventListener('input', () => {
  // Düzəldilən aşağı əminlikli söz yoxlanmış sayılır
  const node = getSelection()?.anchorNode
  const tok = (node?.nodeType === 3 ? node.parentElement : node)?.closest?.('.tok')
  if (tok?.classList.contains('is-low')) {
    tok.classList.remove('is-low')
    tok.classList.add('is-checked')
    tok.removeAttribute('title')
  }
  updateCaption()
})
// Aşağı əminlikli sözə klik — bütöv seçilir ki, dərhal yenisi yazılsın
share.caption.addEventListener('click', (e) => {
  const tok = e.target.closest('.tok.is-low')
  if (!tok) return
  const r = document.createRange()
  r.selectNodeContents(tok)
  const sel = getSelection()
  sel.removeAllRanges()
  sel.addRange(r)
})

function updateCaption() {
  const low = $$('.tok.is-low', share.caption).length
  const has = !!share.caption.textContent.trim()
  share.caption.classList.toggle('is-empty', !has)
  $('#cap-publish').disabled = !has
  $('#cap-note').textContent = low
    ? `${low} söz yoxlanmalıdır (altı xətli): əminlik aşağıdır və ya dekoder hərfləri düzəldib. Üzərinə klikləyib düzəlt.`
    : has ? 'Altyazını paylaşmadan əvvəl bir daha oxu — istədiyin yerini düzəldə bilərsən.' : ''
  hideConfirm()
}

$('#word-done').addEventListener('click', () => share.s2t?.finish())
$('#word-undo').addEventListener('click', () => share.s2t?.undo())
$('#cap-clear').addEventListener('click', () => {
  share.caption.textContent = ''
  share.s2t?.clear()
  updateCaption()
})

/* Kamerasız sınaq: kamera sahəsində klaviatura ilə hərf */
$('#cam').addEventListener('keydown', (e) => {
  const s2t = share.s2t
  if (!s2t || e.target !== $('#cam')) return
  if (e.key === 'Enter') { e.preventDefault(); s2t.finish(); return }
  if (e.key === 'Backspace') { e.preventDefault(); s2t.undo(); return }
  const ch = azLower(e.key)
  if (ch.length === 1 && s2t.decoder.letters.includes(ch)) {
    e.preventDefault()
    s2t.pushLetter(ch, 0.86)
  }
})

/* Təsdiq və paylaşım */
$('#cap-publish').addEventListener('click', () => {
  const low = $$('.tok.is-low', share.caption).length
  if (low && !share.confirmed) {
    $('#confirm-text').textContent = `${low} söz hələ yoxlanmayıb. Yenə də paylaşılsın?`
    $('#confirm').hidden = false
    $('#confirm-no').focus()
    return
  }
  publish()
})
$('#confirm-yes').addEventListener('click', () => {
  if ($('#confirm').hidden) return
  share.confirmed = true
  publish()
})
$('#confirm-no').addEventListener('click', () => {
  hideConfirm()
  // İlk yoxlanmamış söz seçilir — dərhal düzəldilə bilər
  share.caption.focus()
  const tok = $('.tok.is-low', share.caption)
  if (tok) {
    const r = document.createRange()
    r.selectNodeContents(tok)
    getSelection().removeAllRanges()
    getSelection().addRange(r)
  }
})
function hideConfirm() {
  $('#confirm').hidden = true
  share.confirmed = false
}

async function publish() {
  if (share.publishing) return             // cüt klik iki post yaratmasın
  share.s2t?.finish()                      // yarımçıq söz də altyazıya düşsün
  const text = share.caption.innerText.replace(/\s+/g, ' ').trim()
  if (!text) return
  share.publishing = true
  $('#cap-publish').disabled = true
  const video = await recordedVideo().finally(() => { share.publishing = false })
  stopCapture({ keepVideo: true })
  share.fileUrl = null                     // artıq postundur, silinmir
  addPost({ text, video, mirror: false })
  share.caption.textContent = ''
  share.s2t?.clear()
  updateCaption()
  showIdle(true)
  setCamState('')
  share.panel.hidden = true
  $('#btn-sign-share').setAttribute('aria-expanded', 'false')
  $('#btn-sign-share').classList.remove('is-on')
  toast('Paylaşıldı — altyazı ilə birlikdə axtarışda görünür.')
}

/** Yazılmış kamera videosu və ya yüklənmiş fayl (yoxdursa null). */
function recordedVideo() {
  if (share.source === 'file') return Promise.resolve(share.fileUrl)
  const rec = share.recorder
  if (!rec) return Promise.resolve(null)
  const done = () => (share.chunks.length ? URL.createObjectURL(new Blob(share.chunks, { type: rec.mimeType || 'video/webm' })) : null)
  if (rec.state !== 'recording') return Promise.resolve(done())
  return new Promise((resolve) => {
    rec.addEventListener('stop', () => resolve(done()), { once: true })
    rec.stop()
  })
}

/* ══════════════════ başlanğıc ══════════════════ */
async function boot() {
  const [lexicon, poses] = await Promise.all([
    WordLexicon.load(`${WORDS_BASE}index.json`),
    loadPoses(`${ASSET_BASE}poses.json`).catch(() => null),
  ])
  state.lexicon = lexicon
  state.poses = poses
  renderPosts()
  updateCaption()
}
renderPosts()
boot()
