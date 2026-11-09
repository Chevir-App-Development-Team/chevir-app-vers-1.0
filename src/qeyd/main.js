/**
 * /qeyd/ — işarələrin yazılışı (roadmap D4): çatışmayan işarələr (rəqəmlər, gündəlik sözlər, ölkələr) və set C
 * cümlələri (V1). Kamera kadrı cihazda MediaPipe ilə emal olunur (tanıma ilə eyni üç model, landmarks.js); hər
 * yazılışda nöqtələr (tanıma müqaviləsinin sahələri + pose_world) və istəsə video IndexedDB-də saxlanır, ZIP kimi
 * endirilir. Heç nə serverə getmir. Razılıq mətni və vaxtı ZIP-ə yazılır.
 *
 * Nöqtələrin faylı (format "chevir-qeyd-1"): hər sahə LandmarkClip.toRaw()-dakı kimi düz massivdir (T × ölçü,
 * NaN → null), `t` — kadrın yazılışın başından vaxtı (ms). features.js preprocess() bunları birbaşa oxuyur.
 */
import '../styles/base.css'
import '../styles/sections/navbar.css'
import '../tercume/tercume.css'
import '../tercume/tool.css'
import { navbarMarkup, initScrollShadow } from '../sections/navbar.js'
import { createLandmarkers, detectFrame, FRAME_SIZES, EXTRA_SIZES } from '../tercume/landmarks.js'
import { download, zipStore } from '../tercume/tool.js'

document.querySelector('#site-nav').outerHTML = navbarMarkup({ page: 'tercume' })
initScrollShadow(document.querySelector('[data-navbar]'))

const $ = (s) => document.querySelector(s)
const BASE = import.meta.env?.BASE_URL ?? '/'
const CONSENT_VERSION = 1
const FIELDS = { ...FRAME_SIZES, ...EXTRA_SIZES }
const MAX_MS = { sign: 15000, sentence: 40000 }

const st = {
  prompts: [], groups: [], group: null, i: 0, code: '', hand: 'right', facing: 'user', keepVideo: true,
  lms: null, stream: null, recorder: null, chunks: [], frames: null, t0: 0, recording: false, counting: false,
  lastTime: -1, clips: [], last: null, lastTs: 0,
}

/* ── IndexedDB: yazılar ── */
const db = await new Promise((resolve, reject) => {
  const req = indexedDB.open('chevir-qeyd', 1)
  req.onupgradeneeded = () => req.result.createObjectStore('clips', { keyPath: 'key' })
  req.onsuccess = () => resolve(req.result)
  req.onerror = () => reject(req.error)
})
const tx = (mode, fn) => new Promise((resolve, reject) => {
  const t = db.transaction('clips', mode)
  const out = fn(t.objectStore('clips'))
  t.oncomplete = () => resolve(out?.result ?? out)
  t.onerror = () => reject(t.error)
})
const putClip = (c) => tx('readwrite', (s) => s.put(c))
const delClip = (key) => tx('readwrite', (s) => s.delete(key))
const allClips = () => tx('readonly', (s) => s.getAll())

/* ── razılıq ── */
async function boot() {
  const data = await fetch(`${BASE}qeyd/prompts.json`).then((r) => r.json())
  st.prompts = data.prompts
  st.groups = [...new Set(st.prompts.map((p) => p.group))]
  const saved = JSON.parse(localStorage.getItem('chevir-qeyd:last') ?? '{}')
  $('#code').value = saved.code ?? ''
  $('#hand').value = saved.hand ?? 'right'
  $('#facing').value = saved.facing ?? 'user'
  const ok = () => { $('#btn-consent').disabled = !($('#agree').checked && /^[\p{L}\d_-]{1,12}$/u.test($('#code').value.trim())) }
  $('#code').addEventListener('input', ok)
  $('#agree').addEventListener('change', ok)
  ok()
  $('#btn-consent').addEventListener('click', start)
}

async function start() {
  Object.assign(st, {
    code: $('#code').value.trim(), hand: $('#hand').value, facing: $('#facing').value, keepVideo: $('#keep-video').checked,
  })
  const consent = {
    code: st.code, version: CONSENT_VERSION, text: $('#consent-text').innerText.trim(), video: st.keepVideo,
    at: new Date().toISOString(),
  }
  localStorage.setItem(`chevir-qeyd:consent:${st.code}`, JSON.stringify(consent))
  localStorage.setItem('chevir-qeyd:last', JSON.stringify({ code: st.code, hand: st.hand, facing: st.facing }))
  $('#who').textContent = `${st.code} · ${st.hand === 'right' ? 'sağ əl' : 'sol əl'}${st.keepVideo ? '' : ' · videosuz'}`
  $('#consent').hidden = true
  $('#rec').hidden = false
  $('#stage').classList.toggle('no-mirror', st.facing === 'environment')
  $('#group').innerHTML = st.groups.map((g) => `<option>${g}</option>`).join('')
  st.group = st.groups[0]
  await refreshClips()
  const firstTodo = st.prompts.findIndex((p) => !st.clips.some((c) => c.promptId === p.id))
  st.group = st.prompts[Math.max(0, firstTodo)].group
  $('#group').value = st.group
  showPrompt(Math.max(0, firstTodo))
}

/* ── kamera və nöqtələr ── */
async function startCamera() {
  $('#btn-cam').disabled = true
  $('#cam-state').textContent = '· MediaPipe yüklənir'
  try {
    st.lms ??= await createLandmarkers({ face: true })
    st.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: st.facing }, audio: false,
    })
    const v = $('#cam')
    v.srcObject = st.stream
    await v.play()
    $('#cam-state').textContent = `· ${v.videoWidth}×${v.videoHeight}`
    $('#btn-rec').disabled = false
    $('#btn-cam').textContent = 'Kamera işləyir'
    loop()
  } catch (err) {
    console.error(err)
    $('#cam-state').textContent = `· xəta: ${err.message}`
    $('#btn-cam').disabled = false
  }
}

function loop() {
  const v = $('#cam')
  const next = () => (v.requestVideoFrameCallback ? v.requestVideoFrameCallback(step) : requestAnimationFrame(step))
  const step = () => {
    if (!st.stream) return
    if (v.readyState >= 2 && v.currentTime !== st.lastTime) {
      st.lastTime = v.currentTime
      const ts = Math.max(Math.round(performance.now()), st.lastTs + 1)
      st.lastTs = ts
      const f = detectFrame(st.lms, v, ts)
      draw(f)
      if (st.recording) {
        st.frames.push({ t: ts - st.t0, f })
        const ms = ts - st.t0
        $('#rec-time').textContent = (ms / 1000).toFixed(1)
        if (ms > MAX_MS[current().kind]) stopRecording()
      }
    }
    next()
  }
  next()
}

function draw(f) {
  const v = $('#cam'), c = $('#overlay')
  if (c.width !== v.videoWidth) { c.width = v.videoWidth; c.height = v.videoHeight }
  const ctx = c.getContext('2d')
  ctx.clearRect(0, 0, c.width, c.height)
  const dot = (x, y, r, color) => {
    if (!Number.isFinite(x)) return
    ctx.fillStyle = color
    ctx.beginPath(); ctx.arc(x * c.width, y * c.height, r, 0, Math.PI * 2); ctx.fill()
  }
  for (let k = 0; k < 9; k++) dot(f.pose[k * 3], f.pose[k * 3 + 1], 5, '#5fd08a')
  for (let s = 0; s < 2; s++) for (let j = 0; j < 21; j++) dot(f.hand[s * 63 + j * 3], f.hand[s * 63 + j * 3 + 1], 3, s ? '#4da0ff' : '#ff6fb1')
}

/* ── yazılış ── */
const current = () => st.prompts[st.i]

async function toggleRecording() {
  if (st.counting) return
  if (st.recording) return stopRecording()
  if (!st.stream) return
  st.counting = true
  $('#btn-rec').textContent = 'Hazırlaşın…'
  for (const n of [3, 2, 1]) {
    $('#countdown').hidden = false
    $('#countdown').textContent = n
    await new Promise((r) => setTimeout(r, 700))
  }
  $('#countdown').hidden = true
  st.counting = false
  st.frames = []
  st.chunks = []
  if (st.keepVideo && window.MediaRecorder) {
    const mime = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'].find((m) => MediaRecorder.isTypeSupported(m))
    st.recorder = new MediaRecorder(st.stream, mime ? { mimeType: mime, videoBitsPerSecond: 2_500_000 } : {})
    st.recorder.ondataavailable = (e) => { if (e.data.size) st.chunks.push(e.data) }
    st.recorder.start(500)
  }
  st.t0 = Math.round(performance.now())
  st.recording = true
  $('#rec-badge').hidden = false
  $('#btn-rec').textContent = 'Dayandır (boşluq)'
  $('#btn-rec').classList.add('is-rec')
}

async function stopRecording() {
  if (!st.recording) return
  st.recording = false
  const wall = Math.round(performance.now()) - st.t0
  $('#rec-badge').hidden = true
  $('#btn-rec').textContent = 'Yaz (boşluq)'
  $('#btn-rec').classList.remove('is-rec')
  let video = null, mime = null
  if (st.recorder) {
    const r = st.recorder
    st.recorder = null
    await new Promise((resolve) => { r.onstop = resolve; r.stop() })
    mime = r.mimeType
    video = new Blob(st.chunks, { type: mime })
  }
  const frames = st.frames
  const p = current()
  const take = 1 + Math.max(0, ...st.clips.filter((c) => c.promptId === p.id).map((c) => c.take))
  const v = $('#cam')
  const T = frames.length
  const data = {}
  for (const [k, n] of Object.entries(FIELDS)) {
    const a = new Float32Array(T * n)
    frames.forEach((fr, t) => a.set(fr.f[k], t * n))
    data[k] = a
  }
  const dur = wall
  const span = T > 1 ? frames.at(-1).t - frames[0].t : 0
  const has = (k, n) => frames.filter((fr) => Number.isFinite(fr.f[k][0]) || (n && Number.isFinite(fr.f[k][n]))).length / Math.max(1, T)
  const clip = {
    key: `${st.code}/${p.id}/${take}`, code: st.code, promptId: p.id, text: p.text, group: p.group, kind: p.kind,
    split: p.split ?? null, take, hand: st.hand, camera: st.facing, created: new Date().toISOString(),
    size: [v.videoWidth, v.videoHeight], fps: span ? +(((T - 1) * 1000) / span).toFixed(2) : 0, durationMs: dur,
    t: Int32Array.from(frames, (fr) => fr.t), data, video, mime, exported: false,
    quality: { pose: +has('pose').toFixed(3), hands: +has('hand', 63).toFixed(3), face: +has('face').toFixed(3) },
  }
  await putClip(clip)
  st.last = clip.key
  await refreshClips()
  showPrompt(st.i)
  const q = clip.quality
  // Nöqtələr cihazda hesablanır: yavaş cihazda kadr az olur. Video saxlanırsa nöqtələr sonra videodan yenidən çıxarılır
  const slow = clip.fps < 15 ? ` Cihaz yavaşdır (${clip.fps} kadr/s)${st.keepVideo ? ': nöqtələr sonra videodan çıxarılacaq.' : ': videonu saxlamaq daha yaxşıdır.'}` : ''
  const warn = (q.pose < 0.8 ? 'Bədən kadrda az görünür: geri çəkilin.' : q.hands < 0.3 ? 'Əllər az görünür.' : 'Yaxşıdır.') + slow
  $('#take-info').textContent = `Yazı ${take}: ${(dur / 1000).toFixed(1)} s, ${T} kadr; bədən ${Math.round(q.pose * 100)} %, əllər ${Math.round(q.hands * 100)} %, üz ${Math.round(q.face * 100)} %. ${warn}`
  $('#take-info').classList.toggle('bad', warn !== 'Yaxşıdır.' || Boolean(slow))
}

/* ── tapşırıqlar ── */
function showPrompt(i) {
  st.i = i
  const p = current()
  const takes = st.clips.filter((c) => c.promptId === p.id)
  $('#prompt-group').textContent = p.group
  const inGroup = st.prompts.filter((x) => x.group === p.group)
  $('#prompt-pos').textContent = `${inGroup.indexOf(p) + 1}/${inGroup.length}`
  $('#prompt-text').textContent = p.text
  $('#prompt-note').textContent = p.note ?? ''
  $('#prompt-how').textContent = p.kind === 'sentence'
    ? 'Cümləni öz təbii işarə dilinizdə göstərin, mətni sözbəsöz izləməyin. Başda və sonda əllər aşağıda.'
    : 'İşarəni bir dəfə göstərin. Başda və sonda əllər aşağıda.'
  $('#take-info').textContent = takes.length ? `${takes.length} yazı var.` : ''
  $('#take-info').classList.remove('bad')
  $('#btn-del').disabled = !takes.length
  $('#btn-prev').disabled = i === 0
  $('#btn-next').disabled = i === st.prompts.length - 1
  if (st.group !== p.group) { st.group = p.group; $('#group').value = p.group }
  renderList()
}

function renderList() {
  const list = st.prompts.map((p, k) => [p, k]).filter(([p]) => p.group === st.group)
  const n = (p) => st.clips.filter((c) => c.promptId === p.id).length
  $('#items').innerHTML = list.map(([p, k]) => `<button data-k="${k}" class="${n(p) ? 'done' : ''}${k === st.i ? ' is-on' : ''}"
    title="${p.text.replace(/"/g, '&quot;')}">${p.kind === 'sentence' ? p.id : p.text}${n(p) > 1 ? ` ×${n(p)}` : ''}</button>`).join('')
  const done = st.prompts.filter((p) => n(p)).length
  $('#done-count').textContent = `${done}/${st.prompts.length}`
}

async function refreshClips() {
  st.clips = (await allClips()).filter((c) => c.code === st.code)
  const bytes = st.clips.reduce((s, c) => s + (c.video?.size ?? 0) + c.t.length * 562 * 4, 0)
  const fresh = st.clips.filter((c) => !c.exported).length
  $('#store-info').textContent = `${st.clips.length} yazı · ${(bytes / 1e6).toFixed(1)} MB · ${fresh} yeni`
}

/* ── endirmə ── */
const round = (x) => (Number.isFinite(x) ? Math.round(x * 1e5) / 1e5 : null)

async function exportZip(onlyNew) {
  const clips = st.clips.filter((c) => !onlyNew || !c.exported)
  if (!clips.length) return
  const consent = localStorage.getItem(`chevir-qeyd:consent:${st.code}`)
  const files = [{ name: 'consent.json', data: consent ?? '{}' }]
  for (const c of clips) {
    const base = `clips/${c.promptId}_t${c.take}`
    const json = {
      format: 'chevir-qeyd-1', key: c.key, code: c.code, prompt: c.promptId, text: c.text, group: c.group, kind: c.kind,
      split: c.split, take: c.take, hand: c.hand, camera: c.camera, created: c.created, size: c.size, fps: c.fps,
      durationMs: c.durationMs, quality: c.quality, T: c.t.length, fields: FIELDS, t: [...c.t],
    }
    for (const k of Object.keys(FIELDS)) json[k] = Array.from(c.data[k], round)
    files.push({ name: `${base}.json`, data: JSON.stringify(json) })
    if (c.video) files.push({ name: `${base}.${c.mime?.includes('mp4') ? 'mp4' : 'webm'}`, data: new Uint8Array(await c.video.arrayBuffer()) })
  }
  const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, '')
  download(`chevir_qeyd_${st.code}_${stamp}.zip`, zipStore(files))
  for (const c of clips) await putClip({ ...c, exported: true })
  await refreshClips()
}

async function clearExported() {
  const done = st.clips.filter((c) => c.exported)
  if (!done.length) return
  if (!confirm(`Endirilmiş ${done.length} yazı bu cihazdan silinsin? ZIP faylı komandaya verilibsə, silmək olar.`)) return
  for (const c of done) await delClip(c.key)
  await refreshClips()
  showPrompt(st.i)
}

/* ── düymələr ── */
$('#btn-cam').addEventListener('click', startCamera)
$('#btn-rec').addEventListener('click', toggleRecording)
$('#btn-del').addEventListener('click', async () => {
  const takes = st.clips.filter((c) => c.promptId === current().id).sort((a, b) => b.take - a.take)
  if (!takes.length) return
  await delClip(takes[0].key)
  await refreshClips()
  showPrompt(st.i)
})
$('#btn-prev').addEventListener('click', () => showPrompt(st.i - 1))
$('#btn-next').addEventListener('click', () => showPrompt(st.i + 1))
$('#group').addEventListener('change', (e) => {
  st.group = e.target.value
  showPrompt(st.prompts.findIndex((p) => p.group === st.group))
})
$('#items').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-k]')
  if (b) showPrompt(Number(b.dataset.k))
})
$('#btn-export-new').addEventListener('click', () => exportZip(true))
$('#btn-export-all').addEventListener('click', () => exportZip(false))
$('#btn-clear').addEventListener('click', clearExported)
document.addEventListener('keydown', (e) => {
  if (e.code !== 'Space' || $('#rec').hidden || ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return
  e.preventDefault()
  toggleRecording()
})

boot().catch((err) => { console.error(err); $('#who').textContent = `yüklənmədi: ${err.message}` })
