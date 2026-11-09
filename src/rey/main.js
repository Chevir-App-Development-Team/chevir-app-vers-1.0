/**
 * /rey/ — işarə dili daşıyıcılarının rəyi (roadmap V2, set D).
 *
 * Bəndlər public/rey/items.json-dadır (tools/build_review_items.mjs). Mətn → işarə: avatar cümləni göstərir,
 * rəy verən əvvəl nə başa düşdüyünü yazır, sonra mətni görüb anlaşıqlığı və təbiiliyi qiymətləndirir. İşarə →
 * mətn: avatar iştirakçının işarələrini təkrarlayır, proqramın yazdığı mətnin mənası və dili qiymətləndirilir.
 * Cavablar localStorage-də (buraxılış + kod üzrə) saxlanır, CSV kimi endirilir; heç nə serverə getmir.
 */
import '../styles/base.css'
import '../styles/sections/navbar.css'
import '../tercume/tercume.css'
import '../tercume/tool.css'
import { navbarMarkup, initScrollShadow } from '../sections/navbar.js'
import { ASSET_BASE, WORDS_BASE } from '../tercume/paths.js'
import { WordLexicon } from '../tercume/words.js'
import { TextToSign, loadPoses } from '../tercume/t2s.js'
import { csvLine, download } from '../tercume/tool.js'

document.querySelector('#site-nav').outerHTML = navbarMarkup({ page: 'tercume' })
initScrollShadow(document.querySelector('[data-navbar]'))

const $ = (s) => document.querySelector(s)
const $$ = (s) => [...document.querySelectorAll(s)]
const BASE = import.meta.env?.BASE_URL ?? '/'
const COLUMNS = ['release', 'reviewer', 'item', 'direction', 'source', 'text', 'signs', 'answer', 'comprehension',
  'naturalness', 'adequacy', 'fluency', 'correction', 'comment', 'plays', 'seconds', 'saved']

const st = { items: [], release: '', code: '', i: 0, answers: {}, t2s: null, avatar: null, started: 0, plays: 0 }
const key = () => `chevir-rey:${st.release}:${st.code}`
const saveAll = () => localStorage.setItem(key(), JSON.stringify({ i: st.i, answers: st.answers }))

// 1–5 düymələri
for (const fs of $$('.scale')) {
  fs.querySelector('.scale-row').innerHTML = [1, 2, 3, 4, 5]
    .map((n) => `<label><input type="radio" name="${fs.dataset.key}" value="${n}"><span>${n}</span></label>`).join('')
}

async function boot() {
  const data = await fetch(`${BASE}rey/items.json`).then((r) => r.json())
  st.items = data.items
  st.release = data.release
  $('#release').textContent = `buraxılış ${data.release}`
  const code = $('#code')
  code.value = localStorage.getItem('chevir-rey:code') ?? ''
  const ok = () => { $('#btn-begin').disabled = !/^[\p{L}\d_-]{1,12}$/u.test(code.value.trim()) }
  code.addEventListener('input', ok)
  ok()
  $('#btn-begin').addEventListener('click', begin)
}

async function begin() {
  st.code = $('#code').value.trim()
  localStorage.setItem('chevir-rey:code', st.code)
  const saved = JSON.parse(localStorage.getItem(key()) ?? 'null')
  if (saved) Object.assign(st, { i: Math.min(saved.i, st.items.length - 1), answers: saved.answers })
  $('#start').hidden = true
  $('#review').hidden = false
  renderList()
  show(st.i)
  await bootAvatar()
}

async function bootAvatar() {
  const [poses, lexicon] = await Promise.all([loadPoses(`${ASSET_BASE}poses.json`), WordLexicon.load(`${WORDS_BASE}index.json`)])
  st.t2s = new TextToSign({ poses, lexicon, targets: [] })
  try {
    const { VrmAvatar } = await import('../tercume/vrm.js')
    st.avatar = new VrmAvatar($('#avatar'))
    await st.avatar.load(`${ASSET_BASE}avatar.vrm`)
    st.avatar.setHand('right')
    st.t2s.targets = [st.avatar]
    $('#avatar-state').textContent = '· hazır'
  } catch (err) {
    console.error(err)
    $('#avatar-state').textContent = `· yüklənmədi: ${err.message}`
  }
  $('#btn-play').disabled = !st.avatar
  window.addEventListener('resize', () => st.avatar?.resize())
}

function play() {
  const it = st.items[st.i]
  if (!st.t2s) return
  st.t2s.opts.speed = Number($('#speed').value)
  st.plays++
  if (it.dir === 't2s') st.t2s.play(it.text)
  else st.t2s.playSteps(st.t2s.planSigns(it.signs))
}

function show(i) {
  st.t2s?.stop()
  st.i = i
  const it = st.items[i]
  const a = st.answers[it.id] ?? {}
  st.started = performance.now()
  st.plays = a.plays ?? 0
  const t2s = it.dir === 't2s'
  $('#item-title').textContent = `${it.id} · ${t2s ? 'mətn → işarə' : 'işarə → mətn'}`
  $('#progress').textContent = `${i + 1}/${st.items.length}`
  for (const el of $$('[data-dir]')) el.hidden = el.dataset.dir !== it.dir
  $('#answer').value = a.answer ?? ''
  $('#correction').value = a.correction ?? ''
  $('#comment').value = a.comment ?? ''
  $('#t2s-text').textContent = t2s ? it.text : ''
  $('#s2t-text').textContent = t2s ? '' : it.text
  $('#reveal').hidden = !(t2s && a.revealed)
  $('#btn-reveal').hidden = Boolean(t2s && a.revealed)
  for (const k of ['comprehension', 'naturalness', 'adequacy', 'fluency']) {
    for (const r of $$(`input[name="${k}"]`)) r.checked = String(a[k] ?? '') === r.value
  }
  $('#btn-play').textContent = t2s ? 'Göstər' : 'İşarələri göstər'
  $('#btn-prev').disabled = i === 0
  $('#btn-next').textContent = i === st.items.length - 1 ? 'Saxla' : 'Saxla və növbəti →'
  renderList()
}

function collect() {
  const it = st.items[st.i]
  const prev = st.answers[it.id] ?? {}
  const pick = (k) => Number($(`input[name="${k}"]:checked`)?.value ?? 0) || null
  const a = {
    answer: $('#answer').value.trim(), correction: $('#correction').value.trim(), comment: $('#comment').value.trim(),
    revealed: !$('#reveal').hidden, plays: st.plays,
    seconds: Math.round((prev.seconds ?? 0) + (performance.now() - st.started) / 1000),
  }
  for (const k of ['comprehension', 'naturalness', 'adequacy', 'fluency']) a[k] = pick(k)
  const done = it.dir === 't2s' ? Boolean(a.comprehension && a.naturalness) : Boolean(a.adequacy && a.fluency)
  st.answers[it.id] = { ...a, done, saved: new Date().toISOString() }
  saveAll()
}

function renderList() {
  const done = st.items.filter((it) => st.answers[it.id]?.done).length
  $('#done-count').textContent = `${done}/${st.items.length}`
  $('#items').innerHTML = st.items.map((it, k) => `<button data-k="${k}" class="${st.answers[it.id]?.done ? 'done' : ''}${k === st.i ? ' is-on' : ''}"
    title="${it.dir === 't2s' ? 'mətn → işarə' : 'işarə → mətn'}">${it.id}</button>`).join('')
}

function exportCsv() {
  if (st.items.length) collect()
  const lines = [COLUMNS.join(',')]
  for (const it of st.items) {
    const a = st.answers[it.id]
    if (!a || !(a.done || a.answer || a.correction || a.comment)) continue    // yalnız baxılıb keçilən bənd
    const signs = it.dir === 's2t' ? it.recognized.join(' ') : ''
    lines.push(csvLine([st.release, st.code, it.id, it.dir, it.source, it.text, signs, a.answer, a.comprehension,
      a.naturalness, a.adequacy, a.fluency, a.correction, a.comment, a.plays, a.seconds, a.saved]))
  }
  const name = `chevir_rey_${st.release.replace(/\s+/g, '_')}_${st.code}.csv`
  download(name, new Blob(['﻿' + lines.join('\r\n') + '\r\n'], { type: 'text/csv;charset=utf-8' }))
}

$('#btn-play').addEventListener('click', play)
$('#btn-reveal').addEventListener('click', () => { $('#reveal').hidden = false; $('#btn-reveal').hidden = true })
$('#btn-prev').addEventListener('click', () => { collect(); show(st.i - 1) })
$('#btn-next').addEventListener('click', () => { collect(); if (st.i < st.items.length - 1) show(st.i + 1); else renderList() })
$('#items').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-k]')
  if (b) { collect(); show(Number(b.dataset.k)) }
})
$('#btn-export').addEventListener('click', exportCsv)

boot().catch((err) => { console.error(err); $('#release').textContent = `yüklənmədi: ${err.message}` })
