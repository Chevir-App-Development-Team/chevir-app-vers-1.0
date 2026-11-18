/*
 * Yerli rəy paketi (E36, tools/build_native_review.py): modulsuz adi skript, ona görə index.html file:// ilə də açılır.
 * Bəndlər data.js-dədir (window.REVIEW), videolar clips/-də. Cavablar localStorage-də (buraxılış + kod üzrə), bir
 * CSV kimi endirilir: tools/apply_native_review.py xəritələri yeniləyir, V2 sətirlərini tools/review_summary.mjs oxuyur.
 */
(function () {
  'use strict'
  const R = window.REVIEW
  const $ = (s) => document.querySelector(s)
  const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]))
  const COLUMNS = ['release', 'reviewer', 'section', 'item', 'direction', 'source', 'text', 'signs', 'a', 'b', 'answer',
    'comprehension', 'naturalness', 'adequacy', 'fluency', 'correction', 'comment', 'plays', 'seconds', 'saved']
  const SCALES = {
    t2s: [['comprehension', '2. Nə qədər başa düşülür?', '1 - heç başa düşülmür · 5 - tam başa düşülür'],
      ['naturalness', '3. İşarə dilinə nə qədər oxşayır?', '1 - oxşamır · 5 - doğma işarə dili kimidir']],
    s2t: [['adequacy', '1. Mətn göstərilən işarələrin mənasını verir?', '1 - mənası tamam başqadır · 5 - mənası tamamilə düzdür'],
      ['fluency', '2. Mətn düzgün Azərbaycan dilindədir?', '1 - oxunmur · 5 - qüsursuzdur']],
  }
  // [kod, düymənin yazısı, əlavə sahənin sualı (varsa)]
  const OPTIONS = {
    slovo: (it) => [['yes', `Bəli, Azərbaycan işarə dilində də «${it.word}» belədir`],
      ['no', `Xeyr, «${it.word}» başqa işarə ilə göstərilir`],
      ['other', 'Bu işarə başqa söz deməkdir', 'Hansı söz?'], ['unsure', 'Bilmirəm']],
    pair: () => [['yes', 'Bəli, eyni işarədir'], ['no', 'Xeyr, ayrı işarələrdir'], ['unsure', 'Bilmirəm']],
    literal: () => [['yes', 'Doğma işarədir, belə göstərilir'],
      ['no', 'Sözbəsöz tərcümədir', 'Doğma işarə necədir? (məsələn DOĞULMAQ)'], ['unsure', 'Bilmirəm']],
    alt: () => [['yes', 'Bəli, düzdür'], ['no', 'Xeyr', 'Hansı işarə olmalıdır?'], ['unsure', 'Bilmirəm']],
    poly: () => [['yes', 'Bir işarədir'], ['no', 'Bir neçə işarə var', 'Hansı mənada hansı işarə?'], ['unsure', 'Bilmirəm']],
    rule: () => [['yes', 'Bəli, belə göstərilir'], ['no', 'Xeyr', 'Necə göstərilməlidir?'], ['unsure', 'Bilmirəm']],
  }
  const isV2 = (it) => it.kind === 't2s' || it.kind === 's2t'

  const st = { code: '', answers: {}, list: [], i: 0, section: null, started: 0, plays: 0, cur: {} }
  const storeKey = () => `chevir-native:${R.release}:${st.code}`
  const save = () => localStorage.setItem(storeKey(), JSON.stringify(st.answers))
  const isDone = (it) => {
    const a = st.answers[it.id]
    if (!a) return false
    if (it.kind === 't2s') return Boolean(a.comprehension && a.naturalness)
    if (it.kind === 's2t') return Boolean(a.adequacy && a.fluency)
    return Boolean(a.answer)
  }

  // ── başlanğıc ──────────────────────────────────────────────────────────────────────────────────────────
  function home() {
    stopVideos()
    $('#review').hidden = true
    $('#home').hidden = false
    const done = R.items.filter(isDone).length
    $('#total').textContent = `Cəmi ${R.items.length} sual, cavablanıb ${done}.`
    $('#sections').innerHTML = R.sections.filter((s) => s.count).map((s) => {
      const items = R.items.filter((it) => it.section === s.id)
      const d = items.filter(isDone).length
      return `<section class="card sec">
        <h2>${esc(s.title)}</h2><p>${esc(s.intro)}</p>
        <div class="meter"><i style="width:${(100 * d) / items.length}%"></i></div>
        <div class="row"><span class="badge">${d} / ${items.length}</span>
          <button class="btn ${d < items.length ? 'btn-primary' : ''}" data-sec="${s.id}" ${st.code ? '' : 'disabled'}>${d ? (d < items.length ? 'Davam et' : 'Bax') : 'Başla'}</button></div>
      </section>`
    }).join('')
  }

  function setCode(v) {
    st.code = v.trim()
    if (!/^[\p{L}\d_-]{1,12}$/u.test(st.code)) st.code = ''
    if (st.code) {
      localStorage.setItem('chevir-native:code', st.code)
      st.answers = JSON.parse(localStorage.getItem(storeKey()) || '{}')
    }
    home()
  }

  function openSection(id) {
    st.section = id
    st.list = R.items.filter((it) => it.section === id)
    const first = st.list.findIndex((it) => !isDone(it))
    $('#home').hidden = true
    $('#review').hidden = false
    $('#sec-title').textContent = R.sections.find((s) => s.id === id).title
    show(first < 0 ? 0 : first)
  }

  // ── bənd ───────────────────────────────────────────────────────────────────────────────────────────────
  function video(src, { auto = true } = {}) {
    return `<video src="${esc(src)}" muted playsinline preload="auto" ${auto ? 'autoplay loop' : 'controls'}></video>`
  }

  function stopVideos() {
    for (const v of document.querySelectorAll('video')) { v.pause(); v.removeAttribute('src'); v.load() }
  }

  function show(i) {
    stopVideos()
    document.activeElement?.blur()           // gizlənən sahədə qalan fokus növbəti bəndin klavişlərini udmasın
    st.i = i
    const it = st.list[i]
    const a = st.answers[it.id] || {}
    st.cur = { answer: a.answer || '' }
    st.started = performance.now()
    st.plays = a.plays || 0
    $('#progress').textContent = `${i + 1} / ${st.list.length}`

    // videolar
    let media = ''
    if (it.avatar) {
      media += `<div class="group avatar main"><h3>Avatar</h3><div class="vids">${video(it.avatar, { auto: !isV2(it) })}</div></div>`
    }
    for (const g of it.groups || []) {
      media += `<div class="group${g.main ? ' main' : ''}"><h3>${esc(g.label)}</h3><div class="vids">${g.srcs.map((s) => video(s)).join('')}</div>
        ${g.example ? `<p class="ex">Nümunə: ${esc(g.example)}</p>` : ''}</div>`
    }
    $('#media').innerHTML = media || '<p class="muted">Bu sual üçün video yoxdur.</p>'
    $('#media').className = `media ${it.kind}`
    for (const v of document.querySelectorAll('#media video')) {
      v.addEventListener('loadedmetadata', () => { v.playbackRate = Number($('#speed').value) })
      if (isV2(it)) v.addEventListener('play', () => { if (v.currentTime < 0.1) st.plays++ })
    }

    // sual
    let p = ''
    if (it.kind === 'slovo') {
      p = `<p class="word">${esc(it.word)} <small>(rusca: ${esc(it.a)})</small></p>
        <p class="q">Bu rus işarəsi Azərbaycan işarə dilində «${esc(it.word)}» deməkdir?</p>`
    } else if (it.kind === 'pair') {
      p = `<p class="q">«${esc(it.a)}» və «${esc(it.b)}» eyni işarədir?</p>${it.hint ? `<p class="hint-box">${esc(it.hint)}</p>` : ''}`
    } else if (it.kind === 'literal') {
      p = `<p class="word">${esc(it.word)}</p><p class="q">AzSLD iştirakçıları «${esc(it.word)}» ifadəsini belə göstərib. Bu doğma işarədir?</p>`
    } else if (it.kind === 't2s') {
      p = `<label class="field"><span>1. Avatar nə dedi? Öz sözlərinizlə yazın (mətni görmədən)</span>
          <textarea id="v2-answer" rows="2">${esc(a.answer)}</textarea></label>
        <div class="row"><button class="btn btn-ghost" id="btn-reveal" ${a.revealed ? 'hidden' : ''}>Mətni göstər</button></div>
        <div id="reveal" ${a.revealed ? '' : 'hidden'}><p class="reveal-text">Mətn: <b>${esc(it.text)}</b></p>
          ${it.glosses ? `<p class="gloss">Avatar: ${esc(it.glosses)}</p>` : ''}${scales('t2s', a)}</div>`
    } else if (it.kind === 's2t') {
      p = `<p class="hint">Avatar bir iştirakçının göstərdiyi işarələri təkrarlayır. Proqram həmin cümləni kameradan tanıyıb bu mətni yazıb:</p>
        <p class="reveal-text"><b>${esc(it.text)}</b></p>${scales('s2t', a)}
        <label class="field"><span>3. Düzgün mətn necə olmalıdır? (istəyə bağlı)</span>
          <textarea id="v2-correction" rows="2">${esc(a.correction)}</textarea></label>`
    } else {
      p = `<p class="q">${esc(it.question)}</p>`
      if (it.kind === 'rule') p = `<p class="word" style="font-size:1.4rem">${esc(it.text)}</p>` + p
      if (it.glosses) p += `<p class="gloss">Avatar: ${esc(it.glosses)}</p>`
      if (it.hint) p += `<p class="hint-box">${esc(it.hint)}</p>`
      if (it.note) p += `<p class="hint">${esc(it.note)}</p>`
    }
    $('#prompt').innerHTML = p
    $('#btn-reveal')?.addEventListener('click', () => { $('#reveal').hidden = false; $('#btn-reveal').hidden = true })

    // cavab düymələri
    const opts = isV2(it) ? [] : OPTIONS[it.kind](it)
    $('#answers').innerHTML = opts.map(([code, label], k) =>
      `<button class="btn${st.cur.answer === code ? ' on' : ''}" data-code="${code}"><b>${k + 1}</b>${esc(label)}</button>`).join('')
    $('#answers').hidden = !opts.length
    $('#other').value = isV2(it) ? '' : a.correction || ''
    $('#comment').value = a.comment || ''
    otherField()
    $('#btn-prev').disabled = i === 0
    $('#btn-next').textContent = i === st.list.length - 1 ? 'Saxla və bölmələrə qayıt' : 'Saxla və növbəti →'
    strip()
  }

  function scales(dir, a) {
    return SCALES[dir].map(([key, legend, small]) => `<fieldset class="scale"><legend>${legend}</legend><div class="opts">
      ${[1, 2, 3, 4, 5].map((n) => `<label><input type="radio" name="${key}" value="${n}" ${Number(a[key]) === n ? 'checked' : ''}><span>${n}</span></label>`).join('')}
      </div><small>${small}</small></fieldset>`).join('')
  }

  function otherField() {
    const it = st.list[st.i]
    const opt = isV2(it) ? null : OPTIONS[it.kind](it).find(([code]) => code === st.cur.answer)
    $('#other-field').hidden = !(opt && opt[2])
    if (opt && opt[2]) $('#other-label').textContent = opt[2]
  }

  function collect() {
    const it = st.list[st.i]
    if (!it) return
    const prev = st.answers[it.id] || {}
    const a = {
      comment: $('#comment').value.trim(), plays: st.plays,
      seconds: Math.round((prev.seconds || 0) + (performance.now() - st.started) / 1000),
    }
    if (isV2(it)) {
      a.answer = ($('#v2-answer')?.value || '').trim()
      a.correction = ($('#v2-correction')?.value || '').trim()
      a.revealed = it.kind === 't2s' ? !$('#reveal').hidden : undefined
      for (const [key] of SCALES[it.kind]) a[key] = Number(document.querySelector(`input[name="${key}"]:checked`)?.value || 0) || null
    } else {
      a.answer = st.cur.answer
      a.correction = $('#other-field').hidden ? '' : $('#other').value.trim()
    }
    const empty = !a.answer && !a.comment && !a.correction && !a.comprehension && !a.naturalness && !a.adequacy && !a.fluency
    if (empty && !prev.saved) return
    st.answers[it.id] = { ...a, saved: new Date().toISOString() }
    st.started = performance.now()           // vaxt iki dəfə sayılmasın (cavab, sonra növbəti)
    save()
  }

  function next() {
    collect()
    if (st.i < st.list.length - 1) show(st.i + 1)
    else home()
  }

  function choose(code) {
    st.cur.answer = code
    for (const b of document.querySelectorAll('#answers button')) b.classList.toggle('on', b.dataset.code === code)
    otherField()
    if (!$('#other-field').hidden) { $('#other').focus(); return }
    collect()
    strip()
    setTimeout(() => { if (st.list[st.i] && st.cur.answer === code) next() }, 250)
  }

  function strip() {
    $('#strip').innerHTML = st.list.map((it, k) => {
      const a = st.answers[it.id]
      const cls = isDone(it) ? (a.answer === 'unsure' ? 'unsure' : 'done') : ''
      // mətn → işarə bəndinin mətni ipucuda görünməsin (əvvəl avatar, sonra mətn)
      return `<button data-k="${k}" class="${cls}${k === st.i ? ' is-on' : ''}" title="${esc(isV2(it) ? it.rey_id : it.text)}"></button>`
    }).join('')
  }

  // ── CSV ────────────────────────────────────────────────────────────────────────────────────────────────
  const cell = (v) => {
    const s = v === null || v === undefined ? '' : String(v)
    return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }

  function exportCsv() {
    if (!$('#review').hidden) collect()
    if (!st.code) { alert('Əvvəl kodunuzu yazın.'); return }
    const lines = [COLUMNS.join(',')]
    for (const it of R.items) {
      const a = st.answers[it.id]
      if (!a) continue
      const v2 = isV2(it)
      const row = {
        release: R.release, reviewer: st.code, section: it.section, item: v2 ? it.rey_id : it.id,
        direction: v2 ? it.kind : '', source: it.source || '', text: it.text,
        signs: it.kind === 's2t' ? (it.recognized || []).join(' ') : (it.glosses || ''), a: it.a, b: it.b, ...a,
      }
      lines.push(COLUMNS.map((c) => cell(row[c])).join(','))
    }
    const blob = new Blob(['﻿' + lines.join('\r\n') + '\r\n'], { type: 'text/csv;charset=utf-8' })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = `chevir_reyi_${R.release.replace(/\s+/g, '_')}_${st.code}.csv`
    document.body.appendChild(link)
    link.click()
    setTimeout(() => { URL.revokeObjectURL(link.href); link.remove() }, 1000)
  }

  // ── hadisələr ──────────────────────────────────────────────────────────────────────────────────────────
  $('#release').textContent = `buraxılış ${R.release}`
  $('#code').value = localStorage.getItem('chevir-native:code') || ''
  $('#code').addEventListener('input', (e) => setCode(e.target.value))
  $('#sections').addEventListener('click', (e) => { const b = e.target.closest('button[data-sec]'); if (b) openSection(b.dataset.sec) })
  $('#btn-home').addEventListener('click', () => { collect(); home() })
  $('#btn-export').addEventListener('click', exportCsv)
  $('#btn-prev').addEventListener('click', () => { collect(); if (st.i > 0) show(st.i - 1) })
  $('#btn-next').addEventListener('click', next)
  $('#answers').addEventListener('click', (e) => { const b = e.target.closest('button[data-code]'); if (b) choose(b.dataset.code) })
  $('#strip').addEventListener('click', (e) => { const b = e.target.closest('button[data-k]'); if (b) { collect(); show(Number(b.dataset.k)) } })
  $('#speed').addEventListener('change', () => { for (const v of document.querySelectorAll('#media video')) v.playbackRate = Number($('#speed').value) })
  $('#other').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); next() } })
  document.addEventListener('keydown', (e) => {
    if ($('#review').hidden || e.ctrlKey || e.metaKey || e.altKey) return
    if (e.target.matches('input[type=text], textarea')) return
    const it = st.list[st.i]
    const opts = isV2(it) ? [] : OPTIONS[it.kind](it)
    if (/^[1-9]$/.test(e.key) && opts[Number(e.key) - 1]) { e.preventDefault(); choose(opts[Number(e.key) - 1][0]) } else if (e.key === 'Enter' || e.key === 'ArrowRight') { e.preventDefault(); next() } else if (e.key === 'ArrowLeft') { e.preventDefault(); collect(); if (st.i > 0) show(st.i - 1) } else if (e.key === ' ') {
      e.preventDefault()
      for (const v of document.querySelectorAll('#media video')) { v.currentTime = 0; v.play() }
    }
  })
  addEventListener('beforeunload', () => { if (!$('#review').hidden) collect() })
  setCode($('#code').value)
})()
