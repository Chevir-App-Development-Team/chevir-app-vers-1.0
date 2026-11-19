/**
 * Cümlənin alternativləri və düzəlişi (N4, ml/EXPERIMENTS.md E32): cümlə bağlananda deşifrə 2–3 variant verir
 * (continuous.js alternatives, s2s.js onSentence), hər birinin payı ilə. İnsan bir toxunuşla başqa variantı seçir və ya
 * mətni özü düzəldir. Düzəliş yalnız bu cihazda saxlanılır (localStorage); fayl yalnız insan "yüklə" düyməsini basanda
 * yaranır, heç nə avtomatik göndərilmir. Saxlanan: göstərilən və seçilən mətn, işarələrin adları, payları — video və
 * nöqtələr yox.
 *
 * İstifadə: /danis/ (src/danis/main.js) və tərcümə sisteminin cümlə rejimi (app.js).
 */
import './correct.css';

const KEY = 'chevir.duzelisler.v1';
const MAX = 1000;
const EVENT = 'chevir:corrections';

export function loadCorrections() {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) ?? '[]');
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}

/** Düzəliş cihazda saxlanılır → saxlanan düzəlişlərin sayı (yaddaş yoxdursa / doludursa null). */
export function saveCorrection(rec) {
  const all = loadCorrections();
  all.push({ t: new Date().toISOString(), ...rec });
  try {
    localStorage.setItem(KEY, JSON.stringify(all.slice(-MAX)));
  } catch {
    return null;
  }
  dispatchEvent(new CustomEvent(EVENT));
  return Math.min(all.length, MAX);
}

export function clearCorrections() {
  try { localStorage.removeItem(KEY); } catch { /* yaddaş yoxdur */ }
  dispatchEvent(new CustomEvent(EVENT));
}

/** Yalnız insanın düyməsi ilə: düzəlişlər bir JSON faylı kimi yüklənir. */
export function exportCorrections() {
  const all = loadCorrections();
  const body = { app: 'chevir', kind: 'sign-to-text corrections', version: 1, exported: new Date().toISOString(),
    note: 'Saved on this device when a sentence was corrected (E32). Sign names and text only, no video.', corrections: all };
  const blob = new Blob([JSON.stringify(body, null, 1)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `chevir-duzelisler-${new Date().toISOString().slice(0, 10)}.json`;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  return all.length;
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const pct = (x) => `${Math.round(100 * x)} %`;

/**
 * s2s.js onSentence-in cümləsi → göstəriş üçün qeyd: { text, signs, conf, alternatives: [{text, signs, conf}], chosen }.
 * alternatives[0] cümlənin özüdür; boş mətnli variant göstərilmir.
 */
export function sentenceEntry({ words, text, alternatives }) {
  const signs = words.map((w) => w.label);
  const alts = (alternatives?.length ? alternatives : [{ text, labels: signs, conf: null }])
    .map((a) => ({ text: a.text, signs: a.labels, conf: a.conf }))
    .filter((a, i) => i === 0 || a.text);
  alts[0] = { ...alts[0], text: text || signs.join(' '), signs };
  return { text: alts[0].text, signs, conf: alts[0].conf, alternatives: alts, chosen: 0, edited: false };
}

/**
 * Bir cümlənin elementi (li): mətn, payı, işarələr, başqa variantlar (toxunanda seçilir) və "Düzəlt" (mətni yazmaq).
 * Seçim / düzəliş qeydi dəyişir, cihazda saxlanılır və onChange(entry) çağırılır.
 */
export function renderSentence(li, entry, { page, onChange } = {}) {
  li.classList.add('sent');
  // əl ilə düzəldilibsə bütün variantlar qalır (geri qayıtmaq olar), yoxsa seçilən mətndə göstərilir
  const others = entry.alternatives.map((a, i) => ({ ...a, i })).filter((a) => entry.edited || a.i !== entry.chosen);
  li.innerHTML = `<span class="t">${esc(entry.text)}`
    + (entry.conf != null && !entry.edited ? ` <span class="conf" title="Variantlar arasında payı">${pct(entry.conf)}</span>` : '')
    + (entry.edited ? ' <span class="conf">düzəldilib</span>' : '')
    + `</span><span class="g">${esc(entry.signs.join(' '))}</span>`
    + `<span class="alts">${others.map((a) => `<button type="button" class="alt" data-i="${a.i}">`
      + `${esc(a.text)}${a.conf != null ? ` <small>${pct(a.conf)}</small>` : ''}</button>`).join('')}`
    + `<button type="button" class="alt alt-edit">Düzəlt</button></span>`;
  const record = (chosen, via) => {
    saveCorrection({ page, via, shown: { text: entry.text, signs: entry.signs, conf: entry.conf },
      chosen, alternatives: entry.alternatives });
  };
  li.querySelectorAll('.alt[data-i]').forEach((b) => b.addEventListener('click', () => {
    const a = entry.alternatives[Number(b.dataset.i)];
    record({ text: a.text, signs: a.signs, conf: a.conf }, 'alternative');
    Object.assign(entry, { text: a.text, signs: a.signs, conf: a.conf, chosen: Number(b.dataset.i), edited: false });
    renderSentence(li, entry, { page, onChange });
    onChange?.(entry);
  }));
  li.querySelector('.alt-edit').addEventListener('click', () => {
    const t = li.querySelector('.t');
    const input = document.createElement('input');
    input.className = 'edit';
    input.value = entry.text;
    input.setAttribute('aria-label', 'Düzgün mətn');
    t.replaceWith(input);
    input.focus();
    input.select();
    let done = false;
    const finish = (save) => {
      if (done) return;
      done = true;
      const v = input.value.trim();
      if (save && v && v !== entry.text) {
        record({ text: v, signs: entry.signs }, 'edit');
        Object.assign(entry, { text: v, edited: true });
        onChange?.(entry);
      }
      renderSentence(li, entry, { page, onChange });
    };
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') finish(true);
      else if (e.key === 'Escape') finish(false);
    });
    input.addEventListener('blur', () => finish(true));
  });
  return li;
}

/**
 * Düzəlişlərin zolağı: neçəsi cihazda saxlanılıb, "Yüklə" (JSON faylı) və "Sil". Düzəliş yoxdursa gizlidir.
 */
export function mountCorrectionsBar(el) {
  el.classList.add('corr-bar');
  const draw = () => {
    const n = loadCorrections().length;
    el.hidden = !n;
    el.innerHTML = n ? `<span>Düzəlişlər: <b>${n}</b> · yalnız bu cihazda</span>`
      + '<button type="button" class="btn btn-ghost btn-sm" data-act="export">Yüklə (JSON)</button>'
      + '<button type="button" class="btn btn-ghost btn-sm" data-act="clear">Sil</button>' : '';
    el.querySelector('[data-act="export"]')?.addEventListener('click', exportCorrections);
    el.querySelector('[data-act="clear"]')?.addEventListener('click', () => {
      if (confirm('Bu cihazda saxlanan düzəlişlər silinsin?')) clearCorrections();
    });
  };
  addEventListener(EVENT, draw);
  addEventListener('storage', (e) => { if (e.key === KEY) draw(); });
  draw();
  return { refresh: draw };
}
