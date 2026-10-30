/**
 * İşarələr → Azərbaycan mətni (jest → mətn rejiminin son addımı).
 *
 * Tanıyıcı işarələrin adlarını verir (MƏN ATA EV GETMƏK BİLMİR), cümlə isə Azərbaycan dilindədir
 * ("Mənim atam evdən çıxa bilmir"). AzSLD-də işarələrin sırası mətnin sırasıdır (cümlələrin 99,4 %-i),
 * ona görə hər işarə bir sözə çevrilir: kök + qrammatik əlamətlər (hal, mənsubiyyət, zaman, şəxs).
 *
 * Öyrənmə: AzSLD cümlələrində hər işarə mətnin sözü ilə tutuşdurulur (eyni kök, morph.js), sözün
 * əlamətləri sintezlə tapılır (inflect.js-in qurduğu formalardan hansıdır) və kontekstlər üzrə sayılır:
 * işarənin özü (qonşu siniflər, sonrakı işarə), növü (ad / fel) sonrakı işarə ilə — felin istədiyi hal
 * (BAKI YAŞAMAQ → Bakıda) — və qonşu siniflərlə. Görünməmiş işarənin kökü və əlamətləri öz adından
 * çıxır (OLACAQ → ol + gələcək zaman). Uzlaşma qaydadır: fel və xəbər mübtədanın şəxsini alır
 * (MƏN ... YAŞAMAQ → yaşayıram), əvəzlikdən sonrakı ad onun mənsubiyyətindədirsə əvəzlik yiyəlik
 * haldadır (MƏN ATA → mənim atam). Barmaqla yığılan hərflər (D O S T) bir sözdür.
 *
 * Model JSON-dur (tools/build_gloss_text.mjs → public/tercume/assets/gloss_text.json), brauzerdə
 * öyrənmə lazım deyil. Ölçmə: tools/eval_gloss_text.mjs.
 */
import { Lemmatizer, candidates, variants } from './morph.js';
import { analyze, realize, PRONOUNS } from './inflect.js';
import { azLower } from './words.js';

export const textWords = (t) => azLower(t).replace(/[.,!?;"()«»“”]+/g, ' ').split(/\s+/).filter(Boolean);
// Annotasiyada İ çox vaxt ı yerinə yazılıb (ALİNMİR): tutuşdurmada ı = i
const norm = (w) => w.replace(/ı/g, 'i');
// İşarənin adında hərf səhvi (OLMUSAM, DUŞHURSUZ, GUN): kök tapılmasa, diakritiksiz yazılış bir hərfə
// qədər fərqlənə bilər
const fold = (w) => norm(w).replace(/ş/g, 's').replace(/ç/g, 'c').replace(/ğ/g, 'g').replace(/ü/g, 'u').replace(/ö/g, 'o');
function near(a, b) {
  if (a === b) return true;
  if (Math.min(a.length, b.length) < 4 || Math.abs(a.length - b.length) > 1) return false;
  let i = 0;
  while (i < a.length && a[i] === b[i]) i++;
  return a.slice(i + 1) === b.slice(i + 1) || a.slice(i) === b.slice(i + 1) || a.slice(i + 1) === b.slice(i);
}

const SUBJECT = { mən: 1, sən: 2, o: 3, biz: 4, siz: 5, onlar: 6 };
const QWORDS = new Set(['nə', 'nədir', 'hara', 'harada', 'necə', 'niyə', 'kim', 'hansı', 'neçə', 'nə vaxt', 'nəyə', 'haçan']);
const MODAL = new Set(['bilmək', 'bilmir', 'olmaq', 'lazım', 'istəmək']);
const EXIST = new Set(['var', 'yox', 'deyil', 'lazım']);
// Şəxsi mübtədadan gələn zamanlar (məsdər, -a, feli sifət, əmr şəxssizdir)
const FINITE = new Set(['pres', 'past', 'perf', 'ev', 'fut', 'futc', 'aor', 'nec', 'cond']);
const isLetter = (l) => [...l].length === 1 && /\p{L}/u.test(l);
// Barmaqla yığılan hərf: tək hərfli işarə, əvəzlik "O" deyilsə və ya yanında başqa hərf varsa (D O S T)
const spelled = (labels, i) => isLetter(labels[i])
  && (!PRONOUNS.includes(labelText(labels[i])) || isLetter(labels[i - 1] ?? '') || isLetter(labels[i + 1] ?? ''));
const isNumber = (l) => /^\d+$/.test(l);
const labelText = (l) => azLower(l.split('/')[0]);

/** İşarənin sinfi (qonşu kimi): əvəzlik öz şəxsi ilə, fel, ad, sual sözü, rəqəm, hərf. */
function classOf(label, lex) {
  if (!label) return 'B';
  const t = labelText(label);
  if (isLetter(label)) return 'L';
  if (isNumber(label)) return 'N';
  if (SUBJECT[t]) return `P${SUBJECT[t]}`;
  if (QWORDS.has(t)) return 'Q';
  if (EXIST.has(t)) return 'E';
  if (MODAL.has(t)) return 'M';
  return lex?.pos === 'verb' ? 'V' : 'A';
}

// Sayılan kontekstlər: işarənin özü (qonşu siniflərlə, sonrakı işarə ilə, təkcə) və növü — sonrakı
// işarə ilə (felin istədiyi hal: BAKI YAŞAMAQ → Bakıda, EV GETMƏK → evə), qonşu siniflərlə, təkcə
function contexts(labels, i, lexOf, pos) {
  const l = labels[i], next = labels[i + 1] ?? '</s>';
  const pc = classOf(labels[i - 1], lexOf(labels[i - 1])), nc = classOf(labels[i + 1], lexOf(labels[i + 1]));
  return [['lc', `${l}|${pc}>${nc}`], ['ln', `${l}|>${next}`], ['l', l],
    ['cn', `${pos}|>${next}`], ['cc', `${pos}|${pc}>${nc}`], ['cr', `${pos}|>${nc}`], ['c', pos]];
}
// Kontekstlərin çəkisi (dev cümlələrində seçilib: tools/eval_gloss_text.mjs --tune)
export const WEIGHTS = { lc: 8, ln: 4, l: 2, cn: 2, cc: 1, cr: 1, c: 0 };

const fkey = (f) => JSON.stringify(f, Object.keys(f).sort());
const inc = (m, k, n = 1) => { m[k] = (m[k] ?? 0) + n; };

/**
 * Öyrənmə. pairs: [{glosses, text}] — işarələr tanıyıcının birləşmiş adlarındadır; merge: sign_merge.json
 * (işarənin digər adları tutuşdurma üçün); labels — cümlələrdə olmayan, ancaq tanınan işarələr də;
 * text — etiketin düzgün yazılışı (ml/label_text.csv).
 */
export const train = (pairs, merge, labels, text) => fit(prepare(pairs, merge, labels, text));

/**
 * Hər cümlənin tutuşdurulması və sözlərin əlamətləri (ağır hissə, bir dəfə) və hər işarənin öz adının
 * təhlili. Kök açarları yalnız işarələrin adlarından qurulur, mətndən yox: bir cümləni çıxarıb sayan
 * ölçmə (leave-one-out) düzgündür. text: etiketin düzgün yazılışı (ml/label_text.csv: OLMUSAM → olmuşam).
 */
export function prepare(pairs, merge = { merge: {}, spelling: {} }, extra = [], text = {}) {
  const members = new Map();
  const addM = (lab, g) => { if (!members.has(lab)) members.set(lab, new Set([lab])); members.get(lab).add(g); };
  const canon = (g) => { const s = merge.spelling?.[g] ?? g; return merge.merge?.[s] ?? s; };
  for (const [g, grp] of Object.entries(merge.merge ?? {})) addM(grp, g);
  for (const [v, g] of Object.entries(merge.spelling ?? {})) addM(merge.merge?.[g] ?? g, v);
  for (const [g, t] of Object.entries(text)) addM(canon(g), t);
  const textOf = (l) => azLower(text[l] ?? labelText(l));
  const formsOf = (lab) => [...(members.get(lab) ?? [lab])].flatMap((g) => g.split('/'));
  const labels = [...new Set([...pairs.flatMap((p) => p.glosses), ...extra])];
  const lem = new Lemmatizer([...new Set(labels.flatMap(formsOf).flatMap((g) => norm(azLower(g)).split(' ')))]);
  const key = (w) => { const n = norm(w); return lem.key(n) ?? n; };
  // açarlar: işarənin bütün sözləri (tutuşdurma) və son sözü (çoxsözlü işarədə dəyişən söz odur)
  const keysOf = new Map(labels.map((l) => [l, new Set(formsOf(l).flatMap((g) => norm(azLower(g)).split(' ').map(key)))]));
  const headKeys = new Map(labels.map((l) => [l, new Set(formsOf(l).map((g) => key(norm(azLower(g)).split(' ').at(-1))))]));

  const guess = {};
  for (const l of labels) if (!spelled([l], 0) && !isNumber(l)) guess[l] = guessOf(l, headKeys.get(l), lem, textOf(l));
  // xüsusi adlar böyük hərflə yazılır (label_text.csv: BAKI → Bakı, ABŞERON RAYON → Abşeron rayonu)
  for (const l of labels) {
    const t = text[l] ?? [...(members.get(l) ?? [])].map((g) => text[g]).find(Boolean);
    const caps = t?.split(' ').map((w) => w[0] !== azLower(w[0]));
    if (caps?.some(Boolean) && guess[l]) guess[l].proper = caps;
  }

  const sentences = pairs.map((p) => {
    const ws = textWords(p.text);
    // rəqəm mətndə şəkilçi ilə yazılır: 6-dan, 2-ci
    const at = align(p.glosses, ws, (l, w) => (isNumber(l) ? w === l || w.startsWith(`${l}-`)
      : keysOf.get(l)?.has(key(w)) || near(fold(w), fold(textOf(l)))));
    const info = p.glosses.map((l, i) => {
      const surface = at[i] === null ? '' : ws[at[i]];
      if (spelled(p.glosses, i)) return { surface: null };           // hərflər işarənin sözü deyil
      if (!surface || isNumber(l)) return { surface };
      if (!headKeys.get(l).has(key(surface))) return { surface };     // çoxsözlü işarənin başqa sözü
      return { surface, ...rootOf(l, surface, headKeys.get(l), lem) };
    });
    return { glosses: p.glosses, words: ws, at, info };
  });
  return { sentences, guess };
}

/** Sayma: prepare()-in cümlələrindən (keep(i) — hansıları) model (JSON). */
export function fit({ sentences, guess }, keep = () => true) {
  const model = { lex: {}, forms: {}, feats: {}, subj: {}, posCtx: {}, guess };
  const used = sentences.filter((_, i) => keep(i));
  const opts = {};
  for (const { glosses, info } of used) {
    glosses.forEach((l, i) => {
      if (info[i].surface === null) return;
      inc(model.forms[l] ??= {}, info[i].surface);
      if (info[i].feats) inc(opts[l] ??= {}, `${info[i].pos}:${info[i].root}`);
    });
  }
  for (const [l, o] of Object.entries(opts)) {
    const list = Object.entries(o).map(([k, n]) => { const [pos, root] = k.split(/:(.*)/s); return { pos, root, n }; })
      .sort((a, b) => b.n - a.n);
    // hər növün (ad, fel) ən çox işlənən kökü
    const options = list.filter((x, k) => list.findIndex((y) => y.pos === x.pos) === k);
    model.lex[l] = { pos: options[0].pos, root: options[0].root, ...(options.length > 1 && { options }) };
  }
  const lexOf = (l) => model.lex[l] ?? guess[l];
  for (const { glosses, info } of used) {
    // mübtəda: adlıq haldakı ilk şəxs əvəzliyi (mənim atam — "mənim" yiyəlikdir, mübtəda deyil)
    const subj = subjectOf(glosses, (i) => info[i].feats && !info[i].feats.case);
    glosses.forEach((l, i) => {
      const a = info[i];
      if (!a.feats) return;
      const nc = classOf(glosses[i + 1], lexOf(glosses[i + 1]));
      inc(model.posCtx[`${l}|>${nc}`] ??= {}, a.pos);
      const f = { ...a.feats };
      // şəxs mübtədadan gəlir: felin şəxsi ayrıca sayılır, adın xəbərliyi yalnız var/yox kimi
      if (a.pos === 'verb') {
        if (FINITE.has(f.tam)) inc(model.subj[subj] ??= {}, String(f.pers));
        delete f.pers;
      }
      if (f.cop) f.cop = true;
      // mənsubiyyət əvvəlki əvəzliyə uyğundursa (mənim atam, sizin ananız) şəxsi yox, uyğunluğu sayılır
      if (f.poss && f.poss === SUBJECT[labelText(glosses[i - 1] ?? '')]) f.poss = 'P';
      const k = fkey(f);
      for (const [name, c] of contexts(glosses, i, lexOf, a.pos)) inc(model.feats[`${name}:${c}`] ??= {}, k);
    });
  }
  return model;
}

/** Monoton tutuşdurma: işarə i ↔ söz j (eyni kök), ən çox uyğunluq. Qaytarır: işarə → söz nömrəsi | null. */
function align(labels, ws, ok) {
  const n = labels.length, m = ws.length;
  const d = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = 1; i <= n; i++) for (let j = 1; j <= m; j++) {
    d[i][j] = Math.max(d[i - 1][j], d[i][j - 1], ok(labels[i - 1], ws[j - 1]) ? d[i - 1][j - 1] + 1 : -1);
  }
  const at = new Array(n).fill(null);
  for (let i = n, j = m; i > 0 && j > 0;) {
    if (ok(labels[i - 1], ws[j - 1]) && d[i][j] === d[i - 1][j - 1] + 1) { at[i - 1] = j - 1; i--; j--; }
    else if (d[i][j] === d[i - 1][j]) i--;
    else j--;
  }
  return at;
}

/**
 * Sözün kökü (yazılışı sözün özündən) və əlamətləri. Kök işarənin öz açarıdır (ata, yaşa, bil-), açarı
 * ona gedən söz (atam) deyil; ən qısa uyğun kök, fel kökü əvvəl (bilərəm → bil + qeyri-qəti, "bilər" yox).
 */
function rootOf(label, surface, keys, lem) {
  const t = labelText(label).split(' ').at(-1);
  if (PRONOUNS.includes(t)) {
    const a = analyze(surface, t, 'noun');
    return a ? { root: t, pos: 'noun', feats: a.feats } : {};
  }
  const roots = new Set([surface]);
  for (const c of candidates(surface)) for (const v of [...variants(c.stem, true), ...variants(c.stem)]) roots.add(v);
  for (const r of [...roots].sort((a, b) => a.length - b.length)) {
    if (!keys?.has(norm(r)) && !keys?.has(`${norm(r)}-`)) continue;
    for (const p of lem.verbs.has(norm(r)) ? ['verb', 'noun'] : ['noun', 'verb']) {
      const a = analyze(surface, r, p);
      if (a) return { root: r, pos: p, feats: a.feats };
    }
  }
  return {};
}

/**
 * İşarənin öz adından kök və əlamətlər (cümlələrdə görünməyən işarə üçün): OLACAQ → ol, gələcək zaman;
 * BİLMİR → bil, inkar; YAŞAMAQ → yaşa (məsdər lüğət formasıdır, əlaməti yoxdur). Çoxsözlü işarənin
 * son sözü dəyişir, qalanı `prefix`-dir (ANADAN OLMAQ → anadan + ol).
 */
function guessOf(label, keys, lem, t = labelText(label)) {
  const toks = t.split(' ');
  const head = toks.at(-1);
  const prefix = toks.slice(0, -1).join(' ');
  const base = { ...(prefix && { prefix }) };
  if (PRONOUNS.includes(head)) return { pos: 'noun', root: head, ...base };
  const a = rootOf(label, head, keys, lem);
  if (a.root) {
    const inherent = a.pos === 'verb' && a.feats.tam !== 'inf' ? { tam: a.feats.tam, neg: a.feats.neg } : null;
    return { pos: a.pos, root: a.root, ...base, ...(inherent && { inherent }) };
  }
  const inf = /^(.+?)(maq|mək)$/.exec(head);
  if (inf) return { pos: 'verb', root: inf[1], ...base };
  return { pos: 'noun', root: head, ...base };
}

/**
 * Cümlənin mübtədası: adlıq haldakı ilk şəxs əvəzliyi (MƏN → 1), yoxdursa 3.
 * `nominative(i)` — i-ci işarə adlıq haldadırmı (öyrənmədə mətndən).
 */
function subjectOf(glosses, nominative = () => true) {
  for (let i = 0; i < glosses.length; i++) {
    const s = SUBJECT[labelText(glosses[i])];
    if (s && nominative(i)) return s;
  }
  return 3;
}

export class GlossToText {
  constructor(model, { weights = WEIGHTS } = {}) { this.m = model; this.w = weights; }

  #lexOf = (l) => this.m.lex[l] ?? this.m.guess[l];

  #best(table) {
    let best = null, n = 0;
    for (const [k, c] of Object.entries(table ?? {})) if (c > n) { best = k; n = c; }
    return best;
  }

  /** İşarənin kökü və növü: iki növü varsa (İŞLƏMƏK: iş / işlə), sonrakı işarənin sinfinə görə. */
  #lex(labels, i) {
    const l = labels[i];
    const g = this.m.guess[l] ?? { pos: 'noun', root: labelText(l) };
    const lx = this.m.lex[l];
    if (!lx) return { ...g, seen: false };
    let pick = lx;
    if (lx.options) {
      const nc = classOf(labels[i + 1], this.#lexOf(labels[i + 1]));
      const pos = this.#best(this.m.posCtx[`${l}|>${nc}`]);
      pick = lx.options.find((o) => o.pos === pos) ?? lx;
    }
    return { pos: pick.pos, root: pick.root, prefix: g.prefix, proper: g.proper, seen: true };
  }

  /** Ən ehtimallı əlamətlər: kontekstlərin paylanmalarının çəkili cəmi (az sayılan kontekst zəifdir). */
  #feats(labels, i, pos) {
    const score = new Map();
    for (const [name, c] of contexts(labels, i, this.#lexOf, pos)) {
      const table = this.m.feats[`${name}:${c}`];
      const tot = Object.values(table ?? {}).reduce((a, b) => a + b, 0);
      if (!tot) continue;
      const w = this.w[name] * (tot / (tot + 1));
      for (const [k, n] of Object.entries(table)) {
        const f = JSON.parse(k);
        if ((pos === 'verb') !== ('tam' in f)) continue;      // başqa növün əlamətləri
        score.set(k, (score.get(k) ?? 0) + w * (n / tot));
      }
    }
    let best = null, s = -1;
    for (const [k, v] of score) if (v > s) { best = k; s = v; }
    return best ? JSON.parse(best) : pos === 'verb' ? { tam: 'pres' } : {};
  }

  /**
   * İşarələr → {text, words: [{labels, word, via}]}. via: 'gen' — kök + əlamətlər, 'form' — işarənin
   * öyrənilmiş sözü (rəqəm), 'letters' — hərflər bir sözdür, 'drop' — mətndə sözü olmayan işarə.
   */
  generate(labels) {
    const items = [];
    for (let i = 0; i < labels.length; i++) {
      const l = labels[i];
      let j = i;
      while (j < labels.length && spelled(labels, j)) j++;
      if (j > i) {                                         // D O S T → dost (tək "O" isə əvəzlikdir)
        items.push({ labels: labels.slice(i, j), word: azLower(labels.slice(i, j).join('')), via: 'letters' });
        i = j - 1;
        continue;
      }
      const forms = this.m.forms[l];
      if (isNumber(l)) {                                   // 6 → 6-dan (öyrənilmiş yazılış), yoxsa 6
        const seen = Object.entries(forms ?? {}).filter(([w]) => w).sort((a, b) => b[1] - a[1]);
        items.push({ labels: [l], word: seen[0]?.[0] ?? l, via: 'form' });
        continue;
      }
      const tot = Object.values(forms ?? {}).reduce((a, b) => a + b, 0);
      if (tot && (forms[''] ?? 0) / tot > 0.5) { items.push({ labels: [l], word: '', via: 'drop' }); continue; }
      const lx = this.#lex(labels, i);
      // kökü tapılmayan, ancaq mətndə sözü olan işarə (adında hərf səhvi): öyrənilmiş söz
      const seen = !lx.seen && Object.entries(forms ?? {}).filter(([w]) => w).sort((a, b) => b[1] - a[1])[0];
      if (seen) { items.push({ labels: [l], word: seen[0], via: 'form' }); continue; }
      const f = this.#feats(labels, i, lx.pos);
      // görünməmiş işarənin adındakı zaman və inkar (OLACAQ, BİLMİR) qorunur
      const inh = !lx.seen && this.m.guess[l]?.inherent;
      items.push({ labels: [l], lx, f: inh ? { ...f, ...inh } : f, via: 'gen' });
    }
    // Yiyə: şəxs əvəzliyindən sonrakı ad onun mənsubiyyətindədirsə (mənim atam), əvəzlik yiyəlik haldadır;
    // ad mənsubiyyət almırsa (yer adı: Mən Bakıda), əvəzlik yiyəlikdə qalmır. Qərarı ad verir: qohumluq,
    // bədən sözləri mənsubiyyətlə işlənir, yer adları yox. Cümlənin əvvəlindəki əvəzlikdən sonra ad başqa
    // haldadırsa (evimə, çantamı), ad mübtəda deyil, əvəzlik mübtəda qalır: Mən evimə gedirəm
    for (let k = 0; k + 1 < items.length; k++) {
      const a = items[k], b = items[k + 1];
      const p = a.lx && SUBJECT[a.lx.root];
      if (!p || !b.lx || b.lx.pos !== 'noun' || SUBJECT[b.lx.root]) continue;
      if (b.f.poss === p || b.f.poss === 'P') {
        b.f = { ...b.f, poss: p };
        a.f = k === 0 && b.f.case && b.f.case !== 'gen' ? {} : { case: 'gen' };
        continue;
      }
      if (a.f.case === 'gen') a.f = {};
    }
    for (const it of items) if (it.f?.poss === 'P') { const { poss, ...rest } = it.f; it.f = rest; }  // yiyəsiz
    const at = items.findIndex((it) => it.lx && SUBJECT[it.lx.root] && !it.f.case);
    const subj = at >= 0 ? SUBJECT[items[at].lx.root] : 3;
    for (const it of items) {
      if (it.via !== 'gen') continue;
      const f = { ...it.f };
      if (it.lx.pos === 'verb' && FINITE.has(f.tam)) {
        f.pers = Number(this.#best(this.m.subj[subj]) ?? subj);
        if (f.tam === 'ev' && ![3, 6].includes(f.pers)) f.tam = 'perf';     // gəlib → mən gəlmişəm
      }
      if (f.cop) f.cop = subj === 6 ? 3 : subj;
      const w = realize(it.lx.root, it.lx.pos, f) ?? it.lx.root;
      const word = it.lx.prefix ? `${it.lx.prefix} ${w}` : w;
      it.word = it.lx.proper
        ? word.split(' ').map((t, j) => (it.lx.proper[j] ? t[0].toLocaleUpperCase('az') + t.slice(1) : t)).join(' ') : word;
    }
    const words = items.map(({ labels: ls, word, via }) => ({ labels: ls, word, via }));
    const text = words.map((w) => w.word).filter(Boolean).join(' ');
    return { text: text ? text[0].toLocaleUpperCase('az') + text.slice(1) : '', words };
  }
}
