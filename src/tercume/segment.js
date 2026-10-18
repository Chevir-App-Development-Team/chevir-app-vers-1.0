/**
 * Söz rejimi: fasiləsiz landmark axınını ayrı-ayrı işarələrə bölür. ml/azsl/segment.py ilə eyni
 * qaydalar (tools/verify_segment.mjs hər ikisini ml/fixtures/segment_v1.json üzərində yoxlayır).
 *
 * 1. Aktivlik: əl istirahət xəttindən yuxarıda görünür — biləyi çiyin xəttindən `restLine` çiyin
 *    eni aşağıdan yuxarıdadır. Bədən tapılmayanda görünən istənilən əl sayılır. AzSLD-də əllər
 *    dizdə dincəlir (çiyindən ~1.25–1.3 çiyin eni aşağıda) və orada çox vaxt aşkarlanmır.
 * 2. Aralıqlar: `minOn` aktiv kadrdan sonra açılır, `minOff` istirahət kadrından sonra bağlanır.
 *    Nüvə — ilk..son aktiv kadr; iki tərəfdən `margin` kadr kontekst saxlanılır (təlimdəki kimi).
 * 3. Aralığın içində işarələr (fasiləsiz jestdə işarələr arasında istirahət yoxdur): bütün aralıq
 *    və mərkəzi nüvədə olan `lens` uzunluqlu, hər `stride` kadrdan bir pəncərələr modelə verilir;
 *    ən əmin pəncərə saxlanılır, onunla qısanın `maxOverlap` hissəsindən çox üst-üstə düşənlər
 *    atılır, `minP`-ə qədər təkrarlanır. Zamana görə düzülür, eyni sözlü qonşular birləşir.
 */
// v2: lens, minP və maxOverlap val cümlələrində seçilib (notebook 05, EXPERIMENTS.md E6)
export const SEGMENT_VERSION = 2;
export const SEGMENT = {
  restLine: 1.0, minOn: 3, minOff: 8, margin: 8, minSpan: 8, maxSpan: 240,
  lens: [16, 24, 32], stride: 3, minP: 0.8, maxOverlap: 0.0,
};

/**
 * Model manifest-indəki söz rejimi parametrləri (notebook 05 val-da seçib, Python adları ilə) →
 * SEGMENT açarları. Versiya fərqlidirsə (başqa qaydalar üçün seçilib) heç nə götürülmür.
 */
export function segmentFromManifest(seg) {
  if (!seg?.params || seg.version !== SEGMENT_VERSION) return {};
  const names = { rest_line: 'restLine', min_on: 'minOn', min_off: 'minOff', margin: 'margin', min_span: 'minSpan',
    max_span: 'maxSpan', lens: 'lens', stride: 'stride', min_p: 'minP', max_overlap: 'maxOverlap' };
  const out = {};
  for (const [k, v] of Object.entries(seg.params)) if (names[k]) out[names[k]] = v;
  return out;
}

/** Bir kadr (landmarks.js frame: hand 126, pose 27) aktivdirmi? */
export function frameActive(frame, aspect, restLine = SEGMENT.restLine) {
  const { hand, pose } = frame;
  const shY = (pose[1 * 3 + 1] + pose[2 * 3 + 1]) / 2;          // POSE_IDX 1, 2 — çiyinlər
  const sw = Math.hypot((pose[1 * 3] - pose[2 * 3]) * aspect, pose[1 * 3 + 1] - pose[2 * 3 + 1]);
  const body = Number.isFinite(shY) && Number.isFinite(sw) && sw > 1e-6;
  for (let s = 0; s < 2; s++) {
    if (!Number.isFinite(hand[s * 63])) continue;              // bu slotda əl yoxdur
    if (!body || (hand[s * 63 + 1] - shY) / sw < restLine) return true;
  }
  return false;
}

/** raw (LandmarkClip.toRaw / features.js girişi) → hər kadr üçün aktivlik. */
export function activity(raw, aspect, restLine = SEGMENT.restLine) {
  const out = new Array(raw.T);
  for (let t = 0; t < raw.T; t++) {
    out[t] = frameActive({ hand: raw.hand.subarray(t * 126, t * 126 + 126),
      pose: raw.pose.subarray(t * 27, t * 27 + 27) }, aspect, restLine);
  }
  return out;
}

/**
 * Onlayn aralıq izləyicisi: hər kadrda push(aktiv) çağırılır; aralıq bağlananda
 * { lo, hi, core: [c0, c1] } qaytarır (mütləq kadr nömrələri, hi və c1 daxil deyil), əks halda null.
 */
export class SignSegmenter {
  constructor(opts = {}) {
    this.p = { ...SEGMENT, ...opts };
    this.reset();
  }

  reset() {
    this.n = 0; this.on = 0; this.off = 0; this.start = null; this.last = null;
  }

  get open() { return this.start !== null; }

  push(active) {
    const p = this.p, n = this.n++;
    if (active) {
      this.on++; this.off = 0;
      if (this.start === null && this.on >= p.minOn) this.start = n - this.on + 1;
      if (this.start !== null) this.last = n;
    } else {
      this.on = 0; this.off++;
    }
    if (this.start !== null && (this.off >= p.minOff || n - this.start + 1 >= p.maxSpan)) {
      const span = this.#close(n + 1);
      this.on = 0; this.off = 0;
      return span;
    }
    return null;
  }

  /** Axın bitdi: açıq aralıq varsa bağlayır. */
  flush() {
    return this.start === null ? null : this.#close(this.n);
  }

  #close(endN) {
    const p = this.p, { start, last } = this;
    this.start = null; this.last = null;
    if (last - start + 1 < p.minSpan) return null;
    return { lo: Math.max(0, start - p.margin), hi: Math.min(endN, last + 1 + p.margin), core: [start, last + 1] };
  }
}

/** Aralığın namizəd pəncərələri, sabit sırada: bütün aralıq, sonra uzunluğa, sonra başlanğıca görə. */
export function windows({ lo, hi, core: [c0, c1] }, p = SEGMENT) {
  const out = [[lo, hi]];
  for (const n of p.lens) {
    for (let a = lo; a <= hi - n; a += p.stride) {
      if (c0 <= a + n / 2 && a + n / 2 < c1 && !(a === lo && a + n === hi)) out.push([a, a + n]);
    }
  }
  return out;
}

/** scored: [{ a, b, label, p }] windows() sırasında → saxlanılan işarələr zaman sırasında. */
export function pick(scored, p = SEGMENT) {
  const order = scored.map((_, i) => i).sort((i, j) => scored[j].p - scored[i].p || i - j);
  const kept = [];
  for (const i of order) {
    const w = scored[i];
    if (w.p < p.minP) break;
    if (kept.every((k) => Math.min(w.b, k.b) - Math.max(w.a, k.a) <= p.maxOverlap * Math.min(w.b - w.a, k.b - k.a))) {
      kept.push(w);
    }
  }
  kept.sort((x, y) => x.a - y.a || x.b - y.b);
  const merged = [];
  for (const w of kept) {
    const prev = merged[merged.length - 1];
    if (prev && prev.label === w.label) {
      merged[merged.length - 1] = { ...prev, b: Math.max(prev.b, w.b), p: Math.max(prev.p, w.p) };
    } else {
      merged.push({ ...w });
    }
  }
  return merged;
}

/** raw-ın [a, b) kadrları — kopyasız (FRAME_SIZES sahələri). */
export function sliceRaw(raw, a, b, sizes) {
  const out = { T: b - a };
  for (const [k, n] of Object.entries(sizes)) out[k] = raw[k].subarray(a * n, b * n);
  return out;
}

/**
 * Bir aralığı işarələrə bölür. raw — aralığın kadrları (0 = span.lo); score(raw) → { label, p, top }.
 * Qaytarır: { signs: [{ a, b, label, p, top }] (mütləq kadr nömrələri ilə), best } — best ən əmin
 * pəncərədir, heç bir söz `minP`-i keçməyəndə interfeys onu "əmin deyil" kimi göstərə bilər.
 */
export async function segmentSpan(span, raw, score, sizes, p = SEGMENT) {
  const scored = [];
  for (const [a, b] of windows(span, p)) {
    const r = await score(sliceRaw(raw, a - span.lo, b - span.lo, sizes));
    scored.push({ a, b, label: r.label, p: r.p, top: r.top });
  }
  const best = scored.reduce((m, w) => (!m || w.p > m.p ? w : m), null);
  return { signs: pick(scored, p), best };
}
