/**
 * Mətn → jest dili: söz işarələri + barmaq əlifbası.
 *
 * Mətn sözlərə bölünür. Lüğətdə olan söz və ifadələr (AzSLD Words 200, bax
 * tools/build_words.py) bütöv işarə kimi, qalanları hərf-hərf göstərilir.
 * HƏM VRM avatarı, HƏM cyber skeleton eyni landmark-larla sürülür.
 *
 * Hərf pozaları: AzSLD barmaq əlifbası (tools/build_poses.py). Statik hərflər
 * bir kadr, dinamik hərflər kadr ardıcıllığıdır.
 */
import { WordLexicon, azLower } from './words.js';

const DEFAULTS = {
  holdMs: 620,        // statik hərfin saxlanma müddəti
  transitionMs: 120,  // addımlar arası keçid (qısaldıldı)
  dynamicFps: 14,     // dinamik hərfin oynatma sürəti
  spaceMs: 250,       // boşluq (söz sonu) fasiləsi (qısaldıldı)
  leadInMs: 380,      // ilk addımda əlavə vaxt: avatarın qolu istirahətdən qalxır
  wordHoldMs: 150,    // söz işarəsinin son pozasında qısa saxlama (qısaldıldı)
  speed: 1,           // ümumi sürət: 0.5 — iki dəfə yavaş (izləyicinin sürət idarəsi)
};

export class TextToSign {
  constructor({ poses, lexicon = new WordLexicon(null), targets = [], onLetter, onState } = {}) {
    this.poses = poses;                 // {letters: {hərf: {type, frames}}}
    this.lexicon = lexicon;
    this.targets = targets;             // [{setHandPose, setWordPose} | {setPose}]
    this.onLetter = onLetter;
    this.onState = onState;
    this.opts = { ...DEFAULTS };
    this.playing = false;
    this.paused = false;
    this._abort = null;
  }

  get alphabet() { return Object.keys(this.poses?.letters ?? {}); }

  /** Mətni oynadıla bilən addımlara çevirir. */
  plan(text) {
    const letters = this.poses?.letters ?? {};
    const steps = [];
    // Hər addım mətndəki yerini daşıyır: from/to — addımın özü, seg — aid olduğu söz
    for (const seg of this.lexicon.segment(text)) {
      const at = { from: seg.from, to: seg.to, seg: { from: seg.from, to: seg.to } };
      if (seg.kind === 'space') {
        steps.push({ kind: 'space', char: ' ', ...at });
      } else if (seg.kind === 'word') {
        steps.push({ kind: 'word', char: seg.text, entry: seg.entry, ...at });
      } else {
        [...seg.text].forEach((ch, k) => {
          const c = azLower(ch);
          const pos = { ...at, from: seg.from + k, to: seg.from + k + 1 };
          steps.push(letters[c] ? { kind: 'letter', char: c, pose: letters[c], ...pos } : { kind: 'unknown', char: ch, ...pos });
        });
      }
    }
    return steps;
  }

  /** Mətndə hansı simvolların pozası yoxdur. */
  missing(text) {
    return [...new Set(this.plan(text).filter((s) => s.kind === 'unknown').map((s) => s.char))];
  }

  /** `meta` ({pose, frame}) avatara metrik 3D və trayektoriya üçün lazımdır. */
  #emit(landmarks, meta) {
    for (const t of this.targets) {
      if (typeof t.setHandPose === 'function') t.setHandPose(landmarks, meta);
      else if (typeof t.setPose === 'function') t.setPose(landmarks);
    }
  }

  /**
   * Söz kadrı: avatar bütöv kadrı alır, skeleton isə əllərin nöqtələrini.
   * Skeletə metrik 3B nöqtələr verilir — pozanın bilək nöqtəsi + əlin öz landmark-ları.
   * Şəkil koordinatları işlədiləndə hər əlin dərinliyi öz biləyinə nisbətən olurdu, ona görə
   * iki əl eyni müstəviyə yığılır və bir-birinin içinə girmiş kimi görünürdü.
   */
  #emitWord(word, f) {
    // Avatar hansı kadrı və dərinlik variantını seçibsə, skeleton da eynisini göstərsin
    const used = this.targets.map((t) => t.wordHands?.(word, f)).find(Boolean) ?? null;
    const pts = [];
    for (const [side, key, wrist] of [['left', 'L', 5], ['right', 'R', 6]]) {
      const info = used ? used[side] : { frame: f, flip: false };
      if (used && !info) continue;                       // avatar bu əli istirahətə endirib
      const fr = word.frames[info.frame] ?? word.frames[f];
      const hand = fr[key];
      if (!hand) continue;
      const base = fr.b?.[wrist];
      const sz = info.flip ? -1 : 1;
      for (const [x, y, z] of hand.w) {
        const o = [x - hand.w[0][0], y - hand.w[0][1], sz * (z - hand.w[0][2])];
        pts.push(base ? { x: base[0] + o[0], y: base[1] + o[1], z: base[2] + o[2] } : { x: o[0], y: o[1], z: o[2] });
      }
    }
    if (pts.length === 42) separateHands(pts);
    for (const t of this.targets) {
      if (typeof t.setWordPose === 'function') t.setWordPose(word, f);
      else if (typeof t.setPose === 'function') t.setPose(pts.length ? pts : null);
    }
  }

  async play(text) {
    this.stop();
    const steps = this.plan(text);
    if (!steps.length) return;

    const token = {};
    this._abort = token;
    this.playing = true;
    this.paused = false;
    this.onState?.({ playing: true, total: steps.length, index: 0 });

    // Söz kadrları əvvəlcədən gətirilir və həll olunur — oynatma ortasında şəbəkə
    // və hesablama gözlənilməsin. Yüklənməyən söz hərf-hərf göstərilir.
    const words = new Map();
    await Promise.all(steps.filter((s) => s.kind === 'word').map(async (s) => {
      words.set(s.entry.id, await this.lexicon.get(s.entry.id).catch(() => null));
    }));
    if (this._abort !== token) return;
    for (const word of words.values()) {
      if (word) for (const t of this.targets) t.prepareWord?.(word);
    }

    const sp = () => Math.max(this.opts.speed, 0.1);
    let first = true;
    for (let i = 0; i < steps.length; i++) {
      if (this._abort !== token) return;
      const step = steps[i];
      this.onLetter?.({ ...step, index: i, total: steps.length });
      this.onState?.({ playing: true, total: steps.length, index: i });

      if (step.kind === 'space') {
        await this.#wait(this.opts.spaceMs / sp(), token);
      } else if (step.kind === 'unknown') {
        await this.#wait(this.opts.transitionMs / sp(), token);
      } else if (step.kind === 'word' && words.get(step.entry.id)) {
        const word = words.get(step.entry.id);
        const lead = first ? this.opts.leadInMs : 0;
        first = false;
        const dt = 1000 / (word.fps * sp());
        for (let f = 0; f < word.frames.length; f++) {
          if (this._abort !== token) return;
          this.#emitWord(word, f);
          await this.#wait(f === 0 ? dt + lead : dt, token);
        }
        await this.#wait((this.opts.wordHoldMs + this.opts.transitionMs) / sp(), token);
      } else if (step.kind === 'word') {
        // Söz yüklənmədi — hərf-hərf
        for (const c of step.char.replace(/\s+/g, '')) {
          const pose = this.poses?.letters?.[c];
          if (!pose || this._abort !== token) continue;
          this.#emit(pose.frames[0], { pose, frame: 0 });
          await this.#wait((this.opts.holdMs + this.opts.transitionMs) / sp(), token);
        }
      } else {
        const { pose } = step;
        const frames = pose.frames;
        const lead = first ? this.opts.leadInMs : 0;
        first = false;
        if (pose.type === 'dynamic' && frames.length > 1) {
          const dt = 1000 / (this.opts.dynamicFps * sp());
          for (let f = 0; f < frames.length; f++) {
            if (this._abort !== token) return;
            this.#emit(frames[f], { pose, frame: f });
            await this.#wait(f === 0 ? dt + lead : dt, token);
          }
          await this.#wait(this.opts.holdMs * 0.45 / sp(), token);
        } else {
          this.#emit(frames[0], { pose, frame: 0 });
          await this.#wait(this.opts.holdMs / sp() + lead, token);
        }
        await this.#wait(this.opts.transitionMs / sp(), token);
      }
    }

    if (this._abort === token) {
      this.#emit(null);
      this.playing = false;
      this.onState?.({ playing: false, total: steps.length, index: steps.length });
    }
  }

  /** Tək hərfi göstərir (əlifba vərəqi üçün). */
  showLetter(ch) {
    this.stop();
    const pose = this.poses?.letters?.[azLower(ch)];
    if (!pose) return false;
    this.#emit(pose.frames[0], { pose, frame: 0 });
    this.onLetter?.({ kind: 'letter', char: azLower(ch), pose, index: 0, total: 1 });
    return true;
  }

  pause() { this.paused = true; }
  resume() { this.paused = false; }

  stop() {
    this._abort = null;
    this.playing = false;
    this.paused = false;
  }

  async #wait(ms, token) {
    const end = performance.now() + ms;
    while (performance.now() < end || this.paused) {
      if (this._abort !== token) return;
      await new Promise((r) => setTimeout(r, 16));
    }
  }
}

// Əlin sümükləri (MediaPipe indeksləri): barmaq zəncirləri və ovucun kənarları
const HAND_BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10],
  [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];

/** İki parça arasındakı ən kiçik məsafə ({x,y,z} nöqtələr). */
function segDist(p1, p2, p3, p4) {
  const ux = p2.x - p1.x, uy = p2.y - p1.y, uz = p2.z - p1.z;
  const vx = p4.x - p3.x, vy = p4.y - p3.y, vz = p4.z - p3.z;
  const wx = p1.x - p3.x, wy = p1.y - p3.y, wz = p1.z - p3.z;
  const a = ux * ux + uy * uy + uz * uz, b = ux * vx + uy * vy + uz * vz, c = vx * vx + vy * vy + vz * vz;
  const d = ux * wx + uy * wy + uz * wz, e = vx * wx + vy * wy + vz * wz;
  const den = a * c - b * b;
  const cl = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
  let s = den > 1e-12 ? cl((b * e - c * d) / den) : 0;
  let t = c > 1e-12 ? (b * s + e) / c : 0;
  if (t < 0) { t = 0; s = a > 1e-12 ? cl(-d / a) : 0; } else if (t > 1) { t = 1; s = a > 1e-12 ? cl((b - d) / a) : 0; }
  return Math.hypot(wx + s * ux - t * vx, wy + s * uy - t * vy, wz + s * uz - t * vz);
}

/**
 * Skeleton üçün iki əl (42 nöqtə, metr) bir-birinin içindən keçməsin. Datada əllərin nisbi
 * dərinliyi pozanın biləklərindən gəlir və səs-küylüdür: toxunan əllər datada 1-3 sm bir-birinin
 * içində görünür, skeletdə sümüklər bir-birini kəsirdi (ölçülüb: tools/measure_hands.js). Əllər
 * ovuclarını birləşdirən ox boyunca aralanır ki, sümük oxları arasında ən azı `min` qalsın —
 * təsvirdəki oynaq kürəsinin diametri qədər: toxunan əllər toxunan qalır. Nöqtələr yerində dəyişir.
 */
export function separateHands(pts, min = 0.02) {
  const gap = () => {
    let g = Infinity;
    for (const [a, b] of HAND_BONES) for (const [c, e] of HAND_BONES) g = Math.min(g, segDist(pts[a], pts[b], pts[21 + c], pts[21 + e]));
    return g;
  };
  const centre = (o) => ['x', 'y', 'z'].map((k) => [0, 5, 9, 17].reduce((s, j) => s + pts[o + j][k], 0) / 4);
  for (let i = 0; i < 6; i++) {
    const g = gap();
    if (g >= min) return;
    // Ox ən yaxın sümüklərdən yox, ovuclardan götürülür: barmaqlar bir-birinə keçəndə ən yaxın
    // cütün oxu hər addımda başqa tərəfə baxır
    const [l, r] = [centre(0), centre(21)];
    let ax = r.map((v, k) => v - l[k]);
    const len = Math.hypot(...ax);
    ax = len > 1e-6 ? ax.map((v) => v / len) : [1, 0, 0];
    const step = (min - g) / 2 + 0.002;
    for (let k = 0; k < 21; k++) {
      pts[k].x -= ax[0] * step; pts[k].y -= ax[1] * step; pts[k].z -= ax[2] * step;
      pts[21 + k].x += ax[0] * step; pts[21 + k].y += ax[1] * step; pts[21 + k].z += ax[2] * step;
    }
  }
}

export async function loadPoses(url = './assets/poses.json') {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`poses.json yüklənmədi (${res.status})`);
  return res.json();
}
