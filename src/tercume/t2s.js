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
  transitionMs: 180,  // addımlar arası keçid
  dynamicFps: 14,     // dinamik hərfin oynatma sürəti
  spaceMs: 420,       // boşluq (söz sonu) fasiləsi
  leadInMs: 380,      // ilk addımda əlavə vaxt: avatarın qolu istirahətdən qalxır
  wordHoldMs: 260,    // söz işarəsinin son pozasında qısa saxlama
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

  /** Söz kadrı: avatar bütöv kadrı alır, skeleton isə görünən əllərin nöqtələrini. */
  #emitWord(word, f) {
    const fr = word.frames[f];
    const pts = [fr.L, fr.R].filter(Boolean).flatMap((h) => h.n.map(([x, y, z]) => ({ x, y, z })));
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
        this.#emit(null);
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

export async function loadPoses(url = './assets/poses.json') {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`poses.json yüklənmədi (${res.status})`);
  return res.json();
}
