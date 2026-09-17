/**
 * Mətn → işarə dili (barmaq əlifbası ilə hecalama).
 *
 * Verilən mətni hərflərə ayırır, hər hərf üçün real əl pozasını lüğətdən
 * götürür və HƏM VRM avatarını, HƏM cyber skeleti eyni landmark-larla
 * canlandırır — yəni iki görüntü eyni mənbədən sürülür, uyğunsuzluq olmur.
 *
 * Poza mənbəyi: AzSLD barmaq əlifbası videolarından çıxarılmış MediaPipe
 * landmark-ları (bax tools/build_poses.py). Statik hərflər bir kadr,
 * dinamik hərflər kadr ardıcıllığıdır.
 */

const DEFAULTS = {
  holdMs: 620,        // statik hərfin saxlanma müddəti
  transitionMs: 180,  // hərflər arası keçid
  dynamicFps: 14,     // dinamik hərfin oynatma sürəti
  spaceMs: 420,       // boşluq (söz sonu) fasiləsi
  leadInMs: 380,      // ilk hərfdə əlavə vaxt: avatarın qolu istirahətdən qalxır
};

export class TextToSign {
  constructor({ poses, targets = [], onLetter, onState } = {}) {
    this.poses = poses;                 // {letters: {hərf: {type, frames}}}
    this.targets = targets;             // [{setPose(lm)|setHandPose(lm)}]
    this.onLetter = onLetter;
    this.onState = onState;
    this.opts = { ...DEFAULTS };
    this.queue = [];
    this.playing = false;
    this.paused = false;
    this._abort = null;
  }

  get alphabet() { return Object.keys(this.poses?.letters ?? {}); }

  /** Mətni oynadıla bilən addımlara çevirir. */
  plan(text) {
    const steps = [];
    const letters = this.poses?.letters ?? {};
    const words = this.poses?.words ?? {};
    
    // Regex matches words and spaces
    const tokens = text.toLowerCase().match(/\S+|\s+/g) || [];
    
    for (const token of tokens) {
      if (token.trim() === '') {
        steps.push({ kind: 'space', char: ' ' });
      } else if (words[token]) {
        steps.push({ kind: 'word', char: token, pose: words[token] });
      } else {
        // Fallback to spelling
        for (const raw of [...token]) {
          if (letters[raw]) {
            steps.push({ kind: 'letter', char: raw, pose: letters[raw] });
          } else {
            steps.push({ kind: 'unknown', char: raw });
          }
        }
      }
    }
    return steps;
  }

  /** Mətndə hansı simvolların pozası yoxdur. */
  missing(text) {
    return [...new Set(this.plan(text)
      .filter((s) => s.kind === 'unknown').map((s) => s.char))];
  }

  /** `meta` ({pose, frame}) avatara metrik 3D və trayektoriya üçün lazımdır. */
  #emit(landmarks, meta) {
    for (const t of this.targets) {
      if (typeof t.setHandPose === 'function') t.setHandPose(landmarks, meta);
      else if (typeof t.setPose === 'function') t.setPose(landmarks);
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

    let first = true;
    for (let i = 0; i < steps.length; i++) {
      if (this._abort !== token) return;
      const step = steps[i];
      this.onLetter?.({ ...step, index: i, total: steps.length });
      this.onState?.({ playing: true, total: steps.length, index: i });

      if (step.kind === 'space') {
        this.#emit(null);
        await this.#wait(this.opts.spaceMs, token);
      } else if (step.kind === 'unknown') {
        await this.#wait(this.opts.transitionMs, token);
      } else {
        const { pose } = step;
        const frames = pose.frames;
        const lead = first ? this.opts.leadInMs : 0;
        first = false;
        if (pose.type === 'dynamic' && frames.length > 1) {
          const dt = 1000 / this.opts.dynamicFps;
          for (let f = 0; f < frames.length; f++) {
            if (this._abort !== token) return;
            this.#emit(frames[f], { pose, frame: f });
            await this.#wait(f === 0 ? dt + lead : dt, token);
          }
          await this.#wait(this.opts.holdMs * 0.45, token);
        } else {
          this.#emit(frames[0], { pose, frame: 0 });
          await this.#wait(this.opts.holdMs + lead, token);
        }
        await this.#wait(this.opts.transitionMs, token);
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
    const pose = this.poses?.letters?.[ch.toLowerCase()];
    if (!pose) return false;
    this.#emit(pose.frames[0], { pose, frame: 0 });
    this.onLetter?.({ kind: 'letter', char: ch.toLowerCase(), pose, index: 0, total: 1 });
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
      if (this.paused) continue;
    }
  }
}

export async function loadPoses(url = './assets/poses.json') {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`poses.json yüklənmədi (${res.status})`);
  return res.json();
}
