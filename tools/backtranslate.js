/**
 * V3 geri tərcümə (roadmap V3), brauzer hissəsi: mətn → TextToSign-in planı → avatar (tətbiqdəki oynatma,
 * virtual saatla) → hər 1/30 s-də MediaPipe formasında landmark kadrı. Kadrları tools/backtranslate.mjs cümlə
 * modelinə verir və işarələri niyyət olunan işarələrlə tutuşdurur.
 *
 * Landmark-lar avatarın sümüklərindən "mükəmməl izləyici" kimi qurulur (render + MediaPipe yox): sabit virtual
 * kamera AzSLD-nin ön kamerasının (Cam2, 1280×960) median kadrına uyğunlaşdırılır — çiyinlərin ortası y = 0.566,
 * çiyin eni 0.316 kadr hündürlüyü. Ölçülər ml/azsl/landmarks.py ilə eyni sahələrdir:
 *   pose        9 nöqtə (burun, çiyinlər, dirsəklər, biləklər, budlar): şəkil x, y (z = 0, model işlətmir)
 *   hand        21 nöqtə: şəkil x, y; z — biləyə nisbətən dərinlik, x miqyasının 0.46-sı (AzSLD-də ölçülüb)
 *   hand_world  21 nöqtə: kameranın oxlarında metr, mərkəz — nöqtələrin ortası, ölçü MediaPipe-in median əlinə
 *               (bilək → orta barmağın kökü 9 sm) gətirilir: MediaPipe avatarın kiçik əlini də bu ölçüdə verərdi
 *   üz          yoxdur (NaN): cümlə modeli üzü işlətmir (face_drop 1.0)
 * Əl kadrın aşağısından çıxanda (ovuc y > 0.97) aşkarlanmır — AzSLD-də dizdəki əllər kimi.
 *
 * `source` rejimi: avatar əvəzinə iştirakçının öz datası (lüğət klipinin kadrları) eyni kamera ilə — avatarın nə
 * qədər itirdiyini ayırmaq üçün istinad. Hərflər bu rejimdə göstərilmir.
 */
import * as THREE from 'three';
import { handModel, avatarHand, handGap, bodyDepth } from './measure_hands.js';
import { azLower } from '../src/tercume/words.js';

export const ASPECT = 4 / 3;
export const FPS = 30;
const VIEW = { shoulderY: 0.566, shoulderW: 0.316, dist: 2.0 };
const MP_PALM = 0.090;          // MediaPipe əl world: bilək → orta barmağın kökü, median (AzSLD)
const Z_SCALE = 0.46;           // MediaPipe əl şəkli: z / x miqyası (AzSLD Cam2)
const HAND_MAX_Y = 0.97;        // ovuc bundan aşağıdadırsa əl aşkarlanmır
const POSE_BONES = [null, 'leftUpperArm', 'rightUpperArm', 'leftLowerArm', 'rightLowerArm', 'leftHand', 'rightHand',
  'leftUpperLeg', 'rightUpperLeg'];

/** Sabit virtual kamera avatarın ölçülərindən. */
export function makeCamera(avatar) {
  const b = avatar.body;
  const Hm = b.shoulderWidth / VIEW.shoulderW;               // kadrın hündürlüyü metrlə (çiyin müstəvisində)
  const c = new THREE.Vector3(b.mid.x, b.mid.y + (VIEW.shoulderY - 0.5) * Hm, b.mid.z + VIEW.dist);
  return {
    Hm, c,
    /** Dünya nöqtəsi → [x, y] kadrda (0..1) və dərinlik (metr). */
    project(p) {
      const depth = c.z - p.z;
      const k = VIEW.dist / depth / Hm;
      return [0.5 + (p.x - c.x) * k / ASPECT, 0.5 - (p.y - c.y) * k, depth];
    },
  };
}

/** Avatarın burnu başın lokal koordinatında (bədən ölçülən anda). */
function noseLocal(avatar) {
  const head = avatar.bone('head');
  avatar.vrm.scene.updateMatrixWorld(true);
  return head.worldToLocal(avatar.body.nose.clone());
}

const NAN = (n) => new Float32Array(n).fill(NaN);

/** 21 dünya nöqtəsi ([x,y,z]) → kadrın `hand` və `hand_world` sahələri (slot). */
function writeHand(cam, P, slot, out, world = null) {
  const proj = P.map((p) => cam.project(new THREE.Vector3(...p)));
  const palm = [0, 5, 9, 17].reduce((s, j) => s + proj[j][1], 0) / 4;
  const px = [0, 5, 9, 17].reduce((s, j) => s + proj[j][0], 0) / 4;
  if (palm > HAND_MAX_Y || palm < 0 || px < 0 || px > 1) return;
  const kx = VIEW.dist / proj[0][2] / cam.Hm / ASPECT;      // x miqyası biləyin dərinliyində (kadr eni / metr)
  const o = slot * 63;
  for (let j = 0; j < 21; j++) {
    out.hand[o + j * 3] = proj[j][0];
    out.hand[o + j * 3 + 1] = proj[j][1];
    out.hand[o + j * 3 + 2] = Z_SCALE * kx * (proj[j][2] - proj[0][2]);
  }
  // world: kameranın oxları (x sağa, y aşağı, z uzağa), mərkəz ortada, MediaPipe-in əl ölçüsü
  const W = world ?? P.map(([x, y, z]) => [x, -y, -z]);
  const size = Math.hypot(...[0, 1, 2].map((k) => W[9][k] - W[0][k])) || 1;
  const s = world ? 1 : MP_PALM / size;
  const m = [0, 1, 2].map((k) => W.reduce((a, p) => a + p[k], 0) / 21);
  for (let j = 0; j < 21; j++) for (let k = 0; k < 3; k++) out.hand_world[o + j * 3 + k] = (W[j][k] - m[k]) * s;
}

function blank() {
  return { hand: NAN(126), hand_world: NAN(126), pose: NAN(27) };
}

/** Avatarın cari pozası → bir kadr. */
export function avatarFrame(avatar, model, cam, nose) {
  avatar.vrm.scene.updateMatrixWorld(true);
  const out = blank();
  const v = new THREE.Vector3();
  POSE_BONES.forEach((name, k) => {
    const p = name ? avatar.bone(name).getWorldPosition(v) : avatar.bone('head').localToWorld(v.copy(nose));
    const [x, y] = cam.project(p);
    out.pose[k * 3] = x; out.pose[k * 3 + 1] = y; out.pose[k * 3 + 2] = 0;
  });
  // slot 0 = iştirakçının (avatarın) SAĞ əli
  writeHand(cam, avatarHand(avatar, model, 'right'), 0, out);
  writeHand(cam, avatarHand(avatar, model, 'left'), 1, out);
  return out;
}

/** İştirakçının söz kadrı (lüğət) → avatarın çiyin çərçivəsinə köçürülüb eyni kamera ilə bir kadr. */
export function sourceFrame(avatar, cam, fr) {
  const out = blank();
  if (!fr?.b) return out;
  const P = fr.b.map(([x, y, z]) => new THREE.Vector3(x, -y, -z));
  const mid = P[1].clone().add(P[2]).multiplyScalar(0.5);
  const k = avatar.body.shoulderWidth / P[1].distanceTo(P[2]);
  const place = (p) => p.clone().sub(mid).multiplyScalar(k).add(avatar.body.mid);
  P.forEach((p, i) => {
    const [x, y] = cam.project(place(p));
    out.pose[i * 3] = x; out.pose[i * 3 + 1] = y; out.pose[i * 3 + 2] = 0;
  });
  for (const [key, wrist, slot] of [['R', 6, 0], ['L', 5, 1]]) {
    const h = fr[key];
    if (!h) continue;
    const base = place(P[wrist]);
    const pts = h.w.map(([x, y, z]) => [base.x + (x - h.w[0][0]) * k, base.y - (y - h.w[0][1]) * k, base.z - (z - h.w[0][2]) * k]);
    writeHand(cam, pts, slot, out, h.w);
  }
  return out;
}

/** Oynatmanın hədəfi `source` rejimində: göstərilən söz kadrını yadda saxlayır. */
class SourceTarget {
  constructor() { this.cur = null; }
  setWordPose(word, f) { this.cur = { word, f }; }
  setHandPose() { this.cur = null; }
  clearHandPose() { this.cur = null; }
}

/**
 * Səhifədə bir dəfə qurulur: avatar, oynadıcı, virtual saat. `run(text, mode)` bir cümlə üçün kadrları və
 * niyyət olunan işarələri (addım-addım, kadr aralıqları ilə) qaytarır.
 */
export async function setup({ avatar, t2s, lexicon }) {
  const model = handModel(avatar);
  avatar.paused = true;
  avatar.life = false;
  avatar.update(0);
  const cam = makeCamera(avatar);
  const nose = noseLocal(avatar);

  // Virtual saat: t2s-in gözləmələri (performance.now + setTimeout) yalnız kadr irəlilədikcə bitir
  // (tools/render_avatar_clips.mjs kimi)
  const vt = { now: performance.now() };
  const q = [];
  let seq = 0;
  performance.now = () => vt.now;
  window.setTimeout = (fn, ms = 0, ...a) => { const id = ++seq; q.push({ t: vt.now + Math.max(0, Number(ms) || 0), id, fn, a }); return id; };
  window.clearTimeout = (id) => { const i = q.findIndex((x) => x.id === id); if (i >= 0) q.splice(i, 1); };
  const flush = () => new Promise((r) => { const ch = new MessageChannel(); ch.port1.onmessage = () => r(); ch.port2.postMessage(0); });
  async function advance(to) {
    for (;;) {
      let k = -1;
      for (let i = 0; i < q.length; i++) {
        if (q[i].t <= to && (k < 0 || q[i].t < q[k].t || (q[i].t === q[k].t && q[i].id < q[k].id))) k = i;
      }
      if (k < 0) break;
      const x = q.splice(k, 1)[0];
      vt.now = Math.max(vt.now, x.t);
      if (typeof x.fn === 'function') x.fn(...x.a);
      await flush();
      await flush();
    }
    vt.now = to;
    await flush();
  }
  const source = new SourceTarget();
  const avatarTargets = t2s.targets;

  /** Bir zaman addımı (1/30 s: iki 60 Hz yeniləmə, brauzerin rAF-ı kimi). */
  async function tick() {
    for (let i = 0; i < 2; i++) {
      await advance(vt.now + 1000 / 60);
      avatar.update(1 / 60);
    }
  }

  /**
   * @param {{mode?: string, pre?: number, tail?: number, maxSeconds?: number, signs?: object[] | null,
   *          render?: number, opts?: object}} o render — hər N kadrdan bir avatarın şəkli (JPEG, kontakt vərəqi üçün);
   *          opts — yalnız bu cümlə üçün TextToSign.opts
   */
  async function run(text, { mode = 'avatar', pre = 0.5, tail = 0.6, maxSeconds = 60, signs = null, render = 0, opts = null,
    phase = 0, dual = false } = {}) {
    const keep = { ...t2s.opts };
    if (opts) Object.assign(t2s.opts, opts);
    // Hər cümlə saatın eyni fazasından başlayır (`phase` ms): oynadıcının 16 ms-lik gözləmələri ilə 60 Hz yeniləmələrin
    // nisbəti kadrları azca dəyişir; faza əvvəlki cümlədən asılı olsa nəticə əvvəlki cümlələrdən asılı olardı
    await advance(Math.ceil(vt.now / 1000) * 1000 + 1000 + phase);
    try {
      return await play(text, { mode, pre, tail, maxSeconds, signs, render, dual });
    } finally {
      t2s.opts = keep;
    }
  }

  async function play(text, { mode, pre, tail, maxSeconds, signs, render, dual }) {
    const steps = signs ? t2s.planSigns(signs) : t2s.plan(text);
    const words = new Map();
    await Promise.all(steps.filter((s) => s.kind === 'word').map(async (s) => {
      words.set(s.entry.id, await lexicon.get(s.entry.id).catch(() => null));
    }));
    t2s.stop();
    // dual (avatar rejimində): iştirakçının klipi də eyni anda izlənir — hər kadrda avatarın və klipin kadrı yan-yana
    // (tools/v3_swap.mjs onları qarışdırıb itkinin hansı hissədən gəldiyini ölçür)
    t2s.targets = mode === 'source' ? [source] : dual ? [...avatarTargets, source] : avatarTargets;
    source.cur = null;
    avatar.clearHandPose();
    for (let i = 0; i < 90; i++) avatar.update(1 / 60);       // istirahət
    avatar.snap();

    const frames = [], shots = [], srcFrames = [];
    // Oynatmada əllərin yoxlanması (avatar): iki əl bir-birinə > 5 mm girir, əl bədənə > 5 mm girir, ovucun bir
    // kadrdakı (1/30 s) ən böyük yerdəyişməsi; yalnız qaldırılmış əllər (istirahətdəki yox)
    const check = { overlap: 0, body: 0, maxStep: 0, path: 0, up: 0, elbow: 0, right: 0 };
    const elbowUp = (side) => {          // E28: dirsək çiyindən 15 sm-dən az aşağıda və 12 sm-dən çox çöldə
      const rig = avatar.rigs[side], E = avatar.bone(side + 'LowerArm').getWorldPosition(new THREE.Vector3());
      return E.y - rig.S.y > -0.15 && (E.x - rig.S.x) * rig.out > 0.12;
    };
    let prevPalm = null;
    const palmOf = (P) => [0, 1, 2].map((k) => (P[0][k] + P[5][k] + P[9][k] + P[17][k]) / 4);
    const sample = () => {
      const fr = mode === 'source' && source.cur
        ? sourceFrame(avatar, cam, source.cur.word.frames[source.cur.f])
        : avatarFrame(avatar, model, cam, nose);
      frames.push(fr);
      if (dual) srcFrames.push(source.cur ? sourceFrame(avatar, cam, source.cur.word.frames[source.cur.f]) : fr);
      if (render && (frames.length - 1) % render === 0) {
        avatar.renderer.render(avatar.scene, avatar.camera);
        shots.push({ t: frames.length - 1, url: avatar.renderer.domElement.toDataURL('image/jpeg', 0.85) });
      }
      if (mode !== 'avatar') return;
      const L = avatar.last?.left ? avatarHand(avatar, model, 'left') : null;
      const R = avatar.last?.right ? avatarHand(avatar, model, 'right') : null;
      if (L && R && handGap(L, R, model.radius).gap < -0.005) check.overlap++;
      for (const [side, P] of [['left', L], ['right', R]]) {
        if (P && bodyDepth(avatar, model.radius, side, P).hand > 0.005) { check.body++; break; }
      }
      const now = ['left', 'right'].map((s) => palmOf(avatarHand(avatar, model, s)));
      if (prevPalm) {
        const st = Math.max(...now.map((p, k) => Math.hypot(...p.map((v, j) => v - prevPalm[k][j]))));
        check.maxStep = Math.max(check.maxStep, st);
        check.path += st;
      }
      if (L || R) check.up++;
      if (R) { check.right++; if (elbowUp('right')) check.elbow++; }
      prevPalm = now;
    };
    for (let i = 0; i < Math.round(pre * FPS); i++) { await tick(); sample(); }

    // Hər addımın başladığı kadr (onLetter): niyyət olunan işarənin kadr aralığı
    const starts = new Array(steps.length).fill(null);
    const onLetter = t2s.onLetter;
    t2s.onLetter = (s) => { if (starts[s.index] == null) starts[s.index] = frames.length; };
    const done = t2s.playSteps(steps);
    await flush();
    let left = Math.round(tail * FPS);
    while (frames.length < maxSeconds * FPS) {
      await tick();
      sample();
      if (!t2s.playing && --left <= 0) break;
    }
    t2s.stop();
    await advance(vt.now + 100);
    await done;
    t2s.onLetter = onLetter;
    t2s.targets = avatarTargets;
    const end = frames.length;
    const intended = [];
    steps.forEach((s, i) => {
      if (s.kind !== 'word' && s.kind !== 'letter') return;
      const a = starts[i] ?? end;
      const nextStart = starts.slice(i + 1).find((x) => x != null) ?? end;
      const word = s.kind === 'word' ? words.get(s.entry.id) : null;
      intended.push({
        kind: s.kind, a, b: nextStart,
        id: s.entry?.id ?? null,
        text: s.kind === 'word' ? s.entry.text : s.char,
        label: word?.label ?? null,
        via: s.via ?? null,
        seg: s.seg?.from ?? null,
        q: s.gram?.q ?? null,
      });
    });
    // Kadrlar düz massivlərə (Node-a ötürmək üçün)
    const T = frames.length;
    const pack = (k, n, fs = frames) => { const a = new Array(T * n); fs.forEach((f, t) => { for (let j = 0; j < n; j++) a[t * n + j] = f[k][j]; }); return a; };
    return {
      text, T, aspect: ASPECT, fps: FPS, intended, check, shots,
      hand: pack('hand', 126), hand_world: pack('hand_world', 126), pose: pack('pose', 27),
      src: dual ? { hand: pack('hand', 126, srcFrames), hand_world: pack('hand_world', 126, srcFrames), pose: pack('pose', 27, srcFrames) } : null,
    };
  }
  return { run, cam, model };
}

export { azLower };
