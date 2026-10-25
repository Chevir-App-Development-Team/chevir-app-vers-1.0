/**
 * VRM avatarı — işarə pozasını bütün qola ötürür (çiyin → dirsək → bilək → barmaqlar).
 *
 * Əvvəlki versiyada yalnız barmaq bükülməsi landmark-lardan gəlirdi; qol və bilək
 * sabit idi, ona görə əlin istiqaməti itirdi ("l"-də barmaqlar aşağı baxmalıdır,
 * avatar isə yuxarı göstərirdi). İndi:
 *   - əlin oriyentasiyası, qol IK və barmaq oynaqları → retarget.js
 *   - hər sümük kritik sönümlü yayla hərəkət edir: sürət kəsilmir, ani sıçrayış
 *     yoxdur, ağır seqmentlər (qol) barmaqlardan yavaş çatır
 *   - canlılıq: nəfəs, göz qırpma, baxış kameraya, başın kiçik hərəkəti;
 *     sözlər arasında əl hazır vəziyyətə keçir, işarə bitəndə qol aşağı enir
 *
 * Avatar VRM 1.0-dır (VRoid). three-vrm normallaşdırılmış humanoid verir:
 * istirahətdə bütün lokal dönmələr vahiddir və avatar +Z-yə baxır.
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import {
  FINGER_CHAINS, measureRig, handTarget, frameSource, signAnchor, solveArm, solveFingers,
  restArm, readyHand, relaxedFingers, trajectory, quatToVec, vecToQuat,
} from './retarget.js';

const SIDES = ['left', 'right'];
/** Yayların təbii tezliyi (rad/s); hədəfə ~5.8/ω saniyədə çatır: qol ≈0.5 s, barmaq ≈0.28 s. */
const OMEGA = { arm: 16.5, hand: 22.5, finger: 31.5, body: 9 };
const PAUSE_TO_REST = 0.9;   // s — yeni poza gəlməsə qol aşağı enir
const BODY = ['spine', 'chest', 'upperChest', 'neck', 'head'];
const UP = new THREE.Vector3(0, 1, 0), DOWN = new THREE.Vector3(0, -1, 0);
const DEG = Math.PI / 180;
// Söz işarəsinin üzü (tools/build_avatar_words.py, kadrın `f` sahəsi): MediaPipe-in blendshape-ləri
// VRM-in standart ifadələrinə. Hər blendshape lüğətdəki öz diapazonuna görə ölçülür: median
// (siqnalçıların sakit üzü) → 0, 99-cu faiz → tam (MediaPipe-in dəyərləri sıxılmışdır: ağzın
// geniş açılması belə jawOpen ≈ 0.16). Ağız formaları (aa, ih, ou, ee, oh) dodaqların hərəkətidir;
// surprised / angry yalnız qaşlar üçündür, bütün üzü dəyişdikləri üçün zəif saxlanılır.
const FACE_MAP = {
  aa: [['jawOpen'], 0.8], oh: [['mouthFunnel'], 0.6], ou: [['mouthPucker'], 0.6],
  ee: [['mouthStretchLeft', 'mouthStretchRight'], 0.5], ih: [['mouthSmileLeft', 'mouthSmileRight'], 0.6],
  blink: [['eyeBlinkLeft', 'eyeBlinkRight'], 1],
  surprised: [['browInnerUp', 'browOuterUpLeft', 'browOuterUpRight'], 0.6],
  angry: [['browDownLeft', 'browDownRight'], 0.5],
};
const VISEMES = ['aa', 'ih', 'ou', 'ee', 'oh'];
const HEAD_MAX = 20 * DEG;      // başın sözdəki hərəkəti (baş sallama, yelləmə) bu həddə qədər

const CONTACT = [4, 8, 12, 16, 20, 5, 17];   // üzə toxuna bilən əl nöqtələri: barmaq ucları və ovucun kənarları
const TIP_BONES = ['ThumbDistal', 'IndexDistal', 'MiddleDistal', 'RingDistal', 'LittleDistal'];
// Toqquşma üçün sümüklərin qalınlığı (metr). Avatarın mesh-i ölçülüb: barmağın dərisi
// oxdan ~9 mm, ovucun ~23 mm, saidin (köynəklə) ~38 mm. Ovuc və said yastı/konusvaridir,
// ona görə kapsul radiusu enə görə deyil, qalınlığa görə götürülür: iki ovuc üz-üzə
// toxunanda (dua pozası) oxlar ~20 mm aralı olur və bu, girmə sayılmamalıdır.
// Biləyin həddini aşmağın yuxarı sərhədi (solveArm-ın `hard` ölçüsü, ~0.035 ≈ 11°)
const HARD_CEIL = 0.035;

const R_FINGER = 0.008;
const R_PALM = 0.010;
const R_FOREARM = 0.030;
const R_UPPERARM = 0.040;
// Oynatmada əllərin bir-birinə girməsinə icazə (metr): səthlər toxuna bilər, keçə bilməz
const GUARD_SLACK = 0.003;

const FINGER_BONE_NAMES = {
  Thumb: ['Metacarpal', 'Proximal', 'Distal'],
  Index: ['Proximal', 'Intermediate', 'Distal'],
  Middle: ['Proximal', 'Intermediate', 'Distal'],
  Ring: ['Proximal', 'Intermediate', 'Distal'],
  Little: ['Proximal', 'Intermediate', 'Distal'],
};

/**
 * Kapsul dəsti maneə kimi: əvvəlcədən hazırlanır, sonra hər nöqtə üçün allokasiyasız
 * hesablanır. Bu funksiya IK-nın daxili döngəsində minlərlə dəfə çağırılır — vektor
 * obyektləri yaratmaq həlli saniyələrlə uzadırdı (ölçülüb).
 */
function prepareCapsules(parts, margin) {
  return parts.map(([a, b, r]) => {
    const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
    return { ax: a.x, ay: a.y, az: a.z, dx, dy, dz,
             inv: 1 / Math.max(dx * dx + dy * dy + dz * dz, 1e-9), r: r + margin };
  });
}

/** Nöqtənin hazırlanmış kapsullara girmə dərinliyi (müsbət = içəridə). */
function capsuleDepth(caps, p) {
  let depth = -Infinity;
  for (const c of caps) {
    let t = ((p.x - c.ax) * c.dx + (p.y - c.ay) * c.dy + (p.z - c.az) * c.dz) * c.inv;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const qx = p.x - (c.ax + c.dx * t), qy = p.y - (c.ay + c.dy * t), qz = p.z - (c.az + c.dz * t);
    const d = c.r - Math.sqrt(qx * qx + qy * qy + qz * qz);
    if (d > depth) depth = d;
  }
  return depth;
}

/** İki dönmə arasındakı bucaq (radian). */
function quatAngle(a, b) {
  return 2 * Math.acos(Math.min(1, Math.abs(a.dot(b))));
}

/**
 * İki sümük parçası arasındakı ən qısa məsafə; ən yaxın nöqtələr `pa`, `pb`-yə yazılır.
 * Bir parametr sərhədə kəsiləndə digəri yenidən hesablanır: ikisini ayrıca kəsmək ən yaxın
 * cütü itirir və məsafəni şişirdir (barmaq ucu ovucun kənarına yaxınlaşanda olur).
 * Allokasiyasızdır — oynatmada hər kadr yüzlərlə dəfə çağırılır.
 */
function segmentClosest(p1, p2, p3, p4, pa, pb) {
  const ux = p2.x - p1.x, uy = p2.y - p1.y, uz = p2.z - p1.z;
  const vx = p4.x - p3.x, vy = p4.y - p3.y, vz = p4.z - p3.z;
  const wx = p1.x - p3.x, wy = p1.y - p3.y, wz = p1.z - p3.z;
  const a = ux * ux + uy * uy + uz * uz, b = ux * vx + uy * vy + uz * vz, c = vx * vx + vy * vy + vz * vz;
  const d = ux * wx + uy * wy + uz * wz, e = vx * wx + vy * wy + vz * wz;
  const den = a * c - b * b;
  const cl = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
  let s = den > 1e-12 ? cl((b * e - c * d) / den) : 0;
  let t = c > 1e-12 ? (b * s + e) / c : 0;
  if (t < 0) { t = 0; s = a > 1e-12 ? cl(-d / a) : 0; }
  else if (t > 1) { t = 1; s = a > 1e-12 ? cl((b - d) / a) : 0; }
  pa.set(p1.x + s * ux, p1.y + s * uy, p1.z + s * uz);
  pb.set(p3.x + t * vx, p3.y + t * vy, p3.z + t * vz);
  return pa.distanceTo(pb);
}

/**
 * Əlin kapsulları 21 nöqtədən (MediaPipe sırası): avatarın sümükləri də, siqnalçının
 * landmark-ları da eyni hissələrə bölünür və eyni radiuslarla ölçülür. Hər barmağın üç
 * sümüyü (sonuncusu uca qədər), ovuc — bilək → barmaq kökləri yelpiyi və köklər xətti.
 */
function handCapsules(P, shape) {
  const out = [];
  Object.keys(FINGER_BONE_NAMES).forEach((name, i) => {
    const j = 1 + 4 * i;
    for (let k = 0; k < 3; k++) out.push([P[j + k], P[j + k + 1], shape.radius[name][k], name]);
  });
  for (const j of [5, 9, 13, 17]) out.push([P[0], P[j], shape.radius.palm, 'ovuc']);
  out.push([P[5], P[17], shape.radius.palm, 'ovuc'], [P[0], P[1], shape.radius.Thumb[0], 'ovuc']);
  return out;
}

/**
 * İki hissə dəsti (sol, sağ) arasında ən kiçik səth məsafəsi (mənfi = içəri girib), ən yaxın
 * nöqtələr və ayrılma oxu (soldan sağa).
 */
function closestParts(A, B, skip = null) {
  const pa = new THREE.Vector3(), pb = new THREE.Vector3();
  let best = null;
  for (const a of A) for (const b of B) {
    if (skip?.(a, b)) continue;
    const d = segmentClosest(a[0], a[1], b[0], b[1], pa, pb) - a[2] - b[2];
    if (!best || d < best.gap) best = { gap: d, pair: `${a[3]}/${b[3]}`, pa: pa.clone(), pb: pb.clone(), a, b };
  }
  if (!best) return null;
  const axis = best.pb.clone().sub(best.pa);
  // Parçalar kəsişirsə ən yaxın nöqtələr üst-üstə düşür və istiqamət itir:
  // onda toqquşan iki sümüyün mərkəzləri arasındakı ox götürülür.
  if (axis.lengthSq() < 1e-8) axis.copy(best.b[0]).add(best.b[1]).sub(best.a[0]).sub(best.a[1]);
  if (axis.lengthSq() < 1e-8) axis.set(1, 0, 0);
  return { gap: best.gap, pair: best.pair, axis: axis.normalize(), pa: best.pa, pb: best.pb, a: best.a, b: best.b };
}

/**
 * Əlin tərəfi (sol/sağ) landmark-ların həndəsəsindən: (şəhadət−bilək) × (çeçələ−bilək) ·
 * (baş barmaq−bilək) işarəsi. Sol əldə −1, sağ əldə +1 (MediaPipe-in dünya koordinatları).
 * Dərinlik güzgülənəndə işarə də dəyişir, ona görə bu, güzgünü aşkarlamağın birbaşa yoludur.
 */
function handChirality(w) {
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const i = sub(w[5], w[0]), l = sub(w[17], w[0]), t = sub(w[1], w[0]);
  const c = [i[1] * l[2] - i[2] * l[1], i[2] * l[0] - i[0] * l[2], i[0] * l[1] - i[1] * l[0]];
  return Math.sign(c[0] * t[0] + c[1] * t[1] + c[2] * t[2]) || 1;
}

/** Siqnalçının ovucunun biləkdən məsafəsi (əl ölçüsü üçün). */
function palmRadius(hand) {
  const off = [0, 1, 2].map((k) => (hand[5][k] + hand[9][k] + hand[17][k] - 3 * hand[0][k]) / 4);
  return Math.hypot(...off);
}

/** Siqnalçının iki əli arasındakı ən kiçik məsafə (metr): bilək + barmaq ucları. */
function handGap(fr) {
  const pts = (hand, wrist) => [0, 4, 8, 12, 16, 20].map(
    (j) => [0, 1, 2].map((k) => fr.b[wrist][k] + hand.w[j][k] - hand.w[0][k]));
  const L = pts(fr.L, 5), R = pts(fr.R, 6);
  let gap = Infinity;
  for (const a of L) for (const b of R) gap = Math.min(gap, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
  return gap;
}

/** Siqnalçının əlinin 21 nöqtəsi poza koordinatlarında (bilək + landmark-ların biləyə nisbəti). */
function signerPoints(fr, hand, wrist, flipDepth, scale) {
  return hand.w.map((p) => {
    const o = [0, 1, 2].map((k) => p[k] - hand.w[0][k]);
    if (flipDepth) o[2] = -o[2];
    return new THREE.Vector3(fr.b[wrist][0] + o[0], fr.b[wrist][1] + o[1], fr.b[wrist][2] + o[2])
      .multiplyScalar(scale);
  });
}

/**
 * Siqnalçının iki əli arasındakı ən kiçik səth məsafəsi, avatarın əl ölçüsünə miqyaslanıb və
 * avatarın öz kapsulları ilə — avatarın aralığı ilə birbaşa müqayisə olunur. Mütləq hədd yanlış
 * meyardır: bir çox işarədə əllər həqiqətən toxunur (datada 1-2 mm), bəzilərində isə 15 sm
 * aralıdır. Datada əllərin nisbi dərinliyi pozanın biləklərindən gəlir və səs-küylüdür:
 * toxunan əllər datada 1-3 sm bir-birinin içində görünə bilir (ölçülüb), ona görə mənfi
 * qiymət "toxunur" deməkdir, "girməlidir" yox.
 */
function signerGap(fr, flipL, flipR, shape, scale) {
  const parts = (hand, wi, flip, elbow) => {
    const o = handCapsules(signerPoints(fr, hand, wi, flip, scale), shape);
    const e = fr.b[elbow], w = fr.b[wi];
    if (e && w) {
      o.push([new THREE.Vector3(...e).multiplyScalar(scale), new THREE.Vector3(...w).multiplyScalar(scale),
              shape.radius.forearm, 'said']);
    }
    return o;
  };
  const L = parts(fr.L, 5, flipL, 3), R = parts(fr.R, 6, flipR, 4);
  const pa = new THREE.Vector3(), pb = new THREE.Vector3();
  let gap = Infinity;
  for (const a of L) for (const b of R) {
    gap = Math.min(gap, segmentClosest(a[0], a[1], b[0], b[1], pa, pb) - a[2] - b[2]);
  }
  return gap;
}

/**
 * Siqnalçının əl nöqtəsi pozanın koordinatlarında: pozanın biləyi + əl landmark-ının
 * biləyə nisbətən yeri (metrlə, oxlar eynidir). j = 'palm' — ovuc mərkəzi, avatarın
 * rig.palmOff-u ilə eyni nöqtələrdən: bilək, şəhadət, orta və çeçələ barmağın kökü.
 */
function signerPoint(wrist, hand, j, flipDepth) {
  const off = j === 'palm'
    ? [0, 1, 2].map((k) => (hand[5][k] + hand[9][k] + hand[17][k] - 3 * hand[0][k]) / 4)
    : [0, 1, 2].map((k) => hand[j][k] - hand[0][k]);
  if (flipDepth) off[2] = -off[2];
  return wrist.map((v, k) => v + off[k]);
}

export class VrmAvatar {
  #solved = new WeakMap();       // landmark kadrı → {left, right} həll (poza datası dəyişmir)
  #spikes = new WeakMap();       // dinamik poza → {left, right} kadr xəritəsi
  #words = new WeakMap();        // söz → kadrlar üzrə {left, right} həll
  #guard = null;                 // oynatma qoruyucusunun təkrar istifadə olunan vektorları
  #guardOff = null;              // qoruyucunun hər qol üçün saxlanan düzəlişi
  #euler = new THREE.Euler();
  #v = new THREE.Vector3();
  #q = new THREE.Quaternion();
  #q2 = new THREE.Quaternion();

  constructor(container) {
    this.container = container;
    this.vrm = null;
    this.hand = 'right';           // barmaq əlifbası üçün əsas əl
    this.rigs = null;
    this.springs = new Map();      // sümük adı → {node, q, w, base, offset, target, omega}
    this.mode = 'rest';            // rest | sign | pause
    this.idle = 0;
    this.time = 0;
    this.life = true;              // false → nəfəs/baş/göz qırpma söndürülür (ölçmə üçün)
    this.guard = true;             // false → oynatmada əllər arası qoruyucu söndürülür (ölçmə üçün)
    this.blink = { next: 1.5 + Math.random() * 2, t: -1 };
    // Üz: hədəf və cari ifadə çəkiləri, başın sözdəki fırlanması (radian, x/y/z)
    this.face = { target: {}, cur: {}, head: [0, 0, 0], headCur: [0, 0, 0] };
    this.faceCache = new WeakMap();
    this.lastPose = null;
    this.last = null;              // son həll (sazlama və ölçmə üçün)
    this.lastTime = performance.now();

    const w = container.clientWidth || 480, h = container.clientHeight || 480;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0e0c0a);

    this.camera = new THREE.PerspectiveCamera(34, w / h, 0.1, 40);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);

    this.scene.add(new THREE.AmbientLight(0xbfd4ff, 1.5));
    const key = new THREE.DirectionalLight(0xffffff, 2.0);
    key.position.set(1.2, 2.2, 2.4);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x3fc1e8, 1.4);
    rim.position.set(-1.8, 1.2, -1.6);
    this.scene.add(rim);
    const fill = new THREE.DirectionalLight(0xff4f97, 0.6);
    fill.position.set(2.0, 0.2, -1.2);
    this.scene.add(fill);

    this._onResize = () => this.resize();
    addEventListener('resize', this._onResize);
    this.#loop();
  }

  async load(url) {
    if (this.vrm) {
      this.scene.remove(this.vrm.scene);
      VRMUtils.deepDispose(this.vrm.scene);
      this.vrm = null;
    }
    // Keşlənmiş həllər əvvəlki avatarın sümük uzunluqları ilə hesablanıb
    this.#solved = new WeakMap();
    this.#spikes = new WeakMap();
    this.#words = new WeakMap();
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));
    const gltf = await loader.loadAsync(url);
    const vrm = gltf.userData.vrm;
    VRMUtils.removeUnnecessaryVertices(gltf.scene);
    VRMUtils.combineSkeletons(gltf.scene);
    vrm.scene.traverse((o) => { o.frustumCulled = false; });
    this.vrm = vrm;
    this.scene.add(vrm.scene);

    // Ölçmələr T-pozada aparılır (bütün normallaşdırılmış dönmələr hələ vahiddir)
    vrm.scene.updateMatrixWorld(true);
    this.rigs = {};
    this.presets = {};
    for (const side of SIDES) {
      const rig = measureRig(vrm.humanoid, side);
      this.rigs[side] = rig;
      const readyAnchor = signAnchor(rig, UP).add(new THREE.Vector3(0, -0.05, -0.02));
      this.presets[side] = {
        rest: restArm(rig),
        ready: solveArm(rig, readyHand(rig), readyAnchor),
        relaxed: relaxedFingers(rig),
      };
    }

    this.solid = this.#measureSolid();
    this.body = this.#measureBody(vrm.humanoid);
    this.handShape = this.#measureHands(vrm.humanoid);
    this.#guard = null;                // qoruyucu bədən ölçülərini yeni avatardan götürsün
    this.#guardOff = null;

    this.#initSprings();
    for (const side of SIDES) this.#toRest(side);
    this.snap();
    if (vrm.lookAt) vrm.lookAt.target = this.camera;   // göz izləyiciyə baxır
    this.#frameCamera();
    return vrm;
  }

  bone(name) {
    return this.vrm?.humanoid?.getNormalizedBoneNode(name) ?? null;
  }

  /** Aktiv əli dəyişir; digər qol istirahətə enir. */
  setHand(side) {
    this.hand = side === 'left' ? 'left' : 'right';
    if (!this.rigs) return;
    this.#toRest(this.hand === 'left' ? 'right' : 'left');
    this.#frameCamera();
    if (this.mode === 'sign' && this.lastPose) {
      this.setHandPose(this.lastPose.landmarks, this.lastPose.meta);
    } else if (this.mode === 'pause') {
      this.clearHandPose();
    }
  }

  /**
   * Hərf pozasını aktiv qola tətbiq edir (söz işarələri üçün setWordPose).
   * @param {Array<{x,y,z}>|null} landmarks 21 normallaşdırılmış nöqtə (null = hazır vəziyyət)
   * @param {{pose?: object, frame?: number}} meta poses.json-dakı kadr indeksi
   */
  setHandPose(landmarks, meta = {}) {
    if (!landmarks || landmarks.length !== 21) { this.clearHandPose(); return; }
    this.lastPose = { landmarks, meta };
    if (!this.rigs) return;

    const side = this.hand;
    const { pose } = meta;
    const asked = meta.frame ?? 0;
    const frame = this.#frameFor(pose, asked, side);
    const lm = frame === asked ? landmarks : pose.frames[frame];
    const cached = this.#solved.get(lm);
    let sol = cached?.[side];
    if (!sol) {
      // dinamik hərfdə əvvəlki kadrın həlli verilir — dirsək kadrdan kadra sıçramasın
      const prev = frame > 0 && this.last?.pose === pose && this.last.side === side ? this.last.arm.info : null;
      sol = this.#solve(side, lm, pose, frame, prev);
      if (!sol) return;
      this.#solved.set(lm, { ...cached, [side]: sol });
    }
    this.#setSide(side, sol.arm, sol.fingers);
    this.#toRest(side === 'left' ? 'right' : 'left');
    this.mode = 'sign';
    this.idle = 0;
    this.last = { side, pose, frame, ...sol };
    this.#frameCamera();
  }

  /**
   * Dinamik hərfdə tək kadrlıq sıçrayış: ovuc hər iki qonşu kadrdan 60°-dən çox
   * fərqlənir, qonşular isə bir-birinə yaxındır. 70 ms-də bu real hərəkət deyil,
   * MediaPipe səs-küyüdür — həmin kadr əvvəlkini təkrarlayır.
   */
  #frameFor(pose, frame, side) {
    if (pose?.type !== 'dynamic' || !(pose.frames?.length > 2)) return frame;
    let map = this.#spikes.get(pose)?.[side];
    if (!map) {
      const rig = this.rigs[side];
      const R = pose.frames.map((_, f) => handTarget(rig, frameSource(pose, f, side)).R);
      const ang = (a, b) => 2 * Math.acos(Math.min(1, Math.abs(a.dot(b))));
      const limit = THREE.MathUtils.degToRad(60);
      map = pose.frames.map((_, f) => f);
      for (let f = 1; f < R.length - 1; f++) {
        if (ang(R[f], R[f - 1]) > limit && ang(R[f], R[f + 1]) > limit && ang(R[f - 1], R[f + 1]) < limit * 0.75) {
          map[f] = map[f - 1];
        }
      }
      this.#spikes.set(pose, { ...this.#spikes.get(pose), [side]: map });
    }
    return map[frame] ?? frame;
  }

  #solve(side, landmarks, pose, frame, prev) {
    const rig = this.rigs[side];
    // Əl həndəsəsi avatarın tərəfinə gətirilir (sol əl datası sağ əldə güzgülənir
    // və əksinə), MediaPipe-in dərinliyi tərs qiymətləndirdiyi kadrlar düzəldilir
    const src = frameSource(pose, frame, side, landmarks);
    const target = handTarget(rig, src);
    const anchor = signAnchor(rig, target.d);
    const path = trajectory(pose);
    if (path) {
      const step = path[frame];
      anchor.add(new THREE.Vector3(src.mirror ? -step.x : step.x, step.y, step.z));
    }
    const arm = solveArm(rig, target.R, anchor, prev);
    if (!arm) return null;
    return {
      target, arm, anchor, mirror: src.mirror,
      fingers: solveFingers(rig, target.shape, target.shapeRinv),
    };
  }

  /**
   * Söz işarəsinin bir kadrı — hər iki qol (format: tools/build_words.py).
   * Əlin yeri siqnalçının çiyinlərinə nisbətən ölçülüb avatarın çiyinlərinə
   * köçürülür: yanağa, çənəyə, sinəyə toxunan işarələr öz yerində qalır.
   * Əlin forması və oriyentasiyası hərflərdəki kimi landmark-lardan gəlir.
   */
  setWordPose(word, frame) {
    if (!this.rigs) return;
    const sol = this.#wordFrame(word, frame);
    for (const side of SIDES) {
      if (sol[side]) this.#setSide(side, sol[side].arm, sol[side].fingers);
      else this.#toRest(side);
    }
    this.mode = 'sign';
    this.idle = 0;
    this.last = { word, frame, ...sol };
    this.#frameCamera('word', this.#wordBounds(word));
    const face = this.#wordFace(word)[frame];
    this.face.target = face?.w ?? {};
    this.face.head = face?.h ?? [0, 0, 0];
  }

  /**
   * Sözün kadrlarında üz: ifadə çəkiləri və başın fırlanması. Blendshape lüğət üzrə medianı
   * (`faceNeutral`, sakit üz) ilə 99-cu faizi (`faceFull`) arasında ölçülür. Başın fırlanmasından
   * sözün öz medianı çıxılır: kameraya görə duruş siqnalçıdan asılıdır, sözün içindəki
   * hərəkət (baş sallama, yelləmə, əyilmə) isə işarəyə aiddir.
   */
  #wordFace(word) {
    if (this.faceCache.has(word)) return this.faceCache.get(word);
    const keys = word.faceKeys ?? [];
    const idx = new Map(keys.map((k, i) => [k, i]));
    const neutral = word.faceNeutral ?? keys.map(() => 0);
    const full = word.faceFull ?? keys.map(() => 100);
    const heads = word.frames.filter((fr) => fr.f).map((fr) => fr.f.h);
    const med = [0, 1, 2].map((k) => {
      const v = heads.map((h) => h[k]).sort((a, b) => a - b);
      return v.length ? v[v.length >> 1] : 0;
    });
    const out = word.frames.map((fr) => {
      if (!fr.f || !keys.length) return null;
      const val = (k) => {
        const i = idx.get(k);
        return i == null ? 0 : Math.min(1, Math.max(0, (fr.f.s[i] - neutral[i]) / Math.max(5, full[i] - neutral[i])));
      };
      const w = {};
      for (const [name, [src, cap]] of Object.entries(FACE_MAP)) {
        w[name] = cap * src.reduce((a, k) => a + val(k), 0) / src.length;
      }
      const lips = VISEMES.reduce((a, k) => a + w[k], 0);
      if (lips > 1) for (const k of VISEMES) w[k] /= lips;
      const h = fr.f.h.map((v, k) => Math.max(-HEAD_MAX, Math.min(HEAD_MAX, (v - med[k]) * DEG)));
      return { w, h };
    });
    this.faceCache.set(word, out);
    return out;
  }

  /** Sözü əvvəlcədən həll edir (oynatmadan qabaq, ilk kadrda ləngimə olmasın). */
  prepareWord(word) {
    if (this.rigs) this.#wordBounds(word);
  }

  /**
   * Söz kadrında hər əl üçün hansı data kadrı və dərinlik variantı seçilib.
   * Cyber skeleton eyni şeyi göstərsin deyə lazımdır: avatar MediaPipe-in güzgülədiyi
   * dərinliyi düzəldir və tək kadrlıq sıçrayışı oynatmır, xam data isə bunları bilmir.
   */
  wordHands(word, f) {
    if (!this.rigs) return null;
    const sol = this.#wordSolve(word)[f];
    if (!sol) return null;
    const out = {};
    for (const side of SIDES) {
      if (sol[side]) out[side] = { frame: sol[side].frame, flip: sol[side].flip };
    }
    return out;
  }

  #wordFrame(word, f) {
    return this.#wordSolve(word)[f];
  }

  /**
   * Sözün bütün kadrları həll olunur: hər kadrda əlin tərəfi landmark-lardan oxunur
   * (MediaPipe dərinliyi güzgü həll edəndə düzəldilir), qol iki-sümüklü IK ilə qurulur,
   * tək kadrlıq sıçrayış süzülür və iki əl bir-birinə girməsin deyə düzəldilir.
   */
  /** Nöqtənin bədənə girmə dərinliyi (müsbət = içəridə) — IK-nın qiymət funksiyası üçün. */
  #avoidBody() {
    const s = this.solid;
    if (!s) return null;
    const head = s.head;
    const step = s.slices.length > 1 ? s.slices[1].y - s.slices[0].y : 1;
    const y0 = s.slices[0].y;
    // Yalnız ədəd qaytaran sürətli yol: IK-nın döngəsində obyekt yaratmaq olmaz
    return (p) => {
      let dist = Infinity;
      if (head) {
        const dx = p.x - head.center.x, dy = p.y - head.center.y, dz = p.z - head.center.z;
        dist = Math.sqrt(dx * dx + dy * dy + dz * dz) - head.radius;
      }
      if (p.y >= s.yMin && p.y <= s.yMax) {
        let i = Math.round((p.y - y0) / step);
        i = i < 0 ? 0 : i >= s.slices.length ? s.slices.length - 1 : i;
        const sl = s.slices[i];
        const u = (p.x - sl.cx) / sl.rx, w = (p.z - sl.cz) / sl.rz;
        const k = Math.sqrt(u * u + w * w) || 1e-6;
        const ex = (u / k) * sl.rx, ez = (w / k) * sl.rz;
        const d = (k - 1) * Math.sqrt(ex * ex + ez * ez);
        if (d < dist) dist = d;
      }
      return 0.025 - dist;                                  // 2.5 sm sümük qalınlığı payı
    };
  }

  #wordSolve(word) {
    let sol = this.#words.get(word);
    if (sol) return sol;
    sol = word.frames.map(() => ({ left: null, right: null }));
    const places = word.frames.map((fr) => (fr.b ? this.#signerToAvatar(fr.b) : null));
    const scale = this.#wordScale(word);
    const avoid = this.#avoidBody();
    const ang = (a, b) => 2 * Math.acos(Math.min(1, Math.abs(a.dot(b))));   // iki dönmə arasında bucaq

    for (const [side, key, wrist] of [['left', 'L', 5], ['right', 'R', 6]]) {
      const rig = this.rigs[side];
      const pick = word.frames.map((fr, f) => {
        const hand = fr[key];
        if (!hand || !fr.b) return null;
        const place = places[f];
        // Dizin üstündə dayanan (işarədə iştirak etməyən) əl istirahətə enir; baza əl
        // (biler, saat, bayraq) bu həddən yuxarıda qalır
        if (place.height(fr.b[wrist]) < -1.05) return null;
        const norm = hand.n.map(([x, y, z]) => ({ x, y, z }));
        // MediaPipe tək kameradan dərinliyi bəzi kadrlarda güzgü həll edir — onda landmark
        // dəsti qarşı əlin həndəsəsinə çevrilir və barmaqlar avatarda geriyə bükülür.
        // Əlin tərəfi landmark-lardan oxunur və rig-in tərəfi ilə uyğunlaşdırılır.
        const want = side === 'left' ? -1 : 1;
        const flip = handChirality(hand.w) !== want;            // metrik landmark-lar (barmaq forması)
        const flipN = handChirality(hand.n) !== want;           // şəkil landmark-ları (oriyentasiya)
        let target = handTarget(rig, { world: hand.w, norm, size: word.size, flipWorld: flip, flipNorm: flipN });
        // Biləyin oriyentasiyası şəkil koordinatlarından, barmaqların forması isə metrik
        // landmark-lardan gəlir. İki mənbə kəskin fərqlənirsə, MediaPipe-in metrik
        // rekonstruksiyası etibarsızdır — bilək bir tərəfə, barmaqlar başqa tərəfə baxır və
        // oynaqlar həddə dirənir (ölçülüb: fərq 60°-dən çox olanda kadrların yarıdan çoxu
        // pozulur, 20°-dən az olanda 1%-i). Belə kadrlarda barmaqlar da şəkildən qurulur.
        const frameGap = 2 * Math.acos(Math.min(1, Math.abs(target.R.clone().multiply(target.shapeRinv).w)));
        if (frameGap > 50 * DEG) target = handTarget(rig, { norm, size: word.size, flipDepth: flipN });
        let anchor = this.#keepOut(this.#wordPalm(place, fr.b[wrist], hand.w, flip, rig, scale));
        const elbow = place.body(fr.b[wrist - 2]);
        let arm = solveArm(rig, target.R, anchor, null, { elbow, shiftCost: 300, avoid });
        // Barmaqların forması metrik landmark-lardan məcburi götürülür (əlin tərəfi düzgün
        // olsun), oriyentasiya isə şəkil landmark-larının dərinliyindən gəlir və o, səs-küylü
        // ola bilər. Bilək həmin oriyentasiyaya çatmırsa, dərinliyin əks variantı yoxlanılır:
        // əlin forması dəyişmir, yalnız biləyin çata biləcəyi istiqamət seçilir.
        if (arm && arm.info.hard > 0.03) {
          const alt = handTarget(rig, { world: hand.w, norm, size: word.size, flipWorld: flip, flipNorm: !flipN });
          const altAnchor = this.#keepOut(this.#wordPalm(place, fr.b[wrist], hand.w, flip, rig, scale));
          const altArm = solveArm(rig, alt.R, altAnchor, null, { elbow, shiftCost: 300, avoid });
          if (altArm && altArm.info.hard < arm.info.hard - 0.01) {
            target = alt; anchor = altAnchor; arm = altArm;
          }
        }
        if (!arm) return null;
        // goal — datadan gələn ilkin yer: sonrakı düzəlişlər anchor-u dəyişir, ölçmə ona baxır
        return { target, arm, anchor, goal: anchor.clone(), elbow, flip, frame: f, cost: arm.info.cost };
      });

      // Tək kadrlıq sıçrayış (MediaPipe əli bir kadr səhv tapır): üç əlaməti var —
      // həll qonşulardan qat-qat baha; əl hər iki qonşudan kəskin fərqli oriyentasiyada,
      // qonşular isə bir-birinə yaxın; ya da ovuc bir kadrda çıxıb geri qayıdır (ölçülüb:
      // 41 sm-ə qədər atılma). Belə kadr qonşuların ortası ilə əvəz olunur — pozanı
      // saxlamaqdan fərqli olaraq hərəkət hamar qalır.
      const R = (f) => pick[f].target.R;
      const spike = pick.map((c, f) => {
        const a = pick[f - 1], b = pick[f + 1];
        if (!c || !a || !b) return false;
        const jump = c.anchor.distanceTo(a.anchor), back = b.anchor.distanceTo(a.anchor);
        const turn = ang(R(f), R(f - 1)), backTurn = ang(R(f - 1), R(f + 1));
        return (c.cost > 4 && c.cost > 3 * Math.max(a.cost, b.cost)) ||
          (turn > 50 * DEG && ang(R(f), R(f + 1)) > 50 * DEG && backTurn < 35 * DEG) ||
          (turn > 45 * DEG && turn > 2.5 * backTurn) ||
          (jump > 0.05 && jump > 2.5 * back);
      });
      pick.forEach((c, f) => {
        if (!c) return;
        let use = c;
        if (spike[f]) {
          const a = pick[f - 1], b = pick[f + 1];
          const mid = a.target.R.clone().slerp(b.target.R, 0.5);
          const anchor = a.anchor.clone().lerp(b.anchor, 0.5);
          const elbow = a.elbow && b.elbow ? a.elbow.clone().lerp(b.elbow, 0.5) : c.elbow;
          const arm = solveArm(rig, mid, anchor, null, { elbow, shiftCost: 300, avoid });
          use = arm
            ? { ...a, target: { ...a.target, R: mid }, anchor, goal: anchor.clone(), elbow, arm, frame: f }
            : a;
        }
        sol[f][side] = { ...use, fingers: solveFingers(rig, use.target.shape, use.target.shapeRinv) };
      });
    }
    this.#fixBody(word, sol);
    this.#fixTwoHanded(word, sol, places, scale);
    this.#fixBody(word, sol);
    // Hamarlama ən sonda: əvvəlki düzəlişlər özləri də tək kadrlıq atılma yarada bilir.
    // İki keçid, çünki yan-yana iki pozulmuş kadrda ikinci keçid artıq təmiz qonşuya
    // əsaslanır. Yenidən həll bədəni nəzərə alır, ona görə qol bədənə qaytarılmır.
    this.#fixSpikes(word, sol, avoid);
    this.#fixSpikes(word, sol, avoid);
    // Əllər bir də ayrılır: bədən düzəlişi və hamarlama qolları bir-bir yenidən həll edir və
    // qarşı əli görmür — artıq ayrılmış əllər yenidən bir-birinə girirdi (ölçülüb: "gədə"-nin
    // 5-ci kadrı +1.2 sm-dən −2.3 sm-ə). Nisbi həndəsə burada təkrar köçürülmür, yalnız ayrılır.
    this.#fixTwoHanded(word, sol, places, scale, { relative: false });
    this.#words.set(word, sol);
    return sol;
  }

  /**
   * İki əlli işarələrdə əllər bir-birinin (və qarşı saidin) içinə girirdi: hər qol ayrıca
   * həll olunur, iki müstəqil xəta toplanır. İki addımda düzəldilir:
   *   1. Əllər yaxındırsa (toxunan işarə), aralarındakı vektor datadan götürülür — cüt
   *      bütövlükdə bədənə görə yerləşir, amma bir-birinə nisbəti qorunur.
   *   2. Poza həqiqətən qurulub ölçülür: barmaq, ovuc və qol kapsulları arasındakı ən kiçik
   *      SƏTH məsafəsi. Mənfidirsə əllər bir istiqamət boyunca ayrılır və qollar yenidən
   *      həll olunur. Hədəf datadan gəlir: siqnalçıda əllər toxunursa yalnız səthlər
   *      ayrılır, aralıdırsa tam aralıq bərpa olunur.
   * Hər addımda ölçülür və yalnız ən yaxşı nəticə saxlanılır — düzəlmirsə ilkin poza qalır.
   * `relative: false` — yalnız ayırma (son keçid: yerləşmə artıq bədənə görə düzəldilib).
   */
  #fixTwoHanded(word, sol, places, scale, { relative = true } = {}) {
    const touched = new Set();
    const NEAR = 0.16;        // m — siqnalçıda bundan yaxın əllər "toxunan" sayılır
    const MESH = 0.002;       // m — səthlər toxunur, amma bir-birinin içinə girmir
    const FULL = 0.020;       // m — datada əllər aralıdırsa tələb olunan aralıq
    for (let f = 0; f < word.frames.length; f++) {
      const L = sol[f].left, R = sol[f].right;
      const fr = word.frames[f];
      if (!L || !R || !fr.L || !fr.R || !fr.b) continue;
      const place = places[f];

      // 1. Nisbi həndəsə: siqnalçının ovucları arasındakı vektor əl ölçüsünə görə köçürülür
      const sg = handGap(fr);
      if (relative && sg < NEAR) {
        const pl = signerPoint(fr.b[5], fr.L.w, 'palm', L.flip);
        const pr = signerPoint(fr.b[6], fr.R.w, 'palm', R.flip);
        const rel = place.dir(pr.map((v, k) => v - pl[k])).multiplyScalar(scale);
        const mid = L.anchor.clone().add(R.anchor).multiplyScalar(0.5);
        const wantL = mid.clone().addScaledVector(rel, -0.5);
        const wantR = mid.clone().addScaledVector(rel, 0.5);
        // Hər əl ən çoxu 4 sm tərpənir: data ilə bədən xəritəsi arasındakı böyük fərq
        // adətən pozanın səhvidir. İstisna: data əlləri indikindən açıq-aşkar aralı
        // göstərirsə, pozanı qorumaq əlləri bir-birinin içində saxlamaq deməkdir.
        const cap = rel.length() - L.anchor.distanceTo(R.anchor) > 0.03 ? 0.07 : 0.04;
        if (wantL.distanceTo(L.anchor) < cap && wantR.distanceTo(R.anchor) < cap) {
          this.#reanchor(L, 'left', wantL);
          this.#reanchor(R, 'right', wantR);
        }
      }

      // 1.5. Qarşı qol IK-nın qiymətinə maneə kimi verilir. Bədəndə ölçülüb: maneəni
      //      həllin içində nəzərə almaq sonradan kənardan itələməkdən qat-qat effektivdir,
      //      çünki bilək həddi (hard) həlli yerində kilidləyir və itələmə onu tərpədə bilmir.
      const bodyAvoid = this.#avoidBody();
      // Əlləri ayırmaq naminə qolu bədənə salmaq olmaz: hər dəyişiklikdən sonra bədən aralığı
      // da ölçülür və əvvəlkindən dərinə girən variant rədd olunur. Son keçiddən sonra bədən
      // düzəlişi yoxdur, ona görə bu yoxlanış olmasa ayrılan əl sinəyə, said gövdəyə girirdi
      // (ölçülüb: 2 sm-ə qədər).
      let bodyFloor = null;                           // {left, right}: bədən aralığının alt həddi
      const bodyOk = () => {
        bodyFloor ??= {
          left: Math.min(this.#bodyGap('left', L).gap, 0) - 0.003,
          right: Math.min(this.#bodyGap('right', R).gap, 0) - 0.003,
        };
        return this.#bodyGap('left', L).gap >= bodyFloor.left && this.#bodyGap('right', R).gap >= bodyFloor.right;
      };
      const probe0 = this.#probeGap(L, R);
      if (probe0 && probe0.gap < 0) {
        bodyOk();                                     // başlanğıc bədən aralığı yadda saxlanır
        for (const [side, c, oside, other] of [['left', L, 'right', R], ['right', R, 'left', L]]) {
          const caps = prepareCapsules(
            this.#withPose([[oside, other]], () => this.#contactParts(oside)), 0.012);
          const avoid = bodyAvoid
            ? (p) => Math.max(bodyAvoid(p), capsuleDepth(caps, p))
            : (p) => capsuleDepth(caps, p);
          const arm = solveArm(this.rigs[side], c.target.R, c.anchor, null,
            { elbow: c.elbow, shiftCost: 300, avoid });
          if (!arm || arm.info.hard > Math.min(HARD_CEIL, Math.max(0.04, c.arm.info.hard + 0.03))) continue;
          const before = this.#probeGap(L, R)?.gap ?? -9;
          const keep = c.arm;
          c.arm = arm;
          const after = this.#probeGap(L, R)?.gap ?? -9;
          if (after <= before + 0.001 || !bodyOk()) c.arm = keep;
          else touched.add(f);
        }
      }

      // 2. Qalan toqquşma: real pozada ölçülüb ayrılır. Hədəf aralıq datadan gəlir —
      //    toxunan işarədə yalnız həndəsənin tələb etdiyi qədər (barmaq üst-üstə düşməsin),
      //    datada aralı olan işarədə tam aralıq. Ayırma oxu iki ovucun arasıdır: ən yaxın
      //    nöqtələrin oxu əli bədənə və ya arxaya itələyə bilər.
      const sb = signerGap(fr, L.flip, R.flip, this.handShape, scale);
      const want = THREE.MathUtils.clamp(sb, MESH, FULL);
      const palmAxis = R.anchor.clone().sub(L.anchor);
      if (palmAxis.lengthSq() < 1e-8) continue;
      palmAxis.normalize();
      const first = this.#probeGap(L, R);
      if (!first) continue;
      if (first.gap < want) bodyOk();
      // Büdcə girmənin dərinliyinə görə: əl saidin içinə 4 sm girəndə 2.5 sm kifayət etmir
      const budget = THREE.MathUtils.clamp(want - first.gap + 0.02, 0.025, 0.08);
      const snap = () => ({ la: L.anchor.clone(), lm: L.arm, ra: R.anchor.clone(), rm: R.arm });
      const restore = (k) => { L.anchor = k.la; L.arm = k.lm; R.anchor = k.ra; R.arm = k.rm; };
      let best = { gap: first.gap, ...snap() };
      let moved = 0, boost = 1, stuck = 0, sepDir = null;
      // Namizəd istiqamətlər: toqquşma oxu (və ya ovuc oxu), sonra ovuc oxu, önə/arxaya,
      // yuxarı/aşağı. Əllər bir-birinin önündə olanda ovuc oxu arxadakı əli sinəyə itələyir:
      // bədən yoxlanışı onu saxlayır və nisbi məsafəni öndəki əl təkbaşına açır.
      const dirs = [palmAxis, new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, -1),
                    new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, -1, 0)];
      const nextDir = () => {
        const d = dirs.shift();
        if (!d) return false;
        restore(best);                                // yeni istiqamət ən yaxşı vəziyyətdən başlayır
        sepDir = d; moved = 0; boost = 1; stuck = 0;
        return true;
      };
      const floor = bodyFloor ?? {};
      for (let iter = 0; iter < 14 && moved < budget; iter++) {
        const probe = this.#probeGap(L, R);
        if (!probe || probe.gap >= want) break;
        // Toqquşma qarşı tərəfin saidi ilədirsə, əvvəlcə dirsək fırladılır: bilək və əlin
        // forması olduğu yerdə qalır, yalnız qol kənara çıxır — işarə heç dəyişmir.
        if (this.#armAside(L, R, probe)) {
          const aside = this.#probeGap(L, R);
          if (aside && aside.gap > best.gap + 0.0005 && bodyOk()) best = { gap: aside.gap, ...snap() };
          if (aside && aside.gap >= want) break;
          continue;
        }
        // İstiqamət sabit qalır, növbətiyə yalnız iki addım irəliləyiş olmayanda keçilir. Hər
        // addımda ən dərin toqquşmanın oxunu götürmək təsadüfi gəzintiyə çevrilirdi: əllər
        // 36 mm tərpənir, amma oxlar bir-birini kompensasiya etdiyi üçün aralıq açılmırdı (ölçülüb).
        if (!sepDir) {
          sepDir = probe.axis.dot(palmAxis) < -0.5 ? dirs.shift() : probe.axis.clone();
        } else if (stuck >= 2 && !nextDir()) {
          break;
        }
        const dir = sepDir;
        const push = Math.min(((want - probe.gap) / 2 + 0.003) * boost, budget - moved);
        // Əllər bir-birinin içindədirsə, biləyin həddinə bir az güzəşt etmək iki əlin
        // üst-üstə düşməsindən yaxşıdır (həddi aşan hissə solveArm-da onsuz da kəsilir).
        const slack = probe.gap < -0.01 ? 0.05 : probe.gap < 0 ? 0.03 : 0.01;
        const okL = this.#reanchor(L, 'left', L.anchor.clone().addScaledVector(dir, -push), 8000, slack, floor.left);
        const okR = this.#reanchor(R, 'right', R.anchor.clone().addScaledVector(dir, push), 8000, slack, floor.right);
        // Bir tərəf tərpənə bilmirsə (bilək həddi və ya bədən), nisbi məsafəni digər tərəf
        // iki qat gedərək qapayır
        if (okL !== okR) {
          const c = okL ? L : R, side = okL ? 'left' : 'right', sgn = okL ? -1 : 1;
          this.#reanchor(c, side, c.anchor.clone().addScaledVector(dir, sgn * push), 8000, slack, floor[side]);
        }
        if (!okL && !okR) {
          // Bu istiqamət bileyi həddindən çıxarır (ölçülüb: hard 0.005 → 0.13) və ya əli
          // bədənə salır — növbəti namizəd sınanır
          if (nextDir()) continue;
          break;                                      // qol daha yaxşı həll tapmır
        }
        moved += push;
        const after = this.#probeGap(L, R);
        if (after && after.gap > best.gap + 0.0005 && bodyOk()) {
          best = { gap: after.gap, ...snap() };
          boost = 1;
          stuck = 0;
        } else {
          // Ən yaxın cüt dəyişdi, aralıq açılmadı: növbəti addım daha güclü olur
          boost *= 1.8;
          stuck++;
        }
      }
      // Ən yaxşı ölçülmüş vəziyyət saxlanır: heç nə düzəlməyibsə ilkin poza qalır,
      // yəni əlləri ayırmaq cəhdi işarəni korlamır
      restore(best);
      if (best.gap !== first.gap) touched.add(f);
    }
    return touched;
  }

  /**
   * Yekun hamarlama. Bütün düzəlişlərdən sonra əlin yolunda tək kadrlıq atılma qala bilər
   * (əl bir kadrda çıxıb dərhal geri qayıdır — insan belə hərəkət etmir, bu, tanımanın
   * səhvidir). Belə kadr qonşuların ortası ilə əvəz olunur: həm hərəkət hamarlanır, həm
   * işarənin gedişi qorunur.
   */
  #fixSpikes(word, sol, avoid) {
    for (const side of SIDES) {
      for (let f = 1; f < word.frames.length - 1; f++) {
        const a = sol[f - 1][side], c = sol[f][side], b = sol[f + 1][side];
        if (!a || !b || !c) continue;
        const jump = c.arm.info.W.distanceTo(a.arm.info.W);
        const back = b.arm.info.W.distanceTo(a.arm.info.W);
        const turn = quatAngle(c.target.R, a.target.R);
        const backTurn = quatAngle(b.target.R, a.target.R);
        if (!(jump > 0.045 && jump > 2.5 * back) && !(turn > 45 * DEG && turn > 2.5 * backTurn)) continue;
        const mid = a.target.R.clone().slerp(b.target.R, 0.5);
        const anchor = this.#keepOut(a.anchor.clone().lerp(b.anchor, 0.5));
        const elbow = a.elbow && b.elbow ? a.elbow.clone().lerp(b.elbow, 0.5) : c.elbow;
        const arm = solveArm(this.rigs[side], mid, anchor, null, { elbow, shiftCost: 300, avoid });
        // Hamarlıq naminə biləyi qırmaq olmaz: qonşulardan nəzərəçarpacaq pis həll rədd olunur
        if (!arm || arm.info.hard > Math.max(0.05, a.arm.info.hard + 0.02)) continue;
        sol[f][side] = {
          ...a, target: { ...a.target, R: mid }, anchor, elbow, arm, frame: f,
          goal: a.goal.clone().lerp(b.goal, 0.5),
          fingers: solveFingers(this.rigs[side], a.target.shape, a.target.shapeRinv),
        };
      }
    }
  }

  /** Bir qolun bədənə girməsi: pozanı tətbiq edib ölçür. */
  #bodyGap(side, c) {
    return this.#withPose([[side, c]], () => this.#bodyProbe(side));
  }

  /**
   * Qol bədənin içindən keçməsin. Real insanda dirsək gövdəyə çatanda yana açılır, ona görə
   * əvvəlcə qolun artıq sərbəstlik dərəcəsi işlədilir (bilək və əlin forması dəyişmir);
   * yalnız bu kifayət etməyəndə ovuc bədənin səthindən bayıra çəkilir. Hər addım ölçülür,
   * yaxşılaşma yoxdursa poza olduğu kimi qalır.
   */
  #fixBody(word, sol, only = null) {
    if (!this.solid) return;
    const avoid = this.#avoidBody();
    for (let f = 0; f < word.frames.length; f++) {
      if (only && !only.has(f)) continue;
      for (const side of SIDES) {
        const c = sol[f][side];
        if (!c) continue;
        const start = this.#bodyGap(side, c);
        if (!start.normal || start.gap >= 0) continue;
        let best = { gap: start.gap, anchor: c.anchor.clone(), arm: c.arm };
        const start0 = c.arm.info;
        const hardCap = Math.min(HARD_CEIL, Math.max(0.04, c.arm.info.hard + 0.03));

        // 1. Dirsəyin yeri: çiyin→bilək oxu ətrafında tam süpürgə. Dirsək bu oxun ətrafında
        //    çevrə cızır (qolun artıq sərbəstlik dərəcəsi) — bilək və əl yerində qalır, ona
        //    görə işarə dəyişmir. Ölçülüb: bədəndən keçən kadrlarda dirsək gövdənin o biri
        //    tərəfinə və kürəyin arxasına düşür, kiçik itələmə onu oradan çıxarmır.
        const rig = this.rigs[side];
        const toW = c.arm.info.W.clone().sub(rig.S);
        const reach = toW.length();
        const axis = toW.clone().normalize();
        const down = new THREE.Vector3(0, -1, 0).addScaledVector(axis, axis.y).normalize();
        const side2 = new THREE.Vector3().crossVectors(axis, down).normalize();
        const dataElbow = c.elbow ?? c.arm.info.E;
        // Namizədlər IK-nın özünün hesabladığı maneə qiyməti ilə sıralanır (arm.info.body):
        // bu, pulsuzdur, tam ölçmə isə skeletin bütün sümüklərini yeniləməyi tələb edir və
        // söz həllini saniyələrlə uzadır (ölçülüb).
        const score = (arm, gap) => gap * (gap < 0 ? 12 : 1)
          - 3 * Math.max(0, arm.info.hard - start0.hard)
          - 0.8 * arm.info.E.distanceToSquared(dataElbow);
        // Hər namizəd skelet üzərində həqiqətən ölçülür: ucuz göstərici (IK-nın maneə
        // qiyməti) yalnız beş nöqtəyə baxır və barmaqları görmür — onunla sıralamaq
        // nəticəni pisləşdirir (ölçülüb: 51 → 81 problemli söz).
        let bestScore = score(c.arm, best.gap);
        for (let k = 0; k < 12; k++) {
          const a = k * 30 * DEG;
          const dir = down.clone().multiplyScalar(Math.cos(a)).addScaledVector(side2, Math.sin(a));
          const guide = rig.S.clone().addScaledVector(axis, reach * 0.55).addScaledVector(dir, reach * 0.45);
          const arm = solveArm(rig, c.target.R, c.anchor, null, { elbow: guide, shiftCost: 8000, avoid });
          if (!arm || arm.info.hard > hardCap) continue;
          const keepArm = c.arm;
          c.arm = arm;
          const g = this.#bodyGap(side, c);
          const sc = score(arm, g.gap);
          if (sc > bestScore) { bestScore = sc; best = { gap: g.gap, anchor: c.anchor.clone(), arm }; }
          c.arm = keepArm;
        }
        c.anchor = best.anchor.clone();
        c.arm = best.arm;

        // 1b. Dirsəyin yeri çox vaxt əlin oriyentasiyası ilə kilidlənir: qol bədəndən
        //     kənara çıxanda bilək həddi aşılır (ölçülüb — `hard` qiyməti 100-ə qalxır,
        //     dirsək yönləndirməsi 4-dür, ona görə heç nəyi dəyişə bilmir). Çıxış yolu
        //     əlin özünü önqol oxu ətrafında bir neçə dərəcə döndərməkdir: bu, işarənin
        //     oxunuşuna təsir etməyən sapmadır, qolun bədəndən keçməsi isə açıq səhvdir.
        if (best.gap < 0) {
          const fAxis = c.arm.info.W.clone().sub(c.arm.info.E).normalize();
          const nAxis = rig.n0.clone().applyQuaternion(c.target.R).normalize();
          for (const [ax, deg] of [[fAxis, -12], [fAxis, 12], [fAxis, -24], [fAxis, 24],
                                   [fAxis, -36], [fAxis, 36], [nAxis, -15], [nAxis, 15]]) {
            const R2 = new THREE.Quaternion().setFromAxisAngle(ax, deg * DEG).multiply(c.target.R);
            const arm = solveArm(rig, R2, c.anchor, null, { elbow: c.elbow, shiftCost: 8000, avoid });
            if (!arm || arm.info.hard > hardCap) continue;
            const keepArm = c.arm, keepR = c.target.R;
            c.arm = arm; c.target = { ...c.target, R: R2 };
            const g = this.#bodyGap(side, c);
            if (g.gap > best.gap + 0.002) {
              best = { gap: g.gap, anchor: c.anchor.clone(), arm, R: R2 };
              if (g.gap >= 0) break;
            }
            c.arm = keepArm; c.target = { ...c.target, R: keepR };
          }
          if (best.R) c.target = { ...c.target, R: best.R };
          c.arm = best.arm;
        }

        // 1c. Hələ də bədəndən keçirsə, bu kadrın oriyentasiyası etibarsızdır (MediaPipe
        //     tək kadrda dərinliyi səhv həll edir). Qonşu kadrların ortası götürülür:
        //     həm anatomik olaraq mümkün olur, həm hərəkət hamar qalır.
        if (best.gap < -0.01 && f > 0 && f < word.frames.length - 1) {
          const prev = sol[f - 1][side], next = sol[f + 1][side];
          if (prev && next) {
            const R2 = prev.target.R.clone().slerp(next.target.R, 0.5);
            const arm = solveArm(rig, R2, c.anchor, null, { elbow: c.elbow, shiftCost: 8000, avoid });
            if (arm && arm.info.hard <= HARD_CEIL) {
              const keepArm = c.arm, keepR = c.target.R;
              c.arm = arm; c.target = { ...c.target, R: R2 };
              const g = this.#bodyGap(side, c);
              if (g.gap > best.gap + 0.002) best = { gap: g.gap, anchor: c.anchor.clone(), arm, R: R2 };
              else { c.arm = keepArm; c.target = { ...c.target, R: keepR }; }
            }
          }
          if (best.R) c.target = { ...c.target, R: best.R };
          c.arm = best.arm;
        }

        // 2. Ovuc səthdən bayıra
        let moved = 0;
        for (let iter = 0; iter < 4 && best.gap < 0 && moved < 0.07; iter++) {
          const probe = this.#bodyGap(side, c);
          if (!probe.normal || probe.gap >= 0) break;
          const push = Math.min(-probe.gap + 0.004, 0.07 - moved);
          if (!this.#reanchor(c, side, c.anchor.clone().addScaledVector(probe.normal, push), 8000, 0.03)) break;
          moved += push;
          const g = this.#bodyGap(side, c);
          if (g.gap <= best.gap + 0.001) break;
          best = { gap: g.gap, anchor: c.anchor.clone(), arm: c.arm };
        }
        c.anchor = best.anchor.clone();
        c.arm = best.arm;
      }
    }
  }

  /**
   * Əl qarşı qolun içinə girəndə həmin qolun dirsəyini kənara fırladır. Qolun artıq
   * sərbəstlik dərəcəsi (çiyin-bilək oxu ətrafında fırlanma) bileyi tərpətmədən saidi
   * aparır, ona görə işarənin forması qorunur. Yalnız aralığı açan variant saxlanılır.
   */
  #armAside(L, R, probe) {
    const parts = (probe.pair ?? '').split('/');
    let better = false;
    for (const [i, side] of [[0, 'left'], [1, 'right']]) {
      if (parts[i] !== 'said' && parts[i] !== 'çiyin') continue;
      const c = side === 'left' ? L : R;
      const sgn = i === 0 ? -1 : 1;                 // sol hissə oxun əksinə itələnir
      const base = c.arm.info.E.clone();
      for (const d of [0.08, 0.16, 0.24]) {
        const elbow = base.clone().addScaledVector(probe.axis, sgn * d);
        const arm = solveArm(this.rigs[side], c.target.R, c.anchor, null, { elbow, shiftCost: 8000, avoid: this.#avoidBody() });
        if (!arm || arm.info.hard > Math.min(HARD_CEIL, Math.max(0.04, c.arm.info.hard + 0.03))) continue;
        const keep = c.arm;
        c.arm = arm;
        const after = this.#probeGap(L, R);
        if (after && after.gap > probe.gap + 0.001) { better = true; break; }
        c.arm = keep;
      }
    }
    return better;
  }

  /** Siqnalçının əl ölçüsü ilə avatarın ovucu arasındakı nisbət (söz üzrə median). */
  #wordScale(word) {
    const target = this.rigs.left.palmOff.length();
    const r = [];
    for (const fr of word.frames) {
      for (const h of [fr.L, fr.R]) {
        const pr = h && palmRadius(h.w);
        if (pr > 1e-3) r.push(target / pr);
      }
    }
    if (!r.length) return 1;
    r.sort((a, b) => a - b);
    return THREE.MathUtils.clamp(r[r.length >> 1], 0.7, 1.3);
  }

  /**
   * Ovucun hədəfini dəyişib qolu yenidən həll edir. Yeni həll biləyi həddən çıxarırsa və ya
   * əvvəlkindən nəzərəçarpacaq dərəcədə pisdirsə, köhnə poza saxlanılır: əlləri ayırmaq
   * naminə qolu qeyri-təbii vəziyyətə salmaq daha pis nəticədir.
   */
  #reanchor(c, side, anchor, shiftCost = 1500, slack = 0.01, bodyFloor = null) {
    const next = this.#keepOut(anchor);
    // Bədən yoxlanışı ilə çağırılanda (əlləri ayırmaq) dirsək olduğu yerdə saxlanılır: siqnalçının
    // dirsəyinə qayıtmaq #fixBody-nin bədəndən çıxardığı qolu yenidən gövdəyə salırdı və hər
    // addım rədd olunurdu (ölçülüb: "çəkmək", "tərcümə")
    const elbow = bodyFloor != null ? c.arm.info.E : c.elbow;
    const arm = solveArm(this.rigs[side], c.target.R, next, null, { elbow, shiftCost, avoid: this.#avoidBody() });
    // Güzəştin mütləq tavanı var: biləyi həddindən ~10°-dən çox çıxarmaq əlləri ayırmaqdan
    // pisdir — istifadəçi onu "anormal əl" kimi görür.
    const cap = Math.min(HARD_CEIL, Math.max(slack * 2, c.arm.info.hard + slack));
    if (!arm || arm.info.hard > cap) return false;
    // `bodyFloor` verilibsə, qol bədənə ondan dərin girməməlidir (bütün qol ölçülür:
    // IK-nın maneə qiyməti yalnız beş nöqtəyə baxır)
    if (bodyFloor != null && this.#bodyGap(side, { ...c, arm }).gap < bodyFloor) return false;
    c.anchor = next;
    c.arm = arm;
    return true;
  }

  /**
   * İki əlin ən yaxın nöqtələri arasındakı məsafə — poza müvəqqəti olaraq sümüklərə tətbiq
   * olunur, çünki barmaqlar oynaq həddləri ilə dəyişir və landmark-lardan təxmin etmək
   * kifayət qədər dəqiq deyil (ölçülüb: median 3 sm xəta).
   */
  /** Verilən həlli skeletə tətbiq edir, ölçmə funksiyasını çağırır və pozanı geri qaytarır. */
  #withPose(list, fn) {
    const saved = [];
    for (const [side, c] of list) {
      const set = (name, q) => {
        const node = this.bone(side + name);
        if (node) { saved.push([node, node.quaternion.clone()]); node.quaternion.copy(q); }
      };
      set('UpperArm', c.arm.upper);
      set('LowerArm', c.arm.lower);
      set('Hand', c.arm.hand);
      for (const [b, q] of Object.entries(c.fingers)) set(b, q);
    }
    this.scene.updateMatrixWorld(true);
    const out = fn();
    for (const [node, q] of saved) node.quaternion.copy(q);
    return out;
  }

  #probeGap(L, R) {
    const [pl, pr] = this.#withPose([['left', L], ['right', R]],
      () => [this.#contactParts('left'), this.#contactParts('right')]);
    return closestParts(pl, pr);
  }

  /**
   * İki əl toqquşması üçün bir tərəfin hissələri: mesh-dən ölçülmüş əl kapsulları (barmaq
   * ucları daxil), said və bazunun dirsəyə yaxın yarısı. Bədənlə toqquşma köhnə, nazik
   * modellə qalır (#collisionParts): orada barmaq ucunun çənəyə, sinəyə toxunması işarənin
   * özüdür və onu "bədənə girmə" saymaq olmaz.
   */
  #contactParts(side, points = null) {
    if (!this.handShape) return this.#collisionParts(side);
    const parts = handCapsules(points ?? this.#handPoints(side), this.handShape);
    const at = (b) => this.bone(side + b)?.getWorldPosition(new THREE.Vector3());
    const upper = at('UpperArm'), lower = at('LowerArm'), hand = at('Hand');
    parts.push([lower, hand, this.handShape.radius.forearm, 'said']);
    parts.push([upper.clone().lerp(lower, 0.5), lower, R_UPPERARM, 'çiyin']);
    return parts;
  }

  /** Avatarın əlinin 21 nöqtəsi (MediaPipe sırası) — skeletin dünya matrisləri yenilənmiş olmalıdır. */
  #handPoints(side) {
    const at = (b) => this.bone(side + b).getWorldPosition(new THREE.Vector3());
    const P = [at('Hand')];
    for (const [name, bones] of Object.entries(FINGER_BONE_NAMES)) {
      for (const b of bones) P.push(at(name + b));
      P.push(this.bone(side + name + bones[2]).localToWorld(this.handShape.tips[side][name].clone()));
    }
    return P;
  }

  /**
   * Əlin toqquşma forması mesh-dən (T-pozada, yükləmədə bir dəfə): hər barmaq sümüyünün
   * kapsul radiusu ona bağlı vertex-lərin oxdan median məsafəsidir; barmağın son sümüyü
   * modelin uc sümüyünə qədər uzanır (uc sümüyü yoxdursa — vertex-lərin ən uzaq proyeksiyasına);
   * ovucun radiusu ovucun ortasındakı yarım qalınlığıdır. Əvvəlki sabit model (8/10 mm,
   * uc sümüyü yox) barmağın son 2-3 sm-ini və ovucun yarısını görmürdü: həll əlləri "ayrı"
   * sayırdı, halbuki onlar 1-2 sm bir-birinin içində idi (ölçülüb).
   */
  #measureHands(humanoid) {
    const raw = (n) => humanoid.getRawBoneNode(n);
    const W = (o) => o.getWorldPosition(new THREE.Vector3());
    const verts = new Map();
    const want = new Set();
    for (const side of SIDES) {
      want.add(raw(side + 'Hand')).add(raw(side + 'LowerArm'));
      for (const [name, bones] of Object.entries(FINGER_BONE_NAMES)) for (const b of bones) want.add(raw(side + name + b));
    }
    const v = new THREE.Vector3();
    this.vrm.scene.traverse((m) => {
      if (!m.isSkinnedMesh) return;
      const pos = m.geometry.attributes.position;
      const si = m.geometry.attributes.skinIndex, sw = m.geometry.attributes.skinWeight;
      if (!pos || !si || !sw) return;
      for (let i = 0; i < pos.count; i++) {
        let bi = -1, best = 0;
        for (let k = 0; k < 4; k++) {
          const w = sw.getComponent(i, k);
          if (w > best) { best = w; bi = si.getComponent(i, k); }
        }
        const bone = m.skeleton.bones[bi];
        if (best < 0.5 || !want.has(bone)) continue;
        v.fromBufferAttribute(pos, i);
        m.applyBoneTransform(i, v);
        v.applyMatrix4(m.matrixWorld);
        if (!verts.has(bone)) verts.set(bone, []);
        verts.get(bone).push(v.clone());
      }
    });
    const median = (arr) => {
      const s = arr.filter(Number.isFinite).sort((a, b) => a - b);
      return s.length ? s[s.length >> 1] : NaN;
    };
    const radiusOf = (bone, a, b) => {
      const ab = b.clone().sub(a);
      return median((verts.get(bone) ?? []).map((p) => {
        const t = THREE.MathUtils.clamp(p.clone().sub(a).dot(ab) / ab.lengthSq(), 0, 1);
        return p.distanceTo(a.clone().addScaledVector(ab, t));
      }));
    };
    // Mesh-də bir hissə tapılmasa köhnə sabitlər qalır
    const FALLBACK = [R_FINGER, R_FINGER, R_FINGER * 0.75];
    const acc = { palm: [], forearm: [] };
    const tips = {};
    for (const side of SIDES) {
      tips[side] = {};
      for (const [name, bones] of Object.entries(FINGER_BONE_NAMES)) {
        const r = bones.map((b) => raw(side + name + b));
        const P = r.map(W);
        const end = r[2].children.find((c) => c.isBone);
        let tip = end ? W(end) : null;
        if (!tip || tip.distanceTo(P[2]) < 0.005) {
          const dir = P[2].clone().sub(P[1]).normalize();
          const far = Math.max(0.012, ...(verts.get(r[2]) ?? []).map((p) => p.clone().sub(P[2]).dot(dir)));
          tip = P[2].clone().addScaledVector(dir, far - 0.004);
        }
        const norm = this.bone(side + name + bones[2]);
        tips[side][name] = norm.worldToLocal(tip.clone());
        const chain = [...P, tip];
        (acc[name] ??= [[], [], []]).forEach((list, k) => list.push(radiusOf(r[k], chain[k], chain[k + 1])));
      }
      const hand = raw(side + 'Hand'), w0 = W(hand);
      const mid = W(raw(side + 'MiddleProximal')), idx = W(raw(side + 'IndexProximal'));
      const lit = W(raw(side + 'LittleProximal'));
      const d = mid.clone().sub(w0), len = d.length();
      d.normalize();
      const n = d.clone().cross(idx.clone().sub(lit)).normalize();
      const band = (verts.get(hand) ?? []).filter((p) => {
        const t = p.clone().sub(w0).dot(d) / len;
        return t > 0.45 && t < 0.9;
      });
      acc.palm.push(median(band.map((p) => Math.abs(p.clone().sub(w0).dot(n)))));
      const lower = raw(side + 'LowerArm');
      acc.forearm.push(radiusOf(lower, W(lower), w0));
    }
    const avg = (list, fallback) => {
      const ok = list.filter(Number.isFinite);
      return ok.length ? ok.reduce((a, b) => a + b, 0) / ok.length : fallback;
    };
    const radius = { palm: avg(acc.palm, R_PALM), forearm: avg(acc.forearm, R_FOREARM) };
    for (const name of Object.keys(FINGER_BONE_NAMES)) radius[name] = acc[name].map((l, k) => avg(l, FALLBACK[k]));
    return { tips, radius };
  }

  /**
   * Oynatmada iki əl bir-birinin içindən keçməsin. Kadrlar həll olunanda ayrılır, amma yaylar
   * kadrlar arasında (və qol istirahətdən qalxanda) hər qolu müstəqil fırladır: hər iki kadrda
   * aralı olan əllər yolda bir-birinin içindən keçə bilir (ölçülüb: "lazım"-da kadrlarda əllər
   * ən azı 12 sm aralıdır, qollar qalxarkən 1.4 sm bir-birinə girirdi). Yaylardan sonra
   * göstərilən poza ölçülür; girmə varsa hər bazu çiyin ətrafında azca döndərilir və əllər
   * aralanır — əlin forması və biləyi dəyişmir, qol bədənə əvvəlkindən dərin girmir. Düzəliş
   * yayların vəziyyətinə yazılmır: hər qol üçün ayrıca saxlanır, kadrdan kadra sönür və
   * toqquşma bitəndə öz-özünə yox olur. `fresh` — əvvəlki düzəliş atılır (snap).
   */
  #guardHands(dt = 0, fresh = false) {
    if (!this.guard || !this.handShape || !this.rigs) return;
    this.vrm.humanoid.normalizedHumanBonesRoot.updateWorldMatrix(true, true);
    const g = this.#guard ??= {
      q: new THREE.Quaternion(), pq: new THREE.Quaternion(), p: new THREE.Vector3(), one: new THREE.Quaternion(),
      avoid: this.#avoidBody() ?? (() => -Infinity),
      dirs: [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        .map((v) => new THREE.Vector3(...v)),
    };
    const palm = (P) => P[0].clone().add(P[5]).add(P[9]).add(P[17]).multiplyScalar(0.25);
    const shoulder = { left: this.bone('leftUpperArm'), right: this.bone('rightUpperArm') };
    // Yalnız əlin iştirak etdiyi toqquşmalar: çarpazlaşan qollar (said/bazu cütləri) bəzi
    // işarələrin özüdür, onları çiyindən itələmək saidi gövdəyə salırdı (ölçülüb)
    const isArm = (x) => x[3] === 'said' || x[3] === 'çiyin';
    const armOnly = (a, b) => isArm(a) && isArm(b);
    // Hər qolun düzəlişi (çiyin ətrafında dünya fırlanması) kadrdan kadra saxlanır və tədricən
    // sönür. Hər kadr sıfırdan başlayanda qoruyucu bəzi kadrlarda istiqamət tapmırdı və düzəliş
    // birdən itirdi — əllər titrəyirdi (ölçülüb: "yeni"-nin son pozasında aralıq kadrdan kadra
    // −0.2 → −2.0 → +0.2 sm). İndi yalnız çatmayan hissə əlavə olunur, toqquşma bitəndə
    // düzəliş ~0.3 s-də yox olur.
    const off = this.#guardOff ??= { left: new THREE.Quaternion(), right: new THREE.Quaternion() };
    const turn = (side, q) => {
      const upper = shoulder[side];
      upper.parent.getWorldQuaternion(g.pq);
      // Dünya fırlanması lokala: q_lokal' = P⁻¹ · Δ · P · q_lokal
      upper.quaternion.premultiply(g.pq.clone().invert().multiply(q).multiply(g.pq));
      upper.updateWorldMatrix(false, true);
    };
    for (const side of SIDES) {
      if (fresh) off[side].identity();
      else if (dt > 0) off[side].slerp(g.one, 1 - Math.exp(-dt / 0.12));
      if (1 - Math.abs(off[side].w) > 1e-9) turn(side, off[side]);
    }
    for (let iter = 0; iter < 4; iter++) {
      const PL = this.#handPoints('left'), PR = this.#handPoints('right');
      const parts = { left: this.#contactParts('left', PL), right: this.#contactParts('right', PR) };
      const hit = closestParts(parts.left, parts.right, armOnly);
      // Toxunmaq olar (aralıq ≈ 0 — toxunan işarələr belə qalır), bir-birinə keçmək olmaz
      if (!hit || hit.gap >= -GUARD_SLACK) return;
      // Bir addımda ən çoxu 2 sm (üç qatı 6 sm): daha böyük girmə yayın deyil, həllin səhvidir
      const need = Math.min(-GUARD_SLACK - hit.gap + 0.001, 0.02);
      const S = { left: shoulder.left.getWorldPosition(new THREE.Vector3()),
                  right: shoulder.right.getWorldPosition(new THREE.Vector3()) };
      const contact = { left: hit.pa, right: hit.pb };
      // Hər əlin real yerdəyişməsi: çiyin fırlanması δ-nin yalnız r-ə perpendikulyar hissəsini verir
      const reach = (side, dir, amount) => {
        const r = contact[side].clone().sub(S[side]);
        const d = dir.clone().multiplyScalar(amount);
        return d.addScaledVector(r, -d.dot(r) / Math.max(r.lengthSq(), 1e-4));
      };
      // Ayırma istiqaməti sınaqla seçilir. Ən yaxın nöqtələrin oxu barmaqlar bir-birinə
      // keçəndə hər addımda əks tərəfə dönür və itələmələr bir-birini sıfırlayır, ovucların
      // oxu isə əllər bir-birinin önündə olanda yavaş açır (ölçülüb). Namizədlər — ovuc oxu,
      // toqquşma oxu və bədənin üç oxu — hissələri sürüşdürməklə yoxlanılır. Barmaqlar bir-
      // birinə keçəndə kiçik addım heç bir istiqamətdə aralığı açmır (yeni cüt ən yaxın olur),
      // ona görə istiqamət üç qat addımla seçilir, addımın özü isə ən kiçik kifayət edəndir.
      const shift = (list, d) => list.map(([a, b, r, n]) => [a.clone().add(d), b.clone().add(d), r, n]);
      // Əl bədənə doğru itələnməsin: ovuc, barmaq ucları və saidin ortası bədənə əvvəlkindən
      // dərin girən istiqamət rədd olunur (bədən qolun öz IK-sında da maneədir, bura onu pozmasın)
      const P = { left: PL, right: PR };
      const depth = (side, d) => {
        let worst = -Infinity;
        for (const j of [0, 4, 8, 12, 16, 20, 5, 17]) worst = Math.max(worst, g.avoid(g.p.copy(P[side][j]).add(d)));
        const said = parts[side].find((x) => x[3] === 'said');
        if (said) worst = Math.max(worst, g.avoid(g.p.copy(said[0]).lerp(said[1], 0.5).addScaledVector(d, 0.5)));
        return worst;
      };
      const zero = new THREE.Vector3();
      // Qolun real pozasında bədənə girmə: dirsək, saidin üç nöqtəsi, bilək və barmaq ucları
      const bodyNow = (side) => {
        const e = this.bone(side + 'LowerArm').getWorldPosition(new THREE.Vector3());
        const h = this.bone(side + 'Hand').getWorldPosition(new THREE.Vector3());
        let worst = -Infinity;
        for (const t of [0, 0.33, 0.66, 1]) worst = Math.max(worst, g.avoid(g.p.copy(e).lerp(h, t)));
        const Pn = this.#handPoints(side);
        for (const j of [4, 8, 12, 16, 20, 5, 17]) worst = Math.max(worst, g.avoid(Pn[j]));
        return worst;
      };
      const limit = { left: Math.max(depth('left', zero), 0) + 0.003, right: Math.max(depth('right', zero), 0) + 0.003 };
      const trial = (dir, k) => {
        const dl = reach('left', dir, (-need * k) / 2), dr = reach('right', dir, (need * k) / 2);
        if (depth('left', dl) > limit.left || depth('right', dr) > limit.right) return { gap: -Infinity, dl, dr };
        return { gap: closestParts(shift(parts.left, dl), shift(parts.right, dr), armOnly)?.gap ?? -Infinity, dl, dr };
      };
      let dir = null, far = -Infinity;
      for (const d of [palm(PR).sub(palm(PL)).normalize(), hit.axis, ...g.dirs]) {
        const t = trial(d, 3);
        if (t.gap > far) { far = t.gap; dir = d; }
      }
      if (!dir) return;                                            // hər istiqamət bədənə aparır
      let best = null;
      for (const k of [1, 2, 3]) {
        best = trial(dir, k);
        if (best.gap >= -GUARD_SLACK) break;
      }
      if (best.gap <= hit.gap + 0.0005) return;                   // heç bir istiqamət açmır
      let moved = false;
      for (const side of SIDES) {
        const r = contact[side].clone().sub(S[side]);
        // Kiçik dönmə ω: nöqtə ω × r qədər yerini dəyişir, ω = r × δ / |r|²
        const w = r.clone().cross(side === 'left' ? best.dl : best.dr).divideScalar(Math.max(r.lengthSq(), 0.01));
        const ang = Math.min(w.length(), 10 * DEG);
        if (ang < 1e-6) continue;
        g.q.setFromAxisAngle(w.normalize(), ang);
        const was = bodyNow(side);
        turn(side, g.q);
        // Sınaq əlin yerdəyişməsinə baxır, çiyin fırlanması isə dirsəyi başqa tərəfə apara bilər:
        // qol bədənə əvvəlkindən dərin girirsə bu tərəfin dönməsi geri alınır (ölçülüb: "ünvan")
        if (bodyNow(side) > Math.max(was, 0) + 0.003) {
          turn(side, g.q.clone().invert());
          continue;
        }
        off[side].premultiply(g.q);
        moved = true;
      }
      if (!moved) return;
    }
  }

  /**
   * Əlin sümük parçaları (ovuc üçbucağı + hər barmağın iki sümüyü). Toqquşma yalnız
   * ovuc və barmaq uclarına görə yoxlananda barmaqların aralıq sümükləri bir-birinin
   * içindən keçirdi, ona görə bütün parçalar götürülür.
   */
  #handSegments(side) {
    const at = (b) => this.bone(side + b)?.getWorldPosition(new THREE.Vector3());
    const hand = at('Hand');
    const out = [];
    for (const [name, bones] of Object.entries(FINGER_BONE_NAMES)) {
      const chain = bones.map((b) => at(name + b));
      if (chain.some((p) => !p)) continue;
      out.push([chain[0], chain[1], R_FINGER, name], [chain[1], chain[2], R_FINGER, name]);
      if (hand) out.push([hand, chain[0], R_PALM, 'ovuc']);        // ovuc sümüyü
    }
    return out;
  }

  /**
   * Toqquşma üçün bir tərəfin bütün hissələri: barmaqlar, ovuc və qolun kapsulları.
   * Əl bəzi işarələrdə qarşı tərəfin saidinin içindən keçirdi — yalnız əl-əl yoxlanışı
   * bunu görmür, çünki said heç bir əl sümüyü deyil.
   */
  #collisionParts(side) {
    const parts = this.#handSegments(side);
    const at = (b) => this.bone(side + b)?.getWorldPosition(new THREE.Vector3());
    const upper = at('UpperArm'), lower = at('LowerArm'), hand = at('Hand');
    if (lower && hand) parts.push([lower, hand, R_FOREARM, 'said']);
    // Bazunun yalnız dirsəyə yaxın yarısı: çiyin oynağı anatomik olaraq gövdənin içindədir,
    // bütöv sümük götürüləndə hər kadr "bədənə girmiş" sayılır (ölçülüb: 292 sözün hamısı).
    if (upper && lower) parts.push([upper.clone().lerp(lower, 0.5), lower, R_UPPERARM, 'çiyin']);
    return parts;
  }

  /** Sözün bütün kadrlarında əllərin tutduğu sahə (baş və çiyinlər də daxil) — kamera üçün. */
  #wordBounds(word) {
    const sol = this.#wordSolve(word);
    if (!sol.bounds) {
      const b = this.body;
      const box = new THREE.Box3()
        .expandByPoint(b.headCenter.clone().add(new THREE.Vector3(0, b.headRadius, 0)))
        .expandByPoint(b.mid.clone().add(new THREE.Vector3(0, -0.25, 0)))
        .expandByPoint(this.rigs.left.S).expandByPoint(this.rigs.right.S);
      for (const fr of sol) {
        for (const side of SIDES) if (fr[side]) box.expandByPoint(fr[side].anchor);
      }
      box.expandByVector(new THREE.Vector3(0.13, 0.13, 0));   // barmaqlar ovucdan kənara çıxır
      sol.bounds = box;
    }
    return sol.bounds;
  }

  /**
   * Siqnalçının çiyin çərçivəsi qurulur və nöqtə avatarın gövdəsinə köçürülür;
   * miqyas çiyin eninin nisbətidir. Çiyin xətti bədənin dönməsini nəzərə alır.
   * "Yuxarı" şaquldur (kamera düz dayanır), gövdə oxu deyil: oturan siqnalçının
   * budları etibarsız tapılır və gövdə oxu ~20° kameraya əyilir — öndəki əllər
   * avatarda yuxarı qalxardı. Baş isə gövdə ilə birgə önə əyilir, ona görə üzə
   * yaxın nöqtə (yanaq, çənə, qulaq) burundan ölçülüb avatarın üzünə köçürülür;
   * arada iki yerləşmə məsafəyə görə qarışdırılır. Pozanın dərinliyi zəifdir
   * (qulaqdakı əl burundan önə düşür), ona görə üzə yaxın dərinlik yarıya sıxılır.
   */
  #signerToAvatar(b) {
    const P = b.map(([x, y, z]) => new THREE.Vector3(x, -y, -z));
    const mid = P[1].clone().add(P[2]).multiplyScalar(0.5);
    const ax = P[1].clone().sub(P[2]);
    const k = this.body.shoulderWidth / ax.length();
    ax.normalize();
    const up = new THREE.Vector3(0, 1, 0);
    up.addScaledVector(ax, -up.dot(ax)).normalize();
    const fwd = new THREE.Vector3().crossVectors(ax, up);
    const vec = ([x, y, z]) => new THREE.Vector3(x, -y, -z);
    const local = (d, depth = 1) => new THREE.Vector3(d.dot(ax), d.dot(up), depth * d.dot(fwd));
    return {
      /** Çiyinlərə nisbətən yer (dirsək, gövdə önündəki əl). */
      body: (q) => local(vec(q).sub(mid)).multiplyScalar(k).add(this.body.mid),
      /** Burundan ölçülən yer, dərinlik yarıya sıxılır. */
      face: (q) => local(vec(q).sub(P[0]), 0.5).multiplyScalar(k).add(this.body.nose),
      /** Üzə yaxınlıq: 1 — üzdə, 0 — ≥26 sm uzaqda. */
      near: (q) => 1 - THREE.MathUtils.smoothstep(local(vec(q).sub(P[0]), 0.5).length(), 0.12, 0.26),
      /** Siqnalçının istiqamət vektoru avatarın oxlarında (miqyassız). */
      dir: (d) => local(vec(d)),
      /** Çiyin ortasından şaquli məsafə, siqnalçının çiyin eni vahidində. */
      height: (q) => (local(vec(q).sub(mid)).y * k) / this.body.shoulderWidth,
    };
  }

  /**
   * Söz kadrında ovucun hədəfi. Gövdə önündə ovuc çiyinlərə nisbətən köçürülür.
   * Üzə yaxın işarədə isə üzə ən yaxın əl nöqtəsi (barmaq ucu, ovucun kənarı) üzə
   * nisbətən köçürülür və ovuc ondan avatarın əl ölçüsü ilə geri hesablanır: anime
   * avatarın başı böyük, əli kiçikdir — qulağa, yanağa toxunma belə saxlanır.
   */
  #wordPalm(map, wrist, hand, flip, rig, scale) {
    const palm = signerPoint(wrist, hand, 'palm', flip);
    const body = map.body(palm);
    let contact = palm, near = map.near(palm);
    for (const j of CONTACT) {
      const q = signerPoint(wrist, hand, j, flip);
      const n = map.near(q);
      if (n > near) { near = n; contact = q; }
    }
    if (near <= 0) return body;
    const back = map.dir(palm.map((v, k) => v - contact[k])).multiplyScalar(scale);
    return map.face(contact).add(back).lerp(body, 1 - near);
  }

  /** Anime avatarın başı böyükdür: ovuc başın və gövdənin içinə girməsin. */
  #keepOut(p) {
    const b = this.body;
    const d = p.clone().sub(b.headCenter);
    const r = b.headRadius + 0.03;
    if (d.length() < r) {
      if (d.z < 0.25 * d.length()) d.z = 0.25 * d.length();   // üzə tərəf itələnir
      p.copy(b.headCenter).addScaledVector(d.normalize(), r);
    }
    if (this.solid) {
      // Ölçülmüş gövdə: ovuc səthdən ən azı 2 sm kənarda. İstiqamət seçimi vacibdir —
      // işarə məkanı bədənin ÖNÜNDƏDİR, ona görə ellipsin normalı arxaya baxırsa ovuc
      // yana deyil, önə çıxarılır (əks halda əl kürəyin arxasına keçir).
      const hit = this.#solidDistance(p);
      if (hit.dist < 0.02) {
        const dir = hit.normal.z < 0.2
          ? new THREE.Vector3(hit.normal.x * 0.4, 0, 1).normalize()
          : hit.normal;
        const step = hit.normal.z < 0.2 ? this.#frontPush(p) : 0.02 - hit.dist;
        p.addScaledVector(dir, Math.max(step, 0.02 - hit.dist));
      }
    } else if (p.y < b.torsoTop && p.y > b.torsoBottom
               && Math.abs(p.x - b.mid.x) < b.torsoHalfWidth && p.z < b.chestZ) {
      p.z = b.chestZ;
    }
    return p;
  }

  /**
   * Gövdənin həndəsəsi mesh-dən ölçülür: hips-dən çiyinlərə qədər 12 dilim, hər dilimdə
   * ellipsin mərkəzi və yarımoxları vertex buludundan. Qolun bədəndən keçməsini yoxlamaq
   * üçün müstəvi kifayət etmir — gövdə öndən yastı, yanlardan enlidir, ona görə qol yandan
   * baxanda "kənarda", üstdən baxanda isə bədənin içində ola bilir.
   */
  #measureSolid() {
    const humanoid = this.vrm?.humanoid;
    if (!humanoid) return null;
    const raw = (n) => humanoid.getRawBoneNode(n);
    const group = (names) => new Set(names.map(raw).filter(Boolean));
    const torso = group(['hips', 'spine', 'chest', 'upperChest']);
    const headSet = group(['head', 'neck']);
    const pts = [], headPts = [];
    const v = new THREE.Vector3();
    this.vrm.scene.traverse((m) => {
      if (!m.isSkinnedMesh) return;
      const pos = m.geometry.attributes.position;
      const si = m.geometry.attributes.skinIndex, sw = m.geometry.attributes.skinWeight;
      if (!pos || !si || !sw) return;
      const bones = m.skeleton.bones;
      for (let i = 0; i < pos.count; i++) {
        let bi = -1, best = 0;
        for (let k = 0; k < 4; k++) {
          const w = sw.getComponent(i, k);
          if (w > best) { best = w; bi = si.getComponent(i, k); }
        }
        const bone = bones[bi];
        if (best < 0.5 || !bone) continue;
        const inTorso = torso.has(bone), inHead = headSet.has(bone);
        if (!inTorso && !inHead) continue;
        v.fromBufferAttribute(pos, i);
        if (m.applyBoneTransform) m.applyBoneTransform(i, v);
        else m.boneTransform?.(i, v);
        v.applyMatrix4(m.matrixWorld);
        (inTorso ? pts : headPts).push(v.clone());
      }
    });
    if (pts.length < 200) return null;
    const q = (arr, t) => arr.sort((a, b) => a - b)[Math.min(arr.length - 1, Math.floor(t * arr.length))];
    const ys = pts.map((p) => p.y);
    const yMin = q(ys.slice(), 0.02), yMax = q(ys.slice(), 0.99);
    const N = 12, slices = [];
    for (let i = 0; i < N; i++) {
      const lo = yMin + (yMax - yMin) * i / N, hi = yMin + (yMax - yMin) * (i + 1) / N;
      const sel = pts.filter((p) => p.y >= lo && p.y <= hi);
      if (sel.length < 20) { slices.push(null); continue; }
      // Mərkəz medianla deyil, kənarların ortası ilə: qalstuk/qırış kimi qeyri-simmetrik
      // detallar medianı 5 sm sürüşdürür, kəsiyin kənarları isə sabit qalır.
      const span = (vals) => {
        const a = q(vals.slice(), 0.03), b = q(vals.slice(), 0.97);
        return [(a + b) / 2, Math.max((b - a) / 2, 0.04)];
      };
      const [cx, rx] = span(sel.map((p) => p.x));
      const [cz, rz] = span(sel.map((p) => p.z));
      slices.push({ y: (lo + hi) / 2, cx, cz, rx, rz });
    }
    // Qonşu dilimlər arasında sıçrayış olmasın: 3 nöqtəli hamarlama
    const raw3 = slices.filter(Boolean);
    const smooth = raw3.map((c, i) => {
      const n = [raw3[i - 1], c, raw3[i + 1]].filter(Boolean);
      const avg = (k) => n.reduce((a, x) => a + x[k], 0) / n.length;
      return { y: c.y, cx: avg('cx'), cz: avg('cz'), rx: avg('rx'), rz: avg('rz') };
    });
    let head = null;
    if (headPts.length > 100) {
      const c = headPts.reduce((a, p) => a.add(p), new THREE.Vector3()).divideScalar(headPts.length);
      // Burun: göz səviyyəsinə yaxın zolaqda ən öndəki nöqtə. Əvvəl sabit sürüşmə ilə
      // hesablanırdı və o, yalnız bir avatara uyğun gəlirdi (ölçülüb: yeni avatarda 3.4 sm
      // səhv). Üzə toxunan işarələr (yanaq, burun, qulaq) bu nöqtəyə görə yerləşir.
      const eye = humanoid.getNormalizedBoneNode('leftEye')?.getWorldPosition(new THREE.Vector3());
      const level = eye ? eye.y : c.y - 0.03;
      const band = headPts.filter((p) => Math.abs(p.y - level) < 0.05);
      const nose = (band.length ? band : headPts).reduce((a, p) => (p.z > a.z ? p : a));
      head = { center: c, radius: q(headPts.map((p) => p.distanceTo(c)), 0.85), nose: nose.clone() };
    }
    return { slices: smooth, yMin, yMax, head };
  }

  /** Nöqtəni gövdənin ÖN səthinə çıxarmaq üçün lazım olan z-addımı. */
  #frontPush(p) {
    const s = this.solid;
    if (!s || p.y < s.yMin || p.y > s.yMax) return 0.02;
    let sl = s.slices[0];
    for (const c of s.slices) if (Math.abs(c.y - p.y) < Math.abs(sl.y - p.y)) sl = c;
    const u = (p.x - sl.cx) / sl.rx;
    const front = sl.cz + sl.rz * Math.sqrt(Math.max(0, 1 - u * u));
    return Math.max(0.02, front + 0.02 - p.z);
  }

  /**
   * Nöqtənin bədən səthinə məsafəsi (mənfi — içəridə) və bayıra doğru normal.
   * Gövdə dilim-dilim ellips, baş isə sferadır.
   */
  #solidDistance(p) {
    const s = this.solid;
    let dist = Infinity, normal = null;
    if (s?.head) {
      const d = p.clone().sub(s.head.center);
      const len = d.length() || 1e-6;
      if (len - s.head.radius < dist) { dist = len - s.head.radius; normal = d.divideScalar(len); }
    }
    if (s && p.y >= s.yMin && p.y <= s.yMax) {
      let sl = s.slices[0];
      for (const c of s.slices) if (Math.abs(c.y - p.y) < Math.abs(sl.y - p.y)) sl = c;
      const u = (p.x - sl.cx) / sl.rx, w = (p.z - sl.cz) / sl.rz;
      const k = Math.hypot(u, w) || 1e-6;
      // ellipsin səthinə təqribi məsafə: radial nisbət × həmin istiqamətdəki yarımox
      const reff = Math.hypot(u / k * sl.rx, w / k * sl.rz);
      const d = (k - 1) * reff;
      if (d < dist) {
        dist = d;
        normal = new THREE.Vector3(u / (sl.rx * k), 0, w / (sl.rz * k)).normalize();
      }
    }
    return { dist, normal: normal ?? new THREE.Vector3(0, 0, 1) };
  }

  /** Qolun hissələri bədənə nə qədər girib: ən pis nöqtə, onun aralığı və bayır normalı. */
  #bodyProbe(side) {
    let worst = { gap: Infinity, normal: null, part: null };
    // Bazu bədənə söykənə bilər (qollar aşağı olanda insanda da belədir), said yüngül
    // toxuna bilər, əl isə toxunduğu yerdə dayanmalıdır.
    const ALLOW = { 'çiyin': 0.035, said: 0.008 };
    for (const [a, b, r0, name] of this.#collisionParts(side)) {
      const r = r0 - (ALLOW[name] ?? 0);
      const steps = Math.max(2, Math.ceil(a.distanceTo(b) / 0.03));
      for (let i = 0; i <= steps; i++) {
        const p = a.clone().lerp(b, i / steps);
        const hit = this.#solidDistance(p);
        const gap = hit.dist - r;
        if (gap < worst.gap) worst = { gap, normal: hit.normal, part: name };
      }
    }
    return worst;
  }

  /** Söz işarələri üçün bədən ölçüləri — T-pozada, yaylardan əvvəl çağırılır. */
  #measureBody(humanoid) {
    const pos = (n) => humanoid.getNormalizedBoneNode(n)?.getWorldPosition(new THREE.Vector3());
    const L = this.rigs.left.S, R = this.rigs.right.S;
    const head = pos('head'), hips = pos('hips'), chest = pos('upperChest') ?? pos('chest');
    const eyeL = pos('leftEye'), eyeR = pos('rightEye');
    // Başın mərkəzi, radiusu və burnu mesh-dən ölçülür (bax: #measureSolid). Sabit
    // sürüşmələr yalnız bir avatara uyğun gəlirdi; avatar dəyişəndə 2-3 sm səhv verirdi.
    const measured = this.solid?.head;
    const headCenter = measured
      ? measured.center.clone()
      : head.clone().add(new THREE.Vector3(0, 0.085, 0.015));
    const nose = measured?.nose
      ? measured.nose.clone()
      : (eyeL && eyeR
        ? eyeL.clone().add(eyeR).multiplyScalar(0.5).add(new THREE.Vector3(0, -0.04, 0.095))
        : headCenter.clone().add(new THREE.Vector3(0, -0.062, 0.1)));
    return {
      mid: L.clone().add(R).multiplyScalar(0.5),
      shoulderWidth: L.distanceTo(R),
      headCenter,
      nose,
      headRadius: measured ? measured.radius : 0.115,
      chestZ: chest.z + 0.13,
      torsoTop: L.y + 0.02,
      torsoBottom: hips.y,
      torsoHalfWidth: 0.17,
    };
  }

  /** Poza bitdi: əl hazır vəziyyətə keçir, bir az sonra qol aşağı enir. */
  clearHandPose() {
    if (!this.rigs || this.mode === 'rest') return;
    const p = this.presets[this.hand];
    this.#setSide(this.hand, p.ready, p.relaxed);
    this.#toRest(this.hand === 'left' ? 'right' : 'left');   // sözdən sonra ikinci əl də enir
    this.mode = 'pause';
    this.idle = 0;
    this.face.target = {};
    this.face.head = [0, 0, 0];
  }

  /** Yayları dərhal hədəfə qoyur (ilk kadr və ölçmə üçün). */
  snap() {
    for (const s of this.springs.values()) {
      s.target.copy(s.offset).multiply(s.base);
      s.q.copy(s.target);
      s.w.set(0, 0, 0);
      s.node.quaternion.copy(s.q);
    }
    this.#guardHands(0, true);
    if (this.camGoal) {
      this.camera.position.copy(this.camGoal.pos);
      this.camTarget.copy(this.camGoal.target);
      this.camera.lookAt(this.camTarget);
    }
    this.vrm?.update(0);
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  #initSprings() {
    const add = (name, omega) => {
      const node = this.bone(name);
      if (!node) return;
      this.springs.set(name, {
        node, omega, q: node.quaternion.clone(), w: new THREE.Vector3(),
        base: new THREE.Quaternion(), offset: new THREE.Quaternion(), target: new THREE.Quaternion(),
      });
    };
    for (const b of BODY) add(b, OMEGA.body);
    for (const side of SIDES) {
      add(side + 'UpperArm', OMEGA.arm);
      add(side + 'LowerArm', OMEGA.arm);
      add(side + 'Hand', OMEGA.hand);
      for (const { bones } of Object.values(FINGER_CHAINS)) {
        for (const b of bones) add(side + b, OMEGA.finger);
      }
    }
  }

  #setSide(side, arm, fingers) {
    const set = (name, q) => this.springs.get(side + name)?.base.copy(q);
    set('UpperArm', arm.upper);
    set('LowerArm', arm.lower);
    set('Hand', arm.hand);
    for (const [b, q] of Object.entries(fingers)) set(b, q);
  }

  #toRest(side) {
    const p = this.presets[side];
    this.#setSide(side, p.rest, p.relaxed);
  }

  /**
   * Kamera: hərfdə işarə edən əlin tərəfinə yaxın, sözdə gövdənin mərkəzinə
   * (iki əl, üz və sinə birlikdə görünsün). Çərçivələr arasında yumşaq keçir.
   */
  #frameCamera(kind = 'letter', bounds = null) {
    if (!this.rigs) return;
    let target, dist;
    if (kind === 'word') {
      // Sahə kadra sığsın: şaquli və üfüqi görmə bucağına görə məsafə, ən azı 1.4 m
      const size = bounds.getSize(new THREE.Vector3());
      const tan = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
      target = bounds.getCenter(new THREE.Vector3());
      target.z = this.body.mid.z + 0.08;
      dist = THREE.MathUtils.clamp(Math.max(size.y / 2 / tan, size.x / 2 / (tan * this.camera.aspect)) + 0.1, 1.4, 2.4);
    } else {
      const rig = this.rigs[this.hand];
      const c = this.camFrame ?? { bias: 0.78, dist: 1.25, lift: 0 };
      const space = signAnchor(rig, UP).lerp(signAnchor(rig, DOWN), 0.5);
      target = rig.head.clone().lerp(space, c.bias);
      target.y += c.lift;
      dist = c.dist;
    }
    this.camGoal = { target, pos: new THREE.Vector3(target.x, target.y, target.z + dist) };
    if (!this.camTarget) {
      this.camTarget = target.clone();
      this.camera.position.copy(this.camGoal.pos);
      this.camera.lookAt(this.camTarget);
    }
  }

  #loop = () => {
    this._raf = requestAnimationFrame(this.#loop);
    const now = performance.now();
    const dt = Math.min((now - this.lastTime) / 1000, 0.05);
    this.lastTime = now;
    if (this.paused) return;        // sınaqda zaman addımı xaricdən verilir
    this.update(dt);
    this.renderer.render(this.scene, this.camera);
  };

  /** Bir zaman addımı (saniyə): canlılıq, yaylar, VRM yeniləməsi. */
  update(dt) {
    if (!this.vrm) return;
    this.time += dt;
    if (this.mode === 'pause' && (this.idle += dt) > PAUSE_TO_REST) {
      for (const side of SIDES) this.#toRest(side);
      this.mode = 'rest';
    }
    if (this.mode !== 'sign') { this.face.target = {}; this.face.head = [0, 0, 0]; }
    const kf = 1 - Math.exp(-14 * dt);
    for (const name of Object.keys(FACE_MAP)) {
      const c = this.face.cur[name] ?? 0;
      this.face.cur[name] = c + ((this.face.target[name] ?? 0) - c) * kf;
      if (name !== 'blink') this.vrm.expressionManager?.setValue(name, this.face.cur[name]);
    }
    for (let k = 0; k < 3; k++) this.face.headCur[k] += (this.face.head[k] - this.face.headCur[k]) * kf;
    this.#animateLife(dt);
    this.#stepSprings(dt);
    this.#guardHands(dt);
    this.vrm.update(dt);
    if (this.camGoal) {
      const k = 1 - Math.exp(-4 * dt);
      this.camera.position.lerp(this.camGoal.pos, k);
      this.camTarget.lerp(this.camGoal.target, k);
      this.camera.lookAt(this.camTarget);
    }
  }

  /** Nəfəs, başın kiçik hərəkəti, qolun titrəməyən yüngül yırğalanması, göz qırpma. */
  #animateLife(dt) {
    const on = this.life;
    const t = this.time;
    const breath = on ? Math.sin(t * 2 * Math.PI * 0.2) : 0;
    const n1 = on ? Math.sin(t * 0.37) * 0.6 + Math.sin(t * 0.93 + 1.3) * 0.4 : 0;
    const n2 = on ? Math.sin(t * 0.29 + 2.1) * 0.6 + Math.sin(t * 0.71 + 0.4) * 0.4 : 0;
    const active = on && this.mode !== 'rest' ? 1 : 0;
    // +Y ətrafında müsbət dönmə üzü +X-ə (avatarın soluna) çevirir
    const toward = this.hand === 'left' ? 1 : -1;
    const e = this.#euler;
    const off = (name, x, y, z) => {
      const s = this.springs.get(name);
      if (s) s.offset.setFromEuler(e.set(x, y, z));
    };
    off('spine', 0.004 * breath, 0.02 * active * toward, 0);
    off('chest', 0.008 * breath, 0.015 * active * toward, 0);
    off('upperChest', 0.010 * breath, 0, 0);
    off('neck', 0.012 * n2, 0.015 * n1, 0);
    // işarə edəndə baş azca ələ tərəf dönür və əyilir; üstəlik siqnalçının sözdəki baş hərəkəti
    const [hx, hy, hz] = this.face.headCur;
    off('head', 0.02 * n2 + 0.03 * active + hx, 0.03 * n1 + 0.05 * active * toward + hy,
      -0.035 * active * toward + hz);
    for (const side of SIDES) {
      const ph = side === 'left' ? 1.7 : 0;
      const k = on ? 1 : 0;
      off(side + 'UpperArm', k * 0.008 * Math.sin(t * 0.83 + ph), 0, k * 0.006 * Math.sin(t * 0.61 + ph));
      off(side + 'LowerArm', k * 0.010 * Math.sin(t * 1.13 + ph), 0, 0);
    }

    const b = this.blink;
    let v = 0;
    if (b.t < 0) {
      b.next -= dt;
      if (b.next <= 0) b.t = 0;
    }
    if (b.t >= 0) {
      b.t += dt;
      v = b.t < 0.07 ? b.t / 0.07 : Math.max(0, 1 - (b.t - 0.07) / 0.12);
      if (b.t >= 0.19) {
        b.t = -1;
        b.next = Math.random() < 0.15 ? 0.18 : 2.2 + Math.random() * 3.5;   // bəzən qoşa qırpma
      }
    }
    this.vrm.expressionManager?.setValue('blink', Math.max(on ? v : 0, this.face.cur.blink ?? 0));
  }

  /**
   * Kritik sönümlü fırlanma yayı (sümük başına): ω' = k·xəta − c·ω, c = 2√k.
   * Hədəf dəyişəndə sürət kəsilmir — hərəkət təcillə başlayır və yumşaq dayanır.
   */
  #stepSprings(dt) {
    const steps = Math.max(1, Math.ceil(dt * 120));
    const h = dt / steps;
    const err = this.#v, dq = this.#q, inv = this.#q2;
    for (const s of this.springs.values()) {
      s.target.copy(s.offset).multiply(s.base);
      const k = s.omega * s.omega, c = 2 * s.omega;
      for (let i = 0; i < steps; i++) {
        dq.copy(s.target).multiply(inv.copy(s.q).invert());
        quatToVec(dq, err);
        s.w.addScaledVector(err, k * h).multiplyScalar(1 / (1 + c * h));
        vecToQuat(err.copy(s.w).multiplyScalar(h), dq);
        s.q.premultiply(dq).normalize();
      }
      s.node.quaternion.copy(s.q);
    }
  }

  dispose() {
    cancelAnimationFrame(this._raf);
    removeEventListener('resize', this._onResize);
    if (this.vrm) VRMUtils.deepDispose(this.vrm.scene);
    this.renderer.dispose();
    this.container.removeChild(this.renderer.domElement);
  }
}
