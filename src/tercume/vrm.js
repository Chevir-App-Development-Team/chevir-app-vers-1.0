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
const OMEGA = { arm: 11, hand: 15, finger: 21, body: 6 };
const PAUSE_TO_REST = 0.9;   // s — yeni poza gəlməsə qol aşağı enir
const BODY = ['spine', 'chest', 'upperChest', 'neck', 'head'];
const UP = new THREE.Vector3(0, 1, 0), DOWN = new THREE.Vector3(0, -1, 0);

export class VrmAvatar {
  #solved = new WeakMap();       // landmark kadrı → {left, right} həll (poza datası dəyişmir)
  #spikes = new WeakMap();       // dinamik poza → {left, right} kadr xəritəsi
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
    this.blink = { next: 1.5 + Math.random() * 2, t: -1 };
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
   * Bir hərf pozasını bütün qola tətbiq edir.
   * @param {Array<{x,y,z}>|null} landmarks 21 normallaşdırılmış nöqtə (null → hazır vəziyyət)
   * @param {{pose?: object, frame?: number}} meta poses.json-dakı hərf və kadr indeksi
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

  /** Poza bitdi: əl hazır vəziyyətə keçir, bir az sonra qol aşağı enir. */
  clearHandPose() {
    if (!this.rigs || this.mode === 'rest') return;
    const p = this.presets[this.hand];
    this.#setSide(this.hand, p.ready, p.relaxed);
    this.mode = 'pause';
    this.idle = 0;
  }

  /** Yayları dərhal hədəfə qoyur (ilk kadr və ölçmə üçün). */
  snap() {
    for (const s of this.springs.values()) {
      s.target.copy(s.offset).multiply(s.base);
      s.q.copy(s.target);
      s.w.set(0, 0, 0);
      s.node.quaternion.copy(s.q);
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
   * Kamera sabitdir: baş və işarə məkanı birlikdə kadrdadır. Əlin arxasınca
   * getmir — hərəkət kamera sarsıntısı kimi yox, bədənin hərəkəti kimi görünür.
   */
  #frameCamera() {
    const rig = this.rigs?.[this.hand];
    if (!rig) return;
    const c = this.camFrame ?? { bias: 0.78, dist: 1.25, lift: 0 };
    const space = signAnchor(rig, UP).lerp(signAnchor(rig, DOWN), 0.5);
    const target = rig.head.clone().lerp(space, c.bias);
    target.y += c.lift;
    this.camera.position.set(target.x, target.y, target.z + c.dist);
    this.camera.lookAt(target);
    this.camTarget = target;
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
    this.#animateLife(dt);
    this.#stepSprings(dt);
    this.vrm.update(dt);
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
    // işarə edəndə baş azca ələ tərəf dönür və əyilir
    off('head', 0.02 * n2 + 0.03 * active, 0.03 * n1 + 0.05 * active * toward, -0.035 * active * toward);
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
    this.vrm.expressionManager?.setValue('blink', on ? v : 0);
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
