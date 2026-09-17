/**
 * Cyber skeleton — 21 əl nöqtəsinin Three.js ilə 3B neon təsviri.
 *
 * VRM avatarı yanında ikinci görüntü kimi işləyir: barmaq sümükləri
 * çevrilmədən, birbaşa landmark-lardan çəkildiyi üçün modelin/animasiyanın
 * nə gördüyünü olduğu kimi göstərir — yəni sazlama (debug) görüntüsü də,
 * təqdimat effekti də olur.
 *
 * Bloom üçün EffectComposer + UnrealBloomPass işlədilir (əsl parıltı,
 * kölgə-imitasiyası deyil).
 */
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { HAND_CONNECTIONS } from './hands.js';

// Saytın kanal rəngləri: danışıq (mavi), jest (çəhrayı)
const CYAN = 0x3fc1e8;
const MAGENTA = 0xff4f97;
const AMBER = 0xf5b94f;
const BG = 0x0e0c0a;
const TRAIL_COUNT = 6;

export class CyberHand {
  constructor(container, { showGrid = true } = {}) {
    this.container = container;
    this.pose = null;         // hədəf poza (21 × Vector3)
    this.current = Array.from({ length: 42 }, () => new THREE.Vector3());
    this.hasPose = false;
    this.trails = [];
    this.startTime = performance.now();
    this.lastTime = this.startTime;

    const w = container.clientWidth || 480;
    const h = container.clientHeight || 480;

    this.scene = new THREE.Scene();
    // Fon scene.background ilə verilir: composer-in render hədəfində clear color
    // rəng məkanı çevrilməsindən keçmir və fon avatar pəncərəsindən açıq görünürdü
    this.scene.background = new THREE.Color(BG);
    this.scene.fog = new THREE.FogExp2(BG, 0.09);

    this.camera = new THREE.PerspectiveCamera(42, w / h, 0.1, 100);
    this.camera.position.set(0, 0.15, 5.0);
    this.camera.lookAt(0, 0, 0);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.setClearColor(BG, 1);
    container.appendChild(this.renderer.domElement);

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloomBase = 0.55;
    this.bloom = new UnrealBloomPass(new THREE.Vector2(w, h), this.bloomBase, 0.42, 0.30);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    this.root = new THREE.Group();
    this.root.visible = false;        // poza gələnə qədər gizli
    this.fade = 0;                    // 0 = tam sönük, 1 = tam görünən
    this.scene.add(this.root);

    if (showGrid) this.#buildGrid();
    this.#buildJoints();
    this.#buildBones();
    this.#buildTrails();

    this.scene.add(new THREE.AmbientLight(0x3a342d, 1.2));
    const key = new THREE.PointLight(CYAN, 12, 12);
    key.position.set(1.5, 1.5, 2.5);
    this.scene.add(key);

    this._onResize = () => this.resize();
    addEventListener('resize', this._onResize);
    this.#loop();
  }

  #buildGrid() {
    const grid = new THREE.GridHelper(14, 28, CYAN, 0x2a241e);
    grid.material.transparent = true;
    grid.material.opacity = 0.18;
    grid.position.y = -1.9;
    this.scene.add(grid);

    // Üfüq xətti — dərinlik hissi üçün
    const geo = new THREE.PlaneGeometry(14, 6);
    const mat = new THREE.MeshBasicMaterial({
      color: 0x1a1612, transparent: true, opacity: 0.35,
      side: THREE.DoubleSide, depthWrite: false,
    });
    const plane = new THREE.Mesh(geo, mat);
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = -1.905;
    this.scene.add(plane);
  }

  #buildJoints() {
    this.joints = [];
    const tips = new Set([4, 8, 12, 16, 20, 21+4, 21+8, 21+12, 21+16, 21+20]);
    for (let i = 0; i < 42; i++) {
      const isTip = tips.has(i);
      const isWrist = i === 0 || i === 21;
      const r = isWrist ? 0.105 : isTip ? 0.075 : 0.055;
      const geo = new THREE.SphereGeometry(r, 20, 20);
      const mat = new THREE.MeshBasicMaterial({
        color: isWrist ? AMBER : isTip ? MAGENTA : CYAN,
      });
      const mesh = new THREE.Mesh(geo, mat);
      this.root.add(mesh);

      // Additive halo — bloom-un tutacağı parıltı mənbəyi
      const halo = new THREE.Sprite(new THREE.SpriteMaterial({
        map: haloTexture(), color: mat.color, blending: THREE.AdditiveBlending,
        transparent: true, depthWrite: false, opacity: isTip ? 0.45 : 0.22,
      }));
      halo.scale.setScalar(isWrist ? 0.70 : isTip ? 0.52 : 0.34);
      mesh.add(halo);
      this.joints.push(mesh);
    }
  }

  #buildBones() {
    this.bones = [...HAND_CONNECTIONS, ...HAND_CONNECTIONS].map(() => {
      const geo = new THREE.CylinderGeometry(0.022, 0.022, 1, 8, 1, true);
      geo.translate(0, 0.5, 0);                 // baza nöqtədə dayansın
      const mat = new THREE.MeshBasicMaterial({
        color: CYAN, transparent: true, opacity: 0.75,
        blending: THREE.AdditiveBlending, depthWrite: false,
      });
      const mesh = new THREE.Mesh(geo, mat);
      this.root.add(mesh);
      return mesh;
    });
  }

  #buildTrails() {
    for (let t = 0; t < TRAIL_COUNT; t++) {
      const positions = new Float32Array(HAND_CONNECTIONS.length * 4 * 3);
      
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      const mat = new THREE.LineBasicMaterial({
        color: MAGENTA, transparent: true,
        opacity: 0.30 * (1 - t / TRAIL_COUNT),
        blending: THREE.AdditiveBlending, depthWrite: false,
      });
      const line = new THREE.LineSegments(geo, mat);
      this.root.add(line);
      this.trails.push({ line, positions });
    }
  }

  /**
   * Hədəf pozanı təyin edir.
   * @param {Array<{x:number,y:number,z:number}>|null} landmarks normallaşdırılmış 21 və ya 42 nöqtə
   */
  setPose(landmarks) {
    if (!landmarks || (landmarks.length !== 21 && landmarks.length !== 42)) { this.pose = null; return; }
    this.isTwoHand = landmarks.length === 42;

    // Bilək mərkəzə, ölçü normallaşdırılır → kadrdaki yer/ölçü təsvirə təsir etməsin
    const wrist = landmarks[0];
    const pts = landmarks.map((p) => new THREE.Vector3(
      (p.x - wrist.x), -(p.y - wrist.y), -(p.z || 0) * 0.8,
    ));
    let span = 0;
    for (const p of pts) span = Math.max(span, p.length());
    const s = span > 1e-6 ? 1.55 / span : 1;
    pts.forEach((p) => p.multiplyScalar(s));
    // Sabit sürüşmə əvəzinə öz kütlə mərkəzinə görə mərkəzləşdiririk:
    // bəzi hərflərdə əl aşağı baxır və sabit sürüşmə onu kadrdan çıxarırdı.
    const c = new THREE.Vector3();
    for (const p of pts) c.add(p);
    c.divideScalar(pts.length);
    pts.forEach((p) => p.sub(c));
    this.pose = pts;
    if (!this.hasPose) {
      // İlk poza: nöqtələri sıçratmadan birbaşa yerinə qoy
      for (let i = 0; i < 42; i++) this.current[i].copy(pts[i]);
      this.hasPose = true;
    }
    this.root.visible = true;
  }

  clearPose() { this.pose = null; }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.composer.setSize(w, h);
  }

  #loop = () => {
    this._raf = requestAnimationFrame(this.#loop);
    const now = performance.now();
    const dt = Math.min((now - this.lastTime) / 1000, 0.05);
    const t = (now - this.startTime) / 1000;
    this.lastTime = now;

    // Hədəfə yumşaq yaxınlaşma (kritik sönümlü yaxınlaşma kimi)
    const k = 1 - Math.exp(-14 * dt);
    if (this.pose) {
      for (let i = 0; i < 42; i++) this.current[i].lerp(this.pose[i], k);
      this.fade = Math.min(1, this.fade + dt * 5);
    } else {
      // Poza yoxdursa mərkəzə yığmırıq (bloom-da nəhəng ləkə olur) — söndürürük
      this.fade = Math.max(0, this.fade - dt * 3);
      if (this.fade <= 0.001) { this.root.visible = false; this.composer.render(); return; }
    }
    this.#applyFade();

    this.joints.forEach((m, i) => m.position.copy(this.current[i]));
    this.#updateBones();
    this.#updateTrails();

    // Yüngül avtomatik dönmə — 3B dərinlik oxunsun
    this.root.rotation.y = Math.sin(t * 0.35) * 0.28;
    this.root.rotation.x = Math.sin(t * 0.23) * 0.06;
    this.bloom.strength = (this.bloomBase + Math.sin(t * 1.6) * 0.07) * this.fade;

    this.composer.render();
  };

  #applyFade() {
    const f = this.fade;
    for (const j of this.joints) {
      j.material.opacity = f; j.material.transparent = f < 1;
      if (j.children[0]) j.children[0].material.opacity = f * 0.7;
    }
    for (const b of this.bones) b.material.opacity = 0.75 * f;
  }

  #updateBones() {
    const a = new THREE.Vector3(), b = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0), dir = new THREE.Vector3();
    const conns = this.isTwoHand ? [...HAND_CONNECTIONS, ...HAND_CONNECTIONS.map(([i,j])=>[i+21, j+21])] : HAND_CONNECTIONS;
    
    // Hide all bones first
    for(let b of this.bones) b.visible = false;
    
    conns.forEach(([i, j], n) => {
      a.copy(this.current[i]); b.copy(this.current[j]);
      const bone = this.bones[n];
      bone.visible = true;
      dir.subVectors(b, a);
      const len = dir.length();
      bone.position.copy(a);
      bone.scale.set(1, Math.max(len, 1e-4), 1);
      if (len > 1e-5) bone.quaternion.setFromUnitVectors(up, dir.normalize());
    });
  }

  #updateTrails() {
    const head = this.trails.pop();
    const conns = this.isTwoHand ? [...HAND_CONNECTIONS, ...HAND_CONNECTIONS.map(([i,j])=>[i+21, j+21])] : HAND_CONNECTIONS;
    conns.forEach(([i, j], n) => {
      const o = n * 6;
      head.positions[o] = this.current[i].x;
      head.positions[o + 1] = this.current[i].y;
      head.positions[o + 2] = this.current[i].z;
      head.positions[o + 3] = this.current[j].x;
      head.positions[o + 4] = this.current[j].y;
      head.positions[o + 5] = this.current[j].z;
    });
    // zero out remaining positions if 1 hand
    if (!this.isTwoHand) {
       for(let n=conns.length; n<this.bones.length; n++) {
          const o = n * 6;
          head.positions.fill(0, o, o+6);
       }
    }
    head.line.geometry.attributes.position.needsUpdate = true;
    this.trails.unshift(head);
    this.trails.forEach((tr, idx) => {
      tr.line.material.opacity = 0.26 * (1 - idx / TRAIL_COUNT);
    });
  }

  dispose() {
    cancelAnimationFrame(this._raf);
    removeEventListener('resize', this._onResize);
    this.renderer.dispose();
    this.container.removeChild(this.renderer.domElement);
  }
}

/** Radial qradiyent halo teksturu (bloom üçün yem). */
let _halo = null;
function haloTexture() {
  if (_halo) return _halo;
  const size = 128;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.25, 'rgba(255,255,255,0.55)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, size, size);
  _halo = new THREE.CanvasTexture(c);
  return _halo;
}
