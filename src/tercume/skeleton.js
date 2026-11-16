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
const BG = 0x060d24;
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
    // Fon bir dəfə yaradılan kiçik canvas-dan gələn şaquli gradient (vrm.js-dəki eyni texnika) -
    // applyTheme() hər tema dəyişikliyində YALNIZ bu canvas-ı yenidən boyayır (renderer/WebGL
    // context toxunulmur). Composer-in render hədəfində clear color rəng məkanı çevrilməsindən
    // keçmir - texture isə scene.background kimi düzgün rənglənir.
    this._bgCanvas = document.createElement('canvas');
    this._bgCanvas.width = 4;
    this._bgCanvas.height = 256;
    this._bgCtx = this._bgCanvas.getContext('2d');
    this._bgTexture = new THREE.CanvasTexture(this._bgCanvas);
    this._bgTexture.colorSpace = THREE.SRGBColorSpace;
    this.scene.background = this._bgTexture; // applyTheme() aşağıda dərhal boyayır
    this.scene.fog = new THREE.FogExp2(BG, 0.09); // applyTheme() rəngi/density-ni idarə edir

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

    this.showGrid = showGrid;
    // Tema dəyişənə qədər defolt (dark) dəyərlər - applyTheme() dərhal üstələyir.
    this.glowBase = 0.7;
    this.bloomBase = 0.55;
    this.boneRadiusMul = 1;
    this.jointScaleMul = 1;
    this.boneOpacityBase = 0.75;
    this.trailOpacityBase = 0.26;

    this.#buildJoints();
    this.#buildBones();
    this.#buildTrails();

    this.scene.add(new THREE.AmbientLight(0x3a342d, 1.2));
    const key = new THREE.PointLight(CYAN, 12, 12);
    key.position.set(1.5, 1.5, 2.5);
    this.scene.add(key);

    this._onResize = () => this.resize();
    addEventListener('resize', this._onResize);
    this._onTheme = () => this.applyTheme();
    document.addEventListener('chevir:theme', this._onTheme);
    this.applyTheme();                // CSS dəyişənlərdən rəngləri oxuyur, grid-i qurur
    this.#loop();
  }

  /**
   * html[data-theme]-dən CSS dəyişənlərini oxuyub bütün rəngləri YERİNDƏ
   * yeniləyir (WebGL context/renderer YENİDƏN YARADILMIR). GridHelper
   * istisna - rəngləri construct zamanı vertex-ə bağlandığı üçün
   * material.color.set() ilə dəyişdirilə bilmir, ona görə dispose+rebuild
   * edilir (yüngül LineSegments, renderer-ə toxunmur).
   */
  applyTheme() {
    const cs = getComputedStyle(document.body);
    const read = (name, fallback) => cs.getPropertyValue(name).trim() || fallback;
    const readNum = (name, fallback) => {
      const v = parseFloat(cs.getPropertyValue(name));
      return Number.isFinite(v) ? v : fallback;
    };
    // Yalnız grid/plane chrome-u VƏ bloom threshold-u üçün - bunlar --t-skl-additive-dən ASILI
    // DEYİL (additive indi hər iki temada 1-dir), amma panelin öz fonuna görə fərqli davranmalıdır.
    const isLight = document.documentElement.dataset.theme === 'light';

    const cyan = read('--t-skl-cyan', '#3fc1e8');
    const amber = read('--t-skl-amber', '#f5b94f');
    const pink = read('--t-skl-pink', '#ff4f97');
    // Eyni cüt vrm.js-dəki avatar fonu ilə paylaşılır (dizayn hədəfi: kamera/avatar/skeleton
    // panelləri HAMISI eyni gradient) - dark-da hər iki dayanma '#060d24' fallback-ına düşür
    // (ayrıca --t-scene-bg tokeninə ehtiyac qalmadı, piksel-identik).
    const bgTop = read('--t-panel-bg-top', '#060d24');
    const bgBottom = read('--t-panel-bg-bottom', '#060d24');

    // Light-da --t-skl-glow/--t-skl-bloom dark-ın yarısıdır (CSS-dən) - parıltı zəifləşir, amma
    // rənglər/ölçülər dəyişmir.
    this.glowBase = readNum('--t-skl-glow', 0.7);
    this.bloomBase = readNum('--t-skl-bloom', 0.55);
    // --t-skl-additive hər iki temada 1-dir (bloom.enabled, fog, ölçü - dark ilə eyni budaq).
    // Fərqli "additive=0" budaqlar (aşağıda `:` tərəfi) hazırda işə düşmür - CSS bayrağı
    // dəyişsə gələcəkdə yenidən aktivləşə bilsin deyə saxlanılıb.
    this.additive = readNum('--t-skl-additive', 1) > 0;
    const blend = this.additive ? THREE.AdditiveBlending : THREE.NormalBlending;
    this.boneRadiusMul = this.additive ? 1 : 1.15;
    this.jointScaleMul = this.additive ? 1 : 1.12;
    // Kök səbəb (xətlər ağa qarışırdı): açıq fonda additive sümük rəngi arxa fonla toplanıb
    // 255-ə kəsilir (ölçülüb - hər üç rəng tam ağa çıxır). Light-da sümüklər NormalBlending-ə
    // keçir (aşağı, bones loop-u) - bu YALNIZ `isLight`-a bağlıdır, `this.additive`-ə yox (əks
    // halda dark-ın additive+bloom görünüşü də dəyişərdi). Opaklıq da buna uyğun tam (1).
    this.boneOpacityBase = isLight ? 1 : 0.75;
    this.trailOpacityBase = this.additive ? 0.26 : 0.2;
    this.haloScaleMul = this.additive ? 1 : 0.85 / this.jointScaleMul;
    this.haloAccentMul = this.additive ? 1 : 2.5;

    const ctx = this._bgCtx, h = this._bgCanvas.height;
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, bgTop);
    grad.addColorStop(1, bgBottom);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, this._bgCanvas.width, h);
    this._bgTexture.needsUpdate = true;
    // Duman (fog) üfüqdə fonla qarışıb dərinlik hiss etdirir - hər iki temada aktiv (density
    // 0.09, additive=1). Rəng alt dayanmaya (bgBottom) bağlanır - dark-da iki dayanma eyni
    // olduğu üçün fərq yoxdur.
    if (this.scene.fog) {
      this.scene.fog.color.set(bgBottom);
      this.scene.fog.density = this.additive ? 0.09 : 0;
    }

    for (const mesh of this.joints) {
      const { isWrist, isTip, haloBase } = mesh.userData;
      const c = isWrist ? amber : isTip ? pink : cyan;
      mesh.material.color.set(c);
      mesh.scale.setScalar(this.jointScaleMul);
      if (mesh.children[0]) {
        mesh.children[0].material.color.set(c);
        mesh.children[0].material.blending = blend;
        mesh.children[0].scale.setScalar(haloBase * this.haloScaleMul);
      }
    }
    for (const b of this.bones) {
      b.material.color.set(cyan);
      // Light-da NormalBlending + depthWrite:true - sümük öz əsl rənginə opaq görünür (yuxarı
      // şərhə bax). Dark-da dəyişmir (additive + depthWrite:false, piksel-identik).
      b.material.blending = isLight ? THREE.NormalBlending : blend;
      b.material.depthWrite = isLight ? true : !this.additive;
    }
    for (const tr of this.trails) {
      tr.line.material.color.set(pink);
      tr.line.material.blending = blend;
      // depthWrite HƏMİŞƏ false qalır (hər iki temada) - trail xətt seqmentləri öz aralarında
      // üst-üstə düşəndə sərt kəsilmə YOX, yumşaq qarışma lazımdır (sümüklərdən fərqli olaraq).
    }
    this.bloom.enabled = this.additive;
    // --t-panel-bg-top-un (#c9dcf5) öz parlaqlığı (luminance ≈0.70) UnrealBloomPass-ın defolt
    // threshold-undan (0.30) xeyli yuxarıdır - FON ÖZÜ bloom-a düşüb bütöv paneli ağardardı.
    // Dark-ın fonu (~0 luminance) heç vaxt bu həddə yaxın olmadığı üçün dark-da threshold-u
    // qaldırmaq TƏSİRSİZDİR (piksel-identik qalır) - yalnız light-da fonun luminance-ından
    // yuxarı qaldırılır ki, YALNIZ halo-nun additive-stacked parlaq nöqtələri bloom-a düşsün.
    this.bloom.threshold = isLight ? 0.85 : 0.3;

    if (this.showGrid) this.#buildGrid(isLight);
    else this.#disposeGrid();
  }

  #disposeGrid() {
    if (this.grid) {
      this.scene.remove(this.grid);
      this.grid.geometry.dispose();
      this.grid.material.dispose();
      this.grid = null;
    }
    if (this.gridPlane) {
      this.scene.remove(this.gridPlane);
      this.gridPlane.geometry.dispose();
      this.gridPlane.material.dispose();
      this.gridPlane = null;
    }
  }

  // isLight: grid/plane --t-skl-additive-dən ASILI DEYİL (bax applyTheme()-dəki şərh) - çağıran
  // tərəfdən ötürülür ki, eyni DOM oxuması iki dəfə təkrarlanmasın.
  #buildGrid(isLight) {
    this.#disposeGrid();
    const cs = getComputedStyle(document.body);
    const read = (name, fallback) => cs.getPropertyValue(name).trim() || fallback;
    const cyan = read('--t-skl-cyan', '#3fc1e8');
    const gridColor = read('--t-skl-grid', '#2a241e');
    // Mərkəz xətti ayrıca tokendir, YALNIZ light-da (bones/joint-lərin cyan-ından DEKUPLAJ
    // edilib) - grid-in mərkəz xətti "ağ-mavi ton" olmalıdır. Dark-da əvvəllər cyan fallback-ı
    // mərkəz xəttini adi grid xətlərindən kəskin fərqləndirib panelin ortasında nazik şaquli/üfüqi
    // xətt kimi görünürdü (GridHelper-in iki mərkəz xətti bu kamera bucağından belə proyeksiya olunur)
    // - indi dark-da adi gridColor-la eynidir, seçilmir.
    const gridCenter = isLight ? read('--t-skl-grid-center', cyan) : gridColor;
    const planeColor = read('--t-skl-plane', '#1a1612');

    const grid = new THREE.GridHelper(14, 28, new THREE.Color(gridCenter), new THREE.Color(gridColor));
    grid.material.transparent = true;
    // Light-da xətlər - 0.55 çox sıx idi, horizonta yaxın sıxlaşan xətlər bir-birinə qarışıb
    // boz zolaq kimi görünürdü (ölçülüb). 0.22-də ayrı-ayrı xətlər qalır, zolaq itir. Dark dəyişmir.
    grid.material.opacity = isLight ? 0.22 : 0.18;
    grid.position.y = -1.9;
    this.scene.add(grid);
    this.grid = grid;

    // Üfüq xətti — dərinlik hissi üçün. Light-da gradient fonla dəqiq uyğunlaşa bilməyən düz
    // rəngli səth görünən zolaq yaradırdı - ona görə tam şəffaf edilir (dark dəyişmir).
    const geo = new THREE.PlaneGeometry(14, 6);
    const mat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(planeColor), transparent: true, opacity: isLight ? 0 : 0.35,
      side: THREE.DoubleSide, depthWrite: false,
    });
    const plane = new THREE.Mesh(geo, mat);
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = -1.905;
    this.scene.add(plane);
    this.gridPlane = plane;
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
      const haloBase = isWrist ? 0.70 : isTip ? 0.52 : 0.34;
      mesh.userData = { isWrist, isTip, haloBase };
      this.root.add(mesh);

      // Additive halo — bloom-un tutacağı parıltı mənbəyi (dark). Opaklıq #applyFade()-də hər
      // kadr hesablanır (bu construction-time dəyər yalnız ilk, pozasız kadrda görünür).
      const halo = new THREE.Sprite(new THREE.SpriteMaterial({
        map: haloTexture(), color: mat.color, blending: THREE.AdditiveBlending,
        transparent: true, depthWrite: false, opacity: isTip ? 0.45 : 0.22,
      }));
      halo.scale.setScalar(haloBase);
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
    // Dərinlik də biləyə nisbətəndir: metrik 3B nöqtələr veriləndə (söz işarələri, iki əl)
    // mütləq z iki əli eyni müstəviyə yığırdı. Normallaşdırılmış əl datasında z[0] = 0,
    // ona görə bu dəyişiklik orada heç nəyi dəyişmir.
    let pts = landmarks.map((p) => new THREE.Vector3(
      (p.x - wrist.x), -(p.y - wrist.y), -((p.z || 0) - (wrist.z || 0)) * 0.8,
    ));
    
    // Pad to 42 if only 21 points (hide second hand far off-screen)
    if (pts.length === 21) {
      for (let i = 0; i < 21; i++) pts.push(new THREE.Vector3(0, 0, -100));
    }
    
    let span = 0;
    for (const p of pts) if (p.z > -50) span = Math.max(span, p.length());
    const s = span > 1e-6 ? 1.55 / span : 1;
    pts.forEach((p) => { if (p.z > -50) p.multiplyScalar(s); });
    // Sabit sürüşmə əvəzinə öz kütlə mərkəzinə görə mərkəzləşdiririk:
    const c = new THREE.Vector3();
    let visCount = 0;
    for (const p of pts) { if (p.z > -50) { c.add(p); visCount++; } }
    if (visCount > 0) c.divideScalar(visCount);
    pts.forEach((p) => { if (p.z > -50) p.sub(c); });
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
      if (j.children[0]) {
        // Dark-da hamısı eyni (f*glowBase, piksel-identik) - light-da uc/bilək halo-su
        // haloAccentMul ilə bir az güclənir, adi oynaqlar baza (çox zəif) qalır.
        const { isWrist, isTip } = j.userData;
        const mul = isWrist || isTip ? this.haloAccentMul : 1;
        j.children[0].material.opacity = f * this.glowBase * mul;
      }
    }
    for (const b of this.bones) b.material.opacity = this.boneOpacityBase * f;
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
      bone.scale.set(this.boneRadiusMul, Math.max(len, 1e-4), this.boneRadiusMul);
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
      tr.line.material.opacity = this.trailOpacityBase * (1 - idx / TRAIL_COUNT);
    });
  }

  dispose() {
    cancelAnimationFrame(this._raf);
    removeEventListener('resize', this._onResize);
    document.removeEventListener('chevir:theme', this._onTheme);
    this.#disposeGrid();
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
