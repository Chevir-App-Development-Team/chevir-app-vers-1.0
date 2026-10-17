import * as THREE from 'three'
import gsap from 'gsap'
import { HAND_POSES, HAND_CONNECTIONS } from './handPoses.js'
import { BREAKPOINTS } from './gsapSetup.js'

// --hand-left/--hand-right dəyərlərini :root-dan oxuyur ki, rənglər CSS-dəki
// tək mənbədən gəlsin - bu dəyişənlər heç yerdə override olunmur (base.css).
function readCssColor(el, varName, fallback) {
  const value = getComputedStyle(el).getPropertyValue(varName).trim()
  return value || fallback
}

// ---- Əl topologiyasından (HAND_CONNECTIONS, handPoses.js - TOXUNULMUR)
// kapsul/oynaq görünüş məlumatı qurulur. Yalnız render qatı: hansı seqmentin
// nə qalınlıqda çəkiləcəyi, ovuc-daxili bağlantıların (palm mesh örtdüyü üçün)
// kapsul kimi çəkilməyəcəyi. ----
const LOCAL = (i) => (i < 21 ? i : i - 21)

const THUMB_RADIUS = { '0-1': 0.085, '1-2': 0.075, '2-3': 0.062, '3-4': 0.05 }
const FINGER_PROXIMAL = new Set(['5-6', '9-10', '13-14', '17-18'])
const FINGER_MIDDLE = new Set(['6-7', '10-11', '14-15', '18-19'])
const FINGER_DISTAL = new Set(['7-8', '11-12', '15-16', '19-20'])

function segmentRadius(a, b) {
  const key = `${LOCAL(a)}-${LOCAL(b)}`
  if (THUMB_RADIUS[key] != null) return THUMB_RADIUS[key]
  if (FINGER_PROXIMAL.has(key)) return 0.062
  if (FINGER_MIDDLE.has(key)) return 0.05
  if (FINGER_DISTAL.has(key)) return 0.04
  return null // ovuc-daxili bağlantılar (0-5, 5-9, 9-13, 13-17, 0-17) - palm mesh örtür
}

const CAPSULE_CONNECTIONS = HAND_CONNECTIONS.map(([a, b]) => ({ a, b, radius: segmentRadius(a, b) })).filter(
  (c) => c.radius != null
)

// Bilək(0) → baş barmaq(1-4) → işarə(5-8) → orta(9-12) → adsız(13-16) → çeçələ(17-20).
// Bilməyə/ovuca yaxın oynaqlar daha iri, dırnaq ucuna doğru kiçilir - anatomik
// yumru "knuckle" hissi üçün (konusun düz kəsiyini də örtür, bax applyHandMeshes).
const JOINT_RADIUS_BY_LOCAL = [
  0.095, 0.08, 0.07, 0.058, 0.05, 0.07, 0.055, 0.047, 0.042, 0.072, 0.057, 0.048, 0.042, 0.068, 0.054, 0.046, 0.041,
  0.062, 0.05, 0.043, 0.038,
]
const PALM_LOCAL_INDICES = [0, 5, 9, 13, 17] // bilək + 5 MCP - ovuc fan-mesh üçün

// Əl 25-35% böyüdülür (tapşırıq tələbi) - pivot səviyyəsində tətbiq olunur
// (bax initHand3D-dəki frame/pivot/model üç-qatlı struktur), mərkəzləşdirmə
// koordinatları özü XAM qalır.
const HAND_SCALE = 1.3
// Hər kənarda tələb olunan boşluq (6-8% aralığı) - kamera məsafəsi bu
// marja ilə hesablanır, heç bir pozada (ilk VƏ son daxil) kəsilmə olmasın.
const MARGIN = 0.075
// Əl "yuxarı-mərkəzə" qalxsın tələbi - boş qalan şaquli yerin bu faizi
// pivot-u yuxarı itələyir, marja tələbini pozmadan (bax layoutScene).
const UPWARD_BIAS = 0.6

// Bütün pozalar üzrə OX-BAZLI sərhəd qutusu (tək radius yox) - kamera
// məsafəsi və "yuxarı-mərkəz" sürüşdürməsi bu qutuya görə hesablanır, hər iki
// ox üçün ayrıca (sfera-uyğunlaşma əvəzinə), heç bir pozada kəsilmə qalmasın.
function computeHandFraming(poses) {
  const min = [Infinity, Infinity, Infinity]
  const max = [-Infinity, -Infinity, -Infinity]
  poses.forEach((pose) => {
    pose.forEach(([x, y, z]) => {
      min[0] = Math.min(min[0], x)
      max[0] = Math.max(max[0], x)
      min[1] = Math.min(min[1], y)
      max[1] = Math.max(max[1], y)
      min[2] = Math.min(min[2], z)
      max[2] = Math.max(max[2], z)
    })
  })
  // Ən iri oynaq radiusuna (bilək, 0.095) + kiçik ehtiyat - mesh-in öz
  // ölçüsü sərhəd qutusundan kənara çıxmasın.
  const pad = 0.12
  const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2]
  const halfExtents = [(max[0] - min[0]) / 2 + pad, (max[1] - min[1]) / 2 + pad, (max[2] - min[2]) / 2 + pad]
  return { center, halfExtents }
}

const HAND_FRAME = computeHandFraming(HAND_POSES)

const isDesktop = window.matchMedia(BREAKPOINTS.isDesktop).matches
const isCoarsePointer = window.matchMedia('(pointer: coarse)').matches
// Toxunma cihazlarında (coarse pointer və ya <980px) barmaqla scroll əl
// çevirməyə getməsin deyə orbit idarəsi tamamilə sönür - əl yalnız avtomatik
// fırlanır. Masaüstündə kursor idarəsi olduğu kimi qalır.
const enablePointerOrbit = isDesktop && !isCoarsePointer

// Verilmiş yarı-ölçülü qutunu (üfüqi/şaquli ayrıca, dərinlik ehtiyat kimi)
// kadra tam sığdıran kamera məsafəsi - hər ox üçün ayrı FOV-əsaslı tutma,
// maksimumu götürülür ki, heç biri kəsilməsin. Marja (MARGIN) hər iki oxda
// tələb olunan boşluğu təmin edir.
function fitDistanceForBox(camera, halfExtents) {
  const vHalf = (camera.fov * Math.PI) / 360
  const hHalf = Math.atan(Math.tan(vHalf) * camera.aspect)
  const reqV = halfExtents[1] * (1 + MARGIN)
  const reqH = halfExtents[0] * (1 + MARGIN)
  const vDist = reqV / Math.sin(vHalf)
  const hDist = reqH / Math.sin(hHalf)
  // Dərinlik (z) ehtiyatı - qutunun kameraya ən yaxın üzü də sığsın.
  return Math.max(vDist, hDist) + halfExtents[2]
}

// Fon müstəvisi həmişə bu sabit dünya-z-də qalır (kameranın məsafəsindən
// asılı olmayaraq) - yalnız ölçüsü hər relayout-da kameranın yeni məsafəsinə
// görə frustumu tam doldurmaq üçün yenidən hesablanır.
const BG_PLANE_Z = -4

const WAVE_VERTEX = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`

// Səhifənin tünd fon rəngi (#091638 - base.css-dəki --midnight dəyəri,
// loop.css-də də eyni sabit işlədilir). Hero həmişə tünd qaldığı üçün (bax:
// hero.css) bu, dəyişən deyil, birbaşa yazılıb - hero ilə digər tünd bölmələr
// arasında rəng tikişi qalmasın deyə. GLSL CSS dəyişənini oxuya bilmədiyi
// üçün dəyər burda TƏKRARLANIR - --midnight dəyişsə, bura da yenilənməlidir.
//
// TAM SABİT: heç bir uniform vaxtdan asılı deyil (uTime yoxdur) - premium hiss
// hərəkətdən yox, iki aydın işıq ləkəsindən və grain-dən gəlir.
const WAVE_FRAGMENT = /* glsl */ `
  uniform float uAspect; // canvas eni/hündürlüyü - UV məsafəsini ekran-mütənasib etmək üçün
  varying vec2 vUv;

  // Presizyona davamlı hash (Dave Hoskins, "Hash without Sine"). Köhnə versiya
  // full-resolution gl_FragCoord kimi böyük dəyərləri BİRBAŞA böyük əmsalla
  // (123.34/456.21) vurub fract() edirdi - float32-də bu böyüklükdə fract()
  // presizyon itirir və nəticə real təsadüfi yox, GPU-ya görə dəyişən "blok"
  // naxışı kimi görünürdü (əvvəlki "iri, boz, bərabər grain" şikayətinin əsl
  // səbəbi). Burda əvvəlcə kiçik əmsalla (0.13) miqyaslanır, aralıq dəyərlər
  // kiçik qalır, presizyon itmir - hər piksel fərdi təsadüfi qalır.
  float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.13);
    p3 += dot(p3, p3.yzx + 3.333);
    return fract((p3.x + p3.y) * p3.z);
  }

  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
  }

  float fbm(vec2 p) {
    float sum = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 4; i++) {
      sum += amp * noise(p);
      p *= 2.0;
      amp *= 0.5;
    }
    return sum;
  }

  // Fırlanmış (anizotrop) yumşaq ellips - diaqonal uzanan ləkə forması üçün.
  float blob(vec2 p, vec2 center, float angle, vec2 sigma) {
    vec2 d = p - center;
    float ca = cos(angle);
    float sa = sin(angle);
    vec2 dr = vec2(d.x * ca - d.y * sa, d.x * sa + d.y * ca);
    return exp(-(dr.x * dr.x / (2.0 * sigma.x * sigma.x) + dr.y * dr.y / (2.0 * sigma.y * sigma.y)));
  }

  void main() {
    vec2 uv = vUv;
    vec2 p = vec2(uv.x * uAspect, uv.y);

    // Sol-yuxarı: diaqonal (45°) uzanan işıq ləkəsi - küncdə, kiçik nüvə.
    vec2 c1 = vec2(0.05 * uAspect, 0.94);
    float b1 = blob(p, c1, 0.7853981634, vec2(0.32, 0.12));

    // Sağ-aşağı: daha kiçik, daha sönük ləkə (amma hələ aydın görünən).
    vec2 c2 = vec2(0.92 * uAspect, 0.08);
    float b2 = blob(p, c2, -0.5, vec2(0.26, 0.13)) * 0.65;

    // Statik üzvi təhrif - hərəkət yoxdur, forma riyazi ellips kimi süni
    // görünməsin deyə (bax: fbm, vaxtdan asılı deyil).
    float organic = fbm(p * 1.6 + 11.0);
    float light = (b1 + b2) * mix(0.85, 1.05, organic);

    // Kontrastı artır - "bərabər boz duman" yerinə aydın işıq/qaranlıq fərqi.
    // Aşağı/yuxarı hədlər yüksək saxlanılıb ki, ləkə yalnız öz nüvəsində tam
    // parlaq olsun (kiçik, "aydın" nüvə), böyük bir hissə ekranı "yuyub
    // aparmasın" - keçid özü hələ də yumşaqdır (smoothstep).
    light = smoothstep(0.15, 0.80, light);

    // Əlin sahəsini (kanvasın mərkəzi - kamera həmişə dünya mərkəzinə baxır)
    // nisbətən qaranlıq saxla ki, ağ əl fərqlənsin.
    vec2 centered = vec2((uv.x - 0.5) * uAspect, uv.y - 0.5);
    float handMask = smoothstep(0.08, 0.30, length(centered));
    light *= mix(0.35, 1.0, handMask);

    vec3 base = vec3(0.0353, 0.0863, 0.2196); // #091638 (--midnight)
    vec3 color = base + vec3(light) * 0.55;

    // Grain - işığın gücünə bağlı (qaranlıqda demək olar yox, işıqda aydın),
    // sabit (uTime yoxdur, kadr-kadr eyni), fərdi piksel təsadüfiliyi.
    float grainAmt = mix(0.012, 0.16, light);
    float grain = (hash(gl_FragCoord.xy) - 0.5) * grainAmt;
    color += vec3(grain);

    gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
  }
`

// Toon material-ın 3-pilləli kölgə keçidi üçün kiçik, Nearest-filtrli
// gradient toxuması - bir dəfə qurulur, bütün toon material-lar paylaşır.
let toonGradientCache = null
function getToonGradientTexture() {
  if (toonGradientCache) return toonGradientCache
  const steps = [70, 150, 210, 255]
  const canvas = document.createElement('canvas')
  canvas.width = steps.length
  canvas.height = 1
  const ctx = canvas.getContext('2d')
  steps.forEach((v, i) => {
    ctx.fillStyle = `rgb(${v},${v},${v})`
    ctx.fillRect(i, 0, 1, 1)
  })
  const texture = new THREE.CanvasTexture(canvas)
  texture.minFilter = THREE.NearestFilter
  texture.magFilter = THREE.NearestFilter
  texture.generateMipmaps = false
  toonGradientCache = texture
  return texture
}

// Yüngül Fresnel-əsaslı "rim" işıq - toon material-ın öz kölgə pilləsindən
// asılı olmayan, kənarda incə parıltı. Material-ın öz view-direction
// varying-indən deyil, burda ayrıca təyin olunandan istifadə edir (üç.js
// versiyaları arasında daxili adlandırma fərqlərindən asılı olmasın deyə).
function applyRimLight(material) {
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uRimPower = { value: 2.1 }
    shader.uniforms.uRimIntensity = { value: 0.35 }
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vRimViewDir;')
      .replace(
        '#include <begin_vertex>',
        '#include <begin_vertex>\nvRimViewDir = normalize(-(modelViewMatrix * vec4(position, 1.0)).xyz);'
      )
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        '#include <common>\nvarying vec3 vRimViewDir;\nuniform float uRimPower;\nuniform float uRimIntensity;'
      )
      .replace(
        '#include <dithering_fragment>',
        `
        float rimFactor = pow(1.0 - max(dot(normalize(normal), vRimViewDir), 0.0), uRimPower);
        gl_FragColor.rgb += rimFactor * uRimIntensity;
        #include <dithering_fragment>
        `
      )
  }
}

// İnvert-edilmiş-hull kənar xətti: eyni instance matrisini paylaşan ikinci
// mesh dəsti, normal boyunca bir az təpilib (BackSide) - əsl mesh-in
// "arxasından" görünərək nazik tünd kontur yaradır. Bütün hissələr (oynaq,
// seqment) bu TƏK material-ı paylaşır.
// Təpik (push) obyekt-yerli normal boyunca, unit-geometriyanın (radius=1)
// öz ölçüsünə nisbətən kiçik sabit faiz - instanceMatrix-in miqyas hissəsi
// (`instanceMatrix` three.js tərəfindən avtomatik təyin olunur, InstancedMesh
// üçün ƏLAVƏ bəyan etmək "redefinition" compile xətası verir - ölçülüb) bunu
// öz-özünə düzgün nisbətdə böyüdür, ona görə nazik seqmentlərdə kənar xətti
// bütün formanı udmur.
const OUTLINE_PUSH = 0.08
const outlineMaterial = new THREE.ShaderMaterial({
  side: THREE.BackSide,
  vertexShader: /* glsl */ `
    void main() {
      vec3 pushed = position + normal * ${OUTLINE_PUSH.toFixed(3)};
      #ifdef USE_INSTANCING
      vec4 worldPos = instanceMatrix * vec4(pushed, 1.0);
      #else
      vec4 worldPos = vec4(pushed, 1.0);
      #endif
      gl_Position = projectionMatrix * modelViewMatrix * worldPos;
    }
  `,
  fragmentShader: /* glsl */ `
    void main() {
      gl_FragColor = vec4(0.0353, 0.0863, 0.2196, 1.0); // --midnight
    }
  `,
})

function createPalmGeometry() {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(7 * 3), 3))
  geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(7 * 3), 3))
  geometry.setIndex(new THREE.BufferAttribute(new Uint16Array([0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5, 0, 5, 6]), 1))
  return geometry
}

// canvas ekrandan çıxanda (IntersectionObserver) və ya ölçüsü dəyişəndə
// (ResizeObserver) render dayanır/uyğunlaşır - performans üçün.
//
// TƏK SƏHNƏ: fon (noise dalğa + grain shader-li müstəvi) və 3D əl eyni
// renderer/scene-də birləşdirilib - ikinci WebGL konteksti yaradılmır.
// Qaraltma və radial maska (yalnız əlin sahəsi) birbaşa şu shader-in içindədir,
// ona görə əl heç vaxt qaralmır (ayrı material, shader-dən təsirlənmir).
// Müstəvi 980px-dən aşağı ekranlarda ümumiyyətlə yaradılmır (statik CSS
// gradient qalır), yalnız əl render olunur.
export function initHand3D(canvas) {
  // --hand-left/--hand-right base.css-də :root-da sabit təyin olunub, heç
  // yerdə override olunmur - əlin iki tərəfi bu iki rəngi daşıyır.
  const jointColor = readCssColor(document.documentElement, '--hand-left', '#0071ef')
  const boneColor = readCssColor(document.documentElement, '--hand-right', '#6fb1ff')
  const color1 = new THREE.Color(jointColor)
  const color2 = new THREE.Color(boneColor)

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, isDesktop ? 2 : 1.5))
  renderer.setClearAlpha(0)

  const scene = new THREE.Scene()
  const camera = new THREE.PerspectiveCamera(35, 1, 0.05, 100)

  // Üç-qatlı struktur, hər qat TƏK məsuliyyət daşıyır:
  // model  - xam landmark koordinatlarını mərkəzləşdirir (translate yalnız)
  // pivot  - fırlanma + HAND_SCALE miqyası, mərkəzi HƏMİŞƏ (0,0,0)-da qalır
  //          (əks halda "yuxarı-mərkəz" sürüşdürməsi fırlanma oxunu da
  //          sürüşdürər, əl "yırğalanardı")
  // frame  - yalnız ekran-çərçivələmə üçün şaquli sürüşdürmə (resize-də
  //          yenilənir), rotasiyadan tamamilə ayrıdır
  const frame = new THREE.Group()
  const pivot = new THREE.Group()
  const model = new THREE.Group()
  model.position.set(-HAND_FRAME.center[0], -HAND_FRAME.center[1], -HAND_FRAME.center[2])
  pivot.add(model)
  pivot.scale.setScalar(HAND_SCALE)
  frame.add(pivot)
  scene.add(frame)

  // MeshToonMaterial işıqsız tam qara render olunur - toon material-lar
  // səhnə işığına ehtiyac duyur (əvvəlki MeshBasicMaterial/LineBasicMaterial
  // işıqsız işləyirdi, ona görə səhnədə heç vaxt işıq yox idi).
  const keyLight = new THREE.DirectionalLight(0xffffff, 2.2)
  keyLight.position.set(1.1, 1.8, 2.4)
  scene.add(keyLight)
  const fillLight = new THREE.HemisphereLight(0xcfe3ff, 0x0b1430, 0.9)
  scene.add(fillLight)

  // QEYD: vertexColors:true + InstancedMesh.setColorAt() MeshToonMaterial
  // ilə birgə qara render olunur (ölçülüb - eyni material flat `color`
  // olaraq işlədiləndə düzgün işləyir, instanceColor aktiv olanda yox).
  // Ona görə hər əl üçün AYRICA material+InstancedMesh cütü işlədilir (ovuc
  // mesh-i ilə eyni, artıq sınanmış pattern) - instanceColor ümumiyyətlə
  // işlədilmir.
  const toonMaterialLeft = new THREE.MeshToonMaterial({ color: color1, gradientMap: getToonGradientTexture() })
  const toonMaterialRight = new THREE.MeshToonMaterial({ color: color2, gradientMap: getToonGradientTexture() })
  applyRimLight(toonMaterialLeft)
  applyRimLight(toonMaterialRight)

  const jointCount = HAND_POSES[0].length
  const jointsPerHand = jointCount / 2
  const jointGeometry = new THREE.SphereGeometry(1, 10, 8)
  const jointsLeft = new THREE.InstancedMesh(jointGeometry, toonMaterialLeft, jointsPerHand)
  const jointsRight = new THREE.InstancedMesh(jointGeometry, toonMaterialRight, jointsPerHand)
  const jointsOutline = new THREE.InstancedMesh(jointGeometry, outlineMaterial, jointCount)
  model.add(jointsLeft)
  model.add(jointsRight)
  model.add(jointsOutline)

  const capsuleGeometry = new THREE.CylinderGeometry(1, 1, 1, 8)
  const capsulesLeft = CAPSULE_CONNECTIONS.filter((c) => c.a < 21)
  const capsulesRight = CAPSULE_CONNECTIONS.filter((c) => c.a >= 21)
  const segmentsLeft = new THREE.InstancedMesh(capsuleGeometry, toonMaterialLeft, capsulesLeft.length)
  const segmentsRight = new THREE.InstancedMesh(capsuleGeometry, toonMaterialRight, capsulesRight.length)
  const segmentsOutline = new THREE.InstancedMesh(capsuleGeometry, outlineMaterial, CAPSULE_CONNECTIONS.length)
  model.add(segmentsLeft)
  model.add(segmentsRight)
  model.add(segmentsOutline)

  // Ovuc içi - hər əl üçün ayrıca kiçik fan-mesh (bilək + 5 MCP), hər kadr
  // yenilənir (bax applyHandMeshes). Kiçik/çox örtülü element olduğu üçün
  // kənar xətti burda işlədilmir (düz tək-normal-lı mnogougolnik üçün
  // normal-boyunca-təpik effektsiz olardı).
  const palmMaterialLeft = toonMaterialLeft
  const palmMaterialRight = toonMaterialRight
  const palmGeometryLeft = createPalmGeometry()
  const palmGeometryRight = createPalmGeometry()
  const palmLeft = new THREE.Mesh(palmGeometryLeft, palmMaterialLeft)
  const palmRight = new THREE.Mesh(palmGeometryRight, palmMaterialRight)
  model.add(palmLeft)
  model.add(palmRight)

  const current = new Float32Array(jointCount * 3)

  // ---- hər kadr yenilənən köməkçi obyektlər - heç biri burda deyil, tick
  // içində YENİDƏN yaradılmır (GC donması olmasın deyə). ----
  const _dummy = new THREE.Object3D()
  const _segDir = new THREE.Vector3()
  const _segMid = new THREE.Vector3()
  const _segQuat = new THREE.Quaternion()
  const _segScale = new THREE.Vector3()
  const _segMatrix = new THREE.Matrix4()
  const _upAxis = new THREE.Vector3(0, 1, 0)
  const _palmEdgeA = new THREE.Vector3()
  const _palmEdgeB = new THREE.Vector3()
  const _palmNormal = new THREE.Vector3()

  function updatePalm(handOffset, geometry) {
    const pos = geometry.attributes.position.array
    let cx = 0
    let cy = 0
    let cz = 0
    PALM_LOCAL_INDICES.forEach((local) => {
      const idx = handOffset + local
      cx += current[idx * 3]
      cy += current[idx * 3 + 1]
      cz += current[idx * 3 + 2]
    })
    cx /= PALM_LOCAL_INDICES.length
    cy /= PALM_LOCAL_INDICES.length
    cz /= PALM_LOCAL_INDICES.length
    pos[0] = cx
    pos[1] = cy
    pos[2] = cz
    PALM_LOCAL_INDICES.forEach((local, i) => {
      const idx = handOffset + local
      const o = (i + 1) * 3
      pos[o] = current[idx * 3]
      pos[o + 1] = current[idx * 3 + 1]
      pos[o + 2] = current[idx * 3 + 2]
    })
    // Fanı bağlayan 7-ci təpə - ilk xarici nöqtənin (bilək) təkrarı.
    pos[18] = pos[3]
    pos[19] = pos[4]
    pos[20] = pos[5]
    geometry.attributes.position.needsUpdate = true

    // Düz normal (bütün təpələr üçün eyni) - iki kənar vektorundan tək dəfə.
    _palmEdgeA.set(pos[3] - pos[0], pos[4] - pos[1], pos[5] - pos[2])
    _palmEdgeB.set(pos[6] - pos[0], pos[7] - pos[1], pos[8] - pos[2])
    _palmNormal.crossVectors(_palmEdgeA, _palmEdgeB).normalize()
    const normalAttr = geometry.attributes.normal.array
    for (let i = 0; i < 7; i++) {
      normalAttr[i * 3] = _palmNormal.x
      normalAttr[i * 3 + 1] = _palmNormal.y
      normalAttr[i * 3 + 2] = _palmNormal.z
    }
    geometry.attributes.normal.needsUpdate = true
  }

  function applyHandMeshes() {
    for (let i = 0; i < jointCount; i++) {
      const local = LOCAL(i)
      _dummy.position.set(current[i * 3], current[i * 3 + 1], current[i * 3 + 2])
      _dummy.scale.setScalar(JOINT_RADIUS_BY_LOCAL[local])
      _dummy.updateMatrix()
      ;(i < 21 ? jointsLeft : jointsRight).setMatrixAt(local, _dummy.matrix)
      jointsOutline.setMatrixAt(i, _dummy.matrix)
    }
    jointsLeft.instanceMatrix.needsUpdate = true
    jointsRight.instanceMatrix.needsUpdate = true
    jointsOutline.instanceMatrix.needsUpdate = true

    let leftIdx = 0
    let rightIdx = 0
    CAPSULE_CONNECTIONS.forEach((conn, i) => {
      const { a, b, radius } = conn
      _segDir.set(current[b * 3] - current[a * 3], current[b * 3 + 1] - current[a * 3 + 1], current[b * 3 + 2] - current[a * 3 + 2])
      const length = _segDir.length()
      _segDir.normalize()
      _segQuat.setFromUnitVectors(_upAxis, _segDir)
      _segMid.set(
        (current[a * 3] + current[b * 3]) / 2,
        (current[a * 3 + 1] + current[b * 3 + 1]) / 2,
        (current[a * 3 + 2] + current[b * 3 + 2]) / 2
      )
      _segScale.set(radius, length, radius)
      _segMatrix.compose(_segMid, _segQuat, _segScale)
      if (a < 21) {
        segmentsLeft.setMatrixAt(leftIdx++, _segMatrix)
      } else {
        segmentsRight.setMatrixAt(rightIdx++, _segMatrix)
      }
      segmentsOutline.setMatrixAt(i, _segMatrix)
    })
    segmentsLeft.instanceMatrix.needsUpdate = true
    segmentsRight.instanceMatrix.needsUpdate = true
    segmentsOutline.instanceMatrix.needsUpdate = true

    updatePalm(0, palmGeometryLeft)
    updatePalm(21, palmGeometryRight)
  }

  function setFromPose(pose) {
    for (let i = 0; i < jointCount; i++) {
      current[i * 3] = pose[i][0]
      current[i * 3 + 1] = pose[i][1]
      current[i * 3 + 2] = pose[i][2]
    }
    applyHandMeshes()
  }
  setFromPose(HAND_POSES[0])

  // ---- pozalar arasında yumşaq keçid (nöqtələr sıçramır) ----
  let poseIndex = 0
  const poseT = { value: 0 }
  let poseTimer = null

  function morphToNextPose() {
    const from = HAND_POSES[poseIndex]
    const nextIndex = (poseIndex + 1) % HAND_POSES.length
    const to = HAND_POSES[nextIndex]
    // Keçid müddəti: videonun effektiv kadr sürəti 20fps → 0.05s per kadr.
    const isLoopRestart = nextIndex === 0
    poseT.value = 0
    gsap.to(poseT, {
      value: 1,
      duration: 0.085,
      ease: 'none',
      onUpdate: () => {
        const t = poseT.value
        for (let i = 0; i < jointCount; i++) {
          current[i * 3] = from[i][0] + (to[i][0] - from[i][0]) * t
          current[i * 3 + 1] = from[i][1] + (to[i][1] - from[i][1]) * t
          current[i * 3 + 2] = from[i][2] + (to[i][2] - from[i][2]) * t
        }
        applyHandMeshes()
      },
      onComplete: () => {
        poseIndex = nextIndex
        // Dövrənin yenidən başlamasında qısa pauza - təbii nəfəs hissi.
        poseTimer = gsap.delayedCall(isLoopRestart ? 1.5 : 0, morphToNextPose)
      },
    })
  }
  poseTimer = gsap.delayedCall(1, morphToNextPose)

  // ---- fırlanma: yavaş avtomatik dövr + kursor/toxunma ilə orbit ----
  const userOffset = { x: 0, y: 0 }
  const drag = { active: false, lastX: 0, lastY: 0, pointerId: null }

  function onPointerDown(event) {
    drag.active = true
    drag.pointerId = event.pointerId
    drag.lastX = event.clientX
    drag.lastY = event.clientY
    canvas.setPointerCapture(event.pointerId)
    gsap.killTweensOf(userOffset)
  }
  function onPointerMove(event) {
    if (!drag.active) return
    const dx = event.clientX - drag.lastX
    const dy = event.clientY - drag.lastY
    drag.lastX = event.clientX
    drag.lastY = event.clientY
    userOffset.y += dx * 0.008
    userOffset.x = THREE.MathUtils.clamp(userOffset.x + dy * 0.008, -0.6, 0.6)
  }
  function onPointerUp(event) {
    if (drag.pointerId !== null) {
      try {
        canvas.releasePointerCapture(drag.pointerId)
      } catch {
        /* pointer capture may already be released */
      }
    }
    drag.active = false
    drag.pointerId = null
    // Buraxılanda yavaş-yavaş öz avtomatik fırlanmasına qayıdır.
    gsap.to(userOffset, { x: 0, y: 0, duration: 1.4, ease: 'power2.out' })
  }

  if (enablePointerOrbit) {
    canvas.style.touchAction = 'none'
    canvas.addEventListener('pointerdown', onPointerDown)
    canvas.addEventListener('pointermove', onPointerMove)
    canvas.addEventListener('pointerup', onPointerUp)
    canvas.addEventListener('pointercancel', onPointerUp)
  } else {
    // Canvas toxunma/scroll hadisələrini tutmasın - barmaq sürüşdürməsi
    // altındakı səhifəyə keçsin.
    canvas.style.pointerEvents = 'none'
  }

  // ---- fon: noise dalğa + grain shader-i (yalnız masaüstündə) ----
  let wavePlane = null
  let waveUniforms = null

  if (isDesktop) {
    waveUniforms = {
      uAspect: { value: 1 },
    }

    const waveMaterial = new THREE.ShaderMaterial({
      vertexShader: WAVE_VERTEX,
      fragmentShader: WAVE_FRAGMENT,
      uniforms: waveUniforms,
      depthWrite: false,
    })
    wavePlane = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), waveMaterial)
    wavePlane.position.z = BG_PLANE_Z
    scene.add(wavePlane)
  }

  // ---- kamera/fon yerləşməsi: ox-bazlı sərhəd qutusuna və cari aspect-ə görə ----
  function layoutScene() {
    const { clientWidth: w, clientHeight: h } = canvas
    if (!w || !h) return
    renderer.setSize(w, h, false)
    camera.aspect = w / h

    const scaledHalf = [HAND_FRAME.halfExtents[0] * HAND_SCALE, HAND_FRAME.halfExtents[1] * HAND_SCALE, HAND_FRAME.halfExtents[2] * HAND_SCALE]
    const distance = fitDistanceForBox(camera, scaledHalf)
    camera.position.set(0, 0, distance)
    camera.lookAt(0, 0, 0)
    camera.far = distance - BG_PLANE_Z + 20
    camera.updateProjectionMatrix()

    // Qalan boş şaquli yerin bir hissəsi əli yuxarı itələyir - marja
    // tələbini (MARGIN) pozmadan, çünki spare həmişə ≥ 0 və sürüşdürmə
    // spare-dən az qalır.
    const vHalf = (camera.fov * Math.PI) / 360
    const visibleHalfHeight = distance * Math.tan(vHalf)
    const requiredHalf = scaledHalf[1] * (1 + MARGIN)
    const spare = Math.max(0, visibleHalfHeight - requiredHalf)
    frame.position.y = spare * UPWARD_BIAS

    if (wavePlane) {
      const depth = distance - BG_PLANE_Z
      const vFov = (camera.fov * Math.PI) / 180
      const planeHeight = 2 * Math.tan(vFov / 2) * depth
      const planeWidth = planeHeight * camera.aspect
      wavePlane.geometry.dispose()
      wavePlane.geometry = new THREE.PlaneGeometry(planeWidth, planeHeight)
      waveUniforms.uAspect.value = camera.aspect
    }
  }
  layoutScene()

  // ---- görünürlük: hero ekrandan çıxanda render dayanır ----
  let isVisible = true
  const io = new IntersectionObserver(
    (entries) => entries.forEach((entry) => {
      isVisible = entry.isIntersecting
    }),
    { threshold: 0 }
  )
  io.observe(canvas)

  const ro = new ResizeObserver(layoutScene)
  ro.observe(canvas)

  let rafId = null
  function tick() {
    rafId = requestAnimationFrame(tick)
    if (!isVisible) return
    pivot.rotation.y = userOffset.y
    pivot.rotation.x = userOffset.x

    renderer.render(scene, camera)
  }
  tick()

  return {
    destroy() {
      cancelAnimationFrame(rafId)
      io.disconnect()
      ro.disconnect()
      poseTimer?.kill()
      gsap.killTweensOf(poseT)
      gsap.killTweensOf(userOffset)
      canvas.removeEventListener('pointerdown', onPointerDown)
      canvas.removeEventListener('pointermove', onPointerMove)
      canvas.removeEventListener('pointerup', onPointerUp)
      canvas.removeEventListener('pointercancel', onPointerUp)
      if (wavePlane) {
        wavePlane.geometry.dispose()
        wavePlane.material.dispose()
      }
      jointGeometry.dispose()
      capsuleGeometry.dispose()
      palmGeometryLeft.dispose()
      palmGeometryRight.dispose()
      toonMaterialLeft.dispose()
      toonMaterialRight.dispose()
      renderer.dispose()
    },
  }
}
