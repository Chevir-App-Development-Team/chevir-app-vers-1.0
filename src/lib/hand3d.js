import * as THREE from 'three'
import gsap from 'gsap'
import { HAND_POSES, HAND_CONNECTIONS } from './handPoses.js'
import { BREAKPOINTS } from './gsapSetup.js'

// Hero-nun --signed/--spoken dəyərlərini oxuyur ki, rənglər CSS-dəki tək
// mənbədən gəlsin (hero.css-də hər ikisi ağa çəkilib).
function readCssColor(el, varName, fallback) {
  const value = getComputedStyle(el).getPropertyValue(varName).trim()
  return value || fallback
}

const JOINT_RADIUS = 0.045

// 21 nöqtənin (bütün aktiv pozalar üzrə) sərhəd qutusunun mərkəzini və bu
// mərkəzdən ən uzaq nöqtəyə olan məsafəni hesablayır. Kamera bu radiusa görə
// uyğunlaşır (bax: fitDistanceForRadius) - sabit kamera məsafəsi əvəzinə.
function computeHandFraming(poses) {
  const min = [Infinity, Infinity, Infinity]
  const max = [-Infinity, -Infinity, -Infinity]
  poses.forEach((pose) => {
    pose.forEach(([x, y, z]) => {
      min[0] = Math.min(min[0], x); max[0] = Math.max(max[0], x)
      min[1] = Math.min(min[1], y); max[1] = Math.max(max[1], y)
      min[2] = Math.min(min[2], z); max[2] = Math.max(max[2], z)
    })
  })
  const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2]
  let radius = 0
  poses.forEach((pose) => {
    pose.forEach(([x, y, z]) => {
      const dx = x - center[0]
      const dy = y - center[1]
      const dz = z - center[2]
      radius = Math.max(radius, Math.sqrt(dx * dx + dy * dy + dz * dz))
    })
  })
  return { center, radius: radius + JOINT_RADIUS }
}

const HAND_FRAME = computeHandFraming(HAND_POSES)

const isDesktop = window.matchMedia(BREAKPOINTS.isDesktop).matches
const isCoarsePointer = window.matchMedia('(pointer: coarse)').matches
// Toxunma cihazlarında (coarse pointer və ya <980px) barmaqla scroll əl
// çevirməyə getməsin deyə orbit idarəsi tamamilə sönür - əl yalnız avtomatik
// fırlanır. Masaüstündə kursor idarəsi olduğu kimi qalır.
const enablePointerOrbit = isDesktop && !isCoarsePointer

// Radiuslu sferanı (bütün pozalar + istənilən fırlanma bucağı - sfera
// fırlanmaya görə dəyişməz olduğu üçün) kadra tam sığdıran kamera məsafəsi.
// Sfera-uyğunlaşma düsturu riyazi cəhətdən kəsilməyə qarşı təminatlıdır
// (istənilən MARGIN_FILL<1 üçün). Canvas indi tam ekranı örtdüyündən
// (hero.css), mobildə əl wordmark-a qədər böyüyüb onunla kəsişməsin deyə
// <980px-də daha kiçik dəyər işlədilir.
// 2 əl (42 nöqtə) üçün sərhəd qutusu daha genişdir, buna görə standart
// MARGIN_FILL dumanlı/kiçik görünməsinə səbəb olur. Kameranı yaxınlaşdırmaq
// üçün dəyərlər artırılıb. (Desktopda əllərin daha böyük görünməsi üçün 1.75)
const MARGIN_FILL = isDesktop ? 1.75 : 1.0

function fitDistanceForRadius(camera, radius) {
  const vHalf = (camera.fov * Math.PI) / 360
  const hHalf = Math.atan(Math.tan(vHalf) * camera.aspect)
  const effRadius = radius / MARGIN_FILL
  return Math.max(effRadius / Math.sin(vHalf), effRadius / Math.sin(hHalf))
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

// Səhifənin tünd fon rəngi (#16130F - base.css-dəki --paper tündrejim dəyəri,
// loop.css-də də eyni sabit işlədilir). Hero həmişə tünd qaldığı üçün (bax:
// hero.css) bu, dəyişən deyil, birbaşa yazılıb - hero ilə pipeline (dark mode-da)
// arasında rəng tikişi qalmasın deyə.
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

    vec3 base = vec3(0.0863, 0.0745, 0.0588); // #16130F
    vec3 color = base + vec3(light) * 0.55;

    // Grain - işığın gücünə bağlı (qaranlıqda demək olar yox, işıqda aydın),
    // sabit (uTime yoxdur, kadr-kadr eyni), fərdi piksel təsadüfiliyi.
    float grainAmt = mix(0.012, 0.16, light);
    float grain = (hash(gl_FragCoord.xy) - 0.5) * grainAmt;
    color += vec3(grain);

    gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
  }
`

// Nöqtə halosu üçün yumşaq radial sprite - bir dəfə çəkilir, bütün joint
// glow-ları paylaşır.
let glowTextureCache = null
function getGlowTexture() {
  if (glowTextureCache) return glowTextureCache
  const size = 64
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = size
  const ctx = canvas.getContext('2d')
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2)
  gradient.addColorStop(0, 'rgba(255,255,255,1)')
  gradient.addColorStop(0.4, 'rgba(255,255,255,0.45)')
  gradient.addColorStop(1, 'rgba(255,255,255,0)')
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, size, size)
  glowTextureCache = new THREE.CanvasTexture(canvas)
  return glowTextureCache
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
  // hero.css overrides --signed and --spoken to white.
  // We bypass that by reading from the root document, or using our fallback colors
  // so the two hands have distinct, readable colors.
  const jointColor = readCssColor(document.documentElement, '--signed', '#d6006c')
  const boneColor = readCssColor(document.documentElement, '--spoken', '#0088b0')

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, isDesktop ? 2 : 1.5))
  renderer.setClearAlpha(0)

  const scene = new THREE.Scene()
  const camera = new THREE.PerspectiveCamera(35, 1, 0.05, 100)

  // pivot/model ayrımı - fırlanma əlin sərhəd qutusunun mərkəzi ətrafında
  // olsun deyə (HAND_FRAME.center), koordinatların özü toxunulmaz qalır.
  const pivot = new THREE.Group()
  const model = new THREE.Group()
  // Desktopda əllər böyüyəndə yuxarıdan kəsilməsin deyə modeli aşağı çəkirik
  const yShift = isDesktop ? -0.4 : -0.2
  model.position.set(-HAND_FRAME.center[0], -HAND_FRAME.center[1] + yShift, -HAND_FRAME.center[2])
  pivot.add(model)
  scene.add(pivot)

  const jointCount = HAND_POSES[0].length
  const jointGeometry = new THREE.SphereGeometry(JOINT_RADIUS, 10, 8)
  const jointMaterial = new THREE.MeshBasicMaterial()
  const joints = new THREE.InstancedMesh(jointGeometry, jointMaterial, jointCount)
  const color1 = new THREE.Color(jointColor)
  const color2 = new THREE.Color(boneColor)
  for (let i = 0; i < jointCount; i++) {
    joints.setColorAt(i, i < 21 ? color1 : color2)
  }
  joints.instanceColor.needsUpdate = true
  model.add(joints)

  const boneGeometry = new THREE.BufferGeometry()
  const bonePositions = new Float32Array(HAND_CONNECTIONS.length * 2 * 3)
  const boneColors = new Float32Array(HAND_CONNECTIONS.length * 2 * 3)
  HAND_CONNECTIONS.forEach(([a, b], i) => {
    const isLeft = a < 21
    const col = isLeft ? color1 : color2
    boneColors[i * 6] = col.r
    boneColors[i * 6 + 1] = col.g
    boneColors[i * 6 + 2] = col.b
    boneColors[i * 6 + 3] = col.r
    boneColors[i * 6 + 4] = col.g
    boneColors[i * 6 + 5] = col.b
  })
  boneGeometry.setAttribute('position', new THREE.BufferAttribute(bonePositions, 3))
  boneGeometry.setAttribute('color', new THREE.BufferAttribute(boneColors, 3))
  const boneMaterial = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: false,
    opacity: 1.0,
  })
  const bones = new THREE.LineSegments(boneGeometry, boneMaterial)
  model.add(bones)

  const current = new Float32Array(jointCount * 3)
  const glowColors = new Float32Array(jointCount * 3)
  for (let i = 0; i < jointCount; i++) {
    const col = i < 21 ? color1 : color2
    glowColors[i * 3] = col.r
    glowColors[i * 3 + 1] = col.g
    glowColors[i * 3 + 2] = col.b
  }
  const dummy = new THREE.Object3D()

  // Ağ halo - additiv "glow" nöqtələri
  const glowGeometry = new THREE.BufferGeometry()
  glowGeometry.setAttribute('position', new THREE.BufferAttribute(current, 3))
  glowGeometry.setAttribute('color', new THREE.BufferAttribute(glowColors, 3))
  const glowMaterial = new THREE.PointsMaterial({
    map: getGlowTexture(),
    vertexColors: true,
    size: 0.28,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.95,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  })
  const glow = new THREE.Points(glowGeometry, glowMaterial)
  model.add(glow)

  function applyJoints() {
    for (let i = 0; i < jointCount; i++) {
      dummy.position.set(current[i * 3], current[i * 3 + 1], current[i * 3 + 2])
      dummy.updateMatrix()
      joints.setMatrixAt(i, dummy.matrix)
    }
    joints.instanceMatrix.needsUpdate = true
    glowGeometry.attributes.position.needsUpdate = true

    const posAttr = boneGeometry.attributes.position
    HAND_CONNECTIONS.forEach(([a, b], i) => {
      posAttr.setXYZ(i * 2, current[a * 3], current[a * 3 + 1], current[a * 3 + 2])
      posAttr.setXYZ(i * 2 + 1, current[b * 3], current[b * 3 + 1], current[b * 3 + 2])
    })
    posAttr.needsUpdate = true
  }

  function setFromPose(pose) {
    for (let i = 0; i < jointCount; i++) {
      current[i * 3] = pose[i][0]
      current[i * 3 + 1] = pose[i][1]
      current[i * 3 + 2] = pose[i][2]
    }
    applyJoints()
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
        applyJoints()
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
  let autoRotation = 0
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

  // ---- kamera/fon yerləşməsi: sferanın radiusuna və cari aspect-ə görə ----
  function layoutScene() {
    const { clientWidth: w, clientHeight: h } = canvas
    if (!w || !h) return
    renderer.setSize(w, h, false)
    camera.aspect = w / h

    const distance = fitDistanceForRadius(camera, HAND_FRAME.radius)
    camera.position.set(0, 0, distance)
    camera.lookAt(0, 0, 0)
    camera.far = distance - BG_PLANE_Z + 20
    camera.updateProjectionMatrix()

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
      jointMaterial.dispose()
      boneGeometry.dispose()
      boneMaterial.dispose()
      glowGeometry.dispose()
      glowMaterial.dispose()
      renderer.dispose()
    },
  }
}
