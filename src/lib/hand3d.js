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

const JOINT_RADIUS = 0.045

// Bütün pozalar üzrə OX-BAZLI sərhəd qutusu (tək radius yox, əvvəlki
// origin/main versiyasında olduğu kimi) - kamera məsafəsi və "yuxarı-mərkəz"
// sürüşdürməsi bu qutuya görə hesablanır, hər iki ox üçün ayrıca (sfera-
// uyğunlaşma əvəzinə), heç bir pozada kəsilmə qalmasın (axis-based framing).
//
// `halfExtents` BÜTÜN 40 pozanın BİRLİKDƏ (union) sərhəd qutusudur - bu,
// kəsilmə-TƏHLÜKƏSİZLİYİ üçün lazımdır (kameranın HEÇ BİR pozada heç bir
// nöqtəni kəsməməsini təmin edən məcburi minimum). AMMA bu, TEK bir pozanın
// EKRANDA TUTDUĞU HƏQİQİ sahə ilə QARIŞDIRILMAMALIDIR - ölçülüb: tək-poza
// maksimum hündürlük union-un cəmi ~50%-i, maksimum en ~72%-idir (əllər
// müxtəlif pozalarda müxtəlif istiqamətlərə uzanır, heç bir TƏK poza hər
// iki oxda eyni anda ekstremuma çatmır). Ona görə `maxSpanHalf` DƏ
// hesablanır - bu, istənilən TƏK pozanın öz maksimum en/hündürlüyüdür
// (bütün pozalar üzrə maksimum), EKRAN ÖLÇÜSÜ hədəfləməsi üçün işlədilir
// (bax computeFramingDistance), `halfExtents` isə YALNIZ kəsilmə-təhlükəsizlik
// məhdudiyyəti kimi qalır.
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
  // Ən iri oynaq radiusuna + kiçik ehtiyat - mesh-in öz ölçüsü sərhəd
  // qutusundan kənara çıxmasın.
  const pad = JOINT_RADIUS + 0.03
  const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2]
  const halfExtents = [(max[0] - min[0]) / 2 + pad, (max[1] - min[1]) / 2 + pad, (max[2] - min[2]) / 2 + pad]

  let maxSpanW = 0
  let maxSpanH = 0
  poses.forEach((pose) => {
    let pMinX = Infinity, pMaxX = -Infinity, pMinY = Infinity, pMaxY = -Infinity
    pose.forEach(([x, y]) => {
      pMinX = Math.min(pMinX, x)
      pMaxX = Math.max(pMaxX, x)
      pMinY = Math.min(pMinY, y)
      pMaxY = Math.max(pMaxY, y)
    })
    maxSpanW = Math.max(maxSpanW, pMaxX - pMinX + 2 * pad)
    maxSpanH = Math.max(maxSpanH, pMaxY - pMinY + 2 * pad)
  })
  const maxSpanHalf = [maxSpanW / 2, maxSpanH / 2]

  return { center, halfExtents, maxSpanHalf }
}

const HAND_FRAME = computeHandFraming(HAND_POSES)

const isDesktop = window.matchMedia(BREAKPOINTS.isDesktop).matches
const isCoarsePointer = window.matchMedia('(pointer: coarse)').matches
// Toxunma cihazlarında (coarse pointer və ya <980px) barmaqla scroll əl
// çevirməyə getməsin deyə orbit idarəsi tamamilə sönür - əl yalnız avtomatik
// fırlanır. Masaüstündə kursor idarəsi olduğu kimi qalır.
const enablePointerOrbit = isDesktop && !isCoarsePointer

// Əl vizual olaraq pivot səviyyəsində bu qədər böyüdülür (bax initHand3D-dəki
// frame/pivot/model üç-qatlı struktur).
const HAND_SCALE = 1.6
// Əl "yuxarı-mərkəzə" qalxsın tələbi - boş qalan şaquli yerin bu faizi
// pivot-u yuxarı itələyir (bax layoutScene).
const UPWARD_BIAS = 0.6

// Kamera məsafəsi ekranda tutduğu SAHƏNİ birbaşa faizlə hədəfləyir.
// Perspektiv-həndəsədə tan(yarı-FOV) = yarı-ölçü/məsafə, ona görə tələb
// olunan faizi vermək üçün məsafə = ölçü/(faiz*tan(yarı-FOV)).
//
// VACİB: "ölçü" olaraq `HAND_FRAME.maxSpanHalf` işlədilir (TƏK bir pozanın
// öz maksimum en/hündürlüyü), YOX `halfExtents` (bütün 40 pozanın BİRLİKDƏ
// union qutusu) - ölçülüb, əks halda hədəf ~2x həddindən artıq böyük çıxır
// (heç bir tək poza union-un tam ölçüsünə çatmır). `halfExtents` YALNIZ
// TƏHLÜKƏSİZLİK DÖŞƏMƏSİ (safety floor) kimi işlədilir - nəticə məsafə HEÇ
// VAXT bu döşəmədən aşağı düşmür, kəsilmə riski olmadan.
const TARGET_FILL_HEIGHT = 0.5 // 45-55% aralığının ortası (masaüstü hədəfi, TƏK poza üçün)
const TARGET_FILL_WIDTH = 0.75 // 70-80% aralığının ortası (mobil hədəfi, TƏK poza üçün)
const SAFETY_FILL = 0.97 // kəsilmə-təhlükəsizlik döşəməsi: union qutusu HEÇ VAXT bunu keçməsin (3% ehtiyat)

function computeFramingDistance(camera) {
  const vHalf = (camera.fov * Math.PI) / 360
  const tanV = Math.tan(vHalf)
  const aspect = camera.aspect

  const scaledSpanHalfX = HAND_FRAME.maxSpanHalf[0] * HAND_SCALE
  const scaledSpanHalfY = HAND_FRAME.maxSpanHalf[1] * HAND_SCALE
  const scaledUnionHalfX = HAND_FRAME.halfExtents[0] * HAND_SCALE
  const scaledUnionHalfY = HAND_FRAME.halfExtents[1] * HAND_SCALE
  const scaledUnionHalfZ = HAND_FRAME.halfExtents[2] * HAND_SCALE

  const distanceForHeightTarget = scaledSpanHalfY / (TARGET_FILL_HEIGHT * tanV)
  const distanceForWidthTarget = scaledSpanHalfX / (TARGET_FILL_WIDTH * tanV * aspect)
  const distanceForHeightSafety = scaledUnionHalfY / (SAFETY_FILL * tanV)
  const distanceForWidthSafety = scaledUnionHalfX / (SAFETY_FILL * tanV * aspect)

  // Dərinlik (z) ehtiyatı - qutunun kameraya ən yaxın üzü də sığsın.
  return Math.max(distanceForHeightTarget, distanceForWidthTarget, distanceForHeightSafety, distanceForWidthSafety) + scaledUnionHalfZ
}

// Render qatında TƏHLÜKƏSİZ bayraq - true olanda əl pivotu Y oxu ətrafında
// 180° çevrilir, barmaqlar kameraya doğru baxar (bax tick() aşağıda).
// Pozaların datasına (handPoses.js) VƏ sıralamasına TOXUNULMUR. Default OFF
// (indiki hal) - ekranda baxıb qərar veriləcək, DƏYİŞDİRİLMƏYİB.
//
// QEYD (yan təsir): bu fırlanma X oxunu da güzgüləyir - sol əl (--hand-left,
// tünd mavi) ekranın sağına, sağ əl (--hand-right, açıq mavi) soluna keçər
// (rənglərin ekran-mövqeyi dəyişir, əllərin özü dəyişmir, çünki hər ikisi
// artıq bir-birinin güzgü əksidir).
const HAND_FACE_VIEWER_FLIP = false

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
// Fon artıq bu canvas-da DEYİL (CSS-dədir, bax hero.css .hero__bg/.hero__grain) -
// renderer tam şəffaf (alpha:true, setClearAlpha(0)), bu canvas YALNIZ əli
// çəkir. Yüklənmə ilə tam açılış arasında fon heç vaxt dəyişmir, çünki tək
// (CSS) qat var.
export function initHand3D(canvas) {
  // --hand-left/--hand-right base.css-də :root-da sabit təyin olunub, heç
  // yerdə override olunmur - əlin iki tərəfi bu iki rəngi daşıyır.
  const jointColor = readCssColor(document.documentElement, '--hand-left', '#0071ef')
  const boneColor = readCssColor(document.documentElement, '--hand-right', '#6fb1ff')

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
  HAND_CONNECTIONS.forEach(([a], i) => {
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

  // ---- kamera yerləşməsi: ox-bazlı sərhəd qutusuna və cari aspect-ə görə ----
  // Fon artıq CSS-dədir (.hero__bg/.hero__grain) - bu canvas YALNIZ əli
  // çəkir, şəffaf fon üzərində (renderer alpha:true + setClearAlpha(0)).
  function layoutScene() {
    const { clientWidth: w, clientHeight: h } = canvas
    if (!w || !h) return
    renderer.setSize(w, h, false)
    camera.aspect = w / h

    const distance = computeFramingDistance(camera)
    camera.position.set(0, 0, distance)
    camera.lookAt(0, 0, 0)
    camera.far = distance + 20
    camera.updateProjectionMatrix()

    // Qalan boş şaquli yer HƏQİQİ (HAND_SCALE ilə miqyaslanmış) əl ölçüsünə
    // görə hesablanır.
    const vHalf = (camera.fov * Math.PI) / 360
    const visibleHalfHeight = distance * Math.tan(vHalf)
    const requiredHalf = HAND_FRAME.halfExtents[1] * HAND_SCALE
    const spare = Math.max(0, visibleHalfHeight - requiredHalf)
    frame.position.y = spare * UPWARD_BIAS
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
    pivot.rotation.y = userOffset.y + (HAND_FACE_VIEWER_FLIP ? Math.PI : 0)
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
