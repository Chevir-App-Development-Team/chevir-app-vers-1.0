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

const isDesktop = window.matchMedia(BREAKPOINTS.isDesktop).matches

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

// Problemin kökü: əllər pozalarda bir-birindən uzaqlaşanda "iki əl birgə"
// sərhəd qutusu böyüyür, kamera geri çəkilir, hər əl kiçik görünür (ölçülüb:
// ~8% en, 19% hündürlük). Düzəliş: hər pozada hər əli (formasına/bucağına
// TOXUNMADAN, bütöv rigid sürüşdürmə ilə) iki əlin ortaq mərkəzinə doğru
// sıxırıq - bax compressPoseGap. Bu, statik olaraq BİR DƏFƏ (modul
// yüklənəndə) ADJUSTED_POSES-ə tətbiq olunur, sonra HAND_FRAME və bütün
// joint-mövqe istifadələri (setFromPose/morphToNextPose) bu massivdən
// oxuyur - computeFramingDistance-in özü dəyişmir, pozalar boyu kamera
// məsafəsi/mərkəz/ölçü eyni statik sxemlə (BİR dəfə) hesablanır.
function compressPoseGap(pose, compress) {
  function handCenter(lo, hi) {
    let cx = 0, cy = 0, cz = 0
    for (let i = lo; i < hi; i++) {
      cx += pose[i][0]
      cy += pose[i][1]
      cz += pose[i][2]
    }
    const n = hi - lo
    return [cx / n, cy / n, cz / n]
  }
  const leftCenter = handCenter(0, 21)
  const rightCenter = handCenter(21, 42)
  const mid = [
    (leftCenter[0] + rightCenter[0]) / 2,
    (leftCenter[1] + rightCenter[1]) / 2,
    (leftCenter[2] + rightCenter[2]) / 2,
  ]
  function shiftFor(center) {
    return [
      (center[0] - mid[0]) * (compress - 1),
      (center[1] - mid[1]) * (compress - 1),
      (center[2] - mid[2]) * (compress - 1),
    ]
  }
  const leftShift = shiftFor(leftCenter)
  const rightShift = shiftFor(rightCenter)
  return pose.map((landmark, i) => {
    const shift = i < 21 ? leftShift : rightShift
    return [landmark[0] + shift[0], landmark[1] + shift[1], landmark[2] + shift[2]]
  })
}

// Masaüstündə 0.6, mobildə (dar ekran) daha aqressiv 0.45 - iki yeganə
// "tənzimlənən rəqəm" (tapşırıqda tələb olunduğu kimi).
const GAP_COMPRESS = isDesktop ? 0.6 : 0.45
const gapCompressedPoses = HAND_POSES.map((pose) => compressPoseGap(pose, GAP_COMPRESS))

// SAFETY_FILL (0.97) toxunulmur - union-sərhəd (computeFramingDistance-dəki
// "təhlükəsizlik döşəməsi") HƏLƏ DƏ bir neçə "outlier" (həddindən artıq
// enli/hündür) pozanın təyin etdiyi UCA görə hesablanırdı. Əvəzinə: 40
// pozanın (gap-sıxmadan SONRA) öz en/hündürlüyünün 80-ci faizini (percentile)
// limit götürürük - bu limiti aşan HAZIR AZ SAYDA poza (bax aşağıdakı
// ölçmə - 16/40) rigid (formasına toxunmadan, yalnız öz ortaq mərkəzinə
// doğru bərabər miqyasla) kiçildilir, limit/ölçü əmsalı ilə (≤1, heç vaxt
// böyütmür). Nəticədə union-sərhəd kiçilir, kamera bir az da yaxınlaşır -
// SAFETY_FILL-in özü DƏYİŞMƏDƏN. Ölçülüb (70 və 80-ci faiz sınanıb): 80
// daha az pozaya toxunur VƏ bir az böyük orta ölçü verir, ona görə seçilib.
const OUTLIER_PERCENTILE = 80

function poseBoxWH(pose) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
  pose.forEach(([x, y]) => {
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  })
  return { w: maxX - minX, h: maxY - minY, cx: (minX + maxX) / 2, cy: (minY + maxY) / 2 }
}

function percentile(values, p) {
  const sorted = values.slice().sort((a, b) => a - b)
  const idx = (p / 100) * (sorted.length - 1)
  const lo = Math.floor(idx)
  const hi = Math.ceil(idx)
  if (lo === hi) return sorted[lo]
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo)
}

function capOutlierPoses(poses, pct) {
  const boxes = poses.map(poseBoxWH)
  const widthLimit = percentile(boxes.map((b) => b.w), pct)
  const heightLimit = percentile(boxes.map((b) => b.h), pct)
  return poses.map((pose, i) => {
    const box = boxes[i]
    const factor = Math.min(1, widthLimit / box.w, heightLimit / box.h)
    if (factor >= 1) return pose
    return pose.map(([x, y, z]) => [box.cx + (x - box.cx) * factor, box.cy + (y - box.cy) * factor, z * factor])
  })
}

// Hər pozanın öz iki-əl-birgə mərkəzi (x/y) FƏRQLİ idi - bu, "ən pis poza"nın
// (kənara sürüşmüş mərkəzli) tagline-a nə qədər yaxınlaşdığını təyin edən
// əsas amil idi (ölçülüb: cəmi ~18-19px marja). Düzəliş: hər pozanı (gap-
// sıxmadan SONRA, outlier-limitdən ƏVVƏL - limit artıq mərkəzlənmiş
// pozalara görə hesablansın deyə) rigid sürüşdürüb öz mərkəzini BÜTÜN
// pozalar üçün EYNİ sabit nöqtəyə (40 pozanın öz mərkəzlərinin ortalaması)
// köçürürük - z TOXUNULMUR. Nəticə: pozalar arası mövqe fərqi (yalnız ÖLÇÜ
// fərqi qalır) aradan qalxır, union-sərhəd xeyli kiçilir.
function recenterPoses(poses) {
  const centers = poses.map((pose) => {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
    pose.forEach(([x, y]) => {
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    })
    return [(minX + maxX) / 2, (minY + maxY) / 2]
  })
  const fixedX = centers.reduce((sum, c) => sum + c[0], 0) / centers.length
  const fixedY = centers.reduce((sum, c) => sum + c[1], 0) / centers.length
  return poses.map((pose, i) => {
    const dx = fixedX - centers[i][0]
    const dy = fixedY - centers[i][1]
    return pose.map(([x, y, z]) => [x + dx, y + dy, z])
  })
}

const recenteredPoses = recenterPoses(gapCompressedPoses)
const ADJUSTED_POSES = capOutlierPoses(recenteredPoses, OUTLIER_PERCENTILE)

// Sıxılmamış pozaların çərçivəsi YALNIZ aşağıdakı JOINT_RADIUS_COMPENSATION
// hesabı üçün saxlanılır (bax initHand3D) - joint-mövqe məntiqi bundan
// istifadə ETMİR, yalnız ADJUSTED_POSES/HAND_FRAME işlədilir.
const ORIGINAL_HAND_FRAME = computeHandFraming(HAND_POSES)
const HAND_FRAME = computeHandFraming(ADJUSTED_POSES)

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

// `frame` parametrləşdirilib (əvvəlcə birbaşa modul-səviyyəli HAND_FRAME-i
// oxuyurdu) - JOINT_RADIUS_COMPENSATION hesabında (aşağıda) EYNİ düsturu
// həm sıxılmış (HAND_FRAME), həm orijinal (ORIGINAL_HAND_FRAME) sərhədlə
// çağırmaq üçün lazımdır, başqa heç nə dəyişmir.
function computeFramingDistance(camera, frame) {
  const vHalf = (camera.fov * Math.PI) / 360
  const tanV = Math.tan(vHalf)
  const aspect = camera.aspect

  const scaledSpanHalfX = frame.maxSpanHalf[0] * HAND_SCALE
  const scaledSpanHalfY = frame.maxSpanHalf[1] * HAND_SCALE
  const scaledUnionHalfX = frame.halfExtents[0] * HAND_SCALE
  const scaledUnionHalfY = frame.halfExtents[1] * HAND_SCALE
  const scaledUnionHalfZ = frame.halfExtents[2] * HAND_SCALE

  const distanceForHeightTarget = scaledSpanHalfY / (TARGET_FILL_HEIGHT * tanV)
  const distanceForWidthTarget = scaledSpanHalfX / (TARGET_FILL_WIDTH * tanV * aspect)
  const distanceForHeightSafety = scaledUnionHalfY / (SAFETY_FILL * tanV)
  const distanceForWidthSafety = scaledUnionHalfX / (SAFETY_FILL * tanV * aspect)

  // Dərinlik (z) ehtiyatı - qutunun kameraya ən yaxın üzü də sığsın.
  return Math.max(distanceForHeightTarget, distanceForWidthTarget, distanceForHeightSafety, distanceForWidthSafety) + scaledUnionHalfZ
}

// Sıxılmış sərhəd kiçik olduğu üçün kamera YAXINLAŞIR (əl böyüyür) - bax
// yuxarı. Nöqtə (joint) radiusu dünya-vahidlərində SABİT qaldığından, bu
// yaxınlaşma onu da (əl ilə bərabər) böyüdərdi. Tapşırıqda "nöqtə radiusu
// artmasın, əvvəlki piksel ölçüsündə qalsın" tələb olunduğu üçün, kameranın
// yaxınlaşma nisbəti (bir dəfə, 35°-lik FOV və tipik masaüstü 1.6 aspektlə)
// hesablanıb JOINT_RADIUS-a TƏRS tətbiq olunur (bax initHand3D) - nəticədə
// ekrandakı nöqtə ölçüsü təxminən əvvəlki kimi qalır, yalnız əlin öz
// sahəsi (barmaq açılımı/sümük uzunluğu) böyüyür.
const REFERENCE_CAMERA = { fov: 35, aspect: 1.6 }
const JOINT_RADIUS_COMPENSATION =
  computeFramingDistance(REFERENCE_CAMERA, HAND_FRAME) / computeFramingDistance(REFERENCE_CAMERA, ORIGINAL_HAND_FRAME)

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
  const jointGeometry = new THREE.SphereGeometry(JOINT_RADIUS * JOINT_RADIUS_COMPENSATION, 10, 8)
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
    size: 0.28 * JOINT_RADIUS_COMPENSATION,
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
  setFromPose(ADJUSTED_POSES[0])

  // ---- pozalar arasında yumşaq keçid (nöqtələr sıçramır) ----
  let poseIndex = 0
  const poseT = { value: 0 }
  let poseTimer = null

  function morphToNextPose() {
    const from = ADJUSTED_POSES[poseIndex]
    const nextIndex = (poseIndex + 1) % ADJUSTED_POSES.length
    const to = ADJUSTED_POSES[nextIndex]
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

    const distance = computeFramingDistance(camera, HAND_FRAME)
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
    // Əlləri ümumi bir qədər aşağı sal - başlıq iki sətirli böyük heading-ə
    // çevrildiyi üçün (bax hero.css .hero__tagline-line) yuxarı yer azaldı.
    // QEYD: bunu `recenterPoses()`-in sabit nöqtəsində ETMƏK OLMAZ - istənilən
    // bircins sürüşdürmə `model.position = -HAND_FRAME.center` addımında
    // avtomatik ləğv olunur (riyazi sübut edilib: tərcümə edilmiş data-nın
    // öz hesablanan mərkəzi EYNİ miqdarda sürüşür, fərq sabit qalır). Ona
    // görə düzəliş BURADA, ekran-nisbi faizlə (hero hündürlüyünün bu qədər
    // hissəsi) - yeganə pipeline nöqtəsi harda faktiki təsir edir.
    const DOWNWARD_SHIFT_FRACTION = 0.34
    const downwardShift = DOWNWARD_SHIFT_FRACTION * (2 * visibleHalfHeight)
    frame.position.y = spare * UPWARD_BIAS - downwardShift
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
