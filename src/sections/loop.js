import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { BREAKPOINTS } from '../lib/gsapSetup.js'

export const loopMarkup = `
<section class="loop" id="loop-diagram" aria-labelledby="loop-heading">
  <div class="loop__intro">
    <h2 id="loop-heading" class="loop__heading">
      Chevir dövrəni hər iki istiqamətdə bağlayır
    </h2>
    <p class="loop__lead">
      Eşitmə məhdudiyyətli vətəndaşın jesti mətnə və səsə çevrilir, eşidən tərəfin cavabı
      yenidən jest dilinə qayıdır. Heç bir tərəf gözləmir.
    </p>
  </div>
  <div class="loop__stage">
    <div class="loop__side loop__side--signed" data-frame>
      <h3 class="loop__side-title">Eşitmə məhdudiyyətli şəxs</h3>
      <p class="loop__side-desc">Jest dilində danışır, kameraya baxır.</p>
    </div>

    <svg
      class="loop__svg loop__svg--desktop"
      viewBox="0 0 600 260"
      aria-hidden="false"
      focusable="false"
    >
      <path
        class="loop__path loop__path--signed"
        pathLength="100"
        d="M30,190 C220,60 380,220 570,90"
        aria-hidden="true"
      />
      <path
        class="loop__path-halo loop__path-halo--outer"
        pathLength="100"
        d="M570,170 C380,40 220,200 30,70"
        aria-hidden="true"
      />
      <path
        class="loop__path-halo loop__path-halo--mid"
        pathLength="100"
        d="M570,170 C380,40 220,200 30,70"
        aria-hidden="true"
      />
      <path
        class="loop__path loop__path--spoken"
        pathLength="100"
        d="M570,170 C380,40 220,200 30,70"
        aria-hidden="true"
      />
      <circle class="loop__marker loop__marker--signed" r="6" aria-hidden="true" />
      <circle class="loop__marker loop__marker--spoken" r="6" aria-hidden="true" />
      <text class="loop__stage-label" data-path="a" data-stage="0">jest</text>
      <text class="loop__stage-label" data-path="a" data-stage="1">gloss</text>
      <text class="loop__stage-label" data-path="a" data-stage="2">mətn + səs</text>
      <text class="loop__stage-label" data-path="b" data-stage="0">mətn + səs</text>
      <text class="loop__stage-label" data-path="b" data-stage="1">gloss</text>
      <text class="loop__stage-label" data-path="b" data-stage="2">jest</text>
    </svg>

    <svg
      class="loop__svg loop__svg--mobile"
      viewBox="0 0 220 520"
      aria-hidden="false"
      focusable="false"
    >
      <path
        class="loop__path loop__path--signed"
        pathLength="100"
        d="M75,25 C0,180 195,340 75,495"
        aria-hidden="true"
      />
      <path
        class="loop__path-halo loop__path-halo--outer"
        pathLength="100"
        d="M145,495 C220,340 25,180 145,25"
        aria-hidden="true"
      />
      <path
        class="loop__path-halo loop__path-halo--mid"
        pathLength="100"
        d="M145,495 C220,340 25,180 145,25"
        aria-hidden="true"
      />
      <path
        class="loop__path loop__path--spoken"
        pathLength="100"
        d="M145,495 C220,340 25,180 145,25"
        aria-hidden="true"
      />
      <circle class="loop__marker loop__marker--signed" r="6" aria-hidden="true" />
      <circle class="loop__marker loop__marker--spoken" r="6" aria-hidden="true" />
      <text class="loop__stage-label" data-path="a" data-stage="0">jest</text>
      <text class="loop__stage-label" data-path="a" data-stage="1">gloss</text>
      <text class="loop__stage-label" data-path="a" data-stage="2">mətn + səs</text>
      <text class="loop__stage-label" data-path="b" data-stage="0">mətn + səs</text>
      <text class="loop__stage-label" data-path="b" data-stage="1">gloss</text>
      <text class="loop__stage-label" data-path="b" data-stage="2">jest</text>
    </svg>

    <div class="loop__side loop__side--spoken" data-frame>
      <h3 class="loop__side-title">Sosial xidmət agenti</h3>
      <p class="loop__side-desc">Danışıq dilində eşidir və cavab verir.</p>
    </div>
  </div>
</section>
`

export function initLoop() {
  const section = document.querySelector('.loop')
  if (!section) return

  const stageEl = section.querySelector('.loop__stage')
  const sides = Array.from(section.querySelectorAll('[data-frame]'))

  function positionLabels(pathEl, labels, viewBoxWidth, offset, offsets, vertical) {
    const length = pathEl.getTotalLength()
    labels.forEach((el, i) => {
      const pt = pathEl.getPointAtLength(offsets[i] * length)
      if (vertical) {
        el.setAttribute('x', pt.x + offset)
        el.setAttribute('y', pt.y)
        el.setAttribute('text-anchor', offset < 0 ? 'end' : 'start')
      } else {
        el.setAttribute('x', pt.x)
        el.setAttribute('y', pt.y + offset)
        el.setAttribute(
          'text-anchor',
          pt.x < viewBoxWidth * 0.3 ? 'start' : pt.x > viewBoxWidth * 0.7 ? 'end' : 'middle'
        )
      }
    })
  }

  function configureVariant(svg) {
    const vertical = svg.classList.contains('loop__svg--mobile')
    const pathA = svg.querySelector('.loop__path--signed')
    const pathB = svg.querySelector('.loop__path--spoken')
    // Nazik xəttin "parıltısı" CSS filter:drop-shadow ilə işləmir (ölçülüb,
    // bax loop.css-dəki şərh) - əvəzinə pathB-nin arxasında, eyni "d" ilə,
    // daha enli/sönük iki əlavə xətt (halo) çəkilir. dashoffset-ləri pathB
    // ilə sinxron saxlanmalıdır ki, birlikdə "çəkilsinlər".
    const pathBHaloOuter = svg.querySelector('.loop__path-halo--outer')
    const pathBHaloMid = svg.querySelector('.loop__path-halo--mid')
    const markerA = svg.querySelector('.loop__marker--signed')
    const markerB = svg.querySelector('.loop__marker--spoken')
    const labelsA = Array.from(svg.querySelectorAll('[data-path="a"]')).sort(
      (a, b) => Number(a.dataset.stage) - Number(b.dataset.stage)
    )
    const labelsB = Array.from(svg.querySelectorAll('[data-path="b"]')).sort(
      (a, b) => Number(a.dataset.stage) - Number(b.dataset.stage)
    )
    const viewBoxWidth = svg.viewBox.baseVal.width
    if (vertical) {
      positionLabels(pathA, labelsA, viewBoxWidth, -14, [0.08, 0.4, 0.86], true)
      positionLabels(pathB, labelsB, viewBoxWidth, 14, [0.08, 0.6, 0.86], true)
    } else {
      positionLabels(pathA, labelsA, viewBoxWidth, -14, [0.1, 0.38, 0.88], false)
      positionLabels(pathB, labelsB, viewBoxWidth, 22, [0.1, 0.62, 0.88], false)
    }
    return { pathA, pathB, pathBHaloOuter, pathBHaloMid, markerA, markerB, labelsA, labelsB }
  }

  const desktopVariant = configureVariant(section.querySelector('.loop__svg--desktop'))
  const mobileVariant = configureVariant(section.querySelector('.loop__svg--mobile'))

  function setStaticEnd(variant) {
    const { pathA, pathB, pathBHaloOuter, pathBHaloMid, labelsA, labelsB, markerA, markerB } = variant
    gsap.set([pathA, pathB, pathBHaloOuter, pathBHaloMid], { strokeDashoffset: 0 })
    gsap.set([markerA, markerB], { opacity: 0 })
    labelsA.forEach((el, i) => el.classList.toggle('is-active', i === labelsA.length - 1))
    labelsB.forEach((el, i) => el.classList.toggle('is-active', i === labelsB.length - 1))
  }

  function setStaticStart(variant) {
    const { pathA, pathB, pathBHaloOuter, pathBHaloMid, labelsA, labelsB, markerA, markerB } = variant
    gsap.set([pathA, pathB, pathBHaloOuter, pathBHaloMid], { strokeDashoffset: 100 })
    gsap.set([markerA, markerB], { opacity: 0 })
    labelsA.forEach((el) => el.classList.remove('is-active'))
    labelsB.forEach((el) => el.classList.remove('is-active'))
  }

  function updateStageLabels(labels, t) {
    const stage = t < 0.33 ? 0 : t < 0.66 ? 1 : 2
    labels.forEach((el, i) => el.classList.toggle('is-active', i === stage))
  }

  // getPointAtLength() ölçülüb: scroll zamanı onUpdate-in hər tick-ində
  // çağırılanda (köhnə kod) profil 4x CPU throttle-da 386ms-lik tək task
  // göstərdi - mobildə bu animasiya pin-ə bağlı olmadığı üçün (aşağıda)
  // istifadəçi hələ sərbəst scroll edərkən main thread-i tutub "geri atma"
  // hissi yaradırdı. Həll: nöqtələr ANİMASİYA BAŞLAMAZDAN ƏVVƏL (bir dəfə,
  // aşağıda buildPointCache-də) əvvəlcədən hesablanır, onUpdate isə yalnız
  // keşdən oxuyub iki qonşu nöqtə arasında xətti interpolyasiya edir - CPU
  // baxımından demək olar pulsuzdur. Keş resize-də YENİDƏN QURULMUR: path
  // "d" atributu (deməli getTotalLength/getPointAtLength) SVG-nin öz
  // viewBox koordinatındadır, ekran ölçüsündən asılı deyil (ölçülüb: 390px
  // və 1440px-də eyni uzunluq) - yalnız desktop/mobil ARASINDA keçiddə
  // (mm.add aşağıda) hər variant üçün ayrıca, təzə keş qurulur.
  function buildPointCache(pathEl) {
    const length = pathEl.getTotalLength()
    // Uzunluğa mütənasib addım sayı (200-400 aralığında) - path nə qədər
    // uzundursa, hamar görünmək üçün bir o qədər çox nöqtə lazımdır, amma
    // yaddaşa görə yuxarı hədd qoyulub.
    const sampleCount = Math.min(400, Math.max(200, Math.round(length / 1.5)))
    const points = new Array(sampleCount + 1)
    for (let i = 0; i <= sampleCount; i++) {
      const pt = pathEl.getPointAtLength((i / sampleCount) * length)
      points[i] = { x: pt.x, y: pt.y }
    }
    return points
  }

  function pointAtProgress(points, t) {
    const scaled = t * (points.length - 1)
    const i0 = Math.floor(scaled)
    const i1 = Math.min(i0 + 1, points.length - 1)
    const frac = scaled - i0
    const p0 = points[i0]
    const p1 = points[i1]
    return {
      x: p0.x + (p1.x - p0.x) * frac,
      y: p0.y + (p1.y - p0.y) * frac,
    }
  }

  function buildPathProgress(pathEl, markerEl, labels, syncEls = []) {
    const proxy = { t: 0 }
    const points = buildPointCache(pathEl)
    return {
      proxy,
      onUpdate: () => {
        const offset = String(100 * (1 - proxy.t))
        pathEl.style.strokeDashoffset = offset
        syncEls.forEach((el) => {
          el.style.strokeDashoffset = offset
        })
        const pt = pointAtProgress(points, proxy.t)
        markerEl.setAttribute('cx', pt.x)
        markerEl.setAttribute('cy', pt.y)
        updateStageLabels(labels, proxy.t)
      },
    }
  }

  function buildTimeline(variant, scrollTriggerVars) {
    const { pathA, pathB, pathBHaloOuter, pathBHaloMid, markerA, markerB, labelsA, labelsB } = variant
    const a = buildPathProgress(pathA, markerA, labelsA)
    const b = buildPathProgress(pathB, markerB, labelsB, [pathBHaloOuter, pathBHaloMid])

    return gsap.timeline({
      paused: !scrollTriggerVars,
      scrollTrigger: scrollTriggerVars || undefined,
    })
      .to(markerA, { opacity: 1, duration: 0.01 }, 0)
      .to(a.proxy, { t: 1, duration: 4.5, ease: 'none', onUpdate: a.onUpdate }, 0)
      .to(markerA, { opacity: 0, duration: 0.01 }, 4.49)
      .to(markerB, { opacity: 1, duration: 0.01 }, 4.5)
      .to(b.proxy, { t: 1, duration: 4.5, ease: 'none', onUpdate: b.onUpdate }, 4.5)
      .to(markerB, { opacity: 0, duration: 0.01 }, 8.99)
      .to(sides, { scale: 1.035, duration: 0.4, ease: 'power2.out' }, 9)
      .to(sides, { scale: 1, duration: 0.4, ease: 'power2.inOut' }, 9.4)
  }

  const mm = gsap.matchMedia()

  mm.add(BREAKPOINTS, (context) => {
    const { isDesktop, reduceMotion } = context.conditions
    const activeVariant = isDesktop ? desktopVariant : mobileVariant

    if (reduceMotion) {
      setStaticEnd(activeVariant)
      return
    }

    setStaticStart(activeVariant)

    if (isDesktop) {
      buildTimeline(activeVariant, {
        trigger: stageEl,
        start: 'center center',
        end: '+=250%',
        pin: true,
        scrub: 1,
      })
      ScrollTrigger.refresh()
      return
    }

    const tl = buildTimeline(activeVariant, null)
    tl.timeScale(3.5) // scrub durations are tuned for scroll speed, not real time; speed up for a one-shot autoplay
    ScrollTrigger.create({
      trigger: section,
      start: 'top 75%',
      once: true,
      onEnter: () => tl.play(),
    })
  })

  document.fonts?.ready?.then(() => ScrollTrigger.refresh())
}
