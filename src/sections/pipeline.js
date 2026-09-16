import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { BREAKPOINTS } from '../lib/gsapSetup.js'

// public/ altındakı asset-lərə mütləq "/..." yolla deyil, BASE_URL ilə
// istinad olunur (kök yolda "/" verir, amma alt qovluqda yayımlansa da işləyər).
const BASE_URL = import.meta.env.BASE_URL

export const pipelineMarkup = `
<section class="pipeline" id="how-it-works" aria-labelledby="pipeline-heading">
  <div class="pipeline__intro">
    <h2 id="pipeline-heading" class="pipeline__heading">Necə işləyir</h2>
    <p class="pipeline__lead">
      İşarə dilində danışanlarla ünsiyyət hər iki istiqamətdə asanlaşır -
      işarələr mətnə, mətn isə işarə dilinə tərcümə olunur.
    </p>
  </div>
  <div class="pipeline__grid">
    <div class="pipeline__media" aria-hidden="true"></div>
    <ol class="pipeline__steps">
      <li class="pipeline__step" data-step="0" data-channel="signed">
        <span class="pipeline__number">01</span>
        <h3 class="pipeline__step-title" id="pipeline-step-title-0">
          Görüntü və izləmə
        </h3>
        <p class="pipeline__step-desc">
          Kamera əl, barmaq və mimika hərəkətlərini real vaxtda izləyir və
          AZİD (Azərbaycan İşarə Dili) işarələrinin ardıcıllığını qeyd edir.
        </p>
        <div class="pipeline__step-media">
          <video
            class="pipeline__video"
            data-step-video="0"
            muted
            loop
            playsinline
            preload="metadata"
            poster="${BASE_URL}media/track.jpg"
            aria-labelledby="pipeline-step-title-0"
            aria-describedby="pipeline-step-video-desc-0"
          >
            <source src="${BASE_URL}media/track.webm" type="video/webm" />
            <source src="${BASE_URL}media/track.mp4" type="video/mp4" />
          </video>
          <p class="visually-hidden" id="pipeline-step-video-desc-0">
            Video: kamera əl və barmaq hərəkətlərini nöqtələrlə izləyir.
          </p>
        </div>
      </li>
      <li class="pipeline__step" data-step="1" data-channel="signed">
        <span class="pipeline__number">02</span>
        <h3 class="pipeline__step-title" id="pipeline-step-title-1">Tanıma</h3>
        <p class="pipeline__step-desc">
          Model işarələri tanıyır; diqqət xəritəsi əl formasının hansı
          nöqtələrinə əsaslandığını göstərir.
        </p>
        <div class="pipeline__step-media">
          <video
            class="pipeline__video"
            data-step-video="1"
            muted
            loop
            playsinline
            preload="metadata"
            poster="${BASE_URL}media/gradcam.jpg"
            aria-labelledby="pipeline-step-title-1"
            aria-describedby="pipeline-step-video-desc-1"
          >
            <source src="${BASE_URL}media/gradcam.webm" type="video/webm" />
            <source src="${BASE_URL}media/gradcam.mp4" type="video/mp4" />
          </video>
          <p class="visually-hidden" id="pipeline-step-video-desc-1">
            Video: modelin diqqət xəritəsi əl formasının hansı nöqtələrini
            əsas götürdüyünü rəngli overlay ilə göstərir.
          </p>
        </div>
      </li>
      <li class="pipeline__step" data-step="2" data-channel="spoken">
        <span class="pipeline__number">03</span>
        <h3 class="pipeline__step-title" id="pipeline-step-title-2">
          Tərcümə və səs
        </h3>
        <p class="pipeline__step-desc">
          Tanınan işarələr danışıq Azərbaycan dilinə çevrilir və səsli və ya
          yazılı şəkildə ötürülür.
        </p>
      </li>
      <li class="pipeline__step" data-step="3" data-channel="spoken">
        <span class="pipeline__number">04</span>
        <h3 class="pipeline__step-title" id="pipeline-step-title-3">
          Cavab jest dilində
        </h3>
        <p class="pipeline__step-desc">
          Eşidən tərəfin cavabı AZİD-ə çevrilir və avatar vasitəsilə jest
          dilində göstərilir. Funksiya prototip mərhələsindədir.
        </p>
        <div class="pipeline__step-media">
          <video
            class="pipeline__video"
            data-step-video="3"
            muted
            loop
            playsinline
            preload="metadata"
            poster="${BASE_URL}media/demo.jpg"
            aria-labelledby="pipeline-step-title-3"
            aria-describedby="pipeline-step-video-desc-3"
          >
            <source src="${BASE_URL}media/demo.webm" type="video/webm" />
            <source src="${BASE_URL}media/demo.mp4" type="video/mp4" />
          </video>
          <p class="visually-hidden" id="pipeline-step-video-desc-3">
            Video: avatar mətni işarə dilinə çevirib jest dilində göstərir.
          </p>
        </div>
      </li>
    </ol>
  </div>
</section>
`

export function initPipeline() {
  const section = document.querySelector('.pipeline')
  if (!section) return

  const mediaPanel = section.querySelector('.pipeline__media')
  const gridEl = section.querySelector('.pipeline__grid')
  const steps = Array.from(section.querySelectorAll('.pipeline__step'))
  const videos = Array.from(section.querySelectorAll('.pipeline__video'))
  // "Tərcümə və səs" (03) addımının öz medyası yoxdur - sticky paneldə
  // əvvəlki addımın (02, gradcam) videosu dəyişmədən qalır.
  const NO_MEDIA_STEP_INDEX = 2

  // Desktop qolunda ilkin setActiveStep(0) səhifə açılan kimi, scroll-dan
  // əvvəl çağırıldığı üçün .play()-i bölmə görünənə qədər gecikdirmək
  // lazımdır. mediaReady yalnız aşağıdakı IntersectionObserver kəsişəndə true olur.
  let mediaReady = false
  let activeIndex = 0

  function setActiveStep(index) {
    activeIndex = index
    steps.forEach((step, i) => {
      step.classList.toggle('is-active', i === index)
    })
    if (index === NO_MEDIA_STEP_INDEX) return
    videos.forEach((video) => {
      const isActiveVideo = Number(video.dataset.stepVideo) === index
      video.classList.toggle('is-active', isActiveVideo)
      if (isActiveVideo) {
        if (mediaReady) video.play().catch(() => {})
      } else {
        video.pause()
      }
    })
  }

  // Mənfi alt rootMargin bölmənin dəqiq ekran kənarına toxunduğu (amma hələ
  // görünmədiyi) anda tetiklənməsinin qarşısını alır - bölmə həqiqətən bir
  // qədər görünənə qədər gözləyir.
  const lazyMediaObserver = new IntersectionObserver(
    (entries) => {
      if (!entries.some((entry) => entry.isIntersecting)) return
      mediaReady = true
      setActiveStep(activeIndex)
      lazyMediaObserver.disconnect()
    },
    { rootMargin: '0px 0px -100px 0px' }
  )
  lazyMediaObserver.observe(section)

  function setAllStepsStatic() {
    steps.forEach((step) => step.classList.add('is-active'))
  }

  function returnVideosToSteps() {
    steps.forEach((step) => {
      const slot = step.querySelector('.pipeline__step-media')
      const video = slot?.querySelector('.pipeline__video')
      if (video && video.parentElement !== slot) slot.appendChild(video)
    })
  }

  let visibilityObserver = null
  function watchVideoVisibility() {
    visibilityObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) entry.target.play().catch(() => {})
          else entry.target.pause()
        })
      },
      { threshold: 0.1, rootMargin: '200px' }
    )
    videos.forEach((video) => visibilityObserver.observe(video))
    return () => visibilityObserver.disconnect()
  }

  const mm = gsap.matchMedia()

  mm.add(BREAKPOINTS, (context) => {
    const { isDesktop, reduceMotion } = context.conditions
    returnVideosToSteps()

    if (isDesktop && !reduceMotion) {
      section.classList.add('pipeline--desktop')
      videos.forEach((video) => mediaPanel.appendChild(video))
      setActiveStep(0)

      const pinTrigger = ScrollTrigger.create({
        trigger: mediaPanel,
        start: 'center center',
        endTrigger: gridEl,
        end: 'bottom bottom',
        pin: mediaPanel,
      })

      const stepTriggers = steps.map((step, i) =>
        ScrollTrigger.create({
          trigger: step,
          start: 'top center',
          end: 'bottom center',
          onEnter: () => setActiveStep(i),
          onEnterBack: () => setActiveStep(i),
        })
      )

      ScrollTrigger.refresh()

      return () => {
        pinTrigger.kill()
        stepTriggers.forEach((trigger) => trigger.kill())
        videos.forEach((video) => video.pause())
      }
    }

    section.classList.remove('pipeline--desktop')
    setAllStepsStatic()
    const stopWatching = watchVideoVisibility()

    return () => {
      stopWatching()
      videos.forEach((video) => video.pause())
    }
  })

  document.fonts?.ready?.then(() => ScrollTrigger.refresh())
}
