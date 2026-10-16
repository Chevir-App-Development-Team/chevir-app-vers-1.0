import gsap from 'gsap'
import { prefersReducedMotion } from '../lib/gsapSetup.js'
import { BRAND_LOGO_MARKUP_WHITE, buildBrandLogoMarkupBlue, BRAND_LOGO_VIEWBOX } from '../lib/brandLogo.js'

export const splashMarkup = `
<div id="splash" role="status" aria-live="polite">
  <span class="visually-hidden">Chevir yüklənir</span>
</div>
`

function revealNavbar() {
  document.querySelector('[data-navbar]')?.classList.add('is-revealed')
}

// Naviqasiya loqosunu ANİ (keçidsiz) görünən edir - navbar.css-də bu elementin
// öz "opacity 0.4s" keçidi var, amma klon-dan ötürməni KƏSİKSİZ etmək üçün elə
// həmin anda dəqiq üst-üstə düşməlidir, tədricən sönən/yanan iki fərqli müddət
// yox. transition:none + məcburi reflow + transition-u geri qaytarmaq həmin
// "ani dəyişmə" texnikasıdır. İnline opacity YALNIZ müvəqqətidir: çağıran
// kod .is-revealed sinfini əlavə edəndən sonra clearInstantOverride() ilə
// silinməlidir - əks halda inline stil CSS-in .is-light keçid qaydasını
// (naviqasiyanın göy/ağ nüsxələri arasında) həmişəlik bloklayardı.
function revealNavbarLogoInstantly(navbarLogoEl) {
  // getComputedStyle (yox offsetHeight) ilə məcburi stil-yenidənhesablama -
  // sınaqda offsetHeight/layout-reflow bu konkret ardıcıllıqda "transition:none"
  // halını həmişə etibarlı commit etmirdi (opacity yenə də CSS-dəki keçidlə
  // tədricən dəyişirdi); getComputedStyle oxusu hər addımdan sonra etibarlı işlədi.
  navbarLogoEl.style.transition = 'none'
  getComputedStyle(navbarLogoEl).opacity
  navbarLogoEl.style.opacity = '1'
  getComputedStyle(navbarLogoEl).opacity
  navbarLogoEl.style.transition = ''
}

function clearInstantOverride(navbarLogoEl) {
  // Bu an CSS kaskadı artıq eyni dəyəri (1) verir (.is-revealed əlavə olunub,
  // .is-light hələ yoxdur) - inline override-u silmək görünüşü dəyişmir,
  // sadəcə gələcək .is-light keçidlərini CSS-ə qaytarır.
  navbarLogoEl.style.opacity = ''
}

function finishSplash(lenis) {
  const splashEl = document.getElementById('splash')
  document.documentElement.classList.remove('preloading')
  if (lenis) lenis.start()
  if (splashEl) splashEl.hidden = true
}

function runSplashAnimation(lenis, onDone) {
  const splashEl = document.getElementById('splash')
  const desc = document.querySelector('.hero__desc')
  const nsosyalBtn = document.querySelector('.hero__nsosyal-btn')
  const cta = document.querySelector('.navbar__actions')
  const scrollBtn = document.querySelector('.hero__scroll')
  // Naviqasiya səhifə açılan anda HƏMİŞƏ default (ağ, .is-light DEYİL) rejimdədir
  // (hero tünddür, scroll hələ başlamayıb) - klon də ona görə ağla ENMƏLİDİR.
  const navbarLogoDark = document.querySelector('.navbar__logo-svg--dark')

  // Splash əsl loqoları HƏRƏKƏT ETDİRMİR - özünün ayrıca klonunu işlədir.
  // Klon NAVBAR-a enir (hero-ya yox): hero loqosu splash bitən kimi 3D əllə
  // əvəz olunur, ona görə klon hero-ya ensəydi boş yerə enmiş olardı.
  //
  // Splash fonu açıqdır (Cloud) - ona görə klon göy (gradient) nüsxə ilə
  // BAŞLAYIR (ağ-üstündə-ağ olmasın), amma hədəf (naviqasiya) bu anda HƏMİŞƏ
  // ağ default rejimdədir - ona görə iki nüsxə (göy + ağ) üst-üstə qoyulur,
  // uçuşla EYNİ vaxt pəncərəsində bir-birinin üzərinə keçir (navbar.js-dəki
  // eyni texnika) ki, eniş anında rəng "sıçramasın".
  const splashLogo = document.createElement('div')
  splashLogo.classList.add('splash__logo')
  splashLogo.innerHTML = `
    <svg class="splash__logo-svg splash__logo-svg--light" viewBox="${BRAND_LOGO_VIEWBOX}">${buildBrandLogoMarkupBlue('splash')}</svg>
    <svg class="splash__logo-svg splash__logo-svg--dark" viewBox="${BRAND_LOGO_VIEWBOX}">${BRAND_LOGO_MARKUP_WHITE}</svg>
  `
  splashEl.appendChild(splashLogo)

  const svgLight = splashLogo.querySelector('.splash__logo-svg--light')
  const svgDark = splashLogo.querySelector('.splash__logo-svg--dark')
  const blobs = Array.from(svgLight.querySelectorAll('[data-logo-part^="blob-"]'))
  const dots = Array.from(svgLight.querySelectorAll('[data-logo-part^="dot-"]'))

  const vw = window.innerWidth
  const vh = window.innerHeight
  const splashSize = Math.min(vw, vh) * 0.28
  const splashX = vw / 2 - splashSize / 2
  const splashY = vh / 2 - splashSize / 2

  gsap.set(blobs, { opacity: 0, scale: 0.85, transformOrigin: '50% 50%' })
  gsap.set(dots, { opacity: 0, scale: 0, transformOrigin: '50% 50%' })
  gsap.set(svgLight, { opacity: 1 })
  gsap.set(svgDark, { opacity: 0 })
  gsap.set(splashLogo, {
    position: 'fixed',
    top: 0,
    left: 0,
    margin: 0,
    width: splashSize,
    height: splashSize,
    x: splashX,
    y: splashY,
    transformOrigin: 'top left',
  })

  const FLIGHT_START = 0.55
  const FLIGHT_DURATION = 0.3
  const FLIGHT_END = FLIGHT_START + FLIGHT_DURATION

  // Ümumi müddət < 1s (dizayn qaydası, köhnə 3.5s-lik versiyadan əvəz edib) -
  // qövslər/damlalar yumşaq görünür, sonra naviqasiyaya uçur.
  gsap
    .timeline({
      onComplete: () => {
        finishSplash(lenis)
        onDone()
      },
    })
    .to(blobs, { opacity: 1, scale: 1, duration: 0.35, ease: 'power2.out', stagger: 0.05 }, 0)
    .to(dots, { opacity: 1, scale: 1, duration: 0.25, ease: 'back.out(1.6)', stagger: 0.05 }, 0.2)
    // Funksiya-əsaslı dəyərlər: GSAP bunları YALNIZ bu tween başlayanda çağırır -
    // hədəf (navbar) ancaq indi ölçülür, əvvəlcədən yox.
    .to(
      splashLogo,
      {
        x: () => navbarLogoDark.getBoundingClientRect().left,
        y: () => navbarLogoDark.getBoundingClientRect().top,
        width: () => navbarLogoDark.getBoundingClientRect().width,
        height: () => navbarLogoDark.getBoundingClientRect().height,
        duration: FLIGHT_DURATION,
        ease: 'power3.inOut',
      },
      FLIGHT_START
    )
    // Göy→ağ keçid uçuşla EYNİ vaxt pəncərəsində - eniş anında hədəfin ağ
    // rənginə tam uyğun gəlsin.
    .to(svgLight, { opacity: 0, duration: FLIGHT_DURATION, ease: 'power2.inOut' }, FLIGHT_START)
    .to(svgDark, { opacity: 1, duration: FLIGHT_DURATION, ease: 'power2.inOut' }, FLIGHT_START)
    // clip-path uçuşla eyni anda başlayır, ondan bir az gec bitir ki, ardıcıl
    // (uçuş → tam açılış) görünsün.
    .to(splashEl, { clipPath: 'circle(0% at 50% 50%)', duration: FLIGHT_DURATION + 0.15, ease: 'power2.inOut' }, FLIGHT_START)
    // Uçuş bitdiyi an: klon HƏDƏFDƏ SABİT dayanır (sönmür), naviqasiyanın ağ
    // loqosu EYNİ anda keçidsiz opacity:1 olur - iki eyni forma tam üst-üstə
    // düşdüyü üçün göz heç nə görmür. Bir kadr sonra klon DOM-dan silinir.
    .call(
      () => {
        revealNavbarLogoInstantly(navbarLogoDark)
        revealNavbar()
        clearInstantOverride(navbarLogoDark)
        requestAnimationFrame(() => splashLogo.remove())
      },
      [],
      FLIGHT_END
    )
    .to([desc, nsosyalBtn, cta, scrollBtn].filter(Boolean), { autoAlpha: 1, y: 0, duration: 0.4, stagger: 0.05, ease: 'power2.out' }, FLIGHT_END)
}

export function initPreloader(lenis, onDone) {
  // #splash birbaşa body-nin övladı olmalıdır ki, position:fixed riyaziyyatı
  // heç bir transform-lu valideynin altına düşməsin.
  const splashEl = document.getElementById('splash')
  if (splashEl && splashEl.parentElement !== document.body) {
    document.body.appendChild(splashEl)
  }

  if (prefersReducedMotion) {
    revealNavbar()
    finishSplash(lenis)
  } else {
    lenis.stop()
    runSplashAnimation(lenis, onDone)
  }
}
