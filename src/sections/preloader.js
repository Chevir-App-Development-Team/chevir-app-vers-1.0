import gsap from 'gsap'
import { prefersReducedMotion } from '../lib/gsapSetup.js'

export const splashMarkup = `
<div id="splash" role="status" aria-live="polite">
  <span class="splash__counter" aria-hidden="true" data-splash-count>0</span>
  <span class="visually-hidden">Chevir yüklənir</span>
</div>
`

function revealNavbar() {
  document.querySelector('[data-navbar]')?.classList.add('is-revealed')
}

// Naviqasiya loqosunu ANİ (keçidsiz) görünən edir - navbar.css-də bu elementin
// öz "opacity 0.35s" keçidi var, amma klon-dan ötürməni KƏSİKSİZ etmək üçün elə
// həmin anda dəqiq üst-üstə düşməlidir, tədricən sönən/yanan iki fərqli müddət
// yox. transition:none + məcburi reflow + transition-u geri qaytarmaq həmin
// "ani dəyişmə" texnikasıdır (gələcək color keçidlərinə (is-on-dark) toxunmur).
function revealNavbarLogoInstantly(navbarLogoEl) {
  // getComputedStyle (yox offsetHeight) ilə məcburi stil-yenidənhesablama -
  // sınaqda offsetHeight/layout-reflow bu konkret ardıcıllıqda "transition:none"
  // halını həmişə etibarlı commit etmirdi (opacity yenə də CSS-dəki 0.35s
  // keçidlə tədricən dəyişirdi); getComputedStyle oxusu hər addımdan sonra
  // etibarlı işlədi.
  navbarLogoEl.style.transition = 'none'
  getComputedStyle(navbarLogoEl).opacity
  navbarLogoEl.style.opacity = '1'
  getComputedStyle(navbarLogoEl).opacity
  navbarLogoEl.style.transition = ''
}

function finishSplash(lenis) {
  const splashEl = document.getElementById('splash')
  document.documentElement.classList.remove('preloading')
  if (lenis) lenis.start()
  if (splashEl) splashEl.hidden = true
}

function runSplashAnimation(logo, lenis, onDone) {
  const splashEl = document.getElementById('splash')
  const countEl = splashEl.querySelector('[data-splash-count]')
  const desc = document.querySelector('.hero__desc')
  const cta = document.querySelector('.navbar__cta')
  const scrollBtn = document.querySelector('.hero__scroll')
  const navbarLogo = document.querySelector('.navbar__logo-svg')

  // Splash əsl loqoları HƏRƏKƏT ETDİRMİR - özünün ayrıca klonunu işlədir (bax:
  // əvvəlki şərh - əsl elementi fixed etmək transform-lu ancestor-un containing
  // block-una salırdı). Klon indi HERO-ya yox, NAVBAR-a enir: hero loqosu splash
  // bitən kimi 3D əllə əvəz olunur (hero.js, özündən asılı), ona görə klon hero-ya
  // ensəydi boş yerə enmiş olardı. Navbar loqosu isə səhifə boyu sabit qalır.
  const splashLogo = logo.cloneNode(true)
  splashLogo.removeAttribute('id')
  splashLogo.classList.add('splash__logo')
  splashEl.appendChild(splashLogo)

  const arcSigned = splashLogo.querySelector('#arc-signed')
  const arcSpoken = splashLogo.querySelector('#arc-spoken')
  const headSigned = splashLogo.querySelector('#head-signed')
  const headSpoken = splashLogo.querySelector('#head-spoken')
  const dot = splashLogo.querySelector('#dot')
  // Klonun id-ləri əsl loqonun id-ləri ilə toqquşmasın deyə silinir - istinadlar
  // artıq yuxarıdaki element referanslarında saxlanılıb.
  ;[arcSigned, arcSpoken, headSigned, headSpoken, dot].forEach((el) => el?.removeAttribute('id'))

  // Naviqasiya başlanğıcda görünməzdir (navbar.css: opacity:0, display:none YOX),
  // ona görə layout-da yerini saxlayır və getBoundingClientRect() bu mərhələdə
  // də düzgün nəticə verir. İntro ölçüsü (splashSize) hədəfdən asılı deyil -
  // ona görə burda hədəfi ölçməyə ehtiyac yoxdur, uçuş başlamazdan BİRBAŞA
  // ƏVVƏL (aşağıda, funksiya-əsaslı dəyərlərlə) təzədən ölçülür ki, aradakı
  // ~2 saniyədə mümkün layout sürüşməsi (məs. şrift yüklənməsi) köhnəlməsin.
  const vw = window.innerWidth
  const vh = window.innerHeight
  const splashSize = Math.min(vw, vh) * 0.28
  const splashX = vw / 2 - splashSize / 2
  const splashY = vh / 2 - splashSize / 2
  const arcLength = arcSigned.getTotalLength()

  gsap.set([arcSigned, arcSpoken], { strokeDasharray: arcLength, strokeDashoffset: arcLength })
  gsap.set([headSigned, headSpoken], { opacity: 0, scale: 0, transformOrigin: '50% 50%' })
  // Splash fonu açıqdır (paper) - nöqtə burda TÜND başlayır (currentColor
  // ağ olardı, işıqlı fonda itərdi). Navbar-a uçuşla eyni anda ağa keçir
  // (bax aşağıda), çünki hədəf (navbar, is-on-dark) ağ mətnlidir.
  gsap.set(dot, { opacity: 0, scale: 0, svgOrigin: '50 50', fill: '#1a1817' })
  gsap.set(splashLogo, {
    position: 'fixed',
    top: 0,
    left: 0,
    margin: 0,
    // Əsl piksel ölçüsü (scale yox) - SVG öz viewBox-unu qutuya uyğunlaşdırır,
    // ona görə uçuş zamanı width/height-i birbaşa animasiya etmək (aşağıda)
    // son kadrda scale-dən qaynaqlanan yuvarlaqlaşdırma xətasını aradan qaldırır.
    width: splashSize,
    height: splashSize,
    x: splashX,
    y: splashY,
    transformOrigin: 'top left',
  })

  const counter = { value: 0 }

  const FLIGHT_START = 2.0
  const FLIGHT_DURATION = 1.1
  const FLIGHT_END = FLIGHT_START + FLIGHT_DURATION

  // Ümumi müddət ~3.3s (dizayn qaydası: maksimum 3.5s). Qövslərin çəkilməsi əsas
  // andır və ən çox vaxtı alır (~1.4s) - hər şey ondan sonra tərpənir.
  gsap
    .timeline({
      onComplete: () => {
        finishSplash(lenis)
        onDone()
      },
    })
    // Əsas an: qövslər çəkilir. İkinci qövs birincidən azca gecikməklə başlayır,
    // eyni anda yox.
    .to(arcSigned, { strokeDashoffset: 0, duration: 1.25, ease: 'power2.inOut' }, 0)
    .to(arcSpoken, { strokeDashoffset: 0, duration: 1.25, ease: 'power2.inOut' }, 0.15)
    // Sayğad qövslərlə eyni müddətdə gedir - tez bitib gözləmir.
    .to(
      counter,
      {
        value: 100,
        duration: 1.4,
        ease: 'power1.out',
        onUpdate: () => {
          countEl.textContent = Math.round(counter.value)
        },
      },
      0
    )
    .to([headSigned, headSpoken], { opacity: 1, scale: 1, duration: 0.25, ease: 'back.out(2)', stagger: 0.06 }, 1.28)
    .to(dot, { opacity: 1, scale: 1, duration: 0.2, ease: 'back.out(2)' }, 1.5)
    // Qısa fasilə (0.3s, 1.7→2.0) - göz loqonu tam görsün, sonra uçuş başlayır.
    // clip-path uçuşla eyni anda başlayır, amma ondan bir az gec bitir ki,
    // ardıcıl (uçuş → tam açılış) görünsün.
    .to(splashEl, { clipPath: 'circle(0% at 50% 50%)', duration: FLIGHT_DURATION + 0.15, ease: 'power2.inOut' }, FLIGHT_START)
    // Funksiya-əsaslı dəyərlər: GSAP bunları YALNIZ bu tween başlayanda (3.1-də
    // yox, məhz FLIGHT_START-da) çağırır - ona görə hədəf ANCAQ indi ölçülür,
    // ~2 saniyə əvvəl yox. Həm mövqe (x/y), həm əsl ölçü (width/height) - scale
    // yox - hədəflə bit-bit üst-üstə düşür.
    .to(
      splashLogo,
      {
        x: () => navbarLogo.getBoundingClientRect().left,
        y: () => navbarLogo.getBoundingClientRect().top,
        width: () => navbarLogo.getBoundingClientRect().width,
        height: () => navbarLogo.getBoundingClientRect().height,
        duration: FLIGHT_DURATION,
        ease: 'power3.inOut',
      },
      FLIGHT_START
    )
    // Nöqtə tünddən ağa keçir (navbar.is-on-dark --ink: #f1ede9) - uçuşla
    // eyni anda bitir ki, hədəfə çatanda artıq rəng fərqi görünməsin.
    .to(dot, { fill: '#f1ede9', duration: FLIGHT_DURATION, ease: 'power3.inOut' }, FLIGHT_START)
    // Uçuş bitdiyi an: klon HƏDƏFDƏ SABİT dayanır (sönmür), navbar loqosu
    // EYNİ anda keçidsiz opacity:1 olur - iki eyni forma tam üst-üstə düşdüyü
    // üçün göz heç nə görmür. Bir kadr sonra klon DOM-dan silinir.
    .call(
      () => {
        revealNavbarLogoInstantly(navbarLogo)
        revealNavbar()
        requestAnimationFrame(() => splashLogo.remove())
      },
      [],
      FLIGHT_END
    )
    .to([desc, cta, scrollBtn], { autoAlpha: 1, y: 0, duration: 0.4, stagger: 0.05, ease: 'power2.out' }, 2.9)
}

export function initPreloader(lenis, logo, onDone) {
  // #splash birbaşa body-nin övladı olmalıdır ki, position:fixed riyaziyyatı
  // heç bir transform-lu valideynin altına düşməsin (bax: yuxarıdakı şərh).
  const splashEl = document.getElementById('splash')
  if (splashEl && splashEl.parentElement !== document.body) {
    document.body.appendChild(splashEl)
  }

  if (prefersReducedMotion) {
    revealNavbar()
    finishSplash(lenis)
  } else {
    lenis.stop()
    runSplashAnimation(logo, lenis, onDone)
  }
}
