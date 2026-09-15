import gsap from 'gsap'
import { prefersReducedMotion } from '../lib/gsapSetup.js'
import { HERO_TAGLINE } from '../lib/i18n.js'

// public/wordmark.svg-nin dəyişməz kopyası (currentColor - hero-da ağ olur).
const WORDMARK_MARKUP = `
  <path d="M72.418 13.7919C72.418 8.92787 75.49 6.46387 80.354 6.46387V9.98387C77.794 9.98387 76.258 11.2319 76.258 13.7919V22.7519H72.418V13.7919Z" fill="currentColor"/>
  <path d="M66.6309 2.112C66.6309 0.959999 67.5589 0 68.7429 0C69.9269 0 70.8549 0.959999 70.8549 2.112C70.8549 3.328 69.9269 4.256 68.7429 4.256C67.5589 4.256 66.6309 3.328 66.6309 2.112ZM66.8229 22.752V6.752H70.6629V22.752H66.8229Z" fill="currentColor"/>
  <path d="M55.8488 22.752L49.8008 6.75195H53.9288L57.8328 17.6L61.7368 6.75195H65.8648L59.8168 22.752H55.8488Z" fill="currentColor"/>
  <path d="M38.0373 12.5118H46.0692C45.3972 10.9438 43.9573 10.0478 42.0373 10.0478C40.1493 10.0478 38.7093 10.9438 38.0373 12.5118ZM42.0373 19.5198C43.6372 19.5198 44.8853 18.9118 45.6533 17.7918H49.8452C48.7892 21.0238 45.9412 23.0398 42.0692 23.0398C37.1092 23.0398 33.7812 19.7118 33.7812 14.7518C33.7812 9.82383 37.1092 6.52783 42.0692 6.52783C47.0292 6.52783 50.3252 9.82383 50.3252 14.7518C50.3252 15.1998 50.2932 15.6158 50.2612 16.0318H37.7492C38.1652 18.2398 39.7653 19.5198 42.0373 19.5198Z" fill="currentColor"/>
  <path d="M21.4455 0.352051V7.87205C22.3735 7.00805 23.6855 6.46405 25.4775 6.46405C30.2455 6.46405 32.3255 10.4321 32.3255 13.2161V22.7521H28.4855V13.3121C28.4855 11.0401 26.8215 10.0161 24.9655 10.0161C23.1095 10.0161 21.4455 11.0401 21.4455 13.3121V22.7521H17.6055V0.352051H21.4455Z" fill="currentColor"/>
  <path d="M3.84 14.7838C3.84 17.7598 5.6 19.5198 8.256 19.5198C10.208 19.5198 11.648 18.5918 12.32 16.9598H16.32C15.456 20.6718 12.48 23.0398 8.288 23.0398C3.328 23.0398 0 19.7118 0 14.7518C0 9.82383 3.328 6.52783 8.288 6.52783C12.448 6.52783 15.456 8.86383 16.288 12.5118H12.288C11.616 10.9438 10.176 10.0478 8.256 10.0478C5.6 10.0478 3.84 11.8078 3.84 14.7838Z" fill="currentColor"/>
`

// public/logo.svg-nin dəyişməz kopyası - 3D əl yüklənməyəndə (reduced-motion/
// WebGL yoxdursa) statik fallback kimi qalır.
// fill="none" hər qövsdə birbaşa yazılıb: bu svg üçün ayrıca <svg> elementinin
// özündə fill="none" yoxdur, ona görə miras gözləmək əvəzinə açıq təyin olunur.
const BRAND_LOGO_MARKUP = `
  <path id="arc-signed" fill="none" d="M50.5 88.5C45.3784 88.5 40.307 87.4912 35.5753 85.5313C30.8436 83.5714 26.5443 80.6986 22.9228 77.0772C19.3013 73.4557 16.4286 69.1563 14.4687 64.4247C12.5088 59.693 11.5 54.6215 11.5 49.5" stroke="#D6006C" stroke-width="7"/>
  <path id="arc-spoken" fill="none" d="M49.5 10.5C54.6216 10.5 59.693 11.5088 64.4247 13.4687C69.1564 15.4286 73.4557 18.3014 77.0772 21.9228C80.6986 25.5443 83.5714 29.8436 85.5313 34.5753C87.4912 39.307 88.5 44.3784 88.5 49.5" stroke="#0088B0" stroke-width="7"/>
  <path id="head-signed" d="M11.5 34L21.4593 49.75H1.54071L11.5 34Z" fill="#D6006C"/>
  <path id="head-spoken" d="M88.5 65L78.5407 49.25L98.4593 49.25L88.5 65Z" fill="#0088B0"/>
  <circle id="dot" cx="50" cy="50" r="4" fill="currentColor"/>
`

// Naviqasiya paneli üçün loqonun sadə, animasiyasız kopyası - id toqquşmasının
// qarşısını almaq üçün id-siz.
const NAVBAR_LOGO_MARKUP = BRAND_LOGO_MARKUP.replace(/\sid="[^"]*"/g, '')

export const navbarMarkup = `
<header class="navbar" data-navbar>
  <a class="navbar__brand" href="#top">
    <svg class="navbar__logo-svg" viewBox="0 0 100 100" aria-hidden="true">${NAVBAR_LOGO_MARKUP}</svg>
    <span class="navbar__name">Chevir</span>
  </a>
  <a class="navbar__cta" href="#pilot">Pilot tərəfdaş olun</a>
</header>
`

export const heroMarkup = `
<section class="hero" id="top" aria-label="Chevir">
  <div class="hero__bg" aria-hidden="true"></div>
  <canvas class="hero__scene" data-hero-canvas aria-hidden="true"></canvas>
  <svg class="hero__wordmark" viewBox="0 0 81 24" aria-hidden="true">${WORDMARK_MARKUP}</svg>
  <div class="hero__content">
    <p class="hero__tagline hero__desc">${HERO_TAGLINE}</p>

    <div class="hero__logo-slot" aria-hidden="true">
      <svg
        id="brand-logo"
        class="brand-logo"
        viewBox="0 0 100 100"
      >${BRAND_LOGO_MARKUP}</svg>
    </div>
  </div>
</section>
`

function isWebGLAvailable() {
  try {
    const canvas = document.createElement('canvas')
    return !!(
      window.WebGLRenderingContext &&
      (canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
    )
  } catch {
    return false
  }
}

// Statik #brand-logo yalnız FALLBACK kimi görünür (WebGL yoxdursa, reduced-motion,
// ya da 3D yüklənmə uğursuz olsa) - default gizlidir (hero.css), ona görə uğurlu
// 3D yüklənmədə heç vaxt görünüb-yoxa çıxmır (əvvəlki "yanıb-sönmə" bugı).
function setFallbackVisible(visible) {
  document.querySelector('.hero__logo-slot')?.classList.toggle('is-fallback', visible)
}

// Fon (noise dalğa) + 3D əl nöqtə buludu eyni canvas-da (hand3d.js, tək
// renderer/scene) - yalnız hərəkət azaldılmayıb və WebGL dəstəklənirsə
// yüklənir (dinamik import, three.js ilk yükləməni ağırlaşdırmır).
// Əks halda statik logo.svg + CSS fon görünməkdə davam edir.
function initHandVisual() {
  if (prefersReducedMotion) {
    setFallbackVisible(true)
    return
  }

  const canvas = document.querySelector('[data-hero-canvas]')
  const logoSlot = document.querySelector('.hero__logo-slot')
  const heroEl = document.querySelector('.hero')
  if (!canvas || !logoSlot || !heroEl || !isWebGLAvailable()) {
    setFallbackVisible(true)
    return
  }

  function activate() {
    import('../lib/hand3d.js')
      .then(({ initHand3D }) => {
        initHand3D(canvas)
        canvas.classList.add('is-active')
      })
      .catch(() => {
        // 3D yüklənə bilmədi - statik logo.svg fallback kimi göstərilir.
        setFallbackVisible(true)
      })
  }

  // Splash (preloader) bitənə qədər gözləyir ki, loqonun splash-dan hero-ya
  // uçuş animasiyası (preloader.js) pozulmasın - sonra yerini 3D-yə verir.
  if (document.documentElement.classList.contains('preloading')) {
    const observer = new MutationObserver(() => {
      if (!document.documentElement.classList.contains('preloading')) {
        activate()
        observer.disconnect()
      }
    })
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
  } else {
    activate()
  }
}

export function setupHero() {
  initHandVisual()

  const logo = document.getElementById('brand-logo')
  const tagline = document.querySelector('.hero__tagline')
  const cta = document.querySelector('.navbar__cta')

  // tagline/cta burda yalnız gizlədilir (instant set) - görünən reveal
  // tween-i preloader.js-in öz splash timeline-ındadır (`.hero__desc` seçicisi
  // ilə tapır, t=2.9-da açır). Wordmark və əl splash örtüyünün altında adi
  // görünür qalır, əvvəlki başlıq/demo kimi.
  if (prefersReducedMotion) {
    gsap.set([tagline, cta], { autoAlpha: 1 })
    return { logo }
  }

  gsap.set([tagline, cta], { autoAlpha: 0, y: 16 })

  return { logo }
}

export function initNavbar() {
  const navbarEl = document.querySelector('[data-navbar]')
  if (!navbarEl) return

  // Bütün bölmələr tünddür, ona görə naviqasiya həmişə
  // tünd rejimdədir - açıq/tünd keçid məntiqi lazım deyil. Yalnız scroll
  // edildikdə tünd yarımşəffaf blur qatı görünür.
  const toggleScrolledBg = () => navbarEl.classList.toggle('is-scrolled', window.scrollY > 40)
  window.addEventListener('scroll', toggleScrolledBg, { passive: true })
  toggleScrolledBg()
}
