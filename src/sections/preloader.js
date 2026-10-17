import gsap from 'gsap'
import { prefersReducedMotion } from '../lib/gsapSetup.js'
import {
  BRAND_LOGO_VIEWBOX,
  BRAND_LOGO_BLOB_A_D,
  BRAND_LOGO_BLOB_B_D,
  BRAND_LOGO_DOT_LEFT,
  BRAND_LOGO_DOT_RIGHT,
} from '../lib/brandLogo.js'

export const splashMarkup = `
<div id="splash" role="status" aria-live="polite">
  <span class="visually-hidden">Chevir yüklənir</span>
</div>
`

// Bütün zaman nöqtələri BURDA, tək yerdə - "_START" dəyərləri əvvəlki
// mərhələdən TÖRƏNİR ki, bir müddəti dəyişdirəndə sonrakılar avtomatik
// sürüşsün. Ümumi müddət ≈2.7s.
const TIMING = {
  ARCS_START: 0.3, // 0.00-0.30 boş splash fon
  ARCS_DURATION: 0.7, // 0.30-1.00 iki qövs eyni anda, yalnız opacity
  DOTS_DURATION: 0.45, // 1.00-1.45 iki göz birlikdə, yalnız opacity
  BLINK_DOWN_DURATION: 0.18, // 1.45-1.95 sağ göz vurur
  BLINK_UP_DURATION: 0.25,
  BLINK_HOLD: 0.07, // vurmadan sonra qısa dayanma (down+up+hold=0.50)
  FLIGHT_DURATION: 0.75, // 1.95-2.70 uçuş + clip-path bağlanma
  HERO_LEAD: 0.15, // hero məzmunu uçuş bitməzdən bu qədər əvvəl başlayır
  HERO_DURATION: 0.18,
  HERO_STAGGER: 0.02,
}
TIMING.DOTS_START = TIMING.ARCS_START + TIMING.ARCS_DURATION // 1.00
TIMING.BLINK_START = TIMING.DOTS_START + TIMING.DOTS_DURATION // 1.45
TIMING.BLINK_DOWN_END = TIMING.BLINK_START + TIMING.BLINK_DOWN_DURATION // 1.63
TIMING.BLINK_UP_END = TIMING.BLINK_DOWN_END + TIMING.BLINK_UP_DURATION // 1.88
TIMING.FLIGHT_START = TIMING.BLINK_UP_END + TIMING.BLINK_HOLD // 1.95
TIMING.FLIGHT_END = TIMING.FLIGHT_START + TIMING.FLIGHT_DURATION // 2.70
TIMING.HERO_START = TIMING.FLIGHT_END - TIMING.HERO_LEAD // 2.55

// Göz vurma - sağ gözün ry-si bu radiusa qədər sıxılır (cx/cy toxunulmur,
// ona görə mərkəz heç vaxt sürüşmür).
const EYE_RADIUS = BRAND_LOGO_DOT_LEFT.r // 57.5 - sol/sağ eyni
const EYE_BLINK_RY = 8

function revealNavbar() {
  document.querySelector('[data-navbar]')?.classList.add('is-revealed')
}

// revealNavbar()-ın analoqu - hero məzmunu (tagline/nsosyal-btn) üçün.
// HƏR İKİ çağırış yerində (splash var/yox) revealNavbar() ilə BİRLİKDƏ
// çağırılır ki, animasiya skriptindən asılı olmayan, həmişə görünən
// struktur təmin olunsun (bax hero.css `.hero.is-revealed`).
function revealHero() {
  document.querySelector('.hero')?.classList.add('is-revealed')
}

// Naviqasiya loqosunu ANİ (keçidsiz) görünən edir - navbar.css-də bu elementin
// öz "opacity 0.4s" keçidi var, amma uçan elementdən ötürməni KƏSİKSİZ etmək
// üçün elə həmin anda dəqiq üst-üstə düşməlidir, tədricən sönən/yanan iki
// fərqli müddət yox. transition:none + məcburi reflow + transition-u geri
// qaytarmaq həmin "ani dəyişmə" texnikasıdır. İnline opacity YALNIZ
// müvəqqətidir: çağıran kod .is-revealed sinfini əlavə edəndən sonra
// clearInstantOverride() ilə silinməlidir - əks halda inline stil CSS-in
// .is-light keçid qaydasını (naviqasiyanın göy/ağ nüsxələri arasında)
// həmişəlik bloklayardı.
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

// Splash-a məxsus TƏK markup - brandLogo.js-dəki paylaşılan
// buildBrandLogoMarkupBlue()-dan FƏRQLİ olaraq gözlər <circle> yox,
// <ellipse> (rx=ry=EYE_RADIUS) - göz vurma zamanı yalnız ry animasiya
// olunsun deyə (cx/cy sabit qalır, mərkəz heç vaxt sürüşmür). Path/radius
// dəyərləri brandLogo.js-dəki paylaşılan sabitlərdən gəlir (tək mənbə),
// brandLogo.js-in ÖZÜ dəyişmir (navbar/hero fallback ona görə toxunulmaz).
// Gradient stop-ları "stop-color" GSAP attr animasiyası ilə (aşağıda,
// FLIGHT zamanı) göydən ağa keçir - bu, İKİNCİ bir ağ nüsxə/klon tələb
// etmədən rəng keçidini TƏK elementin daxilində edir.
function buildSplashLogoMarkup() {
  return `
  <defs>
    <linearGradient id="splash-grad-dot-left" x1="365.9" y1="594.3" x2="480.9" y2="594.3" gradientUnits="userSpaceOnUse">
      <stop class="splash-grad-stop" offset="0" stop-color="#0071ef"/>
      <stop class="splash-grad-stop" offset="1" stop-color="#0021bf"/>
    </linearGradient>
    <linearGradient id="splash-grad-dot-right" x1="599.1" y1="594.3" x2="714.1" y2="594.3" gradientUnits="userSpaceOnUse">
      <stop class="splash-grad-stop" offset="0" stop-color="#0071ef"/>
      <stop class="splash-grad-stop" offset="1" stop-color="#0021bf"/>
    </linearGradient>
    <linearGradient id="splash-grad-a" x1="19.7" y1="540.2" x2="528" y2="540.2" gradientUnits="userSpaceOnUse">
      <stop class="splash-grad-stop" offset="0" stop-color="#0071ef"/>
      <stop class="splash-grad-stop" offset="1" stop-color="#0021bf"/>
    </linearGradient>
    <linearGradient id="splash-grad-b" x1="552" y1="540" x2="1060.3" y2="540" gradientUnits="userSpaceOnUse">
      <stop class="splash-grad-stop" offset="0" stop-color="#0071ef"/>
      <stop class="splash-grad-stop" offset="1" stop-color="#0021bf"/>
    </linearGradient>
  </defs>
  <path data-logo-part="blob-a" fill="url(#splash-grad-a)" d="${BRAND_LOGO_BLOB_A_D}"/>
  <path data-logo-part="blob-b" fill="url(#splash-grad-b)" d="${BRAND_LOGO_BLOB_B_D}"/>
  <ellipse data-logo-part="dot-left" fill="url(#splash-grad-dot-left)" cx="${BRAND_LOGO_DOT_LEFT.cx}" cy="${BRAND_LOGO_DOT_LEFT.cy}" rx="${EYE_RADIUS}" ry="${EYE_RADIUS}"/>
  <ellipse data-logo-part="dot-right" fill="url(#splash-grad-dot-right)" cx="${BRAND_LOGO_DOT_RIGHT.cx}" cy="${BRAND_LOGO_DOT_RIGHT.cy}" rx="${EYE_RADIUS}" ry="${EYE_RADIUS}"/>
  `
}

function runSplashAnimation(lenis, onDone) {
  const splashEl = document.getElementById('splash')
  const desc = document.querySelector('.hero__desc')
  const nsosyalBtn = document.querySelector('.hero__nsosyal-btn')
  // navbar__actions əvəzinə iki ayrı element (bax navbar.js/hero.js) - grid
  // 1fr auto 1fr düzümündə linklər və CTA artıq ortaq konteynerdə deyil.
  const navLinks = document.querySelector('.navbar__links')
  const navCta = document.querySelector('.navbar__cta')
  const scrollBtn = document.querySelector('.hero__scroll')
  // Hədəf nüsxə (ağ/tünd-fon üçün, yoxsa göy/açıq-fon üçün) SABİT seçilmir -
  // hero indi açıq olsa da, səhifə scroll bərpası ilə (brauzerin öz
  // scrollRestoration-ı) başqa (tünd) bölmənin üzərində açıla bilər, onda
  // navbar `.is-light` DEYİL və ağ nüsxə doğru hədəfdir. Qərar FLIGHT
  // başlamazdan dərhal əvvəl, addFlightLandingTweens()-də verilir (bax aşağı) -
  // bu an initNavbar()-un sinxron ilk `update()` çağırışından sonradır, ona
  // görə `.is-light` dəyəri artıq doğrudur (main.js: initPreloader → initNavbar,
  // eyni tick; FLIGHT isə 2×rAF+TIMING.FLIGHT_START qədər sonra başlayır).
  const navbarEl = document.querySelector('[data-navbar]')
  const navbarLogoDark = document.querySelector('.navbar__logo-svg--dark')
  const navbarLogoLight = document.querySelector('.navbar__logo-svg--light')
  // Hər iki nüsxə eyni konteynerin üzərində üst-üstədir (navbar.css:
  // `.navbar__logo-svg{position:absolute;inset:0}`) - ölçü HANSININ
  // sorğulandığından asılı deyil, tək layout oxuması kifayətdir (FLIGHT
  // tween daxilində hər tick getBoundingClientRect() çağırmaq əvəzinə).
  const navbarRect = navbarLogoDark.getBoundingClientRect()

  // TƏK loqo elementi - klon/ikinci nüsxə YOXDUR. Bütün ardıcıllıq (yaranma,
  // gözlər, göz vurma, uçuş) elə bu BİR elementin üzərində gedir. #splash-ın
  // (clip-path tətbiq olunan) DAXİLİNƏ YOX, birbaşa body-ə əlavə olunur
  // (z-index CSS-də #splash-dan yuxarı) - əks halda dairə bağlananda loqonun
  // özü də kəsilərdi.
  const logoEl = document.createElement('div')
  logoEl.classList.add('splash__logo')
  logoEl.innerHTML = `<svg class="splash__logo-svg" viewBox="${BRAND_LOGO_VIEWBOX}">${buildSplashLogoMarkup()}</svg>`
  document.body.appendChild(logoEl)

  const svg = logoEl.querySelector('.splash__logo-svg')
  const blobs = Array.from(svg.querySelectorAll('[data-logo-part^="blob-"]'))
  const dots = Array.from(svg.querySelectorAll('[data-logo-part^="dot-"]'))
  const dotRight = svg.querySelector('[data-logo-part="dot-right"]')
  const gradientStops = Array.from(svg.querySelectorAll('.splash-grad-stop'))

  const vw = window.innerWidth
  const vh = window.innerHeight
  const splashSize = Math.min(vw, vh) * 0.28
  const splashX = vw / 2 - splashSize / 2
  const splashY = vh / 2 - splashSize / 2

  // Qövslərin/gözlərin başlanğıc gizli vəziyyəti (opacity:0) CSS-dədir (bax
  // splash.css), burda TƏKRAR təyin olunmur. VACİB: bu elementlərin HEÇ
  // BİRİNƏ transform/scale verilmir - hər path/ellipse SVG-nin öz, son,
  // düzgün koordinatında sabit qalır. "Yaranma" scale-i YALNIZ bütöv loqo
  // elementinə (logoEl) verilir.
  gsap.set(logoEl, {
    position: 'fixed',
    top: 0,
    left: 0,
    margin: 0,
    width: splashSize,
    height: splashSize,
    x: splashX,
    y: splashY,
    scale: 0.92,
    transformOrigin: '50% 50%',
  })

  // Dairənin bağlanma mərkəzi uçan loqonun ÖZ mərkəzini izləyir (sabit
  // 50%/50% yox) - splash loqosu da təxminən ekran mərkəzindən başladığı
  // üçün bu iki nöqtə FLIGHT_START-da artıq yaxındır, FLIGHT boyu hamar
  // şəkildə naviqasiya loqosunun mərkəzinə keçir.
  const flightStartCenter = { x: splashX + splashSize / 2, y: splashY + splashSize / 2 }
  const flightEndCenter = { x: navbarRect.left + navbarRect.width / 2, y: navbarRect.top + navbarRect.height / 2 }
  // Ekranın istənilən nöqtəsindən tam örtmə üçün kifayət qədər böyük başlanğıc
  // radiusu - diaqonal məsafə hər zaman kifayətdir.
  const maxRadius = Math.hypot(vw, vh)

  // `paused:true` - timeline YARADILIR, amma dərhal İŞLƏMİR (bax aşağıda
  // 2×rAF). GSAP-ın saatı yalnız .play() çağırılanda "sayma"ğa başlayır,
  // ona görə main.js-dəki ağır sinxron iş (initNavbar/Pipeline/Loop/Content,
  // initPreloader-dan DƏRHAL sonra, eyni JS tick-ində çağırılır) bu paused
  // vəziyyətdə problemsiz bitir - animasiyanın "vaxtı" hələ başlamayıb.
  const tl = gsap.timeline({
    paused: true,
    onComplete: () => {
      finishSplash(lenis)
      onDone()
    },
  })

  // a) İki qövs EYNİ ANDA (stagger yoxdur), YALNIZ opacity. Paralel olaraq
  // bütöv loqo elementi 0.92→1 scale ilə "yaranma" hissini verir - fərdi
  // hissələrə YOX, yalnız bu bir elementə. Tween özü scale-i FLIGHT-dan
  // (d, aşağıda) ƏVVƏL tam 1-ə çatdırır - əlavə `clearProps` ÇAĞIRILMIR:
  // GSAP transform-un bütün komponentlərini (x/y/scale) TƏK CSS `transform`
  // dəyərində birləşdirir, `clearProps:'scale'` bunu AYIRA bilmədiyi üçün
  // bütöv transform-u (x/y DAXİL) silir - bu, "loqo guya (0,0)-a sıçrayır"
  // bugının səbəbi idi (ölçülüb). Tween scale=1-də bitdiyi üçün əlavə
  // təmizləməyə ehtiyac yoxdur.
  tl.to(blobs, { opacity: 1, duration: TIMING.ARCS_DURATION, ease: 'power2.out' }, TIMING.ARCS_START)
  tl.to(logoEl, { scale: 1, duration: TIMING.ARCS_DURATION, ease: 'power2.out' }, TIMING.ARCS_START)

  // b) İki göz EYNİ ANDA (stagger yoxdur), YALNIZ opacity - qövslər artıq
  // yerindədir, gözlər ortada görünür (uçuş hələ başlamayıb).
  tl.to(dots, { opacity: 1, duration: TIMING.DOTS_DURATION, ease: 'power2.out' }, TIMING.DOTS_START)

  // c) Göz vurma - YALNIZ sağ gözün "ry" atributu (GSAP attr, transform
  // DEYİL) kiçilib-böyüyür. cx/cy toxunulmur, ona görə mərkəz HEÇ VAXT
  // sürüşmür (əvvəlki "hoppanma" bugının səbəbi transform-origin idi, bu
  // texnika ilə mümkün deyil). Sol göz tam sabit qalır.
  tl.to(dotRight, { attr: { ry: EYE_BLINK_RY }, duration: TIMING.BLINK_DOWN_DURATION, ease: 'power2.in' }, TIMING.BLINK_START)
  tl.to(dotRight, { attr: { ry: EYE_RADIUS }, duration: TIMING.BLINK_UP_DURATION, ease: 'power2.out' }, TIMING.BLINK_DOWN_END)

  // d) FLIP uçuşu - hədəf (navbarRect) artıq yuxarıda BİR DƏFƏ ölçülüb,
  // burda sabit ədədlərlə işlədilir (funksiya-əsaslı dəyər yoxdur).
  tl.to(
    logoEl,
    {
      x: navbarRect.left,
      y: navbarRect.top,
      width: navbarRect.width,
      height: navbarRect.height,
      duration: TIMING.FLIGHT_DURATION,
      ease: 'power3.inOut',
      onUpdate: function () {
        // `this.ratio` GSAP-ın artıq ease tətbiq etdiyi proqresdir (0-1) -
        // dairənin mərkəzi/radiusu EYNİ bu dəyərlə hesablanır ki, loqo ilə
        // dairə hər kadrda dəqiq sinxron qalsın, FLIGHT_END-də ikisi də
        // tam eyni anda, eyni nöqtədə bitsin (heç bir splash qalığı qalmır).
        const r = this.ratio
        const cx = flightStartCenter.x + (flightEndCenter.x - flightStartCenter.x) * r
        const cy = flightStartCenter.y + (flightEndCenter.y - flightStartCenter.y) * r
        const radius = maxRadius * (1 - r)
        splashEl.style.clipPath = `circle(${radius}px at ${cx}px ${cy}px)`
      },
    },
    TIMING.FLIGHT_START
  )

  // Rəng-keçidi VƏ eniş hədəfi navbarın O ANKI (`.is-light`) vəziyyətinə
  // görə seçilir - BUILD zamanı yox (main.js sırası: initPreloader
  // initNavbar-dan ƏVVƏL çağırılır, o anda `.is-light` hələ doğru dəyəri
  // daşımaya bilər), tl.play()-dən bilavasitə əvvəl (aşağıdakı 2×rAF-ın
  // içində) - bu anda initNavbar()-un sinxron ilk `update()` çağırışı
  // artıq bitib, `.is-light` etibarlıdır (həm scroll=0/hero, həm scroll
  // bərpası ilə başqa/tünd bölmənin üzərində açılma halları üçün).
  function addFlightLandingTweens() {
    const isLight = navbarEl?.classList.contains('is-light') ?? false
    const target = isLight ? navbarLogoLight : navbarLogoDark

    // Göy→ağ keçid YALNIZ tünd-fon hədəfinə enəndə lazımdır (splash loqosu
    // onsuz da başdan göy gradient-dədir, açıq-fon/göy hədəflə artıq
    // piksel-dəqiq eynidir - bax brandLogo.js, eyni stop-color-lar). Uçuşla
    // EYNİ vaxt pəncərəsində, İKİNCİ element/klon yox, elə BU elementin öz
    // gradient-lərinin stop-color-u dəyişir (GSAP attr).
    if (!isLight) {
      tl.to(gradientStops, { attr: { 'stop-color': '#ffffff' }, duration: TIMING.FLIGHT_DURATION, ease: 'power2.inOut' }, TIMING.FLIGHT_START)
    }

    // Uçuş bitdiyi an: klon HƏDƏFDƏ SABİT dayanır (sönmür), naviqasiyanın
    // müvafiq nüsxəsi EYNİ anda keçidsiz opacity:1 olur - iki eyni forma
    // tam üst-üstə düşdüyü üçün göz heç nə görmür. Bir kadr sonra element
    // DOM-dan silinir.
    tl.call(
      () => {
        revealNavbarLogoInstantly(target)
        revealNavbar()
        revealHero()
        clearInstantOverride(target)
        requestAnimationFrame(() => logoEl.remove())
      },
      [],
      TIMING.FLIGHT_END
    )
  }

  // e) Hero məzmunu - uçuş bitməzdən HERO_LEAD qədər əvvəl başlayır (overlap).
  tl.to(
    [desc, nsosyalBtn, navLinks, navCta, scrollBtn].filter(Boolean),
    { autoAlpha: 1, y: 0, duration: TIMING.HERO_DURATION, stagger: TIMING.HERO_STAGGER, ease: 'power2.out' },
    TIMING.HERO_START
  )

  // İLK BOYANMADAN sonra başlat - birinci rAF növbəti boyanmadan ƏVVƏL,
  // ikinci rAF onun ARDINCA işə düşür (brauzerin "boya, sonra işlə" ardıcıllığı
  // ilə uyğunlaşan sübut olunmuş naviqasiya). Bu anda main.js-dəki
  // initNavbar/Pipeline/Loop/Content (initPreloader-dan dərhal sonra, eyni
  // sinxron tick-də çağırılır) artıq bitmiş olur - GSAP saatı yalnız indi
  // "sayma"ğa başlayır, ona görə həmin ağır iş animasiyanın ilk kadrlarını
  // "yeyə" bilmir.
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      addFlightLandingTweens()
      tl.play()
    })
  })

  return tl
}

export function initPreloader(lenis, onDone) {
  // #splash birbaşa body-nin övladı olmalıdır ki, position:fixed riyaziyyatı
  // heç bir transform-lu valideynin altına düşməsin.
  const splashEl = document.getElementById('splash')
  if (splashEl && splashEl.parentElement !== document.body) {
    document.body.appendChild(splashEl)
  }

  // index.html-dəki head script-i naviqasiya növünə (reload/xarici keçid
  // göstərir, sayt-daxili same-origin keçid/back_forward atlanır) görə bu
  // bayrağı paint-dən ƏVVƏL təyin edir - burda yalnız oxunur.
  const shouldShow = window.__SPLASH_SHOW__ !== false

  if (prefersReducedMotion || !shouldShow) {
    revealNavbar()
    revealHero()
    finishSplash(lenis)
  } else {
    lenis.stop()
    runSplashAnimation(lenis, onDone)
  }
}
