import { BRAND_LOGO_MARKUP_WHITE, buildBrandLogoMarkupBlue, BRAND_LOGO_VIEWBOX } from '../lib/brandLogo.js'
import { t, currentLang } from '../lib/i18n.js'

// Naviqasiya paneli üçün loqonun iki nüsxəsi (ağ + göy) üst-üstə yerləşdirilir
// və navbar.css-dəki .is-light sinfinə görə bir-birinin üzərinə sönür/yanır -
// göy nüsxənin gradient id-ləri "navbar" açarı ilə unikaldır (bax brandLogo.js),
// ona görə preloader klonu ilə (başqa açar) heç vaxt toqquşmur.
const NAVBAR_LOGO_MARKUP_BLUE = buildBrandLogoMarkupBlue('navbar')

const BASE = import.meta.env.BASE_URL

const LINKS = {
  home: { href: '', label: t('common.nav.home') },
  tercume: { href: 'tercume/', label: t('common.nav.tercume') },
  nsosyal: { href: 'nsosyal/', label: t('common.nav.nsosyal') },
  investors: { href: 'https://bakuid.seeder.so/r/bakuid-2026/s/chevir', label: t('common.nav.investors'), external: true },
}

// Bütün səhifələrdə eyni mobil/planşet hamburger davranışı (bax navbar.css
// ≤1023px bloku - selektorlar `[data-nav-menu]`-a bağlıdır, səhifə adına
// yox, ona görə bura səhifə əlavə etmək CSS-ə TOXUNMADAN kifayətdir).
const MENU_ENABLED_PAGES = ['home', 'tercume', 'nsosyal']

// Səhifənin EN qarşılığı. Qarşılığı olmayan səhifə (hələlik nsosyal)
// EN düyməsində ana səhifənin EN nüsxəsinə (fallback) aparır.
const PAGE_EN_PATH = { home: 'en/', tercume: 'en/tercume/' }

// /tercume-ə xas tema (dark/light) keçid düyməsi - yalnız bu səhifənin
// navbar-ına əlavə olunur (bax navbarMarkup-dakı `page === 'tercume'` şərti),
// digər səhifələrin markup-u dəyişmir. localStorage/klik məntiqi
// src/tercume/layout.js-dədir (id ilə seçir, DOM-dakı yerindən asılı deyil).
const TERCUME_THEME_TOGGLE_MARKUP = `
    <button type="button" id="theme-toggle" class="theme-toggle" aria-pressed="false" aria-label="${t('common.nav.themeToggle.switchToLight')}">
      <svg class="theme-toggle__sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="12" cy="12" r="5" />
        <line x1="12" y1="1" x2="12" y2="3" />
        <line x1="12" y1="21" x2="12" y2="23" />
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
        <line x1="1" y1="12" x2="3" y2="12" />
        <line x1="21" y1="12" x2="23" y2="12" />
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
      </svg>
      <svg class="theme-toggle__moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
      </svg>
    </button>`

// "AZ | EN" keçidi - iki yerdə işlədilir (masaüstündə CTA-nın yanında,
// hamburger menyusunda ayrıca sətirdə) - bax navbarMarkup aşağıda.
// modifierClass ilə iki nüsxə eyni DOM/CSS əsasını paylaşır, yalnız
// görünmə şərti (media query) fərqlidir.
function langSwitchMarkup(page, modifierClass = '') {
  const azHref = `${BASE}${LINKS[page]?.href ?? ''}`
  const enHref = `${BASE}${PAGE_EN_PATH[page] ?? 'en/'}`
  const isEn = currentLang === 'en'
  const azLabel = t('common.nav.lang.az')
  const enLabel = t('common.nav.lang.en')
  return `
    <span class="navbar__lang${modifierClass}">
      <a class="navbar__lang-link${!isEn ? ' is-active' : ''}" href="${azHref}" lang="az" hreflang="az"${!isEn ? ' aria-current="page"' : ''}>${azLabel}</a>
      <a class="navbar__lang-link${isEn ? ' is-active' : ''}" href="${enHref}" lang="en" hreflang="en"${isEn ? ' aria-current="page"' : ''}>${enLabel}</a>
    </span>`
}

/**
 * Ana səhifə, /tercume/ və /nsosyal/ üçün eyni naviqasiya.
 * Ana səhifədə loqo və CTA səhifə daxili anchor-lardır; digər səhifələrdə naviqasiya
 * splash gözləmədən görünür və keçidlər ana səhifəyə qayıdır. Dar ekranda yalnız
 * birinci keçid qalır.
 * @param {{page?: 'home'|'tercume'|'nsosyal', light?: boolean}} opts
 */
export function navbarMarkup({ page = 'home', light = false } = {}) {
  const home = page === 'home'
  const hasMenu = MENU_ENABLED_PAGES.includes(page)
  // EN-də bu səhifənin özünün yox, QARDAŞ səhifələrin keçidləri EN qarşılığına
  // yönəlməlidir (məs. /en/tercume/-dən "Ana səhifə" linki /en/-ə, yox /-ə getsin).
  // EN qarşılığı olmayan səhifə (nsosyal) öz AZ href-inə düşür - PAGE_EN_PATH-də yoxdur.
  const homeBase = currentLang === 'en' ? `${BASE}${PAGE_EN_PATH.home}` : BASE
  const links = Object.entries(LINKS)
    .filter(([key]) => key !== page)
    .sort(([a], [b]) => (a === 'home' ? -1 : b === 'home' ? 1 : 0))
    .map(([key, l], i) => {
      const relHref = currentLang === 'en' ? (PAGE_EN_PATH[key] ?? l.href) : l.href
      const href = l.external ? l.href : `${BASE}${relHref}`
      const extAttrs = l.external ? ' target="_blank" rel="noopener noreferrer"' : ''
      return `<a class="navbar__link${i ? ' navbar__link--extra' : ''}" href="${href}"${extAttrs}>${l.label}</a>`
    })
    .join('\n    ')
  // Hamburger-dropdown-u olan səhifələrdə AZ|EN ayrıca sətirdə dropdown-un
  // sonunda da təkrarlanır (bax navbar.css ≤1023px bloku) - digər
  // səhifələrdə .navbar__links həmişə sətir-daxili göründüyü üçün bura
  // ƏLAVƏ OLUNMUR (əks halda masaüstü sətrində iki dəfə görünərdi).
  const menuLangBlock = hasMenu
    ? `\n    <span class="navbar__lang-divider" aria-hidden="true"></span>${langSwitchMarkup(page, ' navbar__lang--menu')}`
    : ''
  // CTA ≤600px-də sətir-daxili nüsxə (.navbar__end-dəki) gizlənir - home-da
  // hero-nun öz düymələri əvəz edir, lakin tercume/nsosyal-da əvəzedici
  // yoxdur. Ona görə hamburger-dropdown-u olan HƏR səhifədə CTA dropdown-un
  // son sətri kimi də təkrarlanır (bütün səhifələrdə eyni mobil davranış).
  const menuCtaBlock = hasMenu
    ? `\n    <a class="navbar__cta navbar__cta--menu" href="${home ? '' : homeBase}#pilot">${t('common.nav.cta')}</a>`
    : ''
  const cls = `${home ? '' : ' is-revealed'}${light ? ' is-light' : ''}`
  const navMenuAttr = hasMenu ? ' data-nav-menu' : ''
  return `
<header class="navbar${cls}" data-navbar data-page="${page}"${navMenuAttr}>
  <a class="navbar__brand" href="${home ? '#top' : homeBase}">
    <span class="navbar__logo" aria-hidden="true">
      <svg class="navbar__logo-svg navbar__logo-svg--dark" viewBox="${BRAND_LOGO_VIEWBOX}">${BRAND_LOGO_MARKUP_WHITE}</svg>
      <svg class="navbar__logo-svg navbar__logo-svg--light" viewBox="${BRAND_LOGO_VIEWBOX}">${NAVBAR_LOGO_MARKUP_BLUE}</svg>
    </span>
    <span class="navbar__name">${t('common.brand.name')}</span>
  </a>
  <nav class="navbar__links" id="navbar-links" aria-label="${t('common.nav.ariaLabel')}">
    ${links}${menuLangBlock}${menuCtaBlock}
  </nav>
  <div class="navbar__end">
    ${langSwitchMarkup(page)}${page === 'tercume' ? TERCUME_THEME_TOGGLE_MARKUP : ''}
    <a class="navbar__cta" href="${home ? '' : homeBase}#pilot">${t('common.nav.cta')}</a>
  </div>
  <button class="navbar__toggle" type="button" aria-expanded="false" aria-controls="navbar-links" aria-label="${t('common.nav.toggleAriaLabel')}">
    <span class="navbar__toggle-bar"></span>
    <span class="navbar__toggle-bar"></span>
  </button>
</header>
`
}

// Scroll > 40px-dən sonra görünən nazik "kölgə" xətti (.is-scrolled) - home-da
// hero.js-dəki initNavbar() özü idarə edir (digər bölmə detektoru ilə
// birləşik), digər səhifələrdə (tercume/nsosyal) həmin EYNİ sinif/hədd üçün
// bu kiçik köməkçi istifadə olunur ki, xətt scroll vəziyyətindən asılı
// olmayaraq həmişə görünməsin (əvvəlki bug: markup-a hardcode olunurdu).
export function initScrollShadow(navbarEl) {
  if (!navbarEl) return
  const update = () => navbarEl.classList.toggle('is-scrolled', scrollY > 40)
  update()
  addEventListener('scroll', update, { passive: true })
}

// Mobil/planşet hamburger menyusu (yalnız `[data-nav-menu]` olan
// səhifələrdə, bax MENU_ENABLED_PAGES + navbar.css) - tək, modul-səviyyəli
// delegated listener (navbar.js hər üç səhifənin main.js-i tərəfindən
// import olunur, ona görə buraya yazılan kod hər yerdə bir dəfə qoşulur;
// `data-nav-menu` olmayan səhifələrdə düymə display:none qaldığı üçün bu kod
// orda effektsizdir, həmin səhifələrin öz kodu TOXUNULMUR).
function closeNavbarMenu(navbarEl) {
  navbarEl.classList.remove('is-menu-open')
  navbarEl.querySelector('.navbar__toggle')?.setAttribute('aria-expanded', 'false')
}

document.addEventListener('click', (event) => {
  const toggle = event.target.closest('.navbar__toggle')
  if (toggle) {
    const navbarEl = toggle.closest('[data-navbar]')
    if (!navbarEl) return
    const isOpen = navbarEl.classList.toggle('is-menu-open')
    toggle.setAttribute('aria-expanded', String(isOpen))
    return
  }

  const openNavbar = document.querySelector('.navbar.is-menu-open')
  if (openNavbar && !event.target.closest('.navbar__links')) {
    closeNavbarMenu(openNavbar)
  }
})

document.addEventListener('keydown', (event) => {
  if (event.key !== 'Escape') return
  const openNavbar = document.querySelector('.navbar.is-menu-open')
  if (!openNavbar) return
  closeNavbarMenu(openNavbar)
  openNavbar.querySelector('.navbar__toggle')?.focus()
})
