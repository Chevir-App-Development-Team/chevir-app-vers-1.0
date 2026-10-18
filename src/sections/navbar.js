import { BRAND_LOGO_MARKUP_WHITE, buildBrandLogoMarkupBlue, BRAND_LOGO_VIEWBOX } from '../lib/brandLogo.js'

// Naviqasiya paneli üçün loqonun iki nüsxəsi (ağ + göy) üst-üstə yerləşdirilir
// və navbar.css-dəki .is-light sinfinə görə bir-birinin üzərinə sönür/yanır -
// göy nüsxənin gradient id-ləri "navbar" açarı ilə unikaldır (bax brandLogo.js),
// ona görə preloader klonu ilə (başqa açar) heç vaxt toqquşmur.
const NAVBAR_LOGO_MARKUP_BLUE = buildBrandLogoMarkupBlue('navbar')

const BASE = import.meta.env.BASE_URL

const LINKS = {
  home: { href: '', label: 'Ana səhifə' },
  tercume: { href: 'tercume/', label: 'Sistemi sına' },
  nsosyal: { href: 'nsosyal/', label: 'NSosyal' },
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
  const links = Object.entries(LINKS)
    .filter(([key]) => key !== page)
    .sort(([a], [b]) => (a === 'home' ? -1 : b === 'home' ? 1 : 0))
    .map(([, l], i) => `<a class="navbar__link${i ? ' navbar__link--extra' : ''}" href="${BASE}${l.href}">${l.label}</a>`)
    .join('\n    ')
  const cls = `${home ? '' : ' is-revealed is-scrolled'}${light ? ' is-light' : ''}`
  return `
<header class="navbar${cls}" data-navbar data-page="${page}">
  <a class="navbar__brand" href="${home ? '#top' : BASE}">
    <span class="navbar__logo" aria-hidden="true">
      <svg class="navbar__logo-svg navbar__logo-svg--dark" viewBox="${BRAND_LOGO_VIEWBOX}">${BRAND_LOGO_MARKUP_WHITE}</svg>
      <svg class="navbar__logo-svg navbar__logo-svg--light" viewBox="${BRAND_LOGO_VIEWBOX}">${NAVBAR_LOGO_MARKUP_BLUE}</svg>
    </span>
    <span class="navbar__name">Chevir</span>
  </a>
  <nav class="navbar__links" id="navbar-links" aria-label="Əsas keçidlər">
    ${links}
  </nav>
  <a class="navbar__cta" href="${home ? '' : BASE}#pilot">Pilot tərəfdaş olun</a>
  <button class="navbar__toggle" type="button" aria-expanded="false" aria-controls="navbar-links" aria-label="Menyu">
    <span class="navbar__toggle-bar"></span>
    <span class="navbar__toggle-bar"></span>
  </button>
</header>
`
}

// Mobil hamburger menyusu (yalnız data-page="home" + ≤600px-də görünür, bax
// navbar.css) - tək, modul-səviyyəli delegated listener (navbar.js hər üç
// səhifənin main.js-i tərəfindən import olunur, ona görə buraya yazılan kod
// hər yerdə bir dəfə qoşulur; /tercume və /nsosyal-da düymə display:none
// qaldığı üçün bu kod orda effektsizdir, həmin səhifələrin öz kodu TOXUNULMUR).
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
