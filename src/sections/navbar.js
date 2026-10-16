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
  nsosyal: { href: 'nsosyal/', label: 'NSosyal demosu' },
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
<header class="navbar${cls}" data-navbar>
  <a class="navbar__brand" href="${home ? '#top' : BASE}">
    <span class="navbar__logo" aria-hidden="true">
      <svg class="navbar__logo-svg navbar__logo-svg--dark" viewBox="${BRAND_LOGO_VIEWBOX}">${BRAND_LOGO_MARKUP_WHITE}</svg>
      <svg class="navbar__logo-svg navbar__logo-svg--light" viewBox="${BRAND_LOGO_VIEWBOX}">${NAVBAR_LOGO_MARKUP_BLUE}</svg>
    </span>
    <span class="navbar__name">Chevir</span>
  </a>
  <nav class="navbar__actions" aria-label="Əsas keçidlər">
    ${links}
    <a class="navbar__cta" href="${home ? '' : BASE}#pilot">Pilot tərəfdaş olun</a>
  </nav>
</header>
`
}
