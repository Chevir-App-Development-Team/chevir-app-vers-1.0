import { BRAND_LOGO_MARKUP } from '../lib/brandLogo.js'

// Naviqasiya paneli üçün loqonun sadə, animasiyasız kopyası - id toqquşmasının
// qarşısını almaq üçün id-siz.
const NAVBAR_LOGO_MARKUP = BRAND_LOGO_MARKUP.replace(/\sid="[^"]*"/g, '')

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
    <svg class="navbar__logo-svg" viewBox="0 0 100 100" aria-hidden="true">${NAVBAR_LOGO_MARKUP}</svg>
    <span class="navbar__name">Chevir</span>
  </a>
  <nav class="navbar__actions" aria-label="Əsas keçidlər">
    ${links}
    <a class="navbar__cta" href="${home ? '' : BASE}#pilot">Pilot tərəfdaş olun</a>
  </nav>
</header>
`
}
