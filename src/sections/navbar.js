import { BRAND_LOGO_MARKUP } from '../lib/brandLogo.js'

// Naviqasiya paneli üçün loqonun sadə, animasiyasız kopyası - id toqquşmasının
// qarşısını almaq üçün id-siz.
const NAVBAR_LOGO_MARKUP = BRAND_LOGO_MARKUP.replace(/\sid="[^"]*"/g, '')

const BASE = import.meta.env.BASE_URL

/**
 * Ana səhifə və /tercume/ üçün eyni naviqasiya.
 * Ana səhifədə loqo və CTA səhifə daxili anchor-lardır; digər səhifələrdə naviqasiya
 * splash gözləmədən görünür və keçidlər ana səhifəyə qayıdır.
 */
export function navbarMarkup({ home = true } = {}) {
  const link = home
    ? `<a class="navbar__link" href="${BASE}tercume/">Sistemi sına</a>`
    : `<a class="navbar__link" href="${BASE}">Ana səhifə</a>`
  return `
<header class="navbar${home ? '' : ' is-revealed is-scrolled'}" data-navbar>
  <a class="navbar__brand" href="${home ? '#top' : BASE}">
    <svg class="navbar__logo-svg" viewBox="0 0 100 100" aria-hidden="true">${NAVBAR_LOGO_MARKUP}</svg>
    <span class="navbar__name">Chevir</span>
  </a>
  <nav class="navbar__actions" aria-label="Əsas keçidlər">
    ${link}
    <a class="navbar__cta" href="${home ? '' : BASE}#pilot">Pilot tərəfdaş olun</a>
  </nav>
</header>
`
}
