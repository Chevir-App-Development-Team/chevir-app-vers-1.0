// public/brand/chevir-logo-blue.svg və chevir-logo-white.svg-nin JS kopyası -
// naviqasiya, hero fallback, preloader klonu və dekorativ fon buradan qurulur.
// Göy versiya gradient-dolğuludur (currentColor ilə ifadə oluna bilməz), ona
// görə funksiya kimi ixrac olunur: hər çağırış unikal linearGradient id-ləri
// yaradır ki, sənəddə eyni anda bir neçə nüsxə (məs. naviqasiya + preloader
// klonu) bir-birinin gradientini əvəz etməsin.
export const BRAND_LOGO_VIEWBOX = '0 0 1080 1080'

// Dekorativ fon (content.js logoBackgroundMarkup) da bu həndəsi məlumatı
// işlədir - təkrarlanan path/dairə dəyərlərinin tək mənbəyi.
export const BRAND_LOGO_DOT_LEFT = { cx: 423.4, cy: 594.3, r: 57.5 }
export const BRAND_LOGO_DOT_RIGHT = { cx: 656.6, cy: 594.3, r: 57.5 }
export const BRAND_LOGO_BLOB_A_D =
  'M195.3,662.5c0-85.3,30.5-153.3,91.4-203.9,40.9-33.9,92.5-56.5,154.8-67.6,50.2-9,86.5-53.1,86.5-104.1h0c0-68-63.1-118-129.5-103.2-47,10.5-90.6,26.5-131,48.1-78.6,42.1-139.6,100.3-182.9,174.6-43.3,74.3-64.9,159.7-64.9,256s0,4.5,0,6.7c1.4,149.1,140.7,256.9,286.1,224.1,6.2-1.4,12.5-2.8,18.7-4-86.2-42.4-129.3-118-129.3-226.8Z'
export const BRAND_LOGO_BLOB_B_D =
  'M996.3,406.5c-42.7-74.4-103.3-132.6-182-174.6-40.9-21.9-85.3-38.1-133-48.6-66.3-14.6-129.3,35.3-129.3,103.3h0c0,51.2,36.4,95.3,86.7,104.2,62.8,11.1,114.7,33.7,155.5,67.9,60.3,50.6,90.5,118.6,90.5,203.9s-14,111.6-42.1,151.8c-22.3,31.9-51.5,56.8-87.6,74.9,6.4,1.3,12.7,2.7,19.1,4.1,145.5,32.9,284.8-75.1,286.1-224.2s0-4.4,0-6.6c0-96.3-21.3-181.7-64-256Z'

const DOT_LEFT = BRAND_LOGO_DOT_LEFT
const DOT_RIGHT = BRAND_LOGO_DOT_RIGHT
const BLOB_A_D = BRAND_LOGO_BLOB_A_D
const BLOB_B_D = BRAND_LOGO_BLOB_B_D

// Hissələr "id" yox, "data-logo-part" ilə seçilir - eyni sənəddə bir neçə
// nüsxə (navbar + hero fallback kimi) eyni anda olanda HTML id təkrarlanması
// problemi yaranmasın deyə. Gradient id-ləri isə (aşağıda) hər nüsxə üçün
// unikal suffiks aldığından adi "id" qalır.
export const BRAND_LOGO_MARKUP_WHITE = `
  <path data-logo-part="blob-a" fill="#fff" d="${BLOB_A_D}"/>
  <path data-logo-part="blob-b" fill="#fff" d="${BLOB_B_D}"/>
  <circle data-logo-part="dot-left" fill="#fff" cx="${DOT_LEFT.cx}" cy="${DOT_LEFT.cy}" r="${DOT_LEFT.r}"/>
  <circle data-logo-part="dot-right" fill="#fff" cx="${DOT_RIGHT.cx}" cy="${DOT_RIGHT.cy}" r="${DOT_RIGHT.r}"/>
`

/**
 * @param {string} idSuffix - sənəddə bu nüsxəni unikal edən açar (məs. 'navbar', 'splash').
 */
export function buildBrandLogoMarkupBlue(idSuffix) {
  const gradDotLeft = `logo-grad-dot-left-${idSuffix}`
  const gradDotRight = `logo-grad-dot-right-${idSuffix}`
  const gradA = `logo-grad-a-${idSuffix}`
  const gradB = `logo-grad-b-${idSuffix}`
  return `
  <defs>
    <linearGradient id="${gradDotLeft}" x1="365.9" y1="594.3" x2="480.9" y2="594.3" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#0071ef"/>
      <stop offset="1" stop-color="#0021bf"/>
    </linearGradient>
    <linearGradient id="${gradDotRight}" x1="599.1" y1="594.3" x2="714.1" y2="594.3" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#0071ef"/>
      <stop offset="1" stop-color="#0021bf"/>
    </linearGradient>
    <linearGradient id="${gradA}" x1="19.7" y1="540.2" x2="528" y2="540.2" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#0071ef"/>
      <stop offset="1" stop-color="#0021bf"/>
    </linearGradient>
    <linearGradient id="${gradB}" x1="552" y1="540" x2="1060.3" y2="540" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#0071ef"/>
      <stop offset="1" stop-color="#0021bf"/>
    </linearGradient>
  </defs>
  <path data-logo-part="blob-a" fill="url(#${gradA})" d="${BLOB_A_D}"/>
  <path data-logo-part="blob-b" fill="url(#${gradB})" d="${BLOB_B_D}"/>
  <circle data-logo-part="dot-left" fill="url(#${gradDotLeft})" cx="${DOT_LEFT.cx}" cy="${DOT_LEFT.cy}" r="${DOT_LEFT.r}"/>
  <circle data-logo-part="dot-right" fill="url(#${gradDotRight})" cx="${DOT_RIGHT.cx}" cy="${DOT_RIGHT.cy}" r="${DOT_RIGHT.r}"/>
`
}
