// public/logo.svg-nin dəyişməz kopyası - 3D əl yüklənməyəndə (reduced-motion/
// WebGL yoxdursa) statik fallback kimi qalır; naviqasiya loqosu da bundan qurulur.
// fill="none" hər qövsdə birbaşa yazılıb: bu svg üçün ayrıca <svg> elementinin
// özündə fill="none" yoxdur, ona görə miras gözləmək əvəzinə açıq təyin olunur.
export const BRAND_LOGO_MARKUP = `
  <path id="arc-signed" fill="none" d="M50.5 88.5C45.3784 88.5 40.307 87.4912 35.5753 85.5313C30.8436 83.5714 26.5443 80.6986 22.9228 77.0772C19.3013 73.4557 16.4286 69.1563 14.4687 64.4247C12.5088 59.693 11.5 54.6215 11.5 49.5" stroke="#D6006C" stroke-width="7"/>
  <path id="arc-spoken" fill="none" d="M49.5 10.5C54.6216 10.5 59.693 11.5088 64.4247 13.4687C69.1564 15.4286 73.4557 18.3014 77.0772 21.9228C80.6986 25.5443 83.5714 29.8436 85.5313 34.5753C87.4912 39.307 88.5 44.3784 88.5 49.5" stroke="#0088B0" stroke-width="7"/>
  <path id="head-signed" d="M11.5 34L21.4593 49.75H1.54071L11.5 34Z" fill="#D6006C"/>
  <path id="head-spoken" d="M88.5 65L78.5407 49.25L98.4593 49.25L88.5 65Z" fill="#0088B0"/>
  <circle id="dot" cx="50" cy="50" r="4" fill="currentColor"/>
`
