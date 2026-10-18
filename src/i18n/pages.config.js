// Tək mənbə: hansı səhifə hansı JSON namespace-lərini işlədir, və hansı HTML
// giriş nöqtəsi hansı dilə/yola aiddir. Həm build-time HTML plugin-i
// (vite.config.js, Node tərəfi), həm client-side yükləyici (lib/i18n.js,
// browser tərəfi) bu faylı import edir - dəyərlər İKİ YERDƏ YAZILMIR.
// Yeni səhifəni dil üçün qoşmaq: NAMESPACES_BY_PAGE-ə bölmələrini,
// HTML_ENTRIES-ə hər dil üçün bir sətir əlavə et.

export const SITE_URL = 'https://chevirapp.com'

export const DEFAULT_LOCALE = 'az'

// page-açarı → həmin səhifənin istifadə etdiyi i18n JSON fayllarının adları
// (src/i18n/<lang>/<namespace>.json). Hələlik yalnız "home" aktivdir.
export const NAMESPACES_BY_PAGE = {
  home: ['common', 'home'],
}

// HTML giriş nöqtəsi (layihə kökündən nisbi yol, vite.config.js-dəki
// rollupOptions.input ilə uyğun) → hansı səhifə/dil/kanonik yol.
export const HTML_ENTRIES = {
  'index.html': { page: 'home', lang: 'az', path: '/' },
  'en/index.html': { page: 'home', lang: 'en', path: '/en/' },
}
