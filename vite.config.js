import { defineConfig } from 'vite'
import { resolve } from 'node:path'
import { i18nHtmlPlugin } from './src/i18n/html-plugin.js'

// /tercume, /nsosyal, /en (slash-sız) → slash-lı: Vite-in SPA fallback-i
// slash-sız ünvanda ana səhifəni qaytarır. Production-da eyni yönləndirmə
// vercel.json-dadır.
const PAGES = ['/tercume', '/nsosyal', '/en']
function redirectPages(req, res, next) {
  const [path, query] = req.url.split('?')
  if (!PAGES.includes(path)) return next()
  res.statusCode = 301
  res.setHeader('Location', `${path}/${query ? `?${query}` : ''}`)
  res.end()
}

// Dörd səhifə: sayt (/ + EN nüsxəsi /en/), tərcümə sistemi (/tercume/),
// NSosyal inteqrasiya demosu (/nsosyal/)
export default defineConfig({
  plugins: [
    {
      name: 'pages-trailing-slash',
      // Hook heç nə qaytarmamalıdır: qaytarılan funksiyanı Vite sonradan çağırır
      configureServer(server) {
        server.middlewares.use(redirectPages)
      },
      configurePreviewServer(server) {
        server.middlewares.use(redirectPages)
      },
    },
    i18nHtmlPlugin(import.meta.dirname),
  ],
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, 'index.html'),
        en: resolve(import.meta.dirname, 'en/index.html'),
        tercume: resolve(import.meta.dirname, 'tercume/index.html'),
        nsosyal: resolve(import.meta.dirname, 'nsosyal/index.html'),
      },
    },
  },
})
