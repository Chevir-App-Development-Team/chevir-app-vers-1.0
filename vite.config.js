import { defineConfig } from 'vite'
import { resolve } from 'node:path'
import { i18nHtmlPlugin } from './src/i18n/html-plugin.js'

// /tercume, /nsosyal, /en (slash-sız) → slash-lı: Vite-in SPA fallback-i
// slash-sız ünvanda ana səhifəni qaytarır. Production-da eyni yönləndirmə
// vercel.json-dadır.
const PAGES = ['/tercume', '/nsosyal', '/en', '/rey', '/qeyd', '/danis']
function redirectPages(req, res, next) {
  const [path, query] = req.url.split('?')
  if (!PAGES.includes(path)) return next()
  res.statusCode = 301
  res.setHeader('Location', `${path}/${query ? `?${query}` : ''}`)
  res.end()
}

// /danis/?debug=1 (yalnız dev): səhifə ölçmə sətirlərini bura göndərir, server konsola yazır — kamera ilə sınaqda
// (telefon da) sətirləri brauzerin konsolu olmadan oxumaq üçün
function danisLog(req, res, next) {
  if (req.method !== 'POST' || req.url !== '/__danis') return next()
  let body = ''
  req.on('data', (c) => { body += c })
  req.on('end', () => {
    try {
      const { line } = JSON.parse(body)
      console.log(`[danis] ${new Date().toISOString().slice(11, 23)} ${line}`)
    } catch { /* pozuq sətir */ }
    res.statusCode = 204
    res.end()
  })
}

// Səhifələr: sayt (/ + EN nüsxəsi /en/), tərcümə sistemi (/tercume/), işarə dilində danışmaq (/danis/),
// NSosyal inteqrasiya demosu (/nsosyal/), komanda alətləri (/rey/, /qeyd/)
export default defineConfig({
  plugins: [
    {
      name: 'pages-trailing-slash',
      // Hook heç nə qaytarmamalıdır: qaytarılan funksiyanı Vite sonradan çağırır
      configureServer(server) {
        server.middlewares.use(redirectPages)
        server.middlewares.use(danisLog)
      },
      configurePreviewServer(server) {
        server.middlewares.use(redirectPages)
      },
    },
    i18nHtmlPlugin(import.meta.dirname),
  ],
  // onnxruntime-web öz .wasm faylını import.meta.url ilə tapır: əvvəlcədən paketlənsə (dev) yol
  // pozulur, build isə faylı assets/-ə özü çıxarır — public/-də ayrıca nüsxə lazım deyil
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, 'index.html'),
        en: resolve(import.meta.dirname, 'en/index.html'),
        tercume: resolve(import.meta.dirname, 'tercume/index.html'),
        danis: resolve(import.meta.dirname, 'danis/index.html'),
        nsosyal: resolve(import.meta.dirname, 'nsosyal/index.html'),
        // komanda alətləri: işarə dili daşıyıcılarının rəyi (V2) və yazılış (D4); naviqasiyada yoxdur
        rey: resolve(import.meta.dirname, 'rey/index.html'),
        qeyd: resolve(import.meta.dirname, 'qeyd/index.html'),
      },
    },
  },
})
