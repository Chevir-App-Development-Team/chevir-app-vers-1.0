import { defineConfig } from 'vite'
import { resolve } from 'node:path'

// /tercume, /nsosyal (slash-sız) → slash-lı: Vite-in SPA fallback-i slash-sız ünvanda
// ana səhifəni qaytarır. Production-da eyni yönləndirmə vercel.json-dadır.
const PAGES = ['/tercume', '/nsosyal']
function redirectPages(req, res, next) {
  const [path, query] = req.url.split('?')
  if (!PAGES.includes(path)) return next()
  res.statusCode = 301
  res.setHeader('Location', `${path}/${query ? `?${query}` : ''}`)
  res.end()
}

// Üç səhifə: sayt (/), tərcümə sistemi (/tercume/), NSosyal inteqrasiya demosu (/nsosyal/)
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
  ],
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, 'index.html'),
        tercume: resolve(import.meta.dirname, 'tercume/index.html'),
        nsosyal: resolve(import.meta.dirname, 'nsosyal/index.html'),
      },
    },
  },
})
