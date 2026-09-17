import { defineConfig } from 'vite'
import { resolve } from 'node:path'

// /tercume (slash-sız) → /tercume/: Vite-in SPA fallback-i slash-sız ünvanda ana
// səhifəni qaytarır. Production-da eyni yönləndirmə vercel.json-dadır.
function redirectTercume(req, res, next) {
  const [path, query] = req.url.split('?')
  if (path !== '/tercume') return next()
  res.statusCode = 301
  res.setHeader('Location', `/tercume/${query ? `?${query}` : ''}`)
  res.end()
}

// İki səhifə: sayt (/) və tərcümə sistemi (/tercume/)
export default defineConfig({
  plugins: [
    {
      name: 'tercume-trailing-slash',
      // Hook heç nə qaytarmamalıdır: qaytarılan funksiyanı Vite sonradan çağırır
      configureServer(server) {
        server.middlewares.use(redirectTercume)
      },
      configurePreviewServer(server) {
        server.middlewares.use(redirectTercume)
      },
    },
  ],
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, 'index.html'),
        tercume: resolve(import.meta.dirname, 'tercume/index.html'),
        trending: resolve(import.meta.dirname, 'trending/index.html'),
      },
    },
  },
})
