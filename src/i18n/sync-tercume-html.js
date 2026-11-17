import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import path from 'node:path'

const GENERATED_NOTICE = '<!-- AUTO-GENERATED from tercume/index.html by src/i18n/sync-tercume-html.js - do not hand-edit, changes are overwritten on every dev/build start. -->\n'

// tercume/index.html-in bütün görünən mətni {{tercume.*}}/{{common.*}} açarlarıdır
// (lang daxil: <html lang="{{tercume.meta.htmlLang}}">), ona görə AZ/EN mənbəyi
// mətn baxımından İDENTİKDİR - fərq yalnız html-plugin.js-in hər giriş nöqtəsi
// üçün ayrı dict-lə doldurduğu açarlardadır. Generator sadəcə köçürür.
function generate(root) {
  const src = path.join(root, 'tercume', 'index.html')
  const dest = path.join(root, 'en', 'tercume', 'index.html')
  const html = readFileSync(src, 'utf-8')
  mkdirSync(path.dirname(dest), { recursive: true })
  writeFileSync(dest, GENERATED_NOTICE + html)
}

export function syncEnTercumeHtmlPlugin(root) {
  generate(root)

  return {
    name: 'chevir-sync-en-tercume-html',
    configureServer(server) {
      server.watcher.add(path.join(root, 'tercume', 'index.html'))
      server.watcher.on('change', (file) => {
        if (file.replace(/\\/g, '/') !== path.join(root, 'tercume', 'index.html').replace(/\\/g, '/')) return
        generate(root)
        server.ws.send({ type: 'full-reload' })
      })
    },
    configurePreviewServer(server) {
      server.watcher?.add(path.join(root, 'tercume', 'index.html'))
    },
  }
}
