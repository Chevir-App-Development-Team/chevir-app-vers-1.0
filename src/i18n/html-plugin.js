import { readFileSync } from 'node:fs'
import path from 'node:path'
import { NAMESPACES_BY_PAGE, HTML_ENTRIES, SITE_URL, DEFAULT_LOCALE } from './pages.config.js'

function loadNamespace(root, lang, namespace) {
  const file = path.join(root, 'src', 'i18n', lang, `${namespace}.json`)
  return JSON.parse(readFileSync(file, 'utf-8'))
}

function buildDict(root, lang, namespaces) {
  const dict = {}
  for (const ns of namespaces) {
    const data = loadNamespace(root, lang, ns)
    for (const [key, value] of Object.entries(data)) {
      dict[`${ns}.${key}`] = value
    }
  }
  return dict
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

// {{namespace.key}} yer tutucularını (yalnız <head>-dəki title/meta mətnləri
// üçün - body tam JS-dən render olunur) müvafiq dilin JSON-undan əvəz edir,
// və hər HTML giriş nöqtəsinə canonical + hreflang (az/en/x-default) link
// teqlərini AVTOMATİK əlavə edir (HTML-ə əl ilə yazılmır). Yeni dil/səhifə
// qoşmaq üçün pages.config.js-ə bir sətir kifayətdir - bu plugin özü
// dəyişmir.
export function i18nHtmlPlugin(root) {
  const normalizedRoot = root.replace(/\\/g, '/')

  return {
    name: 'chevir-i18n-html',
    transformIndexHtml: {
      order: 'pre',
      handler(html, ctx) {
        const filename = ctx.filename.replace(/\\/g, '/')
        const entryKey = Object.keys(HTML_ENTRIES).find((key) => filename === `${normalizedRoot}/${key}`)
        if (!entryKey) return html

        const entry = HTML_ENTRIES[entryKey]
        const namespaces = NAMESPACES_BY_PAGE[entry.page] || []
        const dict = buildDict(root, entry.lang, namespaces)

        const out = html.replace(/\{\{([\w.]+)\}\}/g, (match, key) => {
          if (!(key in dict)) {
            console.warn(`[i18n] missing key in ${entryKey}: ${key}`)
            return match
          }
          return escapeHtml(dict[key])
        })

        const siblings = Object.values(HTML_ENTRIES).filter((e) => e.page === entry.page)
        const tags = [
          { tag: 'link', attrs: { rel: 'canonical', href: `${SITE_URL}${entry.path}` }, injectTo: 'head' },
          ...siblings.map((sibling) => ({
            tag: 'link',
            attrs: { rel: 'alternate', hreflang: sibling.lang, href: `${SITE_URL}${sibling.path}` },
            injectTo: 'head',
          })),
        ]
        const defaultSibling = siblings.find((sibling) => sibling.lang === DEFAULT_LOCALE) || siblings[0]
        if (defaultSibling) {
          tags.push({
            tag: 'link',
            attrs: { rel: 'alternate', hreflang: 'x-default', href: `${SITE_URL}${defaultSibling.path}` },
            injectTo: 'head',
          })
        }

        return { html: out, tags }
      },
    },
  }
}
