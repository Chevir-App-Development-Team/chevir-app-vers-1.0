import azCommon from '../i18n/az/common.json'
import azHome from '../i18n/az/home.json'
import azTercume from '../i18n/az/tercume.json'
import enCommon from '../i18n/en/common.json'
import enHome from '../i18n/en/home.json'
import enTercume from '../i18n/en/tercume.json'
import { NAMESPACES_BY_PAGE } from '../i18n/pages.config.js'

const NAMESPACE_FILES = {
  az: { common: azCommon, home: azHome, tercume: azTercume },
  en: { common: enCommon, home: enHome, tercume: enTercume },
}

function buildDict(lang, namespaces) {
  const dict = {}
  namespaces.forEach((ns) => {
    const file = NAMESPACE_FILES[lang][ns]
    Object.entries(file).forEach(([key, value]) => {
      dict[`${ns}.${key}`] = value
    })
  })
  return dict
}

// `<html lang>` statik HTML-də (index.html / en/index.html / tercume/index.html /
// en/tercume/index.html) artıq doğru qoyulub - bu modul yüklənəndə element çoxdan
// parse olunub, ona görə burda sinxron oxumaq təhlükəsizdir (DOMContentLoaded
// gözləmək lazım deyil).
export const currentLang = document.documentElement.lang === 'en' ? 'en' : 'az'

// `<html data-page>` ilə işarələnir (bax pages.config.js HTML_ENTRIES) - tanınmayan/
// yoxdursa 'home'-a düşür (nsosyal kimi i18n-ə qoşulmayan səhifələr daxil).
const pageAttr = document.documentElement.dataset.page
export const currentPage = pageAttr && NAMESPACES_BY_PAGE[pageAttr] ? pageAttr : 'home'

const dict = buildDict(currentLang, NAMESPACES_BY_PAGE[currentPage])

// `params` verilsə, `{token}` yer tutucularını əvəz edir (məs. t('tercume.status.wordCount', { n: 12 })).
export function t(key, params) {
  if (!(key in dict)) {
    console.warn(`[i18n] missing key: ${key}`)
    return key
  }
  const value = dict[key]
  if (!params) return value
  return value.replace(/\{(\w+)\}/g, (match, token) => (token in params ? String(params[token]) : match))
}
