import azCommon from '../i18n/az/common.json'
import azHome from '../i18n/az/home.json'
import enCommon from '../i18n/en/common.json'
import enHome from '../i18n/en/home.json'
import { NAMESPACES_BY_PAGE } from '../i18n/pages.config.js'

const NAMESPACE_FILES = {
  az: { common: azCommon, home: azHome },
  en: { common: enCommon, home: enHome },
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

// `<html lang>` statik HTML-də (index.html / en/index.html) artıq doğru
// qoyulub - bu modul yüklənəndə element çoxdan parse olunub, ona görə burda
// sinxron oxumaq təhlükəsizdir (DOMContentLoaded gözləmək lazım deyil).
export const currentLang = document.documentElement.lang === 'en' ? 'en' : 'az'

const dict = buildDict(currentLang, NAMESPACES_BY_PAGE.home)

export function t(key) {
  if (!(key in dict)) {
    console.warn(`[i18n] missing key: ${key}`)
    return key
  }
  return dict[key]
}
