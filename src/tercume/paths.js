// Sistemin statik faylları public/tercume/-dadır. Yollar Vite-in base-inə görə
// mütləq qurulur ki, səhifə həm /tercume, həm /tercume/ ünvanında işləsin.
// `import.meta.env` yalnız Vite-dədir; Node-dakı sınaq skriptləri üçün ehtiyat dəyər
const BASE = `${import.meta.env?.BASE_URL ?? '/'}tercume/`

export const ASSET_BASE = `${BASE}assets/`
export const WASM_BASE = `${BASE}wasm`
export const WORDS_BASE = `${BASE}words/`
// Söz modeli (model.onnx + manifest.json)
export const WORD_MODEL_BASE = `${BASE}assets/word_model/`
export const CONT_MODEL_BASE = `${BASE}assets/cont_model/`
export const LETTER_MODEL_BASE = `${BASE}assets/letter_model/`
