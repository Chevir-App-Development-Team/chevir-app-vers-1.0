// Sistemin statik faylları public/tercume/-dadır. Yollar Vite-in base-inə görə
// mütləq qurulur ki, səhifə həm /tercume, həm /tercume/ ünvanında işləsin.
const BASE = `${import.meta.env.BASE_URL}tercume/`

export const ASSET_BASE = `${BASE}assets/`
export const WASM_BASE = `${BASE}wasm`
