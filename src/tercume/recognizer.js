/**
 * Söz tanıyıcı: bir işarənin landmark-ları → lüğətdəki sözlərin ehtimalları.
 *
 * Bütün arxa uclar eyni interfeysi verir:
 *   const rec = await LocalRecognizer.load();
 *   const r = await rec.recognize(clip.toRaw(a, b), aspect, { face, topK });
 *   // r = { top: [{ label, p, idx }], probs, logits, frames, ms, backend }
 * `raw` features.js-dəki giriş müqaviləsidir (LandmarkClip.toRaw()), `aspect` — kadrın en/hündürlüyü.
 *
 * LocalRecognizer brauzerdə işləyir (onnxruntime-web, WASM, tək axın — saytın xüsusi HTTP
 * başlıqlarına ehtiyacı yoxdur). Default və ehtiyat yoldur. Sonra əlavə olunacaq ServerRecognizer
 * eyni girişi — yalnız landmark rəqəmlərini — landmark serverinə göndərəcək; video heç vaxt
 * cihazdan çıxmır.
 *
 * Model manifest.json ilə gəlir (ml/azsl/train.py write_manifest): ön emalın versiyası
 * (FEATURE_VERSION) tutmursa model yüklənmir, giriş sütunları manifest-dəki qruplardan seçilir.
 */
import { assertCompatible, preprocess } from './features.js';
import { WORD_MODEL_BASE } from './paths.js';

let ortPromise = null;

/**
 * onnxruntime-web yalnız lazım olanda yüklənir. 'onnxruntime-web/wasm' JS hissəni özündə saxlayır,
 * .wasm faylını isə paketin yanından tapır (Vite build onu assets/-ə çıxarır, versiya həmişə
 * paketlə eynidir). wasmBase — başqa yerdən vermək üçün (məs. CDN); adətən lazım deyil.
 */
export function loadOrt({ wasmBase = null } = {}) {
  ortPromise ??= import('onnxruntime-web/wasm').then((ort) => {
    if (wasmBase) ort.env.wasm.wasmPaths = { wasm: `${wasmBase}ort-wasm-simd-threaded.wasm` };
    ort.env.wasm.numThreads = 1;      // çox axın SharedArrayBuffer (COOP/COEP başlıqları) tələb edir
    return ort;
  });
  return ortPromise;
}

export function softmax(logits) {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) { out[i] = Math.exp(logits[i] - max); sum += out[i]; }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

export function topK(probs, labels, k = 5) {
  return Array.from(probs.keys())
    .sort((a, b) => probs[b] - probs[a])
    .slice(0, k)
    .map((i) => ({ label: labels[i], p: probs[i], idx: i }));
}

export class LocalRecognizer {
  constructor(manifest, session, ort) {
    this.manifest = manifest;
    this.session = session;
    this.ort = ort;
    this.labels = manifest.labels;
    this.groups = manifest.input.groups;
    this.backend = 'local';
  }

  /**
   * @param {string} [base] model qovluğu: manifest.json + manifest-də adı çəkilən .onnx
   * @param {{ fetchImpl?: typeof fetch, wasmBase?: string | null }} [opts] sınaq skriptləri üçün (Node)
   */
  static async load(base = WORD_MODEL_BASE, { fetchImpl = fetch, wasmBase } = {}) {
    const res = await fetchImpl(`${base}manifest.json`);
    if (!res.ok) throw new Error(`manifest.json yüklənmədi (${res.status})`);
    const manifest = await res.json();
    assertCompatible(manifest);
    if (!manifest.model_file) throw new Error('manifest-də model faylı yoxdur (ONNX ixracı alınmayıb)');
    const bytes = new Uint8Array(await (await fetchImpl(`${base}${manifest.model_file}`)).arrayBuffer());
    const ort = await loadOrt({ wasmBase });
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
    return new LocalRecognizer(manifest, session, ort);
  }

  /**
   * @param {object} raw LandmarkClip.toRaw() — bir işarənin kadrları
   * @param {number} aspect kadrın eni / hündürlüyü
   * @param {{ face?: boolean, topK?: number }} [opts] face=false — məxfilik rejimi (üz nöqtələri yoxdur)
   */
  async recognize(raw, aspect, { face = true, topK: k = 5 } = {}) {
    const t0 = performance.now();
    const x = preprocess(raw, aspect, { groups: this.groups, face });
    const { name: inName, n_features: nf } = this.manifest.input;
    if (x.F !== nf) throw new Error(`giriş ${x.F} sütun, model ${nf} gözləyir`);
    const out = await this.session.run({ [inName]: new this.ort.Tensor('float32', x.data, [1, x.T, x.F]) });
    const logits = out[this.manifest.output.name].data;
    const probs = softmax(logits);
    return { top: topK(probs, this.labels, k), probs, logits, frames: x.T, ms: performance.now() - t0, backend: this.backend };
  }
}
