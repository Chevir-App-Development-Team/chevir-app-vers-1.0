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

/**
 * Bir işarənin etiketləri → bir sinif. AzSLD söz formalarını ayrı etiketləyir (ACAM, ACMİŞAM), bəziləri
 * isə eyni işarədir; `merge` (tools/build_sign_merge.py → sign_merge.json: {spelling, merge}) onları
 * birləşdirir. Qaytarır: labels — birləşmiş siniflər (ilk rast gəlinən sırada, <blank> yerində qalır),
 * classOf — modelin hər sinfinin birləşmiş sinfi (birləşmə yoxdursa null), group — istənilən glosun sinfi.
 */
export function signGroups(labels, merge = null) {
  const group = (g) => { const c = merge?.spelling?.[g] ?? g; return merge?.merge?.[c] ?? c; };
  const out = [], index = new Map();
  const classOf = new Int32Array(labels.length);
  labels.forEach((l, i) => {
    const g = group(l);
    if (!index.has(g)) { index.set(g, out.length); out.push(g); }
    classOf[i] = index.get(g);
  });
  return { labels: out, classOf: out.length < labels.length ? classOf : null, group };
}

export class LocalRecognizer {
  constructor(manifest, session, ort, merge = null) {
    this.manifest = manifest;
    this.session = session;
    this.ort = ort;
    // Eyni işarənin etiketləri bir sinifdir: ehtimalları toplanır (softmax-dan sonra, ona görə dəqiqdir)
    const g = signGroups(manifest.labels, merge);
    this.labels = g.labels;
    this.classOf = g.classOf;
    this.group = g.group;
    this.groups = manifest.input.groups;
    this.backend = 'local';
  }

  /**
   * @param {string} [base] model qovluğu: manifest.json + manifest-də adı çəkilən .onnx
   * @param {{ fetchImpl?: typeof fetch, wasmBase?: string | null, merge?: string | object | null }} [opts]
   *   fetchImpl/wasmBase — sınaq skriptləri üçün (Node); merge — sign_merge.json-un ünvanı və ya özü
   *   (yüklənməsə modelin öz etiketləri qalır)
   */
  static async load(base = WORD_MODEL_BASE, { fetchImpl = fetch, wasmBase, merge = null } = {}) {
    const res = await fetchImpl(`${base}manifest.json`);
    if (!res.ok) throw new Error(`manifest.json yüklənmədi (${res.status})`);
    const manifest = await res.json();
    assertCompatible(manifest);
    if (!manifest.model_file) throw new Error('manifest-də model faylı yoxdur (ONNX ixracı alınmayıb)');
    const bytes = new Uint8Array(await (await fetchImpl(`${base}${manifest.model_file}`)).arrayBuffer());
    const ort = await loadOrt({ wasmBase });
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
    if (typeof merge === 'string') {
      merge = await fetchImpl(merge).then((r) => (r.ok ? r.json() : null)).catch(() => null);
      if (!merge) console.warn('sign_merge.json yüklənmədi: modelin öz etiketləri işlədilir');
    }
    return new LocalRecognizer(manifest, session, ort, merge);
  }

  /** Modelin siniflərinin ehtimalları → birləşmiş siniflərin ehtimalları (cəm). */
  #mergeProbs(probs) {
    if (!this.classOf) return probs;
    const out = new Float32Array(this.labels.length);
    for (let i = 0; i < probs.length; i++) out[this.classOf[i]] += probs[i];
    return out;
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
    let logits = out[this.manifest.output.name].data;
    const probs = this.#mergeProbs(softmax(logits));
    if (this.classOf) logits = probs.map(Math.log);        // birləşmiş siniflərin logit-i: softmax-ı eyni probs verir
    return { top: topK(probs, this.labels, k), probs, logits, frames: x.T, ms: performance.now() - t0, backend: this.backend };
  }

  /**
   * Davamlı model (manifest.kind = 'continuous'): pəncərənin HƏR kadrı üçün ehtimallar.
   * @returns {{ T: number, C: number, probs: Float32Array, ms: number }} probs = T×C, sətir-sətir softmax
   */
  async frameProbs(raw, aspect, { face = true } = {}) {
    const t0 = performance.now();
    const x = preprocess(raw, aspect, { groups: this.groups, face, maxLen: this.manifest.input.max_len ?? null });
    const { name: inName, n_features: nf } = this.manifest.input;
    if (x.F !== nf) throw new Error(`giriş ${x.F} sütun, model ${nf} gözləyir`);
    const out = await this.session.run({ [inName]: new this.ort.Tensor('float32', x.data, [1, x.T, x.F]) });
    const logits = out[this.manifest.output.name].data;
    const M = this.manifest.labels.length, C = this.labels.length;
    const probs = new Float32Array(x.T * C);
    for (let t = 0; t < x.T; t++) probs.set(this.#mergeProbs(softmax(logits.subarray(t * M, (t + 1) * M))), t * C);
    return { T: x.T, C, probs, ms: performance.now() - t0 };
  }
}
