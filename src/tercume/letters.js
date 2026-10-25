/**
 * Barmaq əlifbası, landmark modeli (ml/azsl/letters.py, notebook 07): yalnız əlin forması və hərəkəti.
 *
 * Hərf rejimi yalnız əl izləyicisini işlədir, ona görə model bədən və üz görmür (strip). Pəncərədə daha
 * çox kadrda görünən əl 0-cı slota keçir (canonical): tək əl həmişə 0-cı slotdadır, sağ və sol əllə
 * göstərilən hərf modelə eyni görünür. Sonra features.js preprocess — söz modeli ilə eyni müqavilə.
 * tools/verify_letters.mjs bunları ml/fixtures/letters_v<N> üzərində Python ilə tutuşdurur.
 */
import { preprocess } from './features.js';
import { LETTER_MODEL_BASE } from './paths.js';
import { LocalRecognizer, softmax } from './recognizer.js';

export const LETTERS_VERSION = 1;    // ml/azsl/letters.py LETTERS_VERSION
const SLOT = 63;                      // bir əlin 21 nöqtəsi × 3

/** Bədən, üz və blendshape-lər yoxdur (NaN): hərf rejimində yalnız əl izlənir. */
export function strip(raw) {
  const out = { ...raw };
  for (const k of ['pose', 'face', 'blend']) out[k] = new Float32Array(raw[k].length).fill(NaN);
  return out;
}

/** Daha çox kadrda görünən əl 0-cı slot olur (bərabərdirsə sıra qalır). */
export function canonical(raw) {
  const n = [0, 0];
  for (let t = 0; t < raw.T; t++) {
    for (let s = 0; s < 2; s++) if (Number.isFinite(raw.hand[t * 2 * SLOT + s * SLOT])) n[s]++;
  }
  if (n[1] <= n[0]) return raw;
  const swap = (a) => {
    const o = new Float32Array(a.length);
    for (let t = 0; t < raw.T; t++) {
      o.set(a.subarray(t * 2 * SLOT + SLOT, (t + 1) * 2 * SLOT), t * 2 * SLOT);
      o.set(a.subarray(t * 2 * SLOT, t * 2 * SLOT + SLOT), t * 2 * SLOT + SLOT);
    }
    return o;
  };
  return { ...raw, hand: swap(raw.hand), hand_world: swap(raw.hand_world) };
}

/** letters.py preprocess: strip → canonical → features.preprocess (üzsüz, modelin qrupları). */
export function preprocessLetters(raw, aspect, groups) {
  return preprocess(canonical(strip(raw)), aspect, { groups, face: false });
}

/** Hərf modeli: son `window` kadr → hərflər + '<none>' üzrə ehtimallar. */
export class LetterModel {
  constructor(rec) {
    const m = rec.manifest;
    if (m.kind !== 'letters') throw new Error('bu model hərf modeli deyil');
    if ((m.letters_version ?? 1) > LETTERS_VERSION) {
      throw new Error(`model letters_version ${m.letters_version} istəyir, brauzer ${LETTERS_VERSION}-dir`);
    }
    this.rec = rec;
    this.kind = 'letters';
    this.manifest = m;
    this.labels = m.labels;
    this.none = m.none_label;
    this.window = m.input.window;
    this.dynamic = new Set(m.dynamic ?? []);
    this.groups = m.input.groups;
  }

  static async load(base = LETTER_MODEL_BASE, opts = {}) {
    return new LetterModel(await LocalRecognizer.load(base, opts));
  }

  /** @returns {Promise<Float32Array>} ehtimallar, `labels` sırası ilə */
  async predictRaw(raw, aspect) {
    const x = preprocessLetters(raw, aspect, this.groups);
    const { name, n_features: nf } = this.manifest.input;
    if (x.F !== nf) throw new Error(`giriş ${x.F} sütun, model ${nf} gözləyir`);
    const out = await this.rec.session.run({ [name]: new this.rec.ort.Tensor('float32', x.data, [1, x.T, x.F]) });
    return softmax(out[this.manifest.output.name].data);
  }

  /** Ən ehtimallı k sinif: [{label, p, idx}] ('<none>' də daxil). */
  topK(probs, k = 3) {
    return Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, k)
      .map((i) => ({ label: this.labels[i], p: probs[i], idx: i }));
  }
}

/**
 * Hərf rejiminin modeli: yeni landmark modeli, alınmasa köhnə model (model.json + model.bin).
 * @returns {Promise<LetterModel | import('./model.js').FingerspellModel>}
 */
export async function loadLetterModel({ assetBase, base = LETTER_MODEL_BASE } = {}) {
  try {
    return await LetterModel.load(base);
  } catch (err) {
    console.warn('Hərf modeli yüklənmədi, köhnə model işlədilir:', err);
    const { FingerspellModel } = await import('./model.js');
    return FingerspellModel.load(`${assetBase}model.json`, `${assetBase}model.bin`);
  }
}
