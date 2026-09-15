/**
 * Barmaq əlifbası modelinin brauzerdə irəli keçidi.
 *
 * Müəllimin model.h5-i sadə Sequential MLP-dir:
 *   Input(20, 63) -> Flatten(1260) -> Dense(300,relu) -> Dense(256,relu)
 *                 -> Dense(128,relu) -> Dense(35,softmax)
 * Dropout qatları yalnız təlimdə işlədiyi üçün nəzərə alınmır.
 *
 * Çəkilər model.bin-də ardıcıl float32 kimidir: hər qat üçün əvvəl kernel
 * (giriş × çıxış, row-major), sonra bias. Keras kernel-i (giriş, çıxış)
 * saxladığı üçün transpose lazım deyil.
 */

export class FingerspellModel {
  constructor(meta, buffer) {
    this.meta = meta;
    this.labels = meta.labels;
    this.flatSize = meta.input.flat;

    const f32 = new Float32Array(buffer);
    let off = 0;
    this.layers = meta.layers.map((l, i) => {
      const nW = l.in * l.out;
      const W = f32.subarray(off, off + nW); off += nW;
      const b = f32.subarray(off, off + l.out); off += l.out;
      return { ...l, W, b, act: meta.activations[i] };
    });

    if (off !== f32.length) {
      throw new Error(`çəki ölçüsü uyğun gəlmir: ${off} != ${f32.length}`);
    }
    // Təkrar ayırmadan qaçmaq üçün hər qat üçün çıxış buferi
    this.buffers = this.layers.map((l) => new Float32Array(l.out));
  }

  static async load(metaUrl, binUrl) {
    const [meta, buf] = await Promise.all([
      fetch(metaUrl).then((r) => r.json()),
      fetch(binUrl).then((r) => r.arrayBuffer()),
    ]);
    return new FingerspellModel(meta, buf);
  }

  /**
   * @param {Float32Array} x 1260 ölçülü giriş (20 kadr × 63)
   * @returns {Float32Array} 35 sinif üzrə ehtimallar
   */
  predict(x) {
    if (x.length !== this.flatSize) {
      throw new Error(`giriş ${this.flatSize} olmalıdır, ${x.length} gəldi`);
    }
    let input = x;
    for (let li = 0; li < this.layers.length; li++) {
      const { W, b, in: nIn, out: nOut, act } = this.layers[li];
      const y = this.buffers[li];
      // y = b + xᵀW
      y.set(b);
      for (let i = 0; i < nIn; i++) {
        const xi = input[i];
        if (xi === 0) continue;           // seyrək giriş (aşkarlanmayan kadrlar sıfırdır)
        const row = i * nOut;
        for (let j = 0; j < nOut; j++) y[j] += xi * W[row + j];
      }
      if (act === 'relu') {
        for (let j = 0; j < nOut; j++) if (y[j] < 0) y[j] = 0;
      } else if (act === 'softmax') {
        let max = -Infinity;
        for (let j = 0; j < nOut; j++) if (y[j] > max) max = y[j];
        let sum = 0;
        for (let j = 0; j < nOut; j++) { y[j] = Math.exp(y[j] - max); sum += y[j]; }
        for (let j = 0; j < nOut; j++) y[j] /= sum;
      }
      input = y;
    }
    return input;
  }

  /** Ehtimal vektorundan ən yüksək k sinif: [{label, p, idx}] */
  topK(probs, k = 3) {
    const idx = Array.from(probs.keys());
    idx.sort((a, b) => probs[b] - probs[a]);
    return idx.slice(0, k).map((i) => ({ label: this.labels[i], p: probs[i], idx: i }));
  }
}
