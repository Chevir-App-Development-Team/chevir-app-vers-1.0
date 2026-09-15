"""JS portunu yoxlamaq üçün numpy referans irəli keçidi.

Determinik giriş yaradır, h5-dən birbaşa oxuyub Keras-ın etdiyi hesabı
numpy ilə təkrar edir, nəticəni fixture kimi yazır. Node tərəfi eyni girişlə
eyni çıxışı verməlidir.
"""
from pathlib import Path
import h5py, numpy as np, json

SRC = Path(__file__).resolve().parent.parent / "model" / "fingerspelling_33.h5"
OUT = Path(__file__).resolve().parent / "fixture"
OUT.mkdir(exist_ok=True)
LAYERS = ["dense_12", "dense_13", "dense_14", "dense_15"]

rng = np.random.default_rng(1234)
# Real MediaPipe landmark-ları kimi: x,y ∈ [0,1], z kiçik mənfi/müsbət
x = rng.random((20, 63)).astype(np.float32)
x[:, 2::3] = (rng.random((20, 21)).astype(np.float32) - 0.5) * 0.2
flat = x.reshape(-1)
OUT.joinpath("input.bin").write_bytes(flat.tobytes())

h = flat.astype(np.float32)
with h5py.File(SRC, "r") as f:
    g = f["model_weights"]
    for i, name in enumerate(LAYERS):
        W = np.array(g[name][name]["kernel:0"], dtype=np.float32)
        b = np.array(g[name][name]["bias:0"], dtype=np.float32)
        h = h @ W + b
        if i < 3:
            h = np.maximum(h, 0)
        else:
            e = np.exp(h - h.max()); h = e / e.sum()

top = np.argsort(h)[::-1][:5]
ref = {"probs": [float(v) for v in h], "top5": [int(t) for t in top]}
OUT.joinpath("expected.json").write_text(json.dumps(ref))
print("numpy referans (ilk 8):", " ".join(f"{v:.6f}" for v in h[:8]))
print("cəm:", float(h.sum()))
print("top5 indeks:", list(top), "| dəyər:", [round(float(h[t]),5) for t in top])
