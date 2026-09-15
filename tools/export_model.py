"""
Keras model.h5 -> brauzer üçün xam çəki faylı + meta JSON.

Model sadə Sequential MLP-dir (Flatten + 4 Dense), ona görə TensorFlow.js
lazım deyil: çəkiləri float32 binar kimi yazırıq, irəli keçidi JS-də
4 matmul ilə özümüz hesablayırıq.

Keras Dense kernel: (giriş, çıxış)  |  bizim JS matmul: eyni sıra ilə oxuyuruq
(row-major, giriş sətir) — transpose lazım deyil.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "model" / "fingerspelling_33.h5"
OUT_DIR = ROOT / "public" / "tercume" / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Müəllimin notebook-undaki sıra (Beam Search + Lexicon Verification.ipynb, cell 2)
ACTIONS = ['a', 'b', 'backspace', 'c', 'ç', 'd', 'e', 'ə', 'enter', 'f', 'g', 'ğ',
           'h', 'ı', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'ö', 'p', 'q', 'r', 's',
           'ş', ' ', 't', 'u', 'ü', 'v', 'x', 'y', 'z']

LAYERS = ["dense_12", "dense_13", "dense_14", "dense_15"]


def main() -> None:
    blobs, meta_layers = [], []
    with h5py.File(SRC, "r") as f:
        grp = f["model_weights"]
        for name in LAYERS:
            g = grp[name][name]
            W = np.array(g["kernel:0"], dtype=np.float32)   # (giriş, çıxış)
            b = np.array(g["bias:0"], dtype=np.float32)     # (çıxış,)
            meta_layers.append({"name": name, "in": int(W.shape[0]),
                                "out": int(W.shape[1])})
            blobs += [W.ravel(order="C"), b]
            print(f"  {name}: W{W.shape} b{b.shape}")

    flat = np.concatenate(blobs).astype(np.float32)
    (OUT_DIR / "model.bin").write_bytes(flat.tobytes())

    meta = {
        "input": {"frames": 20, "features": 63, "flat": 1260},
        "layers": meta_layers,
        "activations": ["relu", "relu", "relu", "softmax"],
        "labels": ACTIONS,
        "params": int(flat.size),
    }
    (OUT_DIR / "model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"\nmodel.bin  : {flat.size:,} float32 ({flat.nbytes/1048576:.2f} MB)")
    print(f"model.json : {len(ACTIONS)} sinif")


if __name__ == "__main__":
    main()
