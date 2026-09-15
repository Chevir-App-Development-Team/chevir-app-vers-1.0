"""
AzSLD barmaq əlifbası şəkillərindən hərf pozaları çıxarır (poses.json).

Nə edir:
  1. Hər hərf qovluğundaki şəkillərə MediaPipe HandLandmarker tətbiq edir
     (brauzerdə işlədilən EYNİ .task modeli — landmark konvensiyası uyğun olsun).
  2. Müəllimin modelini həmin landmark-larla yoxlayır: 20 kadr × 63 giriş
     qurulur, proqnoz düzgün hərfdirsə sayılır → real datada dəqiqlik.
  3. Statik hərf üçün medoid poza seçilir (digərlərinə məsafəsi minimal olan,
     yəni ən tipik nümunə), dinamik hərf üçün ən yaxşı ardıcıllıq seçilir.

Çıxış: public/tercume/assets/poses.json + konsolda yoxlama cədvəli.
"""
import argparse, json, re
from pathlib import Path

import cv2
import numpy as np
import h5py
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import mediapipe as mp

ROOT = Path(__file__).resolve().parent.parent
FS_DIR = ROOT / "data" / "AzSLD_Fingerspelling"   # Zenodo 14222948; --fs-dir ilə dəyişdirilir
TASK = ROOT / "public" / "tercume" / "assets" / "hand_landmarker.task"
MODEL_H5 = ROOT / "model" / "fingerspelling_33.h5"
OUT = ROOT / "public" / "tercume" / "assets" / "poses.json"

FOLDER_TO_LETTER = {
    'A': 'a', 'B': 'b', 'C': 'c', 'Ç': 'ç', 'D': 'd', 'E': 'e', 'Ə': 'ə',
    'F': 'f', 'Ğ': 'ğ', 'G': 'g', 'H': 'h', 'I': 'ı', 'İ': 'i', 'J': 'j',
    'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'O': 'o', 'Ö': 'ö', 'P': 'p',
    'Q': 'q', 'R': 'r', 'S': 's', 'Ş': 'ş', 'T': 't', 'U': 'u', 'Ü': 'ü',
    'V': 'v', 'X': 'x', 'Y': 'y', 'Z': 'z',
}
ACTIONS = ['a','b','backspace','c','ç','d','e','ə','enter','f','g','ğ','h','ı','i','j',
           'k','l','m','n','o','ö','p','q','r','s','ş',' ','t','u','ü','v','x','y','z']
LAYERS = ["dense_12", "dense_13", "dense_14", "dense_15"]
FRAMES, FEAT = 20, 63


def load_model():
    Ws, bs = [], []
    with h5py.File(MODEL_H5, 'r') as f:
        g = f['model_weights']
        for n in LAYERS:
            Ws.append(np.array(g[n][n]['kernel:0'], dtype=np.float32))
            bs.append(np.array(g[n][n]['bias:0'], dtype=np.float32))
    def predict(x):
        h = x.astype(np.float32).reshape(-1)
        for i, (W, b) in enumerate(zip(Ws, bs)):
            h = h @ W + b
            h = np.maximum(h, 0) if i < 3 else h
        e = np.exp(h - h.max())
        return e / e.sum()
    return predict


def frame_num(p: Path) -> int:
    m = re.findall(r'(\d+)', p.stem)
    return int(m[-1]) if m else 0


def detect(landmarker, path: Path):
    """→ (normallaşdırılmış 21×3, metrik dünya 21×3, (W, H), əl etiketi) və ya None.

    Normallaşdırılmış x/y şəklin eninə/hündürlüyünə bölünür, dataset isə qarışıq
    ölçülüdür (720×1280, 960×1280 …) — bucaqlar təhrif olunur. Avatar üçün
    MediaPipe-in metrik dünya koordinatları (metr, əl mərkəzli) də saxlanılır.
    """
    img = cv2.imread(str(path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.hand_landmarks:
        return None
    norm = np.array([[l.x, l.y, l.z] for l in res.hand_landmarks[0]], dtype=np.float32)
    world = np.array([[l.x, l.y, l.z] for l in res.hand_world_landmarks[0]], dtype=np.float32)
    label = res.handedness[0][0].category_name if res.handedness else ''
    return norm, world, (img.shape[1], img.shape[0]), label


def to_input(seq: np.ndarray) -> np.ndarray:
    """(n,21,3) -> (20,63); az olsa təkrarlanır, çox olsa bərabər seyrəklənir."""
    flat = seq.reshape(len(seq), -1)
    if len(flat) == FRAMES:
        return flat
    idx = np.linspace(0, len(flat) - 1, FRAMES).round().astype(int)
    return flat[idx]


def medoid(poses: np.ndarray) -> int:
    """Digərlərinə orta məsafəsi ən kiçik olan pozanın indeksi (bilək mərkəzli)."""
    c = poses - poses[:, :1, :]
    scale = np.linalg.norm(c.reshape(len(c), -1), axis=1, keepdims=True) + 1e-6
    n = c.reshape(len(c), -1) / scale
    d = ((n[:, None, :] - n[None, :, :]) ** 2).sum(-1)
    return int(d.sum(1).argmin())


def clip_runs(kept, poses, sizes, max_gap=20, max_jump=0.25):
    """Aşkarlanmış kadrları ayrı video kliplərinə bölür (indeks massivləri).

    Qovluqda bir neçə klip ardıcıl nömrələnib; kadr nömrələri klip daxilində 1–9
    addımla gedir. Sərhəd: nömrə boşluğu > max_gap, şəkil ölçüsü dəyişir və ya
    bilək bir addımda kadrın dörddə birindən çox sıçrayır.
    """
    runs, cur = [], [0]
    for i in range(1, len(kept)):
        gap = frame_num(kept[i]) - frame_num(kept[i - 1])
        jump = float(np.linalg.norm(poses[i, 0, :2] - poses[i - 1, 0, :2]))
        if gap <= max_gap and sizes[i] == sizes[i - 1] and jump <= max_jump:
            cur.append(i)
        else:
            runs.append(cur); cur = [i]
    runs.append(cur)
    return [np.array(r) for r in runs]


CURL_PAIRS = [(5, 8), (9, 12), (13, 16), (17, 20), (6, 8), (10, 12), (14, 16), (18, 20)]


def curl_evidence(w):
    """Bükülmənin işarəsi (dünya koordinatları): müsbət → sağ əl həndəsəsi (retarget.js ilə eyni)."""
    d = w[9] - w[0]; d = d / np.linalg.norm(d)
    s = w[5] - w[17]; s = s / np.linalg.norm(s)
    n = np.cross(s, d); n = n / np.linalg.norm(n)
    size = float(np.linalg.norm(w[9] - w[0])) or 1.0
    return float(sum(np.dot(w[t] - w[m], n) for m, t in CURL_PAIRS) + 2 * np.dot(w[4] - w[1], n)) / size


def palm_basis(w, right):
    """Ovuc çərçivəsi (sütunlar: barmaq istiqaməti, ovuc normalı, yan ox)."""
    d = w[9] - w[0]; d = d / np.linalg.norm(d)
    s = w[5] - w[17]; s = s / np.linalg.norm(s)
    n = np.cross(s, d) if right else np.cross(d, s)
    n = n - np.dot(n, d) * d; n = n / np.linalg.norm(n)
    return np.stack([d, n, np.cross(d, n)], axis=1)


def jitter(ws, right, limit_deg=60):
    """Ardıcıl kadrlar arasında ovucun `limit_deg`-dən çox döndüyü keçidlərin payı.

    Oynatma ~14 kadr/s-dir: 70 ms-də 60°+ dönmə real hərəkət deyil, MediaPipe
    səs-küyü və ya dərinliyi güzgü həll edilmiş kadrdır.
    """
    Rs = [palm_basis(w, right) for w in ws]
    big = sum(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))) > limit_deg
              for A, B in zip(Rs, Rs[1:]))
    return big / max(len(Rs) - 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-static', type=int, default=40)
    ap.add_argument('--dyn-frames', type=int, default=14)
    ap.add_argument('--fs-dir', type=Path, default=FS_DIR, help='AzSLD_Fingerspelling qovluğu')
    args = ap.parse_args()

    predict = load_model()
    lm = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(TASK)),
        running_mode=RunningMode.IMAGE, num_hands=1,
        min_hand_detection_confidence=0.4,
    ))

    letters, report = {}, []
    for folder in sorted(FOLDER_TO_LETTER, key=lambda f: FOLDER_TO_LETTER[f]):
        d = args.fs_dir / folder
        if not d.is_dir():
            continue
        letter = FOLDER_TO_LETTER[folder]
        files = sorted(d.iterdir(), key=frame_num)
        dynamic = len(files) > 200
        # Dinamik hərf seyrəldilmir: hərəkət üçün ardıcıl kadrlar lazımdır. Əvvəl
        # seyrəltmə klipləri dağıdırdı və 12 kadr müxtəlif videolardan gəlirdi.
        if not dynamic and len(files) > args.max_static:
            files = [files[i] for i in np.linspace(0, len(files) - 1, args.max_static).round().astype(int)]

        poses, worlds, sizes, labels, kept = [], [], [], [], []
        for p in files:
            r = detect(lm, p)
            if r is not None:
                poses.append(r[0]); worlds.append(r[1]); sizes.append(r[2]); labels.append(r[3])
                kept.append(p)
        if not poses:
            report.append((letter, len(files), 0, 0.0, 'əl aşkarlanmadı'))
            continue
        poses, worlds = np.stack(poses), np.stack(worlds)

        # --- modeli real datada yoxla: hər kadrı 20 dəfə təkrarlayıb proqnoz ---
        target = ACTIONS.index(letter)
        hits = 0
        for k in poses:
            p = predict(np.repeat(k.reshape(1, -1), FRAMES, axis=0))
            if int(p.argmax()) == target:
                hits += 1
        acc = hits / len(poses)

        if dynamic:
            # Tək bir klip seçilir: model onu yaxşı tanısın, əlin tərəfi kadrlar boyu
            # ardıcıl olsun və oynadılacaq kadrlarda ovuc titrəməsin
            runs = clip_runs(kept, poses, sizes)
            pick = lambda ids: ids[np.linspace(0, len(ids) - 1, min(args.dyn_frames, len(ids))).round().astype(int)]
            best, best_key, best_info = None, None, (0.0, 0.0, 0.0)
            for ids in runs:
                if len(ids) < args.dyn_frames:
                    continue
                score = float(predict(to_input(poses[ids]))[target])
                ev = np.array([curl_evidence(worlds[i]) for i in ids])
                right = ev.sum() > 0
                consistent = float(((ev > 0) == right).mean())
                jit = jitter([worlds[i] for i in pick(ids)], right)
                key = score + 0.25 * consistent - 0.5 * jit
                if best_key is None or key > best_key:
                    best, best_key, best_info = ids, key, (score, consistent, jit)
            if best is None:
                best = max(runs, key=len)
                best_info = (float(predict(to_input(poses[best]))[target]), 0.0, 0.0)
            sel = pick(best)
            kind, conf = 'dynamic', best_info[0]
            print(f"     klip: {len(runs)} klipdən seçildi, {len(best)} kadr "
                  f"({kept[best[0]].name} … {kept[best[-1]].name}), "
                  f"tərəf ardıcıllığı {best_info[1]:.0%}, titrəmə {best_info[2]:.0%}", flush=True)
        else:
            mi = medoid(poses)
            sel = np.array([mi])
            conf = float(predict(np.repeat(poses[mi].reshape(1, -1), FRAMES, axis=0))[target])
            kind = 'static'
        frames = poses[sel]

        letters[letter] = {
            'type': kind,
            'samples': int(len(poses)),
            'modelConfidence': round(conf, 4),
            'frames': [[{'x': round(float(x), 5), 'y': round(float(y), 5),
                         'z': round(float(z), 5)} for x, y, z in f] for f in frames],
            # avatar üçün: metrik 3D (metr), mənbə şəklin ölçüsü, MediaPipe əl etiketi
            'world': [[[round(float(v), 5) for v in pt] for pt in worlds[i]] for i in sel],
            'size': [list(sizes[i]) for i in sel],
            'handedness': [labels[i] for i in sel],
            'files': [kept[i].name for i in sel],   # mənbə şəkil (yoxlama üçün)
        }
        report.append((letter, len(files), len(poses), acc, f'{kind} conf={conf:.3f}'))
        print(f"  {letter}  {kind:8s} şəkil={len(files):4d} əl={len(poses):4d} "
              f"model-acc={acc*100:5.1f}%  conf={conf:.3f}", flush=True)

    OUT.write_text(json.dumps({
        'meta': {
            'source': 'AzSLD Fingerspelling (Zenodo 14222948, CC BY 4.0)',
            'landmarker': 'MediaPipe hand_landmarker.task (brauzerlə eyni)',
            'hand': 'right',
            'letters': len(letters),
        },
        'letters': letters,
    }, ensure_ascii=False), encoding='utf8')

    ok = [r for r in report if r[2] > 0]
    accs = [r[3] for r in ok]
    print(f"\n{'='*64}")
    print(f"poses.json: {len(letters)} hərf, {OUT.stat().st_size/1024:.0f} KB")
    print(f"MODELİN REAL AzSLD DATASINDA DƏQİQLİYİ (kadr-səviyyə, top-1):")
    print(f"  orta: {np.mean(accs)*100:.1f}%   median: {np.median(accs)*100:.1f}%")
    print(f"  ən yaxşı: " + ", ".join(f"{l}={a*100:.0f}%" for l,_,_,a,_ in sorted(ok,key=lambda r:-r[3])[:6]))
    print(f"  ən zəif : " + ", ".join(f"{l}={a*100:.0f}%" for l,_,_,a,_ in sorted(ok,key=lambda r:r[3])[:6]))


if __name__ == '__main__':
    main()
