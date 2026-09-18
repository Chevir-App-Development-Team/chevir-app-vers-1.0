"""
AzSLD Words 200 videolarından avatar üçün söz işarələri çıxarır.

Hər söz üçün bir neçə namizəd video emal olunur və ən təmizi seçilir:
  - MediaPipe Pose (dünya koordinatları, metr): çiyinlər, dirsəklər, biləklər,
    budlar — əlin BƏDƏNƏ GÖRƏ yeri (yanaq, çənə, sinə, yan) buradan gəlir
  - MediaPipe Hands (2 əl): normallaşdırılmış + dünya landmark-ları — əlin
    forması və oriyentasiyası (hərf pozaları ilə eyni format)
  - sağ/sol əl pozanın bilək nöqtələrinə yaxınlığa görə təyin olunur: ekrandakı
    x mövqeyi əllər çarpazlaşanda və ya biri itəndə yanılır

Çıxış: public/tercume/words/index.json + hər söz üçün <id>.json (brauzer lazım
olanda yükləyir). Mənbə: AzSLD (Zenodo 14222948, CC BY 4.0).

İstifadə:
  .venv/bin/python tools/build_words.py \\
      --words-dir data/AzSLD_Words_200 --pose-model data/models/pose_landmarker_lite.task
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, PoseLandmarker, PoseLandmarkerOptions, RunningMode,
)

ROOT = Path(__file__).resolve().parent.parent
HAND_MODEL = ROOT / 'public' / 'tercume' / 'assets' / 'hand_landmarker.task'
OUT_DIR = ROOT / 'public' / 'tercume' / 'words'

# Saxlanan bədən nöqtələri (MediaPipe Pose indeksləri), sıra JS ilə eynidir
BODY = [0, 11, 12, 13, 14, 15, 16, 23, 24]   # burun, çiyin L/R, dirsək L/R, bilək L/R, bud L/R
L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP = 11, 12, 15, 16, 23, 24

# Tək hərfli etiketlər barmaq əlifbası ilə qarışır ("o" əvəzliyi və rəqəmlər qalır)
SKIP = {'A', 'D', 'Ə', 'İ', 'J', 'M', 'N', 'S'}
# Datasetin etiketlərindəki yazı xətaları (axtarış və mətndə tanınma düzgün yazılışla işləsin)
ALIAS = {
    'GUN': 'gün', 'IKI': 'iki', 'İŞİGİ': 'işığı', 'SÖNDURƏ': 'söndürə', 'ÜNVANLİ': 'ünvanlı',
    'XİRDA': 'xırda', 'RƏSİM': 'rəsm', 'EŞİTMƏ MƏHDÜDİYYƏTLİ': 'eşitmə məhdudiyyətli',
}

TRANSLIT = str.maketrans({'ə': 'e', 'ı': 'i', 'ö': 'o', 'ü': 'u', 'ş': 's', 'ç': 'c', 'ğ': 'g', ' ': '-'})


def az_lower(s: str) -> str:
    """Azərbaycan qaydası ilə kiçik hərf: I → ı, İ → i (Python-un lower() səhv edir)."""
    return s.replace('I', 'ı').replace('İ', 'i').lower()


def word_id(text: str) -> str:
    return text.translate(TRANSLIT)


_hand = _pose = None


def _init():
    global _hand, _pose
    _hand = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=RunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=0.4,
    ))
    _pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ARGS.pose_model)),
        running_mode=RunningMode.IMAGE, num_poses=1, min_pose_detection_confidence=0.5,
    ))


def _assign(hands, img_pts, aspect):
    """Aşkarlanan əlləri pozanın biləklərinə görə L/R-ə bölür."""
    out = {'L': None, 'R': None}
    if not hands:
        return out
    if img_pts is None:
        # Poza yoxdursa: kameraya baxan adamın sağ əli kadrın solundadır
        for h in sorted(hands, key=lambda h: h['n'][0][0])[:2]:
            out['R' if out['R'] is None else 'L'] = h
        return out

    def dist(h, j):
        (x, y), (px, py) = h['n'][0][:2], img_pts[j]
        return np.hypot((x - px) * aspect, y - py)

    if len(hands) >= 2:
        a, b = hands[:2]
        if dist(a, L_WR) + dist(b, R_WR) <= dist(a, R_WR) + dist(b, L_WR):
            out['L'], out['R'] = a, b
        else:
            out['L'], out['R'] = b, a
    else:
        h = hands[0]
        out['L' if dist(h, L_WR) < dist(h, R_WR) else 'R'] = h
    return out


def process_video(path: Path, stride: int):
    cap = cv2.VideoCapture(str(path))
    W, H = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5) or 30
    aspect = W / H
    frames, i = [], -1
    while True:
        ok, img = cap.read()
        if not ok:
            break
        i += 1
        if i % stride:
            continue
        im = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pr, hr = _pose.detect(im), _hand.detect(im)

        body = up = img_pts = None
        if pr.pose_world_landmarks:
            wl, il = pr.pose_world_landmarks[0], pr.pose_landmarks[0]
            body = [[round(wl[j].x, 4), round(wl[j].y, 4), round(wl[j].z, 4)] for j in BODY]
            img_pts = {j: (il[j].x, il[j].y) for j in (L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP)}
            # Əl "işarə məkanındadır": bilək gövdənin aşağı dörddə birindən yuxarıdadır
            sh_y = (il[L_SH].y + il[R_SH].y) / 2
            hip_y = (il[L_HIP].y + il[R_HIP].y) / 2
            limit = sh_y + 0.75 * (hip_y - sh_y)
            up = [il[L_WR].y < limit, il[R_WR].y < limit]

        hands = []
        for k, hl in enumerate(hr.hand_landmarks or []):
            hands.append({
                'n': [[round(p.x, 4), round(p.y, 4), round(p.z, 4)] for p in hl],
                'w': [[round(p.x, 4), round(p.y, 4), round(p.z, 4)] for p in hr.hand_world_landmarks[k]],
            })
        sides = _assign(hands, img_pts, aspect)
        frames.append({'b': body, 'u': up, 'L': sides['L'], 'R': sides['R']})
    cap.release()
    return {'frames': frames, 'size': [W, H], 'fps': round(fps / stride, 2)}


def trim_and_score(rec):
    """Aktiv hissəni kəsir, əlin görünmədiyi kadrlarda formanı qonşudan götürür, keyfiyyət verir."""
    fr = rec['frames']
    n = len(fr)
    if not n:
        return None, 0.0
    pose_ok = [f['b'] is not None for f in fr]
    active = [bool(f['u'] and any(f['u'])) for f in fr]
    if not any(active):
        return None, 0.0
    a = [k for k, v in enumerate(active) if v]
    lo, hi = max(0, a[0] - 2), min(n, a[-1] + 3)
    fr = fr[lo:hi]

    # Hər tərəf üçün: işarə məkanında olduğu kadrlarda əl nə qədər tapılıb
    cover, used = [], ''
    for s, j in (('L', 0), ('R', 1)):
        up_idx = [k for k, f in enumerate(fr) if f['u'] and f['u'][j]]
        if len(up_idx) < 2:
            for f in fr:
                f[s] = None             # əl aşağıdadır — işarədə iştirak etmir
            continue
        used += s
        found = [k for k in up_idx if fr[k][s] is not None]
        cover.append(len(found) / len(up_idx))
        if not found:
            continue
        last = fr[found[0]][s]         # başdakı boşluqlar ilk tapılan forma ilə dolur
        for k, f in enumerate(fr):
            if f[s] is not None:
                last = f[s]
            elif f['u'] and f['u'][j]:
                f[s] = last
            else:
                f[s] = None
    if not used:
        return None, 0.0

    body_last = next((f['b'] for f in fr if f['b']), None)
    for f in fr:                        # pozanın qısa itkisi — əvvəlki kadr
        if f['b'] is None:
            f['b'] = body_last
        body_last = f['b']
    score = (sum(pose_ok) / n) * (sum(cover) / len(cover)) * min(1.0, len(a) / 6)
    rec = {**rec, 'frames': fr, 'hands': used}
    return rec, score


def build_word(label: str):
    folder = ARGS.words_dir / label
    videos = sorted(p for p in folder.iterdir() if p.suffix == '.mp4')[:ARGS.candidates]
    best, best_score = None, -1.0
    for v in videos:
        rec, score = trim_and_score(process_video(v, ARGS.stride))
        if rec and score > best_score:
            best, best_score = {**rec, 'video': v.name}, score
    return label, best, best_score, len(videos)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob('*.json'):
        old.unlink()
    labels = sorted(p.name for p in ARGS.words_dir.iterdir() if p.is_dir() and p.name not in SKIP)
    if ARGS.only:
        labels = [l for l in labels if az_lower(l) in ARGS.only]
    print(f'{len(labels)} söz, hər biri üçün ≤{ARGS.candidates} namizəd video', flush=True)

    index, dropped = [], []
    with ProcessPoolExecutor(ARGS.workers, initializer=_init) as ex:
        for label, rec, score, nvid in ex.map(build_word, labels):
            text = ALIAS.get(label, az_lower(label))
            if not rec or score < ARGS.min_score:
                dropped.append((text, round(score, 2)))
                print(f'  ✗ {text:<22} keyfiyyət {score:.2f}', flush=True)
                continue
            wid = word_id(text)
            data = {'id': wid, 'text': text, 'label': label, **rec}
            (OUT_DIR / f'{wid}.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')),
                                                  encoding='utf8')
            index.append({'id': wid, 'text': text, 'hands': rec['hands'], 'frames': len(rec['frames'])})
            print(f'  ✓ {text:<22} {rec["hands"]:<2} {len(rec["frames"]):>3} kadr  keyfiyyət {score:.2f}  ({rec["video"]})',
                  flush=True)

    index.sort(key=lambda w: w['text'])
    (OUT_DIR / 'index.json').write_text(json.dumps({
        'source': 'AzSLD Words 200 (Zenodo 14222948, CC BY 4.0)',
        'count': len(index),
        'words': index,
    }, ensure_ascii=False, separators=(',', ':')), encoding='utf8')
    total = sum(p.stat().st_size for p in OUT_DIR.glob('*.json'))
    print(f'\n{len(index)} söz yazıldı ({total / 1024:.0f} KB), {len(dropped)} söz atıldı: {dropped}')


ap = argparse.ArgumentParser()
ap.add_argument('--words-dir', type=Path, default=ROOT / 'data' / 'AzSLD_Words_200')
ap.add_argument('--pose-model', type=Path, default=ROOT / 'data' / 'models' / 'pose_landmarker_lite.task')
ap.add_argument('--candidates', type=int, default=6)
ap.add_argument('--stride', type=int, default=2, help='30 kadr/s videodan hər neçə kadrdan biri (2 → 15 kadr/s)')
ap.add_argument('--min-score', type=float, default=0.45)
ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--only', nargs='*', help='yalnız bu sözlər (sınaq üçün)')
ARGS = ap.parse_args()

if __name__ == '__main__':
    main()
