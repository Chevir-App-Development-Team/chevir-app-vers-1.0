"""Which MediaPipe pose model gives the avatar the best arms? lite / full / heavy, IMAGE / VIDEO mode.

On real AzSLD sign instances (Cam2), per frame:
  wrist error  pixel distance between the pose wrist and the hand landmarker's wrist (the hand model
               finds the wrist precisely; the avatar's arm is built from the pose wrist and elbow)
  jitter       mean |second difference| of the world wrist and elbow (mm / frame^2)
  missing      frames without a pose
    .venv/Scripts/python.exe -I tools/compare_pose_models.py data/azsld 40   (EXPERIMENTS.md E10)
"""
import csv
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (HandLandmarker, HandLandmarkerOptions, PoseLandmarker,
                                           PoseLandmarkerOptions, RunningMode)

from azsld_zip import AzsldZip, cam2

DATA = Path(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
MODELS = DATA.parent / 'models'      # pose_landmarker_{lite,full,heavy}.task (float16/1)
HAND = ROOT / 'public' / 'tercume' / 'assets' / 'hand_landmarker.task'
CONFIGS = [(m, mode) for m in ('lite', 'full', 'heavy') for mode in ('IMAGE', 'VIDEO')]
MARGIN = 8
WR = {'R': 16, 'L': 15}       # pose wrists; signer's right = MediaPipe "Left" on an unmirrored image
ELB = {'R': 14, 'L': 13}


def frames_of(path, a, b):
    cap = cv2.VideoCapture(str(path))
    out, i = [], 0
    while True:
        ok, img = cap.read()
        if not ok or i > b:
            break
        if i >= a:
            out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return out


def run_clip(job):
    path, a, b = job
    frames = frames_of(path, a, b)
    if len(frames) < 10:
        return None
    H, W = frames[0].shape[:2]
    hand = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND)), running_mode=RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))
    ref = []                                   # per frame: {side: (x, y) px}
    for k, f in enumerate(frames):
        r = hand.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=f), k * 33)
        d = {}
        for lm, hd in zip(r.hand_landmarks, r.handedness):
            side = 'R' if hd[0].category_name == 'Left' else 'L'
            d[side] = (lm[0].x * W, lm[0].y * H)
        ref.append(d)
    hand.close()
    res = {}
    for model, mode in CONFIGS:
        pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / f'pose_landmarker_{model}.task')),
            running_mode=getattr(RunningMode, mode), num_poses=1))
        t0 = time.perf_counter()
        img, world = [], []
        for k, f in enumerate(frames):
            im = mp.Image(image_format=mp.ImageFormat.SRGB, data=f)
            r = pose.detect(im) if mode == 'IMAGE' else pose.detect_for_video(im, k * 33)
            if r.pose_landmarks:
                il, wl = r.pose_landmarks[0], r.pose_world_landmarks[0]
                img.append({s: (il[j].x * W, il[j].y * H) for s, j in WR.items()})
                world.append(np.array([[wl[j].x, wl[j].y, wl[j].z] for j in (13, 14, 15, 16)]))
            else:
                img.append(None)
                world.append(None)
        ms = (time.perf_counter() - t0) / len(frames) * 1000
        pose.close()
        # Tərəf etiketinə yox, ən yaxın tərəfə görə (iki əl varsa, ən yaxşı cütləşmə)
        errs = []
        for k in range(len(frames)):
            if not img[k] or not ref[k]:
                continue
            hs = list(ref[k].values())
            d = lambda h, s: np.hypot(*np.subtract(img[k][s], h))
            if len(hs) == 2:
                a1 = [d(hs[0], 'R'), d(hs[1], 'L')]
                a2 = [d(hs[0], 'L'), d(hs[1], 'R')]
                errs += a1 if sum(a1) <= sum(a2) else a2
            else:
                errs.append(min(d(hs[0], 'R'), d(hs[0], 'L')))
        jit = [np.linalg.norm(world[k + 1] - 2 * world[k] + world[k - 1], axis=1).mean() * 1000
               for k in range(1, len(frames) - 1) if all(world[j] is not None for j in (k - 1, k, k + 1))]
        res[(model, mode)] = dict(err=errs, jit=jit, miss=sum(x is None for x in img), n=len(frames), ms=ms)
    return res


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    z = AzsldZip(DATA / 'AzSLD_Sentences.zip')
    rows = [r for r in csv.DictReader(open(ROOT / 'ml/splits/glosses_split.csv', encoding='utf8'))
            if r['start'] and r['dup'] == 'False' and len(r['gloss']) > 1]
    rows = [r for r in rows if 15 <= float(r['end']) - float(r['start']) + 1 <= 45
            and z.available(cam2(r['rec_id']))]
    rng = random.Random(0)
    by_signer = {}
    for r in rng.sample(rows, len(rows)):
        by_signer.setdefault(r['signer'], []).append(r)
    pick = []
    while len(pick) < N and any(by_signer.values()):
        for s in list(by_signer):
            if by_signer[s] and len(pick) < N:
                pick.append(by_signer[s].pop())
    tmp = DATA / 'tmp_videos'
    tmp.mkdir(exist_ok=True)
    jobs = []
    for r in pick:
        p = tmp / cam2(r['rec_id']).replace('/', '__')
        if not p.exists():
            p.write_bytes(z.read(cam2(r['rec_id'])))
        jobs.append((p, max(0, int(float(r['start'])) - 1 - MARGIN), int(float(r['end'])) - 1 + MARGIN))
    print(f'{len(jobs)} sign instances, {len({r["signer"] for r in pick})} signers, '
          f'{len({r["gloss"] for r in pick})} glosses', flush=True)
    agg = {c: dict(err=[], jit=[], miss=0, n=0, ms=[]) for c in CONFIGS}
    with ProcessPoolExecutor(8) as ex:
        for res in ex.map(run_clip, jobs):
            if not res:
                continue
            for c, v in res.items():
                agg[c]['err'] += v['err']; agg[c]['jit'] += v['jit']
                agg[c]['miss'] += v['miss']; agg[c]['n'] += v['n']; agg[c]['ms'].append(v['ms'])
    print(f'\n{"model":<6} {"mode":<6} {"wrist err px (median / p90)":>28} {"jitter mm":>10} {"missing":>8} {"ms/frame":>9}')
    for (m, mode), v in agg.items():
        e = np.array(v['err'])
        print(f'{m:<6} {mode:<6} {np.median(e):>14.1f} / {np.percentile(e, 90):>9.1f} {np.mean(v["jit"]):>10.1f}'
              f' {v["miss"] / v["n"]:>8.3f} {np.mean(v["ms"]):>9.1f}')


if __name__ == '__main__':
    main()
