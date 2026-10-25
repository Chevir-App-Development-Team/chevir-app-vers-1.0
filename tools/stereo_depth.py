"""
How far in front of the body are the hands? The side camera checks MediaPipe's single-view depth.

AzSLD films every sentence from the front (Cam2, the view the avatar's words come from) and from the
signer's right, above (Cam1). The same moment seen from both must give the same wrist position in a
frame built from the body itself (shoulder axis; "up" from the shoulders toward the ears; forward = axis
× up). From the side, the distance of the hands in front of the body lies in the image plane, so a model
that perceives depth gives the same answer from both views; one that guesses does not.

Triangulating the two views directly was tried and does not work on this data: the cameras are not
frame-locked (hands move 10-30 px between the views' frames, the audio tracks are silent), their focal
lengths are unknown and barely observable when both look at one spot, and the slow body points (nose,
ears, shoulders) are too close together to fix the geometry where the hands are (ml/EXPERIMENTS.md E11).

    .venv/Scripts/python.exe -I tools/stereo_depth.py extract data/azsld --sessions 12 --per 10 --out stereo2
    .venv/Scripts/python.exe -I tools/stereo_depth.py crossview data/azsld --out stereo2

Needs data/models/pose_landmarker_heavy.task (float16/1). Results: ml/EXPERIMENTS.md E11.
"""
import argparse
import csv
import json
import random
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (HandLandmarker, HandLandmarkerOptions, PoseLandmarker,
                                           PoseLandmarkerOptions, RunningMode)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'ml'))
from azsld_zip import AzsldZip                       # noqa: E402

HAND = ROOT / 'public' / 'tercume' / 'assets' / 'hand_landmarker.task'
NOSE, L_EAR, R_EAR, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 7, 8, 11, 12, 13, 14, 15, 16


# ───────────────────────── extract ─────────────────────────

def video(z, rec, cam):
    folder, stamp = rec.split('/', 1)
    return f'AzSLD_Sentences/{folder}/{cam}/{stamp}.mp4'


def landmarks(path, models, world):
    """VIDEO-mode pose (each model) + hands for every frame. Pose image coords in pixels."""
    cap = cv2.VideoCapture(str(path))
    W, H = int(cap.get(3)), int(cap.get(4))
    poses = {m: PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(p)), running_mode=RunningMode.VIDEO, num_poses=1))
        for m, p in models.items()}
    hand = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND)), running_mode=RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.4))
    out = defaultdict(list)
    k = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        im = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for m, pose in poses.items():
            r = pose.detect_for_video(im, k * 33)
            if r.pose_landmarks:
                out[f'{m}_img'].append([[p.x * W, p.y * H, p.visibility] for p in r.pose_landmarks[0]])
                out[f'{m}_world'].append([[p.x, p.y, p.z] for p in r.pose_world_landmarks[0]])
            else:
                out[f'{m}_img'].append(np.full((33, 3), np.nan))
                out[f'{m}_world'].append(np.full((33, 3), np.nan))
        r = hand.detect_for_video(im, k * 33)
        out['hands'].append([[[p.x * W, p.y * H] for p in hl] for hl in r.hand_landmarks][:2])
        k += 1
    cap.release()
    for p in poses.values():
        p.close()
    hand.close()
    res = {key: np.asarray(v, np.float32) for key, v in out.items() if key != 'hands' and (world or 'img' in key)}
    # hands → slots by the nearest pose wrist of the first model (0 = signer's left, 1 = right)
    ref = res[f'{next(iter(models))}_img']
    hs = np.full((k, 2, 21, 2), np.nan, np.float32)
    for t, hl in enumerate(out['hands']):
        if not hl or not np.isfinite(ref[t]).all():
            continue
        wr = ref[t][[L_WR, R_WR], :2]
        d = [[np.linalg.norm(np.asarray(h)[0] - wr[s]) for s in (0, 1)] for h in hl]
        if len(hl) == 2:
            pair = (0, 1) if d[0][0] + d[1][1] <= d[0][1] + d[1][0] else (1, 0)
            for i, s in enumerate(pair):
                hs[t, s] = hl[i]
        else:
            hs[t, int(np.argmin(d[0]))] = hl[0]
    res['hands'] = hs
    res['size'] = np.array([W, H])
    return res


def extract_one(job):
    zip_path, rec, out, models = job
    out = Path(out)
    if out.exists():
        return rec
    z = AzsldZip.__new__(AzsldZip)
    z.path, z.pieces = Path(zip_path), Path(zip_path + '.pieces')
    z.ents = json.loads(Path(zip_path + '.ents.json').read_text())
    arrays = {}
    for cam in ('Cam2', 'Cam1'):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=out.parent) as tmp:
            tmp.write(z.read(video(z, rec, cam)))
        try:
            for key, v in landmarks(tmp.name, models, world=True).items():
                arrays[f'{cam}_{key}'] = v
        finally:
            Path(tmp.name).unlink()
    np.savez_compressed(out, **arrays)
    return rec


def extract(args):
    rows = list(csv.DictReader(open(ROOT / 'ml' / 'splits' / 'sentences_split.csv', encoding='utf8')))
    z = AzsldZip(args.data / 'AzSLD_Sentences.zip')
    Path(str(z.path) + '.ents.json').write_text(json.dumps(z.ents))
    by = defaultdict(list)
    for r in rows:
        if video(z, r['rec_id'], 'Cam1') in z.ents and video(z, r['rec_id'], 'Cam2') in z.ents:
            by[r['session']].append(r['rec_id'])
    rng = random.Random(0)
    sessions = sorted(s for s, v in by.items() if len(v) >= 20)
    pick = rng.sample(sessions, min(args.sessions, len(sessions)))
    out_dir = args.data / args.out
    out_dir.mkdir(exist_ok=True)
    models = {'heavy': str(args.data.parent / 'models' / 'pose_landmarker_heavy.task'),
              'lite': str(ROOT / 'public' / 'tercume' / 'assets' / 'pose_landmarker_lite.task')}
    jobs = []
    plan = {}
    for s in pick:
        recs = rng.sample(by[s], args.per)
        plan[s] = recs
        for rec in recs:
            jobs.append((str(z.path), rec, str(out_dir / (rec.replace('/', '__') + '.npz')), models))
    (out_dir / 'plan.json').write_text(json.dumps(plan, indent=1))
    print(f'{len(pick)} sessions × {args.per} recordings, both cameras', flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for n, _ in enumerate(ex.map(extract_one, jobs), 1):
            if n % 20 == 0:
                print(f'  {n}/{len(jobs)}', flush=True)



# ───────────────────────── helpers ─────────────────────────

def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def lag_of(d):
    from azsl.views import align
    return align(d['Cam2_heavy_img'][:, :, :3], d['Cam1_heavy_img'][:, :, :3])


def matched(d, lag):
    """Cam1 frame i shows Cam2 frame i + lag."""
    t1, t2 = len(d['Cam1_heavy_img']), len(d['Cam2_heavy_img'])
    i = np.arange(max(0, -lag), min(t1, t2 - lag))
    return i, i + lag


# ───────────────────────── cross-view ─────────────────────────
#
# Without calibration: the same moment seen from the front and from the side must give the same wrist
# position in a frame built from the body itself (shoulder axis; "up" from the shoulders toward the ears;
# forward = axis × up). From the side, the distance of the hands in front of the body lies in the image
# plane, so the side view's estimate of it depends far less on guessing depth than the front view's.

def body_frame_head(W):
    """MediaPipe world (x right, y down, z away) → point → (lateral, height, forward) / shoulder width."""
    P = np.asarray(W, float) * np.array([1, -1, -1])
    mid = (P[L_SH] + P[R_SH]) / 2
    ax = P[L_SH] - P[R_SH]
    sw = np.linalg.norm(ax)
    ax = ax / sw
    head = (P[L_EAR] + P[R_EAR]) / 2 - mid
    up = head - (head @ ax) * ax
    up = up / np.linalg.norm(up)
    fwd = np.cross(ax, up)
    return lambda q: np.array([(q * [1, -1, -1] - mid) @ ax, (q * [1, -1, -1] - mid) @ up,
                               (q * [1, -1, -1] - mid) @ fwd]) / sw


def crossview(args):
    out_dir = args.data / args.out
    plan = json.loads((out_dir / 'plan.json').read_text())
    need = [L_SH, R_SH, L_EAR, R_EAR]
    rows = []                         # model, view-pair stats per wrist: front (lat, h, fwd), side (lat, h, fwd)
    for ses, recs in plan.items():
        for rec in recs:
            p = out_dir / (rec.replace('/', '__') + '.npz')
            if not p.exists():
                continue
            d = load(p)
            al = lag_of(d)
            if not (al['corr'] >= 0.3 and al['margin'] >= 0.1):
                continue
            i1, i2 = matched(d, al['lag'])
            for m in ('lite', 'heavy'):
                F, S = d[f'Cam2_{m}_world'][i2], d[f'Cam1_{m}_world'][i1]
                vF, vS = d[f'Cam2_{m}_img'][i2][:, :, 2], d[f'Cam1_{m}_img'][i1][:, :, 2]
                for t in range(len(i2)):
                    if not (np.isfinite(F[t]).all() and np.isfinite(S[t]).all()):
                        continue
                    if (vF[t, need] < 0.5).any() or (vS[t, need] < 0.5).any():
                        continue
                    bf, bs = body_frame_head(F[t]), body_frame_head(S[t])
                    for wr in (L_WR, R_WR):
                        if vF[t, wr] < 0.5 or vS[t, wr] < 0.5:
                            continue
                        rows.append((m, ses, *bf(F[t][wr]), *bs(S[t][wr])))
    cols = ['f_lat', 'f_h', 'f_fwd', 's_lat', 's_h', 's_fwd']
    with open(out_dir / 'crossview.csv', 'w', encoding='utf8') as fh:
        fh.write('model,session,' + ','.join(cols) + chr(10))
        for r in rows:
            fh.write(f'{r[0]},{r[1]},' + ','.join(f'{v:.4f}' for v in r[2:]) + chr(10))
    data = {m: np.asarray([r[2:] for r in rows if r[0] == m], float) for m in ('lite', 'heavy')}
    print(f"{len(data['lite'])} / {len(data['heavy'])} wrist positions (lite / heavy), "
          f"{len({r[1] for r in rows})} sessions")
    print('Wrist in shoulder widths (median [p10, p90]); front = Cam2, side = Cam1, same moments.')
    q = lambda a: f'{np.median(a):5.2f} [{np.percentile(a, 10):5.2f},{np.percentile(a, 90):5.2f}]'
    for m, x in data.items():
        for k, nm in ((2, 'forward'), (1, 'height'), (0, 'lateral')):
            a, b = x[:, k], x[:, 3 + k]
            print(f'  {m:5s} {nm:8s} front {q(a)}  side {q(b)}  |front - side| {np.median(np.abs(a - b)):.2f}'
                  f'  r {np.corrcoef(a, b)[0, 1]:.2f}')
    # the same moments in all four estimates: does each one track the others?
    joint = []
    for ses, recs in plan.items():
        for rec in recs:
            p = out_dir / (rec.replace('/', '__') + '.npz')
            if not p.exists():
                continue
            d = load(p)
            al = lag_of(d)
            if not (al['corr'] >= 0.3 and al['margin'] >= 0.1):
                continue
            i1, i2 = matched(d, al['lag'])
            keys = [(m, c, i) for m in ('lite', 'heavy') for c, i in (('Cam2', i2), ('Cam1', i1))]
            W = [d[f'{c}_{m}_world'][i] for m, c, i in keys]
            V = [d[f'{c}_{m}_img'][i][:, :, 2] for m, c, i in keys]
            for t in range(len(i2)):
                if not all(np.isfinite(w[t]).all() and (v[t, need] >= 0.5).all() for w, v in zip(W, V)):
                    continue
                for wr in (L_WR, R_WR):
                    if all(v[t, wr] >= 0.5 for v in V):
                        joint.append([body_frame_head(w[t])(w[t][wr])[2] for w in W])
    a = np.asarray(joint)
    names = ['lite front', 'lite side', 'heavy front', 'heavy side']
    print(f'Forward depth, the {len(a)} wrist moments seen in all four: correlation')
    print(' ' * 12 + ''.join(f'{k:>12s}' for k in names))
    for i, k in enumerate(names):
        print(f'{k:12s}' + ''.join(f'{np.corrcoef(a[:, i], a[:, j])[0, 1]:12.2f}' for j in range(4)))


# ───────────────────────── main ─────────────────────────

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['extract', 'crossview'])
    ap.add_argument('data', type=Path)
    ap.add_argument('--sessions', type=int, default=12)
    ap.add_argument('--per', type=int, default=10)
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--out', default='stereo', help='folder in data/ for the landmarks')
    args = ap.parse_args()
    {'extract': extract, 'crossview': crossview}[args.stage](args)


if __name__ == '__main__':
    main()
