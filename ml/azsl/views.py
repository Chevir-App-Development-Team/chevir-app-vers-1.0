"""The side camera (Cam1) of the Sentences recordings as a second view of the same signs.

The annotations belong to the frontal camera (Cam2). Cam1 starts and stops at slightly different
moments, so every recording gets a frame offset `lag`: Cam1 frame i shows Cam2 frame i + lag. It is the
lag at which the arms' vertical movement agrees best in the two views. Image height barely depends on
the camera's horizontal angle, and the signals do not depend on which wrist is which (the pose model
swaps them more often from the side).
"""
import numpy as np

L_EL, R_EL, L_WR, R_WR = 13, 14, 15, 16


def motion(pose):
    """T×33×(3|4) pose (image coordinates) → T×3: mean wrist height, the higher wrist, mean elbow height,
    each minus its 1 s moving average (posture drifts are not movement). NaN where the pose is missing."""
    y = np.asarray(pose, np.float64)[:, :, 1]
    sig = np.stack([(y[:, L_WR] + y[:, R_WR]) / 2, np.minimum(y[:, L_WR], y[:, R_WR]),
                    (y[:, L_EL] + y[:, R_EL]) / 2], 1)
    out = np.full_like(sig, np.nan)
    ok = np.isfinite(sig).all(1)
    if ok.sum() < 2:
        return out
    t = np.arange(len(sig))
    filled = np.stack([np.interp(t, t[ok], sig[ok, c]) for c in range(sig.shape[1])], 1)
    k = 31
    pad = np.pad(filled, ((k // 2, k // 2), (0, 0)), mode='edge')
    smooth = np.stack([np.convolve(pad[:, c], np.ones(k) / k, mode='valid') for c in range(sig.shape[1])], 1)
    out[ok] = (filled - smooth)[ok]
    return out


def _corr(a, b, min_frames):
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    if ok.sum() < min_frames:
        return np.nan
    a, b = a[ok], b[ok]
    a = (a - a.mean(0)) / (a.std(0) + 1e-9)
    b = (b - b.mean(0)) / (b.std(0) + 1e-9)
    return float((a * b).mean())                 # mean over the signals of Pearson's r


def align(front_pose, side_pose, max_lag=45, min_frames=60):
    """→ dict(lag, corr, second, margin): Cam1 (side) frame i ↔ Cam2 (front) frame i + lag.
    corr = agreement at that lag; second = the best agreement more than 5 frames away from it."""
    f, s = motion(front_pose), motion(side_pose)
    rs = {}
    for lag in range(-max_lag, max_lag + 1):
        t0, t1 = max(0, -lag), min(len(s), len(f) - lag)
        if t1 - t0 >= min_frames:
            rs[lag] = _corr(f[t0 + lag:t1 + lag], s[t0:t1], min_frames)
    rs = {k: v for k, v in rs.items() if np.isfinite(v)}
    if not rs:
        return dict(lag=0, corr=np.nan, second=np.nan, margin=np.nan)
    best = max(rs, key=rs.get)
    second = max((v for k, v in rs.items() if abs(k - best) > 5), default=np.nan)
    return dict(lag=int(best), corr=round(rs[best], 4), second=round(second, 4),
                margin=round(rs[best] - second, 4) if np.isfinite(second) else np.nan)


def align_store(side, front, ids=None, log=print):
    """side, front: LandmarkStore of Cam1 / Cam2 → DataFrame (id, lag, corr, second, margin, n_side, n_front)."""
    import pandas as pd
    rows = []
    ids = list(side.index.index if ids is None else ids)
    fi, si = front.index, side.index        # in part order, so each part file is loaded once
    ids.sort(key=lambda r: ((fi.at[r, 'dir'], fi.at[r, 'part']) if r in fi.index else ('', ''),
                            si.at[r, 'dir'], si.at[r, 'part']))
    for n, rid in enumerate(ids):
        row = {'id': rid, 'n_side': int(side.index.at[rid, 'length'])}
        if rid in front.index.index:
            row['n_front'] = int(front.index.at[rid, 'length'])
            row.update(align(front.get(rid)['pose'], side.get(rid)['pose']))
        rows.append(row)
        if (n + 1) % 500 == 0:
            log(f'  aligned {n + 1}/{len(ids)}')
    return pd.DataFrame(rows)
