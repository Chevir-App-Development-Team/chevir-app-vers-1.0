"""Landmarks → per-frame model input (numpy only, so the browser can mirror it in JS).

Raw (from the extraction, see landmarks.py), reduced to what the model reads:
  hand        T×2×21×3   image coords (slot 0 = signer's right hand)
  hand_world  T×2×21×3   metric hand shape
  pose        T×9×3      POSE_IDX: nose, shoulders, elbows, wrists, hips (x, y, z — only x, y are used)
  face        T×102×2    landmarks.FACE_IDX, x, y
  blend       T×52       face blendshapes

Features per frame (N_FEAT = 656), all body-relative: centre = mid-shoulders, unit = shoulder
width (medians over the clip), x first multiplied by width/height so geometry is square:
  hand positions 126 · hand shape (world, ×10) 126 · hand velocity 126 · pose xy 18 ·
  face xy 204 · blendshapes 52 · presence flags 4 (right hand, left hand, face, pose)
Missing values are 0 with the presence flag telling the model.

Augmentation (training only) works on the raw coordinates: temporal crop/jitter around the
sign, speed change, frame drop, horizontal flip (hands, pose, face points and blendshapes
swap sides — a mirrored sign is the same sign), small rotation/scale/shear, hand dropout.
"""
import numpy as np

from .landmarks import FACE_IDX

# Bump on ANY change to reduce(), to_square() or features(): models, the browser port and the
# landmark server compare it and refuse a mismatch.
FEATURE_VERSION = 1

POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]
POSE_FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7]
N_FEAT = 126 * 3 + 18 + 204 + 52 + 4
MAX_LEN = 64
TRAIN_FPS = 30          # the training landmarks: every frame of 30 fps video (stride 1)
# column ranges of features(), for ablations (train cfg 'groups')
GROUPS = {'hands': (0, 126), 'hand_shape': (126, 252), 'hand_vel': (252, 378), 'pose': (378, 396),
          'face': (396, 600), 'blend': (600, 652), 'flags': (652, 656)}
assert GROUPS['flags'][1] == N_FEAT


def columns(groups=None) -> np.ndarray:
    """Feature columns of the named groups (None = all)."""
    if not groups:
        return np.arange(N_FEAT)
    return np.concatenate([np.arange(*GROUPS[g]) for g in groups])

BLEND_NAMES = [
    '_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft',
    'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft',
    'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft',
    'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft',
    'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft',
    'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
    'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
    'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight',
]   # MediaPipe face_landmarker order (checked against a live detection)


def _swap_lr(name):
    for a, b in (('Left', 'Right'), ('Right', 'Left')):
        if name.endswith(a):
            return name[:-len(a)] + b
    return name


BLEND_FLIP = [BLEND_NAMES.index(_swap_lr(n)) for n in BLEND_NAMES]

# face-mesh mirror pairs (topology of the lip/eye/brow contours; brows, nose and outer lips
# also checked on a flipped frame)
_MIRROR = {
    61: 291, 185: 409, 40: 270, 39: 269, 37: 267, 146: 375, 91: 321, 181: 405, 84: 314,
    78: 308, 191: 415, 80: 310, 81: 311, 82: 312, 95: 324, 88: 318, 178: 402, 87: 317,
    263: 33, 249: 7, 390: 163, 373: 144, 374: 145, 380: 153, 381: 154, 382: 155, 362: 133,
    466: 246, 388: 161, 387: 160, 386: 159, 385: 158, 384: 157, 398: 173,
    276: 46, 283: 53, 282: 52, 295: 65, 285: 55, 300: 70, 293: 63, 334: 105, 296: 66, 336: 107,
    98: 327, 234: 454, 468: 473,
}
_MIRROR.update({b: a for a, b in list(_MIRROR.items())})
FACE_FLIP = [FACE_IDX.index(_MIRROR.get(i, i)) for i in FACE_IDX]
assert sorted(FACE_FLIP) == list(range(len(FACE_IDX))) and sorted(BLEND_FLIP) == list(range(52))


def reduce(arrays: dict) -> dict:
    """Store arrays (landmarks.FIELDS) → the compact raw dict the model needs (float16)."""
    return {
        'hand': arrays['hand'].astype(np.float16),
        'hand_world': arrays['hand_world'].astype(np.float16),
        'pose': arrays['pose'][:, POSE_IDX, :3].astype(np.float16),
        'face': arrays['face'][..., :2].astype(np.float16),
        'blend': arrays['blend'].astype(np.float16),
    }


def _affine(rng, strength=1.0):
    a = np.deg2rad(rng.uniform(-10, 10)) * strength
    s = 1 + rng.uniform(-0.15, 0.15) * strength
    sh = rng.uniform(-0.12, 0.12) * strength
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    M = R @ np.array([[1, sh], [0, 1]]) * s
    return M, a


def augment(raw: dict, rng, aspect: float, core=None, return_idx=False):
    """raw: square-space dict (to_square), `core` = (lo, hi) frame range of the sign inside it.
    return_idx: also the input frame each output frame came from (per-frame labels follow it)."""
    T = raw['hand'].shape[0]
    lo, hi = core if core is not None else (0, T)
    # temporal crop: jitter each boundary by up to ±4 frames inside the stored margin
    lo = int(np.clip(lo + rng.integers(-4, 5), 0, max(0, hi - 2)))
    hi = int(np.clip(hi + rng.integers(-4, 5), lo + 2, T))
    idx = np.arange(lo, hi)
    # speed 0.8–1.25× by resampling indices; drop up to 10% of frames
    speed = np.exp(rng.uniform(np.log(0.8), np.log(1.25)))
    n = max(2, int(round(len(idx) / speed)))
    idx = idx[np.clip(np.round(np.linspace(0, len(idx) - 1, n)).astype(int), 0, len(idx) - 1)]
    if len(idx) > 8 and rng.random() < 0.5:
        keep = rng.random(len(idx)) > rng.uniform(0, 0.1)
        idx = idx[keep] if keep.sum() >= 2 else idx
    out = {k: v[idx].astype(np.float32) for k, v in raw.items()}

    if rng.random() < 0.5:                     # mirror: the same sign signed with the other hand
        out = flip(out, aspect)

    # rotation / scale / shear about the shoulder centre, in square (aspect-corrected) space
    M, ang = _affine(rng)
    c = _centre(out)
    for k in ('hand', 'pose', 'face'):
        xy = out[k][..., :2]
        out[k][..., :2] = (xy - c) @ M.T + c + rng.normal(0, 0.003, xy.shape)
    cw, sw = np.cos(ang), np.sin(ang)
    hw = out['hand_world'][..., :2].copy()
    out['hand_world'][..., 0] = hw[..., 0] * cw - hw[..., 1] * sw
    out['hand_world'][..., 1] = hw[..., 0] * sw + hw[..., 1] * cw

    # a hand missing for a stretch (detector misses happen live too)
    for s in (0, 1):
        if rng.random() < 0.15:
            L = len(idx)
            span = max(1, int(L * rng.uniform(0.1, 0.4)))
            st = rng.integers(0, max(1, L - span + 1))
            out['hand'][st:st + span, s] = np.nan
            out['hand_world'][st:st + span, s] = np.nan
    return (out, idx) if return_idx else out


def flip(raw: dict, aspect: float) -> dict:
    """Mirror a square-space dict (x ∈ [0, aspect])."""
    out = {k: v.copy() for k, v in raw.items()}
    W = aspect
    out['hand'] = out['hand'][:, ::-1].copy()
    out['hand_world'] = out['hand_world'][:, ::-1].copy()
    out['hand'][..., 0] = W - out['hand'][..., 0]
    out['hand_world'][..., 0] = -out['hand_world'][..., 0]
    out['pose'] = out['pose'][:, POSE_FLIP]
    out['pose'][..., 0] = W - out['pose'][..., 0]
    out['face'] = out['face'][:, FACE_FLIP]
    out['face'][..., 0] = W - out['face'][..., 0]
    out['blend'] = out['blend'][:, BLEND_FLIP]
    return out


def to_square(raw: dict, aspect: float) -> dict:
    """Image x ∈ [0,1] → x·width/height, so distances are the same in x and y."""
    out = {k: v.astype(np.float32) for k, v in raw.items()}
    for k in ('hand', 'pose', 'face'):
        out[k][..., 0] *= aspect
    return out


def _centre(d):
    sh = d['pose'][:, 1:3, :2]
    c = np.nanmedian(sh.reshape(-1, 2), axis=0) if np.isfinite(sh).any() else np.array([np.nan, np.nan])
    if not np.isfinite(c).all():
        pts = d['hand'][..., :2].reshape(-1, 2)
        c = np.nanmean(pts, axis=0) if np.isfinite(pts).any() else np.array([0.5, 0.5])
    return c.astype(np.float32)


def _scale(d):
    sh = d['pose'][:, 1:3, :2]
    w = np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1)
    w = np.nanmedian(w) if np.isfinite(w).any() else np.nan
    return float(w) if np.isfinite(w) and w > 1e-3 else 0.3


def features(d: dict, max_len=MAX_LEN) -> np.ndarray:
    """Square-space compact dict → T×N_FEAT float32 (at most max_len frames, uniformly resampled;
    max_len None = every frame, for the continuous model's windows)."""
    T = d['hand'].shape[0]
    if max_len and T > max_len:
        sel = np.round(np.linspace(0, T - 1, max_len)).astype(int)
        d = {k: v[sel] for k, v in d.items()}
        T = max_len
    c, s = _centre(d), _scale(d)
    hand = d['hand'].copy()
    hand[..., :2] = (hand[..., :2] - c) / s
    hand[..., 2] = hand[..., 2] / s
    vel = np.zeros_like(hand)
    vel[1:] = hand[1:] - hand[:-1]
    pose = (d['pose'][..., :2] - c) / s
    face = (d['face'] - c) / s
    flags = np.stack([np.isfinite(d['hand'][:, 0, 0, 0]), np.isfinite(d['hand'][:, 1, 0, 0]),
                      np.isfinite(d['face'][:, 0, 0]), np.isfinite(d['pose'][:, 1, 0])], 1).astype(np.float32)
    x = np.concatenate([hand.reshape(T, -1), (d['hand_world'] * 10).reshape(T, -1), vel.reshape(T, -1),
                        pose.reshape(T, -1), face.reshape(T, -1), d['blend'].reshape(T, -1), flags], 1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, -10, 10).astype(np.float32)


# ---------------------------------------------------------------- the input contract

def drop_face(d: dict) -> dict:
    """Privacy mode / modality dropout: no face points, no blendshapes (the face flag becomes 0)."""
    out = dict(d)
    out['face'] = np.full_like(d['face'], np.nan)
    out['blend'] = np.full_like(d['blend'], np.nan)
    return out


def preprocess(raw: dict, aspect: float, groups=None, face: bool = True, max_len=MAX_LEN) -> np.ndarray:
    """THE entry point from landmarks to model input, the same for training evaluation, the browser
    (ported to JS) and the landmark server.

    raw:    compact dict (reduce() layout: hand, hand_world, pose, face, blend), float, NaN = missing.
            Without face points (privacy mode) pass face=False, or NaN face/blend arrays.
    aspect: frame width / height of the camera the landmarks come from.
    max_len: longer clips are resampled to this many frames (None: every frame, continuous model).
    → T'×F float32, T' = min(T, max_len), F = len(columns(groups)).
    """
    d = to_square(raw, aspect)
    if not face:
        d = drop_face(d)
    x = features(d, max_len)
    return x if not groups else x[:, columns(groups)]


def spec() -> dict:
    """Machine-readable description of the contract, written into every model manifest."""
    return {
        'feature_version': FEATURE_VERSION,
        'raw': {'hand': [2, 21, 3], 'hand_world': [2, 21, 3], 'pose': [len(POSE_IDX), 3],
                'face': [len(FACE_IDX), 2], 'blend': [len(BLEND_NAMES)]},
        'raw_notes': ('per frame; image coords x, y in [0, 1] of the frame (x NOT pre-multiplied by '
                      'width/height), hand slot 0 = signer\'s right hand (pose wrists decide), '
                      'pose = MediaPipe pose landmarks POSE_IDX with (x, y, z; only x, y are used), '
                      'face = face-mesh FACE_IDX (x, y), blend = face blendshapes in BLEND_NAMES order, '
                      'NaN = not detected'),
        'pose_idx': POSE_IDX, 'face_idx': list(FACE_IDX), 'blend_names': BLEND_NAMES,
        'n_features': N_FEAT, 'groups': {k: list(v) for k, v in GROUPS.items()},
        'max_len': MAX_LEN, 'resample': 'clips longer than max_len are uniformly resampled to max_len frames',
        'train_fps': TRAIN_FPS,
        'normalization': ('x *= width/height; centre = median mid-shoulder, unit = median shoulder width '
                          '(fallbacks: mean hand point, 0.3); hand xy, z, velocity, pose xy, face xy are '
                          '(p - centre) / unit; hand_world * 10; NaN → 0; clip to ±10'),
    }
