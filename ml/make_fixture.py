"""Golden files for the feature contract (features.preprocess, FEATURE_VERSION) and for word-mode
segmentation (segment.py, SEGMENT_VERSION).

    python ml/make_fixture.py           # write ml/fixtures/features_v<N>.{json,bin}, segment_v<M>.json
    python ml/make_fixture.py --check   # recompute and compare — fails if features or segmentation
                                        # changed but their version was not bumped

Synthetic, seeded landmarks (no real person's data). The JS port for the browser (Phase 5) and the
landmark server must reproduce `expected` from `raw` — the same pattern as tools/verify_forward.*.
Binary: little-endian float32, NaN kept; the JSON says where each array starts.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from azsl import features as FT  # noqa: E402
from azsl import segment as SG  # noqa: E402
from azsl.landmarks import assign_hands  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
KEYS = ('hand', 'hand_world', 'pose', 'face', 'blend')


def synthetic(T, rng):
    """Plausible frontal signer: shoulders ~0.35 apart, hands moving in front of the chest."""
    t = np.linspace(0, 1, T)[:, None]
    pose = np.zeros((T, len(FT.POSE_IDX), 3), np.float32)
    base = np.array([[0.50, 0.25], [0.68, 0.45], [0.32, 0.45], [0.75, 0.65], [0.25, 0.65],
                     [0.62, 0.55], [0.38, 0.55], [0.62, 0.95], [0.38, 0.95]])
    pose[..., :2] = base + 0.01 * rng.standard_normal((T, len(FT.POSE_IDX), 2))
    pose[..., 2] = 0.99
    hand = np.zeros((T, 2, 21, 3), np.float32)
    for s, cx in ((0, 0.40), (1, 0.60)):
        centre = np.stack([cx + 0.1 * np.sin(6 * t[:, 0] + s), 0.55 - 0.1 * t[:, 0]], 1)
        hand[:, s, :, :2] = centre[:, None, :] + 0.04 * rng.standard_normal((21, 2))
        hand[:, s, :, 2] = 0.02 * rng.standard_normal((T, 21))
    hand_world = (0.05 * rng.standard_normal((T, 2, 21, 3))).astype(np.float32)
    face = (np.array([0.5, 0.25]) + 0.05 * rng.standard_normal((T, len(FT.FACE_IDX), 2))).astype(np.float32)
    blend = rng.random((T, len(FT.BLEND_NAMES))).astype(np.float32)
    return {'hand': hand, 'hand_world': hand_world, 'pose': pose, 'face': face, 'blend': blend}


def cases():
    rng = np.random.default_rng(20261018)
    a = synthetic(12, rng)
    a['hand'][3:6, 1] = np.nan                      # left hand lost for three frames
    a['hand_world'][3:6, 1] = np.nan
    b = synthetic(70, rng)                          # > MAX_LEN → uniform resampling
    b['face'][:, :] = np.nan                        # face not detected at all
    b['blend'][:, :] = np.nan
    c = synthetic(20, rng)
    c['pose'][:, :, :] = np.nan                     # no body found → centre from hands, unit 0.3
    return [
        {'name': 'short_missing_hand', 'raw': a, 'aspect': 16 / 9, 'face': True, 'groups': None},
        {'name': 'no_pose_fallback', 'raw': c, 'aspect': 16 / 9, 'face': True, 'groups': None},
        {'name': 'long_resampled_no_face_detected', 'raw': b, 'aspect': 4 / 3, 'face': True, 'groups': None},
        {'name': 'privacy_mode', 'raw': a, 'aspect': 16 / 9, 'face': False, 'groups': None},
        {'name': 'hands_only_groups', 'raw': a, 'aspect': 16 / 9, 'face': True,
         'groups': ['hands', 'hand_shape', 'hand_vel', 'flags']},
    ]


def hand_slot_cases(n=300):
    """Random hand/pose layouts → which detection goes to the right / left slot (assign_hands)."""
    rng = np.random.default_rng(7)
    out = []
    for _ in range(n):
        k = int(rng.integers(0, 3))
        wrists = [[float(np.float32(v)) for v in rng.uniform(0.15, 0.85, 2)] for _ in range(k)]
        pose = None if rng.random() < 0.2 else [[float(np.float32(v)) for v in rng.uniform(0.2, 0.8, 2)]
                                                 for _ in range(2)]
        aspect = float(rng.choice([16 / 9, 4 / 3, 1.0]))
        out.append({'wrists': wrists, 'pose_wrists': pose, 'aspect': aspect,
                    'slots': assign_hands(wrists, pose, aspect)})
    return out


def _nan_list(a):
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, np.float32).ravel()]


def segment_cases():
    """Random inputs for each step of segment.py and its answers (browser twin: segment.js)."""
    rng = np.random.default_rng(2026)
    variant = {**SG.SEGMENT, 'min_on': 2, 'min_off': 5, 'margin': 4, 'min_span': 5, 'max_span': 50,
               'lens': (8, 11), 'stride': 2, 'min_p': 0.4, 'max_overlap': 0.3}
    act = []
    for k in range(8):
        T = 40
        hand = np.full((T, 2, 21, 3), np.nan, np.float32)
        seen = rng.random((T, 2)) < 0.6
        hand[..., :2] = np.where(seen[:, :, None, None], rng.uniform(0.1, 1.3, (T, 2, 21, 2)), np.nan)
        pose = rng.uniform(0.2, 0.8, (T, 9, 3)).astype(np.float32)
        pose[rng.random(T) < (0.5 if k == 0 else 0.15)] = np.nan          # no body in some frames
        aspect = float(rng.choice([16 / 9, 4 / 3, 1.0]))
        raw = {'hand': hand, 'pose': pose}
        act.append({'T': T, 'aspect': aspect, 'rest_line': SG.SEGMENT['rest_line'],
                    'hand': _nan_list(hand), 'pose': _nan_list(pose),
                    'active': [bool(a) for a in SG.activity(raw, aspect)]})
    spans, wins, picks = [], [], []
    for k in range(60):
        p = SG.SEGMENT if k % 2 else variant
        runs, a = [], bool(rng.random() < 0.5)
        while sum(len(r) for r in runs) < int(rng.integers(10, 300)):
            runs.append([a] * int(rng.integers(1, 40 if a else 15)))
            a = not a
        seq = [x for r in runs for x in r]
        spans.append({'params': _jsonable(p), 'active': seq, 'spans': [list(s) for s in SG.spans(seq, p)]})
    for k in range(30):
        p = SG.SEGMENT if k % 2 else variant
        lo = int(rng.integers(0, 50))
        c0 = lo + int(rng.integers(0, 10))
        c1 = c0 + int(rng.integers(1, 80))
        span = (lo, c1 + int(rng.integers(0, 10)), c0, c1)
        ws = SG.windows(span, p)
        wins.append({'params': _jsonable(p), 'span': list(span), 'windows': [list(w) for w in ws]})
        labels = ['A', 'B', 'C', 'D']
        scored = [(a, b, labels[int(rng.integers(0, 4))], float(np.round(rng.random(), 2))) for a, b in ws]
        picks.append({'params': _jsonable(p), 'scored': [list(w) for w in scored],
                      'kept': [list(w) for w in SG.pick(scored, p)]})
    return {'segment_version': SG.SEGMENT_VERSION, 'params': _jsonable(SG.SEGMENT),
            'activity': act, 'spans': spans, 'windows': wins, 'pick': picks}


def continuous_cases():
    """Golden cases for the browser's continuous mode (continuous.js): the streaming schedule, decode()
    on synthetic frame probabilities, and features of windows without the 64-frame cap."""
    from azsl import continuous as CT
    rng = np.random.default_rng(20261022)
    blobs, off = [], 0

    def put(a):
        nonlocal off
        a = np.asarray(a, np.float32)
        blobs.append(a.ravel())
        ref = {'offset': off, 'shape': list(a.shape)}
        off += a.size
        return ref

    stream = [{'T': T, **st, 'windows': [list(w) for w in CT.stream_windows(T, **st)]}
              for T in (1, 7, 24, 95, 96, 97, 150, 333)
              for st in (CT.STREAM, dict(window=64, stride=4, lookahead=12))]
    dec = []
    for i, p in enumerate([CT.DECODE, dict(smooth=1, min_p=0.3, min_len=4, gap=2), dict(smooth=9, min_p=0.35, min_len=8, gap=4),
                           dict(smooth=3, min_p=0.4, min_len=6, gap=0),
                           dict(smooth=3, min_p=0.3, min_len=10, gap=2, min_len_short=4)]):   # v2
        T, C = 180, 9
        lab = np.zeros(T, int)
        t = int(rng.integers(3, 12))
        while t < T - 5:
            L = int(rng.integers(4, 12 if 'min_len_short' in p else 30))     # v2 case: letter-like short signs
            lab[t:t + L] = rng.integers(1, C)
            t += L + int(rng.integers(0, 10))
        logits = rng.normal(0, 1.0, (T, C))
        logits[np.arange(T), lab] += rng.uniform(0.5, 4.0, T)
        P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        unknown = C - 1 if i % 2 else None
        short = [1, 2, 3] if 'min_len_short' in p else []
        exp = [[a, b, c, pm] for a, b, c, pm in CT.decode(P.astype(np.float32), p, unknown, set(short))]
        dec.append({'T': T, 'C': C, 'params': dict(p), 'unknown': unknown, 'short': short, 'probs': put(P),
                    'expected': exp})
    feats = []
    for T, aspect in ((96, 16 / 9), (150, 4 / 3)):
        raw = synthetic(T, rng)
        raw['hand'][20:31, 0] = np.nan
        raw['hand_world'][20:31, 0] = np.nan
        entry = {'T': T, 'aspect': aspect, 'groups': ['hands', 'hand_shape', 'hand_vel', 'flags', 'pose']}
        for k in KEYS:
            entry[k] = put(raw[k])
        entry['expected'] = put(FT.preprocess(raw, aspect, entry['groups'], False, max_len=None))
        feats.append(entry)
    return {'continuous_version': CT.CONTINUOUS_VERSION, 'stream': stream, 'decode': dec, 'features': feats,
            'tolerance': 1e-5}, np.concatenate(blobs).astype('<f4')


def _jsonable(p):
    return {k: list(v) if isinstance(v, tuple) else v for k, v in p.items()}


def build():
    blobs, meta, off = [], [], 0
    for c in cases():
        entry = {'name': c['name'], 'aspect': c['aspect'], 'use_face': c['face'], 'groups': c['groups']}
        entry['T'] = int(c['raw']['hand'].shape[0])
        for k in KEYS:
            arr = np.asarray(c['raw'][k], np.float32)
            entry[k] = {'offset': off, 'shape': list(arr.shape)}
            blobs.append(arr.ravel())
            off += arr.size
        exp = FT.preprocess(c['raw'], c['aspect'], c['groups'], c['face']).astype(np.float32)
        entry['expected'] = {'offset': off, 'shape': list(exp.shape)}
        blobs.append(exp.ravel())
        off += exp.size
        meta.append(entry)
    return {'feature_version': FT.FEATURE_VERSION, 'dtype': 'float32-le', 'tolerance': 1e-5,
            'contract': FT.spec(), 'cases': meta, 'hand_slots': hand_slot_cases()},         np.concatenate(blobs).astype('<f4')


def check_or_write_continuous(check):
    from azsl import continuous as CT
    meta, blob = continuous_cases()
    stem = os.path.join(OUT, f'continuous_v{CT.CONTINUOUS_VERSION}')
    if check:
        if not os.path.exists(stem + '.bin'):
            sys.exit(f'no fixture for continuous_version {CT.CONTINUOUS_VERSION} — run without --check to create it')
        with open(stem + '.json', encoding='utf-8') as f:
            same = json.load(f) == json.loads(json.dumps(meta))
        ref = np.fromfile(stem + '.bin', dtype='<f4')
        if not same or ref.shape != blob.shape or not np.allclose(ref, blob, atol=1e-6, equal_nan=True):
            sys.exit(f'continuous decoding changed but CONTINUOUS_VERSION is still {CT.CONTINUOUS_VERSION}: bump it, '
                     'update continuous.js and regenerate the fixture')
        print(f'continuous mode matches fixture v{CT.CONTINUOUS_VERSION}')
        return
    blob.tofile(stem + '.bin')
    with open(stem + '.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, separators=(',', ':'))
    print(f'wrote {stem}.json/.bin ({blob.nbytes / 1024:.0f} KB)')


def main():
    check_or_write_continuous('--check' in sys.argv)
    meta, blob = build()
    stem = os.path.join(OUT, f'features_v{FT.FEATURE_VERSION}')
    seg = json.loads(json.dumps(segment_cases()))
    seg_path = os.path.join(OUT, f'segment_v{SG.SEGMENT_VERSION}.json')
    if '--check' in sys.argv:
        if not os.path.exists(seg_path):
            sys.exit(f'no fixture for segment_version {SG.SEGMENT_VERSION} — run without --check to create it')
        with open(seg_path, encoding='utf-8') as f:
            if json.load(f) != seg:
                sys.exit(f'segmentation changed but SEGMENT_VERSION is still {SG.SEGMENT_VERSION}: bump it, '
                         'update segment.js and regenerate the fixture')
        print(f'segmentation matches fixture v{SG.SEGMENT_VERSION}')
        if not os.path.exists(stem + '.bin'):
            sys.exit(f'no fixture for feature_version {FT.FEATURE_VERSION} — run without --check to create it')
        ref = np.fromfile(stem + '.bin', dtype='<f4')
        if ref.shape != blob.shape or not np.allclose(ref, blob, atol=1e-5, equal_nan=True):
            sys.exit(f'features changed but FEATURE_VERSION is still {FT.FEATURE_VERSION}: bump it, '
                     're-export the models and regenerate the fixture')
        print(f'features match fixture v{FT.FEATURE_VERSION} ({len(meta["cases"])} cases)')
        return
    os.makedirs(OUT, exist_ok=True)
    with open(seg_path, 'w', encoding='utf-8') as f:
        json.dump(seg, f, separators=(',', ':'))
    print(f'wrote {seg_path} ({os.path.getsize(seg_path) / 1024:.0f} KB)')
    blob.tofile(stem + '.bin')
    with open(stem + '.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f'wrote {stem}.json/.bin ({blob.nbytes / 1024:.0f} KB, {len(meta["cases"])} cases)')


if __name__ == '__main__':
    main()
