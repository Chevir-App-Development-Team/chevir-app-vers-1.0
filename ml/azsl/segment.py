"""Word mode: cut a continuous landmark stream into signs. Browser twin: src/tercume/segment.js
(the same rules; tools/verify_segment.mjs checks both on ml/fixtures/segment_v1.json).

1. Activity. A frame is active when a hand is seen above the rest line: its wrist is higher than the
   shoulder line + `rest_line` shoulder widths. Without a body, any seen hand counts. In AzSLD the
   hands rest in the lap (wrists ~1.25-1.3 shoulder widths below the shoulders) and are mostly not
   detected there.
2. Spans. A span opens after `min_on` active frames and closes after `min_off` resting ones. Its core is
   first..last active frame; `margin` frames of context are kept on both sides (the training margin).
   Spans whose core is shorter than `min_span` are dropped; one longer than `max_span` is closed anyway.
3. Signs inside a span (continuous signing has no rest between signs). Candidate windows are the whole
   span plus windows of `lens` frames every `stride` frames whose centre is inside the core. Each is
   scored by the model's top-1 probability. Greedy non-maximum suppression keeps the most confident
   window, drops windows that overlap it by more than `max_overlap` of the shorter one, and repeats
   down to `min_p`. Kept windows in time order; neighbours with the same label merge.
"""
import numpy as np

SEGMENT_VERSION = 1
SEGMENT = dict(rest_line=1.0, min_on=3, min_off=8, margin=8, min_span=8, max_span=240,
               lens=(12, 18, 26), stride=3, min_p=0.5, max_overlap=0.25)


def activity(raw: dict, aspect: float, rest_line: float = SEGMENT['rest_line']) -> np.ndarray:
    """raw: features.preprocess() input (hand T×2×21×3, pose T×9×3 in POSE_IDX order) → T bools."""
    hand, pose = np.asarray(raw['hand'], np.float64), np.asarray(raw['pose'], np.float64)
    seen = np.isfinite(hand[:, :, 0, 0])                                    # T×2
    sh_y = (pose[:, 1, 1] + pose[:, 2, 1]) / 2                              # POSE_IDX 1, 2 = shoulders
    sw = np.hypot((pose[:, 1, 0] - pose[:, 2, 0]) * aspect, pose[:, 1, 1] - pose[:, 2, 1])
    body = np.isfinite(sh_y) & np.isfinite(sw) & (sw > 1e-6)
    with np.errstate(invalid='ignore', divide='ignore'):
        up = (hand[:, :, 0, 1] - sh_y[:, None]) / sw[:, None] < rest_line     # T×2, NaN → False
    return (seen & (up | ~body[:, None])).any(1)


def spans(active, p=SEGMENT):
    """→ [(lo, hi, core_lo, core_hi)], hi / core_hi exclusive. Offline form of the browser's online
    state machine: a span is reported once min_off resting frames follow it (or the stream ends)."""
    out, run_on, run_off, start, last = [], 0, 0, None, None
    T = len(active)

    def close(end_n):
        if last - start + 1 >= p['min_span']:
            out.append((max(0, start - p['margin']), min(end_n, last + 1 + p['margin']), start, last + 1))

    for n, a in enumerate(active):
        if a:
            run_on, run_off = run_on + 1, 0
            if start is None and run_on >= p['min_on']:
                start = n - run_on + 1
            if start is not None:
                last = n
        else:
            run_on, run_off = 0, run_off + 1
        if start is not None and (run_off >= p['min_off'] or n - start + 1 >= p['max_span']):
            close(n + 1)
            start, last, run_off = None, None, 0
            run_on = 0
    if start is not None:
        close(T)
    return out


def windows(span, p=SEGMENT):
    """Candidate windows of one span, in a fixed order: whole span, then by length, then by start."""
    lo, hi, c0, c1 = span
    out = [(lo, hi)]
    for n in p['lens']:
        for a in range(lo, hi - n + 1, p['stride']):
            if c0 <= a + n / 2 < c1 and (a, a + n) != (lo, hi):
                out.append((a, a + n))
    return out


def pick(scored, p=SEGMENT):
    """scored: [(a, b, label, prob)] in windows() order → kept signs [(a, b, label, prob)] in time order."""
    order = sorted(range(len(scored)), key=lambda i: (-scored[i][3], i))
    kept = []
    for i in order:
        a, b, lab, pr = scored[i]
        if pr < p['min_p']:
            break
        if all(min(b, B) - max(a, A) <= p['max_overlap'] * min(b - a, B - A) for A, B, _, _ in kept):
            kept.append((a, b, lab, pr))
    kept.sort(key=lambda w: (w[0], w[1]))
    merged = []
    for w in kept:
        if merged and merged[-1][2] == w[2]:
            a, b, lab, pr = merged[-1]
            merged[-1] = (a, max(b, w[1]), lab, max(pr, w[3]))
        else:
            merged.append(w)
    return merged


def segment(raw, aspect, score, p=SEGMENT):
    """Full offline pass. score(a, b) → (label, prob) for frames [a, b). → [(a, b, label, prob)]."""
    out = []
    for sp in spans(activity(raw, aspect, p['rest_line']), p):
        out += pick([(a, b, *score(a, b)) for a, b in windows(sp, p)], p)
    return out
