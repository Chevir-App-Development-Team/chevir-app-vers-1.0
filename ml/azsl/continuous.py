"""Continuous recognition: one label for every frame of a signed sentence, so the words of a whole
sentence come out in one run, while it is being signed.

Labels: 0 = BLANK (no sign: rests, transitions, before and after), then the words, last UNKNOWN (rare
signs). Training data are the whole sentence recordings: every frame counts, also the half of all frames
between signs that the isolated-sign model never saw, and the side camera's view (shifted by its lag).
Frames within `edge` of an annotated boundary are not scored (the annotations are ±1–2 frames uncertain).

The model is the isolated-sign network with a label per frame instead of one per clip (SeqNet), trained on
random windows of 48–160 frames. In the browser it runs on the last `window` frames every `stride` new
frames, and a frame's probabilities are fixed once it is `lookahead` frames from the window's right edge
(stream_windows / stream_probs: the browser twin does exactly this). decode() turns frame probabilities
into words; its settings are chosen on val sentences (notebook 06).
"""
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import features as FT
from . import segment_eval as SE
from . import train as TR
from .store import LandmarkStore

CONTINUOUS_VERSION = 2    # streaming schedule + decode(); continuous.js and ml/fixtures/continuous_v<N> follow it
                          # v2: decode's optional min_len_short for single-character labels (fingerspelled letters)
BLANK = '<blank>'
IGNORE = -100
STREAM = dict(window=96, stride=8, lookahead=24)        # browser: 3.2 s of context, a word 0.8 s after it ends
DECODE = dict(smooth=5, min_p=0.5, min_len=6, gap=3)     # defaults; notebook 06 chooses them on val
DEFAULTS = dict(TR.DEFAULTS, epochs=80, batch=32, crop=(48, 160), repeats=2, edge=1, init=None,
                balance_power=0.5, select='f1_all',
                splice=0.0,          # P(a training sample is a sentence spliced from signs of other recordings)
                splice_power=0.0,    # spliced signs drawn with weight (signs of their word)^-power; 0 = uniform
                fps_aug=0.0,         # P(a training window comes as a slower camera gives it, refilled to 30 fps)
                fps_min=8.0,         # its frame rate: uniform in [fps_min, 30]
                spell=0.0,           # P(a training sample is a spelled sequence: 3-8 letter signs of other recordings)
                stream=None)         # browser streaming (window, stride, lookahead); None = STREAM


# ---------------------------------------------------------------- labels and data

def frame_labels(T, anns, index, unknown=None, exclude=(), edge=1):
    """anns [(start, end, gloss)], 1-based inclusive frames → int64[T]: 0 = BLANK, a word's index, the
    UNKNOWN index for other glosses (IGNORE without one), IGNORE on `exclude` glosses and within `edge`
    frames of every boundary."""
    y = np.zeros(T, np.int64)
    for s, e, g in sorted(anns, key=lambda a: (a[0], a[1])):
        s0, e0 = max(0, int(s) - 1), min(T, int(e))
        if e0 - s0 < 2:
            continue
        y[s0:e0] = IGNORE if g in exclude else index.get(g, index[unknown] if unknown else IGNORE)
    if edge:
        for t in np.flatnonzero(y[1:] != y[:-1]) + 1:
            y[max(0, t - edge):t + edge] = IGNORE
    return y


class SeqBank:
    """Whole recordings: concatenated compact landmarks (float16) + per-frame labels (int16)."""

    def __init__(self):
        self.chunks, self.lab, self.rows, self.n = {k: [] for k in TR.KEYS}, [], [], 0

    def add(self, raw, y, aspect, rec_id):
        for k in TR.KEYS:
            self.chunks[k].append(raw[k])
        self.lab.append(y.astype(np.int16))
        self.rows.append((self.n, len(y), float(aspect), rec_id))
        self.n += len(y)

    def done(self):
        self.data = {k: np.concatenate(v) if v else np.zeros((0,), np.float16) for k, v in self.chunks.items()}
        self.labels = np.concatenate(self.lab) if self.lab else np.zeros(0, np.int16)
        self.chunks = self.lab = None
        self.ids = [r[3] for r in self.rows]
        return self

    def __len__(self):
        return len(self.rows)

    def get(self, i):
        off, T, aspect, rid = self.rows[i]
        return {k: self.data[k][off:off + T] for k in TR.KEYS}, self.labels[off:off + T].astype(np.int64), aspect, rid


def load_recordings(sentence_dirs, recs, anns, index, unknown=None, exclude=(), edge=1, side=None, log=print):
    """recs: DataFrame (rec_id, split) of the recordings to use; anns {rec_id: [(start, end, gloss)]}.
    side: (Cam1 store dirs, {rec_id: lag}) → their side views too: train in 'train', val / test in
    '<split>_side' (frame i of the side view = frame i + lag of the annotated one)."""
    split_of = dict(zip(recs.rec_id, recs.split))
    banks = {s: SeqBank() for s in ('train', 'val', 'test')}
    views = [(LandmarkStore(sentence_dirs), None)]
    if side is not None:
        views.append((LandmarkStore(side[0]), side[1]))
        banks.update(val_side=SeqBank(), test_side=SeqBank())
    t0 = time.time()
    for store, lags in views:
        idx = store.index
        ids = [r for r in split_of if r in idx.index and (lags is None or r in lags)]
        ids.sort(key=lambda r: (idx.at[r, 'dir'], idx.at[r, 'part']))
        front = views[0][0].index
        for n, rid in enumerate(ids):
            raw = FT.reduce(store.get(rid, dtype=np.float16))
            T = raw['hand'].shape[0]
            if T < 8:
                continue
            aspect = float(idx.at[rid, 'width']) / float(idx.at[rid, 'height'])
            if lags is None:
                y = frame_labels(T, anns.get(rid, []), index, unknown, exclude, edge)
                bank = banks[split_of[rid]]
            else:                                # labels of the annotated view, shifted by the lag
                Tf = int(front.at[rid, 'length']) if rid in front.index else T
                yf = frame_labels(Tf, anns.get(rid, []), index, unknown, exclude, edge)
                src = np.arange(T) + int(lags[rid])
                ok = (src >= 0) & (src < Tf)
                y = np.full(T, IGNORE, np.int64)
                y[ok] = yf[src[ok]]
                sp = split_of[rid]
                bank = banks['train' if sp == 'train' else f'{sp}_side']
            bank.add(raw, y, aspect, rid if lags is None else rid + '|side')
            if n and n % 2000 == 0:
                log(f'  {n} recordings ({time.time() - t0:.0f} s)')
    banks = {k: b.done() for k, b in banks.items()}
    log('recordings: ' + ', '.join(f'{k} {len(b)} ({b.n:,} frames)' for k, b in banks.items()))
    return banks


def sign_units(bank, edge=2):
    """Every annotated sign of a bank with the transition after it: (recording, a, b, label), a..b frames
    from `edge` frames before the sign to `edge` frames before the next one. Units of the side camera are
    kept apart (a spliced sentence never mixes the two views)."""
    out = {False: [], True: []}
    for i in range(len(bank)):
        _, y, _, rid = bank.get(i)
        sign = y > 0
        starts = np.flatnonzero(sign & ~np.r_[False, sign[:-1]])
        for k, a in enumerate(starts):
            b = starts[k + 1] if k + 1 < len(starts) else len(y)
            out['|side' in rid].append((i, max(0, int(a) - edge), max(int(a) + 1, int(b) - edge), int(y[a])))
    return out


def unit_cdf(units, power):
    """Cumulative sampling weights of sign units: (signs of the unit's word)^-power (0 = uniform)."""
    if not units:
        return np.zeros(0)
    lab = np.array([u[3] for u in units])
    w = np.bincount(lab)[lab].astype(np.float64) ** -power
    return np.cumsum(w / w.sum())


def _shoulders(raw):
    sh = raw['pose'][:, 1:3, :2].astype(np.float32)
    c = np.nanmedian(sh.reshape(-1, 2), 0) if np.isfinite(sh).any() else np.array([0.5, 0.5], np.float32)
    w = np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1)
    w = float(np.nanmedian(w)) if np.isfinite(w).any() else 0.3
    return c, (w if w > 1e-3 else 0.3)


def _moved(raw, c, w, c0, w0):
    """Image coordinates of a unit moved and scaled so its shoulders sit where the first unit's do."""
    out = dict(raw)
    for k in ('hand', 'pose', 'face'):
        v = raw[k].astype(np.float32).copy()
        v[..., :2] = (v[..., :2] - c) / w * w0 + c0
        if k == 'hand':
            v[..., 2] = v[..., 2] / w * w0
        out[k] = v
    return out


class SeqDataset(torch.utils.data.Dataset):
    """Train: a random window of each recording, augmented like the isolated signs (the labels follow the
    resampled frames). With `splice`, a sample is instead a new sentence: 2–6 signs (each with its
    transition) from random recordings of the same camera, moved to one body position and joined with a
    3-frame cross-fade, so the model cannot lean on the 492 sentences' word order. With `spell`, a sample is
    a spelled sequence the same way, 3–8 signs of the `short` labels (fingerspelled letters and digits), drawn
    with weight (signs of their letter)^-0.5: fast letter-to-letter runs are where letters are lost. Eval: the
    whole recording through features.preprocess (no length cap)."""

    def __init__(self, bank, train=False, groups=None, face=True, crop=(48, 160), face_drop=0.0, splice=0.0,
                 splice_power=0.0, fps_aug=0.0, fps_min=8.0, spell=0.0, short=()):
        self.bank, self.train, self.groups, self.face, self.crop, self.face_drop = bank, train, groups, face, crop, face_drop
        self.fps_aug, self.fps_min = fps_aug, fps_min
        self.cols = FT.columns(groups) if groups else None
        self.splice, self.spell = splice, spell
        self.units = sign_units(bank) if train and (splice or spell) else None
        self.cdf = {k: unit_cdf(u, splice_power) for k, u in self.units.items()} if self.units else None
        if self.units and spell:
            self.letters = {k: [x for x in u if x[3] in short] for k, u in self.units.items()}
            self.letter_cdf = {k: unit_cdf(u, 0.5) for k, u in self.letters.items()}

    def _spliced(self, rng, side, letters=False):
        units, cdf = (self.letters[side], self.letter_cdf[side]) if letters else (self.units[side], self.cdf[side])
        parts, ys, aspect, ref = [], [], None, None
        for _ in range(int(rng.integers(3, 9) if letters else rng.integers(2, 7))):
            i, a, b, _ = units[min(int(np.searchsorted(cdf, rng.random(), side='right')), len(units) - 1)]
            raw, y, asp, _ = self.bank.get(i)
            raw = {k: v[a:b].astype(np.float32) for k, v in raw.items()}
            c, w = _shoulders(raw)
            if ref is None:
                ref, aspect = (c, w), asp
            parts.append(_moved(raw, c, w, *ref))
            ys.append(y[a:b].copy())
        raw, y = parts[0], ys[0]
        for nxt, yn in zip(parts[1:], ys[1:]):
            n = min(3, len(y), len(yn))
            wgt = np.linspace(0, 1, n + 2, dtype=np.float32)[1:-1]
            blend = {k: raw[k][-n:] * (1 - wgt.reshape(-1, *[1] * (raw[k].ndim - 1)))
                     + nxt[k][:n] * wgt.reshape(-1, *[1] * (raw[k].ndim - 1)) for k in raw}
            raw = {k: np.concatenate([raw[k][:-n], blend[k], nxt[k][n:]]) for k in raw}
            y = np.concatenate([y[:-n], np.full(n, IGNORE, np.int64), yn[n:]])
        return raw, y, aspect

    def __len__(self):
        return len(self.bank)

    def __getitem__(self, i):
        raw, y, aspect, _ = self.bank.get(i)
        if not self.train:
            return torch.from_numpy(FT.preprocess(raw, aspect, self.groups, self.face, max_len=None)), torch.from_numpy(y)
        rng = np.random.default_rng()
        side = bool(self.units) and bool(self.units[True]) and '|side' in self.bank.get(i)[3]
        if self.units and self.spell and self.letters[side] and rng.random() < self.spell:
            raw, y, aspect = self._spliced(rng, side, letters=True)
        elif self.units and self.splice and rng.random() < self.splice:
            raw, y, aspect = self._spliced(rng, side)
        T, L = len(y), int(rng.integers(self.crop[0], self.crop[1] + 1))
        if T > L:
            s = int(rng.integers(0, T - L + 1))
            raw, y = {k: v[s:s + L] for k, v in raw.items()}, y[s:s + L]
        if self.fps_aug and rng.random() < self.fps_aug:     # landmarks at a lower frame rate, as in a slow browser
            raw = camera_resample(raw, float(rng.uniform(self.fps_min, 30.0)))
        d, idx = FT.augment(FT.to_square(raw, aspect), rng, aspect, return_idx=True)
        if not self.face or (self.face_drop and rng.random() < self.face_drop):
            d = FT.drop_face(d)
        x = FT.features(d, max_len=None)
        return torch.from_numpy(x if self.cols is None else x[:, self.cols]), torch.from_numpy(y[idx])


def collate(batch):
    T = max(x.shape[0] for x, _ in batch)
    X = torch.zeros(len(batch), T, batch[0][0].shape[1])
    Y = torch.full((len(batch), T), IGNORE, dtype=torch.long)
    M = torch.zeros(len(batch), T, dtype=torch.bool)
    for i, (x, y) in enumerate(batch):
        X[i, :len(x)], Y[i, :len(y)], M[i, :len(x)] = x, y, True
    return X, M, Y


class SeqNet(TR.Net):
    """The isolated-sign network, with a label per frame instead of a pooled one."""

    def forward(self, x, valid=None):           # → B×T×C logits
        m = TR._mask(x, valid)
        h = self.stem_norm(self.stem(x)) * m
        for b in self.blocks:
            h = b(h, valid)
        return self.head(self.drop(self.norm(h)))


def load_checkpoint(path, device='cpu'):
    """A saved best.pt → (SeqNet in eval mode, its training cfg, its labels, its epoch)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = {**DEFAULTS, **ck['cfg']}
    assert ck.get('feature_version') == FT.FEATURE_VERSION, 'checkpoint from another feature version'
    net = SeqNet(len(FT.columns(cfg['groups'])), len(ck['vocab']), cfg['dim'], cfg['stages'], cfg['convs'],
                 cfg['kernel'], cfg['heads'], cfg['drop'], cfg['head_drop']).to(device)
    net.load_state_dict(ck['model'])
    return net.eval(), cfg, list(ck['vocab']), ck.get('epoch')


def init_from(model, ckpt_path, vocab_list, log=print):
    """Start from an isolated-sign checkpoint: every layer, and the head's row of each shared label."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state, old = ck['model'], {g: i for i, g in enumerate(ck['vocab'])}
    own = model.state_dict()
    body = {k: v for k, v in state.items() if not k.startswith('head.') and k in own and own[k].shape == v.shape}
    model.load_state_dict(body, strict=False)
    shared = [(i, old[g]) for i, g in enumerate(vocab_list) if g in old]
    with torch.no_grad():
        for i, j in shared:
            model.head.weight[i] = state['head.weight'][j]
            model.head.bias[i] = state['head.bias'][j]
    log(f'init from {ckpt_path}: {len(body)} tensors, head rows for {len(shared)} of {len(vocab_list)} labels')


# ---------------------------------------------------------------- streaming and decoding

def stream_windows(T, window=96, stride=8, lookahead=24):
    """The browser's schedule → [(a, b, fix_from, fix_to)]: the model runs on frames [a, b), and frames
    [fix_from, fix_to) take their probabilities from that run (each frame exactly once)."""
    assert lookahead + stride <= window
    out, fixed, n = [], 0, 0
    while fixed < T:
        n = min(T, n + stride)
        a = max(0, n - window)
        upto = T if n == T else max(fixed, n - lookahead)
        if upto > fixed:
            out.append((a, n, fixed, upto))
            fixed = upto
    return out


def smooth(P, k):
    """Centred moving average over time (k frames, shorter at the ends)."""
    P = np.asarray(P, np.float64)            # float64 like the browser's numbers
    if k <= 1:
        return P
    c = np.cumsum(np.vstack([np.zeros((1, P.shape[1])), P]), 0)
    T, h = len(P), k // 2
    lo, hi = np.clip(np.arange(T) - h, 0, T), np.clip(np.arange(T) + k - h, 0, T)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def short_classes(labels):
    """Indices of the single-character labels: fingerspelled letters and digits, shorter than words."""
    return {i for i, g in enumerate(labels) if len(g) == 1}


def decode(P, p=DECODE, unknown=None, short=()):
    """T×C frame probabilities (column 0 = BLANK) → [(a, b, label index, mean probability)], frames
    [a, b). Runs of one label (after smoothing), joined across gaps of at most `gap` frames, at least
    `min_len` frames long (labels in `short`: `min_len_short`, if p has it) and `min_p` probable on
    average; UNKNOWN runs are dropped."""
    Ps = smooth(P, int(p['smooth']))
    lab = Ps.argmax(1)
    runs, t, T = [], 0, len(lab)
    while t < T:
        u = t
        while u < T and lab[u] == lab[t]:
            u += 1
        if lab[t] != 0:
            runs.append([t, u, int(lab[t])])
        t = u
    joined = []
    for r in runs:
        if joined and joined[-1][2] == r[2] and r[0] - joined[-1][1] <= p['gap']:
            joined[-1][1] = r[1]
        else:
            joined.append(r)
    out = []
    for a, b, c in joined:
        pm = float(Ps[a:b, c].mean())
        need = p['min_len_short'] if c in short and 'min_len_short' in p else p['min_len']
        if b - a >= need and pm >= p['min_p'] and c != unknown:
            out.append((a, b, c, pm))
    return out


@torch.no_grad()
def window_probs(net, x, wins, device, batch=64):
    """x = (compact raw dict, aspect, groups, face) of one recording; each window [a, b) is preprocessed
    on its own, as in the browser → list of (b − a) × C softmax arrays."""
    raw, aspect, groups, face = x
    feats = [FT.preprocess({k: v[a:b] for k, v in raw.items()}, aspect, groups, face, max_len=None) for a, b, *_ in wins]
    out = [None] * len(feats)
    order = sorted(range(len(feats)), key=lambda i: len(feats[i]))
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        T = max(len(feats[i]) for i in idx)
        X = torch.zeros(len(idx), T, feats[idx[0]].shape[1])
        M = torch.zeros(len(idx), T, dtype=torch.bool)
        for r, i in enumerate(idx):
            X[r, :len(feats[i])], M[r, :len(feats[i])] = torch.from_numpy(feats[i]), True
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            p = net(X.to(device), M.to(device)).float().softmax(-1).cpu().numpy()
        for r, i in enumerate(idx):
            out[i] = p[r, :len(feats[i])]
    return out


def camera_resample(raw, fps, src_fps=30.0):
    """The recording as a camera (or landmark tracker) at `fps` delivers it, filled back to 30 fps as the
    browser does (continuous.js Resampler: linear interpolation onto the 30 fps grid). The end is held at
    the last delivered frame, so the result keeps the recording's length and its frame labels."""
    T = raw['hand'].shape[0]
    if fps >= src_fps:
        return raw
    dt = 1000.0 / src_fps
    out, prev, nxt, k = {key: [] for key in raw}, None, None, 0
    while True:
        ts = k * 1000.0 / fps
        i = int(np.floor(ts / dt + 0.5))                 # Math.round, as the browser's frame times would
        if i >= T:
            break
        fr = {key: raw[key][i].astype(np.float32) for key in raw}
        if prev is None:
            for key in raw:
                out[key].append(fr[key])
            nxt = ts + dt
        else:
            fa, ta = prev
            while nxt <= ts:
                w = 1.0 if ts == ta else (nxt - ta) / (ts - ta)
                for key in raw:
                    out[key].append(fa[key] + (fr[key] - fa[key]) * w)
                nxt += dt
        prev, k = (fr, ts), k + 1
    for key in raw:
        out[key] += [out[key][-1]] * (T - len(out[key]))
    return {key: np.stack(v[:T]) for key, v in out.items()}


def stream_probs(net, raw, aspect, groups, face, device, window=96, stride=8, lookahead=24):
    """Frame probabilities exactly as the browser fixes them while the sentence is being signed."""
    T = raw['hand'].shape[0]
    wins = stream_windows(T, window, stride, lookahead)
    probs = window_probs(net, (raw, aspect, groups, face), wins, device)
    out = None
    for (a, b, f0, f1), p in zip(wins, probs):
        if out is None:
            out = np.zeros((T, p.shape[1]), np.float32)
        out[f0:f1] = p[f0 - a:f1 - a]
    return out


@torch.no_grad()
def offline_probs(net, bank, groups, face, device, batch=32):
    """Whole recordings in one pass each (faster, for per-epoch validation)."""
    net.eval()
    dl = torch.utils.data.DataLoader(SeqDataset(bank, groups=groups, face=face), batch_size=batch,
                                     shuffle=False, num_workers=2, collate_fn=collate)
    out = []
    for X, M, _ in dl:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            p = net(X.to(device), M.to(device)).float().softmax(-1).cpu().numpy()
        out += [p[i, :int(M[i].sum())] for i in range(len(p))]
    return out


def subset(bank, ids):
    """The recordings of a bank whose id (without the camera suffix) is in `ids`, as a new bank."""
    out = SeqBank()
    for i, r in enumerate(bank.ids):
        if r.split('|')[0] in ids:
            raw, y, aspect, rid = bank.get(i)
            out.add(raw, y, aspect, rid)
    return out.done()


def score(probs, bank, anns, vocab_list, decode_p, unknown, fam=None, keep=None, dets_out=None):
    """Word-level scores (segment_eval.Tally: precision, recall over all annotated signs, F1, WER) of
    decoded probabilities against the annotations, + frame accuracy on the scored frames.
    keep: only annotations of these glosses count (e.g. words with training data), and detections that
    overlap only the other annotations are left out; dets_out: a list that receives (rec_id, a, b, word, p)
    of every detection."""
    tally, right, total = SE.Tally(), 0, 0
    known = set(vocab_list) - {BLANK, TR.UNKNOWN}
    short = short_classes(vocab_list)
    hits = lambda a, b, xs: any(min(b, e) - max(a, s) > 0 for s, e, _, _ in xs)      # noqa: E731
    for i, P in enumerate(probs):
        _, y, _, rid = bank.get(i)
        rid = rid.split('|')[0]
        ok = y != IGNORE
        right, total = right + int((P.argmax(1)[ok] == y[ok]).sum()), total + int(ok.sum())
        T = len(P)
        a_ = [(max(0, int(s) - 1), min(int(e), T), g, g in known) for s, e, g in anns.get(rid, [])]
        found = decode(P, decode_p, unknown, short)
        dets = [(a, b, vocab_list[c]) for a, b, c, _ in found]
        if keep is not None:
            gone = [x for x in a_ if x[2] not in keep]
            a_ = [x for x in a_ if x[2] in keep]
            dets = [d for d in dets if not (hits(d[0], d[1], gone) and not hits(d[0], d[1], a_))]
        if dets_out is not None:
            dets_out += [(bank.ids[i], a, b, vocab_list[c], round(pm, 4)) for a, b, c, pm in found]
        tally.add(dets, a_, fam or {})
    m = tally.metrics()
    m['frame_acc'] = right / max(1, total)
    return m


# ---------------------------------------------------------------- training

def fit(cfg, banks, vocab_list, anns, out_dir, log=print, unseen_keep=None):
    """cfg['select']: the val score that picks the checkpoint ('f1_all', or on banks['val_unseen']:
    'f1_unseen', 'f1_unseen_learnable' = only annotations of the glosses in unseen_keep)."""
    cfg = {**DEFAULTS, **cfg}
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    C, groups = len(vocab_list), cfg['groups']
    unk = vocab_list.index(TR.UNKNOWN) if TR.UNKNOWN in vocab_list else None
    face = cfg['face_drop'] < 1.0
    tr = banks['train']
    lab = tr.labels[tr.labels != IGNORE].astype(np.int64)
    counts = np.bincount(lab, minlength=C).astype(np.float64)
    w = np.where(counts > 0, np.maximum(counts, 1) ** -cfg['balance_power'], 0.0)
    w = w / (w * counts).sum() * counts.sum()             # mean frame weight 1
    weight = torch.tensor(w, dtype=torch.float32, device=device)

    model = SeqNet(len(FT.columns(groups)), C, cfg['dim'], cfg['stages'], cfg['convs'], cfg['kernel'], cfg['heads'],
                   cfg['drop'], cfg['head_drop']).to(device)
    if cfg['init']:
        init_from(model, cfg['init'], vocab_list, log)
    sampler = torch.utils.data.RandomSampler(tr, replacement=True, num_samples=len(tr) * cfg['repeats'])
    dl = torch.utils.data.DataLoader(SeqDataset(tr, train=True, groups=groups, face=face, crop=tuple(cfg['crop']),
                                                face_drop=cfg['face_drop'], splice=cfg['splice'],
                                                splice_power=cfg['splice_power'], fps_aug=cfg['fps_aug'],
                                                fps_min=cfg['fps_min'], spell=cfg['spell'],
                                                short=short_classes(vocab_list)),
                                     batch_size=cfg['batch'], sampler=sampler, num_workers=cfg['workers'],
                                     collate_fn=collate, drop_last=True, persistent_workers=cfg['workers'] > 0,
                                     pin_memory=device.type == 'cuda')
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    steps, warm = cfg['epochs'] * len(dl), max(1, cfg['warmup'] * len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, steps - warm)))))
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    n_params = sum(p.numel() for p in model.parameters())
    val = banks.get('val')
    has_val = val is not None and len(val) > 0
    val_unseen = banks.get('val_unseen')
    log(f'{device} | {n_params / 1e6:.2f} M params | {len(FT.columns(groups))} features | {C} labels | '
        f'{len(tr)} train recordings ({tr.n:,} frames, {len(lab):,} scored, blank {counts[0] / counts.sum():.1%}) | '
        f'{len(dl)} steps/epoch' + (f' | val {len(val)} recordings' if has_val else ' | no val'))

    history, best, best_ep = [], -1.0, 0
    for ep in range(1, cfg['epochs'] + 1):
        model.train()
        t0, tl, tc, tn = time.time(), 0.0, 0, 0
        for X, M, Y in dl:
            X, M, Y = X.to(device, non_blocking=True), M.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                logits = model(X, M)
            loss = F.cross_entropy(logits.float().reshape(-1, C), Y.reshape(-1), weight=weight,
                                   ignore_index=IGNORE, label_smoothing=cfg['label_smoothing'])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ok = Y != IGNORE
            tl += loss.item() * int(ok.sum())
            tc, tn = tc + int((logits.argmax(-1)[ok] == Y[ok]).sum()), tn + int(ok.sum())
        row = {'epoch': ep, 'lr': opt.param_groups[0]['lr'], 'train_loss': tl / max(1, tn), 'train_frame_acc': tc / max(1, tn)}
        if has_val:
            m = score(offline_probs(model, val, groups, face, device), val, anns, vocab_list, DECODE, unk)
            row.update({f'val_{k}': m[k] for k in ('frame_acc', 'precision', 'recall_all', 'f1_all', 'wer_all')})
            if val_unseen is not None and len(val_unseen):         # sentences no training recording has, streamed
                model.eval()
                pu = [stream_probs(model, r, a, groups, face, device, **(cfg['stream'] or STREAM))
                      for r, _, a, _ in (val_unseen.get(i) for i in range(len(val_unseen)))]
                m = score(pu, val_unseen, anns, vocab_list, DECODE, unk)
                row.update({'val_f1_unseen': m['f1_all'], 'val_wer_unseen': m['wer_all']})
                if unseen_keep is not None:                         # only words with training signs
                    m = score(pu, val_unseen, anns, vocab_list, DECODE, unk, keep=unseen_keep)
                    row.update({'val_f1_unseen_learnable': m['f1_all'], 'val_wer_unseen_learnable': m['wer_all']})
        row['sec'] = time.time() - t0
        key = float(row.get(f"val_{cfg['select']}", 0.0))
        key = key if math.isfinite(key) else 0.0          # no word found yet → F1 undefined
        row['best'] = key > best if not cfg['save_epoch'] else ep == cfg['save_epoch']
        if row['best']:
            best, best_ep = key, ep
            torch.save({'model': model.state_dict(), 'cfg': cfg, 'vocab': vocab_list, 'epoch': ep, 'kind': 'continuous',
                        'feature_version': FT.FEATURE_VERSION}, os.path.join(out_dir, 'best.pt'))
        history.append(row)
        with open(os.path.join(out_dir, 'history.csv'), 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(row))
            wr.writeheader()
            wr.writerows(history)
        log(' '.join(f'{k} {v:.4f}' if isinstance(v, float) else f'{k} {v}' for k, v in row.items() if k != 'lr'))
    ck = torch.load(os.path.join(out_dir, 'best.pt'), map_location=device)
    model.load_state_dict(ck['model'])
    model.eval()
    with open(os.path.join(out_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab_list, f, ensure_ascii=False)
    return model, {'best_epoch': best_ep, 'params': n_params, 'n_features': len(FT.columns(groups)), 'cfg': cfg}


def write_manifest(out_dir, cfg, vocab_list, result, n_in, onnx_ok, decode_p, stream=STREAM):
    """manifest.json next to model.onnx for the browser's continuous mode."""
    m = {
        'kind': 'continuous', 'feature_version': FT.FEATURE_VERSION, 'continuous_version': CONTINUOUS_VERSION,
        'model_file': 'model.onnx' if onnx_ok else None, 'opset': 17,
        'labels': vocab_list, 'blank_label': BLANK,
        'unknown_label': TR.UNKNOWN if TR.UNKNOWN in vocab_list else None,
        'input': {'name': 'x', 'shape': [1, 'T', n_in], 'dtype': 'float32', 'n_features': n_in,
                  'groups': cfg['groups'] or list(FT.GROUPS), 'max_len': None,
                  'padding': 'none: one window per call, every frame (no resampling)'},
        'output': {'name': 'logits', 'shape': [1, 'T', len(vocab_list)], 'note': 'per frame; softmax not applied'},
        'stream': dict(stream), 'decode': dict(decode_p),
        'face_drop': cfg['face_drop'], 'works_without_face': bool(cfg['face_drop']) or not TR.reads_face(cfg['groups']),
        'contract': FT.spec(),
        'run': os.path.basename(os.path.normpath(out_dir)), 'seed': cfg['seed'], 'split_id': cfg.get('split_id'),
        'best_epoch': result.get('best_epoch'), 'params': result.get('params'),
        'metrics': {k: v for k, v in result.items() if k.startswith(('test', 'val'))},
        'cfg': {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
    }
    with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(m, f, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- held-out evaluation

def decode_grid(grid):
    import itertools
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]


def evaluate(net, banks, anns, vocab_list, groups, face, device, grid, stream=STREAM, subsets=None,
             lookaheads=(), log=print, choose='f1_all', fixed=None, fps=None, fps_checks=()):
    """Streaming probabilities of every val / test recording (as the browser fixes them) → decode settings
    chosen on val (`choose`: 'f1_all' on all val recordings, or 'f1_<subset name>', e.g.
    'f1_unseen_learnable') → test scored once with them. subsets {name: (rec_ids, glosses or None)}: val and
    test scores on those recordings too, counting only those glosses' annotations if given.
    lookaheads: val scores with other delays. fixed: decode settings chosen before (no val is used).
    fps: every recording as a camera at that frame rate gives it (camera_resample). fps_checks: val and
    test scores again at these frame rates, with the decoding chosen at the normal one.
    Test detections are returned in out['test_detections']."""
    unk = vocab_list.index(TR.UNKNOWN) if TR.UNKNOWN in vocab_list else None
    net.eval()
    t0 = time.time()

    def probs(bank, at=fps, **st):
        out = []
        for i in range(len(bank)):
            raw, _, aspect, _ = bank.get(i)
            raw = camera_resample(raw, at) if at else raw
            out.append(stream_probs(net, raw, aspect, groups, face, device, **st))
        return out

    out = {'stream': dict(stream)}
    subsets = subsets or {}

    def part(name, bank, P):
        ids, keep = subsets[name]
        sel = [i for i, r in enumerate(bank.ids) if r.split('|')[0] in ids]
        return [P[i] for i in sel], subset(bank, ids), keep

    if fixed:                       # final model, val signers trained on: settings chosen before, test only
        best = {k: fixed[k] for k in grid}
        out.update(decode=best, val_grid=[])
    else:
        pv = probs(banks['val'], **stream)
        parts = {name: part(name, banks['val'], pv) for name in subsets}
        rows = []
        for p in decode_grid(grid):
            m = score(pv, banks['val'], anns, vocab_list, p, unk)
            row = {**p, **{k: m[k] for k in ('precision', 'recall_all', 'f1_all', 'wer_all', 'frame_acc')}}
            for name, (Pu, bu, keep) in parts.items():
                row[f'f1_{name}'] = score(Pu, bu, anns, vocab_list, p, unk, keep=keep)['f1_all']
            rows.append(row)
        assert choose in rows[0], f'{choose}: not a val score ({list(rows[0])})'
        rows.sort(key=lambda r: -r[choose])
        best = {k: rows[0][k] for k in grid}
        out.update(decode=best, val_grid=rows, val=score(pv, banks['val'], anns, vocab_list, best, unk))
        out['val_offline'] = score(offline_probs(net, banks['val'], groups, face, device), banks['val'], anns,
                                   vocab_list, best, unk)
        for name, (Pv, bv, keep) in parts.items():
            out[f'val_{name}'] = score(Pv, bv, anns, vocab_list, best, unk, keep=keep)
            out[f'val_{name}']['recordings'] = len(Pv)
        log(f'val, streaming: f1_all {out["val"]["f1_all"]:.4f} with {best} (offline {out["val_offline"]["f1_all"]:.4f}, '
            f'{time.time() - t0:.0f} s)')
        for la in lookaheads:
            st = {**stream, 'lookahead': la}
            out[f'val_lookahead_{la}'] = score(probs(banks['val'], **st), banks['val'], anns, vocab_list, best, unk)
    pt = probs(banks['test'], **stream)
    dets = []
    out['test'] = score(pt, banks['test'], anns, vocab_list, best, unk, dets_out=dets)
    out['test_detections'] = dets
    for name in subsets:
        Pt, bt, keep = part(name, banks['test'], pt)
        out[f'test_{name}'] = score(Pt, bt, anns, vocab_list, best, unk, keep=keep)
        out[f'test_{name}']['recordings'] = len(Pt)
    if len(banks.get('test_side', [])):
        out['test_side'] = score(probs(banks['test_side'], **stream), banks['test_side'], anns, vocab_list, best, unk)
    for f in fps_checks:
        for part_name in ('val', 'test') if not fixed else ('test',):
            out[f'{part_name}_fps{f}'] = score(probs(banks[part_name], at=f, **stream), banks[part_name], anns,
                                               vocab_list, best, unk)
    log('test, streaming: ' + ', '.join(f'{k} f1_all {v["f1_all"]:.4f} wer_all {v["wer_all"]:.4f}'
                                       for k, v in out.items() if k.startswith('test') and isinstance(v, dict)))
    return out
