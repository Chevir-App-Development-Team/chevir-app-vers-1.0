"""Fingerspelled letters for the app's letter mode (notebook 07): one hand's shape and movement, no body.

The letter mode spells words letter by letter with the hand tracker alone. Its model reads only the hand
(`GROUPS`: metric hand shape + hand movement), so the body and face are removed from every sample
(`strip`): photos rarely show a body, and the browser runs no body tracker in this mode. Three AzSLD sources:

  sentences  letter signs annotated in the sentence recordings (both cameras), the signer split. "O" is
             mostly the pronoun "o": an O counts as a letter only next to another spelled letter.
  photos     AzSLD_Fingerspelling static letters (one image each), split by contributor; a photo becomes
             a clip of PHOTO_LEN copies (augmentation adds per-frame jitter).
  clips      AzSLD_Fingerspelling moving letters: frames sampled from clips, put back on their frame
             numbers and filled in linearly; runs split where frames are more than MAX_GAP apart.

`<none>` (no letter) comes from the gaps between annotated signs, so the model can say "nothing".
`canonical` puts the hand that is present in more frames into slot 0: a lone hand in the browser (no
body tracker) is always slot 0, and right- and left-handed letters then look alike to the model.
"""
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import features as FT
from . import train as TR
from .store import LandmarkStore

LETTERS_VERSION = 1      # canonical() and the window rule; src/tercume/letters.js follows them
LETTERS = list('abcçdeəfgğhıijklmnoöpqrsştuüvxyz')
NONE = '<none>'
LABELS = LETTERS + [NONE]
DYNAMIC = set('çdgkyzöü')          # AzSLD's moving letters
GROUPS = ['hand_shape', 'hand_vel']
WINDOW = 24                        # frames the browser classifies at a time (0.8 s at 30 fps)
PHOTO_LEN = 24
MAX_GAP = 12
DEFAULTS = dict(TR.DEFAULTS, epochs=60, batch=64, dim=128, groups=GROUPS, margin=6, none_per_letter=1.0,
                save_epoch=None, select='letters_f1')


def letter_of(gloss):
    """A sentence gloss → its lowercase letter (İ → i, I → ı), or None."""
    if not isinstance(gloss, str) or len(gloss) != 1:
        return None
    ch = {'İ': 'i', 'I': 'ı'}.get(gloss, gloss.lower())
    return ch if ch in LETTERS else None


def letter_tags(tags):
    """Annotated letter signs of the sentence recordings (glosses_split rows) with a `letter` column."""
    t = tags.sort_values(['rec_id', 'start']).copy()
    t['letter'] = t.gloss.map(letter_of)
    spelled = t.letter.notna() & (t.letter != 'o')
    prev = spelled.groupby(t.rec_id).shift(1, fill_value=False)
    nxt = spelled.groupby(t.rec_id).shift(-1, fill_value=False)
    t.loc[(t.letter == 'o') & ~(prev | nxt), 'letter'] = None       # the pronoun "o"
    return t[t.letter.notna()]


def none_tags(tags, n_per_split, rng, min_gap=12, max_len=30):
    """Gaps of at least `min_gap` frames between annotated signs → `<none>` pseudo-signs (at most
    `max_len` frames, centred), n_per_split[split] of them drawn at random."""
    rows = []
    for rid, g in tags.sort_values(['rec_id', 'start']).groupby('rec_id'):
        ends, starts = g.end.to_numpy()[:-1], g.start.to_numpy()[1:]
        for a, b in zip(ends, starts):
            if b - a - 3 >= min_gap:
                lo, hi = int(a) + 2, int(b) - 2
                if hi - lo > max_len:
                    lo = (lo + hi - max_len) // 2
                    hi = lo + max_len
                rows.append((rid, lo, hi, g.split.iloc[0]))
    df = pd.DataFrame(rows, columns=['rec_id', 'start', 'end', 'split'])
    out = []
    for split, d in df.groupby('split'):
        n = min(len(d), int(n_per_split.get(split, 0)))
        out.append(d.iloc[rng.choice(len(d), n, replace=False)] if n else d.iloc[:0])
    out = pd.concat(out, ignore_index=True) if out else df.iloc[:0]
    out['gloss'] = NONE
    out['key'] = ['none:' + r + f':{s}' for r, s in zip(out.rec_id, out.start)]
    return out


def strip(raw):
    """No body, no face: what the letter mode's hand tracker gives (NaN = not detected)."""
    out = dict(raw)
    for k in ('pose', 'face', 'blend'):
        out[k] = np.full_like(raw[k], np.nan)
    return out


def canonical(raw):
    """Slot 0 = the hand present in more frames (ties keep the order)."""
    n = np.isfinite(raw['hand'][:, :, 0, 0]).sum(0)
    if n[1] <= n[0]:
        return raw
    out = dict(raw)
    out['hand'] = raw['hand'][:, ::-1].copy()
    out['hand_world'] = raw['hand_world'][:, ::-1].copy()
    return out


def preprocess(raw, aspect, groups=GROUPS):
    """THE way from landmarks to the letter model's input: strip, canonical, features.preprocess."""
    return FT.preprocess(canonical(strip(raw)), aspect, groups, face=False)


def _strip_bank(bank):
    for k in ('pose', 'face', 'blend'):
        bank.data[k] = np.full_like(bank.data[k], np.nan)
    return bank


def load_sentences(sentence_dirs, tags, index, margin, side=None, log=print):
    """Letter signs (+ `<none>` rows of `tags`) → banks per split, frontal and side views, body removed."""
    t = tags.copy()
    t['gloss'] = t.letter.where(t.gloss != NONE, NONE) if 'letter' in t else t.gloss
    banks = TR.load_sentence_signs(sentence_dirs, t, index, margin=margin, side=side, log=log)
    return {k: _strip_bank(b) for k, b in banks.items()}


def load_photos(fs_dirs, fsplit, index, log=print):
    """Static letter photos → a bank per split: each photo repeated PHOTO_LEN frames."""
    store = LandmarkStore(fs_dirs)
    banks = {s: TR.Bank() for s in ('train', 'val', 'test')}
    rows = fsplit[(fsplit.kind == 'static') & fsplit.split.notna() & fsplit.id.isin(store.index.index)]
    for r in rows.itertuples():
        raw = FT.reduce(store.get(r.id, dtype=np.float16))
        if not np.isfinite(raw['hand'][0, :, 0, 0]).any():
            continue
        aspect = float(store.index.at[r.id, 'width']) / float(store.index.at[r.id, 'height'])
        banks[r.split].add({k: np.repeat(v, PHOTO_LEN, 0) for k, v in strip(raw).items()}, index[r.letter],
                           aspect, (0, PHOTO_LEN), 'photo:' + r.id)
    log('photos: ' + ', '.join(f'{s} {len(b)}' for s, b in banks.items()))
    return {s: b.done() for s, b in banks.items()}


def fill_in(frames, times):
    """Frames (dict of T=1 arrays) at increasing frame numbers → every frame in between, linearly."""
    t0 = times[0]
    out = {k: [] for k in frames[0]}
    for (fa, ta), (fb, tb) in zip(zip(frames, times), zip(frames[1:], times[1:])):
        for t in range(ta, tb):
            w = (t - ta) / (tb - ta)
            for k in out:
                out[k].append(fa[k][0].astype(np.float32) * (1 - w) + fb[k][0].astype(np.float32) * w)
    for k in out:
        out[k].append(frames[-1][k][0].astype(np.float32))
    assert all(len(v) == times[-1] - t0 + 1 for v in out.values())
    return {k: np.stack(v).astype(np.float16) for k, v in out.items()}


def load_clips(fs_dirs, fsplit, index, max_gap=MAX_GAP, min_len=6, log=print):
    """Moving-letter frames → clips per run (frames > max_gap apart start a new clip)."""
    store = LandmarkStore(fs_dirs)
    banks = {s: TR.Bank() for s in ('train', 'val', 'test')}
    rows = fsplit[(fsplit.kind == 'dynamic') & fsplit.split.notna() & fsplit.id.isin(store.index.index)
                  & (fsplit.frame_no >= 0)]
    n_seg = 0
    for run, g in rows.sort_values('frame_no').groupby('run'):
        g = g.drop_duplicates('frame_no')
        cut = np.flatnonzero(np.diff(g.frame_no.to_numpy()) > max_gap) + 1
        for seg in np.split(np.arange(len(g)), cut):
            s = g.iloc[seg]
            if len(s) < 2 or s.frame_no.iloc[-1] - s.frame_no.iloc[0] + 1 < min_len:
                continue
            frames = [strip(FT.reduce(store.get(i, dtype=np.float16))) for i in s.id]
            raw = fill_in(frames, [int(f) for f in s.frame_no])
            aspect = float(store.index.at[s.id.iloc[0], 'width']) / float(store.index.at[s.id.iloc[0], 'height'])
            T = raw['hand'].shape[0]
            banks[s.split.iloc[0]].add(raw, index[s.letter.iloc[0]], aspect, (0, T), f'clip:{run}:{seg[0]}')
            n_seg += 1
    log(f'clips: {n_seg} from {rows.run.nunique()} runs: ' + ', '.join(f'{s} {len(b)}' for s, b in banks.items()))
    return {s: b.done() for s, b in banks.items()}


def source_of(sample_id):
    s = str(sample_id)
    return ('none' if s.startswith('none:') else 'photo' if s.startswith('photo:') else
            'clip' if s.startswith('clip:') else 'side' if s.endswith('|side') else 'sentence')


class LetterDataset(torch.utils.data.Dataset):
    """Train: the word model's augmentation (crop jitter, speed, mirror, rotation, per-frame jitter, hand
    dropout), then canonical. Eval: the sample's core through `preprocess`, as the browser does."""

    def __init__(self, bank, train=False, groups=GROUPS):
        self.bank, self.train, self.groups, self.cols = bank, train, groups, FT.columns(groups)

    def __len__(self):
        return len(self.bank)

    def __getitem__(self, i):
        raw, label, aspect, (lo, hi) = self.bank.raw(i)
        if not self.train:
            return torch.from_numpy(preprocess({k: v[lo:hi] for k, v in raw.items()}, aspect, self.groups)), label
        rng = np.random.default_rng()
        d = canonical(FT.augment(FT.to_square(raw, aspect), rng, aspect, (lo, hi)))
        return torch.from_numpy(FT.features(d)[:, self.cols]), label


@torch.no_grad()
def predict(model, bank, device, groups=GROUPS, batch=128, workers=2):
    model.eval()
    dl = torch.utils.data.DataLoader(LetterDataset(bank, groups=groups), batch_size=batch, shuffle=False,
                                     num_workers=workers, collate_fn=TR.collate)
    out = []
    for x, m, _ in dl:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            out.append(model(x.to(device), m.to(device)).float().softmax(1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, len(LABELS)), np.float32)


def letter_scores(probs, y, ids, letters_only=True):
    """Scores per source and over all; letters_only: the argmax among the letters (a sample that is a
    letter is never answered `<none>`), scored on letter samples only."""
    none = LABELS.index(NONE)
    keep = y != none
    out = {}
    if letters_only:
        p = probs.copy()
        p[:, none] = -1
    else:
        p = probs
    src = np.array([source_of(i) for i in ids])
    for name, sel in [('all', keep)] + [(s, keep & (src == s)) for s in ('sentence', 'side', 'photo', 'clip')]:
        if sel.sum():
            out[name] = TR.scores(p[sel], y[sel])
    if (~keep).any():                                    # how often a gap is called a letter
        out['none_as_letter'] = float((probs[~keep].argmax(1) != none).mean())
        out['letter_as_none'] = float((probs[keep].argmax(1) == none).mean())
    return out


def fit(cfg, banks, out_dir, log=print):
    cfg = {**DEFAULTS, **cfg}
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    C, groups = len(LABELS), cfg['groups']
    tr = banks['train']
    counts = np.bincount(tr.labels, minlength=C)
    weights = counts[tr.labels].astype(np.float64) ** -cfg['balance_power']
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(weights), num_samples=len(tr), replacement=True)
    dl = torch.utils.data.DataLoader(LetterDataset(tr, train=True, groups=groups), batch_size=cfg['batch'],
                                     sampler=sampler, num_workers=cfg['workers'], collate_fn=TR.collate,
                                     drop_last=True, persistent_workers=cfg['workers'] > 0,
                                     pin_memory=device.type == 'cuda')
    model = TR.Net(len(FT.columns(groups)), C, cfg['dim'], cfg['stages'], cfg['convs'], cfg['kernel'], cfg['heads'],
                   cfg['drop'], cfg['head_drop']).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    steps, warm = cfg['epochs'] * len(dl), max(1, cfg['warmup'] * len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, steps - warm)))))
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    val = banks.get('val')
    has_val = val is not None and len(val) > 0
    log(f'{device} | {n_params / 1e6:.2f} M params | {len(FT.columns(groups))} features | {C} labels | '
        f'{len(tr)} train samples ({dict(zip(*np.unique([source_of(i) for i in tr.ids], return_counts=True)))}) | '
        + (f'val {len(val)}' if has_val else 'no val'))
    history, best, best_ep = [], -1.0, 0
    for ep in range(1, cfg['epochs'] + 1):
        model.train()
        t0, tl, tc, tn = time.time(), 0.0, 0, 0
        for x, m, y in dl:
            x, m, y = x.to(device, non_blocking=True), m.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                logits = model(x, m)
            loss = F.cross_entropy(logits.float(), y, label_smoothing=cfg['label_smoothing'])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl, tc, tn = tl + loss.item() * len(y), tc + int((logits.argmax(1) == y).sum()), tn + len(y)
        row = {'epoch': ep, 'train_loss': tl / tn, 'train_acc': tc / tn}
        if has_val:
            s = letter_scores(predict(model, val, device, groups), val.labels, val.ids)
            row.update({f'val_{k}_{m}': v[m] for k, v in s.items() if isinstance(v, dict) for m in ('top1', 'macro_f1')})
            row.update({f'val_{k}': v for k, v in s.items() if not isinstance(v, dict)})
            row['val_letters_f1'] = s['all']['macro_f1']
        row['sec'] = time.time() - t0
        key = float(row.get(f"val_{cfg['select']}", 0.0))
        row['best'] = key > best if not cfg['save_epoch'] else ep == cfg['save_epoch']
        if row['best']:
            best, best_ep = key, ep
            torch.save({'model': model.state_dict(), 'cfg': cfg, 'labels': LABELS, 'epoch': ep, 'kind': 'letters',
                        'feature_version': FT.FEATURE_VERSION, 'letters_version': LETTERS_VERSION},
                       os.path.join(out_dir, 'best.pt'))
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(out_dir, 'history.csv'), index=False)
        log(' '.join(f'{k} {v:.4f}' if isinstance(v, float) else f'{k} {v}' for k, v in row.items()))
    ck = torch.load(os.path.join(out_dir, 'best.pt'), map_location=device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, {'best_epoch': best_ep, 'params': n_params, 'n_features': len(FT.columns(groups)), 'cfg': cfg}


def write_manifest(out_dir, cfg, result, onnx_ok):
    m = {
        'kind': 'letters', 'feature_version': FT.FEATURE_VERSION, 'letters_version': LETTERS_VERSION,
        'model_file': 'model.onnx' if onnx_ok else None, 'opset': 17,
        'labels': LABELS, 'none_label': NONE, 'dynamic': sorted(DYNAMIC),
        'input': {'name': 'x', 'shape': [1, 'T', result['n_features']], 'dtype': 'float32',
                  'n_features': result['n_features'], 'groups': cfg['groups'], 'max_len': FT.MAX_LEN,
                  'window': WINDOW, 'body': 'none (pose, face and blendshapes are NaN)',
                  'slots': 'canonical: the hand present in more frames of the window is slot 0'},
        'output': {'name': 'logits', 'shape': [1, len(LABELS)], 'note': 'softmax not applied'},
        'contract': FT.spec(),
        'run': os.path.basename(os.path.normpath(out_dir)), 'seed': cfg['seed'], 'split_id': cfg.get('split_id'),
        'best_epoch': result.get('best_epoch'), 'params': result.get('params'),
        'metrics': {k: v for k, v in result.items() if k.startswith(('test', 'val'))},
        'cfg': {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
    }
    with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(m, f, ensure_ascii=False, indent=1, default=str)


# ---------------------------------------------------------------- the app's previous letter model

class OldModel:
    """The letter mode's model until now (public/tercume/assets/model.json + model.bin, src/tercume/model.js):
    a dense network on 20 frames × the first hand's 21 image points (x, y, z), zeros where no hand."""

    def __init__(self, meta_path, bin_path):
        with open(meta_path, encoding='utf-8') as f:
            self.meta = json.load(f)
        w = np.fromfile(bin_path, dtype='<f4')
        self.layers, off = [], 0
        for l, act in zip(self.meta['layers'], self.meta['activations']):
            W = w[off:off + l['in'] * l['out']].reshape(l['in'], l['out'])
            off += l['in'] * l['out']
            self.layers.append((W, w[off:off + l['out']], act))
            off += l['out']
        assert off == len(w), 'weights do not match the layers'
        self.labels = self.meta['labels']

    def window(self, raw, lo, hi):
        """The sample's core resampled to 20 frames; per frame slot 0's hand if present, else slot 1's."""
        idx = np.round(np.linspace(lo, hi - 1, self.meta['input']['frames'])).astype(int)
        h = raw['hand'][idx].astype(np.float32)
        x = np.where(np.isfinite(h[:, 0, :1, :1]), h[:, 0], h[:, 1])
        return np.nan_to_num(x, nan=0.0).reshape(-1)

    def probs(self, X):
        for W, b, act in self.layers:
            X = X @ W + b
            if act == 'relu':
                X = np.maximum(X, 0)
        e = np.exp(X - X.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict_bank(self, bank):
        """→ probabilities over LABELS (its letters mapped in, the control gestures dropped)."""
        X = np.stack([self.window(*(lambda r: (r[0], *r[3]))(bank.raw(i))) for i in range(len(bank))]) \
            if len(bank) else np.zeros((0, 1260), np.float32)
        p = self.probs(X)
        out = np.zeros((len(p), len(LABELS)), np.float32)
        for j, lab in enumerate(self.labels):
            if lab in LABELS:
                out[:, LABELS.index(lab)] = p[:, j]
        return out
