"""Phase 2: isolated-sign recognition on landmarks (PyTorch), signer-independent.

Data
  train   every annotated sign of the 200-word vocabulary cut from the sentence recordings of
          train signers, with a margin of real context around it (augment() crops inside it)
  val     reported every epoch: the Words_200 clips of val signers — the benchmark — and the
          sentence-cut signs of val signers (a bigger, steadier estimate)
  test    scored once, with the best-on-val checkpoint
Model     Conv1D (depthwise, ECA) + Transformer blocks, as in the top Kaggle isolated-sign
          solutions; ~1.7 M parameters, exported to ONNX for the browser.
Classes are very uneven (MƏN ≈ 30% of Words_200), so batches are drawn with weights n_c^-0.5
and checkpoints are chosen by val macro-F1; plain accuracy is always shown next to the
majority-class baseline.

Unknown signs (cfg 'unknown' > 0): sentence signs outside the vocabulary become one extra class,
UNKNOWN, so word mode can say "not one of my words" instead of naming the closest one. Their
sampling weight is the usual n^-0.5 times cfg['unknown']. Val/test out-of-vocabulary signs are kept
in separate banks (val_unk, test_unk): the word and sign metrics stay comparable with earlier runs,
and the open-set ones are reported next to them.
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
from .store import LandmarkStore

KEYS = ('hand', 'hand_world', 'pose', 'face', 'blend')
UNKNOWN = '<unk>'          # the extra class for out-of-vocabulary signs (last label of the vocabulary)
DEFAULTS = dict(epochs=60, batch=64, lr=1e-3, weight_decay=0.05, warmup=3, label_smoothing=0.1,
                balance_power=0.5, dim=192, stages=2, convs=3, kernel=17, heads=4, drop=0.2,
                head_drop=0.4, margin=8, workers=4, seed=0, groups=None,
                face_drop=0.0,          # P(train clip loses face + blendshapes): one model for both privacy modes
                unknown=0.0,            # sampling weight of the UNKNOWN class (0 = known words only)
                select='words',         # checkpoint on val Words_200 macro-F1; 'open': val open-set accuracy
                fold=None, split_id=None)


# ---------------------------------------------------------------- data

class Bank:
    """Compact raw landmarks of many samples: one concatenated array per field + an index."""

    def __init__(self):
        self.chunks, self.rows, self.n = {k: [] for k in KEYS}, [], 0

    def add(self, raw, label, aspect, core, sample_id):
        T = raw['hand'].shape[0]
        for k in KEYS:
            self.chunks[k].append(raw[k])
        self.rows.append((self.n, T, int(label), float(aspect), int(core[0]), int(core[1]), sample_id))
        self.n += T

    def done(self):
        self.data = {k: np.concatenate(v) if v else np.zeros((0,), np.float16) for k, v in self.chunks.items()}
        self.chunks = None
        self.labels = np.array([r[2] for r in self.rows], np.int64)
        self.ids = [r[6] for r in self.rows]
        return self

    def __len__(self):
        return len(self.rows)

    def raw(self, i):
        off, T, label, aspect, lo, hi, _ = self.rows[i]
        return {k: self.data[k][off:off + T] for k in KEYS}, label, aspect, (lo, hi)


def ambiguous_oov(glosses, vocab_words, prefix=4):
    """Out-of-vocabulary glosses that share their first `prefix` letters with a vocabulary word: mostly
    other forms or misspellings of it (İSTƏYİR ~ İSTƏMƏK, QARABAQ ~ QARABAĞ, ALMALIYAM ~ ALMAQ), so
    probably the same sign. They are left out of the unknown class instead of teaching a contradiction."""
    stems = {v[:prefix] for v in vocab_words if len(v) >= prefix}
    return sorted({g for g in glosses if g not in vocab_words and len(g) >= prefix and g[:prefix] in stems})


def load_sentence_signs(sentence_dirs, tags, vocab, margin=8, log=print, unknown=None, exclude=()):
    """tags (glosses_split.csv rows) → Bank per split, sliced from whole-recording landmarks.
    unknown: a label in vocab (UNKNOWN) → out-of-vocabulary signs are kept too, with that label: in
    'train' for training, in 'val_unk' / 'test_unk' for the open-set metrics. exclude: glosses left
    out entirely (ambiguous_oov)."""
    store = LandmarkStore(sentence_dirs)
    idx = store.index
    n0 = len(tags)
    known = (tags.gloss.isin(vocab) if unknown is None else tags.gloss.notna()) & ~tags.gloss.isin(list(exclude))
    tags = tags[tags.rec_id.isin(idx.index) & known & tags.start.notna() & tags.end.notna()].copy()
    if len(tags) < n0:
        log(f'  {n0 - len(tags)} signs dropped (no landmarks, outside the vocabulary or no frame range)')
    tags['_order'] = list(zip(tags.rec_id.map(idx['dir']), tags.rec_id.map(idx['part'])))
    banks = {s: Bank() for s in ('train', 'val', 'test')}
    if unknown is not None:
        banks.update(val_unk=Bank(), test_unk=Bank())
    t0, skipped = time.time(), 0
    for n, (rid, g) in enumerate(tags.sort_values(['_order', 'rec_id']).groupby('rec_id', sort=False)):
        raw = FT.reduce(store.get(rid, dtype=np.float16))
        T = raw['hand'].shape[0]
        aspect = float(idx.at[rid, 'width']) / float(idx.at[rid, 'height'])
        for r in g.itertuples():
            s0, e0 = int(r.start) - 1, min(int(r.end), T)        # 1-based inclusive → [s0, e0)
            if s0 < 0 or e0 - s0 < 2:
                skipped += 1
                continue
            lo, hi = max(0, s0 - margin), min(T, e0 + margin)
            oov = r.gloss not in vocab
            bank = banks[r.split if not oov or r.split == 'train' else f'{r.split}_unk']
            bank.add({k: v[lo:hi] for k, v in raw.items()}, vocab[unknown if oov else r.gloss], aspect,
                     (s0 - lo, e0 - lo), r.key)
        if n and n % 2000 == 0:
            log(f'  {n} recordings sliced ({time.time() - t0:.0f} s)')
    log('sentence signs: ' + ', '.join(f'{s} {len(b)}' for s, b in banks.items()) + f' (skipped {skipped})')
    return {s: b.done() for s, b in banks.items()}


def load_words200(words_dirs, wsplit, vocab, log=print):
    store = LandmarkStore(words_dirs)
    idx = store.index
    w = wsplit[wsplit.key.isin(idx.index)].copy()
    w['_order'] = list(zip(w.key.map(idx['dir']), w.key.map(idx['part'])))
    banks = {s: Bank() for s in ('train', 'val', 'test')}
    for r in w.sort_values('_order').itertuples():
        raw = FT.reduce(store.get(r.key, dtype=np.float16))
        T = raw['hand'].shape[0]
        if T < 2:
            continue
        aspect = float(idx.at[r.key, 'width']) / float(idx.at[r.key, 'height'])
        banks[r.split].add(raw, vocab[r.gloss], aspect, (0, T), r.key)
    log('Words_200 clips: ' + ', '.join(f'{s} {len(b)}' for s, b in banks.items()))
    return {s: b.done() for s, b in banks.items()}


class SignDataset(torch.utils.data.Dataset):
    """Evaluation goes through features.preprocess() — the exact code the landmark server runs and the
    browser port mirrors. Training adds augmentation (and face dropout) on the same functions."""

    def __init__(self, bank, train=False, flip=False, groups=None, face=True, face_drop=0.0):
        self.bank, self.train, self.flip, self.groups, self.face, self.face_drop = bank, train, flip, groups, face, face_drop
        self.cols = FT.columns(groups) if groups else None

    def __len__(self):
        return len(self.bank)

    def __getitem__(self, i):
        raw, label, aspect, core = self.bank.raw(i)
        lo, hi = core
        if not self.train and not self.flip:
            x = FT.preprocess({k: v[lo:hi] for k, v in raw.items()}, aspect, self.groups, self.face)
            return torch.from_numpy(x), label
        sq = FT.to_square(raw, aspect)
        if self.train:
            rng = np.random.default_rng()
            d = FT.augment(sq, rng, aspect, core)
            if self.face_drop and rng.random() < self.face_drop:
                d = FT.drop_face(d)
        else:                                     # mirror TTA
            d = FT.flip({k: v[lo:hi] for k, v in sq.items()}, aspect)
            if not self.face:
                d = FT.drop_face(d)
        x = FT.features(d)
        return torch.from_numpy(x if self.cols is None else x[:, self.cols]), label


def collate(batch):
    T = max(x.shape[0] for x, _ in batch)
    X = torch.zeros(len(batch), T, batch[0][0].shape[1])
    M = torch.zeros(len(batch), T, dtype=torch.bool)
    for i, (x, _) in enumerate(batch):
        X[i, :len(x)] = x
        M[i, :len(x)] = True
    return X, M, torch.tensor([y for _, y in batch])


# ---------------------------------------------------------------- model

def _mask(x, valid):
    """B×T×1 float mask; without `valid` all ones (shape taken from x, so ONNX keeps T dynamic)."""
    return torch.ones_like(x[..., :1]) if valid is None else valid[:, :, None].to(x.dtype)


class ECA(nn.Module):
    """Efficient channel attention over the masked time average."""

    def __init__(self, k=5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x, m):                    # x B×C×T, m B×1×T
        y = (x * m).sum(-1) / m.sum(-1).clamp(min=1)
        return x * torch.sigmoid(self.conv(y[:, None, :]))[:, 0, :, None]


class ConvBlock(nn.Module):
    def __init__(self, dim, k=17, expand=2, drop=0.2):
        super().__init__()
        h = dim * expand
        self.norm, self.pw1 = nn.LayerNorm(dim), nn.Linear(dim, h)
        self.dw = nn.Conv1d(h, h, k, padding=k // 2, groups=h)
        self.bn, self.eca, self.pw2, self.drop = nn.BatchNorm1d(h), ECA(), nn.Linear(h, dim), nn.Dropout(drop)

    def forward(self, x, valid):
        m = _mask(x, valid)
        y = F.silu(self.pw1(self.norm(x))) * m
        y = y.transpose(1, 2)
        y = self.eca(F.silu(self.bn(self.dw(y))), m.transpose(1, 2))
        return x + self.drop(self.pw2(y.transpose(1, 2))) * m


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, drop=0.2):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Dropout(drop), nn.Linear(dim * 2, dim))
        self.drop = nn.Dropout(drop)

    def forward(self, x, valid):
        h = self.n1(x)
        if valid is None:
            a = self._attend(h)
        else:
            a, _ = self.attn(h, h, h, key_padding_mask=~valid, need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.n2(x)))

    def _attend(self, h):
        """self.attn without a mask, written out with its own weights. Its forward() bakes the traced
        length into a reshape, so the exported ONNX would accept only that many frames."""
        mha = self.attn
        qkv = F.linear(h, mha.in_proj_weight, mha.in_proj_bias).unflatten(-1, (3, mha.num_heads, -1))
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)                     # each B×heads×T×d
        w = (q @ k.transpose(-1, -2)) * (mha.head_dim ** -0.5)
        return mha.out_proj((w.softmax(-1) @ v).transpose(1, 2).flatten(2))


class Net(nn.Module):
    def __init__(self, n_feat, n_cls, dim=192, stages=2, convs=3, kernel=17, heads=4, drop=0.2, head_drop=0.4):
        super().__init__()
        self.stem, self.stem_norm = nn.Linear(n_feat, dim), nn.LayerNorm(dim)
        blocks = []
        for _ in range(stages):
            blocks += [ConvBlock(dim, kernel, drop=drop) for _ in range(convs)] + [TransformerBlock(dim, heads, drop)]
        self.blocks = nn.ModuleList(blocks)
        self.norm, self.drop, self.head = nn.LayerNorm(dim), nn.Dropout(head_drop), nn.Linear(dim, n_cls)

    def forward(self, x, valid=None):           # valid B×T; None = every frame is real (one clip)
        m = _mask(x, valid)
        h = self.stem_norm(self.stem(x)) * m
        for b in self.blocks:
            h = b(h, valid)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.drop(self.norm(pooled)))


class Exportable(nn.Module):
    """One un-padded clip in, logits out — what the browser runs."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):                        # 1×T×N_FEAT
        return self.net(x)


# ---------------------------------------------------------------- metrics

def scores(logits: np.ndarray, y: np.ndarray) -> dict:
    """top-1/5, and macro-F1 / balanced accuracy over the classes present in y."""
    pred = logits.argmax(1)
    top5 = np.argsort(-logits, 1)[:, :5]
    f1s, recalls = [], []
    for c in np.unique(y):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        recalls.append(r)
    return {'top1': float((pred == y).mean()), 'top5': float((top5 == y[:, None]).any(1).mean()),
            'macro_f1': float(np.mean(f1s)), 'balanced_acc': float(np.mean(recalls)), 'n': int(len(y))}


@torch.no_grad()
def predict(model, bank, device, batch=128, workers=2, tta=False, groups=None, face=True):
    model.eval()
    outs = []
    for flip in ([False, True] if tta else [False]):
        dl = torch.utils.data.DataLoader(SignDataset(bank, flip=flip, groups=groups, face=face), batch_size=batch, shuffle=False,
                                         num_workers=workers, collate_fn=collate)
        logits = []
        for x, m, _ in dl:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                logits.append(model(x.to(device), m.to(device)).float().softmax(1).cpu())
        outs.append(torch.cat(logits).numpy())
    return np.mean(outs, 0)


def reads_face(groups) -> bool:
    """Does the model input change when the face is missing? Also true for 'flags' alone: one of the
    four presence flags is "face detected", so a hands + pose model still sees the face tracker."""
    return not groups or bool({'face', 'blend', 'flags'} & set(groups))


# ---------------------------------------------------------------- training

def fit(cfg, signs, words, vocab_list, out_dir, log=print):
    """signs/words: dict split → Bank. Writes history.csv, best.pt, metrics.json, … to out_dir."""
    cfg = {**DEFAULTS, **cfg}
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    C = len(vocab_list)
    tr = signs['train']
    counts = np.bincount(tr.labels, minlength=C)
    weights = counts[tr.labels].astype(np.float64) ** -cfg['balance_power']
    unk = vocab_list.index(UNKNOWN) if UNKNOWN in vocab_list else None
    if unk is not None:
        weights[tr.labels == unk] *= cfg['unknown']          # 0: this run learns the known words only
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(weights), num_samples=len(tr), replacement=True)
    cols, groups = FT.columns(cfg['groups']), cfg['groups']
    dl = torch.utils.data.DataLoader(SignDataset(tr, train=True, groups=groups, face_drop=cfg['face_drop']),
                                     batch_size=cfg['batch'], sampler=sampler,
                                     num_workers=cfg['workers'], collate_fn=collate, drop_last=True,
                                     persistent_workers=cfg['workers'] > 0, pin_memory=device.type == 'cuda')
    evals = {k: b for k, b in (('val_words', words.get('val')), ('val_signs', signs.get('val')),
                               ('val_unk', signs.get('val_unk') if unk is not None else None)) if b is not None and len(b)}
    major = int(np.bincount(tr.labels[tr.labels != unk] if unk is not None else tr.labels, minlength=C).argmax())

    model = Net(len(cols), C, cfg['dim'], cfg['stages'], cfg['convs'], cfg['kernel'], cfg['heads'],
                cfg['drop'], cfg['head_drop']).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    steps, warm = cfg['epochs'] * len(dl), max(1, cfg['warmup'] * len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, steps - warm)))))
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')

    log(f'{device} | {n_params / 1e6:.2f} M params | {len(cols)} features | {len(tr)} train signs | {C} classes | '
        f'{len(dl)} steps/epoch | majority class "{vocab_list[major]}"' +
        (f' | {int((tr.labels == unk).sum())} unknown train signs, weight × {cfg["unknown"]}' if unk is not None else ''))
    for k, b in evals.items():
        log(f'  {k}: {len(b)} samples, majority-class accuracy {(b.labels == major).mean():.3f}')
    hist_path = os.path.join(out_dir, 'history.csv')
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
        row = {'epoch': ep, 'lr': opt.param_groups[0]['lr'], 'train_loss': tl / tn, 'train_acc': tc / tn}
        for k, b in evals.items():
            p = predict(model, b, device, workers=min(2, cfg['workers']), groups=groups)
            if k == 'val_unk':
                row['val_unk_rejected'] = float((p.argmax(1) == unk).mean())
                continue
            row.update({f'{k}_{m}': v for m, v in scores(p, b.labels).items() if m != 'n'})
            if unk is not None:
                row[f'{k}_unk_rate'] = float((p.argmax(1) == unk).mean())
                if k == 'val_signs':
                    row['_val_signs_correct'] = int((p.argmax(1) == b.labels).sum())
        if 'val_unk_rejected' in row and '_val_signs_correct' in row:   # known signs right + unknown rejected
            nk, nu = len(evals['val_signs']), len(evals['val_unk'])
            row['val_open_acc'] = (row.pop('_val_signs_correct') + row['val_unk_rejected'] * nu) / (nk + nu)
        row.pop('_val_signs_correct', None)
        row['sec'] = time.time() - t0
        key = 'val_words_macro_f1' if 'val_words' in evals else 'val_signs_macro_f1'
        if cfg['select'] == 'open' and 'val_open_acc' in row:     # unknown class: rejection counts too
            key = 'val_open_acc'
        row['best'] = row.get(key, 0) > best
        if row['best']:
            best, best_ep = row[key], ep
            torch.save({'model': model.state_dict(), 'cfg': cfg, 'vocab': vocab_list, 'epoch': ep,
                        'feature_version': FT.FEATURE_VERSION}, os.path.join(out_dir, 'best.pt'))
        history.append(row)
        with open(hist_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(history)
        log(_line(row, ep == 1))

    # test once, with the best-on-val weights (plain and with mirror TTA)
    ck = torch.load(os.path.join(out_dir, 'best.pt'), map_location=device)
    model.load_state_dict(ck['model'])
    result = {'best_epoch': best_ep, 'params': n_params, 'n_features': int(len(cols)), 'cfg': cfg,
              'majority_class': vocab_list[major]}
    preds = {}
    for name, bank in (('test_words', words.get('test')), ('test_signs', signs.get('test')),
                       ('val_words', words.get('val')), ('val_signs', signs.get('val')),
                       ('test_unk', signs.get('test_unk')), ('val_unk', signs.get('val_unk'))):
        if bank is None or not len(bank) or (name.endswith('_unk') and unk is None):
            continue
        nw = min(2, cfg['workers'])
        p = predict(model, bank, device, workers=nw, groups=groups)
        preds.update({f'{name}_probs': p.astype(np.float16), f'{name}_labels': bank.labels,
                      f'{name}_ids': np.array(bank.ids)})
        if name.endswith('_unk'):                            # out-of-vocabulary signs: rejected?
            result[name] = {'rejected': float((p.argmax(1) == unk).mean()),
                            'mean_p_unknown': float(p[:, unk].mean()), 'n': int(len(bank))}
            continue
        result[name] = scores(p, bank.labels)
        if unk is not None:                                  # known words wrongly rejected; scores without UNKNOWN
            result[name]['unk_rate'] = float((p.argmax(1) == unk).mean())
            pk = p.copy()
            pk[:, unk] = 0
            result[name + '_known'] = scores(pk, bank.labels)
        result[name]['majority_acc'] = float((bank.labels == major).mean())
        result[name + '_tta'] = scores(predict(model, bank, device, workers=nw, tta=True, groups=groups), bank.labels)
        if reads_face(groups):                                      # privacy mode: same model, no face points
            result[name + '_noface'] = scores(predict(model, bank, device, workers=nw, groups=groups, face=False),
                                              bank.labels)
        if name == 'test_words':
            _per_class(p, bank.labels, vocab_list, out_dir)
    for sp in ('val', 'test'):                               # open set: known signs right + unknown rejected
        if f'{sp}_unk' in result and f'{sp}_signs' in result:
            nk, nu = result[f'{sp}_signs']['n'], result[f'{sp}_unk']['n']
            result[f'{sp}_open_acc'] = (result[f'{sp}_signs']['top1'] * nk + result[f'{sp}_unk']['rejected'] * nu) / (nk + nu)
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab_list, f, ensure_ascii=False)
    # per-sample probabilities, for analyses without retraining (e.g. merging inflected word forms)
    np.savez_compressed(os.path.join(out_dir, 'predictions.npz'), **preds)
    ok = export_onnx(model, os.path.join(out_dir, 'model.onnx'), log, n_in=len(cols))
    write_manifest(out_dir, cfg, vocab_list, result, n_in=len(cols), onnx_ok=ok)
    return result


def write_manifest(out_dir, cfg, vocab_list, result, n_in, onnx_ok=True):
    """manifest.json next to model.onnx: everything the browser or the landmark server needs to feed
    the model and to refuse a mismatch (feature_version), plus where the model came from."""
    m = {
        'feature_version': FT.FEATURE_VERSION,
        'model_file': 'model.onnx' if onnx_ok else None, 'opset': 17,
        'labels': vocab_list,
        'input': {'name': 'x', 'shape': [1, 'T', n_in], 'dtype': 'float32', 'n_features': n_in,
                  'groups': cfg['groups'] or list(FT.GROUPS), 'max_len': FT.MAX_LEN,
                  'padding': 'none — one clip per call (T frames, T <= max_len after preprocess)'},
        'output': {'name': 'logits', 'shape': [1, len(vocab_list)], 'note': 'softmax not applied; '
                   'ensembles average the softmax probabilities'},
        # without face = trained with the face randomly removed, or an input the face cannot change
        'face_drop': cfg['face_drop'], 'works_without_face': bool(cfg['face_drop']) or not reads_face(cfg['groups']),
        # a window whose top-1 is this label is "not one of the words": word mode drops it
        'unknown_label': UNKNOWN if UNKNOWN in vocab_list else None,
        'contract': FT.spec(),
        'run': os.path.basename(os.path.normpath(out_dir)), 'seed': cfg['seed'], 'fold': cfg.get('fold'),
        'split_id': cfg.get('split_id'), 'best_epoch': result.get('best_epoch'), 'params': result.get('params'),
        'metrics': {k: v for k, v in result.items() if k.startswith(('test_', 'val_'))},
        'cfg': cfg,
    }
    with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(m, f, ensure_ascii=False, indent=1)


def fit_many(runs: dict, signs, words, vocab_list, out_root, log=print):
    """Several configurations on the same loaded data, one after another → out_root/<name>/ +
    summary.csv. A failing configuration is logged and skipped, the others still run."""
    rows = []
    for name, cfg in runs.items():
        log(f'\n===== {name}: {cfg}')
        t0 = time.time()
        try:
            r = fit(cfg, signs, words, vocab_list, os.path.join(out_root, name), log)
        except Exception as e:  # noqa: BLE001
            log(f'{name} FAILED: {e!r}')
            rows.append({'run': name, 'error': repr(e)[:200]})
            continue
        row = {'run': name, 'cfg': json.dumps(cfg), 'best_epoch': r['best_epoch'], 'params': r['params'],
               'n_features': r['n_features'], 'minutes': round((time.time() - t0) / 60, 1)}
        for k in ('test_words', 'test_signs', 'val_words', 'test_words_noface', 'test_signs_noface'):
            if k in r:
                row.update({f'{k}_{m}': round(r[k][m], 4) for m in ('top1', 'top5', 'macro_f1', 'balanced_acc')})
        row['test_words_majority'] = round(r['test_words']['majority_acc'], 4) if 'test_words' in r else None
        for k in ('val_open_acc', 'test_open_acc'):
            if k in r:
                row[k] = round(r[k], 4)
        for k in ('val_unk', 'test_unk'):
            if k in r:
                row[f'{k}_rejected'] = round(r[k]['rejected'], 4)
        for k in ('test_words', 'test_signs'):
            if k in r and 'unk_rate' in r[k]:
                row[f'{k}_unk_rate'] = round(r[k]['unk_rate'], 4)
        rows.append(row)
        with open(os.path.join(out_root, 'summary.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for rr in rows for k in rr}, key=lambda k: (k != 'run', k)))
            w.writeheader()
            w.writerows(rows)
    return rows


def _line(row, header):
    cols = [('epoch', 'ep', '{:>3d}'), ('train_loss', 'loss', '{:6.3f}'), ('train_acc', 'tr_acc', '{:6.3f}'),
            ('val_words_top1', 'W_top1', '{:6.3f}'), ('val_words_top5', 'W_top5', '{:6.3f}'),
            ('val_words_macro_f1', 'W_mF1', '{:6.3f}'), ('val_signs_top1', 'S_top1', '{:6.3f}'),
            ('val_signs_macro_f1', 'S_mF1', '{:6.3f}'), ('val_unk_rejected', 'U_rej', '{:6.3f}'),
            ('val_open_acc', 'open', '{:6.3f}'), ('sec', 'sec', '{:5.0f}')]
    cols = [c for c in cols if c[0] in row]
    s = '  '.join(f.format(row[k]) for k, _, f in cols) + ('  *' if row.get('best') else '')
    return ('  '.join(f'{h:>{len(f.format(row[k]))}}' for k, h, f in cols) + '\n' + s) if header else s


def _per_class(p, y, vocab, out_dir):
    pred = p.argmax(1)
    rows = []
    for c in np.unique(y):
        tp = int(((pred == c) & (y == c)).sum())
        n, npred = int((y == c).sum()), int((pred == c).sum())
        confused = np.bincount(pred[(y == c) & (pred != c)], minlength=len(vocab))
        rows.append({'gloss': vocab[c], 'n': n, 'recall': tp / n, 'precision': tp / npred if npred else 0.0,
                     'most_confused_with': vocab[int(confused.argmax())] if confused.any() else ''})
    with open(os.path.join(out_dir, 'per_class_test_words.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r['recall']))


def export_onnx(model, path, log=print, n_in=FT.N_FEAT):
    try:
        model = model.float().eval().cpu()
        # eval() on the wrapper too: export() restores the wrapper's mode afterwards, and a fresh module
        # is in training mode — that would switch the model to dropout + batch statistics
        wrapped = Exportable(model).eval()
        dummy = torch.zeros(1, 32, n_in)
        kw = dict(input_names=['x'], output_names=['logits'], dynamic_axes={'x': {1: 'T'}}, opset_version=17)
        try:
            torch.onnx.export(wrapped, (dummy,), path, dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(wrapped, (dummy,), path, **kw)
        log(f'ONNX: {path} ({os.path.getsize(path) / 1e6:.1f} MB)')
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(path)
        except ImportError:
            sess = None
            log('  (onnxruntime not installed — ONNX not run, only the export path checked in torch)')
        worst = 0.0
        with torch.no_grad():                    # the reference is the training/eval path (masked attention)
            for T_ in (7, 32, 64):               # other lengths than the traced 32
                x = torch.randn(1, T_, n_in)
                ref = model(x, torch.ones(1, T_, dtype=torch.bool)).numpy()
                outs = {'torch export path': wrapped(x).numpy()}
                if sess is not None:
                    outs['onnxruntime'] = sess.run(None, {'x': x.numpy()})[0]
                for what, got in outs.items():
                    d = float(np.abs(ref - got).max())
                    worst = max(worst, d)
                    log(f'  {what} vs torch, T={T_}: max abs diff {d:.2e}')
        if worst > 1e-3:
            log('  export does not reproduce the model: model.onnx is NOT marked as usable')
            return False
        return True
    except Exception as e:  # noqa: BLE001 — the browser export must not lose a finished training run
        log(f'ONNX export failed (training results are saved): {e!r}')
        return False
