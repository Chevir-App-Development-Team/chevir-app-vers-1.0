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
DEFAULTS = dict(epochs=60, batch=64, lr=1e-3, weight_decay=0.05, warmup=3, label_smoothing=0.1,
                balance_power=0.5, dim=192, stages=2, convs=3, kernel=17, heads=4, drop=0.2,
                head_drop=0.4, margin=8, workers=4, seed=0, groups=None)


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


def load_sentence_signs(sentence_dirs, tags, vocab, margin=8, log=print):
    """tags (glosses_split.csv rows) → Bank per split, sliced from whole-recording landmarks."""
    store = LandmarkStore(sentence_dirs)
    idx = store.index
    n0 = len(tags)
    tags = tags[tags.rec_id.isin(idx.index) & tags.gloss.isin(vocab) & tags.start.notna() & tags.end.notna()].copy()
    if len(tags) < n0:
        log(f'  {n0 - len(tags)} signs dropped (no landmarks, outside the vocabulary or no frame range)')
    tags['_order'] = list(zip(tags.rec_id.map(idx['dir']), tags.rec_id.map(idx['part'])))
    banks = {s: Bank() for s in ('train', 'val', 'test')}
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
            banks[r.split].add({k: v[lo:hi] for k, v in raw.items()}, vocab[r.gloss], aspect,
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
    def __init__(self, bank, train=False, flip=False, cols=None):
        self.bank, self.train, self.flip, self.cols = bank, train, flip, cols

    def __len__(self):
        return len(self.bank)

    def __getitem__(self, i):
        raw, label, aspect, core = self.bank.raw(i)
        sq = FT.to_square(raw, aspect)
        if self.train:
            d = FT.augment(sq, np.random.default_rng(), aspect, core)
        else:
            lo, hi = core
            d = {k: v[lo:hi] for k, v in sq.items()}
            if self.flip:
                d = FT.flip(d, aspect)
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
        m = valid[:, :, None].to(x.dtype)
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
        a, _ = self.attn(h, h, h, key_padding_mask=~valid, need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.n2(x)))


class Net(nn.Module):
    def __init__(self, n_feat, n_cls, dim=192, stages=2, convs=3, kernel=17, heads=4, drop=0.2, head_drop=0.4):
        super().__init__()
        self.stem, self.stem_norm = nn.Linear(n_feat, dim), nn.LayerNorm(dim)
        blocks = []
        for _ in range(stages):
            blocks += [ConvBlock(dim, kernel, drop=drop) for _ in range(convs)] + [TransformerBlock(dim, heads, drop)]
        self.blocks = nn.ModuleList(blocks)
        self.norm, self.drop, self.head = nn.LayerNorm(dim), nn.Dropout(head_drop), nn.Linear(dim, n_cls)

    def forward(self, x, valid):
        m = valid[:, :, None].to(x.dtype)
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
        return self.net(x, torch.ones(x.shape[:2], dtype=torch.bool, device=x.device))


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
def predict(model, bank, device, batch=128, workers=2, tta=False, cols=None):
    model.eval()
    outs = []
    for flip in ([False, True] if tta else [False]):
        dl = torch.utils.data.DataLoader(SignDataset(bank, flip=flip, cols=cols), batch_size=batch, shuffle=False,
                                         num_workers=workers, collate_fn=collate)
        logits = []
        for x, m, _ in dl:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                logits.append(model(x.to(device), m.to(device)).float().softmax(1).cpu())
        outs.append(torch.cat(logits).numpy())
    return np.mean(outs, 0)


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
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(weights), num_samples=len(tr), replacement=True)
    cols = FT.columns(cfg['groups'])
    dl = torch.utils.data.DataLoader(SignDataset(tr, train=True, cols=cols), batch_size=cfg['batch'], sampler=sampler,
                                     num_workers=cfg['workers'], collate_fn=collate, drop_last=True,
                                     persistent_workers=cfg['workers'] > 0, pin_memory=device.type == 'cuda')
    evals = {k: b for k, b in (('val_words', words.get('val')), ('val_signs', signs.get('val'))) if b is not None and len(b)}
    major = int(np.bincount(tr.labels, minlength=C).argmax())

    model = Net(len(cols), C, cfg['dim'], cfg['stages'], cfg['convs'], cfg['kernel'], cfg['heads'],
                cfg['drop'], cfg['head_drop']).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    steps, warm = cfg['epochs'] * len(dl), max(1, cfg['warmup'] * len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, steps - warm)))))
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')

    log(f'{device} | {n_params / 1e6:.2f} M params | {len(cols)} features | {len(tr)} train signs | {C} classes | '
        f'{len(dl)} steps/epoch | majority class "{vocab_list[major]}"')
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
            s = scores(predict(model, b, device, workers=min(2, cfg['workers']), cols=cols), b.labels)
            row.update({f'{k}_{m}': v for m, v in s.items() if m != 'n'})
        row['sec'] = time.time() - t0
        key = 'val_words_macro_f1' if 'val_words' in evals else 'val_signs_macro_f1'
        row['best'] = row.get(key, 0) > best
        if row['best']:
            best, best_ep = row[key], ep
            torch.save({'model': model.state_dict(), 'cfg': cfg, 'vocab': vocab_list, 'epoch': ep}, os.path.join(out_dir, 'best.pt'))
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
                       ('val_words', words.get('val')), ('val_signs', signs.get('val'))):
        if bank is None or not len(bank):
            continue
        nw = min(2, cfg['workers'])
        p = predict(model, bank, device, workers=nw, cols=cols)
        preds.update({f'{name}_probs': p.astype(np.float16), f'{name}_labels': bank.labels,
                      f'{name}_ids': np.array(bank.ids)})
        result[name] = scores(p, bank.labels)
        result[name]['majority_acc'] = float((bank.labels == major).mean())
        result[name + '_tta'] = scores(predict(model, bank, device, workers=nw, tta=True, cols=cols), bank.labels)
        if name == 'test_words':
            _per_class(p, bank.labels, vocab_list, out_dir)
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab_list, f, ensure_ascii=False)
    # per-sample probabilities, for analyses without retraining (e.g. merging inflected word forms)
    np.savez_compressed(os.path.join(out_dir, 'predictions.npz'), **preds)
    export_onnx(model, os.path.join(out_dir, 'model.onnx'), log, n_in=len(cols))
    return result


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
        for k in ('test_words', 'test_signs', 'val_words'):
            if k in r:
                row.update({f'{k}_{m}': round(r[k][m], 4) for m in ('top1', 'top5', 'macro_f1', 'balanced_acc')})
        row['test_words_majority'] = round(r['test_words']['majority_acc'], 4) if 'test_words' in r else None
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
            ('val_signs_macro_f1', 'S_mF1', '{:6.3f}'), ('sec', 'sec', '{:5.0f}')]
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
        dummy = torch.zeros(1, 32, n_in)
        kw = dict(input_names=['x'], output_names=['logits'], dynamic_axes={'x': {1: 'T'}}, opset_version=17)
        try:
            torch.onnx.export(Exportable(model), (dummy,), path, dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(Exportable(model), (dummy,), path, **kw)
        log(f'ONNX: {path} ({os.path.getsize(path) / 1e6:.1f} MB)')
        try:                                     # same logits as torch, for a short and a long clip?
            import onnxruntime as ort
            sess = ort.InferenceSession(path)
            for T_ in (12, 64):
                x = torch.randn(1, T_, n_in)
                ref = Exportable(model)(x).detach().numpy()
                got = sess.run(None, {'x': x.numpy()})[0]
                log(f'  onnxruntime vs torch, T={T_}: max abs diff {float(np.abs(ref - got).max()):.2e}')
        except ImportError:
            log('  (onnxruntime not installed — not compared)')
    except Exception as e:  # noqa: BLE001 — the browser export must not lose a finished training run
        log(f'ONNX export failed (training results are saved): {e!r}')
