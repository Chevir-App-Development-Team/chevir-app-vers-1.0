"""Held-out evaluation of word mode (segment.py) on whole sentence recordings (notebook 05).

Every val / test recording goes through word mode the way the browser runs it: activity → spans →
candidate windows → the word model scores each window (one clip, face off when the deployed model
runs without the face tracker) → pick → detected words. They are compared with the annotated glosses:

  match       a detection and an annotation of the same word whose frames overlap; one to one,
              largest overlap (IoU) first
  recall      matched / annotated signs in the 200-word vocabulary (the model cannot know the others)
  precision   matched / all detections: a word shown on an out-of-vocabulary sign is still wrong for
              the user. precision_inv leaves out detections that only overlap out-of-vocabulary signs.
  WER         word error rate of the detected word sequence against the in-vocabulary glosses
  merged      the same with the confirmed word-form families as one word (ml/lemma_groups.csv)

A model with the unknown class (train.UNKNOWN in its vocabulary): an unknown window takes part in the
non-maximum suppression like any word, so it can suppress weaker guesses, but is never output.

Baselines: "whole span" (lens = (): every span is one word, no cutting) and "oracle" (the annotated
boundaries are given; top-1 on the in-vocabulary signs = the recall a perfect cutter would reach).
Parameters are chosen on val by F1; test is scored once with them.
"""
import itertools
import time

import torch

from . import features as FT
from . import segment as SG
from . import train as TR

GRID = dict(rest_line=(0.8, 1.0, 1.2),
            lens=((), (12, 18, 26), (10, 14, 18, 24, 30), (16, 24, 32), (12, 18, 26, 36)),
            min_p=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
            max_overlap=(0.0, 0.25, 0.5))


def configs(grid=GRID):
    keys = list(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield {**SG.SEGMENT, **dict(zip(keys, vals))}


def load_net(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    cfg = {**TR.DEFAULTS, **ck['cfg']}
    groups = cfg['groups']
    net = TR.Net(len(FT.columns(groups)), len(ck['vocab']), cfg['dim'], cfg['stages'], cfg['convs'],
                 cfg['kernel'], cfg['heads'], cfg['drop'], cfg['head_drop'])
    net.load_state_dict(ck['model'])
    return net.to(device).eval(), ck['vocab'], groups


class Scorer:
    """Top-1 (label index, probability) of frame windows of one recording, batched with masks
    (the training forward pass, equal to the exported single-clip graph within 5e-7)."""

    def __init__(self, net, groups, face, device, batch=256):
        self.net, self.groups, self.face, self.device, self.batch = net, groups, face, device, batch

    @torch.no_grad()
    def __call__(self, raw, aspect, wins):
        xs = [FT.preprocess({k: v[a:b] for k, v in raw.items()}, aspect, self.groups, self.face) for a, b in wins]
        out = {}
        order = sorted(range(len(wins)), key=lambda i: len(xs[i]))
        for s in range(0, len(order), self.batch):
            idx = order[s:s + self.batch]
            T = max(len(xs[i]) for i in idx)
            X = torch.zeros(len(idx), T, xs[idx[0]].shape[1])
            M = torch.zeros(len(idx), T, dtype=torch.bool)
            for r, i in enumerate(idx):
                X[r, :len(xs[i])] = torch.from_numpy(xs[i])
                M[r, :len(xs[i])] = True
            p = self.net(X.to(self.device), M.to(self.device)).float().softmax(1)
            pv, pi = p.max(1)
            for r, i in enumerate(idx):
                out[wins[i]] = (int(pi[r]), float(pv[r]))
        return out


def match(dets, anns, fam=None):
    """dets [(a, b, label)], anns [(s, e, gloss, in_vocab)] → matched (det i, ann j) pairs."""
    f = (lambda x: fam.get(x, x)) if fam else (lambda x: x)
    pairs = []
    for i, (a, b, lab) in enumerate(dets):
        for j, (s, e, g, inv) in enumerate(anns):
            ov = min(b, e) - max(a, s)
            if ov > 0 and inv and f(lab) == f(g):
                pairs.append((ov / (max(b, e) - min(a, s)), -i, -j))
    used_d, used_a, out = set(), set(), []
    for _, i, j in sorted(pairs, reverse=True):
        if -i in used_d or -j in used_a:
            continue
        used_d.add(-i)
        used_a.add(-j)
        out.append((-i, -j))
    return out


def edits(hyp, ref):
    """Levenshtein distance between two word sequences."""
    d = list(range(len(ref) + 1))
    for i, h in enumerate(hyp, 1):
        prev, d[0] = d[0], i
        for j, r in enumerate(ref, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (h != r))
    return d[len(ref)]


class Tally:
    """Counts for one configuration over many recordings."""
    KEYS = ('n_det', 'n_det_inv', 'tp', 'tp_m', 'n_ann', 'edits', 'edits_inv', 'edits_m', 'start_err', 'end_err')

    def __init__(self):
        self.c = dict.fromkeys(self.KEYS, 0)

    def add(self, dets, anns, fam):
        inv_anns = [x for x in anns if x[3]]
        hits = lambda a, b, want: any(min(b, e) - max(a, s) > 0 for s, e, _, inv in anns if inv == want)  # noqa: E731
        oov_only = [not hits(a, b, True) and hits(a, b, False) for a, b, _ in dets]
        tp = match(dets, anns)
        c = self.c
        c['n_det'] += len(dets)
        c['n_det_inv'] += len(dets) - sum(oov_only)
        c['tp'] += len(tp)
        c['tp_m'] += len(match(dets, anns, fam))
        c['n_ann'] += len(inv_anns)
        ref = [g for _, _, g, _ in sorted(inv_anns)]
        timed = sorted(zip(dets, oov_only))
        hyp = [lab for (_, _, lab), _ in timed]
        hyp_inv = [lab for (_, _, lab), oov in timed if not oov]
        c['edits'] += edits(hyp, ref)
        c['edits_inv'] += edits(hyp_inv, ref)
        c['edits_m'] += edits([fam.get(x, x) for x in hyp], [fam.get(x, x) for x in ref])
        for i, j in tp:
            c['start_err'] += abs(dets[i][0] - anns[j][0])
            c['end_err'] += abs(dets[i][1] - anns[j][1])

    def metrics(self):
        c = self.c
        div = lambda a, b: a / b if b else float('nan')                    # noqa: E731
        p, r = div(c['tp'], c['n_det']), div(c['tp'], c['n_ann'])
        pm, rm = div(c['tp_m'], c['n_det']), div(c['tp_m'], c['n_ann'])
        return {'precision': p, 'recall': r, 'f1': div(2 * p * r, p + r),
                'precision_inv': div(c['tp'], c['n_det_inv']),
                'wer': div(c['edits'], c['n_ann']), 'wer_inv': div(c['edits_inv'], c['n_ann']),
                'precision_merged': pm, 'recall_merged': rm, 'f1_merged': div(2 * pm * rm, pm + rm),
                'wer_merged': div(c['edits_m'], c['n_ann']),
                'start_err_frames': div(c['start_err'], c['tp']), 'end_err_frames': div(c['end_err'], c['tp']),
                'detections': c['n_det'], 'annotated_in_vocab': c['n_ann'], 'matched': c['tp']}


def config_key(p):
    return (p['rest_line'], tuple(p['lens']), p['min_p'], p['max_overlap'])


def evaluate(load, recs, anns_by_rec, scorer, vocab, fam, cfgs=None, log=print, keep=None):
    """load(rec_id) → reduced raw (features.reduce layout); recs [(rec_id, aspect)];
    anns_by_rec {rec_id: [(start, end, gloss, in_vocab)]} with the annotation's 1-based inclusive frames;
    cfgs: the configurations to score (default: all of GRID).
    → ({config_key: Tally}, oracle counts, detections [(rec_id, a, b, word)] of the `keep` config)."""
    cfgs = list(configs()) if cfgs is None else list(cfgs)
    tallies = {config_key(p): Tally() for p in cfgs}
    groups = {}                                    # (rest_line, lens) → [(min_p, max_overlap)]
    for p in cfgs:
        groups.setdefault((p['rest_line'], tuple(p['lens'])), []).append((p['min_p'], p['max_overlap']))
    rest_lines = sorted({rl for rl, _ in groups})
    oracle = {'n': 0, 'top1': 0, 'top1_m': 0}
    kept, t0 = [], time.time()
    for n, (rid, aspect) in enumerate(recs):
        raw = load(rid)
        T = raw['hand'].shape[0]
        anns = [(max(0, int(s) - 1), min(int(e), T), g, bool(inv)) for s, e, g, inv in anns_by_rec.get(rid, [])]
        spans = {rl: SG.spans(SG.activity(raw, aspect, rl), {**SG.SEGMENT, 'rest_line': rl}) for rl in rest_lines}
        wins = set()
        for rl, lens in groups:
            for sp in spans[rl]:
                wins.update(SG.windows(sp, {**SG.SEGMENT, 'lens': lens}))
        oracle_anns = [(s, e, g) for s, e, g, inv in anns if inv and e - s >= 2]
        scores = scorer(raw, aspect, sorted(wins | {(s, e) for s, e, _ in oracle_anns}))
        for s, e, g in oracle_anns:
            lab = vocab[scores[(s, e)][0]]
            oracle['n'] += 1
            oracle['top1'] += lab == g
            oracle['top1_m'] += fam.get(lab, lab) == fam.get(g, g)
        for (rl, lens), choices in groups.items():
            p0 = {**SG.SEGMENT, 'rest_line': rl, 'lens': lens}
            scored = [[(a, b, vocab[scores[(a, b)][0]], scores[(a, b)][1]) for a, b in SG.windows(sp, p0)]
                      for sp in spans[rl]]
            for mp, mo in choices:
                p = {**p0, 'min_p': mp, 'max_overlap': mo}
                dets = [(a, b, lab) for sc in scored for a, b, lab, _ in SG.pick(sc, p) if lab != TR.UNKNOWN]
                tallies[config_key(p)].add(dets, anns, fam)
                if keep is not None and config_key(p) == config_key(keep):
                    kept += [(rid, a, b, lab) for a, b, lab in dets]
        if (n + 1) % 200 == 0:
            log(f'  {n + 1}/{len(recs)} recordings ({time.time() - t0:.0f} s)')
    return tallies, oracle, kept
