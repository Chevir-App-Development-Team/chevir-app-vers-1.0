"""
Sözün formasının modeli (roadmap N2, docs/NLP_PLAN.md S3, E33): işarələr ardıcıllığından hər işarənin əlamətləri.

  node tools/build_pseudo_gloss.mjs data/corpus/hplt_sentences.txt data/corpus/pseudo_azsld_e33.jsonl
  node tools/build_pseudo_gloss.mjs data/corpus/hplt_sentences.txt data/corpus/pseudo_native_e33.jsonl --native
  node tools/build_form_data.mjs --out D:/chevir/e33/form_data2.jsonl \
      --pseudo data/corpus/pseudo_azsld_e33.jsonl,data/corpus/pseudo_native_e33.jsonl --names hplt,native [--part k/12]
  D:/chevir/venv/Scripts/python ml/formtag.py train --data D:/chevir/e33/form_data2.jsonl --out D:/chevir/e33/r2 \
      --epochs 4 --ft-epochs 6                                # E33-ün seçimi
  node tools/eval_gloss_text.mjs --tagger D:/chevir/e33/r2 --lambda 0.6
  D:/chevir/venv/Scripts/python ml/formtag.py install --model D:/chevir/e33/r2/all.json   # → ml/form_tagger.json
  node tools/build_gloss_text.mjs --tagger ml/form_tagger.json     # gloss_text.json-un içinə, on: true
  D:/chevir/venv/Scripts/python ml/formtag.py fixture --model ml/form_tagger.json --data D:/chevir/e33/form_data2.jsonl

train: hər model öz JSON-una (<out>/<ad>.json, src/tercume/formtag.js-in oxuduğu format, çəkilər float16):
  f0–f9 — korpus + AzSLD dev-in o biri 9 qatı (dev cümlələrinin ölçməsi, tools/eval_gloss_text.mjs --tagger),
  dev — korpus + bütün dev (test cümlələri və yeni cümlə ölçməsində test cümlələri üçün), all — korpus + 492
  cümlə (brauzerin modeli). Hər modelin öz lüğəti və sinifləri var (yalnız öz öyrənmə məlumatından).
  --ft-epochs > 0: əsas model korpusda bir dəfə (<out>/base.pt), hər model ondan AzSLD (çəki 10) + korpusun 20 000
  cümləsi ilə davam edir. Hesabat: AzSLD-nin çıxarılmış qatlarında əlamətlərin düzgünlüyü (<out>/report.json).
install: brauzerin modeli — lüğətdə yalnız tanıyıcının etiketləri (manifest.json) və <NUM>, <YEAR>, <LET> qalır
  (korpusun tanıyıcıda olmayan işarələri brauzerdə heç vaxt gəlmir; ehtimallar dəyişmir) → ml/form_tagger.json.
Giriş (formtag.js signTokens): işarənin adı, adından çıxan növü (ad / fel), adının son 3 hərfi. Model: üç embedding →
  iki istiqamətli GRU (PyTorch r, z, n) → xətti qat → əlamətlər dəstəsi (softmax). Öyrənmədə işarənin adı 10 %
  ehtimalla <UNK> olur: görünməmiş işarə adının sonu və növü ilə qiymətləndirilir.
"""
import argparse
import base64
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
FT_VERSION = 1
FOLDS = 10


def load(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def split(rows, name):
    """Modelin öyrənmə və yoxlama cümlələri: f<k> — dev-in k qatı çıxarılıb, dev — test çıxarılıb, all — hamısı."""
    az = [r for r in rows if r['set'] == 'azsld']
    other = [r for r in rows if r['set'] != 'azsld']
    if name.startswith('f'):
        k = name[1:]
        return other + [r for r in az if r['fold'] not in (k, 'test')], [r for r in az if r['fold'] == k]
    if name == 'dev':
        return other + [r for r in az if r['fold'] != 'test'], [r for r in az if r['fold'] == 'test']
    if name == 'all':
        return other + az, []
    raise ValueError(name)


class Tagger(nn.Module):
    def __init__(self, n_tok, n_suf, n_pos, n_cls, e_tok=48, e_suf=24, e_pos=4, hidden=96, layers=2, dropout=0.2):
        super().__init__()
        self.emb_tok = nn.Embedding(n_tok, e_tok)
        self.emb_suf = nn.Embedding(n_suf, e_suf)
        self.emb_pos = nn.Embedding(n_pos, e_pos)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(e_tok + e_suf + e_pos, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0)
        self.out = nn.Linear(2 * hidden, n_cls)
        self.hidden, self.layers = hidden, layers

    def forward(self, tok, suf, pos, lengths):
        x = torch.cat([self.emb_tok(tok), self.emb_suf(suf), self.emb_pos(pos)], -1)
        x = self.drop(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=tok.shape[1])
        return self.out(self.drop(h))


def vocab(rows, key, min_count=1):
    c = Counter(x[key] for r in rows for x in r['x'])
    return ['<PAD>', '<UNK>'] + sorted(t for t, n in c.items() if n >= min_count and t not in ('<PAD>', '<UNK>'))


MAXLEN = 48


def encode(rows, voc, classes, weights=None):
    """Bir dəfə: (N, MAXLEN) massivləri — tok, suf, pos, y (-100: hədəf yoxdur), çəki; uzunluqlar."""
    ix = {k: {t: i for i, t in enumerate(v)} for k, v in voc.items()}
    ci = {c: i for i, c in enumerate(classes)}
    N = len(rows)
    A = {k: np.zeros((N, MAXLEN), np.int64) for k in ('tok', 'suf', 'pos')}
    Y = np.full((N, MAXLEN), -100, np.int64)
    lens = np.zeros(N, np.int64)
    for b, r in enumerate(rows):
        xs = r['x'][:MAXLEN]
        lens[b] = len(xs)
        for t, x in enumerate(xs):
            A['tok'][b, t] = ix['tok'].get(x['tok'], 1)
            A['suf'][b, t] = ix['suf'].get(x['suf'], 1)
            A['pos'][b, t] = ix['pos'].get(x['pos'], 1)
            y = r['y'][t]
            Y[b, t] = ci.get(y, -100) if y else -100
    w = np.ones(N, np.float32) if weights is None else np.asarray(weights, np.float32)
    return {'tok': A['tok'], 'suf': A['suf'], 'pos': A['pos'], 'y': Y, 'w': w, 'len': lens}


def batches(enc, bs, shuffle, rng, dev, unk_p=0.0):
    N = len(enc['len'])
    order = np.arange(N)
    if shuffle:
        rng.shuffle(order)
    for s in range(0, N, bs):
        idx = order[s:s + bs]
        lens = torch.from_numpy(enc['len'][idx])
        L = int(lens.max())
        g = lambda k: torch.from_numpy(enc[k][idx, :L]).to(dev)  # noqa: E731
        T, Sf, P, Y = g('tok'), g('suf'), g('pos'), g('y')
        if unk_p:                                 # işarənin adı bəzən <UNK>: görünməmiş işarə üçün
            T = torch.where((torch.rand(T.shape, device=dev) < unk_p) & (T > 1), torch.ones_like(T), T)
        W = torch.from_numpy(enc['w'][idx]).to(dev)[:, None].expand(-1, L)
        yield T, Sf, P, Y, W, lens


def accuracy(model, enc, dev):
    if enc is None or not len(enc['len']):
        return None
    model.eval()
    ok = n = 0
    with torch.no_grad():
        for T, Sf, P, Y, W, lens in batches(enc, 512, False, None, dev):
            pred = model(T, Sf, P, lens).argmax(-1)
            m = Y != -100
            ok += int((pred[m] == Y[m]).sum())
            n += int(m.sum())
    return ok / max(1, n), n


def f16(t):
    a = t.detach().cpu().numpy().astype(np.float16)
    return {'shape': list(a.shape), 'data': base64.b64encode(a.astype('<f2').tobytes()).decode()}


def export(model, voc, classes, args, keep=None):
    """formtag.js-in formatı. GRU-nun çəkiləri hər qat və istiqamət üçün ayrıca (gru<L>.f / gru<L>.b). keep: lüğətdə
    qalan işarələr (modelin öyrənmədə görmədiyi işarə <UNK> olur: əsas modelin lüğətində çıxarılmış qatın işarələri var)."""
    rows = [i for i, t in enumerate(voc['tok']) if keep is None or i < 2 or t in keep]
    voc = {**voc, 'tok': [voc['tok'][i] for i in rows]}
    W = {'emb_tok': f16(model.emb_tok.weight[rows]), 'emb_suf': f16(model.emb_suf.weight), 'emb_pos': f16(model.emb_pos.weight),
         'out.w': f16(model.out.weight), 'out.b': f16(model.out.bias)}
    for L in range(model.layers):
        for d, suf in (('f', ''), ('b', '_reverse')):
            for a, b in (('w_ih', 'weight_ih'), ('w_hh', 'weight_hh'), ('b_ih', 'bias_ih'), ('b_hh', 'bias_hh')):
                W[f'gru{L}.{d}.{a}'] = f16(getattr(model.gru, f'{b}_l{L}{suf}'))
    return {'version': FT_VERSION, 'dims': {'H': model.hidden, 'layers': model.layers}, 'vocab': voc, 'classes': classes,
            'weights': W, 'train': {k: v for k, v in vars(args).items() if k not in ('cmd', 'data', 'out', 'models')}}


def round_f16(model):
    """Çəkilər float16-ya yuvarlaqlaşdırılır: Python-un hesabatı və fixture brauzerin çəkiləri ilə."""
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.half().float())


def fit(model, enc, args, dev, epochs, lr, rng, name, enc_va=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, epochs * math.ceil(len(enc['len']) / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for T, Sf, P, Y, W, lens in batches(enc, args.bs, True, rng, dev, args.unk):
            z = model(T, Sf, P, lens)
            loss = nn.functional.cross_entropy(z.reshape(-1, z.shape[-1]), Y.reshape(-1), reduction='none', ignore_index=-100)
            w = W.reshape(-1) * (Y.reshape(-1) != -100)
            loss = (loss * w).sum() / w.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
        acc = accuracy(model, enc_va, dev)
        print(f'  {name} ep {ep + 1}: loss {tot:.1f}' + (f', AzSLD held-out acc {acc[0]:.3f} ({acc[1]})' if acc else ''),
              f'{time.time() - t0:.0f}s', flush=True)


def new_model(voc, classes, args, dev):
    return Tagger(len(voc['tok']), len(voc['suf']), len(voc['pos']), len(classes), e_tok=args.e_tok, e_suf=args.e_suf,
                  hidden=args.hidden, layers=args.layers, dropout=args.dropout).to(dev)


def row_weight(r, args):
    return args.azsld_weight if r['set'] == 'azsld' else (args.native_weight if r['set'] == 'native' else 1.0)


def pretrain(rows, args, dev, out):
    """Əsas model (--ft-epochs > 0): yalnız korpusun cütləri; lüğətdə AzSLD-nin bütün işarələri də var (qatın modeli
    çıxarılmış qatın işarələrini ixrac etmir), siniflər korpusdan."""
    path = out / 'base.pt'
    pseudo = [r for r in rows if r['set'] != 'azsld']
    voc = {'tok': vocab(rows, 'tok'), 'suf': vocab(rows, 'suf', 2), 'pos': ['<PAD>', '<UNK>', '-', 'noun', 'verb']}
    cnt = Counter(t for r in pseudo for t in r['y'] if t)
    classes = sorted(c for c, n in cnt.items() if n >= args.min_class)
    model = new_model(voc, classes, args, dev)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=dev))
        print(f'base: {path}')
    else:
        rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        fit(model, encode(pseudo, voc, classes, [row_weight(r, args) for r in pseudo]), args, dev, args.epochs, args.lr, rng, 'base')
        torch.save(model.state_dict(), path)
    return model, voc, classes


def train_one(rows, name, args, dev, base=None):
    tr, va = split(rows, name)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if base is None:                              # birlikdə: korpus + AzSLD (çəki ilə) sıfırdan
        voc = {'tok': vocab(tr, 'tok'), 'suf': vocab(tr, 'suf', 2), 'pos': ['<PAD>', '<UNK>', '-', 'noun', 'verb']}
        cnt = Counter(t for r in tr for t in r['y'] if t)
        classes = sorted(c for c, n in cnt.items() if n >= args.min_class)
        model = new_model(voc, classes, args, dev)
        enc_va = encode(va, voc, classes)
        fit(model, encode(tr, voc, classes, [row_weight(r, args) for r in tr]), args, dev, args.epochs, args.lr, rng, name, enc_va)
        keep = None
    else:                                         # əsas modeldən: AzSLD + korpusun bir hissəsi ilə davam
        bm, voc, classes = base
        model = new_model(voc, classes, args, dev)
        model.load_state_dict(bm.state_dict())
        az = [r for r in tr if r['set'] == 'azsld']
        ps = [r for r in tr if r['set'] != 'azsld']
        sub = [ps[i] for i in rng.choice(len(ps), min(len(ps), args.ft_pseudo), replace=False)]
        ft = az + sub
        enc_va = encode(va, voc, classes)
        fit(model, encode(ft, voc, classes, [row_weight(r, args) for r in ft]), args, dev, args.ft_epochs, args.ft_lr, rng, name, enc_va)
        keep = {x['tok'] for r in tr for x in r['x']}
    round_f16(model)
    acc = accuracy(model, enc_va, dev)
    return model, voc, classes, acc, keep


def cmd_train(args):
    rows = load(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f'{len(rows)} cümlə ({sum(r["set"] == "azsld" for r in rows)} AzSLD), {dev}')
    names = args.models.split(',') if args.models else [f'f{k}' for k in range(FOLDS)] + ['dev', 'all']
    report = json.loads((out / 'report.json').read_text()) if (out / 'report.json').exists() else {}
    base = pretrain(rows, args, dev, out) if args.ft_epochs else None
    for name in names:
        model, voc, classes, acc, keep = train_one(rows, name, args, dev, base)
        (out / f'{name}.json').write_text(json.dumps(export(model.cpu(), voc, classes, args, keep)), encoding='utf-8')
        report[name] = {'acc': acc[0], 'n': acc[1]} if acc else {}
        print(f'{name}: {len(classes)} classes, vocab {len(voc["tok"])}, acc {report[name]}', flush=True)
        model.to(dev)
    folds = [report[f'f{k}'] for k in range(FOLDS) if report.get(f'f{k}', {}).get('n')]
    if folds:
        report['dev_cv'] = sum(f['acc'] * f['n'] for f in folds) / sum(f['n'] for f in folds)
        print(f'dev 10-fold acc {report["dev_cv"]:.3f}')
    (out / 'report.json').write_text(json.dumps(report, indent=1))


def cmd_fixture(args):
    """Brauzerin forward-u üçün yoxlama: modelin JSON-u, bir neçə cümlə və Python-un ehtimalları."""
    m = json.loads(Path(args.model).read_text(encoding='utf-8'))
    voc, classes = m['vocab'], m['classes']
    W = {k: torch.tensor(np.frombuffer(base64.b64decode(v['data']), dtype='<f2').astype(np.float32).reshape(v['shape']))
         for k, v in m['weights'].items()}
    H, L = m['dims']['H'], m['dims']['layers']
    model = Tagger(len(voc['tok']), len(voc['suf']), len(voc['pos']), len(classes), e_tok=W['emb_tok'].shape[1],
                   e_suf=W['emb_suf'].shape[1], e_pos=W['emb_pos'].shape[1], hidden=H, layers=L, dropout=0)
    with torch.no_grad():
        model.emb_tok.weight.copy_(W['emb_tok']); model.emb_suf.weight.copy_(W['emb_suf']); model.emb_pos.weight.copy_(W['emb_pos'])
        model.out.weight.copy_(W['out.w']); model.out.bias.copy_(W['out.b'])
        for li in range(L):
            for d, suf in (('f', ''), ('b', '_reverse')):
                for a, b in (('w_ih', 'weight_ih'), ('w_hh', 'weight_hh'), ('b_ih', 'bias_ih'), ('b_hh', 'bias_hh')):
                    getattr(model.gru, f'{b}_l{li}{suf}').copy_(W[f'gru{li}.{d}.{a}'])
    model.eval()
    rows = load(args.data)
    rng = random.Random(0)
    other = [r for r in rows if r['set'] != 'azsld']
    pick = [r for r in rows if r['set'] == 'azsld'][:6] + rng.sample(other, min(6, len(other)))
    # görünməmiş işarə: lüğətdə olmayan ad <UNK>-dur
    pick.append({'glosses': ['MƏN', 'QƏRİBƏSÖZ', 'GETMƏK'], 'x': [{'tok': 'MƏN', 'pos': 'noun', 'suf': 'mən'},
                 {'tok': 'QƏRİBƏSÖZ', 'pos': 'noun', 'suf': 'söz'}, {'tok': 'GETMƏK', 'pos': 'verb', 'suf': 'mək'}], 'y': [None] * 3})
    cases = []
    for r in pick:
        e = encode([r], voc, classes)
        n = int(e['len'][0])
        with torch.no_grad():
            z = model(*(torch.from_numpy(e[k][:, :n]) for k in ('tok', 'suf', 'pos')), torch.from_numpy(e['len']))
        p = torch.softmax(z[0].double(), -1).numpy()
        cases.append({'glosses': r['glosses'], 'x': r['x'], 'azsld': r.get('set') == 'azsld', 'probs': p.round(7).tolist()})
    out = ROOT / 'fixtures' / f'formtag_v{FT_VERSION}.json'
    out.write_text(json.dumps({'model': args.model_name or args.model, 'cases': cases}), encoding='utf-8')
    print(f'{out}: {len(cases)} cümlə')


def cmd_install(args):
    """Brauzerin modeli: lüğətdə tanıyıcının etiketləri və xüsusi tokenlər qalır (çəkilər olduğu kimi, float16)."""
    m = json.loads(Path(args.model).read_text(encoding='utf-8'))
    man = json.loads(Path(args.labels).read_text(encoding='utf-8'))
    keep = set(man['labels']) | {'<PAD>', '<UNK>', '<NUM>', '<YEAR>', '<LET>'}
    toks = m['vocab']['tok']
    rows = [i for i, t in enumerate(toks) if t in keep]
    w = m['weights']['emb_tok']
    a = np.frombuffer(base64.b64decode(w['data']), dtype='<f2').reshape(w['shape'])[rows]
    m['weights']['emb_tok'] = {'shape': list(a.shape), 'data': base64.b64encode(a.tobytes()).decode()}
    m['vocab'] = {**m['vocab'], 'tok': [toks[i] for i in rows]}
    m['source'] = f'{args.model} (E33); tokens kept: {len(rows)} of {len(toks)} (labels of {args.labels})'
    Path(args.out).write_text(json.dumps(m), encoding='utf-8')
    print(f'{args.out}: {len(rows)} of {len(toks)} tokens, {Path(args.out).stat().st_size / 1024:.0f} KB')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    i = sub.add_parser('install')
    i.add_argument('--model', required=True)
    i.add_argument('--labels', default=str(ROOT.parent / 'public/tercume/assets/cont_model/manifest.json'))
    i.add_argument('--out', default=str(ROOT / 'form_tagger.json'))
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--models', default='')
    t.add_argument('--hidden', type=int, default=96)
    t.add_argument('--layers', type=int, default=2)
    t.add_argument('--e-tok', type=int, default=48)
    t.add_argument('--e-suf', type=int, default=24)
    t.add_argument('--dropout', type=float, default=0.2)
    t.add_argument('--unk', type=float, default=0.1)
    t.add_argument('--epochs', type=int, default=6)
    t.add_argument('--bs', type=int, default=256)
    t.add_argument('--lr', type=float, default=3e-3)
    t.add_argument('--azsld-weight', type=float, default=10.0)
    t.add_argument('--native-weight', type=float, default=1.0)
    t.add_argument('--min-class', type=int, default=3)
    t.add_argument('--ft-epochs', type=int, default=0, help='>0: əsas model korpusda bir dəfə, hər model ondan davam edir')
    t.add_argument('--ft-lr', type=float, default=1e-3)
    t.add_argument('--ft-pseudo', type=int, default=20000, help='davam edəndə korpusdan neçə cümlə')
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--cpu', action='store_true')
    f = sub.add_parser('fixture')
    f.add_argument('--model', required=True)
    f.add_argument('--data', required=True)
    f.add_argument('--model-name', default='', help='fixture-də yazılan yol (susmaya görə --model)')
    args = ap.parse_args()
    {'train': cmd_train, 'fixture': cmd_fixture, 'install': cmd_install}[args.cmd](args)


if __name__ == '__main__':
    main()
