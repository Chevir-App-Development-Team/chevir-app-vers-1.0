"""
Üz kanalı (R3, E31): cümlə sualdırmı, inkardırmı — üzdən və başdan (ml/azsl/nonmanual.py).

  D:/chevir/venv/Scripts/python ml/nonmanual.py cache --kaggle D:/chevir/kaggle --landmarks D:/chevir/landmarks \
      --out D:/chevir/eval/nonmanual
  D:/chevir/venv/Scripts/python ml/nonmanual.py train --out D:/chevir/eval/nonmanual [--neutral rest] [--groups head,upper]
  D:/chevir/venv/Scripts/python ml/nonmanual.py fit --lam 0.001,0.003,0.01,0.03,0.1 --run D:/chevir/eval/nonmanual/final
  cp D:/chevir/eval/nonmanual/final/nonmanual.json public/tercume/assets/
  D:/chevir/venv/Scripts/python ml/nonmanual.py fixture      # ml/fixtures/nonmanual_v<N> (tools/verify_nonmanual.mjs)

train: λ, neytral və əlamət qrupları üzrə cədvəl (logistik reqressiya, val / test / JestDili.az). fit: son model
  (bəli/xeyr sualı və inkar başlıqları, hədlər val-da) → <run>/nonmanual.json + report.json.
cache: AzSLD-nin ön kamerasının (Cam2) bütün cümlə yazıları və JestDili.az-ın 500 cümləsi → kadr əlamətləri (30 fps),
  əllərin görünməsi, etiketlər. Yan kamerada (Cam1) üz yoxdur.
Etiketlər: AzSLD — mətndəki "?" (sual), sual sözü varsa 'wh', yoxsa 'yn'; inkar mətndən və qlosdan
  (nonmanual.weak_labels). JestDili.az-da "?" yoxdur: qaydalar + əl ilə yoxlanmış siyahı
  (<out>/jestdili_labels.csv, repoda deyil; 'amb' — qərarsız, ölçməyə girmir).
Bölgü: AzSLD-nin iştirakçı bölgüsü (ml/splits/sentences_split.csv: train / val / test).
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from azsl import features as FT  # noqa: E402
from azsl import nonmanual as NM  # noqa: E402
from azsl.store import LandmarkStore  # noqa: E402

# AzSLD: mətndə "?" yoxdur, amma sual kimi oxunur (FAQ başlığı) — nə sual, nə nəqli sayılır
AZSLD_AMBIGUOUS = {'441', '444'}


def _pack(items):
    """[(id, X T×F, up T)] → bir npz-ə yazılacaq massivlər."""
    lens = np.array([len(x) for _, x, _ in items], np.int64)
    off = np.concatenate([[0], np.cumsum(lens)])
    X = np.concatenate([x for _, x, _ in items]).astype(np.float32)
    up = np.concatenate([u for _, _, u in items]).astype(bool)
    return {'ids': np.array([i for i, _, _ in items]), 'off': off, 'X': X, 'up': up}


def cache(args):
    os.makedirs(args.out, exist_ok=True)
    import pandas as pd
    t0 = time.time()
    # AzSLD
    sp = pd.read_csv(ROOT / 'splits' / 'sentences_split.csv', dtype={'folder': str})
    dirs = sorted(glob.glob(os.path.join(args.kaggle, '02-sentences-shard*', 'landmarks', 'sentences_cam2', 'shard*')))
    store = LandmarkStore(dirs)
    idx = store.index
    ids = [r for r in sp.rec_id if r in idx.index]
    ids.sort(key=lambda r: (idx.at[r, 'dir'], idx.at[r, 'part']))
    items = []
    for n, rid in enumerate(ids):
        raw = FT.reduce(store.get(rid, dtype=np.float16))
        raw = NM.to_fps(raw, float(idx.at[rid, 'fps']))
        aspect = float(idx.at[rid, 'width']) / float(idx.at[rid, 'height'])
        items.append((rid, NM.frame_features(raw, aspect), NM.hands_up(raw, aspect)))
        if n % 1000 == 0:
            print(f'  AzSLD {n}/{len(ids)} ({time.time() - t0:.0f} s)', flush=True)
    np.savez(os.path.join(args.out, 'azsld_frames.npz'), **_pack(items))
    meta = sp.set_index('rec_id').loc[[i for i, _, _ in items]].reset_index()
    rows = []
    for r in meta.itertuples():
        q, qt, neg = NM.weak_labels(r.text, True, str(r.glosses))
        amb = r.folder in AZSLD_AMBIGUOUS
        rows.append({'id': r.rec_id, 'split': r.split, 'signer': r.signer, 'folder': r.folder,
                     'heldout': r.heldout_sentence, 'question': 'amb' if amb else q, 'qtype': qt, 'neg': neg,
                     'text': r.text})
    pd.DataFrame(rows).to_csv(os.path.join(args.out, 'azsld_labels.csv'), index=False, encoding='utf-8')
    print(f'AzSLD: {len(items)} recordings ({time.time() - t0:.0f} s)', flush=True)

    # JestDili.az (50 fps → 30)
    jd = LandmarkStore(os.path.join(args.landmarks, 'jestdili_sentences'))
    items = []
    for rid in jd.manifest.id:
        raw = FT.reduce(jd.get(rid, dtype=np.float16))
        raw = NM.to_fps(raw, float(jd.index.at[rid, 'fps']))
        aspect = float(jd.index.at[rid, 'width']) / float(jd.index.at[rid, 'height'])
        items.append((rid, NM.frame_features(raw, aspect), NM.hands_up(raw, aspect)))
    np.savez(os.path.join(args.out, 'jestdili_frames.npz'), **_pack(items))
    lab = os.path.join(args.out, 'jestdili_labels.csv')
    if not os.path.exists(lab):                      # qaydaların ilk siyahısı; sonra əl ilə düzəldilir
        with open(lab, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['id', 'question', 'qtype', 'neg', 'text'])
            for r in jd.manifest.itertuples():
                q, qt, neg = NM.weak_labels(r.text, False)
                w.writerow([r.id, q, qt, neg, r.text])
    print(f'JestDili.az: {len(items)} sentences ({time.time() - t0:.0f} s)')


def load(out, name):
    z = np.load(os.path.join(out, f'{name}_frames.npz'))
    off = z['off']
    X, up = z['X'], z['up']
    return list(z['ids']), [X[off[i]:off[i + 1]] for i in range(len(off) - 1)], \
        [up[off[i]:off[i + 1]] for i in range(len(off) - 1)]


# ---------------------------------------------------------------- model: standartlaşdırma + logistik reqressiya

def logreg(X, y, w, lam, iters=100):
    """Çəkili L2 logistik reqressiya (Nyuton addımları). X standartlaşdırılıb. → (coef d, bias)."""
    n, d = X.shape
    A = np.hstack([X, np.ones((n, 1))]).astype(np.float64)
    b = np.zeros(d + 1)
    R = np.eye(d + 1) * lam * w.sum()
    R[-1, -1] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(A @ b, -30, 30)))
        g = A.T @ (w * (p - y)) + R @ b
        H = (A * (w * p * (1 - p))[:, None]).T @ A + R + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(H, g)
        b -= step
        if np.abs(step).max() < 1e-7:
            break
    return b[:-1], b[-1]


def balanced(y):
    w = np.ones(len(y))
    for c in (0, 1):
        if (y == c).any():
            w[y == c] = len(y) / (2 * (y == c).sum())
    return w


def auc(y, s):
    y = np.asarray(y).astype(bool)
    if y.all() or not y.any():
        return float('nan')
    r = np.argsort(np.argsort(s, kind='mergesort'), kind='mergesort') + 1.0
    # bərabər balların orta rütbəsi
    order = np.argsort(s, kind='mergesort')
    ss = np.asarray(s)[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def binary(y, s, t):
    y = np.asarray(y).astype(bool)
    p = np.asarray(s) >= t
    tp, fp, fn, tn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum(), (~p & ~y).sum()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    spec = tn / max(1, tn + fp)
    return {'f1': f1, 'prec': prec, 'rec': rec, 'spec': spec, 'bacc': (rec + spec) / 2, 'acc': (tp + tn) / max(1, len(y))}


def best_threshold(y, s):
    """Val-da F1-i ən böyük edən hədd."""
    best, bt = -1, 0.5
    for t in np.unique(np.round(s, 3)):
        f = binary(y, s, t)['f1']
        if f > best:
            best, bt = f, float(t)
    return bt


def clip_matrix(X, up, mode, cols, signer_of=None):
    """Kliplərin vektorları. mode 'signer': neytral iştirakçının bütün yazılarının dinc kadrlarından (AzSLD-də
    ölçmək üçün; brauzerdə sessiyanın dinc kadrları)."""
    base = {}
    if mode == 'signer':
        by = {}
        for x, u, s in zip(X, up, signer_of):
            ok = np.isfinite(x[:, 0]) & ~u
            by.setdefault(s, []).append(x[ok])
        base = {s: np.median(np.concatenate(v), 0) if sum(len(a) for a in v) else np.zeros(NM.N_FRAME, np.float32)
                for s, v in by.items()}
    out = []
    for i, (x, u) in enumerate(zip(X, up)):
        b = base[signer_of[i]] if mode == 'signer' else NM.neutral(x, u, mode)
        out.append(NM.clip_features(x, u, b, cols))
    return np.stack(out)


def _cols(groups):
    return None if groups == 'all' else sorted(sum((NM.GROUPS[g] for g in groups.split('+')), []))


def train(args):
    import pandas as pd
    ids, X, up = load(args.out, 'azsld')
    lab = pd.read_csv(os.path.join(args.out, 'azsld_labels.csv'), dtype={'folder': str}).set_index('id').loc[ids]
    jids, JX, Jup = load(args.out, 'jestdili')
    jl = pd.read_csv(os.path.join(args.out, 'jestdili_labels.csv'), dtype=str).set_index('id').loc[jids]
    split = lab.split.values
    target = args.target
    yq = lab.question.astype(str).values
    keep = yq != 'amb'
    y = (lab[target].astype(str).values == '1').astype(float) if target == 'neg' else (yq == '1').astype(float)
    jy_raw = jl[target].astype(str).values if target == 'neg' else jl.question.astype(str).values
    jkeep = jy_raw != 'amb'
    jy = (jy_raw == '1').astype(float)
    signer = lab.signer.values
    results = []
    for mode in args.neutral.split(','):
        for groups in args.groups.split(','):
            cols = _cols(groups)
            F = clip_matrix(X, up, mode, cols, signer)
            JF = clip_matrix(JX, Jup, 'rest' if mode == 'signer' else mode, cols)
            fr = F[:, -1]
            tr = (split == 'train') & keep & (fr >= args.min_face)
            mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
            Z, JZ = (F - mu) / sd, (JF - mu) / sd
            for lam in [float(v) for v in args.lam.split(',')]:
                coef, b0 = logreg(Z[tr], y[tr], balanced(y[tr]), lam)
                s = 1 / (1 + np.exp(-(Z @ coef + b0)))
                js = 1 / (1 + np.exp(-(JZ @ coef + b0)))
                va = (split == 'val') & keep & (fr >= args.min_face)
                te = (split == 'test') & keep & (fr >= args.min_face)
                jt = jkeep & (JF[:, -1] >= args.min_face)
                t = best_threshold(y[va], s[va])
                r = {'neutral': mode, 'groups': groups, 'lam': lam, 't': t,
                     'val_auc': auc(y[va], s[va]), 'val': binary(y[va], s[va], t),
                     'test_auc': auc(y[te], s[te]), 'test': binary(y[te], s[te], t),
                     'jd_auc': auc(jy[jt], js[jt]), 'jd': binary(jy[jt], js[jt], t), 'n_jd': int(jt.sum())}
                results.append(r)
                print(f"{mode:6} {groups:18} lam {lam:<7g} | val AUC {r['val_auc']:.3f} F1 {r['val']['f1']:.3f} "
                      f"| test AUC {r['test_auc']:.3f} F1 {r['test']['f1']:.3f} bacc {r['test']['bacc']:.3f} "
                      f"| JD AUC {r['jd_auc']:.3f} F1 {r['jd']['f1']:.3f} bacc {r['jd']['bacc']:.3f} (n {r['n_jd']})", flush=True)
    with open(os.path.join(args.out, f'grid_{target}.json'), 'w') as f:
        json.dump(results, f, indent=1, default=float)


# ---------------------------------------------------------------- son model: başlıqlar, hədlər, brauzer üçün JSON

# Cümlə modelinin sual sözləri (manual kanal): belə işarə tanınıbsa cümlə sualdır (wh), üz bəli/xeyr sualını verir
WH_LABELS = ['HANSI', 'HARA', 'HARDA', 'KİM', 'KİM?', 'NECƏ', 'NECƏYƏDİR', 'NEÇƏ', 'NƏ VAXT', 'NƏ ZAMAN', 'NƏDİR',
             'NİYƏ', 'NƏ']
P2_LABELS = {'SİZ', 'SƏN'}


def _wh(words):
    return any(w.upper().rstrip('?') in {x.rstrip('?') for x in WH_LABELS} for w in words)


def fit(args):
    """Başlıqlar: yn — bəli/xeyr sualı nəqli cümləyə qarşı (sual sözlü suallar öyrənməyə girmir), neg — inkar.
    Öyrənmə train iştirakçılarında, λ və hədd val-da: yn-in həddi sual aşkarlamasının F1-i (sual sözü işarəsi
    VƏ YA üz ≥ hədd; val-da işarə annotasiyadan), neg-inki öz F1-i. Test və JestDili.az yalnız ölçülür."""
    import pandas as pd
    ids, X, up = load(args.out, 'azsld')
    lab = pd.read_csv(os.path.join(args.out, 'azsld_labels.csv'), dtype={'folder': str}).set_index('id').loc[ids]
    sp = pd.read_csv(ROOT / 'splits' / 'sentences_split.csv').set_index('rec_id').loc[ids]
    jids, JX, Jup = load(args.out, 'jestdili')
    jl = pd.read_csv(os.path.join(args.out, 'jestdili_labels.csv'), dtype=str, keep_default_na=False).set_index('id').loc[jids]
    F = clip_matrix(X, up, args.neutral, None)
    JF = clip_matrix(JX, Jup, args.neutral, None)
    fr, jfr = F[:, -1], JF[:, -1]
    split = lab.split.values
    yq = lab.question.astype(str).values
    qt = lab.qtype.fillna('').values
    face = fr >= args.min_face
    keep = (yq != 'amb') & face
    is_q = yq == '1'
    neg = lab.neg.astype(int).values == 1
    ann_wh = np.array([_wh(g.replace('?', ' ').split()) for g in sp.glosses.fillna('')])
    tr, va, te = [(split == s) & keep for s in ('train', 'val', 'test')]
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    Z, JZ = (F - mu) / sd, (JF - mu) / sd
    sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731

    heads, report = {}, {}
    # yn: bəli/xeyr sualı (wh sualları öyrənmədən çıxır)
    m_tr = tr & ~(is_q & (qt == 'wh'))
    best = None
    for lam in [float(v) for v in args.lam.split(',')]:
        coef, b0 = logreg(Z[m_tr], is_q[m_tr].astype(float), balanced(is_q[m_tr].astype(float)), lam)
        s = sig(Z @ coef + b0)
        m_va = va & ~(is_q & (qt == 'wh'))
        a = auc(is_q[m_va], s[m_va])
        if best is None or a > best[0]:
            best = (a, lam, coef, b0, s)
    _, lam, coef, b0, s = best
    t, bf = 0.5, -1
    for c in np.unique(np.round(s[va], 3)):
        f = binary(is_q[va], (ann_wh[va] | (s[va] >= c)).astype(float), 0.5)['f1']
        if f > bf:
            bf, t = f, float(c)
    heads['yn'] = (coef, b0, t, lam)
    js = sig(JZ @ coef + b0)
    jy = jl.question.values
    jkeep = (jy != 'amb') & (jfr >= args.min_face)
    jq = jy == '1'
    jqt = jl.qtype.values
    m_te = te & ~(is_q & (qt == 'wh'))
    m_jd = jkeep & ~(jq & (jqt == 'wh'))
    report['yn'] = {
        'lam': lam, 't': t, 'val_auc_yn': best[0], 'val_f1_with_annotated_wh': bf,
        'test_auc_yn': auc(is_q[m_te], s[m_te]), 'test_yn': binary(is_q[m_te], s[m_te], t),
        'test_auc_all_q': auc(is_q[te], s[te]), 'test_face_only_all_q': binary(is_q[te], s[te], t),
        'test_with_annotated_wh': binary(is_q[te], (ann_wh[te] | (s[te] >= t)).astype(float), 0.5),
        'jd_auc_yn': auc(jq[m_jd], js[m_jd]), 'jd_yn': binary(jq[m_jd], js[m_jd], t),
        'jd_auc_all_q': auc(jq[jkeep], js[jkeep]), 'jd_face_only_all_q': binary(jq[jkeep], js[jkeep], t),
        'n': {'train': int(m_tr.sum()), 'val': int(va.sum()), 'test': int(te.sum()), 'jd': int(jkeep.sum())},
    }
    # neg
    jneg = jl.neg.values == '1'
    jkeep_n = jfr >= args.min_face
    best = None
    for lam in [float(v) for v in args.lam.split(',')]:
        c_, b_ = logreg(Z[tr], neg[tr].astype(float), balanced(neg[tr].astype(float)), lam)
        s_ = sig(Z @ c_ + b_)
        a = auc(neg[va], s_[va])
        if best is None or a > best[0]:
            best = (a, lam, c_, b_, s_)
    _, lam_n, c_n, b_n, s_n = best
    t_n = best_threshold(neg[va], s_n[va])
    heads['neg'] = (c_n, b_n, t_n, lam_n)
    js_n = sig(JZ @ c_n + b_n)
    report['neg'] = {'lam': lam_n, 't': t_n, 'val_auc': best[0], 'val': binary(neg[va], s_n[va], t_n),
                     'test_auc': auc(neg[te], s_n[te]), 'test': binary(neg[te], s_n[te], t_n),
                     'jd_auc': auc(jneg[jkeep_n], js_n[jkeep_n]), 'jd': binary(jneg[jkeep_n], js_n[jkeep_n], t_n)}
    report['coverage'] = {'azsld_face_seen': float(face.mean()), 'jd_face_seen': float((jfr >= args.min_face).mean())}

    # Brauzerdəki kimi: tanınan işarələrdə sual sözü VƏ YA üz ≥ hədd. AzSLD test — cümlə modeli test iştirakçılarını
    # görməyib (--det, v5 protokolu); JestDili.az — modelin görmədiyi yarısı (--jd-det). Üz görünməyən klip: yalnız işarə.
    if args.det and os.path.exists(args.det):
        det = {}
        with open(args.det, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                det.setdefault(r['rec_id'], []).append(r['word'])
        rec_wh = np.array([_wh(det.get(i, [])) for i in ids])
        has = np.array([i in det for i in ids])
        allq = (yq != 'amb')
        for name, m in [('test', allq & (split == 'test') & has), ('test_heldout_sentences',
                                                                     allq & (split == 'test') & has & lab.heldout.values)]:
            pred = rec_wh[m] | (face[m] & (s[m] >= t))
            report[f'recognized_{name}'] = {'n': int(m.sum()), 'q': int(is_q[m].sum()),
                                            'sign_only': binary(is_q[m], rec_wh[m].astype(float), 0.5),
                                            'sign_or_face': binary(is_q[m], pred.astype(float), 0.5),
                                            'yn_recall': float(pred[is_q[m] & (qt[m] == 'yn')].mean()),
                                            'wh_recall': float(pred[is_q[m] & (qt[m] == 'wh')].mean()),
                                            'statement_false_alarm': float(pred[~is_q[m]].mean())}
    if args.jd_det and os.path.exists(args.jd_det):
        with open(args.jd_det, encoding='utf-8') as f:
            sj = {r['id']: r['signs'] for r in json.load(f)}
        jwh = np.array([_wh(sj.get(i, [])) for i in jids])
        jface = jfr >= args.min_face
        for name, m in [('jd_heldout', np.isin(jids, list(sj)) & (jy != 'amb')),
                        ('jd_heldout_face_seen', np.isin(jids, list(sj)) & (jy != 'amb') & jface)]:
            pred = jwh[m] | (jface[m] & (js[m] >= t))
            report[f'recognized_{name}'] = {'n': int(m.sum()), 'q': int(jq[m].sum()),
                                            'sign_only': binary(jq[m], jwh[m].astype(float), 0.5),
                                            'sign_or_face': binary(jq[m], pred.astype(float), 0.5),
                                            'statement_false_alarm': float(pred[~jq[m]].mean())}

    # brauzer üçün: standartlaşdırma çəkilərə qatlanır (z = (f - mu) / sd → coef / sd)
    model = {
        'kind': 'nonmanual', 'nm_version': NM.NM_VERSION, 'feature_version': FT.FEATURE_VERSION, 'fps': NM.FPS,
        'frame_names': NM.FRAME_NAMES, 'stats': NM.STATS, 'neutral': args.neutral, 'min_rest': 5,
        'min_face': args.min_face, 'n_clip': int(F.shape[1]), 'wh_labels': WH_LABELS,
        'heads': {k: {'coef': [round(float(v), 7) for v in c / sd], 'bias': round(float(b - (c * mu / sd).sum()), 7),
                      't': round(t_, 4)} for k, (c, b, t_, _) in heads.items()},
        'trained_on': 'AzSLD train signers (ml/splits/sentences_split.csv), E31',
    }
    os.makedirs(args.run, exist_ok=True)
    with open(os.path.join(args.run, 'nonmanual.json'), 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False, separators=(',', ':'))
    with open(os.path.join(args.run, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1, default=float)
    np.save(os.path.join(args.run, 'scores_azsld.npy'), np.stack([s, s_n], 1))
    np.save(os.path.join(args.run, 'scores_jd.npy'), np.stack([js, js_n], 1))
    print(json.dumps(report, indent=1, default=lambda v: round(float(v), 4)))


def predict(model: dict, raw: dict, aspect: float, fps: float = NM.FPS):
    """Brauzerdəki kimi bir klip: {'yn': p, 'neg': p, 'face_rate': r} (üz azdırsa p None)."""
    raw = NM.to_fps(raw, fps)
    x, u = NM.frame_features(raw, aspect), NM.hands_up(raw, aspect)
    f = NM.clip_features(x, u, NM.neutral(x, u, model['neutral'], model['min_rest']))
    out = {'face_rate': float(f[-1])}
    for k, h in model['heads'].items():
        z = float(np.dot(np.asarray(h['coef'], np.float64), f.astype(np.float64)) + h['bias'])
        out[k] = 1 / (1 + np.exp(-z)) if f[-1] >= model['min_face'] else None
        out[f'z_{k}'] = z
    return out


def _synthetic(T, rng, template, template_blend, kind):
    """Sintetik iştirakçı (real adamın verisi yoxdur): üz — AzSLD train kadrlarının orta forması (çoxlu adamın
    ortası), başın dönməsi, əyilməsi, yerdəyişməsi; blendshape-lər ortalarının ətrafında hamar təsadüfi (onların orta
    və std-si də çoxlu kadrdan); əllər ortada qalxır."""
    from azsl.features import POSE_IDX
    t = np.linspace(0, 1, T)
    face = np.empty((T, len(NM.FACE_IDX), 2), np.float32)
    yaw = 0.08 * np.sin(2 * np.pi * t * (3 if kind == 'shake' else 1)) + 0.02 * rng.standard_normal(T)
    pitch = 0.05 * np.sin(np.pi * t) * (2 if kind == 'nod' else 1)
    roll = 0.04 * np.sin(3 * t)
    c = template.mean(0)
    for i in range(T):
        R = np.array([[np.cos(roll[i]), -np.sin(roll[i])], [np.sin(roll[i]), np.cos(roll[i])]])
        p = (template - c) * [1 - abs(yaw[i]) * 0.3, 1 - abs(pitch[i]) * 0.2]
        p = p @ R.T + c + [yaw[i] * 0.03, pitch[i] * 0.03]
        face[i] = p + 0.0005 * rng.standard_normal(p.shape)
    lvl, spread = template_blend
    blend = np.clip(lvl + spread * (np.sin(np.outer(t, rng.uniform(1, 6, 52)) * np.pi) + 0.3 * rng.standard_normal((T, 52))),
                    0, 1).astype(np.float32)
    blend[:, 0] = 0
    pose = np.zeros((T, len(POSE_IDX), 3), np.float32)
    base = np.array([[0.50, 0.25], [0.66, 0.42], [0.34, 0.42], [0.72, 0.62], [0.28, 0.62],
                     [0.62, 0.55], [0.38, 0.55], [0.60, 0.95], [0.40, 0.95]])
    pose[..., :2] = base + 0.004 * rng.standard_normal((T, len(POSE_IDX), 2))
    hand = np.full((T, 2, 21, 3), np.nan, np.float32)
    a, b = int(T * 0.2), int(T * 0.8)
    for s, cx in ((0, 0.42), (1, 0.58)):
        hand[a:b, s, :, 0] = cx + 0.03 * rng.standard_normal((b - a, 21))
        hand[a:b, s, :, 1] = 0.5 + 0.03 * rng.standard_normal((b - a, 21))
        hand[a:b, s, :, 2] = 0.0
    hand[:a // 2, 0, :, :2] = [0.4, 0.98]               # dizdəki əl: görünür, amma qalxmayıb
    hand[:a // 2, 0, :, 2] = 0
    if kind == 'gaps':                                   # üz bəzi kadrlarda yoxdur
        face[rng.random(T) < 0.3] = np.nan
    if kind == 'noface':
        face[:] = np.nan
        blend[:] = np.nan
    if kind == 'norest':                                 # əllər bütün klipdə qalxıb: neytral bütün klipin medianı
        hand[:, 1, :, 0] = 0.58
        hand[:, 1, :, 1] = 0.5
        hand[:, 1, :, 2] = 0
    face[..., 0] = np.where(np.isfinite(face[..., 0]), face[..., 0], np.nan)
    return {'hand': hand, 'pose': pose, 'face': face, 'blend': blend}


def fixture(args):
    """ml/fixtures/nonmanual_v<N>.{json,bin}: sintetik kliplər + Python-un kadr əlamətləri, əllər, neytral, klip
    vektoru və başlıqların ehtimalları (installed modellə). tools/verify_nonmanual.mjs eyni rəqəmləri tələb edir."""
    with open(args.model, encoding='utf-8') as f:
        model = json.load(f)
    ids, X, up = load(args.out, 'azsld')
    # üz şablonu: AzSLD train kadrlarının orta forması — reduce()-dan əvvəlki xy, kadr başına normallaşdırılmış
    import pandas as pd
    lab = pd.read_csv(os.path.join(args.out, 'azsld_labels.csv'), dtype={'folder': str}).set_index('id').loc[ids]
    dirs = sorted(glob.glob(os.path.join(args.kaggle, '02-sentences-shard0', 'landmarks', 'sentences_cam2', 'shard*')))
    store = LandmarkStore(dirs)
    tr_ids = [i for i in lab.index[lab.split == 'train'] if i in store.index.index][:200]
    acc, bl = [], []
    for rid in tr_ids:
        a = store.get(rid)
        fc = a['face'][:, :, :2]
        ok = np.isfinite(fc[:, 0, 0]) & np.isfinite(a['blend'][:, 0])
        acc.append(fc[ok] - fc[ok].mean(1, keepdims=True))
        bl.append(a['blend'][ok])
    template = np.concatenate(acc).mean(0) + [0.5, 0.2]
    bl = np.concatenate(bl)
    template_blend = (bl.mean(0), 0.5 * bl.std(0))
    rng = np.random.default_rng(31)
    cases, blobs, off = [], [], 0

    def put(a):
        nonlocal off
        a = np.ascontiguousarray(a, np.float32)
        blobs.append(a.tobytes())
        d = {'offset': off, 'shape': list(a.shape)}
        off += a.size
        return d
    for name, T, kind, aspect, fps in [('plain', 90, 'plain', 4 / 3, 30), ('head shake', 120, 'shake', 4 / 3, 30),
                                       ('nod, 16:9', 100, 'nod', 16 / 9, 30), ('face gaps', 80, 'gaps', 4 / 3, 30),
                                       ('hands up throughout', 60, 'norest', 4 / 3, 30), ('no face', 50, 'noface', 4 / 3, 30),
                                       ('50 fps camera', 150, 'plain', 16 / 9, 50)]:
        raw = _synthetic(T, rng, template, template_blend, kind)
        r30 = NM.to_fps(raw, fps)
        x, u = NM.frame_features(r30, aspect), NM.hands_up(r30, aspect)
        base = NM.neutral(x, u, model['neutral'], model['min_rest'])
        cf = NM.clip_features(x, u, base)
        p = predict(model, raw, aspect, fps)
        cases.append({'name': name, 'T': T, 'aspect': aspect, 'fps': fps,
                      **{k: put(raw[k]) for k in ('hand', 'pose', 'face', 'blend')},
                      'frames': put(x), 'up': [int(v) for v in u], 'active': [int(v) for v in NM.active_range(u)],
                      'neutral': put(base), 'clip': put(cf),
                      'p': {k: (None if v is None else float(v)) for k, v in p.items()}})
    meta = {'nm_version': NM.NM_VERSION, 'frame_names': NM.FRAME_NAMES, 'stats': NM.STATS,
            'tolerance': 2e-4, 'p_tolerance': 1e-4, 'cases': cases,
            'note': 'synthetic, seeded (face shape = mean of many AzSLD train frames, no real person); ml/nonmanual.py fixture'}
    path = os.path.join(ROOT, 'fixtures', f'nonmanual_v{NM.NM_VERSION}')
    with open(path + '.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=0)
    with open(path + '.bin', 'wb') as f:
        f.write(b''.join(blobs))
    print(f'{path}.json/.bin: {len(cases)} cases, {off * 4 / 1e6:.2f} MB')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['cache', 'train', 'fit', 'fixture'])
    ap.add_argument('--run', default='D:/chevir/eval/nonmanual/final')
    ap.add_argument('--model', default='public/tercume/assets/nonmanual.json')
    ap.add_argument('--det', default='D:/chevir/runs/v5_ctc/cont/seq_splice_ctc/detections_test.csv')
    ap.add_argument('--jd-det', default='D:/chevir/eval/final_ctc_hold/sentences.json')
    ap.add_argument('--kaggle', default='D:/chevir/kaggle')
    ap.add_argument('--landmarks', default='D:/chevir/landmarks')
    ap.add_argument('--out', default='D:/chevir/eval/nonmanual')
    ap.add_argument('--target', default='question', choices=['question', 'neg'])
    ap.add_argument('--neutral', default='rest')
    ap.add_argument('--groups', default='all')
    ap.add_argument('--lam', default='0.0001,0.001,0.01,0.1')
    ap.add_argument('--min-face', type=float, default=0.5)
    a = ap.parse_args()
    {'cache': cache, 'train': train, 'fit': fit, 'fixture': fixture}[a.cmd](a)
