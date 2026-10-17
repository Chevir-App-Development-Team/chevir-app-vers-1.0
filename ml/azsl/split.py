"""Signer-independent splits for AzSLD.

AzSLD ships no signer ids and Words_200 clips are UUID-named, which is how the old
I3D repo ended up with the same signers in train and test. Here every Sentences
recording gets a pseudo-signer:

  1. faces — a mean SFace embedding per recording (landmarks.Extractor._embed),
     clustered per RECORDING (average linkage, cosine) at similarity `sim`. Not per
     sitting: 29 of 179 sittings have two people taking turns, and a sitting's average
     resembles neither. The clusters are identical from 0.50 to 0.60 (a clear gap).
  2. split groups — pseudo-signers are tied if any of their sittings look alike
     (split_groups), so a look-alike can never sit on both sides of the split.
  3. one group → split assignment, chosen so the 200-word vocabulary is covered in
     val/test, and reused for every task (words, sentence-cut glosses, sentences),
     so a test signer is never seen in training by any model.

Sessions (recording timestamps, `gap_min`) only define sittings for step 2 and give
recordings without a face their sitting's signer. Real signer ids from the dataset
authors can be passed as an override CSV (rec_id, signer).
"""
import numpy as np


def sessions(recs, gap_min: float = 15.0):
    """recs: DataFrame with rec_id, stamp → Series rec_id → session id."""
    import pandas as pd
    r = recs.dropna(subset=['stamp']).sort_values('stamp')
    sid, prev, out = -1, None, {}
    for rid, t in zip(r.rec_id, r.stamp):
        if prev is None or t.date() != prev.date() or (t - prev).total_seconds() > gap_min * 60:
            sid += 1
        out[rid] = f's{sid:04d}'
        prev = t
    return pd.Series(out, name='session')


def _norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def cluster_signers(rec_ids, emb, sim: float = 0.55):
    """→ (dict rec_id → signer, report dict). `emb` rows are L2-normalised SFace means."""
    from sklearn.cluster import AgglomerativeClustering
    if len(rec_ids) == 1:
        labels = np.zeros(1, int)
    else:
        labels = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average',
                                         distance_threshold=1 - sim).fit_predict(emb)
    signer = {rid: f'p{lab:03d}' for rid, lab in zip(rec_ids, labels)}
    sizes = np.bincount(labels)
    labs = np.unique(labels)
    C = np.stack([_norm(emb[labels == lab].mean(0)) for lab in labs])
    S = C @ C.T
    np.fill_diagonal(S, -1)
    # purity: share of face pairs < 0.4 inside a cluster (two people in one row shows up here)
    rng, mixed = np.random.default_rng(0), []
    for lab in labs[sizes[labs] >= 20]:
        ix = np.where(labels == lab)[0]
        ix = rng.choice(ix, min(400, len(ix)), replace=False)
        P = emb[ix] @ emb[ix].T
        mixed.append(float((P[np.triu_indices(len(ix), 1)] < 0.4).mean()))
    report = {'signers': int(len(labs)), 'tiny_signers_lt20_recordings': int((sizes < 20).sum()),
              'worst_cluster_mixed_pairs': round(max(mixed, default=0.0), 4),
              'nearest_other_signer_sim': [round(float(x), 3) for x in np.sort(S.max(1))[::-1][:10]]}
    return signer, report


def assign_signers(recs, emb_ids, emb, sim=0.55, gap_min=15.0, override=None):
    """recs → DataFrame rec_id, session, signer, has_emb. Recordings without a face fall back to
    their session's signer; with neither they become 'unk-<session>'."""
    ses = sessions(recs, gap_min)
    out = recs[['rec_id']].copy()
    out['session'] = out.rec_id.map(ses).fillna('nosession')
    has = set(emb_ids)
    out['has_emb'] = out.rec_id.isin(has)
    signer, report = ({}, {'signers': 0})
    if len(emb_ids):
        signer, report = cluster_signers(list(emb_ids), _norm(np.asarray(emb, np.float32)), sim)
    out['signer'] = out.rec_id.map(signer)
    maj = out.dropna(subset=['signer']).groupby('session').signer.agg(lambda s: s.value_counts().index[0])
    miss = out.signer.isna()
    out.loc[miss, 'signer'] = out.loc[miss, 'session'].map(maj)
    out['signer'] = out.signer.fillna('unk-' + out.session)
    if override is not None and len(override):
        o = dict(zip(override.rec_id, override.signer.astype(str)))
        out['signer'] = [o.get(r, s) for r, s in zip(out.rec_id, out.signer)]
        report['override'] = len(o)
    # same signer recording the same sentence twice is rare (retakes); many means merged people
    folder = out.rec_id.str.split('/').str[0]
    rep = (out.assign(folder=folder).groupby(['signer', 'folder']).size() > 1).sum()
    report['signer_sentence_repeats'] = int(rep)
    report['recordings'] = len(out)
    report['without_face'] = int((~out.has_emb).sum())
    return out, report


def _units(signers, emb_ids, emb):
    """(session, signer) units — one sitting of one person — and their centroids."""
    sig = dict(zip(signers.rec_id, signers.signer))
    ses = dict(zip(signers.rec_id, signers.session))
    E = np.asarray(emb, np.float32)
    groups = {}
    for i, rid in enumerate(emb_ids):
        if rid in sig:
            groups.setdefault((ses[rid], sig[rid]), []).append(i)
    keys = sorted(groups)
    C = np.stack([_norm(E[groups[k]].mean(0)) for k in keys]) if keys else np.zeros((0, 1))
    return keys, C, {k: len(v) for k, v in groups.items()}


def split_groups(signers, emb_ids, emb, link_sim=0.50):
    """Tie pseudo-signers that share a look-alike sitting: if ANY (sitting, signer) unit of one is
    >= link_sim similar to any unit of the other, they always land in the same split. Units, not
    cluster averages: an average over many recordings hides one look-alike sitting. On AzSLD one
    person's sittings score 0.6+ and different people mostly < 0.5.
    Union-find over the pairs → dict signer → group id."""
    keys, C, _ = _units(signers, emb_ids, emb)
    parent = {n: n for n in signers.signer.unique()}

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    if len(keys) > 1:
        I, J = np.where(np.triu(C @ C.T, 1) >= link_sim)
        for i, j in zip(I, J):
            if keys[i][1] != keys[j][1]:
                parent[root(keys[i][1])] = root(keys[j][1])
    roots = sorted({root(n) for n in parent})
    gid = {r: f'g{k:02d}' for k, r in enumerate(roots)}
    return {n: gid[root(n)] for n in parent}


def closest_pairs(signers, emb_ids, emb, top=8, cross_split_only=False):
    """Most similar sittings of different pseudo-signers, with their splits — read next to the
    contact sheet. With split_groups() no pair at or above link_sim straddles two splits."""
    keys, C, n = _units(signers, emb_ids, emb)
    split = dict(zip(signers.signer, signers.split))
    S = np.triu(C @ C.T, 1)
    order = np.dstack(np.unravel_index(np.argsort(-S, axis=None), S.shape))[0]
    out = []
    for i, j in order:
        if j <= i or keys[i][1] == keys[j][1]:
            continue
        a, b = keys[i], keys[j]
        if cross_split_only and split.get(a[1]) == split.get(b[1]):
            continue
        out.append({'a': f'{a[1]} {a[0]}', 'b': f'{b[1]} {b[0]}', 'sim': round(float(S[i, j]), 3),
                    'splits': f'{split.get(a[1])}/{split.get(b[1])}', 'recordings': f'{n[a]}/{n[b]}'})
        if len(out) >= top:
            break
    return out


def pick_split(df, group_col: str, label_col: str, fracs=(0.7, 0.15, 0.15), seed: int = 0,
               trials: int = 4000, forced_train=()):
    """Random search over whole groups → dict group → 'train'|'val'|'test'.

    Score: classes (with ≥3 groups) missing from test/val, then distance from target
    sample fractions. Groups in `forced_train` (unknown provenance) never go to val/test.
    """
    rng = np.random.default_rng(seed)
    counts = df.groupby(group_col).size()
    labels_of = df.groupby(group_col)[label_col].agg(set)
    groups = [g for g in counts.index if g not in set(forced_train)]
    n_groups_per_label = df[~df[group_col].isin(forced_train)].groupby(label_col)[group_col].nunique()
    testable = set(n_groups_per_label[n_groups_per_label >= 3].index)
    N = counts[groups].sum()
    best, best_score = None, None
    for _ in range(trials):
        order = rng.permutation(groups)
        assign, n_test, n_val = {}, 0, 0
        for g in order:
            if n_test < fracs[2] * N:
                assign[g], n_test = 'test', n_test + counts[g]
            elif n_val < fracs[1] * N:
                assign[g], n_val = 'val', n_val + counts[g]
            else:
                assign[g] = 'train'
        cov = {s: set().union(*[labels_of[g] for g in assign if assign[g] == s] or [set()]) for s in ('val', 'test')}
        miss_t, miss_v = len(testable - cov['test']), len(testable - cov['val'])
        dev = abs(n_test / N - fracs[2]) + abs(n_val / N - fracs[1])
        score = (miss_t, miss_v, round(float(dev), 3))
        if best_score is None or score < best_score:
            best, best_score = assign, score
    for g in forced_train:
        best[g] = 'train'
    return best, {'missing_test': best_score[0], 'missing_val': best_score[1],
                  'fraction_error': best_score[2], 'testable_classes': len(testable)}


def folds(df, group_col: str, k: int = 5):
    """Train groups → fold 0..k-1, balancing sample counts (largest groups first)."""
    sizes = df.groupby(group_col).size().sort_values(ascending=False)
    load, out = [0] * k, {}
    for g, n in sizes.items():
        f = int(np.argmin(load))
        out[g], load[f] = f, load[f] + n
    return out


def baselines(df, label_col: str, split_col: str = 'split') -> dict:
    """Per split: samples, classes, and what always answering the most frequent train class scores."""
    train = df[df[split_col] == 'train']
    top = train[label_col].value_counts()
    major = top.index[0] if len(top) else None
    out = {'majority_class': major, 'train_classes': int(train[label_col].nunique())}
    for s in ('train', 'val', 'test'):
        d = df[df[split_col] == s]
        out[s] = {
            'samples': int(len(d)),
            'classes': int(d[label_col].nunique()),
            'majority_acc': round(float((d[label_col] == major).mean()), 4) if len(d) else None,
            'chance_balanced_acc': round(1 / max(1, d[label_col].nunique()), 4),
            'unseen_in_train': int(len(set(d[label_col]) - set(train[label_col]))) if s != 'train' else 0,
        }
    return out
