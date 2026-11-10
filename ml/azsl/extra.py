"""AzSLD-dən kənar təlim datası: JestDili.az lüğətinin videoları (bir nəfər, studiya, ayaq üstə, 1080p;
tools/extract_jestdili.py → LandmarkStore). Söz modeli üçün təcrid olunmuş işarələr (train.Bank), cümlə modeli üçün
"yazılar" (continuous.SeqBank: istirahət <blank>, işarə öz etiketi): spliced cümlələrə də girir, yeni adam və kamera
kimi. Etiket: JestDili.az-ın sözü modelin lüğətinin sözüdürsə (etiketin düzəldilmiş yazılışı = sözün yazılışı)."""
import csv
import os

import numpy as np

from . import features as FT
from . import train as TR
from .store import LandmarkStore

IGNORE = -100


def az_lower(s):
    return s.replace('I', 'ı').replace('İ', 'i').lower()


def label_text(path):
    """ml/label_text.csv: annotasiyanın yazı xətaları (GUN → gün)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding='utf8') as f:
        return {r['label']: r['text'] for r in csv.DictReader(f)}


def jestdili_glosses(store, labels, fix=None, spelling=None, merge=None):
    """{item_id: model label} JestDili.az sözlərindən modelin lüğətində olanlara. labels: modelin etiketləri
    (birləşmədən sonra); fix: label_text; spelling/merge: notebook-dakı yazılış və eyni-işarə birləşmələri."""
    fix, spelling, merge = fix or {}, spelling or {}, merge or {}
    canon = lambda g: merge.get(spelling.get(g, g), spelling.get(g, g))      # noqa: E731
    by_text = {}
    for g in labels:
        by_text.setdefault(az_lower(fix.get(g, g)), canon(g))
    out = {}
    for item_id, r in store.index.iterrows():
        name = az_lower(str(r.get('name', ''))).split('(')[0].strip()
        if name in by_text and by_text[name] in labels:
            out[item_id] = by_text[name]
    return out


def holdout(gloss_of, frac):
    """JestDili.az sözlərinin bir hissəsi təlimdən kənarda (bu adamın görünməmiş sözləri ilə ölçmək üçün): etiketin
    md5-i ilə, hər dəfə eyni. → (təlim üçün {item_id: etiket}, kənarda qalan etiketlər)"""
    import hashlib
    out = {g for g in set(gloss_of.values())
           if int(hashlib.md5(g.encode('utf-8')).hexdigest()[:8], 16) / 0xFFFFFFFF < frac}
    return {k: g for k, g in gloss_of.items() if g not in out}, sorted(out)


def active_span(raw, pad=2):
    """İşarənin kadrları: əlin göründüyü ilk və son kadr (ayaq üstə dayanan adamın istirahətdəki əlləri kadrdan
    aşağıdadır). Əl yoxdursa (0, T)."""
    seen = np.isfinite(raw['hand'][:, :, 0, 0].astype(np.float32)).any(1)
    idx = np.flatnonzero(seen)
    T = len(seen)
    if len(idx) < 3:
        return 0, T
    return max(0, int(idx[0]) - pad), min(T, int(idx[-1]) + 1 + pad)


def load_signs(dirs, gloss_of, vocab, margin=8, whole=False, tag='jd', log=print):
    """Söz modeli: hər klipin işarəsi ±margin kontekstlə → train.Bank (hamısı təlim üçün). whole: klip bütövlükdə
    işarədir (Slovo), yoxsa işarə əlin göründüyü kadrlardır (JestDili.az)."""
    store = LandmarkStore(dirs)
    bank = TR.Bank()
    for item_id, g in gloss_of.items():
        if g not in vocab:
            continue
        raw = FT.reduce(store.get(item_id, dtype=np.float16))
        T = raw['hand'].shape[0]
        a, b = (0, T) if whole else active_span(raw)
        if b - a < 4:
            continue
        lo, hi = max(0, a - margin), min(T, b + margin)
        aspect = float(store.index.at[item_id, 'width']) / float(store.index.at[item_id, 'height'])
        bank.add({k: v[lo:hi] for k, v in raw.items()}, vocab[g], aspect, (a - lo, b - lo), f'{tag}|{item_id}')
    log(f'{tag}: {len(bank)} sign clips of {len(set(gloss_of.values()) & set(vocab))} vocabulary words')
    return bank.done()


def slovo_glosses(dirs, rename):
    """Slovo klipləri → {item_id: AzSLD etiketi}, yalnız ml/slovo_azsld.csv-də AzSLD adı olan işarələr."""
    idx = LandmarkStore(dirs).index
    return {i: rename[str(n)] for i, n in idx['name'].items() if str(n) in rename}


def load_recordings(dirs, gloss_of, index, edge=2, log=print):
    """Cümlə modeli: hər klip bir "yazı": istirahət <blank> (0), işarə öz etiketi, sərhədin ±edge kadrı IGNORE."""
    from . import continuous as CT
    store = LandmarkStore(dirs)
    bank = CT.SeqBank()
    for item_id, g in gloss_of.items():
        if g not in index:
            continue
        raw = FT.reduce(store.get(item_id, dtype=np.float16))
        T = raw['hand'].shape[0]
        a, b = active_span(raw, pad=0)
        if b - a < 4 or T < 8:
            continue
        y = np.zeros(T, np.int64)
        y[a:b] = index[g]
        for e in (a, b):
            y[max(0, e - edge):min(T, e + edge)] = IGNORE
        aspect = float(store.index.at[item_id, 'width']) / float(store.index.at[item_id, 'height'])
        bank.add(raw, y, aspect, f'jd|{item_id}')
    log(f'JestDili.az: {len(bank)} recordings ({bank.n:,} frames)')
    return bank.done()


def merge_seq(a, b):
    """İki hazır SeqBank → bir (a-nın yazıları əvvəl)."""
    from . import continuous as CT
    if not len(b):
        return a
    out = CT.SeqBank()
    out.chunks = out.lab = None
    out.data = {k: np.concatenate([a.data[k], b.data[k]]) for k in TR.KEYS}
    out.labels = np.concatenate([a.labels, b.labels])
    out.rows = a.rows + [(off + a.n, *rest) for off, *rest in b.rows]
    out.n = a.n + b.n
    out.ids = a.ids + b.ids
    return out


def subset_view(bank, ids, side):
    """SeqBank-ın bu yazıları, yalnız bir kamera: side=False ön, True yan (id-də "|side")."""
    from . import continuous as CT
    out = CT.SeqBank()
    for i, r in enumerate(bank.ids):
        if r.split('|')[0] in ids and ('|side' in r) == side:
            raw, y, aspect, rid = bank.get(i)
            out.add(raw, y, aspect, rid)
    return out.done()


def load_slovo(dirs, margin=8, rename=None, log=print):
    """Slovo (rus işarə dili, tools/extract_slovo.py) → ({'train', 'val'}: train.Bank, etiketlər). Hər klip bütöv
    bir işarədir (Kaggle versiyasında videolar begin..end-ə kəsilib; "no_event" işarəsizdir). Slovo-nun öz
    bölgüsü: train → train, test → val (ön öyrətmənin seçimi üçün; adamlar ayrıdır). rename: {rus adı: AzSLD
    etiketi} (ml/slovo_azsld.csv): ortaq işarələr AzSLD-nin adını alır, eyni etiketə düşənlər bir sinif olur."""
    store = LandmarkStore(dirs)
    idx = store.index
    rename = rename or {}
    label = lambda n: rename.get(str(n), str(n))     # noqa: E731
    names = sorted(set(map(label, idx['name'])))
    vocab = {g: i for i, g in enumerate(names)}
    banks = {'train': TR.Bank(), 'val': TR.Bank()}
    skipped = 0
    for item_id, r in idx.iterrows():
        raw = FT.reduce(store.get(item_id, dtype=np.float16))
        T = raw['hand'].shape[0]
        a, b = 0, T          # Kaggle-dakı videolar artıq işarənin kadrlarına (begin..end) kəsilib
        if b - a < 4:
            skipped += 1
            continue
        lo, hi = max(0, a - margin), min(T, b + margin)
        aspect = float(r['width']) / float(r['height'])
        split = 'train' if str(r['split']) == 'train' else 'val'
        banks[split].add({k: v[lo:hi] for k, v in raw.items()}, vocab[label(r['name'])], aspect, (a - lo, b - lo),
                         f'slovo|{item_id}')
    log(f"Slovo: {len(banks['train'])} train / {len(banks['val'])} val signs, {len(names)} labels, {skipped} too short")
    return {k: b.done() for k, b in banks.items()}, names
