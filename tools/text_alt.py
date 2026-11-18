"""
ml/text_alt.csv (eyni işarənin başqa yazılışları: burda, rəhmətə getmək, birdəfəlik) → avatar lüğətinin `alt` sahələri
(public/tercume/words: index.json və sözün öz faylı).

AzSLD sözlərinin alt-ı yalnız bu cədvəldəndir (tools/build_avatar_words.py). Sonradan əlavə olunan mənbənin
(index.json "sources", JestDili.az "jd-") sözlərinin öz alt-ları var ("xəstə, xəstəlik" → alt xəstəlik, E27): onlar
qalır. İşlədənlər: build_avatar_words.py --alt-only və apply_native_review.py (n8).
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / 'ml' / 'text_alt.csv'


def alt_map(rows=None):
    """text → [alt, ...] (cədvəldəki sıra ilə, təkrarsız). rows: csv.DictReader sətirləri; None — ml/text_alt.csv."""
    if rows is None:
        with open(CSV, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    m = {}
    for r in rows:
        alts = m.setdefault(r['text'], [])
        if r['alt'] not in alts:
            alts.append(r['alt'])
    return m


def sync_alt(words_dir, after, before=None, dry_run=False):
    """
    Lüğətin alt sahələri `after`-ə (text → [alt]) uyğunlaşır: cədvəldən gələn köhnə alt-lar (`before`) çıxır, `after`-in
    alt-ları əlavə olunur, qalanları (mənbənin öz alt-ları) qalır. `before` None: AzSLD sözünün bütün alt-ı (onlar
    yalnız cədvəldəndir) və əlavə mənbənin sözündə `after`-dəkilər; əlavə mənbənin sözündən cədvəldən çıxmış sətrin
    alt-ını yalnız `before` verilib çıxarır (apply_native_review.py). Qaytarır: (dəyişən söz sayı, lüğətdə olmayan mətnlər).
    """
    words_dir = Path(words_dir)
    index_path = words_dir / 'index.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    prefixes = tuple(s['prefix'] for s in index.get('sources', []))
    changed = 0
    for entry in index['words']:
        t, old = entry['text'], entry.get('alt', [])
        if before is not None:
            drop = before.get(t, [])
        else:
            drop = after.get(t, []) if entry['id'].startswith(prefixes) else old
        alt = [x for x in old if x not in drop]
        alt += [x for x in after.get(t, []) if x not in alt]
        if alt == old:
            continue
        changed += 1
        if dry_run:
            continue
        path = words_dir / f"{entry['id']}.json"
        data = json.loads(path.read_text(encoding='utf-8'))
        for d in (entry, data):
            if alt:
                d['alt'] = alt            # sahə varsa yerində qalır (faylın qalanı dəyişmir)
            else:
                d.pop('alt', None)
        path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    if changed and not dry_run:
        index_path.write_text(json.dumps(index, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    return changed, sorted(set(after) - {w['text'] for w in index['words']})
