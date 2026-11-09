"""
JestDili.az lüğətinin videolarından avatarın söz işarələri (roadmap D2; icazə: docs/DATA.md).

Hər söz bir video, bir nəfər, studiyada, qarşıdan: build_avatar_words.py-nin emalı (MediaPipe Pose lite + Hands +
Face, VIDEO rejimi, hər 2 kadrdan biri, istirahət kadrları kəsilir) eynilə işləyir; "8 iştirakçının ən tipiki" əvəzinə
keyfiyyət həddi (--min-score). Adın mötərizəsi məna qeydidir: "bel (alət)" → mətn "bel", sense "alət";
"almanya (variant 2)" → eyni sözün ikinci işarəsi.

  .venv/Scripts/python -I tools/build_jestdili_words.py [--all] [--limit N] [--workers 16]
  → data/jestdili/avatar/<id>.json + index.json (public/tercume/words/ ilə eyni format; yalnız yerli)
--all: lüğətimizdə olan sözləri də (müqayisə üçün); standart — yalnız yeni sözlər.

  .venv/Scripts/python -I tools/build_jestdili_words.py --install
  → hər sözün ilk variantı (eyni yazılışın başqa mənaları da) public/tercume/words/-ə; indeksdə AzSLD sözlərindən
    sonra, "sources"-da JestDili.az kadrlarından üzün diapazonu (words.js "jd-" sözlərinə onu qoşur)
"""
import argparse
import json
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_avatar_words import fingers_close, process  # noqa: E402
from build_words import az_lower, word_id  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
J = ROOT / 'data' / 'jestdili'
WORDS = ROOT / 'public' / 'tercume' / 'words'
PREFIX = 'jd-'
SOURCE = 'JestDili.az (icazə ilə, docs/DATA.md), tools/build_jestdili_words.py'
SENSE = re.compile(r'^(.*?)\s*\(([^()]*)\)\s*$')
VARIANT = re.compile(r'variant\s*(\d+)', re.I)


def parse(name):
    """"bel (alət)" → ("bel", "alət", 1); "almanya (variant 2)" → ("almanya", None, 2)."""
    name = ' '.join(name.split())
    m = SENSE.match(name)
    text, note = (m.group(1), m.group(2).strip()) if m else (name, None)
    v = VARIANT.search(note or '')
    return az_lower(text), (None if v else note), int(v.group(1)) if v else 1


def install():
    """data/jestdili/avatar/ → public/tercume/words/ (E24, D2)."""
    import numpy as np
    src = J / 'avatar'
    jd = json.loads((src / 'index.json').read_text(encoding='utf8'))
    pick = [w for w in jd['words'] if w.get('variant', 1) == 1]
    datas = {w['id']: json.loads((src / f"{w['id']}.json").read_text(encoding='utf8')) for w in pick}
    # Eyni yazılışın bir neçə işarəsi (bel: alət / bədən): mətn ən keyfiyyətlisinə gedir (words.js birincini götürür)
    pick.sort(key=lambda w: (w['text'], -datas[w['id']]['quality']))
    # Üzün diapazonu bu adamın öz kadrlarından (build_avatar_words.py kimi: median sakit üz, 99-cu faiz güclü ifadə)
    faces = np.asarray([fr['f']['s'] for d in datas.values() for fr in d['frames'] if fr.get('f')])
    index = json.loads((WORDS / 'index.json').read_text(encoding='utf8'))
    if index.get('face') != jd.get('face'):
        sys.exit('üz blendshape-lərinin siyahısı AzSLD lüğətindəki ilə eyni deyil')
    for old in WORDS.glob(f'{PREFIX}*.json'):
        old.unlink()
    for w in pick:
        shutil.copyfile(src / f"{w['id']}.json", WORDS / f"{w['id']}.json")
    index['words'] = [w for w in index['words'] if not w['id'].startswith(PREFIX)] + pick
    index['count'] = len(index['words'])
    index['sources'] = [s for s in index.get('sources', []) if s['prefix'] != PREFIX] + [{
        'prefix': PREFIX, 'source': SOURCE, 'count': len(pick),
        'faceNeutral': [int(v) for v in np.median(faces, axis=0)],
        'faceFull': [int(v) for v in np.percentile(faces, 99, axis=0)], 'chest': None}]
    (WORDS / 'index.json').write_text(json.dumps(index, ensure_ascii=False, separators=(',', ':')), encoding='utf8')
    size = sum((WORDS / f"{w['id']}.json").stat().st_size for w in pick)
    print(f"{len(pick)} JestDili.az işarəsi ({size / 1e6:.0f} MB), lüğətdə cəmi {index['count']} söz; "
          f"{len(pick) - len({w['text'] for w in pick})} işarə eyni yazılışın ikinci mənasıdır")


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--install', action='store_true', help='hazır işarələri tətbiqin lüğətinə köçür (emal etmədən)')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--min-score', type=float, default=0.45)
    ap.add_argument('--pose-model', type=Path, default=ROOT / 'public' / 'tercume' / 'assets' / 'pose_landmarker_lite.task')
    args = ap.parse_args()
    if args.install:
        return install()

    ours = json.loads((WORDS / 'index.json').read_text(encoding='utf8'))
    have = {az_lower(t) for w in ours['words'] if not w['id'].startswith(PREFIX) for t in [w['text'], *w.get('alt', [])]}
    cats = {c['id']: c['name'] for c in json.loads((J / 'categories.json').read_text(encoding='utf8'))}
    words = json.loads((J / 'words.json').read_text(encoding='utf8'))
    todo = []
    for w in words:
        text, sense, variant = parse(w['name'])
        video = J / 'words' / f"{w['id']}.mp4"
        if not video.exists() or (text in have and not args.all):
            continue
        todo.append({'jd': w['id'], 'text': text, 'sense': sense, 'variant': variant, 'category': cats.get(w['categoryId']),
                     'description': w.get('description'), 'video': video})
    todo = todo[:args.limit] if args.limit else todo
    cache = J / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(None, str(t['video']), 1, None, str(args.pose_model), 3, 8, 2, str(cache / f"{t['jd']}.json")) for t in todo]
    print(f'{len(todo)} söz emal olunacaq', flush=True)
    results = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, r in enumerate(ex.map(process, jobs, chunksize=2), 1):
            results.append(r)
            if k % 100 == 0:
                print(f'  {k}/{len(jobs)}', flush=True)

    out = J / 'avatar'
    out.mkdir(exist_ok=True)
    for old in out.glob('*.json'):
        old.unlink()
    index, dropped, used = [], [], set()
    face_keys = next((r['face'] for r in results if r.get('face')), None)
    for t, r in zip(todo, results):
        if not r['rec'] or r['score'] < args.min_score:
            dropped.append((t['text'], round(r['score'], 2)))
            continue
        rec = r['rec']
        wid = word_id('jd-' + t['text'] + (f"-{t['variant']}" if t['variant'] > 1 else ''), used)
        used.add(wid)
        data = {'id': wid, 'text': t['text'], 'label': t['text'].upper(), **rec, 'video': t['jd'],
                'source': 'JestDili.az (icazə ilə)', 'quality': round(r['score'], 3), 'candidates': 1}
        for k in ('sense', 'category', 'description'):
            if t[k]:
                data[k] = t[k]
        if t['variant'] > 1:
            data['variant'] = t['variant']
        if rec['hands'] == 'LR':
            data['contact'] = bool(fingers_close(rec) > 0.25)
        (out / f'{wid}.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf8')
        entry = {'id': wid, 'text': t['text'], 'hands': rec['hands'], 'frames': len(rec['frames'])}
        if t['sense']:
            entry['sense'] = t['sense']
        if t['variant'] > 1:
            entry['variant'] = t['variant']
        index.append(entry)
    index.sort(key=lambda w: (w['text'], w.get('variant', 1)))
    (out / 'index.json').write_text(json.dumps({'source': 'JestDili.az (icazə ilə), tools/build_jestdili_words.py',
                                                'count': len(index), 'face': face_keys, 'words': index},
                                               ensure_ascii=False, separators=(',', ':')), encoding='utf8')
    total = sum(p.stat().st_size for p in out.glob('*.json'))
    print(f'{len(index)} söz yazıldı ({total / 1e6:.1f} MB), {len(dropped)} atıldı: {dropped[:15]}')


if __name__ == '__main__':
    main()
