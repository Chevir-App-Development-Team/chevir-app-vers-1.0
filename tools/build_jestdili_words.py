"""
JestDili.az lüğətinin videolarından avatarın söz işarələri (roadmap D2; icazə: docs/DATA.md).

Hər söz bir video, bir nəfər, studiyada, qarşıdan: build_avatar_words.py-nin emalı (MediaPipe Pose lite + Hands +
Face, VIDEO rejimi, hər 2 kadrdan biri, istirahət kadrları kəsilir) eynilə işləyir; "8 iştirakçının ən tipiki" əvəzinə
keyfiyyət həddi (--min-score). Adın mötərizəsi məna qeydidir: "bel (alət)" → mətn "bel", sense "alət";
"almanya (variant 2)" → eyni sözün ikinci işarəsi.

  .venv/Scripts/python -I tools/build_jestdili_words.py [--all] [--limit N] [--workers 16]
  → data/jestdili/avatar/<id>.json + index.json (public/tercume/words/ ilə eyni format; yalnız yerli)
--all: lüğətimizdə olan sözləri də (müqayisə üçün); standart — yalnız yeni sözlər.
"""
import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_avatar_words import fingers_close, process  # noqa: E402
from build_words import az_lower, word_id  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
J = ROOT / 'data' / 'jestdili'
SENSE = re.compile(r'^(.*?)\s*\(([^()]*)\)\s*$')
VARIANT = re.compile(r'variant\s*(\d+)', re.I)


def parse(name):
    """"bel (alət)" → ("bel", "alət", 1); "almanya (variant 2)" → ("almanya", None, 2)."""
    name = ' '.join(name.split())
    m = SENSE.match(name)
    text, note = (m.group(1), m.group(2).strip()) if m else (name, None)
    v = VARIANT.search(note or '')
    return az_lower(text), (None if v else note), int(v.group(1)) if v else 1


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--min-score', type=float, default=0.45)
    ap.add_argument('--pose-model', type=Path, default=ROOT / 'public' / 'tercume' / 'assets' / 'pose_landmarker_lite.task')
    args = ap.parse_args()

    ours = json.loads((ROOT / 'public' / 'tercume' / 'words' / 'index.json').read_text(encoding='utf8'))
    have = {az_lower(t) for w in ours['words'] for t in [w['text'], *w.get('alt', [])]}
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
