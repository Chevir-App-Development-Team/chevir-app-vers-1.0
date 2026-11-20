"""
JestDili.az videolarının landmark-ları (roadmap D2, R4, V4): AzSLD-nin Kaggle çıxarışı ilə eyni boru
(ml/azsl/landmarks.Extractor: eyni .task modelləri md5 ilə, VIDEO rejimi, iki əl + poza + üz + blendshape) və eyni
anbar formatı (ml/azsl/store.PartWriter: part-NNNNN.npz + manifest.csv), ona görə LandmarkStore və təlim kodu
onları AzSLD kimi oxuyur.

  .venv/Scripts/python -I tools/extract_jestdili.py --out D:/chevir/landmarks [--workers 20] [--only words|sentences]
  → <out>/jestdili_words/ (2,766 söz, hər biri bir video), <out>/jestdili_sentences/ (500 cümlə, mətni ilə)

Fasilə verilə bilər: yenidən işə salınanda hazır olanlar keçilir. Videolar və landmark-lar yerli qalır (git-də yox).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'ml'))
from azsl import landmarks, pipeline, store  # noqa: E402

J = ROOT / 'data' / 'jestdili'


def _file_job(task):
    """Diskdəki video (AzSLD-dəki _video_job zip üzvünü oxuyur)."""
    item_id, path, extra = task
    t0 = time.time()
    arrays, meta, face = pipeline._EX.video(path)
    meta.update(extra)
    meta['sec'] = round(time.time() - t0, 2)
    return item_id, arrays, meta, face


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--models', type=Path, default=None, help='.task modellərinin qovluğu (yoxdursa endirilir, md5 yoxlanır)')
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--only', choices=['words', 'sentences'])
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    paths = landmarks.ensure_models(str(args.models or args.out / 'models'))

    jobs = []
    if args.only != 'sentences':
        cats = {c['id']: c['name'] for c in json.loads((J / 'categories.json').read_text(encoding='utf8'))}
        tasks = []
        for w in json.loads((J / 'words.json').read_text(encoding='utf8')):
            v = J / 'words' / f"{w['id']}.mp4"
            if v.exists():
                tasks.append((w['id'], str(v), {'name': w['name'], 'category': cats.get(w.get('categoryId'), '')}))
        jobs.append(('jestdili_words', tasks))
    if args.only != 'words':
        tasks = []
        for s in json.loads((J / 'sentences.json').read_text(encoding='utf8')):
            v = J / 'sentences' / f"{s['id']}.mp4"
            text = next((t['title'] for t in s.get('translations', []) if t.get('languageCode') == 'az'), '')
            if v.exists():
                tasks.append((s['id'], str(v), {'text': text}))
        jobs.append(('jestdili_sentences', tasks))

    for name, tasks in jobs:
        tasks = tasks[:args.limit] if args.limit else tasks
        writer = store.PartWriter(str(args.out / name), items_per_part=200)
        pipeline.run_pool(tasks, _file_job, writer, (paths, True, False, None), workers=args.workers,
                          time_budget_h=48, log_every=100, label=name)


if __name__ == '__main__':
    main()
