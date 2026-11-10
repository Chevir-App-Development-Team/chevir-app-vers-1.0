"""
Slovo (rus işarə dili, 1,000 işarə, 194 adam, CC BY-SA 4.0; roadmap D3, R2) videolarının landmark-ları: AzSLD və
JestDili.az ilə eyni boru (ml/azsl/landmarks.Extractor, md5 ilə eyni .task modelləri) və eyni anbar formatı
(store.PartWriter), ona görə LandmarkStore və təlim kodu onları AzSLD kimi oxuyur. Ön öyrətmə (R2) üçündür;
videolar və landmark-lar yerli qalır (git-də yox).

  D:/chevir/venv/Scripts/python -I tools/extract_slovo.py --src D:/chevir/slovo/raw --out D:/chevir/landmarks [--workers 16]
  → <out>/slovo/ (hər video bir element: name = işarənin rusca adı, user, split, begin, end)

--src: Kaggle kapitanov/slovo arxivinin açıldığı qovluq (annotations.csv və train/, test/ videoları içində harada olsa).
Fasilə verilə bilər: yenidən işə salınanda hazır olanlar keçilir.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'ml'))
from azsl import landmarks, pipeline, store  # noqa: E402


def _file_job(task):
    item_id, path, extra = task
    t0 = time.time()
    arrays, meta, face = pipeline._EX.video(path)
    meta.update(extra)
    meta['sec'] = round(time.time() - t0, 2)
    return item_id, arrays, meta, face


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--models', type=Path, default=None, help='.task modellərinin qovluğu (yoxdursa endirilir, md5 yoxlanır)')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--skip-no-event', action='store_true', help='"no_event" klipləri (işarəsiz) çıxarılmır')
    args = ap.parse_args()
    paths = landmarks.ensure_models(str(args.models or args.out / 'models'))

    ann = sorted(args.src.rglob('annotations.csv'), key=lambda p: len(p.parts))
    if not ann:
        sys.exit(f'{args.src}: annotations.csv yoxdur')
    videos = {p.stem: p for p in args.src.rglob('*.mp4')}
    tasks = []
    with open(ann[-1], encoding='utf-8') as f:
        sep = '\t' if '\t' in f.readline() else ','
        f.seek(0)
        for r in csv.DictReader(f, delimiter=sep):
            vid = videos.get(r['attachment_id'])
            if vid is None or (args.skip_no_event and r['text'] == 'no_event'):
                continue
            split = 'train' if str(r.get('train', '')).strip().lower() in ('true', '1') else 'test'
            tasks.append((r['attachment_id'], str(vid), {'name': r['text'], 'user': r['user_id'], 'split': split,
                                                         'begin': int(float(r['begin'])), 'end': int(float(r['end']))}))
    print(f'{ann[-1]}: {len(tasks)} video ({len(videos)} mp4 tapıldı)')
    tasks = tasks[:args.limit] if args.limit else tasks
    writer = store.PartWriter(str(args.out / 'slovo'), items_per_part=500)
    pipeline.run_pool(tasks, _file_job, writer, (paths, True, False, None), workers=args.workers,
                      time_budget_h=48, log_every=500, label='slovo')


if __name__ == '__main__':
    main()
