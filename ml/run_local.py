"""
Kaggle notebook-larını yerli GPU-da işlədir (04 söz modeli, 06 cümlə modeli, 07 hərflər): notebook-un kod xanaları
eynən icra olunur, yalnız giriş (Kaggle-ın /kaggle/input-u əvəzinə endirilmiş çıxışlar) və çıxış qovluğu dəyişir.
Konfiqurasiya xanasından sonra --set ilə dəyişikliklər (Python ifadəsi) əlavə olunur: RUNS, TRAIN_ON_VAL və s.

  D:/chevir/venv/Scripts/python ml/run_local.py 06 --input D:/chevir/kaggle --out D:/chevir/runs/cont_all \
      --set "TRAIN_ON_ALL = True" --set "RUNS = {...}"

Notebook-lar ml/build_notebooks.py-dən qurulur (python ml/build_notebooks.py); paket xanaları ml/azsl-in
mənbəyini yazır, ona görə yerli kod Kaggle-dakı ilə eynidir. Giriş qovluğunda Kaggle çıxışları öz adları ilə:
01-index-words200-fingerspelling/, 02-sentences-shard0/, …, 02-sentences-cam1-shard3/, 03-split-benchmark/ və
lazım olan model qovluqları (best.pt + manifest.json).
"""
import argparse
import json
import os
import sys
from pathlib import Path

NB = {'04': '04_train_words.ipynb', '05': '05_eval_word_mode.ipynb', '06': '06_train_continuous.ipynb',
      '07': '07_train_letters.ipynb'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('nb', choices=sorted(NB))
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--set', action='append', default=[], help='konfiqurasiya xanasından sonra icra olunan Python')
    ap.add_argument('--workers', type=int, default=0, help='DataLoader işçiləri (Windows: 0 ən etibarlısıdır)')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    os.environ.setdefault('MPLBACKEND', 'Agg')

    path = Path(__file__).parent / 'notebooks' / NB[args.nb]
    cells = [''.join(c['source']) for c in json.loads(path.read_text(encoding='utf8'))['cells'] if c['cell_type'] == 'code']
    os.makedirs(args.out, exist_ok=True)
    os.chdir(args.out)
    ns = {'__name__': '__nb__', 'display': lambda x: print(x.to_string() if hasattr(x, 'to_string') else x)}
    sys.path.insert(0, str(Path(__file__).parent))      # ml/azsl birbaşa (Kaggle-da %%writefile ilə yazılır)
    for k, src in enumerate(cells):
        if src.lstrip().startswith('%%writefile'):
            continue
        src = src.replace("INPUT = '/kaggle/input' if ON_KAGGLE else OUT", f'INPUT = {args.input!r}')
        src = src.replace("OUT = '/kaggle/working' if ON_KAGGLE else os.path.abspath('azsl_out')", f'OUT = {args.out!r}')
        exec(compile(src, f'{path.name}[{k}]', 'exec'), ns)
        if k == 0:                     # konfiqurasiya xanası
            for s in args.set:
                exec(s, ns)
            for r in ns.get('RUNS', {}).values():
                r['workers'] = args.workers
            if 'CFG' in ns:
                ns['CFG']['workers'] = args.workers
            print('RUNS:', json.dumps(ns.get('RUNS'), ensure_ascii=False, default=str)[:600])


if __name__ == '__main__':
    main()
