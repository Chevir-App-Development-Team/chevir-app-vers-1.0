"""Run the ml/notebooks on Kaggle from this machine with the Kaggle CLI.

Auth: the token lives in ~/.kaggle/access_token (put there by you, never passed through here).
The CLI is installed isolated (`uv tool install --python 3.13 kaggle`) because the system
Python's protobuf 4.25 breaks it.

    python ml/kaggle_run.py whoami
    python ml/kaggle_run.py push 01 [--smoke]      # 01 | 02-0 … 02-3 | 02 (all shards) | 03
                                                   # 02c1-0 … 02c1-3 | 02c1: side camera (Cam1) shards
    python ml/kaggle_run.py status                 # every notebook
    python ml/kaggle_run.py log 02-1               # last lines of a run's log
    python ml/kaggle_run.py splits                 # 03's splits/ → ml/splits/ (a few MB)
    python ml/kaggle_run.py results --to run4      # 04's summary/history/metrics/predictions → ml/results/run4/
                                                   # (--models adds best.pt + model.onnx,
                                                   #  --run no_face fetches only that run; only the
                                                   #  latest version, and nothing while one is queued)
    python ml/kaggle_run.py eval --to wordmode1    # 05's eval/ (summary, val grid, test detections;
                                                   #  --models adds the chosen model's files)
    python ml/kaggle_run.py cont --to cont1 [--models]   # 06's cont/ (continuous recognition runs)
    python ml/kaggle_run.py model-dataset ml/results/run11_model/final_side_noface
                                                   # a run's best.pt + manifest as a private dataset, for a
                                                   # run whose notebook version ended in an error (Kaggle
                                                   # attaches only complete versions as inputs)

Each push creates a new version of the notebook on Kaggle and starts running it. All notebooks
are private, CPU-only; 01/02 have internet on (they read Zenodo), 03 reads 01+02 outputs.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.join(HERE, '.kaggle_stage')
SHARDS = 4

# slugs match the notebooks already imported on Kaggle by hand (title = file name, Kaggle turns _ into -)
NOTEBOOKS = {
    '01': ('01_index_words200_fingerspelling.ipynb', '01-index-words200-fingerspelling'),
    **{f'02-{k}': (f'02_sentences_shard{k}.ipynb', f'02-sentences-shard{k}') for k in range(SHARDS)},
    **{f'02c1-{k}': (f'02_sentences_cam1_shard{k}.ipynb', f'02-sentences-cam1-shard{k}') for k in range(SHARDS)},
    '03': ('03_split_benchmark.ipynb', '03-split-benchmark'),
    '04': ('04_train_words.ipynb', '04-train-words'),
    '05': ('05_eval_word_mode.ipynb', '05-eval-word-mode'),
    '06': ('06_train_continuous.ipynb', '06-train-continuous'),
    '07': ('07_train_letters.ipynb', '07-train-letters'),
}
EXTRACTION = ['01'] + [f'02-{k}' for k in range(SHARDS)]
CAM1 = [f'02c1-{k}' for k in range(SHARDS)]
# per notebook: GPU, internet, which earlier notebooks' outputs it reads (01/02 default: CPU + internet)
SETTINGS = {
    # Cam2: for the alignment; the shard's own previous output: a rerun resumes from it
    **{k: dict(gpu=False, internet=True, sources=[f'02-{j}' for j in range(SHARDS)] + [k]) for k in CAM1},
    '03': dict(gpu=False, internet=False, sources=EXTRACTION),
    '04': dict(gpu=True, internet=True, sources=EXTRACTION + CAM1 + ['03']),   # internet: pip onnx if missing
    # CPU: ~2 ms/window; its own previous output keeps models that 04 has since replaced
    '05': dict(gpu=False, internet=False, sources=[f'02-{k}' for k in range(SHARDS)] + ['03', '04', '05'],
               datasets=['azsl-model-final-side-noface']),
    # continuous recognition: whole recordings of both cameras; 04 for run 10a (the starting point)
    # datasets: models kept for evaluation-only runs (kaggle_run.py model-dataset)
    '06': dict(gpu=True, internet=True, sources=[f'02-{k}' for k in range(SHARDS)] + CAM1 + ['03', '04'],
               datasets=['azsl-model-seq-splice', 'azsl-model-final-splice']),
    # letters: fingerspelling landmarks (01), letter signs of both cameras, the app's old letter model to compare
    '07': dict(gpu=True, internet=True, sources=['01'] + [f'02-{k}' for k in range(SHARDS)] + CAM1 + ['03'],
               datasets=['chevir-letter-model-old']),
}


def kaggle_exe():
    exe = shutil.which('kaggle') or os.path.join(os.path.expanduser('~'), '.local', 'bin', 'kaggle.exe')
    if not os.path.exists(exe):
        sys.exit('Kaggle CLI not found — install it with: uv tool install --python 3.13 kaggle')
    return exe


def kg(*args, check=True, capture=True):
    env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}   # the CLI prints → and crashes on cp1252
    r = subprocess.run([kaggle_exe(), *args], capture_output=capture, text=True, encoding='utf-8', errors='replace', env=env)
    if check and r.returncode:
        sys.exit(f'kaggle {" ".join(args)} failed:\n{(r.stdout or "") + (r.stderr or "")}')
    return (r.stdout or '') + (r.stderr or '')


def username():
    if os.environ.get('KAGGLE_USERNAME'):
        return os.environ['KAGGLE_USERNAME']
    out = kg('config', 'view', check=False)
    m = re.search(r'username:\s*(\S+)', out)
    if m and m.group(1).lower() != 'none':
        return m.group(1)
    sys.exit('Could not tell the Kaggle username from the token — run with KAGGLE_USERNAME=<name> set.\n' + out)


def stage(key, user, smoke, cpu=False):
    file, slug = NOTEBOOKS[key]
    d = os.path.join(STAGE, slug)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    with open(os.path.join(HERE, 'notebooks', file), encoding='utf-8') as f:
        nb = json.load(f)
    if smoke:
        for c in nb['cells']:
            if c['cell_type'] == 'code':
                c['source'] = [s.replace('SMOKE = False', 'SMOKE = True') for s in c['source']] \
                    if isinstance(c['source'], list) else c['source'].replace('SMOKE = False', 'SMOKE = True')
    with open(os.path.join(d, file), 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    st = SETTINGS.get(key, dict(gpu=False, internet=True, sources=[]))
    if cpu:                                     # e.g. 06 evaluating saved models: no GPU quota
        st = dict(st, gpu=False)
    meta = {
        'id': f'{user}/{slug}', 'title': os.path.splitext(file)[0], 'code_file': file, 'language': 'python', 'kernel_type': 'notebook',
        'is_private': 'true', 'enable_gpu': str(st['gpu']).lower(), 'enable_internet': str(st['internet']).lower(),
        'machine_shape': 'NvidiaTeslaT4' if st['gpu'] else '',
        'dataset_sources': [f'{user}/{d}' for d in st.get('datasets', [])], 'competition_sources': [],
        'model_sources': [], 'kernel_sources': [f'{user}/{NOTEBOOKS[k][1]}' for k in st['sources']],
    }
    with open(os.path.join(d, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    return d, f'{user}/{slug}'


def keys(arg):
    if arg == '02':
        return [f'02-{k}' for k in range(SHARDS)]
    if arg == '02c1':
        return CAM1
    if arg == 'all':
        return list(NOTEBOOKS)
    if arg not in NOTEBOOKS:
        sys.exit(f'unknown notebook {arg!r}; use one of {", ".join(NOTEBOOKS)}, 02 or all')
    return [arg]


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # Windows console is cp1252; glosses aren't
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['whoami', 'push', 'stage', 'status', 'log', 'splits', 'results', 'eval',
                                    'model-dataset', 'cont', 'letters', 'files-dataset'])
    ap.add_argument('which', nargs='?', default='all')
    ap.add_argument('--smoke', action='store_true', help='push with SMOKE = True (a few files, ~3 min)')
    ap.add_argument('--cpu', action='store_true', help='push without a GPU (evaluation only)')
    ap.add_argument('--user', help='Kaggle username (else read from the CLI config)')
    ap.add_argument('--to', help='results: folder name under ml/results/ (default: latest)')
    ap.add_argument('--models', action='store_true', help='results: also best.pt + model.onnx (bigger downloads)')
    ap.add_argument('--run', help='results: only this run of the batch (e.g. no_face), to save bandwidth')
    ap.add_argument('files', nargs='*', help='files-dataset: the files to upload (which = the dataset slug)')
    a = ap.parse_args()
    user = a.user or (None if a.cmd == 'stage' else username())

    if a.cmd == 'model-dataset':                       # which = a run folder with best.pt + manifest.json
        with open(os.path.join(a.which, 'manifest.json'), encoding='utf-8') as f:
            run = json.load(f)['run']
        slug = 'azsl-model-' + run.replace('_', '-')
        d = os.path.join(STAGE, slug)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        for name in ('best.pt', 'manifest.json', 'vocab.json', 'metrics.json'):
            shutil.copy2(os.path.join(a.which, name), d)
        with open(os.path.join(d, 'dataset-metadata.json'), 'w', encoding='utf-8') as f:
            json.dump({'title': slug, 'id': f'{user}/{slug}', 'licenses': [{'name': 'other'}]}, f)
        exists = 'ref' in kg('datasets', 'list', '--mine', '-s', slug, check=False).lower() and slug in             kg('datasets', 'list', '--mine', '-s', slug, check=False)
        out = kg('datasets', 'version', '-p', d, '-m', 'update', check=False) if exists else             kg('datasets', 'create', '-p', d, check=False)            # private unless --public
        print(f'{user}/{slug}:', out.strip()[-300:])
    elif a.cmd == 'files-dataset':                    # which = slug; files → a private dataset of them
        d = os.path.join(STAGE, a.which)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        for name in a.files:
            shutil.copy2(name, d)
        with open(os.path.join(d, 'dataset-metadata.json'), 'w', encoding='utf-8') as f:
            json.dump({'title': a.which, 'id': f'{user}/{a.which}', 'licenses': [{'name': 'other'}]}, f)
        exists = a.which in kg('datasets', 'list', '--mine', '-s', a.which, check=False)
        out = kg('datasets', 'version', '-p', d, '-m', 'update', check=False) if exists else \
            kg('datasets', 'create', '-p', d, check=False)
        print(f'{user}/{a.which}:', out.strip()[-300:])
    elif a.cmd == 'whoami':
        print('kaggle user:', user)
        print(kg('kernels', 'list', '--mine', '--page-size', '5', check=False))
    elif a.cmd in ('stage', 'push'):
        for k in keys(a.which):
            d, ref = stage(k, user or 'USERNAME', a.smoke, a.cpu)
            if a.cmd == 'push':
                print(f'push {ref}{" (SMOKE)" if a.smoke else ""}:', kg('kernels', 'push', '-p', d).strip())
            else:
                print('staged', d)
    elif a.cmd == 'status':
        for k in keys(a.which):
            ref = f'{user}/{NOTEBOOKS[k][1]}'
            print(f'{k:5s} {kg("kernels", "status", ref, check=False).strip()}')
    elif a.cmd == 'log':
        for k in keys(a.which):
            ref = f'{user}/{NOTEBOOKS[k][1]}'
            tmp = tempfile.mkdtemp(prefix='kaggle_log_')
            kg('kernels', 'output', ref, '-p', tmp, '--file-pattern', r'\.log$', '-q', check=False)
            for name in os.listdir(tmp):
                with open(os.path.join(tmp, name), encoding='utf-8', errors='replace') as f:
                    text = f.read()
                try:   # Kaggle logs are a JSON list of {stream_name, time, data}
                    text = ''.join(e.get('data', '') for e in json.loads(text))
                except ValueError:
                    pass
                lines = [ln for ln in text.splitlines() if not ln.startswith(('W0000', 'I0000'))]
                print(f'===== {k} ({name})\n' + '\n'.join(lines[-40:]))
            shutil.rmtree(tmp, ignore_errors=True)
    elif a.cmd == 'splits':
        ref = f'{user}/{NOTEBOOKS["03"][1]}'
        tmp = tempfile.mkdtemp(prefix='kaggle_splits_')
        print(kg('kernels', 'output', ref, '-p', tmp, '--file-pattern', r'^splits/', check=False))
        src = os.path.join(tmp, 'splits')
        if not os.path.isdir(src):
            sys.exit(f'no splits/ in the output of {ref} yet')
        dst = os.path.join(HERE, 'splits')
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        shutil.rmtree(tmp, ignore_errors=True)
        print('wrote', dst, os.listdir(dst))
    elif a.cmd == 'results':
        ref = f'{user}/{NOTEBOOKS["04"][1]}'   # always the latest version: the CLI ignores a /<version> suffix
        tmp = tempfile.mkdtemp(prefix='kaggle_results_')
        small = r'history\.csv|metrics\.json|manifest\.json|vocab\.json|per_class_test_\w+\.csv|predictions\.npz'
        files = small + (r'|model\.onnx|best\.pt' if a.models else '')
        run = re.escape(a.run) if a.run else '[^/]+'
        kg('kernels', 'output', ref, '-p', tmp, '--file-pattern', rf'^train/(summary\.csv|({run}/)?({files}))$',
           '-q', check=False)
        src = os.path.join(tmp, 'train')
        if not os.path.isdir(src):
            sys.exit(f'no train/ in the output of {ref} yet')
        dst = os.path.join(HERE, 'results', a.to or 'latest')   # other runs' folders are left alone
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        shutil.rmtree(tmp, ignore_errors=True)
        print('wrote', dst, os.listdir(dst))

    elif a.cmd == 'cont':                             # 06's cont/ (summary, per run: metrics, history, grid)
        ref = f'{user}/{NOTEBOOKS["06"][1]}'
        tmp = tempfile.mkdtemp(prefix='kaggle_cont_')
        files = r'history\.csv|metrics\.json|manifest\.json|vocab\.json|grid_val\.csv|detections_test\.csv' + (
            r'|model\.onnx|best\.pt' if a.models else '')
        kg('kernels', 'output', ref, '-p', tmp, '--file-pattern', rf'^cont/(summary\.csv|[^/]+/({files}))$', '-q', check=False)
        src = os.path.join(tmp, 'cont')
        if not os.path.isdir(src):
            sys.exit(f'no cont/ in the output of {ref} yet')
        dst = os.path.join(HERE, 'results', a.to or 'cont_latest')
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        shutil.rmtree(tmp, ignore_errors=True)
        print('wrote', dst, os.listdir(dst))
    elif a.cmd == 'letters':                          # 07's letters/ (summary, per run: metrics, history, predictions)
        ref = f'{user}/{NOTEBOOKS["07"][1]}'
        tmp = tempfile.mkdtemp(prefix='kaggle_letters_')
        files = r'history\.csv|metrics\.json|manifest\.json|predictions_[a-z]+\.npz' + (
            r'|model\.onnx|best\.pt' if a.models else '')
        kg('kernels', 'output', ref, '-p', tmp, '--file-pattern',
           rf'^letters/(summary\.csv|spelled_[a-z]+\.(npz|json)|[^/]+/({files}))$',
           '-q',
           check=False)
        src = os.path.join(tmp, 'letters')
        if not os.path.isdir(src):
            sys.exit(f'no letters/ in the output of {ref} yet')
        dst = os.path.join(HERE, 'results', a.to or 'letters_latest')
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        shutil.rmtree(tmp, ignore_errors=True)
        print('wrote', dst, os.listdir(dst))
    elif a.cmd == 'eval':
        ref = f'{user}/{NOTEBOOKS["05"][1]}'
        tmp = tempfile.mkdtemp(prefix='kaggle_eval_')
        pattern = r'^eval/([^/]+|model/[^/]+/[^/]+)$' if a.models else r'^eval/[^/]+$'   # --models: the chosen model too
        kg('kernels', 'output', ref, '-p', tmp, '--file-pattern', pattern, '-q', check=False)
        src = os.path.join(tmp, 'eval')
        if not os.path.isdir(src):
            sys.exit(f'no eval/ in the output of {ref} yet')
        dst = os.path.join(HERE, 'results', a.to or 'eval_latest')
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        shutil.rmtree(tmp, ignore_errors=True)
        print('wrote', dst, os.listdir(dst))


if __name__ == '__main__':
    main()
