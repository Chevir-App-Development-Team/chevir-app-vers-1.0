"""Run the ml/notebooks on Kaggle from this machine with the Kaggle CLI.

Auth: the token lives in ~/.kaggle/access_token (put there by you, never passed through here).
The CLI is installed isolated (`uv tool install --python 3.13 kaggle`) because the system
Python's protobuf 4.25 breaks it.

    python ml/kaggle_run.py whoami
    python ml/kaggle_run.py push 01 [--smoke]      # 01 | 02-0 … 02-3 | 02 (all shards) | 03
    python ml/kaggle_run.py status                 # every notebook
    python ml/kaggle_run.py log 02-1               # last lines of a run's log
    python ml/kaggle_run.py splits                 # 03's splits/ → ml/splits/ (a few MB)
    python ml/kaggle_run.py results                # 04's history/metrics/ONNX/best.pt → ml/results/ (~15 MB)

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
    '03': ('03_split_benchmark.ipynb', '03-split-benchmark'),
    '04': ('04_train_words.ipynb', '04-train-words'),
}
EXTRACTION = ['01'] + [f'02-{k}' for k in range(SHARDS)]
# per notebook: GPU, internet, which earlier notebooks' outputs it reads (01/02 default: CPU + internet)
SETTINGS = {
    '03': dict(gpu=False, internet=False, sources=EXTRACTION),
    '04': dict(gpu=True, internet=True, sources=EXTRACTION + ['03']),   # internet: pip onnx if missing
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


def stage(key, user, smoke):
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
    meta = {
        'id': f'{user}/{slug}', 'title': os.path.splitext(file)[0], 'code_file': file, 'language': 'python', 'kernel_type': 'notebook',
        'is_private': 'true', 'enable_gpu': str(st['gpu']).lower(), 'enable_internet': str(st['internet']).lower(),
        'machine_shape': 'NvidiaTeslaT4' if st['gpu'] else '', 'dataset_sources': [], 'competition_sources': [],
        'model_sources': [], 'kernel_sources': [f'{user}/{NOTEBOOKS[k][1]}' for k in st['sources']],
    }
    with open(os.path.join(d, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    return d, f'{user}/{slug}'


def keys(arg):
    if arg == '02':
        return [f'02-{k}' for k in range(SHARDS)]
    if arg == 'all':
        return list(NOTEBOOKS)
    if arg not in NOTEBOOKS:
        sys.exit(f'unknown notebook {arg!r}; use one of {", ".join(NOTEBOOKS)}, 02 or all')
    return [arg]


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # Windows console is cp1252; glosses aren't
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['whoami', 'push', 'stage', 'status', 'log', 'splits', 'results'])
    ap.add_argument('which', nargs='?', default='all')
    ap.add_argument('--smoke', action='store_true', help='push with SMOKE = True (a few files, ~3 min)')
    ap.add_argument('--user', help='Kaggle username (else read from the CLI config)')
    a = ap.parse_args()
    user = a.user or (None if a.cmd == 'stage' else username())

    if a.cmd == 'whoami':
        print('kaggle user:', user)
        print(kg('kernels', 'list', '--mine', '--page-size', '5', check=False))
    elif a.cmd in ('stage', 'push'):
        for k in keys(a.which):
            d, ref = stage(k, user or 'USERNAME', a.smoke)
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
        ref = f'{user}/{NOTEBOOKS["04"][1]}'
        tmp = tempfile.mkdtemp(prefix='kaggle_results_')
        kg('kernels', 'output', ref, '-p', tmp, '--file-pattern',
           r'^train/(history\.csv|metrics\.json|vocab\.json|per_class_test_words\.csv|model\.onnx|best\.pt)$', '-q', check=False)
        src = os.path.join(tmp, 'train')
        if not os.path.isdir(src):
            sys.exit(f'no train/ in the output of {ref} yet')
        dst = os.path.join(HERE, 'results')
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        shutil.rmtree(tmp, ignore_errors=True)
        print('wrote', dst, os.listdir(dst))


if __name__ == '__main__':
    main()
