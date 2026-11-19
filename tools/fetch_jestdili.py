"""
JestDili.az lüğətinin sözləri və cümlələri (videolar + mətn), komandanın icazəsi ilə (docs/DATA.md, roadmap D2).
Siyahılar API-dən (data/jestdili/words.json, sentences.json, categories.json, alphabets.json), videolar
data/jestdili/{words,sentences}/<id>.mp4. Yarımçıq qalan fayl yenidən yüklənir; tam olanlar keçilir.

  JESTDILI_API=<ünvan> python tools/fetch_jestdili.py [--lists-only] [--workers 4]
  python tools/fetch_jestdili.py --alphabet D:/chevir/jestdili_alphabet     # əlifba şəkilləri (API lazım deyil)

API-nin ünvanı repoda saxlanmır (repo açıqdır): JestDili.az komandasından alınır, JESTDILI_API və ya --api ilə verilir.
--alphabet: data/jestdili/alphabets.json-dakı şəkil keçidləri (hər hərfə coverImageUrl, image1Url) ayrıca qovluğa,
<hərf id>_<sahə>.jpg adı ilə; yanında index.csv (fayl, hərf, sahə).
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = os.environ.get('JESTDILI_API', '').rstrip('/')
OUT = Path('data/jestdili')


def get(path):
    with urllib.request.urlopen(API + path, timeout=60) as r:
        return json.load(r)


def lists():
    out = {}
    for ep in ['words', 'sentences', 'categories', 'alphabets']:
        items, page = [], 1
        while True:
            d = get(f'/api/{ep}?page={page}&size=500')
            data = d.get('data') if isinstance(d, dict) else d
            items += data or []
            if not data or len(data) < 500:
                break
            page += 1
            time.sleep(0.5)
        (OUT / f'{ep}.json').write_text(json.dumps(items, ensure_ascii=False), encoding='utf8')
        out[ep] = items
        print(f'{ep}: {len(items)}', flush=True)
    return out


def fetch(job):
    url, dst = job
    req = urllib.request.Request(url, method='HEAD')
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                size = int(r.headers.get('Content-Length', 0))
            if dst.exists() and dst.stat().st_size == size:
                return 'skip', size
            tmp = dst.with_suffix('.part')
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, 'wb') as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if tmp.stat().st_size != size:
                raise IOError(f'short read {tmp.stat().st_size} / {size}')
            tmp.replace(dst)
            return 'ok', size
        except Exception as e:                      # noqa: BLE001 (network: retry, then report)
            err = e
            time.sleep(2 * (attempt + 1))
    return f'fail {err}', 0


def alphabet(out, workers):
    """Əlifba şəkilləri (alphabets.json-dakı keçidlər) → out/<id>_<sahə>.jpg + out/index.csv."""
    items = json.loads((OUT / 'alphabets.json').read_text(encoding='utf8'))
    out.mkdir(parents=True, exist_ok=True)
    rows, jobs = [], []
    for a in items:
        for field in ('coverImageUrl', 'image1Url'):
            url = a.get(field)
            if url:
                dst = out / f"{a['id']}_{field}{Path(url).suffix or '.jpg'}"
                jobs.append((url, dst))
                rows.append(f"{dst.name},{a['character']},{field}")
    (out / 'index.csv').write_text('file,character,field\n' + '\n'.join(rows) + '\n', encoding='utf8')
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(fetch, jobs))
    fails = [(j[0], s) for j, (s, _) in zip(jobs, res) if s.startswith('fail')]
    mb = sum(n for _, n in res) / 2**20
    print(f'{len(jobs)} şəkil, {mb:.1f} MB, {len(fails)} alınmadı')
    for url, s in fails[:5]:
        print(' ', s)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--lists-only', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--api', help='JestDili.az API ünvanı (və ya JESTDILI_API)')
    ap.add_argument('--alphabet', type=Path, metavar='DIR', help='yalnız əlifba şəkilləri, alphabets.json-dan DIR-ə')
    args = ap.parse_args()
    if args.alphabet:
        return alphabet(args.alphabet, args.workers)
    global API
    API = (args.api or API).rstrip('/')
    if not API:
        sys.exit('API ünvanı lazımdır: JESTDILI_API=... və ya --api ...')
    OUT.mkdir(parents=True, exist_ok=True)
    d = lists()
    if args.lists_only:
        return
    jobs = [(w['videoUrl'], OUT / 'words' / f"{w['id']}.mp4") for w in d['words'] if w.get('videoUrl')]
    jobs += [(s['video']['url'], OUT / 'sentences' / f"{s['id']}.mp4") for s in d['sentences'] if s.get('video', {}).get('url')]
    for _, dst in jobs:
        dst.parent.mkdir(parents=True, exist_ok=True)
    done = total = fails = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        for k, (status, size) in enumerate(ex.map(fetch, jobs), 1):
            total += size
            fails += status.startswith('fail')
            if k % 100 == 0 or k == len(jobs):
                mb = total / 2**20
                print(f'{k}/{len(jobs)} files, {mb:.0f} MB, {mb / max(1, time.time() - t0):.1f} MB/s, {fails} failed', flush=True)
    print('done' if not fails else f'{fails} failed: run again to retry')


if __name__ == '__main__':
    main()
