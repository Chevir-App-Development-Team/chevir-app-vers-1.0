"""Resumable segmented download of the AzSLD Zenodo record (14222948), md5-checked.

Pieces of every file go through one queue of WORKERS threads (Zenodo throttles more than 4).
Each piece is written at its offset in a preallocated file; finished pieces are recorded in
<file>.pieces so a rerun continues where it stopped.

    python tools/download_azsld.py data/azsld

tools/azsld_zip.py reads single videos out of a zip while this is still running.
"""
import hashlib
import json
import os
import queue
import sys
import threading
import time
import urllib.request

RECORD = 'https://zenodo.org/api/records/14222948'
WORKERS = 4
PIECE = 32 << 20

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
files = json.load(urllib.request.urlopen(RECORD, timeout=60))['files']
files.sort(key=lambda f: -f['size'])

lock = threading.Lock()
state = {}
q = queue.Queue()
for f in files:
    path = os.path.join(out, f['key'])
    done_path = path + '.pieces'
    if os.path.exists(path) and not os.path.exists(done_path):
        continue                                   # finished and verified earlier
    n = (f['size'] + PIECE - 1) // PIECE
    done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
    if not os.path.exists(path):
        with open(path, 'wb') as fh:
            fh.truncate(f['size'])
        json.dump([], open(done_path, 'w'))            # a rerun must not take it for a finished file
    state[f['key']] = {'f': f, 'path': path, 'done_path': done_path, 'done': done, 'n': n}
    for i in range(n):
        if i not in done:
            q.put((f['key'], i))

total = q.qsize()
got = [0]
t0 = time.time()
print(f'{total} pieces of {PIECE >> 20} MB to fetch', flush=True)


def fetch(key, i):
    s = state[key]
    a = i * PIECE
    b = min(a + PIECE, s['f']['size']) - 1
    url = s['f']['links']['self']
    for attempt in range(12):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={a}-{b}'})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise IOError(f'status {r.status}')
                data = r.read()
            if len(data) != b - a + 1:
                raise IOError(f'short read {len(data)}')
            with open(s['path'], 'r+b') as fh:
                fh.seek(a)
                fh.write(data)
            return
        except Exception as e:
            wait = min(60, 2 ** attempt)
            print(f'  {key} piece {i}: {e}; retry in {wait}s', flush=True)
            time.sleep(wait)
    raise RuntimeError(f'{key} piece {i} failed')


def worker():
    while True:
        try:
            key, i = q.get_nowait()
        except queue.Empty:
            return
        fetch(key, i)
        with lock:
            s = state[key]
            s['done'].add(i)
            got[0] += 1
            if got[0] % 10 == 0 or len(s['done']) == s['n']:
                json.dump(sorted(s['done']), open(s['done_path'], 'w'))
            if got[0] % 20 == 0:
                rate = got[0] * PIECE / (time.time() - t0) / 1e6
                left = (total - got[0]) * PIECE / 1e6 / rate / 60
                print(f'{got[0]}/{total} pieces, {rate:.1f} MB/s, ~{left:.0f} min left', flush=True)


threads = [threading.Thread(target=worker) for _ in range(WORKERS)]
for t in threads:
    t.start()
for t in threads:
    t.join()

for key, s in state.items():
    assert len(s['done']) == s['n'], key
    h = hashlib.md5()
    with open(s['path'], 'rb') as fh:
        for chunk in iter(lambda: fh.read(64 << 20), b''):
            h.update(chunk)
    want = s['f']['checksum'].split(':')[1]
    ok = h.hexdigest() == want
    print(f'{key}: md5 {"ok" if ok else "MISMATCH " + h.hexdigest()}', flush=True)
    if ok:
        os.remove(s['done_path'])
print(f'done in {(time.time() - t0) / 60:.0f} min', flush=True)
