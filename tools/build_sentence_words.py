"""
AzSLD Sentences datasetindən avatar üçün yeni söz işarələri çıxarır.

Words 200 dəstində cəmi 200 söz var. Cümlə datasetində isə hər sözün kadr aralığı
etiketlənib (10 104 cümlə, ~49 000 söz nümunəsi, 841 fərqli söz), yəni istənilən sözü
təkbaşına klip kimi kəsmək mümkündür. 43.8 GB-lıq arxiv endirilmir: zip-in mərkəzi
kataloqu oxunur və yalnız lazım olan fayllar HTTP range sorğusu ilə götürülür.

Boru xətti: annotasiya indeksi → hədəf sözlərin seçimi → yalnız lazımi videolar →
ffmpeg ilə kəsim → build_words.py ilə poza/əl çıxarışı → mövcud lüğətə birləşdirmə.

İstifadə:
  .venv/bin/python tools/build_sentence_words.py --count 100
  .venv/bin/python tools/build_sentence_words.py --count 100 --dry-run   # yalnız siyahı

Qeydlər:
  - Zenodo 4 paraleldən çoxunu boğur, WORKERS artırmayın.
  - Endirilənlər `data/azsld_sentences/` altında keşlənir (git-ə düşmür), təkrar
    işlətmədə yenidən endirilmir.
  - Cam2 götürülür: 1280×960-dır və annotasiyadakı kadr nömrələri ona aiddir
    (Cam1 640×480-dir və kadr sayı 1-2 kadr fərqlənir).
"""
import argparse
import collections
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
from build_words import ALIAS, SKIP, az_lower, word_id   # noqa: E402  (eyni qaydalar)

URL = 'https://zenodo.org/records/14222948/files/AzSLD_Sentences.zip'
WORDS_DIR = ROOT / 'public' / 'tercume' / 'words'
WORKERS = 4
RETRIES = 5
FPS = 30
MARGIN = 3          # kəsimdə sözün əvvəlinə/sonuna əlavə olunan kadr


# ───────────────────────── zip-i endirmədən oxumaq ─────────────────────────

def _range(start, end):
    req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def zip_index(cache: Path):
    """Arxivin mərkəzi kataloqu: fayl adı → [metod, sıxılmış ölçü, həqiqi ölçü, offset]."""
    if cache.exists():
        return json.loads(cache.read_text())
    size = int(urllib.request.urlopen(urllib.request.Request(URL, method='HEAD'), timeout=60)
               .headers['Content-Length'])
    tail = _range(size - 400000, size - 1)
    j = tail.rfind(b'PK\x06\x06')                      # zip64 (arxiv 4 GB-dan böyükdür)
    _, cd_size, cd_off = struct.unpack('<QQQ', tail[j + 32:j + 56])
    cd = b''
    while len(cd) < cd_size:
        cd += _range(cd_off + len(cd), min(cd_off + cd_size, cd_off + len(cd) + (8 << 20)) - 1)
    ents, p = {}, 0
    while p < len(cd) and cd[p:p + 4] == b'PK\x01\x02':
        meth, = struct.unpack('<H', cd[p + 10:p + 12])
        csz, usz = struct.unpack('<II', cd[p + 20:p + 28])
        nl, el, cl = struct.unpack('<HHH', cd[p + 28:p + 34])
        off, = struct.unpack('<I', cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nl].decode('utf8', 'replace')
        extra = cd[p + 46 + nl:p + 46 + nl + el]
        if 0xFFFFFFFF in (csz, usz, off):              # böyük fayllar zip64 sahəsindədir
            q = 0
            while q < len(extra):
                hid, hsz = struct.unpack('<HH', extra[q:q + 4])
                blk, k = extra[q + 4:q + 4 + hsz], 0
                if hid == 1:
                    if usz == 0xFFFFFFFF: usz, = struct.unpack('<Q', blk[k:k + 8]); k += 8
                    if csz == 0xFFFFFFFF: csz, = struct.unpack('<Q', blk[k:k + 8]); k += 8
                    if off == 0xFFFFFFFF: off, = struct.unpack('<Q', blk[k:k + 8]); k += 8
                q += 4 + hsz
        if not name.endswith('/'):
            ents[name] = [meth, csz, usz, off]
        p += 46 + nl + el + cl
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(ents))
    return ents


def extract_file(name, ents):
    """Arxivdən bir fayl — bir sorğu ilə (lokal başlıq + data birlikdə)."""
    meth, csz, _, off = ents[name]
    blob = _range(off, off + 30 + 1024 + csz - 1)
    nl, el = struct.unpack('<HH', blob[26:30])
    data = blob[30 + nl + el: 30 + nl + el + csz]
    return zlib.decompressobj(-15).decompress(data) if meth == 8 else data


def _parallel(names, fn, label):
    """Zenodo-nu boğmadan paralel işləmə; uğursuzlar geri qaytarılır."""
    lock, state = threading.Lock(), {'n': 0, 'fail': [], 't0': time.time()}

    def run(name):
        for attempt in range(RETRIES):
            try:
                fn(name)
                with lock:
                    state['n'] += 1
                    if state['n'] % 200 == 0:
                        rate = state['n'] / (time.time() - state['t0'])
                        print(f'  {label}: {state["n"]}/{len(names)} · {rate:.1f}/s · '
                              f'qalan ≈{(len(names) - state["n"]) / rate / 60:.0f} dəq', flush=True)
                return
            except Exception:
                time.sleep(0.5 * (attempt + 1))
        with lock:
            state['fail'].append(name)

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(run, names))
    return state['fail']


# ───────────────────────── annotasiya indeksi ─────────────────────────

def annotations(cache_dir: Path, ents):
    """Hər cümlənin sözləri və kadr aralıqları (bir dəfə endirilir, sonra keşdən)."""
    out = cache_dir / 'annotations.jsonl'
    done = set()
    if out.exists():
        with out.open(encoding='utf8') as f:
            for line in f:
                try:
                    done.add(json.loads(line)['ann'])
                except Exception:
                    pass
    todo = [n for n in ents if n.endswith('.json') and n not in done]
    if todo:
        print(f'{len(todo)} annotasiya endirilir (cəmi ~20 MB)…', flush=True)
        fh = out.open('a', encoding='utf8')
        lock = threading.Lock()

        def grab(name):
            d = json.loads(extract_file(name, ents))
            session = name.split('/')[1]
            base = name.split('/')[-1][:-5]
            cams = {c: f'AzSLD_Sentences/{session}/{c}/{base}'
                    for c in ('Cam1', 'Cam2') if f'AzSLD_Sentences/{session}/{c}/{base}' in ents}
            row = {'ann': name, 'session': session, 'cams': cams,
                   'tags': [{'name': t['name'], 'frames': t.get('frameRange')} for t in d.get('tags', [])]}
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + '\n')

        fail = _parallel(todo, grab, 'annotasiya')
        fh.close()
        if fail:
            print(f'  {len(fail)} annotasiya alınmadı, yenidən işlədəndə tamamlanacaq', flush=True)
    rows = []
    with out.open(encoding='utf8') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def word_instances(rows):
    """söz → [{video, frames, ann}] (Cam2 üstünlüklə)."""
    words = collections.defaultdict(list)
    for r in rows:
        for t in r['tags']:
            fr = t.get('frames')
            cam = r['cams'].get('Cam2') or r['cams'].get('Cam1')
            if fr and cam:
                words[t['name']].append({'video': cam, 'frames': fr, 'ann': r['ann']})
    return words


def pick_targets(words, existing_texts, count, min_samples, per_word):
    """Lüğətdə olmayan, kifayət qədər nümunəsi olan sözlər (ən çox rast gələndən başlayaraq)."""
    cands = []
    for name, inst in words.items():
        if name in SKIP or len(name) == 1 or name.isdigit():
            continue                                    # tək hərflər barmaq əlifbasındadır
        text = ALIAS.get(name, az_lower(name))
        if text in existing_texts:
            continue
        good = [i for i in inst if 12 <= i['frames'][1] - i['frames'][0] + 1 <= 45]
        if len(good) < min_samples:
            continue
        good.sort(key=lambda i: abs((i['frames'][1] - i['frames'][0] + 1) - 22))
        cands.append((name, text, good[:per_word], len(inst)))
    cands.sort(key=lambda c: -c[3])
    return cands[:count]


# ───────────────────────── video → klip ─────────────────────────

def fetch_videos(targets, cache_dir: Path, ents):
    raw = cache_dir / 'videos'
    raw.mkdir(parents=True, exist_ok=True)
    need = sorted({i['video'] for _, _, inst, _ in targets for i in inst})
    todo = [v for v in need if not (raw / v.replace('/', '__')).exists()]
    print(f'{len(need)} video lazımdır, {len(need) - len(todo)} keşdə var, {len(todo)} endirilir', flush=True)
    if not todo:
        return raw

    def grab(name):
        dst = raw / name.replace('/', '__')
        tmp = dst.with_suffix('.part')
        tmp.write_bytes(extract_file(name, ents))
        tmp.replace(dst)

    fail = _parallel(todo, grab, 'video')
    if fail:
        print(f'  {len(fail)} video alınmadı: {fail[:3]}', flush=True)
    return raw


def cut_clips(targets, raw: Path, clips: Path):
    """Kadr aralığına görə sözü kəsir; hər klipin yanında mənbə qeydi saxlanılır."""
    jobs = []
    for label, _, inst, _ in targets:
        for i in inst:
            jobs.append((label, i))
    made = {}

    def cut(job):
        label, i = job
        src = raw / i['video'].replace('/', '__')
        if not src.exists():
            return None
        a, b = i['frames']
        out_dir = clips / label
        out_dir.mkdir(parents=True, exist_ok=True)
        cid = hashlib.md5(f"{i['ann']}|{a}|{b}".encode()).hexdigest()[:24]
        out = out_dir / f'{cid}.mp4'
        if not out.exists():
            r = subprocess.run(
                ['ffmpeg', '-v', 'error', '-y', '-ss', f'{max(0, a - MARGIN) / FPS:.3f}', '-i', str(src),
                 '-t', f'{(b - a + 1 + 2 * MARGIN) / FPS:.3f}', '-c:v', 'libx264', '-preset', 'veryfast',
                 '-crf', '20', '-an', '-r', str(FPS), str(out)],
                capture_output=True, text=True)
            if r.returncode or not out.exists() or out.stat().st_size < 2000:
                return None
        return (f'{cid}.mp4', {'sentence': i['video'], 'frames': i['frames']})

    with ThreadPoolExecutor(6) as ex:
        for res in ex.map(cut, jobs):
            if res:
                made[res[0]] = res[1]
    (clips / 'provenance.json').write_text(json.dumps(made, ensure_ascii=False), encoding='utf8')
    print(f'{len(made)} klip kəsildi ({len(targets)} söz)', flush=True)
    return made


# ───────────────────────── çıxarış və birləşdirmə ─────────────────────────

def run_extraction(clips: Path, staging: Path, pose_model: Path, workers: int):
    cmd = [sys.executable, str(ROOT / 'tools' / 'build_words.py'),
           '--words-dir', str(clips), '--pose-model', str(pose_model),
           '--out-dir', str(staging), '--workers', str(workers),
           '--source', 'AzSLD Sentences (Zenodo 14222948, CC BY 4.0)']
    print('çıxarış başlayır:', ' '.join(cmd[-8:]), flush=True)
    subprocess.run(cmd, check=True)


def merge(staging: Path, provenance: dict):
    """Yeni sözləri mövcud lüğətə əlavə edir: id-lər unikal, mənbə hər sözdə qeyd olunur."""
    index = json.loads((WORDS_DIR / 'index.json').read_text(encoding='utf8'))
    used = {w['id'] for w in index['words']}
    texts = {w['text'] for w in index['words']}
    new_index = json.loads((staging / 'index.json').read_text(encoding='utf8'))
    added, skipped = [], []
    for w in new_index['words']:
        if w['text'] in texts:
            skipped.append(w['text'])
            continue
        data = json.loads((staging / f"{w['id']}.json").read_text(encoding='utf8'))
        wid = word_id(w['text'], used)
        used.add(wid)
        prov = provenance.get(data.get('video'), {})
        data.update({'id': wid, 'source': 'AzSLD Sentences (Zenodo 14222948, CC BY 4.0)',
                     'sentence': prov.get('sentence'), 'sentenceFrames': prov.get('frames')})
        (WORDS_DIR / f'{wid}.json').write_text(
            json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf8')
        index['words'].append({'id': wid, 'text': w['text'], 'hands': w['hands'], 'frames': w['frames']})
        added.append(w['text'])
    index['words'].sort(key=lambda w: w['text'])
    index['count'] = len(index['words'])
    index['source'] = ('AzSLD Words 200 + AzSLD Sentences (Zenodo 14222948, CC BY 4.0) — '
                       'cümlə datasetindəki sözlər kadr aralığına görə kəsilib')
    (WORDS_DIR / 'index.json').write_text(
        json.dumps(index, ensure_ascii=False, separators=(',', ':')), encoding='utf8')
    print(f'\nlüğət: {index["count"]} söz ({len(added)} yeni)')
    if skipped:
        print(f'lüğətdə onsuz da olan {len(skipped)} söz atıldı: {skipped[:8]}')
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', type=int, default=100, help='neçə yeni söz əlavə olunsun')
    ap.add_argument('--min-samples', type=int, default=6)
    ap.add_argument('--per-word', type=int, default=6, help='hər söz üçün neçə namizəd klip')
    ap.add_argument('--cache', type=Path, default=ROOT / 'data' / 'azsld_sentences')
    ap.add_argument('--pose-model', type=Path, default=ROOT / 'data' / 'models' / 'pose_landmarker_lite.task')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--dry-run', action='store_true', help='yalnız seçilən sözləri göstər')
    args = ap.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    ents = zip_index(args.cache / 'zip_index.json')
    rows = annotations(args.cache, ents)
    words = word_instances(rows)
    print(f'{len(rows)} cümlə · {sum(len(v) for v in words.values())} söz nümunəsi · {len(words)} fərqli söz')

    existing = {w['text'] for w in json.loads((WORDS_DIR / 'index.json').read_text(encoding='utf8'))['words']}
    targets = pick_targets(words, existing, args.count, args.min_samples, args.per_word)
    print(f'{len(targets)} yeni söz seçildi (lüğətdə {len(existing)} söz var)')
    print('   ' + ', '.join(f'{t[1]}({t[3]})' for t in targets[:20]) + (' …' if len(targets) > 20 else ''))
    if args.dry_run:
        return

    raw = fetch_videos(targets, args.cache, ents)
    clips = args.cache / 'clips'
    prov = cut_clips(targets, raw, clips)
    staging = args.cache / 'staging'
    run_extraction(clips, staging, args.pose_model, args.workers)
    merge(staging, prov)


if __name__ == '__main__':
    main()
