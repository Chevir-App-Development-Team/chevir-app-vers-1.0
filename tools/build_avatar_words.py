"""
AzSLD-nin bütün işarələrindən avatar lüğəti — yerli datasetdən (data/azsld/AzSLD_Sentences.zip).

Hər glos üçün (ən azı --min-instances nümunə, tək hərflər barmaq əlifbasındadır) bir neçə
siqnalçının nümunəsi emal olunur və ən tipik təmiz nümunə seçilir:
  - MediaPipe Pose lite + Hands + Face, VIDEO rejimi (kadrdan kadra izləmə): IMAGE rejiminə görə
    titrəmə iki dəfə az. Əlin bədəndən nə qədər irəlidə olduğunu heavy görür, lite isə demək olar
    sabit təxmin edir (yan kamera ilə yoxlanılıb, EXPERIMENTS E11). Avatarın qol həlli hələ lite-yə
    uyğundur: heavy ilə əllər və qollar gövdəyə daha tez-tez girir, ona görə lüğət lite ilədir.
    Heavy variantı: --pose-model .../pose_landmarker_heavy.task --chest (siqnalçıların sinə xətti
    lüğətin özündən ölçülür, `chest` index.json-da, və avatar sinəyə toxunan əli öz sinəsinə köçürür; E12).
  - keyfiyyət: build_words.trim_and_score (poza, işarə məkanında əllərin tapılması, uzunluq);
  - tipiklik: namizədlər arasında DTW məsafəsi (əlin bədənə görə yeri + əlin forması). Keyfiyyəti
    ən yaxşınınkının 85 %-indən aşağı olmayanlardan bütün namizədlərə ən yaxını (medoid) seçilir —
    bir siqnalçının özünəməxsus icrası və ya solaxay icra lüğətə düşmür.
Kadr formatı build_words.py ilə eynidir (15 kadr/s, koordinatlar 1 mm / 0.001 dəqiqliklə), üstəlik üz: `f` = {s: 52 blendshape (0–100),
h: başın fırlanması [x, y, z] dərəcə ilə (kameraya görə)} — jest dilinin üz ifadələri (sual, inkar)
üçün; adlar index.json-da `face`. Mənbə hər sözün faylında qeyd olunur.

İstifadə:
  .venv/Scripts/python.exe -I tools/build_avatar_words.py                 # bütün lüğət
  .venv/Scripts/python.exe -I tools/build_avatar_words.py --only salam ev --out-dir data/azsld/lexicon_test
Ara nəticələr data/azsld/lexicon_cache/-də saxlanır: təkrar işlədəndə emal olunmuş nümunə yenidən
emal olunmur.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import itertools
import random
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (FaceLandmarker, FaceLandmarkerOptions, HandLandmarker,
                                           HandLandmarkerOptions, PoseLandmarker, PoseLandmarkerOptions,
                                           RunningMode)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
from azsld_zip import AzsldZip, cam2                                   # noqa: E402
from build_words import (ALIAS, ALT, BODY, L_HIP, L_SH, L_WR, R_HIP, R_SH, R_WR, SKIP,   # noqa: E402
                         _assign, az_lower, trim_and_score, word_id)

HAND_MODEL = ROOT / 'public' / 'tercume' / 'assets' / 'hand_landmarker.task'
FACE_MODEL = ROOT / 'public' / 'tercume' / 'assets' / 'face_landmarker.task'
SOURCE = 'AzSLD Sentences (Zenodo 14222948, CC BY 4.0)'


# ───────────────────────── glos → söz, namizədlər ─────────────────────────

def read_csv(path):
    with open(path, encoding='utf8') as f:
        return list(csv.DictReader(f))


# Annotasiyada Azərbaycan hərfləri tez-tez latın oxşarı ilə yazılıb (YARDİM, QİRMİZİ, OLACAGİQ, ŞEXS)
SWAPS = {'i': 'ı', 'ı': 'i', 'g': 'ğ', 'c': 'ç', 's': 'ş', 'u': 'ü', 'o': 'ö', 'e': 'ə', 'a': 'ə'}


def speller(sentences):
    """
    Glosun yazılışını cümlə mətnlərinin (düzgün yazılmış) sözlərinə görə düzəldir. Söz məlumdursa
    (mətndə var, mətndəki sözün başlanğıcıdır, ya da məsdərin gövdəsi mətndə işlənib) dəyişmir;
    yoxsa ən az hərfi dəyişən məlum variant götürülür, o da yoxdursa söz olduğu kimi qalır.
    """
    vocab = {t for r in sentences for t in re.findall(r"[\w'-]+", az_lower(r['text']))}
    stems = {v[:k] for v in vocab for k in range(3, len(v) + 1)}

    def known(w):
        stem = w[:-3] if w.endswith(('maq', 'mək')) else w
        return w in vocab or w in stems or (stem != w and len(stem) >= 2 and stem in stems)

    def fix(word):
        if known(word) or not word.isalpha():
            return word
        pos = [k for k, c in enumerate(word) if c in SWAPS][:10]
        for n in range(1, min(4, len(pos)) + 1):
            hits = []
            for sub in itertools.combinations(pos, n):
                w = list(word)
                for k in sub:
                    w[k] = SWAPS[w[k]]
                w = ''.join(w)
                if known(w):
                    hits.append(w)
            if hits:
                return min(hits, key=lambda w: (w not in vocab, w))
        return word

    return lambda text: ' '.join(fix(t) for t in text.split())


def entries(splits: Path, min_instances: int, words200=None):
    """
    Mətn → {labels, instances}. Eyni yazılışa gələn gloslar (GÜN / GUN, YARDİM / YARDIM) birləşir.
    Tək hərflər barmaq əlifbasıdır; "O" isə əvəzlikdir, əgər yanında başqa hərf yoxdursa.
    Cümlə annotasiyalarında heç olmayan Words_200 sözü (UĞUR) onun kliplərindən götürülür.
    """
    spelling = {r['variant']: r['gloss'] for r in read_csv(ROOT / 'ml' / 'gloss_spelling.csv')}
    sentences = read_csv(splits / 'sentences_split.csv')
    frontal = {r['rec_id']: float(r['frontal'] or 'nan') for r in sentences}
    fix = speller(sentences)

    def text_of(g):
        c = spelling.get(g, g)
        return ALIAS.get(c) or ALIAS.get(g) or fix(az_lower(c))

    rows = [r for r in read_csv(splits / 'glosses_split.csv') if r['dup'] != 'True' and r['start']]
    letter = lambda g: len(g) == 1 and not g.isdigit()
    letters_at = collections.defaultdict(list)
    for r in rows:
        if letter(r['gloss']) and r['gloss'] != 'O':
            letters_at[r['rec_id']].append((int(float(r['start'])), int(float(r['end']))))

    def spelled(r):
        a, b = int(float(r['start'])), int(float(r['end']))
        return any(s - 15 <= b and a <= e + 15 for s, e in letters_at[r['rec_id']])

    out = collections.defaultdict(lambda: {'labels': collections.Counter(), 'instances': []})
    for r in rows:
        g = r['gloss']
        if g in SKIP or (letter(g) and g != 'O') or (g == 'O' and spelled(r)):
            continue
        a, b = int(float(r['start'])), int(float(r['end']))
        e = out[text_of(g)]
        e['labels'][g] += 1
        e['instances'].append({'rec': r['rec_id'], 'a': a, 'b': b, 'signer': r['signer'],
                               'frontal': frontal.get(r['rec_id'], float('nan'))})
    if words200:
        z = AzsldZip(words200)
        for name in sorted(n for n in z.ents if n.endswith('.mp4')):
            label = name.split('/')[-2]
            text = text_of(label)
            if text in out and not out[text].get('clips'):
                continue
            e = out[text]
            e['clips'] = True
            e['labels'][label] += 1
            # Klip sözün özüdür (kənarsız): bütün kadrları, siqnalçısı məlum deyil
            e['instances'].append({'rec': None, 'zip': str(words200), 'member': name, 'a': 1, 'b': 10 ** 6,
                                   'signer': name, 'frontal': float('nan')})
    return {t: e for t, e in out.items() if len(e['instances']) >= min_instances}


def pick_candidates(text, e, k, available):
    """Fərqli siqnalçılardan k nümunə: kameraya düz baxan və uzunluğu tipik olanlar əvvəl."""
    if e.get('clips'):
        inst = list(e['instances'])
        random.Random(text).shuffle(inst)
        return inst[:k]
    inst = [i for i in e['instances'] if 6 <= i['b'] - i['a'] + 1 <= 60 and available(cam2(i['rec']))]
    if not inst:
        return []
    med = float(np.median([i['b'] - i['a'] + 1 for i in inst]))
    rng = random.Random(text)
    rng.shuffle(inst)
    fr = lambda i: i['frontal'] if i['frontal'] == i['frontal'] else 1.0
    inst.sort(key=lambda i: (round(fr(i) / 0.05), abs(i['b'] - i['a'] + 1 - med) / med))
    out, seen = [], set()
    for i in inst:
        if i['signer'] not in seen:
            out.append(i)
            seen.add(i['signer'])
        if len(out) == k:
            break
    return out


# ───────────────────────── bir nümunənin emalı ─────────────────────────

def process(job):
    """Videonun [a-1-warm, b-1+margin] kadrları, VIDEO rejimində; çıxış a-1-margin-dən, hər stride-dan bir."""
    zip_path, member, a, b, pose_model, margin, warm, stride, cache = job
    cache = Path(cache)
    if cache.exists():
        return json.loads(cache.read_text(encoding='utf8'))
    data = AzsldZip.__new__(AzsldZip)            # kataloq hər prosesdə yenidən oxunmasın
    data.path, data.pieces = Path(zip_path), Path(zip_path + '.pieces')
    data.ents = json.loads(Path(zip_path + '.ents.json').read_text())
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=cache.parent) as tmp:
        tmp.write(data.read(member))
    first, last, out0 = max(0, a - 1 - warm), b - 1 + margin, max(0, a - 1 - margin)
    cap = cv2.VideoCapture(tmp.name)
    W, H = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5) or 30
    imgs, i = [], 0
    while i <= last:
        ok, img = cap.read()
        if not ok:
            break
        if i >= first:
            imgs.append((i, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    os.unlink(tmp.name)

    hand = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)), running_mode=RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=0.4))
    pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model)), running_mode=RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=0.5))
    face = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL)), running_mode=RunningMode.VIDEO,
        num_faces=1, output_face_blendshapes=True, output_facial_transformation_matrixes=True))
    aspect = W / H
    frames, face_keys = [], None
    for i, rgb in imgs:
        im = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(i * 1000 / fps)
        pr, hr, fr = pose.detect_for_video(im, ts), hand.detect_for_video(im, ts), face.detect_for_video(im, ts)
        if i < out0 or (i - out0) % stride:
            continue
        f = None
        if fr.face_blendshapes:
            cats = [c for c in fr.face_blendshapes[0] if c.category_name != '_neutral']
            face_keys = face_keys or [c.category_name for c in cats]
            f = {'s': [int(round(c.score * 100)) for c in cats], 'h': head_angles(fr.facial_transformation_matrixes[0])}
        body = up = img_pts = None
        if pr.pose_world_landmarks:
            wl, il = pr.pose_world_landmarks[0], pr.pose_landmarks[0]
            body = [[round(wl[j].x, 3), round(wl[j].y, 3), round(wl[j].z, 3)] for j in BODY]
            img_pts = {j: (il[j].x, il[j].y) for j in (L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP)}
            sh_y = (il[L_SH].y + il[R_SH].y) / 2
            hip_y = (il[L_HIP].y + il[R_HIP].y) / 2
            limit = sh_y + 0.75 * (hip_y - sh_y)
            up = [il[L_WR].y < limit, il[R_WR].y < limit]
        hands = [{'n': [[round(p.x, 3), round(p.y, 3), round(p.z, 3)] for p in hl],
                  'w': [[round(p.x, 3), round(p.y, 3), round(p.z, 3)] for p in hr.hand_world_landmarks[k]]}
                 for k, hl in enumerate(hr.hand_landmarks or [])]
        sides = _assign(hands, img_pts, aspect)
        frames.append({'b': body, 'u': up, 'L': sides['L'], 'R': sides['R'], 'f': f})
    hand.close()
    pose.close()
    face.close()
    rec_out, score = trim_and_score({'frames': frames, 'size': [W, H], 'fps': round(fps / stride, 2)})
    res = {'rec': rec_out, 'score': score, 'face': face_keys}
    cache.write_text(json.dumps(res, separators=(',', ':')), encoding='utf8')
    return res


def head_angles(m):
    """Üzün 4×4 çevrilmə matrisindən başın fırlanması: [x (aşağı-yuxarı), y (sağa-sola), z (yana əyilmə)], dərəcə."""
    r = np.asarray(m, dtype=float)[:3, :3]
    x = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    y = np.degrees(np.arcsin(-np.clip(r[2, 0], -1, 1)))
    z = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    return [round(float(x), 1), round(float(y), 1), round(float(z), 1)]


# ───────────────────────── tipiklik ─────────────────────────

def features(rec):
    """Kadr başına: hər əlin biləyi çiyinlərin ortasına görə (çiyin eni ilə) + əlin forması."""
    out = []
    for f in rec['frames']:
        b = np.asarray(f['b'], dtype=float)
        mid, sw = (b[1] + b[2]) / 2, max(np.linalg.norm(b[1] - b[2]), 0.1)
        v = []
        for s, wi in (('L', 5), ('R', 6)):
            h = f[s]
            if h:
                w = np.asarray(h['w'], dtype=float)
                v += list((b[wi] - mid) / sw) + list(((w[1:] - w[0]) * 5).ravel())
            else:
                v += [0.0] * 63
        out.append(v)
    return np.asarray(out)


FINGER_SEGS = [(a, b) for f in ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
               for a, b in zip(f[:-1], f[1:])]
_T = np.linspace(0, 1, 5)[:, None]


def fingers_close(rec, limit=0.016):
    """Share of two-handed frames in which a finger of one hand is within `limit` (m, bone axes) of a
    finger of the other: the signer's metric hands (pose wrist + hand world landmarks)."""
    n = close = 0
    for fr in rec['frames']:
        if not (fr.get('L') and fr.get('R') and fr.get('b')):
            continue
        pts = []
        for s, wi in (('L', 5), ('R', 6)):
            W = np.asarray(fr[s]['w'], float)
            P = np.asarray(fr['b'][wi], float) + W - W[0]
            pts.append(np.concatenate([P[a] + _T * (P[b] - P[a]) for a, b in FINGER_SEGS]))
        d = np.linalg.norm(pts[0][:, None] - pts[1][None], axis=-1).min()
        n += 1
        close += d < limit
    return close / n if n else 0.0


def chest_line(words):
    """
    The signers' chest surface in the avatar's frame (vrm.js #signerToAvatar: shoulder axis, camera
    vertical, forward; shoulder widths): hands stop at the body, so the 3rd percentile of the palm's
    forward distance at each height (palm in front of the torso) traces it. Seated signers lean toward
    the camera, so the line slants: forward = a + b × height. → {a, b} or None.
    """
    pts = []
    for w in words:
        for fr in w['frames']:
            b = fr.get('b')
            if not b:
                continue
            P = np.asarray(b, float) * [1, -1, -1]
            mid, ax = (P[1] + P[2]) / 2, P[1] - P[2]
            sw = np.linalg.norm(ax)
            ax = ax / sw
            up = np.array([0, 1.0, 0]) - ax[1] * ax
            up /= np.linalg.norm(up)
            fwd = np.cross(ax, up)
            for s, wi in (('L', 5), ('R', 6)):
                h = fr.get(s)
                if h:
                    W = np.asarray(h['w'], float)
                    q = (np.asarray(b[wi], float) + W[[0, 5, 9, 17]].mean(0) - W[0]) * [1, -1, -1] - mid
                    pts.append((q @ ax / sw, q @ up / sw, q @ fwd / sw))
    pts = np.asarray(pts)
    if not len(pts):
        return None
    pts = pts[np.abs(pts[:, 0]) < 0.45]
    hs, fs = [], []
    for lo in np.arange(-1.3, -0.2, 0.1):
        m = (pts[:, 1] >= lo) & (pts[:, 1] < lo + 0.1)
        if m.sum() >= 150:
            hs.append(lo + 0.05)
            fs.append(np.percentile(pts[m, 2], 3))
    if len(hs) < 4:
        return None
    b, a = np.polyfit(hs, fs, 1)
    return {'a': round(float(a), 3), 'b': round(float(b), 3)}


def dtw(x, y):
    d = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)
    acc = np.full((len(x) + 1, len(y) + 1), np.inf)
    acc[0, 0] = 0
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            acc[i, j] = d[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return acc[-1, -1] / (len(x) + len(y))


def choose(results, min_score, keep=0.85):
    """Keyfiyyət süzgəci, sonra medoid. Qaytarır: (indeks, orta DTW) və ya None."""
    ok = [k for k, r in enumerate(results) if r['rec'] and r['score'] >= min_score]
    if not ok:
        return None
    best = max(results[k]['score'] for k in ok)
    good = [k for k in ok if results[k]['score'] >= keep * best]
    if len(ok) < 3:
        return max(ok, key=lambda k: results[k]['score']), None
    feats = {k: features(results[k]['rec']) for k in ok}
    mean = {k: np.mean([dtw(feats[k], feats[j]) for j in ok if j != k]) for k in good}
    k = min(mean, key=mean.get)
    return k, round(float(mean[k]), 3)


# ───────────────────────── əsas ─────────────────────────

def main():
    sys.stdout.reconfigure(encoding='utf-8')      # -I ilə PYTHONIOENCODING oxunmur
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default=str(ROOT / 'data' / 'azsld' / 'AzSLD_Sentences.zip'))
    ap.add_argument('--words200', default=str(ROOT / 'data' / 'azsld' / 'AzSLD_Words_200.zip'),
                    help='cümlələrdə olmayan sözlər üçün kliplər (boş — istifadə olunmur)')
    ap.add_argument('--splits', type=Path, default=ROOT / 'ml' / 'splits')
    ap.add_argument('--pose-model', type=Path, default=ROOT / 'public' / 'tercume' / 'assets' / 'pose_landmarker_lite.task',
                    help='heavy: data/models/pose_landmarker_heavy.task (float16/1 relizi)')
    ap.add_argument('--chest', action='store_true',
                    help='siqnalçıların sinə xəttini index.json-a yaz: avatar dərinliyi sinədən ölçür (heavy ilə; E12)')
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'public' / 'tercume' / 'words')
    ap.add_argument('--cache', type=Path, default=ROOT / 'data' / 'azsld' / 'lexicon_cache')
    ap.add_argument('--candidates', type=int, default=8)
    ap.add_argument('--min-instances', type=int, default=3)
    ap.add_argument('--min-score', type=float, default=0.45)
    ap.add_argument('--margin', type=int, default=3, help='annotasiyadan əvvəl/sonra saxlanan kadr')
    ap.add_argument('--warm', type=int, default=8, help='izləmənin oturması üçün əvvəldən emal olunan kadr')
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--workers', type=int, default=22)
    ap.add_argument('--only', nargs='*', help='yalnız bu sözlər (sınaq)')
    ap.add_argument('--only-index', type=Path, help='yalnız bu index.json-dakı sözlər (müqayisə üçün)')
    ap.add_argument('--partial', action='store_true', help='endirmə bitməyibsə, olan videolarla işlə')
    ap.add_argument('--prefetch', action='store_true',
                    help='endirmə gedərkən: son namizəd siyahısından diskdə olanları keşə emal et, lüğət yazma')
    args = ap.parse_args()

    z = AzsldZip(args.zip)
    if z.pieces.exists() and not (args.partial or args.prefetch):
        sys.exit('endirmə bitməyib (--partial ilə olan videolarla işləmək olar)')
    Path(args.zip + '.ents.json').write_text(json.dumps(z.ents))
    w200 = args.words200 if args.words200 and Path(args.words200).exists() else None
    if w200:
        Path(w200 + '.ents.json').write_text(json.dumps(AzsldZip(w200).ents))
    ents = entries(args.splits, args.min_instances, w200)
    if args.only_index:
        args.only = [w['text'] for w in json.loads(args.only_index.read_text(encoding='utf8'))['words']]
    if args.only:
        ents = {t: e for t, e in ents.items() if t in set(args.only)}
    args.cache.mkdir(parents=True, exist_ok=True)
    tag = f'{args.pose_model.stem}-face-r3-m{args.margin}-w{args.warm}-s{args.stride}'
    plan, jobs = {}, []
    for text, e in sorted(ents.items()):
        cands = pick_candidates(text, e, args.candidates, z.__contains__ if args.prefetch else z.available)
        plan[text] = cands
        for c in cands:
            if args.prefetch and c['rec'] and not z.available(cam2(c['rec'])):
                continue
            src = c['rec'] or c['member']
            h = hashlib.md5(f"{src}|{c['a']}|{c['b']}|{tag}".encode()).hexdigest()[:20]
            jobs.append((c.get('zip', args.zip), c['member'] if c['rec'] is None else cam2(c['rec']), c['a'], c['b'],
                         str(args.pose_model), args.margin, args.warm, args.stride, str(args.cache / f'{h}.json')))
    print(f'{len(ents)} söz, {len(jobs)} nümunə emal olunacaq ({tag})', flush=True)
    results = []
    with ProcessPoolExecutor(args.workers) as ex:
        for n, r in enumerate(ex.map(process, jobs, chunksize=4), 1):
            results.append(r)
            if n % 200 == 0:
                print(f'  {n}/{len(jobs)}', flush=True)
    if args.prefetch:
        print(f'{len(results)} nümunə keşdədir')
        return

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob('*.json'):
        old.unlink()
    index, dropped, used, k = [], [], set(), 0
    face_keys = next((r['face'] for r in results if r.get('face')), None)
    for text, cands in plan.items():
        res = results[k:k + len(cands)]
        k += len(cands)
        pick = choose(res, args.min_score)
        if not pick:
            dropped.append(text)
            continue
        j, typ = pick
        c, rec = cands[j], res[j]['rec']
        wid = word_id(az_lower(text), used)        # Bakı → baki (böyük hərf id-dən düşməsin)
        used.add(wid)
        label = ents[text]['labels'].most_common(1)[0][0]
        where = ({'video': c['rec'], 'source': SOURCE, 'sentence': cam2(c['rec']), 'sentenceFrames': [c['a'], c['b']]}
                 if c['rec'] else {'video': c['member'], 'source': 'AzSLD Words 200 (Zenodo 14222948, CC BY 4.0)'})
        data = {'id': wid, 'text': text, 'label': label, **rec, **where,
                'quality': round(res[j]['score'], 3), 'candidates': len(cands), 'typicality': typ}
        # Əllərin bir-birinə toxunduğu işarə (BİRLİKDƏ, TƏZYİQ, KİTAB): sözün siqnalçılarının çoxunda
        # barmaqlar kadrların dörddə birindən çoxunda 1.6 sm-dən yaxındır. Avatar belə sözdə
        # siqnalçının öz əllərinin üst-üstə düşməsinə icazə verir (barmaqlar bir-birinin arasından
        # keçir, ovuc ovucun üstündədir); təsadüfi bir siqnalçının dərinlik xətası səs vermir.
        two = [r for r in res if r['rec'] and r['rec']['hands'] == 'LR']
        if len(two) >= 3 and rec['hands'] == 'LR':
            data['contact'] = bool(sum(fingers_close(r['rec']) > 0.25 for r in two) / len(two) >= 0.5)
        entry = {'id': wid, 'text': text, 'hands': rec['hands'], 'frames': len(rec['frames'])}
        if text in ALT:
            data['alt'] = entry['alt'] = ALT[text]
        (out_dir / f'{wid}.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')),
                                             encoding='utf8')
        index.append(entry)
    index.sort(key=lambda w: w['text'])
    words = [json.loads(p.read_text(encoding='utf8')) for p in out_dir.glob('*.json') if p.name != 'index.json']
    chest = chest_line(words) if args.chest else None
    # Hər blendshape-in diapazonu lüğətin bütün kadrları üzrə: median — siqnalçıların sakit üzü,
    # 99-cu faiz — güclü ifadə (avatar bu aralığı öz ifadəsinin 0…tam aralığına çevirir)
    faces = np.asarray([fr['f']['s'] for w in words for fr in w['frames'] if fr.get('f')])
    face_neutral = [int(v) for v in np.median(faces, axis=0)] if len(faces) else None
    face_full = [int(v) for v in np.percentile(faces, 99, axis=0)] if len(faces) else None
    (out_dir / 'index.json').write_text(json.dumps({
        'source': f'{SOURCE} — hər işarə üçün bir neçə siqnalçının ən tipik təmiz nümunəsi '
                  f'(MediaPipe Pose {args.pose_model.stem.split("_")[-1]} + Hands + Face, VIDEO rejimi; '
                  f'tools/build_avatar_words.py)',
        'count': len(index), 'face': face_keys, 'faceNeutral': face_neutral, 'faceFull': face_full,
        'chest': chest, 'words': index,
    }, ensure_ascii=False, separators=(',', ':')), encoding='utf8')
    total = sum(p.stat().st_size for p in out_dir.glob('*.json'))
    print(f'{len(index)} söz yazıldı ({total / 1e6:.1f} MB), {len(dropped)} söz atıldı: {dropped[:20]}')


if __name__ == '__main__':
    main()
