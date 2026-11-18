"""
İşarə dili daşıyıcısının yerli rəy paketi (E36): doğma istifadəçini gözləyən hər şey bir qovluqda, bir neçə dəqiqəlik
suallar, internetsiz. index.html iki kliklə açılır (Chrome, Edge, Firefox; server və quraşdırma lazım deyil), videolar
yanındakı clips/ qovluğundadır, cavablar brauzerdə saxlanır və bir CSV kimi endirilir. Komanda o CSV-ni
tools/apply_native_review.py ilə tətbiq edir.

Bölmələr (hamısı repodakı fayllardan, Slovo-nun 55 uyğunluğu D:/chevir/slovo/r4-dən):
  slovo_new_installed  ml/slovo_new.csv, rank ≤ 10: tanıyıcıya artıq əlavə olunan 101 sinif (RSL, təsdiqlənməyib)
  slovo_map            ml/slovo_azsld.csv (xəritə v2): Slovo klipləri AzSLD etiketi ilə öyrədilir
  slovo_azsld55        Slovo-nun AzSLD etiketlərinə uyğun gələn, xəritəyə hələ girməyən 55 adı
  slovo_new_unchecked  ml/slovo_new.csv, rank > 10: yoxlanmamış yeni siniflər
  d5                   ml/review/word_forms_review_w500.csv: söz formaları eyni işarədir?
  literal              tanıyıcının çoxsözlü AzSLD etiketləri: doğma işarə, yoxsa sözbəsöz tərcümə (ANADAN OLMAQ)?
  n8                   ml/text_alt.csv (sinonim və idiomlar), maşın → AVTOMOBİL təklifi, çoxmənalı işarələr
  n9                   açıq doğma qaydalar (fel + YOX + zaman, BİRLİKDƏ, üçüncü şəxs, ...): avatarın videosu ilə
  v2                   public/rey/items.json-un 50 bəndi (/rey/ səhifəsi kimi; tools/review_summary.mjs oxuyur)

Videolar: Slovo (zip-dən, işarəyə kəsilmiş), AzSLD (Words 500 zip-i, hər etiketin iki iştirakçısı), JestDili.az
(data/jestdili/words), avatar (tools/render_avatar_clips.mjs, başsız Chrome). Hamısı yerli qalır: paket repoya
girmir, JestDili.az API ünvanı paketə də yazılmır.

  D:/chevir/venv/Scripts/python -I tools/build_native_review.py [--out D:/chevir/native_review] [--zip]
      [--no-avatar] [--slovo-clips 2] [--azsld-clips 2] [--workers 6]
Yenidən işə salınanda hazır videolar keçilir.
"""
import argparse
import csv
import datetime
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / 'tools' / 'native_review'


def az_lower(s):
    return s.replace('I', 'ı').replace('İ', 'i').lower()


def az_upper(s):
    return s.replace('i', 'İ').replace('ı', 'I').upper()


def read_csv(path, delimiter=','):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def rank_of(basis):
    last = basis.split()[-1] if basis.split() else ''
    return int(last) if last.isdigit() else None


# ── videolar ────────────────────────────────────────────────────────────────────────────────────────────────

class Clips:
    """Videoları paketin clips/ qovluğuna kiçik H.264 mp4 kimi yazır (hazırdırsa keçir)."""

    def __init__(self, out, workers):
        self.out = out
        self.jobs = {}            # nisbi yol → (mənbə, hündürlük)
        self.workers = workers

    def add(self, rel, source, height):
        self.jobs.setdefault(rel, (source, height))
        return rel

    def run(self):
        todo = [(rel, src, h) for rel, (src, h) in self.jobs.items() if not (self.out / rel).exists()]
        print(f'videolar: {len(self.jobs)}, yazılacaq {len(todo)}')
        zips, lock = {}, threading.Lock()

        def open_zip(path):                     # bir ZipFile bütün axınlara (zipfile oxumanı kilidlə bölüşdürür)
            with lock:
                if path not in zips:
                    zips[path] = zipfile.ZipFile(path)
                return zips[path]

        def one(job):
            rel, src, h = job
            dst = self.out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = None
            if isinstance(src, tuple):           # (zip, üzv): müvəqqəti fayla çıxarılır (mp4-ün moov-u sonda ola bilər)
                zpath, member = src
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                    f.write(open_zip(zpath).read(member))
                    tmp = f.name
                src = tmp
            part = dst.with_suffix('.part.mp4')
            r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(src), '-an', '-vf', f'scale=-2:{h}',
                                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28', '-pix_fmt', 'yuv420p',
                                '-movflags', '+faststart', str(part)], capture_output=True, text=True)
            if tmp:
                Path(tmp).unlink(missing_ok=True)
            if r.returncode:
                print(f'  ffmpeg xətası {rel}: {r.stderr.strip()[:200]}')
                part.unlink(missing_ok=True)
                return
            part.replace(dst)

        with ThreadPoolExecutor(self.workers) as ex:
            for k, _ in enumerate(ex.map(one, todo), 1):
                if k % 100 == 0:
                    print(f'  {k}/{len(todo)}')


class Slovo:
    def __init__(self, zpath):
        self.zpath = zpath
        with zipfile.ZipFile(zpath) as z:
            names = set(z.namelist())
            rows = list(csv.DictReader(z.read('slovo/annotations.csv').decode('utf-8').splitlines(), delimiter='\t'))
        self.by = defaultdict(list)
        for r in rows:
            member = f"slovo/{'train' if r['train'] == 'True' else 'test'}/{r['attachment_id']}.mp4"
            if member in names:
                self.by[r['text']].append((r['attachment_id'], r['user_id'], float(r['length']), member))

    def clips(self, gloss, n, add):
        """Bir adın n klipi, fərqli adamlardan, ən azı 12 kadr (bərabər seçim: id sırası ilə)."""
        out, users = [], set()
        for aid, user, length, member in sorted(self.by.get(gloss, [])):
            if user in users or length < 12:
                continue
            users.add(user)
            out.append(add(f'clips/slovo/{aid}.mp4', (self.zpath, member), 480))
            if len(out) == n:
                break
        return out


class Azsld:
    """AzSLD etiketinin klipləri: Words 500 zip-i cümlə yazılarını annotasiya açarları ilə kəsilmiş saxlayır (E20)."""

    def __init__(self, zpath):
        self.zpath = zpath
        with zipfile.ZipFile(zpath) as z:
            self.member = {Path(n).name.rsplit('_', 2)[1]: n for n in z.namelist() if n.endswith('.mp4')}
        spelling = {r['variant']: r['gloss'] for r in read_csv(ROOT / 'ml/gloss_spelling.csv')}
        self.by = defaultdict(list)
        for r in read_csv(ROOT / 'ml/splits/glosses_split.csv'):
            if r['dup'] == 'True' or r['key'] not in self.member:
                continue
            g = spelling.get(r['gloss'], r['gloss'])
            self.by[g].append((r['signer'], r['key'], float(r['end']) - float(r['start'])))

    def clips(self, label, n, add):
        out, signers = [], set()
        for signer, key, length in sorted(self.by.get(label, [])):
            if signer in signers or length < 8:
                continue
            signers.add(signer)
            out.append(add(f'clips/azsld/{key}.mp4', (self.zpath, self.member[key]), 480))
            if len(out) == n:
                break
        return out


class JestDili:
    """JestDili.az sözlərinin klipləri (data/jestdili/words/<id>.mp4). words.json-un video ünvanları oxunmur."""

    def __init__(self, root):
        self.dir = root / 'words'
        self.by = defaultdict(list)
        for w in json.loads((root / 'words.json').read_text(encoding='utf-8')):
            if not (self.dir / f"{w['id']}.mp4").exists():
                continue
            name = az_lower(w['name'].strip())
            keys = {name, re.split(r'\s*[(,/]', name)[0].strip()}
            keys |= {p.strip() for p in re.split(r'[,/]', re.sub(r'\(.*?\)?$', '', name)) if p.strip()}
            for k in keys:
                self.by[k].append(w['id'])

    def clip_id(self, cid, add):
        return add(f'clips/jd/{cid}.mp4', self.dir / f'{cid}.mp4', 360)

    def clips(self, word, n, add):
        return [self.clip_id(cid, add) for cid in sorted(set(self.by.get(az_lower(word), [])))[:n]]


# ── bəndlər ─────────────────────────────────────────────────────────────────────────────────────────────────

N9 = [
    # (id, mətn, sual, izah)
    ('neg-past', 'Mən gəlmədim.', 'Keçmiş zamanda inkar: fel + YOX + OLMAQ. Sıra düzdür?',
     'Avatar inkarı fellə YOX-la, keçmiş zamanı OLMAQ ilə göstərir. Bəlkə OLMAQ YOX-dan əvvəl gəlir, ya da heç lazım deyil?'),
    ('neg-future', 'Mən sabah gəlməyəcəyəm.', 'Gələcək zamanda inkar: fel + YOX + OLACAQ. Sıra düzdür?', ''),
    ('with-meet', 'Mən dostumla görüşdüm.', '«görüşmək» kimi qarşılıqlı feldə BİRLİKDƏ lazımdır?',
     '«-la / ilə» insanla olanda avatar BİRLİKDƏ göstərir. Görüşmək onsuz da iki adamın işidir.'),
    ('with-agree', 'Mən səninlə razıyam.', '«razı olmaq» ilə BİRLİKDƏ lazımdır?', ''),
    ('third-person', 'Gəldi.', 'Üçüncü şəxs: mətndə «o» yoxdur, avatar da O göstərmir. Belə düzdür, yoxsa O göstərilməlidir?', ''),
    ('subject', 'Gəldim.', 'Mətndə «mən» yoxdur, avatar MƏN göstərir. Düzdür?', ''),
    ('future', 'Sən sabah işləyəcəksən.', 'Gələcək zaman: fel + OLACAQ. SABAH olanda da OLACAQ lazımdır?', ''),
    ('with-person', 'Anamla bazara getdim.', '«ilə» insanla: BİRLİKDƏ. Düzdür?', ''),
    ('must', 'Mən getməliyəm.', '«-malı»: fel + LAZIM. Düzdür, yoxsa MÜTLƏQ işarəsi olmalıdır?',
     'MÜTLƏQ işarəsi indi JestDili.az-dan var; avatar hələ LAZIM göstərir.'),
    ('possessive', 'Mənim anam həkimdir.', '«mənim» MƏN işarəsi ilə göstərilir. Düzdür, yoxsa MƏNİM işarəsi lazımdır?', ''),
    ('question-yn', 'Sən həkimsən?', 'Hə / yox sualı: qaşlar qalxır, baş bir az önə. Sual olduğu üzdən başa düşülür?', ''),
    ('question-wh', 'Sən harada işləyirsən?', 'Sual sözü olan sual: qaşlar enir, baş bir az əyilir. Üz düzdür?', ''),
    ('neg-shake', 'Mən işləmirəm.', 'İnkarda baş yellənir. Düzdür?', ''),
]

N8_PROPOSALS = [('avtomobil', 'maşın', 'təklif: tək sözlü sinonim (maşın həm də «mexanizm» deməkdir)')]
N8_POLYSEMY = [
    # (etiket, söz, sual)
    ('ALMAQ', 'almaq', '«almaq» bir işarədir, yoxsa mənaya görə (almaq, təhsil almaq, satın almaq) bir neçə işarə var?'),
    ('BİLMƏK', 'bilmək', '«bilmək» (bilmək / bacarmaq) bir işarədir, yoxsa iki?'),
    (None, 'eşitmək', '«eşitmək» (eşitmək, qulaq asmaq, təsadüfən eşitmək) neçə işarədir? Hansı mənada hansı?'),
]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('D:/chevir/native_review'))
    ap.add_argument('--r4', type=Path, default=Path('D:/chevir/slovo/r4'))
    ap.add_argument('--slovo-zip', type=Path, default=Path('D:/chevir/slovo/download/slovo.zip'))
    ap.add_argument('--azsld-zip', type=Path, default=ROOT / 'data/azsld/AzSLD_Words_500.zip')
    ap.add_argument('--jestdili', type=Path, default=ROOT / 'data/jestdili')
    ap.add_argument('--slovo-clips', type=int, default=2)
    ap.add_argument('--azsld-clips', type=int, default=2)
    ap.add_argument('--v2', default='13,12,10,15', help='V2: müsahibə, AzSLD, set C (mətn → işarə), işarə → mətn sayı')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--no-avatar', action='store_true', help='avatar videolarını yenidən çəkmə')
    ap.add_argument('--zip', action='store_true', help='qovluğu <out>.zip kimi də yaz')
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    clips = Clips(out, args.workers)
    add = clips.add

    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    today = datetime.date.today().isoformat()
    release = f'{today} {commit}'
    slovo, azsld, jd = Slovo(args.slovo_zip), Azsld(args.azsld_zip), JestDili(args.jestdili)
    label_text = {r['label']: r['text'] for r in read_csv(ROOT / 'ml/label_text.csv')}
    word_of = lambda label: label_text.get(label, az_lower(label))      # noqa: E731
    manifest = json.loads((ROOT / 'public/tercume/assets/word_model/manifest.json').read_text(encoding='utf-8'))
    labels = set(manifest['labels'])
    verify = defaultdict(list)                         # rus adı → JestDili.az klipləri (E28, r4/verify.json)
    for v in json.loads((args.r4 / 'verify.json').read_text(encoding='utf-8')):
        verify[v['ru']].append(v['clip'])
    items = []

    def ref(label, word, jd_ids=()):
        """Azərbaycan tərəfi: AzSLD iştirakçıları və JestDili.az (müqayisə üçün)."""
        srcs = azsld.clips(label, args.azsld_clips, add) if label else []
        jds = [jd.clip_id(c, add) for c in sorted(set(jd_ids))[:2]] or jd.clips(word, 1, add)
        return [{'label': 'AzSLD iştirakçıları', 'srcs': srcs}] * bool(srcs) + [{'label': 'JestDili.az', 'srcs': jds}] * bool(jds)

    def slovo_item(section, ru, label, word, basis, jd_ids=()):
        items.append({
            'id': f'{section}/{ru}', 'section': section, 'kind': 'slovo', 'a': ru, 'b': label, 'word': word,
            'text': f'{ru} → {word}', 'basis': basis,
            'groups': [{'label': 'Rus işarə dili (Slovo)', 'srcs': slovo.clips(ru, args.slovo_clips, add), 'main': True},
                       *ref(label if label in azsld.by else None, word, jd_ids)],
        })

    # 1–4. Slovo
    new_rows = read_csv(ROOT / 'ml/slovo_new.csv')
    for r in new_rows:
        rank = rank_of(r['basis'])
        if rank is not None and rank <= 10:
            slovo_item('slovo_new_installed', r['slovo'], r['azsld'], word_of(r['azsld']), r['basis'], verify.get(r['slovo'], ()))
    for r in read_csv(ROOT / 'ml/slovo_azsld.csv'):
        slovo_item('slovo_map', r['slovo'], r['azsld'], word_of(r['azsld']), r['basis'])
    matches = json.loads((args.r4 / 'matches.json').read_text(encoding='utf-8'))
    mapped = {r['slovo'] for r in read_csv(ROOT / 'ml/slovo_azsld.csv')}
    for ru, label in matches['azsld']:
        if ru not in mapped:
            slovo_item('slovo_azsld55', ru, label, word_of(label), 'ru_az.txt: AzSLD etiketi')
    for r in new_rows:
        rank = rank_of(r['basis'])
        if rank is None or rank > 10:
            slovo_item('slovo_new_unchecked', r['slovo'], r['azsld'], word_of(r['azsld']), r['basis'], verify.get(r['slovo'], ()))

    # 5. D5: söz formaları
    for r in read_csv(ROOT / 'ml/review/word_forms_review_w500.csv'):
        a, b = r['gloss_a'], r['gloss_b']
        items.append({
            'id': f"d5/{r['id']}", 'section': 'd5', 'kind': 'pair', 'a': a, 'b': b, 'text': f'{a} / {b}',
            'hint': r['question_az'] if r['category'] == 'meaning flip' else '',
            'groups': [{'label': a, 'srcs': azsld.clips(a, args.azsld_clips, add), 'example': r['example_a']},
                       {'label': b, 'srcs': azsld.clips(b, args.azsld_clips, add), 'example': r['example_b']}],
        })

    # 6. çoxsözlü etiketlər: doğma işarə, yoxsa sözbəsöz tərcümə?
    new_labels = {r['azsld'] for r in new_rows}
    for label in sorted(l for l in labels if ' ' in l.strip() and l not in new_labels and l in azsld.by):
        items.append({
            'id': f'literal/{label}', 'section': 'literal', 'kind': 'literal', 'a': label, 'b': '', 'word': word_of(label),
            'text': label, 'groups': [{'label': 'AzSLD iştirakçıları', 'srcs': azsld.clips(label, 3, add), 'main': True}],
        })

    # 7. N8: sinonimlər, idiomlar, çoxmənalılıq
    jobs = []
    for k, r in enumerate(read_csv(ROOT / 'ml/text_alt.csv'), 1):
        jid = f'n8-{k:02d}'
        jobs.append({'id': jid, 'text': r['alt']})
        items.append({'id': f"n8/{r['text']}|{r['alt']}", 'section': 'n8', 'kind': 'alt', 'a': r['text'], 'b': r['alt'],
                      'text': f"{r['alt']} → {r['text']}", 'avatar': f'clips/avatar/{jid}.mp4', 'job': jid,
                      'question': f'Mətndə «{r["alt"]}» yazılanda avatar «{r["text"]}» işarəsini göstərir. Düzdür?'})
    for k, (text, alt, note) in enumerate(N8_PROPOSALS, 1):
        jid = f'n8-new-{k:02d}'
        jobs.append({'id': jid, 'text': text})
        items.append({'id': f'n8/{text}|{alt}', 'section': 'n8', 'kind': 'alt', 'a': text, 'b': alt, 'proposal': True,
                      'text': f'{alt} → {text}', 'avatar': f'clips/avatar/{jid}.mp4', 'job': jid, 'note': note,
                      'question': f'«{alt}» sözü üçün də bu işarə («{text}») göstərilsin?'})
    for label, word, q in N8_POLYSEMY:
        groups = [{'label': 'AzSLD iştirakçıları', 'srcs': azsld.clips(label, 2, add)}] if label else []
        jds = jd.clips(word, 3, add)
        if jds:
            groups.append({'label': 'JestDili.az', 'srcs': jds})
        items.append({'id': f'n8/poly/{word}', 'section': 'n8', 'kind': 'poly', 'a': word, 'b': '', 'text': word,
                      'question': q, 'groups': groups})

    # 8. N9: açıq doğma qaydalar
    for rid, text, q, hint in N9:
        jid = f'n9-{rid}'
        jobs.append({'id': jid, 'text': text})
        items.append({'id': f'n9/{rid}', 'section': 'n9', 'kind': 'rule', 'a': rid, 'b': '', 'text': text,
                      'question': q, 'hint': hint, 'avatar': f'clips/avatar/{jid}.mp4', 'job': jid})

    # 9. V2: /rey/ bəndlərinin 50-si
    rey = json.loads((ROOT / 'public/rey/items.json').read_text(encoding='utf-8'))
    n_m4, n_az, n_c, n_s = map(int, args.v2.split(','))
    pick = ([x for x in rey['items'] if x['source'].startswith('müsahibə')][:n_m4]
            + [x for x in rey['items'] if x['source'].startswith('AzSLD test') and x['dir'] == 't2s'][:n_az]
            + [x for x in rey['items'] if x['source'].startswith('set C')][:n_c]
            + [x for x in rey['items'] if x['dir'] == 's2t'][:n_s])
    for x in pick:
        jid = f"v2-{x['id']}"
        jobs.append({'id': jid, **({'text': x['text']} if x['dir'] == 't2s' else {'signs': x['signs']})})
        items.append({'id': f"v2/{x['id']}", 'section': 'v2', 'kind': x['dir'], 'a': x['id'], 'b': '',
                      'rey_id': x['id'], 'source': x['source'], 'text': x['text'],
                      'recognized': x.get('recognized'), 'reference': x.get('reference'),
                      'avatar': f'clips/avatar/{jid}.mp4', 'job': jid})

    clips.run()

    # avatar videoları
    adir = out / 'clips' / 'avatar'
    adir.mkdir(parents=True, exist_ok=True)
    (adir / 'jobs.json').write_text(json.dumps(jobs, ensure_ascii=False, indent=1), encoding='utf-8')
    plans_path = adir / 'plans.json'
    plans = json.loads(plans_path.read_text(encoding='utf-8')) if plans_path.exists() else {}
    todo = [j['id'] for j in jobs if not (adir / f"{j['id']}.mp4").exists() or plans.get(j['id'], {}).get('text') != j.get('text')]
    if todo and not args.no_avatar:
        print(f'avatar: {len(todo)} video')
        subprocess.run(['node', 'tools/render_avatar_clips.mjs', '--jobs', str(adir / 'jobs.json'), '--out', str(adir),
                        '--only', ','.join(todo)], cwd=ROOT, check=True)
        plans = json.loads(plans_path.read_text(encoding='utf-8'))
    for it in items:
        if 'job' in it:
            it['glosses'] = plans.get(it.pop('job'), {}).get('glosses', '')

    # yoxlama: videosu olmayan bəndlər
    have = lambda rel: (out / rel).exists()                                  # noqa: E731
    for it in items:
        for g in it.get('groups', []):
            g['srcs'] = [s for s in g['srcs'] if have(s)]
        it['groups'] = [g for g in it.get('groups', []) if g['srcs']]
        if it.get('avatar') and not have(it['avatar']):
            print(f"  avatar videosu yoxdur: {it['id']}")
            it.pop('avatar')
    no_main = [it['id'] for it in items if it['kind'] == 'slovo' and not any(g.get('main') for g in it['groups'])]
    if no_main:
        print(f'  Slovo klipi tapılmadı: {no_main}')

    sections = json.loads((PAGE / 'sections.json').read_text(encoding='utf-8'))
    for s in sections:
        s['count'] = sum(it['section'] == s['id'] for it in items)
    data = {'release': release, 'built': datetime.datetime.now().isoformat(timespec='seconds'),
            'rey_release': rey['release'], 'sections': sections, 'items': items}
    (out / 'data.js').write_text('window.REVIEW = ' + json.dumps(data, ensure_ascii=False, indent=0) + ';\n', encoding='utf-8')
    for name in ('index.html', 'review.js', 'review.css', 'OXU.txt'):
        shutil.copy(PAGE / name, out / name)
    size = sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 2 ** 20
    print(f'{out}: buraxılış {release}, {len(items)} bənd · ' +
          ', '.join(f"{s['id']} {s['count']}" for s in sections) + f' · {size:.0f} MB')
    if args.zip:
        z = shutil.make_archive(str(out), 'zip', root_dir=out.parent, base_dir=out.name)
        print(f'{z}: {Path(z).stat().st_size / 2 ** 20:.0f} MB')


if __name__ == '__main__':
    main()
