"""Review sheet: every word-form decision (merge into one sign or keep apart) for signers to confirm.
.venv/Scripts/python.exe -I tools/same_sign/build_review.py <families_all.json> <out.csv>"""
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent))
import signdist as S  # noqa: E402

ROOT = S.ROOT
fam_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
ACC, RATIO = 0.82, 1.20
low = S.B.az_lower
read = lambda p: list(csv.DictReader(open(ROOT / p, encoding='utf8')))

# Frequency and one example sentence per gloss (the sentence where it is signed most often)
rows = [r for r in read('ml/splits/glosses_split.csv') if r['dup'] != 'True']
freq = Counter(r['gloss'] for r in rows)
where = defaultdict(Counter)
for r in rows:
    where[r['gloss']][r['rec_id'].split('/')[0]] += 1
text = {s['folder']: s['text'] for s in json.load(open(ROOT / 'public/tercume/assets/cont_model/sentences.json', encoding='utf8'))}
example = lambda g: text.get(where[g].most_common(1)[0][0], '') if where[g] else ''

confirmed = {r['gloss']: r['group'] for r in read('ml/lemma_groups.csv') if r['confirmed'] == 'yes'}
same = read('ml/same_sign.csv')
new_same = {r['gloss']: r['group'] for r in same if 'hand paths' in r['evidence']}
old_same = {r['gloss']: r['group'] for r in same if 'hand paths' not in r['evidence']}
group_of = {**confirmed, **old_same, **new_same}
for g in list(group_of.values()):
    group_of.setdefault(g, g)
resolve = lambda g: group_of.get(group_of.get(g, g), group_of.get(g, g))


def status(a, b):
    if resolve(a) != resolve(b):
        return 'separate'
    for src, name in ((new_same, 'merged by the clip test (new, not confirmed)'),
                      (old_same, 'merged by the model-confusion rule (not confirmed)')):
        if a in src or b in src:
            return name
    return 'confirmed family'


cache = {}
def measure(a, b):
    ta, tb = S.text_of.get(a), S.text_of.get(b)
    if not ta or not tb or ta == tb:
        return None
    k = tuple(sorted((ta, tb)))
    if k not in cache:
        cache[k] = S.compare(*k)
    return cache[k]


same_rule = lambda m: m is not None and m['bal_acc'] <= ACC and m['ratio'] <= RATIO
BOGUS = {'hə', 'qan', 'tə', 'in', 'işt', 'ət'}           # morph.js put different words together
NOT_NEG = {'GƏZMƏYƏ'}                                    # gəzməyə = "to walk" (dative), not a negation
SYN = [('KÖMƏK', 'YARDİM'), ('TƏHSİL', 'ÖYRƏNMƏK'), ('KƏND', 'RAYON'), ('RUS', 'RUSİYA')]

out, seen = [], set()
def add(cat, a, b, family, proposal):
    k = frozenset((a, b))
    if a == b or k in seen:
        return
    seen.add(k)
    m = measure(a, b)
    says = 'not measured (too few clips)' if m is None else 'same sign' if same_rule(m) else 'different'
    out.append({'category': cat, 'proposal': proposal, 'status': status(a, b), 'family': family,
                'gloss_a': a, 'gloss_b': b, 'example_a': example(a), 'example_b': example(b),
                'signs_a': freq[a], 'signs_b': freq[b], 'hands_say': says,
                'nn_balanced_acc': '' if m is None else m['bal_acc'], 'distance_ratio': '' if m is None else m['ratio']})


# 1-3: merges in place that signers never confirmed, and confirmed families the hands dispute
for g, grp in new_same.items():
    add('1 new merge (clip test)', g, resolve(grp), low(resolve(grp)), 'merge')
for g, grp in old_same.items():
    add('2 merge by model confusion', g, resolve(grp), low(resolve(grp)), 'merge')
for g, grp in confirmed.items():
    m = measure(g, grp)
    if m and not same_rule(m):
        add('3 confirmed, but the hands differ', g, grp, low(grp), 'check again')

# 4-6: family forms that stay apart
for f in json.load(open(fam_path, encoding='utf8')):
    if f['key'] in BOGUS:
        continue
    forms = [x for x in f['forms'] if not (f['key'] == 'it' and x['label'] == 'İT')]
    for x in forms:
        if x['label'] in NOT_NEG:
            x['neg'] = False
    pos = [x for x in forms if not x['neg']]
    if not pos:
        continue
    bases = [x['label'] for x in pos if x['base']] or [max((x['label'] for x in pos), key=lambda g: freq[g])]
    base = sorted(bases, key=lambda g: (not g.endswith(('MAQ', 'MƏK')), -freq[g]))[0]
    labels = {x['label'] for x in forms}
    for x in forms:
        g = x['label']
        if g == base:
            continue
        if x['neg']:
            add('6 negated form', g, base, f['key'], 'keep apart (merging loses the negation)')
        elif resolve(g) != resolve(base):
            rep = resolve(g) if resolve(g) in labels else g       # a merged group is asked once
            m = measure(rep, base)
            if m is None:
                add('5 not measured', rep, base, f['key'], 'decide')
            elif same_rule(m):
                add('4a hands look the same, not merged', rep, base, f['key'], 'decide')
            else:
                add('4b hands look different', rep, base, f['key'], 'keep apart')
for a, b in SYN:
    add('7 different words, maybe one sign', a, b, '', 'decide')

QUESTION = {'merge': 'Bu iki etiket eyni işarədir? (birləşdirilsin?)',
            'check again': 'Təsdiqlənib, amma əllərin hərəkəti fərqli görünür: eyni işarədir?',
            'keep apart': 'Əllərin hərəkəti fərqli görünür: ayrı işarələrdir?',
            'keep apart (merging loses the negation)': 'İnkar forması: ayrı işarədir?',
            'decide': 'Eyni işarədir, yoxsa ayrı?'}
cols = ['id', 'category', 'proposal', 'question_az', 'gloss_a', 'gloss_b', 'example_a', 'example_b',
        'same_sign_answer (bəli / xeyr / bilmirəm)', 'reviewer', 'comment', 'status', 'hands_say', 'nn_balanced_acc', 'distance_ratio',
        'signs_a', 'signs_b', 'family']
out.sort(key=lambda r: (r['category'], -(r['signs_a'] + r['signs_b'])))
with open(out_path, 'w', newline='', encoding='utf-8-sig') as fh:   # BOM: Excel shows ə, ş, ı right
    w = csv.DictWriter(fh, cols, lineterminator='\n')
    w.writeheader()
    for i, r in enumerate(out, 1):
        w.writerow({'id': i, 'question_az': QUESTION[r['proposal']], 'same_sign_answer (bəli / xeyr / bilmirəm)': '', 'reviewer': '',
                    'comment': '', **r})
print(f'{len(out)} rows:', dict(Counter(r['category'] for r in out)))
print('hands say:', dict(Counter(r['hands_say'] for r in out)))
