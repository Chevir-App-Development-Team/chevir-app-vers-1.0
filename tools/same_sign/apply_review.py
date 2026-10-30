"""Signers' answers on ml/review/word_forms_review.csv → ml/lemma_groups.csv and ml/same_sign.csv.

The answer column asks "same sign?": bəli → the two labels are one sign (a confirmed row in lemma_groups.csv),
xeyr → two signs (a merge that was in place is undone: confirmed = no, or the row leaves same_sign.csv),
maybe / did not understand → nothing changes. Every reviewed same_sign.csv row leaves that file: a confirmed
merge moves to lemma_groups.csv, a rejected one is dropped. Groups are named so that no chain is left
(sign_merge() looks a gloss up once): an existing confirmed family keeps its name, two different words that are
one sign get "A/B", otherwise the existing same_sign name, then the infinitive, then the most frequent label.

python -I tools/same_sign/apply_review.py ml/review/word_forms_review.csv
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding='utf-8')
review_path = Path(sys.argv[1])
TAG = 'word-form review 2026-10'


def read(p, enc='utf-8'):
    with open(p, newline='', encoding=enc) as f:
        return list(csv.DictReader(f))


def answer(a):
    a = a.strip().lower()
    if 'anlamadim' in a or 'basha dushmedim' in a:
        return 'unclear'
    if re.match(r'^(bəli|beli)', a):
        return 'yes'
    if a.startswith('xeyr'):
        return 'no'
    return 'unclear'


review = read(review_path, 'utf-8-sig')
col = next(k for k in review[0] if k.startswith('same_sign_answer'))
lemma = read(ROOT / 'ml/lemma_groups.csv')
same = read(ROOT / 'ml/same_sign.csv')
freq = Counter(r['gloss'] for r in read(ROOT / 'ml/splits/glosses_split.csv') if r['dup'] != 'True')

confirmed = {r['gloss']: r['group'] for r in lemma if r['confirmed'] == 'yes'}
same_group = {r['gloss']: r['group'] for r in same}
yes, no, unclear = [], [], []
for r in review:
    a, b, ans = r['gloss_a'], r['gloss_b'], answer(r[col])
    note = f"{TAG}: {r[col].strip()}" + (f"; {r['comment'].strip()}" if r['comment'].strip() else '')
    (yes if ans == 'yes' else no if ans == 'no' else unclear).append((a, b, note, r['category']))

# Union of everything that is one sign after the review: confirmed families minus the rejected pairs, plus yes
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def union(a, b):
    parent[find(a)] = find(b)

rejected = {frozenset((a, b)) for a, b, _, _ in no}
for g, grp in confirmed.items():
    if frozenset((g, grp)) not in rejected:
        union(g, grp)
for a, b, _, _ in yes:
    union(a, b)
comps = defaultdict(set)
for x in list(parent):
    comps[find(x)].add(x)

synonyms = {frozenset((a, b)) for a, b, _, c in yes if c.startswith('7')}
families = set(confirmed.values())
def name(members):
    fam = sorted(m for m in members if m in families and confirmed.get(m) == m)
    syn = [p for p in synonyms if p <= members]
    if syn:
        heads = sorted(next(iter(syn)), key=lambda g: -freq[g])
        return '/'.join(heads)
    if fam:
        return fam[0]
    old = sorted({same_group[m] for m in members if m in same_group} & members)
    if old:
        return old[0]
    inf = [m for m in members if m.endswith(('MAQ', 'MƏK'))]
    return sorted(inf or members, key=lambda g: (-freq[g], g))[0]

group_of = {}
for members in comps.values():
    if len(members) > 1:
        n = name(members)
        for m in members:
            group_of[m] = n

# lemma_groups.csv: existing rows (rejected → no, renamed when their family got a new name) + new confirmed rows
notes = {}
for a, b, note, _ in yes + no:
    notes.setdefault(a, note)
    notes.setdefault(b, note)
out, seen = [], set()
for r in lemma:
    r = dict(r)
    g = r['gloss']
    if r['confirmed'] == 'yes' and frozenset((g, r['group'])) in rejected:
        r['confirmed'] = 'no'
        r['note'] = (r['note'] + '; ' if r['note'] else '') + notes[g]
    elif g in group_of and r['confirmed'] == 'yes':
        r['group'] = group_of[g]
    out.append(r)
    seen.add(g)
new = sorted((grp, g) for g, grp in group_of.items() if g not in seen)
for grp, g in new:
    out.append({'group': grp, 'gloss': g, 'confirmed': 'yes', 'note': notes.get(g, f'{TAG}: base form of the group')})
with open(ROOT / 'ml/lemma_groups.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, ['group', 'gloss', 'confirmed', 'note'], lineterminator='\n')
    w.writeheader()
    w.writerows(out)

# same_sign.csv: every reviewed row leaves (confirmed → lemma_groups.csv, rejected → dropped)
reviewed = {g for a, b, _, c in yes + no + unclear if c[0] in '12' for g in (a, b)}
keep = [r for r in same if r['gloss'] not in reviewed and r['group'] not in reviewed]
with open(ROOT / 'ml/same_sign.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, ['group', 'gloss', 'pair_bal_acc', 'evidence'], lineterminator='\n')
    w.writeheader()
    w.writerows(keep)

# No chains: a group name is never a member of another group
bad = [(g, grp) for g, grp in group_of.items() if group_of.get(grp, grp) != grp]
print(f'yes {len(yes)}, no {len(no)}, unclear {len(unclear)}; {len(new)} new confirmed rows, '
      f'{sum(r["confirmed"] == "no" for r in out) - sum(r["confirmed"] == "no" for r in lemma)} families undone; '
      f'same_sign.csv {len(same)} → {len(keep)} rows; chains: {bad or "none"}')
by = defaultdict(list)
for g, grp in group_of.items():
    by[grp].append(g)
for grp in sorted(by):
    print(f'  {grp}: ' + ', '.join(sorted(m for m in by[grp] if m != grp)))
print('unclear (unchanged):', ', '.join(f'{a}/{b}' for a, b, _, _ in unclear))
