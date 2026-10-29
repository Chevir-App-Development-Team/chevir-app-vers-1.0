"""Labels that are one sign: the one place every reader takes them from (the notebooks via build_notebooks.py,
analyze.py, install_continuous.py and the browser's sign_merge.json via tools/build_sign_merge.py).
Standard library only, so the browser file can be rebuilt without torch.

  ml/gloss_spelling.csv  one word typed two ways in the annotations (variant → gloss), applied first
  ml/lemma_groups.csv    word-form families; confirmed = yes / no is the signers' decision (E8, E15)
  ml/same_sign.csv       forms found to be one sign but not reviewed yet (gloss → group)
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
YES = ('yes', 'y', 'bəli', 'hə', 'he')
NO = ('no', 'n', 'xeyr', 'yox')


def load_groups(path, confirmed_only=True):
    """gloss → family. A form marked yes joins its family; the family's base form (gloss == group)
    joins automatically unless it is marked no. Empty or no = the form stays its own class."""
    with open(path, newline='', encoding='utf-8') as f:
        rows = [{k: (v or '').strip() for k, v in r.items()} for r in csv.DictReader(f)]
    out = {}
    for r in rows:
        if not confirmed_only or r['confirmed'].lower() in YES:
            out[r['gloss']] = r['group']
    if confirmed_only:
        for r in rows:
            if r['gloss'] == r['group'] and r['group'] in out.values() and r['confirmed'].lower() not in NO:
                out[r['gloss']] = r['group']
    # a family needs at least two forms to merge anything
    sizes = {}
    for fam in out.values():
        sizes[fam] = sizes.get(fam, 0) + 1
    return {g: fam for g, fam in out.items() if sizes[fam] > 1}


def sign_merge(lemma_groups=None, same_sign=None):
    """One sign, several glosses → {gloss: group}: the confirmed word-form families (ml/lemma_groups.csv)
    and the forms not reviewed yet (ml/same_sign.csv). Other files (e.g. an earlier version) can be given."""
    out = dict(load_groups(lemma_groups or os.path.join(HERE, 'lemma_groups.csv')))
    with open(same_sign or os.path.join(HERE, 'same_sign.csv'), encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        out[r['gloss']] = r['group']
    for r in rows:                         # a group that joins a confirmed family takes its name
        out[r['group']] = out.get(r['group'], r['group'])
    return {g: grp for g, grp in out.items() if g != grp}


def spelling():
    """ml/gloss_spelling.csv → {variant: gloss}: the same word typed two ways in the annotations."""
    with open(os.path.join(HERE, 'gloss_spelling.csv'), encoding='utf-8') as f:
        return {r['variant']: r['gloss'] for r in csv.DictReader(f)}
