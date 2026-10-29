"""Merge groups from the DTW pair scores: within a morph family, complete linkage (every pair in a group
must pass: 1-NN balanced accuracy <= ACC and distance ratio <= RATIO). Pairs that were not measured block a merge.
python -I tools/same_sign/merge_groups.py data/azsld/same_sign/pairs.json [ACC RATIO]"""
import json, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')
res = json.load(open(sys.argv[1], encoding='utf8'))
ACC, RATIO = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (0.81, 1.20)
ok = lambda r: r['bal_acc'] <= ACC and r['ratio'] <= RATIO
neg = res['random_different']
print(f'rule: bal_acc <= {ACC} and ratio <= {RATIO}; random different-word pairs passing: '
      f'{sum(ok(r) for r in neg)} of {len(neg)}; confirmed families passing: '
      f'{sum(ok(r) for r in res["confirmed"])} of {len(res["confirmed"])}')
score, fam = {}, defaultdict(set)
for r in res['families']:
    score[frozenset((r['a'], r['b']))] = r
    fam[r['key']] |= {r['a'], r['b']}
groups = []
for key, members in sorted(fam.items()):
    cl = [{m} for m in sorted(members)]
    while True:
        best = None
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                pairs = [score.get(frozenset((a, b))) for a in cl[i] for b in cl[j]]
                if all(p and ok(p) for p in pairs):
                    worst = max(p['ratio'] for p in pairs)
                    if best is None or worst < best[0]:
                        best = (worst, i, j)
        if not best:
            break
        _, i, j = best
        cl[i] |= cl.pop(j)
    groups += [(key, sorted(c)) for c in cl if len(c) > 1]
print(f'{len(groups)} groups, {sum(len(g) for _, g in groups)} forms -> {len(groups)} signs:')
for key, g in groups:
    print(f'  {key:9s} ' + ' = '.join(g))
if len(sys.argv) > 4:
    worst = lambda g: {k: max(score[frozenset((a, b))][k] for a in g for b in g if a < b) for k in ('bal_acc', 'ratio')}
    json.dump([{'key': k, 'texts': g, **worst(g)} for k, g in groups], open(sys.argv[4], 'w', encoding='utf8'),
              ensure_ascii=False)
