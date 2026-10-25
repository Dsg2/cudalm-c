# Document-level selection for v3pre proxy anneals.
#   docsel.py pool <fp dir> <nblocks>                  -> <fp>/pool_spans.txt (source-proportional random blocks)
#   docsel.py build <v3pre> <fp> <out root> <tokens> <arm> <rep> [...]  arm in R L50 H50 X50 D25 D10 HW, one corpus per arm+rep
# Scores come from `cudalm eval` with CUDALM_EVAL_SPANS: <fp>/docs_micro.tsv (LS-Micro2) and <fp>/docs_ref.tsv (LS2.5).
#   L50/H50: lowest/highest half of each source's tokens by LS-Micro2 loss per token
#   X50: highest half by excess loss (LS-Micro2 - LS2.5 per token): learnable docs the small model hasn't learned yet
import sys, os, random, shutil
import numpy as np
from collections import defaultdict
cmd, a = sys.argv[1], sys.argv[2:]
if cmd == 'pool':
    fp, n = a[0], int(a[1]); rng = random.Random(77)
    blk = [l.split('\t') for l in open(os.path.join(fp, 'blocks.tsv')).read().split('\n') if l]
    blk = [(int(b[1]), int(b[2]), b[3]) for b in blk if int(b[2]) >= 4097]
    pick = sorted(rng.sample(blk, n))
    with open(os.path.join(fp, 'pool_spans.txt'), 'w') as f:
        for st, ln, _ in pick: f.write(f'{st} {ln}\n')
    with open(os.path.join(fp, 'pool.tsv'), 'w') as f:
        for st, ln, s in pick: f.write(f'{st}\t{ln}\t{s}\n')
    print(f'{n} blocks, {sum(p[1] for p in pick)} tokens'); sys.exit()
src_dir, fp, root, tokens = a[0], a[1], a[2], int(a[3]); jobs = list(zip(a[4::2], map(int, a[5::2])))
pool = [l.split('\t') for l in open(os.path.join(fp, 'pool.tsv')).read().split('\n') if l]
spans = sorted((int(st), int(ln), s) for st, ln, s in pool)
def rd(fn):
    d = {}
    for l in open(os.path.join(fp, fn)):
        ds, n, sl = l.split('\t'); d[int(ds)] = (int(n), float(sl))
    return d
mi = rd('docs_micro.tsv'); rf = rd('docs_ref.tsv') if os.path.exists(os.path.join(fp, 'docs_ref.tsv')) else {k: (n, sl) for k, (n, sl) in mi.items()}   # no ref: excess 0
docs = defaultdict(list)  # src -> [(start, n, loss/tok, excess/tok)]
import bisect
starts = [s[0] for s in spans]
for ds, (n, sl) in mi.items():
    if ds not in rf or rf[ds][0] != n: continue
    s = spans[bisect.bisect_right(starts, ds) - 1][2]
    docs[s].append((ds, n, sl / n, (sl - rf[ds][1]) / n))
tot = sum(d[1] for v in docs.values() for d in v)
if not os.path.exists(os.path.join(fp, 'docstats.txt')):
    with open(os.path.join(fp, 'docstats.txt'), 'w') as f:
        f.write('source        docs    tokens     loss  excess  corr(loss,excess)\n')
        for s, v in sorted(docs.items(), key=lambda x: -sum(d[1] for d in x[1])):
            w = np.array([d[1] for d in v], float); x = np.array([d[2] for d in v]); y = np.array([d[3] for d in v])
            c = np.corrcoef(x, y)[0, 1] if len(v) > 2 else 0
            f.write(f'{s:13s} {len(v):6d} {int(w.sum()):10d}  {np.average(x, weights=w):.3f}  {np.average(y, weights=w):.3f}  {c:.3f}\n')
def half(v, key, top, frac=0.5):  # docs making up frac of the source's tokens, best first by key
    v = sorted(v, key=key, reverse=top); out, acc, h = [], 0, sum(d[1] for d in v) * frac
    for d in v:
        if acc >= h: break
        out.append(d); acc += d[1]
    return out
sel = {'R': lambda v: v, 'L50': lambda v: half(v, lambda d: d[2], False), 'H50': lambda v: half(v, lambda d: d[2], True),
       'X50': lambda v: half(v, lambda d: d[3], True),
       'D25': lambda v: half(v, lambda d: d[2], True, 0.75), 'D10': lambda v: half(v, lambda d: d[2], True, 0.9)}
PROSE = {'ultrafineweb', 'ufw-l3', 'cosmopedia-v2', 'finemath'}   # HW: H50 for prose sources, random elsewhere
tb = np.memmap(os.path.join(src_dir, 'train.bin'), dtype=np.uint16, mode='r')
for arm, rep in jobs:
    rng = random.Random(f'{arm}-{rep}'); out = []; ml = mx = mt = 0
    for s, v in docs.items():
        budget = tokens * sum(d[1] for d in v) / tot
        cand = (sel['H50'](v) if s in PROSE else v)[:] if arm == 'HW' else sel[arm](v)[:]; rng.shuffle(cand); acc = 0
        for d in cand:
            if acc >= budget: break
            out.append(d); acc += d[1]
    rng.shuffle(out)
    d = os.path.join(root, f'{arm}_{rep}'); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'train.bin'), 'wb') as f:
        for ds, n, l, x in out:
            seg = tb[ds:ds + n]; f.write(seg.tobytes())
            if seg[-1] != 0: f.write(b'\0\0')
            ml += l * n; mx += x * n; mt += n
    for fn in ('val.bin', 'tokenizer.json', 'tokenizer.bin', 'template.txt', 'meta.json'):
        shutil.copy(os.path.join(src_dir, fn), d)
    print(f'{arm}_{rep}: {len(out)} docs, {mt} tokens, loss/tok {ml/mt:.3f}, excess/tok {mx/mt:.3f}')
