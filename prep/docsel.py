# Document-level selection for v3pre proxy anneals.
#   docsel.py pool <fp dir> <nblocks>                  -> <fp>/pool_spans.txt (source-proportional random blocks)
#   docsel.py build <v3pre> <fp> <out root> <tokens> <arm> <rep> [...]  one corpus per arm+rep
# arm = base[+extraN...]; base in R L50 H50 X50 D25 D10 (all sources) or HW* (prose hard half, see HWV);
#   +extraN mixes N% of the tokens from <fp>/extra/<extra>.bin (uint16 eot-separated docs), e.g. HWM+arith3+cloze6
# Scores come from `cudalm eval` with CUDALM_EVAL_SPANS: <fp>/docs_micro.tsv (LS-Micro2) and <fp>/docs_ref.tsv (LS2.5).
#   L50/H50: lowest/highest half of each source's tokens by LS-Micro2 loss per token
#   X50: highest half by excess loss (LS-Micro2 - LS2.5 per token): learnable docs the small model hasn't learned yet
import sys, os, random, shutil, re, bisect
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
PROSE = {'ultrafineweb', 'ufw-l3', 'cosmopedia-v2', 'finemath'}
# HW variants: (sources filtered to their hard half, source weight multipliers; shares renormalised)
#   HWK = HW control with fresh samples; HWW = web weight cut by 1/3 (pair with +dclmN to replace it)
HWV = {'HW': (PROSE, {}), 'HWK': (PROSE, {}), 'HWM': (PROSE - {'finemath'}, {}), 'HWC': (PROSE, {'github-python': 10 / 21}),
       'HWMC': (PROSE - {'finemath'}, {'github-python': 10 / 21}), 'HWW': (PROSE, {'ultrafineweb': 2 / 3}),
       'HWK5': (PROSE, {}), 'HWK9': (PROSE, {})}   # HW corpora for the KD arms (alpha 0.5 / 0.9)
XRE = re.compile(r'([a-z]+)(\d+(?:\.\d+)?)$')
def extra_docs(name):
    arr = np.fromfile(os.path.join(fp, 'extra', name + '.bin'), dtype=np.uint16); e = np.flatnonzero(arr == 0)
    return arr, list(zip(np.r_[0, e[:-1] + 1], e + 1))
tb = np.memmap(os.path.join(src_dir, 'train.bin'), dtype=np.uint16, mode='r')
for arm, rep in jobs:
    rng = random.Random(f'{arm}-{rep}'); out = []; ml = mx = mt = 0
    base, *ex = arm.split('+'); ex = [(XRE.match(e).group(1), float(XRE.match(e).group(2)) / 100) for e in ex]
    btok = tokens * (1 - sum(f for _, f in ex))
    filt, mul = HWV.get(base, (None, {}))
    w = {s: sum(d[1] for d in v) * mul.get(s, 1) for s, v in docs.items()}; wt = sum(w.values())
    for s, v in docs.items():
        budget = btok * w[s] / wt
        cand = ((sel['H50'](v) if s in filt else v) if filt is not None else sel[base](v))[:]; rng.shuffle(cand); acc = 0
        for d in cand:
            if acc >= budget: break
            out.append(d); acc += d[1]
    xs = []
    for nm, fr in ex:
        arr, dl = extra_docs(nm); acc = 0; k = 0; order = list(range(len(dl))); rng.shuffle(order)
        while acc < tokens * fr:
            i0, i1 = dl[order[k % len(order)]]; xs.append(arr[i0:i1]); acc += i1 - i0; k += 1
        print(f'  {arm}_{rep}: extra {nm} {acc} tokens ({k / len(dl):.2f} passes over {len(dl)} docs)')
    items = [('d', d) for d in out] + [('x', x) for x in xs]; rng.shuffle(items)
    d = os.path.join(root, f'{arm}_{rep}'); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'train.bin'), 'wb') as f:
        for kind, it in items:
            if kind == 'x': f.write(it.tobytes()); mt += len(it); continue
            ds, n, l, x = it; seg = tb[ds:ds + n]; f.write(seg.tobytes())
            if seg[-1] != 0: f.write(b'\0\0')
            ml += l * n; mx += x * n; mt += n
    for fn in ('val.bin', 'tokenizer.json', 'tokenizer.bin', 'template.txt', 'meta.json'):
        shutil.copy(os.path.join(src_dir, fn), d)
    print(f'{arm}_{rep}: {len(out)} docs + {len(xs)} extra, {mt} tokens, pool loss/tok {ml/max(1,mt):.3f}')
