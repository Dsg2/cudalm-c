# Full-scale corpus for a docsel recipe (Stage 2 decays): base[+extraN...] over ALL v3pre docs scored by LS-Micro2.
#   d1build.py <v3pre dir> <fdec dir> <extra dir> <out dir> <recipe> [tokens=1.8e9] [seed=2027]
# base: HW* variant as in docsel.py (filtered sources keep a random sample of their higher-loss half, others random docs;
# shares = v3pre source tokens x multiplier). Extras: N% of tokens from <extra dir>/<name>.bin, capped at 4 passes.
import sys, os, json, random, bisect, shutil, re
import numpy as np
from collections import defaultdict
src, fd, xd, out, recipe = sys.argv[1:6]; TOK = float(sys.argv[6]) if len(sys.argv) > 6 else 1.8e9
rng = random.Random(int(sys.argv[7]) if len(sys.argv) > 7 else 2027)
PROSE = {'ultrafineweb', 'ufw-l3', 'cosmopedia-v2', 'finemath'}
HWV = {'HW': (PROSE, {}), 'HWK': (PROSE, {}), 'HWM': (PROSE - {'finemath'}, {}), 'HWC': (PROSE, {'github-python': 10 / 21}),
       'HWMC': (PROSE - {'finemath'}, {'github-python': 10 / 21}), 'HWW': (PROSE, {'ultrafineweb': 2 / 3})}
base, *ex = recipe.split('+'); filt, mul = HWV[base]
ex = [(m.group(1), float(m.group(2)) / 100) for m in (re.fullmatch(r'([a-z]+)(\d+(?:\.\d+)?)', e) for e in ex)]
xs = {}
for nm, fr in ex:
    arr = np.fromfile(os.path.join(xd, nm + '.bin'), dtype=np.uint16); e = np.flatnonzero(arr == 0)
    fr2 = min(fr, 4 * len(arr) / TOK); xs[nm] = (arr, list(zip(np.r_[0, e[:-1] + 1], e + 1)), fr2)
    print(f'extra {nm}: {len(arr)} tokens, {fr:.1%} asked -> {fr2:.2%} ({fr2 * TOK / len(arr):.2f} passes)')
blk = [l.split('\t') for l in open(os.path.join(fd, 'blocks.tsv')).read().split('\n') if l]
starts = [int(b[1]) for b in blk]; bsrc = [b[3] for b in blk]
docs = defaultdict(list)
for fn in ('docs_micro.tsv', 'docs_rest.tsv'):
    for l in open(os.path.join(fd, fn)):
        ds, n, sl = l.split('\t'); ds, n = int(ds), int(n)
        docs[bsrc[bisect.bisect_right(starts, ds) - 1]].append((ds, n, float(sl) / n))
btok = TOK * (1 - sum(v[2] for v in xs.values()))
w = {s: sum(d[1] for d in v) * mul.get(s, 1) for s, v in docs.items()}; wt = sum(w.values())
order, stats = [], {}
for s, v in docs.items():
    if s in filt:
        v = sorted(v, key=lambda d: -d[2]); h, acc, cand = sum(d[1] for d in v) / 2, 0, []
        for d in v:
            if acc >= h: break
            cand.append(d); acc += d[1]
    else: cand = v[:]
    rng.shuffle(cand); budget = btok * w[s] / wt; acc = 0; k = 0
    for d in cand:
        if acc >= budget: break
        order.append(('d', d)); acc += d[1]; k += 1
    stats[s] = dict(tokens=acc, docs=k, budget=int(budget), short=acc < budget * 0.99)
    print(f'  {s:14s} {acc:>11d} tok {k:>8d} docs (budget {int(budget)}{", SHORT" if acc < budget * 0.99 else ""})')
for nm, (arr, dl, fr) in xs.items():
    acc = 0; i = 0; perm = list(range(len(dl))); rng.shuffle(perm)
    while acc < TOK * fr:
        a, b = dl[perm[i % len(perm)]]; order.append(('x', (nm, a, b))); acc += b - a; i += 1
    stats['extra:' + nm] = dict(tokens=int(acc), docs=i)
rng.shuffle(order); os.makedirs(out, exist_ok=True)
tb = np.memmap(os.path.join(src, 'train.bin'), dtype=np.uint16, mode='r'); n_out = 0
with open(os.path.join(out, 'train.bin'), 'wb', buffering=1 << 24) as f:
    for kind, it in order:
        if kind == 'x': nm, a, b = it; f.write(xs[nm][0][a:b].tobytes()); n_out += b - a; continue
        ds, n, _ = it; seg = tb[ds:ds + n]; f.write(seg.tobytes()); n_out += n
        if seg[-1] != 0: f.write(b'\0\0'); n_out += 1
for fn in ('val.bin', 'tokenizer.json', 'tokenizer.bin', 'template.txt'):
    shutil.copy(os.path.join(src, fn), out)
m = json.load(open(os.path.join(src, 'meta.json')))
m.update(profile='v3pre-' + recipe, train_tokens=n_out, total_tokens=n_out + m['val_tokens'], recipe=recipe,
         filter='d1build.py: docsel recipe at full scale (see FILTER_PLAN.md); val = v3pre val', sources=stats)
json.dump(m, open(os.path.join(out, 'meta.json'), 'w'), indent=2, default=int)
print(f'{recipe}: {n_out} train tokens -> {out}')
