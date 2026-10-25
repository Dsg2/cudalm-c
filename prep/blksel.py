# Builds proxy-anneal corpora from scored v3pre blocks. Every arm has NB blocks with v3pre's source shares.
#   R<k>: random blocks (seed k)      Q<k>: blocks from loss quintile k (1 = lowest) within each source
# usage: blksel.py <v3pre dir> <fp dir> <out root> <NB> [arms...]   (default arms R1 R2 R3 Q1..Q5)
import sys, os, shutil, random
from collections import defaultdict
src_dir, fp, root, NB = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
arms = sys.argv[5:] or ['R1', 'R2', 'R3', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']
blk = [l.split('\t') for l in open(os.path.join(fp, 'blocks.tsv')).read().split('\n') if l]
wl = defaultdict(list)
for l in open(os.path.join(fp, 'winloss.tsv')):
    o, v = l.split(); wl[int(o)].append(float(v))
B = []  # (start, len, src, score or None, w1, w2)
for i, st, n, s, so in blk:
    st, n = int(st), int(n); a, b = wl.get(st), wl.get(st + n // 2)
    B.append((st, n, s, (a[0] + b[0]) / 2 if a and b else None, a and a[0], b and b[0]))
bysrc = defaultdict(list)
for b in B:
    if b[3] is not None: bysrc[b[2]].append(b)
tot = sum(len(v) for v in bysrc.values())
raw = {s: NB * len(v) / tot for s, v in bysrc.items()}
share = {s: int(r) for s, r in raw.items()}
for s in sorted(raw, key=lambda s: raw[s] - share[s], reverse=True)[:NB - sum(share.values())]: share[s] += 1
with open(os.path.join(fp, 'sources.txt'), 'w') as f:
    f.write('source        blocks  mean   q20    q40    q60    q80   w1~w2 corr  share\n')
    for s, v in sorted(bysrc.items(), key=lambda x: -len(x[1])):
        sc = sorted(x[3] for x in v); q = [sc[int(len(sc) * p)] for p in (0.2, 0.4, 0.6, 0.8)]
        x, y = [b[4] for b in v], [b[5] for b in v]; mx, my = sum(x) / len(x), sum(y) / len(y)
        cv = sum((a - mx) * (b - my) for a, b in zip(x, y)); vx = sum((a - mx) ** 2 for a in x); vy = sum((b - my) ** 2 for b in y)
        f.write(f'{s:13s} {len(v):6d} {sum(sc)/len(sc):.3f}  ' + '  '.join(f'{t:.3f}' for t in q) + f'   {cv/(vx*vy)**0.5:.3f}     {share[s]}\n')
print(open(os.path.join(fp, 'sources.txt')).read())
tb = open(os.path.join(src_dir, 'train.bin'), 'rb')
for arm in arms:
    rng = random.Random(1000 + int(arm[1:])); pick = []
    for s, v in bysrc.items():
        if arm[0] == 'R': pool = v
        else:
            v = sorted(v, key=lambda b: b[3]); k = int(arm[1:]); pool = v[len(v) * (k - 1) // 5: len(v) * k // 5]
        pick += rng.sample(pool, min(share[s], len(pool)))
    rng.shuffle(pick)
    d = os.path.join(root, arm); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'train.bin'), 'wb') as f:
        for st, n, *_ in pick: tb.seek(st * 2); f.write(tb.read(n * 2))
    for fn in ('val.bin', 'tokenizer.json', 'tokenizer.bin', 'template.txt', 'meta.json'):
        shutil.copy(os.path.join(src_dir, fn), d)
    ms = sum(b[3] for b in pick) / len(pick)
    print(f'{arm}: {len(pick)} blocks, {sum(b[1] for b in pick)} tokens, mean block loss {ms:.3f}')
