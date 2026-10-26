# Builds v3pre-hw from v3pre + LS-Micro2 per-document scores (the HW recipe tested in fdoc/fdec/fscr):
#   prose sources (ultrafineweb, ufw-l3, cosmopedia-v2, finemath): the higher-loss half of each source's tokens
#   every other source: a random half of its documents, so v3pre's source shares are kept
# usage: hwbuild.py <v3pre dir> <fdec dir> <out dir>      val.bin = v3pre's (unfiltered)
import sys, os, json, random, bisect, shutil
import numpy as np
from collections import defaultdict
src, fd, out = sys.argv[1:4]
PROSE = {'ultrafineweb', 'ufw-l3', 'cosmopedia-v2', 'finemath'}
blk = [l.split('\t') for l in open(os.path.join(fd, 'blocks.tsv')).read().split('\n') if l]
starts = [int(b[1]) for b in blk]; bsrc = [b[3] for b in blk]
docs = defaultdict(list)  # src -> [(start, n, loss/tok)]
for fn in ('docs_micro.tsv', 'docs_rest.tsv'):
    for l in open(os.path.join(fd, fn)):
        ds, n, sl = l.split('\t'); ds, n = int(ds), int(n)
        docs[bsrc[bisect.bisect_right(starts, ds) - 1]].append((ds, n, float(sl) / n))
rng = random.Random(2026); keep = {}; stats = {}
for s, v in docs.items():
    tot = sum(d[1] for d in v)
    if s in PROSE: v = sorted(v, key=lambda d: -d[2])
    else: v = v[:]; rng.shuffle(v)
    k, acc = [], 0
    for d in v:
        if acc >= tot / 2: break
        k.append(d); acc += d[1]
    k.sort(); keep[s] = k
    stats[s] = dict(docs=len(v), tokens=tot, kept_docs=len(k), kept_tokens=acc,
                    min_loss_kept=round(min(d[2] for d in k), 4) if s in PROSE else None)
os.makedirs(out, exist_ok=True)
tb = np.memmap(os.path.join(src, 'train.bin'), dtype=np.uint16, mode='r')
order = [(s, d) for s, k in keep.items() for d in k]; rng.shuffle(order)   # interleave sources
n_out = 0
with open(os.path.join(out, 'train.bin'), 'wb', buffering=1 << 24) as f:
    for s, (ds, n, _) in order:
        seg = tb[ds:ds + n]; f.write(seg.tobytes()); n_out += n
        if seg[-1] != 0: f.write(b'\0\0'); n_out += 1
for fn in ('val.bin', 'tokenizer.json', 'tokenizer.bin', 'template.txt'):
    shutil.copy(os.path.join(src, fn), out)
m = json.load(open(os.path.join(src, 'meta.json')))
m.update(profile='v3pre-hw', train_tokens=n_out, total_tokens=n_out + m['val_tokens'],
         filter='HW: prose sources keep the higher-loss half (LS-Micro2 loss/token), other sources a random half; val = v3pre val',
         hw_sources=stats)
json.dump(m, open(os.path.join(out, 'meta.json'), 'w'), indent=2)
print(f'v3pre-hw: {n_out} train tokens')
for s, st in sorted(stats.items(), key=lambda x: -x[1]['tokens']):
    print(f"  {s:14s} {st['kept_tokens']:>11d} of {st['tokens']:>11d} tok  {st['kept_docs']:>8d} docs" + (f"  loss>={st['min_loss_kept']}" if st['min_loss_kept'] else ''))
