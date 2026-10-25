# Replays build_corpus.py's weighted round-robin interleave from meta.json to label every train.bin block with its
# source. Writes <out>/blocks.tsv (idx start len src srcoff) and <out>/offsets.txt (2 scoring windows per block).
import json, sys, os
data, out, T = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2048
m = json.load(open(os.path.join(data, 'meta.json')))
BLOCK, TARGET, REP = 1 << 16, m['target_tokens'], m['max_repeat']
cats = []
for cat, w in m['weights'].items():
    srcs = list(m['categories'][cat]['sources'].items())
    lens = [s['unique'] for _, s in srcs]
    cats.append(dict(names=[n for n, _ in srcs], lens=lens, caps=[int(l * REP) for l in lens], pos=[0] * len(srcs),
                     contrib=[0] * len(srcs), want=int(TARGET * w), emitted=0, dead=False))
def next_block(c):
    while True:
        cand = [j for j in range(len(c['lens'])) if c['pos'][j] < c['caps'][j] and c['lens'][j]]
        if not cand: c['dead'] = True; return None
        j = min(cand, key=lambda x: c['contrib'][x])
        off = c['pos'][j] % c['lens'][j]; n = min(BLOCK, c['lens'][j] - off)
        if n <= 0: c['pos'][j] = c['caps'][j]; continue
        c['pos'][j] += n; c['contrib'][j] += n; c['emitted'] += n
        return c['names'][j], off, n
val_target = max(BLOCK, int(TARGET * 0.005)); vw = 0; written = 0; tpos = 0; rows = []
while written < TARGET:
    live = [c for c in cats if not c['dead'] and c['emitted'] < c['want']]
    if not live: break
    c = min(live, key=lambda x: x['emitted'] / x['want'])
    b = next_block(c)
    if b is None: continue
    src, off, n = b
    if vw < val_target:
        take = min(n, val_target - vw); vw += take
        if take < n: rows.append((tpos, n - take, src, off + take)); tpos += n - take
    else:
        rows.append((tpos, n, src, off)); tpos += n
    written += n
assert tpos == m['train_tokens'], (tpos, m['train_tokens'])
for c in cats:
    for j, nm in enumerate(c['names']):
        for cat in m['categories'].values():
            if nm in cat['sources']: assert cat['sources'][nm]['emitted'] == c['contrib'][j], (nm, c['contrib'][j])
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, 'blocks.tsv'), 'w') as f, open(os.path.join(out, 'offsets.txt'), 'w') as g:
    for i, (st, n, src, so) in enumerate(rows):
        f.write(f'{i}\t{st}\t{n}\t{src}\t{so}\n')
        if n > 2 * (T + 1):
            g.write(f'{st}\n{st + n // 2}\n')
from collections import Counter
cnt = Counter(r[2] for r in rows); tok = Counter()
for r in rows: tok[r[2]] += r[1]
print(f'{len(rows)} blocks, {tpos} train tokens (matches meta)')
for s, k in cnt.most_common(): print(f'  {s:14s} {k:6d} blocks {tok[s]:>11d} tok')
