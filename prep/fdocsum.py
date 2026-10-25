# Summarises fdoc.res: per arm mean +- sd over replicates of val / index / margin, and the difference vs R (Welch t).
import sys, re, statistics as st
from collections import defaultdict
d = defaultdict(lambda: defaultdict(list))
for l in open(sys.argv[1]):
    m = re.match(r'(\w+?)_(\d+) iters .*\| val (\S+) index (\S+) margin (\S+)', l)
    if m:
        for k, v in zip(('val', 'index', 'margin'), m.groups()[2:]): d[m.group(1)][k].append(float(v))
def ms(x): return st.mean(x), (st.stdev(x) if len(x) > 1 else 0)
print('arm   n   val (mean+-sd)       index           margin           | vs R: d val (t)   d index (t)   d margin (t)')
for a, v in d.items():
    row = f'{a:4s} {len(v["val"]):2d}  ' + '  '.join(f'{ms(v[k])[0]:.4f}+-{ms(v[k])[1]:.4f}' if k == 'val' else f'{ms(v[k])[0]:.3f}+-{ms(v[k])[1]:.3f}' for k in ('val', 'index', 'margin'))
    if a != 'R' and 'R' in d:
        for k in ('val', 'index', 'margin'):
            x, y = v[k], d['R'][k]; df = st.mean(x) - st.mean(y)
            se = ((ms(x)[1] ** 2) / len(x) + (ms(y)[1] ** 2) / len(y)) ** 0.5
            row += f'   {df:+.4f} ({df / se if se else 0:+.1f})' if k == 'val' else f'   {df:+.3f} ({df / se if se else 0:+.1f})'
    print(row)
