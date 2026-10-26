# Gate for proxy arms: paired (by rep = seed) comparison of <arm>_<rep>.items.tsv against a control arm.
#   gate.py <items dir> <control> <arm> [arm...]      -> per-arm table + "DECISION <arm> ADOPT|EXTEND|KILL|HOLD"
#   gate.py rate <items dir>                          -> accuracy-per-margin exchange rate over every run in the dir
# Rule (pre-registered): ADOPT if d_margin >= +0.10 and t >= 3; EXTEND (to 8 reps) if +0.05 <= d < +0.10, or d >= +0.10
# with t < 3 and n < 8; KILL if d <= 0; otherwise HOLD. Soft accuracy = mean sigmoid(margin / s_task), s_task frozen as
# 0.5 x sd of LS-Micro2's margins (logs\ga\r0.items.tsv). Dev sets (<job>.dev.items.tsv) are reported, not gated.
import sys, os, glob, math, re
import numpy as np
W = {'hellaswag': 1, 'arc': 1, 'piqa': 1, 'arithmark': 0.65}; CH = {'hellaswag': 25, 'arc': 25, 'piqa': 50, 'arithmark': 25}
REF = r'E:\cudalm\logs\ga\r0.items.tsv'
def load(p):
    t = {}
    for l in open(p):
        f = l.rstrip('\n').split('\t')
        if len(f) != 4 or f[0] == 'task': continue
        t.setdefault(f[0], {})[int(f[1])] = (int(f[2]), float(f[3]))
    return t
ref = load(REF) if os.path.exists(REF) else {}
S = {k: 0.5 * float(np.std([m for _, m in v.values()])) for k, v in ref.items()}
def comp(t, fn):  # index over HS, ARC (mean of E/C), PIQA, AM with fn(task items) -> per-task value
    v = {k: fn(k, t[k]) for k in t}; out = {}
    for k in ('hellaswag', 'piqa', 'arithmark'):
        if k in v: out[k] = v[k]
    a = [v[k] for k in ('arc_easy', 'arc_challenge') if k in v]
    if a: out['arc'] = sum(a) / len(a)
    return out
def index(c, norm):
    num = den = 0
    for k, x in c.items():
        num += W[k] * (100 * (x - CH[k]) / (100 - CH[k]) if norm else x); den += W[k]
    return num / den if den else float('nan')
acc = lambda k, d: 100 * np.mean([c for c, _ in d.values()])
soft = lambda k, d: 100 * np.mean([1 / (1 + math.exp(-m / S.get(k, 0.05))) for _, m in d.values()])
mar = lambda k, d: 100 * np.mean([m for _, m in d.values()])
def stats(p):
    t = load(p); return {'II': index(comp(t, acc), 1), 'soft': index(comp(t, soft), 1), 'margin': index(comp(t, mar), 0),
                         'tasks': {k: (acc(k, d), mar(k, d)) for k, d in t.items()}}
def runs(d, arm):
    r = {}
    for p in glob.glob(os.path.join(d, glob.escape(arm) + '_*.items.tsv')):
        m = re.fullmatch(re.escape(arm) + r'_(\d+)\.items\.tsv', os.path.basename(p))
        if m and not p.endswith('.dev.items.tsv'): r[int(m.group(1))] = p
    return r
def mt(x):
    x = np.array(x, float); n = len(x); sd = x.std(ddof=1) if n > 1 else float('nan')
    return x.mean(), sd, x.mean() / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float('nan')
if sys.argv[1] == 'rate':
    d = sys.argv[2]; P = [p for p in glob.glob(os.path.join(d, '*.items.tsv')) if not p.endswith('.dev.items.tsv')]
    st = [stats(p) for p in P]; x = np.array([s['margin'] for s in st]); y = np.array([s['II'] for s in st]); z = np.array([s['soft'] for s in st])
    b = np.polyfit(x, y, 1); bs = np.polyfit(x, z, 1)
    print(f'{len(P)} runs | II = {b[1]:.2f} + {b[0]:.3f} x margin (r {np.corrcoef(x, y)[0, 1]:.2f}, resid sd {np.std(y - np.polyval(b, x)):.2f})')
    print(f'soft II = {bs[1]:.2f} + {bs[0]:.3f} x margin (r {np.corrcoef(x, z)[0, 1]:.2f}, resid sd {np.std(z - np.polyval(bs, x)):.2f})')
    print(f'-> +0.10 margin ~ {10 * b[0]:+.2f} II;  II 10 needs margin ~ {(10 - b[1]) / b[0]:.2f}'); sys.exit()
d, ctl, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
C = {k: stats(p) for k, p in runs(d, ctl).items()}
cm = mt([s['margin'] for s in C.values()]) if C else (float('nan'),) * 3
print(f'control {ctl}: n {len(C)}  margin {cm[0]:.3f} +-{cm[1]:.3f}  II {np.mean([s["II"] for s in C.values()]):.2f}  soft {np.mean([s["soft"] for s in C.values()]):.2f}')
for a in arms:
    A = {k: stats(p) for k, p in runs(d, a).items()}; ks = sorted(set(A) & set(C))
    if not ks: print(f'DECISION {a} NODATA'); continue
    dm, sd, t = mt([A[k]['margin'] - C[k]['margin'] for k in ks]); di = mt([A[k]['II'] - C[k]['II'] for k in ks])
    ds = mt([A[k]['soft'] - C[k]['soft'] for k in ks]); n = len(ks)
    print(f'{a:16s} n {n}  d_margin {dm:+.3f} +-{sd:.3f} (t {t:+.1f})  d_II {di[0]:+.2f} (t {di[2]:+.1f})  d_soft {ds[0]:+.2f} (t {ds[2]:+.1f})')
    tk = sorted(set.intersection(*[set(A[k]['tasks']) & set(C[k]['tasks']) for k in ks]))
    print('    ' + '  '.join(f'{x} {np.mean([A[k]["tasks"][x][0] - C[k]["tasks"][x][0] for k in ks]):+.1f}/{np.mean([A[k]["tasks"][x][1] - C[k]["tasks"][x][1] for k in ks]):+.2f}' for x in tk) + '   (acc/margin)')
    dv = [(load(os.path.join(d, f'{a}_{k}.dev.items.tsv')), load(os.path.join(d, f'{ctl}_{k}.dev.items.tsv'))) for k in ks
          if os.path.exists(os.path.join(d, f'{a}_{k}.dev.items.tsv')) and os.path.exists(os.path.join(d, f'{ctl}_{k}.dev.items.tsv'))]
    for x in sorted(set.intersection(*[set(p) & set(q) for p, q in dv])) if dv else []:
        print(f'    dev {x}: d_acc {np.mean([acc(x, p[x]) - acc(x, q[x]) for p, q in dv]):+.2f}  d_margin {np.mean([mar(x, p[x]) - mar(x, q[x]) for p, q in dv]):+.3f}  (n {len(dv)})')
    dec = 'ADOPT' if dm >= 0.10 and t >= 3 else 'KILL' if dm <= 0 else 'EXTEND' if (dm >= 0.05 and n < 8) or (dm >= 0.10 and n < 8) else 'HOLD'
    print(f'DECISION {a} {dec} {dm:+.3f} {t:+.1f}')
