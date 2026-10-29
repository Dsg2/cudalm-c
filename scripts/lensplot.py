# Plot `cudalm lens` tables (CUDALM_LENS_OUT): next-token CE and agreement with the full model's top-1 per stage,
# logit lens vs tuned lens (when the table has the t* columns). Several tables = several models on one figure.
#   py -3 lensplot.py <out.png> <a.lens.tsv> [b.lens.tsv ...]
import csv, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COL = ['#2a6fdb', '#e8590c', '#2b9348', '#9c36b5']

def load(p):
    with open(p, encoding='utf-8') as f: return list(csv.DictReader(f, delimiter='\t'))

def main():
    out, tabs = sys.argv[1], sys.argv[2:]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, constrained_layout=True)
    names = None
    for k, p in enumerate(tabs):
        r = load(p); names = names or [x['stage'] for x in r]; x = range(len(r)); c = COL[k % len(COL)]
        lab = os.path.basename(p).replace('.lens.tsv', '')
        tuned = 'tce' in r[0] and r[0]['tce'] != ''
        a1.plot(x, [float(v['ce']) for v in r], color=c, lw=2, ls='--' if tuned else '-', marker='o', ms=3, label=f'{lab} logit lens')
        a2.plot(x, [100 * float(v['agree']) for v in r], color=c, lw=2, ls='--' if tuned else '-', marker='o', ms=3)
        if tuned:
            a1.plot(x, [float(v['tce']) for v in r], color=c, lw=2, marker='o', ms=3, label=f'{lab} tuned lens')
            a2.plot(x, [100 * float(v['tagree']) for v in r], color=c, lw=2, marker='o', ms=3)
    a1.set_ylabel('next-token CE (nats)'); a1.set_yscale('log'); a1.set_title('Loss of each stage read out through the head')
    a2.set_ylabel('agrees with final top-1 (%)'); a2.set_ylim(0, 102); a2.set_title('How often the stage already has the model\'s answer')
    a2.set_xticks(range(len(names))); a2.set_xticklabels(names, rotation=60, fontsize=8)
    for a in (a1, a2): a.grid(alpha=0.25); a.spines[['top', 'right']].set_visible(False)
    a1.legend(frameon=False, fontsize=9)
    fig.savefig(out, dpi=140); print(out)

main()
