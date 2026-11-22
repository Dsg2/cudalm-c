"""lm-evaluation-harness vs `cudalm lmeval`, item by item.

    compare.py <cudalm items.tsv> <bench_dir> <lm-eval output dir>...

items.tsv: `CUDALM_LMEVAL_ITEMS=<file> cudalm lmeval <model> all`; bench_dir: the .tsv task files cudalm read;
each output dir: one `lm_eval run ... --log_samples --output_path <dir>` on the `cudalm hf` export (tasks hellaswag,
arc_easy, arc_challenge, piqa, lambada_openai, arithmark3). Per task: lm-eval's acc / acc_norm (acc_norm divides by
characters of the choice), the same log-likelihoods divided by UTF-8 bytes of the continuation as cudalm does, how
often that pick agrees with cudalm's, and the per-item margin difference (gold logP/byte minus the best wrong one)."""
import glob, json, math, os, re, sys

items, bench, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
TASKS = {"hellaswag": "hellaswag", "arc_easy": "arc_easy", "arc_challenge": "arc_challenge", "piqa": "piqa",
         "lambada_openai": "lambada", "arithmark3": "arithmark"}
unesc = lambda s: re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), s)
nb = lambda s: max(1, len(s.encode("utf-8")))
ours, key = {}, {}                                   # task -> {item: (correct, margin)}, {(context, first cont): item}
for ln in open(items, encoding="utf-8"):
    t, i, c, m = ln.rstrip("\n").split("\t")
    if t != "task": ours.setdefault(t, {})[int(i)] = (int(c), float(m))
for t in TASKS.values():
    key[t] = {}
    for i, ln in enumerate(open(os.path.join(bench, t + ".tsv"), encoding="utf-8")):
        f = [unesc(x) for x in ln.rstrip("\n").split("\t")]
        key[t].setdefault((f[1], f[2]), i)


def args_of(s):
    a = s["arguments"]
    return [(x[0], x[1]) for x in ([[v["arg_0"], v["arg_1"]] for v in a.values()] if isinstance(a, dict) else a)]


def ii(sc):                                          # Open SLM intelligence index from acc_norm %
    n = [(sc["hellaswag"] - 25) / .75, ((sc["arc_easy"] + sc["arc_challenge"]) / 2 - 25) / .75, (sc["piqa"] - 50) / .5,
         (sc["arithmark"] - 25) / .75]
    return (n[0] + n[1] + n[2] + .65 * n[3]) / 3.65


for run in runs:
    print(f"\n=== {run}")
    print(f"  {'task':14s} {'n':>6s} {'acc':>7s} {'acc_norm':>8s} {'byte-norm':>9s} {'cudalm':>7s} {'agree':>7s}"
          f" {'d margin':>9s} {'|d|':>7s} {'corr':>6s} {'unmatched':>9s}")
    sc_l, sc_b, sc_o = {}, {}, {}
    for lt, ct in TASKS.items():
        fs = sorted(glob.glob(os.path.join(run, "**", f"samples_{lt}_*.jsonl"), recursive=True))
        if not fs:
            print(f"  {lt:14s} no samples"); continue
        A = AN = BN = OK = AG = n = um = 0; pm = []
        for ln in open(fs[-1], encoding="utf-8"):
            s = json.loads(ln); ar = args_of(s); ll = [float(r[0]) for r in s["filtered_resps"]]
            k = key[ct].get(ar[0])
            if k is None or k not in ours.get(ct, {}):
                um += 1; continue
            ok, m = ours[ct][k]; n += 1; A += float(s["acc"]); OK += ok
            if ct == "lambada":                      # greedy match: no choices, margin = logP/byte of the target
                BN += float(s["acc"]); AG += ok == int(s["acc"]); pm.append((m, ll[0] / nb(ar[0][1]))); continue
            g = int(s["target"])
            pb = [l / nb(c) for l, (_, c) in zip(ll, ar)]
            bi = max(range(len(pb)), key=pb.__getitem__)
            AN += float(s["acc_norm"]); BN += bi == g; AG += (bi == g) == bool(ok)
            pm.append((m, pb[g] - max(p for j, p in enumerate(pb) if j != g)))
        if not n:
            print(f"  {lt:14s} nothing matched ({um} unmatched)"); continue
        d = [b - a for a, b in pm]
        ma, mb = sum(a for a, _ in pm) / n, sum(b for _, b in pm) / n
        va, vb = sum((a - ma) ** 2 for a, _ in pm), sum((b - mb) ** 2 for _, b in pm)
        r = sum((a - ma) * (b - mb) for a, b in pm) / math.sqrt(va * vb) if va > 0 and vb > 0 else float("nan")
        an = f"{100 * AN / n:8.2f}" if ct != "lambada" else f"{'-':>8s}"
        print(f"  {lt:14s} {n:6d} {100 * A / n:7.2f} {an} {100 * BN / n:9.2f} {100 * OK / n:7.2f} {100 * AG / n:6.2f}%"
              f" {sum(d) / n:+9.4f} {sum(abs(x) for x in d) / n:7.4f} {r:6.3f} {um:9d}")
        if ct != "lambada":
            sc_l[ct], sc_b[ct], sc_o[ct] = 100 * AN / n, 100 * BN / n, 100 * OK / n
    if len(sc_l) == 5:
        print(f"  II: lm-eval acc_norm {ii(sc_l):.2f} | lm-eval byte-normalised {ii(sc_b):.2f} | cudalm {ii(sc_o):.2f}")
