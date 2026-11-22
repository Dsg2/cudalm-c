"""HF port check: the transformers model vs the cudalm fp32 reference (CPU build).

    hf_check.py <cudalm_cpu.exe> <model.bin> <workdir> [--val val.bin]... [--quick] [--threads N]

Exports <model> to <workdir>/hf (cudalm hf + tltk2hf + the export/hf modules), then compares
  0. chat template generation prompt == cmd_chat's;
  1. logits on fixed sequences (val docs, one full-T window, 1-3 token sequences) with `cudalm logits`; a sequence
     that first diverges at an exact MoE routing tie (gap < 1e-5) is excluded from there on (at most max(1, 10%) of the sequences);
  2. 8 greedy continuations of 64 tokens (+1 crossing the sliding window): HF with KV cache == HF without cache ==
     generate(), and cudalm's argmax reproduces every generated token (teacher-forced through `cudalm logits`);
  3. bf16: mean KL and top-1 agreement on positions whose fp32 top-2 margin is >= 0.1.
"""
import json, os, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tltk2hf  # noqa: E402

a = sys.argv[1:]
vals = [a[i + 1] for i, x in enumerate(a) if x == "--val"]
quick = "--quick" in a
thr = int(a[a.index("--threads") + 1]) if "--threads" in a else 0
pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--val", "--threads"))]
exe, model, work = pos[0], os.path.abspath(pos[1]), os.path.abspath(pos[2])
import torch  # noqa: E402
if thr:
    torch.set_num_threads(thr)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

hf = os.path.join(work, "hf")
os.makedirs(work, exist_ok=True)


def run(*args):
    r = subprocess.run([exe, *args], capture_output=True, text=True, errors="replace")
    if r.returncode:
        sys.exit(f"{' '.join(args[:1])} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


run("hf", model, hf)
for f in ("configuration_cudalm.py", "modeling_cudalm.py"):
    shutil.copy(os.path.join(HERE, "hf", f), hf)
tltk2hf.convert(tltk2hf.tltk_blob(model), hf)
cfg = json.load(open(os.path.join(hf, "config.json")))
T, V, eot = cfg["max_position_embeddings"], cfg["vocab_size"], cfg["eos_token_id"]
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
if cfg["cudalm"]["template"] == "chatml":
    ct = tok.apply_chat_template([{"role": "user", "content": "Hi"}], add_generation_prompt=True, tokenize=False)
    print("[0] chat template prompt", "== cmd_chat -> PASS" if ct == "<|user|>\nHi\n<|assistant|>\n" else f"{ct!r} -> FAIL")
win = max(cfg["layer_window"])

# ---- sequences
seqs = [[eot][:1], tok.encode("The")[:1] + tok.encode(" cat"), tok.encode("Once upon a time")[:3]]
texts = ["The capital of France is", "def fib(n):\n    if n < 2:\n        return n\n",
         "<|user|>\nWhat is 12 + 30?\n<|assistant|>\n", "Water boils at 100 degrees Celsius because"]
seqs += [tok.encode(t) for t in texts]
rng = np.random.default_rng(1)
for v in vals:
    d = np.fromfile(v, np.uint16)
    starts = np.flatnonzero(d == eot)[:-1] + 1
    for s in rng.choice(starts, 2 if quick else 5, replace=False):
        seqs.append(d[s:s + 256].tolist())
    if not quick:
        s0 = int(rng.integers(0, len(d) - T))
        seqs.append(d[s0:s0 + T].tolist())                # full window: crosses docs, exercises the sliding window
seqs = [s for s in seqs if s]


def ref_logits(name, sq):
    p = os.path.join(work, name)
    open(p + ".txt", "w").write("\n".join(" ".join(map(str, s)) for s in sq) + "\n")
    t0 = time.time()
    run("logits", model, p + ".txt", p + ".f32")
    b = np.fromfile(p + ".f32", np.uint8)
    assert bytes(b[:4]) == b"LGT1"
    ns, vv = np.frombuffer(b[4:12], np.int32)
    out, o = [], 12
    for _ in range(ns):
        n = int(np.frombuffer(b[o:o + 4], np.int32)[0]); o += 4
        out.append(np.frombuffer(b[o:o + 4 * n * vv], np.float32).reshape(n, vv)); o += 4 * n * vv
    os.remove(p + ".f32"); os.remove(p + ".txt")
    print(f"  cudalm logits: {ns} sequences, {sum(len(s) for s in sq)} tokens ({time.time() - t0:.1f}s)")
    return out


print(f"[1] logits on {len(seqs)} sequences (lengths {sorted(len(s) for s in seqs)})")
ref = ref_logits("seqs", seqs)
m32 = AutoModelForCausalLM.from_pretrained(hf, trust_remote_code=True, dtype=torch.float32).eval()
gaps = {}                          # MoE layer -> per-position min selection gap over picks whose competitor weight > 1e-3


def gap_hook(l):
    def f(mod, inp, out):
        xf = inp[0].reshape(-1, inp[0].shape[-1])
        p = torch.softmax(mod.router(xf).float(), -1); taken = torch.zeros_like(p, dtype=torch.bool)
        g = torch.full((p.shape[0],), float("inf"))
        for k in range(mod.K):
            sc = (p + mod.router_bias[min(k, mod.S - 1)].float() if mod.S else p).masked_fill(taken, float("-inf"))
            t2 = sc.topk(2, -1)
            w = torch.maximum(p.gather(1, t2.indices[:, :1]), p.gather(1, t2.indices[:, 1:2]))[:, 0]
            g = torch.where(w > 1e-3, torch.minimum(g, t2.values[:, 0] - t2.values[:, 1]), g)
            taken.scatter_(1, t2.indices[:, :1], True)
        gaps[l] = g
    return f


hooks = [ly.mlp.register_forward_hook(gap_hook(l)) for l, ly in enumerate(m32.model.layers) if hasattr(ly.mlp, "router")]
worst = tot_abs = tot_n = agree = npos = cut = ncut = 0
untied = True
with torch.no_grad():
    for s, r in zip(seqs, ref):
        gaps.clear()
        g = m32(torch.tensor([s])).logits[0].float().numpy()      # the default call, as lm-eval makes it
        d = np.abs(g - r); n = len(s)
        bad = np.flatnonzero(d.max(-1) > 1e-3)
        if len(bad):                  # an exact routing tie broken differently diverges from there on: exclude, if explained
            p0 = int(bad[0]); gm = min(float(v[p0]) for v in gaps.values()) if gaps else float("inf")
            print(f"  len {n}: max|d| {d.max():.2e}, first > 1e-3 at pos {p0}, router gap there {gm:.1e}"
                  f"{' (tie: positions from there excluded)' if gm < 1e-5 else ''}")
            if gm < 1e-5:
                cut += n - p0; ncut += 1; d, g, r, n = d[:p0], g[:p0], r[:p0], p0
            else:
                untied = False
        if n:
            worst = max(worst, d.max()); tot_abs += d.sum(); tot_n += d.size
            agree += (g.argmax(-1) == r.argmax(-1)).sum(); npos += n
for h in hooks:
    h.remove()
mean = tot_abs / tot_n
ok1 = worst <= 1e-3 and mean <= 1e-4 and agree == npos and untied and ncut <= max(1, len(seqs) // 10)
print(f"  fp32: max|dlogit| {worst:.2e}, mean {mean:.2e}, argmax {agree}/{npos}"
      f"{f', {cut} positions after {ncut} routing tie(s) excluded' if cut else ''} -> {'PASS' if ok1 else 'FAIL'}")

# ---- greedy continuations
prompts = [s[:32] for s in seqs if len(s) >= 32][:4] + [tok.encode(t) for t in texts]
prompts = prompts[:8]
long = [s for s in seqs if len(s) == T]
if win and long and not quick:                            # decoding across the window edge (positions win-16 .. win+48)
    prompts.append(long[0][:win - 16])
print(f"[2] greedy continuations, {len(prompts)} x 64 tokens{' (one crosses the window)' if len(prompts) > 8 else ''}")
gen_c, ok2 = [], True
with torch.no_grad():
    for p in prompts:
        ids = torch.tensor([p]); out = m32(ids, use_cache=True); cache = out.past_key_values; g = []
        for _ in range(64):
            nx = int(out.logits[0, -1].argmax()); g.append(nx)
            out = m32(torch.tensor([[nx]]), past_key_values=cache, use_cache=True)
        full = list(p)                                    # no cache: full recompute each step
        for _ in range(64):
            full.append(int(m32(torch.tensor([full]), use_cache=False).logits[0, -1].argmax()))
        gg = m32.generate(ids, max_new_tokens=64, do_sample=False, pad_token_id=eot,     # stops after eot
                          attention_mask=torch.ones_like(ids))[0, len(p):].tolist()
        if eot in gg:
            gg = gg[:gg.index(eot) + 1]
        same = full[len(p):] == g and g[:len(gg)] == gg
        ok2 &= same
        if not same:
            print(f"  prompt len {len(p)}: cache/no-cache/generate differ")
        gen_c.append((p, g))
refg = ref_logits("gen", [p + g for p, g in gen_c])
for (p, g), r in zip(gen_c, refg):
    am = r[len(p) - 1:len(p) - 1 + len(g)].argmax(-1).tolist()
    if am != g:
        k = next(i for i in range(len(g)) if am[i] != g[i]); row = np.sort(r[len(p) - 1 + k])
        print(f"  cudalm argmax differs at step {k} (top-2 margin {row[-1] - row[-2]:.2e})"); ok2 = False
print(f"  cache == no cache == generate(), cudalm reproduces all {sum(len(g) for _, g in gen_c)} tokens -> {'PASS' if ok2 else 'FAIL'}")
print("  sample:", repr(tok.decode(gen_c[7 if len(gen_c) > 7 else -1][1][:40])))

# ---- bf16
print("[3] bf16 vs the fp32 reference (near-ties flip under bf16 rounding, so agreement is gated on margin >= 0.1)")
del m32
mb = AutoModelForCausalLM.from_pretrained(hf, trust_remote_code=True, dtype=torch.bfloat16).eval()
ag, mg, kl = [], [], []
with torch.no_grad():
    for s, r in zip(seqs, ref):
        g = mb(torch.tensor([s]), use_cache=False).logits[0].float()
        r = torch.from_numpy(r.copy())
        t = r.topk(2, -1).values; mg.append(t[:, 0] - t[:, 1]); ag.append(g.argmax(-1) == r.argmax(-1))
        lp, lq = r.log_softmax(-1), g.log_softmax(-1); kl.append((lp.exp() * (lp - lq)).sum(-1))
ag, mg, kl = torch.cat(ag).float(), torch.cat(mg), torch.cat(kl)
sure = mg >= 0.1
ok3 = kl.mean() <= 5e-3 and ag[sure].mean() >= 0.995
print(f"  argmax agreement all {ag.mean():.2%}, margin>=0.1 {ag[sure].mean():.2%} ({sure.float().mean():.0%} of positions); "
      f"mean KL {kl.mean():.2e} nats -> {'PASS' if ok3 else 'FAIL'}")
print("RESULT", "PASS" if ok1 and ok2 and ok3 and (cfg["cudalm"]["template"] != "chatml" or ct == "<|user|>\nHi\n<|assistant|>\n") else "FAIL")
