"""Diagnose large fp32 logit deltas: MoE selection near-ties, or cross-sequence state in `cudalm logits`?
    hf_diag.py <cudalm_cpu.exe> <model.bin> <hf_dir> <val.bin>... [--threads N] [--batch]
Rebuilds hf_check's full-mode sequences; --batch scores all of them in one `cudalm logits` call (as hf_check does),
otherwise each 2048 window alone."""
import json, os, subprocess, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

a = sys.argv[1:]
thr = int(a[a.index("--threads") + 1]) if "--threads" in a else 0
batch = "--batch" in a
pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] != "--threads")]
exe, model, hf, vals = pos[0], pos[1], pos[2], pos[3:]
if thr:
    torch.set_num_threads(thr)
m = AutoModelForCausalLM.from_pretrained(hf, trust_remote_code=True, dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
T, eot = m.config.max_position_embeddings, m.config.eos_token_id
seqs = [[eot][:1], tok.encode("The")[:1] + tok.encode(" cat"), tok.encode("Once upon a time")[:3]]
seqs += [tok.encode(t) for t in ["The capital of France is", "def fib(n):\n    if n < 2:\n        return n\n",
                                 "<|user|>\nWhat is 12 + 30?\n<|assistant|>\n", "Water boils at 100 degrees Celsius because"]]
rng = np.random.default_rng(1)
for v in vals:
    d = np.fromfile(v, np.uint16)
    starts = np.flatnonzero(d == eot)[:-1] + 1
    for s in rng.choice(starts, 5, replace=False):
        seqs.append(d[s:s + 256].tolist())
    s0 = int(rng.integers(0, len(d) - T)); seqs.append(d[s0:s0 + T].tolist())
seqs = [s for s in seqs if s]
if not batch:
    seqs = [s for s in seqs if len(s) == T]


def ref(sq):
    open("dg_ids.txt", "w").write("\n".join(" ".join(map(str, s)) for s in sq) + "\n")
    subprocess.run([exe, "logits", model, "dg_ids.txt", "dg.f32"], check=True, capture_output=True)
    b = np.fromfile("dg.f32", np.uint8); V = int(np.frombuffer(b[8:12], np.int32)[0]); out, o = [], 12
    for _ in sq:
        n = int(np.frombuffer(b[o:o + 4], np.int32)[0]); o += 4
        out.append(np.frombuffer(b[o:o + 4 * n * V], np.float32).reshape(n, V)); o += 4 * n * V
    os.remove("dg.f32"); os.remove("dg_ids.txt")
    return out


gaps = {}                          # layer -> (min gap over picks whose competing weight > 1e-3, that weight)
def hook(l):
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
for l, layer in enumerate(m.model.layers):
    if hasattr(layer.mlp, "router"):
        layer.mlp.register_forward_hook(hook(l))

R = ref(seqs)
print(f"threads {thr or 'default'}, {'batch of ' + str(len(seqs)) if batch else 'windows alone'}")
for si, (s, r) in enumerate(zip(seqs, R)):
    with torch.no_grad():
        g = m(torch.tensor([s]), use_cache=False).logits[0].numpy()
    d = np.abs(g - r).max(-1)
    if len(s) < T and d.max() <= 1e-3:
        continue
    L = sorted(gaps); G = torch.stack([gaps[l] for l in L]); gmin, gl = G.min(0)
    bad = np.flatnonzero(d > 1e-3)
    print(f"seq {si} (len {len(s)}): max|d| {d.max():.2e} at {int(d.argmax())}, {len(bad)} positions > 1e-3; "
          f"weighted near-ties (gap < 1e-5): {int((gmin < 1e-5).sum())}")
    for p in bad[:10]:
        print(f"  pos {p}: |d| {d[p]:.2e}, min weighted gap {gmin[p]:.2e} (layer {L[gl[p]]})")
