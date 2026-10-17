#!/usr/bin/env python3
"""bin2gguf.py - convert a tinylm checkpoint (.bin) to GGUF for llama.cpp.

Reads the on-disk formats produced by c/tinylm.c (and the CUDA build):
  TLM1-TLM5  fp32 checkpoints (TLM2+ bundle the byte-level BPE tokenizer)
  TLQ1/TLQ2  quantized checkpoints (fp16, or Q8_64: fp16 scale + 64 int8,
             or cudalm q4n: fp32 scale + 128 nibbles; tinylm target only)

tinylm's FFN is top-1 MoE whose output is `gate * Expert_best(x)`, where `gate`
is the raw softmax probability of the winning expert. Two stock llama.cpp
architectures can host that:

  deepseek (default, EXACT)
      build_moe_ffn(..., norm_w=false, ...) keeps the raw router probability, so
      the gate survives. The arch also demands a shared expert, which tinylm has
      no counterpart for -- we emit one whose down projection is all zeros, so it
      contributes exactly 0.0 and the graph reduces to tinylm's. That costs
      3*D*F zero weights per layer (5-10% of the file).

  llama (--arch llama, APPROXIMATE for MoE; the default and exact for dense models)
      build_moe_ffn(..., norm_w=true, ...) divides the selected weights by their
      own sum; with one expert used that forces gate == 1.0 and drops the router
      probability entirely. On LS-63M that costs 2.9x perplexity (10.24 -> 29.66
      on verify_gguf.py's corpus). Kept as a debugging baseline -- do not ship it.

Sliding-window attention has no equivalent in either arch, so llama.cpp always
runs full causal attention. Within `win` tokens the two are identical, so the
context length is capped at `win` (1024) by default; --ctx lifts the cap and
prints a warning, since past that point llama.cpp attends to tokens the model
never saw during training.

The tokenizer is carried over as a GPT-2-style byte-level BPE so llama.cpp
tokenizes with the exact training vocab, and the chatml prompt format from
cmd_chat() is written as a Jinja chat template.

Usage:
  python bin2gguf.py <model.bin> [out.gguf] [--type q4n|q4|q8|fp16|f32] [--target tinylm|llama]
  (or `cudalm gguf <name> [type] [target]`, which applies the model's .env first)
"""

import argparse
import os
import shutil
import struct
import subprocess
import sys

import numpy as np

GGUF_ALIGNMENT = 32
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_I8 = 24   # raw int8/byte tensor (tinylm target: TLTK blob, q4 digits)

QK8_0 = 32          # values per Q8_0 block
Q8_0_BLOCK = 2 + QK8_0   # fp16 scale + 32 int8


def type_nbytes(ttype, n):
    if ttype == GGML_TYPE_Q8_0:
        return n // QK8_0 * Q8_0_BLOCK
    if ttype == GGML_TYPE_I8:
        return n
    return n * (2 if ttype == GGML_TYPE_F16 else 4)


def encode_q8_0(a):
    """Quantize to ggml Q8_0: per 32 values, an fp16 scale + 32 int8.

    Near-free for these checkpoints: TLQ2 already stores `scale * int8` over
    groups of 64, so every 32-value block is already an integer lattice and
    re-quantizing to a 32-value block only re-centres the scale.
    """
    b = a.reshape(-1, QK8_0).astype(np.float32)
    amax = np.abs(b).max(1, keepdims=True)
    d = (amax / 127.0).astype(np.float16).astype(np.float32)
    q = np.where(d > 0, np.rint(b / np.where(d > 0, d, 1.0)), 0.0)
    out = np.empty((b.shape[0], Q8_0_BLOCK), np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = np.clip(q, -127, 127).astype(np.int8).view(np.uint8)
    return out.tobytes()

# GGUF KV value types (v3)
V_UINT32 = 4
V_INT32 = 5
V_FLOAT32 = 6
V_BOOL = 7
V_STRING = 8
V_ARRAY = 9

ROPE_BASE = 10000.0
RMS_EPS = 1e-5

# tinylm cmd_chat(): "<|user|>\n%s\n<|assistant|>\n" for tmpl==2 (chatml).
# The markers are ordinary text to this BPE, not special tokens.
CHATML_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role'] == 'user' %}<|user|>\n{{ m['content'] }}\n"
    "{% elif m['role'] == 'assistant' %}<|assistant|>\n{{ m['content'] }}\n"
    "{% else %}{{ m['content'] }}\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)
ALPACA_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role'] == 'user' %}### Instruction:\n{{ m['content'] }}\n\n"
    "{% else %}### Response:\n{{ m['content'] }}\n\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}### Response:\n{% endif %}"
)


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class GGUFWriter:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.kv = []            # (key, vtype, payload bytes)
        self.tensors = []       # (name, dims, ggml_type, producer)

    # ---- key/value metadata -------------------------------------------------
    @staticmethod
    def _str(s):
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def kv_str(self, key, val):
        self.kv.append((key, V_STRING, self._str(val)))

    def kv_bool(self, key, val):
        self.kv.append((key, V_BOOL, struct.pack("<?", bool(val))))

    def kv_uint32(self, key, val):
        self.kv.append((key, V_UINT32, struct.pack("<I", val)))

    def kv_float32(self, key, val):
        self.kv.append((key, V_FLOAT32, struct.pack("<f", val)))

    def kv_array_int32(self, key, vals):
        p = struct.pack("<IQ", V_INT32, len(vals)) + np.asarray(vals, "<i4").tobytes()
        self.kv.append((key, V_ARRAY, p))

    def kv_array_str(self, key, vals):
        parts = [struct.pack("<IQ", V_STRING, len(vals))]
        parts += [self._str(s) for s in vals]
        self.kv.append((key, V_ARRAY, b"".join(parts)))

    # ---- tensors ------------------------------------------------------------
    def add_tensor(self, name, dims, ggml_type, producer):
        """producer() -> np.ndarray of exactly prod(dims) values, in ggml order.

        1-D tensors are forced to F32: the RMSNorm gains reach ggml_mul, whose
        CPU kernel rejects an f16 second operand ("binary_op: unsupported types").
        The MoE router stays F32 too -- it decides an argmax, and llama.cpp never
        quantizes ffn_gate_inp either.
        """
        if len(dims) == 1 or name.endswith("ffn_gate_inp.weight"):
            ggml_type = GGML_TYPE_F32
        elif ggml_type == GGML_TYPE_Q8_0 and dims[0] % QK8_0:
            # ggml blocks run along ne[0]; LS-220M has n_ff_exp=707, so its
            # ffn_down_* rows are not a whole number of blocks.
            ggml_type = GGML_TYPE_F16
        self.tensors.append((name, dims, ggml_type, producer))

    def add_raw(self, name, dims, ggml_type, producer):
        """Append a tensor with no llama.cpp-oriented coercion (tinylm target).

        The 1-D-forces-F32 / Q8_0-fallback logic in add_tensor exists only to
        satisfy stock ggml kernels; the tinylm loader reads these tensors back
        verbatim, so a flat 1-D param body may stay F16 and a byte blob stays I8.
        """
        self.tensors.append((name, dims, ggml_type, producer))

    def write(self):
        f = self.f
        f.write(b"GGUF")
        f.write(struct.pack("<IQQ", 3, len(self.tensors), len(self.kv)))
        for key, vtype, payload in self.kv:
            f.write(self._str(key))
            f.write(struct.pack("<I", vtype))
            f.write(payload)
        off = 0
        for name, dims, ttype, _ in self.tensors:
            f.write(self._str(name))
            f.write(struct.pack("<I", len(dims)))
            f.write(struct.pack("<%dQ" % len(dims), *dims))
            f.write(struct.pack("<IQ", ttype, off))
            nbytes = type_nbytes(ttype, int(np.prod(dims)))
            off += (nbytes + GGUF_ALIGNMENT - 1) & ~(GGUF_ALIGNMENT - 1)
        for name, dims, ttype, producer in self.tensors:
            while f.tell() % GGUF_ALIGNMENT:
                f.write(b"\x00")
            a = producer()
            assert a.size == int(np.prod(dims)), f"{name}: {a.size} != {np.prod(dims)}"
            if ttype == GGML_TYPE_Q8_0:
                f.write(encode_q8_0(a))
            elif ttype == GGML_TYPE_I8:
                f.write(np.ascontiguousarray(a, dtype=np.uint8).tobytes())
            else:
                f.write(np.ascontiguousarray(
                    a, dtype=np.float16 if ttype == GGML_TYPE_F16 else np.float32).tobytes())
        f.close()


class ParamReader:
    """Random access into the flat parameter body, dequantizing on demand.

    qtype: 0 = fp16 params, 1 = q8 (fp16 scale + `group` int8 per block),
    else fp32. read(start, count) returns float32 values in the flat checkpoint
    order (the [in][out] row-major layout of the C map_weights()).
    """

    def __init__(self, data, base_off, n, qtype, group):
        self.data, self.base, self.n = data, base_off, n
        self.qtype, self.group = qtype, group

    def read(self, start, count):
        end = start + count
        assert end <= self.n, "param read past end of body"
        if self.qtype == 1:
            G = self.group
            b0, b1 = start // G, (end - 1) // G + 1
            blk = np.frombuffer(self.data, np.uint8, (b1 - b0) * (2 + G),
                                self.base + b0 * (2 + G)).reshape(-1, 2 + G)
            scale = blk[:, :2].copy().view(np.float16).astype(np.float32)
            vals = (blk[:, 2:].view(np.int8).astype(np.float32) * scale).reshape(-1)
            return vals[start - b0 * G:end - b0 * G]
        if self.qtype == 0:
            return np.frombuffer(self.data, np.float16, count,
                                 self.base + 2 * start).astype(np.float32)
        return np.frombuffer(self.data, np.float32, count, self.base + 4 * start)


def parse_checkpoint(path):
    data = open(path, "rb").read()
    magic = data[:4]
    # nh (header int count) per version: TLM6/TLQ3 add n_shared+qknorm (15),
    # TLM7/TLQ4 add topk (16). Quantized mirrors carry the same header, then
    # two extra ints (qtype, group) before the body.
    QVER = {b"TLQ1": 4, b"TLQ2": 5, b"TLQ3": 6, b"TLQ4": 7}
    MVER = {b"TLM1": 1, b"TLM2": 2, b"TLM3": 3, b"TLM4": 4,
            b"TLM5": 5, b"TLM6": 6, b"TLM7": 7}
    NH = {1: 8, 2: 9, 3: 10, 4: 11, 5: 13, 6: 15, 7: 16}
    if magic in QVER:
        ver = QVER[magic]
        nh = NH[ver]
        hdr = struct.unpack_from("<%di" % nh, data, 4)
        qtype, group = struct.unpack_from("<2i", data, 4 + 4 * nh)
        body_off = 4 + 4 * nh + 8
    elif magic in MVER:
        ver = MVER[magic]
        nh = NH[ver]
        hdr = struct.unpack_from("<%di" % nh, data, 4)
        qtype, group = -1, 0
        body_off = 4 + 4 * nh
    else:
        sys.exit(f"{path}: not a tinylm model (magic {magic!r})")

    V, D, L, H, KV, F, T, eot = hdr[0:8]
    cfg = dict(V=V, D=D, L=L, H=H, KV=KV, F=F, T=T, eot=eot,
               tmpl=hdr[8] if ver >= 2 else 0,
               n_mtp=hdr[9] if ver >= 3 else 0,
               n_exp=hdr[10] if ver >= 4 else 0,
               win=hdr[11] if ver >= 5 else 0,
               full_every=hdr[12] if ver >= 5 else 0)
    cfg["hd"] = D // H
    cfg["KD"] = KV * cfg["hd"]
    cfg["E"] = cfg["n_exp"] if cfg["n_exp"] > 0 else 1
    # cudalm `quantize q4n` writes qtype 3 under TLQ1/TLQ2 (tinylm's own qtype 3,
    # selective q4, only exists under TLQ3/TLQ4). GGUF code 4 = q4n, unambiguous.
    if qtype == 3 and ver <= 5:
        qtype = 4
    cfg["ver"], cfg["qtype"], cfg["group"] = ver, qtype, group
    cfg["n_shared"] = hdr[13] if ver >= 6 else 0
    cfg["qknorm"] = hdr[14] if ver >= 6 else 0
    # MoE top-K routing: baked at hdr[15] in TLM7/TLQ4 (0 = not recorded).
    # CUDALM_TOPK overrides, mirroring moe_topk() in tinylm.c.
    topk = hdr[15] if ver >= 7 else 0
    env_k = os.environ.get("CUDALM_TOPK")
    if env_k and env_k.isdigit() and int(env_k) > 0:
        topk = int(env_k)
    cfg["topk"] = topk

    E, n_exp, KD = cfg["E"], cfg["n_exp"], cfg["KD"]
    per_layer = (D + D * D + D * KD + D * KD + D * D + D
                 + (D * n_exp if n_exp > 0 else 0) + E * (2 * D * F + F * D))
    n = V * D + L * per_layer + D + cfg["n_mtp"] * D * D
    cfg["n_params"] = n

    pr = ParamReader(data, body_off, n, qtype, group)
    if qtype >= 2:
        # Full-q4 (2) and selective-q4 (3) have variable per-tensor layouts we
        # don't replicate here; the body is copied verbatim into the tinylm GGUF
        # regardless. Locate its end via the TLTK tokenizer blob (ver>=2 always
        # writes `int32 len` + "TLTK..." immediately after the body).
        # "TLTK" can occur inside nibble data: accept only a match whose int32
        # length prefix runs exactly to EOF.
        tpos = data.find(b"TLTK", body_off + 4)
        while tpos >= 0 and struct.unpack_from("<i", data, tpos - 4)[0] != len(data) - tpos:
            tpos = data.find(b"TLTK", tpos + 1)
        if tpos < 0:
            sys.exit(f"{path}: cannot locate tokenizer after qtype={qtype} body")
        pos = tpos - 4
    else:
        pos = body_off + (n * 2 if qtype == 0 else n * 4 if qtype < 0 else
                          ((n + group - 1) // group) * (2 + group))
    cfg["body_off"], cfg["body_bytes"] = body_off, pos - body_off
    tk = None
    tltk = None       # raw TLTK blob bytes, carried verbatim into the tinylm GGUF
    if ver >= 2 and pos + 4 <= len(data):
        tl = struct.unpack_from("<i", data, pos)[0]
        if tl > 0 and pos + 4 + tl <= len(data):
            tltk = data[pos + 4:pos + 4 + tl]
            tk = parse_tokenizer(tltk)
    return cfg, pr, tk, tltk


def parse_tokenizer(blob):
    if blob[:4] != b"TLTK":
        sys.exit("model has no TLTK tokenizer blob")
    V, eot, nmerges = struct.unpack_from("<3i", blob, 4)
    p = 16
    dec = []
    for _ in range(V):
        l = struct.unpack_from("<i", blob, p)[0]
        p += 4
        dec.append(blob[p:p + l])
        p += l
    base = list(struct.unpack_from("<256i", blob, p))
    p += 1024
    merges = [struct.unpack_from("<3i", blob, p + 12 * i) for i in range(nmerges)]
    return {"V": V, "eot": eot, "nmerges": nmerges, "dec": dec,
            "base": base, "merges": merges}


def build_tokenizer_kv(tk):
    b2u = bytes_to_unicode()
    tokens, types = [], []
    for i in range(tk["V"]):
        if i == tk["eot"]:
            tokens.append("<|endoftext|>")
            types.append(3)  # CONTROL
        else:
            tokens.append("".join(b2u[b] for b in tk["dec"][i]))
            types.append(1)  # NORMAL
    merges = [f"{tokens[a]} {tokens[b]}" for a, b, _ in tk["merges"]]
    return tokens, types, merges


# ---------------------------------------------------------------------------
# weight reshaping
#
# tinylm stores every linear weight row-major [in][out] so Y = X@W is a plain
# matmul. ggml wants ne = [n_in, n_out] with n_in contiguous, i.e. [out][in] in
# memory -- so every 2-D weight needs a transpose. (The old converter shipped
# attn_output untransposed, which alone was enough to produce garbage.)
# ---------------------------------------------------------------------------

def transpose_2d(src, in_rows, out_cols):
    return np.ascontiguousarray(src.reshape(in_rows, out_cols).T)


def transpose_exps(src, E, A, B):
    return np.ascontiguousarray(src.reshape(E, A, B).transpose(0, 2, 1))


def _pad_axis(a, axis, width):
    """Widen one axis to `width` with zeros.

    Used to round n_ff_exp up to a whole number of quantization blocks. Exact,
    not an approximation: the extra hidden units get zero columns in gate/up, so
    silu(0)*0 = 0, and zero rows in down, so they contribute nothing downstream.
    """
    if a.shape[axis] == width:
        return np.ascontiguousarray(a)
    shape = list(a.shape)
    shape[axis] = width
    out = np.zeros(shape, np.float32)
    out[(slice(None),) * axis + (slice(0, a.shape[axis]),)] = a
    return out


# The four FFN reshapes, each producing ggml's [out][in] order with n_ff padded
# to FP. Experts are the leading (slowest) axis in ggml's *_exps tensors.
def ffn_in_exps(src, E, D, F, FP):      # gate/up: [E][D][F] -> [E][FP][D]
    return _pad_axis(src.reshape(E, D, F).transpose(0, 2, 1), 1, FP)


def ffn_down_exps(src, E, F, D, FP):    # down:    [E][F][D] -> [E][D][FP]
    return _pad_axis(src.reshape(E, F, D).transpose(0, 2, 1), 2, FP)


def ffn_in_dense(src, D, F, FP):        # gate/up: [D][F] -> [FP][D]
    return _pad_axis(src.reshape(D, F).T, 0, FP)


def ffn_down_dense(src, F, D, FP):      # down:    [F][D] -> [D][FP]
    return _pad_axis(src.reshape(F, D).T, 1, FP)


def rope_permute(src, in_rows, out_cols, hd):
    """Rewrite a q/k projection so llama.cpp's INTERLEAVED rope reproduces
    tinylm's PAIRED rope, then transpose to ggml order.

    tinylm rotates (v[p], v[p+hd/2]) by theta_p. ggml's NORM rope rotates
    (v[2p], v[2p+1]) by the same theta_p. Placing model column p at ggml column
    2p and model column p+half at ggml column 2p+1 makes the two identical.
    Applied within each head's hd columns. Only needed for archs whose rope type
    is NORM (llama, deepseek); NEOX archs match tinylm directly.
    """
    a = src.reshape(in_rows, out_cols)
    half = hd // 2
    within = np.arange(out_cols) % hd
    perm = (np.arange(out_cols) // hd) * hd + (within % 2) * half + within // 2
    return np.ascontiguousarray(a[:, perm].T)


# tinylm suffixes describing how the *checkpoint* was stored. They say nothing
# about the GGUF's own tensor type, and "-q8" in particular collides with the
# GGUF quant tag, so strip them before naming the output.
SRC_TAGS = ("-q8", "-fp16", "-f16", "-f32")


def base_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    for tag in GGUF_TAGS.values():
        if stem.upper().endswith("-" + tag):
            stem = stem[:-len(tag) - 1]
            break
    return stem


def size_label(n):
    return f"{n/1e9:.1f}B" if n >= 1e9 else f"{round(n/1e6)}M"


# The tag Hugging Face parses out of the filename to label the quantization.
# It matches GGMLFileQuantizationType names, so "-q8" does not count -- it has
# to be exactly "Q8_0" / "F16" / "F32".
GGUF_TAGS = {"q8_0": "Q8_0", "f16": "F16", "f32": "F32"}


def default_out(model_path, out_type, kind=""):
    stem = os.path.splitext(os.path.basename(model_path))[0]
    for tag in SRC_TAGS:
        if stem.lower().endswith(tag):
            stem = stem[:-len(tag)]
            break
    return os.path.join(os.path.dirname(model_path),
                        f"{stem}{kind}-{GGUF_TAGS[out_type]}.gguf")


ARCHS = {
    # name: (rope type is NORM -> needs permute, exact gate, needs shared expert)
    "deepseek": dict(permute=True, exact=True, shexp=True),
    "llama": dict(permute=True, exact=False, shexp=False),
}


def convert(cfg, pr, tk, out_path, out_type, arch, ctx):
    V, D, L, H, KV, F = cfg["V"], cfg["D"], cfg["L"], cfg["H"], cfg["KV"], cfg["F"]
    hd, KD, E, n_exp = cfg["hd"], cfg["KD"], cfg["E"], cfg["n_exp"]
    spec = ARCHS[arch]
    ggml = {"f32": GGML_TYPE_F32, "f16": GGML_TYPE_F16,
            "q8_0": GGML_TYPE_Q8_0}[out_type]
    # ggml quantization blocks run along ne[0], which for ffn_down_* is n_ff_exp.
    # LS-220M has n_ff_exp=707, so without this its largest tensors would fall
    # back to f16 and the file would be ~40% bigger than the checkpoint.
    FP = F
    if ggml == GGML_TYPE_Q8_0 and F % QK8_0:
        FP = (F + QK8_0 - 1) // QK8_0 * QK8_0
        print(f"[bin2gguf] padding n_ff_exp {F} -> {FP} (a whole number of "
              f"{QK8_0}-value blocks); the added units are zeroed, so the "
              f"result is unchanged")
    g = GGUFWriter(out_path)

    if n_exp <= 0 and arch == "deepseek":
        sys.exit("--arch deepseek is for MoE checkpoints; use --arch llama for dense")

    # ---- metadata ----------------------------------------------------------
    g.kv_str("general.architecture", arch)
    g.kv_str("general.type", "model")
    g.kv_str("general.name", os.path.splitext(os.path.basename(out_path))[0])
    g.kv_str("general.basename", base_name(out_path))
    g.kv_str("general.size_label", size_label(cfg["n_params"]))
    g.kv_uint32("general.file_type",
                {"f32": 0, "f16": 1, "q8_0": 7}[out_type])  # llama_ftype
    # GGML_QNT_VERSION. llama.cpp writes this on every file it produces, and
    # Hugging Face's GGUF viewer expects it -- without it the upload shows no
    # quantization at all.
    g.kv_uint32("general.quantization_version", 2)
    g.kv_uint32(f"{arch}.context_length", ctx)
    g.kv_uint32(f"{arch}.embedding_length", D)
    g.kv_uint32(f"{arch}.block_count", L)
    g.kv_uint32(f"{arch}.feed_forward_length", FP)
    g.kv_uint32(f"{arch}.attention.head_count", H)
    g.kv_uint32(f"{arch}.attention.head_count_kv", KV)
    g.kv_uint32(f"{arch}.attention.key_length", hd)
    g.kv_uint32(f"{arch}.attention.value_length", hd)
    g.kv_uint32(f"{arch}.vocab_size", V)
    g.kv_uint32(f"{arch}.rope.dimension_count", hd)
    g.kv_float32(f"{arch}.rope.freq_base", ROPE_BASE)
    g.kv_float32(f"{arch}.attention.layer_norm_rms_epsilon", RMS_EPS)
    if n_exp > 0:
        g.kv_uint32(f"{arch}.expert_count", E)
        g.kv_uint32(f"{arch}.expert_used_count", 1)
    if arch == "deepseek":
        # required by llama_model_deepseek::load_arch_hparams
        g.kv_uint32("deepseek.expert_feed_forward_length", FP)
        g.kv_uint32("deepseek.expert_shared_count", 1)
        g.kv_uint32("deepseek.leading_dense_block_count", 0)
        # expert_weights_scale deliberately omitted: it defaults to 0.0, and
        # build_moe_ffn only applies a scale when it is neither 0 nor 1.

    if tk is None:
        sys.exit("model has no tokenizer to write; GGUF requires one (pack it first)")
    tokens, types, merges = build_tokenizer_kv(tk)
    g.kv_str("tokenizer.ggml.model", "gpt2")
    g.kv_str("tokenizer.ggml.pre", "gpt-2")
    g.kv_array_str("tokenizer.ggml.tokens", tokens)
    g.kv_array_int32("tokenizer.ggml.token_type", types)
    g.kv_array_str("tokenizer.ggml.merges", merges)
    g.kv_uint32("tokenizer.ggml.eos_token_id", tk["eot"])
    g.kv_uint32("tokenizer.ggml.bos_token_id", tk["eot"])
    g.kv_uint32("tokenizer.ggml.unknown_token_id", tk["eot"])
    g.kv_bool("tokenizer.ggml.add_bos_token", False)
    g.kv_bool("tokenizer.ggml.add_eos_token", False)
    if cfg["tmpl"] == 2:
        g.kv_str("tokenizer.chat_template", CHATML_TEMPLATE)
    elif cfg["tmpl"] == 1:
        g.kv_str("tokenizer.chat_template", ALPACA_TEMPLATE)

    # ---- tensors -----------------------------------------------------------
    def rd(start, count):
        return lambda: pr.read(start, count)

    off = 0
    emb = off
    off += V * D
    # ggml ne=[D,V], memory [V][D] token-major -- tinylm's layout already.
    # output.weight is omitted: both archs fall back to token_embd (tied).
    g.add_tensor("token_embd.weight", [D, V], ggml, rd(emb, V * D))

    zeros = {}

    def zero(n):
        if n not in zeros:
            zeros[n] = np.zeros(n, np.float32)
        return lambda: zeros[n]

    for l in range(L):
        def T(name, dims, start, fn):
            g.add_tensor(f"blk.{l}.{name}", dims, ggml, fn)

        an1 = off; off += D
        wq = off; off += D * D
        wk = off; off += D * KD
        wv = off; off += D * KD
        wo = off; off += D * D
        an2 = off; off += D
        wr = None
        if n_exp > 0:
            wr = off; off += D * n_exp
        w1 = off; off += E * D * F
        w3 = off; off += E * D * F
        w2 = off; off += E * F * D

        perm = spec["permute"]
        T("attn_norm.weight", [D], an1, rd(an1, D))
        T("attn_q.weight", [D, D], wq,
          (lambda s=wq: rope_permute(pr.read(s, D * D), D, D, hd)) if perm
          else (lambda s=wq: transpose_2d(pr.read(s, D * D), D, D)))
        T("attn_k.weight", [D, KD], wk,
          (lambda s=wk: rope_permute(pr.read(s, D * KD), D, KD, hd)) if perm
          else (lambda s=wk: transpose_2d(pr.read(s, D * KD), D, KD)))
        T("attn_v.weight", [D, KD], wv, lambda s=wv: transpose_2d(pr.read(s, D * KD), D, KD))
        T("attn_output.weight", [D, D], wo, lambda s=wo: transpose_2d(pr.read(s, D * D), D, D))
        T("ffn_norm.weight", [D], an2, rd(an2, D))
        if n_exp > 0:
            T("ffn_gate_inp.weight", [D, E], wr, lambda s=wr: transpose_2d(pr.read(s, D * E), D, E))
            T("ffn_gate_exps.weight", [D, FP, E], w1,
              lambda s=w1: ffn_in_exps(pr.read(s, E * D * F), E, D, F, FP))
            T("ffn_up_exps.weight", [D, FP, E], w3,
              lambda s=w3: ffn_in_exps(pr.read(s, E * D * F), E, D, F, FP))
            T("ffn_down_exps.weight", [FP, D, E], w2,
              lambda s=w2: ffn_down_exps(pr.read(s, E * F * D), E, F, D, FP))
            if spec["shexp"]:
                # Dummy shared expert. down_shexp is all zeros, so the whole
                # branch contributes exactly 0.0 and the deepseek graph reduces
                # to tinylm's `gate * Expert_best(x)`. up/gate are zeroed too --
                # deepseek adds this branch with a plain ggml_add (no division),
                # so there is no NaN path here.
                T("ffn_up_shexp.weight", [D, FP], 0, zero(D * FP))
                T("ffn_gate_shexp.weight", [D, FP], 0, zero(D * FP))
                T("ffn_down_shexp.weight", [FP, D], 0, zero(FP * D))
        else:
            T("ffn_gate.weight", [D, FP], w1,
              lambda s=w1: ffn_in_dense(pr.read(s, D * F), D, F, FP))
            T("ffn_up.weight", [D, FP], w3,
              lambda s=w3: ffn_in_dense(pr.read(s, D * F), D, F, FP))
            T("ffn_down.weight", [FP, D], w2,
              lambda s=w2: ffn_down_dense(pr.read(s, F * D), F, D, FP))

    nf = off; off += D
    g.add_tensor("output_norm.weight", [D], ggml, rd(nf, D))
    if cfg["n_mtp"] > 0:
        print(f"[bin2gguf] warning: dropping {cfg['n_mtp']} MTP heads (no GGUF counterpart)")
    assert off + cfg["n_mtp"] * D * D == cfg["n_params"], "param layout mismatch"

    g.write()
    return len(g.tensors)


# ---------------------------------------------------------------------------
# tinylm-native target
#
# A faithful round-trip GGUF that tinylm's own C loader reads back with a
# near-memcpy: the whole flat parameter body as one tensor in exact map_weights
# order, the raw TLTK tokenizer blob, and the n-gram sidecar (Wng + q4 digits +
# per-group scales). No transpose / rope_permute / FFN padding -- those exist
# only to satisfy stock llama.cpp and are pure loss/complexity for tinylm.
# ---------------------------------------------------------------------------

def _splitmix64(x):
    M = (1 << 64) - 1
    x = (x + 0x9e3779b97f4a7c15) & M
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & M
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & M
    return x ^ (x >> 31)


def _isprime(p):
    if p < 2:
        return False
    d = 2
    while d * d <= p:
        if p % d == 0:
            return False
        d += 1
    return True


def eg_scheme(h):
    """NGR2 hash scheme from the header: (S, Rp, primes, multipliers) exactly as cudalm17."""
    H, no, De, R = h[0], h[1], h[6], h[7]
    S = H * no
    al = 128 // De if 128 % De == 0 else 1
    Rp = (R + al - 1) // al * al
    primes, p = [], R + 1
    for _ in range(S):
        p -= 1
        while p > 1 and not _isprime(p):
            p -= 1
        primes.append(p)
    mults = [_splitmix64(0x5EED0E17A11 + i) | 1 for i in range(S * 4)]
    return S, Rp, primes, mults


def q4_pack(x, group=128):
    """fp32 -> (packed two's-complement nibbles, per-group absmax/7 scales), cudalm layout."""
    n = x.size
    xp = np.zeros((n + group - 1) // group * group, np.float32)
    xp[:n] = x
    g = xp.reshape(-1, group)
    sc = np.maximum(np.abs(g).max(1) / 7.0, 1e-12).astype(np.float32)
    d = np.clip(np.rint(g / sc[:, None]), -7, 7).astype(np.int8).reshape(-1)[:n]
    if n & 1:
        d = np.append(d, np.int8(0))
    u = (d.astype(np.uint8) & 0xF)
    return (u[0::2] | (u[1::2] << 4)).astype(np.uint8), sc


def load_engram(path, data, D, V, L, f32_table=False):
    """cudalm17 NGR2 sidecar -> dict(hdr, hash, canon, w, digits+scales | table).
    Refuses anything it cannot reproduce exactly (header, hash scheme, size)."""
    h = list(struct.unpack_from("<24i", data, 4))
    H, no, De, R, q4, np_, conv, canon, gate = h[0], h[1], h[6], h[7], h[8], h[10], h[19], h[20], h[21]
    orders, lays = h[2:2 + no], h[11:11 + np_]
    ok = (H >= 1 and 1 <= no <= 4 and H * no <= 32 and 1 <= De <= 1024 and R >= 2 and 1 <= np_ <= 8
          and q4 in (0, 1) and conv in (0, 4) and canon in (0, 1) and gate in (0, 1) and h[22] == D
          and all(1 <= o <= 4 for o in orders) and all(0 <= l < L for l in lays)
          and lays == sorted(set(lays)))
    if not ok:
        sys.exit(f"{path}: unsupported NGR2 header {h} (model D={D} L={L})")
    S, Rp, primes, mults = eg_scheme(h)
    off = 4 + 96
    fp = list(struct.unpack_from(f"<{S}i", data, off)); off += 4 * S
    fm = list(struct.unpack_from(f"<{S * 4}Q", data, off)); off += 32 * S
    if fp != primes or fm != mults:
        sys.exit(f"{path}: hash primes/multipliers differ from the NGR2 scheme")
    hash_blob = np.frombuffer(data, np.uint8, 36 * S, 4 + 96).copy()
    cn = None
    if canon:
        cn = np.frombuffer(data, np.uint8, 2 * V, off).copy(); off += 2 * V
    nw, ws = S * De * D, []
    def take(n):
        nonlocal off
        a = np.frombuffer(data, np.float32, n, off).copy()
        off += n * 8                      # fp32 value + bf16 M + bf16 V
        return a
    for _ in range(np_):
        ws.append(take(nw))
        if gate:
            ws.append(take(nw))
        if conv:
            ws.append(take(D)); ws.append(take(4 * D))
    nt = S * Rp * De
    ngrp = (nt + 127) // 128
    out = dict(hdr=h, hash=hash_blob, canon=cn, w=np.concatenate(ws))
    if q4:
        off += 4                          # sref (training only)
        out["digits"] = np.frombuffer(data, np.uint8, (nt + 1) // 2, off).copy()
        off += (nt + 1) // 2 + 3 * nt     # digits, residual, m8, v8
        out["scales"] = np.frombuffer(data, np.float32, ngrp, off).copy()
        off += 12 * ngrp                  # s, ms, vs
    else:
        tbl = np.frombuffer(data, np.float32, nt, off).copy()
        off += 8 * nt
        if f32_table:
            out["table"] = tbl
        else:
            out["digits"], out["scales"] = q4_pack(tbl)
            h[8] = 1
    if off != len(data):
        sys.exit(f"{path}: NGR2 size mismatch ({len(data)} bytes, layout needs {off})")
    tb = out.get("table")
    print(f"[bin2gguf] engram: S={S} (H={H} x orders {orders}) De={De} R={R} points={lays} "
          f"gate={gate} conv={conv} canon={canon} table={'fp32' if tb is not None else 'q4'}"
          f"{' (quantized from fp32)' if not q4 and tb is None else ''} iter {h[23]}")
    return out


def load_ngram(model_path, D, V=0, L=0, f32_table=False):
    """Parse the `<model>.opt.ng` NGR1 sidecar, returning the inference-only
    data tinylm needs: (meta, Wng[nw f32], digits[nd u8], scales[ngrp f32]).

    Mirrors ngram_load() in tinylm.c, including the SLIM/FULL layout split.
    Returns None when the sidecar is absent.
    """
    path = model_path + ".opt.ng"
    if not os.path.exists(path):
        return None
    data = open(path, "rb").read()
    if data[:4] == b"NGR2":
        return load_engram(path, data, D, V, L, f32_table)
    if data[:4] != b"NGR1":
        sys.exit(f"{path}: unknown ngram magic {data[:4]!r} (need NGR1/NGR2)")
    R, De, order, q4, _iter = struct.unpack_from("<5i", data, 4)
    if not (0 < R < (1 << 24) and 0 < De < 1024 and order == 2 and q4 == 1 and De % 2 == 0):
        sys.exit(f"{path}: unsupported ngram header R={R} De={De} order={order} q4={q4}")
    nw = De * D
    nt = R * De
    nd = (nt + 1) // 2
    group = 128
    ngrp = (nt + 127) // 128
    ng4 = ngrp * 4

    off = 4 + 20                          # magic + 5 ints
    wng = np.frombuffer(data, np.float32, nw, off).copy()
    off += nw * 4

    left = len(data) - off
    slim_left = nd + ng4 + 4
    if left == slim_left:                 # SLIM: digits, scales, sref
        digits = np.frombuffer(data, np.uint8, nd, off).copy()
        off += nd
        scales = np.frombuffer(data, np.float32, ngrp, off).copy()
    else:                                 # FULL: WM/WV bf16 + sref + digits + resid+m8+v8 + scales
        off += 4 * nw + 4
        digits = np.frombuffer(data, np.uint8, nd, off).copy()
        off += nd + 3 * nt
        scales = np.frombuffer(data, np.float32, ngrp, off).copy()

    meta = dict(R=R, De=De, order=order, group=group, ngrp=ngrp)
    print(f"[bin2gguf] ngram: R={R} De={De} order={order} "
          f"Wng={nw} digits={nd/1e6:.1f}MB scales={ngrp} ({'slim' if left == slim_left else 'full'})")
    return meta, wng, digits, scales


def load_rbias(model_path, cfg):
    """cudalm `<model>.opt.t1` router bias [L][S][E] (selection-only); None if absent."""
    path = model_path + ".opt.t1"
    if not os.path.exists(path) or cfg["n_exp"] <= 0:
        return None
    data = open(path, "rb").read()
    if data[:4] != b"T1S1":
        sys.exit(f"{path}: bad t1 magic {data[:4]!r}")
    attg, L, D, gate, De, mb, tri, layer, it = struct.unpack_from("<9i", data, 4)
    if mb <= 0 or L != cfg["L"] or mb % cfg["n_exp"]:
        return None
    off = 40 + (L * D * D * 8 if attg else 0) + (De * D * 8 if gate else 0)
    b = np.frombuffer(data, np.float32, L * mb, off).copy()
    S = mb // cfg["n_exp"]
    print(f"[bin2gguf] router bias: S={S} (iter {it})")
    # The bias has one row per routing slot, so S is the trained top-K.
    if cfg.get("topk", 0) <= 0 and S > 1:
        cfg["topk"] = S
    elif cfg.get("topk", 0) > 0 and cfg["topk"] != S:
        print(f"[bin2gguf] WARNING: topk={cfg['topk']} but router bias has S={S} slots")
    return b


def convert_tinylm(cfg, pr, tltk, ngram, out_path, out_type, rbias=None):
    """Write a tinylm-native GGUF: one flat param tensor + raw TLTK blob +
    optional ngram tensors + a `tinylm.cfg` int32 header array."""
    n = cfg["n_params"]
    ggml = GGML_TYPE_F16 if out_type == "f16" else GGML_TYPE_F32
    g = GGUFWriter(out_path)

    g.kv_str("general.architecture", "tinylm")
    g.kv_str("general.name", os.path.splitext(os.path.basename(out_path))[0])
    g.kv_str("general.size_label", size_label(n))
    # The 15 ints of a TLM6 header, in header order. TLM4/TLQ1 lack the last two
    # (n_shared, qknorm); default them to 0 so the C loader synthesizes a valid
    # TLM6 config via cfg_from_hdr().
    g.kv_array_int32("tinylm.cfg", [
        cfg["V"], cfg["D"], cfg["L"], cfg["H"], cfg["KV"], cfg["F"], cfg["T"],
        cfg["eot"], cfg["tmpl"], cfg["n_mtp"], cfg["n_exp"],
        cfg["win"], cfg["full_every"], cfg.get("n_shared", 0), cfg.get("qknorm", 0),
    ])
    g.kv_uint32("tinylm.n_params", n)

    # Quantized checkpoint (TLQ*): preserve the raw quant body verbatim so the C
    # loader dequantizes it exactly as model_load_full() does and runs the same
    # int8 kernels -- bit-identical to the .bin, not a dequantized F32 blob.
    is_quant = cfg["qtype"] >= 0
    if is_quant:
        g.kv_uint32("tinylm.qtype", cfg["qtype"])
        g.kv_uint32("tinylm.group", cfg["group"])

    # Self-describe MoE top-K routing so the C loader needs no CUDALM_TOPK env.
    if cfg.get("topk", 0) > 0:
        g.kv_uint32("tinylm.topk", cfg["topk"])

    if tltk is None:
        sys.exit("model has no TLTK tokenizer blob; cannot build a tinylm GGUF")
    g.kv_uint32("tinylm.tok_len", len(tltk))

    eg = ngram if isinstance(ngram, dict) else None
    if eg:
        ngram = None
        g.kv_array_int32("tinylm.engram", eg["hdr"])
    has_ng = ngram is not None
    g.kv_bool("tinylm.has_ngram", has_ng)
    if has_ng:
        meta, wng, digits, scales = ngram
        g.kv_array_int32("tinylm.ngram",
                         [meta["R"], meta["De"], meta["order"], meta["group"], meta["ngrp"]])

    # ---- tensors: flat param body, tokenizer blob, ngram ----
    if is_quant:
        nb = cfg["body_bytes"]
        qbody = np.frombuffer(pr.data, np.uint8, nb, pr.base)
        g.add_raw("tinylm.qparams", [nb], GGML_TYPE_I8, lambda: qbody)
    else:
        g.add_raw("tinylm.params", [n], ggml, lambda: pr.read(0, n))
    tok_arr = np.frombuffer(tltk, np.uint8)
    g.add_raw("tinylm.tokenizer", [len(tltk)], GGML_TYPE_I8, lambda: tok_arr)
    if has_ng:
        meta, wng, digits, scales = ngram
        g.add_raw("tinylm.ngram.wng", [wng.size], GGML_TYPE_F32, lambda: wng)
        g.add_raw("tinylm.ngram.digits", [digits.size], GGML_TYPE_I8, lambda: digits)
        g.add_raw("tinylm.ngram.scales", [scales.size], GGML_TYPE_F32, lambda: scales)
    if eg:
        for k, t in (("hash", GGML_TYPE_I8), ("canon", GGML_TYPE_I8), ("w", GGML_TYPE_F32),
                     ("digits", GGML_TYPE_I8), ("scales", GGML_TYPE_F32), ("table", GGML_TYPE_F32)):
            a = eg.get(k)
            if a is not None:
                g.add_raw(f"tinylm.engram.{k}", [a.size], t, lambda a=a: a)
    if rbias is not None:
        g.add_raw("tinylm.router_bias", [rbias.size], GGML_TYPE_F32, lambda: rbias)

    g.write()
    return len(g.tensors)


TYPE_ALIAS = {"f16": "fp16", "q8_0": "q8", "fp32": "f32"}
QTYPE = {"fp16": 0, "q8": 1, "q4": 2, "q4n": 4}            # cudalm quantize mode -> parsed qtype
QNAME = {v: k for k, v in QTYPE.items()} | {3: "q4s"}


def find_cudalm(path):
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (path, os.environ.get("CUDALM_EXE"), os.path.join(here, "..", "cudalm.exe"),
              os.path.join(here, "..", "cudalm"), shutil.which("cudalm")):
        if c and os.path.isfile(c):
            return c
    sys.exit("[bin2gguf] cudalm executable not found (needed for q4n/q4/q8); pass --cudalm PATH")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="tinylm .bin checkpoint (TLM1-5 or TLQ1-2)")
    ap.add_argument("out", nargs="?", default=None,
                    help="output .gguf (default: model path with .gguf)")
    ap.add_argument("--target", choices=["tinylm", "llama"], default="tinylm",
                    help="tinylm (default): a faithful GGUF the tinylm C loader "
                         "reads back exactly, including the n-gram. llama: a stock "
                         "llama.cpp-compatible GGUF (lossy for MoE / ngram).")
    ap.add_argument("--type", type=lambda t: TYPE_ALIAS.get(t.lower(), t.lower()),
                    choices=["q4n", "q4", "q8", "fp16", "f32"], default=None,
                    help="cudalm quantize modes. tinylm target: q4n/q4/q8 run "
                         "`cudalm quantize` on an fp32 model first (q4n is lossless "
                         "for QLM); fp16/f32 store floats; default = the source's own "
                         "precision. llama target: q8 (default, Q8_0), fp16, f32.")
    ap.add_argument("--cudalm", default=None,
                    help="cudalm executable for q4n/q4/q8 (default: CUDALM_EXE, "
                         "../cudalm.exe next to this script, or PATH)")
    ap.add_argument("--arch", choices=sorted(ARCHS), default=None,
                    help="llama.cpp architecture (default: deepseek for MoE, exact; llama for dense)")
    ap.add_argument("--ctx", type=int, default=0,
                    help="context length to advertise (default: the training "
                         "sliding window, beyond which llama.cpp diverges)")
    ap.add_argument("--ng-f32", action="store_true",
                    help="keep an fp32 engram table fp32 (default: export it as q4)")
    args = ap.parse_args()

    cfg, pr, tk, tltk = parse_checkpoint(args.model)
    src = QNAME.get(cfg["qtype"], f"qtype {cfg['qtype']}")

    if args.target == "tinylm":
        t, model, tmp = args.type, args.model, None
        if cfg["qtype"] >= 0 and t not in (None, src):
            sys.exit(f"[bin2gguf] {args.model} is already {src}; re-export from the fp32 model to get {t}")
        if cfg["qtype"] < 0 and t in QTYPE:
            base = args.out or os.path.join(os.path.dirname(args.model), base_name(args.model) + ".gguf")
            tmp = os.path.splitext(base)[0] + ".tmp.bin"
            exe = find_cudalm(args.cudalm)
            print(f"[bin2gguf] {exe} quantize -> {t}", flush=True)
            if subprocess.call([exe, "quantize", args.model, tmp, t]) != 0 or not os.path.exists(tmp):
                sys.exit("[bin2gguf] cudalm quantize failed")
            cfg, pr, tk, tltk = parse_checkpoint(tmp)
            model = tmp
        out_type = "f16" if t == "fp16" else "f32"
        try:
            out = tinylm_export(args, cfg, pr, tltk, model, out_type)
        finally:
            for f in ((tmp, tmp + ".opt.ng", tmp + ".opt.t1") if tmp else ()):
                if os.path.exists(f):
                    os.remove(f)
        size = os.path.getsize(out)
        print(f"[bin2gguf] {args.model} -> {out}  (target=tinylm, {QNAME.get(cfg['qtype'], 'f16' if out_type == 'f16' else 'f32')})")
        print(f"[bin2gguf] {cfg['n_params']:,} params, {size/1e6:.1f} MB")
        return

    if args.type in ("q4n", "q4"):
        sys.exit("[bin2gguf] --target llama has no exact 4-bit form (Q4_0 blocks run along the transposed "
                 "rows and cut across cudalm's quant groups); use --type q8, or --target tinylm for q4n/q4")
    if cfg["qtype"] >= 2:
        sys.exit(f"[bin2gguf] {src} sources are tinylm-target only; use --target tinylm or the fp32 model")
    args.type = {"q8": "q8_0", "fp16": "f16", "f32": "f32", None: "q8_0"}[args.type]
    llama_export(args, cfg, pr, tk)


def tinylm_export(args, cfg, pr, tltk, model, out_type):
    if cfg["qtype"] >= 0:
        # Quantized source: the body is preserved verbatim, so the F32/F16
        # tensor-type flag is moot. Tag the filename by the native quant.
        tag = {0: "F16", 1: "Q8_64", 2: "Q4", 3: "Q4S", 4: "Q4N"}.get(cfg["qtype"], "Q")
        stem = os.path.splitext(os.path.basename(args.model))[0]
        for t in SRC_TAGS + ("-q4n", "-q4"):
            if stem.lower().endswith(t):
                stem = stem[:-len(t)]
                break
        out = args.out or os.path.join(os.path.dirname(args.model),
                                       f"{stem}-{tag}.gguf")
    else:
        out = args.out or default_out(args.model, out_type)
    ngram = load_ngram(model, cfg["D"], cfg["V"], cfg["L"], args.ng_f32)
    rbias = load_rbias(model, cfg)
    if cfg["n_exp"] > 0:
        print(f"[bin2gguf] MoE top-K = {cfg['topk'] or '1 (not recorded)'}")
    convert_tinylm(cfg, pr, tltk, ngram, out, out_type, rbias)
    print(f"[bin2gguf] V={cfg['V']} D={cfg['D']} L={cfg['L']} H={cfg['H']} "
          f"KV={cfg['KV']} F={cfg['F']} experts={cfg['n_exp']} "
          f"ngram={('engram' if isinstance(ngram, dict) else 'yes') if ngram else 'no'}")
    return out


def llama_export(args, cfg, pr, tk):
    args.arch = args.arch or ("deepseek" if cfg["n_exp"] > 0 else "llama")
    if args.arch == "deepseek" and cfg["n_exp"] == 0:
        sys.exit("[bin2gguf] --arch deepseek is for MoE checkpoints; use --arch llama for dense")
    if cfg["topk"] > 1 and cfg["n_exp"] > 0:
        print(f"[bin2gguf] WARNING: model routes top-{cfg['topk']}; the llama target is top-1.")
    if os.path.exists(args.model + ".opt.ng"):
        print(f"[bin2gguf] WARNING: {args.model}.opt.ng (ngram/engram memory) has no llama.cpp "
              f"equivalent and is DROPPED; the output will not match tinylm. Use --target tinylm.")
    out = args.out or default_out(args.model, args.type, "-llama")

    win, T = cfg["win"], cfg["T"]
    ctx = args.ctx or (min(T, win) if win > 0 else T)
    if win > 0 and ctx > win:
        print(f"[bin2gguf] WARNING: ctx={ctx} exceeds the {win}-token training "
              f"window. llama.cpp has no sliding-window attention for "
              f"arch={args.arch}, so beyond {win} tokens it attends over a span "
              f"the model never saw in training and output will degrade.")
    if not ARCHS[args.arch]["exact"] and cfg["n_exp"] > 0:
        print(f"[bin2gguf] WARNING: arch={args.arch} normalizes the MoE router "
              f"weights, forcing the top-1 gate to 1.0. This is NOT equivalent "
              f"to tinylm and costs several points of perplexity. Use "
              f"--arch deepseek for an exact conversion.")

    nt = convert(cfg, pr, tk, out, args.type, args.arch, ctx)
    size = os.path.getsize(out)
    print(f"[bin2gguf] {args.model} -> {out}")
    print(f"[bin2gguf] arch={args.arch} V={cfg['V']} D={cfg['D']} L={cfg['L']} "
          f"H={cfg['H']} KV={cfg['KV']} F={cfg['F']} experts={cfg['n_exp']} "
          f"win={cfg['win'] or 'full'} ctx={ctx} {args.type}")
    print(f"[bin2gguf] {cfg['n_params']:,} params, {nt} tensors, {size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
