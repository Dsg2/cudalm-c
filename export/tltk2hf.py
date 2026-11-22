"""TLTK (cudalm tokenizer) -> Hugging Face tokenizer files.

    python tltk2hf.py <tokenizer.bin | model.bin> <out_dir> [--max-len N]

Writes tokenizer.json, tokenizer_config.json and chat_template.jinja. The encoding matches cudalm's tok_encode:
special tokens (<|endoftext|> and every vocab entry spelled <|name|>) are matched literally anywhere, the rest is
split with the GPT-2 ByteLevel regex and BPE-merged per piece in merge-rank order (first occurrence of a pair wins).
"""
import json, os, struct, sys


def b2u():
    bs = list(range(33, 127)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def tltk_blob(path):
    b = open(path, "rb").read()
    if b[:4] == b"TLTK":
        return b
    p = b.rfind(b"TLTK")                        # model .bin: ... [int len][TLTK blob] at the end
    while p >= 4:
        if struct.unpack_from("<i", b, p - 4)[0] == len(b) - p:
            return b[p:]
        p = b.rfind(b"TLTK", 0, p)
    sys.exit(f"{path}: no TLTK tokenizer found")


def is_special(t):                              # = tok_isspecial in cudalm.cu
    return 5 <= len(t) <= 32 and t[:2] == b"<|" and t[-2:] == b"|>" and \
        all(c in b"abcdefghijklmnopqrstuvwxyz0123456789_/" for c in t[2:-2])


def parse(blob):
    V, eot, nm = struct.unpack_from("<3i", blob, 4)
    p, toks = 16, []
    for _ in range(V):
        n, = struct.unpack_from("<i", blob, p); toks.append(blob[p + 4:p + 4 + n]); p += 4 + n
    base = struct.unpack_from("<256i", blob, p); p += 1024
    nm = min(nm, (len(blob) - p) // 12)
    merges = [struct.unpack_from("<3i", blob, p + 12 * i) for i in range(nm)]
    return V, eot, toks, base, merges


CHAT = ("{%- set R = {'system': 'system', 'developer': 'system', 'user': 'user', 'assistant': 'assistant', "
        "'tool': 'result', 'ipython': 'result'} -%}\n"
        "{%- for m in messages -%}\n"
        "{{- ('\\n' if not loop.first else '') + '<|' + R.get(m.role, m.role) + '|>\\n' + (m.content | trim) -}}\n"
        "{%- endfor -%}\n"
        "{%- if add_generation_prompt -%}\n{{- ('\\n' if messages else '') + '<|assistant|>\\n' -}}\n"
        "{%- elif messages and messages[-1].role == 'assistant' -%}\n{{- '<|endoftext|>' -}}\n{%- endif -%}\n")


def convert(blob, out, max_len=2048):
    V, eot, toks, base, merges = parse(blob)
    u = b2u()
    sp = [i for i in range(V) if i != eot and is_special(toks[i])]
    s = ["<|endoftext|>" if i == eot else toks[i].decode("latin-1") if i in sp else "".join(u[c] for c in toks[i])
         for i in range(V)]
    vocab, dup = {}, []
    for i, t in enumerate(s):
        (dup.append(i) if t in vocab else vocab.__setitem__(t, i))
    bad_base = [c for c in range(256) if base[c] < 0 or s[base[c]] != u[c] or vocab.get(u[c]) != base[c]]
    seen, ml, bad_m = set(), [], 0
    for a, c, nw in merges:
        if not (0 <= a < V and 0 <= c < V) or (a, c) in seen:
            continue
        seen.add((a, c))
        if not (0 <= nw < V) or toks[nw] != toks[a] + toks[c] or vocab.get(s[nw]) != nw:
            bad_m += 1
        ml.append([s[a], s[c]])
    added = [{"id": i, "content": s[i], "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": True} for i in sorted([eot] + sp)]
    bl = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}
    tj = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": added, "normalizer": None,
          "pre_tokenizer": bl, "post_processor": None, "decoder": dict(bl),
          "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                    "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                    "vocab": vocab, "merges": ml}}
    os.makedirs(out, exist_ok=True)
    json.dump(tj, open(os.path.join(out, "tokenizer.json"), "w", encoding="utf-8"), ensure_ascii=False)
    cfg = {"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": None, "eos_token": "<|endoftext|>",
           "pad_token": "<|endoftext|>", "unk_token": None, "add_bos_token": False, "add_eos_token": False,
           "clean_up_tokenization_spaces": False, "model_max_length": max_len,
           "extra_special_tokens": [s[i] for i in sp]}
    json.dump(cfg, open(os.path.join(out, "tokenizer_config.json"), "w"), indent=2)
    open(os.path.join(out, "chat_template.jinja"), "w", newline="\n").write(CHAT)
    rep = {"vocab": V, "eot": eot, "specials": len(sp), "merges": len(ml), "dup_strings": len(dup),
           "bad_base_bytes": len(bad_base), "bad_merges": bad_m}
    if dup or bad_base or bad_m:
        print("WARNING: HF BPE cannot reproduce this tokenizer exactly:", rep, file=sys.stderr)
    return rep


if __name__ == "__main__":
    a = sys.argv[1:]
    ml = 2048
    if "--max-len" in a:
        i = a.index("--max-len"); ml = int(a[i + 1]); del a[i:i + 2]
    if len(a) != 2:
        sys.exit(__doc__)
    print(json.dumps(convert(tltk_blob(a[0]), a[1], ml)))
