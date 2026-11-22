"""HF tokenizer check (H1).  hf_tokcheck.py <tokenizer.bin|model.bin> <hf_dir> <adv_out.u16> <val.bin[:maxtok]>... [--chat <val.bin>]...
   Then: cudalm tokcheck <model> <adv_out.u16>  (re-encodes the adversarial set with tok_encode; 0 mismatched docs = HF == cudalm)"""
import random, re, sys, time
from array import array
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import tltk2hf
from tokenizers import Tokenizer

a = sys.argv[1:]
chats = [a[i + 1] for i, x in enumerate(a) if x == "--chat"]
a = [x for i, x in enumerate(a) if x != "--chat" and (i == 0 or a[i - 1] != "--chat")]
tokbin, hf, advout, vals = a[0], a[1], a[2], a[3:]
print("convert:", tltk2hf.convert(tltk2hf.tltk_blob(tokbin), hf))
V, eot, toks, _, _ = tltk2hf.parse(tltk2hf.tltk_blob(tokbin))
tk = Tokenizer.from_file(hf + "\\tokenizer.json")
fails = 0


def docs(path, maxtok):
    import os
    t = array("H"); t.fromfile(open(path, "rb"), min(maxtok, os.path.getsize(path) // 2))
    t = t.tolist(); i = 0
    while i < len(t):
        j = i
        while j < len(t) and t[j] != eot: j += 1
        if j > i: yield t[i:j]
        i = j + 1


def check(ids_list, texts, name):
    global fails
    bad = rt = 0; ntok = sum(map(len, ids_list)); t0 = time.time()
    enc = tk.encode_batch(texts, add_special_tokens=False)
    for k, (ref, e, s) in enumerate(zip(ids_list, enc, texts)):
        if e.ids != ref:
            bad += 1
            if bad <= 3:
                p = next(q for q in range(min(len(e.ids), len(ref)) + 1) if q >= min(len(e.ids), len(ref)) or e.ids[q] != ref[q])
                print(f"  {name} doc {k} differs at {p}: hf {e.ids[max(0,p-2):p+3]} want {ref[max(0,p-2):p+3]}",
                      repr(b"|".join(toks[x] for x in ref[max(0, p - 2):p + 3])))
        if tk.decode(e.ids, skip_special_tokens=False) != s:
            rt += 1
            if rt <= 3: print(f"  {name} doc {k} decode round trip differs")
    print(f"{name}: {len(texts)} docs, {ntok} tokens, {bad} id mismatches, {rt} round-trip errors ({time.time()-t0:.1f}s)")
    fails += bad + rt


def from_corpus(spec):
    path, _, mx = spec.rpartition(":") if spec.rpartition(":")[2].isdigit() else (spec, "", "")
    ids, texts, nonutf = [], [], 0
    for d in docs(path, int(mx) if mx else 1 << 40):
        try: texts.append(b"".join(toks[x] for x in d).decode("utf-8")); ids.append(d)
        except UnicodeDecodeError: nonutf += 1
    if nonutf: print(f"  {path}: {nonutf} docs are not valid UTF-8 (skipped)")
    return ids, texts


for v in vals:
    check(*from_corpus(v), v)

# chat template: chat docs -> messages -> template -> must equal the doc text (+ eot after a final assistant turn)
import jinja2
tpl = jinja2.Environment(trim_blocks=True, lstrip_blocks=True).from_string(open(hf + "\\chat_template.jinja").read())
mk = re.compile(r"(?:^|\n)<\|(system|user|assistant|result|call|tools)\|>\n")
for c in chats:
    n = ok = skip = 0
    for ids, s in zip(*from_corpus(c)):
        parts = mk.split(s)
        if parts[0] or len(parts) < 3: skip += 1; continue
        msgs = [{"role": {"result": "tool"}.get(r, r), "content": t} for r, t in zip(parts[1::2], parts[2::2])]
        want = s + ("<|endoftext|>" if msgs[-1]["role"] == "assistant" else "")
        n += 1; same = tpl.render(messages=msgs, add_generation_prompt=False) == want; ok += same
        if not same and n - ok == 1: print("  template differs:", repr(want[:200]))
    gen = tpl.render(messages=[{"role": "user", "content": "Hi"}], add_generation_prompt=True)
    print(f"chat template {c}: {ok}/{n} docs re-rendered exactly ({skip} not chat-shaped); prompt {gen!r}",
          "== cmd_chat" if gen == "<|user|>\nHi\n<|assistant|>\n" else "!= cmd_chat")
    fails += (n - ok) + (gen != "<|user|>\nHi\n<|assistant|>\n")

# adversarial strings: HF ids go to <advout>; `cudalm tokcheck` re-encodes them with tok_encode and must agree
adv = ["Café naïve résumé", "日本語のテキストです。", "مرحبا بالعالم", "हिन्दी भाषा", "한국어 텍스트", "ภาษาไทยไม่มีช่องว่าง",
       "👍🏽 👨‍👩‍👧‍👦 🇺🇸🇯🇵", "é ä", "‏RTL‎", "a b", "a　b", "a b", "a\u0085b", "﻿bom",
       "zw​sp", "mvs᠎x", "a b c", "1234567890" * 5, "3.14159", "1,000,000", "x2y3z", "٣٤٥ ١٢", "½ ² Ⅻ ⅷ",
       "a  b", "a   b", "\t\tx", "x\n\n\ny", "l1\r\nl2\r\n", "abc   ", "   abc", " \n ", "\n \n", "a \tb", "     ", "\f\v",
       "I'm can't we're they've I'll he'd it's", "I'M CAN'T", "'s'''t", "rock'n'roll", "!!!???", "a...b", "--->", "<<>>",
       "<|", "<|notspecial|>", "<|user|", "|>", "<|user|><|assistant|>", "x<|user|>y", "<|USER|>", "<|end|>text",
       "<|system|>\nYou are helpful.\n<|user|>\nHi\n<|assistant|>\nHello!", "def f(x):\n    return x  # c\n\tpass\n",
       '{"a": [1, 2.5e-3, null], "b": "\\n"}', "<div class=\"x\">&nbsp;</div>", "a" * 500, "9" * 300, "ab" * 300,
       "abc日本def", "x²", "naïve123", "\U0010fffd﷐", "ǅungla ǈ", "ﬁ ﬀ Ⅸ", " ", "\n", "'", "' s"]
R = random.Random(1)
pool = list(" \t\n\r'sdmtlrev0123456789aZ.,!?-_<|>()[]{}") + ["<|user|>", "<|assistant|>", "<|end|>", "é", "日", "😀",
        " ", "́", "²", "　", "  ", "'ll", "'RE"]
adv += ["".join(R.choice(pool) for _ in range(R.randint(1, 120))) for _ in range(3000)]
def cp():
    while True:
        c = R.randint(1, 0x10FFFF if R.random() < .3 else 0xFFFF)
        if not 0xD800 <= c <= 0xDFFF: return chr(c)
adv += ["".join(cp() if R.random() < .7 else R.choice(" \n'0a") for _ in range(R.randint(1, 60))) for _ in range(3000)]
enc = tk.encode_batch(adv, add_special_tokens=False)
rt = sum(tk.decode(e.ids, skip_special_tokens=False) != s for e, s in zip(enc, adv))
out = array("H")
for e in enc: out.extend(e.ids); out.append(eot)
out.tofile(open(advout, "wb"))
print(f"adversarial: {len(adv)} strings, {len(out) - len(adv)} tokens -> {advout}; {rt} round-trip errors")
fails += rt
print("FAILS", fails)
