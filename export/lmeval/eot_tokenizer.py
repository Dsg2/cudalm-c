"""Tokenizer copy for lm-evaluation-harness that starts every sequence with EOT, as `cudalm lmeval` does.

    eot_tokenizer.py <hf_dir> <out_dir>

Then pass `tokenizer=<out_dir>,add_bos_token=True` in lm-eval's --model_args. The exported tokenizer itself adds no
special tokens; this copy adds EOT only when the caller asks for special tokens (add_special_tokens=True)."""
import json, os, shutil, sys

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if f.startswith("tokenizer") or f in ("special_tokens_map.json", "chat_template.jinja"):
        shutil.copy(os.path.join(src, f), dst)
eid = json.load(open(os.path.join(src, "config.json")))["eos_token_id"]
p = os.path.join(dst, "tokenizer.json"); tj = json.load(open(p, encoding="utf-8"))
eot = next(t["content"] for t in tj["added_tokens"] if t["id"] == eid)
S, A, B = {"SpecialToken": {"id": eot, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 0}}
tj["post_processor"] = {"type": "TemplateProcessing", "single": [S, A], "pair": [S, A, B],
                        "special_tokens": {eot: {"id": eot, "ids": [eid], "tokens": [eot]}}}
json.dump(tj, open(p, "w", encoding="utf-8"), ensure_ascii=False)
p = os.path.join(dst, "tokenizer_config.json"); tc = json.load(open(p, encoding="utf-8")); tc["bos_token"] = eot
json.dump(tc, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(dst, trust_remote_code=True)
x = "Question: 2 + 2?\nAnswer:"
a, b = t(x, add_special_tokens=True)["input_ids"], t(x, add_special_tokens=False)["input_ids"]
assert a == [eid] + b, (a, b)
print(f"{dst}: EOT {eot!r} (id {eid}) prepended with add_special_tokens=True")
