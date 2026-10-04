#!/usr/bin/env python3
"""Live schema guard for the 4B post-training sources (POSTTRAIN_PLAN.md 3).

The silent-zero-doc bug is always a column-name mismatch: the parser reads a
column the shard does not have, iterates nothing, and the source contributes 0
tokens without raising. This probes each new source's REAL schema via the HF
datasets-server /first-rows endpoint and asserts the exact field the parser reads
is present and non-empty in a live sample -- catching the mismatch before a
multi-hour build silently drops a whole category.

  python prep/test_parsers.py

Cheap by design: /first-rows returns a handful of rows, so this never downloads
a parquet shard. It checks the schema, not the parser body -- the prose/code
parsers are one-liners over a single column, so a present, non-empty column is
the whole contract.
"""
import json, sys, urllib.request

FIRST_ROWS = "https://datasets-server.huggingface.co/first-rows?dataset={d}&config={c}&split={s}"

# (label, repo, config, split, probe) where probe(row0)->bool asserts the field
# the source's parser reads is present and non-empty.
def has_text(r):     return bool((r.get("text") or "").strip())
def has_code(r):     return bool((r.get("code") or "").strip())
def has_messages(r):
    m = r.get("messages")
    return isinstance(m, list) and len(m) > 0 and isinstance(m[0], dict) \
        and bool(m[0].get("content"))

CASES = [
 # general/edu + synthetic all flow through src_cosmopedia -> reads "text"
 ("fineweb-edu",   "HuggingFaceFW/fineweb-edu",      "sample-10BT",   "train", has_text),
 ("cosmopedia-v2", "HuggingFaceTB/cosmopedia",       "web_samples_v2","train", has_text),
 ("finemath",      "HuggingFaceTB/finemath",         "finemath-4plus","train", has_text),
 # code -> src_code reads "code"
 ("github-python", "codeparrot/github-code-clean",   "Python-all",    "train", has_code),
 # chat -> src_messages reads native "messages":[{role,content}]
 ("tulu3",         "allenai/tulu-3-sft-mixture",     "default",       "train", has_messages),
]


def first_row(repo, config, split):
    url = FIRST_ROWS.format(d=repo, c=config, s=split)
    req = urllib.request.Request(url, headers={"User-Agent": "tinylm-schema-probe"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    rows = data.get("rows") or []
    if not rows:
        raise RuntimeError("no rows returned")
    return rows[0].get("row") or {}


def main():
    fails = 0
    for label, repo, config, split, probe in CASES:
        try:
            row = first_row(repo, config, split)
            ok = probe(row)
            cols = ",".join(sorted(row.keys()))
            if ok:
                print(f"[PASS] {label:14s} {repo}  cols=[{cols}]")
            else:
                fails += 1
                print(f"[FAIL] {label:14s} {repo}  parser field MISSING/EMPTY  cols=[{cols}]")
        except Exception as e:
            fails += 1
            print(f"[FAIL] {label:14s} {repo}  probe error: {type(e).__name__} {e}")
    print(f"\n{len(CASES)-fails}/{len(CASES)} sources OK")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
