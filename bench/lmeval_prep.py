# Download (if missing) and convert the lmeval benchmarks into cudalm lmeval .tsv files
# (prompt formats follow lm-eval-harness). Usage: python lmeval_prep.py <bench_dir>
# line: gold \t context \t cont0 \t cont1 ...   (one cont = greedy/ppl task); \n,\t,\\ escaped
import json, re, sys, os, urllib.request
import pyarrow.parquet as pq

d = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(d, exist_ok=True)
SRC = {
    "hs_val.parquet": "https://huggingface.co/datasets/Rowan/hellaswag/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
    "arce_test.parquet": "https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Easy/test-00000-of-00001.parquet",
    "arcc_test.parquet": "https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/test-00000-of-00001.parquet",
    "arithmark-3.jsonl": "https://huggingface.co/datasets/AxiomicLabs/Arithmark-3.0/resolve/main/arithmark-3.jsonl",
    "piqa_valid.jsonl": "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/valid.jsonl",
    "piqa_valid-labels.lst": "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/valid-labels.lst",
    "lambada_test.jsonl": "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl",
}
for name, url in SRC.items():
    p = os.path.join(d, name)
    if not os.path.exists(p):
        print("download", url, flush=True)
        urllib.request.urlretrieve(url, p + ".part")
        os.replace(p + ".part", p)

esc = lambda s: s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")
def out(name, rows):
    with open(os.path.join(d, name), "w", encoding="utf-8", newline="\n") as f:
        for g, c, conts in rows: f.write("\t".join([str(g), esc(c)] + [esc(x) for x in conts]) + "\n")
    print(name, len(rows))

def hs_pre(t):
    t = t.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", t).replace("  ", " ")
rows = []
for r in pq.read_table(os.path.join(d, "hs_val.parquet")).to_pylist():
    ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
    rows.append((int(r["label"]), hs_pre(r["activity_label"] + ": " + ctx), [" " + hs_pre(e) for e in r["endings"]]))
out("hellaswag.tsv", rows)

for src, name in (("arce_test.parquet", "arc_easy.tsv"), ("arcc_test.parquet", "arc_challenge.tsv")):
    rows = []
    for r in pq.read_table(os.path.join(d, src)).to_pylist():
        lab = r["choices"]["label"]
        rows.append((lab.index(r["answerKey"]), "Question: " + r["question"] + "\nAnswer:", [" " + t for t in r["choices"]["text"]]))
    out(name, rows)

rows = []   # ArithMark 3.0: unfinished word problem + 4 endings (leading space kept), as its bencharithmark-3.py scores it
for ln in open(os.path.join(d, "arithmark-3.jsonl"), encoding="utf-8"):
    r = json.loads(ln)
    rows.append((int(r["label"]), r["ctx"], r["endings"]))
out("arithmark.tsv", rows)

labs = open(os.path.join(d, "piqa_valid-labels.lst")).read().split()
rows = []
for ln, g in zip(open(os.path.join(d, "piqa_valid.jsonl"), encoding="utf-8"), labs):
    r = json.loads(ln)
    rows.append((int(g), "Question: " + r["goal"] + "\nAnswer:", [" " + r["sol1"], " " + r["sol2"]]))
out("piqa.tsv", rows)

rows = []
for ln in open(os.path.join(d, "lambada_test.jsonl"), encoding="utf-8"):
    c, w = json.loads(ln)["text"].rsplit(" ", 1)
    rows.append((0, c, [" " + w]))
out("lambada.tsv", rows)
