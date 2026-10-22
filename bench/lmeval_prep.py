# Convert raw benchmark files into cudalm lmeval .tsv (prompt formats follow lm-eval-harness).
# line: gold \t context \t cont0 \t cont1 ...   (one cont = greedy/ppl task); \n,\t,\\ escaped
import json, re, sys, os
import pyarrow.parquet as pq

d = sys.argv[1] if len(sys.argv) > 1 else "."
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

rows = []
for r in pq.read_table(os.path.join(d, "arce_test.parquet")).to_pylist():
    lab = r["choices"]["label"]
    rows.append((lab.index(r["answerKey"]), "Question: " + r["question"] + "\nAnswer:", [" " + t for t in r["choices"]["text"]]))
out("arc_easy.tsv", rows)

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
