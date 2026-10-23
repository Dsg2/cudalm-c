"""fetch.py -- text getter for `cudalm data fetch` (the only Python step of the cudalm data toolchain).
Writes plain .jsonl ({"text","cat","src"} per line, chat already rendered with the chatml markers) that
`cudalm data tokenizer` / `cudalm data encode` read directly.

  python fetch.py list                                   profiles and their sources
  python fetch.py get <profile|url> <out> [tokens=1e9] [template]   download (resumable) + parse; raw shards are deleted
       profile hf:<user/dataset>[:config[:split]][@template]  any Hugging Face dataset (its parquet files)
  python fetch.py convert <file|folder> <out> [cat] [template]      parquet/json/jsonl -> jsonl
  python fetch.py peek <file|folder> [rows=3]                       first rows as JSON (for the TUI preview)

template: which columns become the document (empty = auto-detect):
  text=a+b                     plain text, columns joined by a blank line ("a" alone = text=a)
  chat=col                     a list of turns ({role|from, content|value})
  user=q+ctx,assistant=a       one prompt/answer exchange
  + system=col, tools=col      optional, with chat= or user=/assistant=

HF_TOKEN (env) is sent when set (needed for gated sets such as xlam60k).
"""
import os, sys, json, time, re, threading, queue, urllib.request, urllib.error, urllib.parse

CPT, SLACK = 4.0, 1.15            # chars per token (v3 tokenizer, approx), over-fetch slack
M = dict(system="<|system|>\n", tools="<|tools|>\n", user="<|user|>\n",
         assistant="<|assistant|>\n", call="<|call|>\n", result="<|result|>\n")
ROLE = {"system": "system", "human": "user", "user": "user", "prompter": "user",
        "gpt": "assistant", "assistant": "assistant", "chatgpt": "assistant",
        "tool": "result", "observation": "result", "function_response": "result",
        "function_call": "call", "tool_call": "call"}

def render(turns):
    parts = [M[r] + c.strip() for r, c in turns if (c or "").strip()]
    return "\n".join(parts) if len(parts) >= 2 else None

def as_list(v):
    if isinstance(v, str):
        try: return json.loads(v)
        except Exception: return None
    return v

def jtxt(v): return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)

def conv_turns(convs, tools=None, system=None):
    turns = []
    if system: turns.append(("system", system))
    if tools: turns.append(("tools", jtxt(tools)))
    for c in as_list(convs) or []:
        if not isinstance(c, dict): continue
        r = ROLE.get((c.get("from") or c.get("role") or "").strip().lower())
        v = c.get("value") or c.get("content") or ""
        if r and v: turns.append((r, jtxt(v)))
    return turns

# ------------------------------ readers ------------------------------------
def iter_jsonl(p):
    with open(p, encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                try: yield json.loads(line)
                except json.JSONDecodeError: pass

def iter_json_array(p):
    raw = open(p, encoding="utf-8-sig", errors="ignore").read()
    dec = json.JSONDecoder(); i = raw.find("[") + 1; n = len(raw)
    while i < n:
        while i < n and raw[i] in " \t\r\n,": i += 1
        if i >= n or raw[i] == "]": break
        try: obj, i = dec.raw_decode(raw, i)
        except json.JSONDecodeError: break
        yield obj

def iter_rows(p, cols=None):
    if p.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(p)
        have = [c for c in (cols or pf.schema_arrow.names) if c in pf.schema_arrow.names]
        if not have: return
        for b in pf.iter_batches(batch_size=256, columns=have):
            yield from b.to_pylist()
    else:
        with open(p, encoding="utf-8-sig", errors="ignore") as f: head = f.read(4096).lstrip()
        yield from (iter_json_array(p) if head.startswith("[") else iter_jsonl(p))

def auto_doc(r, minlen=1):
    """any row -> text: chat (messages/conversations [+tools/system]) or the first text-like column"""
    if not isinstance(r, dict): return None
    if isinstance(r.get("text"), str): return r["text"].strip() or None
    for k in ("messages", "conversations", "conversation"):
        if r.get(k): return render(conv_turns(r[k], tools=r.get("tools"), system=r.get("system")))
    fs = lambda ks: next((r[k].strip() for k in ks if isinstance(r.get(k), str) and r[k].strip()), None)
    q, a = fs(("instruction", "prompt", "question")), fs(("output", "response", "answer"))
    if q and a:                                                    # alpaca-style pair
        i = fs(("input", "context"))
        return render(conv_turns([], tools=r.get("tools"), system=r.get("system")) + [("user", q + "\n\n" + i if i else q), ("assistant", a)])
    for k in ("content", "code", "markdown", "body"):
        v = r.get(k)
        if isinstance(v, str) and len(v.strip()) >= minlen: return v.strip()
    return None

# ------------------------------ parsers (profile sources) -------------------
def p_col(col, minlen):
    def f(p):
        for r in iter_rows(p, [col]):
            t = (r.get(col) or "").strip()
            if len(t) > minlen: yield t
    return f

def p_conv(key="conversations", tools=False, system=False):
    def f(p):
        for r in iter_rows(p):
            d = render(conv_turns(r.get(key), tools=r.get("tools") if tools else None, system=r.get("system") if system else None))
            if d: yield d
    return f

TOOLCALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
def p_hermes_fc(p):
    for r in iter_rows(p):
        turns = []
        for c in r.get("conversations") or []:
            role = ROLE.get((c.get("from") or "").lower()); v = c.get("value") or ""
            if not role or not v: continue
            if role == "assistant" and "<tool_call>" in v:
                pos = 0
                for m in TOOLCALL.finditer(v):
                    if v[pos:m.start()].strip(): turns.append(("assistant", v[pos:m.start()]))
                    turns.append(("call", m.group(1))); pos = m.end()
                if v[pos:].strip(): turns.append(("assistant", v[pos:]))
            else: turns.append((role, v))
        d = render(turns)
        if d: yield d

GLAIVE = re.compile(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)")
def p_glaive(p):
    for r in iter_rows(p):
        s = (r.get("system") or "").strip()
        if s.startswith("SYSTEM:"): s = s[7:].strip()
        parts = GLAIVE.split((r.get("chat") or "").replace("<|endoftext|>", " "))
        turns = [("system", s)] if s else []
        for i in range(1, len(parts), 2):
            lab, c = parts[i], (parts[i + 1] if i + 1 < len(parts) else "").strip()
            if not c: continue
            if lab == "USER:": turns.append(("user", c))
            elif lab == "FUNCTION RESPONSE:": turns.append(("result", c))
            elif "<functioncall>" in c:
                pre, _, call = c.partition("<functioncall>")
                if pre.strip(): turns.append(("assistant", pre))
                turns.append(("call", call))
            else: turns.append(("assistant", c))
        d = render(turns)
        if d: yield d

def p_xlam(p):
    for r in iter_rows(p):
        turns = []
        if r.get("tools"): turns.append(("tools", jtxt(r["tools"])))
        if r.get("query"): turns.append(("user", r["query"]))
        if r.get("answers"): turns.append(("call", jtxt(r["answers"])))
        d = render(turns)
        if d: yield d

TPL_KEYS = ("text", "chat", "user", "assistant", "system", "tools")
def parse_tpl(s):
    m = {}
    for part in filter(None, (s or "").split(",")):
        k, eq, v = part.partition("=")
        k, v = (k.strip(), v) if eq else ("text", k)
        if k not in TPL_KEYS: sys.exit(f"template: unknown key {k!r} - use {' / '.join(TPL_KEYS)}")
        m[k] = [c.strip() for c in v.split("+") if c.strip()]
    if ("user" in m) != ("assistant" in m): sys.exit("template: user= and assistant= go together")
    return m

def cell(r, cols):                                                  # columns -> one string (non-strings as JSON)
    return "\n\n".join(t for t in ((v if isinstance(v, str) else jtxt(v)).strip() for v in (r.get(c) for c in cols) if v is not None) if t)

def tpl_doc(r, m):
    if not isinstance(r, dict): return None
    if "text" in m: return cell(r, m["text"]) or None
    sysm, tools = cell(r, m.get("system", [])) or None, cell(r, m.get("tools", [])) or None
    if "chat" in m: return render(conv_turns(r.get(m["chat"][0]), tools=tools, system=sysm))
    if "user" in m: return render(conv_turns([], tools=tools, system=sysm) + [("user", cell(r, m["user"])), ("assistant", cell(r, m["assistant"]))])
    return None

def p_tpl(m):
    cols = list(dict.fromkeys(c for v in m.values() for c in v))
    def f(p):
        for r in iter_rows(p, cols):
            d = tpl_doc(r, m)
            if d: yield d
    return f

def p_auto(p):
    for r in iter_rows(p):
        d = auto_doc(r)
        if d: yield d

# ------------------------------ sources / profiles ---------------------------
HF = "https://huggingface.co/datasets"
def shards(fmt, n, start=0): return [fmt.format(i) for i in range(start, start + n)]
_UCH = f"{HF}/HuggingFaceH4/ultrachat_200k/resolve/main/data"
S = {  # name: (urls, parser, license)
  "ultrafineweb":  (shards(f"{HF}/openbmb/Ultra-FineWeb/resolve/main/data/ultrafineweb_en/ultrafineweb-en-part-{{:04d}}-of-2048.parquet", 40, 1),
                    p_col("content", 200), "openbmb/Ultra-FineWeb (ODC-By-1.0; derived from FineWeb/CommonCrawl)"),
  "ufw-l3":        (shards(f"{HF}/openbmb/Ultra-FineWeb-L3/resolve/main/data/ultrafineweb_en_l3/multi_style/part-{{:05d}}-f36f5a53-4a77-434b-b9bc-67ed69b93fe2-c000.snappy.parquet", 30),
                    p_col("content", 200), "openbmb/Ultra-FineWeb-L3 (ODC-By-1.0; synthetic-styled web)"),
  "ufw-l3-qa":     (shards(f"{HF}/openbmb/Ultra-FineWeb-L3/resolve/main/data/ultrafineweb_en_l3/qa/part-{{:05d}}-37dc9f21-f87f-4f43-8dd2-134424f1537a-c000.snappy.parquet", 60),
                    p_col("content", 200), "openbmb/Ultra-FineWeb-L3 qa (Apache-2.0; synthetic Q&A-styled web)"),
  "ufw-l3-ms":     (shards(f"{HF}/openbmb/Ultra-FineWeb-L3/resolve/main/data/ultrafineweb_en_l3/multi_style/part-{{:05d}}-f36f5a53-4a77-434b-b9bc-67ed69b93fe2-c000.snappy.parquet", 60, 30),
                    p_col("content", 200), "openbmb/Ultra-FineWeb-L3 multi_style, shards 30+ (unused by v3pre) (Apache-2.0)"),
  "github-python": (shards("https://huggingface.co/api/datasets/codeparrot/github-code-clean/parquet/Python-all/train/{}.parquet", 16),
                    p_col("code", 100), "codeparrot/github-code-clean (per-file OSS licences, permissive subset)"),
  "cosmopedia-v2": (shards(f"{HF}/HuggingFaceTB/cosmopedia/resolve/main/data/web_samples_v2/train-{{:05d}}-of-00118.parquet", 30),
                    p_col("text", 200), "HuggingFaceTB/cosmopedia (Apache-2.0; synthetic)"),
  "finemath":      (shards(f"{HF}/HuggingFaceTB/finemath/resolve/main/finemath-4plus/train-{{:05d}}-of-00064.parquet", 40),
                    p_col("text", 200), "HuggingFaceTB/finemath (ODC-By-1.0)"),
  "hermes2.5":     ([f"{HF}/teknium/OpenHermes-2.5/resolve/main/openhermes2_5.json"], p_conv(), "teknium/OpenHermes-2.5 (research/mixed source licences)"),
  "ultrachat":     ([f"{_UCH}/train_sft-00000-of-00003-a3ecf92756993583.parquet", f"{_UCH}/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
                     f"{_UCH}/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet"], p_conv("messages"), "HuggingFaceH4/ultrachat_200k (MIT)"),
  "toolace":       ([f"{HF}/Team-ACE/ToolACE/resolve/main/data.json"], p_conv(system=True), "Team-ACE/ToolACE (Apache-2.0)"),
  "hermes-fc":     ([f"{HF}/NousResearch/hermes-function-calling-v1/resolve/main/func-calling.json"], p_hermes_fc, "NousResearch/hermes-function-calling-v1 (Apache-2.0)"),
  "glaive-v2":     ([f"{HF}/glaiveai/glaive-function-calling-v2/resolve/main/glaive-function-calling-v2.json"], p_glaive, "glaiveai/glaive-function-calling-v2 (Apache-2.0)"),
  "hermes-single": ([f"{HF}/NousResearch/hermes-function-calling-v1/resolve/main/func-calling-singleturn.json"], p_hermes_fc, "NousResearch/hermes-function-calling-v1 (Apache-2.0)"),
  "hermes-gl5k":   ([f"{HF}/NousResearch/hermes-function-calling-v1/resolve/main/glaive-function-calling-5k.json"], p_hermes_fc, "NousResearch/hermes-function-calling-v1 (Apache-2.0)"),
  "apigen-mt":     ([f"{HF}/Salesforce/APIGen-MT-5k/resolve/main/apigen-mt_5k.json"], p_conv(), "Salesforce/APIGen-MT-5k (CC-BY-4.0)"),
  "xlam60k":       ([f"{HF}/Salesforce/xlam-function-calling-60k/resolve/main/xlam_function_calling_60k.json"], p_xlam, "Salesforce/xlam-function-calling-60k (CC-BY-4.0, gated: needs HF_TOKEN)"),
  "tulu3":         (shards(f"{HF}/allenai/tulu-3-sft-mixture/resolve/main/data/train-{{:05d}}-of-00006.parquet", 6), p_conv("messages"), "allenai/tulu-3-sft-mixture (ODC-By-1.0)"),
  "smoltalk2":     (shards(f"{HF}/HuggingFaceTB/smoltalk2/resolve/main/SFT/OpenHermes_2.5_no_think-{{:05d}}-of-00002.parquet", 2), p_conv("messages"), "HuggingFaceTB/smoltalk2 (Apache-2.0)"),
  "openthoughts3": (shards(f"{HF}/open-thoughts/OpenThoughts3-1.2M/resolve/main/data/train-{{:05d}}-of-00120.parquet", 12), p_conv(), "open-thoughts/OpenThoughts3-1.2M (Apache-2.0)"),
  "hermes-rtu":    ([f"{HF}/interstellarninja/hermes_reasoning_tool_use/resolve/main/data/train-00000-of-00001.parquet"], p_conv(tools=True), "interstellarninja/hermes_reasoning_tool_use (Apache-2.0)"),
}
P = {  # profile: (description, {cat: weight}, [(cat, source)])
  "v3pre":  ("pretraining blend (web, code, synthetic, math, chat, tools)",
             {"web": .34, "code": .20, "synthetic": .16, "math": .14, "chat": .10, "tool": .06},
             [("web", "ultrafineweb"), ("code", "github-python"), ("synthetic", "cosmopedia-v2"), ("synthetic", "ufw-l3"),
              ("math", "finemath"), ("chat", "hermes2.5"), ("chat", "ultrachat"), ("tool", "toolace"), ("tool", "hermes-fc"),
              ("tool", "glaive-v2"), ("tool", "hermes-single"), ("tool", "hermes-gl5k"), ("tool", "apigen-mt")]
             + ([("tool", "xlam60k")] if os.environ.get("HF_TOKEN") else [])),
  "v3sft":  ("instruction / reasoning / tool-use SFT blend",
             {"general": .24, "chat": .20, "code": .18, "reasoning": .14, "tool": .14, "agentic": .10},
             [("general", "tulu3"), ("chat", "smoltalk2"), ("code", "github-python"), ("reasoning", "openthoughts3"),
              ("tool", "toolace"), ("tool", "hermes-fc"), ("agentic", "hermes-rtu")]),
  "ufwl3-en": ("Ultra-FineWeb-L3 English, qa + multi_style at the card's token ratio", {"qa": .6, "style": .4},
               [("qa", "ufw-l3-qa"), ("style", "ufw-l3-ms")]),
  "v3chat": ("plain chat SFT (tulu3 + smoltalk2)", {"general": .62, "chat": .38}, [("general", "tulu3"), ("chat", "smoltalk2")]),
}
SINGLE = {"web": ("web", "ultrafineweb"), "code": ("code", "github-python"), "math": ("math", "finemath"),
          "cosmopedia": ("synthetic", "cosmopedia-v2"), "chat": ("chat", "ultrachat"), "tools": ("tool", "glaive-v2")}
for k, (c, s) in SINGLE.items(): P[k] = (f"only {s} ({S[s][2].split(' (')[0]})", {c: 1.0}, [(c, s)])

# ------------------------------ any Hugging Face dataset ---------------------
def hf_json(url):
    h = {"User-Agent": "Mozilla/5.0"}
    if os.environ.get("HF_TOKEN"): h["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60))

def hf_source(spec, tpl=None):
    """hf:<repo>[:config[:split]][@template] -> (name, urls, parser, licence). Files: the hub's parquet conversion
    (datasets-server), else the repo's own .parquet/.jsonl/.json files. Config 'default' and split 'train' when not given."""
    s, col = (spec[3:].rsplit("@", 1) + [None])[:2] if "@" in spec else (spec[3:], None)
    tpl = tpl or col
    repo, cfg, split = (s.split(":") + [None, None])[:3]
    q = urllib.parse.quote(repo)
    try: files = hf_json(f"https://datasets-server.huggingface.co/parquet?dataset={q}")
    except Exception as e: files = {"parquet_files": []}; print(f"  (no parquet conversion: {e})", flush=True)
    pf = files.get("parquet_files", [])
    if pf:
        cfgs = list(dict.fromkeys(f["config"] for f in pf)); cfg = cfg or ("default" if "default" in cfgs else cfgs[0])
        sp = list(dict.fromkeys(f["split"] for f in pf if f["config"] == cfg)); split = split or ("train" if "train" in sp else sp[0] if sp else "train")
        urls = [f["url"] for f in sorted((f for f in pf if f["config"] == cfg and f["split"] == split), key=lambda f: f["filename"])]
        if files.get("partial"): print("  note: the hub's parquet conversion of this dataset is partial (about the first 5 GB per split)", flush=True)
    else: urls = []
    if not urls:                                                   # repo files, preferring paths that mention the split
        tree = hf_json(f"https://huggingface.co/api/datasets/{q}/tree/main?recursive=true")
        fs = [t["path"] for t in tree if t.get("type") == "file" and t["path"].endswith((".parquet", ".jsonl", ".json")) and not t["path"].startswith(".")]
        pick = [f for f in fs if (split or "train") in f.lower()] or fs
        if (split or cfg) and not pick: sys.exit(f"no files for {spec}")
        urls = [f"{HF}/{repo}/resolve/main/{urllib.parse.quote(f)}" for f in sorted(pick)]
    if not urls: sys.exit(f"{repo}: no parquet / jsonl / json files found (script-only datasets are not supported)")
    lic = "?"
    try: lic = next((t[8:] for t in hf_json(f"https://huggingface.co/api/datasets/{q}").get("tags", []) if t.startswith("license:")), "?")
    except Exception: pass
    name = re.sub(r"[^\w.-]", "_", repo.split("/")[-1] + (f"-{cfg}" if cfg and cfg != "default" else "") + (f"-{split}" if split and split != "train" else ""))
    print(f"[hf  ] {repo} config {cfg or '?'} split {split or '?'}: {len(urls)} file(s), template {tpl or 'auto'}, licence {lic}", flush=True)
    return name, urls, p_tpl(parse_tpl(tpl)) if tpl else p_auto, f"{repo} ({lic})"

# ------------------------------ download ------------------------------------
def download(url, dest, retries=6):
    for attempt in range(retries):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        try:
            h = {"Range": f"bytes={have}-", "User-Agent": "Mozilla/5.0"}
            if os.environ.get("HF_TOKEN"): h["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
            r = urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=180)
            if have and getattr(r, "status", 206) != 206: have = 0
            with open(dest, "ab" if have else "wb") as f:
                while True:
                    c = r.read(1 << 20)
                    if not c: break
                    f.write(c)
            r.close()
            print(f"  downloaded {os.path.basename(dest)} {os.path.getsize(dest)/1e6:.0f} MB", flush=True); return dest
        except urllib.error.HTTPError as e:
            if e.code == 416 and have: return dest
            if e.code in (401, 403, 404): print(f"  HTTP {e.code} {url}" + (" (gated: set HF_TOKEN)" if e.code != 404 else ""), flush=True); return None
            print(f"  retry {attempt+1}/{retries} HTTP {e.code} {os.path.basename(dest)}", flush=True); time.sleep(3)
        except Exception as e:
            print(f"  retry {attempt+1}/{retries} {type(e).__name__} {os.path.basename(dest)}", flush=True); time.sleep(3)
    return None

def rm(p):
    try: os.remove(p)
    except OSError: pass

# ------------------------------ get ------------------------------------------
def get_source(out, cat, name, urls, parser, budget, state, save):
    path = os.path.join(out, f"{cat}__{name}.jsonl"); raw = os.path.join(out, "_raw"); os.makedirs(raw, exist_ok=True)
    st = state.setdefault(name, {"shards": 0, "chars": 0, "bytes": 0, "docs": 0, "done": False})
    if st["done"] and os.path.exists(path):
        print(f"[skip] {name}: done ({st['chars']/1e6:.0f}M chars)", flush=True); return st
    if os.path.exists(path):
        with open(path, "r+b") as f: f.truncate(st["bytes"])
    else: st.update(shards=0, chars=0, bytes=0, docs=0)
    q = queue.Queue(maxsize=1); stop = threading.Event()
    def producer():
        for si in range(st["shards"], len(urls)):
            if stop.is_set(): break
            ext = ".parquet" if ".parquet" in urls[si] else ".json"
            q.put((si, download(urls[si], os.path.join(raw, f"{name}_{si:04d}{ext}"))))
        q.put(None)
    threading.Thread(target=producer, daemon=True).start()
    print(f"[get ] {name} ({cat}): want {budget/1e6:.0f}M chars (~{budget/CPT/1e6:.0f}M tokens), {len(urls)} shard(s), from shard {st['shards']}", flush=True)
    t0 = time.time()
    with open(path, "ab") as f:
        while st["chars"] < budget:
            it = q.get()
            if it is None: break
            si, p = it
            if p:
                for d in parser(p):
                    f.write((json.dumps({"text": d, "cat": cat, "src": name}, ensure_ascii=False) + "\n").encode("utf-8"))
                    st["chars"] += len(d); st["docs"] += 1
                    if st["chars"] >= budget: break
                rm(p)
            f.flush(); os.fsync(f.fileno()); st["bytes"] = f.tell(); st["shards"] = si + 1; save()
            print(f"       {name} shard {si}: {st['chars']/1e6:.0f}/{budget/1e6:.0f}M chars, {st['docs']} docs, {time.time()-t0:.0f}s", flush=True)
    stop.set()
    while True:
        try: it = q.get(timeout=1)
        except queue.Empty: break
        if it is None: break
        if it[1]: rm(it[1])
    st["done"] = True; save(); open(path + ".done", "w").close(); return st   # .done lets `data encode ... follow=1` finish

def cmd_get(spec, out, tokens, tpl=None):
    os.makedirs(out, exist_ok=True)
    if spec.startswith("http"):
        name = re.sub(r"[^\w.-]", "_", os.path.splitext(spec.rstrip("/").split("/")[-1].split("?")[0])[0])
        prof = ("url", {"text": 1.0}, [("text", name)]); S[name] = ([spec], p_tpl(parse_tpl(tpl)) if tpl else p_auto, spec)
    elif spec.startswith("hf:"):
        name, urls, parser, lic = hf_source(spec, tpl); S[name] = (urls, parser, lic); prof = (spec, {name: 1.0}, [(name, name)])
    elif tpl: sys.exit("a template applies to hf: and URL sources, not to the built-in profiles")
    elif spec in P: prof = P[spec]
    else: sys.exit(f"unknown profile {spec!r} - run: fetch.py list")
    desc, w, srcs = prof
    if any(".parquet" in u for _, nm in srcs for u in S[nm][0]):
        try: import pyarrow.parquet
        except ImportError: sys.exit(f"pyarrow is missing for {sys.executable}: run  {sys.executable} -m pip install pyarrow  (or set CUDALM_PYTHON)")
    spath = os.path.join(out, "_state.json")
    try: state = json.load(open(spath))
    except Exception: state = {}
    def save():
        json.dump(state, open(spath + ".tmp", "w"), indent=1); os.replace(spath + ".tmp", spath)
    n = {c: sum(1 for cc, _ in srcs if cc == c) for c in w}
    print(f"[plan] {spec}: ~{tokens/1e6:.0f}M tokens -> {out}", flush=True)
    for cat, name in srcs:
        b = tokens * w[cat] * CPT * SLACK
        try: get_source(out, cat, name, S[name][0], S[name][1], int(min(b, b * 2.0 / n[cat])), state, save)
        except Exception as e: print(f"  ! {name} failed: {type(e).__name__} {e}", flush=True)
    json.dump({"profile": spec, "target_tokens": tokens, "weights": w,
               "sources": {nm: dict(state.get(nm, {}), cat=c, license=S[nm][2]) for c, nm in srcs}},
              open(os.path.join(out, "sources.json"), "w"), indent=2)
    try: os.rmdir(os.path.join(out, "_raw"))
    except OSError: pass
    tot = sum(state.get(nm, {}).get("chars", 0) for _, nm in srcs)
    print(f"[done] {out}: {tot/1e6:.0f}M chars (~{tot/CPT/1e6:.0f}M tokens). Next: Build corpus with this folder as input.", flush=True)

# ------------------------------ convert --------------------------------------
def cmd_convert(src, out, cat=None, tpl=None):
    os.makedirs(out, exist_ok=True)
    files = [src] if os.path.isfile(src) else sorted(os.path.join(dp, f) for dp, _, fs in os.walk(src) for f in fs
                                                      if f.endswith((".parquet", ".json", ".jsonl")))
    if not files: sys.exit(f"no .parquet/.json/.jsonl under {src}")
    m = parse_tpl(tpl) if tpl else None
    if m: print(f"[conv] template {tpl}", flush=True)
    for p in files:
        stem = os.path.splitext(os.path.basename(p))[0]; c = cat or stem.split("__")[0]
        dst = os.path.join(out, (stem if stem == c or stem.startswith(c + "__") else f"{c}__{stem}") + ".jsonl")
        if os.path.abspath(dst) == os.path.abspath(p): dst = os.path.join(out, f"{c}__{stem}.conv.jsonl")
        n = ch = bad = 0; t0 = time.time()
        with open(dst + ".tmp", "w", encoding="utf-8", newline="\n") as f:
            for r in iter_rows(p):
                d = tpl_doc(r, m) if m else auto_doc(r)
                if not d: bad += 1; continue
                rc = r.get("cat") if not cat and isinstance(r.get("cat"), str) else c
                f.write(json.dumps({"text": d, "cat": rc, "src": stem}, ensure_ascii=False) + "\n"); n += 1; ch += len(d)
        os.replace(dst + ".tmp", dst)
        print(f"[conv] {p} -> {dst}: {n} docs, {ch/1e6:.1f}M chars (~{ch/CPT/1e6:.1f}M tokens), {bad} rows without text, {time.time()-t0:.0f}s", flush=True)
        if not n:
            try: import pyarrow.parquet as pq; print("       columns:", pq.ParquetFile(p).schema_arrow.names if p.endswith(".parquet") else "?")
            except Exception: pass

def cmd_list():
    for k, (d, w, srcs) in P.items():
        print(f"{k:11s} {d}")
        for c, s in srcs: print(f"{'':13s}{c:10s} {w[c]*100:4.0f}%  {s:14s} {S[s][2]}")

def cmd_peek(src, n=3):
    """first rows of the first file, shaped like datasets-server /first-rows ({"rows":[{"row":{...}}]})"""
    files = [src] if os.path.isfile(src) else sorted(os.path.join(dp, f) for dp, _, fs in os.walk(src) for f in fs
                                                      if f.endswith((".parquet", ".json", ".jsonl")) and f != "sources.json" and not f.startswith("_"))
    try:
        if not files: raise RuntimeError(f"no .parquet/.json/.jsonl under {src}")
        rows = []
        for r in iter_rows(files[0]):
            if isinstance(r, dict): rows.append({"row": r})
            if len(rows) >= n: break
        o = {"file": files[0], "files": len(files), "rows": rows}
    except Exception as e: o = {"error": f"{type(e).__name__}: {e}" + (" (pip install pyarrow)" if isinstance(e, ImportError) else "")}
    sys.stdout.buffer.write(json.dumps(o, ensure_ascii=False, default=str).encode("utf-8")); sys.stdout.flush()

if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("list", "get", "convert", "peek"): print(__doc__); sys.exit(1)
    if a[0] == "list": cmd_list()
    elif a[0] == "get" and len(a) >= 3: cmd_get(a[1], a[2], int(float(a[3])) if len(a) > 3 and a[3] else 1_000_000_000, a[4] if len(a) > 4 and a[4] else None)
    elif a[0] == "convert" and len(a) >= 3: cmd_convert(a[1], a[2], a[3] if len(a) > 3 and a[3] else None, a[4] if len(a) > 4 and a[4] else None)
    elif a[0] == "peek" and len(a) >= 2: cmd_peek(a[1], int(a[2]) if len(a) > 2 else 3)
    else: print(__doc__); sys.exit(1)
