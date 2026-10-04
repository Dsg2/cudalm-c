"""Optimised, resumable, streaming corpus builder for cudalm (v3).

Improvements over prepare_mix.py (which wrote train.bin only once at the very end,
so any crash forfeited the whole run):

  * RESUMABLE   -- each source writes an append-only work/<cat>__<name>.u16 and a
                   per-shard checkpoint. A crash loses only the in-flight shard;
                   a rerun skips finished shards and continues appending.
  * PARALLEL    -- a producer thread downloads shard i+1 while the main thread
                   tokenises shard i (network and CPU overlap).
  * BATCHED     -- tok.encode_batch() (multi-core Rust) instead of per-doc encode.
  * BOUNDED RAM -- nothing accumulates in a Python list; tokens stream to disk.
                   The final interleave memmaps the work files and copies blocks.
  * PROVENANCE  -- per-source token counts + licences recorded in meta.json.

Reuses an EXISTING tokenizer only (MIX_TOKENIZER=...); this tool never trains a
BPE. Wire format is identical to prepare_mix.py: uint16, each doc ends in EOT,
chatml markers as plain text.

Env:
  MIX_PROFILE       v3pre | v3sft            (selects blend + sources)
  MIX_TOKENIZER     path to tokenizer.json   (required)
  MIX_TARGET_TOKENS integer, e.g. 6000000000
  MIX_MAX_REPEAT    float (default: 1.0 for v3pre, 2.0 for v3sft)
  MIX_OUT           output dir (train.bin/val.bin/meta.json/template.txt)
  MIX_WORK          scratch dir for per-source .u16 + checkpoint (default OUT/work)
  MIX_KEEP_WORK=1   keep work/ after interleave (default: delete)

Run:  python -u build_corpus.py
"""
import os, sys, json, gzip, time, re, threading, queue, urllib.request, urllib.error
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _abs(p, default):
    p = (os.environ.get(p, "").strip() or default)
    return p if os.path.isabs(p) else os.path.join(HERE, p)

PROFILE       = os.environ.get("MIX_PROFILE", "").strip()
TOKENIZER     = os.environ.get("MIX_TOKENIZER", "").strip()
TARGET_TOKENS = int(os.environ.get("MIX_TARGET_TOKENS", 1_000_000_000))
OUT           = _abs("MIX_OUT", os.path.join("data", PROFILE or "mix"))
WORK          = _abs("MIX_WORK", os.path.join(OUT, "work"))
RAW           = os.path.join(HERE, "data", "raw")
KEEP_WORK     = os.environ.get("MIX_KEEP_WORK", "") == "1"
EOT           = "<|endoftext|>"
SLACK         = 1.15
MB            = 1024 * 1024
BLOCK         = 1 << 16          # interleave granularity, tokens
BATCH         = 1000            # docs per encode_batch call
FLUSH_BYTES   = 8 * MB
os.makedirs(OUT, exist_ok=True); os.makedirs(WORK, exist_ok=True); os.makedirs(RAW, exist_ok=True)

# ----------------------------- markers / render ---------------------------
M = dict(system="<|system|>\n", tools="<|tools|>\n", user="<|user|>\n",
         assistant="<|assistant|>\n", call="<|call|>\n", result="<|result|>\n")

def render(turns):
    parts = []
    for role, content in turns:
        content = (content or "").strip()
        if content: parts.append(M[role] + content)
    return "\n".join(parts) if len(parts) >= 2 else None

def raw_doc(text):
    text = (text or "").strip()
    return text if len(text) > 200 else None

ROLE = {"system": "system", "human": "user", "user": "user", "prompter": "user",
        "gpt": "assistant", "assistant": "assistant", "chatgpt": "assistant",
        "tool": "result", "observation": "result", "function_response": "result",
        "function_call": "call", "tool_call": "call"}

def as_list(v):
    if isinstance(v, str):
        try: return json.loads(v)
        except Exception: return None
    return v

def conv_turns(convs, tools=None, system=None):
    convs = as_list(convs); turns = []
    if system: turns.append(("system", system))
    if tools:  turns.append(("tools", tools if isinstance(tools, str) else json.dumps(tools, ensure_ascii=False)))
    for c in convs or []:
        if not isinstance(c, dict): continue
        r = (c.get("from") or c.get("role") or "").strip().lower()
        v = c.get("value") or c.get("content") or ""
        r = ROLE.get(r)
        if r and v: turns.append((r, v))
    return turns

# ----------------------------- download -----------------------------------
def download(url, dest, max_bytes=None, retries=6):
    if max_bytes and os.path.exists(dest) and os.path.getsize(dest) >= max_bytes:
        return dest
    for attempt in range(retries):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        end = f"{max_bytes-1}" if max_bytes else ""
        try:
            _h = {"Range": f"bytes={have}-{end}", "User-Agent": "Mozilla/5.0"}
            if os.environ.get("HF_TOKEN"): _h["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
            req = urllib.request.Request(url, headers=_h)
            r = urllib.request.urlopen(req, timeout=180)
            if have > 0 and getattr(r, "status", 206) != 206: have = 0
            cr = r.headers.get("Content-Range", "")
            full = (int(cr.split("/")[-1]) if "/" in cr and cr.rsplit("/", 1)[-1].isdigit()
                    else have + int(r.headers.get("Content-Length", 0)))
            target = min(full, max_bytes) if max_bytes else full
            with open(dest, "ab" if have else "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk: break
                    f.write(chunk)
                    if max_bytes and f.tell() >= max_bytes: break
            r.close(); have = os.path.getsize(dest)
            if have >= target:
                print(f"  dl {os.path.basename(dest)} {have/1e6:.0f}MB", flush=True); return dest
        except urllib.error.HTTPError as e:
            if e.code == 416 and os.path.exists(dest) and os.path.getsize(dest) > 0: return dest
            print(f"  retry {attempt+1}/{retries} HTTP{e.code} {os.path.basename(dest)}", flush=True); time.sleep(3)
        except Exception as e:
            print(f"  retry {attempt+1}/{retries} {type(e).__name__} {os.path.basename(dest)}", flush=True); time.sleep(3)
    return dest if os.path.exists(dest) else None

# ------------------------- tolerant readers -------------------------------
def iter_jsonl(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: yield json.loads(line)
            except json.JSONDecodeError: pass

def iter_json_array(path):
    raw = open(path, encoding="utf-8", errors="ignore").read()
    dec = json.JSONDecoder(); i = raw.find("[") + 1; n = len(raw)
    while i < n:
        while i < n and raw[i] in " \t\r\n,": i += 1
        if i >= n or raw[i] == "]": break
        try: obj, i = dec.raw_decode(raw, i)
        except json.JSONDecodeError: break
        yield obj

def iter_auto(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        head = f.read(4096).lstrip()
    return iter_json_array(path) if head.startswith("[") else iter_jsonl(path)

def iter_parquet(path, col):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    if col not in pf.schema_arrow.names: return
    for batch in pf.iter_batches(batch_size=512, columns=[col]):
        for v in batch.column(0):
            s = v.as_py()
            if s: yield s

def iter_parquet_rows(path, cols):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    have = [c for c in cols if c in pf.schema_arrow.names]
    if not have: return
    for batch in pf.iter_batches(batch_size=256, columns=have):
        for row in batch.to_pylist(): yield row

# ------------------------------ parsers -----------------------------------
def src_cosmopedia(p):
    for t in iter_parquet(p, "text"):
        d = raw_doc(t)
        if d: yield d

def src_content(p):
    for t in iter_parquet(p, "content"):
        d = raw_doc(t)
        if d: yield d

def src_text(p):                      # generic single "text" column, len>100
    for t in iter_parquet(p, "text"):
        t = (t or "").strip()
        if len(t) > 100: yield t

def src_code(p):
    for c in iter_parquet(p, "code"):
        c = (c or "").strip()
        if len(c) > 100: yield c

def src_conv_pq(p):
    for r in iter_parquet_rows(p, ["conversations"]):
        d = render(conv_turns(r.get("conversations")))
        if d: yield d

def src_conv_tools_pq(p):
    for r in iter_parquet_rows(p, ["conversations", "tools"]):
        d = render(conv_turns(r.get("conversations"), tools=r.get("tools")))
        if d: yield d

def src_pq_messages(p):
    for r in iter_parquet_rows(p, ["messages"]):
        d = render(conv_turns(r.get("messages")))
        if d: yield d

def src_conversations(p):
    for r in iter_auto(p):
        d = render(conv_turns(r.get("conversations")))
        if d: yield d

def src_toolace(p):
    for r in iter_auto(p):
        d = render(conv_turns(r.get("conversations"), system=r.get("system")))
        if d: yield d

TOOLCALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
def src_hermes_fc(p):
    for r in iter_auto(p):
        turns = []
        for c in r.get("conversations") or []:
            role = ROLE.get((c.get("from") or "").lower()); v = c.get("value") or ""
            if not role or not v: continue
            if role == "assistant" and "<tool_call>" in v:
                pos = 0
                for m in TOOLCALL_TAG.finditer(v):
                    pre = v[pos:m.start()].strip()
                    if pre: turns.append(("assistant", pre))
                    turns.append(("call", m.group(1))); pos = m.end()
                tail = v[pos:].strip()
                if tail: turns.append(("assistant", tail))
            else:
                turns.append((role, v))
        d = render(turns)
        if d: yield d

def src_apigen(p):
    for r in iter_auto(p):
        d = render(conv_turns(r.get("conversations"), tools=r.get("tools"), system=r.get("system")))
        if d: yield d

GLAIVE_SPLIT = re.compile(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)")
def src_glaive_v2(p):
    for r in iter_auto(p):
        sysmsg = (r.get("system") or "").strip()
        if sysmsg.startswith("SYSTEM:"): sysmsg = sysmsg[7:].strip()
        chat = (r.get("chat") or "").replace("<|endoftext|>", " ")
        turns = []
        if sysmsg: turns.append(("system", sysmsg))
        parts = GLAIVE_SPLIT.split(chat); i = 1
        while i < len(parts):
            label = parts[i]; content = (parts[i+1] if i+1 < len(parts) else "").strip(); i += 2
            if not content: continue
            if label == "USER:": turns.append(("user", content))
            elif label == "FUNCTION RESPONSE:": turns.append(("result", content))
            else:  # ASSISTANT:
                if "<functioncall>" in content:
                    pre, _, call = content.partition("<functioncall>")
                    pre = pre.strip()
                    if pre: turns.append(("assistant", pre))
                    turns.append(("call", call.strip()))
                else:
                    turns.append(("assistant", content))
        d = render(turns)
        if d: yield d

def src_xlam(p):
    for r in iter_auto(p):
        turns = []
        tl = r.get("tools")
        if tl: turns.append(("tools", tl if isinstance(tl, str) else json.dumps(tl, ensure_ascii=False)))
        if r.get("query"): turns.append(("user", r["query"]))
        an = r.get("answers")
        if an: turns.append(("call", an if isinstance(an, str) else json.dumps(an, ensure_ascii=False)))
        d = render(turns)
        if d: yield d

# ============================ source tables ===============================
HF     = "https://huggingface.co/datasets"
GHCODE = "https://huggingface.co/api/datasets/codeparrot/github-code-clean/parquet/Python-all/train"
_COSMO = f"{HF}/HuggingFaceTB/cosmopedia/resolve/main/data/web_samples_v2"
_FMATH = f"{HF}/HuggingFaceTB/finemath/resolve/main/finemath-4plus"
_TULU  = f"{HF}/allenai/tulu-3-sft-mixture/resolve/main/data"
_UFW   = f"{HF}/openbmb/Ultra-FineWeb/resolve/main/data/ultrafineweb_en"
_UFWL3 = f"{HF}/openbmb/Ultra-FineWeb-L3/resolve/main/data/ultrafineweb_en_l3/multi_style"
_L3U   = "f36f5a53-4a77-434b-b9bc-67ed69b93fe2"
_OT3   = f"{HF}/open-thoughts/OpenThoughts3-1.2M/resolve/main/data"
_ST2   = f"{HF}/HuggingFaceTB/smoltalk2/resolve/main/SFT"
_HRTU  = f"{HF}/interstellarninja/hermes_reasoning_tool_use/resolve/main/data"

COSMO_SHARDS = [f"{_COSMO}/train-{i:05d}-of-00118.parquet"      for i in range(30)]
FMATH_SHARDS = [f"{_FMATH}/train-{i:05d}-of-00064.parquet"      for i in range(40)]
GHCODE_SHARDS= [f"{GHCODE}/{i}.parquet"                         for i in range(16)]
TULU_SHARDS  = [f"{_TULU}/train-{i:05d}-of-00006.parquet"       for i in range(6)]
UFW_SHARDS   = [f"{_UFW}/ultrafineweb-en-part-{i:04d}-of-2048.parquet" for i in range(1, 41)]
UFWL3_SHARDS = [f"{_UFWL3}/part-{i:05d}-{_L3U}-c000.snappy.parquet"    for i in range(30)]
OT3_SHARDS   = [f"{_OT3}/train-{i:05d}-of-00120.parquet"        for i in range(12)]
ST2_SHARDS   = [f"{_ST2}/OpenHermes_2.5_no_think-{i:05d}-of-00002.parquet" for i in range(2)]

HERMES25 = (f"{HF}/teknium/OpenHermes-2.5/resolve/main/openhermes2_5.json", "hermes25.json")
TOOLACE  = (f"{HF}/Team-ACE/ToolACE/resolve/main/data.json", "toolace.json")
HERMESFC = (f"{HF}/NousResearch/hermes-function-calling-v1/resolve/main/func-calling.json", "hermes_fc_multi.json")
GLAIVEV2 = (f"{HF}/glaiveai/glaive-function-calling-v2/resolve/main/glaive-function-calling-v2.json", "glaive_v2.json")
HERMESSGL= (f"{HF}/NousResearch/hermes-function-calling-v1/resolve/main/func-calling-singleturn.json", "hermes_single.json")
HERMESGL5K=(f"{HF}/NousResearch/hermes-function-calling-v1/resolve/main/glaive-function-calling-5k.json", "hermes_gl5k.json")
APIGENMT = (f"{HF}/Salesforce/APIGen-MT-5k/resolve/main/apigen-mt_5k.json", "apigen_mt.json")
XLAM     = (f"{HF}/Salesforce/xlam-function-calling-60k/resolve/main/xlam_function_calling_60k.json", "xlam60k.json")
_UCHAT   = f"{HF}/HuggingFaceH4/ultrachat_200k/resolve/main/data"
UCHAT_SHARDS = [
    f"{_UCHAT}/train_sft-00000-of-00003-a3ecf92756993583.parquet",
    f"{_UCHAT}/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
    f"{_UCHAT}/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet",
]

# (cat, name, urls, fname, cap_bytes, parser)
V3_PRE = dict(
  weights={"web":0.34, "code":0.20, "synthetic":0.16, "math":0.14, "chat":0.10, "tool":0.06},
  max_repeat=1.0,
  sources=[
    ("web",       "ultrafineweb",  UFW_SHARDS,    "ufw.parquet",      None, src_content),
    ("code",      "github-python", GHCODE_SHARDS, "ghpy.parquet",     None, src_code),
    ("synthetic", "cosmopedia-v2", COSMO_SHARDS,  "cosmo_v2.parquet", None, src_cosmopedia),
    ("synthetic", "ufw-l3",        UFWL3_SHARDS,  "ufwl3.parquet",    None, src_content),
    ("math",      "finemath",      FMATH_SHARDS,  "finemath.parquet", None, src_cosmopedia),
    ("chat",      "hermes2.5",     HERMES25[0],   HERMES25[1],        None, src_conversations),
    ("chat",      "ultrachat",     UCHAT_SHARDS,  "uchat.parquet",    None, src_pq_messages),
    ("tool",      "toolace",       TOOLACE[0],    TOOLACE[1],         None, src_toolace),
    ("tool",      "hermes-fc",     HERMESFC[0],   HERMESFC[1],        None, src_hermes_fc),
    ("tool",      "glaive-v2",     GLAIVEV2[0],   GLAIVEV2[1],        None, src_glaive_v2),
    ("tool",      "hermes-single", HERMESSGL[0],  HERMESSGL[1],       None, src_hermes_fc),
    ("tool",      "hermes-gl5k",   HERMESGL5K[0], HERMESGL5K[1],      None, src_hermes_fc),
    ("tool",      "apigen-mt",     APIGENMT[0],   APIGENMT[1],        None, src_conversations),
  ] + ([("tool", "xlam60k", XLAM[0], XLAM[1], None, src_xlam)] if os.environ.get("HF_TOKEN") else []))

V3_SFT = dict(
  weights={"general":0.24, "chat":0.20, "code":0.18, "reasoning":0.14, "tool":0.14, "agentic":0.10},
  max_repeat=2.0,
  sources=[
    ("general",  "tulu3",         TULU_SHARDS,   "tulu3.parquet",  None, src_pq_messages),
    ("chat",     "smoltalk2",     ST2_SHARDS,    "st2.parquet",    None, src_pq_messages),
    ("code",     "github-python", GHCODE_SHARDS, "ghpy.parquet",   None, src_code),
    ("reasoning","openthoughts3", OT3_SHARDS,    "ot3.parquet",    None, src_conv_pq),
    ("tool",     "toolace",       TOOLACE[0],    TOOLACE[1],       None, src_toolace),
    ("tool",     "hermes-fc",     HERMESFC[0],   HERMESFC[1],      None, src_hermes_fc),
    ("agentic",  "hermes-rtu",    f"{_HRTU}/train-00000-of-00001.parquet", "hermes_rtu.parquet", None, src_conv_tools_pq),
  ])

V3_CHAT = dict(
  weights={"general":0.62, "chat":0.38},
  max_repeat=1.3,
  sources=[s for s in V3_SFT["sources"] if s[1] in ("tulu3","smoltalk2")])

LICENSES = {
  "ultrafineweb":  "openbmb/Ultra-FineWeb (ODC-By-1.0; derived from FineWeb/CommonCrawl)",
  "ufw-l3":        "openbmb/Ultra-FineWeb-L3 (ODC-By-1.0; synthetic-styled web)",
  "github-python": "codeparrot/github-code-clean (per-file OSS licences, permissive subset)",
  "cosmopedia-v2": "HuggingFaceTB/cosmopedia (Apache-2.0; synthetic)",
  "finemath":      "HuggingFaceTB/finemath (ODC-By-1.0)",
  "hermes2.5":     "teknium/OpenHermes-2.5 (research/mixed source licences)",
  "toolace":       "Team-ACE/ToolACE (Apache-2.0)",
  "hermes-fc":     "NousResearch/hermes-function-calling-v1 (Apache-2.0)",
  "hermes-single": "NousResearch/hermes-function-calling-v1 (Apache-2.0)",
  "hermes-gl5k":   "NousResearch/hermes-function-calling-v1 (Apache-2.0)",
  "glaive-v2":     "glaiveai/glaive-function-calling-v2 (Apache-2.0)",
  "apigen-mt":     "Salesforce/APIGen-MT-5k (CC-BY-4.0)",
  "xlam60k":       "Salesforce/xlam-function-calling-60k (CC-BY-4.0, gated)",
  "ultrachat":     "HuggingFaceH4/ultrachat_200k (MIT)",
  "tulu3":         "allenai/tulu-3-sft-mixture (ODC-By-1.0)",
  "smoltalk2":     "HuggingFaceTB/smoltalk2 (Apache-2.0)",
  "openthoughts3": "open-thoughts/OpenThoughts3-1.2M (Apache-2.0)",
  "hermes-rtu":    "interstellarninja/hermes_reasoning_tool_use (Apache-2.0)",
}

BLEND = {"v3pre": V3_PRE, "v3sft": V3_SFT, "v3chat": V3_CHAT}.get(PROFILE)
if BLEND is None:
    sys.exit(f"MIX_PROFILE must be v3pre or v3sft (got {PROFILE!r})")
WEIGHTS   = BLEND["weights"]
SOURCES   = BLEND["sources"]
MAX_REPEAT = float(os.environ.get("MIX_MAX_REPEAT", BLEND["max_repeat"]))
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, f"{PROFILE} weights sum != 1"

# ------------------------------ tokenizer ---------------------------------
if not TOKENIZER or not os.path.exists(TOKENIZER):
    sys.exit(f"MIX_TOKENIZER not found: {TOKENIZER!r}")
from tokenizers import Tokenizer
tok = Tokenizer.from_file(TOKENIZER)
eot = tok.token_to_id(EOT)
if eot is None: sys.exit(f"tokenizer has no {EOT}")
VOCAB = tok.get_vocab_size()
print(f"[tok] {TOKENIZER} vocab={VOCAB} eot={eot}", flush=True)

# ------------------------------ checkpoint --------------------------------
CKPATH = os.path.join(WORK, "checkpoint.json")
def load_ck():
    try: return json.load(open(CKPATH))
    except Exception: return {}
def save_ck(ck):
    tmp = CKPATH + ".tmp"
    json.dump(ck, open(tmp, "w"))
    os.replace(tmp, CKPATH)

def batched(it, n):
    b = []
    for x in it:
        b.append(x)
        if len(b) >= n: yield b; b = []
    if b: yield b

# ---------------------- per-source download+encode ------------------------
def build_source(cat, name, urls, fname, cap, parser, budget, ck):
    outpath = os.path.join(WORK, f"{cat}__{name}.u16")
    st = ck.get(name) or {"shards": 0, "tokens": 0, "done": False}
    if st.get("done") and os.path.exists(outpath):
        print(f"[skip] {name}: done, {st['tokens']/1e6:.1f}M tok", flush=True)
        return st["tokens"], outpath
    # truncate any partial tail written after the last checkpoint
    if os.path.exists(outpath):
        with open(outpath, "r+b") as f: f.truncate(st["tokens"] * 2)
    else:
        st = {"shards": 0, "tokens": 0, "done": False}
    ck[name] = st
    urls = urls if isinstance(urls, list) else [urls]

    q = queue.Queue(maxsize=2); stop = threading.Event()
    def producer():
        for si in range(st["shards"], len(urls)):
            if stop.is_set(): break
            shard = fname if len(urls) == 1 else f"{si:04d}_{fname}"
            p = download(urls[si], os.path.join(RAW, shard), max_bytes=cap)
            q.put((si, p))
        q.put(None)
    t = threading.Thread(target=producer, daemon=True); t.start()

    fout = open(outpath, "ab"); got = st["tokens"]; t0 = time.time()
    print(f"[get ] {name} ({cat}) budget {budget/1e6:.0f}M tok  from shard {st['shards']}", flush=True)
    while got < budget:
        item = q.get()
        if item is None: break
        si, p = item
        if p:
            buf = bytearray()
            for grp in batched(parser(p), BATCH):
                for e in tok.encode_batch(grp):
                    ids = e.ids
                    a = np.empty(len(ids) + 1, dtype=np.uint16); a[:-1] = ids; a[-1] = eot
                    buf += a.tobytes(); got += len(a)
                if len(buf) >= FLUSH_BYTES:
                    fout.write(buf); buf.clear()
                if got >= budget: break
            if buf: fout.write(buf)
            try: os.remove(p)
            except OSError: pass
        fout.flush(); os.fsync(fout.fileno())
        st["shards"] = si + 1; st["tokens"] = got; save_ck(ck)
        print(f"       {name} shard{si} -> {got/1e6:.1f}/{budget/1e6:.0f}M tok  {time.time()-t0:.0f}s", flush=True)
        if got >= budget: break
    stop.set()
    # drain the queue so the producer thread can exit; delete any prefetched raw
    while True:
        try: item = q.get_nowait()
        except queue.Empty: break
        if item and item[1]:
            try: os.remove(item[1])
            except OSError: pass
        if item is None: break
    st["done"] = True; st["tokens"] = got; save_ck(ck)
    fout.close()
    return got, outpath

# ------------------------------- collect ----------------------------------
nsrc = {c: sum(1 for s in SOURCES if s[0] == c) for c in WEIGHTS}
def srccap(cat):
    b = TARGET_TOKENS * WEIGHTS[cat] * SLACK
    return int(min(b, b * 2.0 / max(1, nsrc[cat])))

print(f"[plan] {PROFILE}: {TARGET_TOKENS/1e6:.0f}M tok, max_repeat {MAX_REPEAT}", flush=True)
for c, w in WEIGHTS.items():
    print(f"       {c:9s} {w*100:4.1f}%  {TARGET_TOKENS*w/1e6:5.0f}M tok  ({nsrc[c]} src, cap {srccap(c)/1e6:.0f}M each)", flush=True)

ck = load_ck()
cats = {c: [] for c in WEIGHTS}       # cat -> [(name, path, tokens)]
persrc = {}
for cat, name, urls, fname, cap, parser in SOURCES:
    try:
        got, path = build_source(cat, name, urls, fname, cap, parser, srccap(cat), ck)
        if got > 0: cats[cat].append((name, path, got))
        persrc[name] = got
        if got == 0: print(f"       !! {name} produced ZERO tokens", flush=True)
    except Exception as e:
        persrc[name] = 0
        print(f"       ! {name} failed: {type(e).__name__} {e}", flush=True)

if not any(cats.values()): sys.exit("no data collected")

# ------------------------------ interleave --------------------------------
# Weighted round-robin over memmapped work files. Categories are drawn to hit
# WEIGHTS proportions; within a category sources are drawn by tokens contributed
# (furthest-behind first), wrapping for repeat up to tokens*MAX_REPEAT.
print("\n[interleave]", flush=True)
train = open(os.path.join(OUT, "train.bin"), "wb")
val   = open(os.path.join(OUT, "val.bin"), "wb")
val_target = max(BLOCK, int(TARGET_TOKENS * 0.005)); val_written = 0

class Cat:
    __slots__ = ("mms", "lens", "caps", "pos", "contrib", "want", "emitted", "dead")
    def __init__(self, srcs, want):
        self.mms  = [np.memmap(p, dtype=np.uint16, mode="r") for _, p, _ in srcs]
        self.lens = [len(m) for m in self.mms]
        self.caps = [int(l * MAX_REPEAT) for l in self.lens]
        self.pos  = [0] * len(srcs); self.contrib = [0] * len(srcs)
        self.want = want; self.emitted = 0; self.dead = False
    def next_block(self):
        cand = [j for j in range(len(self.mms)) if self.pos[j] < self.caps[j] and self.lens[j]]
        if not cand: self.dead = True; return None
        j = min(cand, key=lambda x: self.contrib[x])
        off = self.pos[j] % self.lens[j]
        blk = self.mms[j][off: off + BLOCK]
        if len(blk) == 0: self.pos[j] = self.caps[j]; return self.next_block()
        self.pos[j] += len(blk); self.contrib[j] += len(blk); self.emitted += len(blk)
        return blk

C = {}
for cat in WEIGHTS:
    if cats[cat]:
        C[cat] = Cat(cats[cat], int(TARGET_TOKENS * WEIGHTS[cat]))

written = 0
while written < TARGET_TOKENS:
    live = [c for c in C.values() if not c.dead and c.emitted < c.want]
    if not live: break
    c = min(live, key=lambda x: x.emitted / x.want)
    blk = c.next_block()
    if blk is None: continue
    if val_written < val_target:                 # fill val.bin first
        take = min(len(blk), val_target - val_written)
        blk[:take].tofile(val); val_written += take
        if take < len(blk): blk[take:].tofile(train)
    else:
        blk.tofile(train)
    written += len(blk)
train.close(); val.close()

# ------------------------------ meta --------------------------------------
report = {}
for cat in WEIGHTS:
    if cat in C:
        c = C[cat]
        report[cat] = dict(
            emitted=int(c.emitted), want=int(c.want),
            pct=round(c.emitted / max(1, written) * 100, 2),
            sources={cats[cat][j][0]: dict(unique=int(c.lens[j]), emitted=int(c.contrib[j]),
                                           repeat=round(c.contrib[j] / max(1, c.lens[j]), 2))
                     for j in range(len(cats[cat]))})
    else:
        report[cat] = dict(emitted=0, want=int(TARGET_TOKENS * WEIGHTS[cat]), pct=0.0, sources={})

json.dump(dict(profile=PROFILE, vocab_size=VOCAB, eot_id=eot,
               total_tokens=int(written), train_tokens=int(written - val_written),
               val_tokens=int(val_written), target_tokens=TARGET_TOKENS,
               max_repeat=MAX_REPEAT, weights=WEIGHTS, tokenizer=TOKENIZER,
               categories=report,
               sources={n: dict(tokens=int(t), license=LICENSES.get(n, "see dataset card"))
                        for n, t in persrc.items()}),
          open(os.path.join(OUT, "meta.json"), "w"), indent=2)
open(os.path.join(OUT, "template.txt"), "w").write("chatml")
import shutil as _sh
_sh.copy(TOKENIZER, os.path.join(OUT, "tokenizer.json"))
try:
    sys.path.insert(0, HERE)
    from tokbin import write_tokenizer_bin
    write_tokenizer_bin(OUT)
except Exception as e:
    print(f"[warn] tokbin: {e}", flush=True)

if not KEEP_WORK:
    for cat in WEIGHTS:
        for _, p, _ in cats.get(cat, []):
            try: os.remove(p)
            except OSError: pass
    try: os.remove(CKPATH)
    except OSError: pass

print(f"\n[done] {OUT}", flush=True)
print(f"  {written:,} tokens -> train {written-val_written:,} / val {val_written:,}", flush=True)
for cat, r in report.items():
    print(f"  {cat:9s} {r['pct']:5.1f}%  ({len(r['sources'])} src)", flush=True)
