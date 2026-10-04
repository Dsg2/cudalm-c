"""Build a balanced general/chat/instruct/code/tool/agentic corpus for tinylm.

Six categories, each capped to a token share so no single source dominates:

  general knowledge  Cosmopedia-100k (synthetic textbooks), SQuAD v2, Dolly-15k,
                     OpenHermes-2.5 (prefix)
  multi-turn chat    ShareGPT-V3 (prefix), OASST1, OASST2  (reply trees)
  instruct           Alpaca-cleaned, WizardLM-evol-70k
  code               Evol-CodeAlpaca, Magicoder-OSS, glaive-code-assistant, CodeAlpaca
  tool calling       glaive-function-calling-v2, Hermes-FC (single+multi), ToolACE
  agentic            APIGen-MT-5k, Agent-FLAN (x5), Gorilla APIBench (x3)

Every record schema here was verified against a live prefix of the real file
before this script was written -- see the notes on each parser.

Wire format (plain text, NOT BPE special tokens): the C tokenizer in tok_encode
is pure byte-level BPE with no special-token path, so a special <|user|> would
tokenize differently at inference than in training and silently break parity.
As frequent plain text, BPE merges each marker to 1-3 tokens anyway.

    <|system|>\n...\n<|tools|>\n[{...}]\n<|user|>\n...\n<|assistant|>\n...
    <|call|>\n{"name":..,"arguments":{..}}\n<|result|>\n...\n<|assistant|>\n...

Prose sources (Cosmopedia) are emitted raw, with no markers -- the model needs
continuous text for factual fluency, not only Q&A pairs.

Writes to data/mix/: tokenizer.json, tokenizer.bin, train.bin, val.bin,
meta.json, template.txt ("chatml", which the C side reads as tmpl=2).

Run:  python prepare_mix.py            (needs: tokenizers, numpy, pyarrow)
"""
import os, sys, json, gzip, random, time, re, array, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.environ.get("MIX_OUT", "").strip() or os.path.join(HERE, "data", "mix")
if not os.path.isabs(OUT): OUT = os.path.join(HERE, OUT)
RAW  = os.path.join(HERE, "data", "raw")
os.makedirs(OUT, exist_ok=True); os.makedirs(RAW, exist_ok=True)

# ----------------------------- config ------------------------------------
TARGET_TOKENS = int(os.environ.get("MIX_TARGET_TOKENS", 1_000_000_000))
VOCAB         = int(os.environ.get("MIX_VOCAB", 8192))
EOT           = "<|endoftext|>"
CHARS_PER_TOK = 3.5      # byte-BPE @8k on mixed English+code; used only for budgeting
SLACK         = 1.15     # collect a little extra so the shuffle has choices
MAX_REPEAT    = float(os.environ.get("MIX_MAX_REPEAT", "2.0"))  # cap oversampling; set 1.0 for a fully-unique build
SEED          = 0
INCLUDE_WIKIPEDIA = os.environ.get("MIX_WIKIPEDIA", "") == "1"   # +420MB parquet

# Post-training modes (see POSTTRAIN_PLAN.md).
#   MIX_REUSE_TOK=<tokenizer.json>  reuse an existing tokenizer instead of training
#     a fresh BPE. MANDATORY when continuing an existing checkpoint: a retrained BPE
#     silently remaps token<->text and corrupts the model (V stays 8192, so `train`
#     won't even error). Point this at the tokenizer.json that built the checkpoint.
#   MIX_STREAM=1  tokenize each source as it is parsed and DELETE its raw file before
#     the next, so the raw corpus is never fully materialised on disk (the 4B build
#     needs this: 8GB train.bin barely fits a small disk). Requires
#     MIX_REUSE_TOK, since BPE training would need the whole text corpus buffered.
REUSE_TOK = os.environ.get("MIX_REUSE_TOK", "").strip()
STREAM    = os.environ.get("MIX_STREAM", "") == "1"

# Post-training mix (POSTTRAIN_PLAN.md 2): 4B unique tokens, quality over volume.
# Dominated by clean web/edu prose + synthetic textbook/math, which the 1B mix
# lacked entirely -- that mix was all instruction/chat/tool turns and never taught
# broad world knowledge, so MMLU stayed at chance. Every source here is large
# enough to fill its share without oversampling (build runs MIX_MAX_REPEAT=1.0).
WEIGHTS = {
    "general":   0.40,   # fineweb-edu: clean educational web prose
    "code":      0.20,   # github-code-clean Python (raw source, not Q&A)
    "chat":      0.175,  # OpenHermes-2.5 + tulu-3 instruction/chat
    "synthetic": 0.15,   # cosmopedia textbooks + finemath math prose
    "tool":      0.075,  # ToolACE + hermes-fc + glaive-fc2 function calling
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

MB = 1024 * 1024

# markers
M = dict(system="<|system|>\n", tools="<|tools|>\n", user="<|user|>\n",
         assistant="<|assistant|>\n", call="<|call|>\n", result="<|result|>\n")

def render(turns):
    """turns: list of (role, content). Returns the wire-format document."""
    parts = []
    for role, content in turns:
        content = (content or "").strip()
        if not content:
            continue
        parts.append(M[role] + content)
    if len(parts) < 2:
        return None
    return "\n".join(parts)

def raw_doc(text):
    """Prose passthrough (no chat markers)."""
    text = (text or "").strip()
    return text if len(text) > 200 else None

# ----------------------------- download -----------------------------------
def download(url, dest, max_bytes=None, retries=6):
    """Resumable range download; stops at max_bytes if given."""
    if max_bytes and os.path.exists(dest) and os.path.getsize(dest) >= max_bytes:
        print(f"  {os.path.basename(dest)}: cached", flush=True); return dest
    for attempt in range(retries):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        end = f"{max_bytes-1}" if max_bytes else ""
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-{end}",
                                                       "User-Agent": "Mozilla/5.0"})
            r = urllib.request.urlopen(req, timeout=180)
            if have > 0 and getattr(r, "status", 206) != 206:
                have = 0
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
            r.close()
            have = os.path.getsize(dest)
            if have >= target:
                print(f"  {os.path.basename(dest)}: {have/1e6:.0f} MB", flush=True); return dest
            print(f"  {os.path.basename(dest)}: {have/1e6:.0f}/{target/1e6:.0f} MB, resuming...", flush=True)
        except urllib.error.HTTPError as e:
            if e.code == 416 and os.path.exists(dest) and os.path.getsize(dest) > 0:
                print(f"  {os.path.basename(dest)}: complete", flush=True); return dest
            print(f"  retry {attempt+1}/{retries} (HTTP {e.code})", flush=True); time.sleep(3)
        except Exception as e:
            sz = os.path.getsize(dest) if os.path.exists(dest) else 0
            print(f"  retry {attempt+1}/{retries} ({type(e).__name__}) at {sz/1e6:.0f} MB", flush=True)
            time.sleep(3)
    return dest if os.path.exists(dest) else None

# ------------------------- tolerant readers -------------------------------
def iter_jsonl(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: yield json.loads(line)
            except json.JSONDecodeError: pass        # drops a truncated tail line

def iter_gz_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: yield json.loads(line)
            except (json.JSONDecodeError, EOFError): pass

def iter_json_array(path):
    """Stream objects out of a JSON array, tolerating a truncated prefix download."""
    raw = open(path, encoding="utf-8", errors="ignore").read()
    dec = json.JSONDecoder()
    i = raw.find("[") + 1; n = len(raw)
    while i < n:
        while i < n and raw[i] in " \t\r\n,": i += 1
        if i >= n or raw[i] == "]": break
        try: obj, i = dec.raw_decode(raw, i)
        except json.JSONDecodeError: break
        yield obj

def iter_auto(path):
    """Sniff array-vs-JSONL rather than trusting the extension: several of these
    sources ship JSONL under a .json name (Gorilla APIBench does)."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        head = f.read(4096).lstrip()
    return iter_json_array(path) if head.startswith("[") else iter_jsonl(path)

def iter_parquet(path, col):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=512, columns=[col]):
        for v in batch.column(0):
            s = v.as_py()
            if s: yield s

def iter_parquet_rows(path, cols):
    """Row dicts for the multi-column parquet sources."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    have = [c for c in cols if c in pf.schema_arrow.names]
    if not have: return
    for batch in pf.iter_batches(batch_size=256, columns=have):
        for row in batch.to_pylist():
            yield row

# --------------------------- role mapping ---------------------------------
ROLE = {"system": "system", "human": "user", "user": "user", "prompter": "user",
        "gpt": "assistant", "assistant": "assistant", "chatgpt": "assistant",
        "tool": "result", "observation": "result", "function_response": "result",
        "function_call": "call", "tool_call": "call"}

def as_list(v):
    """Parquet columns frequently store conversation lists as JSON *strings*
    rather than native list-of-struct (orca-agentinstruct, wildchat and
    pythonic-fc all do). Iterating the raw string yields characters, not dicts,
    and the source silently produces zero documents -- so coerce here."""
    if isinstance(v, str):
        try: return json.loads(v)
        except Exception: return None
    return v

def conv_turns(convs, tools=None, system=None):
    """Normalise a [{from|role, value|content}] conversation list."""
    convs = as_list(convs)
    turns = []
    if system: turns.append(("system", system))
    if tools:
        turns.append(("tools", tools if isinstance(tools, str) else json.dumps(tools, ensure_ascii=False)))
    for c in convs or []:
        if not isinstance(c, dict): continue
        r = (c.get("from") or c.get("role") or "").strip().lower()
        v = c.get("value") or c.get("content") or ""
        r = ROLE.get(r)
        if r and v: turns.append((r, v))
    return turns

# ============================ source parsers ==============================
# Each yields rendered documents. Schemas verified against live prefixes.

def src_cosmopedia(p):                       # parquet: column "text" (prose)
    for t in iter_parquet(p, "text"):
        d = raw_doc(t)
        if d: yield d

def src_content(p):                          # parquet: column "content" (prose)
    for t in iter_parquet(p, "content"):     # Ultra-FineWeb / Ultra-FineWeb-L3
        d = raw_doc(t)
        if d: yield d

def src_conv_pq(p):                          # parquet: column "conversations"
    for r in iter_parquet_rows(p, ["conversations"]):     # OpenThoughts3
        d = render(conv_turns(r.get("conversations")))
        if d: yield d

def src_conv_tools_pq(p):                    # parquet: "conversations" + "tools"
    for r in iter_parquet_rows(p, ["conversations", "tools"]):   # hermes-rtu
        d = render(conv_turns(r.get("conversations"), tools=r.get("tools")))
        if d: yield d

def src_code(p):                             # parquet: column "code" (raw source)
    for c in iter_parquet(p, "code"):
        c = (c or "").strip()
        if len(c) > 100: yield c             # raw code, emitted without chat markers

def src_wikipedia(p):                        # parquet: "title","text" (prose)
    for t in iter_parquet(p, "text"):
        d = raw_doc(t)
        if d: yield d

def src_squad(p):
    """data[].paragraphs[].{context, qas[].{question, answers[].text, is_impossible}}
    One doc per paragraph = grounded multi-turn QA over a shared context."""
    obj = json.load(open(p, encoding="utf-8", errors="ignore"))
    for art in obj.get("data", []):
        for para in art.get("paragraphs", []):
            ctx = (para.get("context") or "").strip()
            qas = [q for q in para.get("qas", [])
                   if not q.get("is_impossible") and q.get("answers")]
            if not ctx or not qas: continue
            turns, first = [], True
            for q in qas[:6]:
                ans = (q["answers"][0].get("text") or "").strip()
                if not ans: continue
                qt = (q.get("question") or "").strip()
                turns.append(("user", f"{ctx}\n\nQuestion: {qt}" if first else qt))
                turns.append(("assistant", ans)); first = False
            d = render(turns)
            if d: yield d

def src_dolly(p):
    for r in iter_jsonl(p):
        i, c, o = r.get("instruction"), r.get("context"), r.get("response")
        u = f"{i}\n\n{c}" if c else i
        d = render([("user", u), ("assistant", o)])
        if d: yield d

def src_conversations(p):
    """OpenHermes-2.5 / ShareGPT: {conversations:[{from,value}]}"""
    for r in iter_auto(p):
        d = render(conv_turns(r.get("conversations")))
        if d: yield d

def src_oasst(p):
    """Reply-tree jsonl.gz. Root {prompt:{text,role,lang,replies[]}}; each reply
    has role prompter/assistant, text, rank (0 = best), replies[]. Walk the
    best-ranked chain to get one coherent multi-turn conversation."""
    for tree in iter_gz_jsonl(p):
        node = tree.get("prompt")
        if not isinstance(node, dict) or node.get("lang") != "en": continue
        turns = []
        while isinstance(node, dict):
            r = ROLE.get((node.get("role") or "").lower())
            t = (node.get("text") or "").strip()
            if r and t: turns.append((r, t))
            reps = [x for x in (node.get("replies") or []) if isinstance(x, dict)]
            if not reps: break
            reps.sort(key=lambda x: (x.get("rank") if x.get("rank") is not None else 99))
            node = reps[0]
        d = render(turns)
        if d: yield d

def src_alpaca(p):
    for r in iter_auto(p):
        i, inp, o = r.get("instruction"), r.get("input"), r.get("output")
        u = f"{i}\n\n{inp}" if inp else i
        d = render([("user", u), ("assistant", o)])
        if d: yield d

def src_instr_out(p):
    for r in iter_auto(p):
        d = render([("user", r.get("instruction")), ("assistant", r.get("output"))])
        if d: yield d

def src_magicoder(p):
    for r in iter_auto(p):
        d = render([("user", r.get("problem")), ("assistant", r.get("solution"))])
        if d: yield d

def src_glaive_code(p):
    for r in iter_auto(p):
        d = render([("user", r.get("question")),
                    ("assistant", r.get("answer") or r.get("response"))])
        if d: yield d

# --- tool calling ---------------------------------------------------------
GLAIVE_SPLIT = re.compile(r"(?m)^\s*(USER|ASSISTANT|FUNCTION RESPONSE):\s*")

def src_glaive_fc2(p):
    """{system: "SYSTEM: ...{schema}", chat: "USER: .. ASSISTANT: .. <functioncall>
    {..} FUNCTION RESPONSE: {..} ASSISTANT: .."}  -> split the chat string."""
    for r in iter_auto(p):
        sysmsg = (r.get("system") or "").strip()
        if sysmsg.startswith("SYSTEM:"): sysmsg = sysmsg[7:].strip()
        chat = r.get("chat") or ""
        parts = GLAIVE_SPLIT.split(chat)
        turns = [("system", sysmsg)] if sysmsg else []
        for k in range(1, len(parts) - 1, 2):
            who, body = parts[k], parts[k+1].replace("<|endoftext|>", "").strip()
            if not body: continue
            if who == "USER":
                turns.append(("user", body))
            elif who == "FUNCTION RESPONSE":
                turns.append(("result", body))
            else:                                   # ASSISTANT, maybe a call
                if "<functioncall>" in body:
                    pre, _, callpart = body.partition("<functioncall>")
                    if pre.strip(): turns.append(("assistant", pre.strip()))
                    turns.append(("call", callpart.strip()))
                else:
                    turns.append(("assistant", body))
        d = render(turns)
        if d: yield d

TOOLCALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)

def src_hermes_fc(p):
    """{conversations:[{from: system/human/gpt/tool, value}], tools:[...]}
    gpt turns may embed <tool_call>{...}</tool_call>; lift those to call turns
    so the whole corpus uses one convention."""
    for r in iter_auto(p):
        turns = []
        for c in r.get("conversations") or []:
            role = ROLE.get((c.get("from") or "").lower())
            v = c.get("value") or ""
            if not role or not v: continue
            if role == "assistant" and "<tool_call>" in v:
                pos = 0
                for m in TOOLCALL_TAG.finditer(v):
                    pre = v[pos:m.start()].strip()
                    if pre: turns.append(("assistant", pre))
                    turns.append(("call", m.group(1)))
                    pos = m.end()
                tail = v[pos:].strip()
                if tail: turns.append(("assistant", tail))
            else:
                turns.append((role, v))
        d = render(turns)
        if d: yield d

def src_toolace(p):
    for r in iter_auto(p):
        d = render(conv_turns(r.get("conversations"), system=r.get("system")))
        if d: yield d

def src_apigen(p):
    """{conversations:[{from: human/gpt/function_call/observation}], tools, system}"""
    for r in iter_auto(p):
        d = render(conv_turns(r.get("conversations"), tools=r.get("tools"),
                              system=r.get("system")))
        if d: yield d

# --- agentic --------------------------------------------------------------
def src_agentflan(p):
    """{conversation:[{role,content}], id}"""
    for r in iter_auto(p):
        d = render(conv_turns(r.get("conversation")))
        if d: yield d

def src_gorilla(p):
    """{code: "###Instruction: ..\\n###Output: ..", api_call, api_data}"""
    for r in iter_auto(p):
        code = r.get("code") or ""
        if "###Output:" not in code: continue
        instr, _, out = code.partition("###Output:")
        instr = instr.replace("###Instruction:", "").strip()
        d = render([("user", instr), ("assistant", out.strip())])
        if d: yield d

# --- added for the 1B build (schemas probed live before writing these) -------
def src_messages(p):
    """Code-Feedback: {id, messages:[{role,content}]} -- multi-turn code."""
    for r in iter_auto(p):
        d = render(conv_turns(r.get("messages")))
        if d: yield d

def src_instr_resp(p):
    """Magicoder-Evol-Instruct: {instruction, response}"""
    for r in iter_auto(p):
        d = render([("user", r.get("instruction")), ("assistant", r.get("response"))])
        if d: yield d

def src_pq_instr_resp(p):
    """self-oss-instruct parquet: instruction / response"""
    for r in iter_parquet_rows(p, ["instruction", "response"]):
        d = render([("user", r.get("instruction")), ("assistant", r.get("response"))])
        if d: yield d

def src_pq_messages(p):
    """orca-agentinstruct parquet: messages:[{role,content}]  (reasoning data,
    filed under `general` -- despite the dataset name these splits are brain
    teasers / MCQ / reading comprehension, not agent trajectories)."""
    for r in iter_parquet_rows(p, ["messages"]):
        d = render(conv_turns(r.get("messages")))
        if d: yield d

def src_apigen_query(p):
    """argilla/apigen-function-calling parquet: {query, tools, answers}, where
    answers is the list of calls. Rendered as a one-shot tool-call example."""
    for r in iter_parquet_rows(p, ["query", "tools", "answers"]):
        q, tools, ans = r.get("query"), r.get("tools"), r.get("answers")
        if not q or not ans: continue
        turns = []
        if tools: turns.append(("tools", tools if isinstance(tools, str) else json.dumps(tools, ensure_ascii=False)))
        turns.append(("user", q))
        turns.append(("call", ans if isinstance(ans, str) else json.dumps(ans, ensure_ascii=False)))
        d = render(turns)
        if d: yield d

def src_pythonic_fc(p):
    """driaforall/pythonic-function-calling parquet: {tools (python defs),
    conversations:[{role,content}]}"""
    for r in iter_parquet_rows(p, ["tools", "conversations"]):
        d = render(conv_turns(r.get("conversations"), tools=r.get("tools")))
        if d: yield d

def src_swegym(p):
    """SWE-Gym parquet: real repo issues + the patch that fixes them."""
    for r in iter_parquet_rows(p, ["repo", "problem_statement", "patch", "hints_text"]):
        prob, patch = r.get("problem_statement"), r.get("patch")
        if not prob or not patch: continue
        hint = (r.get("hints_text") or "").strip()
        u = f"Repository: {r.get('repo')}\n\n{prob}" + (f"\n\nHints:\n{hint}" if hint else "")
        d = render([("user", u), ("assistant", f"```diff\n{patch}\n```")])
        if d: yield d

def src_wildchat(p):
    """WildChat parquet: {conversation:[{role,content}], language, toxic}"""
    for r in iter_parquet_rows(p, ["conversation", "language", "toxic"]):
        if r.get("toxic"): continue
        if (r.get("language") or "English") != "English": continue
        d = render(conv_turns(r.get("conversation")))
        if d: yield d

# ============================ source table ================================
HF = "https://huggingface.co/datasets"
# Auto-converted parquet endpoint yields per-config (Python-filtered) shards for
# github-code-clean; the raw resolve/ files are 880 mixed-language shards instead.
GHCODE = "https://huggingface.co/api/datasets/codeparrot/github-code-clean/parquet/Python-all/train"
_FW    = f"{HF}/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT"
_COSMO = f"{HF}/HuggingFaceTB/cosmopedia/resolve/main/data/web_samples_v2"
_FMATH = f"{HF}/HuggingFaceTB/finemath/resolve/main/finemath-4plus"
_TULU  = f"{HF}/allenai/tulu-3-sft-mixture/resolve/main/data"

# A source's url may be a LIST of parquet shards. The collect loop downloads,
# parses and (in stream mode) deletes them one at a time, stopping as soon as the
# per-category cap is reached -- so listing more shards than needed is harmless.
FINEWEB_SHARDS = [f"{_FW}/{i:03d}_00000.parquet"                 for i in range(14)]
COSMO_SHARDS   = [f"{_COSMO}/train-{i:05d}-of-00118.parquet"     for i in range(30)]
FMATH_SHARDS   = [f"{_FMATH}/train-{i:05d}-of-00064.parquet"     for i in range(30)]
GHCODE_SHARDS  = [f"{GHCODE}/{i}.parquet"                        for i in range(10)]
TULU_SHARDS    = [f"{_TULU}/train-{i:05d}-of-00006.parquet"      for i in range(6)]

# (category, name, url|[urls], local, max_bytes, parser)  -- parquet: max_bytes=None
# (the footer sits at end of file, so a byte-prefix breaks pyarrow).
SOURCES = [
 # ---- general/edu web prose (40%): clean educational web text ----------------
 ("general","fineweb-edu",  FINEWEB_SHARDS, "fineweb.parquet", None, src_cosmopedia),
 # ---- synthetic textbooks + math (15%) ---------------------------------------
 ("synthetic","cosmopedia-v2", COSMO_SHARDS, "cosmo_v2.parquet", None, src_cosmopedia),
 ("synthetic","finemath",      FMATH_SHARDS, "finemath.parquet", None, src_cosmopedia),
 # ---- code (20%): raw Python source, not instruction Q&A ---------------------
 ("code",   "github-python", GHCODE_SHARDS, "ghpy.parquet", None, src_code),
 # ---- instruction / chat (17.5%) ---------------------------------------------
 ("chat",   "hermes2.5",  f"{HF}/teknium/OpenHermes-2.5/resolve/main/openhermes2_5.json",
                          "hermes25.json", None, src_conversations),
 ("chat",   "tulu3",      TULU_SHARDS, "tulu3.parquet", None, src_messages),
 # ---- tool / function calling (7.5%) -----------------------------------------
 ("tool",   "toolace",    f"{HF}/Team-ACE/ToolACE/resolve/main/data.json",
                          "toolace.json", None, src_toolace),
 ("tool",   "hermes-fc",  f"{HF}/NousResearch/hermes-function-calling-v1/resolve/main/func-calling.json",
                          "hermes_fc_multi.json", None, src_hermes_fc),
 ("tool",   "glaive-fc2", f"{HF}/glaiveai/glaive-function-calling-v2/resolve/main/glaive-function-calling-v2.json",
                          "glaive_fc2.json", 220*MB, src_glaive_fc2),
]

# ============================ v3 corpus (2026-09-20) ======================
# Fresh high-quality blend with its OWN V=16384 tokenizer (NOT token-compatible
# with mix/mix4b -- accepted). Selected by MIX_PROFILE: "v3pre" (~6B, stream, no
# upsampling) or "v3sft" (~1.2B, batch, upsample small sources <=2x). Column
# names + shard paths verified live before wiring. Every TB-scale source is
# subset-capped by listing only a handful of shards; the collect loop stops at
# the per-category cap anyway, so over-listing is harmless.
_UFW    = f"{HF}/openbmb/Ultra-FineWeb/resolve/main/data/ultrafineweb_en"
_UFWL3  = f"{HF}/openbmb/Ultra-FineWeb-L3/resolve/main/data/ultrafineweb_en_l3/multi_style"
_L3UUID = "f36f5a53-4a77-434b-b9bc-67ed69b93fe2"
_OT3    = f"{HF}/open-thoughts/OpenThoughts3-1.2M/resolve/main/data"
_ST2    = f"{HF}/HuggingFaceTB/smoltalk2/resolve/main/SFT"
_HRTU   = f"{HF}/interstellarninja/hermes_reasoning_tool_use/resolve/main/data"

UFW_SHARDS   = [f"{_UFW}/ultrafineweb-en-part-{i:04d}-of-2048.parquet" for i in range(1, 21)]
UFWL3_SHARDS = [f"{_UFWL3}/part-{i:05d}-{_L3UUID}-c000.snappy.parquet"  for i in range(20)]
OT3_SHARDS   = [f"{_OT3}/train-{i:05d}-of-00120.parquet"                for i in range(12)]
ST2_SHARDS   = [f"{_ST2}/OpenHermes_2.5_no_think-{i:05d}-of-00002.parquet" for i in range(2)]

V3_PRE_WEIGHTS = {
    "web":       0.34,   # Ultra-FineWeb EN (content): top-quality filtered web
    "code":      0.20,   # github-code-clean Python (raw source)
    "synthetic": 0.16,   # cosmopedia-v2 textbooks + Ultra-FineWeb-L3 EN
    "math":      0.14,   # finemath-4plus prose
    "chat":      0.10,   # OpenHermes-2.5
    "tool":      0.06,   # ToolACE + hermes-fc
}
V3_PRE_SOURCES = [
 ("web",      "ultrafineweb",  UFW_SHARDS,    "ufw.parquet",      None, src_content),
 ("code",     "github-python", GHCODE_SHARDS, "ghpy.parquet",     None, src_code),
 ("synthetic","cosmopedia-v2", COSMO_SHARDS,  "cosmo_v2.parquet", None, src_cosmopedia),
 ("synthetic","ufw-l3",        UFWL3_SHARDS,  "ufwl3.parquet",    None, src_content),
 ("math",     "finemath",      FMATH_SHARDS,  "finemath.parquet", None, src_cosmopedia),
 ("chat",     "hermes2.5",     f"{HF}/teknium/OpenHermes-2.5/resolve/main/openhermes2_5.json",
                               "hermes25.json", None, src_conversations),
 ("tool",     "toolace",       f"{HF}/Team-ACE/ToolACE/resolve/main/data.json",
                               "toolace.json", None, src_toolace),
 ("tool",     "hermes-fc",     f"{HF}/NousResearch/hermes-function-calling-v1/resolve/main/func-calling.json",
                               "hermes_fc_multi.json", None, src_hermes_fc),
]

V3_SFT_WEIGHTS = {
    "general":   0.24,   # Tulu-3 SFT mixture
    "chat":      0.20,   # SmolTalk2 (OpenHermes-2.5 no_think)
    "code":      0.18,   # github-code-clean Python
    "reasoning": 0.14,   # OpenThoughts3-1.2M
    "tool":      0.14,   # ToolACE + hermes-fc
    "agentic":   0.10,   # hermes_reasoning_tool_use
}
V3_SFT_SOURCES = [
 ("general",  "tulu3",         TULU_SHARDS,   "tulu3.parquet",    None, src_pq_messages),
 ("chat",     "smoltalk2",     ST2_SHARDS,    "st2.parquet",      None, src_pq_messages),
 ("code",     "github-python", GHCODE_SHARDS, "ghpy.parquet",     None, src_code),
 ("reasoning","openthoughts3", OT3_SHARDS,    "ot3.parquet",      None, src_conv_pq),
 ("tool",     "toolace",       f"{HF}/Team-ACE/ToolACE/resolve/main/data.json",
                               "toolace.json", None, src_toolace),
 ("tool",     "hermes-fc",     f"{HF}/NousResearch/hermes-function-calling-v1/resolve/main/func-calling.json",
                               "hermes_fc_multi.json", None, src_hermes_fc),
 ("agentic",  "hermes-rtu",    f"{_HRTU}/train-00000-of-00001.parquet",
                               "hermes_rtu.parquet", None, src_conv_tools_pq),
]

_PROFILE = os.environ.get("MIX_PROFILE", "").strip()
if _PROFILE == "v3pre":
    WEIGHTS, SOURCES = V3_PRE_WEIGHTS, V3_PRE_SOURCES
elif _PROFILE == "v3sft":
    WEIGHTS, SOURCES = V3_SFT_WEIGHTS, V3_SFT_SOURCES
if _PROFILE:
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, f"{_PROFILE} weights sum != 1"
    print(f"[profile] {_PROFILE}: cats={list(WEIGHTS)} nsrc={len(SOURCES)}")

# ------------------------- OLD 1B mix (kept for reference) -----------------
_SOURCES_1B = [
 # ---- general knowledge: capped to ~460 MB raw on request -----------------
 ("general","cosmopedia", f"{HF}/HuggingFaceTB/cosmopedia-100k/resolve/main/data/train-00000-of-00002.parquet",
                          "cosmopedia0.parquet", None, src_cosmopedia),
 ("general","squad2",     "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
                          "squad2.json", None, src_squad),
 ("general","dolly15k",   f"{HF}/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
                          "dolly15k.jsonl", None, src_dolly),
 ("general","hermes2.5",  f"{HF}/teknium/OpenHermes-2.5/resolve/main/openhermes2_5.json",
                          "hermes25.json", 250*MB, src_conversations),
 # ---- general: orca-agentinstruct reasoning splits (NOT agent traces) ------
 ("general","orca-rc",    f"{HF}/microsoft/orca-agentinstruct-1M-v1/resolve/main/data/rc-00000-of-00001.parquet",
                          "orca_rc.parquet", None, src_pq_messages),
 ("general","orca-rag",   f"{HF}/microsoft/orca-agentinstruct-1M-v1/resolve/main/data/rag-00000-of-00001.parquet",
                          "orca_rag.parquet", None, src_pq_messages),
 ("general","orca-mcq",   f"{HF}/microsoft/orca-agentinstruct-1M-v1/resolve/main/data/mcq-00000-of-00001.parquet",
                          "orca_mcq.parquet", None, src_pq_messages),
 ("general","orca-cot",   f"{HF}/microsoft/orca-agentinstruct-1M-v1/resolve/main/data/fs_cot_flow-00000-of-00001.parquet",
                          "orca_cot.parquet", None, src_pq_messages),
 # ---- multi-turn chat -----------------------------------------------------
 ("chat",   "sharegpt",   f"{HF}/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
                          "sharegpt.json", 400*MB, src_conversations),
 ("chat",   "wildchat",   f"{HF}/allenai/WildChat-1M/resolve/main/data/train-00000-of-00014.parquet",
                          "wildchat0.parquet", None, src_wildchat),
 ("chat",   "oasst1",     f"{HF}/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz",
                          "oasst1.jsonl.gz", None, src_oasst),
 ("chat",   "oasst2",     f"{HF}/OpenAssistant/oasst2/resolve/main/2023-11-05_oasst2_ready.trees.jsonl.gz",
                          "oasst2.jsonl.gz", None, src_oasst),
 # ---- instruct ------------------------------------------------------------
 ("instruct","alpaca",    f"{HF}/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json",
                          "alpaca_clean.json", None, src_alpaca),
 ("instruct","wizardlm",  f"{HF}/WizardLMTeam/WizardLM_evol_instruct_70k/resolve/main/alpaca_evol_instruct_70k.json",
                          "wizardlm70k.json", None, src_instr_out),
 # ---- code ----------------------------------------------------------------
 ("code",   "codealpaca", f"{HF}/sahil2801/CodeAlpaca-20k/resolve/main/code_alpaca_20k.json",
                          "codealpaca.json", None, src_alpaca),
 ("code",   "evol-code",  f"{HF}/theblackcat102/evol-codealpaca-v1/resolve/main/train.jsonl",
                          "evolcode.jsonl", 160*MB, src_instr_out),
 ("code",   "magicoder",  f"{HF}/ise-uiuc/Magicoder-OSS-Instruct-75K/resolve/main/data-oss_instruct-decontaminated.jsonl",
                          "magicoder.jsonl", 160*MB, src_magicoder),
 ("code",   "glaive-code",f"{HF}/glaiveai/glaive-code-assistant/resolve/main/c9bc9129-eba0-4b10-8292-4ae70fc7fa0d.json",
                          "glaive_code.json", 160*MB, src_glaive_code),
 # Code-Feedback is the genuinely multi-turn code source
 ("code",   "codefeedback",f"{HF}/m-a-p/Code-Feedback/resolve/main/Code-Feedback.jsonl",
                          "codefeedback.jsonl", None, src_messages),
 ("code",   "magicoder-evol",f"{HF}/ise-uiuc/Magicoder-Evol-Instruct-110K/resolve/main/data-evol_instruct-decontaminated.jsonl",
                          "magicoder_evol.jsonl", None, src_instr_resp),
 ("code",   "selfoss",    f"{HF}/bigcode/self-oss-instruct-sc2-exec-filter-50k/resolve/main/data/train-00000-of-00001.parquet",
                          "selfoss.parquet", None, src_pq_instr_resp),
 # ---- tool calling --------------------------------------------------------
 ("tool",   "glaive-fc2", f"{HF}/glaiveai/glaive-function-calling-v2/resolve/main/glaive-function-calling-v2.json",
                          "glaive_fc2.json", 220*MB, src_glaive_fc2),
 ("tool",   "hermes-fc",  f"{HF}/NousResearch/hermes-function-calling-v1/resolve/main/func-calling-singleturn.json",
                          "hermes_fc_single.json", None, src_hermes_fc),
 ("tool",   "hermes-fc-mt",f"{HF}/NousResearch/hermes-function-calling-v1/resolve/main/func-calling.json",
                          "hermes_fc_multi.json", None, src_hermes_fc),
 ("tool",   "toolace",    f"{HF}/Team-ACE/ToolACE/resolve/main/data.json",
                          "toolace.json", None, src_toolace),
 ("tool",   "hermes-agentic",f"{HF}/NousResearch/hermes-function-calling-v1/resolve/main/json-mode-agentic.json",
                          "hermes_agentic.json", None, src_hermes_fc),
 ("tool",   "hermes-glaive5k",f"{HF}/NousResearch/hermes-function-calling-v1/resolve/main/glaive-function-calling-5k.json",
                          "hermes_glaive5k.json", None, src_hermes_fc),
 # not gated, unlike Salesforce/xlam-function-calling-60k (401)
 ("tool",   "argilla-apigen",f"{HF}/argilla/apigen-function-calling/resolve/main/data/train-00000-of-00001.parquet",
                          "argilla_apigen.parquet", None, src_apigen_query),
 ("tool",   "pythonic-fc",f"{HF}/driaforall/pythonic-function-calling/resolve/main/data/train-00000-of-00001.parquet",
                          "pythonic_fc.parquet", None, src_pythonic_fc),
 # ---- agentic -------------------------------------------------------------
 ("agentic","apigen-mt",  f"{HF}/Salesforce/APIGen-MT-5k/resolve/main/apigen-mt_5k.json",
                          "apigen_mt5k.json", None, src_apigen),
 ("agentic","aflan-react",f"{HF}/internlm/Agent-FLAN/resolve/main/data/agent_instruct_react.jsonl",
                          "aflan_react.jsonl", None, src_agentflan),
 ("agentic","aflan-tflan",f"{HF}/internlm/Agent-FLAN/resolve/main/data/agent_instruct_tflan.jsonl",
                          "aflan_tflan.jsonl", None, src_agentflan),
 ("agentic","aflan-tb3k", f"{HF}/internlm/Agent-FLAN/resolve/main/data/toolbench_instruct_j1s1_3k.jsonl",
                          "aflan_tb3k.jsonl", None, src_agentflan),
 ("agentic","aflan-tbreact",f"{HF}/internlm/Agent-FLAN/resolve/main/data/toolbench_react_10p.jsonl",
                          "aflan_tbreact.jsonl", None, src_agentflan),
 ("agentic","gorilla-hf", f"{HF}/gorilla-llm/APIBench/resolve/main/huggingface_train.json",
                          "gorilla_hf.jsonl", None, src_gorilla),
 ("agentic","gorilla-tf", f"{HF}/gorilla-llm/APIBench/resolve/main/tensorflow_train.json",
                          "gorilla_tf.jsonl", None, src_gorilla),
 ("agentic","gorilla-th", f"{HF}/gorilla-llm/APIBench/resolve/main/torchhub_train.json",
                          "gorilla_th.jsonl", None, src_gorilla),
 ("agentic","swegym",     f"{HF}/SWE-Gym/SWE-Gym/resolve/main/data/train-00000-of-00001.parquet",
                          "swegym.parquet", None, src_swegym),
]
if INCLUDE_WIKIPEDIA:
    SOURCES.insert(0, ("general","wikipedia",
        [f"{HF}/wikimedia/wikipedia/resolve/main/20231101.en/train-{i:05d}-of-00041.parquet" for i in range(41)],
        "wikipedia.parquet", None, src_wikipedia))

# ---------------------- tokenizer (reuse or defer) ------------------------
# In reuse/stream mode the tokenizer must exist BEFORE parsing so docs can be
# encoded on the fly. In plain batch mode `tok` stays None and the BPE is trained
# after collection (unchanged behaviour).
from tokenizers import Tokenizer
tok = None; eot = None
if REUSE_TOK:
    if not os.path.exists(REUSE_TOK):
        sys.exit(f"MIX_REUSE_TOK not found: {REUSE_TOK}")
    tok = Tokenizer.from_file(REUSE_TOK)
    eot = tok.token_to_id(EOT)
    if eot is None:
        sys.exit(f"reuse tokenizer has no {EOT} token: {REUSE_TOK}")
    if tok.get_vocab_size() != VOCAB:
        print(f"[warn] reuse tokenizer vocab {tok.get_vocab_size()} != MIX_VOCAB {VOCAB}")
    print(f"[tok] reusing {REUSE_TOK}  vocab={tok.get_vocab_size()} eot={eot}")
if STREAM and tok is None:
    sys.exit("MIX_STREAM=1 requires MIX_REUSE_TOK (cannot train a BPE without buffering the text corpus)")

def encode_doc(text):
    """Encode one document to a uint16 array with a trailing EOT (stream mode)."""
    ids = tok.encode(text).ids
    a = np.empty(len(ids) + 1, dtype=np.uint16)
    a[:len(ids)] = ids; a[-1] = eot
    return a

# ============================ collect =====================================
# In stream mode the per-source "size" is measured in TOKENS (docs are already
# encoded); in batch mode it is CHARS (docs are text, encoded later).
random.seed(SEED)
UNIT = "tok" if STREAM else "chars"
def budget_of(w):
    return int(TARGET_TOKENS * w * SLACK * (1.0 if STREAM else CHARS_PER_TOK))
budget_chars = {c: budget_of(w) for c, w in WEIGHTS.items()}
nsrc = {c: sum(1 for s in SOURCES if s[0] == c) for c in WEIGHTS}
# Per-source read cap. Without this the first source listed in a category can
# swallow the whole budget and later ones get skipped entirely -- ShareGPT ate
# all of `chat` and starved OASST1/2 on the first run. 2x fair share lets a big
# source cover for a small one without erasing it.
srccap = {c: int(min(budget_chars[c], budget_chars[c] * 2.0 / max(1, nsrc[c]))) for c in WEIGHTS}
buckets = {c: {} for c in WEIGHTS}          # cat -> source name -> [docs]
chars   = {c: 0 for c in WEIGHTS}
persrc  = {}

print(f"[plan] target {TARGET_TOKENS/1e6:.0f}M tokens, vocab {VOCAB}")
for c, w in WEIGHTS.items():
    print(f"       {c:9s} {w*100:4.1f}%  ->  {TARGET_TOKENS*w/1e6:5.1f}M tok  "
          f"({budget_chars[c]/1e6:.0f}M {UNIT}, {nsrc[c]} src, cap {srccap[c]/1e6:.0f}M each)")

for cat, name, url, fname, cap, parser in SOURCES:
    print(f"[get ] {name} ({cat})")
    urls = url if isinstance(url, list) else [url]
    try:
        lst, got = [], 0
        for si, u in enumerate(urls):
            if got >= srccap[cat]: break            # cap reached; skip remaining shards
            shard = fname if len(urls) == 1 else f"{si:04d}_{fname}"
            p = download(u, os.path.join(RAW, shard), max_bytes=cap)
            if not p:
                print(f"       ! shard {si} download failed"); continue
            for d in parser(p):
                item = encode_doc(d) if STREAM else d   # uint16 array (stream) or text (batch)
                lst.append(item); got += len(item)
                if got >= srccap[cat]: break
            if STREAM:                              # free the raw shard before the next
                try: os.remove(p)
                except OSError: pass
        if not lst and not urls: raise RuntimeError("download failed")
        buckets[cat][name] = lst; chars[cat] += got
        persrc[name] = (len(lst), got)
        if not lst:
            print(f"       !! {name} produced ZERO docs - parser/schema mismatch, not a download problem")
        else:
            print(f"       + {len(lst):>7,} docs  {got/1e6:7.1f}M {UNIT}"
                  f"   [{cat} total {chars[cat]/1e6:.0f}/{budget_chars[cat]/1e6:.0f}M]")
    except Exception as e:
        persrc[name] = (0, 0)
        print(f"       ! {name} failed: {type(e).__name__} {e}")

if not any(buckets[c] for c in buckets):
    sys.exit("no data collected")

# ---------------------- balance to the target shares ----------------------
# Two levels of balancing: categories hit their WEIGHTS share, and within a
# category sources are drawn round-robin by characters contributed (always take
# from whichever source is furthest behind), capped at MAX_REPEAT each.
print("\n[balance]")
docs = []
report = {}
for cat, w in WEIGHTS.items():
    want = int(TARGET_TOKENS * w * (1.0 if STREAM else CHARS_PER_TOK))
    names = [n for n, v in buckets[cat].items() if v]
    srcs  = [buckets[cat][n] for n in names]
    for s in srcs: random.shuffle(s)
    if not srcs:
        print(f"  {cat:9s} EMPTY - no source produced data")
        report[cat] = dict(docs=0, chars=0, sources={}); continue
    avail = [sum(len(d) for d in s) for s in srcs]
    limit = min(want, int(sum(avail) * MAX_REPEAT))
    idx     = [0] * len(srcs)
    contrib = [0] * len(srcs)
    sel, acc = [], 0
    while acc < limit:
        cands = [j for j in range(len(srcs)) if idx[j] < len(srcs[j]) * MAX_REPEAT]
        if not cands: break
        j = min(cands, key=lambda x: contrib[x])
        d = srcs[j][idx[j] % len(srcs[j])]; idx[j] += 1
        sel.append(d); acc += len(d); contrib[j] += len(d)
    short = "" if acc >= want * 0.98 else f"  (SHORT of {want/1e6:.0f}M)"
    print(f"  {cat:9s} {len(sel):>8,} docs  {acc/1e6:7.1f}M {UNIT}{short}")
    srep = {}
    for n, s, c, ix in zip(names, srcs, contrib, idx):
        rep = ix / max(1, len(s))
        srep[n] = dict(size=c, repeat=round(rep, 2))
        print(f"      {n:14s} {c/1e6:6.1f}M {UNIT}  repeat {rep:4.2f}x")
    report[cat] = dict(docs=len(sel), chars=acc, sources=srep)
    docs.extend(sel)

random.shuffle(docs)
total_chars = sum(len(d) for d in docs)   # chars (batch) or tokens (stream)
if STREAM:
    print(f"[data] {len(docs):,} docs  {total_chars/1e6:.0f}M tokens")
else:
    print(f"[data] {len(docs):,} docs  {total_chars/1e6:.0f}M chars  "
          f"(~{total_chars/CHARS_PER_TOK/1e6:.0f}M tokens estimated)")

# --------------------------- train / reuse the BPE ------------------------
if tok is None:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=[EOT],
                                  show_progress=True,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    def corpus_iter(max_chars=200_000_000):
        """Sample across the shuffled mix so the vocab covers every category."""
        used = 0
        for d in docs:
            used += len(d)
            if used > max_chars: break
            yield d
    print(f"[tok] training BPE vocab={VOCAB} on <=200M chars")
    tok.train_from_iterator(corpus_iter(), trainer)
    eot = tok.token_to_id(EOT)
tok.save(os.path.join(OUT, "tokenizer.json"))
print(f"[tok] vocab={tok.get_vocab_size()} eot={eot}")

# ------------------------- encode -> uint16 -------------------------------
print("[enc] encoding...")
if STREAM:
    # docs are already uint16 arrays (each ends in EOT); just concatenate.
    arr = np.concatenate(docs).astype(np.uint16, copy=False)
else:
    ids = array.array("H")
    CHUNK = 1000
    t0 = time.time()
    for i in range(0, len(docs), CHUNK):
        for e in tok.encode_batch(docs[i:i+CHUNK]):
            ids.extend(e.ids); ids.append(eot)
        if (i // CHUNK) % 20 == 0:
            print(f"  {i:>9,}/{len(docs):,}  {len(ids)/1e6:.1f}M tok  {time.time()-t0:.0f}s",
                  end="\r", flush=True)
    arr = np.frombuffer(ids, dtype=np.uint16)
nval = max(1, int(len(arr) * 0.005))
arr[:nval].tofile(os.path.join(OUT, "val.bin"))
arr[nval:].tofile(os.path.join(OUT, "train.bin"))

real_cpt = 0.0 if STREAM else total_chars / max(1, len(arr))
json.dump(dict(vocab_size=tok.get_vocab_size(), eot_id=eot,
               total_tokens=int(len(arr)), train_tokens=int(len(arr)-nval),
               val_tokens=int(nval), target_tokens=TARGET_TOKENS,
               chars_per_token=round(real_cpt, 3), weights=WEIGHTS,
               stream=STREAM, reuse_tok=REUSE_TOK or None, max_repeat=MAX_REPEAT,
               unit=UNIT, categories=report, sources={k: dict(docs=v[0], chars=v[1])
                                           for k, v in persrc.items()}),
          open(os.path.join(OUT, "meta.json"), "w"), indent=2)
open(os.path.join(OUT, "template.txt"), "w").write("chatml")
sys.path.insert(0, HERE)
from tokbin import write_tokenizer_bin
write_tokenizer_bin(OUT)

print(f"\n[done] {OUT}")
print(f"  {len(arr):,} tokens  ->  train {len(arr)-nval:,} / val {nval:,}")
if not STREAM:
    print(f"  measured {real_cpt:.2f} chars/token (budgeting assumed {CHARS_PER_TOK})")
for c, r in report.items():
    print(f"  {c:9s} {r['chars']/total_chars*100 if total_chars else 0:5.1f}% of {UNIT}"
          f"   ({len(r.get('sources', {}))} sources)")
