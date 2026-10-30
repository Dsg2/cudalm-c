/* tui.cuh -- interactive menu when cudalm is started with no arguments (Windows console).
 * Each command is a form; Run launches this same exe as a child with the built argv, so every
 * command gets a fresh process (exit() paths and CUDA state stay as in normal CLI use) and inherits
 * the environment edited on the Environment screen (every setting in g_envreg, or a loaded *env.bat).
 * New-model fields of kind F_ENV/F_MEM are pushed into the child's environment; cmd_new saves them
 * (with the rest of the environment) as the model config models\<name>.bin.env.
 * The child runs in its own process group. Ctrl+C during Train writes models\<name>.bin.stop (clean save+exit at
 * the next 20-iter check -- never a hard kill of a CUDA process); for other commands it is forwarded as Ctrl+Break,
 * i.e. what plain CLI Ctrl+C does. Working dir is set to the exe dir. The child's output is shown live and
 * also saved to logs\<command>-<model>-<time>.log; the result screen offers the next step with the output pre-filled.
 * Data forms show a live panel (resolved output folder, free space, size estimate, problems) and ask before running
 * with problems. */
#ifdef _WIN32
#include <conio.h>
#include <stdarg.h>
#include <shellapi.h>
#pragma comment(lib,"shell32.lib")

enum { F_TEXT, F_MODEL, F_DATA, F_FILE, F_CHOICE, F_DIR, F_PATH, F_TOKD, F_OPT, F_ENV, F_MEM, F_FLAG };
/* arg: choices "a|b" (F_CHOICE), the --flag (F_OPT, F_FLAG = valueless, passed when 1) or the env var (F_ENV).
 * vis: 0 always, 1 ngram only, 2 engram only, 3 any memory, 4 advanced (F2 shows). ext: "a|b" choices, a file glob, or F_PATH suffixes ".a;.b".
 * F_DATA = corpus picker, F_TOKD = tokenizer/corpus picker, F_DIR / F_PATH = folder / file-or-folder browser. */
typedef struct { const char*label,*def; int kind; const char*arg; int vis; const char*ext; const char*help; } TFld;   /* help: '\n' = next line */
#define T_NF 32
typedef struct { const char*cmd,*title,*help; TFld f[T_NF]; const char*run; int grp; } TCmd;   /* run: help on [ Run ]; grp 1 = Data tools, 2 = Diagnostics, 3 = Data tools (advanced) */
#define T_TEXTIN ".jsonl;.json;.txt"
#define T_PRESETS "nano|micro|m1|m3|m6|m8|m16|m32|m70|m50|m64e|m220e|a8m|a8m9|s8x|s8w"
#define T_MIXP(n) {"corpus " n,"",F_DATA,0,0,0,"A corpus to blend in; the next pair appears once this one is set."},\
    {" weight " n,"",F_TEXT,0,0,0,"Relative weight (any scale: 3 and 1 = 75% / 25% of the train tokens)."}
static const TCmd T_CMDS[]={
 {"train","Train","continue training a model (create it first with New model)",
  {{"model","",F_MODEL},{"data dir","",F_DATA},{"epochs","1",F_TEXT,0,0,0,
    "Passes over THIS corpus from now on. A model trained before starts a new stage here (warmup + LR schedule\n"
    "restart, sampling from epoch 0 of this corpus). Re-running after a stop finishes the stage (models\\<name>.bin.run)."},
   {"bench every","10000",F_ENV,"CUDALM_BENCH_EVERY",0,0,"0 = off (default 10000). N = every N steps of this stage, lmeval the checkpoint just saved (training waits,\n"
    "the time is left out of tok/s) and log the index + per-task scores to models\\<name>.bin.bench.tsv. Also runs at the end."},
   {" bench items","0",F_ENV,"CUDALM_BENCH_N",0,0,"Items per task (0 = full sets, the default).\n"
    "Tasks: CUDALM_BENCH_TASKS (default all: hellaswag,arc_easy,arc_challenge,piqa,lambada,arithmark)."}}},
 {"run","Generate","sample one continuation of a prompt",
  {{"model","",F_MODEL},{"prompt","Once upon a time",F_TEXT},{"tokens","120",F_TEXT},{"temp","0.8",F_TEXT},{"top-k","40",F_TEXT}}},
 {"serve","Serve","HTTP /completion server for the model (used by agentenv)",
  {{"model","",F_MODEL},{"port","8090",F_TEXT,0,0,0,"TCP port. Stop the server with Ctrl+C."}},
  "Run: load the model and serve until Ctrl+C."},
 {"chatbench","Chatbench","fixed chat prompts, N samples each (the saved log is the input for judgerun.bat)",
  {{"model","",F_MODEL},{"bench file","chatbench.txt",F_FILE},{"gen tokens","128",F_TEXT},{"temp","0.7",F_TEXT},
   {"top-p","0.9",F_TEXT},{"rep pen","1.15",F_TEXT},{"samples","4",F_TEXT},{"ctx T","256",F_TEXT}}},
 {"lmeval","LM eval","zero-shot HellaSwag / ARC-Easy / PIQA / LAMBADA on the full sets",
  {{"model","",F_MODEL},
   {"tasks","all",F_CHOICE,"all|hellaswag|arc_easy|arc_challenge|piqa|lambada|arithmark",0,0,"all = the six tasks; Enter -> type a value for a comma list (e.g. arc_easy,piqa)\n"
    "or a .tsv path. Task files live in bench\\ (CUDALM_BENCH_DIR); missing ones are downloaded once (python + pyarrow)."},
   {"items","0",F_TEXT,0,0,0,"0 = every item; N = only the first N of each task (quick check)."},
   {"max ctx","1024",F_TEXT,0,0,0,"Longest sequence scored (capped at the model's context); longer LAMBADA passages are cut from the left."}},
  "Run: acc / acc_norm (+-std err) per task, LAMBADA greedy last-word acc + ppl. FP8 GEMMs are off (exact path).\n~25 s for a 10M model, ~2 min for 250M."},
 {"probe","Probe","one generation per line of a prompts file",
  {{"model","",F_MODEL},{"prompts file","",F_FILE},{"tokens","80",F_TEXT},{"temp","0",F_TEXT},{"top-k","1",F_TEXT},
   {"top-p","0",F_TEXT},{"rep pen","1",F_TEXT}}},
 {"eval","Eval","forward-only val loss + MoE utilisation (safe during training)",
  {{"model","",F_MODEL},{"data dir","",F_DATA},{"batches","20",F_TEXT},{"batch size","1",F_TEXT}}},
 {"new","New model","create models\\<name>.bin + config .bin.env (other settings: Environment screen, also saved)",
  {{"name","",F_TEXT,0,0,0,"File name: creates models\\<name>.bin plus its config models\\<name>.bin.env.\n"
     "Refused if the model already exists. Train it next with Train (model = this name)."},
   {"preset","s8w",F_CHOICE,T_PRESETS "|custom",0,0,"Model shape: width D, layers L, attention heads H (KV = shared key/value heads),\n"
     "FFN width F (per expert), default context T and vocab V. custom = set every shape value yourself below."},
   {" dim","160",F_ENV,"CUDALM_D",5,0,"Model width D (residual stream). Must be divisible by heads, with an even head dim (D/heads).\n"
     "Embedding params = vocab x D, so width is the costliest knob at small sizes."},
   {" layers","11",F_ENV,"CUDALM_L",5,0,"Number of transformer blocks (1..64). Each adds attention + FFN (or experts) params."},
   {" heads","5",F_ENV,"CUDALM_H",5,0,"Attention (query) heads; head dim = dim / heads, must be even."},
   {" kv heads","1",F_ENV,"CUDALM_KV",5,0,"Shared key/value heads (GQA); heads must be divisible by this. 1 = MQA, = heads = full MHA."},
   {" ffn","432",F_ENV,"CUDALM_F",5,0,"FFN hidden width (SwiGLU, 3 x D x F params), per expert when experts >= 2."},
   {"experts","32",F_TEXT,0,0,0,"0 = dense FFN. >=2 = Mixture of Experts: each layer gets this many FFN experts (max 256)\n"
     "and every token uses only top-k of them -- total params grow with experts, compute per token does not."},
   {"ctx","0",F_TEXT,0,0,0,"Context length in tokens (0 = preset default, min 16). Longer = more activation VRAM\n"
     "and, with full attention, more compute per token. Fixed at creation."},
   {"window","0",F_TEXT,0,0,0,"Sliding attention window (0 = full causal attention). Each token attends to the last N tokens,\n"
     "so attention cost is O(window) not O(ctx). Reach = layers x window; a window >= ctx means full."},
   {"full every","0",F_TEXT,0,0,0,"With a window: every Nth layer attends to the whole prefix again (0 = never),\n"
     "restoring exact long-range lookup. Ignored without a window."},
   {"top-k","2",F_ENV,"CUDALM_TOPK",0,0,"MoE only: experts each token is routed to (1..8). Active FFN params scale with k.\n"
     "Saved in the config; can be changed later (it is not part of the weights)."},
   {"unipool","0",F_ENV,"CUDALM_UNIPOOL",0,"0|1","1 = ONE expert pool shared by all layers (each layer keeps its own router) instead of\n"
     "a pool per layer: same compute, far fewer total params. CUDA only: no quantize/grow/GGUF. Fixed at creation."},
   {"attn gate","0",F_ENV,"CUDALM_ATTN_GATE",0,"0|1","1 = learned sigmoid gate (a D x D matrix per layer) that scales each layer's attention output\n"
     "per token and channel (+L*D*D params)."},
   {"emb dim","0",F_ENV,"CUDALM_EMB_E",0,0,"Factorized embedding: 0 = plain vocab x D table. E (< D, e.g. 96) = vocab x E table + E x D projection,\n"
     "saving (vocab - E) x (D - E) params. Fixed at creation; QLM is not supported."},
   {"memory","engram",F_MEM,0,0,"none|ngram|engram","Hashed n-gram lookup memory, added to the residual stream. Huge in total params but tiny when active\n"
     "(one table row per sub-table per token). none = off; ngram = one table, one order, one layer; engram = multi-head x multi-order, several layers."},
   {" rows","524287",F_ENV,"NGRAM_ROWS",3,0,"Hash table rows. ngram: whole table; engram: per sub-table (rounded down to a prime).\n"
     "Table params = rows x dim (x sub-tables). More rows = fewer hash collisions, more VRAM."},
   {" dim","32",F_ENV,"NGRAM_DIM",3,0,"Width of each table row; projected up to D at each injection point (dim x D weights).\n"
     "q4 fold needs 128 divisible by dim (16, 32, 64, 128)."},
   {" q4 table","1",F_ENV,"NGRAM_Q4",3,"0|1","1 = table stored as 4-bit values with integer Adam moments: a fraction of the fp32 VRAM.\n"
     "0 = full-precision table (only sensible for small tables)."},
   {" q4 fold","1",F_ENV,"NGRAM_Q4_FOLD",3,"0|1","1 = keep only a compact gradient buffer for the q4 table instead of a full fp32 one (saves VRAM).\n"
     "Needs q4 table = 1 and 128 % dim == 0."},
   {" gate","1",F_ENV,"NGRAM_GATE",3,"0|1","1 = context gate: the hidden state decides per token how much of the looked-up memory to add.\n"
     "Adds another dim x D matrix per injection point."},
   {" order","2",F_ENV,"NGRAM_ORDER",1,0,"n-gram length hashed: 2 = current + previous token (bigram), 3 = trigram, ..."},
   {" tri","0",F_ENV,"NGRAM_TRI",1,"0|1","1 = split the table in two halves: order and order+1 grams, both looked up per token."},
   {" layer","0",F_ENV,"NGRAM_LAYER",1,0,"Layer (0-based) whose input receives the memory output."},
   {" heads","4",F_ENV,"NGRAM_HEADS",2,0,"Independent hash functions per order. Each head is its own sub-table; collisions in one head\n"
     "are averaged out by the others. Sub-tables = heads x orders (max 32)."},
   {" orders","2,3",F_ENV,"NGRAM_ORDERS",2,0,"Comma list of n-gram orders, each 1..4, up to 4 values (2 = bigram, 3 = trigram)."},
   {" layers","1,4",F_ENV,"NGRAM_LAYERS",2,0,"Comma list of injection layers (0-based, max 8). The table lookup is shared;\n"
     "each layer gets its own projection (+gate/conv)."},
   {" conv","1",F_ENV,"NGRAM_CONV",2,"0|1","1 = norm + short causal convolution (kernel 4) on the memory output at each point\n"
     "(+5*D params per point), letting neighbouring lookups mix."},
   {" canon map","canon_v3pre.bin",F_ENV,"NGRAM_CANON",2,"canon*.bin","Token -> canonical-token map applied before hashing, so spelling variants of a token\n"
     "share table rows. Needs one uint16 per vocab entry of the training data (canon_v3pre.bin = v3 vocab). Empty = off."},
   {" lr mult","",F_ENV,"NGRAM_LRMULT",2,0,"Learning-rate multiplier for the engram table only (empty = 1)."}}},
 {"info","Model info","config, saved settings, total / ngram-engram / active params",{{"model","",F_MODEL}}},
 {"bench","Bench","training-step speed of a preset (blank batch = auto)",
  {{"preset","s8w",F_CHOICE,T_PRESETS},{"iters","20",F_TEXT},{"batch","",F_TEXT},{"experts","",F_TEXT},
   {"ctx","",F_TEXT},{"window","",F_TEXT}}},
 {"grow","Grow","add layers / widen FFN of a checkpoint",
  {{"in model","",F_MODEL},{"out .bin","",F_TEXT},{"layers","",F_OPT,"--layers"},{"ffn","",F_OPT,"--ffn"},
   {"noise","",F_OPT,"--noise"}}},
 {"quantize","Quantize","shrink a checkpoint on disk",
  {{"in model","",F_MODEL},{"out .bin","",F_TEXT},{"mode","q4n",F_CHOICE,"q8|q4|q4n|fp16"}}},
 {"gguf","Export GGUF","model -> .gguf (runs export\\bin2gguf.py: python + numpy)",
  {{"model","",F_MODEL},
   {"type","q4n",F_CHOICE,"q4n|q4|q8|fp16|f32",0,0,"Same modes as Quantize, applied to an fp32 model first.\n"
    "q4n = lossless for QLM models (smallest exact). q4 = 4-bit, lossy. q8 = 8-bit.\n"
    "fp16/f32 = floats. A quantized input keeps its own type (leave type at its mode)."},
   {"target","tinylm",F_CHOICE,"tinylm|llama",0,0,"tinylm = exact GGUF for the tinylm C runtime (top-k MoE, ngram/engram kept).\n"
    "llama = stock llama.cpp (q8|fp16|f32 only): top-1 routing, ngram/engram dropped -> lossy."},
   {"out .gguf","",F_TEXT,0,0,0,"Blank = next to the model: <model>-<TYPE>.gguf (llama target: <model>-llama-<TYPE>.gguf)."},
   {"ctx","",F_OPT,"--ctx",0,0,"llama target: context to advertise (default = the training window)."},
   {"ng f32","0",F_FLAG,"--ng-f32",0,"0|1","1 = keep an fp32 engram table fp32 in the GGUF (default 0: exported as q4)."}}},
 /* ---- Diagnostics (grp 2) ---- */
 {"qlmcheck","QLM check","distinct values per weight group (proves a QLM model really holds 4-bit weights)",
  {{"model","",F_MODEL},{"group","128",F_TEXT}},0,2},
 {"selftest","Selftest","GPU kernels vs CPU references",{{0}},0,2},
 {"decheck","Decode check","incremental decoder logits vs the full forward (2 slots, different prefixes)",
  {{"model","",F_MODEL},{"tokens","200",F_TEXT},{"top-k","40",F_TEXT}},0,2},
 {"tokcheck","Tokenizer check","re-encode the docs of a token file and compare with the stored tokens",
  {{"model","",F_MODEL},{"tokens file","",F_PATH,0,0,".bin;.u16","uint16 token file made with this model's tokenizer, e.g. a corpus's val.bin."},
   {"max docs","1000000",F_TEXT},{"encoder","new",F_CHOICE,"new|old",0,0,"old = the pre-18b whole-string BPE (for comparison)."}},0,2},
 /* ---- Data tools (grp 1): text -> tokenizer -> corpus; all CPU ---- */
 {"data fetch","Download text","Hugging Face blend -> .jsonl text (resumable; runs prep\\fetch.py: python + pyarrow)",
  {{"profile","v3pre",F_CHOICE,"v3pre|v3sft|v3chat|web|code|math|cosmopedia|chat|tools",0,0,
     "v3pre = pretraining blend (web 34, code 20, synthetic 16, math 14, chat 10, tools 6 %); v3sft / v3chat = instruction blends;\n"
     "web, code, math, cosmopedia, chat, tools = one source each. Or type the https URL of a single .parquet/.json file,\n"
     "or hf:<user/dataset>[:config[:split]][@column] for any Hugging Face dataset (Browse Hugging Face fills this in)."},
   {"out folder","text_v3pre",F_TEXT,0,0,0,"Folder for the text (<category>__<source>.jsonl). A bare name goes under the data root with the most free space.\n"
     "Run again with the same folder to resume an interrupted download."},
   {"tokens","1B",F_TEXT,0,0,0,"Roughly how many tokens of text to fetch (k / M / B suffixes; 1B ~ 4.6 GB of text), split by the profile's weights.\n"
     "Get 10-20% more than you plan to train on."},
   {"template","",F_OPT,"--template",0,0,"Which columns become each document. Enter opens the editor: the columns, and the first rows exactly as the model will see them.\n"
     "Empty = auto (text, messages / conversations, instruction + output, content / code). Or type: text=a+b | chat=col |\n"
     "user=q+ctx,assistant=a  (+ system=col, tools=col). Chats already rendered with ChatML / Llama-3 / Gemma / Mistral / MiniCPM\n"
     "markers (and Alpaca / USER:-ASSISTANT: strings in chat=) are split into turns and get cudalm's markers."}},
  "Run: download and parse each source in turn (raw shards are deleted once parsed).\nNext: Build corpus with this folder as text (tokenizer v3tok), or Train tokenizer first for a new vocab.",1},
 {"data fetch","Convert files","parquet / json / jsonl from elsewhere -> plain .jsonl text (python + pyarrow)",
  {{"source","",F_PATH,0,0,".parquet;.json;.jsonl","A .parquet / .json / .jsonl file, or a folder (every such file inside, recursively).\n"
     "Rows with messages / conversations become chats; otherwise the text / content / code column is used.\n"
     "Set template to choose the columns yourself (prompt + answer, several text columns, a system column...)."},
   {"out folder","converted",F_TEXT,0,0,0,"Folder for the converted .jsonl (one per input file). A bare name goes under a data root."},
   {"category","",F_TEXT,0,0,0,"Category stored with every doc and shown in the corpus stats. Empty = the file name."},
   {"template","",F_OPT,"--template",0,0,"Which columns become each document. Enter opens the editor: the columns, and the first rows exactly as the model will see them.\n"
     "Empty = auto (text, messages / conversations, instruction + output, content / code). Or type: text=a+b | chat=col |\n"
     "user=q+ctx,assistant=a  (+ system=col, tools=col). Chats already rendered with ChatML / Llama-3 / Gemma / Mistral / MiniCPM\n"
     "markers (and Alpaca / USER:-ASSISTANT: strings in chat=) are split into turns and get cudalm's markers."}},
  "Run: convert. .jsonl/.json that already has \"text\" or chat fields needs no conversion:\nBuild corpus reads it directly.",1},
 {"data tokenizer","Train tokenizer","byte-level BPE from your text -> tokenizer.json (HF) + tokenizer.bin + canon.bin",
  {{"out name","mytok",F_TEXT,0,0,0,"Folder for the new tokenizer (a bare name goes under the data root with most free space).\n"
     "Use it as the tokenizer in Build corpus. A model can only ever train on data from the tokenizer it started with."},
   {"text","",F_PATH,0,0,T_TEXTIN,"Text to learn from: .jsonl / .json / .txt files or folders, several separated by ';'.\n"
     "Use the same mix as the corpus so every kind of text gets good tokens."},
   {"vocab","16384",F_TEXT,0,0,0,"Vocabulary size including the 256 byte tokens and <|endoftext|> (max 65535, tokens are 16-bit).\n"
     "Bigger = fewer tokens per text but a larger embedding; 16384 suits ~100M-param models."},
   {"sample MB","300",F_TEXT,0,0,0,"MB of text read, an equal share from each file. More = slightly better merges, slower.\n"
     "300 MB: a few minutes, a few GB of RAM."}},
  "Run: count words, then repeatedly merge the most frequent pair until the vocab is full.\nAll current v3 models use v3tok (V=16384); a new tokenizer means new models trained from scratch.",1},
 {"data tokenizer-from","Tokenizer from model","a model's own tokenizer -> tokenizer folder (tokenizer.bin + tokenizer.json + canon.bin)",
  {{"model","",F_MODEL,0,0,0,"The model you will finetune. Its .bin carries the exact tokenizer it was trained with."},
   {"out name","",F_TEXT,0,0,0,"Folder for the tokenizer (a bare name goes under the data root with most free space).\n"
     "Use it as the tokenizer in Build corpus so the finetune data gets the model's own token ids."}},
  "Run: write the model's tokenizer as a folder (seconds).\nNext: Build corpus with this folder as tokenizer, then Train the model on that corpus.",1},
 {"data encode","Build corpus","tokenize text into train.bin / val.bin for Train and Eval (new corpus, or append)",
  {{"mode","new",F_CHOICE,"new|append",0,0,"new = create a corpus folder (refused if it already holds one).\n"
     "append = add docs to an existing corpus's train.bin (val.bin unchanged); needs the identical tokenizer."},
   {"corpus","",F_DATA,0,0,0,"new: a name (bare = under the data root with most free space) or a full path.\n"
     "append: pick the existing corpus."},
   {"text","",F_PATH,0,0,T_TEXTIN,"Input .jsonl / .json / .txt files or folders, several separated by ';'. JSON needs \"text\" or a chat\n"
     "(\"messages\" / \"conversations\"), rendered with the chatml markers. Parquet: Convert files first."},
   {"tokenizer","v3tok",F_TOKD,0,0,0,"Folder with tokenizer.bin: one from Train tokenizer or Tokenizer from model, or any corpus.\n"
     "Must be the tokenizer of the model you will train (v3tok = all v3 models)."},
   {"val %","1",F_TEXT,0,0,0,"Share of docs held out in val.bin (every Nth doc), used by Eval and the training val loss. 0 = none."},
   {"dedup","1",F_CHOICE,"1|0",0,0,"1 = skip docs whose exact text already appeared in this run."},
   {"follow","0",F_CHOICE,"0|1",4,0,"1 = keep reading a .jsonl another program is still writing, until <file>.done appears\n"
     "(e.g. teacher generation). 0 = stop at the end of the input."},
   {"Q/A -> chat","0",F_ENV,"CUDALM_DATA_QA",0,0,"1 = docs shaped \"passage \\n\\nQuestion: q Answer: a ...\" (Ultra-FineWeb-L3 qa) become chat turns:\n"
     "user = passage + 1st question, assistant = its answer, then one user/assistant pair per question. Other docs stay as they are."}},
  "Run: tokenize on all CPU cores (token-exact with the HF tokenizer).\nNext: Corpus info to check it, then Train with this corpus as data dir.",1},
 {"data mix","Mix corpora","weighted, shuffled blend of corpora that share one tokenizer",
  {{"out name","mix",F_TEXT,0,0,0,"New corpus folder (a bare name goes under the data root with most free space)."},
   {"train tokens","0",F_TEXT,0,0,0,"Size of the blend's train.bin (k / M / B suffixes). 0 = as large as the weights allow without reusing\n"
     "any corpus more than max repeat times. The table below shows what each corpus contributes."},
   {"max repeat","1",F_TEXT,0,0,0,"How often one corpus's docs may be reused (1 = each doc at most once, 2 = up to twice).\n"
     "A corpus that runs out contributes less than its weight (a note is printed)."},
   {"val tokens","1M",F_TEXT,0,0,0,"Tokens in val.bin, taken from each corpus's own val.bin by weight."},
   T_MIXP("1"),T_MIXP("2"),T_MIXP("3"),T_MIXP("4"),T_MIXP("5"),T_MIXP("6"),T_MIXP("7"),T_MIXP("8"),T_MIXP("9"),T_MIXP("10")},
  "Run: pick whole docs per corpus, shuffle them together, write train.bin + val.bin + meta.json.\nAll corpora must use the identical tokenizer (checked).",1},
 {"data info","Corpus info","tokenizer, token / doc counts, doc lengths, categories and a decoded sample",
  {{"corpus","",F_TOKD,0,0,0,"A corpus (or tokenizer) folder. Checks every token id against the vocab - catches data built\n"
     "with another tokenizer - and prints one random doc decoded."}},
  "Run: read-only; scanning a multi-GB train.bin takes a few seconds.",1},
 {"data canon","Canon map","canon.bin for the New model 'canon map' field (NGRAM_CANON)",
  {{"tokenizer","v3tok",F_TOKD,0,0,0,"Tokenizer or corpus folder. Tokens differing only by case, leading space or Unicode form\n"
     "map to one id, so they share engram table rows."},
   {"out file","",F_TEXT,0,0,0,"Output .bin (empty = canon.bin inside the tokenizer folder)."}},
  "Run: write one uint16 per token. Train tokenizer already writes canon.bin.",3},
 {"data weights","Loss weights","train.w per-token loss weights for a corpus (CUDALM_TOKW=1 trains with it)",
  {{"corpus","",F_DATA,0,0,0,"A corpus encoded from the rows file in order with val % = 0 and dedup = 0 (doc k = row k)."},
   {"rows","",F_PATH,0,0,".jsonl","jsonl rows: \"w\" = per-doc weight (GRPO advantage), or \"a0\"/\"tl\"/\"tp\" teacher log-probs\n"
    "(on-policy distillation, agentenv tscore)."}},
  "Run: write <corpus>\\train.w (int8 per token).",3},
};
#define T_NCMD (int)(sizeof(T_CMDS)/sizeof(T_CMDS[0]))
/* positional fields the CLI requires (by form title): Run refuses while one is empty */
static const struct { const char*title; int n; } T_REQ[]={
 {"Train",2},{"Generate",1},{"Serve",1},{"Chatbench",2},{"LM eval",1},{"Probe",2},{"Eval",2},{"New model",2},{"Model info",1},
 {"Bench",1},{"Grow",2},{"Quantize",2},{"Export GGUF",1},{"QLM check",1},{"Decode check",1},{"Tokenizer check",2},
 {"Download text",2},{"Convert files",2},{"Train tokenizer",2},{"Tokenizer from model",2},{"Build corpus",4},{"Mix corpora",6},{"Corpus info",1},
 {"Canon map",1},{"Loss weights",2}};
static int t_req(const TCmd*c){ for(size_t i=0;i<sizeof T_REQ/sizeof T_REQ[0];i++) if(!strcmp(T_REQ[i].title,c->title)) return T_REQ[i].n; return 0; }
enum { K_UP=1072, K_DOWN=1080, K_LEFT=1075, K_RIGHT=1077, K_PGUP=1073, K_PGDN=1081, K_HOME=1071, K_END=1079, K_F2=1060, K_F5=1063, K_DEL=1083 };
#define T_MAXL 512
#define T_INFO 220
static char t_items[T_MAXL][260], t_info[T_MAXL][T_INFO];
static char t_profile[260]="(none)";

static int t_key(void){ int c=_getch(); if(c==0||c==0xE0) return 1000+_getch(); return c; }
static int t_rows(void){ CONSOLE_SCREEN_BUFFER_INFO b; if(GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE),&b))
    return b.srWindow.Bottom-b.srWindow.Top+1; return 25; }
static int t_cols(void){ CONSOLE_SCREEN_BUFFER_INFO b; if(GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE),&b))
    return b.srWindow.Right-b.srWindow.Left+1; return 80; }
static void t_head(const char*title){
    printf("\x1b[2J\x1b[H\x1b[1;36mcudalm\x1b[0m  %s   \x1b[2m[%s]  profile: %s\x1b[0m\n\n",title,exe_dir(),t_profile); }
static int t_envwanted(const char*k){ return !_strnicmp(k,"CUDALM_",7)||!_strnicmp(k,"QLM_",4)||!_strnicmp(k,"NGRAM_",6); }

/* single-line editor on the bottom row; returns 0 on Esc (buf unchanged) */
static int t_edit(const char*label, char*buf, int cap){
    char tmp[1024]; snprintf(tmp,sizeof tmp,"%s",buf); int n=(int)strlen(tmp);
    for(;;){
        printf("\x1b[%d;1H\x1b[2K\x1b[1m%s:\x1b[0m %s\x1b[7m \x1b[0m",t_rows(),label,tmp); fflush(stdout);
        int k=t_key();
        if(k==27) return 0;
        if(k==13){ snprintf(buf,cap,"%s",tmp); return 1; }
        if(k==8){ if(n) tmp[--n]=0; }
        else if(k==21){ n=0; tmp[0]=0; }                        /* Ctrl+U */
        else if(k>=32 && k<256 && n<(int)sizeof(tmp)-1 && n<cap-1){ tmp[n++]=(char)k; tmp[n]=0; }
    }
}
/* scrolling list over t_items/t_info; returns index or -1.
 * t_pv: optional preview of the highlighted item under the list; t_multi = k+1: items from k on toggle t_mark with Space */
static void (*t_pv)(int sel); static int t_multi; static char t_mark[T_MAXL];
static int t_pick(const char*title, const char*hint, int n, int sel){
    if(n<=0) return -1; if(sel<0||sel>=n) sel=0; int top=0;
    for(;;){
        int h=t_rows()-6-(t_pv?6:0); if(h<3) h=3;
        if(sel<top) top=sel; if(sel>=top+h) top=sel-h+1;
        t_head(title);
        for(int i=top;i<n && i<top+h;i++)
            printf("%s %s%-44s \x1b[2m%s\x1b[0m\x1b[0m\n", i==sel?"\x1b[7m>":" ", t_multi&&i>=t_multi-1?(t_mark[i]?"[x] ":"[ ] "):"", t_items[i], t_info[i]);
        if(t_pv){ printf("\x1b[%d;1H",t_rows()-7); t_pv(sel); }
        printf("\x1b[%d;1H\x1b[2m%s\x1b[0m",t_rows()-1,hint?hint:"Up/Down PgUp/PgDn  Enter select  Esc back"); fflush(stdout);
        int k=t_key();
        if(k==' ' && t_multi && sel>=t_multi-1) t_mark[sel]^=1;
        else if(k==K_UP && sel>0) sel--; else if(k==K_DOWN && sel<n-1) sel++;
        else if(k==K_PGUP) sel=sel>h?sel-h:0; else if(k==K_PGDN) sel=sel+h<n?sel+h:n-1;
        else if(k==K_HOME) sel=0; else if(k==K_END) sel=n-1;
        else if(k==13) return sel; else if(k==27) return -1;
    }
}
static int t_ends(const char*s,const char*e){ size_t a=strlen(s),b=strlen(e); return a>=b && !_stricmp(s+a-b,e); }
static int t_list_models(void){
    char pat[800]; snprintf(pat,sizeof pat,"%s\\models\\*.bin",exe_dir());
    WIN32_FIND_DATAA fd; HANDLE h=FindFirstFileA(pat,&fd); int n=0;
    if(h!=INVALID_HANDLE_VALUE){ do{ if(n>=T_MAXL || !t_ends(fd.cFileName,".bin")) continue;
        snprintf(t_items[n],260,"%.*s",(int)strlen(fd.cFileName)-4,fd.cFileName);
        SYSTEMTIME st; FileTimeToSystemTime(&fd.ftLastWriteTime,&st);
        double mb=((double)fd.nFileSizeHigh*4294967296.0+fd.nFileSizeLow)/1048576.0;
        char cp[1100]; model_path(t_items[n],cp,sizeof cp-4); strcat(cp,".env");
        snprintf(t_info[n],T_INFO,"%8.1f MB  %04d-%02d-%02d %02d:%02d%s",mb,st.wYear,st.wMonth,st.wDay,st.wHour,st.wMinute,
                 GetFileAttributesA(cp)!=INVALID_FILE_ATTRIBUTES?"  [config]":""); n++;
    }while(FindNextFileA(h,&fd)); FindClose(h); }
    return n;
}
static long long t_meta_tokens(const char*dir){
    char p[800]; snprintf(p,sizeof p,"%s\\meta.json",dir); size_t len; char*b=(char*)read_file(p,&len); if(!b) return -1;
    char*q=strstr(b,"\"total_tokens\""); long long v=q?_atoi64(strchr(q,':')+1):-1; free(b); return v; }
/* short id of a folder's tokenizer.bin (0 = none): equal ids = identical tokenizer, which mix/append require */
static unsigned t_tokhash(const char*dir){
    static char kp[32][1100]; static unsigned kh[32]; static int kn;
    for(int i=0;i<kn&&i<32;i++) if(!strcmp(kp[i],dir)) return kh[i];
    char p[1200]; snprintf(p,sizeof p,"%s\\tokenizer.bin",dir); size_t n; char*b=(char*)read_file(p,&n); if(!b) return 0;
    unsigned h=(unsigned)(d_fnv(b,n)&0xFFFFFF); free(b); if(!h) h=1;
    int i=kn++%32; snprintf(kp[i],sizeof kp[i],"%s",dir); kh[i]=h; return h; }
/* folders under the data roots holding a corpus (train.bin) or, tok=1, any tokenizer.bin */
static int t_list_dirs(int tok){
    char r[1024]; int n=0;
    for(int ri=0;d_root(ri,r,sizeof r);ri++){
        char pat[1100]; snprintf(pat,sizeof pat,"%s\\*",r); WIN32_FIND_DATAA fd; HANDLE h=FindFirstFileA(pat,&fd);
        if(h==INVALID_HANDLE_VALUE) continue;
        do{ if(n>=T_MAXL || !(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY) || fd.cFileName[0]=='.') continue;
            char d[1200], tb[1300], tk[1300]; snprintf(d,sizeof d,"%s\\%s",r,fd.cFileName);
            snprintf(tb,sizeof tb,"%s\\train.bin",d); snprintf(tk,sizeof tk,"%s\\tokenizer.bin",d);
            int hc=d_exists(tb), ht=d_exists(tk); if(tok?!ht:!hc) continue;
            int V=0; FILE*f=fopen(tk,"rb"); if(f){ char m[4]; if(fread(m,1,4,f)!=4||fread(&V,4,1,f)!=1) V=0; fclose(f); }
            snprintf(t_items[n],260,"%s",d); long long t=hc?t_meta_tokens(d):-1;
            unsigned th=t_tokhash(d);
            if(hc&&t>0) snprintf(t_info[n],T_INFO,"%9.1fM tok  V=%d  tok %06x",t/1e6,V,th);
            else if(hc) snprintf(t_info[n],T_INFO,"%9.1fM tok  V=%d  tok %06x",d_fsize(tb)/2e6,V,th);
            else snprintf(t_info[n],T_INFO,"tokenizer    V=%d  tok %06x",V,th);
            n++;
        }while(FindNextFileA(h,&fd)); FindClose(h);
    }
    return n;
}
/* glob in the exe dir and in bench\ (bench files are listed as bench\name) */
static int t_list_glob(const char*glob){
    int n=0;
    for(int d=0;d<2;d++){ char pat[400]; snprintf(pat,sizeof pat,"%s%s",d?"bench\\":"",glob);
        WIN32_FIND_DATAA fd; HANDLE h=FindFirstFileA(pat,&fd);
        if(h!=INVALID_HANDLE_VALUE){ do{ if(n<T_MAXL && !(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)){
            snprintf(t_items[n],260,"%s%s",d?"bench\\":"",fd.cFileName); snprintf(t_info[n],T_INFO,"%8.1f KB",fd.nFileSizeLow/1024.0); n++; } }
            while(FindNextFileA(h,&fd)); FindClose(h); } }
    return n;
}
/* picker with a leading "type it" entry; writes the choice into val */
static void t_browse(const char*label, char*val, int cap, const char*ext, int files);
static void t_choose(const char*label, int n, char*val, int cap, int browse){
    int k=browse?2:1;
    for(int i=(n+k<T_MAXL?n+k:T_MAXL)-1;i>=k;i--){ memcpy(t_items[i],t_items[i-k],260); memcpy(t_info[i],t_info[i-k],T_INFO); }
    snprintf(t_items[0],260,"(type a value...)"); t_info[0][0]=0;
    if(browse){ snprintf(t_items[1],260,"[ browse folders... ]"); t_info[1][0]=0; }
    n=n+k<T_MAXL?n+k:T_MAXL;
    int cur=0; for(int i=k;i<n;i++) if(!strcmp(t_items[i],val)) cur=i;
    char ttl[200]; snprintf(ttl,sizeof ttl,"select %s",label);
    int s=t_pick(ttl,NULL,n,cur);
    if(s==0) t_edit(label,val,cap); else if(browse && s==1) t_browse(label,val,cap,NULL,0); else if(s>0) snprintf(val,cap,"%s",t_items[s]);
}
static int t_extok(const char*nm,const char*ext){ char e[200]; snprintf(e,sizeof e,"%s",ext?ext:"");
    for(char*p=e,*q;*p;p=q){ q=strchr(p,';'); if(q) *q++=0; if(t_ends(nm,p)) return 1; if(!q) break; } return !ext; }
/* folder browser (files=1 also lists files matching ext): drives at the top, data folders annotated */
static void t_browse(const char*label, char*val, int cap, const char*ext, int files){
    char cur[MAX_PATH]="", r[1200]; int sel=0;
    if(val[0]){ const char*s1=strchr(val,';'); char v1[1200]; snprintf(v1,sizeof v1,"%.*s",s1?(int)(s1-val):(int)strlen(val),val);
        data_resolve(v1,r,sizeof r); GetFullPathNameA(r,MAX_PATH,cur,NULL);
        while(cur[0] && !d_isdir(cur)){ char*q=strrchr(cur,'\\'); if(!q){ cur[0]=0; break; } *q=0; } }
    if(!cur[0]){ for(int i=0;d_root(i,r,sizeof r);i++) if(d_isdir(r)){ GetFullPathNameA(r,MAX_PATH,cur,NULL); break; } }
    for(;;){
        size_t L=strlen(cur); if(L && cur[L-1]=='\\') cur[--L]=0;
        int n=0, root=!cur[0];
        snprintf(t_items[n],260,"(type a path...)"); snprintf(t_info[n++],T_INFO,"several inputs: separate with ;");
        if(!root){ snprintf(t_items[n],260,"[ use this folder ]"); snprintf(t_info[n++],T_INFO,"%s",cur);
                   snprintf(t_items[n],260,".."); t_info[n++][0]=0; }
        int d0=n;
        if(root){ char b[256]; GetLogicalDriveStringsA(sizeof b,b);
            for(char*p=b;*p&&n<T_MAXL;p+=strlen(p)+1){ snprintf(t_items[n],260,"%.2s",p); t_info[n++][0]=0; } }
        else for(int pass=0;pass<1+files;pass++){
            char pat[MAX_PATH+4]; snprintf(pat,sizeof pat,"%s\\*",cur); WIN32_FIND_DATAA fd; HANDLE h=FindFirstFileA(pat,&fd);
            if(h==INVALID_HANDLE_VALUE) continue;
            do{ int dir=(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)!=0; if(n>=T_MAXL||fd.cFileName[0]=='.'||dir!=(pass==0)) continue;
                if(!dir && !t_extok(fd.cFileName,ext)) continue;
                snprintf(t_items[n],260,"%s%s",fd.cFileName,dir?"\\":"");
                if(dir){ char d[MAX_PATH+300],x[MAX_PATH+320]; snprintf(d,sizeof d,"%s\\%s",cur,fd.cFileName);
                    snprintf(x,sizeof x,"%s\\train.bin",d); long long t=d_exists(x)?d_fsize(x)/2:-1; snprintf(x,sizeof x,"%s\\tokenizer.bin",d);
                    if(t>=0) snprintf(t_info[n],T_INFO,"corpus %.1fM tok",t/1e6); else if(d_exists(x)) snprintf(t_info[n],T_INFO,"tokenizer"); else t_info[n][0]=0; }
                else snprintf(t_info[n],T_INFO,"%10.1f MB",((double)fd.nFileSizeHigh*4294967296.0+fd.nFileSizeLow)/1048576.0);
                n++; }while(FindNextFileA(h,&fd)); FindClose(h); }
        char ttl[MAX_PATH+200]; snprintf(ttl,sizeof ttl,"%s: %s",label,root?"(drives)":cur);
        int s=t_pick(ttl,files?"Enter open folder / pick file  Esc cancel":"Enter open folder  Esc cancel",n,sel); sel=0;
        if(s<0) return;
        if(s==0){ t_edit(label,val,cap); return; }
        if(!root && s==1){ snprintf(val,cap,"%s",cur); return; }
        if(!root && s==2){ char*q=strrchr(cur,'\\'); if(q) *q=0; else cur[0]=0; continue; }
        if(s>=d0){ char nm[260]; snprintf(nm,sizeof nm,"%s",t_items[s]); size_t l=strlen(nm);
            if(root){ snprintf(cur,sizeof cur,"%s",nm); continue; }
            if(l && nm[l-1]=='\\'){ nm[l-1]=0; size_t c=strlen(cur); snprintf(cur+c,sizeof cur-c,"\\%s",nm); continue; }
            snprintf(val,cap,"%s\\%s",cur,nm); return; }
    }
}
/* ---- data helpers: sizes, free space, previews, pickers ---- */
static int t_is(const TCmd*c,const char*t){ return !strcmp(c->title,t); }
static int t_ci(const char*title){ for(int i=0;i<T_NCMD;i++) if(!strcmp(T_CMDS[i].title,title)) return i; return -1; }
static int t_fi(int ci,const char*label){ if(ci<0) return -1;
    for(int i=0;i<T_NF&&T_CMDS[ci].f[i].label;i++) if(!strcmp(T_CMDS[ci].f[i].label,label)) return i; return -1; }
static const char* t_hb(double b,char*o){ if(b>=1e9) sprintf(o,"%.1f GB",b/1e9); else if(b>=1e6) sprintf(o,"%.0f MB",b/1e6); else sprintf(o,"%.0f KB",b/1e3); return o; }
static const char* t_hn(double v,char*o){ if(v>=1e9) sprintf(o,"%.2fB",v/1e9); else if(v>=1e6) sprintf(o,"%.1fM",v/1e6);
    else if(v>=1e3) sprintf(o,"%.1fk",v/1e3); else sprintf(o,"%.0f",v); return o; }
static double t_tree(const char*p,const char*ext,int*nf){          /* bytes of the files under p matching ext (NULL = all) */
    if(!d_isdir(p)){ if(!d_exists(p)||!t_extok(p,ext)) return 0; (*nf)++; return (double)d_fsize(p); }
    char pat[1200]; snprintf(pat,sizeof pat,"%s\\*",p); WIN32_FIND_DATAA fd; HANDLE h=FindFirstFileA(pat,&fd); double s=0;
    if(h==INVALID_HANDLE_VALUE) return 0;
    do{ if(fd.cFileName[0]=='.') continue; char q[1200]; snprintf(q,sizeof q,"%s\\%s",p,fd.cFileName);
        if(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY) s+=t_tree(q,ext,nf);
        else if(t_extok(fd.cFileName,ext)){ s+=(double)fd.nFileSizeHigh*4294967296.0+fd.nFileSizeLow; (*nf)++; } }while(FindNextFileA(h,&fd));
    FindClose(h); return s; }
static double t_spec(const char*spec,const char*ext,int*nf){      /* ';' list of files/folders (data-root names ok); cached 10 s */
    static char ks[4][1100]; static double kv[4]; static int kn[4],kk; static DWORD kt[4]; char key[1100];
    snprintf(key,sizeof key,"%s|%s",spec,ext?ext:"");
    for(int i=0;i<4;i++) if(!strcmp(ks[i],key)&&GetTickCount()-kt[i]<10000){ *nf=kn[i]; return kv[i]; }
    char s[1024]; snprintf(s,sizeof s,"%s",spec); double b=0; int n=0;
    for(char*p=s,*e;p;p=e){ e=strchr(p,';'); if(e) *e++=0; while(*p==' ') p++; if(!*p) continue; char r[1200]; data_resolve(p,r,sizeof r); b+=t_tree(r,ext,&n); }
    int i=kk++%4; snprintf(ks[i],sizeof ks[i],"%s",key); kv[i]=b; kn[i]=n; kt[i]=GetTickCount(); *nf=n; return b; }
static double t_free(const char*path){                             /* free bytes on the drive of path (or its nearest parent) */
    char d[1200]; if(!GetFullPathNameA(path,sizeof d,d,NULL)) return -1;
    for(;;){ ULARGE_INTEGER f; if(GetDiskFreeSpaceExA(d,&f,NULL,NULL)) return (double)f.QuadPart;
        char*q=strrchr(d,'\\'); if(!q) return -1; if(q<=d+2){ if(q[1]==0) return -1; q[1]=0; } else *q=0; } }
static void t_corpus_pv(int sel){                                  /* corpus / tokenizer preview under a picker */
    const char*d=t_items[sel]; if(!d_isdir(d)) return; char p[1300],a[32],b[32]; size_t ml;
    snprintf(p,sizeof p,"%s\\train.bin",d); double tr=d_exists(p)?d_fsize(p)/2.0:-1;
    snprintf(p,sizeof p,"%s\\val.bin",d); double va=d_exists(p)?d_fsize(p)/2.0:0;
    printf("\x1b[36m%s\x1b[0m\n",d);
    if(tr>=0) printf("  train %s tok   val %s tok   tokenizer %06x",t_hn(tr,a),t_hn(va,b),t_tokhash(d));
    else printf("  tokenizer only   %06x",t_tokhash(d));
    snprintf(p,sizeof p,"%s\\meta.json",d); char*mj=(char*)read_file(p,&ml); if(!mj){ printf("\n"); return; }
    mj=(char*)realloc(mj,ml+1); mj[ml]=0;
    const char*bb=jget(mj,"built_by"); if(bb){ DBuf x={0}; jdec(bb,&x); printf("   built by %s",x.p?x.p:""); free(x.p); }
    const char*h=jget(mj,"history"); int nh=0; if(h) for(const char*e=jfirst(h);e;e=jnext(e)) nh++; if(nh>1) printf("  (+%d appends)",nh-1);
    printf("\n");
    const char*v=jget(mj,"categories"); char line[600]=""; int lo=0,n=0;
    if(v) for(const char*k=jfirst(v);k&&*k=='"'&&n<8;k=jobj_next(k),n++){ const char*o=jobj_val(k); if(!o) break; DBuf nm={0}; jdec(k,&nm);
        const char*x=jget(o,"tokens"); if(!x) x=jget(o,"emitted");
        lo+=snprintf(line+lo,sizeof line-lo,"%s%s %s",n?", ":"",nm.p?nm.p:"",x?t_hn(atof(x),a):"?"); free(nm.p); if(lo>=(int)sizeof line) break; }
    if(lo) printf("  \x1b[2m%.150s\x1b[0m\n",line);
    free(mj); }
static int t_keep_tok(int n,unsigned h){                           /* keep only list entries with tokenizer h */
    int m=0; for(int i=0;i<n;i++) if(t_tokhash(t_items[i])==h){ if(m!=i){ memcpy(t_items[m],t_items[i],260); memcpy(t_info[m],t_info[i],T_INFO); } m++; }
    return m; }
static void t_mark_tok(int n,unsigned h){ for(int i=0;i<n;i++) if(t_tokhash(t_items[i])!=h){ size_t l=strlen(t_info[i]);
    snprintf(t_info[i]+l,T_INFO-l,"  \x1b[33mother tokenizer\x1b[0m"); } }
/* text folders under the data roots (Space marks several, joined with ';') */
static void t_pick_text(const char*label,char*val,int cap,const char*ext){
    int n=0; char r[1100],a[32];
    snprintf(t_items[n],260,"(type a path...)"); snprintf(t_info[n++],T_INFO,"several: separate with ;");
    snprintf(t_items[n],260,"[ browse files / folders... ]"); t_info[n++][0]=0;
    for(int ri=0;d_root(ri,r,sizeof r);ri++){ char pat[1200]; snprintf(pat,sizeof pat,"%s\\*",r); WIN32_FIND_DATAA fd; HANDLE h=FindFirstFileA(pat,&fd);
        if(h==INVALID_HANDLE_VALUE) continue;
        do{ if(n>=T_MAXL||!(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)||fd.cFileName[0]=='.') continue;
            char d[1200],q[1300]; snprintf(d,sizeof d,"%s\\%s",r,fd.cFileName); snprintf(q,sizeof q,"%s\\tokenizer.bin",d); if(d_exists(q)) continue;
            int nf=0; double b=t_tree(d,ext,&nf); if(!nf) continue;
            snprintf(t_items[n],260,"%s",d); snprintf(t_info[n++],T_INFO,"%6d files %10s",nf,t_hb(b,a)); }while(FindNextFileA(h,&fd)); FindClose(h); }
    memset(t_mark,0,sizeof t_mark);
    { char s[1024]; snprintf(s,sizeof s,"%s",val); for(char*p=s,*e;p;p=e){ e=strchr(p,';'); if(e) *e++=0; if(!*p) continue; char x[1200]; data_resolve(p,x,sizeof x);
        for(int i=2;i<n;i++) if(!_stricmp(t_items[i],x)) t_mark[i]=1; } }
    char ttl[300]; snprintf(ttl,sizeof ttl,"select %s: text folders under the data roots",label);
    t_multi=3; int s=t_pick(ttl,"Space mark several   Enter use the marked ones (or the highlighted one)   Esc cancel",n,0); t_multi=0;
    if(s<0) return; if(s==0){ t_edit(label,val,cap); return; } if(s==1){ t_browse(label,val,cap,ext,1); return; }
    char o[T_MAXL]=""; int lo=0; for(int i=2;i<n;i++) if(t_mark[i]) lo+=snprintf(o+lo,sizeof o-lo,"%s%s",lo?";":"",t_items[i]);
    snprintf(val,cap,"%s",lo?o:t_items[s]); }
static int t_yes(const char*fmt,...){                              /* y/N question on the bottom row */
    va_list ap; va_start(ap,fmt); printf("\x1b[%d;1H\x1b[2K\x1b[1;33m",t_rows()); vprintf(fmt,ap); printf("\x1b[0m"); va_end(ap); fflush(stdout);
    while(_kbhit()) _getch(); int k=t_key(); return k=='y'||k=='Y'; }
static void t_note(const char*fmt,...){
    va_list ap; va_start(ap,fmt); printf("\x1b[%d;1H\x1b[2K\x1b[33m",t_rows()); vprintf(fmt,ap); printf("\x1b[0m   \x1b[2mpress any key\x1b[0m"); va_end(ap); fflush(stdout);
    while(_kbhit()) _getch(); t_key(); }
static void t_choice_items(const char*ch, int*n){
    *n=0; const char*p=ch;
    while(*p && *n<T_MAXL){ const char*q=strchr(p,'|'); int l=q?(int)(q-p):(int)strlen(p);
        snprintf(t_items[*n],260,"%.*s",l,p); t_info[*n][0]=0; (*n)++; if(!q) break; p=q+1; }
}
static const char* t_choices(const TFld*f){
    if(f->kind==F_CHOICE) return f->arg;
    return (f->kind==F_ENV||f->kind==F_MEM||f->kind==F_FLAG)&&f->ext&&strchr(f->ext,'|') ? f->ext : NULL; }

/* Windows argv quoting (CommandLineToArgvW / MSVC CRT rules) */
static void t_quote(char*o, size_t cap, const char*a){
    size_t n=strlen(o); int need=!*a || strpbrk(a," \t\"")!=NULL;
    #define PUT(c) do{ if(n+1<cap) o[n++]=(c); }while(0)
    if(n) PUT(' ');
    if(!need){ for(;*a;a++) PUT(*a); o[n]=0; return; }
    PUT('"');
    for(;;){ int bs=0; while(*a=='\\'){ bs++; a++; }
        if(!*a){ for(int i=0;i<bs*2;i++) PUT('\\'); break; }
        if(*a=='"'){ for(int i=0;i<bs*2+1;i++) PUT('\\'); PUT('"'); }
        else { for(int i=0;i<bs;i++) PUT('\\'); PUT(*a); }
        a++; }
    PUT('"'); o[n]=0;
    #undef PUT
}
static volatile DWORD t_child; static char t_stop[1100];
static BOOL WINAPI t_ctrl(DWORD ev){
    if(ev!=CTRL_C_EVENT && ev!=CTRL_BREAK_EVENT) return FALSE;
    if(t_child){
        if(t_stop[0]){ FILE*f=fopen(t_stop,"wb"); if(f) fclose(f);
            printf("\n\x1b[33m[tui] wrote %s - training stops and saves at the next 20-iter check\x1b[0m\n",t_stop); fflush(stdout); }
        else GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT,t_child);
    }
    return TRUE;
}

/* builds "<exe> cmd args"; returns 0 and sets err on a gap in the positional args */
static int t_cmdline(const TCmd*c, char vals[][T_MAXL], char*o, size_t cap, char*err, size_t ecap){
    char exe[MAX_PATH], cw[64]; GetModuleFileNameA(0,exe,sizeof exe); o[0]=0; t_quote(o,cap,exe);
    snprintf(cw,sizeof cw,"%s",c->cmd); for(char*w=strtok(cw," ");w;w=strtok(NULL," ")) t_quote(o,cap,w);
    int last=-1; for(int i=0;i<T_NF&&c->f[i].label;i++) if(c->f[i].kind<F_OPT && vals[i][0]) last=i;
    for(int i=0,r=t_req(c);i<T_NF&&c->f[i].label&&r>0;i++) if(c->f[i].kind<F_OPT){
        if(!vals[i][0]){ snprintf(err,ecap,"'%s' is required",c->f[i].label); return 0; } r--; }
    for(int i=0;i<=last;i++){ if(c->f[i].kind>=F_OPT) continue;
        if(!vals[i][0]){ snprintf(err,ecap,"'%s' is empty but a later field is set",c->f[i].label); return 0; }
        t_quote(o,cap,vals[i]); }
    for(int i=0;i<T_NF&&c->f[i].label;i++) if(c->f[i].kind==F_OPT && vals[i][0]){ t_quote(o,cap,c->f[i].arg); t_quote(o,cap,vals[i]); }
    for(int i=0;i<T_NF&&c->f[i].label;i++) if(c->f[i].kind==F_FLAG && !strcmp(vals[i],"1")) t_quote(o,cap,c->f[i].arg);
    return 1;
}
/* runs cl with stdout+stderr piped back here: shown live and copied to logs\<tag>-<time>.log
 * (LMTUI_CHILD makes the child's stdout unbuffered so the pipe stays live) */
static DWORD t_spawn(const char*cl, const char*stop, const char*tag, char*lpo, size_t lcap){
    printf("\x1b[2J\x1b[H\x1b[1;36m> %s\x1b[0m\n\n",cl); fflush(stdout);
    char lp[600]; SYSTEMTIME tm; GetLocalTime(&tm); CreateDirectoryA("logs",NULL);
    snprintf(lp,sizeof lp,"logs\\%s-%04d%02d%02d-%02d%02d%02d.log",tag,tm.wYear,tm.wMonth,tm.wDay,tm.wHour,tm.wMinute,tm.wSecond);
    for(char*p=lp+5;*p;p++) if(strchr(" /\\:*?\"<>|",*p)) *p='_';
    FILE*lf=fopen(lp,"wb"); if(lf) fprintf(lf,"> %s\n\n",cl);
    SECURITY_ATTRIBUTES sa={sizeof sa,NULL,TRUE}; HANDLE rd=NULL,wr=NULL;
    if(!CreatePipe(&rd,&wr,&sa,0)) rd=wr=NULL; else SetHandleInformation(rd,HANDLE_FLAG_INHERIT,0);
    STARTUPINFOA si; PROCESS_INFORMATION pi; memset(&si,0,sizeof si); si.cb=sizeof si;
    if(wr){ si.dwFlags=STARTF_USESTDHANDLES; si.hStdInput=GetStdHandle(STD_INPUT_HANDLE); si.hStdOutput=si.hStdError=wr; }
    char buf[8192]; snprintf(buf,sizeof buf,"%s",cl); DWORD t0=GetTickCount(); DWORD code=0;
    snprintf(t_stop,sizeof t_stop,"%s",stop?stop:""); lm_setenv("LMTUI_CHILD","1");
    if(!CreateProcessA(NULL,buf,NULL,NULL,TRUE,CREATE_NEW_PROCESS_GROUP,NULL,NULL,&si,&pi)) printf("CreateProcess failed (%lu)\n",GetLastError());
    else { t_child=pi.dwProcessId; if(wr){ CloseHandle(wr); wr=NULL; }
        if(rd){ HANDLE ho=GetStdHandle(STD_OUTPUT_HANDLE); char b[4096]; DWORD n,w;
            while(ReadFile(rd,b,sizeof b,&n,NULL)&&n){ WriteFile(ho,b,n,&w,NULL); if(lf){ fwrite(b,1,n,lf); fflush(lf); } } }
        WaitForSingleObject(pi.hProcess,INFINITE); t_child=0;
        GetExitCodeProcess(pi.hProcess,&code); CloseHandle(pi.hProcess); CloseHandle(pi.hThread); }
    lm_setenv("LMTUI_CHILD",""); if(wr) CloseHandle(wr); if(rd) CloseHandle(rd);
    if(lf){ fprintf(lf,"\n[exit %lu, %.1f s]\n",code,(GetTickCount()-t0)/1000.0); fclose(lf); }
    printf("\n\x1b[1;%dm[exit %lu, %.1f s]\x1b[0m  \x1b[2mlog: %s\\%s\x1b[0m\n",code?31:32,code,(GetTickCount()-t0)/1000.0,exe_dir(),lp);
    if(lpo) snprintf(lpo,lcap,"%s\\%s",exe_dir(),lp);
    return code;
}

/* ---- form env fields: pushed into this process's env (inherited by the child), then restored ---- */
static char t_saved[40][2][T_MAXL]; static int t_nsaved;
static void t_env_push(const char*k,const char*v){
    if(t_nsaved<40){ const char*o=getenv(k); snprintf(t_saved[t_nsaved][0],T_MAXL,"%s",k); snprintf(t_saved[t_nsaved][1],T_MAXL,"%s",o?o:""); t_nsaved++; }
    lm_setenv(k,v); }
static void t_env_pop(void){ while(t_nsaved>0){ t_nsaved--; lm_setenv(t_saved[t_nsaved][0],t_saved[t_nsaved][1]); } }
static int t_cust;                                                 /* New model preset = custom: shape fields (vis 5) shown */
static int t_mode(const TCmd*c, char vals[][T_MAXL]){
    t_cust=0; for(int i=0;i<T_NF&&c->f[i].label;i++) if(!strcmp(c->f[i].label,"preset")) t_cust=!strcmp(vals[i],"custom");
    for(int i=0;i<T_NF&&c->f[i].label;i++) if(c->f[i].kind==F_MEM) return !strcmp(vals[i],"engram")?2:!strcmp(vals[i],"ngram")?1:0;
    return -1; }
static int t_adv;                                                  /* F2: show advanced fields (vis 4) */
static int t_vis(const TFld*f,int mode){ return !f->vis || (f->vis==5 ? t_cust : f->vis==4 ? t_adv : f->vis==3 ? mode>0 : mode==f->vis); }
/* hidden or empty fields are pushed as unset, so a loaded profile cannot leak into the other memory mode */
static void t_form_env(const TCmd*c, char vals[][T_MAXL]){
    int mode=t_mode(c,vals);
    for(int i=0;i<T_NF&&c->f[i].label;i++){ const TFld*f=&c->f[i];
        if(f->kind==F_ENV){ char v[T_MAXL]="";
            if(t_vis(f,mode)&&vals[i][0]){ snprintf(v,sizeof v,"%s",vals[i]);
                if(f->ext&&strchr(f->ext,'*')) GetFullPathNameA(vals[i],sizeof v,v,NULL); }   /* file paths saved absolute */
            t_env_push(f->arg,v); }
        else if(f->kind==F_MEM){ t_env_push("CUDALM_NGRAM",mode?"1":"0"); t_env_push("CUDALM_ENGRAM",mode==2?"1":"0"); } } }
static void t_new_stats(const TCmd*c, char vals[][T_MAXL]){
    int ci=(int)(c-T_CMDS), ip=t_fi(ci,"preset"), ie=t_fi(ci,"experts"); char err[200];
    Cfg cf; if(ip<0||!preset(vals[ip],&cf)){ printf("  \x1b[2m(unknown preset)\x1b[0m\n"); return; }
    cf.n_mtp=0; cf.n_exp=ie<0?0:atoi(vals[ie]); if(cf.n_exp<=1) cf.n_exp=0;
    t_form_env(c,vals);
    if(!shape_env(&cf,err,sizeof err)){ printf("  \x1b[31m%s\x1b[0m\n",err); t_env_pop(); return; }
    cf.pool=env_i("CUDALM_UNIPOOL",0)&&cf.n_exp>1; if(!layout_apply(&cf)){ t_env_pop(); return; } cfg_derive(&cf);
    int kl[64], k=topk_parse(getenv("CUDALM_TOPK"),getenv("CUDALM_TOPK_L"),kl);
    MemCfg m; mem_from_env(&m,cf.L);
    printf("\x1b[1mparams\x1b[0m  D=%d L=%d H=%d KV=%d F=%d \x1b[2m(V=%d from the preset; the first train resizes it to the data's vocab)\x1b[0m\n",
           cf.D,cf.L,cf.H,cf.KV,cf.F,cf.V);
    pstats_print(&cf,&m,k,kl,env_i("CUDALM_ATTN_GATE",0)!=0);
    t_env_pop(); }

/* help for the selected field: own text, else the env registry's; always 3 lines so the screen doesn't jump */
static void t_help(const TFld*f, const char*v, const char*run){
    const char*h=f?f->help:run?run:"Run: create the model (F5 from anywhere).\nThe settings above are saved in its .bin.env and reapplied by every later command."; int n=0;
    if(!h && f->kind==F_ENV){ const EnvReg*r=envreg_find(f->arg,strlen(f->arg)); if(r) h=r->help; }
    printf("\n");
    for(const char*p=h; p&&*p&&n<2; n++){ const char*e=strchr(p,'\n'); int l=e?(int)(e-p):(int)strlen(p);
        printf("  \x1b[36m%.*s\x1b[0m\n",l,p); p=e?e+1:p+l; }
    Cfg cf; if(f && f->kind==F_CHOICE && !strcmp(f->label,"preset") && preset(v,&cf)){
        printf("  \x1b[1;36m%s: D=%d L=%d heads=%d kv=%d ffn=%d ctx=%d vocab=%d%s\x1b[0m\n",v,cf.D,cf.L,cf.H,cf.KV,cf.F,cf.T,cf.V,
            cf.win?" (windowed)":""); n++; }
    for(;n<3;n++) printf("\n"); }

/* ---- template editor: which columns become the document, previewed exactly as the model will see it ---- */
static void t_tplshow(const char*row,const char*tpl,int maxl){
    static DWs w; const char*t; size_t n, i=0; int wd=t_cols()-4, l=0; if(wd<40) wd=40; if(maxl<3) maxl=3;
    if(!tpl_render(row,tpl,&w,&t,&n)){ printf("  \x1b[31m(no document from this row with this template)\x1b[0m\n"); return; }
    for(;i<n&&l<maxl;l++){ size_t e=i,c=i; int k=0; while(e<n&&t[e]!='\n') e++;
        while(c<e&&k<wd){ if(((unsigned char)t[c]&0xC0)!=0x80) k++; c++; } while(c<e&&((unsigned char)t[c]&0xC0)==0x80) c++;
        int mk=e-i>=4&&t[i]=='<'&&t[i+1]=='|'&&t[e-1]=='>';
        printf("  %s%.*s%s\x1b[0m\n",mk?"\x1b[1;35m":"",(int)(c-i),t+i,c<e?"\x1b[2m...":""); i=e+1; }
    if(i<n) printf("  \x1b[2m... %zu characters in all\x1b[0m\n",n); }
static int t_pickcols(const char*ttl,char nm[][64],const char*kd,int nc,int strs,int multi,const char cur[][64],int ncur,char out[][64]){
    int n=0, ix[64]; memset(t_mark,0,sizeof t_mark);
    for(int i=0;i<nc;i++) if(!strs||kd[i]=='s'){ ix[n]=i; snprintf(t_items[n],260,"%s",nm[i]); snprintf(t_info[n],T_INFO,"%s",d_kname(kd[i]));
        for(int j=0;j<ncur;j++) if(!strcmp(cur[j],nm[i])) t_mark[n]=1; n++; }
    if(!n){ t_note("no %scolumns to pick",strs?"text ":""); return 0; }
    t_multi=multi; int s=t_pick(ttl,multi?"Space mark several (joined with a blank line, in list order)   Enter use the marked ones (or the highlighted one)   Esc cancel"
                                      :"Enter pick   Esc cancel",n,0); t_multi=0;
    if(s<0) return 0; int m=0; if(multi) for(int i=0;i<n&&m<8;i++) if(t_mark[i]) snprintf(out[m++],64,"%s",nm[ix[i]]);
    if(!m) snprintf(out[m++],64,"%s",nm[ix[s]]); return m; }
static int t_colmap(const char*title,const char*rj,char*tpl,int cap){   /* 1 = accepted (tpl updated) */
    int nr=0; while(nr<8&&d_row(rj,nr)) nr++; if(!nr){ t_note("no rows to preview"); return 0; }
    DTpl t; char e[200], cur[1024]; if(!tpl_parse(tpl,&t,e,sizeof e)) memset(&t,0,sizeof t); int ri=0;
    for(;;){ const char*row=d_row(rj,ri); char nm[64][64],kd[64]; int nc=d_cols(row,nm,kd,64), shown=nc<12?nc:12, wd=t_cols()-30; if(wd<20) wd=20;
        tpl_fmt(&t,cur,sizeof cur); const char*how=tpl_auto(row);
        t_head(title); printf("  \x1b[1mcolumns\x1b[0m  \x1b[2mrow %d of %d\x1b[0m\n",ri+1,nr);
        for(int i=0;i<shown;i++){ DBuf x={0}; const char*v=jget(row,nm[i]); if(v) jval(v,&x); char*sv=x.p?x.p:(char*)"";
            for(char*q=sv;*q;q++) if(*q=='\n'||*q=='\r'||*q=='\t') *q=' ';
            int c=0,k=0; while(sv[c]&&k<wd){ if(((unsigned char)sv[c]&0xC0)!=0x80) k++; c++; } while(((unsigned char)sv[c]&0xC0)==0x80) c++;
            printf("    \x1b[36m%-16.16s\x1b[0m \x1b[2m%-6s\x1b[0m %.*s%s\n",nm[i],d_kname(kd[i]),c,sv,sv[c]?"...":""); free(x.p); }
        if(nc>shown) printf("    \x1b[2m+%d more\x1b[0m\n",nc-shown);
        if(cur[0]) printf("\n  \x1b[1mtemplate\x1b[0m  %s\n",cur);
        else printf("\n  \x1b[1mtemplate\x1b[0m  auto: %s\n",how?how:"\x1b[31mnothing recognised - pick columns with T / P / C\x1b[0m");
        printf("\n  \x1b[2m--- what the model sees ---\x1b[0m\n"); t_tplshow(row,cur,t_rows()-shown-14);
        printf("\x1b[%d;1H\x1b[2mT text  P prompt + answer  C chat column  S system  O tools  A auto  Left/Right row  Enter use  Esc cancel\x1b[0m",t_rows()); fflush(stdout);
        int k=t_key();
        if(k==27) return 0;
        if(k==13){ snprintf(tpl,cap,"%s",cur); return 1; }
        if(k==K_LEFT&&ri>0) ri--; else if(k==K_RIGHT&&ri<nr-1) ri++;
        else if(k=='a'||k=='A') memset(&t,0,sizeof t);
        else if(k=='t'||k=='T'){ char o[8][64]; int m=t_pickcols("text: the columns that make each document",nm,kd,nc,1,1,t.c[0],t.n[0],o);
            if(m){ memset(&t,0,sizeof t); memcpy(t.c[0],o,sizeof o); t.n[0]=m; } }
        else if(k=='p'||k=='P'){ char u[8][64],a[8][64]; int mu=t_pickcols("prompt (user turn): mark several to join them, e.g. question + context",nm,kd,nc,1,1,t.c[2],t.n[2],u); if(!mu) continue;
            int ma=t_pickcols("answer (assistant turn)",nm,kd,nc,1,1,t.c[3],t.n[3],a); if(!ma) continue;
            t.n[0]=t.n[1]=0; memcpy(t.c[2],u,sizeof u); t.n[2]=mu; memcpy(t.c[3],a,sizeof a); t.n[3]=ma; }
        else if(k=='c'||k=='C'){ char o[8][64]; int m=t_pickcols("chat: the column holding the list of turns ({role, content} or {from, value})",nm,kd,nc,0,0,NULL,0,o);
            if(m){ t.n[0]=t.n[2]=t.n[3]=0; snprintf(t.c[1][0],64,"%s",o[0]); t.n[1]=1; } }
        else if(k=='s'||k=='S'||k=='o'||k=='O'){ int key=k=='s'||k=='S'?4:5;
            if(t.n[key]){ t.n[key]=0; continue; }                   /* pressed again: remove */
            if(!t.n[1]&&!t.n[2]){ t_note("system / tools go with a chat (C) or prompt + answer (P) template"); continue; }
            char o[8][64]; int m=t_pickcols(key==4?"system prompt column":"tools column (non-text values are kept as JSON)",nm,kd,nc,0,0,NULL,0,o);
            if(m){ snprintf(t.c[key][0],64,"%s",o[0]); t.n[key]=1; } } } }
static void t_template(const TCmd*c,char V[][T_MAXL],char*v){       /* Enter on a template field: peek the source, open the editor */
    const char*src=V[0]; if(!src[0]){ t_note("set '%s' first",c->f[0].label); return; }
    if(t_is(c,"Download text")&&_strnicmp(src,"hf:",3)){ t_note("the editor needs an hf: source (Browse Hugging Face); for a URL type the template"); t_edit("template",v,T_MAXL); return; }
    printf("\x1b[%d;1H\x1b[2K  reading the first rows...",t_rows()); fflush(stdout);
    DBuf rj={0}; char e[300]; if(!d_peek(src,&rj,e,sizeof e)){ t_note("%s",e); t_edit("template",v,T_MAXL); free(rj.p); return; }
    char ttl[700]; snprintf(ttl,sizeof ttl,"template for %s",src); t_colmap(ttl,rj.p,v,T_MAXL); free(rj.p); }

/* ---- form values (kept per form for the session; other forms' results are handed in with t_set) ---- */
static char t_vals[T_NCMD][T_NF][T_MAXL]; static int t_vinit[T_NCMD];
static void t_form_init(int ci){
    if(t_vinit[ci]) return; const TCmd*c=&T_CMDS[ci]; char (*V)[T_MAXL]=t_vals[ci]; int nf=0; while(nf<T_NF && c->f[nf].label) nf++;
    int envmem=getenv("CUDALM_NGRAM")||getenv("CUDALM_ENGRAM");
    for(int i=0;i<nf;i++){ const TFld*f=&c->f[i]; const char*e=f->kind==F_ENV?getenv(f->arg):NULL;
        if(e&&*e&&!strncmp(f->arg,"NGRAM_",6)) envmem=1;
        snprintf(V[i],T_MAXL,"%s",e&&*e?e:f->def);
        if(f->kind==F_FILE && f->def[0] && !d_exists(f->def)){ char b[300]; snprintf(b,sizeof b,"bench\\%s",f->def);
            if(d_exists(b)) snprintf(V[i],T_MAXL,"%s",b); } }
    for(int i=0;i<nf;i++) if(c->f[i].kind==F_MEM && envmem){   /* a loaded profile decides the mode */
        MemCfg m; mem_from_env(&m,64); snprintf(V[i],T_MAXL,"%s",mem_name(m.mode)); }
    t_vinit[ci]=1; }
static void t_set(const char*title,const char*label,const char*v){
    int ci=t_ci(title), fi=t_fi(ci,label); if(fi<0) return; t_form_init(ci); snprintf(t_vals[ci][fi],T_MAXL,"%s",v); }
/* Mix: pair k (k>=2) stays hidden until pair k-1 has a corpus */
static int t_mixhide(const TCmd*c,char v[][T_MAXL],int i){ if(!t_is(c,"Mix corpora")||i<6) return 0; int k=(i-4)/2; return !v[4+2*(k-1)][0]&&!v[4+2*k][0]; }

/* ---- live checks + output preview for data forms ---- */
static char t_outp[1200]; static int t_nprob;
static void t_prob(int draw,const char*fmt,...){ t_nprob++; if(!draw) return; va_list ap; va_start(ap,fmt);
    printf("  \x1b[31mx \x1b[0m"); vprintf(fmt,ap); printf("\n"); va_end(ap); }
static const struct { const char*t,*l; double lo,hi; int cnt; } T_NUM[]={
    {"Download text","tokens",1e3,1e13,1},{"Train tokenizer","vocab",512,65535,0},{"Train tokenizer","sample MB",1,1e6,0},
    {"Build corpus","val %",0,50,0},{"Mix corpora","train tokens",0,1e13,1},{"Mix corpora","max repeat",0.01,1e3,0},{"Mix corpora","val tokens",0,1e11,1}};
static double t_mixtab(char v[][T_MAXL],int draw){                 /* live blend table; returns the train tokens written */
    char d[10][1200],a[32],b[32]; double n[10],w[10],ws=0,sum=0; unsigned h[10]; int k=0, idx[10];
    double rep=atof(v[2]); if(rep<=0) rep=1; double tot=d_count(v[1]);
    for(int i=0;i<10;i++){ const char*cv=v[4+2*i],*wv=v[5+2*i]; if(!cv[0]) continue;
        data_resolve(cv,d[k],sizeof d[k]); char p[1300]; snprintf(p,sizeof p,"%s\\train.bin",d[k]); n[k]=d_exists(p)?d_fsize(p)/2.0:0;
        w[k]=wv[0]?atof(wv):0; h[k]=t_tokhash(d[k]); idx[k]=i+1;
        if(w[k]<=0) t_prob(draw,"'weight %d' must be > 0",i+1); else ws+=w[k];
        if(k&&h[k]&&h[0]&&h[k]!=h[0]) t_prob(draw,"corpus %d uses tokenizer %06x, corpus 1 uses %06x: mix refuses different tokenizers",i+1,h[k],h[0]);
        k++; }
    if(!k||ws<=0) return 0;
    if(tot<=0){ tot=1e300; for(int i=0;i<k;i++) if(w[i]>0&&n[i]>0){ double cap=n[i]*rep/(w[i]/ws); if(cap<tot) tot=cap; } if(tot>=1e300) tot=0; }
    if(draw) printf("  \x1b[2m    %-28s %9s %7s %9s %7s  tokenizer\x1b[0m\n","corpus","tokens","share","takes","repeat");
    for(int i=0;i<k;i++){ double want=w[i]>0?tot*w[i]/ws:0, take=want<n[i]*rep?want:n[i]*rep; sum+=take;
        if(draw) printf("  %2d  %-28.28s %9s %6.1f%% %9s %6.2fx  %06x%s\n",idx[i],d_base(d[i]),t_hn(n[i],a),100*w[i]/ws,t_hn(take,b),n[i]>0?take/n[i]:0,h[i],
                        take<want*0.98?"  \x1b[33mruns out (max repeat)\x1b[0m":""); }
    if(draw) printf("  \x1b[2m    blend: %s train tokens\x1b[0m\n",t_hn(sum,a));
    return sum; }
static int t_panel(const TCmd*c,char v[][T_MAXL],int draw){       /* returns the number of problems; sets t_outp */
    char r[1200],o[1200],q[1300],a[32],b[32]; double est=-1; int ci=(int)(c-T_CMDS), mode=t_mode(c,v), isnew=1, follow=0, pbad=0, of=-1;
    t_nprob=0; t_outp[0]=0;
    if(t_is(c,"Build corpus")){ isnew=strcmp(v[0],"append")!=0; follow=!strcmp(v[6],"1"); }
    for(size_t i=0;i<sizeof T_NUM/sizeof T_NUM[0];i++) if(t_is(c,T_NUM[i].t)){ int f=t_fi(ci,T_NUM[i].l); if(f<0||!v[f][0]) continue;
        char*e; double x=T_NUM[i].cnt?d_count(v[f]):strtod(v[f],&e); if(!T_NUM[i].cnt&&*e) x=-1;
        if(x<T_NUM[i].lo||x>T_NUM[i].hi) t_prob(draw,"'%s' must be a number from %g to %g%s",T_NUM[i].l,T_NUM[i].lo,T_NUM[i].hi,T_NUM[i].cnt?" (k / M / B suffixes ok)":""); }
    { int f=t_fi(ci,"template"); DTpl tp; char te[200];
        if(f>=0&&v[f][0]){ if(!tpl_parse(v[f],&tp,te,sizeof te)) t_prob(draw,"'template': %s",te);
            else if(t_is(c,"Download text")&&_strnicmp(v[0],"hf:",3)&&_strnicmp(v[0],"http",4)) t_prob(draw,"'template' applies to hf: and URL sources, not the built-in profiles"); } }
    for(int i=0;i<T_NF&&c->f[i].label;i++){ const TFld*f=&c->f[i]; if(!v[i][0]||!t_vis(f,mode)||t_mixhide(c,v,i)) continue;
        if(f->kind==F_TOKD){ data_resolve(v[i],r,sizeof r);
            if(!t_tokhash(r)) t_prob(draw,"'%s': no tokenizer.bin in %s - Enter to pick one%s",f->label,r,t_is(c,"Build corpus")?", or make one with Train tokenizer or Tokenizer from model":""); }
        else if(f->kind==F_DATA && !(t_is(c,"Build corpus")&&isnew)){ data_resolve(v[i],r,sizeof r); snprintf(q,sizeof q,"%s\\train.bin",r);
            if(!d_exists(q)) t_prob(draw,"'%s': no corpus (train.bin) in %s",f->label,r); }
        else if(f->kind==F_PATH && !follow){ char sp[1024]; snprintf(sp,sizeof sp,"%s",v[i]);
            for(char*p=sp,*e;p;p=e){ e=strchr(p,';'); if(e) *e++=0; while(*p==' ') p++; if(!*p) continue; data_resolve(p,r,sizeof r);
                if(!d_exists(r)){ t_prob(draw,"'%s': not found: %s",f->label,p); pbad=1; } } } }
    int nf=0; double tb;
    if(t_is(c,"Download text")){ of=1; double t=d_count(v[2]); if(t>0) est=t*4.6; }
    else if(t_is(c,"Convert files")){ of=1; if(v[0][0]&&!pbad){ tb=t_spec(v[0],".parquet;.json;.jsonl",&nf); est=tb*2;
        if(draw) printf("  \x1b[2msource %s in %d file(s); parquet grows about 2x as .jsonl\x1b[0m\n",t_hb(tb,a),nf); } }
    else if(t_is(c,"Train tokenizer")){ of=0; if(v[1][0]&&!pbad){ tb=t_spec(v[1],T_TEXTIN,&nf);
        if(!nf) t_prob(draw,"'text': no .jsonl / .json / .txt files there");
        else if(draw) printf("  \x1b[2mtext %s in %d file(s); reads up to %s MB of it\x1b[0m\n",t_hb(tb,a),nf,v[3]); } }
    else if(t_is(c,"Build corpus")){ of=1; if(v[2][0]&&!pbad&&!follow){ tb=t_spec(v[2],T_TEXTIN,&nf); est=tb/4.5*2;
            if(!nf) t_prob(draw,"'text': no .jsonl / .json / .txt files there");
            else if(draw) printf("  \x1b[2mtext %s in %d file(s) -> about %s tokens\x1b[0m\n",t_hb(tb,a),nf,t_hn(tb/4.5,b)); }
        if(!isnew&&v[1][0]&&v[3][0]){ char tk[1200]; data_resolve(v[1],r,sizeof r); data_resolve(v[3],tk,sizeof tk); unsigned h1=t_tokhash(r),h2=t_tokhash(tk);
            if(h1&&h2&&h1!=h2) t_prob(draw,"append: the corpus uses tokenizer %06x, 'tokenizer' is %06x - they must be identical",h1,h2); } }
    else if(t_is(c,"Mix corpora")){ of=0; est=t_mixtab(v,draw)*2; }
    if(of>=0&&v[of][0]){
        if(t_is(c,"Build corpus")&&!isnew) data_resolve(v[of],o,sizeof o); else data_outdir(v[of],o,sizeof o);
        snprintf(t_outp,sizeof t_outp,"%s",o); double fr=t_free(o);
        if(draw){ printf("  \x1b[36moutput\x1b[0m %s",o); if(fr>=0) printf("   \x1b[2m%s free",t_hb(fr,a)); if(est>0) printf(", needs about %s",t_hb(est,b)); printf("\x1b[0m\n"); }
        if(est>0&&fr>=0&&est>fr) t_prob(draw,"not enough space: needs about %s, %s free on that drive",t_hb(est,a),t_hb(fr,b));
        if(draw&&d_isdir(o)&&t_is(c,"Download text")) printf("  \x1b[2mthe folder exists: the download resumes into it\x1b[0m\n");
        if(draw&&d_isdir(o)&&t_is(c,"Convert files")) printf("  \x1b[2mthe folder exists: converted files are added to it\x1b[0m\n");
        snprintf(q,sizeof q,"%s\\tokenizer.bin",o); if(t_is(c,"Train tokenizer")&&d_exists(q)) t_prob(draw,"%s already has a tokenizer - pick a new name",o);
        snprintf(q,sizeof q,"%s\\train.bin",o);
        if(((t_is(c,"Build corpus")&&isnew)||t_is(c,"Mix corpora"))&&d_exists(q)) t_prob(draw,"%s already holds a corpus - pick a new name%s",o,t_is(c,"Build corpus")?" (or mode append)":""); }
    if(t_is(c,"Corpus info")&&v[0][0]) data_resolve(v[0],t_outp,sizeof t_outp);
    return t_nprob; }

/* ---- after a run: hand the output to the next forms and offer the next step ---- */
static const struct { const char*from,*to; } T_NEXT[]={{"Download text","Build corpus"},{"Convert files","Build corpus"},
    {"Train tokenizer","Build corpus"},{"Build corpus","Corpus info"},{"Mix corpora","Corpus info"},{"Corpus info","Train"},{"Loss weights","Train"}};
static void t_handoff(const TCmd*c,const char*o){
    char q[1300]; snprintf(q,sizeof q,"%s\\train.bin",o); int corpus=d_exists(q);
    if(t_is(c,"Download text")||t_is(c,"Convert files")){ t_set("Build corpus","text",o); t_set("Train tokenizer","text",o); }
    else if(t_is(c,"Train tokenizer")){ t_set("Build corpus","tokenizer",o); t_set("Canon map","tokenizer",o); }
    else if(corpus&&(t_is(c,"Build corpus")||t_is(c,"Mix corpora")||t_is(c,"Corpus info"))){
        if(!t_is(c,"Corpus info")) t_set("Corpus info","corpus",o);
        t_set("Train","data dir",o); t_set("Eval","data dir",o); if(t_is(c,"Build corpus")) t_set("Loss weights","corpus",o); } }
static void t_form(const TCmd*c);
static void t_after(const TCmd*c,DWORD code,const char*lp,const char*o){
    const char*nx=NULL; char exe[MAX_PATH],cl[2600],lp2[700];
    if(!code) for(size_t i=0;i<sizeof T_NEXT/sizeof T_NEXT[0];i++) if(t_is(c,T_NEXT[i].from)) nx=T_NEXT[i].to;
    int info=!code&&o[0]&&d_isdir(o)&&(t_is(c,"Build corpus")||t_is(c,"Mix corpora")||t_is(c,"Train tokenizer"));
    if(!code&&o[0]) t_handoff(c,o);
    for(;;){
        printf("\n");
        if(nx) printf("  \x1b[1mEnter\x1b[0m next: %s (filled in)   ",nx);
        if(info) printf("\x1b[1mI\x1b[0m info on the output   ");
        printf("\x1b[1mL\x1b[0m open the log   \x1b[2many other key: back\x1b[0m\n"); fflush(stdout);
        while(_kbhit()) _getch(); int k=t_key();
        if(k=='l'||k=='L'){ ShellExecuteA(NULL,"open",lp,NULL,NULL,SW_SHOWNORMAL); continue; }
        if((k=='i'||k=='I')&&info){ GetModuleFileNameA(0,exe,sizeof exe); cl[0]=0; t_quote(cl,sizeof cl,exe); t_quote(cl,sizeof cl,"data");
            t_quote(cl,sizeof cl,"info"); t_quote(cl,sizeof cl,o); t_spawn(cl,NULL,"data-info",lp2,sizeof lp2); continue; }
        if(k==13&&nx) t_form(&T_CMDS[t_ci(nx)]);
        return; } }

static void t_form(const TCmd*c){
    int ci=(int)(c-T_CMDS), nf=0; while(nf<T_NF && c->f[nf].label) nf++;
    t_form_init(ci); char (*V)[T_MAXL]=t_vals[ci];
    int hasadv=0; for(int i=0;i<nf;i++) if(c->f[i].vis==4) hasadv=1;
    int sel=0; char msg[300]="";
    for(;;){
        int mode=t_mode(c,V), vis[T_NF], nv=0;
        for(int i=0;i<nf;i++) if(t_vis(&c->f[i],mode)&&!t_mixhide(c,V,i)) vis[nv++]=i;
        if(sel>nv) sel=nv;
        char cl[4096], err[200]; int ok=t_cmdline(c,V,cl,sizeof cl,err,sizeof err);
        t_head(c->title); printf("  \x1b[2m%s\x1b[0m\n\n",c->help);
        for(int j=0;j<nv;j++){ const TFld*f=&c->f[vis[j]]; const char*v=V[vis[j]];
            printf("%s %-14s\x1b[0m %s",j==sel?"\x1b[7m>":" ",f->label,v[0]?v:"\x1b[2m-\x1b[0m");
            if(f->kind==F_ENV) printf("   \x1b[2m%s%s\x1b[0m",f->arg,t_choices(f)?"  <- ->":"");
            else if(f->kind==F_MEM||f->kind==F_FLAG) printf("   \x1b[2m<- ->\x1b[0m");
            printf("\n"); }
        if(mode>=0 || c->run) t_help(sel<nv?&c->f[vis[sel]]:NULL,sel<nv?V[vis[sel]]:"",c->run);
        printf("\n%s [ Run ]\x1b[0m\n\n\x1b[2m%s\x1b[0m\n",sel==nv?"\x1b[7m>":" ",ok?cl:err);
        t_panel(c,V,1);
        if(mode>=0){ printf("\n"); t_new_stats(c,V); }
        if(msg[0]) printf("\x1b[33m%s\x1b[0m\n",msg); msg[0]=0;
        printf("\x1b[%d;1H\x1b[2mUp/Down move  Enter edit/pick  Left/Right cycle choice  Del clear  F5 run  %sEsc back\x1b[0m",t_rows(),
               hasadv?(t_adv?"F2 hide advanced  ":"F2 advanced fields  "):""); fflush(stdout);
        int k=t_key();
        if(k==27) return;
        if(k==K_F2&&hasadv){ t_adv^=1; continue; }
        if(k==K_UP) sel=sel>0?sel-1:nv; else if(k==K_DOWN) sel=sel<nv?sel+1:0;
        else if(k==K_F5 || (k==13 && sel==nv)){
            if(!ok){ snprintf(msg,sizeof msg,"%s",err); continue; }
            int np=t_panel(c,V,0); char outp[1200]; snprintf(outp,sizeof outp,"%s",t_outp);
            if(np&&!t_yes("%d problem%s shown above - run anyway? (y/N) ",np,np>1?"s":"")) continue;
            char sp[1100]=""; if(!strcmp(c->cmd,"train")){ model_path(V[0],sp,sizeof sp-5); strcat(sp,".stop"); }
            char tag[200],lp[700]; snprintf(tag,sizeof tag,"%s%s%.80s",c->cmd,nf&&V[0][0]?"-":"",nf?V[0]:"");
            for(char*p=tag;*p;p++) if(*p==' ') *p='-';
            t_form_env(c,V); DWORD code=t_spawn(cl,sp[0]?sp:NULL,tag,lp,sizeof lp); t_env_pop();
            t_after(c,code,lp,outp); }
        else if(sel<nv){
            const TFld*f=&c->f[vis[sel]]; char*v=V[vis[sel]]; const char*ch=t_choices(f); int n; char r[1200];
            if(k==K_DEL) v[0]=0;
            else if((k==K_LEFT||k==K_RIGHT) && ch){ t_choice_items(ch,&n); int cur=0;
                for(int i=0;i<n;i++) if(!strcmp(t_items[i],v)) cur=i;
                cur=(cur+(k==K_RIGHT?1:n-1))%n; snprintf(v,T_MAXL,"%s",t_items[cur]); }
            else if(k==13){
                if(f->kind==F_OPT&&f->arg&&!strcmp(f->arg,"--template")) t_template(c,V,v);
                else if(t_is(c,"Build corpus")&&vis[sel]==1){           /* new: a name to create; append: corpora with the same tokenizer */
                    if(strcmp(V[0],"append")) t_choose(f->label,0,v,T_MAXL,1);
                    else { n=t_list_dirs(0); data_resolve(V[3],r,sizeof r); unsigned h=t_tokhash(r); if(h) n=t_keep_tok(n,h);
                        t_pv=t_corpus_pv; t_choose(f->label,n,v,T_MAXL,1); t_pv=0; } }
                else if(t_is(c,"Mix corpora")&&f->kind==F_DATA){
                    n=t_list_dirs(0); if(vis[sel]!=4&&V[4][0]){ data_resolve(V[4],r,sizeof r); unsigned h=t_tokhash(r); if(h) t_mark_tok(n,h); }
                    t_pv=t_corpus_pv; t_choose(f->label,n,v,T_MAXL,1); t_pv=0;
                    if(v[0]&&!V[vis[sel]+1][0]) snprintf(V[vis[sel]+1],T_MAXL,"1"); }
                else if(f->kind==F_MODEL){ n=t_list_models(); t_choose(f->label,n,v,T_MAXL,0); }
                else if(f->kind==F_DATA||f->kind==F_TOKD){ n=t_list_dirs(f->kind==F_TOKD); t_pv=t_corpus_pv; t_choose(f->label,n,v,T_MAXL,1); t_pv=0; }
                else if(f->kind==F_PATH&&f->ext&&!strcmp(f->ext,T_TEXTIN)) t_pick_text(f->label,v,T_MAXL,f->ext);
                else if(f->kind==F_DIR||f->kind==F_PATH) t_browse(f->label,v,T_MAXL,f->ext,f->kind==F_PATH);
                else if(f->kind==F_FILE){ n=t_list_glob("*.txt"); t_choose(f->label,n,v,T_MAXL,0); }
                else if(f->kind==F_ENV && f->ext && strchr(f->ext,'*')){ n=t_list_glob(f->ext); t_choose(f->label,n,v,T_MAXL,0); }
                else if(ch){ t_choice_items(ch,&n); t_choose(f->label,n,v,T_MAXL,0); }
                else t_edit(f->label,v,T_MAXL);
            }
        }
    }
}

/* ---- settings browser over g_envreg: the process environment, or one model's config file ---- */
#define T_SMAX 200
typedef struct { int file; char path[1100]; int n; char k[T_SMAX][64]; char v[T_SMAX][T_MAXL]; } TStore;
static TStore t_st;
static const char* st_get(TStore*s,const char*k){
    if(!s->file){ const char*e=getenv(k); return e&&*e?e:NULL; }
    for(int i=0;i<s->n;i++) if(!strcmp(s->k[i],k)) return s->v[i];
    return NULL; }
static void st_save(TStore*s){
    FILE*f=fopen(s->path,"wb"); if(!f) return;
    fprintf(f,"# cudalm model config (edited in the TUI). KEY=VALUE; the environment overrides these.\n");
    for(int i=0;i<s->n;i++) fprintf(f,"%s=%s\n",s->k[i],s->v[i]);
    fclose(f); }
static void st_set(TStore*s,const char*k,const char*v){
    if(!s->file){ lm_setenv(k,v); snprintf(t_profile,sizeof t_profile,"(edited)"); return; }
    int i=0; while(i<s->n && strcmp(s->k[i],k)) i++;
    if(!v||!*v){ if(i<s->n){ for(int j=i+1;j<s->n;j++){ memcpy(s->k[j-1],s->k[j],64); memcpy(s->v[j-1],s->v[j],T_MAXL); } s->n--; } }
    else { if(i==s->n){ if(s->n>=T_SMAX) return; s->n++; snprintf(s->k[i],64,"%s",k); } snprintf(s->v[i],T_MAXL,"%s",v); }
    st_save(s); }
static void st_load(TStore*s,const char*path){
    s->file=1; s->n=0; snprintf(s->path,sizeof s->path,"%s",path);
    FILE*f=fopen(path,"rb"); if(!f) return; char l[1024];
    while(fgets(l,sizeof l,f) && s->n<T_SMAX){ l[strcspn(l,"\r\n")]=0; char*eq=strchr(l,'='); if(!eq||eq==l||l[0]=='#') continue; *eq=0;
        snprintf(s->k[s->n],64,"%s",l); snprintf(s->v[s->n],T_MAXL,"%s",eq+1); s->n++; }
    fclose(f); }
static int t_load_profile(const char*path){   /* applies `set K=V` / `set "K=V"` lines of a .bat */
    FILE*f=fopen(path,"rb"); if(!f) return -1; char line[1024]; int n=0;
    while(fgets(line,sizeof line,f)){
        char*s=line; while(*s==' '||*s=='\t'||*s=='@') s++;
        if(_strnicmp(s,"set ",4)) continue; s+=4; while(*s==' ') s++;
        if(!_strnicmp(s,"/a",2)||!_strnicmp(s,"/p",2)) continue;
        size_t l=strcspn(s,"\r\n"); s[l]=0;
        if(*s=='"'){ s++; char*q=strrchr(s,'"'); if(q) *q=0; }
        char*eq=strchr(s,'='); if(!eq||eq==s) continue; *eq=0;
        if(strchr(s,' ')||strchr(eq+1,'%')) continue;               /* skip anything needing cmd expansion */
        lm_setenv(s,eq+1); n++;
    }
    fclose(f); return n;
}
static void t_reg(TStore*s,const char*title){
    static char extra[T_MAXL][64]; int sel=0;
    for(;;){
        int n=0, ne=0;
        for(const EnvReg*r=g_envreg;r->name&&n<T_MAXL-8;r++){ const char*v=st_get(s,r->name), *ev=getenv(r->name);
            char ov[160]=""; if(s->file&&v&&ev&&*ev&&strcmp(ev,v)) snprintf(ov,sizeof ov,"\x1b[0m\x1b[35m  [env overrides: %s]",ev);
            snprintf(t_items[n],260,"%-8s %s",r->grp,r->name);
            if(v) snprintf(t_info[n],T_INFO,"\x1b[0m\x1b[1;33m= %-12s\x1b[0m\x1b[2m %s%s",v,r->help,ov);
            else snprintf(t_info[n],T_INFO,"  %-12s %s",r->def[0]?r->def:"-",r->help);
            n++; }
        if(s->file){ for(int i=0;i<s->n&&ne<T_MAXL;i++) if(!envreg_find(s->k[i],strlen(s->k[i]))) snprintf(extra[ne++],64,"%s",s->k[i]); }
        else { char*blk=GetEnvironmentStringsA();
            for(char*p=blk;*p&&ne<T_MAXL;p+=strlen(p)+1){ const char*eq=strchr(p,'=');
                if(*p!='='&&eq&&t_envwanted(p)&&!envreg_find(p,eq-p)) snprintf(extra[ne++],64,"%.*s",(int)(eq-p),p); }
            FreeEnvironmentStringsA(blk); }
        for(int i=0;i<ne&&n<T_MAXL-4;i++){ const char*v=st_get(s,extra[i]); snprintf(t_items[n],260,"%-8s %s","other",extra[i]);
            snprintf(t_info[n],T_INFO,"\x1b[0m\x1b[1;33m= %s",v?v:""); n++; }
        int a0=n;
        if(!s->file){ snprintf(t_items[n],260,"[ load profile (*.bat) ]"); t_info[n++][0]=0; }
        snprintf(t_items[n],260,"[ add variable ]"); t_info[n++][0]=0;
        snprintf(t_items[n],260,"[ clear all set values ]"); t_info[n++][0]=0;
        int k=t_pick(title,"Enter edit (empty value = unset / default)  Esc back",n,sel);
        if(k<0) return; sel=k;
        if(k<a0){ char key[64]; const char*nm=strchr(t_items[k],' '); while(*nm==' ') nm++; snprintf(key,sizeof key,"%s",nm);
            const EnvReg*r=envreg_find(key,strlen(key)); char v[T_MAXL]; const char*cur=st_get(s,key); snprintf(v,sizeof v,"%s",cur?cur:"");
            t_head(title); printf("  %s\n  %s\n  default: %s%s\n",key,r?r->help:"",r&&r->def[0]?r->def:"-",
                                  r&&r->rt?"\n  per-run setting: never saved into a model config":"");
            if(t_edit(key,v,sizeof v)) st_set(s,key,v); }
        else if(!s->file && k==a0){ int m=t_list_glob("*.bat"); int p=t_pick("load profile: set lines are applied on top of the current vars",NULL,m,0);
            if(p>=0){ char nm[260]; snprintf(nm,sizeof nm,"%s",t_items[p]); t_load_profile(nm); snprintf(t_profile,sizeof t_profile,"%s",nm); } }
        else if(k==n-2){ char kv[T_MAXL]=""; t_head(title); printf("  NAME=value\n");
            if(t_edit("add",kv,sizeof kv)){ char*eq=strchr(kv,'='); if(eq&&eq>kv){ *eq=0; st_set(s,kv,eq+1); } } }
        else { for(const EnvReg*r=g_envreg;r->name;r++) if(st_get(s,r->name)) st_set(s,r->name,"");
            for(int i=0;i<ne;i++) st_set(s,extra[i],""); if(!s->file) snprintf(t_profile,sizeof t_profile,"(none)"); }
    }
}
static void t_model_cfg(void){
    int n=t_list_models(); int p=t_pick("model config: pick a model",NULL,n,0); if(p<0) return;
    char mp[1100], ttl[1300]; model_path(t_items[p],mp,sizeof mp-4); strcat(mp,".env"); st_load(&t_st,mp);
    snprintf(ttl,sizeof ttl,"config %s  (applied by every command on this model; the environment overrides)",mp);
    t_reg(&t_st,ttl); }

/* data roots: built-in, data_roots.txt (add / remove here) or CUDALM_DATA_ROOTS */
static void t_roots(void){
    int sel=0; char fp[1100]; snprintf(fp,sizeof fp,"%s\\data_roots.txt",exe_dir());
    for(;;){ char x[1200],a[32]; const char*env=getenv("CUDALM_DATA_ROOTS"); int ev=env&&*env, n=0, nr=0;
        for(int i=0;d_root(i,x,sizeof x)&&n<T_MAXL-2;i++,nr++){ snprintf(t_items[n],260,"%s",x); double fr=d_isdir(x)?t_free(x):-1;
            char fs[48]; if(fr>=0) snprintf(fs,sizeof fs,"%s free",t_hb(fr,a)); else snprintf(fs,sizeof fs,"(missing)");
            snprintf(t_info[n++],T_INFO,"%-18s %s",ev?"CUDALM_DATA_ROOTS":i<2?"built-in":"data_roots.txt",fs); }
        snprintf(t_items[n],260,"[ add a folder ]"); snprintf(t_info[n++],T_INFO,"browse to it; saved in data_roots.txt");
        snprintf(t_items[n],260,"[ open data_roots.txt ]"); snprintf(t_info[n++],T_INFO,"%s",fp);
        int s=t_pick(ev?"data roots   CUDALM_DATA_ROOTS is set: it replaces this list (edits here apply once it is cleared)"
                       :"data roots   bare text / tokenizer / corpus names are looked up here, in order; new outputs go to the one with most free space",
                     "Enter on a data_roots.txt folder removes it   Esc back",n,sel);
        if(s<0) return; sel=s;
        if(s==nr){ char v[1200]=""; t_browse("add data root",v,sizeof v,NULL,0);
            if(v[0]){ FILE*f=fopen(fp,"a"); if(f){ fprintf(f,"%s\n",v); fclose(f); } } }
        else if(s==nr+1){ FILE*f=fopen(fp,"a"); if(f) fclose(f); ShellExecuteA(NULL,"open",fp,NULL,NULL,SW_SHOWNORMAL); }
        else if(!ev&&s>=2&&t_yes("remove %s from data_roots.txt? The folder itself is not touched. (y/N) ",t_items[s])){
            size_t len; char*b=(char*)read_file(fp,&len); if(!b) continue; b=(char*)realloc(b,len+1); b[len]=0;
            FILE*f=fopen(fp,"wb"); int gone=0;
            for(char*l=b,*e;f&&*l;l=e){ e=strchr(l,'\n'); e=e?e+1:l+strlen(l); char t[1100]; snprintf(t,sizeof t,"%.*s",(int)(e-l),l); t[strcspn(t,"\r\n")]=0;
                if(!gone&&!strcmp(t,t_items[s])){ gone=1; continue; } fwrite(l,1,e-l,f); }
            if(f) fclose(f); free(b); } } }
/* every folder under the data roots: type, size, date; open / info / rename / Recycle Bin */
static void t_manage(void){
    static char pth[T_MAXL][1100]; int sel=0;
    for(;;){ int n=0; char r[1100],a[32];
        printf("\x1b[2J\x1b[H  scanning the data roots...\n"); fflush(stdout);
        for(int ri=0;d_root(ri,r,sizeof r)&&n<T_MAXL;ri++){ char pat[1200]; snprintf(pat,sizeof pat,"%s\\*",r); WIN32_FIND_DATAA fd; HANDLE h=FindFirstFileA(pat,&fd);
            if(h==INVALID_HANDLE_VALUE) continue;
            do{ if(n>=T_MAXL||!(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)||fd.cFileName[0]=='.') continue;
                char d[1200],q[1300]; snprintf(d,sizeof d,"%s\\%s",r,fd.cFileName); int nf=0,tn=0; double sz=t_tree(d,NULL,&nf);
                snprintf(q,sizeof q,"%s\\train.bin",d); const char*ty=d_exists(q)?"corpus":NULL;
                if(!ty){ snprintf(q,sizeof q,"%s\\tokenizer.bin",d); ty=d_exists(q)?"tokenizer":NULL; }
                if(!ty){ t_tree(d,T_TEXTIN,&tn); ty=tn?"text":"other"; }
                SYSTEMTIME st; FileTimeToSystemTime(&fd.ftLastWriteTime,&st);
                snprintf(pth[n],1100,"%s",d); snprintf(t_items[n],260,"%s",d);
                snprintf(t_info[n++],T_INFO,"%-9s %10s %7d files  %04d-%02d-%02d",ty,t_hb(sz,a),nf,st.wYear,st.wMonth,st.wDay); }while(FindNextFileA(h,&fd));
            FindClose(h); }
        if(!n){ t_note("no folders under the data roots yet"); return; }
        int s=t_pick("Manage data folders   (size = everything inside)","Enter actions   Esc back",n,sel); if(s<0) return; sel=s;
        char d[1100]; snprintf(d,sizeof d,"%s",pth[s]);
        snprintf(t_items[0],260,"Open in Explorer"); t_info[0][0]=0;
        snprintf(t_items[1],260,"Info"); snprintf(t_info[1],T_INFO,"cudalm data info (corpus / tokenizer folders)");
        snprintf(t_items[2],260,"Rename..."); t_info[2][0]=0;
        snprintf(t_items[3],260,"Delete to the Recycle Bin..."); snprintf(t_info[3],T_INFO,"can be restored from the Recycle Bin");
        int act=t_pick(d,NULL,4,0);
        if(act==0) ShellExecuteA(NULL,"explore",d,NULL,NULL,SW_SHOWNORMAL);
        else if(act==1){ char exe[MAX_PATH],cl[2600],lp[700]; GetModuleFileNameA(0,exe,sizeof exe); cl[0]=0; t_quote(cl,sizeof cl,exe);
            t_quote(cl,sizeof cl,"data"); t_quote(cl,sizeof cl,"info"); t_quote(cl,sizeof cl,d); t_spawn(cl,NULL,"data-info",lp,sizeof lp); t_note("done"); }
        else if(act==2){ char nm[260]; snprintf(nm,sizeof nm,"%s",d_base(d));
            if(t_edit("new name",nm,sizeof nm)&&nm[0]&&!strpbrk(nm,"\\/:*?\"<>|")){ char to[1400]; snprintf(to,sizeof to,"%.*s%s",(int)(d_base(d)-d),d,nm);
                if(!MoveFileA(d,to)) t_note("rename failed (error %lu) - is it in use?",GetLastError()); } }
        else if(act==3){ char y[16]=""; t_head("delete"); printf("  send %s to the Recycle Bin?\n  type yes and Enter to confirm, Esc to cancel\n",d);
            if(t_edit("confirm",y,sizeof y)&&!strcmp(y,"yes")){ char fr[1102]; memset(fr,0,sizeof fr); snprintf(fr,sizeof fr-1,"%s",d);
                SHFILEOPSTRUCTA op; memset(&op,0,sizeof op); op.wFunc=FO_DELETE; op.pFrom=fr; op.fFlags=FOF_ALLOWUNDO|FOF_NOCONFIRMATION|FOF_WANTNUKEWARNING;
                int rc=SHFileOperationA(&op); if(rc||op.fAnyOperationsAborted) t_note("not deleted (code %d)",rc); } } } }
/* ---- Hugging Face browser: search -> config / split -> columns + first rows -> Download text, filled in ---- */
static HfDs t_hfd[100];
static void t_hf_pv(int sel){ const HfDs*d=&t_hfd[sel]; char a[32], s[400]; int w=t_cols()-4; if(w<40) w=40; if(w>300) w=300;
    printf("\x1b[36m%s\x1b[0m   %s downloads, %d likes%s\n",d->id,t_hn((double)d->dl,a),d->likes,d->gated?"   \x1b[33mgated: needs HF_TOKEN and the terms accepted on the website\x1b[0m":"");
    printf("  \x1b[2mlicence %s   size %s   task %s   language %s   format %s\x1b[0m\n",d->lic[0]?d->lic:"?",d->size[0]?d->size:"?",
           d->task[0]?d->task:"-",d->lang[0]?d->lang:"-",d->fmt[0]?d->fmt:"-");
    int m=0; for(const char*q=d->desc;*q;q++){ char c=*q=='\n'||*q=='\r'||*q=='\t'?' ':*q; if(c!=' '||(m&&s[m-1]!=' ')) s[m++]=c; } s[m]=0;
    for(int l=0,o=0;l<3&&s[o];l++){ int e=o,c=0; while(s[e]&&c<w){ if(((unsigned char)s[e]&0xC0)!=0x80) c++; e++; } while(((unsigned char)s[e]&0xC0)==0x80) e++;
        printf("  %.*s\n",e-o,s+o); o=e; } }
static void t_hf_open(const char*repo,const char*cfg,const char*split,long long bytes){
    DBuf b={0}; char tpl[T_MAXL]="", a[32]; int st=0; const char*r0=NULL;
    if(cfg[0]){ t_head(repo); printf("  loading the first rows...\n"); fflush(stdout); st=hf_rows(repo,cfg,split,&b); if(st==200) r0=d_row(b.p,0); }
    char ttl[400]; snprintf(ttl,sizeof ttl,"template for %s  %s / %s",repo,cfg,split);
    for(;;){
        t_head(repo); printf("  config \x1b[1m%s\x1b[0m   split \x1b[1m%s\x1b[0m",cfg[0]?cfg:"auto",split[0]?split:"auto");
        if(bytes>0) printf("   \x1b[2m%s of parquet, roughly %s tokens\x1b[0m",t_hb((double)bytes,a),t_hn(bytes*0.45,a+16)); printf("\n\n");
        if(!r0){ char e[300]=""; if(b.p) d_jstr(jget(b.p,"error"),e,sizeof e);
            printf("  \x1b[33mno preview%s%s: the download uses the repo's own .parquet / .jsonl / .json files, auto-detected\x1b[0m\n",e[0]?" - ":"",e); }
        else { char nm[64][64],kd[64]; int nc=d_cols(r0,nm,kd,64); const char*how=tpl_auto(r0);
            hf_show_rows(b.p,1,t_cols()-24);
            if(tpl[0]) printf("\n  \x1b[1mtemplate\x1b[0m  %s\n",tpl);
            else printf("\n  \x1b[1mtemplate\x1b[0m  auto: %s\n",how?how:"\x1b[31mnothing recognised - press T to pick the columns\x1b[0m");
            printf("\n  \x1b[2m--- what the model sees (row 0) ---\x1b[0m\n"); t_tplshow(r0,tpl,t_rows()-nc-14); }
        printf("\x1b[%d;1H\x1b[2mEnter download this (fills in Download text)   %sEsc back\x1b[0m",t_rows(),r0?"T template: which columns, previewed   ":""); fflush(stdout);
        int k=t_key();
        if(k==27){ free(b.p); return; }
        if((k=='t'||k=='T')&&r0){ t_colmap(ttl,b.p,tpl,sizeof tpl); continue; }
        if(k!=13) continue;
        if(r0){ static DWs w; const char*t; size_t n; if(!tpl_render(r0,tpl,&w,&t,&n)){ t_note("row 0 gives no document with this template: press T and pick the columns"); continue; } }
        char spec[T_MAXL], nm[200], tk[32]; int n=snprintf(spec,sizeof spec,"hf:%s",repo); if(cfg[0]) snprintf(spec+n,sizeof spec-n,":%s:%s",cfg,split);
        hf_name(repo,cfg,split,nm,sizeof nm); double est=bytes>0?bytes*0.45*1.2:1e9; if(est>1e9) est=1e9; if(est<1e6) est=1e6;
        snprintf(tk,sizeof tk,"%.0fM",est/1e6);
        t_set("Download text","profile",spec); t_set("Download text","out folder",nm); t_set("Download text","tokens",tk); t_set("Download text","template",tpl);
        free(b.p); t_form(&T_CMDS[t_ci("Download text")]); return; } }
static void t_hf_ds(const char*repo){
    static HfSplit sp[256]; int part; char err[300], a[32], b[32];
    t_head(repo); printf("  loading configs and splits...\n"); fflush(stdout);
    int n=hf_splits(repo,sp,256,&part,err,sizeof err);
    if(n<=0){ t_note("no split list (%s) - the download will try the repo's own files",n<0?err:"empty"); t_hf_open(repo,"","",0); return; }
    if(n==1){ t_hf_open(repo,sp[0].cfg,sp[0].split,sp[0].bytes); return; }
    int sel=hf_default(sp,n); char ttl[400]; snprintf(ttl,sizeof ttl,"%s: pick a config / split%s",repo,part?"   (partial parquet conversion: about the first 5 GB of each)":"");
    for(;;){ for(int i=0;i<n;i++){ snprintf(t_items[i],260,"%s / %s",sp[i].cfg,sp[i].split); snprintf(t_info[i],T_INFO,"%10s rows %10s",t_hn((double)sp[i].rows,a),t_hb((double)sp[i].bytes,b)); }
        int s=t_pick(ttl,"Up/Down  Enter preview  Esc back",n,sel); if(s<0) return; sel=s; t_hf_open(repo,sp[s].cfg,sp[s].split,sp[s].bytes); } }
static void t_hf(void){
    static char q[200]; char a[32];
    for(;;){
        t_head("Browse Hugging Face datasets");
        printf("  Search dataset names, e.g. tinystories, fineweb-edu, ultrachat, the-stack. Results come most downloaded first.\n"
               "  Pick a dataset and split, check what fetch will read from its first rows, then download it as .jsonl text.\n"
               "  \x1b[2mGated datasets need HF_TOKEN (environment or profile) and their terms accepted on huggingface.co.\n"
               "  Datasets that only ship a loading script cannot be read.\x1b[0m\n");
        if(!t_edit("search",q,sizeof q)||!q[0]) return;
        printf("\x1b[%d;1H\x1b[2K  searching...",t_rows()); fflush(stdout);
        int n=hf_search(q,t_hfd,100), sel=0;
        if(n<0){ t_note(n==-1?"search failed: no connection":"search failed: HTTP %d",-n); continue; }
        if(!n){ t_note("no datasets match '%s'",q); continue; }
        char ttl[300]; snprintf(ttl,sizeof ttl,"Hugging Face: '%s'   %d result%s, most downloaded first",q,n,n>1?"s":"");
        for(;;){ for(int i=0;i<n;i++){ const HfDs*d=&t_hfd[i]; snprintf(t_items[i],260,"%s",d->id);
                snprintf(t_info[i],T_INFO,"%7s dl %6d likes  %-11s %-14.14s%s",t_hn((double)d->dl,a),d->likes,d->size,d->lic,d->gated?"  \x1b[33mgated\x1b[0m":""); }
            t_pv=t_hf_pv; int s=t_pick(ttl,"Up/Down  Enter open  Esc new search",n,sel); t_pv=0;
            if(s<0) break; sel=s; t_hf_ds(t_hfd[s].id); } } }
static void t_group_menu(int grp){
    int sel=0;
    for(;;){ int n=0, ids[T_NCMD+8];
        if(grp==1){ ids[n]=-4; snprintf(t_items[n],260,"Browse Hugging Face"); snprintf(t_info[n++],T_INFO,"search any dataset, preview its rows, download it as text"); }
        for(int i=0;i<T_NCMD;i++) if(T_CMDS[i].grp==grp){ ids[n]=i; snprintf(t_items[n],260,"%s",T_CMDS[i].title); snprintf(t_info[n++],T_INFO,"%s",T_CMDS[i].help); }
        if(grp==1){
            ids[n]=-1; snprintf(t_items[n],260,"-- advanced --"); t_info[n++][0]=0;
            for(int i=0;i<T_NCMD;i++) if(T_CMDS[i].grp==3){ ids[n]=i; snprintf(t_items[n],260,"%s",T_CMDS[i].title); snprintf(t_info[n++],T_INFO,"%s",T_CMDS[i].help); }
            ids[n]=-1; snprintf(t_items[n],260,"-- folders --"); t_info[n++][0]=0;
            ids[n]=-2; snprintf(t_items[n],260,"Manage data folders"); snprintf(t_info[n++],T_INFO,"sizes, open, rename, delete to the Recycle Bin");
            char r[2048]; data_roots(r,sizeof r); ids[n]=-3; snprintf(t_items[n],260,"Data roots"); snprintf(t_info[n++],T_INFO,"%.200s",r); }
        int s=grp==1?t_pick("Data tools   order: Download / Convert text -> (Train tokenizer) -> Build corpus -> (Mix) -> Corpus info -> Train",
                     "Up/Down  Enter open  Esc back   bare names = folders under the data roots",n,sel)
                    :t_pick("Diagnostics   correctness checks for kernels, decoder, tokenizer and QLM weights",NULL,n,sel);
        if(s<0) return; sel=s;
        if(ids[s]>=0) t_form(&T_CMDS[ids[s]]); else if(ids[s]==-2) t_manage(); else if(ids[s]==-3) t_roots(); else if(ids[s]==-4) t_hf(); } }
/* tuicheck [model]: every form's command line from its defaults (model fields = model), no console needed */
/* tuicheck [model] ["Form:label=value" ...]: every form's command line, plus the live panel of the data forms */
static int tui_check(int argc,char**argv){
    if(argc>3&&!strcmp(argv[2],"hf")){ t_hfd[0].id[0]=0; int n=hf_search(argv[3],t_hfd,5); for(int i=0;i<n;i++){ t_hf_pv(i); printf("\n"); } return n<0; }
    const char*model=argc>2&&!strchr(argv[2],':')?argv[2]:NULL;
    for(int i=model?3:2;i<argc;i++){ char t[600]; snprintf(t,sizeof t,"%s",argv[i]); char*l=strchr(t,':'),*v=l?strchr(l,'='):NULL;
        if(!v){ fprintf(stderr,"tuicheck: expected Form:label=value, got %s\n",argv[i]); return 1; } *l++=0; *v++=0; t_set(t,l,v); }
    for(int ci=0;ci<T_NCMD;ci++){ const TCmd*c=&T_CMDS[ci]; char cl[4096],err[200]; t_form_init(ci); char (*V)[T_MAXL]=t_vals[ci];
        for(int i=0;i<T_NF&&c->f[i].label;i++) if(c->f[i].kind==F_MODEL&&model) snprintf(V[i],T_MAXL,"%s",model);
        int ok=t_cmdline(c,V,cl,sizeof cl,err,sizeof err);
        printf("%-16s %s\n",c->title,ok?cl:err);
        if(c->grp==1||c->grp==3){ int np=t_panel(c,V,1); if(np) printf("  (%d problem%s)\n",np,np>1?"s":""); }
        if(t_is(c,"New model")){ t_mode(c,V); t_new_stats(c,V); } }
    return 0; }
static int tui_main(void){
    HANDLE ho=GetStdHandle(STD_OUTPUT_HANDLE); DWORD m=0;
    if(!GetConsoleMode(ho,&m) || !SetConsoleMode(ho,m|0x0004)) return -1;  /* ENABLE_VIRTUAL_TERMINAL_PROCESSING */
    SetConsoleOutputCP(CP_UTF8); SetConsoleCtrlHandler(t_ctrl,TRUE); SetCurrentDirectoryA(exe_dir());
    int sel=0;
    for(;;){
        int n=0, ids[T_NCMD];
        for(int i=0;i<T_NCMD;i++) if(!T_CMDS[i].grp){ ids[n]=i; snprintf(t_items[n],260,"%s",T_CMDS[i].title); snprintf(t_info[n++],T_INFO,"%s",T_CMDS[i].help); }
        int nc=n;
        snprintf(t_items[n],260,"Data tools..."); snprintf(t_info[n++],T_INFO,"download text, train a tokenizer, build / mix / inspect corpora, loss weights");
        snprintf(t_items[n],260,"Diagnostics..."); snprintf(t_info[n++],T_INFO,"selftest, decode check, tokenizer check, QLM check");
        snprintf(t_items[n],260,"Model config"); snprintf(t_info[n++],T_INFO,"edit a model's saved settings (models\\<name>.bin.env)");
        snprintf(t_items[n],260,"Environment / profile"); snprintf(t_info[n++],T_INFO,"all CUDALM_* QLM_* NGRAM_* settings for the commands");
        snprintf(t_items[n],260,"Quit"); t_info[n++][0]=0;
        int s=t_pick("main menu","Up/Down  Enter open  Esc quit   (cudalm help = CLI usage)",n,sel);
        if(s<0 || s==n-1){ printf("\x1b[2J\x1b[H"); return 0; }
        sel=s;
        if(s<nc) t_form(&T_CMDS[ids[s]]);
        else if(s==nc) t_group_menu(1);
        else if(s==nc+1) t_group_menu(2);
        else if(s==nc+2) t_model_cfg();
        else { t_st.file=0; t_reg(&t_st,"Environment (inherited by every command; saved into a model's config by new/train)"); }
    }
}
#endif
