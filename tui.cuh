/* tui.cuh -- interactive menu when cudalm is started with no arguments (Windows console).
 * Each command is a form; Run launches this same exe as a child with the built argv, so every
 * command gets a fresh process (exit() paths and CUDA state stay as in normal CLI use) and inherits
 * the environment edited on the Environment screen (every setting in g_envreg, or a loaded *env.bat).
 * New-model fields of kind F_ENV/F_MEM are pushed into the child's environment; cmd_new saves them
 * (with the rest of the environment) as the model config models\<name>.bin.env.
 * The child runs in its own process group. Ctrl+C during Train writes models\<name>.bin.stop (clean save+exit at
 * the next 20-iter check -- never a hard kill of a CUDA process); for other commands it is forwarded as Ctrl+Break,
 * i.e. what plain CLI Ctrl+C does. Working dir is set to the exe dir. */
#ifdef _WIN32
#include <conio.h>

enum { F_TEXT, F_MODEL, F_DATA, F_FILE, F_CHOICE, F_DIR, F_PATH, F_TOKD, F_OPT, F_ENV, F_MEM };
/* arg: choices "a|b" (F_CHOICE), the --flag (F_OPT) or the env var (F_ENV).
 * vis: 0 always, 1 ngram only, 2 engram only, 3 any memory. ext: "a|b" choices, a file glob, or F_PATH suffixes ".a;.b".
 * F_DATA = corpus picker, F_TOKD = tokenizer/corpus picker, F_DIR / F_PATH = folder / file-or-folder browser. */
typedef struct { const char*label,*def; int kind; const char*arg; int vis; const char*ext; const char*help; } TFld;   /* help: '\n' = next line */
#define T_NF 26
typedef struct { const char*cmd,*title,*help; TFld f[T_NF]; const char*run; int grp; } TCmd;   /* run: help on [ Run ]; grp 1 = Data tools */
#define T_TEXTIN ".jsonl;.json;.txt"
#define T_PRESETS "nano|micro|m1|m3|m6|m8|m16|m32|m70|m50|m64e|m220e|a8m|a8m9|s8x|s8w"
static const TCmd T_CMDS[]={
 {"train","Train","continue training a model (create it first with New model)",
  {{"model","",F_MODEL},{"data dir","",F_DATA},{"epochs","1",F_TEXT}}},
 {"run","Generate","sample one continuation of a prompt",
  {{"model","",F_MODEL},{"prompt","Once upon a time",F_TEXT},{"tokens","120",F_TEXT},{"temp","0.8",F_TEXT},{"top-k","40",F_TEXT}}},
 {"chatbench","Chatbench","fixed chat prompts, N samples each (feed the output to judgerun.bat)",
  {{"model","",F_MODEL},{"bench file","chatbench.txt",F_FILE},{"gen tokens","128",F_TEXT},{"temp","0.7",F_TEXT},
   {"top-p","0.9",F_TEXT},{"rep pen","1.15",F_TEXT},{"samples","4",F_TEXT},{"ctx T","256",F_TEXT}}},
 {"probe","Probe","one generation per line of a prompts file",
  {{"model","",F_MODEL},{"prompts file","",F_FILE},{"tokens","80",F_TEXT},{"temp","0",F_TEXT},{"top-k","1",F_TEXT},
   {"top-p","0",F_TEXT},{"rep pen","1",F_TEXT}}},
 {"eval","Eval","forward-only val loss + MoE utilisation (safe during training)",
  {{"model","",F_MODEL},{"data dir","",F_DATA},{"batches","20",F_TEXT},{"batch size","1",F_TEXT}}},
 {"new","New model","create models\\<name>.bin + config .bin.env (other settings: Environment screen, also saved)",
  {{"name","",F_TEXT,0,0,0,"File name: creates models\\<name>.bin plus its config models\\<name>.bin.env.\n"
     "Refused if the model already exists. Train it next with Train (model = this name)."},
   {"preset","s8w",F_CHOICE,T_PRESETS,0,0,"Model shape: width D, layers L, attention heads H (KV = shared key/value heads),\n"
     "FFN width F (per expert), default context T and vocab V. Left/Right cycles; the shape is shown below."},
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
   {"ctx","",F_OPT,"--ctx",0,0,"llama target: context to advertise (default = the training window)."}}},
 {"qlmcheck","QLM check","distinct values per weight group",{{"model","",F_MODEL},{"group","128",F_TEXT}}},
 {"selftest","Selftest","GPU kernels vs CPU references",{{0}}},
 /* ---- Data tools (grp 1): text -> tokenizer -> corpus; all CPU ---- */
 {"data fetch","Download text","Hugging Face blend -> .jsonl text (resumable; runs prep\\fetch.py: python + pyarrow)",
  {{"profile","v3pre",F_CHOICE,"v3pre|v3sft|v3chat|web|code|math|cosmopedia|chat|tools",0,0,
     "v3pre = pretraining blend (web 34, code 20, synthetic 16, math 14, chat 10, tools 6 %); v3sft / v3chat = instruction blends;\n"
     "web, code, math, cosmopedia, chat, tools = one source each. Or type the https URL of a single .parquet/.json file."},
   {"out folder","text_v3pre",F_TEXT,0,0,0,"Folder for the text (<category>__<source>.jsonl). A bare name goes under the data root with the most free space.\n"
     "Run again with the same folder to resume an interrupted download."},
   {"tokens","1000000000",F_TEXT,0,0,0,"Roughly how many tokens of text to fetch (1e9 ~ 4.6 GB of text), split by the profile's weights.\n"
     "Get 10-20% more than you plan to train on."}},
  "Run: download and parse each source in turn (raw shards are deleted once parsed).\nNext: Build corpus with this folder as text (tokenizer v3tok), or Train tokenizer first for a new vocab.",1},
 {"data fetch","Convert files","parquet / json / jsonl from elsewhere -> plain .jsonl text (python + pyarrow)",
  {{"source","",F_PATH,0,0,".parquet;.json;.jsonl","A .parquet / .json / .jsonl file, or a folder (every such file inside, recursively).\n"
     "Rows with messages / conversations become chats; otherwise the text / content / code column is used."},
   {"out folder","converted",F_TEXT,0,0,0,"Folder for the converted .jsonl (one per input file). A bare name goes under a data root."},
   {"category","",F_TEXT,0,0,0,"Category stored with every doc and shown in the corpus stats. Empty = the file name."}},
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
 {"data encode","Build corpus","tokenize text into train.bin / val.bin for Train and Eval (new corpus, or append)",
  {{"mode","new",F_CHOICE,"new|append",0,0,"new = create a corpus folder (refused if it already holds one).\n"
     "append = add docs to an existing corpus's train.bin (val.bin unchanged); needs the identical tokenizer."},
   {"corpus","",F_DATA,0,0,0,"new: a name (bare = under the data root with most free space) or a full path.\n"
     "append: pick the existing corpus."},
   {"text","",F_PATH,0,0,T_TEXTIN,"Input .jsonl / .json / .txt files or folders, several separated by ';'. JSON needs \"text\" or a chat\n"
     "(\"messages\" / \"conversations\"), rendered with the chatml markers. Parquet: Convert files first."},
   {"tokenizer","v3tok",F_TOKD,0,0,0,"Folder with tokenizer.bin: one from Train tokenizer, or any corpus.\n"
     "Must be the tokenizer of the model you will train (v3tok = all v3 models)."},
   {"val %","1",F_TEXT,0,0,0,"Share of docs held out in val.bin (every Nth doc), used by Eval and the training val loss. 0 = none."},
   {"dedup","1",F_CHOICE,"1|0",0,0,"1 = skip docs whose exact text already appeared in this run."},
   {"follow","0",F_CHOICE,"0|1",0,0,"1 = keep reading a .jsonl another program is still writing, until <file>.done appears\n"
     "(e.g. teacher generation). 0 = stop at the end of the input."}},
  "Run: tokenize on all CPU cores (token-exact with the HF tokenizer).\nNext: Corpus info to check it, then Train with this corpus as data dir.",1},
 {"data mix","Mix corpora","weighted, shuffled blend of corpora that share one tokenizer",
  {{"out name","mix",F_TEXT,0,0,0,"New corpus folder (a bare name goes under the data root with most free space)."},
   {"train tokens","0",F_TEXT,0,0,0,"Size of the blend's train.bin. 0 = as large as the weights allow without reusing any corpus\n"
     "more than max repeat times."},
   {"max repeat","1",F_TEXT,0,0,0,"How often one corpus's docs may be reused (1 = each doc at most once, 2 = up to twice).\n"
     "A corpus that runs out contributes less than its weight (a note is printed)."},
   {"val tokens","1000000",F_TEXT,0,0,0,"Tokens in val.bin, taken from each corpus's own val.bin by weight."},
   {"corpus 1","",F_DATA,0,0,0,"A corpus to blend in. Fill corpus/weight pairs in order; unused pairs stay empty."},
   {" weight 1","",F_TEXT,0,0,0,"Relative weight (any scale: 3 and 1 = 75% / 25% of the train tokens)."},
   {"corpus 2","",F_DATA,0,0,0,"A corpus to blend in. Fill corpus/weight pairs in order; unused pairs stay empty."},
   {" weight 2","",F_TEXT,0,0,0,"Relative weight (any scale: 3 and 1 = 75% / 25% of the train tokens)."},
   {"corpus 3","",F_DATA,0,0,0,"A corpus to blend in. Fill corpus/weight pairs in order; unused pairs stay empty."},
   {" weight 3","",F_TEXT,0,0,0,"Relative weight (any scale: 3 and 1 = 75% / 25% of the train tokens)."},
   {"corpus 4","",F_DATA,0,0,0,"A corpus to blend in. Fill corpus/weight pairs in order; unused pairs stay empty."},
   {" weight 4","",F_TEXT,0,0,0,"Relative weight (any scale: 3 and 1 = 75% / 25% of the train tokens)."},
   {"corpus 5","",F_DATA,0,0,0,"A corpus to blend in. Fill corpus/weight pairs in order; unused pairs stay empty."},
   {" weight 5","",F_TEXT,0,0,0,"Relative weight (any scale: 3 and 1 = 75% / 25% of the train tokens)."},
   {"corpus 6","",F_DATA,0,0,0,"A corpus to blend in. Fill corpus/weight pairs in order; unused pairs stay empty."},
   {" weight 6","",F_TEXT,0,0,0,"Relative weight (any scale: 3 and 1 = 75% / 25% of the train tokens)."}},
  "Run: pick whole docs per corpus, shuffle them together, write train.bin + val.bin + meta.json.\nAll corpora must use the identical tokenizer (checked).",1},
 {"data info","Corpus info","tokenizer, token / doc counts, doc lengths, categories and a decoded sample",
  {{"corpus","",F_TOKD,0,0,0,"A corpus (or tokenizer) folder. Checks every token id against the vocab - catches data built\n"
     "with another tokenizer - and prints one random doc decoded."}},
  "Run: read-only; scanning a multi-GB train.bin takes a few seconds.",1},
 {"data canon","Canon map","canon.bin for the New model 'canon map' field (NGRAM_CANON)",
  {{"tokenizer","v3tok",F_TOKD,0,0,0,"Tokenizer or corpus folder. Tokens differing only by case, leading space or Unicode form\n"
     "map to one id, so they share engram table rows."},
   {"out file","",F_TEXT,0,0,0,"Output .bin (empty = canon.bin inside the tokenizer folder)."}},
  "Run: write one uint16 per token. Train tokenizer already writes canon.bin.",1},
};
#define T_NCMD (int)(sizeof(T_CMDS)/sizeof(T_CMDS[0]))
enum { K_UP=1072, K_DOWN=1080, K_LEFT=1075, K_RIGHT=1077, K_PGUP=1073, K_PGDN=1081, K_HOME=1071, K_END=1079, K_F5=1063, K_DEL=1083 };
#define T_MAXL 512
#define T_INFO 220
static char t_items[T_MAXL][260], t_info[T_MAXL][T_INFO];
static char t_profile[260]="(none)";

static int t_key(void){ int c=_getch(); if(c==0||c==0xE0) return 1000+_getch(); return c; }
static int t_rows(void){ CONSOLE_SCREEN_BUFFER_INFO b; if(GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE),&b))
    return b.srWindow.Bottom-b.srWindow.Top+1; return 25; }
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
/* scrolling list over t_items/t_info; returns index or -1 */
static int t_pick(const char*title, const char*hint, int n, int sel){
    if(n<=0) return -1; if(sel<0||sel>=n) sel=0; int top=0;
    for(;;){
        int h=t_rows()-6; if(h<3) h=3;
        if(sel<top) top=sel; if(sel>=top+h) top=sel-h+1;
        t_head(title);
        for(int i=top;i<n && i<top+h;i++)
            printf("%s %-44s \x1b[2m%s\x1b[0m\x1b[0m\n", i==sel?"\x1b[7m>":" ", t_items[i], t_info[i]);
        printf("\x1b[%d;1H\x1b[2m%s\x1b[0m",t_rows()-1,hint?hint:"Up/Down PgUp/PgDn  Enter select  Esc back"); fflush(stdout);
        int k=t_key();
        if(k==K_UP && sel>0) sel--; else if(k==K_DOWN && sel<n-1) sel++;
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
            if(hc&&t>0) snprintf(t_info[n],T_INFO,"%9.1fM tok  V=%d",t/1e6,V);
            else if(hc) snprintf(t_info[n],T_INFO,"%9.1fM tok  V=%d",d_fsize(tb)/2e6,V);
            else snprintf(t_info[n],T_INFO,"tokenizer    V=%d",V);
            n++;
        }while(FindNextFileA(h,&fd)); FindClose(h);
    }
    return n;
}
static int t_list_glob(const char*glob){
    WIN32_FIND_DATAA fd; HANDLE h=FindFirstFileA(glob,&fd); int n=0;
    if(h!=INVALID_HANDLE_VALUE){ do{ if(n<T_MAXL && !(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)){
        snprintf(t_items[n],260,"%s",fd.cFileName); snprintf(t_info[n],T_INFO,"%8.1f KB",fd.nFileSizeLow/1024.0); n++; } }
        while(FindNextFileA(h,&fd)); FindClose(h); }
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
static void t_choice_items(const char*ch, int*n){
    *n=0; const char*p=ch;
    while(*p && *n<T_MAXL){ const char*q=strchr(p,'|'); int l=q?(int)(q-p):(int)strlen(p);
        snprintf(t_items[*n],260,"%.*s",l,p); t_info[*n][0]=0; (*n)++; if(!q) break; p=q+1; }
}
static const char* t_choices(const TFld*f){
    if(f->kind==F_CHOICE) return f->arg;
    return (f->kind==F_ENV||f->kind==F_MEM)&&f->ext&&strchr(f->ext,'|') ? f->ext : NULL; }

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
    for(int i=0;i<=last;i++){ if(c->f[i].kind>=F_OPT) continue;
        if(!vals[i][0]){ snprintf(err,ecap,"'%s' is empty but a later field is set",c->f[i].label); return 0; }
        t_quote(o,cap,vals[i]); }
    for(int i=0;i<T_NF&&c->f[i].label;i++) if(c->f[i].kind==F_OPT && vals[i][0]){ t_quote(o,cap,c->f[i].arg); t_quote(o,cap,vals[i]); }
    return 1;
}
static void t_spawn(const char*cl, const char*stop){
    printf("\x1b[2J\x1b[H\x1b[1;36m> %s\x1b[0m\n\n",cl); fflush(stdout);
    STARTUPINFOA si; PROCESS_INFORMATION pi; memset(&si,0,sizeof si); si.cb=sizeof si;
    char buf[8192]; snprintf(buf,sizeof buf,"%s",cl); DWORD t0=GetTickCount(); DWORD code=0;
    snprintf(t_stop,sizeof t_stop,"%s",stop?stop:"");
    if(!CreateProcessA(NULL,buf,NULL,NULL,FALSE,CREATE_NEW_PROCESS_GROUP,NULL,NULL,&si,&pi)) printf("CreateProcess failed (%lu)\n",GetLastError());
    else { t_child=pi.dwProcessId; WaitForSingleObject(pi.hProcess,INFINITE); t_child=0;
        GetExitCodeProcess(pi.hProcess,&code); CloseHandle(pi.hProcess); CloseHandle(pi.hThread); }
    printf("\n\x1b[1;%dm[exit %lu, %.1f s]\x1b[0m  press any key\n",code?31:32,code,(GetTickCount()-t0)/1000.0);
    while(_kbhit()) _getch(); t_key();
}

/* ---- form env fields: pushed into this process's env (inherited by the child), then restored ---- */
static char t_saved[40][2][T_MAXL]; static int t_nsaved;
static void t_env_push(const char*k,const char*v){
    if(t_nsaved<40){ const char*o=getenv(k); snprintf(t_saved[t_nsaved][0],T_MAXL,"%s",k); snprintf(t_saved[t_nsaved][1],T_MAXL,"%s",o?o:""); t_nsaved++; }
    lm_setenv(k,v); }
static void t_env_pop(void){ while(t_nsaved>0){ t_nsaved--; lm_setenv(t_saved[t_nsaved][0],t_saved[t_nsaved][1]); } }
static int t_mode(const TCmd*c, char vals[][T_MAXL]){
    for(int i=0;i<T_NF&&c->f[i].label;i++) if(c->f[i].kind==F_MEM) return !strcmp(vals[i],"engram")?2:!strcmp(vals[i],"ngram")?1:0;
    return -1; }
static int t_vis(const TFld*f,int mode){ return !f->vis || (f->vis==3 ? mode>0 : mode==f->vis); }
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
    Cfg cf; if(!preset(vals[1],&cf)){ printf("  \x1b[2m(unknown preset)\x1b[0m\n"); return; }
    cf.n_mtp=0; cf.n_exp=atoi(vals[2]); if(cf.n_exp<=1) cf.n_exp=0;
    t_form_env(c,vals);
    cf.pool=env_i("CUDALM_UNIPOOL",0)&&cf.n_exp>1; if(!layout_apply(&cf)) return; cfg_derive(&cf);
    int kl[64], k=topk_parse(getenv("CUDALM_TOPK"),getenv("CUDALM_TOPK_L"),kl);
    MemCfg m; mem_from_env(&m,cf.L);
    printf("\x1b[1mparams\x1b[0m \x1b[2m(V=%d from the preset; the first train resizes it to the data's vocab)\x1b[0m\n",cf.V);
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

static void t_form(const TCmd*c){
    static char vals[T_NCMD][T_NF][T_MAXL]; static int init[T_NCMD];
    int ci=(int)(c-T_CMDS), nf=0; while(nf<T_NF && c->f[nf].label) nf++;
    if(!init[ci]){ int envmem=getenv("CUDALM_NGRAM")||getenv("CUDALM_ENGRAM");
        for(int i=0;i<nf;i++){ const TFld*f=&c->f[i]; const char*e=f->kind==F_ENV?getenv(f->arg):NULL;
            if(e&&*e&&!strncmp(f->arg,"NGRAM_",6)) envmem=1;
            snprintf(vals[ci][i],T_MAXL,"%s",e&&*e?e:f->def); }
        for(int i=0;i<nf;i++) if(c->f[i].kind==F_MEM && envmem){   /* a loaded profile decides the mode */
            MemCfg m; mem_from_env(&m,64); snprintf(vals[ci][i],T_MAXL,"%s",mem_name(m.mode)); }
        init[ci]=1; }
    int sel=0; char msg[300]="";
    for(;;){
        int mode=t_mode(c,vals[ci]), vis[T_NF], nv=0;
        for(int i=0;i<nf;i++) if(t_vis(&c->f[i],mode)) vis[nv++]=i;
        if(sel>nv) sel=nv;
        char cl[4096], err[200]; int ok=t_cmdline(c,vals[ci],cl,sizeof cl,err,sizeof err);
        t_head(c->title); printf("  \x1b[2m%s\x1b[0m\n\n",c->help);
        for(int j=0;j<nv;j++){ const TFld*f=&c->f[vis[j]]; const char*v=vals[ci][vis[j]];
            printf("%s %-14s\x1b[0m %s",j==sel?"\x1b[7m>":" ",f->label,v[0]?v:"\x1b[2m-\x1b[0m");
            if(f->kind==F_ENV) printf("   \x1b[2m%s%s\x1b[0m",f->arg,t_choices(f)?"  <- ->":"");
            else if(f->kind==F_MEM) printf("   \x1b[2m<- ->\x1b[0m");
            printf("\n"); }
        if(mode>=0 || c->run) t_help(sel<nv?&c->f[vis[sel]]:NULL,sel<nv?vals[ci][vis[sel]]:"",c->run);
        printf("\n%s [ Run ]\x1b[0m\n\n\x1b[2m%s\x1b[0m\n",sel==nv?"\x1b[7m>":" ",ok?cl:err);
        if(mode>=0){ printf("\n"); t_new_stats(c,vals[ci]); }
        if(msg[0]) printf("\x1b[33m%s\x1b[0m\n",msg); msg[0]=0;
        printf("\x1b[%d;1H\x1b[2mUp/Down move  Enter edit/pick  Left/Right cycle choice  Del clear  F5 run  Esc back\x1b[0m",t_rows()); fflush(stdout);
        int k=t_key();
        if(k==27) return;
        if(k==K_UP) sel=sel>0?sel-1:nv; else if(k==K_DOWN) sel=sel<nv?sel+1:0;
        else if(k==K_F5 || (k==13 && sel==nv)){
            if(!ok){ snprintf(msg,sizeof msg,"%s",err); continue; }
            char sp[1100]=""; if(!strcmp(c->cmd,"train")){ model_path(vals[ci][0],sp,sizeof sp-5); strcat(sp,".stop"); }
            t_form_env(c,vals[ci]); t_spawn(cl,sp[0]?sp:NULL); t_env_pop(); }
        else if(sel<nv){
            const TFld*f=&c->f[vis[sel]]; char*v=vals[ci][vis[sel]]; const char*ch=t_choices(f); int n;
            if(k==K_DEL) v[0]=0;
            else if((k==K_LEFT||k==K_RIGHT) && ch){ t_choice_items(ch,&n); int cur=0;
                for(int i=0;i<n;i++) if(!strcmp(t_items[i],v)) cur=i;
                cur=(cur+(k==K_RIGHT?1:n-1))%n; snprintf(v,T_MAXL,"%s",t_items[cur]); }
            else if(k==13){
                if(f->kind==F_MODEL){ n=t_list_models(); t_choose(f->label,n,v,T_MAXL,0); }
                else if(f->kind==F_DATA||f->kind==F_TOKD){ n=t_list_dirs(f->kind==F_TOKD); t_choose(f->label,n,v,T_MAXL,1); }
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

static void t_data_menu(void){
    int sel=0;
    for(;;){ int n=0, ids[T_NCMD];
        for(int i=0;i<T_NCMD;i++) if(T_CMDS[i].grp==1){ ids[n]=i; snprintf(t_items[n],260,"%s",T_CMDS[i].title); snprintf(t_info[n++],T_INFO,"%s",T_CMDS[i].help); }
        char r[2048]; data_roots(r,sizeof r);
        snprintf(t_items[n],260,"(data roots)"); snprintf(t_info[n++],T_INFO,"%.200s",r);
        int s=t_pick("Data tools   order: Download / Convert text -> (Train tokenizer) -> Build corpus -> (Mix) -> Corpus info -> Train",
                     "Up/Down  Enter open  Esc back   bare names = folders under the data roots (CUDALM_DATA_ROOTS)",n,sel);
        if(s<0) return; sel=s; if(s<n-1) t_form(&T_CMDS[ids[s]]); } }
static int tui_main(void){
    HANDLE ho=GetStdHandle(STD_OUTPUT_HANDLE); DWORD m=0;
    if(!GetConsoleMode(ho,&m) || !SetConsoleMode(ho,m|0x0004)) return -1;  /* ENABLE_VIRTUAL_TERMINAL_PROCESSING */
    SetConsoleOutputCP(CP_UTF8); SetConsoleCtrlHandler(t_ctrl,TRUE); SetCurrentDirectoryA(exe_dir());
    int sel=0;
    for(;;){
        int n=0, ids[T_NCMD];
        for(int i=0;i<T_NCMD;i++) if(!T_CMDS[i].grp){ ids[n]=i; snprintf(t_items[n],260,"%s",T_CMDS[i].title); snprintf(t_info[n++],T_INFO,"%s",T_CMDS[i].help); }
        int nc=n;
        snprintf(t_items[n],260,"Data tools..."); snprintf(t_info[n++],T_INFO,"download text, train a tokenizer, build / mix / inspect corpora");
        snprintf(t_items[n],260,"Model config"); snprintf(t_info[n++],T_INFO,"edit a model's saved settings (models\\<name>.bin.env)");
        snprintf(t_items[n],260,"Environment / profile"); snprintf(t_info[n++],T_INFO,"all CUDALM_* QLM_* NGRAM_* settings for the commands");
        snprintf(t_items[n],260,"Quit"); t_info[n++][0]=0;
        int s=t_pick("main menu","Up/Down  Enter open  Esc quit   (cudalm help = CLI usage)",n,sel);
        if(s<0 || s==n-1){ printf("\x1b[2J\x1b[H"); return 0; }
        sel=s;
        if(s<nc) t_form(&T_CMDS[ids[s]]);
        else if(s==nc) t_data_menu();
        else if(s==nc+1) t_model_cfg();
        else { t_st.file=0; t_reg(&t_st,"Environment (inherited by every command; saved into a model's config by new/train)"); }
    }
}
#endif
