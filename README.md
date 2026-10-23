# cudalm

A single-file CUDA trainer and inference engine for small Mixture-of-Experts language models. It also includes its
own data toolchain (BPE tokenizer training, corpus encoding and mixing, Hugging Face downloads), a console menu, a
chat benchmark, and GGUF export.

- Models are MoE transformers with top-k routing. An optional hashed n-gram "engram" memory is attached to chosen layers.
- QLM mode: 4-bit weights with an integer AdamW optimizer, so a ~100M-parameter model trains in a few GB of VRAM.
- The schedule is WSD/cosine, checkpoints can be resumed, and chat SFT works with an assistant-only loss mask (ChatML).
- Everything except the optional helper tools is C/C++ in `cudalm.cu` plus a few headers. The only external library is cuBLAS.
- No NVIDIA GPU? `python build.py --cpu` builds the same program for the CPU (OpenMP + AVX2). See [CPU build](#cpu-build).

**Tested on:** Windows 11, an RTX 50-series GPU (sm_120), and CUDA 12.9. Other architectures (`--arch 89`, `86`, ...)
and Linux should compile, but they are untested. The data tools and the console menu are Windows-only.

## Build

```
python build.py
```

`build.py` finds or downloads everything it needs:

| Need | How it's found / installed |
|---|---|
| CUDA (nvcc, cudart, cccl, cuBLAS) | `--cuda DIR`, or an installed toolkit (`CUDA_PATH` / `nvcc` on PATH, ≥ 12.8 for sm_120). Otherwise it downloads NVIDIA's redistributable archives (~650 MB, sha256-checked) into `.deps/cuda-12.9.1/`. |
| Host C++ compiler | Windows: MSVC from Visual Studio 2022 Build Tools, found via vswhere. Without Visual Studio it offers a **portable MSVC**: just the x64 compiler and Windows SDK, unpacked from Microsoft's package feed into `.deps/msvc/` (~290 MB download, ~750 MB on disk, no install, no admin). This accepts the VS Build Tools license; pass `--accept-msvc-license` to skip the prompt. `--install-msvc` runs Microsoft's full installer instead (admin, ~3 GB). Linux: `g++`. |
| GPU arch | `--arch` if given, otherwise read from `nvidia-smi`, otherwise 120. |
| NVIDIA driver | Must already be installed (a driver supporting CUDA 12.9 or newer). |

The result is `cudalm.exe`, with the two cuBLAS DLLs copied next to it. Other options:

| Option | What it does |
|---|---|
| `--tools` | Also builds `judge.exe` and `teachgen.exe` (uses gcc if present, else MSVC). |
| `--python-deps` | Runs `pip install -r requirements.txt`. |
| `--dry-run` | Prints the plan without changing anything. |
| `--out NAME` | Uses a different exe name. |
| `--download-cuda` | Downloads CUDA even if an installed toolkit exists. |
| `--cuda-version X.Y.Z` | Picks a different redist version. |
| `--accept-msvc-license` | Downloads the portable MSVC without asking (for scripted builds). |

### CPU build

```
python build.py --cpu                    (needs only g++ with OpenMP; Windows: MSYS2 mingw-w64)
```

This builds `cudalm_cpu.exe` from the same `cudalm.cu` with `-DCUDALM_CPU`. `cpu_rt.h` stands in for the CUDA runtime
and cuBLAS (it has a packed AVX2/FMA SGEMM). `cpu_kern.h` has OpenMP versions of every kernel, including blocked
flash attention with a fused backward. All commands, model files and data are shared with the GPU build. The fp16, FP8 and
tensor-core paths are GPU-only, so the CPU build always runs the fp32 path. It gives the same results as the GPU with
`CUDALM_FP16=0 CUDALM_FP8_GEMM=0 CUDALM_H16=0 CUDALM_TF32=0`: identical greedy text and identical eval loss, checked on the
~120M-parameter 32-expert reference checkpoint.

| `--cpu-arch` | Runs on |
|---|---|
| `native` (default) | the build machine (fastest; the exe may not run on other CPUs) |
| `haswell` | any x86-64 CPU with AVX2 + FMA (2013+) |
| `x86-64-v2` | CPUs without AVX2 (SSE fallbacks, ~3.5x slower) |

Speed on a 6-core i5-9400F: ~1.9k tok/s training the 10M `s8w` preset (B=8, T=1024), ~2k tok/s eval and ~16 tok/s
generation on that ~120M 32-expert top-2 model. That is fine for testing, evaluation, chat and small models. Real pretraining
still wants a GPU.

## Quick start

```
cudalm                                             (no arguments: menu with help text for every option)
cudalm data fetch v3chat text_chat 2e8             download ~200M tokens of chat data as .jsonl
cudalm data tokenizer mytok text_chat 16384        train a 16k byte-level BPE (+ canon.bin)
cudalm data encode new chat text_chat mytok        -> data\chat\train.bin, val.bin, meta.json
runs\pretrain.bat mymodel chat                     the reference recipe (32-expert MoE, QLM, engram)
cudalm run mymodel "Hello" 128 0.7 40
bench\chatbench.bat mymodel
```

`python prep\fetch.py list` prints the available download profiles. `cudalm data fetch <file|folder> <out>` converts
your own parquet, json or jsonl files instead.

## CLI

```
cudalm new       <name> <preset> [experts]
cudalm train     <name> <corpus> [epochs]
cudalm eval      <name> <corpus> [batches] [batch_size]
cudalm run       <name|model.bin> ["prompt"] [n] [temp] [topk]
cudalm chatbench <name> <bench.txt> [ngen] [temp] [topp] [reppen] [nsamp]
cudalm lmeval    <name> [hellaswag,arc_easy,arc_challenge,piqa,lambada,arithmark|all] [n] [Tmax]
cudalm lmeval-vs <A.items.tsv> <B.items.tsv>       paired per-item comparison of two lmeval dumps
cudalm quantize  <in> <out.bin> [q8|q4|q4n|fp16]
cudalm gguf      <name|model.bin> [q4n|q4|q8|fp16|f32] [tinylm|llama] [out.gguf]
cudalm info      <name>
cudalm serve     <name> [port=8090]                  HTTP /completion for agentenv (continuous batching, CUDALM_SERVE_SLOTS)
cudalm data      <fetch|tokenizer|encode|mix|info|canon> ...
cudalm help                                        all commands + every environment variable
```

`cudalm lmeval` ends with the Open SLM intelligence index. Each component is normalised against chance,
N(s, c) = 100 × (s − c) ÷ (100 − c), using acc_norm: HellaSwag (c 25), ARC = mean of ARC-Easy and ARC-Challenge (c 25),
PIQA (c 50), ArithMark-3 (c 25, weight 0.65). Index = (HS + ARC + PIQA + 0.65 × ArithMark) ÷ 3.65. A task that wasn't
run drops out of both sums (at least two components needed), and scores below chance count as negative.

Near chance, accuracy is mostly noise, so `lmeval` also reports a **margin** per task: per item, the log-probability per
byte of the right answer minus the best wrong one (LAMBADA: log-probability per byte of the target word), averaged ± SE.
The **margin index** combines them with the same weights (× 100). `CUDALM_LMEVAL_ITEMS=<file>` writes every item's
margin and correctness; `cudalm lmeval-vs A.tsv B.tsv` pairs two such dumps item by item and prints, per task, the
accuracy flips each way and the mean margin change ± paired SE, then the margin-index change with its z. Treat |z| < 2
as no change.

Training options are `CUDALM_*`, `QLM_*` and `NGRAM_*` environment variables (`cudalm help` lists them all).
`cudalm new` and `cudalm train` save the ones in effect to `models\<name>.bin.env`, so later commands don't need them
again. To stop a training run cleanly, create `models\<name>.bin.stop` (or press Ctrl+C in the menu).

**Epochs and stages.** `[epochs]` is how many passes over *the corpus you name* to train from now on. Training a model
that has been trained before (e.g. chat SFT on a pretrained base) starts a new **stage**: the warmup and the LR schedule
restart over the stage's steps, and sampling starts at epoch 0 of the new corpus. `models\<name>.bin.run` records the
stage (start/end iteration, corpus), so re-running the same command after a stop or crash finishes that stage instead
of adding more. A stage on a trained model defaults to a third of the from-scratch peak LR (re-warming a cooled-down
model to the full peak makes the grad norm blow up); set it with `CUDALM_LR` (1e-4 suits chat SFT) and the cooldown
share with `CUDALM_WSD_DECAY` (fraction of the stage). `CUDALM_ITERS=N` keeps the old behaviour: one schedule over
iterations 0..N of the model's whole life (`runs\sft.bat` uses it). `CUDALM_ITERS`, `CUDALM_WSD_DECAY`,
`CUDALM_REWARMUP` and `CUDALM_PERM_BASE` are per-run settings: they are not saved to `.bin.env`, and old `.bin.env`
files that contain them are ignored for those keys.

**Benchmarks during training.** `CUDALM_BENCH_EVERY=N` (TUI: Train > bench every) runs `lmeval` on the checkpoint saved
at or after every N steps of the stage, and once more at the end. It runs as a child process; training waits, and that time
is left out of tok/s. It prints the Open SLM index with per-task scores and appends a row to `models\<name>.bin.bench.tsv`.
By default it runs every task (HellaSwag, ARC-E, ARC-C, PIQA, LAMBADA, ArithMark) on its full set; `CUDALM_BENCH_TASKS`
(comma list) and `CUDALM_BENCH_N` (items per task) narrow it. Each bench also saves its per-item dump to
`models\<name>.bin.items\it<N>.tsv`, prints the paired comparison with the previous bench, and adds the margin index as
the last column of `bench.tsv`.

In the menu every command is a form (required fields are checked before Run). Its output is shown live and also
saved to `logs\<command>-<model>-<time>.log`, e.g. the chatbench log is the input for `bench\judgerun.bat`. Diagnostics
(selftest, decode / tokenizer / QLM checks) have their own submenu.

### Data

- **Folders:** every run creates `models\`, `data\`, `bench\` and `logs\` next to the exe if they are missing. Models
  are `models\<name>.bin` plus sidecars named after them (`.env`, `.opt*`, `.run`, `.bench.tsv`).
- **Corpus names:** bare names resolve under the data roots: `<exe>\data`, `<exe>\prep\data`, and any extra folders
  listed one per line in `data_roots.txt` next to the exe. Set `CUDALM_DATA_ROOTS` (`;`-separated) to replace all of them.
- **Input files:** `.jsonl`, `.json` or `.txt`. Accepted rows:
  - `text`;
  - chat `messages` / `conversations` (with optional `system`, `tools`);
  - alpaca-style `instruction` / `input` / `output`;
  - `content` / `code` / `markdown` / `body`.
- **Mixing:** `cudalm data mix <out> <train_tokens|0> <max_repeat> <val_tokens> <corpus> <weight> ...` mixes encoded
  corpora that share a tokenizer. Token counts (fetch, mix) accept `k` / `M` / `B` suffixes (`500M`, `1B`).
- **Templates (which columns become the document):** `--template` on `data fetch` (hf:, URL and convert), the
  `template` field in Download text / Convert files, and T in the Hugging Face preview.
  | Template | Result |
  | --- | --- |
  | empty | auto-detect: `text`, chat `messages`/`conversations`, instruction/answer pairs, `content`/`code` |
  | `text=title+body` | plain text, columns joined by a blank line (`col` alone = `text=col`) |
  | `user=question+context,assistant=answer` | one `<\|user\|>` / `<\|assistant\|>` exchange; several columns are joined |
  | `chat=conversation` | a list of turns (`role`/`content` or `from`/`value`) in any column |
  | `+ system=col`, `tools=col` | added to a chat or user/assistant template |

  `cudalm data peek <file|folder|hf:spec> [template]` prints the columns and the first rows exactly as the model will
  see them. The TUI editor shows the same live, with keys T text, P prompt + answer, C chat, S system, O tools, A auto.
- **Any Hugging Face dataset:** `cudalm data fetch hf:<user/dataset>[:config[:split]] <out> [tokens] [--template T]` downloads
  the hub's parquet files for that split, or the repo's own .parquet/.jsonl/.json files when the hub has no conversion.
  The config defaults to `default` and the split to `train`. Text is auto-detected the same way as Convert files:
  `text`, chat `messages`/`conversations`, instruction/answer pairs, or `content`/`code`/`markdown`/`body`.
  `--template` (or `@template` at the end of the spec) picks the columns instead (see Templates).
  - `cudalm data hf search <query>`: lists matching datasets.
  - `cudalm data hf info <user/dataset>`: lists configs and splits with row counts and sizes.
  - `cudalm data hf rows <user/dataset> [config] [split]`: shows the columns, what fetch will read, and the first rows.
  - In the TUI, **Browse Hugging Face** (top of Data tools) does the same interactively and fills in Download text.
  - Gated datasets need `HF_TOKEN`. Script-only datasets can't be read.
- **TUI Data tools:**
  - Each form shows a live panel: the resolved output folder, free space and estimated size, range checks, missing
    tokenizer/corpus, tokenizer mismatches, and outputs that already exist. Mix also shows a blend table with each
    corpus's share, the tokens it contributes and the repeats. F5 with problems asks before running.
  - Pickers show a tokenizer id (`tok 289cba`; equal ids mean an identical `tokenizer.bin`) and preview the
    highlighted corpus. The text picker lists folders under the data roots; Space marks several.
  - After a successful run, the output is filled into the next forms. Enter opens the next step, I shows info on
    the output, and L opens the log.
  - Canon map and Loss weights sit under "advanced". Build corpus `follow` appears with F2.
  - **Manage data folders** lists every folder under the roots with its type, size and date, and can open, show
    info, rename, or send one to the Recycle Bin. **Data roots** adds folders to `data_roots.txt` and removes them.
  - `cudalm tuicheck [model] ["Form:label=value" ...]` prints every form's command line and data panel without the UI.
- **Per-token loss weights (RL):** `cudalm data weights <corpus> <rows.jsonl>` writes `train.w` from each row's `"w"`
  (encode those rows with `val%=0 dedup=0` first). `data mix` carries it through (1.0 for corpora without one), and
  `CUDALM_TOKW=1` trains with it: negative weights push failed rollouts down until their p < `CUDALM_NEG_PMIN`.
- **Python:** `cudalm data fetch` runs `prep\fetch.py` with `py -3`. Override it with `CUDALM_PYTHON`. `HF_TOKEN`
  is sent when set (needed for gated datasets).

### GGUF export

`cudalm gguf <name> [type] [target]` runs `export\bin2gguf.py` (python + numpy, `CUDALM_PYTHON` applies) with the
model's saved `.env`, so MoE top-k is recorded. `type` uses the quantize modes: q4n/q4/q8 quantize an fp32 model first,
fp16/f32 keep floats; a quantized model exports as its own type.

- `tinylm` (default, type default q4n): exact. Keeps top-k routing, router bias and the ngram/engram memory.
- `llama` (type default q8; q8|fp16|f32): stock llama.cpp `deepseek` arch. Routes top-1 and drops ngram/engram, so it is lossy.

## Layout

| Path | What |
|---|---|
| `cudalm.cu` | Engine: model, training, eval, sampling, chatbench, quantize, CLI |
| `qlm.cuh` | 4-bit weights + integer AdamW (QLM) |
| `cpu_rt.h`, `cpu_kern.h` | CPU build only: CUDA runtime/cuBLAS stand-in (AVX2 SGEMM) and OpenMP/AVX2 kernels |
| `tui.cuh` | Console menu (Windows) |
| `data.cuh` | `cudalm data`: BPE trainer, encoder, mixer, canon map (CPU, Windows) |
| `uniclass.h` | Unicode tables for the GPT-2 pre-tokenizer regex |
| `build.py` | Toolchain download and build (`--cpu` for the CPU build) |
| `prep\fetch.py` | Hugging Face downloads and parquet/json → jsonl conversion (called by `cudalm data fetch`) |
| `prep\legacy\` | Older Python corpus builders (numpy + `tokenizers`); kept for reference |
| `runs\` | Reference recipe: `s8w2env.bat` (model env), `pretrain.bat`, `sft.bat` (chat SFT continuing a base model) |
| `bench\chatbench.txt`, `chatbench.bat`, `bench\chatbench_v2.txt` | Chat benchmark: 118 multiple-choice + 24 open-generation prompts (v2 expanded) |
| `bench\judge.c`, `judgerun.bat` | Coherence judge: scores chatbench replies with any instruct GGUF via llama-server (`JUDGE_GGUF`, `LLAMA_SERVER`) |
| `bench\lmeval_prep.py` | Converts HellaSwag (val), ARC-Easy and ARC-Challenge (test), PIQA (val), LAMBADA (OpenAI test) and ArithMark 3.0 (AxiomicLabs, 1000 arithmetic word problems) into the `cudalm lmeval` .tsv files (lm-eval-harness prompts); `cudalm lmeval` reads them from `CUDALM_BENCH_DIR` (default `<exe dir>\bench`) and runs it to download + convert them when a task file is missing |
| `bench\gemmbench.cu` | MoE expert-GEMM micro-benchmark |
| `teach\teachgen.c`, `teachrun.bat` | Synthetic teacher data from llama-server instances (`TEACHER_GGUF`, `LLAMA_SERVER`) |
| `teach\teachtok.c` | Teacher jsonl → train/val token bins with the model's tokenizer (special tokens matched literally) |
| `agent\agentenv.c` | Agent evals against `cudalm serve` (or a teacher server): tool use, code pass@k, task generation |
| `export\bin2gguf.py` | Model `.bin` → GGUF (needs numpy) |

## Recipe notes

- **`runs\pretrain.bat`:**
  - preset `s8w`: 32 experts, top-2;
  - batch 16, LR 7e-4, WSD;
  - QLM 4-bit;
  - a q4 sparse engram on layers 1 and 4.
  - If 20 steps in a row are non-finite, the trainer exits with code 3, and the script resumes from the last checkpoint.
- **`runs\sft.bat base sft chat_corpus <base_iters> <sft_iters>`:** copies the base model, re-warms to LR 1e-4, and
  decays over the last 40% of the SFT steps. `<base_iters>` is the base run's final iteration.
- **Measuring SFT:** use `bench\chatbench.bat` (accuracy) and `bench\judgerun.bat` (coherence). Validation loss alone
  is a poor guide.
- **Speed switches:** each of these is on by default and has an off switch for A/B runs. On a 5060 Ti, LS-Micro (top-1) went from
  254k to 294k tok/s and s8w (top-2) from 108k to about 120k.
  - `CUDALM_TOP1_OLD=1`: runs top-1 on its old fp32 path. By default top-1 runs as top-K with K=1, which uses bf16 fused experts.
  - `CUDALM_MOE_GPERM=0`: builds the expert permutation on the host. By default the GPU builds it with the same stable order.
  - `CUDALM_MOE_MERGE=0`: runs the top-2 passes separately. By default they run as one grouped pass.
  - `CUDALM_FA_CP=0`: turns off the cp.async tile prefetch in the FA2 attention kernels.
  - `CUDALM_LRS_KEEP=0`: makes the LR scaler retrain the steps of the ×1.0 arm instead of keeping it.
  - `CUDALM_CH_ROWS`: sets the head row chunk. It defaults to 2048; 8192 was slower.
  - `CUDALM_NOSYNC`: on by default on Linux.
