# cudalm

A single-file CUDA trainer and inference engine for small Mixture-of-Experts language models. It also includes its
own data toolchain (BPE tokenizer training, corpus encoding and mixing, Hugging Face downloads), a console menu, a
chat benchmark, and GGUF export.

- Models are MoE transformers with top-k routing. An optional hashed n-gram "engram" memory is attached to chosen layers.
- QLM mode: 4-bit weights with an integer AdamW optimizer, so a ~100M-parameter model trains in a few GB of VRAM.
- The schedule is WSD/cosine, checkpoints can be resumed, and chat SFT works with an assistant-only loss mask (ChatML).
- Everything except the optional helper tools is C/C++ in `cudalm.cu` plus four headers. The only external library is cuBLAS.

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
cudalm quantize  <in> <out.bin> [q8|q4|q4n|fp16]
cudalm info      <name>
cudalm data      <fetch|tokenizer|encode|mix|info|canon> ...
cudalm help                                        all commands + every environment variable
```

Training options are `CUDALM_*`, `QLM_*` and `NGRAM_*` environment variables (`cudalm help` lists them all).
`cudalm new` and `cudalm train` save the ones in effect to `models\<name>.bin.env`, so later commands don't need them
again. To stop a training run cleanly, create `models\<name>.bin.stop` (or press Ctrl+C in the menu).

### Data

- **Corpus names:** bare names resolve under the data roots: `<exe>\data`, `<exe>\prep\data`, and any extra folders
  listed one per line in `data_roots.txt` next to the exe. Set `CUDALM_DATA_ROOTS` (`;`-separated) to replace all of them.
- **Input files:** `.jsonl`, `.json` or `.txt`. Accepted rows:
  - `text`;
  - chat `messages` / `conversations` (with optional `system`, `tools`);
  - alpaca-style `instruction` / `input` / `output`;
  - `content` / `code` / `markdown` / `body`.
- **Mixing:** `cudalm data mix <out> <train_tokens|0> <max_repeat> <val_tokens> <corpus> <weight> ...` mixes encoded
  corpora that share a tokenizer.
- **Python:** `cudalm data fetch` runs `prep\fetch.py` with `py -3`. Override it with `CUDALM_PYTHON`. `HF_TOKEN`
  is sent when set (needed for gated datasets).

## Layout

| Path | What |
|---|---|
| `cudalm.cu` | Engine: model, training, eval, sampling, chatbench, quantize, CLI |
| `qlm.cuh` | 4-bit weights + integer AdamW (QLM) |
| `tui.cuh` | Console menu (Windows) |
| `data.cuh` | `cudalm data`: BPE trainer, encoder, mixer, canon map (CPU, Windows) |
| `uniclass.h` | Unicode tables for the GPT-2 pre-tokenizer regex |
| `build.py` | Toolchain download and build |
| `prep\fetch.py` | Hugging Face downloads and parquet/json → jsonl conversion (called by `cudalm data fetch`) |
| `prep\legacy\` | Older Python corpus builders (numpy + `tokenizers`); kept for reference |
| `runs\` | Reference recipe: `s8w2env.bat` (model env), `pretrain.bat`, `sft.bat` (chat SFT continuing a base model) |
| `bench\chatbench.txt`, `chatbench.bat` | Chat benchmark: 118 multiple-choice + 24 open-generation prompts |
| `bench\judge.c`, `judgerun.bat` | Coherence judge: scores chatbench replies with any instruct GGUF via llama-server (`JUDGE_GGUF`, `LLAMA_SERVER`) |
| `bench\gemmbench.cu` | MoE expert-GEMM micro-benchmark |
| `teach\teachgen.c`, `teachrun.bat` | Synthetic teacher data from llama-server instances (`TEACHER_GGUF`, `LLAMA_SERVER`) |
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
