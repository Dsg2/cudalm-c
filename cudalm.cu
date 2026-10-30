/* cudalm.cu - CUDA backend for tinylm (training + inference), data-resident.
 *
 * Port of gpu.c (OpenCL) to CUDA. Shares the TLM1-4 checkpoint format, TLTK
 * tokenizer, models/ dir and data/ files with the CPU build (tinylm.c), so a
 * model trained by one runs in the other.
 *
 * Build (needs nvcc + MSVC host compiler):
 *   nvcc -O3 -arch=sm_75 -o cudalm.exe cudalm.cu -lcublas
 *
 * Commands:
 *   cudalm run      <name|model.bin> ["prompt"] [n] [temp] [topk]
 *   cudalm serve    <name|model.bin> [port=8090]
 *   cudalm train    <name> <data_dir> [epochs]
 *   cudalm bench    <preset> [iters] [batch]
 *   cudalm selftest
 *
 * Differences from gpu.c, all deliberate:
 *  - cuBLAS for the three GEMM shapes instead of hand-written kernels.
 *  - Attention backward is a 3-kernel flash-style split using the forward's
 *    saved log-sum-exp. gpu.c ran B*KV work items (32 at B=16,KV=2) and was
 *    occupancy-bound; this runs B*H*T and B*KV*T, still atomic-free.
 *  - lm_head + cross-entropy are row-chunked so the N*V logit buffer never
 *    has to be materialised in full.
 *  - Inference computes logits for the single row it samples from, not all T.
 *  - Dense only: MoE checkpoints (n_exp>0) are rejected with a message.
 *
 * Attention carried 77.5% of the training step (measured, CUDALM_PROF=1), so
 * two changes target it and nothing else:
 *  - Drow is computed as dA.O instead of re-walking the window (k_attn_drow).
 *    Same quantity, O(head_dim) instead of O(window); that pass alone was 17.9%.
 *  - Q/K/V/dA are cast to __half and read as __half2, halving the bytes AND the
 *    transaction count in the four window loops. Accumulators, softmax, lse,
 *    weights, gradients and AdamW all stay fp32 -- this is mixed precision, not
 *    fp16 training. CUDALM_FP16=0 restores the fp32 operands.
 * Checkpoints are unaffected: the on-disk format and the weights are fp32 as
 * before, so a model is still interchangeable with the CPU build.
 */
#define _CRT_SECURE_NO_WARNINGS
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#ifdef CUDALM_CPU
#include "cpu_rt.h"
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#define KL(k,...) k<<<__VA_ARGS__>>>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#ifdef _WIN32
  #include <windows.h>
  #include <direct.h>
  #include <io.h>               /* _commit/_fileno: fsync for durable checkpoints */
#else
  #include <unistd.h>           /* fsync, readlink, fileno */
  #include <fcntl.h>            /* open, for the directory fsync */
  #include <sys/stat.h>         /* mkdir */
  #include <sys/statvfs.h>      /* disk_low */
  #include <sys/types.h>
  #include <sys/wait.h>         /* WEXITSTATUS */
#endif

/* ---------------- platform shims -----------------------------------------
 * Windows is the reference build; Linux exists for Kaggle (P100). Everything
 * platform-specific is confined here so the two cannot drift.
 * PSEP matters: model_path builds paths by hand.                             */
/* lm_fseek64/lm_ftell64: plain fseek/ftell return `long`, which is 32-BIT on
 * Windows x64. A corpus over 2 GB then reports its size as -1 (or truncated)
 * and read_file silently fails. This is not hypothetical: the 960M-token
 * train.bin was 1.79 GB and worked, the 2.01B-token one is 3.75 GB and did not.
 * Linux's long is 64-bit, so this only ever bit on the reference build. */
#ifdef _WIN32
  #define PSEP "\\"
  #define lm_mkdir(p)      _mkdir(p)
  #define lm_fsync(f)      _commit(_fileno(f))
  #define lm_strtou64(s)   _strtoui64((s),NULL,10)
  #define lm_fseek64(f,o,w) _fseeki64((f),(o),(w))
  #define lm_ftell64(f)     _ftelli64(f)
#else
  #define PSEP "/"
  #define lm_mkdir(p)      mkdir((p),0755)
  #define lm_fsync(f)      fsync(fileno(f))
  #define lm_strtou64(s)   strtoull((s),NULL,10)
  #define lm_fseek64(f,o,w) fseeko((f),(off_t)(o),(w))
  #define lm_ftell64(f)     (long long)ftello(f)
#endif
/* file helpers for the training loop (stop file, *_itN snapshots, disk-full wait) */
static int lm_exists(const char*p){
#ifdef _WIN32
    return GetFileAttributesA(p)!=INVALID_FILE_ATTRIBUTES;
#else
    return access(p,F_OK)==0;
#endif
}
static void lm_sleep_ms(unsigned ms){
#ifdef _WIN32
    Sleep(ms);
#else
    usleep((useconds_t)ms*1000u);
#endif
}
/* 1 on success; overwrites dst */
static int lm_copyfile(const char*src,const char*dst){
#ifdef _WIN32
    return CopyFileA(src,dst,FALSE)!=0;
#else
    FILE*a=fopen(src,"rb"); if(!a) return 0;
    FILE*b=fopen(dst,"wb"); if(!b){ fclose(a); return 0; }
    static char buf[1<<20]; size_t n; int ok=1;
    while((n=fread(buf,1,sizeof buf,a))>0) if(fwrite(buf,1,n,b)!=n){ ok=0; break; }
    if(ferror(a)) ok=0;
    fclose(a); if(fclose(b)!=0) ok=0;
    if(!ok) remove(dst);
    return ok;
#endif
}

#define CH_ROWS 2048      /* lm_head/CE row chunk (caps the logits buffer); lmeval keeps this */
static int g_chrows=CH_ROWS;  /* O8: CUDALM_CH_ROWS, the train/eval head chunk */
/* Timesteps handled per attention-backward launch. Caps launch duration so it
 * does not grow with T -- the TDR precaution. Override with CUDALM_ATTN_CHUNK. */
static int g_attn_chunk=128;
/* Mixed-precision attention: fp16 Q/K/V/dA operands, fp32 everywhere else.
 * On by default where the head dim allows it; CUDALM_FP16=0 forces the fp32
 * kernels back (needed to reproduce a bit-exact CPU-parity run). */
static int g_fp16=1;
static int g_muon=0, g_muon_exp=0;   /* Muon on 2-D tensors (env CUDALM_MUON, default on); exp: set explicitly */

static void ck(cudaError_t e, const char *what){
    if(e != cudaSuccess){ fprintf(stderr,"CUDA error: %s at %s\n", cudaGetErrorString(e), what); exit(1); }
}
static void ckb(cublasStatus_t s, const char *what){
    if(s != CUBLAS_STATUS_SUCCESS){ fprintf(stderr,"cuBLAS error %d at %s\n",(int)s,what); exit(1); }
}
static void *dmalloc(size_t bytes){ void *p=NULL; ck(cudaMalloc(&p,bytes),"cudaMalloc"); return p; }
static float *dzero(size_t n){ float *p=(float*)dmalloc(n*4); ck(cudaMemset(p,0,n*4),"memset"); return p; }

/* ===================== shared host scaffolding (mirrors tinylm.c) ========= */
/* win: sliding-attention window (0 = full causal attention). full_every: every
 * Nth layer ignores the window and attends to the whole prefix (0 = never),
 * which restores exact long-range retrieval that pure SWA approximates. */
typedef struct { int V,D,L,H,KV,F,T,hd,KD,half,eot,tmpl,n_mtp,n_exp,win,full_every,pool; float rope_base;
                 int mod,lt[64],lE[64],lF[64]; int lp,lb,lmap[64]; } Cfg;   /* mod: TLM7 per-layer FFN (lt 0 dense/1 MoE/2 pool); lp/lb/lmap: CUDALM_LOOP */
static void cfg_derive(Cfg*c){ c->hd=c->D/c->H; c->KD=c->KV*c->hd; c->half=c->hd/2; }
static int lyT(const Cfg*c,int l){ return c->mod?c->lt[l]:c->n_exp>0?(c->pool?2:1):0; }
static int lyE(const Cfg*c,int l){ return c->mod?c->lE[l]:c->n_exp; }
static int lyF(const Cfg*c,int l){ return c->mod?c->lF[l]:c->F; }
/* CUDALM_LOOP="a-bxk": layers a..b run k times with shared weights (L becomes the virtual depth,
 * lp the physical one). Each repeat pass keeps its own norm gains, appended after all other params. */
static int lyA(const Cfg*c,int v){ return c->lp&&v>c->lb&&c->lmap[v]<=c->lb; }   /* v is a repeat of an earlier layer */
static size_t loop_xn(const Cfg*c){ size_t n=0; for(int v=0;v<c->L;v++) n+=lyA(c,v); return n*2*c->D; }
static void loop_apply(Cfg*c){
    const char*s=getenv("CUDALM_LOOP"); int a,b,k;
    if(!s||!*s||!strcmp(s,"0")||c->lp) return;
    if(sscanf(s,"%d-%dx%d",&a,&b,&k)!=3||k<2||a<0||b<a||b>=c->L||c->L+(b-a+1)*(k-1)>64){
        fprintf(stderr,"CUDALM_LOOP=%s: want a-bxk with 0<=a<=b<L=%d, k>=2, total depth <=64\n",s,c->L); exit(1); }
    Cfg p=*c; int n=0, m[64];
    for(int l=0;l<a;l++) m[n++]=l;
    for(int r=0;r<k;r++) for(int l=a;l<=b;l++) m[n++]=l;
    for(int l=b+1;l<p.L;l++) m[n++]=l;
    for(int v=0;v<n;v++){ c->lmap[v]=m[v]; c->lt[v]=p.lt[m[v]]; c->lE[v]=p.lE[m[v]]; c->lF[v]=p.lF[m[v]]; }
    c->lp=p.L; c->lb=b; c->L=n; }
static int NPOOL(const Cfg*c){ int n=0; for(int l=0;l<c->L;l++) n+=lyT(c,l)==2; return n; }
static int NSPARSE(const Cfg*c){ int n=0; for(int l=0;l<c->L;l++) n+=lyT(c,l)!=0; return n; }
static int FXW(const Cfg*c){ int m=c->F; for(int l=0;l<c->L;l++) if(!lyT(c,l)&&lyF(c,l)>m) m=lyF(c,l); return m; }
/* CUDALM_LAYOUT="D2560,M28,P,..." (one entry per layer; sets L): D<F> dense, M<E> MoE of E experts,
 * P = UniPool layer (CUDALM_POOL_E experts shared by all P layers). Expert width CUDALM_EXP_F (default F). */
static int layout_apply(Cfg*c){
    const char*s=getenv("CUDALM_LAYOUT"); if(!s||!*s) return 1;
    const char*e=getenv("CUDALM_EXP_F"); if(e&&atoi(e)>0) c->F=atoi(e);
    int PE=(e=getenv("CUDALM_POOL_E"))?atoi(e):0, L=0, mx=0, np=0;
    for(const char*p=s;*p;){ while(*p==','||*p==' ') p++; if(!*p) break;
        if(L>=64){ fprintf(stderr,"layout: >64 layers\n"); return 0; }
        char t=*p++; int v=atoi(p); while(*p&&*p!=',') p++;
        if(t=='D'||t=='d'){ c->lt[L]=0; c->lE[L]=0; c->lF[L]=v>0?v:c->F; }
        else if(t=='M'||t=='m'){ if(v<2||v>256){ fprintf(stderr,"layout: M%d needs 2..256 experts\n",v); return 0; }
            c->lt[L]=1; c->lE[L]=v; c->lF[L]=c->F; }
        else if(t=='P'||t=='p'){ if(PE<2||PE>256){ fprintf(stderr,"layout: P needs CUDALM_POOL_E in 2..256\n"); return 0; }
            c->lt[L]=2; c->lE[L]=PE; c->lF[L]=c->F; np++; }
        else { fprintf(stderr,"layout: bad entry '%c'\n",t); return 0; }
        if(c->lE[L]>mx) mx=c->lE[L]; L++; }
    if(!L) return 0;
    c->mod=1; c->L=L; c->n_exp=mx; c->pool=np>0; return 1; }

static int preset(const char*n, Cfg*c){
    memset(c,0,sizeof(*c)); c->T=256; c->rope_base=10000.0f;
    if(!strcmp(n,"nano")){c->V=512;c->D=64;c->L=4;c->H=4;c->KV=2;c->F=176;}
    else if(!strcmp(n,"micro")){c->V=2048;c->D=96;c->L=4;c->H=6;c->KV=2;c->F=256;}
    else if(!strcmp(n,"m1")) {c->V=2048;c->D=128;c->L=4;c->H=4;c->KV=1;c->F=336;}
    else if(!strcmp(n,"m3")) {c->V=2048;c->D=192;c->L=6;c->H=6;c->KV=2;c->F=512;}
    else if(!strcmp(n,"m6")) {c->V=2048;c->D=256;c->L=8;c->H=8;c->KV=2;c->F=688;}
    else if(!strcmp(n,"m8")) {c->V=2048;c->D=288;c->L=8;c->H=8;c->KV=2;c->F=768;}
    else if(!strcmp(n,"m16")){c->V=2048;c->D=512;c->L=5;c->H=8;c->KV=2;c->F=1408;}
    else if(!strcmp(n,"m32")){c->V=2048;c->D=512;c->L=10;c->H=8;c->KV=2;c->F=1536;}
    else if(!strcmp(n,"m70")){c->V=2048;c->D=768;c->L=11;c->H=12;c->KV=2;c->F=2048;}
    /* ~48.3M dense, the target shape for this backend */
    else if(!strcmp(n,"m50")){c->V=8192;c->D=640;c->L=10;c->H=10;c->KV=2;c->F=1728;c->T=1024;}
    /* MoE base: with 8 experts -> 16.2M active / 63.0M total */
    else if(!strcmp(n,"m64e")){c->V=8192;c->D=512;c->L=8; c->H=8; c->KV=2;c->F=544; c->T=1024;}
    /* Successor to the grown 92M. With 16 experts -> 25.2M active / 220.7M total.
     * F=707 is 1.3x m64e's 544: the predecessor's active FFN was 0.84M against
     * ~2.1M for a dense model of the same width, which is where its factual
     * recall was starved. Doubling experts alone would have left that untouched;
     * widening as well lifts active FFN to 1.09M AND total capacity to 220.7M.
     * Effective capacity (geometric mean of active and total) ~74.6M, matching
     * what 24 experts at F=544 would give while actually fitting in 6.4 GB.
     * State is 3.53 GB, so this runs at B=1 with ~1.6 GB left for the desktop. */
    else if(!strcmp(n,"m220e")){c->V=8192;c->D=512;c->L=12;c->H=8; c->KV=2;c->F=707; c->T=4096;}
    /* Narrow MoE shape for the ~8M-active / ~92M-total top-2 test: emb 3.1M keeps
     * active in reach of 8M. With 35 experts -> ~8M active(top-1) / ~92M total. */
    else if(!strcmp(n,"a8m")){c->V=8192;c->D=384;c->L=6; c->H=6; c->KV=2;c->F=360; c->T=1024;}
    /* a8m scaled 1.5x deep (6->9 layers): ~137M total / ~14.3M active(top-2). */
    else if(!strcmp(n,"a8m9")){c->V=8192;c->D=384;c->L=9; c->H=6; c->KV=2;c->F=360; c->T=1024;}
    /* Sparse MoE + Engram backbone: with 64 experts -> 122.9M total / ~13M active(top-2). */
    else if(!strcmp(n,"s8x")){c->V=8192;c->D=384;c->L=8; c->H=6; c->KV=2;c->F=192; c->T=1024;}
    else if(!strcmp(n,"s8w")){c->V=8192;c->D=384;c->L=8; c->H=6; c->KV=2;c->F=384; c->T=1024;}
    /* modular base (MODULAR_LLM_PLAN.md): pair with CUDALM_LAYOUT; F = expert width */
    else if(!strcmp(n,"mod230")){c->V=16384;c->D=512;c->L=12;c->H=8;c->KV=2;c->F=256;c->T=1024;}
    /* base for a hand-set shape (CUDALM_D/L/H/KV/F); defaults = the LS-Micro shape */
    else if(!strcmp(n,"custom")){c->V=16384;c->D=160;c->L=11;c->H=5;c->KV=1;c->F=432;c->T=2048;}
    else return 0;
    cfg_derive(c); return 1;
}
/* CUDALM_D/_L/_H/_KV/_F shape overrides at new; 0 + err filled if the shape is invalid */
static int shape_env(Cfg*c,char*err,size_t ne){
    int*f[5]={&c->D,&c->L,&c->H,&c->KV,&c->F}; const char*k[5]={"CUDALM_D","CUDALM_L","CUDALM_H","CUDALM_KV","CUDALM_F"};
    for(int i=0;i<5;i++){ const char*e=getenv(k[i]); if(e&&atoi(e)>0) *f[i]=atoi(e); }
    if(c->L>64||c->D%c->H||c->H%c->KV||(c->D/c->H)&1){
        snprintf(err,ne,"bad shape D=%d L=%d H=%d KV=%d (need D%%H==0, H%%KV==0, even head dim, L<=64)",c->D,c->L,c->H,c->KV); return 0; }
    cfg_derive(c); return 1; }
typedef struct { size_t emb; size_t an1[64],wq[64],wk[64],wv[64],wo[64],an2[64],wr[64],w1[64],w3[64],w2[64];
                 size_t nf; size_t mtp[16]; } Off;
/* Layout must match tinylm.c map_weights EXACTLY or checkpoints stop being
 * interchangeable between the CPU and CUDA builds. Note the router sits
 * BEFORE w1/w3/w2, and each expert e lives at w1[l]+e*D*F / w2[l]+e*F*D. */
static size_t calc_offsets(const Cfg*c, Off*o){
    size_t p=0, D=c->D, PE=0; o->emb=p; p+=(size_t)c->V*D;
    for(int l=0;l<c->L;l++){ int t=lyT(c,l);
        if(lyA(c,l)){ int s=c->lmap[l]; o->wq[l]=o->wq[s]; o->wk[l]=o->wk[s]; o->wv[l]=o->wv[s]; o->wo[l]=o->wo[s];
            o->wr[l]=o->wr[s]; o->w1[l]=o->w1[s]; o->w3[l]=o->w3[s]; o->w2[l]=o->w2[s]; continue; }
        o->an1[l]=p;p+=D; o->wq[l]=p;p+=D*D; o->wk[l]=p;p+=D*c->KD;
        o->wv[l]=p;p+=D*c->KD; o->wo[l]=p;p+=D*D; o->an2[l]=p;p+=D;
        o->wr[l]=p; if(t) p+=D*lyE(c,l);
        if(t==2){ PE=lyE(c,l); continue; }
        size_t W=t?(size_t)lyE(c,l)*c->F:(size_t)lyF(c,l);
        o->w1[l]=p;p+=W*D; o->w3[l]=p;p+=W*D; o->w2[l]=p;p+=W*D;
    }
    o->nf=p; p+=D;
    for(int k=0;k<c->n_mtp;k++){ o->mtp[k]=p; p+=D*D; }
    if(PE){ size_t B=PE*D*c->F;                     /* UniPool: P layers alias one pool, after MTP */
        for(int l=0;l<c->L;l++) if(lyT(c,l)==2){ o->w1[l]=p; o->w3[l]=p+B; o->w2[l]=p+2*B; }
        p+=3*B; }
    for(int l=0;l<c->L;l++) if(lyA(c,l)){ o->an1[l]=p; p+=D; o->an2[l]=p; p+=D; }
    return p;
}
static size_t param_count(const Cfg*c){ Off o; return calc_offsets(c,&o); }

/* ---- A2: fused per-layer update state ------------------------------------
 * G normally holds every weight gradient at once -- 4 B/param, the largest
 * single item once the digits are packed. With the fused path on, all layers
 * SHARE one gradient block and the optimizer consumes layer l's gradients at
 * layer_sync() before layer l-1 overwrites them.
 *
 * The cost is that global grad-norm clipping becomes impossible: gpu_adam
 * normally reduces over all of G before touching any parameter, but here the
 * last layer is updated before the first one has a gradient. We clip with the
 * PREVIOUS step's norm -- standard practice, but a real change to the update,
 * which is why this is behind CUDALM_QLM_FUSED and defaults OFF. */
static Off g_goff; static int g_fused=0; static size_t g_gsize=0;
static const Off *g_og=NULL;
static float g_prev_gn=1.0f;
static float *g_fgs=NULL;   /* fused path: device [clip multiplier, norm] from g_prev_gn, for the float-span/side AdamW */
static float g_fused_lr=0.0f; static int g_fused_t=1;
static float *g_gnpart=NULL;                 /* 64 floats per layer */
static size_t g_layer_g0=0, g_layer_gn=0;

/* ---- Phase 2: per-layer P scratch (CUDALM_QLM_PSCRATCH=1) -----------------
 * The rendered fp32 weights P are 4 B/param and redundant with the digits.
 * With pscratch on, P holds only the float spans plus ONE shared layer's
 * worth of quantized weights; each layer is rendered into that block just
 * before its forward and again before its backward. Requires the fused path
 * (the update already runs per layer). Mirrors the A2 gradient compaction. */
static Off g_poff; static int g_pscratch=0; static size_t g_psize=0;
static size_t g_player_p0=0, g_player_pn=0;
static const Off *g_prd=NULL;   /* P read-offset map; NULL => real offsets */

/* ---- bfloat16 storage for the AdamW moments -------------------------------
 * m and v are 8 of the 16 bytes AdamW costs per parameter. At 220.67M params
 * that is 1.77 GB of a 6.4 GB card, and the difference between fitting inside
 * the desktop VRAM reserve and not.
 *
 * bf16 rather than fp16 on purpose: bf16 keeps fp32's 8-bit exponent, so v --
 * which holds SQUARED gradients and routinely runs at 1e-10 and below -- cannot
 * underflow. fp16 bottoms out around 6e-8 and would silently flush v to zero,
 * making the update explode through the 1/sqrt(v) term.
 *
 * The precision cost is benign. bf16 carries 8 mantissa bits, ~0.4% relative.
 * Each step moves m by (1-b1)=10% of itself and v by (1-b2)=5%, so the update
 * is 12-25x the quantisation step and nothing gets rounded away.
 *
 * Done with raw bit ops rather than cuda_bf16.h: bf16 IS the top half of an
 * fp32, and the header's intrinsics want sm_80 while this card is sm_75. */
typedef unsigned short bf16;
__device__ __forceinline__ bf16 d_f2bf(float f){
    unsigned int x=__float_as_uint(f);
    unsigned int r=x+0x7fffu+((x>>16)&1u);      /* round to nearest even */
    return (bf16)(r>>16);
}
__device__ __forceinline__ float d_bf2f(bf16 h){ return __uint_as_float((unsigned int)h<<16); }
static bf16 h_f2bf(float f){
    unsigned int x; memcpy(&x,&f,4);
    unsigned int r=x+0x7fffu+((x>>16)&1u);
    return (bf16)(r>>16);
}
static float h_bf2f(bf16 h){ unsigned int x=(unsigned int)h<<16; float f; memcpy(&f,&x,4); return f; }
/* P + G (fp32) + m + v (bf16) */
static size_t opt_state_bytes(size_t np){ return np*(4+4+2+2) + (g_muon?np*4:0); }

static uint8_t* read_file(const char*path,size_t*len){ FILE*f=fopen(path,"rb"); if(!f)return NULL;
    lm_fseek64(f,0,SEEK_END); long long s=lm_ftell64(f); lm_fseek64(f,0,SEEK_SET);
    if(s<0){ fclose(f); return NULL; }
    uint8_t*b=(uint8_t*)malloc((size_t)(s>0?s:1));
    if(!b){ fclose(f); fprintf(stderr,"read_file: out of memory for %.2f GB (%s)\n",s/1e9,path); return NULL; }
    /* fread is capped at size_t, which is 64-bit here, but read in chunks anyway:
     * some CRTs mis-handle single reads above 2 GB. */
    size_t got=0,n=(size_t)s;
    while(got<n){ size_t want=n-got; if(want>(size_t)1<<30) want=(size_t)1<<30;
        size_t r=fread(b+got,1,want,f); if(!r) break; got+=r; }
    if(got!=n){ fclose(f); free(b);
        fprintf(stderr,"read_file: short read on %s (%zu of %lld bytes)\n",path,got,s); return NULL; }
    fclose(f); *len=n; return b; }
typedef struct { const uint8_t*p,*end; } Rd;
static int rd(Rd*r,void*d,size_t n){ if(r->p+n>r->end)return 0; memcpy(d,r->p,n); r->p+=n; return 1; }

typedef struct { int V,eot,nmerges; uint8_t**dec; int*declen; int base[256]; int*rank,*newid; int nsp,sp[64]; } Tok;
/* special tokens (tokenizer v4): vocab entries spelled <|name|>, never produced by BPE, matched literally */
static int tok_isspecial(const Tok*tk,int i){ int l=tk->declen[i]; const uint8_t*d=tk->dec[i];
    if(l<5||l>32||d[0]!='<'||d[1]!='|'||d[l-2]!='|'||d[l-1]!='>') return 0;
    for(int k=2;k<l-2;k++){ int c=d[k]; if(!((c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='_'||c=='/')) return 0; } return 1; }
static Tok* tok_parse(const uint8_t*data,size_t len,int full){
    Rd r={data,data+len}; char m[4]; if(!rd(&r,m,4)||memcmp(m,"TLTK",4)){fprintf(stderr,"bad tok\n");exit(1);}
    Tok*tk=(Tok*)calloc(1,sizeof(Tok)); rd(&r,&tk->V,4); rd(&r,&tk->eot,4); rd(&r,&tk->nmerges,4);
    tk->dec=(uint8_t**)malloc(sizeof(uint8_t*)*tk->V); tk->declen=(int*)malloc(sizeof(int)*tk->V);
    for(int i=0;i<tk->V;i++){ int l; rd(&r,&l,4); tk->declen[i]=l; tk->dec[i]=(uint8_t*)malloc(l>0?l:1); if(l>0)rd(&r,tk->dec[i],l); }
    rd(&r,tk->base,4*256);
    for(int i=0;i<tk->V&&tk->nsp<64;i++) if(i!=tk->eot&&tok_isspecial(tk,i)) tk->sp[tk->nsp++]=i;
    if(full){ tk->rank=(int*)malloc(sizeof(int)*(size_t)tk->V*tk->V); tk->newid=(int*)malloc(sizeof(int)*(size_t)tk->V*tk->V);
        for(size_t i=0;i<(size_t)tk->V*tk->V;i++){tk->rank[i]=-1;tk->newid[i]=-1;}
        for(int i=0;i<tk->nmerges;i++){ int a,b,nw; rd(&r,&a,4); rd(&r,&b,4); rd(&r,&nw,4);
            size_t idx=(size_t)a*tk->V+b; if(tk->rank[idx]<0){tk->rank[idx]=i;tk->newid[idx]=nw;} } }
    return tk;
}
/* exact training-time encoding (= data.cuh de_text / teachtok): specials and <|endoftext|> matched literally,
 * the rest GPT-2 ByteLevel pre-split, then BPE per piece. Whole-string BPE (the old path) diverged on whitespace runs. */
typedef void (*DPieceFn)(void*,const unsigned char*,int);
static void d_split(const unsigned char*s,int n,unsigned*cp,int*off,DPieceFn f,void*ctx);
typedef struct { const Tok*tk; int*ids; int cnt; } TEnc;
static void te_bpe(void*ctx,const unsigned char*s,int n){
    TEnc*e=(TEnc*)ctx; const Tok*tk=e->tk; int*w=e->ids+e->cnt,c=n;
    for(int i=0;i<n;i++){ int id=tk->base[s[i]]; if(id<0) id=tk->base[(int)' ']; w[i]=id; }
    for(;;){ int best=-1,bi=-1; for(int i=0;i+1<c;i++){ int rr=tk->rank[(size_t)w[i]*tk->V+w[i+1]]; if(rr>=0&&(best<0||rr<best)){ best=rr; bi=i; } }
        if(bi<0) break; w[bi]=tk->newid[(size_t)w[bi]*tk->V+w[bi+1]]; memmove(w+bi+1,w+bi+2,sizeof(int)*(c-bi-2)); c--; }
    e->cnt+=c; }
/* id of the special token spelled at q (or -1); *len = its byte length */
static int tok_special_at(const Tok*tk,const char*q,const char*end,int*len){
    if(end-q<5||q[0]!='<'||q[1]!='|') return -1;
    if(end-q>=13&&!memcmp(q,"<|endoftext|>",13)){ *len=13; return tk->eot; }
    for(int k=0;k<tk->nsp;k++){ int id=tk->sp[k],l=tk->declen[id]; if(end-q>=l&&!memcmp(q,tk->dec[id],l)){ *len=l; return id; } }
    return -1; }
static int* tok_encode(const Tok*tk,const char*s,int*out){
    int n=(int)strlen(s); TEnc e={tk,(int*)malloc(sizeof(int)*(n>0?n:1)),0};
    unsigned*cp=(unsigned*)malloc(4*(size_t)(n+1)); int*off=(int*)malloc(4*(size_t)(n+1));
    const char*p=s,*end=s+n;
    while(p<end){ const char*q=p; int sid=-1,sl=0;
        for(;q<end;q++) if((sid=tok_special_at(tk,q,end,&sl))>=0) break;
        if(q>p) d_split((const unsigned char*)p,(int)(q-p),cp,off,te_bpe,&e);
        if(sid>=0){ e.ids[e.cnt++]=sid; q+=sl; }
        p=q; }
    free(cp); free(off); *out=e.cnt; return e.ids; }
static void tok_print(const Tok*tk,int id){ if(id<0||id>=tk->V)return; fwrite(tk->dec[id],1,tk->declen[id],stdout); }

/* TLM5 adds the attention window. Written only when win>0, so full-attention
 * models stay TLM4 and remain readable by the CPU tinylm build. */
static int hdr_ints(const char*m,int*v){ if(!memcmp(m,"TLM7",4)){*v=7;return 14+3*64;}
    if(!memcmp(m,"TLM6",4)){*v=6;return 14;}
    if(!memcmp(m,"TLM5",4)){*v=5;return 13;}
    if(!memcmp(m,"TLM4",4)){*v=4;return 11;} if(!memcmp(m,"TLM3",4)){*v=3;return 10;}
    if(!memcmp(m,"TLM2",4)){*v=2;return 9;} if(!memcmp(m,"TLM1",4)){*v=1;return 8;} *v=0; return 0; }
static void cfg_from_hdr(Cfg*c,const int*h,int ver){
    memset(c,0,sizeof *c);   /* lp/lb/lmap (CUDALM_LOOP) are not in the header: stack garbage made info/load alias layers */
    c->V=h[0];c->D=h[1];c->L=h[2];c->H=h[3];c->KV=h[4];c->F=h[5];c->T=h[6];c->eot=h[7];
    c->tmpl=ver>=2?h[8]:0; c->n_mtp=ver>=3?h[9]:0; c->n_exp=ver>=4?h[10]:0;
    c->win=ver>=5?h[11]:0; c->full_every=ver>=5?h[12]:0;
    c->pool=ver>=6?h[13]:0; c->rope_base=10000.0f; c->mod=ver>=7;
    for(int l=0;l<64;l++){ c->lt[l]=c->mod?h[14+l]:0; c->lE[l]=c->mod?h[78+l]:0; c->lF[l]=c->mod?h[142+l]:0; }
    cfg_derive(c); }
/* defined below, next to the rest of the quantization code */
static int dequantize_params(Rd*r,float*P,size_t n,int qtype,int group);

static uint8_t*g_tokblob; static int g_tokblobl;     /* raw TLTK blob of the last model_load_full (data tokenizer-from) */
static void tok_keep(const uint8_t*p,int n){ free(g_tokblob); g_tokblob=(uint8_t*)malloc(n); memcpy(g_tokblob,p,n); g_tokblobl=n; }
static float* model_load_full(const char*path,Cfg*c,Tok**tk){
    size_t len; uint8_t*buf=read_file(path,&len); if(!buf){fprintf(stderr,"cannot open %s\n",path);exit(1);}
    Rd r={buf,buf+len}; char m[4]; rd(&r,m,4);
    /* Quantized containers: TLQ1 mirrors TLM4 (11 ints), TLQ2 mirrors TLM5 (13).
     * Weights are dequantized back to fp32 here, so everything downstream --
     * training, inference, the GPU kernels -- is unchanged. */
    int qz = !memcmp(m,"TLQ1",4) ? 1 : (!memcmp(m,"TLQ2",4) ? 2 : 0);
    if(qz){
        int nh = qz==1 ? 11 : 13, ver = qz==1 ? 4 : 5;
        int hdr[256]; if(!rd(&r,hdr,4*nh)){fprintf(stderr,"%s: bad quantized header\n",path);exit(1);}
        cfg_from_hdr(c,hdr,ver);
        if(getenv("CUDALM_LOOP")&&strcmp(getenv("CUDALM_LOOP"),"0")){ fprintf(stderr,"%s: CUDALM_LOOP needs an fp32 checkpoint\n",path); exit(1); }
        int qtype=0,group=0;
        if(!rd(&r,&qtype,4)||!rd(&r,&group,4)||group<=0){fprintf(stderr,"%s: bad quant params\n",path);exit(1);}
        size_t n=param_count(c); float*P=(float*)malloc(n*4);
        if(!dequantize_params(&r,P,n,qtype,group)){fprintf(stderr,"%s: quantized body truncated\n",path);exit(1);}
        *tk=NULL; { int tl=0; if(rd(&r,&tl,4)&&tl>0&&r.p+tl<=r.end){ *tk=tok_parse(r.p,tl,1); tok_keep(r.p,tl); } }
        fprintf(stderr,"[load] %s: %s checkpoint, dequantized to fp32\n",path,qtype==3?"q4n":qtype==2?"q4":(qtype?"q8":"fp16"));
        free(buf); return P;
    }
    int ver,nh=hdr_ints(m,&ver);
    /* must be >= the largest hdr_ints() return (TLM5 = 13) or the read overruns */
    if(!nh){fprintf(stderr,"bad model magic\n");exit(1);} int hdr[256]; rd(&r,hdr,4*nh);
    cfg_from_hdr(c,hdr,ver); loop_apply(c);
    size_t n=param_count(c); float*P=(float*)malloc(n*4);
    if(c->lp){   /* a model saved without the repeat-pass norms grows them, copied from the shared layer */
        size_t x=loop_xn(c), rem=r.end-r.p; int tl=-1; if(rem>=n*4+4) memcpy(&tl,r.p+n*4,4);
        if(tl!=(long long)(rem-n*4-4)){ if(!rd(&r,P,(n-x)*4)){fprintf(stderr,"model body truncated\n");exit(1);}
            Off o; calc_offsets(c,&o);
            for(int v=0;v<c->L;v++) if(lyA(c,v)){ memcpy(P+o.an1[v],P+o.an1[c->lmap[v]],c->D*4); memcpy(P+o.an2[v],P+o.an2[c->lmap[v]],c->D*4); }
            fprintf(stderr,"[loop] %s: %d physical layers -> depth %d, repeat-pass norms initialized from the shared layers\n",path,c->lp,c->L); }
        else if(!rd(&r,P,n*4)){fprintf(stderr,"model body truncated\n");exit(1);}
    } else if(!rd(&r,P,n*4)){fprintf(stderr,"model body truncated\n");exit(1);}
    *tk=NULL; if(ver>=2){ int tl=0; if(rd(&r,&tl,4)&&tl>0&&r.p+tl<=r.end){ *tk=tok_parse(r.p,tl,1); tok_keep(r.p,tl); } }
    free(buf); return P;
}
static const char* exe_dir(void){ static char d[1024]; static int done=0; if(done)return d;
    char p[1024]; size_t n;
#ifdef _WIN32
    DWORD w=GetModuleFileNameA(0,p,sizeof(p)); if(!w||w>=sizeof(p)){strcpy(d,".");done=1;return d;}
    n=w;
#else
    ssize_t w=readlink("/proc/self/exe",p,sizeof(p)-1); if(w<=0){strcpy(d,".");done=1;return d;}
    n=(size_t)w;
#endif
    p[n]=0; char*last=p; for(char*q=p;*q;q++) if(*q=='/'||*q=='\\')last=q; size_t L=last-p; if(!L||L>=sizeof(d)){strcpy(d,".");done=1;return d;}
    memcpy(d,p,L); d[L]=0; done=1; return d; }
static void model_path(const char*name,char*buf,size_t n){ char md[700]; snprintf(md,sizeof(md),"%.690s" PSEP "models",exe_dir()); lm_mkdir(md);
    snprintf(buf,n,"%.700s" PSEP "%.200s.bin",md,name); }
/* Durable, atomic save: write a .tmp, force it to physical disk, then rename
 * over the target. Two failures this protects against, both of which actually
 * happened: (1) fclose only flushes the CRT buffer to the OS -- an unclean
 * shutdown lost a 240 MB checkpoint that NTFS had already sized but never
 * committed, leaving a full-length file of zeros; (2) fopen(path,"wb") truncates
 * the previous GOOD checkpoint before writing the new one, so an interrupted
 * save destroys both. */
static FILE* save_open(const char*path,char*tmp,size_t tn){
    snprintf(tmp,tn,"%s.tmp",path);
    FILE*f=fopen(tmp,"wb"); if(!f){fprintf(stderr,"write %s\n",tmp);exit(1);}
    return f;
}
/* Checkpoint batch: between save_begin() and save_end() every save_commit only
 * flushes, fsyncs and closes its .tmp; save_end() renames them all only if every
 * file in the set was written, else deletes them. A full disk mid-save therefore
 * keeps the whole previous set instead of mixing iterations across files. */
#define SV_MAX 16
static int g_sv_batch=0, g_sv_fail=0, g_sv_n=0; static char g_sv_tmp[SV_MAX][1300], g_sv_dst[SV_MAX][1200];
static int save_rename(const char*tmp,const char*path);
static void save_begin(void){ g_sv_batch=1; g_sv_fail=0; g_sv_n=0; }
/* a writer that gives up on its own (failed fwrite it checked) */
static void save_abort(FILE*f,const char*tmp){ if(f) fclose(f); remove(tmp); if(g_sv_batch) g_sv_fail=1; }
static int save_end(void){
    int ok=!g_sv_fail; g_sv_batch=0;
    for(int i=0;i<g_sv_n;i++){
        if(!ok){ remove(g_sv_tmp[i]); continue; }
        if(!save_rename(g_sv_tmp[i],g_sv_dst[i])){ fprintf(stderr,"rename %s FAILED\n",g_sv_tmp[i]); ok=0; } }
    g_sv_n=0; return ok; }
static int save_commit(FILE*f,const char*tmp,const char*path){
    /* ferror: a large fwrite that hits a full disk can leave the stdio buffer empty,
     * after which fflush, fsync and fclose all succeed on a truncated file */
    int ok = !ferror(f) && fflush(f)==0 && !ferror(f) && lm_fsync(f)==0;   /* fsync: to the platter */
    if(fclose(f)!=0) ok=0;
    if(!ok){ remove(tmp); if(g_sv_batch) g_sv_fail=1; return 0; }
    if(g_sv_batch){
        if(g_sv_fail||g_sv_n>=SV_MAX){ remove(tmp); g_sv_fail=1; return 0; }
        snprintf(g_sv_tmp[g_sv_n],sizeof g_sv_tmp[0],"%s",tmp); snprintf(g_sv_dst[g_sv_n],sizeof g_sv_dst[0],"%s",path);
        g_sv_n++; return 1; }
    return save_rename(tmp,path);
}
static int save_rename(const char*tmp,const char*path){
#ifdef _WIN32
    /* MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH = atomic swap on NTFS */
    return MoveFileExA(tmp,path,MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)!=0;
#else
    /* rename() is atomic and replaces the target -- but on POSIX the DIRECTORY
     * ENTRY is not durable until the containing directory is itself fsynced.
     * Without that a power loss can lose the rename even though the file's data
     * reached the platter, which would quietly weaken this back to the bug that
     * cost 1000 iterations. Windows' WRITE_THROUGH covers the directory update;
     * POSIX makes it the caller's job. */
    if(rename(tmp,path)!=0) return 0;
    char dir[1200]; snprintf(dir,sizeof(dir),"%s",path);
    char*last=NULL; for(char*q=dir;*q;q++) if(*q=='/')last=q;
    if(last) *last=0; else strcpy(dir,".");
    int dfd=open(dir,O_RDONLY);
    if(dfd>=0){ fsync(dfd); close(dfd); }   /* best effort: the rename already stuck */
    return 1;
#endif
}
static void model_save_bundle(const char*path,const Cfg*c,const float*P,size_t n,const uint8_t*blob,int bl){
    char tmp[1200]; FILE*f=save_open(path,tmp,sizeof(tmp));
    Cfg pc; if(c->lp){ pc=*c; pc.L=c->lp; for(int v=c->L-1;v>=0;v--){ int l=c->lmap[v]; pc.lt[l]=c->lt[v]; pc.lE[l]=c->lE[v]; pc.lF[l]=c->lF[v]; } c=&pc; }
    int hdr[14+3*64]={c->V,c->D,c->L,c->H,c->KV,c->F,c->T,c->eot,c->tmpl,c->n_mtp,c->n_exp,c->win,c->full_every,c->pool};
    for(int l=0;l<64;l++){ hdr[14+l]=c->lt[l]; hdr[78+l]=c->lE[l]; hdr[142+l]=c->lF[l]; }
    if(c->mod){ fwrite("TLM7",1,4,f); fwrite(hdr,4,14+3*64,f); }   /* per-layer FFN layout: CUDA-only */
    else if(c->pool){ fwrite("TLM6",1,4,f); fwrite(hdr,4,14,f); }   /* UniPool: CUDA-only */
    else if(c->win>0){ fwrite("TLM5",1,4,f); fwrite(hdr,4,13,f); }
    else        { fwrite("TLM4",1,4,f); fwrite(hdr,4,11,f); }   /* stays CPU-readable */
    fwrite(P,4,n,f); fwrite(&bl,4,1,f); if(bl>0&&blob)fwrite(blob,1,bl,f);
    if(!save_commit(f,tmp,path)){ fprintf(stderr,"checkpoint save FAILED for %s (previous kept)\n",path); } }
/* TOP2 = bf16 moments (half the size of TOPT). TOPT is still READ, converting
 * on the way in, so an optimizer state written before this change still
 * resumes -- LS-GROWN's .opt is 1.77 GB of fp32 and should not become dead. */
static void opt_save(const char*path,int it,const bf16*M,const bf16*V,size_t n){
    char tmp[1200]; FILE*f=save_open(path,tmp,sizeof(tmp));
    fwrite("TOP2",1,4,f); fwrite(&it,4,1,f); fwrite(M,2,n,f); fwrite(V,2,n,f);
    if(!save_commit(f,tmp,path)) fprintf(stderr,"optimizer save FAILED for %s (previous kept)\n",path); }
static int opt_load(const char*path,bf16*M,bf16*V,size_t n){
    FILE*f=fopen(path,"rb"); if(!f)return 0; char m[4]; int it=0;
    if(fread(m,1,4,f)!=4){fclose(f);return 0;}
    int v2=!memcmp(m,"TOP2",4), v1=!memcmp(m,"TOPT",4);
    if(!v2&&!v1){fclose(f);return 0;}
    if(fread(&it,4,1,f)!=1){fclose(f);return 0;}
    if(v2){ size_t n0=n; long long c0=lm_ftell64(f); lm_fseek64(f,0,SEEK_END); long long e=lm_ftell64(f); lm_fseek64(f,c0,SEEK_SET);
        if(e>c0&&(size_t)((e-c0)/4)<n) n0=(size_t)((e-c0)/4);   /* a looped model grew per-pass norms: their moments stay 0 */
        if(fread(M,2,n0,f)!=n0||fread(V,2,n0,f)!=n0){fclose(f);return 0;} fclose(f);
        if(n0<n) printf("[opt] %s holds %zu of %zu params: the new ones start with zero moments\n",path,n0,n); return it; }
    /* legacy fp32: stream it in a chunk at a time rather than allocating a
     * second full-size host buffer next to the one the caller already holds */
    { const size_t CH=1u<<22; float*buf=(float*)malloc(CH*4);
      if(!buf){fclose(f);return 0;}
      for(int pass=0;pass<2;pass++){ bf16*dst=pass?V:M;
        for(size_t o=0;o<n;){ size_t k=n-o; if(k>CH)k=CH;
            if(fread(buf,4,k,f)!=k){ free(buf); fclose(f); return 0; }
            for(size_t j=0;j<k;j++) dst[o+j]=h_f2bf(buf[j]);
            o+=k; } }
      free(buf); }
    fclose(f);
    fprintf(stderr,"[opt] converted legacy fp32 optimizer state to bf16\n");
    return it; }
/* ===================== q8 / fp16 checkpoint quantization =================
 * Ported bit-for-bit from tinylm.c so the two builds produce byte-identical
 * files for the same input. This lives in cudalm rather than tinylm because
 * tinylm's `hdr_ints` only knows TLM1-4: quantizing a TLM5 (windowed) model
 * there would silently drop `win`/`full_every` and the result would run as FULL
 * attention -- wrong outputs, no error.
 *
 * Container mirrors the TLM4/TLM5 split exactly:
 *   TLQ1 = 11 header ints (win==0)  -- stays readable by the CPU build
 *   TLQ2 = 13 header ints (win>0)   -- carries win/full_every
 *
 * This shrinks the file on DISK only. Loading dequantizes back to fp32, so
 * run-time compute and quality are unchanged; q8 costs a little accuracy in the
 * stored weights and nothing else. */
static uint16_t f32_to_f16(float x){
    uint32_t f; memcpy(&f,&x,4);
    uint32_t sign=(f>>16)&0x8000; int32_t exp=(int32_t)((f>>23)&0xff)-112; uint32_t mant=f&0x7fffff;
    if(((f>>23)&0xff)==0xff) return (uint16_t)(sign|0x7c00|(mant?0x200:0));
    if(exp>=0x1f) return (uint16_t)(sign|0x7c00);
    if(exp<=0){
        if(exp<-10) return (uint16_t)sign;
        mant|=0x800000; int sh=14-exp; uint32_t h=mant>>sh;
        uint32_t rem=mant&(((uint32_t)1<<sh)-1), half=(uint32_t)1<<(sh-1);
        if(rem>half||(rem==half&&(h&1))) h++;
        return (uint16_t)(sign|h);
    }
    uint16_t h=(uint16_t)(sign|((uint32_t)exp<<10)|(mant>>13));
    uint32_t rem=mant&0x1fff;
    if(rem>0x1000||(rem==0x1000&&(h&1))) h++;      /* round-to-even */
    return h;
}
static float f16_to_f32(uint16_t h){
    uint32_t sign=(uint32_t)(h&0x8000)<<16, exp=(h>>10)&0x1f, mant=h&0x3ff, f;
    if(exp==0){
        if(mant==0) f=sign;
        else{ exp=113; while(!(mant&0x400)){mant<<=1;exp--;} mant&=0x3ff; f=sign|(exp<<23)|(mant<<13); }
    } else if(exp==0x1f){ f=sign|0x7f800000|(mant<<13); }
    else f=sign|((exp+112)<<23)|(mant<<13);
    float out; memcpy(&out,&f,4); return out;
}
/* qtype 0=fp16 (2B/weight), 1=q8 (fp16 scale + int8 per `group` weights),
 * 2=q4 (fp16 scale + packed signed nibbles -7..7 per `group`, group must be even) */
static uint8_t* quantize_params(const float*P,size_t n,int qtype,int group,size_t*out_len){
    if(qtype==3){  /* q4n: exact QLM export. fp32 scale + 64 nibble bytes per 128; NaN scale = raw fp32 group */
        size_t nb=(n+group-1)/group; uint8_t*q=(uint8_t*)malloc(nb*4+n*4+1); size_t o=0,nraw=0;
        for(size_t i=0;i<n;i+=group){
            int g=(i+(size_t)group<=n)?group:(int)(n-i), ok=0; float amax=0,s=0;
            for(int j=0;j<g;j++){ float a=fabsf(P[i+j]); if(a>amax)amax=a; }
            if(amax==0) ok=1;
            for(int k=7;k>=1&&!ok;k--){ float s0=amax/k, c[5]={s0,nextafterf(s0,0),nextafterf(s0,INFINITY),0,0};
                c[3]=nextafterf(c[1],0); c[4]=nextafterf(c[2],INFINITY);
                for(int t=0;t<5&&!ok;t++){ s=c[t]; ok=1;
                    for(int j=0;j<g&&ok;j++){ float d=rintf(P[i+j]/s); if(fabsf(d)>7||d*s!=P[i+j]) ok=0; } } }
            if(!ok){ float nan=NAN; memcpy(q+o,&nan,4); o+=4; memcpy(q+o,P+i,4*(size_t)g); o+=4*(size_t)g; nraw++; continue; }
            memcpy(q+o,&s,4); o+=4;
            for(int j=0;j<g;j+=2){ int a=s?(int)rintf(P[i+j]/s):0, b=(j+1<g&&s)?(int)rintf(P[i+j+1]/s):0;
                q[o++]=(uint8_t)((a&15)|((b&15)<<4)); }
        }
        fprintf(stderr,"[q4n] %zu/%zu groups raw fp32 (not on a 4-bit grid)\n",nraw,nb);
        *out_len=o; return q;
    }
    if(qtype==0){
        uint16_t*q=(uint16_t*)malloc(n*2>0?n*2:2);
        for(size_t i=0;i<n;i++) q[i]=f32_to_f16(P[i]);
        *out_len=n*2; return (uint8_t*)q;
    }
    if(qtype==2){
        size_t nb=(n+group-1)/group; uint8_t*q=(uint8_t*)malloc(nb*2+n/2+nb+1); size_t o=0;
        for(size_t i=0;i<n;i+=group){
            int g=(i+(size_t)group<=n)?group:(int)(n-i);
            float amax=0; for(int j=0;j<g;j++){ float a=fabsf(P[i+j]); if(a>amax)amax=a; }
            float scale=amax>0?amax/7.0f:1.0f;
            uint16_t sh=f32_to_f16(scale); memcpy(q+o,&sh,2); o+=2; scale=f16_to_f32(sh);
            for(int j=0;j<g;j+=2){
                int a=(int)lrintf(P[i+j]/scale); if(a>7)a=7; if(a<-7)a=-7;
                int b=0; if(j+1<g){ b=(int)lrintf(P[i+j+1]/scale); if(b>7)b=7; if(b<-7)b=-7; }
                q[o++]=(uint8_t)((a&15)|((b&15)<<4)); }  /* two's-complement nibbles = tinylm.c qtype 2 */
        }
        *out_len=o; return q;
    }
    size_t nb=(n+group-1)/group; uint8_t*q=(uint8_t*)malloc(nb*2+n+1); size_t o=0;
    for(size_t i=0;i<n;i+=group){
        int g=(i+(size_t)group<=n)?group:(int)(n-i);
        float amax=0; for(int j=0;j<g;j++){ float a=fabsf(P[i+j]); if(a>amax)amax=a; }
        float scale=amax>0?amax/127.0f:1.0f;
        uint16_t sh=f32_to_f16(scale); memcpy(q+o,&sh,2); o+=2;
        for(int j=0;j<g;j++){ int v=(int)lrintf(P[i+j]/scale); if(v>127)v=127; if(v<-127)v=-127;
            q[o++]=(uint8_t)(int8_t)v; }
    }
    *out_len=o; return q;
}
static int dequantize_params(Rd*r,float*P,size_t n,int qtype,int group){
    if(qtype==3){
        for(size_t i=0;i<n;i+=group){
            int g=(i+(size_t)group<=n)?group:(int)(n-i); float s; if(!rd(r,&s,4)) return 0;
            if(s!=s){ if(!rd(r,P+i,4*(size_t)g)) return 0; continue; }
            for(int j=0;j<g;j+=2){ uint8_t b; if(!rd(r,&b,1)) return 0;
                P[i+j]=(float)((int)((b&15)^8)-8)*s; if(j+1<g) P[i+j+1]=(float)((int)((b>>4)^8)-8)*s; }
        }
        return 1;
    }
    if(qtype==0){
        for(size_t i=0;i<n;i++){ uint16_t h; if(!rd(r,&h,2)) return 0; P[i]=f16_to_f32(h); }
        return 1;
    }
    if(qtype==2){
        for(size_t i=0;i<n;i+=group){
            int g=(i+(size_t)group<=n)?group:(int)(n-i);
            uint16_t sh; if(!rd(r,&sh,2)) return 0; float scale=f16_to_f32(sh);
            for(int j=0;j<g;j+=2){ uint8_t b; if(!rd(r,&b,1)) return 0;
                P[i+j]=(float)((int)((b&15)^8)-8)*scale; if(j+1<g) P[i+j+1]=(float)((int)((b>>4)^8)-8)*scale; }
        }
        return 1;
    }
    for(size_t i=0;i<n;i+=group){
        int g=(i+(size_t)group<=n)?group:(int)(n-i);
        uint16_t sh; if(!rd(r,&sh,2)) return 0; float scale=f16_to_f32(sh);
        for(int j=0;j<g;j++){ int8_t b; if(!rd(r,&b,1)) return 0; P[i+j]=(float)b*scale; }
    }
    return 1;
}

static uint16_t* load_u16(const char*p,size_t*cnt){ size_t s; uint8_t*b=read_file(p,&s); if(!b){fprintf(stderr,"open %s\n",p);exit(1);} *cnt=s/2; return (uint16_t*)b; }

static uint64_t g_rng=88172645463325252ULL;
static float urand(void){ uint64_t x=g_rng; x^=x<<13;x^=x>>7;x^=x<<17;g_rng=x; return (float)((x>>11)*(1.0/9007199254740992.0)); }
/* Full-width draw for picking a window start. `(size_t)(urand()*4294967296.0)`
 * has two defects that the corpus growth made worse: urand() is a FLOAT, so
 * only ~2^24 distinct values exist and starts quantize to multiples of ~256;
 * and 2^32 mod ntr is badly non-uniform once ntr is a large fraction of 2^32 --
 * at 2.01B tokens the first 13.6% of the corpus draws 3 chances against 2 for
 * the rest, a 1.5x oversampling of whatever source happens to be written first.
 * Used for TRAINING windows only. cmd_eval deliberately keeps the old draw, or
 * every val loss ever recorded would stop being comparable. */
static uint64_t urand64(void){ uint64_t x=g_rng; x^=x<<13;x^=x>>7;x^=x<<17;g_rng=x; return x; }
static float rnd_normal(void){ float u1=urand(); if(u1<1e-9f)u1=1e-9f; float u2=urand(); return sqrtf(-2.0f*logf(u1))*cosf(6.28318530718f*u2); }

/* ===================== CUDA kernels ====================================== */
#define TPB 256
static inline int nblk(size_t n,int t){ return (int)((n+t-1)/t); }

/* ---------------- TDR / driver-reset precautions -------------------------
 * On Windows WDDM a kernel that runs longer than the TDR watchdog (default
 * 2 s) makes the driver reset the GPU: every later call fails and the device
 * stays dead until reboot. That is exactly how the first attention-backward
 * took this card down. Three defences, in order of importance:
 *   1. bounded work per launch  - attention backward is chunked (ATTN_CHUNK)
 *      so launch duration does not grow with T. This is the real fix.
 *   2. preflight probe          - time one fwd+bwd before the training loop
 *      and refuse to start if any kernel is near the watchdog budget.
 *   3. per-kernel timing        - CUDALM_DEBUG=1 reports the slowest kernel
 *      so a regression is visible before it is fatal.
 * ------------------------------------------------------------------------ */
/* a save that runs out of disk truncates the last good checkpoint */
#ifdef _WIN32
static int disk_low(const char*p){ char r[4]={p[0],':','\\',0}; ULARGE_INTEGER f;
    return p[1]==':'&&GetDiskFreeSpaceExA(r,&f,0,0)&&f.QuadPart<(3ull<<30); }
#else
static int disk_low(const char*p){ char d[1200]; snprintf(d,sizeof d,"%s",p);
    char*s=strrchr(d,'/'); if(s) *s=0; else snprintf(d,sizeof d,".");
    /* a filesystem that reports no capacity (some FUSE/pseudo mounts) counts as unknown, not full */
    struct statvfs v; if(statvfs(d[0]?d:"/",&v)!=0 || !v.f_blocks || !v.f_frsize) return 0;
    return (unsigned long long)v.f_bavail*v.f_frsize<(3ull<<30); }
#endif
static double wall(void){
#ifdef _WIN32
    LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart/(double)f.QuadPart;
#else
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec + (double)t.tv_nsec*1e-9;
#endif
}

/* Windows: read-only registry probe; key absent => the 2 s default applies.
 * Linux: there is no WDDM watchdog at all, so report a large budget. That makes
 * preflight's "% of TDR budget" check a no-op rather than a false alarm -- the
 * chunking it guards stays in place regardless, since it is also what keeps
 * per-launch memory traffic sane. */
static double tdr_budget_s(void){
#ifdef _WIN32
    DWORD v=0,sz=sizeof(v);
    if(RegGetValueA(HKEY_LOCAL_MACHINE,"SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers",
                    "TdrDelay",RRF_RT_REG_DWORD,NULL,&v,&sz)==ERROR_SUCCESS && v>0) return (double)v;
    return 2.0;
#else
    return 1e9;
#endif
}
static double g_tdr=2.0;

static int g_debug=0;
static double g_tprev=0, g_maxms=0;
static char   g_maxwhat[96]="(none)";

/* CUDALM_PROF=1 accumulates per-kernel wall time by label (summed across layers)
 * so optimisation effort can be aimed at what actually costs, rather than at
 * whatever the FLOP count suggests -- these kernels are memory-bound and the
 * two disagree badly. */
#define PROF_MAX 48
typedef struct { char name[40]; double ms; int n, drain; } ProfEnt;   /* named: nvcc is C++ */
static int g_prof=0, g_prof_n=0;
static ProfEnt g_prof_tab[PROF_MAX];
static int prof_idx(const char*what){
    for(int i=0;i<g_prof_n;i++) if(!strcmp(g_prof_tab[i].name,what)) return i;
    if(g_prof_n>=PROF_MAX) return PROF_MAX-1;
    memset(&g_prof_tab[g_prof_n],0,sizeof(ProfEnt)); snprintf(g_prof_tab[g_prof_n].name,40,"%s",what);
    return g_prof_n++;
}
static void prof_add(const char*what,double ms){ ProfEnt*e=&g_prof_tab[prof_idx(what)]; e->ms+=ms; e->n++; }
/* CUDALM_PROF>=3: no sync. Each label gets the GPU time since the previous label's
 * event; "drained" counts labels where the GPU had already finished all queued work
 * when the host got there, i.e. the host was the bottleneck at that point. */
#define PEV_MAX 4096
static cudaEvent_t g_pev[PEV_MAX]; static short g_pev_lab[PEV_MAX]; static int g_pev_n=0;
static void prof_flush(void){
    if(g_pev_n<2) return;
    cudaEventSynchronize(g_pev[g_pev_n-1]);
    for(int i=1;i<g_pev_n;i++){ float ms=0; cudaEventElapsedTime(&ms,g_pev[i-1],g_pev[i]);
        g_prof_tab[g_pev_lab[i]].ms+=ms; g_prof_tab[g_pev_lab[i]].n++; }
    cudaEvent_t t=g_pev[0]; g_pev[0]=g_pev[g_pev_n-1]; g_pev[g_pev_n-1]=t; g_pev_n=1;
}
static void prof_ev(const char*what){
    static int init=0;
    if(!init){ for(int i=0;i<PEV_MAX;i++) cudaEventCreate(&g_pev[i]); init=1; }
    int k=prof_idx(what);
    if(g_pev_n && cudaEventQuery(g_pev[g_pev_n-1])==cudaSuccess) g_prof_tab[k].drain++;
    cudaEventRecord(g_pev[g_pev_n],0); g_pev_lab[g_pev_n++]=(short)k;
    if(g_pev_n==PEV_MAX) prof_flush();
}
static void prof_report(void){
    if(g_prof>=3) prof_flush();
    if(!g_prof||!g_prof_n) return;
    double tot=0; for(int i=0;i<g_prof_n;i++) tot+=g_prof_tab[i].ms;
    for(int i=0;i<g_prof_n;i++) for(int j=i+1;j<g_prof_n;j++)
        if(g_prof_tab[j].ms>g_prof_tab[i].ms){
            ProfEnt t=g_prof_tab[i]; g_prof_tab[i]=g_prof_tab[j]; g_prof_tab[j]=t; }
    printf("\n[prof] per-kernel totals over the run (%.0f ms tracked%s)\n",tot,g_prof>=3?", async GPU time":"");
    for(int i=0;i<g_prof_n;i++){
        printf("  %-22s %9.1f ms  %5.1f%%   x%d", g_prof_tab[i].name,
               g_prof_tab[i].ms, 100.0*g_prof_tab[i].ms/tot, g_prof_tab[i].n);
        if(g_prof>=3) printf("   drained %d",g_prof_tab[i].drain);
        printf("\n"); }
    fflush(stdout);
}

static void kchk(const char*what,int l){
    if(!g_debug && !g_prof) return;
    if(g_prof>=3 && !g_debug){ prof_ev(what); return; }
    cudaError_t e=cudaDeviceSynchronize();
    if(e==cudaSuccess) e=cudaGetLastError();
    if(e!=cudaSuccess){ fprintf(stderr,"\n[FAULT] after %s (layer %d): %s\n",what,l,cudaGetErrorString(e)); exit(1); }
    double now=wall(), ms=(now-g_tprev)*1000.0; g_tprev=now;
    if(g_prof) prof_add(what,ms);            /* summed across layers */
    if(ms>g_maxms){ g_maxms=ms; snprintf(g_maxwhat,sizeof(g_maxwhat),"%s (layer %d)",what,l); }
    /* loud warning well before the watchdog actually fires */
    if(g_debug && ms > g_tdr*1000.0*0.25)
        fprintf(stderr,"[WARN] %s (layer %d) took %.0f ms - %.0f%% of the %.0fs TDR budget\n",
                what,l,ms,100.0*ms/(g_tdr*1000.0),g_tdr);
}

#ifndef CUDALM_CPU
__global__ void k_embed(const float*P,size_t eo,const int*tok,float*X,int N,int D){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(size_t)N*D)return;
    int n=(int)(i/D), d=(int)(i%D); X[i]=P[eo+(size_t)tok[n]*D+d];
}
__global__ void k_embed_bwd(const int*tok,const float*dout,float*G,size_t eo,int N,int D){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(size_t)N*D)return;
    int n=(int)(i/D), d=(int)(i%D); atomicAdd(&G[eo+(size_t)tok[n]*D+d], dout[i]);
}
/* rmsnorm forward: one block per row, warp-shuffle reduction */
__global__ void k_rmsnorm(const float*X,const float*P,size_t go,float*O,float*rinv,int N,int D){
    int n=blockIdx.x; if(n>=N)return;
    const float*x=X+(size_t)n*D; float*o=O+(size_t)n*D;
    __shared__ float sh[32]; float ss=0;
    for(int i=threadIdx.x;i<D;i+=blockDim.x) ss+=x[i]*x[i];
    for(int off=16;off>0;off>>=1) ss+=__shfl_down_sync(0xffffffff,ss,off);
    if((threadIdx.x&31)==0) sh[threadIdx.x>>5]=ss;
    __syncthreads();
    if(threadIdx.x==0){ float t=0; int nw=(blockDim.x+31)/32; for(int i=0;i<nw;i++)t+=sh[i];
        sh[0]=1.0f/sqrtf(t/D+1e-5f); rinv[n]=sh[0]; }
    __syncthreads();
    float s=sh[0];
    for(int i=threadIdx.x;i<D;i+=blockDim.x) o[i]=x[i]*s*P[go+i];
}
/* rmsnorm backward: dx += ... (one block per row) */
__global__ void k_rmsnorm_bwd(const float*X,const float*P,size_t go,const float*rinv,
                              const float*dy,float*dx,int N,int D){
    int n=blockIdx.x; if(n>=N)return;
    const float*x=X+(size_t)n*D,*d=dy+(size_t)n*D; float*dxr=dx+(size_t)n*D;
    float s=rinv[n]; __shared__ float sh[32]; float c=0;
    for(int i=threadIdx.x;i<D;i+=blockDim.x) c+=x[i]*P[go+i]*d[i];
    for(int off=16;off>0;off>>=1) c+=__shfl_down_sync(0xffffffff,c,off);
    if((threadIdx.x&31)==0) sh[threadIdx.x>>5]=c;
    __syncthreads();
    if(threadIdx.x==0){ float t=0; int nw=(blockDim.x+31)/32; for(int i=0;i<nw;i++)t+=sh[i]; sh[0]=s*s*s/D*t; }
    __syncthreads();
    float coef=sh[0];
    for(int i=threadIdx.x;i<D;i+=blockDim.x) dxr[i]+=s*P[go+i]*d[i]-coef*x[i];
}
/* warp-per-row variants (D%4==0, 16B-aligned rows): a 256-thread block per
 * 384-wide row left most lanes idle; here 8 rows share a block, float4 I/O */
__global__ void k_rmsnorm_w(const float*X,const float*P,size_t go,float*O,float*rinv,int N,int D){
    int n=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5), ln=threadIdx.x&31, D4=D>>2; if(n>=N)return;
    const float4*x=(const float4*)(X+(size_t)n*D); float4*o=(float4*)(O+(size_t)n*D); const float*g=P+go; float ss=0;
    for(int i=ln;i<D4;i+=32){ float4 v=x[i]; ss+=v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w; }
    for(int k=16;k;k>>=1) ss+=__shfl_xor_sync(~0u,ss,k);
    float s=1.0f/sqrtf(ss/D+1e-5f); if(!ln) rinv[n]=s;
    for(int i=ln;i<D4;i+=32){ float4 v=x[i]; const float*gg=g+4*i;
        o[i]=make_float4(v.x*s*gg[0],v.y*s*gg[1],v.z*s*gg[2],v.w*s*gg[3]); }
}
__global__ void k_rmsnorm_bwd_w(const float*X,const float*P,size_t go,const float*rinv,
                                const float*dy,float*dx,int N,int D){
    int n=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5), ln=threadIdx.x&31, D4=D>>2; if(n>=N)return;
    const float4*x=(const float4*)(X+(size_t)n*D),*d=(const float4*)(dy+(size_t)n*D); float4*r=(float4*)(dx+(size_t)n*D);
    const float*g=P+go; float s=rinv[n],c=0;
    for(int i=ln;i<D4;i+=32){ float4 v=x[i],e=d[i]; const float*gg=g+4*i;
        c+=v.x*gg[0]*e.x+v.y*gg[1]*e.y+v.z*gg[2]*e.z+v.w*gg[3]*e.w; }
    for(int k=16;k;k>>=1) c+=__shfl_xor_sync(~0u,c,k);
    float cf=s*s*s/D*c;
    for(int i=ln;i<D4;i+=32){ float4 v=x[i],e=d[i],o=r[i]; const float*gg=g+4*i;
        o.x+=s*gg[0]*e.x-cf*v.x; o.y+=s*gg[1]*e.y-cf*v.y; o.z+=s*gg[2]*e.z-cf*v.z; o.w+=s*gg[3]*e.w-cf*v.w; r[i]=o; }
}
/* dgamma: column sum of x*rinv*dy, one thread per dim (coalesced across d) */
__global__ void k_gamma_grad(const float*X,const float*rinv,const float*dy,float*G,size_t ggo,int N,int D){
    int d=blockIdx.x*blockDim.x+threadIdx.x; if(d>=D)return; float acc=0;
    for(int n=0;n<N;n++) acc+=X[(size_t)n*D+d]*rinv[n]*dy[(size_t)n*D+d];
    G[ggo+d]+=acc;
}
/* Row-chunked dgamma: the one-thread-per-dim form above puts only D threads on
 * the whole GPU, each walking all N rows serially (~2 ms/call at N=32k). Chunk
 * the rows into GG_CH partials, then sum them in fixed order -- no atomics, so
 * the result stays run-to-run deterministic. */
#define GG_CH 128
__global__ void k_gamma_part(const float*X,const float*rinv,const float*dy,float*part,int N,int D){
    int d=blockIdx.x*blockDim.x+threadIdx.x; if(d>=D)return;
    int c=blockIdx.y, r0=(int)((long long)N*c/GG_CH), r1=(int)((long long)N*(c+1)/GG_CH); float acc=0;
    for(int n=r0;n<r1;n++) acc+=X[(size_t)n*D+d]*rinv[n]*dy[(size_t)n*D+d];
    part[(size_t)c*D+d]=acc;
}
__global__ void k_gamma_sum(const float*part,float*G,size_t ggo,int D){
    int d=blockIdx.x*blockDim.x+threadIdx.x; if(d>=D)return; float acc=0;
    for(int c=0;c<GG_CH;c++) acc+=part[(size_t)c*D+d];
    G[ggo+d]+=acc;
}
__global__ void k_rope(float*buf,const float*cosb,const float*sinb,int N,int nh,int hd,int hf,int T,int back){
    size_t gid=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(gid>=(size_t)N*nh*hf) return;
    int i=(int)(gid%hf); size_t r=gid/hf; int h=(int)(r%nh); int n=(int)(r/nh); int t=n%T;
    float*v=buf+(size_t)n*nh*hd+(size_t)h*hd;
    float c=cosb[t*hf+i],s=sinb[t*hf+i],a=v[i],b=v[i+hf];
    if(back){ v[i]=a*c+b*s; v[i+hf]=-a*s+b*c; } else { v[i]=a*c-b*s; v[i+hf]=b*c+a*s; }
}
/* Attention kernels: ONE WARP per output slot, not one thread.
 *
 * The thread-per-slot version (ported from gpu.c) put a float[64] accumulator
 * in local memory and ran a long serial loop; k_attn_bwd_dkv did grp*(T-s)
 * steps per thread and eventually blew past the 2 s WDDM TDR watchdog, which
 * resets the GPU. Per-warp instead: each lane owns dims d = lane, lane+32
 * (so hd <= 64), accumulators are registers, and the q.k / dA.v dots become
 * __shfl_xor butterfly reductions that leave the total in every lane. */
#define WARP 32
/* Tiling: one block handles QPB consecutive queries from the SAME (b,h), so a
 * staged K/V tile in shared memory is reused by all QPB warps instead of each
 * warp re-reading it from global. Profiling showed attention at 78.7% of the
 * step, bound on global traffic once the working set outgrew this card's 1.5MB
 * L2 -- this attacks that directly. */
#define QPB   8                 /* warps (= queries) per block */
#define KTILE 32                /* keys staged per shared-memory tile */

/* causal GQA attention, online softmax. Saves lse so backward need not re-reduce. */
__global__ void k_attn_fwd(const float*Q,const float*K,const float*V,float*O,float*lse,
                           int win,int B,int T,int H,int KV,int hd,int D,int KD){
    int gid=blockIdx.x; if(gid>=B*H*T)return;
    int lane=threadIdx.x, d0=lane, d1=lane+WARP;
    int t=gid%T,h=(gid/T)%H,b=gid/(T*H); int kvh=h/(H/KV);
    float scale=1.0f/sqrtf((float)hd);
    const float*q=Q+(size_t)(b*T+t)*D+h*hd;
    float q0=(d0<hd)?q[d0]:0.f, q1=(d1<hd)?q[d1]:0.f;
    float a0=0.f,a1=0.f,m=-1e30f,l=0.f;
    int s_lo=(win>0 && t>=win)?t-win+1:0;         /* sliding window */
    for(int s=s_lo;s<=t;s++){
        const float*k=K+(size_t)(b*T+s)*KD+kvh*hd;
        float dot=(d0<hd?q0*k[d0]:0.f)+(d1<hd?q1*k[d1]:0.f);
        for(int off=16;off>0;off>>=1) dot+=__shfl_xor_sync(0xffffffff,dot,off);
        dot*=scale;
        float nm=fmaxf(m,dot),corr=expf(m-nm),p=expf(dot-nm);
        l=l*corr+p; const float*vv=V+(size_t)(b*T+s)*KD+kvh*hd;
        a0=a0*corr+(d0<hd?p*vv[d0]:0.f);
        a1=a1*corr+(d1<hd?p*vv[d1]:0.f);
        m=nm;
    }
    float inv=1.0f/l; float*o=O+(size_t)(b*T+t)*D+h*hd;
    if(d0<hd)o[d0]=a0*inv; if(d1<hd)o[d1]=a1*inv;
    if(lane==0) lse[(size_t)(b*H+h)*T+t]=m+logf(l);
}
/* ---- sub-warp mapping ---------------------------------------------------
 * The warp-per-query layout spends a 5-step __shfl_xor reduction on every key
 * while each lane contributes only 2 FMAs (hd=64 across 32 lanes) -- a long
 * dependent chain for almost no work. Here a query is owned by SUBL lanes
 * instead of 32, so each lane does hd/SUBL FMAs and the reduction is
 * log2(SUBL) steps. At SUBL=8: 8 FMAs and 3 steps, i.e. 4x the useful work per
 * shuffle, plus 4 queries in flight per warp instead of 1.
 * Dims are interleaved (lane owns d = sl, sl+SUBL, ...) so the SUBL lanes of a
 * group read consecutive addresses and stay coalesced. */
#define SUBL 8
#define EPLMAX 8                       /* hd/SUBL, hd<=64 */
__device__ __forceinline__ unsigned submask(){
    return ((1u<<SUBL)-1) << (SUBL * ((threadIdx.x & (WARP-1)) / SUBL));
}
__global__ void k_attn_fwd_sub(const float*Q,const float*K,const float*V,float*O,float*lse,
                               int win,int B,int T,int H,int KV,int hd,int D,int KD){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int qidx=tid/SUBL, sl=tid%SUBL;
    if(qidx>=B*H*T) return;            /* whole group exits together */
    int t=qidx%T,h=(qidx/T)%H,b=qidx/(T*H); int kvh=h/(H/KV);
    int EPL=hd/SUBL;
    /* group-local mask: other groups in this warp may have exited or be at a
     * different trip count, so a full-warp mask would be wrong here */
    unsigned gm=submask();
    float scale=1.0f/sqrtf((float)hd);
    const float*q=Q+(size_t)(b*T+t)*D+h*hd;
    float qr[EPLMAX],ac[EPLMAX];
    for(int i=0;i<EPL;i++){ qr[i]=q[sl+i*SUBL]; ac[i]=0.f; }
    float m=-1e30f,l=0.f;
    int s_lo=(win>0 && t>=win)?t-win+1:0;
    for(int s=s_lo;s<=t;s++){
        const float*k=K+(size_t)(b*T+s)*KD+kvh*hd;
        float dot=0; for(int i=0;i<EPL;i++) dot+=qr[i]*k[sl+i*SUBL];
        for(int off=SUBL/2;off>0;off>>=1) dot+=__shfl_xor_sync(gm,dot,off);
        dot*=scale;
        float nm=fmaxf(m,dot),corr=expf(m-nm),p=expf(dot-nm);
        l=l*corr+p;
        const float*v=V+(size_t)(b*T+s)*KD+kvh*hd;
        for(int i=0;i<EPL;i++) ac[i]=ac[i]*corr+p*v[sl+i*SUBL];
        m=nm;
    }
    float inv=1.0f/l; float*o=O+(size_t)(b*T+t)*D+h*hd;
    for(int i=0;i<EPL;i++) o[sl+i*SUBL]=ac[i]*inv;
    if(sl==0) lse[(size_t)(b*H+h)*T+t]=m+logf(l);
}

/* Tiled forward. Requires T % QPB == 0 so a block never straddles a (b,h)
 * boundary -- that is what lets every warp in the block share one K/V tile. */
__global__ void k_attn_fwd_tiled(const float*Q,const float*K,const float*V,float*O,float*lse,
                                 int win,int B,int T,int H,int KV,int hd,int D,int KD){
    extern __shared__ float sm[];
    float *sK=sm, *sV=sm+KTILE*hd;
    int warp=threadIdx.x>>5, lane=threadIdx.x&31;
    int base=blockIdx.x*QPB;
    int t0=base%T, h=(base/T)%H, b=base/(T*H);
    int kvh=h/(H/KV), t=t0+warp;
    float scale=1.0f/sqrtf((float)hd);
    int d0=lane, d1=lane+WARP;
    const float*q=Q+(size_t)(b*T+t)*D+h*hd;
    float q0=(d0<hd)?q[d0]:0.f, q1=(d1<hd)?q[d1]:0.f;
    float a0=0.f,a1=0.f,m=-1e30f,l=0.f;
    int hiS=t0+QPB-1;                       /* furthest key any warp here needs */
    int loS=(win>0)?(t0-win+1):0; if(loS<0) loS=0;
    for(int s0=loS; s0<=hiS; s0+=KTILE){
        int n=KTILE; if(s0+n>T) n=T-s0;
        for(int i=threadIdx.x;i<KTILE*hd;i+=blockDim.x){
            int r=i/hd, d=i%hd, s=s0+r; float kv=0.f,vv=0.f;
            if(r<n){ kv=K[(size_t)(b*T+s)*KD+kvh*hd+d]; vv=V[(size_t)(b*T+s)*KD+kvh*hd+d]; }
            sK[i]=kv; sV[i]=vv;
        }
        __syncthreads();
        for(int r=0;r<n;r++){
            int s=s0+r;
            if(s>t) break;                              /* causal */
            if(win>0 && s<=t-win) continue;             /* outside the window */
            float dot=(d0<hd?q0*sK[r*hd+d0]:0.f)+(d1<hd?q1*sK[r*hd+d1]:0.f);
            for(int off=16;off>0;off>>=1) dot+=__shfl_xor_sync(0xffffffff,dot,off);
            dot*=scale;
            float nm=fmaxf(m,dot),corr=expf(m-nm),p=expf(dot-nm);
            l=l*corr+p;
            a0=a0*corr+(d0<hd?p*sV[r*hd+d0]:0.f);
            a1=a1*corr+(d1<hd?p*sV[r*hd+d1]:0.f);
            m=nm;
        }
        __syncthreads();
    }
    float inv=1.0f/l; float*o=O+(size_t)(b*T+t)*D+h*hd;
    if(d0<hd)o[d0]=a0*inv; if(d1<hd)o[d1]=a1*inv;
    if(lane==0) lse[(size_t)(b*H+h)*T+t]=m+logf(l);
}

/* attn bwd 1/2: per (b,h,t). Writes the softmax row-dot Drow and dQ. Atomic-free:
 * this thread uniquely owns query slot (b,t,h). */
/* Backward is split into three chunked passes so no launch scales with T:
 *   dr  : row-dot  sum_s p*(dA.v), accumulated over s-chunks   (needs all s)
 *   dQ  : accumulated over s-chunks, consumes the finished dr
 *   dKV : accumulated over t-chunks, consumes the finished dr
 * Each warp still owns its own output slot across every chunk, so all three
 * stay atomic-free; the buffers are zeroed once before the chunk loops. */
/* fp16 attention backward: dA is scaled by DSCALE on the cast (see k_f2h below). */
#ifndef DSCALE
#define DSCALE     1024.0f
#endif
#define DSCALE_INV (1.0f/DSCALE)
/* DSCALE is only the ceiling: before each layer's cast the scale is lowered (powers of 2) until
 * max|dA|*scale <= 1, so large attention grads (LS-Micro, norm gains ~0.003) cannot overflow fp16
 * into NaN dQ/dK. Normal models (max|dA| << 1e-3) keep exactly 1024. */
__device__ float g_dsc[2]={DSCALE,DSCALE_INV}; __device__ unsigned g_dsc_amax;
__global__ void k_dsc_amax(const float*x,size_t n){
    float m=0.f; for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x) m=fmaxf(m,fabsf(x[i]));
    for(int o=16;o>0;o>>=1) m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,o));
    if((threadIdx.x&31)==0) atomicMax(&g_dsc_amax,__float_as_uint(m)); }
__global__ void k_dsc_set(void){ float m=__uint_as_float(g_dsc_amax), s=DSCALE; g_dsc_amax=0;
    while(s>1.0f/1024 && m*s>1.0f) s*=0.5f; g_dsc[0]=s; g_dsc[1]=1.0f/s; }
/* Drow without touching a single key.
 *
 * k_attn_bwd_dr below computes  dr_t = sum_s a_s (dA_t . v_s)  by re-walking the
 * whole window, which measured 17.9% of the training step. But the softmax
 * weights are exactly the ones the forward already applied:
 *     sum_s a_s (dA_t . v_s) = dA_t . (sum_s a_s v_s) = dA_t . O_t
 * so it collapses to one dot product per (b,h,t) against the attention output
 * that forward already saved. O(hd) instead of O(window), and no K/V traffic.
 * This is the standard flash-attention "D" term. k_attn_bwd_dr is kept as the
 * independent implementation selftest cross-checks this against.
 *
 * `s` folds in DSCALE so dr carries the same factor as the dp computed from a
 * scaled fp16 dA in the dQ/dKV kernels. */
__global__ void k_attn_drow(const float*dA,const float*O,float*Drow,
                            int B,int T,int H,int hd,int D,float s){
    int gid=blockIdx.x; if(gid>=B*H*T)return;
    int lane=threadIdx.x;
    int t=gid%T,h=(gid/T)%H,b=gid/(T*H);
    const float*da=dA+(size_t)(b*T+t)*D+h*hd, *o=O+(size_t)(b*T+t)*D+h*hd;
    float acc=0.f;
    for(int d=lane;d<hd;d+=WARP) acc+=da[d]*o[d];
    for(int off=16;off>0;off>>=1) acc+=__shfl_xor_sync(0xffffffff,acc,off);
    if(lane==0) Drow[(size_t)(b*H+h)*T+t]=acc*(s<0.f?g_dsc[0]:s);   /* s<0: the adaptive scale */
}
__global__ void k_attn_bwd_dr(const float*Q,const float*K,const float*V,const float*dA,
                              const float*lse,float*Drow,int s0,int s1,int win,
                              int B,int T,int H,int KV,int hd,int D,int KD){
    int gid=blockIdx.x; if(gid>=B*H*T)return;
    int lane=threadIdx.x, d0=lane, d1=lane+WARP;
    int t=gid%T,h=(gid/T)%H,b=gid/(T*H); int kvh=h/(H/KV);
    /* With a window, [s0,s1) are WINDOW OFFSETS (s = t-w), not absolute times:
     * chunking over absolute s at T=8192 dispatches every warp for every chunk
     * and ~85% of them immediately early-return. Offset chunking keeps every
     * warp busy and cuts launches from T/CH to win/CH. */
    int lo,hi;
    if(win>0){ lo=t-(s1-1); hi=t-s0; if(hi<0) return; if(lo<0) lo=0; }
    else     { lo=s0; hi=(t<s1-1)?t:(s1-1); if(lo>hi) return; }
    float scale=1.0f/sqrtf((float)hd);
    const float*q=Q+(size_t)(b*T+t)*D+h*hd;
    const float*da=dA+(size_t)(b*T+t)*D+h*hd;
    float L=lse[(size_t)(b*H+h)*T+t];
    float q0=(d0<hd)?q[d0]:0.f, q1=(d1<hd)?q[d1]:0.f;
    float g0=(d0<hd)?da[d0]:0.f, g1=(d1<hd)?da[d1]:0.f;
    float dr=0.f;
    for(int s=lo;s<=hi;s++){
        const float*k=K+(size_t)(b*T+s)*KD+kvh*hd,*v=V+(size_t)(b*T+s)*KD+kvh*hd;
        float dot=(d0<hd?q0*k[d0]:0.f)+(d1<hd?q1*k[d1]:0.f);
        float dp =(d0<hd?g0*v[d0]:0.f)+(d1<hd?g1*v[d1]:0.f);
        for(int off=16;off>0;off>>=1){ dot+=__shfl_xor_sync(0xffffffff,dot,off);
                                       dp +=__shfl_xor_sync(0xffffffff,dp ,off); }
        dr+=expf(dot*scale-L)*dp;
    }
    if(lane==0) Drow[(size_t)(b*H+h)*T+t]+=dr;
}
__global__ void k_attn_bwd_dq(const float*Q,const float*K,const float*V,const float*dA,
                              const float*lse,float*dQ,const float*Drow,int s0,int s1,int win,
                              int B,int T,int H,int KV,int hd,int D,int KD){
    int gid=blockIdx.x; if(gid>=B*H*T)return;
    int lane=threadIdx.x, d0=lane, d1=lane+WARP;
    int t=gid%T,h=(gid/T)%H,b=gid/(T*H); int kvh=h/(H/KV);
    int lo,hi;                                       /* see k_attn_bwd_dr */
    if(win>0){ lo=t-(s1-1); hi=t-s0; if(hi<0) return; if(lo<0) lo=0; }
    else     { lo=s0; hi=(t<s1-1)?t:(s1-1); if(lo>hi) return; }
    float scale=1.0f/sqrtf((float)hd);
    const float*q=Q+(size_t)(b*T+t)*D+h*hd;
    const float*da=dA+(size_t)(b*T+t)*D+h*hd;
    float L=lse[(size_t)(b*H+h)*T+t], dr=Drow[(size_t)(b*H+h)*T+t];
    float q0=(d0<hd)?q[d0]:0.f, q1=(d1<hd)?q[d1]:0.f;
    float g0=(d0<hd)?da[d0]:0.f, g1=(d1<hd)?da[d1]:0.f;
    float a0=0.f,a1=0.f;
    for(int s=lo;s<=hi;s++){
        const float*k=K+(size_t)(b*T+s)*KD+kvh*hd,*v=V+(size_t)(b*T+s)*KD+kvh*hd;
        float dot=(d0<hd?q0*k[d0]:0.f)+(d1<hd?q1*k[d1]:0.f);
        float dp =(d0<hd?g0*v[d0]:0.f)+(d1<hd?g1*v[d1]:0.f);
        for(int off=16;off>0;off>>=1){ dot+=__shfl_xor_sync(0xffffffff,dot,off);
                                       dp +=__shfl_xor_sync(0xffffffff,dp ,off); }
        float p=expf(dot*scale-L), ds=p*(dp-dr)*scale;
        if(d0<hd)a0+=ds*k[d0]; if(d1<hd)a1+=ds*k[d1];
    }
    float*dq=dQ+(size_t)(b*T+t)*D+h*hd;
    if(d0<hd)dq[d0]+=a0; if(d1<hd)dq[d1]+=a1;
}
/* attn bwd 2/2: per (b,kvh,s). Owns key/value slot (b,s,kvh) so dK/dV are
 * atomic-free too. Uses the saved lse + Drow, so no row re-softmax. */
__global__ void k_attn_bwd_dkv(const float*Q,const float*K,const float*V,const float*dA,
                               const float*lse,const float*Drow,float*dK,float*dV,int t0,int t1,int win,
                               int B,int T,int H,int KV,int hd,int D,int KD){
    int gid=blockIdx.x; if(gid>=B*KV*T)return;
    int lane=threadIdx.x, d0=lane, d1=lane+WARP;
    int s=gid%T,kvh=(gid/T)%KV,b=gid/(T*KV); int grp=H/KV;
    /* query t attends key s iff s <= t <= s+win-1. With a window, [t0,t1) are
     * WINDOW OFFSETS (t = s+w) so every warp has work in every chunk. */
    int lo,hi;
    if(win>0){ lo=s+t0; hi=s+t1; if(hi>s+win) hi=s+win; if(hi>T) hi=T; }
    else     { lo=(s>t0)?s:t0; hi=t1; }
    if(lo>=hi) return;
    float scale=1.0f/sqrtf((float)hd);
    const float*k=K+(size_t)(b*T+s)*KD+kvh*hd,*v=V+(size_t)(b*T+s)*KD+kvh*hd;
    float k0=(d0<hd)?k[d0]:0.f, k1=(d1<hd)?k[d1]:0.f;
    float v0=(d0<hd)?v[d0]:0.f, v1=(d1<hd)?v[d1]:0.f;
    float dk0=0.f,dk1=0.f,dv0=0.f,dv1=0.f;
    for(int hh=0;hh<grp;hh++){ int h=kvh*grp+hh;
        for(int t=lo;t<hi;t++){
            const float*q=Q+(size_t)(b*T+t)*D+h*hd;
            const float*da=dA+(size_t)(b*T+t)*D+h*hd;
            float L=lse[(size_t)(b*H+h)*T+t], dr=Drow[(size_t)(b*H+h)*T+t];
            float q0=(d0<hd)?q[d0]:0.f, q1=(d1<hd)?q[d1]:0.f;
            float g0=(d0<hd)?da[d0]:0.f, g1=(d1<hd)?da[d1]:0.f;
            float dot=q0*k0+q1*k1, dp=g0*v0+g1*v1;
            for(int off=16;off>0;off>>=1){ dot+=__shfl_xor_sync(0xffffffff,dot,off);
                                           dp +=__shfl_xor_sync(0xffffffff,dp ,off); }
            float p=expf(dot*scale-L), ds=p*(dp-dr)*scale;
            dv0+=p*g0; dv1+=p*g1; dk0+=ds*q0; dk1+=ds*q1;
        } }
    float*dkp=dK+(size_t)(b*T+s)*KD+kvh*hd,*dvp=dV+(size_t)(b*T+s)*KD+kvh*hd;
    if(d0<hd){ dkp[d0]+=dk0; dvp[d0]+=dv0; }
    if(d1<hd){ dkp[d1]+=dk1; dvp[d1]+=dv1; }
}

/* ===================== fp16 attention ====================================
 * Q/K/V/dA are re-read O(window) times per query, so these kernels are bound by
 * memory transactions, not FLOPs -- 6 TFLOP/s of fp32 sits idle waiting on a
 * 336 GB/s bus. Storing the operands as __half and loading them as __half2
 * halves both the bytes moved and the transaction count: one 128-byte
 * transaction per warp per 64-dim row instead of two.
 *
 * This is mixed precision, not fp16 training. Only the four attention OPERANDS
 * are rounded to half; every accumulator, the softmax, the log-sum-exp, the
 * weights, the gradients and AdamW all stay fp32. Q/K/V are post-RoPE
 * activations of order 1, comfortably inside fp16's range.
 *
 * dA is the one operand that can underflow (fp16's smallest normal is 6.1e-5),
 * so it is scaled by DSCALE on the cast. Every output of the backward is exactly
 * linear in dA -- dr, ds, dQ, dK and dV all carry one factor of it -- so the
 * scale is undone by a single multiply at the final write, with no extra launch.
 *
 * sm_75 has no tensor cores; the gain here is bandwidth, not math throughput. */

__global__ void k_f2h(const float*src,__half*dst,size_t n,float s){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) dst[i]=__float2half(src[i]*(s<0.f?g_dsc[0]:s));
}
/* forward RoPE straight into the fp16 attention copy; wb: also write the fp32 back */
__global__ void k_rope_h(float*buf,__half*dst,const float*cosb,const float*sinb,int N,int nh,int hd,int hf,int T,int wb){
    size_t gid=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(gid>=(size_t)N*nh*hf) return;
    int i=(int)(gid%hf); size_t r=gid/hf; int h=(int)(r%nh); int n=(int)(r/nh); int t=n%T;
    size_t o=(size_t)n*nh*hd+(size_t)h*hd;
    float c=cosb[t*hf+i],s=sinb[t*hf+i],a=buf[o+i],b=buf[o+i+hf], x=a*c-b*s, y=b*c+a*s;
    dst[o+i]=__float2half(x); dst[o+i+hf]=__float2half(y);
    if(wb){ buf[o+i]=x; buf[o+i+hf]=y; }
}
/* lane owns dims (2*lane, 2*lane+1). hd is even and the row offsets are
 * multiples of hd, so every __half2 access is 4-byte aligned. */
__device__ __forceinline__ float2 ldh2(const __half*base,int lane,int h2){
    if(lane>=h2) return make_float2(0.f,0.f);
    return __half22float2(((const __half2*)base)[lane]);
}
__global__ void k_attn_fwd_h(const __half*Q,const __half*K,const __half*V,float*O,float*lse,
                             int win,int B,int T,int H,int KV,int hd,int D,int KD){
    int gid=blockIdx.x; if(gid>=B*H*T)return;
    int lane=threadIdx.x, h2=hd>>1;
    int t=gid%T,h=(gid/T)%H,b=gid/(T*H); int kvh=h/(H/KV);
    float scale=1.0f/sqrtf((float)hd);
    float2 qv=ldh2(Q+(size_t)(b*T+t)*D+h*hd,lane,h2);
    float a0=0.f,a1=0.f,m=-1e30f,l=0.f;
    int s_lo=(win>0 && t>=win)?t-win+1:0;
    for(int s=s_lo;s<=t;s++){
        float2 kv2=ldh2(K+(size_t)(b*T+s)*KD+kvh*hd,lane,h2);
        float dot=qv.x*kv2.x+qv.y*kv2.y;
        for(int off=16;off>0;off>>=1) dot+=__shfl_xor_sync(0xffffffff,dot,off);
        dot*=scale;
        float nm=fmaxf(m,dot),corr=expf(m-nm),p=expf(dot-nm);
        l=l*corr+p;
        float2 vv2=ldh2(V+(size_t)(b*T+s)*KD+kvh*hd,lane,h2);
        a0=a0*corr+p*vv2.x; a1=a1*corr+p*vv2.y;
        m=nm;
    }
    float inv=1.0f/l; float*o=O+(size_t)(b*T+t)*D+h*hd;
    if(lane<h2){ o[2*lane]=a0*inv; o[2*lane+1]=a1*inv; }
    if(lane==0) lse[(size_t)(b*H+h)*T+t]=m+logf(l);
}
__global__ void k_attn_bwd_dq_h(const __half*Q,const __half*K,const __half*V,const __half*dA,
                                const float*lse,float*dQ,const float*Drow,int s0,int s1,int win,
                                int B,int T,int H,int KV,int hd,int D,int KD){
    int gid=blockIdx.x; if(gid>=B*H*T)return;
    int lane=threadIdx.x, h2=hd>>1;
    int t=gid%T,h=(gid/T)%H,b=gid/(T*H); int kvh=h/(H/KV);
    int lo,hi;
    if(win>0){ lo=t-(s1-1); hi=t-s0; if(hi<0) return; if(lo<0) lo=0; }
    else     { lo=s0; hi=(t<s1-1)?t:(s1-1); if(lo>hi) return; }
    float scale=1.0f/sqrtf((float)hd);
    float L=lse[(size_t)(b*H+h)*T+t], dr=Drow[(size_t)(b*H+h)*T+t];
    float2 qv=ldh2(Q +(size_t)(b*T+t)*D+h*hd,lane,h2);
    float2 gv=ldh2(dA+(size_t)(b*T+t)*D+h*hd,lane,h2);
    float a0=0.f,a1=0.f;
    for(int s=lo;s<=hi;s++){
        float2 kv2=ldh2(K+(size_t)(b*T+s)*KD+kvh*hd,lane,h2);
        float2 vv2=ldh2(V+(size_t)(b*T+s)*KD+kvh*hd,lane,h2);
        float dot=qv.x*kv2.x+qv.y*kv2.y;
        float dp =gv.x*vv2.x+gv.y*vv2.y;
        for(int off=16;off>0;off>>=1){ dot+=__shfl_xor_sync(0xffffffff,dot,off);
                                       dp +=__shfl_xor_sync(0xffffffff,dp ,off); }
        float p=expf(dot*scale-L), ds=p*(dp-dr)*scale;
        a0+=ds*kv2.x; a1+=ds*kv2.y;
    }
    float*dq=dQ+(size_t)(b*T+t)*D+h*hd;
    if(lane<h2){ dq[2*lane]+=a0*g_dsc[1]; dq[2*lane+1]+=a1*g_dsc[1]; }
}
__global__ void k_attn_bwd_dkv_h(const __half*Q,const __half*K,const __half*V,const __half*dA,
                                 const float*lse,const float*Drow,float*dK,float*dV,int t0,int t1,int win,
                                 int B,int T,int H,int KV,int hd,int D,int KD){
    int gid=blockIdx.x; if(gid>=B*KV*T)return;
    int lane=threadIdx.x, h2=hd>>1;
    int s=gid%T,kvh=(gid/T)%KV,b=gid/(T*KV); int grp=H/KV;
    int lo,hi;                                       /* see k_attn_bwd_dkv */
    if(win>0){ lo=s+t0; hi=s+t1; if(hi>s+win) hi=s+win; if(hi>T) hi=T; }
    else     { lo=(s>t0)?s:t0; hi=t1; }
    if(lo>=hi) return;
    float scale=1.0f/sqrtf((float)hd);
    float2 kv2=ldh2(K+(size_t)(b*T+s)*KD+kvh*hd,lane,h2);
    float2 vv2=ldh2(V+(size_t)(b*T+s)*KD+kvh*hd,lane,h2);
    float dk0=0.f,dk1=0.f,dv0=0.f,dv1=0.f;
    for(int hh=0;hh<grp;hh++){ int h=kvh*grp+hh;
        for(int t=lo;t<hi;t++){
            float L=lse[(size_t)(b*H+h)*T+t], dr=Drow[(size_t)(b*H+h)*T+t];
            float2 qv=ldh2(Q +(size_t)(b*T+t)*D+h*hd,lane,h2);
            float2 gv=ldh2(dA+(size_t)(b*T+t)*D+h*hd,lane,h2);
            float dot=qv.x*kv2.x+qv.y*kv2.y, dp=gv.x*vv2.x+gv.y*vv2.y;
            for(int off=16;off>0;off>>=1){ dot+=__shfl_xor_sync(0xffffffff,dot,off);
                                           dp +=__shfl_xor_sync(0xffffffff,dp ,off); }
            float p=expf(dot*scale-L), ds=p*(dp-dr)*scale;
            dv0+=p*gv.x; dv1+=p*gv.y; dk0+=ds*qv.x; dk1+=ds*qv.y;
        } }
    float*dkp=dK+(size_t)(b*T+s)*KD+kvh*hd,*dvp=dV+(size_t)(b*T+s)*KD+kvh*hd;
    if(lane<h2){ dkp[2*lane]+=dk0*g_dsc[1]; dkp[2*lane+1]+=dk1*g_dsc[1];
                 dvp[2*lane]+=dv0*g_dsc[1]; dvp[2*lane+1]+=dv1*g_dsc[1]; }
}

/* ---- lane-per-query flash tile -------------------------------------------
 * Every attention kernel above spends five __shfl_xor steps and two expf per
 * (query,key) pair -- roughly 26 warp-instructions to cover 64 FMAs of useful
 * work. That instruction chain, not bandwidth, is what binds them: fp16
 * operands halved both the bytes AND the transaction count and bought 4%.
 *
 * Here a THREAD owns a query. q and the output accumulator live in registers,
 * and K/V are read from a shared tile that every thread in the block walks in
 * the same order -- so each shared read is a broadcast, and there is no
 * cross-lane reduction anywhere. A warp retires 32 (query,key) pairs per pass
 * of the dim loop, and the two expf now cover 32 pairs instead of one.
 *
 * HD is a template parameter on purpose. qr[]/acc[] must be indexed by fully
 * unrolled loops or nvcc spills them to LOCAL memory -- which is exactly the
 * mistake that took this GPU down the first time. Check `-Xptxas -v` for
 * "spill stores" after touching anything here.
 *
 * The trade is occupancy: 2*HD float registers per thread means few resident
 * warps, so latency has to be hidden by ILP inside the dim loop (four
 * independent partial sums) instead. */
#define LQ_TPB 128        /* threads (= queries) per block */
#define LQ_KT  32         /* keys staged per shared tile */

/* Dot product over HD/2 __half2 pairs, returning an fp32 total. Two forms:
 *
 *  PACKED   32x __hfma2 into an fp16 accumulator. This is the ONLY form that
 *           reaches Pascal's 2x fp16 rate: GP100 gets fp16 throughput by running
 *           two halves per FP32 slot, so scalar __half math is NOT faster, and
 *           there is no fp16-multiply/fp32-accumulate path on Pascal at all
 *           (that arrived with tensor cores on Volta). The accumulator is
 *           therefore fp16 and the precision cost is real -- roughly
 *           sqrt(HD/2)*2^-11 relative, ~0.3% at HD=64.
 *  UNPACKED unpack to float2, accumulate in fp32. Exact, and the same speed as
 *           fp32 everywhere.
 *
 * There is no free lunch here, which is why this is a measured flag
 * (CUDALM_HFMA2=1) and not the default. Only the SHORT (HD-length) dots can use
 * it -- the output accumulators run over the whole window and stay fp32
 * unconditionally. */
template<int HD,bool PACKED>
__device__ __forceinline__ float dot_h2(const __half2*a,const __half2*b){
    if(PACKED){
        __half2 s=__floats2half2_rn(0.f,0.f);
        #pragma unroll
        for(int d=0;d<HD/2;d++) s=__hfma2(a[d],b[d],s);
        float2 f=__half22float2(s); return f.x+f.y;
    } else {
        float x=0,y=0;
        #pragma unroll
        for(int d=0;d<HD/2;d++){ float2 u=__half22float2(a[d]),v=__half22float2(b[d]);
            x+=u.x*v.x; y+=u.y*v.y; }
        return x+y;
    }
}

template<int HD>
__global__ __launch_bounds__(LQ_TPB,2)
void k_attn_fwd_lane(const __half*Q,const __half*K,const __half*V,float*O,float*lse,
                     int win,int B,int T,int H,int KV,int D,int KD){
    __shared__ __half sK[LQ_KT*HD], sV[LQ_KT*HD];
    int bh=blockIdx.y, h=bh%H, b=bh/H, kvh=h/(H/KV);
    int t0=blockIdx.x*LQ_TPB, t=t0+threadIdx.x;
    int act=(t<T);                     /* inactive threads still stage and sync */
    float scale=rsqrtf((float)HD);
    float qr[HD],acc[HD];
    const __half*qp=Q+(size_t)(b*T+(act?t:0))*D+h*HD;
    #pragma unroll
    for(int d=0;d<HD;d++){ qr[d]=act?__half2float(qp[d]):0.f; acc[d]=0.f; }
    float m=-1e30f,l=0.f;
    int hiS=t0+LQ_TPB-1; if(hiS>T-1) hiS=T-1;      /* last key any thread here needs */
    int loS=(win>0)?(t0-win+1):0; if(loS<0) loS=0;
    for(int s0=loS;s0<=hiS;s0+=LQ_KT){
        int n=LQ_KT; if(s0+n>T) n=T-s0;
        #pragma unroll 1
        for(int i=threadIdx.x;i<LQ_KT*HD;i+=LQ_TPB){
            int r=i/HD, d=i-r*HD, s=s0+r; __half kk=__float2half(0.f),vv=kk;
            if(r<n){ kk=K[(size_t)(b*T+s)*KD+kvh*HD+d]; vv=V[(size_t)(b*T+s)*KD+kvh*HD+d]; }
            sK[i]=kk; sV[i]=vv;
        }
        __syncthreads();
        if(act) for(int r=0;r<n;r++){
            int s=s0+r;
            if(s>t) break;                          /* causal */
            if(win>0 && s<=t-win) continue;         /* outside this query's window */
            const __half*kr=sK+r*HD,*vr=sV+r*HD;
            float a0=0,a1=0,a2=0,a3=0;
            #pragma unroll
            for(int d=0;d<HD;d+=4){
                a0+=qr[d  ]*__half2float(kr[d  ]);
                a1+=qr[d+1]*__half2float(kr[d+1]);
                a2+=qr[d+2]*__half2float(kr[d+2]);
                a3+=qr[d+3]*__half2float(kr[d+3]);
            }
            float dot=((a0+a1)+(a2+a3))*scale;
            float nm=fmaxf(m,dot),corr=expf(m-nm),p=expf(dot-nm);
            l=l*corr+p;
            #pragma unroll
            for(int d=0;d<HD;d++) acc[d]=acc[d]*corr+p*__half2float(vr[d]);
            m=nm;
        }
        __syncthreads();
    }
    if(!act) return;
    float inv=1.0f/l; float*o=O+(size_t)(b*T+t)*D+h*HD;
    #pragma unroll
    for(int d=0;d<HD;d++) o[d]=acc[d]*inv;
    lse[(size_t)(b*H+h)*T+t]=m+logf(l);
}
/* Same mapping for dQ. Three HD-sized register arrays instead of two (q, dA and
 * the dQ accumulator), so this is the register-hungriest of the set -- watch
 * `-Xptxas -v`. It also covers the whole window in ONE launch: per-launch work
 * is bounded by the window, not by T, and at W=1024 that is ~65 ms against a
 * 60 s watchdog, so the chunking the warp path needs buys nothing here. */
template<int HD,bool PACKED>
__global__ __launch_bounds__(LQ_TPB,2)
void k_attn_bwd_dq_lane(const __half*Q,const __half*K,const __half*V,const __half*dA,
                        const float*lse,float*dQ,const float*Drow,int win,
                        int B,int T,int H,int KV,int D,int KD){
    __shared__ __align__(8) __half sK[LQ_KT*HD], sV[LQ_KT*HD];
    int bh=blockIdx.y, h=bh%H, b=bh/H, kvh=h/(H/KV);
    int t0=blockIdx.x*LQ_TPB, t=t0+threadIdx.x;
    int act=(t<T);
    float scale=rsqrtf((float)HD);
    /* q and dA stay PACKED as half2. They arrive as fp16 anyway so this costs no
     * precision, and it is what keeps three HD-sized arrays under the 255-register
     * ceiling -- held as float, this kernel spilled to local memory. */
    __half2 qr[HD/2],gr[HD/2]; float acc[HD];
    const __half2*qp=(const __half2*)(Q +(size_t)(b*T+(act?t:0))*D+h*HD);
    const __half2*gp=(const __half2*)(dA+(size_t)(b*T+(act?t:0))*D+h*HD);
    __half2 z=__floats2half2_rn(0.f,0.f);
    #pragma unroll
    for(int d=0;d<HD/2;d++){ qr[d]=act?qp[d]:z; gr[d]=act?gp[d]:z; }
    #pragma unroll
    for(int d=0;d<HD;d++) acc[d]=0.f;
    float L=0.f,dr=0.f;
    if(act){ L=lse[(size_t)(b*H+h)*T+t]; dr=Drow[(size_t)(b*H+h)*T+t]; }
    int hiS=t0+LQ_TPB-1; if(hiS>T-1) hiS=T-1;
    int loS=(win>0)?(t0-win+1):0; if(loS<0) loS=0;
    for(int s0=loS;s0<=hiS;s0+=LQ_KT){
        int n=LQ_KT; if(s0+n>T) n=T-s0;
        #pragma unroll 1
        for(int i=threadIdx.x;i<LQ_KT*HD;i+=LQ_TPB){
            int r=i/HD, d=i-r*HD, s=s0+r; __half kk=__float2half(0.f),vv=kk;
            if(r<n){ kk=K[(size_t)(b*T+s)*KD+kvh*HD+d]; vv=V[(size_t)(b*T+s)*KD+kvh*HD+d]; }
            sK[i]=kk; sV[i]=vv;
        }
        __syncthreads();
        if(act) for(int r=0;r<n;r++){
            int s=s0+r;
            if(s>t) break;
            if(win>0 && s<=t-win) continue;
            const __half2*kr=(const __half2*)(sK+r*HD),*vr=(const __half2*)(sV+r*HD);
            /* the two dot products are deliberately NOT fused: one unrolled loop
             * over four arrays makes ptxas hold ~128 loaded values live at once,
             * which pushed this past 255 registers and into local memory */
            float qk=dot_h2<HD,PACKED>(qr,kr);
            float gv=dot_h2<HD,PACKED>(gr,vr);
            float p=expf(qk*scale-L), ds=p*(gv-dr)*scale;
            #pragma unroll
            for(int d=0;d<HD/2;d++){
                float2 k2=__half22float2(kr[d]);
                acc[2*d]+=ds*k2.x; acc[2*d+1]+=ds*k2.y;
            }
        }
        __syncthreads();
    }
    if(!act) return;
    float*dq=dQ+(size_t)(b*T+t)*D+h*HD;
    #pragma unroll
    for(int d=0;d<HD;d++) dq[d]+=acc[d]*g_dsc[1];
}
/* dK/dV under the same mapping, with the roles swapped: a thread owns a KEY and
 * holds k, v and both accumulators, while q and dA become the broadcast operands
 * staged in shared. Four HD-sized arrays is over the register budget even with
 * k/v packed, so this one spills a little -- measured anyway, because the dQ
 * kernel spilled by a similar amount and was still 46% faster. */
template<int HD,bool PACKED>
__global__ __launch_bounds__(LQ_TPB,2)
void k_attn_bwd_dkv_lane(const __half*Q,const __half*K,const __half*V,const __half*dA,
                         const float*lse,const float*Drow,float*dK,float*dV,int win,
                         int B,int T,int H,int KV,int D,int KD){
    __shared__ __align__(8) __half sQ[LQ_KT*HD], sG[LQ_KT*HD];
    int bkv=blockIdx.y, kvh=bkv%KV, b=bkv/KV, grp=H/KV;
    int sblk=blockIdx.x*LQ_TPB, s=sblk+threadIdx.x;
    int act=(s<T);
    float scale=rsqrtf((float)HD);
    __half2 kr[HD/2],vr[HD/2]; float dk[HD],dv[HD];
    const __half2*kp=(const __half2*)(K+(size_t)(b*T+(act?s:0))*KD+kvh*HD);
    const __half2*vp=(const __half2*)(V+(size_t)(b*T+(act?s:0))*KD+kvh*HD);
    __half2 z=__floats2half2_rn(0.f,0.f);
    #pragma unroll
    for(int d=0;d<HD/2;d++){ kr[d]=act?kp[d]:z; vr[d]=act?vp[d]:z; }
    #pragma unroll
    for(int d=0;d<HD;d++){ dk[d]=0.f; dv[d]=0.f; }
    /* query t attends key s iff s <= t <= s+win-1, so this block spans
     * [sblk, sblk+LQ_TPB-1+win-1] and each thread masks its own slice */
    int loT=sblk, hiT=sblk+LQ_TPB-1+(win>0?win-1:T); if(hiT>T-1) hiT=T-1;
    for(int hh=0;hh<grp;hh++){ int h=kvh*grp+hh;
        for(int t0=loT;t0<=hiT;t0+=LQ_KT){
            int n=LQ_KT; if(t0+n>T) n=T-t0;
            __syncthreads();                 /* previous tile's reads must be done */
            #pragma unroll 1
            for(int i=threadIdx.x;i<LQ_KT*HD;i+=LQ_TPB){
                int r=i/HD, d=i-r*HD, t=t0+r; __half qq=__float2half(0.f),gg=qq;
                if(r<n){ qq=Q [(size_t)(b*T+t)*D+h*HD+d]; gg=dA[(size_t)(b*T+t)*D+h*HD+d]; }
                sQ[i]=qq; sG[i]=gg;
            }
            __syncthreads();
            if(act) for(int r=0;r<n;r++){
                int t=t0+r;
                if(t<s) continue;                     /* causal */
                if(win>0 && t>s+win-1) break;         /* past this key's reach */
                const __half2*qr=(const __half2*)(sQ+r*HD),*gr=(const __half2*)(sG+r*HD);
                float qk=dot_h2<HD,PACKED>(qr,kr);
                float gv=dot_h2<HD,PACKED>(gr,vr);
                float L=lse[(size_t)(b*H+h)*T+t], dr=Drow[(size_t)(b*H+h)*T+t];
                float p=expf(qk*scale-L), ds=p*(gv-dr)*scale;
                #pragma unroll
                for(int d=0;d<HD/2;d++){
                    float2 q2=__half22float2(qr[d]), g2=__half22float2(gr[d]);
                    dk[2*d]+=ds*q2.x; dk[2*d+1]+=ds*q2.y;
                    dv[2*d]+=p *g2.x; dv[2*d+1]+=p *g2.y;
                }
            }
        } }
    if(!act) return;
    float*dkp=dK+(size_t)(b*T+s)*KD+kvh*HD,*dvp=dV+(size_t)(b*T+s)*KD+kvh*HD;
    #pragma unroll
    for(int d=0;d<HD;d++){ dkp[d]+=dk[d]*g_dsc[1]; dvp[d]+=dv[d]*g_dsc[1]; }
}
/* Tensor-core dKV (FA2-style). One block owns TC_BK=64 keys of one (b,kvh);
 * each of the four warps owns 16 of them and keeps its K/V rows as wmma A
 * fragments for the whole kernel. Per 32-query tile:
 *   S = K.Q^T, dP = V.dA^T      (wmma, fp16 in, fp32 acc)
 *   P = exp(S*scale-L), dS = P*(dP-Drow)*scale   (fp32, masked, then fp16)
 *   dV += P.dA, dK += dS.Q      (wmma, fp32 accumulators in registers)
 * Every warp only touches its own 16 rows of the S/P scratch, so only the
 * query-tile staging needs block syncs. dS carries an extra x TC_DSS so the
 * fp16 rounding stays clear of the denormal floor; undone at the dK write.
 * Replaces the one-thread-per-key lane kernel (255 regs + spill). */
#define TC_BK 64
#define TC_BQ 32
#define TC_DSS 16.0f
/* CUDALM_DOCMASK: per-token document start (position of the first token after the
 * last eot before it); keys before it are masked. NULL = plain causal. */
__device__ const int* d_dstart=NULL;
__global__ void k_docstart(const int*tok,int*ds,int B,int T,int eot){
    int b=blockIdx.x*blockDim.x+threadIdx.x; if(b>=B) return;
    const int*tk=tok+(size_t)b*T; int*o=ds+(size_t)b*T, st=0;
    for(int t=0;t<T;t++){ o[t]=st; if(tk[t]==eot) st=t+1; }
}
template<int HD>
__global__ __launch_bounds__(128)
void k_attn_bwd_dkv_tc(const __half*Q,const __half*K,const __half*V,const __half*dA,
                       const float*lse,const float*Drow,float*dK,float*dV,int win,
                       int B,int T,int H,int KV,int D,int KD){
    using namespace nvcuda;
    static_assert(HD%16==0 && HD<=64, "tc dKV: HD in {16,32,64}");
    constexpr int NF=HD/16, NQ=TC_BQ/16;
    __shared__ __align__(32) __half sQ[TC_BQ*HD], sG[TC_BQ*HD];
    __shared__ __align__(32) float  sF[2*TC_BK*TC_BQ];   /* S | dP; K/V staging; output */
    __shared__ __align__(32) __half sP[TC_BK*TC_BQ], sX[TC_BK*TC_BQ];
    int w=threadIdx.x>>5, ln=threadIdx.x&31;
    int bkv=blockIdx.y, kvh=bkv%KV, b=bkv/KV, grp=H/KV;
    int kb=blockIdx.x*TC_BK, k0=kb+16*w;
    float scale=rsqrtf((float)HD);
    __half*sKs=(__half*)sF, *sVs=(__half*)(sF+TC_BK*TC_BQ);
    for(int i=threadIdx.x;i<TC_BK*HD/8;i+=128){
        int r=i/(HD/8), c=(i-r*(HD/8))*8, s=kb+r;
        uint4 kk=make_uint4(0,0,0,0), vv=kk;
        if(s<T){ size_t o=(size_t)(b*T+s)*KD+kvh*HD+c;
                 kk=*(const uint4*)(K+o); vv=*(const uint4*)(V+o); }
        *(uint4*)(sKs+r*HD+c)=kk; *(uint4*)(sVs+r*HD+c)=vv;
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> fk[NF], fv[NF];
    wmma::fragment<wmma::accumulator,16,16,16,float> ak[NF], av[NF];
    #pragma unroll
    for(int f=0;f<NF;f++){
        wmma::load_matrix_sync(fk[f],sKs+16*w*HD+16*f,HD);
        wmma::load_matrix_sync(fv[f],sVs+16*w*HD+16*f,HD);
        wmma::fill_fragment(ak[f],0.f); wmma::fill_fragment(av[f],0.f);
    }
    float *sSw=sF+16*w*TC_BQ, *sDw=sF+TC_BK*TC_BQ+16*w*TC_BQ;
    __half*sPw=sP+16*w*TC_BQ, *sXw=sX+16*w*TC_BQ;
    int loT=kb, hiT=kb+TC_BK-1+(win>0?win-1:T); if(hiT>T-1) hiT=T-1;
    for(int hh=0;hh<grp;hh++){ int h=kvh*grp+hh;
      const float*Lh=lse+(size_t)(b*H+h)*T, *Dh=Drow+(size_t)(b*H+h)*T;
      for(int t0=loT;t0<=hiT;t0+=TC_BQ){
        __syncthreads();             /* prior tile's reads (and the K/V frag loads) done */
        for(int i=threadIdx.x;i<TC_BQ*HD/8;i+=128){
            int r=i/(HD/8), c=(i-r*(HD/8))*8, t=t0+r;
            uint4 qq=make_uint4(0,0,0,0), gg=qq;
            if(t<T){ size_t o=(size_t)(b*T+t)*D+h*HD+c;
                     qq=*(const uint4*)(Q+o); gg=*(const uint4*)(dA+o); }
            *(uint4*)(sQ+r*HD+c)=qq; *(uint4*)(sG+r*HD+c)=gg;
        }
        __syncthreads();
        /* warp-uniform skip when every (key,query) pair of this warp is masked */
        if(k0>=T || t0+TC_BQ-1<k0 || (win>0 && t0>k0+15+win-1)) continue;
        #pragma unroll
        for(int j=0;j<NQ;j++){
            wmma::fragment<wmma::accumulator,16,16,16,float> cs,cd;
            wmma::fill_fragment(cs,0.f); wmma::fill_fragment(cd,0.f);
            #pragma unroll
            for(int f=0;f<NF;f++){
                wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> fb;
                wmma::load_matrix_sync(fb,sQ+16*j*HD+16*f,HD); wmma::mma_sync(cs,fk[f],fb,cs);
                wmma::load_matrix_sync(fb,sG+16*j*HD+16*f,HD); wmma::mma_sync(cd,fv[f],fb,cd);
            }
            wmma::store_matrix_sync(sSw+16*j,cs,TC_BQ,wmma::mem_row_major);
            wmma::store_matrix_sync(sDw+16*j,cd,TC_BQ,wmma::mem_row_major);
        }
        __syncwarp();
        { int t=t0+ln; float L=0.f,dr=0.f; int dst=0; if(t<T){ L=Lh[t]; dr=Dh[t]; if(d_dstart) dst=d_dstart[b*T+t]; }
          #pragma unroll 4
          for(int i=0;i<16;i++){ int s=k0+i; float p=0.f,ds=0.f;
              if(t<T && s<T && t>=s && s>=dst && (win<=0 || t<=s+win-1)){
                  p=expf(sSw[i*TC_BQ+ln]*scale-L);
                  ds=p*(sDw[i*TC_BQ+ln]-dr)*(scale*TC_DSS); }
              sPw[i*TC_BQ+ln]=__float2half(p); sXw[i*TC_BQ+ln]=__float2half(ds); } }
        __syncwarp();
        #pragma unroll
        for(int j=0;j<NQ;j++){
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> fp,fs;
            wmma::load_matrix_sync(fp,sPw+16*j,TC_BQ); wmma::load_matrix_sync(fs,sXw+16*j,TC_BQ);
            #pragma unroll
            for(int f=0;f<NF;f++){
                wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> fb;
                wmma::load_matrix_sync(fb,sG+16*j*HD+16*f,HD); wmma::mma_sync(av[f],fp,fb,av[f]);
                wmma::load_matrix_sync(fb,sQ+16*j*HD+16*f,HD); wmma::mma_sync(ak[f],fs,fb,ak[f]);
            }
        }
      } }
    __syncthreads();                 /* sF is reused below by every warp */
    float*sO=sF+16*w*HD;             /* 16 x HD per warp; 4 warps fill sF at HD=64 */
    #pragma unroll
    for(int f=0;f<NF;f++) wmma::store_matrix_sync(sO+16*f,ak[f],HD,wmma::mem_row_major);
    __syncwarp();
    for(int e=ln;e<16*HD;e+=32){ int r=e/HD, d=e-r*HD, s=k0+r;
        if(s<T) dK[(size_t)(b*T+s)*KD+kvh*HD+d]+=sO[e]*(g_dsc[1]/TC_DSS); }
    __syncwarp();
    #pragma unroll
    for(int f=0;f<NF;f++) wmma::store_matrix_sync(sO+16*f,av[f],HD,wmma::mem_row_major);
    __syncwarp();
    for(int e=ln;e<16*HD;e+=32){ int r=e/HD, d=e-r*HD, s=k0+r;
        if(s<T) dV[(size_t)(b*T+s)*KD+kvh*HD+d]+=sO[e]*g_dsc[1]; }
}
/* 0 = no instantiation (hd not 16/32/64, or rows not 16-byte aligned) */
static int attn_bwd_dkv_tc(const __half*Q,const __half*K,const __half*V,const __half*dA,
                           const float*lse,const float*Drow,float*dK,float*dV,int win,
                           int B,int T,int H,int KV,int hd,int D,int KD){
    if(D%8 || KD%8) return 0;
    dim3 gr((unsigned)nblk((size_t)T,TC_BK),(unsigned)(B*KV));
    #define TC_KV(N) case N: k_attn_bwd_dkv_tc<N><<<gr,128>>>(Q,K,V,dA,lse,Drow,dK,dV,win,B,T,H,KV,D,KD); return 1;
    switch(hd){ TC_KV(64) TC_KV(32) TC_KV(16) default: return 0; }
    #undef TC_KV
}/* Tensor-core dQ, the dKV kernel's mirror. One block owns TC_BK=64 queries of
 * one (b,h); each warp keeps its 16 Q and dA rows as wmma A fragments. Per
 * 32-key tile:
 *   S = Q.K^T, dP = dA.V^T       (wmma, fp16 in, fp32 acc)
 *   dS = P*(dP-Drow)*scale       (fp32, masked, then fp16 with the x TC_DSS)
 *   dQ += dS.K                   (wmma, fp32 accumulators in registers)
 * K/V tiles are shared by all four warps; S/dP/dS scratch is per-warp rows. */
template<int HD>
__global__ __launch_bounds__(128)
void k_attn_bwd_dq_tc(const __half*Q,const __half*K,const __half*V,const __half*dA,
                      const float*lse,const float*Drow,float*dQ,int win,
                      int B,int T,int H,int KV,int D,int KD){
    using namespace nvcuda;
    static_assert(HD%16==0 && HD<=64, "tc dQ: HD in {16,32,64}");
    constexpr int NF=HD/16, NK=TC_BQ/16;
    __shared__ __align__(32) __half sK[TC_BQ*HD], sV[TC_BQ*HD];
    __shared__ __align__(32) float  sF[2*TC_BK*TC_BQ];   /* S | dP; Q/dA staging; output */
    __shared__ __align__(32) __half sX[TC_BK*TC_BQ];
    int w=threadIdx.x>>5, ln=threadIdx.x&31;
    int bh=blockIdx.y, h=bh%H, b=bh/H, kvh=h/(H/KV);
    int qb=blockIdx.x*TC_BK, q0=qb+16*w;
    float scale=rsqrtf((float)HD);
    __half*sQs=(__half*)sF, *sGs=(__half*)(sF+TC_BK*TC_BQ);
    for(int i=threadIdx.x;i<TC_BK*HD/8;i+=128){
        int r=i/(HD/8), c=(i-r*(HD/8))*8, t=qb+r;
        uint4 qq=make_uint4(0,0,0,0), gg=qq;
        if(t<T){ size_t o=(size_t)(b*T+t)*D+h*HD+c;
                 qq=*(const uint4*)(Q+o); gg=*(const uint4*)(dA+o); }
        *(uint4*)(sQs+r*HD+c)=qq; *(uint4*)(sGs+r*HD+c)=gg;
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> fq[NF], fg[NF];
    wmma::fragment<wmma::accumulator,16,16,16,float> aq[NF];
    #pragma unroll
    for(int f=0;f<NF;f++){
        wmma::load_matrix_sync(fq[f],sQs+16*w*HD+16*f,HD);
        wmma::load_matrix_sync(fg[f],sGs+16*w*HD+16*f,HD);
        wmma::fill_fragment(aq[f],0.f);
    }
    float Lr[16],Dr[16];
    { const float*Lh=lse+(size_t)(b*H+h)*T, *Dh=Drow+(size_t)(b*H+h)*T;
      #pragma unroll
      for(int i=0;i<16;i++){ int t=q0+i; Lr[i]=t<T?Lh[t]:0.f; Dr[i]=t<T?Dh[t]:0.f; } }
    float *sSw=sF+16*w*TC_BQ, *sDw=sF+TC_BK*TC_BQ+16*w*TC_BQ;
    __half*sXw=sX+16*w*TC_BQ;
    int hiS=qb+TC_BK-1; if(hiS>T-1) hiS=T-1;
    int loS=(win>0)?(qb-win+1):0; if(loS<0) loS=0;
    for(int s0=loS;s0<=hiS;s0+=TC_BQ){
        __syncthreads();             /* prior tile's reads (and the Q/dA frag loads) done */
        for(int i=threadIdx.x;i<TC_BQ*HD/8;i+=128){
            int r=i/(HD/8), c=(i-r*(HD/8))*8, s=s0+r;
            uint4 kk=make_uint4(0,0,0,0), vv=kk;
            if(s<T){ size_t o=(size_t)(b*T+s)*KD+kvh*HD+c;
                     kk=*(const uint4*)(K+o); vv=*(const uint4*)(V+o); }
            *(uint4*)(sK+r*HD+c)=kk; *(uint4*)(sV+r*HD+c)=vv;
        }
        __syncthreads();
        /* warp-uniform skip when every (query,key) pair of this warp is masked */
        if(q0>=T || s0>q0+15 || (win>0 && s0+TC_BQ-1<=q0-win)) continue;
        #pragma unroll
        for(int j=0;j<NK;j++){
            wmma::fragment<wmma::accumulator,16,16,16,float> cs,cd;
            wmma::fill_fragment(cs,0.f); wmma::fill_fragment(cd,0.f);
            #pragma unroll
            for(int f=0;f<NF;f++){
                wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> fb;
                wmma::load_matrix_sync(fb,sK+16*j*HD+16*f,HD); wmma::mma_sync(cs,fq[f],fb,cs);
                wmma::load_matrix_sync(fb,sV+16*j*HD+16*f,HD); wmma::mma_sync(cd,fg[f],fb,cd);
            }
            wmma::store_matrix_sync(sSw+16*j,cs,TC_BQ,wmma::mem_row_major);
            wmma::store_matrix_sync(sDw+16*j,cd,TC_BQ,wmma::mem_row_major);
        }
        __syncwarp();
        { int s=s0+ln; const int*DSp=d_dstart;
          #pragma unroll
          for(int i=0;i<16;i++){ int t=q0+i; float ds=0.f;
              if(t<T && s<T && s<=t && (win<=0 || s>t-win) && (!DSp || s>=DSp[b*T+t])){
                  float p=expf(sSw[i*TC_BQ+ln]*scale-Lr[i]);
                  ds=p*(sDw[i*TC_BQ+ln]-Dr[i])*(scale*TC_DSS); }
              sXw[i*TC_BQ+ln]=__float2half(ds); } }
        __syncwarp();
        #pragma unroll
        for(int j=0;j<NK;j++){
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> fs;
            wmma::load_matrix_sync(fs,sXw+16*j,TC_BQ);
            #pragma unroll
            for(int f=0;f<NF;f++){
                wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> fb;
                wmma::load_matrix_sync(fb,sK+16*j*HD+16*f,HD); wmma::mma_sync(aq[f],fs,fb,aq[f]);
            }
        }
    }
    __syncthreads();                 /* sF is reused below by every warp */
    float*sO=sF+16*w*HD;
    #pragma unroll
    for(int f=0;f<NF;f++) wmma::store_matrix_sync(sO+16*f,aq[f],HD,wmma::mem_row_major);
    __syncwarp();
    for(int e=ln;e<16*HD;e+=32){ int r=e/HD, d=e-r*HD, t=q0+r;
        if(t<T) dQ[(size_t)(b*T+t)*D+h*HD+d]+=sO[e]*(g_dsc[1]/TC_DSS); }
}
static int attn_bwd_dq_tc(const __half*Q,const __half*K,const __half*V,const __half*dA,
                          const float*lse,const float*Drow,float*dQ,int win,
                          int B,int T,int H,int KV,int hd,int D,int KD){
    if(D%8 || KD%8) return 0;
    dim3 gr((unsigned)nblk((size_t)T,TC_BK),(unsigned)(B*H));
    #define TC_Q(N) case N: k_attn_bwd_dq_tc<N><<<gr,128>>>(Q,K,V,dA,lse,Drow,dQ,win,B,T,H,KV,D,KD); return 1;
    switch(hd){ TC_Q(64) TC_Q(32) TC_Q(16) default: return 0; }
    #undef TC_Q
}/* Tensor-core forward (FA2-style), same tiling as the dQ kernel: one block owns
 * TC_BK=64 queries of one (b,h), each warp 16 of them; 32-key K/V tiles are
 * shared by all warps. S=Q*K^T on wmma, online softmax in fp32 with two lanes
 * per query row, P to fp16, P*V back through smem so the running O -- which
 * needs a per-row rescale -- lives in registers rather than an opaque fragment.
 * Lanes take interleaved columns and the smem tiles are padded (+4 floats /
 * +8 halves) so the row-per-lane-pair reads are 2-way, not 16-way, conflicted. */
#define TC_LS (TC_BQ+4)
#define TC_LP (TC_BQ+8)
template<int HD>
__global__ __launch_bounds__(128)
void k_attn_fwd_tc(const __half*Q,const __half*K,const __half*V,float*O,float*lse,
                   int win,int B,int T,int H,int KV,int D,int KD){
    using namespace nvcuda;
    static_assert(HD%16==0 && HD<=64, "tc fwd: HD in {16,32,64}");
    constexpr int NF=HD/16, NK=TC_BQ/16, HH=HD/2, LT=HD+4;
    __shared__ __align__(32) __half sK[TC_BQ*HD], sV[TC_BQ*HD], sP[TC_BK*TC_LP];
    __shared__ __align__(32) float  sF[TC_BK*TC_LS+TC_BK*LT];   /* S | P*V; Q staging */
    int w=threadIdx.x>>5, ln=threadIdx.x&31, r=ln>>1, hf=ln&1;
    int bh=blockIdx.y, h=bh%H, b=bh/H, kvh=h/(H/KV);
    int qb=blockIdx.x*TC_BK, q0=qb+16*w, t=q0+r;
    int dst=(d_dstart && t<T) ? d_dstart[b*T+t] : 0;
    float scale=rsqrtf((float)HD);
    __half*sQs=(__half*)sF;
    for(int i=threadIdx.x;i<TC_BK*HD/8;i+=128){
        int rr=i/(HD/8), c=(i-rr*(HD/8))*8, tt=qb+rr;
        uint4 qq=make_uint4(0,0,0,0);
        if(tt<T) qq=*(const uint4*)(Q+(size_t)(b*T+tt)*D+h*HD+c);
        *(uint4*)(sQs+rr*HD+c)=qq;
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> fq[NF];
    #pragma unroll
    for(int f=0;f<NF;f++) wmma::load_matrix_sync(fq[f],sQs+16*w*HD+16*f,HD);
    float acc[HH];
    #pragma unroll
    for(int d=0;d<HH;d++) acc[d]=0.f;
    float m=-1e30f,l=0.f;
    float *sSw=sF+16*w*TC_LS, *sTw=sF+TC_BK*TC_LS+16*w*LT;
    __half*sPw=sP+16*w*TC_LP;
    int hiS=qb+TC_BK-1; if(hiS>T-1) hiS=T-1;
    int loS=(win>0)?(qb-win+1):0; if(loS<0) loS=0;
    for(int s0=loS;s0<=hiS;s0+=TC_BQ){
        __syncthreads();             /* prior tile's reads (and the Q frag loads) done */
        for(int i=threadIdx.x;i<TC_BQ*HD/8;i+=128){
            int rr=i/(HD/8), c=(i-rr*(HD/8))*8, s=s0+rr;
            uint4 kk=make_uint4(0,0,0,0), vv=kk;
            if(s<T){ size_t o=(size_t)(b*T+s)*KD+kvh*HD+c;
                     kk=*(const uint4*)(K+o); vv=*(const uint4*)(V+o); }
            *(uint4*)(sK+rr*HD+c)=kk; *(uint4*)(sV+rr*HD+c)=vv;
        }
        __syncthreads();
        if(q0>=T || s0>q0+15 || (win>0 && s0+TC_BQ-1<=q0-win)) continue;
        #pragma unroll
        for(int j=0;j<NK;j++){
            wmma::fragment<wmma::accumulator,16,16,16,float> cs;
            wmma::fill_fragment(cs,0.f);
            #pragma unroll
            for(int f=0;f<NF;f++){
                wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> fb;
                wmma::load_matrix_sync(fb,sK+16*j*HD+16*f,HD); wmma::mma_sync(cs,fq[f],fb,cs);
            }
            wmma::store_matrix_sync(sSw+16*j,cs,TC_LS,wmma::mem_row_major);
        }
        __syncwarp();
        float x[16], mt=-INFINITY;
        #pragma unroll
        for(int k=0;k<16;k++){ int c=2*k+hf, s=s0+c;
            bool ok=t<T && s<T && s<=t && s>=dst && (win<=0 || s>t-win);
            x[k]=ok? sSw[r*TC_LS+c]*scale : -INFINITY; mt=fmaxf(mt,x[k]); }
        mt=fmaxf(mt,__shfl_xor_sync(0xffffffffu,mt,1));
        float nm=fmaxf(m,mt), corr=expf(m-nm), ps=0.f;
        #pragma unroll
        for(int k=0;k<16;k++){ float p=expf(x[k]-nm); ps+=p; sPw[r*TC_LP+2*k+hf]=__float2half(p); }
        ps+=__shfl_xor_sync(0xffffffffu,ps,1);
        l=l*corr+ps; m=nm;
        __syncwarp();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> fp[NK];
        #pragma unroll
        for(int j=0;j<NK;j++) wmma::load_matrix_sync(fp[j],sPw+16*j,TC_LP);
        #pragma unroll
        for(int f=0;f<NF;f++){
            wmma::fragment<wmma::accumulator,16,16,16,float> co;
            wmma::fill_fragment(co,0.f);
            #pragma unroll
            for(int j=0;j<NK;j++){
                wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> fb;
                wmma::load_matrix_sync(fb,sV+16*j*HD+16*f,HD); wmma::mma_sync(co,fp[j],fb,co);
            }
            wmma::store_matrix_sync(sTw+16*f,co,LT,wmma::mem_row_major);
        }
        __syncwarp();
        const float*tr=sTw+r*LT+hf;
        #pragma unroll
        for(int k=0;k<HH;k++) acc[k]=acc[k]*corr+tr[2*k];
    }
    if(t>=T) return;
    float inv=1.0f/l; float*o=O+(size_t)(b*T+t)*D+h*HD+hf;
    #pragma unroll
    for(int k=0;k<HH;k++) o[2*k]=acc[k]*inv;
    if(!hf) lse[(size_t)(b*H+h)*T+t]=m+logf(l);
}
static int attn_fwd_tc(const __half*Q,const __half*K,const __half*V,float*O,float*lse,
                       int win,int B,int T,int H,int KV,int hd,int D,int KD){
    if(D%8 || KD%8) return 0;
    dim3 gr((unsigned)nblk((size_t)T,TC_BK),(unsigned)(B*H));
    #define TC_F(N) case N: k_attn_fwd_tc<N><<<gr,128>>>(Q,K,V,O,lse,win,B,T,H,KV,D,KD); return 1;
    switch(hd){ TC_F(64) TC_F(32) TC_F(16) default: return 0; }
    #undef TC_F
}

/* ---- FA2-style attention on raw mma.sync m16n8k16 (CUDALM_FA, default 1) ----
 * Same contracts as the wmma TC kernels above (fp16 Q/K/V/dA, fp32 O/lse/dQ/dK/dV,
 * dA and Drow scaled by DSCALE, outputs +=), but S/P/dS never leave registers: the
 * fp32 accumulator of S = Q.K^T is, two 16x8 tiles at a time, exactly the A-operand
 * layout of P for P.V. 64x64 tiles, 4 warps x 16 rows, K/V (or Q/dA) tiles in smem
 * padded to HD+8 halves so ldmatrix rows are conflict-free. Softmax in the exp2
 * domain. */
#define FA_B 64
__device__ __forceinline__ void fa_mma(float*c,const unsigned*a,const unsigned*b){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
__device__ __forceinline__ void fa_ld4(unsigned*r,const __half*p){
    unsigned s=(unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(s));
}
__device__ __forceinline__ void fa_ld4t(unsigned*r,const __half*p){
    unsigned s=(unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];\n":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(s));
}
__device__ __forceinline__ unsigned fa_pk(float a,float b){ __half2 h=__floats2half2_rn(a,b); return *(unsigned*)&h; }
/* rows r0.. of a [*,ld] fp16 matrix (columns c0..c0+HD) -> smem [FA_B][HD+8], zero past T */
template<int HD>
__device__ __forceinline__ void fa_tile(__half*s,const __half*g,int r0,int T,int bT,int ld,int c0){
    for(int i=threadIdx.x;i<FA_B*HD/8;i+=128){
        int r=i/(HD/8), c=(i-r*(HD/8))*8;
        uint4 v=make_uint4(0,0,0,0);
        if(r0+r<T) v=*(const uint4*)(g+(size_t)(bT+r0+r)*ld+c0+c);
        *(uint4*)(s+r*(HD+8)+c)=v;
    }
}
/* O6: same tile via cp.async (16 B per copy, rows past T zero-filled with plain stores) */
template<int HD>
__device__ __forceinline__ void fa_tile_cp(__half*s,const __half*g,int r0,int T,int bT,int ld,int c0){
    for(int i=threadIdx.x;i<FA_B*HD/8;i+=128){
        int r=i/(HD/8), c=(i-r*(HD/8))*8; __half*d=s+r*(HD+8)+c;
        if(r0+r<T){ unsigned sa=(unsigned)__cvta_generic_to_shared(d);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(sa),"l"(g+(size_t)(bT+r0+r)*ld+c0+c)); }
        else *(uint4*)d=make_uint4(0,0,0,0);
    }
}
template<int CP,int HD>
__device__ __forceinline__ void fa_fill(__half*s,const __half*g,int r0,int T,int bT,int ld,int c0){
    if(CP) fa_tile_cp<HD>(s,g,r0,T,bT,ld,c0); else fa_tile<HD>(s,g,r0,T,bT,ld,c0); }
__device__ __forceinline__ void fa_commit(void){ asm volatile("cp.async.commit_group;\n"::); }
__device__ __forceinline__ void fa_wait1(void){ asm volatile("cp.async.wait_group 1;\n"::); }
/* B operand from an smem tile whose rows are the mma N index (keys for Q.K^T):
 * x4 = n-tiles 2p,2p+1 at k-step kk */
#define FA_BN(sm,p,kk) ((sm)+(16*(p)+(ln&7)+8*(ln>>4))*LDS+16*(kk)+8*((ln>>3)&1))
/* B operand from an smem tile whose rows are the mma K index (keys for P.V), .trans:
 * x4 = n-tiles 2p,2p+1 (columns 16p..) at k-step j */
#define FA_BK(sm,j,p) ((sm)+(16*(j)+(ln&7)+8*((ln>>3)&1))*LDS+16*(p)+8*(ln>>4))
/* A fragments of a 16 x HD row block straight from global (rows ra, rb) */
#define FA_AG(fr,base,ra,rb,ld,col0,okA,okB) \
    _Pragma("unroll") for(int kk=0;kk<HD/16;kk++){ \
        const __half*pa=(base)+(size_t)(ra)*(ld)+(col0)+16*kk+2*tq, *pb=(base)+(size_t)(rb)*(ld)+(col0)+16*kk+2*tq; \
        fr[kk][0]=(okA)?*(const unsigned*)pa:0u;     fr[kk][1]=(okB)?*(const unsigned*)pb:0u; \
        fr[kk][2]=(okA)?*(const unsigned*)(pa+8):0u; fr[kk][3]=(okB)?*(const unsigned*)(pb+8):0u; }

template<int HD,int CP>
__global__ __launch_bounds__(128)
void k_attn_fwd_fa(const __half*Q,const __half*K,const __half*V,float*O,float*lse,
                   int win,int B,int T,int H,int KV,int D,int KD){
    constexpr int LDS=HD+8, NK=HD/16, ND=HD/8;
    __shared__ __align__(16) __half sKb[2][FA_B*LDS], sVb[2][FA_B*LDS];
    __shared__ int sMin;
    int w=threadIdx.x>>5, ln=threadIdx.x&31, g=ln>>2, tq=ln&3;
    int bh=blockIdx.y, h=bh%H, b=bh/H, kvh=h/(H/KV);
    int qb=(gridDim.x-1-blockIdx.x)*FA_B, q0=qb+16*w, tA=q0+g, tB=tA+8;
    int dA=0,dB=0;
    if(d_dstart){ if(tA<T) dA=d_dstart[b*T+tA]; if(tB<T) dB=d_dstart[b*T+tB]; }
    unsigned qf[NK][4];
    FA_AG(qf,Q,b*T+tA,b*T+tB,D,h*HD,tA<T,tB<T)
    float o[ND][4];
    #pragma unroll
    for(int n=0;n<ND;n++) o[n][0]=o[n][1]=o[n][2]=o[n][3]=0.f;
    float mA=-1e30f,mB=-1e30f,lA=0.f,lB=0.f;
    const float sl2=rsqrtf((float)HD)*1.4426950409f;
    int hi=qb+FA_B-1; if(hi>T-1) hi=T-1;
    int lo=win>0?qb-win+1:0; if(lo<0) lo=0;
    if(d_dstart){                       /* skip key tiles before every row's document start */
        if(threadIdx.x==0) sMin=T; __syncthreads();
        atomicMin(&sMin,min(tA<T?dA:T,tB<T?dB:T)); __syncthreads();
        if(sMin>lo) lo=sMin; }
    int s0=lo/FA_B*FA_B, bf=0;
    if(s0<=hi){ fa_fill<CP,HD>(sKb[0],K,s0,T,b*T,KD,kvh*HD); fa_fill<CP,HD>(sVb[0],V,s0,T,b*T,KD,kvh*HD); }
    fa_commit();
    for(;s0<=hi;s0+=FA_B,bf^=1){
        if(s0+FA_B<=hi){ fa_fill<CP,HD>(sKb[bf^1],K,s0+FA_B,T,b*T,KD,kvh*HD); fa_fill<CP,HD>(sVb[bf^1],V,s0+FA_B,T,b*T,KD,kvh*HD); }
        fa_commit(); fa_wait1(); __syncthreads();
        const __half *sK=sKb[bf], *sV=sVb[bf];
        if(!(q0>=T || s0>q0+15 || (win>0 && s0+FA_B-1<=q0-win))){
        float s[8][4];
        #pragma unroll
        for(int n=0;n<8;n++) s[n][0]=s[n][1]=s[n][2]=s[n][3]=0.f;
        #pragma unroll
        for(int kk=0;kk<NK;kk++)
            #pragma unroll
            for(int p=0;p<4;p++){ unsigned bk[4]; fa_ld4(bk,FA_BN(sK,p,kk));
                fa_mma(s[2*p],qf[kk],bk); fa_mma(s[2*p+1],qf[kk],bk+2); }
        float xA=-INFINITY,xB=-INFINITY;
        #pragma unroll
        for(int n=0;n<8;n++)
            #pragma unroll
            for(int e=0;e<4;e++){ int key=s0+8*n+2*tq+(e&1), t=e<2?tA:tB, ds=e<2?dA:dB;
                bool ok=t<T && key<=t && key>=ds && (win<=0 || key>t-win);
                float v=ok?s[n][e]*sl2:-INFINITY; s[n][e]=v;
                if(e<2) xA=fmaxf(xA,v); else xB=fmaxf(xB,v); }
        xA=fmaxf(xA,__shfl_xor_sync(0xffffffffu,xA,1)); xA=fmaxf(xA,__shfl_xor_sync(0xffffffffu,xA,2));
        xB=fmaxf(xB,__shfl_xor_sync(0xffffffffu,xB,1)); xB=fmaxf(xB,__shfl_xor_sync(0xffffffffu,xB,2));
        float nA=fmaxf(mA,xA), nB=fmaxf(mB,xB), cA=exp2f(mA-nA), cB=exp2f(mB-nB);
        mA=nA; mB=nB; lA*=cA; lB*=cB;
        #pragma unroll
        for(int n=0;n<8;n++){ s[n][0]=exp2f(s[n][0]-nA); s[n][1]=exp2f(s[n][1]-nA);
            s[n][2]=exp2f(s[n][2]-nB); s[n][3]=exp2f(s[n][3]-nB);
            lA+=s[n][0]+s[n][1]; lB+=s[n][2]+s[n][3]; }
        #pragma unroll
        for(int n=0;n<ND;n++){ o[n][0]*=cA; o[n][1]*=cA; o[n][2]*=cB; o[n][3]*=cB; }
        #pragma unroll
        for(int j=0;j<4;j++){
            unsigned pa[4]={fa_pk(s[2*j][0],s[2*j][1]),fa_pk(s[2*j][2],s[2*j][3]),
                            fa_pk(s[2*j+1][0],s[2*j+1][1]),fa_pk(s[2*j+1][2],s[2*j+1][3])};
            #pragma unroll
            for(int p=0;p<NK;p++){ unsigned bv[4]; fa_ld4t(bv,FA_BK(sV,j,p));
                fa_mma(o[2*p],pa,bv); fa_mma(o[2*p+1],pa,bv+2); }
        }
        }
        __syncthreads();
    }
    lA+=__shfl_xor_sync(0xffffffffu,lA,1); lA+=__shfl_xor_sync(0xffffffffu,lA,2);
    lB+=__shfl_xor_sync(0xffffffffu,lB,1); lB+=__shfl_xor_sync(0xffffffffu,lB,2);
    const float LN2=0.6931471806f;
    if(tA<T){ float iv=1.f/lA; float*op=O+(size_t)(b*T+tA)*D+h*HD+2*tq;
        #pragma unroll
        for(int n=0;n<ND;n++) *(float2*)(op+8*n)=make_float2(o[n][0]*iv,o[n][1]*iv);
        if(!tq) lse[(size_t)(b*H+h)*T+tA]=mA*LN2+logf(lA); }
    if(tB<T){ float iv=1.f/lB; float*op=O+(size_t)(b*T+tB)*D+h*HD+2*tq;
        #pragma unroll
        for(int n=0;n<ND;n++) *(float2*)(op+8*n)=make_float2(o[n][2]*iv,o[n][3]*iv);
        if(!tq) lse[(size_t)(b*H+h)*T+tB]=mB*LN2+logf(lB); }
}

/* dQ: rows = 16 queries per warp. S=Q.K^T, dP=dA.V^T, dS=P(dP-Drow)*scale*TC_DSS,
 * dQ += dS.K */
template<int HD,int CP>
__global__ __launch_bounds__(128)
void k_attn_bwd_dq_fa(const __half*Q,const __half*K,const __half*V,const __half*dA,
                      const float*lse,const float*Drow,float*dQ,int win,
                      int B,int T,int H,int KV,int D,int KD){
    constexpr int LDS=HD+8, NK=HD/16, ND=HD/8;
    __shared__ __align__(16) __half sKb[2][FA_B*LDS], sVb[2][FA_B*LDS];
    __shared__ int sMin;
    int w=threadIdx.x>>5, ln=threadIdx.x&31, g=ln>>2, tq=ln&3;
    int bh=blockIdx.y, h=bh%H, b=bh/H, kvh=h/(H/KV);
    int qb=(gridDim.x-1-blockIdx.x)*FA_B, q0=qb+16*w, tA=q0+g, tB=tA+8;
    int dsA=0,dsB=0;
    if(d_dstart){ if(tA<T) dsA=d_dstart[b*T+tA]; if(tB<T) dsB=d_dstart[b*T+tB]; }
    unsigned qf[NK][4], gf[NK][4];
    FA_AG(qf,Q,b*T+tA,b*T+tB,D,h*HD,tA<T,tB<T)
    FA_AG(gf,dA,b*T+tA,b*T+tB,D,h*HD,tA<T,tB<T)
    const float L2E=1.4426950409f, sl2=rsqrtf((float)HD)*L2E, dsc=rsqrtf((float)HD)*TC_DSS;
    const float*Lh=lse+(size_t)(b*H+h)*T, *Dh=Drow+(size_t)(b*H+h)*T;
    float LA=tA<T?Lh[tA]*L2E:0.f, LB=tB<T?Lh[tB]*L2E:0.f, RA=tA<T?Dh[tA]:0.f, RB=tB<T?Dh[tB]:0.f;
    float dq[ND][4];
    #pragma unroll
    for(int n=0;n<ND;n++) dq[n][0]=dq[n][1]=dq[n][2]=dq[n][3]=0.f;
    int hi=qb+FA_B-1; if(hi>T-1) hi=T-1;
    int lo=win>0?qb-win+1:0; if(lo<0) lo=0;
    if(d_dstart){
        if(threadIdx.x==0) sMin=T; __syncthreads();
        atomicMin(&sMin,min(tA<T?dsA:T,tB<T?dsB:T)); __syncthreads();
        if(sMin>lo) lo=sMin; }
    int s0=lo/FA_B*FA_B, bf=0;
    if(s0<=hi){ fa_fill<CP,HD>(sKb[0],K,s0,T,b*T,KD,kvh*HD); fa_fill<CP,HD>(sVb[0],V,s0,T,b*T,KD,kvh*HD); }
    fa_commit();
    for(;s0<=hi;s0+=FA_B,bf^=1){
        if(s0+FA_B<=hi){ fa_fill<CP,HD>(sKb[bf^1],K,s0+FA_B,T,b*T,KD,kvh*HD); fa_fill<CP,HD>(sVb[bf^1],V,s0+FA_B,T,b*T,KD,kvh*HD); }
        fa_commit(); fa_wait1(); __syncthreads();
        const __half *sK=sKb[bf], *sV=sVb[bf];
        if(!(q0>=T || s0>q0+15 || (win>0 && s0+FA_B-1<=q0-win))){
        float s[8][4], dp[8][4];
        #pragma unroll
        for(int n=0;n<8;n++){ s[n][0]=s[n][1]=s[n][2]=s[n][3]=0.f; dp[n][0]=dp[n][1]=dp[n][2]=dp[n][3]=0.f; }
        #pragma unroll
        for(int kk=0;kk<NK;kk++)
            #pragma unroll
            for(int p=0;p<4;p++){ unsigned bk[4],bv[4]; fa_ld4(bk,FA_BN(sK,p,kk)); fa_ld4(bv,FA_BN(sV,p,kk));
                fa_mma(s[2*p],qf[kk],bk); fa_mma(s[2*p+1],qf[kk],bk+2);
                fa_mma(dp[2*p],gf[kk],bv); fa_mma(dp[2*p+1],gf[kk],bv+2); }
        #pragma unroll
        for(int n=0;n<8;n++)
            #pragma unroll
            for(int e=0;e<4;e++){ int key=s0+8*n+2*tq+(e&1), t=e<2?tA:tB, ds=e<2?dsA:dsB;
                bool ok=t<T && key<=t && key>=ds && (win<=0 || key>t-win);
                float pr=ok?exp2f(s[n][e]*sl2-(e<2?LA:LB)):0.f;
                s[n][e]=pr*(dp[n][e]-(e<2?RA:RB))*dsc; }
        #pragma unroll
        for(int j=0;j<4;j++){
            unsigned xa[4]={fa_pk(s[2*j][0],s[2*j][1]),fa_pk(s[2*j][2],s[2*j][3]),
                            fa_pk(s[2*j+1][0],s[2*j+1][1]),fa_pk(s[2*j+1][2],s[2*j+1][3])};
            #pragma unroll
            for(int p=0;p<NK;p++){ unsigned bk[4]; fa_ld4t(bk,FA_BK(sK,j,p));
                fa_mma(dq[2*p],xa,bk); fa_mma(dq[2*p+1],xa,bk+2); }
        }
        }
        __syncthreads();
    }
    const float os=g_dsc[1]/TC_DSS;
    if(tA<T){ float*op=dQ+(size_t)(b*T+tA)*D+h*HD+2*tq;
        #pragma unroll
        for(int n=0;n<ND;n++){ float2 v=*(float2*)(op+8*n); v.x+=dq[n][0]*os; v.y+=dq[n][1]*os; *(float2*)(op+8*n)=v; } }
    if(tB<T){ float*op=dQ+(size_t)(b*T+tB)*D+h*HD+2*tq;
        #pragma unroll
        for(int n=0;n<ND;n++){ float2 v=*(float2*)(op+8*n); v.x+=dq[n][2]*os; v.y+=dq[n][3]*os; *(float2*)(op+8*n)=v; } }
}

/* dK/dV: rows = 16 keys per warp, looping over the H/KV query heads that share this
 * kv head and over 64-query tiles. S^T=K.Q^T, dP^T=V.dA^T, P^T=exp(S^T-lse[q]),
 * dV += P^T.dA, dS^T=P^T(dP^T-Drow[q]), dK += dS^T.Q */
template<int HD,int CP>
__global__ __launch_bounds__(128)
void k_attn_bwd_dkv_fa(const __half*Q,const __half*K,const __half*V,const __half*dA,
                       const float*lse,const float*Drow,float*dK,float*dV,int win,
                       int B,int T,int H,int KV,int D,int KD){
    constexpr int LDS=HD+8, NK=HD/16, ND=HD/8;
    __shared__ __align__(16) __half sQb[2][FA_B*LDS], sGb[2][FA_B*LDS];
    __shared__ float sLb[2][FA_B], sRb[2][FA_B]; __shared__ int sDb[2][FA_B];
    int w=threadIdx.x>>5, ln=threadIdx.x&31, g=ln>>2, tq=ln&3;
    int bkv=blockIdx.y, kvh=bkv%KV, b=bkv/KV, grp=H/KV;
    int kb=blockIdx.x*FA_B, k0=kb+16*w, kA=k0+g, kB=kA+8;
    unsigned kf[NK][4], vf[NK][4];
    FA_AG(kf,K,b*T+kA,b*T+kB,KD,kvh*HD,kA<T,kB<T)
    FA_AG(vf,V,b*T+kA,b*T+kB,KD,kvh*HD,kA<T,kB<T)
    const float L2E=1.4426950409f, sl2=rsqrtf((float)HD)*L2E, dsc=rsqrtf((float)HD)*TC_DSS;
    float dk[ND][4], dv[ND][4];
    #pragma unroll
    for(int n=0;n<ND;n++){ dk[n][0]=dk[n][1]=dk[n][2]=dk[n][3]=0.f; dv[n][0]=dv[n][1]=dv[n][2]=dv[n][3]=0.f; }
    int hiT=kb+FA_B-1+(win>0?win-1:T); if(hiT>T-1) hiT=T-1;
    /* the (head, query tile) pairs as one flat sequence so the prefetch runs across heads */
    int ntq=hiT>=kb?(hiT-kb)/FA_B+1:0, nti=grp*ntq;
    auto fill=[&](int i,int bq){ int h=kvh*grp+i/ntq, t0=kb+(i%ntq)*FA_B;
        fa_fill<CP,HD>(sQb[bq],Q,t0,T,b*T,D,h*HD); fa_fill<CP,HD>(sGb[bq],dA,t0,T,b*T,D,h*HD);
        if(threadIdx.x<FA_B){ int t=t0+threadIdx.x; const float*Lh=lse+(size_t)(b*H+h)*T, *Dh=Drow+(size_t)(b*H+h)*T;
            if(t<T&&CP){ unsigned a0=(unsigned)__cvta_generic_to_shared(&sLb[bq][threadIdx.x]), a1=(unsigned)__cvta_generic_to_shared(&sRb[bq][threadIdx.x]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(a0),"l"(Lh+t));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(a1),"l"(Dh+t));
                if(d_dstart){ unsigned a2=(unsigned)__cvta_generic_to_shared(&sDb[bq][threadIdx.x]);
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(a2),"l"(d_dstart+b*T+t)); }
                else sDb[bq][threadIdx.x]=0; }
            else { bool ok=t<T; sLb[bq][threadIdx.x]=ok?Lh[t]:0.f; sRb[bq][threadIdx.x]=ok?Dh[t]:0.f;
                sDb[bq][threadIdx.x]=ok?(d_dstart?d_dstart[b*T+t]:0):T; } } };
    int bf=0;
    if(nti) fill(0,0);
    fa_commit();
    for(int it=0;it<nti;it++,bf^=1){
        int t0=kb+(it%ntq)*FA_B;
        if(it+1<nti) fill(it+1,bf^1);
        fa_commit(); fa_wait1(); __syncthreads();
        const __half *sQ=sQb[bf], *sG=sGb[bf]; const float *sL=sLb[bf], *sR=sRb[bf]; const int*sD=sDb[bf];
        if(!(k0>=T || t0+FA_B-1<k0 || (win>0 && t0>k0+15+win-1))){
        float s[8][4];
        #pragma unroll
        for(int n=0;n<8;n++) s[n][0]=s[n][1]=s[n][2]=s[n][3]=0.f;
        #pragma unroll
        for(int kk=0;kk<NK;kk++)
            #pragma unroll
            for(int p=0;p<4;p++){ unsigned bq[4]; fa_ld4(bq,FA_BN(sQ,p,kk));
                fa_mma(s[2*p],kf[kk],bq); fa_mma(s[2*p+1],kf[kk],bq+2); }
        unsigned ph[8][2];              /* P^T as packed fp16 pairs: rows kA (0), kB (1) */
        #pragma unroll
        for(int n=0;n<8;n++){
            float pr[4];
            #pragma unroll
            for(int e=0;e<4;e++){ int c=8*n+2*tq+(e&1), t=t0+c, key=e<2?kA:kB;
                bool ok=key<T && t<T && t>=key && key>=sD[c] && (win<=0 || t<=key+win-1);
                pr[e]=ok?exp2f(s[n][e]*sl2-sL[c]*L2E):0.f; }
            ph[n][0]=fa_pk(pr[0],pr[1]); ph[n][1]=fa_pk(pr[2],pr[3]); }
        #pragma unroll
        for(int j=0;j<4;j++){
            unsigned pa[4]={ph[2*j][0],ph[2*j][1],ph[2*j+1][0],ph[2*j+1][1]};
            #pragma unroll
            for(int p=0;p<NK;p++){ unsigned bg[4]; fa_ld4t(bg,FA_BK(sG,j,p));
                fa_mma(dv[2*p],pa,bg); fa_mma(dv[2*p+1],pa,bg+2); }
        }
        #pragma unroll
        for(int n=0;n<8;n++) s[n][0]=s[n][1]=s[n][2]=s[n][3]=0.f;
        #pragma unroll
        for(int kk=0;kk<NK;kk++)
            #pragma unroll
            for(int p=0;p<4;p++){ unsigned bg[4]; fa_ld4(bg,FA_BN(sG,p,kk));
                fa_mma(s[2*p],vf[kk],bg); fa_mma(s[2*p+1],vf[kk],bg+2); }
        #pragma unroll
        for(int n=0;n<8;n++){
            float2 pA=__half22float2(*(__half2*)&ph[n][0]), pB=__half22float2(*(__half2*)&ph[n][1]);
            int c=8*n+2*tq; float r0=sR[c], r1=sR[c+1];
            ph[n][0]=fa_pk(pA.x*(s[n][0]-r0)*dsc,pA.y*(s[n][1]-r1)*dsc);
            ph[n][1]=fa_pk(pB.x*(s[n][2]-r0)*dsc,pB.y*(s[n][3]-r1)*dsc); }
        #pragma unroll
        for(int j=0;j<4;j++){
            unsigned xa[4]={ph[2*j][0],ph[2*j][1],ph[2*j+1][0],ph[2*j+1][1]};
            #pragma unroll
            for(int p=0;p<NK;p++){ unsigned bq[4]; fa_ld4t(bq,FA_BK(sQ,j,p));
                fa_mma(dk[2*p],xa,bq); fa_mma(dk[2*p+1],xa,bq+2); }
        }
        }
        __syncthreads();
    }
    const float ok_=g_dsc[1]/TC_DSS, ov=g_dsc[1];
    #pragma unroll
    for(int r=0;r<2;r++){ int key=r?kB:kA; if(key>=T) continue;
        float*pk=dK+(size_t)(b*T+key)*KD+kvh*HD+2*tq, *pv=dV+(size_t)(b*T+key)*KD+kvh*HD+2*tq;
        #pragma unroll
        for(int n=0;n<ND;n++){ float2 a=*(float2*)(pk+8*n), c=*(float2*)(pv+8*n);
            a.x+=dk[n][2*r]*ok_; a.y+=dk[n][2*r+1]*ok_; c.x+=dv[n][2*r]*ov; c.y+=dv[n][2*r+1]*ov;
            *(float2*)(pk+8*n)=a; *(float2*)(pv+8*n)=c; } }
}
#undef FA_BN
#undef FA_BK
#undef FA_AG
static int g_fa=1;      /* CUDALM_FA=0: wmma TC kernels */
static int g_facp=1;    /* O6 CUDALM_FA_CP=0: synchronous tile loads (no cp.async prefetch) */
static int attn_fwd_fa(const __half*Q,const __half*K,const __half*V,float*O,float*lse,
                       int win,int B,int T,int H,int KV,int hd,int D,int KD){
    if(!g_fa || hd!=64 || D%8 || KD%8) return 0;
    dim3 gr((unsigned)nblk((size_t)T,FA_B),(unsigned)(B*H));
    if(g_facp) k_attn_fwd_fa<64,1><<<gr,128>>>(Q,K,V,O,lse,win,B,T,H,KV,D,KD);
    else       k_attn_fwd_fa<64,0><<<gr,128>>>(Q,K,V,O,lse,win,B,T,H,KV,D,KD); return 1;
}
static int attn_bwd_dq_fa(const __half*Q,const __half*K,const __half*V,const __half*dA,
                          const float*lse,const float*Drow,float*dQ,int win,
                          int B,int T,int H,int KV,int hd,int D,int KD){
    if(!g_fa || hd!=64 || D%8 || KD%8) return 0;
    dim3 gr((unsigned)nblk((size_t)T,FA_B),(unsigned)(B*H));
    if(g_facp) k_attn_bwd_dq_fa<64,1><<<gr,128>>>(Q,K,V,dA,lse,Drow,dQ,win,B,T,H,KV,D,KD);
    else       k_attn_bwd_dq_fa<64,0><<<gr,128>>>(Q,K,V,dA,lse,Drow,dQ,win,B,T,H,KV,D,KD); return 1;
}
static int attn_bwd_dkv_fa(const __half*Q,const __half*K,const __half*V,const __half*dA,
                           const float*lse,const float*Drow,float*dK,float*dV,int win,
                           int B,int T,int H,int KV,int hd,int D,int KD){
    if(!g_fa || hd!=64 || D%8 || KD%8) return 0;
    dim3 gr((unsigned)nblk((size_t)T,FA_B),(unsigned)(B*KV));
    if(g_facp) k_attn_bwd_dkv_fa<64,1><<<gr,128>>>(Q,K,V,dA,lse,Drow,dK,dV,win,B,T,H,KV,D,KD);
    else       k_attn_bwd_dkv_fa<64,0><<<gr,128>>>(Q,K,V,dA,lse,Drow,dK,dV,win,B,T,H,KV,D,KD); return 1;
}

/* Sub-lane dKV: SPL threads cooperate on ONE key, each owning HD/SPL dims.
 *
 * The one-thread-per-key form above holds four HD-sized arrays (k, v and the two
 * accumulators) and compiles to 255 registers with 388 B of spill. Under
 * __launch_bounds__(128,2) that is 8 of a possible 32 warps per SM -- 25%
 * occupancy on the kernel that costs ~30% of the training step, with spill
 * traffic sitting in the inner loop and nothing resident to hide it behind.
 *
 * Splitting the HEAD DIM across SPL threads divides all four arrays by SPL, so
 * SPL=2 targets 128 registers and 4 blocks/SM. The two dot products then span
 * SPL threads and need a depth-log2(SPL) butterfly to finish -- 2 shuffles per
 * (key,query) pair against 256 FMAs, ~1.5% overhead.
 *
 * This is why it beats the obvious alternative of splitting into separate dK and
 * dV kernels: BOTH of those still need qk AND gv, so that form recomputes the
 * two dots twice and costs 1.5x the FMAs. Here the math is unchanged and only
 * the register budget moves.
 *
 * The butterfly must be warp-uniform, so causality and the window become a
 * predicate (`ok`) rather than continue/break. At win=1024 with 128 keys per
 * block ~89% of the swept (key,query) rectangle is live, so the dots thrown
 * away cost ~11% -- less than the occupancy buys back. Both forms stay
 * reachable: CUDALM_DKV_SPLIT=1 selects the kernel above.
 *
 * CRITICAL: the BLOCK grows to LQ_TPB*SPL threads so that keys per block stays
 * at LQ_TPB. The first version instead kept 128 threads and covered LQ_TPB/SPL
 * keys -- and measured 2.4x SLOWER. Staging sQ/sG is sized by the swept QUERY
 * range, which does not shrink with the key span, so halving keys-per-block
 * doubled the global traffic for Q and dA. This kernel is bound by that
 * staging, not by occupancy; the split has to be paid for in threads, not in
 * key coverage. */
template<int HD,bool PACKED,int SPL>
__global__ __launch_bounds__(LQ_TPB*SPL,2)
void k_attn_bwd_dkv_sub(const __half*Q,const __half*K,const __half*V,const __half*dA,
                        const float*lse,const float*Drow,float*dK,float*dV,int win,
                        int B,int T,int H,int KV,int D,int KD){
    enum { HS=HD/SPL, NTH=LQ_TPB*SPL };        /* dims per thread, threads per block */
    __shared__ __align__(8) __half sQ[LQ_KT*HD], sG[LQ_KT*HD];
    int bkv=blockIdx.y, kvh=bkv%KV, b=bkv/KV, grp=H/KV;
    int sub=threadIdx.x%SPL, slot=threadIdx.x/SPL;   /* SPL-aligned within the warp */
    int sblk=blockIdx.x*LQ_TPB, s=sblk+slot;         /* key span unchanged vs SPL=1 */
    int act=(s<T);
    float scale=rsqrtf((float)HD);
    __half2 kr[HS/2],vr[HS/2]; float dk[HS],dv[HS];
    const __half2*kp=(const __half2*)(K+(size_t)(b*T+(act?s:0))*KD+kvh*HD+sub*HS);
    const __half2*vp=(const __half2*)(V+(size_t)(b*T+(act?s:0))*KD+kvh*HD+sub*HS);
    __half2 z=__floats2half2_rn(0.f,0.f);
    #pragma unroll
    for(int d=0;d<HS/2;d++){ kr[d]=act?kp[d]:z; vr[d]=act?vp[d]:z; }
    #pragma unroll
    for(int d=0;d<HS;d++){ dk[d]=0.f; dv[d]=0.f; }
    int loT=sblk, hiT=sblk+LQ_TPB-1+(win>0?win-1:T); if(hiT>T-1) hiT=T-1;
    for(int hh=0;hh<grp;hh++){ int h=kvh*grp+hh;
        for(int t0=loT;t0<=hiT;t0+=LQ_KT){
            int n=LQ_KT; if(t0+n>T) n=T-t0;
            __syncthreads();                 /* previous tile's reads must be done */
            #pragma unroll 1
            for(int i=threadIdx.x;i<LQ_KT*HD;i+=NTH){
                int r=i/HD, d=i-r*HD, t=t0+r; __half qq=__float2half(0.f),gg=qq;
                if(r<n){ qq=Q [(size_t)(b*T+t)*D+h*HD+d]; gg=dA[(size_t)(b*T+t)*D+h*HD+d]; }
                sQ[i]=qq; sG[i]=gg;
            }
            __syncthreads();
            /* NOTE: every thread walks every r, inactive ones included. The
             * butterfly below is a warp-wide collective and diverging here is
             * undefined, so the masking is a predicate, not a branch. */
            for(int r=0;r<n;r++){
                int t=t0+r;
                int ok = act && t>=s && (win<=0 || t<=s+win-1);
                const __half2*qr=(const __half2*)(sQ+r*HD+sub*HS);
                const __half2*gr=(const __half2*)(sG+r*HD+sub*HS);
                float qk=dot_h2<HS,PACKED>(qr,kr);
                float gv=dot_h2<HS,PACKED>(gr,vr);
                #pragma unroll
                for(int o=1;o<SPL;o<<=1){    /* vanishes entirely at SPL=1 */
                    qk+=__shfl_xor_sync(0xffffffff,qk,o);
                    gv+=__shfl_xor_sync(0xffffffff,gv,o);
                }
                if(!ok) continue;
                float L=lse[(size_t)(b*H+h)*T+t], dr=Drow[(size_t)(b*H+h)*T+t];
                float p=expf(qk*scale-L), ds=p*(gv-dr)*scale;
                #pragma unroll
                for(int d=0;d<HS/2;d++){
                    float2 q2=__half22float2(qr[d]), g2=__half22float2(gr[d]);
                    dk[2*d]+=ds*q2.x; dk[2*d+1]+=ds*q2.y;
                    dv[2*d]+=p *g2.x; dv[2*d+1]+=p *g2.y;
                }
            }
        } }
    if(!act) return;
    float*dkp=dK+(size_t)(b*T+s)*KD+kvh*HD+sub*HS;
    float*dvp=dV+(size_t)(b*T+s)*KD+kvh*HD+sub*HS;
    #pragma unroll
    for(int d=0;d<HS;d++){ dkp[d]+=dk[d]*g_dsc[1]; dvp[d]+=dv[d]*g_dsc[1]; }
}

/* returns 0 if this head dim has no instantiation, so the caller can fall back */
static int attn_fwd_lane(const __half*Q,const __half*K,const __half*V,float*O,float*lse,
                         int win,int B,int T,int H,int KV,int hd,int D,int KD){
    dim3 gr((unsigned)nblk((size_t)T,LQ_TPB),(unsigned)(B*H));
    #define LQ_CASE(N) case N: k_attn_fwd_lane<N><<<gr,LQ_TPB>>>(Q,K,V,O,lse,win,B,T,H,KV,D,KD); return 1;
    switch(hd){ LQ_CASE(64) LQ_CASE(32) LQ_CASE(16) LQ_CASE(8) default: return 0; }
    #undef LQ_CASE
}
/* `packed` picks the __hfma2 dot (see dot_h2). Passed explicitly rather than
 * read from the global so selftest can measure BOTH forms in one run. */
static int attn_bwd_dq_lane(const __half*Q,const __half*K,const __half*V,const __half*dA,
                            const float*lse,float*dQ,const float*Drow,int win,
                            int B,int T,int H,int KV,int hd,int D,int KD,int packed){
    dim3 gr((unsigned)nblk((size_t)T,LQ_TPB),(unsigned)(B*H));
    #define LQ_DQ(N) case N: if(packed) k_attn_bwd_dq_lane<N,true ><<<gr,LQ_TPB>>>(Q,K,V,dA,lse,dQ,Drow,win,B,T,H,KV,D,KD); \
                             else       k_attn_bwd_dq_lane<N,false><<<gr,LQ_TPB>>>(Q,K,V,dA,lse,dQ,Drow,win,B,T,H,KV,D,KD); return 1;
    switch(hd){ LQ_DQ(64) LQ_DQ(32) LQ_DQ(16) LQ_DQ(8) default: return 0; }
    #undef LQ_DQ
}
/* spl: 1 = one thread per key (k_attn_bwd_dkv_lane), 2 or 4 = sub-lane split.
 * Passed explicitly rather than read from the global so selftest and bench can
 * measure every form in one run. */
static int attn_bwd_dkv_lane(const __half*Q,const __half*K,const __half*V,const __half*dA,
                             const float*lse,const float*Drow,float*dK,float*dV,int win,
                             int B,int T,int H,int KV,int hd,int D,int KD,int packed,int spl){
    #define KVARGS Q,K,V,dA,lse,Drow,dK,dV,win,B,T,H,KV,D,KD
    if(spl<=1){
        dim3 gr((unsigned)nblk((size_t)T,LQ_TPB),(unsigned)(B*KV));
        #define LQ_KV(N) case N: if(packed) k_attn_bwd_dkv_lane<N,true ><<<gr,LQ_TPB>>>(KVARGS); \
                                 else       k_attn_bwd_dkv_lane<N,false><<<gr,LQ_TPB>>>(KVARGS); return 1;
        switch(hd){ LQ_KV(64) LQ_KV(32) LQ_KV(16) LQ_KV(8) default: return 0; }
        #undef LQ_KV
    }
    if(spl!=2 && spl!=4) return 0;
    if(hd%(2*spl)) return 0;              /* HS must be even for the __half2 loads */
    /* same key span per block as spl=1 -- the extra threads come from a wider
     * block, NOT from covering fewer keys (see the kernel's note) */
    dim3 gr((unsigned)nblk((size_t)T,LQ_TPB),(unsigned)(B*KV));
    #define LQ_SUB(N,S) if(packed) k_attn_bwd_dkv_sub<N,true ,S><<<gr,LQ_TPB*S>>>(KVARGS); \
                        else       k_attn_bwd_dkv_sub<N,false,S><<<gr,LQ_TPB*S>>>(KVARGS); return 1;
    #define LQ_KV(N) case N: if(spl==2){ LQ_SUB(N,2) } else { LQ_SUB(N,4) }
    switch(hd){ LQ_KV(64) LQ_KV(32) LQ_KV(16) LQ_KV(8) default: return 0; }
    #undef LQ_KV
    #undef LQ_SUB
    #undef KVARGS
}

__global__ void k_swiglu(const float*G,const float*U,float*O,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float z=G[i]; O[i]=(z/(1.0f+expf(-z)))*U[i]; }
}
__global__ void k_swiglu_bwd(const float*Gg,const float*U,const float*dact,float*dg,float*du,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float z=Gg[i],sig=1.0f/(1.0f+expf(-z)),silu=z*sig,ds=sig+z*sig*(1.0f-sig);
    dg[i]=dact[i]*U[i]*ds; du[i]=dact[i]*silu;
}
/* float4 variants for the MoE backward: same per-element math, 4 elems/thread */
__global__ void k_swiglu4(const float4*G,const float4*U,float4*O,size_t n4){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n4)return;
    float4 z=G[i],u=U[i],o;
    o.x=(z.x/(1.0f+expf(-z.x)))*u.x; o.y=(z.y/(1.0f+expf(-z.y)))*u.y;
    o.z=(z.z/(1.0f+expf(-z.z)))*u.z; o.w=(z.w/(1.0f+expf(-z.w)))*u.w; O[i]=o;
}
__device__ __forceinline__ void swb1(float z,float u,float da,float*dg,float*du){
    float sig=1.0f/(1.0f+expf(-z)),silu=z*sig,ds=sig+z*sig*(1.0f-sig); *dg=da*u*ds; *du=da*silu;
}
__global__ void k_swiglu_bwd4(const float4*Gg,const float4*U,const float4*dact,float4*dg,float4*du,size_t n4){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n4)return;
    float4 z=Gg[i],u=U[i],a=dact[i],g,v;
    swb1(z.x,u.x,a.x,&g.x,&v.x); swb1(z.y,u.y,a.y,&g.y,&v.y);
    swb1(z.z,u.z,a.z,&g.z,&v.z); swb1(z.w,u.w,a.w,&g.w,&v.w);
    dg[i]=g; du[i]=v;
}
#else
#define CPUK_PART 1
#include "cpu_kern.h"
#endif
/* ---------------------------- MoE (top-1) --------------------------------
 * Mirrors moe_forward/moe_backward in tinylm.c op-for-op. Tokens are permuted
 * so every expert's rows are contiguous, which turns the variable-size
 * per-expert FFN into a plain cuBLAS GEMM over a slice. The permutation is
 * built on the host from `assign` (N ints, ~40 KB) -- deterministic, matches
 * the CPU ordering exactly, and negligible next to a 3 s step. */
/* CUDALM_ROUTE (eval-only diagnostic; leave unset for training)
 *   0 = learned routing, the normal path
 *   1 = every token to expert 0
 *   2 = every token to a deterministic pseudo-random expert
 * Both overrides force gate=1 so the FFN contribution keeps its usual scale --
 * otherwise the ablation would also be scaling the branch down and the result
 * would say nothing about routing.
 *
 * The point: a Switch-style aux loss FORCES balanced utilisation, so "healthy
 * routing" in eval is consistent with eight interchangeable experts. If val loss
 * barely moves when the router's choice is discarded, the experts are not
 * specialised and buying more of them adds capacity nothing can exploit. */
__device__ int g_route_mode_d;
static int g_route_mode=0;              /* host copy; uploaded by cuda_init */
static int g_t1old=0;                   /* CUDALM_TOP1_OLD=1: top-1 keeps its own fp32 path; default runs it as top-K with K=1 (bf16 fused experts, async counts) */
static int g_topk=2;                    /* CUDALM_TOPK: experts routed per token (= max over layers) */
static int g_topkL[64];                 /* CUDALM_TOPK_L: per-layer K (0 = g_topk) */
static float g_mbias_u=0.001f;          /* CUDALM_MOE_BIAS: aux-free balance step u (0 = off) */
static float *g_mbiasH=0,*g_mbiasD=0;   /* selection-only router bias [L*E] */
static int g_mbias_ld=0;                /* bias loaded from .t1: applied even when u=0 */
/* balance levers (cudalm15): BIAS_P proportional bias step, BIAS_SLOT per-slot bias [L*Km*E],
 * RCAP capacity reroute, SINK Sinkhorn selection (both train-only), AUX_L per-layer aux on top of pool aux */
static int g_mbias_p=1,g_mbias_slot=1,g_moe_sink=0,g_moe_train=0; static float g_moe_rcap=0,g_moe_auxL=0;
/* CUDALM_MOE_AUX_LM / CUDALM_MOE_BIAS_LM: per-layer multipliers on the aux loss and the bias step (missing = 1) */
static float g_auxlm[64],g_biaslm[64]; static int g_auxlm_n=0,g_biaslm_n=0;
#define LMUL(v,n,l) ((l)<(n)?(v)[l]:1.0f)
static int parse_flist(const char*s,float*v,int max){ int n=0; while(s&&*s&&n<max){ v[n++]=(float)atof(s); s=strchr(s,','); if(s)s++; } return n; }
static double g_sk_r[2]; static long g_sk_n[2],g_rb_mv,g_rb_tot;
static unsigned short*g_kdI=0,*g_kdP=0; static int g_kdK=0,g_kdrn=0; static float g_kda=0.f;   /* CUDALM_KD, see k_kd_adj (GPU only) */
#ifndef CUDALM_CPU
__global__ void k_router_top1(const float*rt,float*rprobs,int*assign,float*gate,int N,int E,const float*bias){
    int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N)return;
    const float*r=rt+(size_t)n*E; float*pr=rprobs+(size_t)n*E;
    float mx=-1e30f; for(int e=0;e<E;e++) if(r[e]>mx)mx=r[e];
    float s=0; for(int e=0;e<E;e++){ pr[e]=expf(r[e]-mx); s+=pr[e]; }
    float inv=1.0f/s; int best=0;
    for(int e=0;e<E;e++){ pr[e]*=inv; if(pr[e]+(bias?bias[e]:0.f)>pr[best]+(bias?bias[best]:0.f))best=e; }
    int mode=g_route_mode_d;
    if(mode==1){ assign[n]=0; gate[n]=1.0f; return; }
    if(mode==2){ unsigned h=(unsigned)n*2654435761u; h^=h>>15;
                 assign[n]=(int)(h%(unsigned)E); gate[n]=1.0f; return; }
    assign[n]=best; gate[n]=pr[best];
}
/* Top-K router (K>=2). Softmax over all E, then pick the K highest-prob experts.
 * Output is pass-major: expert index in assign[k*N+n], its RAW softmax prob in
 * gate[k*N+n] (no top-K renormalisation -- keeps each pass's gate gradient the
 * plain softmax jacobian, so backward is K independent top-1 gate paths summed).
 * The FFN branch is therefore scaled by sum_k p_k (<1); a learnable Wr absorbs
 * it and the two-pass sum still doubles the active FFN vs top-1, which is the
 * point of top-2. route_mode diagnostics stay on the top-1 kernel only. */
__global__ void k_router_topk(const float*rt,float*rprobs,int*assign,float*gate,int N,int E,int K,const float*bias,int bs){
    int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N)return;
    const float*r=rt+(size_t)n*E; float*pr=rprobs+(size_t)n*E;
    float mx=-1e30f; for(int e=0;e<E;e++) if(r[e]>mx)mx=r[e];
    float s=0; for(int e=0;e<E;e++){ pr[e]=expf(r[e]-mx); s+=pr[e]; }
    float inv=1.0f/s; for(int e=0;e<E;e++) pr[e]*=inv;
    int chosen[8];                                   /* K capped at 8 */
    for(int k=0;k<K;k++){
        int best=-1; float bp=-1e30f;
        for(int e=0;e<E;e++){
            int taken=0; for(int j=0;j<k;j++) if(chosen[j]==e){ taken=1; break; }
            if(taken) continue;
            float sc=pr[e]+(bias?bias[k*bs+e]:0.f);
            if(sc>bp){ bp=sc; best=e; }
        }
        if(best<0) best=k;                           /* NaN scores: stay in range, the step gets skipped */
        chosen[k]=best;
        assign[(size_t)k*N+n]=best; gate[(size_t)k*N+n]=pr[best];
    }
}
/* Xp[j,:] = src[perm[j],:] */
__global__ void k_gather(const float*src,const int*perm,float*Xp,int N,int D){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(size_t)N*D)return;
    int j=(int)(i/D), d=(int)(i%D), n=perm[j]; Xp[i]=n<0?0.f:src[(size_t)n*D+d];
}
/* out[perm[j],:] += gate[perm[j]] * yp[j,:]   (each j owns a distinct row) */
__global__ void k_scatter_gate(const float*yp,const int*perm,const float*gate,float*out,int N,int D){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(size_t)N*D)return;
    int j=(int)(i/D), d=(int)(i%D); int n=perm[j]; if(n<0)return;
    out[(size_t)n*D+d]+=gate[n]*yp[i];
}
/* tgtk[j] = tgt[keep[j]] -- companion to k_gather for the head's row compaction */
__global__ void k_gather_i(const int*src,const int*idx,int*dst,int n){
    int j=blockIdx.x*blockDim.x+threadIdx.x; if(j<n) dst[j]=src[idx[j]];
}
/* dst[idx[j],:] = src[j,:] (assign, not accumulate -- dst is zeroed first) */
__global__ void k_scatter_rows(const float*src,const int*idx,float*dst,int n,int D){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(size_t)n*D)return;
    int j=(int)(i/D), d=(int)(i%D); dst[(size_t)idx[j]*D+d]=src[i];
}
/* row-per-block float4 gather / scatter-add (D4 = D/4) */
#define RB4(D4) ((((D4)+31)&~31)>256?256:(((D4)+31)&~31))
__global__ void k_gather4(const float4*src,const int*perm,float4*Xp,int D4){
    int j=blockIdx.x, n=perm[j]; float4*o=Xp+(size_t)j*D4;
    if(n<0){ for(int d=threadIdx.x;d<D4;d+=blockDim.x) o[d]=make_float4(0,0,0,0); return; }
    const float4*s=src+(size_t)n*D4;
    for(int d=threadIdx.x;d<D4;d+=blockDim.x) o[d]=s[d];
}
__global__ void k_scatter_add4(const float4*dXp,const int*perm,float4*dst,int D4){
    int j=blockIdx.x; if(perm[j]<0)return; float4*o=dst+(size_t)perm[j]*D4; const float4*s=dXp+(size_t)j*D4;
    for(int d=threadIdx.x;d<D4;d+=blockDim.x){ float4 a=o[d],b=s[d];
        a.x+=b.x; a.y+=b.y; a.z+=b.z; a.w+=b.w; o[d]=a; }
}
/* k_moe_dgate + (optionally) the Xp gather in one row pass */
__global__ void k_moe_dgate4(const float4*dout,const float4*yp,const int*perm,const float*gate,
                             float*d_gate,float4*dyp,const float4*src,float4*Xp,int D4){
    int j=blockIdx.x, n=perm[j]; float4*dyR=dyp+(size_t)j*D4;
    if(n<0){ float4 z=make_float4(0,0,0,0);
        for(int d=threadIdx.x;d<D4;d+=blockDim.x){ dyR[d]=z; if(src) Xp[(size_t)j*D4+d]=z; } return; }
    const float4*doR=dout+(size_t)n*D4,*yR=yp+(size_t)j*D4;
    __shared__ float sh[8]; float acc=0, g=gate[n];
    for(int d=threadIdx.x;d<D4;d+=blockDim.x){ float4 a=doR[d],b=yR[d];
        acc+=a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;
        a.x*=g; a.y*=g; a.z*=g; a.w*=g; dyR[d]=a;
        if(src) Xp[(size_t)j*D4+d]=src[(size_t)n*D4+d]; }
    for(int off=16;off>0;off>>=1) acc+=__shfl_down_sync(0xffffffff,acc,off);
    if((threadIdx.x&31)==0) sh[threadIdx.x>>5]=acc;
    __syncthreads();
    if(threadIdx.x==0){ float s=0; int nw=blockDim.x>>5; for(int i=0;i<nw;i++)s+=sh[i]; d_gate[n]=s; }
}
/* d_fnorm[perm[j],:] += dXp[j,:] */
__global__ void k_scatter_add(const float*dXp,const int*perm,float*dst,int N,int D){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(size_t)N*D)return;
    int j=(int)(i/D), d=(int)(i%D); dst[(size_t)perm[j]*D+d]+=dXp[i];
}
/* per permuted row: d_gate[n] = dot(dout[n], yp[j]) ; dyp[j] = gate[n]*dout[n] */
__global__ void k_moe_dgate(const float*dout,const float*yp,const int*perm,const float*gate,
                            float*d_gate,float*dyp,int N,int D){
    int j=blockIdx.x; if(j>=N)return;
    int n=perm[j];
    const float*doR=dout+(size_t)n*D,*yR=yp+(size_t)j*D; float*dyR=dyp+(size_t)j*D;
    __shared__ float sh[32]; float acc=0;
    for(int d=threadIdx.x;d<D;d+=blockDim.x) acc+=doR[d]*yR[d];
    for(int off=16;off>0;off>>=1) acc+=__shfl_down_sync(0xffffffff,acc,off);
    if((threadIdx.x&31)==0) sh[threadIdx.x>>5]=acc;
    __syncthreads();
    if(threadIdx.x==0){ float t=0; int nw=(blockDim.x+31)/32; for(int i=0;i<nw;i++)t+=sh[i]; d_gate[n]=t; }
    float g=gate[n];
    for(int d=threadIdx.x;d<D;d+=blockDim.x) dyR[d]=g*doR[d];
}
/* router backward: load-balance aux + gate path, then softmax back -> drt */
__global__ void k_router_bwd(const float*rprobs,const int*assign,const float*d_gate,
                             const int*cnt,float*drt,int N,int E,float aux){
    int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N)return;
    const float*pr=rprobs+(size_t)n*E; float*d=drt+(size_t)n*E;
    float invN=1.0f/N, dot=0;
    for(int e=0;e<E;e++){ float dp=aux*E*((float)cnt[e]*invN)*invN;
        if(e==assign[n]) dp+=d_gate[n];
        d[e]=dp; dot+=pr[e]*dp; }
    for(int e=0;e<E;e++) d[e]=pr[e]*(d[e]-dot);
}
/* Top-K router backward. One softmax jacobian per token folds in every pass's
 * gate gradient (d_gate is pass-major [k*N+n], assign matches) plus the
 * load-balance aux. cnt[e] is the COMBINED dispatch count over all K passes;
 * the aux fraction f_e = cnt[e]/(K*N) while the mean-prob term keeps its 1/N,
 * so at K=1 this reduces to k_router_bwd exactly (aux*E*cnt[e]/N^2). */
__global__ void k_router_bwd_topk(const float*rprobs,const int*assign,const float*d_gate,
                                  const float*w,float*drt,int N,int E,int K){
    int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N)return;
    const float*pr=rprobs+(size_t)n*E; float*d=drt+(size_t)n*E; float dot=0;
    for(int e=0;e<E;e++) d[e]=w[e];                  /* aux*E*f_e/N, built on the host */
    for(int k=0;k<K;k++){ int e=assign[(size_t)k*N+n]; d[e]+=d_gate[(size_t)k*N+n]; }
    for(int e=0;e<E;e++) dot+=pr[e]*d[e];
    for(int e=0;e<E;e++) d[e]=pr[e]*(d[e]-dot);
}
__global__ void k_add(float*A,const float*B,size_t n){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)A[i]+=B[i]; }
__global__ void k_copy(float*A,const float*B,size_t n){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)A[i]=B[i]; }
__global__ void k_scale(float*G,size_t n,float s){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)G[i]*=s; }
__global__ void k_scale_d(float*G,size_t n,const float*s){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)G[i]*=*s; }
/* softmax + CE, one block per row. nAll = the FULL batch size, so the 1/N
 * gradient scale matches the unchunked reference exactly. */
/* q8 != NULL: also write dl as E4M3 scaled by 448*nAll. |dl| <= 1/nAll, so this
 * static scale equals the per-tensor absmax scale without the extra passes. */
/* wt (optional): per-row loss weight in [-1,1] (CUDALM_TOKW policy gradient); a negative row whose target p is
 * already below negp gets no gradient, so failed rollouts are pushed down only while still likely */
/* wt <= -50 encodes a teacher log p + baseline (wt + 100, on-policy distillation): w = clamp(that - log p_student, -1, 1);
 * d_opd accumulates the advantage (one block per row) so the host can center it */
__device__ float d_opd[2];
__device__ __forceinline__ float tokw_w(const float*wt,int n,float ptg,float negp){
    float w=wt?wt[n]:1.f; if(w<=-50.f){ float g=w+100.f-logf(ptg+1e-12f); if(!threadIdx.x){ atomicAdd(d_opd,g); atomicAdd(d_opd+1,1.f); }
        w=fminf(1.f,fmaxf(-1.f,g)); }
    return w<0.f&&ptg<negp?0.f:w; }
template<class TL> __global__ void k_softmax_ce(const TL*lg,const int*tgt,float*loss,float*dl,__nv_fp8_e4m3*q8,int rows,int V,int nAll,const float*wt,float negp){
    int n=blockIdx.x; if(n>=rows)return;
    const TL*l=lg+(size_t)n*V; float*d=dl+(size_t)n*V;
    __nv_fp8_e4m3*q=q8?q8+(size_t)n*V:NULL; float qs=448.0f*nAll;
    /* tgt < 0 = MASKED position: contribute no loss and no gradient. Used to
     * stop the model being trained to produce the USER's turns -- which is why
     * it kept emitting <|user|> and answering its own questions. dl must still
     * be zeroed, since gpu_head_bwd consumes the whole chunk unconditionally. */
    if(tgt[n]<0){
        for(int i=threadIdx.x;i<V;i+=blockDim.x){ d[i]=0.f; if(q) q[i]=__nv_fp8_e4m3(0.f); }
        if(threadIdx.x==0) loss[n]=0.f;
        return;
    }
    __shared__ float sh[32]; float mx=-1e30f;
    for(int i=threadIdx.x;i<V;i+=blockDim.x){ mx=fmaxf(mx,(float)l[i]); }
    for(int off=16;off>0;off>>=1) mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,off));
    if((threadIdx.x&31)==0) sh[threadIdx.x>>5]=mx;
    __syncthreads();
    if(threadIdx.x==0){ float t=-1e30f; int nw=(blockDim.x+31)/32; for(int i=0;i<nw;i++)t=fmaxf(t,sh[i]); sh[0]=t; }
    __syncthreads();
    float m=sh[0],s=0;
    __syncthreads();          /* every thread must read sh[0] before it is reused below */
    for(int i=threadIdx.x;i<V;i+=blockDim.x) s+=expf((float)l[i]-m);
    for(int off=16;off>0;off>>=1) s+=__shfl_down_sync(0xffffffff,s,off);
    if((threadIdx.x&31)==0) sh[threadIdx.x>>5]=s;
    __syncthreads();
    if(threadIdx.x==0){ float t=0; int nw=(blockDim.x+31)/32; for(int i=0;i<nw;i++)t+=sh[i]; sh[0]=t; }
    __syncthreads();
    int tg=tgt[n]; float inv=1.0f/sh[0], ptg=expf((float)l[tg]-m)*inv, w=tokw_w(wt,n,ptg,negp);
    float iN=w/nAll;
    for(int i=threadIdx.x;i<V;i+=blockDim.x){ float v=expf((float)l[i]-m)*inv*iN; d[i]=v; if(q) q[i]=__nv_fp8_e4m3(v*qs); }
    __syncthreads();          /* d[tg] must be written before thread 0 corrects it */
    if(threadIdx.x==0){ d[tg]-=iN; loss[n]=-logf(ptg+1e-12f);
        if(q) q[tg]=__nv_fp8_e4m3(d[tg]*qs); }
}
/* logit lens: per row out[4n..] = logP(tgt), argmax, entropy (nats), logP(fin) (the full model's top-1; own top-1 if fin NULL) */
__global__ void k_lens(const float*lg,const int*tgt,const int*fin,float*out,int rows,int V){
    int n=blockIdx.x; if(n>=rows) return; const float*l=lg+(size_t)n*V;
    __shared__ float sv[32],ss[32],se[32]; __shared__ int si[32]; int w=threadIdx.x>>5, nw=(blockDim.x+31)>>5;
    float mx=-1e30f; int mi=0;
    for(int i=threadIdx.x;i<V;i+=blockDim.x) if(l[i]>mx){ mx=l[i]; mi=i; }
    for(int k=16;k;k>>=1){ float v=__shfl_down_sync(~0u,mx,k); int j=__shfl_down_sync(~0u,mi,k); if(v>mx||(v==mx&&j<mi)){ mx=v; mi=j; } }
    if(!(threadIdx.x&31)){ sv[w]=mx; si[w]=mi; }
    __syncthreads();
    if(!threadIdx.x) for(int k=1;k<nw;k++) if(sv[k]>sv[0]||(sv[k]==sv[0]&&si[k]<si[0])){ sv[0]=sv[k]; si[0]=si[k]; }
    __syncthreads();
    float m=sv[0],s=0.f,e=0.f;
    for(int i=threadIdx.x;i<V;i+=blockDim.x){ float x=l[i]-m, p=expf(x); s+=p; e+=p*x; }
    for(int k=16;k;k>>=1){ s+=__shfl_down_sync(~0u,s,k); e+=__shfl_down_sync(~0u,e,k); }
    if(!(threadIdx.x&31)){ ss[w]=s; se[w]=e; }
    __syncthreads();
    if(!threadIdx.x){ for(int k=1;k<nw;k++){ ss[0]+=ss[k]; se[0]+=se[k]; }
        float lse=m+logf(ss[0]); float*o=out+4*(size_t)n;
        o[0]=l[tgt[n]]-lse; o[1]=(float)si[0]; o[2]=logf(ss[0])-se[0]/ss[0]; o[3]=fin?l[fin[n]]-lse:-logf(ss[0]); }
}
/* tuned lens: row log-softmax (the full model's reference distribution) */
__device__ float blk_reduce(float v,float*sh,int mx){
    for(int k=16;k;k>>=1){ float o=__shfl_down_sync(~0u,v,k); v=mx?fmaxf(v,o):v+o; }
    int w=threadIdx.x>>5, nw=(blockDim.x+31)>>5; __syncthreads();
    if(!(threadIdx.x&31)) sh[w]=v; __syncthreads();
    if(!threadIdx.x){ for(int k=1;k<nw;k++) sh[0]=mx?fmaxf(sh[0],sh[k]):sh[0]+sh[k]; } __syncthreads();
    return sh[0]; }
__global__ void k_logsm(const float*lg,float*out,int rows,int V){
    int n=blockIdx.x; if(n>=rows) return; const float*l=lg+(size_t)n*V; float*o=out+(size_t)n*V; __shared__ float sh[32];
    float m=-1e30f; for(int i=threadIdx.x;i<V;i+=blockDim.x) m=fmaxf(m,l[i]); m=blk_reduce(m,sh,1);
    float s=0.f; for(int i=threadIdx.x;i<V;i+=blockDim.x) s+=expf(l[i]-m); s=blk_reduce(s,sh,0);
    float lse=m+logf(s); for(int i=threadIdx.x;i<V;i+=blockDim.x) o[i]=l[i]-lse; }
/* KL(p || softmax(lg)) per row with p = exp(lp); lg is overwritten by its gradient (q - p) * sc */
__global__ void k_kld(float*lg,const float*lp,float*loss,int rows,int V,float sc){
    int n=blockIdx.x; if(n>=rows) return; float*l=lg+(size_t)n*V; const float*p=lp+(size_t)n*V; __shared__ float sh[32];
    float m=-1e30f; for(int i=threadIdx.x;i<V;i+=blockDim.x) m=fmaxf(m,l[i]); m=blk_reduce(m,sh,1);
    float s=0.f; for(int i=threadIdx.x;i<V;i+=blockDim.x) s+=expf(l[i]-m); s=blk_reduce(s,sh,0);
    float lse=m+logf(s), kl=0.f;
    for(int i=threadIdx.x;i<V;i+=blockDim.x){ float pi=expf(p[i]), lq=l[i]-lse; kl+=pi*(p[i]-lq); l[i]=(expf(lq)-pi)*sc; }
    kl=blk_reduce(kl,sh,0); if(!threadIdx.x) loss[n]=kl; }
/* translator: Y = H + H A + b, with Y = H A on entry */
__global__ void k_tl_add(float*Y,const float*H,const float*b,size_t n,int D){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) Y[i]+=H[i]+b[i%D]; }
/* h8+h16 head: online max/sum (2 reads of the fp16 logits, not 3), 8 per
 * thread-iteration, dl as bf16 (|dl|<=1/nAll underflows fp16) + the E4M3 copy.
 * Needs V%8==0. */
__device__ __forceinline__ void ms_merge(float&m,float&s,float m2,float s2){
    float mm=fmaxf(m,m2); s=s*__expf(m-mm)+s2*__expf(m2-mm); m=mm; }
__global__ void k_sce_h8(const __half*lg,const int*tgt,float*loss,bf16*dl,__nv_fp8_e4m3*q8,int V,int nAll,const float*wt,float negp){
    int n=blockIdx.x, V8=V>>3, tg=tgt[n];
    const uint4*l=(const uint4*)(lg+(size_t)n*V); uint4*d=(uint4*)(dl+(size_t)n*V); uint2*q=(uint2*)(q8+(size_t)n*V);
    if(tg<0){ for(int i=threadIdx.x;i<V8;i+=blockDim.x){ d[i]=make_uint4(0,0,0,0); q[i]=make_uint2(0,0); }
        if(!threadIdx.x) loss[n]=0.f; return; }
    float m=-1e30f,s=0.f;
    for(int i=threadIdx.x;i<V8;i+=blockDim.x){ uint4 u=l[i]; const __half*h=(const __half*)&u; float x[8],mm=m;
        for(int j=0;j<8;j++){ x[j]=__half2float(h[j]); mm=fmaxf(mm,x[j]); }
        s*=__expf(m-mm); for(int j=0;j<8;j++) s+=__expf(x[j]-mm); m=mm; }
    for(int k=16;k;k>>=1) ms_merge(m,s,__shfl_xor_sync(~0u,m,k),__shfl_xor_sync(~0u,s,k));
    __shared__ float sm[32],ss[32]; int w=threadIdx.x>>5, nw=blockDim.x>>5;
    if(!(threadIdx.x&31)){ sm[w]=m; ss[w]=s; }
    __syncthreads();
    if(w==0){ m=threadIdx.x<nw?sm[threadIdx.x]:-1e30f; s=threadIdx.x<nw?ss[threadIdx.x]:0.f;
        for(int k=16;k;k>>=1) ms_merge(m,s,__shfl_xor_sync(~0u,m,k),__shfl_xor_sync(~0u,s,k));
        if(!threadIdx.x){ sm[0]=m; ss[0]=s; } }
    __syncthreads();
    m=sm[0]; float inv=1.0f/ss[0], qs=448.0f*nAll, ptg=__expf(__half2float(lg[(size_t)n*V+tg])-m)*inv, wr=tokw_w(wt,n,ptg,negp);
    float iN=wr/nAll;
    for(int i=threadIdx.x;i<V8;i+=blockDim.x){ uint4 u=l[i],o; uint2 oq; const __half*h=(const __half*)&u;
        bf16*ob=(bf16*)&o; __nv_fp8_e4m3*qb=(__nv_fp8_e4m3*)&oq;
        for(int j=0;j<8;j++){ float v=__expf(__half2float(h[j])-m)*inv*iN; if(i*8+j==tg) v-=iN;
            ob[j]=d_f2bf(v); qb[j]=__nv_fp8_e4m3(v*qs); }
        d[i]=o; q[i]=oq; }
    if(!threadIdx.x) loss[n]=-logf(ptg+1e-12f);
}
/* CUDALM_KD (20n): cached teacher top-K per input position (cudalm kdcache). After the CE kernel the head
 * gradient p - onehot becomes p - (1-a) onehot - a t: CE against the mixed soft target. 20o default t = the raw
 * teacher top-K + its missing mass 1-sum(top-K) on the true token (rn=0); rn=1 = the 20n top-K renormalised to 1,
 * which gives the true token no teacher mass whenever it is outside the top-K (~28% of positions at K=8).
 * One thread per row, K+1 entries; the E4M3 copy of the patched entries is refreshed. */
__global__ void k_kd_adj(float*dl,bf16*dlb,__nv_fp8_e4m3*q8,const int*tgt,const unsigned short*ki,const __half*kp,int rows,int V,int K,int nAll,float a,int rn){
    int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>=rows||tgt[n]<0) return;
    const __half*p=kp+(size_t)n*K; const unsigned short*id=ki+(size_t)n*K; float s=0.f, iN=a/nAll, qs=448.0f*nAll;
    for(int k=0;k<K;k++) s+=__half2float(p[k]);
    if(s<=0.f) return;
    float sc=rn?1.f/s:1.f, ty=rn?1.f:fminf(s,1.f);
    for(int k=-1;k<K;k++){ size_t x=(size_t)n*V+(k<0?tgt[n]:id[k]); float dv=k<0?iN*ty:-iN*__half2float(p[k])*sc, nv;
        if(dlb){ nv=d_bf2f(dlb[x])+dv; dlb[x]=d_f2bf(nv); } else { nv=dl[x]+dv; dl[x]=nv; }
        if(q8) q8[x]=__nv_fp8_e4m3(nv*qs); } }
/* kdcache: top-K (K <= 16) of one logits row with their full-softmax probabilities, one block per row */
__global__ void k_topk_row(const float*lg,int V,int K,unsigned short*oi,__half*op){
    int n=blockIdx.x; const float*l=lg+(size_t)n*V; float m=-1e30f,s=0.f,tv[16]; int ti[16],nt=0;
    for(int i=threadIdx.x;i<V;i+=blockDim.x){ float x=l[i],mm=fmaxf(m,x); s=s*__expf(m-mm)+__expf(x-mm); m=mm;
        if(nt<K||x>tv[nt-1]){ int j=nt<K?nt++:K-1; while(j>0&&tv[j-1]<x){ tv[j]=tv[j-1]; ti[j]=ti[j-1]; j--; } tv[j]=x; ti[j]=i; } }
    for(int k=16;k;k>>=1) ms_merge(m,s,__shfl_xor_sync(~0u,m,k),__shfl_xor_sync(~0u,s,k));
    __shared__ float sm[32],ss[32],wv[32]; __shared__ int wi[32],win; int w=threadIdx.x>>5, nw=blockDim.x>>5, ptr=0;
    if(!(threadIdx.x&31)){ sm[w]=m; ss[w]=s; }
    __syncthreads();
    if(w==0){ m=threadIdx.x<nw?sm[threadIdx.x]:-1e30f; s=threadIdx.x<nw?ss[threadIdx.x]:0.f;
        for(int k=16;k;k>>=1) ms_merge(m,s,__shfl_xor_sync(~0u,m,k),__shfl_xor_sync(~0u,s,k));
        if(!threadIdx.x){ sm[0]=m; ss[0]=s; } }
    __syncthreads();
    m=sm[0]; float inv=1.0f/ss[0];
    for(int r=0;r<K;r++){ float v=ptr<nt?tv[ptr]:-1e30f; int t=threadIdx.x;
        for(int k=16;k;k>>=1){ float v2=__shfl_xor_sync(~0u,v,k); int t2=__shfl_xor_sync(~0u,t,k); if(v2>v||(v2==v&&t2<t)){ v=v2; t=t2; } }
        if(!(threadIdx.x&31)){ wv[w]=v; wi[w]=t; }
        __syncthreads();
        if(!threadIdx.x){ int bt=wi[0]; float bv=wv[0]; for(int j=1;j<nw;j++) if(wv[j]>bv||(wv[j]==bv&&wi[j]<bt)){ bv=wv[j]; bt=wi[j]; } win=bt; }
        __syncthreads();
        if(threadIdx.x==win){ oi[(size_t)n*K+r]=(unsigned short)ti[ptr]; op[(size_t)n*K+r]=__float2half(__expf(tv[ptr]-m)*inv); ptr++; }
        __syncthreads(); } }
__global__ void k_f2bf(const float*src,bf16*dst,size_t n){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=d_f2bf(src[i]); }
/* sum of squares -> 64 partials (grad-norm clipping) */
__global__ void k_sqsum(const float*G,size_t n,float*part){
    __shared__ float ls[TPB]; size_t gid=(size_t)blockIdx.x*blockDim.x+threadIdx.x, gs=(size_t)gridDim.x*blockDim.x;
    float s=0,s2=0;
    if(!(((uintptr_t)G)&15)){             /* O4: float4 x2 in flight -- 64 scalar-load blocks reached ~40% of DRAM BW */
        const float4*g4=(const float4*)G; size_t n4=n/4, i=gid;
        for(;i+gs<n4;i+=2*gs){ float4 x=g4[i],y=g4[i+gs]; s+=x.x*x.x+x.y*x.y+x.z*x.z+x.w*x.w; s2+=y.x*y.x+y.y*y.y+y.z*y.z+y.w*y.w; }
        if(i<n4){ float4 x=g4[i]; s+=x.x*x.x+x.y*x.y+x.z*x.z+x.w*x.w; }
        for(size_t j=n4*4+gid;j<n;j+=gs) s+=G[j]*G[j];
    } else for(size_t i=gid;i<n;i+=gs) s+=G[i]*G[i];
    ls[threadIdx.x]=s+s2; __syncthreads();
    for(int o=blockDim.x/2;o>0;o>>=1){ if(threadIdx.x<o) ls[threadIdx.x]+=ls[threadIdx.x+o]; __syncthreads(); }
    if(threadIdx.x==0) part[blockIdx.x]=ls[0];
}
/* S2: finish the clip on the device, same summation order as the old host code.
 * gs[0] = clip multiplier (1 when gn<=1, exact), gs[1] = gn. part = G's 64 partials, then
 * nside more slices of 64 (side parameters): they join the norm when clip_side is set, and a
 * non-finite side sum makes gn NaN either way so the step is skipped. */
__global__ void k_gnorm(const float*part,int nside,int clip_side,float*gs){
    double ss=0,sd=0; for(int i=0;i<64;i++) ss+=part[i];
    for(int i=64;i<64*(1+nside);i++) sd+=part[i];
    float gn=(float)sqrt(clip_side?ss+sd:ss); if(!isfinite(sd)) gn=__int_as_float(0x7fc00000);
    gs[0]=gn>1.0f?1.0f/gn:1.0f; gs[1]=gn;
}
__global__ void k_clipmul(float*gs,float gn){ gs[0]=gn>1.0f?1.0f/gn:1.0f; gs[1]=gn; }
#else
#define CPUK_PART 2
#include "cpu_kern.h"
#endif
/* Per-layer learning-rate scaling. The parameter block is laid out as
 * [emb | L identically-sized layer blocks | final norm | mtp], so a parameter's
 * layer is one division -- no per-parameter table (which would be another copy
 * of the model).
 *
 * Why this exists: `grow` creates new layers as exact identities by zeroing wo
 * and w2, the two projections that write into the residual stream. Measured on
 * LS-GROWN after ~80k post-growth iterations, those projections in the four
 * grown layers were still at 0.35x (wo) and 0.55x (w2) of the layers around
 * them, while their inputs (wq 0.84x, w1 0.98x) had caught up. The layers
 * un-froze but never caught up, so a 1.5x parameter count was worth roughly
 * 1.16x in effect. Zero-init is a slow-escape region -- while wo is small the
 * layer barely moves the loss, so the gradient that would grow it is itself
 * small -- and a single decaying LR gives the youngest layers the least room.
 *
 * lmul == NULL reproduces the old single-LR behaviour exactly. */
#include "qlm.cuh"

/* --- Muon: which params it owns, so AdamW can skip them -------------------
 * Muon owns every 2-D tensor in a layer block; the only non-matrices are the
 * two norm gains an1 (offset 0) and an2 (offset a2r). Embedding, final norm
 * and MTP heads stay on AdamW. */
typedef struct { int on; size_t l0,per; int L,D; size_t a2r; } MuSkip;
static const MuSkip MU_OFF={0,0,0,0,0,0};
__device__ __forceinline__ int muon_owns(size_t i,size_t l0,size_t per,int L,int D,size_t a2r){
    if(i<l0) return 0;
    size_t d=i-l0; if(d>=(size_t)L*per) return 0;
    size_t r=d%per;
    if(r<(size_t)D) return 0;                    /* an1 gain */
    if(r>=a2r && r<a2r+(size_t)D) return 0;      /* an2 gain */
    return 1;
}
#ifndef CUDALM_CPU
__global__ void k_adamw(float*P,const float*G,bf16*M,bf16*V,size_t n,
                        float lr,float b1,float b2,float wd,float bc1,float bc2,
                        const float*lmul,size_t l0,size_t per,int L,MuSkip mu,const float*gs=NULL){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    if(mu.on && muon_owns(i,mu.l0,mu.per,mu.L,mu.D,mu.a2r)) return;
    float lp=lr;
    if(lmul && i>=l0){ size_t d=i-l0; if(d<(size_t)L*per) lp=lr*lmul[d/per]; }
    /* moments live in bf16 but every arithmetic step is fp32 -- only the STORE
     * is narrowed, which is what keeps the 0.4% rounding from compounding */
    float g=gs?G[i]*gs[0]:G[i];
    float m=b1*d_bf2f(M[i])+(1.0f-b1)*g;
    float v=b2*d_bf2f(V[i])+(1.0f-b2)*g*g;
    M[i]=d_f2bf(m); V[i]=d_f2bf(v);
    float mh=m/bc1, vh=v/bc2;
    P[i]-=lp*(mh/(sqrtf(vh)+1e-8f)+wd*P[i]);
}
#else
#define CPUK_PART 3
#include "cpu_kern.h"
#endif

/* ===================== cuBLAS GEMM wrappers ==============================
 * cuBLAS is column-major; our buffers are row-major. Row-major C[M,N] is the
 * same memory as column-major C[N,M] with ld=N, so each call is issued
 * "transposed" with the operands swapped. */
static cublasHandle_t CB;
static const float ONE=1.0f, ZERO=0.0f;
/* C[M,N] = A[M,K] @ B[K,N] */
/* ---- fp16 operands / fp32 accumulate, for cards with tensor cores ---------
 * CUDALM_FP16_GEMM=1 routes every GEMM through cublasGemmEx with CUDA_R_16F
 * inputs, CUDA_R_32F output and CUBLAS_COMPUTE_32F.
 *
 * Why it is a flag and not the default: on THIS box (TU116) there are no tensor
 * cores, so cuBLAS falls back internally and the only effect is the cost of
 * converting operands -- measured earlier as no win, which is why the fp32 path
 * stayed. On a T4 (TU104, same sm_75, but WITH tensor cores) the identical
 * binary dispatches to them. One code path, speedup where the hardware has it.
 *
 * fp32 ACCUMULATE is the load-bearing detail. It is what makes this safe for
 * g_atb, the weight-gradient GEMM that reduces over thousands of rows;
 * CUBLAS_COMPUTE_16F would accumulate in fp16 there and lose the small terms.
 * Only the multiply operands are narrowed.
 *
 * Two scratch slots are enough because no call uses more than two distinct
 * operands, and everything runs on the default stream -- so the conversion for
 * call n+1 cannot start before the GEMM of call n has finished. */
static int g_fp16gemm=0;   /* default OFF: fp16 path trains 0.73 worse (val 8.03 vs 7.28) and is slower; it was also the FP8 unaligned fallback. CUDALM_FP16_GEMM=1 to enable */
static __half *ghA=NULL,*ghB=NULL; static size_t ghAcap=0,ghBcap=0;
static __half* hslot(__half**p,size_t*cap,size_t n){
    if(n>*cap){ if(*p) cudaFree(*p); *p=(__half*)dmalloc(n*2); *cap=n; }
    return *p;
}
static void hcvt(const float*src,__half*dst,size_t n){
    KL(k_f2h,nblk(n,TPB),TPB)(src,dst,n,1.0f);
}
/* generic wrapper: same op flags and leading dimensions as the Sgemm it
 * replaces, so the two paths are interchangeable line for line */
static void gemm_ex(cublasOperation_t opB,cublasOperation_t opA,
                    int N,int M,int K,const float*alpha,
                    const float*B,int ldb,const float*A,int lda,
                    const float*beta,float*C,int ldc,
                    size_t nB,size_t nA,const char*what){
    __half *Bh=hslot(&ghB,&ghBcap,nB), *Ah=hslot(&ghA,&ghAcap,nA);
    hcvt(B,Bh,nB); hcvt(A,Ah,nA);
    ckb(cublasGemmEx(CB,opB,opA,N,M,K,alpha,
                     Bh,CUDA_R_16F,ldb, Ah,CUDA_R_16F,lda, beta,
                     C,CUDA_R_32F,ldc, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP), what);
}
/* ---- FP8 E4M3 operands / fp32 accumulate (cublasLt) -----------------------
 * CUDALM_FP8_GEMM=1 routes ALL three GEMM shapes through cublasLtMatmul with
 * E4M3 operands, per-tensor absmax scaling and fp32 output+accumulate. FP8 on
 * cuBLAS 12.9 / sm_120 supports ONLY op(A)=T, op(B)=N -- verified empirically --
 * so the hardware primitive is R=P@Q^T with both operands' contraction dim
 * contiguous. Stage 1: g_abt/g_abt_b (dgrad TN) feed it directly. Stage 2:
 * g_ab (NN forward) and g_atb (NT wgrad) reach the same primitive via
 * transpose-on-cast (k_qtr_e4m3) of the operand whose K is not contiguous.
 * E4M3 has ~3 mantissa bits (~3-4% per-GEMM error at 1024^3); fp32 accumulate +
 * fp32 master weights keep it trainable. Needs the non-QLM fp32-master path.
 * Dims not divisible by 16 fall back to fp16 gemm_ex. Default OFF. */
static int g_fp8gemm=1;   /* default on: Stage-2 FP8 matches fp32 (val 7.29 vs 7.28); CUDALM_FP8_GEMM=0 to disable */
static int g_fp8dbg=0;   /* CUDALM_FP8_DBG=1: sync+check each FP8 GEMM to surface async faults */
static cublasLtHandle_t LT=NULL;
static void* g_fp8ws=NULL; static size_t g_fp8wscap=0;
static float* g_fp8sc=NULL;   /* [amB,deqB,invB, amA,deqA,invA] device scalars */
static __nv_fp8_e4m3 *g8A=NULL,*g8B=NULL; static size_t g8Acap=0,g8Bcap=0;
static __nv_fp8_e4m3* f8slot(__nv_fp8_e4m3**p,size_t*cap,size_t n){
    if(n>*cap){ if(*p) cudaFree(*p); *p=(__nv_fp8_e4m3*)dmalloc(n); *cap=n; }
    return *p;
}
#ifndef CUDALM_CPU
__global__ void k_absmax(const float* x,size_t n,float* out){
    __shared__ float s[256];
    float m=0.f;
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=(size_t)gridDim.x*blockDim.x)
        m=fmaxf(m,fabsf(x[i]));
    s[threadIdx.x]=m; __syncthreads();
    for(int d=blockDim.x>>1; d>0; d>>=1){ if(threadIdx.x<d) s[threadIdx.x]=fmaxf(s[threadIdx.x],s[threadIdx.x+d]); __syncthreads(); }
    if(threadIdx.x==0) atomicMax((int*)out,__float_as_int(s[0])); /* s[0]>=0: int bits monotonic */
}
__global__ void k_setf(float*p,float v){ *p=v; }
__global__ void k_mkscale(const float* amax,float* deq,float* inv){
    float a=*amax; *deq=(a>0.f? a/448.f:1.f); *inv=(a>0.f? 448.f/a:1.f);
}
__global__ void k_quant_e4m3(const float* x,__nv_fp8_e4m3* q,size_t n,const float* inv){
    float f=*inv;
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=(size_t)gridDim.x*blockDim.x)
        q[i]=__nv_fp8_e4m3(x[i]*f);
}
/* transpose-on-cast: src[rows,cols] row-major (cols contiguous) -> E4M3 dst
 * [cols,rows] row-major (rows contiguous), scaled by *inv. 32x32 tiled, +1 pad
 * to avoid shared-bank conflicts; both global reads and writes coalesced. Lets
 * the K-contiguous FP8 TN primitive consume the NN/NT operands whose contraction
 * dim is NOT contiguous in memory (Stage 2). */
__global__ void k_qtr_e4m3(const float* src,__nv_fp8_e4m3* dst,int rows,int cols,const float* inv){
    __shared__ float t[32][33];
    int x=blockIdx.x*32+threadIdx.x, y=blockIdx.y*32+threadIdx.y;   /* src col,row (load) */
    if(x<cols&&y<rows) t[threadIdx.y][threadIdx.x]=src[(size_t)y*cols+x];
    __syncthreads();
    float f=*inv;
    int r=blockIdx.y*32+threadIdx.x, c=blockIdx.x*32+threadIdx.y;   /* src row,col (store) */
    if(c<cols&&r<rows) dst[(size_t)c*rows+r]=__nv_fp8_e4m3(t[threadIdx.x][threadIdx.y]*f);
}
#else
#define CPUK_PART 4
#include "cpu_kern.h"
#endif
/* ---- S6: delayed FP8 scaling (CUDALM_F8_DELAY) ----
 * In training steps each fp8_prep call site quantizes with the amax its previous
 * step measured (satfinite clamps any growth) and records this step's amax inside
 * the quant kernel, removing the separate absmax read pass. A site is keyed by
 * (src, n, tr, occurrence within the step); unseen sites use the exact path. */
static int g_dscale=0, g_ds_on=0; static long g_ds_step=0;
#ifndef CUDALM_CPU
#define DS_MAX 4096
typedef struct { const float*p; size_t n; int tr,occ; long st; } DSk;
static DSk g_dsk[DS_MAX]; static int g_dsn=0; static float*g_dsh=0;
__global__ void k_dsc(float*prev,float*cur,float*sc){ float a=*prev; sc[0]=a; sc[1]=a>0.f?a/448.f:1.f; sc[2]=a>0.f?448.f/a:1.f; *cur=0.f; }
__global__ void k_cp1(const float*s,float*d){ *d=*s; }
__device__ inline void blk_amax(float m,float*am){
    __shared__ float sh[32];
    for(int o=16;o>0;o>>=1) m=fmaxf(m,__shfl_down_sync(0xffffffffu,m,o));
    int t=threadIdx.x+threadIdx.y*blockDim.x, nt=blockDim.x*blockDim.y;
    if((t&31)==0) sh[t>>5]=m;
    __syncthreads();
    if(t==0){ for(int i=1;i<(nt>>5);i++) m=fmaxf(m,sh[i]); atomicMax((int*)am,__float_as_int(m)); } }
__global__ void k_quant_am(const float* x,__nv_fp8_e4m3* q,size_t n,const float* inv,float*am){
    float f=*inv, m=0.f;
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=(size_t)gridDim.x*blockDim.x){
        float v=x[i]; m=fmaxf(m,fabsf(v)); q[i]=__nv_fp8_e4m3(v*f); }
    blk_amax(m,am); }
__global__ void k_qtr_am(const float* src,__nv_fp8_e4m3* dst,int rows,int cols,const float* inv,float*am){
    __shared__ float t[32][33];
    int x=blockIdx.x*32+threadIdx.x, y=blockIdx.y*32+threadIdx.y; float v=0.f;
    if(x<cols&&y<rows) v=src[(size_t)y*cols+x];
    t[threadIdx.y][threadIdx.x]=v;
    __syncthreads();
    float f=*inv;
    int r=blockIdx.y*32+threadIdx.x, c=blockIdx.x*32+threadIdx.y;
    if(c<cols&&r<rows) dst[(size_t)c*rows+r]=__nv_fp8_e4m3(t[threadIdx.x][threadIdx.y]*f);
    blk_amax(fabsf(v),am); }
/* returns 1 when the delayed path quantized; 0 = caller runs the exact path (then cp1 records amax) */
static int ds_prep(const float*src,__nv_fp8_e4m3*dst,int tr,int rows,int cols,float*sc,float**rec){
    size_t n=(size_t)rows*cols; int occ=0,h=-1; *rec=NULL;
    for(int i=0;i<g_dsn;i++){ DSk*k=&g_dsk[i]; if(k->p==src&&k->n==n&&k->tr==tr&&k->st==g_ds_step) occ++; }
    for(int i=0;i<g_dsn;i++){ DSk*k=&g_dsk[i]; if(k->p==src&&k->n==n&&k->tr==tr&&k->occ==occ){ h=i; break; } }
    if(h<0){ if(g_dsn>=DS_MAX) return 0; h=g_dsn++; g_dsk[h].p=src; g_dsk[h].n=n; g_dsk[h].tr=tr; g_dsk[h].occ=occ; g_dsk[h].st=-9; }
    if(!g_dsh){ g_dsh=(float*)dmalloc(2*DS_MAX*sizeof(float)); cudaMemsetAsync(g_dsh,0,2*DS_MAX*sizeof(float),0); }
    int seen=g_dsk[h].st==g_ds_step-1, par=(int)(g_ds_step&1); float *cur=g_dsh+2*h+par, *prev=g_dsh+2*h+(par^1);
    g_dsk[h].st=g_ds_step;
    if(!seen){ *rec=cur; return 0; }
    KL(k_dsc,1,1)(prev,cur,sc);
    if(tr){ dim3 b(32,32),gr((cols+31)/32,(rows+31)/32); KL(k_qtr_am,gr,b)(src,dst,rows,cols,sc+2,cur); }
    else  { int g=(int)((n+255)/256); if(g>1024)g=1024; if(g<1)g=1; KL(k_quant_am,g,256)(src,dst,n,sc+2,cur); }
    return 1;
}
#endif
/* Quantize one operand to E4M3 with per-tensor absmax scaling. tr=0: plain cast
 * of an already-Kc-contiguous [outer,Kc] matrix. tr=1: transpose-cast a row-major
 * [rows,cols] matrix into [cols,rows] (Kc=rows becomes contiguous). Writes deq
 * scale into sc[deqi], scratch amax/inv into sc[deqi-1]/sc[deqi+1]. */
static void fp8_prep_sc(const float* src,__nv_fp8_e4m3* dst,int tr,int rows,int cols,float* sc){
    size_t n=(size_t)rows*cols; float*rec=NULL;
#ifndef CUDALM_CPU
    if(g_ds_on && ds_prep(src,dst,tr,rows,cols,sc,&rec)) return;
#endif
    int g=(int)((n+255)/256); if(g>1024)g=1024; if(g<1)g=1;
    KL(k_absmax,g,256)(src,n,sc);
    KL(k_mkscale,1,1)(sc,sc+1,sc+2);
    if(tr){ dim3 b(32,32),gr((cols+31)/32,(rows+31)/32); KL(k_qtr_e4m3,gr,b)(src,dst,rows,cols,sc+2); }
    else  { KL(k_quant_e4m3,nblk(n,TPB),TPB)(src,dst,n,sc+2); }
#ifndef CUDALM_CPU
    if(rec) KL(k_cp1,1,1)(sc,rec);
#endif
}
static void fp8_prep(const float* src,__nv_fp8_e4m3* dst,int tr,int rows,int cols,int deqi){
    fp8_prep_sc(src,dst,tr,rows,cols,&g_fp8sc[deqi-1]);
}
/* A-operand hold: while A is held, every FP8 GEMM reading it as plain [M,K] reuses one
 * quantized copy (q/k/v share a_norm, w1/w3 share f_norm). sc = {amax,deq,inv}.
 * A producer may also fill buf/sc itself and set valid (fused swiglu -> FP8 act). */
typedef struct { const float*src; int M,K,valid; __nv_fp8_e4m3*buf; size_t cap; float*sc; } F8Hold;
static F8Hold g_hold={0};
static int g_f8hold=1;   /* CUDALM_F8_HOLD=0: off */
static void f8_hold(const float*A,int M,int K){ g_hold.src=A; g_hold.M=M; g_hold.K=K; g_hold.valid=0;
    if(!g_hold.sc) g_hold.sc=(float*)dmalloc(4*sizeof(float)); }
static void f8_drop(void){ g_hold.src=NULL; g_hold.valid=0; }
static __nv_fp8_e4m3* f8_A(const float*A,int M,int K,float**deq){
    if(g_f8hold && A==g_hold.src && M==g_hold.M && K==g_hold.K){
        if(!g_hold.valid){ __nv_fp8_e4m3*b=f8slot(&g_hold.buf,&g_hold.cap,(size_t)M*K);
            cudaMemsetAsync(g_hold.sc,0,sizeof(float),0); fp8_prep_sc(A,b,0,M,K,g_hold.sc); g_hold.valid=1; }
        *deq=g_hold.sc+1; return g_hold.buf; }
    __nv_fp8_e4m3*P8=f8slot(&g8A,&g8Acap,(size_t)M*K); fp8_prep(A,P8,0,M,K,4); *deq=&g_fp8sc[4]; return P8;
}
/* Core FP8 E4M3 TN matmul: R[rowsP,rowsQ] = alpha*(P @ Q^T) + beta*C, with both
 * P8[rowsP,Kc] and Q8[rowsQ,Kc] Kc-contiguous E4M3, fp32 accumulate+output. In
 * cublasLt terms A=Q8(op=T), B=P8(op=N), D=C col-major [rowsQ,rowsP]. deqQ/deqP
 * are DEVICE deq-scale pointers (A_SCALE applies to A=Q8, B_SCALE to B=P8). */
static const void* g_f8cin=NULL;   /* set: the epilogue reads beta*C from here instead of from D */
static cudaDataType_t g_f8dt=CUDA_R_32F; /* D type; CUDA_R_16F only for head logits (CUDALM_H16) */
static void fp8_matmul(const __nv_fp8_e4m3* P8,const __nv_fp8_e4m3* Q8,
                       float* deqP,float* deqQ,int rowsP,int rowsQ,int Kc,
                       const float* alpha,const float* beta,void* C,int ldc,const char* what){
    if(g_fp8dbg){cudaError_t e=cudaDeviceSynchronize(); if(e!=cudaSuccess){fprintf(stderr,"fp8 FAULT pre-matmul %s (rowsP=%d rowsQ=%d Kc=%d): %s\n",what,rowsP,rowsQ,Kc,cudaGetErrorString(e)); exit(3);}}
    /* descs/layouts cached: creating them costs more host time than a small MoE GEMM runs.
     * op keyed by scale ptrs; layouts by (Kc,rowsQ,ldc,dt) with rowsP patched in per call */
    static struct { float *a,*b; cublasLtMatmulDesc_t op; } oc[64]; static int noc=0;
    static struct { int k,q,ld,dt; cublasLtMatrixLayout_t La,Lb,Ld; } lc[64]; static int nlc=0;
    float *pAsc=deqQ, *pBsc=deqP; /* device deq-scale ptrs; SetAttribute copies the ptr VALUE from these host vars */
    int oi=-1, li=-1;
    for(int i=0;i<noc;i++) if(oc[i].a==pAsc&&oc[i].b==pBsc){ oi=i; break; }
    if(oi<0){ if(noc==64){ fprintf(stderr,"fp8 op cache full\n"); exit(1); }
      cublasLtMatmulDesc_t op; ckb((cublasStatus_t)cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F),what);
      cublasOperation_t tT=CUBLAS_OP_T,tN=CUBLAS_OP_N; int8_t fast=1;
      ckb((cublasStatus_t)cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&tT,sizeof(tT)),"f8 TRANSA");
      ckb((cublasStatus_t)cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&tN,sizeof(tN)),"f8 TRANSB");
      ckb((cublasStatus_t)cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&pAsc,sizeof(pAsc)),"f8 ASCALE");
      ckb((cublasStatus_t)cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&pBsc,sizeof(pBsc)),"f8 BSCALE");
      ckb((cublasStatus_t)cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_FAST_ACCUM,&fast,sizeof(fast)),"f8 FASTACC");
      oc[noc].a=pAsc; oc[noc].b=pBsc; oc[noc].op=op; oi=noc++; }
    cublasLtMatmulDesc_t op=oc[oi].op;
    for(int i=0;i<nlc;i++) if(lc[i].k==Kc&&lc[i].q==rowsQ&&lc[i].ld==ldc&&lc[i].dt==(int)g_f8dt){ li=i; break; }
    if(li<0){ if(nlc==64){ fprintf(stderr,"fp8 layout cache full\n"); exit(1); }
      ckb((cublasStatus_t)cublasLtMatrixLayoutCreate(&lc[nlc].La,CUDA_R_8F_E4M3,Kc,rowsQ,Kc),"f8 La"); /* A=Q8: [Kc,rowsQ] col */
      ckb((cublasStatus_t)cublasLtMatrixLayoutCreate(&lc[nlc].Lb,CUDA_R_8F_E4M3,Kc,rowsP,Kc),"f8 Lb"); /* B=P8: [Kc,rowsP] col */
      ckb((cublasStatus_t)cublasLtMatrixLayoutCreate(&lc[nlc].Ld,g_f8dt,rowsQ,rowsP,ldc),"f8 Ld"); /* D=C:  [rowsQ,rowsP] col */
      lc[nlc].k=Kc; lc[nlc].q=rowsQ; lc[nlc].ld=ldc; lc[nlc].dt=(int)g_f8dt; li=nlc++; }
    cublasLtMatrixLayout_t La=lc[li].La, Lb=lc[li].Lb, Ld=lc[li].Ld;
    uint64_t cp=(uint64_t)rowsP;
    ckb((cublasStatus_t)cublasLtMatrixLayoutSetAttribute(Lb,CUBLASLT_MATRIX_LAYOUT_COLS,&cp,sizeof(cp)),"f8 Lb cols");
    ckb((cublasStatus_t)cublasLtMatrixLayoutSetAttribute(Ld,CUBLASLT_MATRIX_LAYOUT_COLS,&cp,sizeof(cp)),"f8 Ld cols");
    /* heuristic lookup costs more host time than a small MoE GEMM runs, so cache
     * the algo per (rowsQ,Kc,ldc,rowsP bucket); queried at the bucket's top rowsP */
    static struct { int q,k,ld,b,dt; cublasLtMatmulAlgo_t algo; } hc[128]; static int nhc=0;
    int bk=(rowsP+255)>>8, hi=-1;
    for(int i=0;i<nhc;i++) if(hc[i].q==rowsQ&&hc[i].k==Kc&&hc[i].ld==ldc&&hc[i].b==bk&&hc[i].dt==(int)g_f8dt){ hi=i; break; }
    cublasLtMatmulHeuristicResult_t h; int nres=0;
    if(hi<0){
        cublasLtMatmulPreference_t pref; ckb((cublasStatus_t)cublasLtMatmulPreferenceCreate(&pref),"f8 pref");
        ckb((cublasStatus_t)cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&g_fp8wscap,sizeof(g_fp8wscap)),"f8 prefws");
        cublasLtMatrixLayout_t Lb2,Ld2; int rp=bk<<8;
        ckb((cublasStatus_t)cublasLtMatrixLayoutCreate(&Lb2,CUDA_R_8F_E4M3,Kc,rp,Kc),"f8 Lb2");
        ckb((cublasStatus_t)cublasLtMatrixLayoutCreate(&Ld2,g_f8dt,rowsQ,rp,ldc),"f8 Ld2");
        ckb((cublasStatus_t)cublasLtMatmulAlgoGetHeuristic(LT,op,La,Lb2,Ld2,Ld2,pref,1,&h,&nres),"f8 heur");
        cublasLtMatrixLayoutDestroy(Lb2); cublasLtMatrixLayoutDestroy(Ld2); cublasLtMatmulPreferenceDestroy(pref);
        if(nres==0){ fprintf(stderr,"fp8: no algo for %s (rowsP=%d rowsQ=%d Kc=%d)\n",what,rowsP,rowsQ,Kc); exit(1); }
        if(nhc<128){ hc[nhc].q=rowsQ; hc[nhc].k=Kc; hc[nhc].ld=ldc; hc[nhc].b=bk; hc[nhc].dt=(int)g_f8dt; hc[nhc].algo=h.algo; nhc++; }
    } else h.algo=hc[hi].algo;
    ckb((cublasStatus_t)cublasLtMatmul(LT,op,alpha,Q8,La,P8,Lb,beta,g_f8cin?g_f8cin:C,Ld,C,Ld,&h.algo,g_fp8ws,g_fp8wscap,0),what);
    if(g_fp8dbg){cudaError_t e=cudaDeviceSynchronize(); if(e!=cudaSuccess){fprintf(stderr,"fp8 FAULT after-matmul %s (rowsP=%d rowsQ=%d Kc=%d): %s\n",what,rowsP,rowsQ,Kc,cudaGetErrorString(e)); exit(2);} else fprintf(stderr,"fp8 OK %s (rowsP=%d rowsQ=%d Kc=%d)\n",what,rowsP,rowsQ,Kc);}
}
/* alignment gate: FP8 tensor-core kernel needs all three GEMM dims divisible by
 * 16 (leading dims here are Kc or rowsQ, so those are covered too). */
static int g_fp8mask=3;   /* default skips g_atb: wgrad double transpose-cast costs more than FP8 saves. CUDALM_FP8_MASK: 1=g_ab(NN) 2=g_abt/g_abt_b(TN) 4=g_atb(NT) */
static inline int fp8_ok(int rowsP,int rowsQ,int Kc,int kind){ return g_fp8gemm && (g_fp8mask&kind) && !((rowsP|rowsQ|Kc)&15); }

/* C[M,N] = A[M,K] @ B[K,N]  (NN). FP8: P=A[M,K] plain, Q=B^T (transpose-cast
 * B[K,N]->[N,K] so K is contiguous). */
static void g_ab(const float*A,const float*B,float*C,int M,int K,int N){
    if(fp8_ok(M,N,K,1)){
        __nv_fp8_e4m3 *Q8=f8slot(&g8B,&g8Bcap,(size_t)K*N); float*dA;
        cudaMemsetAsync(g_fp8sc,0,6*sizeof(float),0);
        __nv_fp8_e4m3 *P8=f8_A(A,M,K,&dA);
        fp8_prep(B,Q8,1,K,N,1);      /* Q=B^T deq at sc[1] */
        fp8_matmul(P8,Q8,dA,&g_fp8sc[1],M,N,K,&ONE,&ZERO,C,N,"g_ab_f8");
        return;
    }
    if(g_fp16gemm){ gemm_ex(CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&ONE,B,N,A,K,&ZERO,C,N,
                            (size_t)K*N,(size_t)M*K,"g_ab_h"); return; }
    ckb(cublasSgemm(CB,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&ONE,B,N,A,K,&ZERO,C,N),"g_ab");
}
/* C[M,N] = A[M,K] @ B[N,K]^T   (B row-major [N,K], both K-contiguous already) */
static void g_abt(const float*A,const float*B,float*C,int M,int K,int N){
    if(fp8_ok(M,N,K,2)){
        __nv_fp8_e4m3 *Q8=f8slot(&g8B,&g8Bcap,(size_t)N*K); float*dA;
        cudaMemsetAsync(g_fp8sc,0,6*sizeof(float),0);
        __nv_fp8_e4m3 *P8=f8_A(A,M,K,&dA);
        fp8_prep(B,Q8,0,N,K,1);
        fp8_matmul(P8,Q8,dA,&g_fp8sc[1],M,N,K,&ONE,&ZERO,C,N,"g_abt_f8");
        return;
    }
    if(g_fp16gemm){ gemm_ex(CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&ONE,B,K,A,K,&ZERO,C,N,
                            (size_t)N*K,(size_t)M*K,"g_abt_h"); return; }
    ckb(cublasSgemm(CB,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&ONE,B,K,A,K,&ZERO,C,N),"g_abt");
}
/* C = A@B + R (residual folded into the FP8 epilogue; CUDALM_RESEPI=0: separate add) */
static int g_resepi=1;
static void g_ab_res(const float*A,const float*B,float*C,const float*R,int M,int K,int N){
    if(g_resepi && fp8_ok(M,N,K,1)){
        __nv_fp8_e4m3 *Q8=f8slot(&g8B,&g8Bcap,(size_t)K*N); float*dA;
        cudaMemsetAsync(g_fp8sc,0,6*sizeof(float),0);
        __nv_fp8_e4m3 *P8=f8_A(A,M,K,&dA);
        fp8_prep(B,Q8,1,K,N,1);
        g_f8cin=R; fp8_matmul(P8,Q8,dA,&g_fp8sc[1],M,N,K,&ONE,&ONE,C,N,"g_ab_res_f8"); g_f8cin=NULL;
        return;
    }
    g_ab(A,B,C,M,K,N); KL(k_add,nblk((size_t)M*N,TPB),TPB)(C,R,(size_t)M*N);
}
/* same but with an explicit beta, so dX = dg@w1^T + du@w3^T accumulates in place */
static void g_abt_b(const float*A,const float*B,float*C,int M,int K,int N,float beta){
    if(fp8_ok(M,N,K,2)){
        __nv_fp8_e4m3 *Q8=f8slot(&g8B,&g8Bcap,(size_t)N*K); float*dA;
        cudaMemsetAsync(g_fp8sc,0,6*sizeof(float),0);
        __nv_fp8_e4m3 *P8=f8_A(A,M,K,&dA);
        fp8_prep(B,Q8,0,N,K,1);
        fp8_matmul(P8,Q8,dA,&g_fp8sc[1],M,N,K,&ONE,&beta,C,N,"g_abt_b_f8");
        return;
    }
    if(g_fp16gemm){ gemm_ex(CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&ONE,B,K,A,K,&beta,C,N,
                            (size_t)N*K,(size_t)M*K,"g_abt_b_h"); return; }
    ckb(cublasSgemm(CB,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&ONE,B,K,A,K,&beta,C,N),"g_abt_b");
}
/* C[K,N] += A[M,K]^T @ B[M,N]  (NT, accumulates). FP8: contraction is M, so BOTH
 * operands transpose-cast to make M contiguous: P=A^T ([M,K]->[K,M]), Q=B^T
 * ([M,N]->[N,M]). */
static void g_atb(const float*A,const float*B,float*C,int M,int K,int N){
    if(fp8_ok(K,N,M,4)){
        __nv_fp8_e4m3 *P8=f8slot(&g8A,&g8Acap,(size_t)M*K), *Q8=f8slot(&g8B,&g8Bcap,(size_t)M*N);
        cudaMemsetAsync(g_fp8sc,0,6*sizeof(float),0);
        fp8_prep(A,P8,1,M,K,4);      /* P=A^T [K,M], Kc=M contiguous */
        fp8_prep(B,Q8,1,M,N,1);      /* Q=B^T [N,M], Kc=M contiguous */
        fp8_matmul(P8,Q8,&g_fp8sc[4],&g_fp8sc[1],K,N,M,&ONE,&ONE,C,N,"g_atb_f8");
        return;
    }
    if(g_fp16gemm){ gemm_ex(CUBLAS_OP_N,CUBLAS_OP_T,N,K,M,&ONE,B,N,A,K,&ONE,C,N,
                            (size_t)M*N,(size_t)M*K,"g_atb_h"); return; }
    ckb(cublasSgemm(CB,CUBLAS_OP_N,CUBLAS_OP_T,N,K,M,&ONE,B,N,A,K,&ONE,C,N),"g_atb");
}

/* ---- grouped MoE expert GEMMs (CUDALM_MOE_GROUPED, default 1) ----
 * One expert GEMM is ~1k x 384 x 360: under one wave of SMs, so E serial
 * launches crawl. mg_* queue the per-expert problems and mg_flush issues each
 * queue as one cublasSgemmGroupedBatched. Queue 1 holds the beta!=0 dgrads that
 * accumulate into a C a queue-0 problem writes (dX = dg@w1^T + du@w3^T), so it
 * runs strictly after queue 0. FP8/FP16-eligible shapes keep their own path. */
static int g_h16=1;       /* default on: FP8 lm_head writes fp16 logits, +2% tok/s, val unchanged; CUDALM_H16=0 to disable */
static int g_headw8=0;    /* CUDALM_HEAD_W8: FP8 wgrad for the tied-embedding dEmb only */
static int g_hbf=1;       /* CUDALM_HEAD_BF16=0: fp32 dlogits + TF32 dEmb in the FP8 head */
static int g_moe_grp=7;   /* bitmask: 1 NN fwd, 2 TN dgrad, 4 NT wgrad (NT needs its own call: mixed into the dgrad call it gains nothing) */
#define MG_MAXP 1024
#define MG_RING 256
typedef struct { int n; cublasOperation_t ta[MG_MAXP],tb[MG_MAXP];
    int m[MG_MAXP],nn[MG_MAXP],k[MG_MAXP],lda[MG_MAXP],ldb[MG_MAXP],ldc[MG_MAXP],gs[MG_MAXP];
    float al[MG_MAXP],be[MG_MAXP]; const float*A[MG_MAXP],*B[MG_MAXP]; float*C[MG_MAXP]; } MGq;
static MGq mgq[3];   /* 0 beta=0, 1 beta=1 dgrad (after 0), 2 NT wgrad */
/* pointer arrays must live on the device; a pinned ring lets the upload stay
 * async. Every forward pass syncs in moe_perm_all, far inside MG_RING runs. */
static void **mg_h=0,**mg_d=0; static int mg_slot=0;
static void mg_push(int q,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,
                    const float*A,int lda,const float*B,int ldb,float beta,float*C,int ldc){
    MGq*g=&mgq[q]; int i=g->n++;
    if(i>=MG_MAXP){ fprintf(stderr,"mg queue overflow\n"); exit(1); }
    g->ta[i]=ta; g->tb[i]=tb; g->m[i]=m; g->nn[i]=n; g->k[i]=k; g->A[i]=A; g->lda[i]=lda;
    g->B[i]=B; g->ldb[i]=ldb; g->C[i]=C; g->ldc[i]=ldc; g->al[i]=1.0f; g->be[i]=beta; g->gs[i]=1;
}
static void mg_run(MGq*g){
    if(!g->n) return;
    if(!mg_h){ ck(cudaHostAlloc((void**)&mg_h,(size_t)MG_RING*3*MG_MAXP*sizeof(void*),0),"mg host");
               ck(cudaMalloc((void**)&mg_d,(size_t)MG_RING*3*MG_MAXP*sizeof(void*)),"mg dev"); }
    size_t s=(size_t)mg_slot*3*MG_MAXP; mg_slot=(mg_slot+1)%MG_RING;
    void **h=mg_h+s, **d=mg_d+s; int n=g->n;
    memcpy(h,g->A,n*sizeof(void*)); memcpy(h+MG_MAXP,g->B,n*sizeof(void*));
    memcpy(h+2*MG_MAXP,g->C,n*sizeof(void*));
    ck(cudaMemcpyAsync(d,h,3*MG_MAXP*sizeof(void*),cudaMemcpyHostToDevice,0),"mg ptr h2d");
    ckb(cublasSgemmGroupedBatched(CB,g->ta,g->tb,g->m,g->nn,g->k,g->al,(const float*const*)d,g->lda,
        (const float*const*)(d+MG_MAXP),g->ldb,g->be,(float*const*)(d+2*MG_MAXP),g->ldc,n,g->gs),"mg grouped");
    g->n=0;
}
static void mg_flush(void){ mg_run(&mgq[0]); mg_run(&mgq[1]); mg_run(&mgq[2]); }
static int g_moe_ntser=0;  /* CUDALM_MOE_NTSER: grouped wgrad only up to this many rows, bigger experts go serial */
static int g_moe_nof8=1;   /* experts whose rows happen to be a multiple of 16 stay in the grouped call; CUDALM_MOE_NOFP8=0 = old serial FP8 */
static inline int mg_off(int P,int K,int Q,int kind){ return !(g_moe_grp&kind)||g_fp16gemm||(!g_moe_nof8&&fp8_ok(P,Q,K,kind)); }
static void mg_ab(const float*A,const float*B,float*C,int M,int K,int N){
    if(mg_off(M,K,N,1)){ g_ab(A,B,C,M,K,N); return; }
    mg_push(0,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,B,N,A,K,0.0f,C,N); }
static void mg_abt_b(const float*A,const float*B,float*C,int M,int K,int N,float beta){
    if(mg_off(M,K,N,2)){ g_abt_b(A,B,C,M,K,N,beta); return; }
    mg_push(beta!=0.0f,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,B,K,A,K,beta,C,N); }
static void mg_abt(const float*A,const float*B,float*C,int M,int K,int N){ mg_abt_b(A,B,C,M,K,N,0.0f); }
static void mg_atb(const float*A,const float*B,float*C,int M,int K,int N){
    if(mg_off(K,M,N,4)||(g_moe_ntser&&M>g_moe_ntser)){ g_atb(A,B,C,M,K,N); return; }
    mg_push(2,CUBLAS_OP_N,CUBLAS_OP_T,N,K,M,B,N,A,K,1.0f,C,N); }

/* ---- capacity-padded strided-batched MoE (CUDALM_MOE_SB, default 0: router max/avg load ~2.7x makes padding unprofitable) ----
 * Each expert gets `cap` rows (max load, rounded to 16) at e*cap; unused rows
 * are perm=-1 pads the gather zeroes and the scatter skips. All E experts then
 * run as one cublasGemmStridedBatchedEx: ~1.4x the grouped path's TF32 rate.
 * Zero pad rows add exact zeros to the wgrad sums. If a pass's cap would
 * overflow the Np = N*capf rows allocated, that pass uses the grouped path. */
static int g_moe_sb=0; static float g_moe_capf=1.25f; static long g_sb_n=0,g_sb_fb=0; static double g_sb_r=0,g_sb_rmax=0,g_sb_ema=0;
static size_t moe_np(int N){ return (g_moe_sb&&g_topk>1)?(((size_t)(N*g_moe_capf)+15)&~(size_t)15):(size_t)N; }
static void sbg(cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,const float*A,int lda,size_t sa,
                const float*B,int ldb,size_t sb,float beta,float*C,int ldc,size_t sc,int E){
    ckb(cublasGemmStridedBatchedEx(CB,ta,tb,m,n,k,&ONE,A,CUDA_R_32F,lda,(long long)sa,B,CUDA_R_32F,ldb,(long long)sb,
        beta!=0.0f?&ONE:&ZERO,C,CUDA_R_32F,ldc,(long long)sc,E,CUBLAS_COMPUTE_32F_FAST_TF32,CUBLAS_GEMM_DEFAULT),"moe sb"); }
/* row-major, per expert: C[M,N]=A[M,K]@B[K,N] / C[M,N]=A[M,K]@B[N,K]^T / C[K,N]+=A[M,K]^T@B[M,N] */
static void sb_ab(const float*A,size_t sa,const float*B,size_t sb,float*C,size_t sc,int M,int K,int N,int E){
    sbg(CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,B,N,sb,A,K,sa,0.0f,C,N,sc,E); }
static void sb_abt(const float*A,size_t sa,const float*B,size_t sb,float*C,size_t sc,int M,int K,int N,float beta,int E){
    sbg(CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,B,K,sb,A,K,sa,beta,C,N,sc,E); }
static void sb_atb(const float*A,size_t sa,const float*B,size_t sb,float*C,size_t sc,int M,int K,int N,int E){
    sbg(CUBLAS_OP_N,CUBLAS_OP_T,N,K,M,B,N,sb,A,K,sa,1.0f,C,N,sc,E); }

/* ===================== model state ======================================= */static Cfg gC; static float *gRC,*gRS;   /* rope cos/sin tables (device) */

/* ---- arena-shaping switches. Declared here because both ga_new and ga_bytes
 *      need them, and those come before the rest of the env-var block. ----
 * CUDALM_MOE_STASH: keep the expert forward's gp/up/yp per layer instead of
 * recomputing w1/w3/w2 in the backward pass. Costs (2*N*F + N*D)*4*L bytes and
 * saves ~9% of all training FLOPs. Cleared automatically if VRAM is short. */
static int g_moe_stash=1;
/* Set by cmd_train when prompt masking is on: the lm_head then runs only over
 * trainable rows. Masked rows produce zero loss and zero gradient, so all three
 * V-wide GEMMs and a V-wide softmax are being spent to compute nothing. */
static int g_mask_head=0;
/* Trainable row count for the current step, or 0 for "all N". */
static int g_nkeep=0;
/* WSD by default (CUDALM_SCHED=cosine reverts) + CUDALM_WSD_DECAY (fraction of the run spent cooling down) */
static int   g_wsd=1;
static float g_wsd_decay=0.15f;
/* CUDALM_REWARMUP=N: ramp the scheduled rate over the first N iterations of THIS
 * session. The built-in warmup keys off the absolute iteration, so it never
 * fires on a resume -- but a resume that also changes corpus or schedule can
 * step the LR by 10x, which is exactly the case continual-pretraining work warns
 * about (re-warm, re-decay, replay). Relative to start_iter, not to zero. */
static int g_rewarm=0, g_start_iter=0;
/* g_stage0: first iteration of the current training stage. `train` on an already trained model starts a new
 * stage (its own warmup + WSD/cosine over the stage's steps); lr_at() works relative to it. */
static int g_stage0=0;

typedef struct { int N, tr;   /* tr: training step (BF16 FFN/MoE paths); eval/serve GAs stay exact */
    float **a_in,**a_norm,**r1,**qh,**kh,**vh,**att,**f_in,**f_norm,**r2,**gg,**uu,**lse;
    float *x_final,*fn,*fnr,*acttmp;
    float *logits,*dlogits,*lossb;                   /* chunked: CH_ROWS rows */
    /* head row compaction (prompt masking only): keep = trainable row indices,
     * fnk/dfnk = those rows gathered together, tgtk = their targets */
    float *fnk,*dfnk;  int *keep,*tgtk,*h_keep;
    float *dout,*dfn,*dact,*dg,*du,*dfnorm,*d2,*dfin,*datt,*dq,*dk,*dv,*danorm,*dain,*drow;
    /* ---- MoE (n_exp>0 only) ---- */
    float **rprobs,**gate;      /* per layer, needed again in backward */
    int   **assign,**perm;      /* per layer */
    int   *hoff;                /* host: L*(E+1) expert row offsets into perm */
    int   *h_assign,*h_perm;    /* host scratch for building the permutation */
    int   *hcap; size_t Np;     /* top-K: per (l,k) SB capacity (0 = compact/grouped); padded rows */
    float *rt,*drt,*d_gate,*Xp,*gp,*up,*ap,*yp,*dyp,*dap,*dgp,*dup,*dXp;
    /* CUDALM_MOE_STASH: per-layer copies of the expert forward's gp/up/yp, so
     * moe_backward does not have to re-run w1/w3/w2. NULL = recompute. */
    float **gpS,**upS,**ypS;
    int   *d_cnt;
    /* ---- fp16 attention operands (g_fp16 only) ----
     * Q/K/V are per layer because backward re-reads the forward's values; dA is
     * a single buffer because backward only ever holds one layer's at a time. */
    __half **qh16,**kh16,**vh16,*datt16;
    float **agz,*attg,*dz;      /* CUDALM_ATTN_GATE: per-layer gate 2σ(z), gated O, dz */
    float **cna,**cnc;          /* CUDALM_CANON: per-layer pre-canon rmsnorm outputs (A, C) */
} GA;
static float g_moe_aux=0.03f;   /* CUDALM_MOE_AUX */
#define MOE_AUX g_moe_aux
/* ---- Tier 1: gated attention (o = SDPA * 2σ(x Wg), Wg zero-init = identity)
 * and n-gram layer/gate/trigram. Float weights outside QLM, AdamW, .t1 sidecar. */
static int g_attg=0, g_ng_layer=0, g_ng_gate=0, g_ng_tri=0;
/* CUDALM_VRES (20n): value residual, v_l <- (v_l + v_0)/2 for l >= 1; g_dv0 collects layer 0's share in backward */
static int g_vres=-1; static float*g_dv0=NULL; static int g_dv0n=0;
static int vres_on(void){ if(g_vres<0){ const char*e=getenv("CUDALM_VRES"); g_vres=e&&atoi(e)!=0; } return g_vres; }
static float *g_Wg=0,*g_WgG=0; static bf16 *g_WgM=0,*g_WgV=0;
#ifndef CUDALM_CPU
__global__ void k_agate_f(const float*att,float*s,float*o,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float g=2.0f/(1.0f+expf(-s[i])); s[i]=g; o[i]=att[i]*g; }
__global__ void k_agate_b(const float*att,const float*s,float*datt,float*dz,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float g=s[i],d=datt[i]; dz[i]=d*att[i]*g*(1.0f-0.5f*g); datt[i]=d*g; }
__global__ void k_mul(const float*a,const float*b,float*o,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=a[i]*b[i]; }
/* 3-way sum over a 128-thread block */
__device__ inline void bsum3(float&a,float&b,float&c){
    __shared__ float sh[3][4];
    for(int o=16;o>0;o>>=1){ a+=__shfl_down_sync(0xffffffff,a,o); b+=__shfl_down_sync(0xffffffff,b,o); c+=__shfl_down_sync(0xffffffff,c,o); }
    int w=threadIdx.x>>5; if((threadIdx.x&31)==0){ sh[0][w]=a; sh[1][w]=b; sh[2][w]=c; }
    __syncthreads();
    a=sh[0][0]+sh[0][1]+sh[0][2]+sh[0][3]; b=sh[1][0]+sh[1][1]+sh[1][2]+sh[1][3]; c=sh[2][0]+sh[2][1]+sh[2][2]+sh[2][3];
    __syncthreads(); }
/* Engram gate, one 128-thread block per row: g=σ(<rms(h),rms(k)>/√D); h+=g*v.
 * Leaves rms(h) in hn, rms(k) in k (in place), and g/rh/rk per row. */
__global__ void k_nggate_f(float*h,const float*v,float*k,float*hn,float*g,float*rh,float*rk,int D){
    size_t o=(size_t)blockIdx.x*D; float a=0,b=0,c=0;
    for(int i=threadIdx.x;i<D;i+=blockDim.x){ float x=h[o+i],y=k[o+i]; a+=x*x; b+=y*y; c+=x*y; }
    bsum3(a,b,c);
    float r1=rsqrtf(a/D+1e-6f), r2=rsqrtf(b/D+1e-6f), gg=1.0f/(1.0f+expf(-c*r1*r2*rsqrtf((float)D)));
    for(int i=threadIdx.x;i<D;i+=blockDim.x){ hn[o+i]=h[o+i]*r1; k[o+i]*=r2; h[o+i]+=gg*v[o+i]; }
    if(threadIdx.x==0){ g[blockIdx.x]=gg; rh[blockIdx.x]=r1; rk[blockIdx.x]=r2; } }
/* backward: dv=g*dy, dk through z, and dy += dh through z (in place) */
__global__ void k_nggate_b(float*dy,const float*v,const float*hn,const float*kn,const float*g,
                           const float*rh,const float*rk,float*dv,float*dk,int D){
    size_t o=(size_t)blockIdx.x*D; float a=0,b=0,c=0;
    for(int i=threadIdx.x;i<D;i+=blockDim.x){ a+=dy[o+i]*v[o+i]; b+=hn[o+i]*kn[o+i]; }
    bsum3(a,b,c);
    float gg=g[blockIdx.x], cz=a*gg*(1.0f-gg)*rsqrtf((float)D), m=b/D, r1=rh[blockIdx.x], r2=rk[blockIdx.x];
    for(int i=threadIdx.x;i<D;i+=blockDim.x){ float d=dy[o+i],x=hn[o+i],y=kn[o+i];
        dv[o+i]=gg*d; dk[o+i]=cz*(x-y*m)*r2; dy[o+i]=d+cz*(y-x*m)*r1; } }
#else
#define CPUK_PART 5
#include "cpu_kern.h"
#endif

/* FFN scratch widths: gg/uu of layer l hold the dense FFN, or the shared expert on sparse layers */
static int ga_shw(void){ const char*e=getenv("CUDALM_SHARED_F"); int v=e?atoi(e):0; return v>0?v:0; }
static int ga_w(int l){ int s=ga_shw(); return lyT(&gC,l)?(s>gC.F?s:gC.F):lyF(&gC,l); }
static int ga_fx(void){ int m=FXW(&gC), s=ga_shw(); return s>m?s:m; }
static GA* ga_new(int N,int train){
    GA*a=(GA*)calloc(1,sizeof(GA)); a->N=N;
    int L=gC.L,D=gC.D,F=gC.F,KD=gC.KD,V=gC.V,H=gC.H,T=gC.T;
    size_t ND=(size_t)N*D,NKD=(size_t)N*KD,NF=(size_t)N*F, NFX=(size_t)N*ga_fx();
    int ch=N<g_chrows?N:g_chrows;
    #define ARR(x) a->x=(float**)malloc(sizeof(float*)*L)
    ARR(a_in);ARR(a_norm);ARR(r1);ARR(qh);ARR(kh);ARR(vh);ARR(att);ARR(f_in);ARR(f_norm);ARR(r2);ARR(gg);ARR(uu);ARR(lse);
    #undef ARR
    for(int l=0;l<L;l++){
        a->a_in[l]=(float*)dmalloc(ND*4);   a->a_norm[l]=(float*)dmalloc(ND*4); a->r1[l]=(float*)dmalloc((size_t)N*4);
        a->qh[l]=(float*)dmalloc(ND*4);     a->kh[l]=(float*)dmalloc(NKD*4);    a->vh[l]=(float*)dmalloc(NKD*4);
        a->att[l]=(float*)dmalloc(ND*4);    a->f_in[l]=(float*)dmalloc(ND*4);   a->f_norm[l]=(float*)dmalloc(ND*4);
        a->r2[l]=(float*)dmalloc((size_t)N*4); a->gg[l]=(float*)dmalloc((size_t)N*ga_w(l)*4); a->uu[l]=(float*)dmalloc((size_t)N*ga_w(l)*4);
        /* per-layer log-sum-exp: lets backward skip re-running the attention
         * forward, at ~6 MB for m50 (L*B*H*T floats). */
        a->lse[l]=(float*)dmalloc((size_t)(N/T)*H*T*4);
    }
    a->x_final=(float*)dmalloc(ND*4); a->fn=(float*)dmalloc(ND*4); a->fnr=(float*)dmalloc((size_t)N*4);
    a->acttmp=(float*)dmalloc(NFX*4);
    if(g_attg){ a->agz=(float**)malloc(sizeof(float*)*L); for(int l=0;l<L;l++) a->agz[l]=(float*)dmalloc(ND*4);
                a->attg=(float*)dmalloc(ND*4); if(train) a->dz=(float*)dmalloc(ND*4); }
    a->logits=(float*)dmalloc((size_t)ch*V*4);
    if(train){
        a->dlogits=(float*)dmalloc((size_t)ch*V*4); a->lossb=(float*)dmalloc((size_t)N*4);
        a->dout=(float*)dmalloc(ND*4); a->dfn=(float*)dmalloc(ND*4); a->dact=(float*)dmalloc(NFX*4);
        a->dg=(float*)dmalloc(NFX*4);   a->du=(float*)dmalloc(NFX*4);  a->dfnorm=(float*)dmalloc(ND*4);
        a->d2=(float*)dmalloc(ND*4);   a->dfin=(float*)dmalloc(ND*4);a->datt=(float*)dmalloc(ND*4);
        a->dq=(float*)dmalloc(ND*4);   a->dk=(float*)dmalloc(NKD*4); a->dv=(float*)dmalloc(NKD*4);
        a->danorm=(float*)dmalloc(ND*4); a->dain=(float*)dmalloc(ND*4);
        a->drow=(float*)dmalloc((size_t)(N/T)*H*T*4);
        if(g_mask_head){                        /* only allocated when masking is on */
            a->fnk=(float*)dmalloc(ND*4); a->dfnk=(float*)dmalloc(ND*4);
            a->keep=(int*)dmalloc((size_t)N*4); a->tgtk=(int*)dmalloc((size_t)N*4);
            a->h_keep=(int*)malloc((size_t)N*4);
        }
    }
    if(g_fp16){
        a->qh16=(__half**)malloc(sizeof(__half*)*L);
        a->kh16=(__half**)malloc(sizeof(__half*)*L);
        a->vh16=(__half**)malloc(sizeof(__half*)*L);
        for(int l=0;l<L;l++){
            a->qh16[l]=(__half*)dmalloc(ND*2);
            a->kh16[l]=(__half*)dmalloc(NKD*2);
            a->vh16[l]=(__half*)dmalloc(NKD*2);
        }
        if(train) a->datt16=(__half*)dmalloc(ND*2);
    }
    if(gC.n_exp>0){
        int E=gC.n_exp;
        a->rprobs=(float**)malloc(sizeof(float*)*L); a->gate=(float**)malloc(sizeof(float*)*L);
        a->assign=(int**)malloc(sizeof(int*)*L);     a->perm=(int**)malloc(sizeof(int*)*L);
        int K=g_topk<1?1:g_topk; if(K>E)K=E;
        size_t Np=a->Np=moe_np(N); ND=Np*D; NF=Np*F;   /* expert scratch is capacity-padded */
        for(int l=0;l<L;l++){
            a->rprobs[l]=(float*)dmalloc((size_t)N*E*4); a->gate[l]=(float*)dmalloc((size_t)K*N*4);
            a->assign[l]=(int*)dmalloc((size_t)K*N*4);    a->perm[l]=(int*)dmalloc((size_t)K*Np*4);
        }
        a->hoff=(int*)calloc((size_t)L*K*(E+1),sizeof(int)); a->hcap=(int*)calloc((size_t)L*K,sizeof(int));
        ck(cudaHostAlloc((void**)&a->h_assign,(size_t)K*N*4,0),"assign pin"); ck(cudaHostAlloc((void**)&a->h_perm,(size_t)K*Np*4,0),"perm pin");
        a->rt=(float*)dmalloc((size_t)N*E*4);
        a->Xp=(float*)dmalloc(ND*4); a->ap=(float*)dmalloc(NF*4);
        /* with the stash on, the shared gp/up/yp are never read -- every use
         * goes through gpS/upS/ypS. Skipping them returns 52 MB, which is the
         * difference between the stash fitting and not. */
        if(!(train && g_moe_stash)){
            a->gp=(float*)dmalloc(NF*4); a->up=(float*)dmalloc(NF*4); a->yp=(float*)dmalloc(ND*4);
        }
        a->d_cnt=(int*)dmalloc((size_t)E*4);
        if(train){
            a->drt=(float*)dmalloc((size_t)N*E*4); a->d_gate=(float*)dmalloc((size_t)(g_topk<1?1:g_topk)*N*4);
            a->dyp=(float*)dmalloc(ND*4); a->dap=(float*)dmalloc(NF*4);
            a->dgp=(float*)dmalloc(NF*4); a->dup=(float*)dmalloc(NF*4); a->dXp=(float*)dmalloc(ND*4);
            if(g_moe_stash){
                a->gpS=(float**)malloc(sizeof(float*)*L); a->upS=(float**)malloc(sizeof(float*)*L);
                a->ypS=(float**)malloc(sizeof(float*)*L);
                for(int l=0;l<L;l++){ if(!lyT(&gC,l)){ a->gpS[l]=a->upS[l]=a->ypS[l]=NULL; continue; }
                    a->gpS[l]=(float*)dmalloc(K*NF*4); a->upS[l]=(float*)dmalloc(K*NF*4);
                    a->ypS[l]=(float*)dmalloc(K*ND*4);
                }
            }
        }
    }
    return a;
}

/* Build the token permutation that groups rows by expert. Done on the host from
 * `assign` so the ordering is deterministic and identical to the CPU build --
 * a device-side atomic partition would reorder rows run-to-run and perturb the
 * GEMM summation order. N ints each way is negligible against a multi-second step. */
static void asn_bad(int l,int e){
    fprintf(stderr,"[moe] layer %d: router picked expert %d (out of range) - aborting\n",l,e); fflush(stderr); exit(4);
}
static void moe_perm(GA*a,int l,int N,int E){
    ck(cudaMemcpy(a->h_assign,a->assign[l],(size_t)N*4,cudaMemcpyDeviceToHost),"assign d2h");
    int *off=a->hoff+(size_t)l*(gC.n_exp+1);
    for(int e=0;e<=E;e++) off[e]=0;
    for(int n=0;n<N;n++){ if((unsigned)a->h_assign[n]>=(unsigned)E) asn_bad(l,a->h_assign[n]); off[a->h_assign[n]+1]++; }
    for(int e=0;e<E;e++) off[e+1]+=off[e];
    int cur[256];                                 /* n_exp is capped at 256 */
    for(int e=0;e<E;e++) cur[e]=off[e];
    for(int n=0;n<N;n++) a->h_perm[cur[a->h_assign[n]]++]=n;
    ck(cudaMemcpyAsync(a->perm[l],a->h_perm,(size_t)N*4,cudaMemcpyHostToDevice,0),"perm h2d");
}
/* Same build for all K top-K passes of layer l in one round-trip: assign/perm
 * slices at k*N, hoff at (l*Km+k). h_perm is pinned and uploaded async; the next
 * layer's blocking assign D2H orders its reuse. */
/* CUDALM_MOE_OVL (default 1): the assign copy is async and the shared expert (which
 * needs no routing) is queued behind it, so the GPU has work while the host waits. */
static int g_moe_ovl=1, g_sh_done=0;
static void (*g_ovl_fn)(GA*,int,float*,int)=NULL;
static cudaEvent_t g_asn_ev=0;
/* O1: the same stable permutation built on the GPU (CUDALM_MOE_GPERM, default 1). Tiles of
 * 1024 tokens: per-tile expert histograms -> one prefix per pass -> stable in-tile ranks via
 * __match_any_sync. Bit-identical to the host build; only K*(E+1) offsets come back. */
#ifndef CUDALM_CPU
#define GP_T 1024
static int g_gperm=1;
__global__ void k_gp_hist(const int*asn,int N,int E,int NC,int*ch,int*bad){
    extern __shared__ int h[]; int c=blockIdx.x, k=blockIdx.y, n=c*GP_T+threadIdx.x;
    for(int e=threadIdx.x;e<E;e+=blockDim.x) h[e]=0; __syncthreads();
    if(n<N){ int e=asn[(size_t)k*N+n]; if((unsigned)e>=(unsigned)E) atomicExch(bad,e|0x40000000); else atomicAdd(&h[e],1); }
    __syncthreads();
    for(int e=threadIdx.x;e<E;e+=blockDim.x) ch[((size_t)k*E+e)*NC+c]=h[e];
}
__global__ void k_gp_scan(int*ch,int E,int NC,int*off){   /* one block per pass; ch -> exclusive tile offsets */
    extern __shared__ int tot[]; int k=blockIdx.x; int*cp=ch+(size_t)k*E*NC;
    for(int e=threadIdx.x;e<E;e+=blockDim.x){ int s=0; for(int c=0;c<NC;c++) s+=cp[(size_t)e*NC+c]; tot[e]=s; }
    __syncthreads();
    if(!threadIdx.x){ int s=0; for(int e=0;e<E;e++){ int t=tot[e]; tot[e]=s; off[k*(E+1)+e]=s; s+=t; } off[k*(E+1)+E]=s; }
    __syncthreads();
    for(int e=threadIdx.x;e<E;e+=blockDim.x){ int s=tot[e]; for(int c=0;c<NC;c++){ int t=cp[(size_t)e*NC+c]; cp[(size_t)e*NC+c]=s; s+=t; } }
}
__global__ void k_gp_scatter(const int*asn,int N,int E,int NC,const int*co,int*perm,size_t Np,int*inv){
    extern __shared__ int wc[];                   /* [32 warps][E] */
    int c=blockIdx.x, k=blockIdx.y, n=c*GP_T+threadIdx.x, w=threadIdx.x>>5, ln=threadIdx.x&31;
    for(int i=threadIdx.x;i<32*E;i+=blockDim.x) wc[i]=0; __syncthreads();
    int e=n<N?asn[(size_t)k*N+n]:-1; unsigned pe=__match_any_sync(0xffffffffu,e);
    int r=__popc(pe&((1u<<ln)-1));
    if(e>=0&&r==0) wc[w*E+e]=__popc(pe);
    __syncthreads();
    if(e<0) return;
    int pos=co[((size_t)k*E+e)*NC+c]+r; for(int j=0;j<w;j++) pos+=wc[j*E+e];
    perm[(size_t)k*Np+pos]=n; if(inv) inv[(size_t)k*N+n]=pos;
}
static int *g_gp_d=0, *g_gp_h=0; static size_t g_gp_n=0;
/* O2 merged top-K (CUDALM_MOE_MERGE, default 1): all K passes as ONE expert-grouped pass of K*N rows
 * (perm entries n' = k*N+n), so each expert GEMM sees ~K x the rows and the launches halve. g_minv
 * (inverse perm, per layer) lets a per-token combine add the K contributions in pass order. */
static int g_moemrg=1, g_mrgL[64]; static int *g_moff=0, *g_minv=0;
/* returns 1 if it built perm + host offsets (all K passes), 0 to fall back to the host build */
static int moe_gperm(GA*a,int l,int N,int E,int Km,int K,float*out,int mrg=0){
    if(!g_gperm||E>256) return 0;
    int NC=(N+GP_T-1)/GP_T, KN=K*N, NCm=mrg?(KN+GP_T-1)/GP_T:0; size_t nh=(size_t)K*(E+1)+(E+1)+1;
    size_t need=(size_t)K*E*NC+(size_t)E*NCm+nh;
    if(need>g_gp_n){ if(g_gp_d){ cudaFree(g_gp_d); cudaFreeHost(g_gp_h); }
        g_gp_d=(int*)dmalloc(need*4); ck(cudaHostAlloc((void**)&g_gp_h,nh*4,0),"gperm pin"); g_gp_n=need; }
    int *ch=g_gp_d, *chm=ch+(size_t)K*E*NC, *off=chm+(size_t)E*NCm, *offm=off+(size_t)K*(E+1), *bad=offm+E+1;
    cudaMemsetAsync(bad,0,4,0);
    KL(k_gp_hist,dim3(NC,K),GP_T,E*4)(a->assign[l],N,E,NC,ch,bad);
    KL(k_gp_scan,K,256,E*4)(ch,E,NC,off);
    if(!mrg) KL(k_gp_scatter,dim3(NC,K),GP_T,32*E*4)(a->assign[l],N,E,NC,ch,a->perm[l],a->Np,(int*)NULL);
    else {      /* per-pass counts above (aux/bias/stats); the merged grouping below */
        if(!g_minv){ g_minv=(int*)dmalloc((size_t)gC.L*KN*4); g_moff=(int*)calloc((size_t)gC.L*(gC.n_exp+1),sizeof(int)); }
        KL(k_gp_hist,dim3(NCm,1),GP_T,E*4)(a->assign[l],KN,E,NCm,chm,bad);
        KL(k_gp_scan,1,256,E*4)(chm,E,NCm,offm);
        KL(k_gp_scatter,dim3(NCm,1),GP_T,32*E*4)(a->assign[l],KN,E,NCm,chm,a->perm[l],0,g_minv+(size_t)l*KN);
    }
    ck(cudaMemcpyAsync(g_gp_h,off,nh*4,cudaMemcpyDeviceToHost,0),"gperm off d2h");
    if(g_ovl_fn&&out){
        if(!g_asn_ev) ck(cudaEventCreateWithFlags(&g_asn_ev,cudaEventDisableTiming),"asn ev");
        ck(cudaEventRecord(g_asn_ev,0),"asn rec"); g_ovl_fn(a,l,out,N); g_sh_done=1;
        ck(cudaEventSynchronize(g_asn_ev),"assign wait");
    } else ck(cudaStreamSynchronize(0),"gperm sync");
    int b=g_gp_h[nh-1]; if(b) asn_bad(l,b&~0x40000000);
    if(mrg) for(int e=0;e<=E;e++) g_moff[(size_t)l*(gC.n_exp+1)+e]=g_gp_h[(size_t)K*(E+1)+e];
    for(int k=0;k<K;k++){
        int *src=g_gp_h+(size_t)k*(E+1), *o=a->hoff+((size_t)l*Km+k)*(gC.n_exp+1), cap=0;
        for(int e=0;e<=E;e++) o[e]=src[e];
        for(int e=0;e<E;e++) if(o[e+1]-o[e]>cap) cap=o[e+1]-o[e];
        double r=(double)E*cap/N; g_sk_n[k>0]++; g_sk_r[k>0]+=r;
        g_sb_n++; g_sb_r+=r; g_sb_ema=g_sb_n>1?0.999*g_sb_ema+0.001*r:r; if(r>g_sb_rmax) g_sb_rmax=r;
        a->hcap[(size_t)l*Km+k]=0;
    }
    return 1;
}
#else
static int g_gperm=0, g_moemrg=0, g_mrgL[64], g_facp=0;
static int moe_gperm(GA*,int,int,int,int,int,float*,int=0){ return 0; }
#endif
static void moe_perm_all(GA*a,int l,int N,int E,int Km,int K,int sb,float*out){
    if(!sb&&moe_gperm(a,l,N,E,Km,K,out)) return;
    if(g_ovl_fn&&out){
        if(!g_asn_ev) ck(cudaEventCreateWithFlags(&g_asn_ev,cudaEventDisableTiming),"asn ev");
        ck(cudaMemcpyAsync(a->h_assign,a->assign[l],(size_t)K*N*4,cudaMemcpyDeviceToHost,0),"assign d2h");
        ck(cudaEventRecord(g_asn_ev,0),"asn rec");
        g_ovl_fn(a,l,out,N); g_sh_done=1;
        ck(cudaEventSynchronize(g_asn_ev),"assign wait");
    } else ck(cudaMemcpy(a->h_assign,a->assign[l],(size_t)K*N*4,cudaMemcpyDeviceToHost),"assign d2h");
    size_t Np=a->Np;
    for(int k=0;k<K;k++){
        int *asn=a->h_assign+(size_t)k*N, *prm=a->h_perm+(size_t)k*Np, *off=a->hoff+((size_t)l*Km+k)*(gC.n_exp+1);
        for(int e=0;e<=E;e++) off[e]=0;
        for(int n=0;n<N;n++){ if((unsigned)asn[n]>=(unsigned)E) asn_bad(l,asn[n]); off[asn[n]+1]++; }
        int cap=0; for(int e=0;e<E;e++) if(off[e+1]>cap) cap=off[e+1];
        for(int e=0;e<E;e++) off[e+1]+=off[e];
        double r=(double)E*cap/N;
        cap=(cap+15)&~15; if(!sb||(size_t)E*cap>Np) cap=0;
        g_sk_n[k>0]++; g_sk_r[k>0]+=r;
        g_sb_n++; g_sb_r+=r; g_sb_ema=g_sb_n>1?0.999*g_sb_ema+0.001*r:r; if(r>g_sb_rmax) g_sb_rmax=r; if(sb&&!cap) g_sb_fb++;
        a->hcap[(size_t)l*Km+k]=cap;
        int cur[256]; for(int e=0;e<E;e++) cur[e]=cap?e*cap:off[e];
        if(cap) memset(prm,0xff,(size_t)E*cap*4);
        for(int n=0;n<N;n++) prm[cur[asn[n]]++]=n;
    }
    ck(cudaMemcpyAsync(a->perm[l],a->h_perm,(size_t)K*Np*4,cudaMemcpyHostToDevice,0),"perm h2d");
}

static size_t ga_bytes(int N,int train){   /* for the pre-flight VRAM report */
    int L=gC.L,D=gC.D,F=gC.F,KD=gC.KD,V=gC.V,H=gC.H,T=gC.T;
    size_t ND=(size_t)N*D,NKD=(size_t)N*KD,NF=(size_t)N*F, ch=(size_t)(N<g_chrows?N:g_chrows), NFX=(size_t)N*ga_fx();
    size_t per=6*ND+2*NKD+2*(size_t)N+(size_t)(N/T)*H*T;
    size_t tot=(size_t)L*per + 2*ND + N + NFX + ch*V;
    for(int l=0;l<L;l++) tot += 2*(size_t)N*ga_w(l);
    if(train) tot += ch*V + N + 6*ND + 3*NFX + 2*ND + 2*NKD + 2*ND + (size_t)(N/T)*H*T;
    if(train && g_mask_head) tot += 2*ND + 2*(size_t)N;      /* fnk/dfnk, keep/tgtk */
    if(gC.n_exp>0){                       /* router state per layer + shared expert scratch */
        size_t E=(size_t)gC.n_exp;
        tot += (size_t)L*((size_t)N*E + 3*(size_t)N);
        tot += (size_t)N*E;                                  /* rt */
        size_t Np=moe_np(N), PD=Np*D, PF=Np*F;              /* expert scratch is capacity-padded */
        tot += (size_t)L*(Np-N)*(g_topk<1?1:g_topk);         /* padded perm */
        if(train && g_moe_stash) tot += PD + PF;             /* Xp, ap only */
        else                     tot += 2*PD + 3*PF;         /* Xp, yp, gp/up/ap */
        if(train) tot += (size_t)N*E + N + 2*PD + 3*PF;      /* drt, d_gate, dyp/dXp, dap/dgp/dup */
        if(train && g_moe_stash) tot += (size_t)NSPARSE(&gC)*(2*PF+PD)*(g_topk<1?1:g_topk); /* gpS/upS/ypS, per pass */
    }
    size_t h16=0;                          /* fp16 attention operands, 2 bytes each */
    if(g_fp16){ h16 = (size_t)L*(ND+2*NKD)*2; if(train) h16 += ND*2; }
    return tot*4 + h16;
}

/* ---- launch helpers ---- */
static void e_rope(float*b,int nh,int N,int back){ KL(k_rope,nblk((size_t)N*nh*gC.half,TPB),TPB)(b,gRC,gRS,N,nh,gC.hd,gC.half,gC.T,back); }
static void e_add(float*A,const float*B,size_t n){ KL(k_add,nblk(n,TPB),TPB)(A,B,n); }
static void e_cpy(float*A,const float*B,size_t n){ KL(k_copy,nblk(n,TPB),TPB)(A,B,n); }
static int g_rmsw=1;       /* CUDALM_RMS_WARP=0: old block-per-row rmsnorm */
#define A16(p) (!(((uintptr_t)(p))&15))
static void e_rmsf(const float*X,const float*P,size_t go,float*O,float*rinv,int N,int D){
    if(g_rmsw&&!(D&3)&&A16(X)&&A16(O)) KL(k_rmsnorm_w,nblk(N,TPB/32),TPB)(X,P,go,O,rinv,N,D);
    else KL(k_rmsnorm,N,TPB)(X,P,go,O,rinv,N,D); }
static void e_rmsb(const float*X,const float*P,size_t go,const float*rinv,const float*dy,float*dx,
                   float*G,size_t ggo,int N,int D){
    if(g_rmsw&&!(D&3)&&A16(X)&&A16(dy)&&A16(dx)) KL(k_rmsnorm_bwd_w,nblk(N,TPB/32),TPB)(X,P,go,rinv,dy,dx,N,D);
    else KL(k_rmsnorm_bwd,N,TPB)(X,P,go,rinv,dy,dx,N,D);
    static float*part=NULL; static int partD=0;
    if(D>partD){ if(part) cudaFree(part); ck(cudaMalloc(&part,(size_t)GG_CH*D*4),"gamma part"); partD=D; }
    KL(k_gamma_part,dim3((unsigned)nblk(D,TPB),GG_CH),TPB)(X,rinv,dy,part,N,D);
    KL(k_gamma_sum,nblk(D,TPB),TPB)(part,G,ggo,D); }
static void e_swig(const float*G,const float*U,float*O,size_t n){
    if(!(n&3) && !(((size_t)G|(size_t)U|(size_t)O)&15)){
        KL(k_swiglu4,nblk(n/4,TPB),TPB)((const float4*)G,(const float4*)U,(float4*)O,n/4); return; }
    KL(k_swiglu,nblk(n,TPB),TPB)(G,U,O,n); }
/* Effective window for layer l: full attention on every full_every-th layer,
 * otherwise the sliding window. 0 means "attend to the whole prefix". */
static int lwin(int l){
    if(gC.win<=0) return 0;
    if(gC.full_every>0 && ((l+1)%gC.full_every)==0) return 0;
    return gC.win;
}
/* Shared-memory tiling is OFF by default: measured 17% SLOWER than the untiled
 * kernel (fwd attn 6746 -> 7888 ms). Consecutive queries share nearly all their
 * keys and land on adjacent warps of the same SM, so L1/L2 was already
 * providing the reuse; explicit staging just adds a __syncthreads per tile.
 * Kept behind CUDALM_TILE=1 because it becomes the right structure again once
 * the working set no longer fits L2 (much larger hd, or many more KV heads). */
static int g_tiled=0;
static int g_sub=0;        /* CUDALM_SUB=1 enables the sub-warp mapping */
/* Lane-per-query flash tile: on by default (measured 1.90x over the warp
 * kernels at B=3, identical loss). CUDALM_LANE=0 falls back to warp-per-output. */
static int g_lane=1;
/* CUDALM_HFMA2=1: packed __hfma2 dots in the backward lane kernels. OFF by
 * default -- it trades precision for speed and only pays where fp16 math is
 * faster than fp32 (Pascal GP100, TU116). Measure per architecture. */
static int g_hfma2=0;
/* CUDALM_DKV_SPLIT: threads cooperating on one key in the dKV backward kernel.
 * 1 = one-thread-per-key (255 regs, 388 B spill) -- and MEASURED FASTEST despite
 *     the low occupancy: the sub-lane forms stage the same query range across a
 *     wider block, so their staging traffic dominates. On the 1660S (bench):
 *       hd=64 T=256  split1 335ms vs split2 467ms  (1.39x)
 *       hd=64 T=2048 split1 1131ms vs split2 2716ms (2.40x) -- worse the longer T
 *       hd=32 T=256  tie. Never slower than 2 in any measured shape.
 * 2/4 = sub-lane split: fewer registers, more blocks/SM, but the extra staging
 *     and the per-(key,query) butterfly cost more than the occupancy buys back.
 * Numerically identical to 1 (selftest: dKV LANE/1 == LANE/2). Default was 2 on
 * an occupancy argument that the measurement overturns. */
static int g_dkv_split=1;
/* CUDALM_DKV_TC: tensor-core (wmma) dKV backward; 0 = the lane kernel above. */
static int g_dkv_tc=1;
static int g_dq_tc=1;
static int g_fwd_tc=1;

/* Bound how much work sits in one command buffer. Preflight syncs after every
 * launch and passes at 6% of the watchdog; real training runs async, so a whole
 * step (~100 launches, seconds of GPU work) can be queued before the driver
 * gets a scheduling point. Syncing per layer is measurably FREE here -- the
 * fully-synced profiling run came out at 4.12 s/iter against 4.22 s/iter
 * unsynced, because each kernel already saturates the GPU -- so this is pure
 * insurance. CUDALM_NOSYNC=1 disables it. */
#ifdef _WIN32
static int g_sync=1;
#else
static int g_sync=0;       /* O1: no WDDM watchdog to feed on Linux */
#endif
static inline void layer_sync(void){ if(g_sync) cudaStreamSynchronize(0); }
static void e_f2h(const float*src,__half*dst,size_t n,float s){
    KL(k_f2h,nblk(n,TPB),TPB)(src,dst,n,s); }
static int g_ropeh=1, g_h16rdy=0;   /* CUDALM_ROPEH=0: separate rope + cast */
static void e_attn(GA*a,int N,int l){
    const float *Qb=a->qh[l],*Kb=a->kh[l],*Vb=a->vh[l];
    float *O=a->att[l],*lse=a->lse[l];
    int B=N/gC.T, T=gC.T;
    if(g_fp16){
        /* cast once per layer; the window loop then re-reads the half copies */
        if(!g_h16rdy){
            e_f2h(Qb,a->qh16[l],(size_t)N*gC.D, 1.0f);
            e_f2h(Kb,a->kh16[l],(size_t)N*gC.KD,1.0f);
            e_f2h(Vb,a->vh16[l],(size_t)N*gC.KD,1.0f); }
        g_h16rdy=0;
        if(attn_fwd_fa(a->qh16[l],a->kh16[l],a->vh16[l],O,lse,lwin(l),
                       B,T,gC.H,gC.KV,gC.hd,gC.D,gC.KD)) return;
        if(g_fwd_tc && attn_fwd_tc(a->qh16[l],a->kh16[l],a->vh16[l],O,lse,lwin(l),
                                   B,T,gC.H,gC.KV,gC.hd,gC.D,gC.KD)) return;        if(g_lane && attn_fwd_lane(a->qh16[l],a->kh16[l],a->vh16[l],O,lse,lwin(l),
                                   B,T,gC.H,gC.KV,gC.hd,gC.D,gC.KD)) return;
        KL(k_attn_fwd_h,B*gC.H*T,WARP)(a->qh16[l],a->kh16[l],a->vh16[l],O,lse,lwin(l),
            B,T,gC.H,gC.KV,gC.hd,gC.D,gC.KD);
        return;
    }
    if(g_sub && (gC.hd%SUBL)==0){
        size_t nq=(size_t)B*gC.H*T, thr=nq*SUBL;
        KL(k_attn_fwd_sub,nblk(thr,TPB),TPB)(Qb,Kb,Vb,O,lse,lwin(l),
            B,T,gC.H,gC.KV,gC.hd,gC.D,gC.KD);
    } else if(g_tiled && (T%QPB)==0){
        size_t shm=(size_t)2*KTILE*gC.hd*sizeof(float);
        KL(k_attn_fwd_tiled,(B*gC.H*T)/QPB,QPB*WARP,shm)(Qb,Kb,Vb,O,lse,lwin(l),
            B,T,gC.H,gC.KV,gC.hd,gC.D,gC.KD);
    } else {
        KL(k_attn_fwd,B*gC.H*T,WARP)(Qb,Kb,Vb,O,lse,lwin(l),B,T,gC.H,gC.KV,gC.hd,gC.D,gC.KD);
    } }
/* Timesteps per attention-backward launch. With a window, per-warp work is
 * already bounded by W, so larger chunks are safe and cut the launch count
 * (T=8192 at chunk 128 would be 1536 launches/step, mostly early-returning). */
static int attn_chunk(void){
    int c = gC.win>0 ? (gC.win<256?gC.win:256) : g_attn_chunk;
    return c<8?8:c;
}

/* MoE FFN forward for layer l: out = f_in + sum_e gate * expert_e(f_norm). */
/* ---- per-expert FP8 for the top-K MoE (CUDALM_MOE_F8, bit 1 fwd NN, bit 2 dgrad TN).
 * F=360 is not %16, so the F-contracting GEMMs pad F->Fp=368 with zeros inside
 * the FP8 buffers; weights are cast once per layer, activations once per pass. */
static int g_moef8=0;  /* pad-F FP8 MoE: correct but no faster than TF32 grouped (small per-expert GEMMs); CUDALM_MOE_F8=3 to enable */
static inline int moef8(int bit){ return g_fp8gemm && (g_moef8&bit) && !(gC.D&15); }
static __nv_fp8_e4m3 *mf8[9]; static size_t mf8c[9]; static float *g_msc=NULL;
static __nv_fp8_e4m3* mf_buf(int i,size_t n){ return f8slot(&mf8[i],&mf8c[i],n); }
static float* mf_sc(int i){ if(!g_msc) g_msc=(float*)dmalloc(48*sizeof(float)); return g_msc+3*i; }  /* [amax,deq,inv] */
#ifndef CUDALM_CPU
__global__ void k_quant_pad(const float* x,__nv_fp8_e4m3* q,size_t rows,int cols,int colsP,const float* inv){
    float f=*inv; size_t n=rows*colsP;
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=(size_t)gridDim.x*blockDim.x){
        size_t r=i/colsP; int c=(int)(i-r*colsP);
        q[i]=__nv_fp8_e4m3(c<cols? x[r*cols+c]*f:0.f); }
}
/* batched transpose-cast: src[e][rows,cols] -> dst[e][cols,rowsP], rows>=rows zero-filled */
__global__ void k_qtrp_b(const float* src,__nv_fp8_e4m3* dst,int rows,int cols,int rowsP,const float* inv){
    __shared__ float t[32][33];
    size_t e=blockIdx.z; src+=e*(size_t)rows*cols; dst+=e*(size_t)cols*rowsP;
    int x=blockIdx.x*32+threadIdx.x, y=blockIdx.y*32+threadIdx.y;
    t[threadIdx.y][threadIdx.x]=(x<cols&&y<rows)? src[(size_t)y*cols+x]:0.f;
    __syncthreads();
    float f=*inv;
    int r=blockIdx.y*32+threadIdx.x, c=blockIdx.x*32+threadIdx.y;
    if(c<cols&&r<rowsP) dst[(size_t)c*rowsP+r]=__nv_fp8_e4m3(t[threadIdx.x][threadIdx.y]*f);
}
#else
#define CPUK_PART 6
#include "cpu_kern.h"
#endif
static void mq_scale(const float* x,size_t n,float* s){
    int g=(int)((n+255)/256); if(g>1024)g=1024; if(g<1)g=1;
    cudaMemsetAsync(s,0,sizeof(float),0);
    KL(k_absmax,g,256)(x,n,s);
    KL(k_mkscale,1,1)(s,s+1,s+2);
}

static int mb_S(void){ int k=g_topk<1?1:g_topk; if(k>gC.n_exp)k=gC.n_exp; return g_mbias_slot?k:1; }
static float* mbias(int l){
    if(g_mbias_u<=0&&!g_mbias_ld) return NULL;
    size_t ls=(size_t)mb_S()*gC.n_exp;
    if(!ls) return NULL;                     /* dense model: no router, and no zero-byte allocations */
    if(!g_mbiasD){ size_t n=(size_t)gC.L*ls; ck(cudaHostAlloc((void**)&g_mbiasH,n*4,0),"mbias pin"); memset(g_mbiasH,0,n*4); g_mbiasD=(float*)dmalloc(n*4);
        ck(cudaMemset(g_mbiasD,0,n*4),"mbias z"); }
    return g_mbiasD+(size_t)l*ls;
}
/* aux-free balancing: b_e += u*sign(m - load_e) (BIAS_P: u*(m-load_e)/m); selection-only bias.
 * cs = ns count rows of E (one per slot with BIAS_SLOT, else the combined count), m = target load */
static void mbias_upd(int l,const int*cs,int ns,int E,float m){
    if(g_mbias_u<=0||!g_mbiasH) return;
    size_t ls=(size_t)mb_S()*gC.n_exp; float *b=g_mbiasH+(size_t)l*ls;
    float u=g_mbias_u*LMUL(g_biaslm,g_biaslm_n,l);
    for(int s=0;s<ns;s++) for(int e=0;e<E;e++){ float c=(float)cs[s*E+e];
        b[s*E+e]+=u*(g_mbias_p?(m-c)/m:(c<m?1.f:c>m?-1.f:0.f)); }
    /* pinned + async: a pageable cudaMemcpy here drained the GPU once per MoE layer. The host
     * copy is next written a step later, after the forward's blocking assign copy ordered this one. */
    ck(cudaMemcpyAsync(g_mbiasD+(size_t)l*ls,b,ls*4,cudaMemcpyHostToDevice,0),"mbias h2d");
}
/* train-only host rebalance of layer l's top-K assignment before the perm build.
 * SINK: Sinkhorn column scales c_e so sum_n P_ne c_e r_n = N/E, then reselect top-K by pr*c_e.
 * RCAP: per slot, experts over C=ceil(rcap*N/E) keep their C highest-pr tokens; the rest go to
 * the best non-full expert (pr+bias) not already used by that token. Gate stays pr[expert]. */
static const float*g_rb_p;
static int rb_cmp(const void*x,const void*y){ float a=g_rb_p[*(const int*)x],b=g_rb_p[*(const int*)y];
    return a<b?1:a>b?-1:0; }
static void moe_rebal(GA*a,int l,int N,int E,int K){
    static float *hp=0,*hg=0; static int *lst=0,*ov=0; static size_t nc=0;
    size_t ne=(size_t)N*E;
    if(nc<ne){ if(hp){ cudaFreeHost(hp); cudaFreeHost(hg); free(lst); free(ov); }
        ck(cudaHostAlloc((void**)&hp,ne*4,0),"rb pin"); ck(cudaHostAlloc((void**)&hg,(size_t)8*N*4,0),"rb pin");
        lst=(int*)malloc((size_t)N*4); ov=(int*)malloc((size_t)N*4); nc=ne; }
    ck(cudaMemcpy(hp,a->rprobs[l],ne*4,cudaMemcpyDeviceToHost),"rb p d2h");
    int *as=a->h_assign; ck(cudaMemcpy(as,a->assign[l],(size_t)K*N*4,cudaMemcpyDeviceToHost),"rb a d2h");
    int bs=g_mbias_slot?E:0; const float*b=(g_mbiasH&&(g_mbias_u>0||g_mbias_ld))?g_mbiasH+(size_t)l*mb_S()*gC.n_exp:NULL;
    if(g_moe_sink){
        float c[256],col[256]; for(int e=0;e<E;e++) c[e]=1.f;
        for(int it=0;it<g_moe_sink;it++){
            for(int e=0;e<E;e++) col[e]=0;
            for(int n=0;n<N;n++){ const float*p=hp+(size_t)n*E; float s=0;
                for(int e=0;e<E;e++) s+=p[e]*c[e]; s=1.f/s;
                for(int e=0;e<E;e++) col[e]+=p[e]*c[e]*s; }
            for(int e=0;e<E;e++){ c[e]*=(float)N/E/(col[e]+1e-6f); c[e]=c[e]<1e-3f?1e-3f:c[e]>1e3f?1e3f:c[e]; }
        }
        for(int n=0;n<N;n++){ const float*p=hp+(size_t)n*E; int ch[8];
            for(int k=0;k<K;k++){ int best=0; float bp=-1e30f;
                for(int e=0;e<E;e++){ int t=0; for(int j=0;j<k;j++) if(ch[j]==e){ t=1; break; }
                    float sc=p[e]*c[e]+(b?b[k*bs+e]:0.f); if(!t&&sc>bp){ bp=sc; best=e; } }
                ch[k]=best; if(as[(size_t)k*N+n]!=best) g_rb_mv++; as[(size_t)k*N+n]=best; } }
    }
    if(g_moe_rcap>0){
        int C=(int)ceilf(g_moe_rcap*N/E); g_rb_p=hp;
        for(int k=0;k<K;k++){ int *ak=as+(size_t)k*N, cnt[256], off[257], no=0;
            for(int e=0;e<E;e++) cnt[e]=0;
            for(int n=0;n<N;n++) cnt[ak[n]]++;
            off[0]=0; for(int e=0;e<E;e++) off[e+1]=off[e]+cnt[e];
            { int cur[256]; for(int e=0;e<E;e++) cur[e]=off[e]; for(int n=0;n<N;n++) lst[cur[ak[n]]++]=n; }
            for(int e=0;e<E;e++) if(cnt[e]>C){ int *q=lst+off[e];
                for(int i=0;i<cnt[e];i++) q[i]=(int)((size_t)q[i]*E+e);     /* flat index for the sort */
                qsort(q,cnt[e],4,rb_cmp);
                for(int i=C;i<cnt[e];i++) ov[no++]=q[i]/E;
                cnt[e]=C; }
            for(int i=0;i<no;i++){ int n=ov[i], best=-1; float bp=-1e30f; const float*p=hp+(size_t)n*E;
                for(int e=0;e<E;e++){ if(cnt[e]>=C) continue; int t=0;
                    for(int j=0;j<K;j++) if(j!=k&&as[(size_t)j*N+n]==e){ t=1; break; }
                    float sc=p[e]+(b?b[k*bs+e]:0.f); if(!t&&sc>bp){ bp=sc; best=e; } }
                if(best<0){ cnt[ak[n]]++; continue; }                    /* nowhere to go: keep (over cap) */
                ak[n]=best; cnt[best]++; g_rb_mv++; }
        }
    }
    g_rb_tot+=(long)K*N;
    for(int k=0;k<K;k++) for(int n=0;n<N;n++) hg[(size_t)k*N+n]=hp[(size_t)n*E+as[(size_t)k*N+n]];
    ck(cudaMemcpy(a->assign[l],as,(size_t)K*N*4,cudaMemcpyHostToDevice),"rb a h2d");
    ck(cudaMemcpy(a->gate[l],hg,(size_t)K*N*4,cudaMemcpyHostToDevice),"rb g h2d");
}
static int g_rpre=0; static const float* rpre_x(const float*P,const Off*rd,GA*a,int l,int N);
static void rpre_bwd(const float*P,float*G,const Off*og,const Off*rd,GA*a,int l,int N);
/* ---- BF16 GEMM core (S2 FFN path + S3b MoE experts) ----
 * bf16 operands, fp32 accumulate: 2x TF32 on sm_120 (gemm19: 45 vs 23 TF/s; grouped
 * 1.8x). Training activation buffers only (a->tr), so eval/serving/decheck keep the
 * exact path and no bf16 weight copy can go stale. */
#ifndef CUDALM_CPU
#include <cuda_bf16.h>
typedef __nv_bfloat16 hbf;
typedef __nv_bfloat162 hbf2;
static int g_bffn=1, g_bfmoe=1;   /* CUDALM_BFFN / CUDALM_BFMOE */
static long g_fwd_id=0;
__global__ void k_f2bf(const float*s,hbf*d,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(4*i+3<n){ float4 v=((const float4*)s)[i];
        ((hbf2*)d)[2*i]=__floats2bfloat162_rn(v.x,v.y); ((hbf2*)d)[2*i+1]=__floats2bfloat162_rn(v.z,v.w); }
    else for(size_t j=4*i;j<n;j++) d[j]=__float2bfloat16(s[j]);
}
__global__ void k_f2bf1(const float*s,hbf*d,size_t n){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=__float2bfloat16(s[i]); }
__global__ void k_swig_bf(const hbf*G,const hbf*U,hbf*O,size_t n2){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n2) return;
    float2 z=__bfloat1622float2(((const hbf2*)G)[i]), u=__bfloat1622float2(((const hbf2*)U)[i]);
    ((hbf2*)O)[i]=__floats2bfloat162_rn(z.x/(1.f+__expf(-z.x))*u.x, z.y/(1.f+__expf(-z.y))*u.y);
}
__global__ void k_swigb_bf(const hbf*G,const hbf*U,const hbf*dA,hbf*dg,hbf*du,size_t n2){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n2) return;
    float2 z=__bfloat1622float2(((const hbf2*)G)[i]), u=__bfloat1622float2(((const hbf2*)U)[i]),
           a=__bfloat1622float2(((const hbf2*)dA)[i]);
    float s0=1.f/(1.f+__expf(-z.x)), s1=1.f/(1.f+__expf(-z.y));
    ((hbf2*)dg)[i]=__floats2bfloat162_rn(a.x*u.x*(s0+z.x*s0*(1.f-s0)), a.y*u.y*(s1+z.y*s1*(1.f-s1)));
    ((hbf2*)du)[i]=__floats2bfloat162_rn(a.x*z.x*s0, a.y*z.y*s1);
}
/* MoE row kernels, one block per permuted row j (n = perm[j]), D even */
__global__ void k_gather_bf(const float*src,const int*perm,hbf*Xp,int D){
    int j=blockIdx.x, n=perm[j]; hbf2*o=(hbf2*)(Xp+(size_t)j*D); const float2*s=(const float2*)(src+(size_t)(n<0?0:n)*D);
    for(int d=threadIdx.x;d<D/2;d+=blockDim.x){ float2 v=n<0?make_float2(0.f,0.f):s[d]; o[d]=__floats2bfloat162_rn(v.x,v.y); }
}
__global__ void k_scatter_gate_bf(const hbf*yp,const int*perm,const float*gate,float*out,int D){
    int j=blockIdx.x, n=perm[j]; if(n<0) return; float g=gate[n];
    const hbf2*y=(const hbf2*)(yp+(size_t)j*D); float2*o=(float2*)(out+(size_t)n*D);
    for(int d=threadIdx.x;d<D/2;d+=blockDim.x){ float2 v=__bfloat1622float2(y[d]), a=o[d]; a.x+=g*v.x; a.y+=g*v.y; o[d]=a; }
}
__global__ void k_scatter_add_bf(const hbf*dXp,const int*perm,float*dst,int D){
    int j=blockIdx.x, n=perm[j]; if(n<0) return;
    const hbf2*s=(const hbf2*)(dXp+(size_t)j*D); float2*o=(float2*)(dst+(size_t)n*D);
    for(int d=threadIdx.x;d<D/2;d+=blockDim.x){ float2 v=__bfloat1622float2(s[d]), a=o[d]; a.x+=v.x; a.y+=v.y; o[d]=a; }
}
/* d_gate[n] = dot(dout[n], yp[j]); dyp[j] = gate[n]*dout[n]; Xp[j] = src[n] (re-gather) */
__global__ void k_moe_dgate_bf(const float*dout,const hbf*yp,const int*perm,const float*gate,float*d_gate,
                               hbf*dyp,const float*src,hbf*Xp,int D){
    int j=blockIdx.x, n=perm[j]; hbf2 *dy=(hbf2*)(dyp+(size_t)j*D), *xp=(hbf2*)(Xp+(size_t)j*D);
    if(n<0){ for(int d=threadIdx.x;d<D/2;d+=blockDim.x){ dy[d]=__floats2bfloat162_rn(0.f,0.f); xp[d]=dy[d]; } return; }
    const float2 *dO=(const float2*)(dout+(size_t)n*D), *sr=(const float2*)(src+(size_t)n*D); const hbf2*y=(const hbf2*)(yp+(size_t)j*D);
    __shared__ float sh[8]; float acc=0.f, g=gate[n];
    for(int d=threadIdx.x;d<D/2;d+=blockDim.x){ float2 a=dO[d], b=__bfloat1622float2(y[d]), s=sr[d];
        acc+=a.x*b.x+a.y*b.y; dy[d]=__floats2bfloat162_rn(a.x*g,a.y*g); xp[d]=__floats2bfloat162_rn(s.x,s.y); }
    for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffffu,acc,o);
    if((threadIdx.x&31)==0) sh[threadIdx.x>>5]=acc;
    __syncthreads();
    if(threadIdx.x==0){ float s=0.f; for(int i=0;i<(int)(blockDim.x>>5);i++) s+=sh[i]; d_gate[n]=s; }
}
/* G[i] += sum_k T[k*st + i] */
__global__ void k_addbfk(float*G,const hbf*T,size_t st,int K,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float s=0.f; for(int k=0;k<K;k++) s+=__bfloat162float(T[(size_t)k*st+i]); G[i]+=s;
}
static void bf_cvt(const float*s,hbf*d,size_t n){
    if(((size_t)s|(size_t)d)&15) KL(k_f2bf1,nblk(n,TPB),TPB)(s,d,n);
    else KL(k_f2bf,nblk((n+3)/4,TPB),TPB)(s,d,n); }
static hbf* bf_slot(hbf**p,size_t*cap,size_t n){ if(n>*cap){ if(*p) cudaFree(*p); *p=(hbf*)dmalloc(n*2); *cap=n; } return *p; }
/* row-major C = A@B, A@B^T, C += A^T@B with bf16 A/B; C fp32, or bf16 when cb */
static void hb_g(cublasOperation_t tb,cublasOperation_t ta,int n,int m,int k,const hbf*B,int ldb,const hbf*A,int lda,
                 float beta,void*C,int cb,int ldc,const char*w){
    ckb(cublasGemmEx(CB,tb,ta,n,m,k,&ONE,B,CUDA_R_16BF,ldb,A,CUDA_R_16BF,lda,&beta,C,cb?CUDA_R_16BF:CUDA_R_32F,ldc,
                     CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT),w); }
static void hb_ab (const hbf*A,const hbf*B,void*C,int M,int K,int N,float beta,int cb){ hb_g(CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,B,N,A,K,beta,C,cb,N,"hb_ab"); }
static void hb_abt(const hbf*A,const hbf*B,void*C,int M,int K,int N,float beta,int cb){ hb_g(CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,B,K,A,K,beta,C,cb,N,"hb_abt"); }
static void hb_atb(const hbf*A,const hbf*B,float*C,int M,int K,int N){ hb_g(CUBLAS_OP_N,CUBLAS_OP_T,N,K,M,B,N,A,K,1.0f,C,0,N,"hb_atb"); }
/* grouped bf16 queue (bf16 C only: cuBLAS has no bf16-in/fp32-out grouped GEMM) */
typedef struct { int n; cublasOperation_t ta[MG_MAXP],tb[MG_MAXP];
    int m[MG_MAXP],nn[MG_MAXP],k[MG_MAXP],lda[MG_MAXP],ldb[MG_MAXP],ldc[MG_MAXP],gs[MG_MAXP];
    float al[MG_MAXP],be[MG_MAXP]; const void*A[MG_MAXP],*B[MG_MAXP]; void*C[MG_MAXP]; } MGbq;
static MGbq mgbq; static void **mgb_h=0,**mgb_d=0; static int mgb_slot=0;
static void mgb_push(cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,const hbf*A,int lda,const hbf*B,int ldb,float beta,hbf*C,int ldc){
    MGbq*g=&mgbq; int i=g->n++;
    if(i>=MG_MAXP){ fprintf(stderr,"mgb queue overflow\n"); exit(1); }
    g->ta[i]=ta; g->tb[i]=tb; g->m[i]=m; g->nn[i]=n; g->k[i]=k; g->A[i]=A; g->lda[i]=lda;
    g->B[i]=B; g->ldb[i]=ldb; g->C[i]=C; g->ldc[i]=ldc; g->al[i]=1.0f; g->be[i]=beta; g->gs[i]=1;
}
static void mgb_run(void){
    MGbq*g=&mgbq; if(!g->n) return;
    if(!mgb_h){ ck(cudaHostAlloc((void**)&mgb_h,(size_t)MG_RING*3*MG_MAXP*sizeof(void*),0),"mgb host");
                ck(cudaMalloc((void**)&mgb_d,(size_t)MG_RING*3*MG_MAXP*sizeof(void*)),"mgb dev"); }
    size_t s=(size_t)mgb_slot*3*MG_MAXP; mgb_slot=(mgb_slot+1)%MG_RING;
    void **h=mgb_h+s, **d=mgb_d+s; int n=g->n;
    memcpy(h,g->A,n*sizeof(void*)); memcpy(h+MG_MAXP,g->B,n*sizeof(void*)); memcpy(h+2*MG_MAXP,g->C,n*sizeof(void*));
    ck(cudaMemcpyAsync(d,h,3*MG_MAXP*sizeof(void*),cudaMemcpyHostToDevice,0),"mgb ptr h2d");
    ckb(cublasGemmGroupedBatchedEx(CB,g->ta,g->tb,g->m,g->nn,g->k,g->al,(const void*const*)d,CUDA_R_16BF,g->lda,
        (const void*const*)(d+MG_MAXP),CUDA_R_16BF,g->ldb,g->be,(void*const*)(d+2*MG_MAXP),CUDA_R_16BF,g->ldc,n,g->gs,
        CUBLAS_COMPUTE_32F),"mgb grouped");
    g->n=0;
}
/* row-major per-expert problems, as mg_ab / mg_abt_b / mg_atb (atb writes, beta 0) */
static void mgb_ab(const hbf*A,const hbf*B,hbf*C,int M,int K,int N){ mgb_push(CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,B,N,A,K,0.f,C,N); }
static void mgb_abt(const hbf*A,const hbf*B,hbf*C,int M,int K,int N,float beta){ mgb_push(CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,B,K,A,K,beta,C,N); }
static void mgb_atb(const hbf*A,const hbf*B,hbf*C,int M,int K,int N){ mgb_push(CUBLAS_OP_N,CUBLAS_OP_T,N,K,M,B,N,A,K,0.f,C,N); }
/* gate|up fused: per expert W13[D,2F] (row d = w1[d] | w3[d]), so the expert FFN is one
 * GEMM into gu[ne,2F] and one dgrad/wgrad each way. W2 stays [F,D]. */
__global__ void k_w13bf(const float*w1,const float*w3,hbf*d,int F,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    size_t r=i/F, j=i%F; d[r*2*F+j]=__float2bfloat16(w1[i]); d[r*2*F+F+j]=__float2bfloat16(w3[i]); }
__global__ void k_swig_bf2(const hbf*GU,hbf*O,int F,size_t n2){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n2) return;
    int F2=F/2; size_t r=i/F2, c=i%F2; const hbf2*g=(const hbf2*)(GU+r*2*F);
    float2 z=__bfloat1622float2(g[c]), u=__bfloat1622float2(g[F2+c]);
    ((hbf2*)O)[i]=__floats2bfloat162_rn(z.x/(1.f+__expf(-z.x))*u.x, z.y/(1.f+__expf(-z.y))*u.y); }
__global__ void k_swigb_bf2(const hbf*GU,const hbf*dA,hbf*dGU,int F,size_t n2){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n2) return;
    int F2=F/2; size_t r=i/F2, c=i%F2; const hbf2*g=(const hbf2*)(GU+r*2*F); hbf2*o=(hbf2*)(dGU+r*2*F);
    float2 z=__bfloat1622float2(g[c]), u=__bfloat1622float2(g[F2+c]), a=__bfloat1622float2(((const hbf2*)dA)[i]);
    float s0=1.f/(1.f+__expf(-z.x)), s1=1.f/(1.f+__expf(-z.y));
    o[c]=__floats2bfloat162_rn(a.x*u.x*(s0+z.x*s0*(1.f-s0)), a.y*u.y*(s1+z.y*s1*(1.f-s1)));
    o[F2+c]=__floats2bfloat162_rn(a.x*z.x*s0, a.y*z.y*s1); }
__global__ void k_addbf13(float*G1,float*G3,const hbf*T,size_t st,int K,int F,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    size_t o=(i/F)*2*F+i%F; float s1=0.f,s3=0.f;
    for(int k=0;k<K;k++){ s1+=__bfloat162float(T[(size_t)k*st+o]); s3+=__bfloat162float(T[(size_t)k*st+o+F]); }
    G1[i]+=s1; G3[i]+=s3; }
/* expert weights W13 (E*D*2F) | W2 (E*F*D) in bf16, cast once per forward; UniPool layers share one cache */
static hbf **g_bfm=0; static long *g_bfm_id=0; static size_t *g_bfm_n=0;
static int bfm_slot(int l){   /* PSCRATCH renders every layer into one buffer, so only share without it */
    if(!g_prd && lyT(&gC,l)==2) for(int i=0;i<gC.L;i++) if(lyT(&gC,i)==2) return i;
    return l; }
static hbf* bfm_w(int l,const float*w1,const float*w3,const float*w2,size_t n,int F){
    if(!g_bfm){ g_bfm=(hbf**)calloc(gC.L,sizeof(hbf*)); g_bfm_id=(long*)calloc(gC.L,sizeof(long)); g_bfm_n=(size_t*)calloc(gC.L,sizeof(size_t)); }
    int c=bfm_slot(l);
    if(g_bfm_n[c]<n){ if(g_bfm[c]) cudaFree(g_bfm[c]); g_bfm[c]=(hbf*)dmalloc(3*n*2); g_bfm_n[c]=n; g_bfm_id[c]=-1; }
    if(g_bfm_id[c]!=g_fwd_id){ KL(k_w13bf,nblk(n,TPB),TPB)(w1,w3,g_bfm[c],F,n); bf_cvt(w2,g_bfm[c]+2*n,n); g_bfm_id[c]=g_fwd_id; }
    return g_bfm[c];
}
static int bfm_on(const GA*a,int f8,int sb){ return g_bfmoe && a->tr && !f8 && !moef8(2) && !sb && !g_fp16gemm && !(gC.D&1) && !(gC.F&1); }
/* gu = Xp W13 (bf16 [N,2F] in the gp slice), ap = swiglu(gu), yp = ap W2 */
static void bfm_ffn(hbf*W,GA*a,const int*off,int N,int E,int D,int F,hbf*gu,hbf*yp){
    size_t DF=(size_t)D*F, EDF=(size_t)E*DF; hbf *Xp=(hbf*)a->Xp, *ap=(hbf*)a->ap;
    for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
        mgb_ab(Xp+(size_t)off[e]*D,W+2*e*DF,gu+(size_t)off[e]*2*F,ne,D,2*F); } mgb_run();
    KL(k_swig_bf2,nblk((size_t)N*F/2,TPB),TPB)(gu,ap,F,(size_t)N*F/2);
    for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
        mgb_ab(ap+(size_t)off[e]*F,W+2*EDF+e*DF,yp+(size_t)off[e]*D,ne,F,D); } mgb_run();
}
static void bfm_fwd(const float*P,const Off*rd,GA*a,int l,int N,int E,int Km,int K,float*out,float*gp0,float*up0,float*yp0){
    int D=gC.D,F=gC.F; size_t Np=a->Np; int rb=RB4(D/2); (void)up0;
    hbf *W=bfm_w(l,P+rd->w1[l],P+rd->w3[l],P+rd->w2[l],(size_t)E*D*F,F);
    for(int k=0;k<K;k++){
        int *off=a->hoff+((size_t)l*Km+k)*(gC.n_exp+1), *prm=a->perm[l]+(size_t)k*Np; size_t sk=a->gpS?(size_t)k:0;
        hbf *gu=(hbf*)(gp0+sk*Np*F), *yp=(hbf*)(yp0+sk*Np*D);
        KL(k_gather_bf,N,rb)(a->f_norm[l],prm,(hbf*)a->Xp,D);
        bfm_ffn(W,a,off,N,E,D,F,gu,yp);
        KL(k_scatter_gate_bf,N,rb)(yp,prm,a->gate[l]+(size_t)k*N,out,D);
        if(g_prof>1) kchk("fm scatter",l);
    }
}
static hbf *g_bfT=0; static size_t g_bfTc=0;
/* O2 merged-pass helpers: row j holds token perm[j]%N of pass perm[j]/N */
__global__ void k_gather_bfm(const float*src,const int*perm,hbf*Xp,int D,int N){
    int j=blockIdx.x, n=perm[j]%N; hbf2*o=(hbf2*)(Xp+(size_t)j*D); const float2*s=(const float2*)(src+(size_t)n*D);
    for(int d=threadIdx.x;d<D/2;d+=blockDim.x){ float2 v=s[d]; o[d]=__floats2bfloat162_rn(v.x,v.y); }
}
/* out[n] += sum_k g[k*N+n] * y[inv[k*N+n]] (g NULL = 1), k ascending: same order as the per-pass scatters */
__global__ void k_combine_bf(const hbf*y,const int*inv,const float*g,float*out,int N,int K,int D){
    int n=blockIdx.x; float2*o=(float2*)(out+(size_t)n*D);
    for(int d=threadIdx.x;d<D/2;d+=blockDim.x){ float2 a=o[d];
        for(int k=0;k<K;k++){ size_t i=(size_t)k*N+n; float w=g?g[i]:1.f;
            float2 v=__bfloat1622float2(((const hbf2*)(y+(size_t)inv[i]*D))[d]); a.x+=w*v.x; a.y+=w*v.y; }
        o[d]=a; }
}
__global__ void k_moe_dgate_bfm(const float*dout,const hbf*yp,const int*perm,const float*gate,float*d_gate,
                                hbf*dyp,const float*src,hbf*Xp,int D,int N){
    int j=blockIdx.x, m=perm[j], n=m%N; hbf2 *dy=(hbf2*)(dyp+(size_t)j*D), *xp=(hbf2*)(Xp+(size_t)j*D);
    const float2 *dO=(const float2*)(dout+(size_t)n*D), *sr=(const float2*)(src+(size_t)n*D); const hbf2*y=(const hbf2*)(yp+(size_t)j*D);
    __shared__ float sh[8]; float acc=0.f, g=gate[m];
    for(int d=threadIdx.x;d<D/2;d+=blockDim.x){ float2 a=dO[d], b=__bfloat1622float2(y[d]), s=sr[d];
        acc+=a.x*b.x+a.y*b.y; dy[d]=__floats2bfloat162_rn(a.x*g,a.y*g); xp[d]=__floats2bfloat162_rn(s.x,s.y); }
    for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffffu,acc,o);
    if((threadIdx.x&31)==0) sh[threadIdx.x>>5]=acc;
    __syncthreads();
    if(threadIdx.x==0){ float s=0.f; for(int i=0;i<(int)(blockDim.x>>5);i++) s+=sh[i]; d_gate[m]=s; }
}
/* merged needs the stash (gu/yp hold all K*N rows in the K pass slices), K=2 (the N-row scratch holds
 * 2N bf16 rows) and no capacity padding */
static int mrg_ok(const GA*a,int N,int K){ return g_moemrg && a->gpS && a->ypS && K==2 && a->Np==(size_t)N && gC.L<=64; }
static hbf *g_mdgu=0; static size_t g_mdguc=0;
static void bfm_fwd_mrg(const float*P,const Off*rd,GA*a,int l,int N,int E,int K,float*out,float*gp0,float*yp0){
    int D=gC.D,F=gC.F,KN=K*N,rb=RB4(D/2); const int*off=g_moff+(size_t)l*(gC.n_exp+1);
    hbf *W=bfm_w(l,P+rd->w1[l],P+rd->w3[l],P+rd->w2[l],(size_t)E*D*F,F);
    KL(k_gather_bfm,KN,rb)(a->f_norm[l],a->perm[l],(hbf*)a->Xp,D,N);
    bfm_ffn(W,a,off,KN,E,D,F,(hbf*)gp0,(hbf*)yp0);
    KL(k_combine_bf,N,rb)((hbf*)yp0,g_minv+(size_t)l*KN,a->gate[l],out,N,K,D);
    if(g_prof>1) kchk("fm scatter",l);
}
/* per pass: dyp = gate*dout, dap = dyp W2^T, dgu = swiglu'(gu), dXp = dgu W13^T scattered into
 * dfnorm; weight grads (own grouped call) go to bf16 temps T[k] = T13|T2, summed into G at the end */
static void bfm_bwd(const float*P,float*G,const Off*og,GA*a,int l,int N,int E,int Km,int K,float*gp0,float*up0,float*yp0){
    int D=gC.D,F=gC.F; size_t Np=a->Np, DF=(size_t)D*F, EDF=(size_t)E*DF, st=3*EDF; int rb=RB4(D/2); (void)up0;
    hbf *W=g_bfm[bfm_slot(l)], *T=bf_slot(&g_bfT,&g_bfTc,(size_t)K*st);
    hbf *Xp=(hbf*)a->Xp, *ap=(hbf*)a->ap, *dyp=(hbf*)a->dyp, *dap=(hbf*)a->dap, *dgu=(hbf*)a->dgp, *dXp=(hbf*)a->dXp;
    for(int k=0;k<K;k++){
        int *off=a->hoff+((size_t)l*Km+k)*(gC.n_exp+1), *prm=a->perm[l]+(size_t)k*Np; size_t sk=a->gpS?(size_t)k:0;
        hbf *gu=(hbf*)(gp0+sk*Np*F), *yp=(hbf*)(yp0+sk*Np*D), *T13=T+k*st, *T2=T13+2*EDF;
        if(!a->gpS){ KL(k_gather_bf,N,rb)(a->f_norm[l],prm,Xp,D); bfm_ffn(W,a,off,N,E,D,F,gu,yp); }   /* no stash: recompute */
        KL(k_moe_dgate_bf,N,rb)(a->dout,yp,prm,a->gate[l]+(size_t)k*N,a->d_gate+(size_t)k*N,dyp,a->f_norm[l],Xp,D);
        KL(k_swig_bf2,nblk((size_t)N*F/2,TPB),TPB)(gu,ap,F,(size_t)N*F/2);
        if(g_prof>1) kchk("bm gather+dgate",l);
        for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
            mgb_abt(dyp+(size_t)off[e]*D,W+2*EDF+e*DF,dap+(size_t)off[e]*F,ne,D,F,0.f); } mgb_run();
        KL(k_swigb_bf2,nblk((size_t)N*F/2,TPB),TPB)(gu,dap,dgu,F,(size_t)N*F/2);
        for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
            mgb_abt(dgu+(size_t)off[e]*2*F,W+2*e*DF,dXp+(size_t)off[e]*D,ne,2*F,D,0.f); } mgb_run();
        if(g_prof>1) kchk("bm dgrad gemms",l);
        KL(k_scatter_add_bf,N,rb)(dXp,prm,a->dfnorm,D);
        for(int e=0;e<E;e++){ int ne=off[e+1]-off[e];
            if(!ne){ cudaMemsetAsync(T13+2*e*DF,0,2*DF*2,0); cudaMemsetAsync(T2+e*DF,0,DF*2,0); continue; }
            mgb_atb(Xp+(size_t)off[e]*D,dgu+(size_t)off[e]*2*F,T13+2*e*DF,ne,D,2*F);
            mgb_atb(ap+(size_t)off[e]*F,dyp+(size_t)off[e]*D,T2+e*DF,ne,F,D); } mgb_run();
        if(g_prof>1) kchk("bm wgrad gemms",l);
    }
    KL(k_addbf13,nblk(EDF,TPB),TPB)(G+og->w1[l],G+og->w3[l],T,st,K,F,EDF);
    KL(k_addbfk,nblk(EDF,TPB),TPB)(G+og->w2[l],T+2*EDF,st,K,EDF);
}
static void bfm_bwd_mrg(const float*P,float*G,const Off*og,GA*a,int l,int N,int E,int K,float*gp0,float*yp0){
    int D=gC.D,F=gC.F,KN=K*N,rb=RB4(D/2); size_t DF=(size_t)D*F, EDF=(size_t)E*DF, st=3*EDF;
    const int*off=g_moff+(size_t)l*(gC.n_exp+1), *inv=g_minv+(size_t)l*KN;
    hbf *W=g_bfm[bfm_slot(l)], *T13=bf_slot(&g_bfT,&g_bfTc,st), *T2=T13+2*EDF, *dgu=bf_slot(&g_mdgu,&g_mdguc,(size_t)KN*2*F);
    hbf *Xp=(hbf*)a->Xp, *ap=(hbf*)a->ap, *dyp=(hbf*)a->dyp, *dap=(hbf*)a->dap, *dXp=(hbf*)a->dXp, *gu=(hbf*)gp0, *yp=(hbf*)yp0;
    KL(k_moe_dgate_bfm,KN,rb)(a->dout,yp,a->perm[l],a->gate[l],a->d_gate,dyp,a->f_norm[l],Xp,D,N);
    KL(k_swig_bf2,nblk((size_t)KN*F/2,TPB),TPB)(gu,ap,F,(size_t)KN*F/2);
    if(g_prof>1) kchk("bm gather+dgate",l);
    for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
        mgb_abt(dyp+(size_t)off[e]*D,W+2*EDF+e*DF,dap+(size_t)off[e]*F,ne,D,F,0.f); } mgb_run();
    KL(k_swigb_bf2,nblk((size_t)KN*F/2,TPB),TPB)(gu,dap,dgu,F,(size_t)KN*F/2);
    for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
        mgb_abt(dgu+(size_t)off[e]*2*F,W+2*e*DF,dXp+(size_t)off[e]*D,ne,2*F,D,0.f); } mgb_run();
    if(g_prof>1) kchk("bm dgrad gemms",l);
    KL(k_combine_bf,N,rb)(dXp,inv,(const float*)NULL,a->dfnorm,N,K,D);
    for(int e=0;e<E;e++){ int ne=off[e+1]-off[e];
        if(!ne){ cudaMemsetAsync(T13+2*e*DF,0,2*DF*2,0); cudaMemsetAsync(T2+e*DF,0,DF*2,0); continue; }
        mgb_atb(Xp+(size_t)off[e]*D,dgu+(size_t)off[e]*2*F,T13+2*e*DF,ne,D,2*F);
        mgb_atb(ap+(size_t)off[e]*F,dyp+(size_t)off[e]*D,T2+e*DF,ne,F,D); } mgb_run();
    if(g_prof>1) kchk("bm wgrad gemms",l);
    KL(k_addbf13,nblk(EDF,TPB),TPB)(G+og->w1[l],G+og->w3[l],T13,st,1,F,EDF);
    KL(k_addbfk,nblk(EDF,TPB),TPB)(G+og->w2[l],T2,st,1,EDF);
}
#else
static int mrg_ok(const GA*,int,int){ return 0; }
static void bfm_fwd_mrg(const float*,const Off*,GA*,int,int,int,int,float*,float*,float*){}
static void bfm_bwd_mrg(const float*,float*,const Off*,GA*,int,int,int,int,float*,float*){}
static int g_bffn=0, g_bfmoe=0; static long g_fwd_id=0;
static int bfm_on(const GA*,int,int){ return 0; }
static void bfm_fwd(const float*,const Off*,GA*,int,int,int,int,int,float*,float*,float*,float*){}
static void bfm_bwd(const float*,float*,const Off*,GA*,int,int,int,int,int,float*,float*,float*){}
#endif
/* router z-loss (CUDALM_ROUTER_Z=c): L += c * mean_n lse(logits_n)^2, training GAs only. Keeps router
 * logits from growing until the softmax saturates and experts die (ST-MoE). lse kept per layer. */
static float g_zl=0, *g_zlse=0; static size_t g_zlc=0;
#ifndef CUDALM_CPU
__global__ void k_rlse(const float*rt,float*lse,int N,int E){
    int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N) return; const float*r=rt+(size_t)n*E;
    float mx=-1e30f; for(int e=0;e<E;e++) mx=fmaxf(mx,r[e]);
    float s=0; for(int e=0;e<E;e++) s+=expf(r[e]-mx); lse[n]=mx+logf(s); }
__global__ void k_zl_bwd(float*drt,const float*pr,const float*lse,int N,int E,float c){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<(size_t)N*E) drt[i]+=c*lse[i/E]*pr[i]; }
#else
#define CPUK_PART 12
#include "cpu_kern.h"
#endif
/* 1/N over every routed token (masked ones included: the router still sees them), as in
 * ST-MoE and as the load-balance aux loss here; not the CE's masked denominator */
static void zl_fwd(GA*a,int l,int N,int E){
    if(g_zl<=0||!a->tr) return;
    size_t need=(size_t)gC.L*N; if(need>g_zlc){ if(g_zlse) cudaFree(g_zlse); g_zlse=(float*)dmalloc(need*4); g_zlc=need; }
    KL(k_rlse,nblk(N,TPB),TPB)(a->rt,g_zlse+(size_t)l*N,N,E); }
static void zl_bwd(GA*a,int l,int N,int E){
    if(g_zl>0&&a->tr) KL(k_zl_bwd,nblk((size_t)N*E,TPB),TPB)(a->drt,a->rprobs[l],g_zlse+(size_t)l*N,N,E,2.f*g_zl/N); }
/* the z-loss of the last training forward, sum over sparse layers of c*mean(lse^2):
 * the CE loss in the log line does not include it */
static double zl_value(int N){
    if(g_zl<=0||!g_zlse||g_zlc<(size_t)gC.L*N) return 0.0;
    size_t n=(size_t)gC.L*N; float*h=(float*)malloc(n*4);
    ck(cudaMemcpy(h,g_zlse,n*4,cudaMemcpyDeviceToHost),"zl d2h");
    double z=0; for(int l=0;l<gC.L;l++){ if(!lyT(&gC,l)) continue;
        double s=0; for(int i=0;i<N;i++){ double v=h[(size_t)l*N+i]; s+=v*v; } z+=g_zl*s/N; }
    free(h); return z; }
static void moe_forward(const float*P,const Off*o,GA*a,int l,float*out,int N){
#define KF(s) if(g_prof>1) kchk(s,l)
    int D=gC.D,F=gC.F,E=lyE(&gC,l), isP=lyT(&gC,l)==2;
    const Off*rd=g_prd?g_prd:o;
    /* with the stash on, the expert outputs land straight in the per-layer
     * buffers; backward then reads them instead of re-running these GEMMs */
    float *gp=a->gpS?a->gpS[l]:a->gp, *up=a->upS?a->upS[l]:a->up, *yp=a->ypS?a->ypS[l]:a->yp;
    g_ab(g_rpre?rpre_x(P,rd,a,l,N):a->f_norm[l],P+rd->wr[l],a->rt,N,D,E);   /* router logits */
    if(g_topk<=1&&(g_t1old||g_route_mode)){
      KL(k_router_top1,nblk(N,TPB),TPB)(a->rt,a->rprobs[l],a->assign[l],a->gate[l],N,E,mbias(l)); zl_fwd(a,l,N,E);
      moe_perm(a,l,N,E);
      int *off=a->hoff+(size_t)l*(gC.n_exp+1);
      e_cpy(out,a->f_in[l],(size_t)N*D);                             /* residual */
      KL(k_gather,nblk((size_t)N*D,TPB),TPB)(a->f_norm[l],a->perm[l],a->Xp,N,D);
      for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
          mg_ab(a->Xp+(size_t)off[e]*D,P+rd->w1[l]+(size_t)e*D*F,gp+(size_t)off[e]*F,ne,D,F);
          mg_ab(a->Xp+(size_t)off[e]*D,P+rd->w3[l]+(size_t)e*D*F,up+(size_t)off[e]*F,ne,D,F); } mg_flush();
      e_swig(gp,up,a->ap,(size_t)N*F);
      for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
          mg_ab(a->ap+(size_t)off[e]*F,P+rd->w2[l]+(size_t)e*F*D,yp+(size_t)off[e]*D,ne,F,D); } mg_flush();
      KL(k_scatter_gate,nblk((size_t)N*D,TPB),TPB)(yp,a->perm[l],a->gate[l],out,N,D);
    } else {
      int Km=g_topk<1?1:g_topk; if(Km>gC.n_exp)Km=gC.n_exp; int K=(g_topkL[l]>0&&g_topkL[l]<Km)?g_topkL[l]:Km; if(K>E)K=E;
      KL(k_router_topk,nblk(N,TPB),TPB)(a->rt,a->rprobs[l],a->assign[l],a->gate[l],N,E,K,mbias(l),g_mbias_slot?E:0); zl_fwd(a,l,N,E);
      if(g_moe_train&&(g_moe_rcap>0||g_moe_sink>0)) moe_rebal(a,l,N,E,K);
      float *gp0=gp, *up0=up, *yp0=yp;
      e_cpy(out,a->f_in[l],(size_t)N*D);                             /* residual, once */
      int f8f=moef8(1), Fp=(F+15)&~15;
      if(f8f){ dim3 tb(32,32);
        mq_scale(P+rd->w1[l],(size_t)E*D*F,mf_sc(0));
        KL(k_qtrp_b,dim3((F+31)/32,(D+31)/32,E),tb)(P+rd->w1[l],mf_buf(3,(size_t)E*F*D),D,F,D,mf_sc(0)+2);
        mq_scale(P+rd->w3[l],(size_t)E*D*F,mf_sc(1));
        KL(k_qtrp_b,dim3((F+31)/32,(D+31)/32,E),tb)(P+rd->w3[l],mf_buf(4,(size_t)E*F*D),D,F,D,mf_sc(1)+2);
        mq_scale(P+rd->w2[l],(size_t)E*F*D,mf_sc(2));
        KL(k_qtrp_b,dim3((D+31)/32,(Fp+31)/32,E),tb)(P+rd->w2[l],mf_buf(5,(size_t)E*D*Fp),F,D,Fp,mf_sc(2)+2);
      }
      int sb=g_moe_sb&&!f8f&&!moef8(2)&&!g_fp16gemm&&!(D&3)&&!(F&3); size_t Np=a->Np;
      int mg=bfm_on(a,f8f,sb)&&mrg_ok(a,N,K);
      KF("fm router+wq"); if(!(mg&&moe_gperm(a,l,N,E,Km,K,out,1))){ mg=0; moe_perm_all(a,l,N,E,Km,K,sb,out); } KF("fm perm");
      if(l<64) g_mrgL[l]=mg;
      if(mg){ bfm_fwd_mrg(P,rd,a,l,N,E,K,out,gp0,yp0); return; }
      if(bfm_on(a,f8f,sb)){ bfm_fwd(P,rd,a,l,N,E,Km,K,out,gp0,up0,yp0); return; }
      for(int k=0;k<K;k++){                                          /* each pass = one top-1 FFN, summed */
        int *off=a->hoff+((size_t)l*Km+k)*(gC.n_exp+1), cap=a->hcap[(size_t)l*Km+k], nr=cap?E*cap:N;
        int *prm=a->perm[l]+(size_t)k*Np;
        KL(k_gather,nblk((size_t)nr*D,TPB),TPB)(a->f_norm[l],prm,a->Xp,nr,D);
        size_t sk=a->gpS?(size_t)k:0;                                /* stash: one slice per pass */
        float *gp=gp0+sk*Np*F, *up=up0+sk*Np*F, *yp=yp0+sk*Np*D;
        size_t cD=(size_t)cap*D, cF=(size_t)cap*F, DF=(size_t)D*F;
        if(f8f){
          __nv_fp8_e4m3 *Xq=mf_buf(0,(size_t)N*D);
          mq_scale(a->Xp,(size_t)N*D,mf_sc(3));
          KL(k_quant_e4m3,nblk((size_t)N*D,TPB),TPB)(a->Xp,Xq,(size_t)N*D,mf_sc(3)+2); KF("fm gather+xq");
          for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
            fp8_matmul(Xq+(size_t)off[e]*D,mf8[3]+(size_t)e*F*D,mf_sc(3)+1,mf_sc(0)+1,ne,F,D,&ONE,&ZERO,gp+(size_t)off[e]*F,F,"moe_w1_f8");
            fp8_matmul(Xq+(size_t)off[e]*D,mf8[4]+(size_t)e*F*D,mf_sc(3)+1,mf_sc(1)+1,ne,F,D,&ONE,&ZERO,up+(size_t)off[e]*F,F,"moe_w3_f8"); } KF("fm w13");
        } else if(cap){
          sb_ab(a->Xp,cD,P+rd->w1[l],DF,gp,cF,cap,D,F,E);
          sb_ab(a->Xp,cD,P+rd->w3[l],DF,up,cF,cap,D,F,E);
        } else {
          for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
            mg_ab(a->Xp+(size_t)off[e]*D,P+rd->w1[l]+(size_t)e*D*F,gp+(size_t)off[e]*F,ne,D,F);
            mg_ab(a->Xp+(size_t)off[e]*D,P+rd->w3[l]+(size_t)e*D*F,up+(size_t)off[e]*F,ne,D,F); } mg_flush();
        }
        e_swig(gp,up,a->ap,(size_t)nr*F);
        if(f8f){
          __nv_fp8_e4m3 *Aq=mf_buf(1,(size_t)N*Fp);
          mq_scale(a->ap,(size_t)N*F,mf_sc(4));
          KL(k_quant_pad,nblk((size_t)N*Fp,TPB),TPB)(a->ap,Aq,(size_t)N,F,Fp,mf_sc(4)+2); KF("fm swig+aq");
          for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
            fp8_matmul(Aq+(size_t)off[e]*Fp,mf8[5]+(size_t)e*D*Fp,mf_sc(4)+1,mf_sc(2)+1,ne,D,Fp,&ONE,&ZERO,yp+(size_t)off[e]*D,D,"moe_w2_f8"); } KF("fm w2");
        } else if(cap) sb_ab(a->ap,cF,P+rd->w2[l],DF,yp,cD,cap,F,D,E);
        else {
          for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
            mg_ab(a->ap+(size_t)off[e]*F,P+rd->w2[l]+(size_t)e*F*D,yp+(size_t)off[e]*D,ne,F,D); } mg_flush();
        }
        KL(k_scatter_gate,nblk((size_t)nr*D,TPB),TPB)(yp,prm,a->gate[l]+(size_t)k*N,out,nr,D); KF("fm scatter");
      }
    }
}

/* MoE FFN backward for layer l: consumes a->dout, produces a->dfnorm. */
static void moe_backward(const float*P,float*G,const Off*o,GA*a,int l,int N){
    const Off*og = g_og ? g_og : o;
    const Off*rd = g_prd ? g_prd : o;
    int D=gC.D,F=gC.F,E=lyE(&gC,l), isP=lyT(&gC,l)==2;
    float *gp=a->gpS?a->gpS[l]:a->gp, *up=a->upS?a->upS[l]:a->up, *yp=a->ypS?a->ypS[l]:a->yp;
  if(g_topk<=1&&(g_t1old||g_route_mode)){
    int *off=a->hoff+(size_t)l*(gC.n_exp+1);
    /* Recompute the expert forward, exactly as the CPU path does -- unless the
     * stash kept it. Xp and ap are rebuilt either way: a gather and an
     * elementwise swiglu are pure memory traffic, while w1/w3/w2 are three of
     * the nine expert GEMMs and ~9% of every training step's FLOPs. Stashed and
     * recomputed values are bit-identical, so this changes no result. */
    KL(k_gather,nblk((size_t)N*D,TPB),TPB)(a->f_norm[l],a->perm[l],a->Xp,N,D);
    if(!a->gpS) for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
        mg_ab(a->Xp+(size_t)off[e]*D,P+rd->w1[l]+(size_t)e*D*F,gp+(size_t)off[e]*F,ne,D,F);
        mg_ab(a->Xp+(size_t)off[e]*D,P+rd->w3[l]+(size_t)e*D*F,up+(size_t)off[e]*F,ne,D,F); } mg_flush();
    e_swig(gp,up,a->ap,(size_t)N*F);
    if(!a->ypS) for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
        mg_ab(a->ap+(size_t)off[e]*F,P+rd->w2[l]+(size_t)e*F*D,yp+(size_t)off[e]*D,ne,F,D); } mg_flush();
    /* d_gate (pre-gate y . dout) and the gated dy */
    KL(k_moe_dgate,N,TPB)(a->dout,yp,a->perm[l],a->gate[l],a->d_gate,a->dyp,N,D);
    for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
        mg_atb(a->ap+(size_t)off[e]*F,a->dyp+(size_t)off[e]*D,G+og->w2[l]+(size_t)e*F*D,ne,F,D);
        mg_abt(a->dyp+(size_t)off[e]*D,P+rd->w2[l]+(size_t)e*F*D,a->dap+(size_t)off[e]*F,ne,D,F); } mg_flush();
    KL(k_swiglu_bwd,nblk((size_t)N*F,TPB),TPB)(gp,up,a->dap,a->dgp,a->dup,(size_t)N*F);
    for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
        mg_atb(a->Xp+(size_t)off[e]*D,a->dgp+(size_t)off[e]*F,G+og->w1[l]+(size_t)e*D*F,ne,D,F);
        mg_atb(a->Xp+(size_t)off[e]*D,a->dup+(size_t)off[e]*F,G+og->w3[l]+(size_t)e*D*F,ne,D,F);
        mg_abt_b(a->dgp+(size_t)off[e]*F,P+rd->w1[l]+(size_t)e*D*F,a->dXp+(size_t)off[e]*D,ne,F,D,0.0f);
        mg_abt_b(a->dup+(size_t)off[e]*F,P+rd->w3[l]+(size_t)e*D*F,a->dXp+(size_t)off[e]*D,ne,F,D,1.0f); } mg_flush();
    ck(cudaMemset(a->dfnorm,0,(size_t)N*D*4),"zero dfnorm");
    KL(k_scatter_add,nblk((size_t)N*D,TPB),TPB)(a->dXp,a->perm[l],a->dfnorm,N,D);
    /* router backward: load-balance aux + gate path */
    { int cnt[256], ps[256]={0};
      for(int e=0;e<E;e++) cnt[e]=off[e+1]-off[e];
      if(isP) for(int ll=0;ll<gC.L;ll++){ if(lyT(&gC,ll)!=2) continue; int *o2=a->hoff+(size_t)ll*(gC.n_exp+1);
          for(int e=0;e<E;e++) ps[e]+=o2[e+1]-o2[e]; }
      ck(cudaMemcpy(a->d_cnt,isP?ps:cnt,(size_t)E*4,cudaMemcpyHostToDevice),"cnt h2d"); mbias_upd(l,cnt,1,E,(float)N/E); }
    KL(k_router_bwd,nblk(N,TPB),TPB)(a->rprobs[l],a->assign[l],a->d_gate,a->d_cnt,
                                      a->drt,N,E,(isP?MOE_AUX/NPOOL(&gC):MOE_AUX)*LMUL(g_auxlm,g_auxlm_n,l));
    zl_bwd(a,l,N,E);
    if(g_rpre) rpre_bwd(P,G,og,rd,a,l,N); else {
    g_abt_b(a->drt,P+rd->wr[l],a->dfnorm,N,E,D,1.0f);      /* dfnorm += drt @ wr^T */
    g_atb(a->f_norm[l],a->drt,G+og->wr[l],N,D,E); }        /* dWr += fnorm^T @ drt */
  } else {
    int Km=g_topk<1?1:g_topk; if(Km>gC.n_exp)Km=gC.n_exp; int K=(g_topkL[l]>0&&g_topkL[l]<Km)?g_topkL[l]:Km; if(K>E)K=E;
    /* Top-K: each pass is one top-1 expert backward, accumulated. G is zeroed
     * once per step and g_atb has beta=1, so an expert receiving rows from both
     * passes sums its weight grads correctly. dfnorm is zeroed once and each
     * pass's dXp scatter-adds in; router grads for all passes fold into one
     * softmax jacobian. With the stash on, each pass has its own gp/up/yp
     * slice (K per layer); without it they alias one buffer and are recomputed. */
    float *gp0=gp, *up0=up, *yp0=yp;
    ck(cudaMemset(a->dfnorm,0,(size_t)N*D*4),"zero dfnorm");
    int f8b=moef8(2), Fp=(F+15)&~15;
    if(f8b){
        mq_scale(P+rd->w2[l],(size_t)E*F*D,mf_sc(5));
        KL(k_quant_e4m3,nblk((size_t)E*F*D,TPB),TPB)(P+rd->w2[l],mf_buf(6,(size_t)E*F*D),(size_t)E*F*D,mf_sc(5)+2);
        mq_scale(P+rd->w1[l],(size_t)E*D*F,mf_sc(6));
        KL(k_quant_pad,nblk((size_t)E*D*Fp,TPB),TPB)(P+rd->w1[l],mf_buf(7,(size_t)E*D*Fp),(size_t)E*D,F,Fp,mf_sc(6)+2);
        mq_scale(P+rd->w3[l],(size_t)E*D*F,mf_sc(7));
        KL(k_quant_pad,nblk((size_t)E*D*Fp,TPB),TPB)(P+rd->w3[l],mf_buf(8,(size_t)E*D*Fp),(size_t)E*D,F,Fp,mf_sc(7)+2);
    }
    size_t Np=a->Np;
    if(l<64&&g_mrgL[l]) bfm_bwd_mrg(P,G,og,a,l,N,E,K,gp0,yp0);
    else if(bfm_on(a,moef8(1),g_moe_sb&&!moef8(1)&&!f8b&&!g_fp16gemm&&!(D&3)&&!(F&3))) bfm_bwd(P,G,og,a,l,N,E,Km,K,gp0,up0,yp0);
    else for(int k=0;k<K;k++){
        int *off=a->hoff+((size_t)l*Km+k)*(gC.n_exp+1), cap=a->hcap[(size_t)l*Km+k], nr=cap?E*cap:N;
        int *prm=a->perm[l]+(size_t)k*Np;
        size_t cD=(size_t)cap*D, cF=(size_t)cap*F, DF=(size_t)D*F;
        float *gk=a->gate[l]+(size_t)k*N, *dgk=a->d_gate+(size_t)k*N;
        int v4=!(D&3)&&!(F&3), D4=D/4, rb=RB4(D4);
        if(!v4) KL(k_gather,nblk((size_t)nr*D,TPB),TPB)(a->f_norm[l],prm,a->Xp,nr,D);
        else if(!a->gpS) KL(k_gather4,nr,rb)((const float4*)a->f_norm[l],prm,(float4*)a->Xp,D4);
        size_t sk=a->gpS?(size_t)k:0;
        float *gp=gp0+sk*Np*F, *up=up0+sk*Np*F, *yp=yp0+sk*Np*D;
        if(!a->gpS){ if(cap){ sb_ab(a->Xp,cD,P+rd->w1[l],DF,gp,cF,cap,D,F,E); sb_ab(a->Xp,cD,P+rd->w3[l],DF,up,cF,cap,D,F,E); }
          else { for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
            mg_ab(a->Xp+(size_t)off[e]*D,P+rd->w1[l]+(size_t)e*D*F,gp+(size_t)off[e]*F,ne,D,F);
            mg_ab(a->Xp+(size_t)off[e]*D,P+rd->w3[l]+(size_t)e*D*F,up+(size_t)off[e]*F,ne,D,F); } mg_flush(); } }
        e_swig(gp,up,a->ap,(size_t)nr*F);
        if(!a->ypS){ if(cap) sb_ab(a->ap,cF,P+rd->w2[l],DF,yp,cD,cap,F,D,E);
          else { for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
            mg_ab(a->ap+(size_t)off[e]*F,P+rd->w2[l]+(size_t)e*F*D,yp+(size_t)off[e]*D,ne,F,D); } mg_flush(); } }
        if(v4) KL(k_moe_dgate4,nr,rb)((const float4*)a->dout,(const float4*)yp,prm,gk,dgk,(float4*)a->dyp,
                                      a->gpS?(const float4*)a->f_norm[l]:NULL,(float4*)a->Xp,D4);
        else KL(k_moe_dgate,N,TPB)(a->dout,yp,prm,gk,dgk,a->dyp,N,D);
        kchk("bm gather+dgate",l);
        if(f8b){
            for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
                mg_atb(a->ap+(size_t)off[e]*F,a->dyp+(size_t)off[e]*D,G+og->w2[l]+(size_t)e*F*D,ne,F,D); } mg_flush();
            __nv_fp8_e4m3 *Xq=mf_buf(0,(size_t)N*D);
            mq_scale(a->dyp,(size_t)N*D,mf_sc(8));
            KL(k_quant_e4m3,nblk((size_t)N*D,TPB),TPB)(a->dyp,Xq,(size_t)N*D,mf_sc(8)+2);
            for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
                fp8_matmul(Xq+(size_t)off[e]*D,mf8[6]+(size_t)e*F*D,mf_sc(8)+1,mf_sc(5)+1,ne,F,D,&ONE,&ZERO,a->dap+(size_t)off[e]*F,F,"moe_w2d_f8"); }
            kchk("bm w2 gemms",l);
        } else if(cap){
            sb_atb(a->ap,cF,a->dyp,cD,G+og->w2[l],DF,cap,F,D,E);
            sb_abt(a->dyp,cD,P+rd->w2[l],DF,a->dap,cF,cap,D,F,0.0f,E); kchk("bm w2 gemms",l);
        } else {
        for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
            mg_atb(a->ap+(size_t)off[e]*F,a->dyp+(size_t)off[e]*D,G+og->w2[l]+(size_t)e*F*D,ne,F,D);
            mg_abt(a->dyp+(size_t)off[e]*D,P+rd->w2[l]+(size_t)e*F*D,a->dap+(size_t)off[e]*F,ne,D,F); } mg_flush(); kchk("bm w2 gemms",l);
        }
        /* nr, not N: a capacity-padded pass (MOE_SB) has E*cap > N rows */
        if(v4) KL(k_swiglu_bwd4,nblk((size_t)nr*F/4,TPB),TPB)((const float4*)gp,(const float4*)up,(const float4*)a->dap,
                                                            (float4*)a->dgp,(float4*)a->dup,(size_t)nr*F/4);
        else KL(k_swiglu_bwd,nblk((size_t)N*F,TPB),TPB)(gp,up,a->dap,a->dgp,a->dup,(size_t)N*F);
        kchk("bm swiglu",l);
        if(f8b){
            for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
                mg_atb(a->Xp+(size_t)off[e]*D,a->dgp+(size_t)off[e]*F,G+og->w1[l]+(size_t)e*D*F,ne,D,F);
                mg_atb(a->Xp+(size_t)off[e]*D,a->dup+(size_t)off[e]*F,G+og->w3[l]+(size_t)e*D*F,ne,D,F); } mg_flush();
            __nv_fp8_e4m3 *Aq=mf_buf(1,(size_t)N*Fp), *Aq2=mf_buf(2,(size_t)N*Fp);
            mq_scale(a->dgp,(size_t)N*F,mf_sc(9));
            KL(k_quant_pad,nblk((size_t)N*Fp,TPB),TPB)(a->dgp,Aq,(size_t)N,F,Fp,mf_sc(9)+2);
            mq_scale(a->dup,(size_t)N*F,mf_sc(10));
            KL(k_quant_pad,nblk((size_t)N*Fp,TPB),TPB)(a->dup,Aq2,(size_t)N,F,Fp,mf_sc(10)+2);
            for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
                fp8_matmul(Aq+(size_t)off[e]*Fp,mf8[7]+(size_t)e*D*Fp,mf_sc(9)+1,mf_sc(6)+1,ne,D,Fp,&ONE,&ZERO,a->dXp+(size_t)off[e]*D,D,"moe_w1d_f8");
                fp8_matmul(Aq2+(size_t)off[e]*Fp,mf8[8]+(size_t)e*D*Fp,mf_sc(10)+1,mf_sc(7)+1,ne,D,Fp,&ONE,&ONE,a->dXp+(size_t)off[e]*D,D,"moe_w3d_f8"); }
            kchk("bm w13 gemms",l);
        } else if(cap){
            sb_atb(a->Xp,cD,a->dgp,cF,G+og->w1[l],DF,cap,D,F,E);
            sb_atb(a->Xp,cD,a->dup,cF,G+og->w3[l],DF,cap,D,F,E);
            sb_abt(a->dgp,cF,P+rd->w1[l],DF,a->dXp,cD,cap,F,D,0.0f,E);
            sb_abt(a->dup,cF,P+rd->w3[l],DF,a->dXp,cD,cap,F,D,1.0f,E); kchk("bm w13 gemms",l);
        } else {
        for(int e=0;e<E;e++){ int ne=off[e+1]-off[e]; if(!ne) continue;
            mg_atb(a->Xp+(size_t)off[e]*D,a->dgp+(size_t)off[e]*F,G+og->w1[l]+(size_t)e*D*F,ne,D,F);
            mg_atb(a->Xp+(size_t)off[e]*D,a->dup+(size_t)off[e]*F,G+og->w3[l]+(size_t)e*D*F,ne,D,F);
            mg_abt_b(a->dgp+(size_t)off[e]*F,P+rd->w1[l]+(size_t)e*D*F,a->dXp+(size_t)off[e]*D,ne,F,D,0.0f);
            mg_abt_b(a->dup+(size_t)off[e]*F,P+rd->w3[l]+(size_t)e*D*F,a->dXp+(size_t)off[e]*D,ne,F,D,1.0f); } mg_flush(); kchk("bm w13 gemms",l);
        }
        if(v4) KL(k_scatter_add4,nr,rb)((const float4*)a->dXp,prm,(float4*)a->dfnorm,D4);
        else KL(k_scatter_add,nblk((size_t)N*D,TPB),TPB)(a->dXp,prm,a->dfnorm,N,D);
        kchk("bm scatter",l);
    }
    { int cnt[256], cs[8*256], ps[256]={0}; for(int e=0;e<E;e++) cnt[e]=0;
      for(int k=0;k<K;k++){ int *off=a->hoff+((size_t)l*Km+k)*(gC.n_exp+1);
          for(int e=0;e<E;e++){ cs[k*E+e]=off[e+1]-off[e]; cnt[e]+=cs[k*E+e]; } }
      static float *pc=NULL; if(!pc) ck(cudaHostAlloc((void**)&pc,64*256*sizeof(float),0),"cnt pin");
      float *hc=pc+(size_t)(l&63)*256, invN=1.0f/N, invKN=1.0f/((float)K*N), A=isP?MOE_AUX/NPOOL(&gC):MOE_AUX;
      if(isP)                                      /* UniPool: pool-level f = sum over the P layers */
          for(int ll=0;ll<gC.L;ll++){ if(lyT(&gC,ll)!=2) continue; int Kl=(g_topkL[ll]>0&&g_topkL[ll]<Km)?g_topkL[ll]:Km; if(Kl>E)Kl=E;
              for(int k=0;k<Kl;k++){ int *o2=a->hoff+((size_t)ll*Km+k)*(gC.n_exp+1);
                  for(int e=0;e<E;e++) ps[e]+=o2[e+1]-o2[e]; } }
      float lm=LMUL(g_auxlm,g_auxlm_n,l);
      for(int e=0;e<E;e++) hc[e]=lm*(A*E*((float)(isP?ps[e]:cnt[e])*invKN)*invN
                                 +(g_moe_auxL>0?g_moe_auxL*E*((float)cnt[e]*invKN)*invN:0.f));
      ck(cudaMemcpyAsync(a->d_cnt,hc,(size_t)E*4,cudaMemcpyHostToDevice,0),"cnt h2d");
      if(g_mbias_slot) mbias_upd(l,cs,K,E,(float)N/E); else mbias_upd(l,cnt,1,E,(float)K*N/E); }
    KL(k_router_bwd_topk,nblk(N,TPB),TPB)(a->rprobs[l],a->assign[l],a->d_gate,(const float*)a->d_cnt,
                                           a->drt,N,E,K);
    zl_bwd(a,l,N,E);
    if(g_rpre) rpre_bwd(P,G,og,rd,a,l,N); else {
    g_abt_b(a->drt,P+rd->wr[l],a->dfnorm,N,E,D,1.0f);      /* dfnorm += drt @ wr^T */
    g_atb(a->f_norm[l],a->drt,G+og->wr[l],N,D,E); }        /* dWr += fnorm^T @ drt */
  }
}

/* ===================== n-gram embeddings (Stage 1: fp32 scaffold) =========
 * CUDALM_NGRAM=1 turns on an independent bigram-embedding path, orthogonal to
 * qlm. Per position t (t within its T-block, t%T!=0), the pair (tok[t-1],tok[t])
 * is hashed into one of R rows of a [R x De] table; the gathered De-vector is
 * projected by Wng[De x D] and ADDED into the block-0 input a_in[0], beside the
 * ordinary token embedding. Backward reuses a->dout (= dL/d a_in[0], shared with
 * k_embed_bwd). Table+Wng are updated densely with the stock bf16 AdamW kernel.
 * Wng is zero-initialised, so at step 0 the path is an exact no-op and the loss
 * equals the baseline; any divergence thereafter is the n-gram learning. No
 * checkpoint persistence yet -- this stage only proves the train-time pipeline. */
static int   g_ngram_on=0;
/* clip multiplier [mult, norm] for every parameter outside P/G (tables, projections, canon, shared
 * expert, attention gate); set each step by gpu_adam (or the fused tail), NULL = unclipped */
static const float *g_side_gs=NULL;
static int   g_ng_R=0, g_ng_De=0, g_ng_order=2;
static int   g_ng_fh[4];   /* NGR1 header of the table being loaded for inference */
static float *g_ng=0, *g_ngG=0;      /* table + its grad (R*De) */
static bf16  *g_ngM=0, *g_ngV=0;     /* table adam moments */
static float *g_ngW=0, *g_ngWG=0;    /* projection De*D + grad */
static bf16  *g_ngWM=0, *g_ngWV=0;   /* projection adam moments */
static float *g_ngbuf=0, *g_ngproj=0, *g_ngdbuf=0;  /* scratch (grow with N) */
static int   *g_ngkey=0, g_ng_ncap=0;
static int   g_ng_q4=0;                         /* NGRAM_Q4: 4-bit table */
static signed char  *g_ngq_d=0,*g_ngq_r=0,*g_ngq_m8=0;
static unsigned char *g_ngq_v8=0;
static float *g_ngq_s=0,*g_ngq_ms=0,*g_ngq_vs=0;
static QSpanD g_ngq_spH; static QSpanD *g_ngq_spD=0;
static int    g_ngq_ngrp=0, g_ngq_grp=128, g_ngq_dmax=7;
/* NGRAM_Q4_FOLD: drop the full-table fp32 grad (g_ngG, R*De*4). Instead scatter
 * this step's gradients into a compact per-touched-group buffer and update only
 * the touched groups with a private copy of the qlm integer-AdamW step. Rows in
 * groups untouched this step keep their state (lazy embedding update) -- rows in
 * a touched group still see the dense update since the whole group is stepped.
 * Requires q4 and group %% De == 0 (each row lies wholly inside one group). */
static int   g_ng_q4fold=0;
static float *g_ngGc=0;                 /* compact grad, g_ng_gcap*grp floats */
static int   *g_ng_cidxD=0,*g_ng_tgiD=0;/* device: pos->compact group, compact->real group */
static int   *g_ng_hk=0,*g_ng_hgi=0,*g_ng_hord=0,*g_ng_hcidx=0,*g_ng_htgi=0;
static int    g_ng_gcap=0, g_ng_Gt=0;   /* Gc capacity (groups), touched-group count */
/* FNV-1a fold over the last `order` tokens ending at n. For order=2 this is
 * byte-identical to the previous 2-arg hash, so the bigram path is unchanged.
 * Positions with < order-1 tokens of history in their sequence emit key=-1
 * (no-op gather/scatter), the same boundary rule the bigram used at n%T==0. */
#ifndef CUDALM_CPU
__global__ void k_ngram_key(const int*tok,int*key,int N,int T,int R,int order,int base){
    int n=(int)(blockIdx.x*blockDim.x+threadIdx.x); if(n>=N)return;
    if(n%T < order-1){ key[n]=-1; return; }
    unsigned h=2166136261u;
    for(int j=order-1;j>=0;j--) h=(h^(unsigned)tok[n-j])*16777619u;
    h^=h>>15;
    key[n]=base+(int)(h%(unsigned)R); }
__global__ void k_ngram_gather(const float*tbl,const int*key,float*buf,int N,int De,int nk){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(size_t)N*De)return;
    int n=(int)(i/De), e=(int)(i%De), k=key[n];
    float s= k>=0 ? tbl[(size_t)k*De+e] : 0.0f;
    for(int j=1;j<nk;j++){ k=key[j*N+n]; if(k>=0) s+=tbl[(size_t)k*De+e]; }
    buf[i]=s; }
__global__ void k_ngram_scatter(float*grad,const int*key,const float*dbuf,int N,int De,int Nd){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(size_t)N*De)return;
    int n=(int)(i/De), e=(int)(i%De), k=key[n];
    if(k>=0) atomicAdd(&grad[(size_t)k*De+e], dbuf[(size_t)(n%Nd)*De+e]); }
/* NGRAM_Q4_FOLD scatter: dbuf -> compact per-touched-group buffer Gc. cidx[n] is
 * the compact group slot of position n's row; the element's offset inside its
 * group is (row*De+e) %% group (rows align to group since group %% De == 0). */
__global__ void k_ng_scatter_compact(float*Gc,const int*key,const int*cidx,
                                     const float*dbuf,int N,int De,int group,int Nd){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(size_t)N*De)return;
    int n=(int)(i/De), e=(int)(i%De), r=key[n]; if(r<0) return;
    int c=cidx[n]; if(c<0) return;
    int o=(int)(((long long)r*De + e) % group);
    atomicAdd(&Gc[(size_t)c*group + o], dbuf[(size_t)(n%Nd)*De+e]);
}
/* Private copy of qlm.cuh k_qlm_step specialised to the ngram's single 4-bit
 * span (packed, adam, lrmode=rel via frozen sref, gscale=1, no stat), driven by
 * a list of touched group indices instead of qspan_of -- so only touched groups
 * are updated and no full-table grad buffer is needed. Math mirrors k_qlm_step
 * exactly; keep in sync if that kernel changes. */
__global__ void k_ng_q4_step(float*P,const float*Gc,
                             signed char*d,signed char*r,float*s,
                             signed char*m8,float*m8s,unsigned char*v8,float*v8s,
                             const int*touched_gi,int group,int dmax,
                             float lr,float b1,float b2,float wd,float sref,
                             float inv_bc1,float inv_sqrt_bc2,
                             unsigned long long t,unsigned long long nt,const float*gs){
    int gi=touched_gi[blockIdx.x];
    unsigned long long j0=(unsigned long long)gi*group;      /* span starts at 0 */
    int n=(int)min((unsigned long long)group, nt - j0);
    int j=threadIdx.x;
    unsigned long long pof=j0, qi0=j0;
    extern __shared__ signed char shd[];
    if(j<blockDim.x) shd[j]=0;
    float sg=s[gi];
    float L=lr*128.0f/sref;                                  /* lrmode=rel(0) */
    float dec=lr*wd*128.0f;
    float ms_old=m8s[gi], vs_old=v8s[gi];
    float mn=0.0f, rv=0.0f;
    if(j<n){
        unsigned long long qi=qi0+j, k=pof+j;
        float gr=Gc[(size_t)blockIdx.x*group + j]*(gs?gs[0]:1.0f);
        float mv=(float)m8[qi]*ms_old;
        float sv=qv_dec((float)v8[qi],vs_old), vv=sv*sv;
        mn=b1*mv+(1.0f-b1)*gr;
        vv=b2*vv+(1.0f-b2)*gr*gr;
        rv=sqrtf(vv);
        float veps=fmaxf(qv_dec(1.0f,vs_old),1e-8f);
        float u=(mn*inv_bc1)/(rv*inv_sqrt_bc2+veps);
        u=fmaxf(-8.0f,fminf(8.0f,u));
        rv=qv_enc(rv);
        int dd=qd_get(d,qi,1);
        int idl=qsri(-(L*u+dec*(float)dd), qi, t);
        int rr=(int)r[qi]+idl;
        int c=(rr>64)?(((rr-65)>>7)+1):(rr<-64)?-((((-rr)-65)>>7)+1):0;
        rr-=c*128; dd+=c; dd=qclmp(dd,-dmax,dmax);
        shd[j]=(signed char)dd; r[qi]=(signed char)rr;
        P[k]=sg*(float)dd;
    }
    __syncthreads();
    qd_write_group(d,qi0,shd,n,j,1);
    __shared__ float sha[4],shv[4];
    float am=(j<n)?fabsf(mn):0.0f, vm=(j<n)?rv:0.0f;
    for(int o=16;o>0;o>>=1){ am=fmaxf(am,__shfl_down_sync(0xffffffff,am,o));
                             vm=fmaxf(vm,__shfl_down_sync(0xffffffff,vm,o)); }
    if((j&31)==0){ sha[j>>5]=am; shv[j>>5]=vm; }
    __syncthreads();
    if(j==0){ int nw=(blockDim.x+31)/32; float a=sha[0],v=shv[0];
        for(int i=1;i<nw;i++){ a=fmaxf(a,sha[i]); v=fmaxf(v,shv[i]); }
        m8s[gi]=a>0.0f?a/127.0f:1e-12f; v8s[gi]=v>0.0f?v/255.0f:1e-12f; }
    __syncthreads();
    if(j<n){ unsigned long long qi=qi0+j;
        m8[qi]=(signed char)qclmp(qsri(mn/m8s[gi],qi,t+0x5EEDULL),-127,127);
        int vq=qclmp(qsri(rv/v8s[gi],qi,t+0xBEEFULL),0,255);
        if(vq==0&&rv>0.0f)vq=1; v8[qi]=(unsigned char)vq;
    }
}
#else
#define CPUK_PART 7
#include "cpu_kern.h"
#endif
/* Quantize the random-init table into a single 4-bit span (reusing the qlm
 * device kernels), freeze the rel-LR reference scale, render back into g_ng.
 * g_ng thereafter is the rendered fp32 view the gather reads; the persistent
 * state is digits+residual+int8 moments. Wng stays fp32 (only 6K params). */
static void ngram_q4_setup(void){
    size_t nt=(size_t)g_ng_R*g_ng_De;
    g_ngq_grp=128; g_ngq_dmax=7; qlm_v4_init();
    g_ngq_ngrp=(int)((nt+g_ngq_grp-1)/g_ngq_grp);
    g_ngq_spH.pof=0; g_ngq_spH.gof=0; g_ngq_spH.qi0=0; g_ngq_spH.n=nt;
    g_ngq_spH.gi0=0; g_ngq_spH.ngrp=g_ngq_ngrp; g_ngq_spH.sref=1.0f;
    g_ngq_d =(signed char*)dmalloc((nt+1)/2);
    g_ngq_r =(signed char*)dmalloc(nt);
    g_ngq_m8=(signed char*)dmalloc(nt);
    g_ngq_v8=(unsigned char*)dmalloc(nt);
    ck(cudaMemset(g_ngq_m8,0,nt),"ngq m0"); ck(cudaMemset(g_ngq_v8,0,nt),"ngq v0");
    g_ngq_s =(float*)dmalloc((size_t)g_ngq_ngrp*4);
    g_ngq_ms=(float*)dmalloc((size_t)g_ngq_ngrp*4);
    g_ngq_vs=(float*)dmalloc((size_t)g_ngq_ngrp*4);
    ck(cudaMemset(g_ngq_ms,0,(size_t)g_ngq_ngrp*4),"ngq ms0");
    ck(cudaMemset(g_ngq_vs,0,(size_t)g_ngq_ngrp*4),"ngq vs0");
    g_ngq_spD=(QSpanD*)dmalloc(sizeof(QSpanD));
    ck(cudaMemcpy(g_ngq_spD,&g_ngq_spH,sizeof(QSpanD),cudaMemcpyHostToDevice),"ngq span h2d");
    cudaDeviceSynchronize(); g_tprev=wall();
    KL(k_qlm_encode,(unsigned)g_ngq_ngrp,g_ngq_grp,g_ngq_grp)(
        g_ng,g_ngq_d,g_ngq_r,g_ngq_s,g_ngq_spD,1,g_ngq_grp,g_ngq_dmax,1);
    kchk("ngq encode",-1);
    { float*sh=(float*)malloc((size_t)g_ngq_ngrp*4); double acc=0;
      ck(cudaMemcpy(sh,g_ngq_s,(size_t)g_ngq_ngrp*4,cudaMemcpyDeviceToHost),"ngq s d2h");
      for(int g=0;g<g_ngq_ngrp;g++) acc+=sh[g];
      g_ngq_spH.sref=(float)(acc/(double)g_ngq_ngrp); free(sh);
      ck(cudaMemcpy(g_ngq_spD,&g_ngq_spH,sizeof(QSpanD),cudaMemcpyHostToDevice),"ngq span h2d 2"); }
    KL(k_qlm_render,(unsigned)g_ngq_ngrp,g_ngq_grp)(
        g_ng,g_ngq_d,g_ngq_s,g_ngq_spD,1,g_ngq_grp,1,0);
    kchk("ngq render",-1);
}
static float *g_ngWk=0,*g_ngWkG=0,*g_ngk=0,*g_nghn=0,*g_ngdv=0,*g_ngdk=0,*g_ngg=0,*g_ngrh=0,*g_ngrk=0;
static bf16 *g_ngWkM=0,*g_ngWkV=0;
static void x1_setup(const Cfg*c); static void x1_update(float lr,int t);
static void x1_save(const char*optpath,int it); static void x1_load(const char*optpath);
static void t1_setup(const Cfg*c){
    const char*e;
    x1_setup(c);
    if((e=getenv("CUDALM_ATTN_GATE"))&&atoi(e)&&!g_Wg){ size_t n=(size_t)c->L*c->D*c->D; g_attg=1;
        g_Wg=(float*)dmalloc(n*4); g_WgG=(float*)dmalloc(n*4); g_WgM=(bf16*)dmalloc(n*2); g_WgV=(bf16*)dmalloc(n*2);
        ck(cudaMemset(g_Wg,0,n*4),"Wg0"); ck(cudaMemset(g_WgM,0,n*2),"WgM0"); ck(cudaMemset(g_WgV,0,n*2),"WgV0");
        printf("[t1] gated attention on (Wg %d x %dx%d, zero-init)\n",c->L,c->D,c->D); }
    if((e=getenv("NGRAM_LAYER"))){ g_ng_layer=atoi(e); if(g_ng_layer<0)g_ng_layer=0; if(g_ng_layer>=c->L)g_ng_layer=c->L-1; }
    if((e=getenv("NGRAM_GATE"))) g_ng_gate=atoi(e)!=0;
    if((e=getenv("NGRAM_TRI")))  g_ng_tri=atoi(e)!=0;
}
static int g_eg_on=0, g_eg_fh[24];   /* sparse engram (NGR2), defined after ngram_infer */
static int eg_wanted(void); static void engram_setup(const Cfg*c); static int engram_load(const char*optpath);
static void ngram_setup(const Cfg*c){
    t1_setup(c);
    const char*e=getenv("CUDALM_NGRAM");
    if(eg_wanted()||(g_eg_fh[0]&&!(e&&!atoi(e)))){ engram_setup(c); return; }
    g_ngram_on=e?atoi(e)!=0:1; if(!g_ngram_on)return;   /* default on (gen2) */
    g_ng_R=655360; { const char*r=getenv("NGRAM_ROWS"); if(r) g_ng_R=atoi(r); }
    g_ng_De=64;   { const char*d=getenv("NGRAM_DIM");  if(d) g_ng_De=atoi(d); }
    { const char*o=getenv("NGRAM_ORDER"); if(o) g_ng_order=atoi(o); }
    { const char*q=getenv("NGRAM_Q4"); g_ng_q4=q?atoi(q)!=0:1; }
    if(g_ng_fh[0]){ g_ng_R=g_ng_fh[0]; g_ng_De=g_ng_fh[1]; g_ng_order=g_ng_fh[2]; g_ng_q4=g_ng_fh[3]; }  /* inference: file decides */
    { const char*f=getenv("NGRAM_Q4_FOLD");        /* group is fixed at 128 below */
      if(f?atoi(f)!=0:g_ng_q4){
        if(g_ng_q4 && (128 % g_ng_De)==0) g_ng_q4fold=1;
        else printf("[ngram] Q4_FOLD ignored: needs NGRAM_Q4=1 and 128%%De==0 (De=%d)\n",g_ng_De); } }
    size_t nt=(size_t)g_ng_R*g_ng_De, nw=(size_t)g_ng_De*c->D;
    g_ng=(float*)dmalloc(nt*4);
    if(!g_ng_q4fold) g_ngG=(float*)dmalloc(nt*4);   /* fold: compact grad instead */
    if(!g_ng_q4){ g_ngM=(bf16*)dmalloc(nt*2); g_ngV=(bf16*)dmalloc(nt*2);
        ck(cudaMemset(g_ngM,0,nt*2),"ng M0"); ck(cudaMemset(g_ngV,0,nt*2),"ng V0"); }
    g_ngW=(float*)dmalloc(nw*4); g_ngWG=(float*)dmalloc(nw*4);
    g_ngWM=(bf16*)dmalloc(nw*2); g_ngWV=(bf16*)dmalloc(nw*2);
    ck(cudaMemset(g_ngW,0,nw*4),"ng W0");            /* zero -> step-0 no-op */
    ck(cudaMemset(g_ngWM,0,nw*2),"ng WM0"); ck(cudaMemset(g_ngWV,0,nw*2),"ng WV0");
    float*h=(float*)malloc(nt*4); unsigned s=0x9e3779b9u;
    for(size_t i=0;i<nt;i++){ s^=s<<13; s^=s>>17; s^=s<<5;
        h[i]=0.02f*(((s&0xffffff)/(float)0x1000000)*2.0f-1.0f); }
    ck(cudaMemcpy(g_ng,h,nt*4,cudaMemcpyHostToDevice),"ng tbl init"); free(h);
    if(g_ng_q4) ngram_q4_setup();
    if(g_ng_gate){ g_ngWk=(float*)dmalloc(nw*4); g_ngWkG=(float*)dmalloc(nw*4);
        g_ngWkM=(bf16*)dmalloc(nw*2); g_ngWkV=(bf16*)dmalloc(nw*2);
        ck(cudaMemset(g_ngWkM,0,nw*2),"ng WkM0"); ck(cudaMemset(g_ngWkV,0,nw*2),"ng WkV0");
        float*w=(float*)malloc(nw*4), sc=1.0f/sqrtf((float)g_ng_De);
        for(size_t i=0;i<nw;i++){ s^=s<<13; s^=s>>17; s^=s<<5; w[i]=sc*(((s&0xffffff)/(float)0x1000000)*2.0f-1.0f); }
        ck(cudaMemcpy(g_ngWk,w,nw*4,cudaMemcpyHostToDevice),"ng Wk init"); free(w); }
    if(g_ng_layer||g_ng_gate||g_ng_tri) printf("[ngram] layer=%d gate=%d tri=%d\n",g_ng_layer,g_ng_gate,g_ng_tri);
    printf("[ngram] on: R=%d De=%d order=%d  table=%.1f MiB (%s)%s\n",
           g_ng_R,g_ng_De,g_ng_order,
           g_ng_q4?(double)((nt+1)/2+3*nt)/1048576.0:(double)nt*4/1048576.0,
           g_ng_q4?"q4 digit+resid+int8 moments":"fp32 scaffold",
           g_ng_q4fold?"  [Q4_FOLD: no full grad, lazy touched-group step]":"");
}
static void ngram_ensure(int N){
    if(N<=g_ng_ncap) return;
    if(g_ngbuf){ cudaFree(g_ngbuf); cudaFree(g_ngproj); cudaFree(g_ngdbuf); cudaFree(g_ngkey); }
    g_ngbuf =(float*)dmalloc((size_t)N*g_ng_De*4);
    g_ngproj=(float*)dmalloc((size_t)N*gC.D*4);
    g_ngdbuf=(float*)dmalloc((size_t)N*g_ng_De*4);
    int NK=N*(g_ng_tri?2:1);
    g_ngkey =(int*)  dmalloc((size_t)NK*4);
    if(g_ng_gate){ size_t ND=(size_t)N*gC.D;
        if(g_ngk){ cudaFree(g_ngk); cudaFree(g_nghn); cudaFree(g_ngdv); cudaFree(g_ngdk); cudaFree(g_ngg); cudaFree(g_ngrh); cudaFree(g_ngrk); }
        g_ngk=(float*)dmalloc(ND*4); g_nghn=(float*)dmalloc(ND*4); g_ngdv=(float*)dmalloc(ND*4); g_ngdk=(float*)dmalloc(ND*4);
        g_ngg=(float*)dmalloc((size_t)N*4); g_ngrh=(float*)dmalloc((size_t)N*4); g_ngrk=(float*)dmalloc((size_t)N*4); }
    if(g_ng_q4fold){
        if(g_ngGc){ cudaFree(g_ngGc); cudaFree(g_ng_cidxD); cudaFree(g_ng_tgiD);
                    cudaFreeHost(g_ng_hk); free(g_ng_hgi); free(g_ng_hord); cudaFreeHost(g_ng_hcidx); cudaFreeHost(g_ng_htgi); }
        g_ng_gcap=NK;                                /* <= NK distinct touched groups */
        g_ngGc=(float*)dmalloc((size_t)NK*g_ngq_grp*4);
        g_ng_cidxD=(int*)dmalloc((size_t)NK*4); g_ng_tgiD=(int*)dmalloc((size_t)NK*4);
        ck(cudaHostAlloc((void**)&g_ng_hk,(size_t)NK*4,0),"ng pin");
        ck(cudaHostAlloc((void**)&g_ng_hcidx,(size_t)NK*4,0),"ng pin");
        ck(cudaHostAlloc((void**)&g_ng_htgi,(size_t)NK*4,0),"ng pin");
        g_ng_hgi=(int*)malloc((size_t)NK*4); g_ng_hord=(int*)malloc((size_t)NK*4);
    }
    g_ng_ncap=N;
}
/* qsort key positions by their touched-group index (negatives, i.e. masked
 * positions, sort to the front and are dropped). */
static const int *g_ng_cmp_gi=0;
static int ng_gi_cmp(const void*a,const void*b){
    int ga=g_ng_cmp_gi[*(const int*)a], gb=g_ng_cmp_gi[*(const int*)b];
    return (ga>gb)-(ga<gb);
}
/* Device-side touched map: a CAS claims each touched group and an atomic counter
 * numbers it, then the slots are released. Slot order is arbitrary (every group
 * steps independently). CUDALM_NG_HOSTMAP=1 = the old host qsort (~15 ms/step). */
#ifndef CUDALM_CPU
__global__ void k_ng_claim(const int*key,int*slot,int*tgi,int*cnt,int N,int De,int grp){
    int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N) return; int r=key[n]; if(r<0) return;
    int gi=(int)(((long long)r*De)/grp);
    if(atomicCAS(&slot[gi],-1,-2)==-1){ int c=atomicAdd(cnt,1); tgi[c]=gi; slot[gi]=c; } }
__global__ void k_ng_cidx(const int*key,const int*slot,int*cidx,int N,int De,int grp){
    int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N) return; int r=key[n];
    cidx[n]= r<0 ? -1 : slot[(int)(((long long)r*De)/grp)]; }
__global__ void k_ng_unclaim(int*slot,const int*tgi,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) slot[tgi[i]]=-1; }
#else
#define CPUK_PART 8
#include "cpu_kern.h"
#endif
/* Build the compact touched-group map for this step: cidx[n] = compact slot of
 * position n's group (-1 if masked), tgi[c] = real group index of slot c. */
static void ng_build_touched(int N){
    int grp=g_ngq_grp, De=g_ng_De;
    static int *slot=0,*cnt=0,cap=0,host=-1;
    if(host<0){ const char*e=getenv("CUDALM_NG_HOSTMAP"); host=e&&atoi(e); }
    if(!host){
        if(cap<g_ngq_ngrp){ if(slot) cudaFree(slot); cap=g_ngq_ngrp;
            slot=(int*)dmalloc((size_t)cap*4+4); cnt=slot+cap; ck(cudaMemset(slot,0xFF,(size_t)cap*4),"ng slot"); }
        ck(cudaMemset(cnt,0,4),"ng cnt");
        KL(k_ng_claim,nblk(N,TPB),TPB)(g_ngkey,slot,g_ng_tgiD,cnt,N,De,grp);
        KL(k_ng_cidx,nblk(N,TPB),TPB)(g_ngkey,slot,g_ng_cidxD,N,De,grp);
        ck(cudaMemcpy(&g_ng_Gt,cnt,4,cudaMemcpyDeviceToHost),"ng Gt d2h");
        if(g_ng_Gt) KL(k_ng_unclaim,nblk(g_ng_Gt,TPB),TPB)(slot,g_ng_tgiD,g_ng_Gt);
        return; }
    ck(cudaMemcpy(g_ng_hk,g_ngkey,(size_t)N*4,cudaMemcpyDeviceToHost),"ng key d2h");
    for(int n=0;n<N;n++){ int r=g_ng_hk[n];
        g_ng_hgi[n] = (r<0) ? -1 : (int)(((long long)r*De)/grp);
        g_ng_hord[n]=n; }
    g_ng_cmp_gi=g_ng_hgi; qsort(g_ng_hord,N,sizeof(int),ng_gi_cmp);
    int Gt=0, prev=-2;
    for(int i=0;i<N;i++){ int n=g_ng_hord[i], gi=g_ng_hgi[n];
        if(gi<0){ g_ng_hcidx[n]=-1; continue; }
        if(gi!=prev){ g_ng_htgi[Gt]=gi; prev=gi; Gt++; }
        g_ng_hcidx[n]=Gt-1; }
    g_ng_Gt=Gt;
    ck(cudaMemcpyAsync(g_ng_cidxD,g_ng_hcidx,(size_t)N*4,cudaMemcpyHostToDevice,0),"ng cidx h2d");
    if(Gt) ck(cudaMemcpyAsync(g_ng_tgiD,g_ng_htgi,(size_t)Gt*4,cudaMemcpyHostToDevice,0),"ng tgi h2d");
}
/* dtw/dpos (decoder): keys from each slot's last tokens instead of the token rows */
__global__ void k_dec_ngkey(const int*tokw,int*key,const int*pos,int B,int R,int order,int base);
static void ngram_forward(GA*a,const int*tokG,int N,const int*dtw=NULL,const int*dpos=NULL){
    int De=g_ng_De, D=gC.D, nk=g_ng_tri?2:1, R=g_ng_R/nk;
    float*h=a->a_in[g_ng_layer];
    ngram_ensure(N);
    if(dtw){ KL(k_dec_ngkey,nblk(N,TPB),TPB)(dtw,g_ngkey,dpos,N,R,g_ng_order,0);
        if(g_ng_tri) KL(k_dec_ngkey,nblk(N,TPB),TPB)(dtw,g_ngkey+N,dpos,N,R,g_ng_order+1,R); }
    else { KL(k_ngram_key,nblk(N,TPB),TPB)(tokG,g_ngkey,N,gC.T,R,g_ng_order,0);
        if(g_ng_tri) KL(k_ngram_key,nblk(N,TPB),TPB)(tokG,g_ngkey+N,N,gC.T,R,g_ng_order+1,R); }   /* (order+1)-gram half */
    kchk("ng key",-1);
    KL(k_ngram_gather,nblk((size_t)N*De,TPB),TPB)(g_ng,g_ngkey,g_ngbuf,N,De,nk); kchk("ng gather",-1);
    g_ab(g_ngbuf,g_ngW,g_ngproj,N,De,D);            /* ngproj[N,D]=buf@Wng */
    if(g_ng_gate){ g_ab(g_ngbuf,g_ngWk,g_ngk,N,De,D);
        KL(k_nggate_f,N,128)(h,g_ngproj,g_ngk,g_nghn,g_ngg,g_ngrh,g_ngrk,D); kchk("ng gate",-1); }
    else e_add(h,g_ngproj,(size_t)N*D);             /* add into the residual stream */
}
static void ngram_backward(GA*a,const int*tokG,int N){
    int De=g_ng_De, D=gC.D, NK=N*(g_ng_tri?2:1);
    const float*dy=a->dout;
    ck(cudaMemset(g_ngWG,0,(size_t)De*D*4),"ng WG0");
    if(g_ng_gate){ KL(k_nggate_b,N,128)(a->dout,g_ngproj,g_nghn,g_ngk,g_ngg,g_ngrh,g_ngrk,g_ngdv,g_ngdk,D);
        kchk("ng gate bwd",-1); dy=g_ngdv; }
    g_atb(g_ngbuf,dy,g_ngWG,N,De,D);                /* dWng += buf^T @ dy */
    g_abt(dy,g_ngW,g_ngdbuf,N,D,De);                /* dbuf[N,De]=dy@Wng^T */
    if(g_ng_gate){ ck(cudaMemset(g_ngWkG,0,(size_t)De*D*4),"ng WkG0");
        g_atb(g_ngbuf,g_ngdk,g_ngWkG,N,De,D); g_abt_b(g_ngdk,g_ngWk,g_ngdbuf,N,D,De,1.0f); }
    if(g_ng_q4fold){
        ng_build_touched(NK);
        if(g_ng_Gt) ck(cudaMemset(g_ngGc,0,(size_t)g_ng_Gt*g_ngq_grp*4),"ng Gc0");
        KL(k_ng_scatter_compact,nblk((size_t)NK*De,TPB),TPB)(
            g_ngGc,g_ngkey,g_ng_cidxD,g_ngdbuf,NK,De,g_ngq_grp,N); kchk("ng scatter compact",-1);
    } else {
        ck(cudaMemset(g_ngG,0,(size_t)g_ng_R*De*4),"ng G0");
        KL(k_ngram_scatter,nblk((size_t)NK*De,TPB),TPB)(g_ngG,g_ngkey,g_ngdbuf,NK,De,N); kchk("ng scatter",-1);
    }
}
static void ng_tbl_step(float lr,int t){
    float b1=0.9f,b2=0.95f,wd=0.0f,bc1=1.0f-powf(b1,(float)t),bc2=1.0f-powf(b2,(float)t);
    size_t nt=(size_t)g_ng_R*g_ng_De;
    if(g_ng_q4fold){
        float inv_bc1=1.0f/bc1, inv_sqrt_bc2=1.0f/sqrtf(bc2);
        if(g_ng_Gt) KL(k_ng_q4_step,(unsigned)g_ng_Gt,g_ngq_grp,g_ngq_grp)(
            g_ng,g_ngGc,g_ngq_d,g_ngq_r,g_ngq_s,g_ngq_m8,g_ngq_ms,g_ngq_v8,g_ngq_vs,
            g_ng_tgiD,g_ngq_grp,g_ngq_dmax,lr,b1,b2,wd,g_ngq_spH.sref,inv_bc1,inv_sqrt_bc2,
            (unsigned long long)t,nt,g_side_gs);
    } else if(g_ng_q4){
        float inv_bc1=1.0f/bc1, inv_sqrt_bc2=1.0f/sqrtf(bc2);
        qlm_step_launch(g_ng,g_ngG,g_ngq_d,g_ngq_r,g_ngq_s,g_ngq_m8,g_ngq_ms,g_ngq_v8,g_ngq_vs,
            g_ngq_spD,1,g_ngq_grp,g_ngq_dmax,lr,b1,b2,wd,1.0f,g_side_gs,inv_bc1,inv_sqrt_bc2,
            0,(unsigned long long)t,1,(long long*)0,1,0,(int)g_ngq_ngrp);
    } else {
        KL(k_adamw,nblk(nt,TPB),TPB)(g_ng,g_ngG,g_ngM,g_ngV,nt,lr,b1,b2,wd,bc1,bc2,
                                      (const float*)0,(size_t)0,(size_t)0,0,MU_OFF,g_side_gs);
    }
}
static void ngram_update(float lr,int t){
    float b1=0.9f,b2=0.95f,wd=0.0f,bc1=1.0f-powf(b1,(float)t),bc2=1.0f-powf(b2,(float)t);
    size_t nw=(size_t)g_ng_De*gC.D;
    KL(k_adamw,nblk(nw,TPB),TPB)(g_ngW,g_ngWG,g_ngWM,g_ngWV,nw,lr,b1,b2,wd,bc1,bc2,
                                  (const float*)0,(size_t)0,(size_t)0,0,MU_OFF,g_side_gs);
    ng_tbl_step(lr,t);
    kchk("ng adam",-1);
}

/* Durable sidecar <optpath>.ng, written beside the main .opt (mirrors qlm .q).
 * Holds Wng + its bf16 moments, and the table state: q4 = digits+resid+int8
 * moments+scales+sref (g_ng is a pure render view, re-rendered on load); fp32 =
 * table + bf16 moments. Header ints validate config against env on resume. */
static void ngram_save(const char*optpath,int it){
    char path[1200]; snprintf(path,sizeof path,"%s.ng",optpath);
    char tmp[1300]; FILE*f=save_open(path,tmp,sizeof tmp);
    size_t nw=(size_t)g_ng_De*gC.D, nt=(size_t)g_ng_R*g_ng_De;
    int hdr[5]={g_ng_R,g_ng_De,g_ng_order,g_ng_q4,it};
    fwrite("NGR1",1,4,f); fwrite(hdr,4,5,f);
    { float*wh=(float*)malloc(nw*4);
      ck(cudaMemcpy(wh,g_ngW,nw*4,cudaMemcpyDeviceToHost),"ng save W");
      fwrite(wh,4,nw,f); free(wh); }
    { bf16*mh=(bf16*)malloc(nw*2),*vh=(bf16*)malloc(nw*2);
      ck(cudaMemcpy(mh,g_ngWM,nw*2,cudaMemcpyDeviceToHost),"ng save WM");
      ck(cudaMemcpy(vh,g_ngWV,nw*2,cudaMemcpyDeviceToHost),"ng save WV");
      fwrite(mh,2,nw,f); fwrite(vh,2,nw,f); free(mh); free(vh); }
    if(g_ng_q4){
        int ng=g_ngq_ngrp; size_t nd=(nt+1)/2;
        float sref=g_ngq_spH.sref; fwrite(&sref,4,1,f);
        { signed char*b=(signed char*)malloc(nd);
          ck(cudaMemcpy(b,g_ngq_d,nd,cudaMemcpyDeviceToHost),"ng save d"); fwrite(b,1,nd,f); free(b); }
        { signed char*b=(signed char*)malloc(nt);
          ck(cudaMemcpy(b,g_ngq_r,nt,cudaMemcpyDeviceToHost),"ng save r"); fwrite(b,1,nt,f); free(b); }
        { signed char*b=(signed char*)malloc(nt);
          ck(cudaMemcpy(b,g_ngq_m8,nt,cudaMemcpyDeviceToHost),"ng save m8"); fwrite(b,1,nt,f); free(b); }
        { unsigned char*b=(unsigned char*)malloc(nt);
          ck(cudaMemcpy(b,g_ngq_v8,nt,cudaMemcpyDeviceToHost),"ng save v8"); fwrite(b,1,nt,f); free(b); }
        { float*sb=(float*)malloc((size_t)ng*4);
          ck(cudaMemcpy(sb,g_ngq_s ,(size_t)ng*4,cudaMemcpyDeviceToHost),"ng save s");  fwrite(sb,4,ng,f);
          ck(cudaMemcpy(sb,g_ngq_ms,(size_t)ng*4,cudaMemcpyDeviceToHost),"ng save ms"); fwrite(sb,4,ng,f);
          ck(cudaMemcpy(sb,g_ngq_vs,(size_t)ng*4,cudaMemcpyDeviceToHost),"ng save vs"); fwrite(sb,4,ng,f); free(sb); }
    } else {
        { float*b=(float*)malloc(nt*4);
          ck(cudaMemcpy(b,g_ng,nt*4,cudaMemcpyDeviceToHost),"ng save tbl"); fwrite(b,4,nt,f); free(b); }
        { bf16*mh=(bf16*)malloc(nt*2),*vh=(bf16*)malloc(nt*2);
          ck(cudaMemcpy(mh,g_ngM,nt*2,cudaMemcpyDeviceToHost),"ng save M");
          ck(cudaMemcpy(vh,g_ngV,nt*2,cudaMemcpyDeviceToHost),"ng save V");
          fwrite(mh,2,nt,f); fwrite(vh,2,nt,f); free(mh); free(vh); }
    }
    if(!save_commit(f,tmp,path)) fprintf(stderr,"ngram save FAILED for %s (previous kept)\n",path);
}
static int ngram_load(const char*optpath){
    char path[1200]; snprintf(path,sizeof path,"%s.ng",optpath);
    FILE*f=fopen(path,"rb"); if(!f) return 0;
    char m[4]; if(fread(m,1,4,f)!=4||memcmp(m,"NGR1",4)){
        if(!memcmp(m,"NGR2",4)){ fprintf(stderr,"[ngram] resume mismatch: %s is an engram (NGR2) table, env has plain ngram\n",path); exit(1); }
        fclose(f);return 0;}
    int hdr[5]; if(fread(hdr,4,5,f)!=5){fclose(f);return 0;}
    if(hdr[0]!=g_ng_R||hdr[1]!=g_ng_De||hdr[2]!=g_ng_order||hdr[3]!=g_ng_q4){
        fprintf(stderr,"[ngram] resume mismatch: file R=%d De=%d order=%d q4=%d vs env R=%d De=%d order=%d q4=%d\n",
                hdr[0],hdr[1],hdr[2],hdr[3],g_ng_R,g_ng_De,g_ng_order,g_ng_q4);
        fclose(f); exit(1);
    }
    int it=hdr[4];
    size_t nw=(size_t)g_ng_De*gC.D, nt=(size_t)g_ng_R*g_ng_De;
    { float*wh=(float*)malloc(nw*4);
      if(fread(wh,4,nw,f)!=nw){free(wh);fclose(f);return 0;}
      ck(cudaMemcpy(g_ngW,wh,nw*4,cudaMemcpyHostToDevice),"ng load W"); free(wh); }
    { bf16*mh=(bf16*)malloc(nw*2),*vh=(bf16*)malloc(nw*2);
      if(fread(mh,2,nw,f)!=nw||fread(vh,2,nw,f)!=nw){free(mh);free(vh);fclose(f);return 0;}
      ck(cudaMemcpy(g_ngWM,mh,nw*2,cudaMemcpyHostToDevice),"ng load WM");
      ck(cudaMemcpy(g_ngWV,vh,nw*2,cudaMemcpyHostToDevice),"ng load WV"); free(mh); free(vh); }
    if(g_ng_q4){
        int ng=g_ngq_ngrp; size_t nd=(nt+1)/2;
        float sref; if(fread(&sref,4,1,f)!=1){fclose(f);return 0;}
        g_ngq_spH.sref=sref;
        ck(cudaMemcpy(g_ngq_spD,&g_ngq_spH,sizeof(QSpanD),cudaMemcpyHostToDevice),"ng load span");
        { signed char*b=(signed char*)malloc(nd);
          if(fread(b,1,nd,f)!=nd){free(b);fclose(f);return 0;}
          ck(cudaMemcpy(g_ngq_d,b,nd,cudaMemcpyHostToDevice),"ng load d"); free(b); }
        { signed char*b=(signed char*)malloc(nt);
          if(fread(b,1,nt,f)!=nt){free(b);fclose(f);return 0;}
          ck(cudaMemcpy(g_ngq_r,b,nt,cudaMemcpyHostToDevice),"ng load r"); free(b); }
        { signed char*b=(signed char*)malloc(nt);
          if(fread(b,1,nt,f)!=nt){free(b);fclose(f);return 0;}
          ck(cudaMemcpy(g_ngq_m8,b,nt,cudaMemcpyHostToDevice),"ng load m8"); free(b); }
        { unsigned char*b=(unsigned char*)malloc(nt);
          if(fread(b,1,nt,f)!=nt){free(b);fclose(f);return 0;}
          ck(cudaMemcpy(g_ngq_v8,b,nt,cudaMemcpyHostToDevice),"ng load v8"); free(b); }
        { float*sb=(float*)malloc((size_t)ng*4);
          if(fread(sb,4,ng,f)!=(size_t)ng){free(sb);fclose(f);return 0;}
          ck(cudaMemcpy(g_ngq_s ,sb,(size_t)ng*4,cudaMemcpyHostToDevice),"ng load s");
          if(fread(sb,4,ng,f)!=(size_t)ng){free(sb);fclose(f);return 0;}
          ck(cudaMemcpy(g_ngq_ms,sb,(size_t)ng*4,cudaMemcpyHostToDevice),"ng load ms");
          if(fread(sb,4,ng,f)!=(size_t)ng){free(sb);fclose(f);return 0;}
          ck(cudaMemcpy(g_ngq_vs,sb,(size_t)ng*4,cudaMemcpyHostToDevice),"ng load vs"); free(sb); }
        KL(k_qlm_render,(unsigned)g_ngq_ngrp,g_ngq_grp)(
            g_ng,g_ngq_d,g_ngq_s,g_ngq_spD,1,g_ngq_grp,1,0);
        kchk("ng load render",-1);
    } else {
        { float*b=(float*)malloc(nt*4);
          if(fread(b,4,nt,f)!=nt){free(b);fclose(f);return 0;}
          ck(cudaMemcpy(g_ng,b,nt*4,cudaMemcpyHostToDevice),"ng load tbl"); free(b); }
        { bf16*mh=(bf16*)malloc(nt*2),*vh=(bf16*)malloc(nt*2);
          if(fread(mh,2,nt,f)!=nt||fread(vh,2,nt,f)!=nt){free(mh);free(vh);fclose(f);return 0;}
          ck(cudaMemcpy(g_ngM,mh,nt*2,cudaMemcpyHostToDevice),"ng load M");
          ck(cudaMemcpy(g_ngV,vh,nt*2,cudaMemcpyHostToDevice),"ng load V"); free(mh); free(vh); }
    }
    fclose(f);
    printf("[ngram] resumed table+Wng from %s at iter %d\n",path,it);
    return it;
}
/* CUDALM_EMB_E (20u): factorized embedding. E [V,Ee] and Up [Ee,D] are the trained parameters;
 * the tied [V,D] slot in P is rebuilt as E*Up after every step, so every reader of P+emb (head,
 * serve, eval, the CPU build) is unchanged. Backward turns dW into dE = dW Up^T, dUp = E^T dW and
 * zeroes dW, which leaves the main AdamW a no-op there. State: sidecar <optpath>.fe. */
static int g_feE=0; static float *g_feP=0,*g_feG=0,*g_feW=0; static bf16 *g_feM=0,*g_feV=0;
static int g_fe_file=-1;   /* info on a trained model: Ee from its .opt.fe (0 = none), not the environment */
static int fe_dim(int D){ const char*e=getenv("CUDALM_EMB_E"); int v=g_fe_file>=0?g_fe_file:e&&*e?atoi(e):0; return v>0&&v<D?v:0; }
static void fe_mat(void){   /* W = E Up */
    const float one=1.0f,zero=0.0f; int V=gC.V,D=gC.D,Ee=g_feE;
    ckb(cublasSgemm(CB,CUBLAS_OP_N,CUBLAS_OP_N,D,V,Ee,&one,g_feP+(size_t)V*Ee,D,g_feP,Ee,&zero,g_feW,D),"fe mat"); }
static void fe_grad(float*dW){
    const float one=1.0f,zero=0.0f; int V=gC.V,D=gC.D,Ee=g_feE; size_t ne=(size_t)V*Ee;
    ckb(cublasSgemm(CB,CUBLAS_OP_T,CUBLAS_OP_N,Ee,V,D,&one,g_feP+ne,D,dW,D,&zero,g_feG,Ee),"fe dE");
    ckb(cublasSgemm(CB,CUBLAS_OP_N,CUBLAS_OP_T,D,Ee,V,&one,dW,D,g_feP,Ee,&zero,g_feG+ne,D),"fe dUp");
    ck(cudaMemset(dW,0,(size_t)V*D*4),"fe zero dW"); }
static void fe_update(float lr,int t){
    if(!g_feE) return;
    static float b2=-1,wd; const char*e;
    if(b2<0){ b2=(e=getenv("CUDALM_B2"))?(float)atof(e):0.99f; wd=(e=getenv("CUDALM_WD"))?(float)atof(e):0.1f; }
    float b1=0.9f,bc1=1.0f-powf(b1,(float)t),bc2=1.0f-powf(b2,(float)t);
    size_t ne=(size_t)gC.V*g_feE, nu=(size_t)g_feE*gC.D;
    KL(k_adamw,nblk(ne,TPB),TPB)(g_feP,g_feG,g_feM,g_feV,ne,lr,b1,b2,wd,bc1,bc2,(const float*)0,(size_t)0,(size_t)0,0,MU_OFF,g_side_gs);
    KL(k_adamw,nblk(nu,TPB),TPB)(g_feP+ne,g_feG+ne,g_feM+ne,g_feV+ne,nu,lr,b1,b2,0.0f,bc1,bc2,(const float*)0,(size_t)0,(size_t)0,0,MU_OFF,g_side_gs);
    fe_mat(); kchk("fe adam",-1); }
static int t1_rw(FILE*f,void*dev,size_t by,int wr);
/* train only: loads <optpath>.fe (the file decides Ee) or starts one from CUDALM_EMB_E at iter 0 */
static int fe_init(const char*optpath,int start,float*W){
    char p[1200]; snprintf(p,sizeof p,"%s.fe",optpath);
    FILE*f=fopen(p,"rb"); char m[4]; int h[4]={0};
    if(f&&(fread(m,1,4,f)!=4||memcmp(m,"FES1",4)||fread(h,4,4,f)!=4||h[1]!=gC.V||h[2]!=gC.D)){
        fprintf(stderr,"[emb] bad or mismatched %s\n",p); exit(1); }
    int Ee=f?h[0]:fe_dim(gC.D); if(!Ee) return 0;
    if(!f&&start>0){ fprintf(stderr,"[emb] CUDALM_EMB_E=%d but %s is missing and the model is at iter %d: "
                                    "factorize a model from new, or unset CUDALM_EMB_E\n",Ee,p,start); exit(1); }
    size_t ne=(size_t)gC.V*Ee, n=ne+(size_t)Ee*gC.D;
    g_feE=Ee; g_feW=W; g_feP=(float*)dmalloc(n*4); g_feG=(float*)dmalloc(n*4);
    g_feM=(bf16*)dmalloc(n*2); g_feV=(bf16*)dmalloc(n*2);
    if(f){ int ok=t1_rw(f,g_feP,n*4,0)&&t1_rw(f,g_feM,n*2,0)&&t1_rw(f,g_feV,n*2,0); fclose(f);
        if(!ok){ fprintf(stderr,"[emb] truncated %s\n",p); exit(1); } }
    else { float*hp=(float*)malloc(n*4), su=1.0f/sqrtf((float)Ee);   /* E*Up ~ N(0, 0.02), as the unfactorized init */
        for(size_t i=0;i<n;i++) hp[i]=(i<ne?0.02f:su)*rnd_normal();
        ck(cudaMemcpy(g_feP,hp,n*4,cudaMemcpyHostToDevice),"fe init"); free(hp);
        ck(cudaMemset(g_feM,0,n*2),"feM0"); ck(cudaMemset(g_feV,0,n*2),"feV0"); }
    fe_mat(); kchk("fe init",-1);
    printf("[emb] factorized %d x %d x %d: %zu params instead of %zu (%s)\n",gC.V,Ee,gC.D,n,(size_t)gC.V*gC.D,f?"resumed":"new");
    return 1; }
static void fe_save(const char*optpath,int it){
    if(!g_feE) return;
    char p[1200],tmp[1300]; snprintf(p,sizeof p,"%s.fe",optpath); FILE*f=save_open(p,tmp,sizeof tmp); if(!f) return;
    size_t n=(size_t)gC.V*g_feE+(size_t)g_feE*gC.D; int h[4]={g_feE,gC.V,gC.D,it};
    fwrite("FES1",1,4,f); fwrite(h,4,4,f);
    t1_rw(f,g_feP,n*4,1); t1_rw(f,g_feM,n*2,1); t1_rw(f,g_feV,n*2,1);
    save_commit(f,tmp,p); }
static void t1_update(float lr,int t){
    x1_update(lr,t); fe_update(lr,t);
    float b1=0.9f,b2=0.95f,bc1=1.0f-powf(b1,(float)t),bc2=1.0f-powf(b2,(float)t);
    if(g_attg){ size_t n=(size_t)gC.L*gC.D*gC.D;
        KL(k_adamw,nblk(n,TPB),TPB)(g_Wg,g_WgG,g_WgM,g_WgV,n,lr,b1,b2,0.0f,bc1,bc2,(const float*)0,(size_t)0,(size_t)0,0,MU_OFF,g_side_gs); }
    if(g_ngram_on&&g_ng_gate){ size_t n=(size_t)g_ng_De*gC.D;
        KL(k_adamw,nblk(n,TPB),TPB)(g_ngWk,g_ngWkG,g_ngWkM,g_ngWkV,n,lr,b1,b2,0.0f,bc1,bc2,(const float*)0,(size_t)0,(size_t)0,0,MU_OFF,g_side_gs); }
    if(g_attg||g_ng_gate) kchk("t1 adam",-1);
}
static int t1_rw(FILE*f,void*dev,size_t by,int wr){
    void*h=malloc(by); int ok=1;
    if(wr){ ck(cudaMemcpy(h,dev,by,cudaMemcpyDeviceToHost),"t1 d2h"); fwrite(h,1,by,f); }
    else if((ok=fread(h,1,by,f)==by)) ck(cudaMemcpy(dev,h,by,cudaMemcpyHostToDevice),"t1 h2d");
    free(h); return ok;
}
/* sidecar <optpath>.t1: Wg(+m,v), ngram Wk(+m,v), router bias; header validates flags */
static void t1_save(const char*optpath,int it){
    x1_save(optpath,it);
    int gate=g_ngram_on&&g_ng_gate, tri=g_ngram_on&&g_ng_tri, mb=g_mbiasH?mb_S()*gC.n_exp:0;
    if(!g_attg&&!gate&&!tri&&!mb) return;
    char path[1200]; snprintf(path,sizeof path,"%s.t1",optpath);
    char tmp[1300]; FILE*f=save_open(path,tmp,sizeof tmp); if(!f) return;
    int hdr[9]={g_attg,gC.L,gC.D,gate,g_ng_De,mb,tri,g_ng_layer,it};
    fwrite("T1S1",1,4,f); fwrite(hdr,4,9,f);
    if(g_attg){ size_t n=(size_t)gC.L*gC.D*gC.D; t1_rw(f,g_Wg,n*4,1); t1_rw(f,g_WgM,n*2,1); t1_rw(f,g_WgV,n*2,1); }
    if(gate){ size_t n=(size_t)g_ng_De*gC.D; t1_rw(f,g_ngWk,n*4,1); t1_rw(f,g_ngWkM,n*2,1); t1_rw(f,g_ngWkV,n*2,1); }
    if(mb) fwrite(g_mbiasH,4,(size_t)gC.L*mb,f);
    save_commit(f,tmp,path);
}
static void t1_load(const char*optpath){
    x1_load(optpath);
    char path[1200]; snprintf(path,sizeof path,"%s.t1",optpath);
    FILE*f=fopen(path,"rb"); if(!f) return;
    char m[4]; int hdr[9];
    if(fread(m,1,4,f)!=4||memcmp(m,"T1S1",4)||fread(hdr,4,9,f)!=9){ fclose(f); return; }
    int gate=g_ngram_on&&g_ng_gate, tri=g_ngram_on&&g_ng_tri;
    if(hdr[0]!=g_attg||(hdr[1]!=gC.L&&!(gC.lp&&hdr[1]==gC.lp))||hdr[2]!=gC.D||hdr[3]!=gate||(gate&&hdr[4]!=g_ng_De)||hdr[6]!=tri||hdr[7]!=g_ng_layer){
        fprintf(stderr,"[t1] %s mismatch: file attg=%d gate=%d tri=%d layer=%d vs env attg=%d gate=%d tri=%d layer=%d\n",
                path,hdr[0],hdr[3],hdr[6],hdr[7],g_attg,gate,tri,g_ng_layer); fclose(f); exit(1); }
    int ok=1;
    if(g_attg){ size_t n=(size_t)gC.L*gC.D*gC.D; ok=t1_rw(f,g_Wg,n*4,0)&&t1_rw(f,g_WgM,n*2,0)&&t1_rw(f,g_WgV,n*2,0); }
    if(ok&&gate){ size_t n=(size_t)g_ng_De*gC.D; ok=t1_rw(f,g_ngWk,n*4,0)&&t1_rw(f,g_ngWkM,n*2,0)&&t1_rw(f,g_ngWkV,n*2,0); }
    if(ok&&hdr[5]) g_mbias_slot=hdr[5]>gC.n_exp;   /* file decides per-slot vs shared bias */
    if(ok&&hdr[5]&&hdr[5]==mb_S()*gC.n_exp){ size_t n=(size_t)hdr[1]*hdr[5]; float*b=(float*)malloc((size_t)gC.L*hdr[5]*4);
        if((ok=fread(b,4,n,f)==n)){ if(hdr[1]!=gC.L){ n=(size_t)gC.L*hdr[5];   /* loop: expand per-layer bias to the virtual depth */
                for(int v=gC.L-1;v>=0;v--) memmove(b+(size_t)v*hdr[5],b+(size_t)gC.lmap[v]*hdr[5],hdr[5]*4); } g_mbias_ld=1; mbias(0); memcpy(g_mbiasH,b,n*4);
            ck(cudaMemcpy(g_mbiasD,b,n*4,cudaMemcpyHostToDevice),"t1 mbias"); }
        free(b); }
    fclose(f);
    printf("[t1] %s %s (iter %d)\n",ok?"loaded":"PARTIAL read of",path,hdr[8]);
}
static void ngram_infer(const char*path){
    char op[1100]; snprintf(op,sizeof(op),"%s.opt",path);
    { char ng[1110]; snprintf(ng,sizeof(ng),"%s.ng",op); FILE*f;   /* default-on ngram: skip for models without a table */
      if(!getenv("CUDALM_NGRAM")&&!eg_wanted()){ if((f=fopen(ng,"rb"))) fclose(f); else { t1_setup(&gC); t1_load(op); return; } }
      if((f=fopen(ng,"rb"))){ char m[4]; g_ng_fh[0]=g_eg_fh[0]=0;
          if(fread(m,1,4,f)==4){
              if(!memcmp(m,"NGR1",4)){ if(fread(g_ng_fh,4,4,f)!=4) g_ng_fh[0]=0; }
              else if(!memcmp(m,"NGR2",4)&&fread(g_eg_fh,4,24,f)!=24) g_eg_fh[0]=0; }
          fclose(f); } }
    ngram_setup(&gC); if(g_ngram_on&&!ngram_load(op)) printf("[ngram] WARNING: no trained table at %s.ng (random init!)\n",op);
    if(g_eg_on&&!engram_load(op)) printf("[engram] WARNING: no trained table at %s.ng (random init!)\n",op);
    t1_load(op);
}

/* ---- Sparse engram (cudalm17, SPARSE_ENGRAM_PLAN P2) ----
 * S=n_orders*H hashed sub-tables, each a row range [s*Rp, s*Rp+prime_s) of the
 * one g_ng table (so q4/fold/optimizer state are shared with the ngram code).
 * Per position the S rows are concatenated (S*De), projected by a per-point Wv
 * (E2: sigmoid(signed-sqrt(<rms h, rms k>)) gate with k=buf@Wk; E3: causal
 * depthwise conv k=4 + SiLU + residual) and added to the residual stream at one
 * or more layers (E4, table gathered once). E5 canonical token map, E6 table LR
 * multiplier. Everything lives in <optpath>.ng as NGR2. */
#define EG_MAXS 32
#define EG_MAXP 8
static int g_eg_H=1, g_eg_no=1, g_eg_ord[4]={2,0,0,0}, g_eg_S=1, g_eg_Rraw=0, g_eg_Rp=0;
static int g_eg_np=1, g_eg_lay[EG_MAXP], g_eg_conv=0, g_eg_gate=0, g_eg_canon=0, g_eg_canon_env=0, g_eg_dil=2;
static float g_eg_lrm=1.0f;
static int g_eg_primeH[EG_MAXS]; static unsigned long long g_eg_multH[EG_MAXS*4];
static unsigned short *g_eg_canonH=0,*g_eg_canonD=0;
__constant__ int c_eg_ord[EG_MAXS], c_eg_prime[EG_MAXS];
__constant__ unsigned long long c_eg_mult[EG_MAXS*4];
typedef struct { int layer;
    float *Wv,*WvG,*Wk,*WkG,*nw,*nwG,*K,*KG; bf16 *WvM,*WvV,*WkM,*WkV,*nwM,*nwV,*KM,*KV;
    float *v,*vg,*kn,*hn,*g,*rh,*rk,*y,*c,*ri; } EgP;
static EgP g_egp[EG_MAXP];
static float *g_eg_dvg=0,*g_eg_dv=0,*g_eg_dk=0,*g_eg_dy=0,*g_eg_dc=0;
static int g_eg_ncap=0;

static unsigned long long splitmix64(unsigned long long x){
    x+=0x9e3779b97f4a7c15ULL; x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL; x=(x^(x>>27))*0x94d049bb133111ebULL; return x^(x>>31); }
static int eg_wanted(void){
    static const char*v[]={"CUDALM_ENGRAM","NGRAM_HEADS","NGRAM_ORDERS","NGRAM_LAYERS","NGRAM_CONV","NGRAM_CANON","NGRAM_LRMULT"};
    const char*e=getenv(v[0]); if(e) return atoi(e)!=0;
    for(int i=1;i<7;i++) if(getenv(v[i])) return 1;
    return 0; }
static int eg_ilist(const char*s,int*o,int mx){ int n=0; while(s&&*s&&n<mx){ o[n++]=atoi(s); s=strchr(s,','); if(s)s++; } return n; }
static int eg_isprime(int p){ if(p<2) return 0; for(int d=2;(long long)d*d<=p;d++) if(p%d==0) return 0; return 1; }
static void eg_par(float**p,float**g,bf16**m,bf16**v,size_t n){
    *p=(float*)dmalloc(n*4); *g=(float*)dmalloc(n*4); *m=(bf16*)dmalloc(n*2); *v=(bf16*)dmalloc(n*2);
    ck(cudaMemset(*p,0,n*4),"eg p0"); ck(cudaMemset(*m,0,n*2),"eg m0"); ck(cudaMemset(*v,0,n*2),"eg v0"); }

static void engram_setup(const Cfg*c){
    const char*e; int*fh=g_eg_fh, D=c->D;
    g_eg_on=1; g_ngram_on=0;
    memset(g_eg_ord,0,sizeof g_eg_ord); g_eg_ord[0]=2;
    g_eg_H=(e=getenv("NGRAM_HEADS"))?atoi(e):1;
    g_eg_no=(e=getenv("NGRAM_ORDERS"))?eg_ilist(e,g_eg_ord,4):1;
    g_ng_De=(e=getenv("NGRAM_DIM"))?atoi(e):64;
    g_eg_Rraw=(e=getenv("NGRAM_ROWS"))?atoi(e):655360;
    g_ng_q4=(e=getenv("NGRAM_Q4"))?atoi(e)!=0:1;
    if((e=getenv("NGRAM_LAYERS"))) g_eg_np=eg_ilist(e,g_eg_lay,EG_MAXP); else { g_eg_np=1; g_eg_lay[0]=g_ng_layer; }
    g_eg_conv=((e=getenv("NGRAM_CONV"))&&atoi(e))?4:0;
    g_eg_gate=g_ng_gate;
    g_eg_lrm=(e=getenv("NGRAM_LRMULT"))?(float)atof(e):1.0f;
    const char*cf=getenv("NGRAM_CANON"); g_eg_canon=cf!=0; g_eg_canon_env=cf!=0;
    if(fh[0]){ g_eg_H=fh[0]; g_eg_no=fh[1]; memcpy(g_eg_ord,fh+2,16); g_ng_De=fh[6]; g_eg_Rraw=fh[7]; g_ng_q4=fh[8];
        g_eg_np=fh[10]; memcpy(g_eg_lay,fh+11,EG_MAXP*4); g_eg_conv=fh[19]; g_eg_canon=fh[20]; g_eg_gate=fh[21];
        if(cf&&!g_eg_canon){ g_eg_canon_env=0; cf=0; } }   /* inference: file decides */
    for(int i=0;i<g_eg_np;i++){ int l=g_eg_lay[i]; g_eg_lay[i]=l<0?0:l>=c->L?c->L-1:l; }
    for(int i=1;i<g_eg_np;i++) for(int j=i;j>0&&g_eg_lay[j]<g_eg_lay[j-1];j--){ int t=g_eg_lay[j]; g_eg_lay[j]=g_eg_lay[j-1]; g_eg_lay[j-1]=t; }
    { int n=0; for(int i=0;i<g_eg_np;i++) if(!n||g_eg_lay[i]!=g_eg_lay[n-1]) g_eg_lay[n++]=g_eg_lay[i]; g_eg_np=n; }
    g_eg_S=g_eg_no*g_eg_H; int De=g_ng_De;
    int bad=g_eg_H<1||g_eg_no<1||g_eg_S>EG_MAXS||De<1||g_eg_Rraw<2||g_eg_np<1;
    g_eg_dil=1; for(int i=0;i<g_eg_no;i++){ if(g_eg_ord[i]<1||g_eg_ord[i]>4) bad=1; if(g_eg_ord[i]>g_eg_dil) g_eg_dil=g_eg_ord[i]; }
    if(bad){ fprintf(stderr,"[engram] bad config: H=%d orders=%d S=%d (max %d, orders 1..4) De=%d R=%d\n",
                     g_eg_H,g_eg_no,g_eg_S,EG_MAXS,De,g_eg_Rraw); exit(1); }
    if(g_eg_canon){ g_eg_canonH=(unsigned short*)calloc(c->V,2);
        if(cf){ FILE*f=fopen(cf,"rb");
            if(!f||fread(g_eg_canonH,2,c->V,f)!=(size_t)c->V){ fprintf(stderr,"[engram] canon map %s: need %d uint16\n",cf,c->V); exit(1); }
            fclose(f); }
        g_eg_canonD=(unsigned short*)dmalloc((size_t)c->V*2);
        ck(cudaMemcpy(g_eg_canonD,g_eg_canonH,(size_t)c->V*2,cudaMemcpyHostToDevice),"eg canon"); }
    int al=(128%De==0)?128/De:1;
    g_eg_Rp=(g_eg_Rraw+al-1)/al*al; g_ng_R=g_eg_S*g_eg_Rp;
    { int p=g_eg_Rraw+1, ordS[EG_MAXS];
      for(int s=0;s<g_eg_S;s++){ do p--; while(p>1&&!eg_isprime(p));
          if(p<2){ fprintf(stderr,"[engram] NGRAM_ROWS=%d too small for %d sub-tables\n",g_eg_Rraw,g_eg_S); exit(1); }
          g_eg_primeH[s]=p; ordS[s]=g_eg_ord[s/g_eg_H];
          for(int k=0;k<4;k++) g_eg_multH[s*4+k]=splitmix64(0x5EED0E17A11ULL+(unsigned long long)(s*4+k))|1ULL; }
      ck(cudaMemcpyToSymbol(c_eg_ord,ordS,g_eg_S*4),"eg ord");
      ck(cudaMemcpyToSymbol(c_eg_prime,g_eg_primeH,g_eg_S*4),"eg prime");
      ck(cudaMemcpyToSymbol(c_eg_mult,g_eg_multH,(size_t)g_eg_S*32),"eg mult"); }
    g_ng_q4fold=0;
    { const char*f=getenv("NGRAM_Q4_FOLD");
      if(f?atoi(f)!=0:g_ng_q4){ if(g_ng_q4&&128%De==0) g_ng_q4fold=1;
          else printf("[engram] Q4_FOLD ignored: needs NGRAM_Q4=1 and 128%%De==0 (De=%d)\n",De); } }
    size_t nt=(size_t)g_ng_R*De, sd=(size_t)g_eg_S*De, nw=sd*D;
    g_ng=(float*)dmalloc(nt*4);
    if(!g_ng_q4fold) g_ngG=(float*)dmalloc(nt*4);
    if(!g_ng_q4){ g_ngM=(bf16*)dmalloc(nt*2); g_ngV=(bf16*)dmalloc(nt*2);
        ck(cudaMemset(g_ngM,0,nt*2),"eg M0"); ck(cudaMemset(g_ngV,0,nt*2),"eg V0"); }
    { float*h=(float*)malloc(nt*4); unsigned s=0x9e3779b9u;
      for(size_t i=0;i<nt;i++){ s^=s<<13; s^=s>>17; s^=s<<5; h[i]=0.02f*(((s&0xffffff)/(float)0x1000000)*2.0f-1.0f); }
      ck(cudaMemcpy(g_ng,h,nt*4,cudaMemcpyHostToDevice),"eg tbl init"); free(h); }
    if(g_ng_q4) ngram_q4_setup();
    unsigned s=0x2545F491u; float sc=1.0f/sqrtf((float)sd);
    float*w=(float*)malloc((nw>(size_t)D?nw:(size_t)D)*4);
    for(int i=0;i<g_eg_np;i++){ EgP*P=&g_egp[i]; memset(P,0,sizeof *P); P->layer=g_eg_lay[i];
        eg_par(&P->Wv,&P->WvG,&P->WvM,&P->WvV,nw);                       /* zero -> step-0 no-op */
        if(g_eg_gate){ eg_par(&P->Wk,&P->WkG,&P->WkM,&P->WkV,nw);
            for(size_t j=0;j<nw;j++){ s^=s<<13; s^=s>>17; s^=s<<5; w[j]=sc*(((s&0xffffff)/(float)0x1000000)*2.0f-1.0f); }
            ck(cudaMemcpy(P->Wk,w,nw*4,cudaMemcpyHostToDevice),"eg Wk init"); }
        if(g_eg_conv){ eg_par(&P->nw,&P->nwG,&P->nwM,&P->nwV,D); eg_par(&P->K,&P->KG,&P->KM,&P->KV,(size_t)D*4);
            for(int j=0;j<D;j++) w[j]=1.0f;
            ck(cudaMemcpy(P->nw,w,(size_t)D*4,cudaMemcpyHostToDevice),"eg nw init"); } }
    free(w);
    char ob[32]={0},lb[64]={0};
    for(int i=0;i<g_eg_no;i++) snprintf(ob+strlen(ob),sizeof ob-strlen(ob),"%s%d",i?",":"",g_eg_ord[i]);
    for(int i=0;i<g_eg_np;i++) snprintf(lb+strlen(lb),sizeof lb-strlen(lb),"%s%d",i?",":"",g_eg_lay[i]);
    printf("[engram] on: S=%d (H=%d x orders %s) De=%d R=%d/sub (pad %d) layers=%s gate=%d conv=%d canon=%d lrmult=%g\n"
           "[engram] table %d rows = %.1f MiB (%s)%s\n",
           g_eg_S,g_eg_H,ob,De,g_eg_Rraw,g_eg_Rp,lb,g_eg_gate,g_eg_conv,g_eg_canon,g_eg_lrm,g_ng_R,
           g_ng_q4?(double)((nt+1)/2+3*nt)/1048576.0:(double)nt*4/1048576.0,
           g_ng_q4?"q4 digit+resid+int8 moments":"fp32",g_ng_q4fold?"  [Q4_FOLD]":"");
}

/* key i=s*N+n: -1 without enough history, else a sub-table row */
#ifndef CUDALM_CPU
__global__ void k_eg_key(const int*tok,const unsigned short*cn,int*key,int N,int T,int S,int Rp){
    int i=(int)(blockIdx.x*blockDim.x+threadIdx.x); if(i>=S*N) return;
    int s=i/N, n=i%N, o=c_eg_ord[s];
    if(n%T<o-1){ key[i]=-1; return; }
    unsigned long long m=0;
    for(int k=0;k<o;k++){ unsigned t=(unsigned)tok[n-k]; if(cn) t=cn[t]; m^=(unsigned long long)t*c_eg_mult[s*4+k]; }
    key[i]=(int)(m%(unsigned long long)c_eg_prime[s])+s*Rp; }
__global__ void k_eg_gather(const float*tbl,const int*key,float*buf,int N,int S,int De){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x, SD=(size_t)S*De; if(i>=(size_t)N*SD) return;
    int n=(int)(i/SD), r=(int)(i%SD), s=r/De, e=r%De, k=key[(size_t)s*N+n];
    buf[i]= k>=0 ? tbl[(size_t)k*De+e] : 0.0f; }
__global__ void k_eg_scatter(float*G,const int*key,const float*db,int N,int S,int De){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x, SD=(size_t)S*De; if(i>=(size_t)N*SD) return;
    int n=(int)(i/SD), r=(int)(i%SD), s=r/De, e=r%De, k=key[(size_t)s*N+n];
    if(k>=0) atomicAdd(&G[(size_t)k*De+e],db[i]); }
__global__ void k_eg_scatter_c(float*Gc,const int*key,const int*cidx,const float*db,int N,int S,int De,int grp){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x, SD=(size_t)S*De; if(i>=(size_t)N*SD) return;
    int n=(int)(i/SD), r=(int)(i%SD), s=r/De, e=r%De; size_t ki=(size_t)s*N+n; int k=key[ki]; if(k<0) return;
    int c=cidx[ki]; if(c<0) return;
    atomicAdd(&Gc[(size_t)c*grp+(int)(((long long)k*De+e)%grp)],db[i]); }
/* E2 gate, one 128-thread block per row: z=<rms h,rms k>/sqrt(D), g=σ(sign(z)sqrt|z|), vg=g*v.
 * h untouched; stores rms(h) in hn, rms(k) in k (in place), g/rh/rk per row. */
__global__ void k_eg_gate_f(const float*h,const float*v,float*k,float*hn,float*vg,float*g,float*rh,float*rk,int D){
    size_t o=(size_t)blockIdx.x*D; float a=0,b=0,c=0;
    for(int i=threadIdx.x;i<D;i+=blockDim.x){ float x=h[o+i],y=k[o+i]; a+=x*x; b+=y*y; c+=x*y; }
    bsum3(a,b,c);
    float r1=rsqrtf(a/D+1e-6f), r2=rsqrtf(b/D+1e-6f), z=c*r1*r2*rsqrtf((float)D);
    float gg=1.0f/(1.0f+expf(-copysignf(sqrtf(fmaxf(fabsf(z),1e-6f)),z)));
    for(int i=threadIdx.x;i<D;i+=blockDim.x){ hn[o+i]=h[o+i]*r1; k[o+i]*=r2; vg[o+i]=gg*v[o+i]; }
    if(threadIdx.x==0){ g[blockIdx.x]=gg; rh[blockIdx.x]=r1; rk[blockIdx.x]=r2; } }
/* backward: dv=g*dvg, dk through z, dh += dh through z. dvg may alias dh. */
__global__ void k_eg_gate_b(const float*dvg,float*dh,const float*v,const float*hn,const float*kn,const float*g,
                            const float*rh,const float*rk,float*dv,float*dk,int D){
    size_t o=(size_t)blockIdx.x*D; float a=0,b=0,c=0;
    for(int i=threadIdx.x;i<D;i+=blockDim.x){ a+=dvg[o+i]*v[o+i]; b+=hn[o+i]*kn[o+i]; }
    bsum3(a,b,c);
    float gg=g[blockIdx.x], m=b/D, z=m*sqrtf((float)D), r1=rh[blockIdx.x], r2=rk[blockIdx.x];
    float cz=a*gg*(1.0f-gg)*0.5f*rsqrtf(fmaxf(fabsf(z),1e-6f))*rsqrtf((float)D);
    for(int i=threadIdx.x;i<D;i+=blockDim.x){ float d=dvg[o+i],x=hn[o+i],y=kn[o+i];
        dv[o+i]=gg*d; dk[o+i]=cz*(x-y*m)*r2; dh[o+i]+=cz*(y-x*m)*r1; } }
/* E3: c[n,d]=sum_j K[d,j]*y[n-j*dil,d] (causal, per sequence); h += vg + SiLU(c) */
__global__ void k_eg_conv_f(float*h,const float*vg,const float*y,const float*K,float*c,int N,int D,int T,int dil){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(size_t)N*D) return;
    int n=(int)(i/D), d=(int)(i%D), p=n%T; float s=0;
    for(int j=0;j<4;j++) if(p>=j*dil) s+=K[d*4+j]*y[i-(size_t)j*dil*D];
    c[i]=s; h[i]+=vg[i]+s/(1.0f+expf(-s)); }
__global__ void k_eg_conv_dc(const float*dout,const float*c,float*dc,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float x=c[i], s=1.0f/(1.0f+expf(-x)); dc[i]=dout[i]*s*(1.0f+x*(1.0f-s)); }
__global__ void k_eg_conv_dy(const float*dc,const float*K,float*dy,int N,int D,int T,int dil){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(size_t)N*D) return;
    int n=(int)(i/D), d=(int)(i%D), p=n%T; float s=0;
    for(int j=0;j<4;j++) if(p+j*dil<T&&n+j*dil<N) s+=K[d*4+j]*dc[i+(size_t)j*dil*D];
    dy[i]=s; }
/* dK, deterministic: one 256-thread block per channel d (overwrites KG) */
__global__ void k_eg_conv_dK(const float*dc,const float*y,float*KG,int N,int D,int T,int dil){
    int d=blockIdx.x, t=threadIdx.x; float s[4]={0,0,0,0};
    for(int n=t;n<N;n+=blockDim.x){ float g=dc[(size_t)n*D+d]; int p=n%T;
        for(int j=0;j<4;j++) if(p>=j*dil) s[j]+=g*y[(size_t)(n-j*dil)*D+d]; }
    __shared__ float sh[4][256];
    for(int j=0;j<4;j++) sh[j][t]=s[j];
    __syncthreads();
    for(int st=128;st>0;st>>=1){ if(t<st) for(int j=0;j<4;j++) sh[j][t]+=sh[j][t+st]; __syncthreads(); }
    if(t<4) KG[d*4+t]=sh[t][0]; }
#else
#define CPUK_PART 9
#include "cpu_kern.h"
#endif

static void eg_re(void*pp,size_t by){ void**p=(void**)pp; if(*p) cudaFree(*p); *p=by?dmalloc(by):0; }
static void eg_ensure(int N){
    if(N<=g_eg_ncap) return;
    int S=g_eg_S, NK=S*N, G=g_eg_gate, C=g_eg_conv;
    size_t ND=(size_t)N*gC.D*4, NR=(size_t)N*4, NS=(size_t)NK*g_ng_De*4;
    eg_re(&g_ngbuf,NS); eg_re(&g_ngdbuf,NS); eg_re(&g_ngkey,(size_t)NK*4);
    eg_re(&g_eg_dvg,C?ND:0); eg_re(&g_eg_dy,C?ND:0); eg_re(&g_eg_dc,C?ND:0);
    eg_re(&g_eg_dv,G?ND:0); eg_re(&g_eg_dk,G?ND:0);
    for(int i=0;i<g_eg_np;i++){ EgP*P=&g_egp[i];
        eg_re(&P->v,ND); eg_re(&P->vg,G?ND:0); eg_re(&P->kn,G?ND:0); eg_re(&P->hn,G?ND:0);
        eg_re(&P->g,G?NR:0); eg_re(&P->rh,G?NR:0); eg_re(&P->rk,G?NR:0);
        eg_re(&P->y,C?ND:0); eg_re(&P->c,C?ND:0); eg_re(&P->ri,C?NR:0); }
    if(g_ng_q4fold){
        if(g_ngGc){ cudaFree(g_ngGc); cudaFree(g_ng_cidxD); cudaFree(g_ng_tgiD);
                    cudaFreeHost(g_ng_hk); free(g_ng_hgi); free(g_ng_hord); cudaFreeHost(g_ng_hcidx); cudaFreeHost(g_ng_htgi); }
        g_ng_gcap=NK;
        g_ngGc=(float*)dmalloc((size_t)NK*g_ngq_grp*4);
        g_ng_cidxD=(int*)dmalloc((size_t)NK*4); g_ng_tgiD=(int*)dmalloc((size_t)NK*4);
        ck(cudaHostAlloc((void**)&g_ng_hk,(size_t)NK*4,0),"eg pin");
        ck(cudaHostAlloc((void**)&g_ng_hcidx,(size_t)NK*4,0),"eg pin");
        ck(cudaHostAlloc((void**)&g_ng_htgi,(size_t)NK*4,0),"eg pin");
        g_ng_hgi=(int*)malloc((size_t)NK*4); g_ng_hord=(int*)malloc((size_t)NK*4);
    }
    g_eg_ncap=N;
}
static EgP* eg_pt(int l){ for(int i=0;i<g_eg_np;i++) if(g_egp[i].layer==l) return &g_egp[i]; return 0; }
static void engram_forward(GA*a,const int*tokG,int N,int l){
    EgP*P=eg_pt(l); if(!P) return;
    int S=g_eg_S, De=g_ng_De, D=gC.D, SD=S*De; size_t ND=(size_t)N*D;
    eg_ensure(N);                                   /* (re)allocates P->v: read it after */
    float*h=a->a_in[l], *vg=P->v;
    if(P==g_egp){                                   /* lowest point: gather once */
        KL(k_eg_key,nblk((size_t)S*N,TPB),TPB)(tokG,g_eg_canonD,g_ngkey,N,gC.T,S,g_eg_Rp);
        KL(k_eg_gather,nblk((size_t)N*SD,TPB),TPB)(g_ng,g_ngkey,g_ngbuf,N,S,De); kchk("eg gather",-1); }
    g_ab(g_ngbuf,P->Wv,P->v,N,SD,D);
    if(g_eg_gate){ g_ab(g_ngbuf,P->Wk,P->kn,N,SD,D);
        KL(k_eg_gate_f,N,128)(h,P->v,P->kn,P->hn,P->vg,P->g,P->rh,P->rk,D); vg=P->vg; }
    if(g_eg_conv){ e_rmsf(vg,P->nw,0,P->y,P->ri,N,D);
        KL(k_eg_conv_f,nblk(ND,TPB),TPB)(h,vg,P->y,P->K,P->c,N,D,gC.T,g_eg_dil); }
    else e_add(h,vg,ND);
    kchk("eg fwd",l);
}
static void eg_scatter(int N){
    int S=g_eg_S, De=g_ng_De; size_t n=(size_t)N*S*De;
    if(g_ng_q4fold){
        ng_build_touched(S*N);
        if(g_ng_Gt) ck(cudaMemset(g_ngGc,0,(size_t)g_ng_Gt*g_ngq_grp*4),"eg Gc0");
        KL(k_eg_scatter_c,nblk(n,TPB),TPB)(g_ngGc,g_ngkey,g_ng_cidxD,g_ngdbuf,N,S,De,g_ngq_grp);
    } else {
        ck(cudaMemset(g_ngG,0,(size_t)g_ng_R*De*4),"eg G0");
        KL(k_eg_scatter,nblk(n,TPB),TPB)(g_ngG,g_ngkey,g_ngdbuf,N,S,De);
    }
    kchk("eg scatter",-1);
}
/* gradient w.r.t. the post-injection residual is in a->dout; on return it holds
 * the gradient w.r.t. the pre-injection residual (gate adds its dh). */
static void engram_backward(GA*a,const int*tokG,int N,int l){
    EgP*P=eg_pt(l); if(!P) return;
    int S=g_eg_S, De=g_ng_De, D=gC.D, SD=S*De; size_t ND=(size_t)N*D, nw=(size_t)SD*D;
    float*dvg=a->dout, *dv;
    if(g_eg_conv){ const float*vg=g_eg_gate?P->vg:P->v;
        KL(k_eg_conv_dc,nblk(ND,TPB),TPB)(a->dout,P->c,g_eg_dc,ND);
        KL(k_eg_conv_dy,nblk(ND,TPB),TPB)(g_eg_dc,P->K,g_eg_dy,N,D,gC.T,g_eg_dil);
        KL(k_eg_conv_dK,D,256)(g_eg_dc,P->y,P->KG,N,D,gC.T,g_eg_dil);
        e_cpy(g_eg_dvg,a->dout,ND); ck(cudaMemset(P->nwG,0,(size_t)D*4),"eg nwG0");
        e_rmsb(vg,P->nw,0,P->ri,g_eg_dy,g_eg_dvg,P->nwG,0,N,D); dvg=g_eg_dvg; }
    dv=dvg;
    if(g_eg_gate){ KL(k_eg_gate_b,N,128)(dvg,a->dout,P->v,P->hn,P->kn,P->g,P->rh,P->rk,g_eg_dv,g_eg_dk,D); dv=g_eg_dv; }
    ck(cudaMemset(P->WvG,0,nw*4),"eg WvG0");
    g_atb(g_ngbuf,dv,P->WvG,N,SD,D);
    if(P==&g_egp[g_eg_np-1]) g_abt(dv,P->Wv,g_ngdbuf,N,D,SD);   /* highest point runs first */
    else g_abt_b(dv,P->Wv,g_ngdbuf,N,D,SD,1.0f);
    if(g_eg_gate){ ck(cudaMemset(P->WkG,0,nw*4),"eg WkG0");
        g_atb(g_ngbuf,g_eg_dk,P->WkG,N,SD,D); g_abt_b(g_eg_dk,P->Wk,g_ngdbuf,N,D,SD,1.0f); }
    kchk("eg bwd",l);
    if(P==g_egp) eg_scatter(N);
}
static void eg_adam(float*p,float*g,bf16*m,bf16*v,size_t n,float lr,float bc1,float bc2){
    KL(k_adamw,nblk(n,TPB),TPB)(p,g,m,v,n,lr,0.9f,0.95f,0.0f,bc1,bc2,(const float*)0,(size_t)0,(size_t)0,0,MU_OFF,g_side_gs); }
static void engram_update(float lr,int t){
    float bc1=1.0f-powf(0.9f,(float)t),bc2=1.0f-powf(0.95f,(float)t);
    size_t nw=(size_t)g_eg_S*g_ng_De*gC.D, D=gC.D;
    for(int i=0;i<g_eg_np;i++){ EgP*P=&g_egp[i];
        eg_adam(P->Wv,P->WvG,P->WvM,P->WvV,nw,lr,bc1,bc2);
        if(g_eg_gate) eg_adam(P->Wk,P->WkG,P->WkM,P->WkV,nw,lr,bc1,bc2);
        if(g_eg_conv){ eg_adam(P->nw,P->nwG,P->nwM,P->nwV,D,lr,bc1,bc2); eg_adam(P->K,P->KG,P->KM,P->KV,D*4,lr,bc1,bc2); } }
    ng_tbl_step(lr*g_eg_lrm,t);
    kchk("eg adam",-1);
}
static void lrs_add(void*d,size_t n);
static void eg_lrs(void){
    size_t nw=(size_t)g_eg_S*g_ng_De*gC.D, D=gC.D;
    for(int i=0;i<g_eg_np;i++){ EgP*P=&g_egp[i];
        lrs_add(P->Wv,nw*4); lrs_add(P->WvM,nw*2); lrs_add(P->WvV,nw*2);
        if(g_eg_gate){ lrs_add(P->Wk,nw*4); lrs_add(P->WkM,nw*2); lrs_add(P->WkV,nw*2); }
        if(g_eg_conv){ lrs_add(P->nw,D*4); lrs_add(P->nwM,D*2); lrs_add(P->nwV,D*2);
                       lrs_add(P->K,D*16); lrs_add(P->KM,D*8); lrs_add(P->KV,D*8); } }
}
/* NGR2 sidecar <optpath>.ng: "NGR2", int hdr[24], primes, multipliers, canon map,
 * per-point weights+moments, table state (same layout as NGR1). */
static void eg_hdr(int*h,int it){
    memset(h,0,24*4); h[0]=g_eg_H; h[1]=g_eg_no; memcpy(h+2,g_eg_ord,16); h[6]=g_ng_De; h[7]=g_eg_Rraw;
    h[8]=g_ng_q4; h[9]=g_ng_q4fold; h[10]=g_eg_np; for(int i=0;i<g_eg_np;i++) h[11+i]=g_egp[i].layer;
    h[19]=g_eg_conv; h[20]=g_eg_canon; h[21]=g_eg_gate; h[22]=gC.D; h[23]=it; }
static int eg_rw_pts(FILE*f,int wr){
    size_t nw=(size_t)g_eg_S*g_ng_De*gC.D, D=gC.D; int ok=1;
    for(int i=0;i<g_eg_np&&ok;i++){ EgP*P=&g_egp[i];
        ok=t1_rw(f,P->Wv,nw*4,wr)&&t1_rw(f,P->WvM,nw*2,wr)&&t1_rw(f,P->WvV,nw*2,wr);
        if(ok&&g_eg_gate) ok=t1_rw(f,P->Wk,nw*4,wr)&&t1_rw(f,P->WkM,nw*2,wr)&&t1_rw(f,P->WkV,nw*2,wr);
        if(ok&&g_eg_conv) ok=t1_rw(f,P->nw,D*4,wr)&&t1_rw(f,P->nwM,D*2,wr)&&t1_rw(f,P->nwV,D*2,wr)
                           &&t1_rw(f,P->K,D*16,wr)&&t1_rw(f,P->KM,D*8,wr)&&t1_rw(f,P->KV,D*8,wr); }
    return ok; }
static int ng_tbl_rw(FILE*f,int wr){
    size_t nt=(size_t)g_ng_R*g_ng_De, ng=(size_t)g_ngq_ngrp*4;
    if(!g_ng_q4) return t1_rw(f,g_ng,nt*4,wr)&&t1_rw(f,g_ngM,nt*2,wr)&&t1_rw(f,g_ngV,nt*2,wr);
    if(wr) fwrite(&g_ngq_spH.sref,4,1,f);
    else { if(fread(&g_ngq_spH.sref,4,1,f)!=1) return 0;
        ck(cudaMemcpy(g_ngq_spD,&g_ngq_spH,sizeof(QSpanD),cudaMemcpyHostToDevice),"eg load span"); }
    int ok=t1_rw(f,g_ngq_d,(nt+1)/2,wr)&&t1_rw(f,g_ngq_r,nt,wr)&&t1_rw(f,g_ngq_m8,nt,wr)&&t1_rw(f,g_ngq_v8,nt,wr)
         &&t1_rw(f,g_ngq_s,ng,wr)&&t1_rw(f,g_ngq_ms,ng,wr)&&t1_rw(f,g_ngq_vs,ng,wr);
    if(ok&&!wr){ KL(k_qlm_render,(unsigned)g_ngq_ngrp,g_ngq_grp)(g_ng,g_ngq_d,g_ngq_s,g_ngq_spD,1,g_ngq_grp,1,0);
        kchk("eg load render",-1); }
    return ok; }
static void engram_save(const char*optpath,int it){
    char path[1200]; snprintf(path,sizeof path,"%s.ng",optpath);
    char tmp[1300]; FILE*f=save_open(path,tmp,sizeof tmp); if(!f) return;
    int h[24]; eg_hdr(h,it);
    fwrite("NGR2",1,4,f); fwrite(h,4,24,f);
    fwrite(g_eg_primeH,4,g_eg_S,f); fwrite(g_eg_multH,8,(size_t)g_eg_S*4,f);
    if(g_eg_canon) fwrite(g_eg_canonH,2,gC.V,f);
    eg_rw_pts(f,1); ng_tbl_rw(f,1);
    if(!save_commit(f,tmp,path)) fprintf(stderr,"engram save FAILED for %s (previous kept)\n",path);
}
static int engram_load(const char*optpath){
    char path[1200]; snprintf(path,sizeof path,"%s.ng",optpath);
    FILE*f=fopen(path,"rb"); if(!f) return 0;
    char m[4]; int h[24],e[24]; eg_hdr(e,0);
    if(fread(m,1,4,f)!=4||memcmp(m,"NGR2",4)||fread(h,4,24,f)!=24){
        fprintf(stderr,"[engram] resume mismatch: %s is not an NGR2 engram table\n",path); fclose(f); exit(1); }
    int bad=0; for(int i=0;i<23;i++) if(i!=9&&h[i]!=e[i]) bad=1;
    int pr[EG_MAXS]; unsigned long long mu[EG_MAXS*4];
    if(!bad&&(fread(pr,4,g_eg_S,f)!=(size_t)g_eg_S||fread(mu,8,(size_t)g_eg_S*4,f)!=(size_t)g_eg_S*4
              ||memcmp(pr,g_eg_primeH,g_eg_S*4)||memcmp(mu,g_eg_multH,(size_t)g_eg_S*32))) bad=2;
    if(bad){ fprintf(stderr,"[engram] resume mismatch (%s): file vs env hdr\n",bad==2?"hash primes/multipliers":"config");
        for(int i=0;i<23;i++) fprintf(stderr," %d:%d/%d",i,h[i],e[i]);
        fprintf(stderr,"\n"); fclose(f); exit(1); }
    if(g_eg_canon){ unsigned short*cm=(unsigned short*)malloc((size_t)gC.V*2);
        if(fread(cm,2,gC.V,f)!=(size_t)gC.V){ free(cm); fclose(f); return 0; }
        if(g_eg_canon_env&&memcmp(cm,g_eg_canonH,(size_t)gC.V*2)){
            fprintf(stderr,"[engram] resume mismatch: NGRAM_CANON map differs from %s\n",path); fclose(f); exit(1); }
        memcpy(g_eg_canonH,cm,(size_t)gC.V*2); free(cm);
        ck(cudaMemcpy(g_eg_canonD,g_eg_canonH,(size_t)gC.V*2,cudaMemcpyHostToDevice),"eg canon load"); }
    int ok=eg_rw_pts(f,0)&&ng_tbl_rw(f,0);
    fclose(f);
    printf("[engram] %s %s at iter %d\n",ok?"resumed table+weights from":"PARTIAL read of",path,h[23]);
    return ok?(h[23]>0?h[23]:1):0;
}
/* selftest: keys/gather/scatter vs CPU; finite-difference grads through two
 * injection points (identity "layer" between them), gate+conv+canon on. */
static float relerr(const float*a,const float*b,size_t n); static int check(const char*name,float e,float tol);
static double eg_st_loss(GA*a,const int*tokG,const float*h0,const float*w,int N){
    size_t ND=(size_t)N*gC.D; float*o=(float*)malloc(ND*4); double L=0;
    ck(cudaMemcpy(a->a_in[0],h0,ND*4,cudaMemcpyHostToDevice),"egst h0");
    engram_forward(a,tokG,N,0); e_cpy(a->a_in[1],a->a_in[0],ND); engram_forward(a,tokG,N,1);
    ck(cudaMemcpy(o,a->a_in[1],ND*4,cudaMemcpyDeviceToHost),"egst out");
    for(size_t i=0;i<ND;i++) L+=(double)w[i]*o[i];
    free(o); return L; }
static void eg_st_set(float*dp,float*h0,size_t i,float v){
    if(dp) ck(cudaMemcpy(dp+i,&v,4,cudaMemcpyHostToDevice),"egst set"); else h0[i]=v; }
static float eg_st_get(const float*dp,const float*h0,size_t i){
    float v; if(dp) ck(cudaMemcpy(&v,dp+i,4,cudaMemcpyDeviceToHost),"egst get"); else v=h0[i]; return v; }
static void eg_st_rnd(float*dp,size_t n,float mu,float sd){
    float*h=(float*)malloc(n*4); for(size_t i=0;i<n;i++) h[i]=mu+sd*rnd_normal();
    ck(cudaMemcpy(dp,h,n*4,cudaMemcpyHostToDevice),"egst rnd"); free(h); }
static int eg_selftest(void){
    Cfg sv=gC; int s8=g_fp8gemm,s16=g_fp16gemm,sg=g_ng_gate,ok=1;
    g_fp8gemm=0; g_fp16gemm=0; gC.D=64; gC.T=16; gC.V=50; gC.L=2;
    static char ev[][24]={"NGRAM_HEADS=2","NGRAM_ORDERS=2,3","NGRAM_DIM=8","NGRAM_ROWS=97",
                          "NGRAM_LAYERS=0,1","NGRAM_CONV=4","NGRAM_Q4=0"};
#ifdef _WIN32
    for(int i=0;i<7;i++) _putenv(ev[i]);
#else
    for(int i=0;i<7;i++) putenv(ev[i]);
#endif
    g_ng_gate=1; engram_setup(&gC); g_ng_gate=sg;
    int N=2*gC.T, S=g_eg_S, De=g_ng_De, D=gC.D, V=gC.V, SD=S*De, NK=S*N;
    size_t ND=(size_t)N*D, nt=(size_t)g_ng_R*De, NS=(size_t)N*SD;
    g_eg_canon=1; g_eg_canonH=(unsigned short*)malloc((size_t)V*2);
    for(int t=0;t<V;t++) g_eg_canonH[t]=(unsigned short)((t*7)%V);
    g_eg_canonD=(unsigned short*)dmalloc((size_t)V*2);
    ck(cudaMemcpy(g_eg_canonD,g_eg_canonH,(size_t)V*2,cudaMemcpyHostToDevice),"egst canon");
    int*tok=(int*)malloc((size_t)N*4),*kc=(int*)malloc((size_t)NK*4),*kg=(int*)malloc((size_t)NK*4);
    for(int n=0;n<N;n++) tok[n]=(int)(urand()*V)%V;
    int*tokG=(int*)dmalloc((size_t)N*4); ck(cudaMemcpy(tokG,tok,(size_t)N*4,cudaMemcpyHostToDevice),"egst tok");
    float*tb=(float*)malloc(nt*4); for(size_t i=0;i<nt;i++) tb[i]=0.5f*rnd_normal();
    ck(cudaMemcpy(g_ng,tb,nt*4,cudaMemcpyHostToDevice),"egst tbl");
    /* ---- keys, gather, scatter ---- */
    for(int s=0;s<S;s++) for(int n=0;n<N;n++){ int o=g_eg_ord[s/g_eg_H], *k=&kc[s*N+n];
        if(n%gC.T<o-1){ *k=-1; continue; }
        unsigned long long m=0; for(int j=0;j<o;j++) m^=(unsigned long long)g_eg_canonH[tok[n-j]]*g_eg_multH[s*4+j];
        *k=(int)(m%(unsigned long long)g_eg_primeH[s])+s*g_eg_Rp; }
    eg_ensure(N);
    KL(k_eg_key,nblk((size_t)NK,TPB),TPB)(tokG,g_eg_canonD,g_ngkey,N,gC.T,S,g_eg_Rp);
    KL(k_eg_gather,nblk(NS,TPB),TPB)(g_ng,g_ngkey,g_ngbuf,N,S,De); kchk("egst gather",-1);
    ck(cudaMemcpy(kg,g_ngkey,(size_t)NK*4,cudaMemcpyDeviceToHost),"egst key");
    int km=0; for(int i=0;i<NK;i++) km+=kg[i]!=kc[i];
    printf("  engram keys            %d/%d mismatch  %s\n",km,NK,km?"FAIL":"OK"); ok&=!km;
    float*bc=(float*)malloc(NS*4),*bg=(float*)malloc(NS*4),*db=(float*)malloc(NS*4),*Gc=(float*)calloc(nt,4),*Gg=(float*)malloc(nt*4);
    for(size_t i=0;i<NS;i++){ int n=(int)(i/SD), r=(int)(i%SD), s=r/De, e=r%De, k=kc[s*N+n];
        bc[i]=k>=0?tb[(size_t)k*De+e]:0.0f; db[i]=rnd_normal(); if(k>=0) Gc[(size_t)k*De+e]+=db[i]; }
    ck(cudaMemcpy(bg,g_ngbuf,NS*4,cudaMemcpyDeviceToHost),"egst buf");
    ok&=check("engram gather",relerr(bg,bc,NS),0.0f);
    ck(cudaMemcpy(g_ngdbuf,db,NS*4,cudaMemcpyHostToDevice),"egst dbuf"); eg_scatter(N);
    ck(cudaMemcpy(Gg,g_ngG,nt*4,cudaMemcpyDeviceToHost),"egst G");
    ok&=check("engram scatter",relerr(Gg,Gc,nt),1e-6f);
    /* ---- finite differences ---- */
    GA*a=(GA*)calloc(1,sizeof(GA)); float*ain[2]={(float*)dmalloc(ND*4),(float*)dmalloc(ND*4)};
    a->a_in=ain; a->dout=(float*)dmalloc(ND*4); a->N=N;
    size_t nw=(size_t)SD*D;
    for(int p=0;p<g_eg_np;p++){ EgP*P=&g_egp[p];
        eg_st_rnd(P->Wv,nw,0,0.3f); eg_st_rnd(P->Wk,nw,0,0.3f); eg_st_rnd(P->nw,D,1,0.1f); eg_st_rnd(P->K,(size_t)D*4,0,0.3f); }
    float*h0=(float*)malloc(ND*4),*w=(float*)malloc(ND*4);
    for(size_t i=0;i<ND;i++){ h0[i]=rnd_normal(); w[i]=rnd_normal(); }
    for(int gt=1;gt>=0;gt--){                       /* gated, then ungated path */
    g_eg_gate=gt; eg_st_loss(a,tokG,h0,w,N);
    ck(cudaMemcpy(a->dout,w,ND*4,cudaMemcpyHostToDevice),"egst dout");
    engram_backward(a,tokG,N,1); engram_backward(a,tokG,N,0);
    float an[64],nu[64]; int nc=0;
    for(int c=0;c<6*2*4+6+6;c++){
        float*dp,*gp; size_t i;
        if(c<48){ EgP*P=&g_egp[c/24]; int t=(c%24)/6;
            if(!gt&&t==1) t=0;
            size_t n= t<2?nw:t==2?(size_t)D:(size_t)D*4; i=(size_t)(urand()*n)%n;
            dp= t==0?P->Wv:t==1?P->Wk:t==2?P->nw:P->K; gp= t==0?P->WvG:t==1?P->WkG:t==2?P->nwG:P->KG; }
        else if(c<54){ int k; do k=kc[(int)(urand()*NK)%NK]; while(k<0); i=(size_t)k*De+(size_t)(urand()*De)%De; dp=g_ng; gp=g_ngG; }
        else { i=(size_t)(urand()*ND)%ND; dp=0; gp=a->dout; }
        ck(cudaMemcpy(&an[nc],gp+i,4,cudaMemcpyDeviceToHost),"egst grad");
        float x=eg_st_get(dp,h0,i), e=1e-2f;
        eg_st_set(dp,h0,i,x+e); double Lp=eg_st_loss(a,tokG,h0,w,N);
        eg_st_set(dp,h0,i,x-e); double Lm=eg_st_loss(a,tokG,h0,w,N);
        eg_st_set(dp,h0,i,x); nu[nc++]=(float)((Lp-Lm)/(2.0*e)); }
    ok&=check(gt?"engram FD grads gate":"engram FD grads nogate",relerr(an,nu,nc),2e-2f);
    { float ea=relerr(an+48,nu+48,6), eh=relerr(an+54,nu+54,6);
      printf("  (table %.2e  dh %.2e)\n",ea,eh); } }
    g_eg_gate=0; ck(cudaGetLastError(),"egst");
    free(tok);free(kc);free(kg);free(tb);free(bc);free(bg);free(db);free(Gc);free(Gg);free(h0);free(w);
    gC=sv; g_fp8gemm=s8; g_fp16gemm=s16;
    return ok;
}

/* ---- cudalm18 X1 extras (all env-gated, off = bit-identical to 17n) ----
 * CUDALM_CANON=A|C|AC: Canon layers (Physics of LMs 4.1) -- depthwise causal conv,
 *   4 taps incl. the current token, residual, zero-init: y[n]=x[n]+sum_j K[j][d]x[n-j].
 *   A = after rmsnorm1 (feeds QKV), C = after rmsnorm2 (feeds router + experts).
 *   Taps stop at the sequence start and, with CUDALM_DOCMASK, the document start.
 * CUDALM_SHARED_F=Fs: dense always-on SwiGLU expert next to the routed ones (MoE only),
 *   W1s/W3s [D,Fs] 0.02-normal, W2t [D,Fs] zero-init so step 0 is a no-op.
 * Float weights, AdamW (wd 0 canon / 0.1 shared), sidecar <optpath>.x1. */
static int g_canA=0,g_canC=0,g_shF=0,g_docmask=0;
static float *g_xP=0,*g_xG=0,*g_xpart=0; static bf16 *g_xM=0,*g_xV=0; static size_t g_xn=0,g_xcn=0;
static int *g_ds=0,g_dsN=0;
#define XCAN(B,l,c) ((B)+((size_t)(l)*(g_canA+g_canC)+((c)&&g_canA))*4*gC.D)
#define XSH(B,l,k) ((B)+g_xcn+((size_t)(l)*3+(k))*gC.D*g_shF)
#define CAN_NCH 64
#ifndef CUDALM_CPU
__global__ void k_canon_f(const float*x,const float*K,float*y,const int*ds,int N,int T,int D){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(size_t)N*D) return;
    int n=(int)(i/D), d=(int)(i-(size_t)n*D), p=n%T, lo=ds?ds[n]:0; float s=x[i];
    #pragma unroll
    for(int j=0;j<4;j++) if(p-j>=lo) s+=K[j*D+d]*x[i-(size_t)j*D];
    y[i]=s; }
__global__ void k_canon_bx(const float*dy,const float*K,float*dx,const int*ds,int N,int T,int D){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(size_t)N*D) return;
    int n=(int)(i/D), d=(int)(i-(size_t)n*D), p=n%T; float s=dy[i];
    #pragma unroll
    for(int j=0;j<4;j++) if(p+j<T && p>=(ds?ds[n+j]:0)) s+=K[j*D+d]*dy[i+(size_t)j*D];
    dx[i]=s; }
/* dK partials: block (32 channels x 8 row lanes) over one CAN_NCH-th of the rows */
__global__ void k_canon_bk(const float*dy,const float*x,float*part,const int*ds,int N,int T,int D){
    __shared__ float sh[4][8][33];
    int tx=threadIdx.x&31, ty=threadIdx.x>>5, d=blockIdx.x*32+tx, c=blockIdx.y;
    int r0=(int)((long long)N*c/CAN_NCH), r1=(int)((long long)N*(c+1)/CAN_NCH);
    float a0=0,a1=0,a2=0,a3=0;
    if(d<D) for(int n=r0+ty;n<r1;n+=8){ int p=n%T, lo=ds?ds[n]:0; size_t i=(size_t)n*D+d; float g=dy[i];
        a0+=g*x[i]; if(p-1>=lo) a1+=g*x[i-D]; if(p-2>=lo) a2+=g*x[i-2*(size_t)D]; if(p-3>=lo) a3+=g*x[i-3*(size_t)D]; }
    sh[0][ty][tx]=a0; sh[1][ty][tx]=a1; sh[2][ty][tx]=a2; sh[3][ty][tx]=a3;
    __syncthreads();
    if(ty<4 && d<D){ float s=0; for(int k=0;k<8;k++) s+=sh[ty][k][tx]; part[((size_t)c*4+ty)*D+d]=s; } }
__global__ void k_canon_bk2(const float*part,float*dK,int D){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=4*D) return;
    float s=0; for(int c=0;c<CAN_NCH;c++) s+=part[(size_t)c*4*D+i]; dK[i]+=s; }
#else
__global__ void k_canon_f(const float*x,const float*K,float*y,const int*ds,int N,int T,int D){
    #pragma omp parallel for schedule(static)
    for(int n=0;n<N;n++){ int p=n%T, lo=ds?ds[n]:0; const float*xr=x+(size_t)n*D; float*yr=y+(size_t)n*D;
        for(int d=0;d<D;d++){ float s=xr[d]; for(int j=0;j<4;j++) if(p-j>=lo) s+=K[j*D+d]*xr[(ptrdiff_t)d-(ptrdiff_t)j*D]; yr[d]=s; } } }
__global__ void k_canon_bx(const float*dy,const float*K,float*dx,const int*ds,int N,int T,int D){
    #pragma omp parallel for schedule(static)
    for(int n=0;n<N;n++){ int p=n%T; const float*gr=dy+(size_t)n*D; float*o=dx+(size_t)n*D;
        for(int d=0;d<D;d++){ float s=gr[d]; for(int j=0;j<4;j++) if(p+j<T && p>=(ds?ds[n+j]:0)) s+=K[j*D+d]*gr[d+(size_t)j*D]; o[d]=s; } } }
__global__ void k_canon_bk(const float*dy,const float*x,float*part,const int*ds,int N,int T,int D){
    #pragma omp parallel for schedule(static)
    for(int c=0;c<CAN_NCH;c++){ int r0=(int)((long long)N*c/CAN_NCH), r1=(int)((long long)N*(c+1)/CAN_NCH);
        float*pc=part+(size_t)c*4*D; memset(pc,0,(size_t)4*D*4);
        for(int n=r0;n<r1;n++){ int p=n%T, lo=ds?ds[n]:0; const float*g=dy+(size_t)n*D,*xr=x+(size_t)n*D;
            for(int j=0;j<4;j++) if(p-j>=lo) for(int d=0;d<D;d++) pc[j*D+d]+=g[d]*xr[(ptrdiff_t)d-(ptrdiff_t)j*D]; } } }
__global__ void k_canon_bk2(const float*part,float*dK,int D){
    for(int i=0;i<4*D;i++){ float s=0; for(int c=0;c<CAN_NCH;c++) s+=part[(size_t)c*4*D+i]; dK[i]+=s; } }
#endif
static void x1_setup(const Cfg*c){
    const char*e;
    if(g_xP) return;
    if((e=getenv("CUDALM_DOCMASK"))&&atoi(e)) g_docmask=1;
    if((e=getenv("CUDALM_CANON"))){ g_canA=strchr(e,'A')||strchr(e,'a'); g_canC=strchr(e,'C')||strchr(e,'c'); }
    if((e=getenv("CUDALM_SHARED_F"))&&atoi(e)>0){
        if(c->n_exp<=0) printf("[x1] CUDALM_SHARED_F ignored: dense model\n");
        else if(!c->mod&&atoi(e)>c->F){ fprintf(stderr,"[x1] CUDALM_SHARED_F=%d > F=%d\n",atoi(e),c->F); exit(1); }
        else g_shF=atoi(e); }
    g_xcn=(size_t)c->L*(g_canA+g_canC)*4*c->D;
    g_xn=g_xcn+(size_t)c->L*3*c->D*g_shF;
    if(!g_xn) return;
    g_xP=(float*)dmalloc(g_xn*4); g_xG=(float*)dmalloc(g_xn*4); g_xM=(bf16*)dmalloc(g_xn*2); g_xV=(bf16*)dmalloc(g_xn*2);
    ck(cudaMemset(g_xM,0,g_xn*2),"xM0"); ck(cudaMemset(g_xV,0,g_xn*2),"xV0");
    float*h=(float*)calloc(g_xn,4); unsigned s=0x2545F491u;
    for(size_t l=0;l<(size_t)c->L&&g_shF;l++) for(int k=0;k<2;k++){
        float*w=h+g_xcn+(l*3+k)*c->D*g_shF;
        for(size_t i=0;i<(size_t)c->D*g_shF;i++){ float u1,u2;
            s^=s<<13; s^=s>>17; s^=s<<5; u1=((s>>8)+1)/16777217.0f;
            s^=s<<13; s^=s>>17; s^=s<<5; u2=(s>>8)/16777216.0f;
            w[i]=0.02f*sqrtf(-2.0f*logf(u1))*cosf(6.2831853f*u2); } }
    ck(cudaMemcpy(g_xP,h,g_xn*4,cudaMemcpyHostToDevice),"x1 init"); free(h);
    if(g_canA||g_canC) g_xpart=(float*)dmalloc((size_t)CAN_NCH*4*c->D*4);
    printf("[x1] canon %s%s, shared expert F=%d: %.2fM extra params\n",g_canA?"A":"",g_canC?"C":"",g_shF,g_xn/1e6);
}
/* per-step doc starts for the attention mask and canon taps (CUDALM_DOCMASK) */
static void x1_docstart(const int*tokG,int N){
    if(!g_docmask) return;
#ifdef CUDALM_CPU
    fprintf(stderr,"[x1] CUDALM_DOCMASK needs the GPU build\n"); exit(1);
#else
    static int chk=0;
    if(!chk){ chk=1;
        if(!(g_fp16&&g_lane&&g_fwd_tc&&g_dq_tc&&g_dkv_tc&&gC.hd%16==0&&gC.hd<=64&&gC.D%8==0&&gC.KD%8==0)){
            fprintf(stderr,"[x1] CUDALM_DOCMASK needs the tensor-core attention path (FP16, LANE, *_TC on; hd 16/32/64)\n"); exit(1); }
        printf("[x1] document-masked attention on (eot %d)\n",gC.eot); }
    if(N>g_dsN){ if(g_ds) cudaFree(g_ds); g_ds=(int*)dmalloc((size_t)N*4); g_dsN=N;
        ck(cudaMemcpyToSymbol(d_dstart,&g_ds,sizeof(int*)),"dstart sym"); }
    int B=N/gC.T; k_docstart<<<(B+63)/64,64>>>(tokG,g_ds,B,gC.T,gC.eot); kchk("docstart",-1);
#endif
}
static const int* x1_ds(void){ return g_docmask?g_ds:NULL; }
static void canon_fwd(const float*x,int l,int c,float*y,int N){
    size_t n=(size_t)N*gC.D;
    KL(k_canon_f,nblk(n,TPB),TPB)(x,XCAN(g_xP,l,c),y,x1_ds(),N,gC.T,gC.D); kchk("canon fwd",l); }
/* dy -> dx (dx must not alias dy); accumulates dK */
static void canon_bwd(const float*dy,const float*x,int l,int c,float*dx,int N){
    size_t n=(size_t)N*gC.D; int D=gC.D;
    KL(k_canon_bx,nblk(n,TPB),TPB)(dy,XCAN(g_xP,l,c),dx,x1_ds(),N,gC.T,D);
    KL(k_canon_bk,dim3((D+31)/32,CAN_NCH),256)(dy,x,g_xpart,x1_ds(),N,gC.T,D);
    KL(k_canon_bk2,nblk((size_t)4*D,TPB),TPB)(g_xpart,XCAN(g_xG,l,c),D); kchk("canon bwd",l); }
static void ga_x1(GA*a){
    if(g_canA&&!a->cna){ a->cna=(float**)malloc(sizeof(float*)*gC.L); for(int l=0;l<gC.L;l++) a->cna[l]=(float*)dmalloc((size_t)a->N*gC.D*4); }
    if(g_canC&&!a->cnc){ a->cnc=(float**)malloc(sizeof(float*)*gC.L); for(int l=0;l<gC.L;l++) a->cnc[l]=(float*)dmalloc((size_t)a->N*gC.D*4); }
}
/* shared expert: out += swiglu(fn W1s, fn W3s) W2t^T; gg/uu/acttmp are free in MoE mode */
/* ---- S2: BF16 FFN path for dense layers and the shared expert (CUDALM_BFFN, default 1) ----
 * Every GEMM of the SwiGLU FFN takes bf16 operands with fp32 accumulate (2x TF32 on
 * sm_120, no FP8 absmax/quant passes), and gate/up/act/dact/dg/du live as bf16 in the
 * existing fp32 buffers, halving the elementwise traffic. Weights are cast once per
 * layer in forward and reused by backward. f_norm/dout get bf16 scratch copies. */
#ifndef CUDALM_CPU
static hbf *g_bfn=0,*g_bfo=0; static size_t g_bfnc=0,g_bfoc=0;
static hbf **g_bfw=0;     /* per layer: w1|w3|w2 (dense) or W1s|W3s|W2t (shared), bf16 */
static hbf* bf_w(int l,const float*w1,const float*w3,const float*w2,size_t n){
    if(!g_bfw){ g_bfw=(hbf**)calloc(gC.L,sizeof(hbf*)); }
    if(!g_bfw[l]) g_bfw[l]=(hbf*)dmalloc(3*n*2);
    bf_cvt(w1,g_bfw[l],n); bf_cvt(w3,g_bfw[l]+n,n); bf_cvt(w2,g_bfw[l]+2*n,n);
    return g_bfw[l];
}
/* dense: w1,w3 [D,F], w2 [F,D]. out = swiglu(fn W1, fn W3) W2 + res */
static void bff_dense_fwd(GA*a,const float*P,const Off*rd,int l,float*out,const float*res,int N){
    int D=gC.D,F=lyF(&gC,l); size_t n=(size_t)D*F, nf=(size_t)N*F;
    hbf*W=bf_w(l,P+rd->w1[l],P+rd->w3[l],P+rd->w2[l],n), *fn=bf_slot(&g_bfn,&g_bfnc,(size_t)N*D);
    hbf *gg=(hbf*)a->gg[l], *uu=(hbf*)a->uu[l], *act=(hbf*)a->acttmp;
    bf_cvt(a->f_norm[l],fn,(size_t)N*D);
    hb_ab(fn,W,gg,N,D,F,0.f,1); hb_ab(fn,W+n,uu,N,D,F,0.f,1);
    KL(k_swig_bf,nblk(nf/2,TPB),TPB)(gg,uu,act,nf/2);
    e_cpy(out,res,(size_t)N*D); hb_ab(act,W+2*n,out,N,F,D,1.f,0);
}
static void bff_dense_bwd(GA*a,float*G,const Off*og,int l,int N){
    int D=gC.D,F=lyF(&gC,l); size_t n=(size_t)D*F, nf=(size_t)N*F;
    hbf *W=g_bfw[l], *fn=bf_slot(&g_bfn,&g_bfnc,(size_t)N*D), *dO=bf_slot(&g_bfo,&g_bfoc,(size_t)N*D);
    hbf *gg=(hbf*)a->gg[l], *uu=(hbf*)a->uu[l], *act=(hbf*)a->acttmp, *dact=(hbf*)a->dact, *dg=(hbf*)a->dg, *du=(hbf*)a->du;
    bf_cvt(a->dout,dO,(size_t)N*D); bf_cvt(a->f_norm[l],fn,(size_t)N*D);
    KL(k_swig_bf,nblk(nf/2,TPB),TPB)(gg,uu,act,nf/2);
    hb_atb(act,dO,G+og->w2[l],N,F,D);
    hb_abt(dO,W+2*n,dact,N,D,F,0.f,1);
    KL(k_swigb_bf,nblk(nf/2,TPB),TPB)(gg,uu,dact,dg,du,nf/2);
    hb_abt(dg,W,a->dfnorm,N,F,D,0.f,0); hb_abt(du,W+n,a->dfnorm,N,F,D,1.f,0);
    hb_atb(fn,dg,G+og->w1[l],N,D,F); hb_atb(fn,du,G+og->w3[l],N,D,F);
}
/* shared: W1s,W3s [D,Fs], W2t [D,Fs]. out += swiglu(fn W1s, fn W3s) W2t^T */
static void bff_shared_fwd(GA*a,int l,float*out,int N){
    int D=gC.D,Fs=g_shF; size_t n=(size_t)D*Fs, nf=(size_t)N*Fs;
    hbf*W=bf_w(l,XSH(g_xP,l,0),XSH(g_xP,l,1),XSH(g_xP,l,2),n), *fn=bf_slot(&g_bfn,&g_bfnc,(size_t)N*D);
    hbf *gg=(hbf*)a->gg[l], *uu=(hbf*)a->uu[l], *act=(hbf*)a->acttmp;
    bf_cvt(a->f_norm[l],fn,(size_t)N*D);
    hb_ab(fn,W,gg,N,D,Fs,0.f,1); hb_ab(fn,W+n,uu,N,D,Fs,0.f,1);
    KL(k_swig_bf,nblk(nf/2,TPB),TPB)(gg,uu,act,nf/2);
    hb_abt(act,W+2*n,out,N,Fs,D,1.f,0);
}
static void bff_shared_bwd(GA*a,int l,int N){
    int D=gC.D,Fs=g_shF; size_t n=(size_t)D*Fs, nf=(size_t)N*Fs;
    hbf *W=g_bfw[l], *fn=bf_slot(&g_bfn,&g_bfnc,(size_t)N*D), *dO=bf_slot(&g_bfo,&g_bfoc,(size_t)N*D);
    hbf *gg=(hbf*)a->gg[l], *uu=(hbf*)a->uu[l], *act=(hbf*)a->acttmp, *dact=(hbf*)a->dact, *dg=(hbf*)a->dg, *du=(hbf*)a->du;
    bf_cvt(a->dout,dO,(size_t)N*D); bf_cvt(a->f_norm[l],fn,(size_t)N*D);
    KL(k_swig_bf,nblk(nf/2,TPB),TPB)(gg,uu,act,nf/2);
    hb_atb(dO,act,XSH(g_xG,l,2),N,D,Fs);
    hb_ab(dO,W+2*n,dact,N,D,Fs,0.f,1);
    KL(k_swigb_bf,nblk(nf/2,TPB),TPB)(gg,uu,dact,dg,du,nf/2);
    hb_abt(dg,W,a->dfnorm,N,Fs,D,1.f,0); hb_abt(du,W+n,a->dfnorm,N,Fs,D,1.f,0);
    hb_atb(fn,dg,XSH(g_xG,l,0),N,D,Fs); hb_atb(fn,du,XSH(g_xG,l,1),N,D,Fs);
}
#else
static void bff_dense_fwd(GA*,const float*,const Off*,int,float*,const float*,int){}
static void bff_dense_bwd(GA*,float*,const Off*,int,int){}
static void bff_shared_fwd(GA*,int,float*,int){}
static void bff_shared_bwd(GA*,int,int){}
#endif
static void shared_fwd(GA*a,int l,float*out,int N){
    int D=gC.D,Fs=g_shF;
    if(g_bffn&&a->tr){ bff_shared_fwd(a,l,out,N); kchk("fwd shared",l); return; }
    f8_hold(a->f_norm[l],N,D);
    g_ab(a->f_norm[l],XSH(g_xP,l,0),a->gg[l],N,D,Fs);
    g_ab(a->f_norm[l],XSH(g_xP,l,1),a->uu[l],N,D,Fs); f8_drop();
    e_swig(a->gg[l],a->uu[l],a->acttmp,(size_t)N*Fs);
    g_abt_b(a->acttmp,XSH(g_xP,l,2),out,N,Fs,D,1.0f);                     kchk("fwd shared",l); }
static void shared_bwd(GA*a,int l,int N){
    int D=gC.D,Fs=g_shF;
    if(g_bffn&&a->tr){ bff_shared_bwd(a,l,N); kchk("bwd shared",l); return; }
    e_swig(a->gg[l],a->uu[l],a->acttmp,(size_t)N*Fs);
    g_atb(a->dout,a->acttmp,XSH(g_xG,l,2),N,D,Fs);
    g_ab(a->dout,XSH(g_xP,l,2),a->dact,N,D,Fs);
    KL(k_swiglu_bwd,nblk((size_t)N*Fs,TPB),TPB)(a->gg[l],a->uu[l],a->dact,a->dg,a->du,(size_t)N*Fs);
    g_abt_b(a->dg,XSH(g_xP,l,0),a->dfnorm,N,Fs,D,1.0f);
    g_abt_b(a->du,XSH(g_xP,l,1),a->dfnorm,N,Fs,D,1.0f);
    g_atb(a->f_norm[l],a->dg,XSH(g_xG,l,0),N,D,Fs);
    g_atb(a->f_norm[l],a->du,XSH(g_xG,l,1),N,D,Fs);                       kchk("bwd shared",l); }
static void x1_update(float lr,int t){
    if(!g_xn) return;
    float b1=0.9f,b2=0.95f,bc1=1.0f-powf(b1,(float)t),bc2=1.0f-powf(b2,(float)t);
    if(g_xcn) KL(k_adamw,nblk(g_xcn,TPB),TPB)(g_xP,g_xG,g_xM,g_xV,g_xcn,lr,b1,b2,0.0f,bc1,bc2,(const float*)0,(size_t)0,(size_t)0,0,MU_OFF,g_side_gs);
    size_t ns=g_xn-g_xcn;
    if(ns) KL(k_adamw,nblk(ns,TPB),TPB)(g_xP+g_xcn,g_xG+g_xcn,g_xM+g_xcn,g_xV+g_xcn,ns,lr,b1,b2,0.1f,bc1,bc2,(const float*)0,(size_t)0,(size_t)0,0,MU_OFF,g_side_gs);
    kchk("x1 adam",-1);
}
/* sidecar <optpath>.x1: header validates the layout; P then M,V */
static void x1_save(const char*optpath,int it){
    if(!g_xn) return;
    char path[1200]; snprintf(path,sizeof path,"%s.x1",optpath);
    char tmp[1300]; FILE*f=save_open(path,tmp,sizeof tmp); if(!f) return;
    int hdr[6]={g_canA,g_canC,g_shF,gC.L,gC.D,it};
    fwrite("X1S1",1,4,f); fwrite(hdr,4,6,f);
    t1_rw(f,g_xP,g_xn*4,1); t1_rw(f,g_xM,g_xn*2,1); t1_rw(f,g_xV,g_xn*2,1);
    save_commit(f,tmp,path);
}
static void x1_load(const char*optpath){
    if(!g_xn) return;
    char path[1200]; snprintf(path,sizeof path,"%s.x1",optpath);
    FILE*f=fopen(path,"rb"); if(!f){ printf("[x1] no %s - canon/shared start from init\n",path); return; }
    char m[4]; int hdr[6];
    if(fread(m,1,4,f)!=4||memcmp(m,"X1S1",4)||fread(hdr,4,6,f)!=6){ fclose(f); fprintf(stderr,"[x1] bad header in %s\n",path); exit(1); }
    if(hdr[0]!=g_canA||hdr[1]!=g_canC||hdr[2]!=g_shF||hdr[3]!=gC.L||hdr[4]!=gC.D){
        fprintf(stderr,"[x1] %s mismatch: file canon A%d C%d shared %d vs env A%d C%d shared %d\n",
                path,hdr[0],hdr[1],hdr[2],g_canA,g_canC,g_shF); fclose(f); exit(1); }
    int ok=t1_rw(f,g_xP,g_xn*4,0)&&t1_rw(f,g_xM,g_xn*2,0)&&t1_rw(f,g_xV,g_xn*2,0);
    fclose(f);
    if(!ok){ fprintf(stderr,"[x1] truncated %s\n",path); exit(1); }
    printf("[x1] loaded %s (iter %d)\n",path,hdr[5]);
}
/* selftest: FD grads of L=sum(w*y) through canon A/C (with a doc boundary) and the shared expert */
static double x1_st_dot(const float*dv,const float*w,size_t n){
    float*h=(float*)malloc(n*4); double s=0; ck(cudaMemcpy(h,dv,n*4,cudaMemcpyDeviceToHost),"x1st d2h");
    for(size_t i=0;i<n;i++) s+=(double)w[i]*h[i]; free(h); return s; }
static float x1_st_rd(const float*d,size_t i){ float v; ck(cudaMemcpy(&v,d+i,4,cudaMemcpyDeviceToHost),"x1st rd"); return v; }
static void x1_st_wr(float*d,size_t i,float v){ ck(cudaMemcpy(d+i,&v,4,cudaMemcpyHostToDevice),"x1st wr"); }
static int x1_selftest(void){
    Cfg sv=gC; int s8=g_fp8gemm,s16=g_fp16gemm,ok=1;
    int sA=g_canA,sC=g_canC,sF=g_shF,sdm=g_docmask,*sds=g_ds; float*sP=g_xP,*sG=g_xG,*spart=g_xpart; size_t sn=g_xn,scn=g_xcn;
    g_fp8gemm=0; g_fp16gemm=0; gC.D=64; gC.T=16; gC.L=1; gC.F=64;
    int N=2*gC.T, D=gC.D; size_t ND=(size_t)N*D;
    g_canA=g_canC=1; g_shF=32; g_docmask=1;
    g_xcn=(size_t)2*4*D; g_xn=g_xcn+(size_t)3*D*g_shF;
    g_xP=(float*)dmalloc(g_xn*4); g_xG=(float*)dmalloc(g_xn*4); g_xpart=(float*)dmalloc((size_t)CAN_NCH*4*D*4);
    eg_st_rnd(g_xP,g_xn,0,0.3f);
    int dsh[32]; for(int n=0;n<N;n++){ int p=n%gC.T; dsh[n]= n<gC.T ? (p>=6?6:0) : (p>=11?11:0); }
    g_ds=(int*)dmalloc((size_t)N*4); ck(cudaMemcpy(g_ds,dsh,(size_t)N*4,cudaMemcpyHostToDevice),"x1st ds");
    float*w=(float*)malloc(ND*4); for(size_t i=0;i<ND;i++) w[i]=rnd_normal();
    float*x=(float*)dmalloc(ND*4),*y=(float*)dmalloc(ND*4),*dy=(float*)dmalloc(ND*4),*dx=(float*)dmalloc(ND*4);
    eg_st_rnd(x,ND,0,1.0f); ck(cudaMemcpy(dy,w,ND*4,cudaMemcpyHostToDevice),"x1st w");
    for(int c=0;c<2;c++){
        float an[24],nu[24]; int nc=0;
        ck(cudaMemset(g_xG,0,g_xn*4),"x1st G0"); canon_bwd(dy,x,0,c,dx,N);
        for(int k=0;k<24;k++){
            float*dp= k<12?x:XCAN(g_xP,0,c); const float*gp= k<12?dx:XCAN(g_xG,0,c);
            size_t i= k<12 ? (size_t)(urand()*ND)%ND : (size_t)(urand()*4*D)%(4*D);
            if(k==0) i=(size_t)(gC.T+11)*D+3;              /* a doc-start row */
            an[nc]=x1_st_rd(gp,i); float v=x1_st_rd(dp,i), e=1e-2f;
            x1_st_wr(dp,i,v+e); canon_fwd(x,0,c,y,N); double Lp=x1_st_dot(y,w,ND);
            x1_st_wr(dp,i,v-e); canon_fwd(x,0,c,y,N); double Lm=x1_st_dot(y,w,ND);
            x1_st_wr(dp,i,v); nu[nc++]=(float)((Lp-Lm)/(2.0*e)); }
        ok&=check(c?"canon C FD grads (docmask)":"canon A FD grads (docmask)",relerr(an,nu,nc),1e-2f); }
    { GA*a=(GA*)calloc(1,sizeof(GA)); a->N=N; a->tr=1; size_t NF=(size_t)N*gC.F;
      float*fn=(float*)dmalloc(ND*4),*gg=(float*)dmalloc(NF*4),*uu=(float*)dmalloc(NF*4);
      a->f_norm=&fn; a->gg=&gg; a->uu=&uu; a->acttmp=(float*)dmalloc(NF*4);
      a->dout=dy; a->dact=(float*)dmalloc(NF*4); a->dg=(float*)dmalloc(NF*4); a->du=(float*)dmalloc(NF*4); a->dfnorm=dx;
      eg_st_rnd(fn,ND,0,1.0f);
      cublasSetMathMode(CB,CUBLAS_DEFAULT_MATH);              /* TF32 GEMMs swamp the FD */
      int bf0=g_bffn; g_bffn=0;
      ck(cudaMemset(y,0,ND*4),"x1st y0"); shared_fwd(a,0,y,N);
      ck(cudaMemset(g_xG,0,g_xn*4),"x1st G0"); ck(cudaMemset(dx,0,ND*4),"x1st dfn0"); shared_bwd(a,0,N);
      float an[24],nu[24]; int nc=0; size_t nw=(size_t)D*g_shF; const float*gps[24]; size_t gis[24];
      for(int k=0;k<24;k++){
          float*dp= k<6?fn:XSH(g_xP,0,(k-6)/6); const float*gp= k<6?dx:XSH(g_xG,0,(k-6)/6);
          size_t i= k<6 ? (size_t)(urand()*ND)%ND : (size_t)(urand()*nw)%nw; gps[k]=gp; gis[k]=i;
          an[nc]=x1_st_rd(gp,i); float v=x1_st_rd(dp,i), e=1e-2f;
          x1_st_wr(dp,i,v+e); ck(cudaMemset(y,0,ND*4),"y0"); shared_fwd(a,0,y,N); double Lp=x1_st_dot(y,w,ND);
          x1_st_wr(dp,i,v-e); ck(cudaMemset(y,0,ND*4),"y0"); shared_fwd(a,0,y,N); double Lm=x1_st_dot(y,w,ND);
          x1_st_wr(dp,i,v); nu[nc++]=(float)((Lp-Lm)/(2.0*e)); }
      ok&=check("shared expert FD grads",relerr(an,nu,nc),2e-2f);
      g_bffn=bf0;
      if(g_bffn){ float ab[24];
          ck(cudaMemset(y,0,ND*4),"y0"); shared_fwd(a,0,y,N);
          ck(cudaMemset(g_xG,0,g_xn*4),"G0"); ck(cudaMemset(dx,0,ND*4),"dfn0"); shared_bwd(a,0,N);
          for(int k=0;k<24;k++) ab[k]=x1_st_rd(gps[k],gis[k]);
          ok&=check("shared expert BF16 grads vs fp32",relerr(ab,an,nc),3e-2f); }
      { const char*t=getenv("CUDALM_TF32"); if(!(t&&atoi(t)==0)) cublasSetMathMode(CB,CUBLAS_TF32_TENSOR_OP_MATH); }
      cudaFree(fn); cudaFree(gg); cudaFree(uu); cudaFree(a->acttmp); cudaFree(a->dact); cudaFree(a->dg); cudaFree(a->du); free(a); }
    cudaFree(x); cudaFree(y); cudaFree(dy); cudaFree(dx); cudaFree(g_ds); cudaFree(g_xP); cudaFree(g_xG); cudaFree(g_xpart); free(w);
    g_canA=sA; g_canC=sC; g_shF=sF; g_docmask=sdm; g_ds=sds; g_xP=sP; g_xG=sG; g_xpart=spart; g_xn=sn; g_xcn=scn;
    gC=sv; g_fp8gemm=s8; g_fp16gemm=s16;
    return ok;
}

/* ---- forward: embed -> L blocks -> final norm. Logits are computed
 *      separately (chunked in training, single-row in inference). ---- */
/* CUDALM_ROUTER_PRE=1: the router reads rmsnorm(f_in - E)*an2, E = the engram injections so far, with a
 * stop-gradient through E (its grad goes to f_in only), instead of f_norm. Tests whether the router keys on
 * the engram signal (collapse sits right after the injection points). Off = untouched code path. */
#ifndef CUDALM_CPU
__global__ void k_egacc(float*A,const float*H,const float*T,size_t n){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) A[i]+=H[i]-T[i]; }
__global__ void k_rpsub(float*U,const float*X,const float*E,size_t n){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) U[i]=X[i]-E[i]; }
#else
CPU_EW(k_egacc,(float*A,const float*H,const float*T,size_t n),A[i]+=H[i]-T[i])
CPU_EW(k_rpsub,(float*U,const float*X,const float*E,size_t n),U[i]=X[i]-E[i])
#endif
static int g_rpN=0; static float *g_egc=0,*g_rptmp=0,*g_rpdx=0,*g_rpG=0,**g_rpu=0,**g_rpr=0;
static void rpre_ga(int N){
    if(N<=g_rpN) return;
    if(g_egc){ cudaFree(g_egc); cudaFree(g_rptmp); cudaFree(g_rpdx); for(int l=0;l<gC.L;l++){ cudaFree(g_rpu[l]); cudaFree(g_rpr[l]); } }
    else { g_rpu=(float**)malloc(sizeof(float*)*gC.L); g_rpr=(float**)malloc(sizeof(float*)*gC.L); g_rpG=(float*)dmalloc((size_t)gC.D*4); }
    size_t n=(size_t)N*gC.D;
    g_egc=(float*)dmalloc(n*4); g_rptmp=(float*)dmalloc(n*4); g_rpdx=(float*)dmalloc(n*4);
    for(int l=0;l<gC.L;l++){ g_rpu[l]=(float*)dmalloc(n*4); g_rpr[l]=(float*)dmalloc((size_t)N*4); }
    g_rpN=N; }
static void eg_fwd(GA*a,const int*tokG,int N,int l){
    if(!g_rpre){ engram_forward(a,tokG,N,l); return; }
    size_t n=(size_t)N*gC.D; e_cpy(g_rptmp,a->a_in[l],n); engram_forward(a,tokG,N,l);
    KL(k_egacc,nblk(n,TPB),TPB)(g_egc,a->a_in[l],g_rptmp,n); kchk("rpre egacc",l); }
static const float* rpre_x(const float*P,const Off*rd,GA*a,int l,int N){
    size_t n=(size_t)N*gC.D;
    KL(k_rpsub,nblk(n,TPB),TPB)(g_rpu[l],a->f_in[l],g_egc,n);
    e_rmsf(g_rpu[l],P,rd->an2[l],g_rptmp,g_rpr[l],N,gC.D); kchk("rpre x",l); return g_rptmp; }
/* dWr += xr^T drt; g_rpdx = d(f_in) of the router path (added to dfin by gpu_backward) */
static void rpre_bwd(const float*P,float*G,const Off*og,const Off*rd,GA*a,int l,int N){
    int D=gC.D,E=lyE(&gC,l); size_t n=(size_t)N*D;
    e_rmsf(g_rpu[l],P,rd->an2[l],g_rptmp,g_rpr[l],N,D);
    g_atb(g_rptmp,a->drt,G+og->wr[l],N,D,E);
    g_abt(a->drt,P+rd->wr[l],g_rptmp,N,E,D);
    ck(cudaMemset(g_rpdx,0,n*4),"rpdx0");
    e_rmsb(g_rpu[l],P,rd->an2[l],g_rpr[l],g_rptmp,g_rpdx,g_rpG,0,N,D); kchk("rpre bwd",l); }
static void gpu_forward(const float*P,const Off*o,GA*a,const int*tokG){
    int N=a->N,D=gC.D,F=gC.F,KD=gC.KD;
    const Off*rd=g_prd?g_prd:o;
    g_fwd_id++; g_ds_on=g_dscale&&a->tr; g_ds_step+=g_ds_on;
    KL(k_embed,nblk((size_t)N*D,TPB),TPB)(P,rd->emb,tokG,a->a_in[0],N,D); kchk("fwd embed",-1);
    x1_docstart(tokG,N); ga_x1(a);
    if(g_rpre){ rpre_ga(N); ck(cudaMemset(g_egc,0,(size_t)N*D*4),"egc0"); }
    if(g_ngram_on&&!g_ng_layer) ngram_forward(a,tokG,N);
    if(g_eg_on) eg_fwd(a,tokG,N,0);
    for(int l=0;l<gC.L;l++){
        if(g_pscratch) qlm_render_layer(&QQ,(float*)P,l);
        if(g_ngram_on&&l&&l==g_ng_layer) ngram_forward(a,tokG,N);
        if(g_eg_on&&l) eg_fwd(a,tokG,N,l);
        e_rmsf(a->a_in[l],P,rd->an1[l],g_canA?a->cna[l]:a->a_norm[l],a->r1[l],N,D); kchk("fwd rmsnorm1",l);
        if(g_canA) canon_fwd(a->cna[l],l,0,a->a_norm[l],N);
        f8_hold(a->a_norm[l],N,D);
        g_ab(a->a_norm[l],P+rd->wq[l],a->qh[l],N,D,D);
        g_ab(a->a_norm[l],P+rd->wk[l],a->kh[l],N,D,KD);
        g_ab(a->a_norm[l],P+rd->wv[l],a->vh[l],N,D,KD); f8_drop();          kchk("fwd qkv",l);
        if(l&&vres_on()){ size_t nv=(size_t)N*KD; e_add(a->vh[l],a->vh[0],nv); KL(k_scale,nblk(nv,TPB),TPB)(a->vh[l],nv,0.5f); kchk("fwd vres",l); }
        if(g_fp16&&g_ropeh){        /* q's fp32 copy is never read again; k's is (decoder prefill) */
            KL(k_rope_h,nblk((size_t)N*gC.H*gC.half,TPB),TPB)(a->qh[l],a->qh16[l],gRC,gRS,N,gC.H,gC.hd,gC.half,gC.T,0);
            KL(k_rope_h,nblk((size_t)N*gC.KV*gC.half,TPB),TPB)(a->kh[l],a->kh16[l],gRC,gRS,N,gC.KV,gC.hd,gC.half,gC.T,1);
            e_f2h(a->vh[l],a->vh16[l],(size_t)N*gC.KD,1.0f); g_h16rdy=1;
        } else { e_rope(a->qh[l],gC.H,N,0); e_rope(a->kh[l],gC.KV,N,0); } kchk("fwd rope",l);
        e_attn(a,N,l);                                                     kchk("fwd attn",l);
        const float*ao=a->att[l];
        if(g_attg){ g_ab(a->a_norm[l],g_Wg+(size_t)l*D*D,a->agz[l],N,D,D);
            KL(k_agate_f,nblk((size_t)N*D,TPB),TPB)(a->att[l],a->agz[l],a->attg,(size_t)N*D); ao=a->attg; }
        g_ab_res(ao,P+rd->wo[l],a->f_in[l],a->a_in[l],N,D,D);               kchk("fwd wo+res",l);
        e_rmsf(a->f_in[l],P,rd->an2[l],g_canC?a->cnc[l]:a->f_norm[l],a->r2[l],N,D); kchk("fwd rmsnorm2",l);
        if(g_canC) canon_fwd(a->cnc[l],l,1,a->f_norm[l],N);
        float*out=(l+1<gC.L)?a->a_in[l+1]:a->x_final;
        if(lyT(&gC,l)){
            g_ovl_fn=g_moe_ovl&&g_shF?shared_fwd:NULL;
            moe_forward(P,o,a,l,out,N);                                    kchk("fwd moe",l);
            if(g_shF&&!g_sh_done) shared_fwd(a,l,out,N);
            g_sh_done=0; g_ovl_fn=NULL;
        } else if(g_bffn&&a->tr){ bff_dense_fwd(a,P,rd,l,out,a->f_in[l],N);       kchk("fwd w2+res",l);
        } else { int F=lyF(&gC,l);
            f8_hold(a->f_norm[l],N,D);
            g_ab(a->f_norm[l],P+rd->w1[l],a->gg[l],N,D,F);
            g_ab(a->f_norm[l],P+rd->w3[l],a->uu[l],N,D,F); f8_drop();
            e_swig(a->gg[l],a->uu[l],a->acttmp,(size_t)N*F);               kchk("fwd swiglu",l);
            g_ab_res(a->acttmp,P+rd->w2[l],out,a->f_in[l],N,F,D);          kchk("fwd w2+res",l);
        }
        layer_sync();
    }
    e_rmsf(a->x_final,P,rd->nf,a->fn,a->fnr,N,D);                           kchk("fwd final norm",-1);
}

/* ---- chunked lm_head + CE + its backward (dfn, dEmb). Row-chunking keeps
 *      the N*V logit buffer bounded; each row is independent so this is
 *      numerically the same as doing it in one shot. ---- */
/* Number of positions the loss is averaged over. Equals N unless prompt masking
 * is on, in which case it is the trainable count -- otherwise masking would
 * silently scale the gradients down by the masked fraction, i.e. act as an
 * unintended LR cut. Set per step by cmd_train; 0 means "use N". */
static int g_loss_denom=0;
/* CUDALM_TOKW=1: per-token loss weights from <corpus>/train.w (int8, weight = q/32, aligned with train.bin) */
static float*g_wG=NULL,*g_wGk=NULL,g_negp=0.05f,g_opdb=0.f; static int g_tokw=0,g_opdauto=1;   /* 2: q<=0 = teacher log p*8 (OPD) */
static void gpu_head_bwd(const float*P,float*G,const Off*o,GA*a,const int*tgtG){
    int N=a->N,D=gC.D,V=gC.V;
    int denom = g_loss_denom>0 ? g_loss_denom : N;
    /* Row compaction. A masked row (tgt<0) yields loss 0, a zero dlogits row and
     * hence a zero dfn row and no contribution to dEmb -- yet it still costs an
     * fn@embT, a V-wide softmax and two more V-wide GEMMs. Gathering the
     * trainable rows first removes all of that. The result is mathematically
     * identical (the dropped terms are exactly zero); it is not BIT-identical,
     * because regrouping rows changes which of them cuBLAS sums together for
     * dEmb. eval never sets g_nkeep, so val loss is unaffected either way. */
    const float *fnsrc=a->fn; const int *tgts=tgtG; float *dfnd=a->dfn; int rowsN=N;
    int comp = (a->keep && g_nkeep>0 && g_nkeep<N);
    if(comp){
        rowsN=g_nkeep;
        KL(k_gather,nblk((size_t)rowsN*D,TPB),TPB)(a->fn,a->keep,a->fnk,rowsN,D);
        KL(k_gather_i,nblk((size_t)rowsN,TPB),TPB)(tgtG,a->keep,a->tgtk,rowsN);
        if(g_wG) KL(k_gather,nblk((size_t)rowsN,TPB),TPB)(g_wG,a->keep,g_wGk,rowsN,1);
        fnsrc=a->fnk; tgts=a->tgtk; dfnd=a->dfnk;
        ck(cudaMemset(a->dfn,0,(size_t)N*D*4),"zero dfn");
        /* the tail of lossb is never written now, and cmd_train sums all N */
        ck(cudaMemset(a->lossb,0,(size_t)N*4),"zero lossb");
    }
    /* emb is constant across chunks: E4M3-cast it once (plain [V,D] for lm_head,
     * transposed [D,V] for dfn), hs = [amax,deq,inv, dlogits deq] */
    int h8 = g_fp8gemm && (g_fp8mask&3)==3 && !(D&15) && !(V&15);
    static __nv_fp8_e4m3 *e8=NULL,*e8t=NULL,*q8=NULL; static float*hs=NULL;
    if(h8){
        if(!e8){ e8=(__nv_fp8_e4m3*)dmalloc((size_t)V*D); e8t=(__nv_fp8_e4m3*)dmalloc((size_t)V*D);
                 q8=(__nv_fp8_e4m3*)dmalloc((size_t)g_chrows*V); hs=(float*)dmalloc(4*sizeof(float)); }
        const float*em=P+o->emb; size_t ne=(size_t)V*D; int gb=(int)((ne+255)/256); if(gb>1024)gb=1024;
        cudaMemsetAsync(hs,0,4*sizeof(float),0);
        KL(k_absmax,gb,256)(em,ne,&hs[0]); KL(k_mkscale,1,1)(&hs[0],&hs[1],&hs[2]);
        KL(k_quant_e4m3,nblk(ne,TPB),TPB)(em,e8,ne,&hs[2]);
        { dim3 b(32,32),gr((D+31)/32,(V+31)/32); KL(k_qtr_e4m3,gr,b)(em,e8t,V,D,&hs[2]); }
        KL(k_setf,1,1)(&hs[3],1.0f/(448.0f*denom));
    }
    for(int r0=0;r0<rowsN;r0+=g_chrows){
        int rows=(rowsN-r0<g_chrows)?(rowsN-r0):g_chrows;
        int f8 = h8 && !(rows&15);
        if(f8){ __nv_fp8_e4m3*P8=f8slot(&g8A,&g8Acap,(size_t)rows*D);
            cudaMemsetAsync(g_fp8sc,0,6*sizeof(float),0);
            fp8_prep(fnsrc+(size_t)r0*D,P8,0,rows,D,4);
            if(g_h16) g_f8dt=CUDA_R_16F;
            fp8_matmul(P8,e8,&g_fp8sc[4],&hs[1],rows,V,D,&ONE,&ZERO,a->logits,V,"head_f8"); g_f8dt=CUDA_R_32F;
        } else g_abt(fnsrc+(size_t)r0*D,P+o->emb,a->logits,rows,D,V);
        kchk("head lm_head",r0);
        int hb = f8 && g_h16 && g_hbf && !g_headw8;
        const float*wr=g_wG?(comp?g_wGk:g_wG)+r0:NULL;
        if(hb) KL(k_sce_h8,rows,TPB)((const __half*)a->logits,tgts+r0,a->lossb+r0,(bf16*)a->dlogits,q8,V,denom,wr,g_negp);
        else if(f8&&g_h16) KL(k_softmax_ce,rows,TPB)((const __half*)a->logits,tgts+r0,a->lossb+r0,a->dlogits,q8,rows,V,denom,wr,g_negp);
        else KL(k_softmax_ce,rows,TPB)(a->logits,tgts+r0,a->lossb+r0,a->dlogits,f8?q8:NULL,rows,V,denom,wr,g_negp);
        kchk("head softmax_ce",r0);
#ifndef CUDALM_CPU
        if(g_kdK&&!comp){ KL(k_kd_adj,nblk((size_t)rows,TPB),TPB)(hb?NULL:a->dlogits,hb?(bf16*)a->dlogits:NULL,f8?q8:NULL,tgts+r0,
                              g_kdI+(size_t)r0*g_kdK,(const __half*)g_kdP+(size_t)r0*g_kdK,rows,V,g_kdK,denom,g_kda,g_kdrn); kchk("head kd",r0); }
#endif
        if(f8) fp8_matmul(q8,e8t,&hs[3],&hs[1],rows,D,V,&ONE,&ZERO,dfnd+(size_t)r0*D,D,"head_dfn_f8");
        else g_ab(a->dlogits,P+o->emb,dfnd+(size_t)r0*D,rows,V,D);
        kchk("head dfn",r0);
        if(hb){ static bf16*fb=NULL; if(!fb) fb=(bf16*)dmalloc((size_t)g_chrows*D*2);
          KL(k_f2bf,nblk((size_t)rows*D,TPB),TPB)(fnsrc+(size_t)r0*D,fb,(size_t)rows*D);
          /* G[V,D] += dl^T @ fn, bf16 operands, fp32 accumulate (same flags as g_atb) */
          ckb(cublasGemmEx(CB,CUBLAS_OP_N,CUBLAS_OP_T,D,V,rows,&ONE,fb,CUDA_R_16BF,D,a->dlogits,CUDA_R_16BF,V,
                           &ONE,G+o->emb,CUDA_R_32F,D,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),"head dEmb bf16");
        } else { int sm=g_fp8mask; if(g_headw8) g_fp8mask|=4;
          g_atb(a->dlogits,fnsrc+(size_t)r0*D,G+o->emb,rows,V,D); g_fp8mask=sm; } kchk("head dEmb",r0);
    }
    if(comp){
        KL(k_scatter_rows,nblk((size_t)rowsN*D,TPB),TPB)(a->dfnk,a->keep,a->dfn,rowsN,D);
        kchk("head scatter dfn",-1);
    }
}

/* ---- backward (mirrors the verified CPU backward op-for-op) ---- */
static void qlm_fused_layer(float*P,float*G,const Off*o,const Off*og,int l);
static void gpu_backward(const float*P,float*G,const Off*o,GA*a,const int*tokG,const int*tgtG){
    int N=a->N,D=gC.D,F=gC.F,KD=gC.KD;
    const Off*og = g_fused ? &g_goff : o;
    const Off*rd = g_prd ? g_prd : o;
    g_og = g_fused ? &g_goff : NULL;
    ck(cudaMemset(G,0,(g_fused?g_gsize:param_count(&gC))*4),"zero G");
    if(g_attg) ck(cudaMemset(g_WgG,0,(size_t)gC.L*D*D*4),"zero WgG");
    if(g_xn) ck(cudaMemset(g_xG,0,g_xn*4),"zero xG");
    gpu_head_bwd(P,G,o,a,tgtG);
    ck(cudaMemset(a->dout,0,(size_t)N*D*4),"zero dout");
    e_rmsb(a->x_final,P,rd->nf,a->fnr,a->dfn,a->dout,G,og->nf,N,D);
    for(int l=gC.L-1;l>=0;l--){
        if(g_pscratch) qlm_render_layer(&QQ,(float*)P,l);
        /* the shared block still holds the PREVIOUS layer's gradients */
        if(g_fused && !getenv("CUDALM_QLM_NOALIAS") && !getenv("CUDALM_QLM_NOREUSE"))
            ck(cudaMemset(G+g_layer_g0,0,g_layer_gn*4),"zero layer G");
        if(lyT(&gC,l)){
            moe_backward(P,G,o,a,l,N);                                     kchk("bwd moe",l);
            if(g_shF) shared_bwd(a,l,N);
        } else if(g_bffn&&a->tr){ bff_dense_bwd(a,G,og,l,N);                     kchk("bwd w1/w3",l);
        } else { int F=lyF(&gC,l);
            e_swig(a->gg[l],a->uu[l],a->acttmp,(size_t)N*F);      /* recompute act */
            g_atb(a->acttmp,a->dout,G+og->w2[l],N,F,D);
            g_abt(a->dout,P+rd->w2[l],a->dact,N,D,F);                       kchk("bwd w2",l);
            KL(k_swiglu_bwd,nblk((size_t)N*F,TPB),TPB)(a->gg[l],a->uu[l],a->dact,a->dg,a->du,(size_t)N*F);
            kchk("bwd swiglu",l);
            g_abt(a->dg,P+rd->w1[l],a->dfnorm,N,F,D);
            g_abt_b(a->du,P+rd->w3[l],a->dfnorm,N,F,D,1.0f);
            g_atb(a->f_norm[l],a->dg,G+og->w1[l],N,D,F);
            g_atb(a->f_norm[l],a->du,G+og->w3[l],N,D,F);                    kchk("bwd w1/w3",l);
        }
        /* O4: the residual gradient stays in dout; both RMSNorm backwards add into it in place */
        float*dfin=a->dout;
        if(g_rpre&&lyT(&gC,l)) e_add(dfin,g_rpdx,(size_t)N*D);
        if(g_canC) canon_bwd(a->dfnorm,a->cnc[l],l,1,a->d2,N);
        e_rmsb(a->f_in[l],P,rd->an2[l],a->r2[l],g_canC?a->d2:a->dfnorm,dfin,G,og->an2[l],N,D); kchk("bwd rmsnorm2",l);
        g_abt(dfin,P+rd->wo[l],a->datt,N,D,D);
        if(g_attg){ size_t nd=(size_t)N*D;
            KL(k_mul,nblk(nd,TPB),TPB)(a->att[l],a->agz[l],a->attg,nd);
            g_atb(a->attg,dfin,G+og->wo[l],N,D,D);
            KL(k_agate_b,nblk(nd,TPB),TPB)(a->att[l],a->agz[l],a->datt,a->dz,nd); }
        else g_atb(a->att[l],dfin,G+og->wo[l],N,D,D);                    kchk("bwd wo",l);
        /* attention backward: chunked so no launch grows with T (see TDR notes).
         * lse was saved per layer in forward, so nothing is recomputed. */
        { int B=N/gC.T, T=gC.T, CH=attn_chunk(), W=lwin(l);
          /* chunk range: window offsets when windowed, absolute time otherwise */
          int RANGE = W>0 ? W : T;
          cudaMemset(a->dq,0,(size_t)N*D*4);
          cudaMemset(a->dk,0,(size_t)N*KD*4); cudaMemset(a->dv,0,(size_t)N*KD*4);
          /* Q/K/V half copies were made in forward and are still valid; only dA
           * is new here. DSCALE keeps it off fp16's denormal floor and is undone
           * at the dQ/dK/dV writes. */
#ifndef CUDALM_CPU
          if(g_fp16){ KL(k_dsc_amax,64,256)(a->datt,(size_t)N*D); KL(k_dsc_set,1,1)(); e_f2h(a->datt,a->datt16,(size_t)N*D,-1.0f); }
#endif
          /* one dot product per query -- no window walk, no memset (it writes) */
          KL(k_attn_drow,B*gC.H*T,WARP)(a->datt,a->att[l],a->drow,
              B,T,gC.H,gC.hd,gC.D, g_fp16?-1.0f:1.0f);
          kchk("bwd attn dr",l);
#ifdef CUDALM_CPU
          cpu_attn_bwd(a->qh[l],a->kh[l],a->vh[l],a->datt,a->lse[l],a->drow,a->dq,a->dk,a->dv,
                       W,B,T,gC.H,gC.KV,gC.hd,gC.D,gC.KD);
#else
          /* the lane kernel covers the whole window in one launch, so it
           * replaces the chunk loop rather than running inside it */
          int lane_dq = g_fp16 && g_lane &&
              (attn_bwd_dq_fa(a->qh16[l],a->kh16[l],a->vh16[l],a->datt16,
                  a->lse[l],a->drow,a->dq,W,B,T,gC.H,gC.KV,gC.hd,gC.D,gC.KD) ||
               (g_dq_tc && attn_bwd_dq_tc(a->qh16[l],a->kh16[l],a->vh16[l],a->datt16,
                  a->lse[l],a->drow,a->dq,W,B,T,gC.H,gC.KV,gC.hd,gC.D,gC.KD))
               || attn_bwd_dq_lane(a->qh16[l],a->kh16[l],a->vh16[l],a->datt16,
                  a->lse[l],a->dq,a->drow,W,B,T,gC.H,gC.KV,gC.hd,gC.D,gC.KD,g_hfma2));
          if(!lane_dq) for(int s0=0;s0<RANGE;s0+=CH){ int s1=(s0+CH<RANGE)?s0+CH:RANGE;
            if(g_fp16)
              KL(k_attn_bwd_dq_h,B*gC.H*T,WARP)(a->qh16[l],a->kh16[l],a->vh16[l],a->datt16,
                  a->lse[l],a->dq,a->drow,s0,s1,W,B,T,gC.H,gC.KV,gC.hd,gC.D,gC.KD);
            else
              KL(k_attn_bwd_dq,B*gC.H*T,WARP)(a->qh[l],a->kh[l],a->vh[l],a->datt,
                  a->lse[l],a->dq,a->drow,s0,s1,W,B,T,gC.H,gC.KV,gC.hd,gC.D,gC.KD); }
          kchk("bwd attn dQ",l);
          int lane_dkv = g_fp16 && g_lane &&
              (attn_bwd_dkv_fa(a->qh16[l],a->kh16[l],a->vh16[l],a->datt16,
                  a->lse[l],a->drow,a->dk,a->dv,W,B,T,gC.H,gC.KV,gC.hd,gC.D,gC.KD) ||
               (g_dkv_tc && attn_bwd_dkv_tc(a->qh16[l],a->kh16[l],a->vh16[l],a->datt16,
                  a->lse[l],a->drow,a->dk,a->dv,W,B,T,gC.H,gC.KV,gC.hd,gC.D,gC.KD))
               || attn_bwd_dkv_lane(a->qh16[l],a->kh16[l],a->vh16[l],a->datt16,
                  a->lse[l],a->drow,a->dk,a->dv,W,B,T,gC.H,gC.KV,gC.hd,gC.D,gC.KD,g_hfma2,g_dkv_split)
               /* head dim too small to split: fall back to one thread per key */
               || attn_bwd_dkv_lane(a->qh16[l],a->kh16[l],a->vh16[l],a->datt16,
                  a->lse[l],a->drow,a->dk,a->dv,W,B,T,gC.H,gC.KV,gC.hd,gC.D,gC.KD,g_hfma2,1));
          if(!lane_dkv) for(int t0=0;t0<RANGE;t0+=CH){ int t1=(t0+CH<RANGE)?t0+CH:RANGE;
            if(g_fp16)
              KL(k_attn_bwd_dkv_h,B*gC.KV*T,WARP)(a->qh16[l],a->kh16[l],a->vh16[l],a->datt16,
                  a->lse[l],a->drow,a->dk,a->dv,t0,t1,W,B,T,gC.H,gC.KV,gC.hd,gC.D,gC.KD);
            else
              KL(k_attn_bwd_dkv,B*gC.KV*T,WARP)(a->qh[l],a->kh[l],a->vh[l],a->datt,
                  a->lse[l],a->drow,a->dk,a->dv,t0,t1,W,B,T,gC.H,gC.KV,gC.hd,gC.D,gC.KD); }
#endif
          kchk("bwd attn dKV",l); }
        e_rope(a->dq,gC.H,N,1); e_rope(a->dk,gC.KV,N,1);                   kchk("bwd rope",l);
        if(vres_on()){ size_t nv=(size_t)N*KD;   /* v_l = (raw_l + v_0)/2: half to this layer's Wv, half to layer 0's */
            if(g_dv0n<N){ if(g_dv0) cudaFree(g_dv0); g_dv0=(float*)dmalloc(nv*4); g_dv0n=N; }
            if(l==gC.L-1) ck(cudaMemset(g_dv0,0,nv*4),"dv0 zero");
            if(l){ KL(k_scale,nblk(nv,TPB),TPB)(a->dv,nv,0.5f); e_add(g_dv0,a->dv,nv); }
            else e_add(a->dv,g_dv0,nv);
            kchk("bwd vres",l); }
        g_abt(a->dq,P+rd->wq[l],a->danorm,N,D,D);
        g_abt_b(a->dk,P+rd->wk[l],a->danorm,N,KD,D,1.0f);
        g_abt_b(a->dv,P+rd->wv[l],a->danorm,N,KD,D,1.0f);
        g_atb(a->a_norm[l],a->dq,G+og->wq[l],N,D,D);
        g_atb(a->a_norm[l],a->dk,G+og->wk[l],N,D,KD);
        g_atb(a->a_norm[l],a->dv,G+og->wv[l],N,D,KD);                       kchk("bwd qkv",l);
        if(g_attg){ g_abt_b(a->dz,g_Wg+(size_t)l*D*D,a->danorm,N,D,D,1.0f);
            g_atb(a->a_norm[l],a->dz,g_WgG+(size_t)l*D*D,N,D,D);           kchk("bwd agate",l); }
        if(g_canA) canon_bwd(a->danorm,a->cna[l],l,0,a->d2,N);
        e_rmsb(a->a_in[l],P,rd->an1[l],a->r1[l],g_canA?a->d2:a->danorm,a->dout,G,og->an1[l],N,D); kchk("bwd rmsnorm1",l);
        if(g_ngram_on&&l&&l==g_ng_layer) ngram_backward(a,tokG,N);
        if(g_eg_on&&l) engram_backward(a,tokG,N,l);
        layer_sync();
        /* layer l's gradients are complete here, and layer l-1 is about to
         * overwrite them -- the only point at which they can be consumed */
        if(g_fused) qlm_fused_layer((float*)P,G,g_pscratch?&g_poff:o,og,l);
    }
    if(g_ngram_on&&!g_ng_layer) ngram_backward(a,tokG,N);   /* gate adds dh into dout */
    if(g_eg_on) engram_backward(a,tokG,N,0);
    KL(k_embed_bwd,nblk((size_t)N*D,TPB),TPB)(tokG,a->dout,G,og->emb,N,D); kchk("bwd embed",-1);
    if(g_feE) fe_grad(G+og->emb);
}

/* Build the compact gradient offsets: emb, nf and mtp keep private slots
 * (they accumulate across the whole backward), every layer aliases one block.
 * Positions WITHIN a layer match the real layout, so the 12 g_atb call sites
 * only swap o-> for og->. */
static size_t calc_goffsets(const Cfg*c, const Off*o, Off*go){
    int E = c->n_exp>0 ? c->n_exp : 1;
    if(getenv("CUDALM_QLM_NOALIAS")){        /* identity map: no memory saving */
        *go = *o;
        g_layer_g0 = o->an1[0];
        g_layer_gn = (size_t)c->D + (size_t)c->D*c->D + 2*(size_t)c->D*c->KD
                   + (size_t)c->D*c->D + (size_t)c->D
                   + (c->n_exp>0 ? (size_t)c->D*c->n_exp : 0)
                   + 2*(size_t)E*c->D*c->F + (size_t)E*c->F*c->D;
        return param_count(c);
    }
    size_t p=0;
    go->emb=p; p+=(size_t)c->V*c->D;
    go->nf=p;  p+=c->D;
    for(int k=0;k<c->n_mtp;k++){ go->mtp[k]=p; p+=(size_t)c->D*c->D; }
    size_t base=p;
    size_t per = (size_t)c->D + (size_t)c->D*c->D + 2*(size_t)c->D*c->KD
               + (size_t)c->D*c->D + (size_t)c->D
               + (c->n_exp>0 ? (size_t)c->D*c->n_exp : 0)
               + 2*(size_t)E*c->D*c->F + (size_t)E*c->F*c->D;
    int reuse = getenv("CUDALM_QLM_NOREUSE") ? 0 : 1;
    for(int l=0;l<c->L;l++){
        size_t b=o->an1[l], lb = base + (reuse ? 0 : (size_t)l*per);
        go->an1[l]=lb+(o->an1[l]-b); go->wq[l]=lb+(o->wq[l]-b);
        go->wk[l] =lb+(o->wk[l] -b); go->wv[l]=lb+(o->wv[l]-b);
        go->wo[l] =lb+(o->wo[l] -b); go->an2[l]=lb+(o->an2[l]-b);
        go->wr[l] =lb+(o->wr[l] -b); go->w1[l]=lb+(o->w1[l]-b);
        go->w3[l] =lb+(o->w3[l] -b); go->w2[l]=lb+(o->w2[l]-b);
    }
    g_layer_g0=base; g_layer_gn=per;
    return base + (reuse ? per : (size_t)c->L*per);
}

/* Compact P layout for the scratch path: emb, the per-layer norms/router, the
 * final norm and MTP heads keep PRIVATE persistent slots (they are float spans
 * and are never re-rendered); every layer's 7 quantized weights ALIAS one
 * shared block, rendered from the digits just before use. */
static size_t calc_poffsets(const Cfg*c, const Off*o, Off*po){
    (void)o; int E = c->n_exp>0 ? c->n_exp : 1;
    size_t p=0; po->emb=p; p+=(size_t)c->V*c->D;
    for(int l=0;l<c->L;l++){
        po->an1[l]=p; p+=c->D;
        po->an2[l]=p; p+=c->D;
        po->wr[l]=p;  if(c->n_exp>0) p+=(size_t)c->D*c->n_exp;
    }
    po->nf=p; p+=c->D;
    for(int k=0;k<c->n_mtp;k++){ po->mtp[k]=p; p+=(size_t)c->D*c->D; }
    size_t base=p, b=0;
    size_t owq=b; b+=(size_t)c->D*c->D;   size_t owk=b; b+=(size_t)c->D*c->KD;
    size_t owv=b; b+=(size_t)c->D*c->KD;  size_t owo=b; b+=(size_t)c->D*c->D;
    size_t ow1=b; b+=(size_t)E*c->D*c->F; size_t ow3=b; b+=(size_t)E*c->D*c->F;
    size_t ow2=b; b+=(size_t)E*c->F*c->D;
    for(int l=0;l<c->L;l++){
        po->wq[l]=base+owq; po->wk[l]=base+owk; po->wv[l]=base+owv; po->wo[l]=base+owo;
        po->w1[l]=base+ow1; po->w3[l]=base+ow3; po->w2[l]=base+ow2;
    }
    g_player_p0=base; g_player_pn=b;
    return base+b;
}

/* ---- AdamW with global grad-norm clipping ---- */
static float *gPart=NULL;
/* QLM_WDF (E2): weight decay for the float spans (embeddings, norm gains,
 * router, final norm); <0 = same as the matrices. Decaying norm gains and
 * router logits toward 0 fights the loss for no regularisation benefit. */
static float g_wdf=-1.0f;
static float wdf_of(float wd){ return g_wdf<0.0f?wd:g_wdf; }
/* per-layer LR multipliers; NULL = one rate for the whole model (the default) */
static float *g_lmul=NULL; static size_t g_lay0=0,g_layper=0;

/* Integer AdamW over the quantized spans; the float spans (embeddings, norms,
 * router, MTP heads) keep the stock bf16 AdamW so only the weight
 * REPRESENTATION is under test, not a mixture of two update rules.
 * gs: device clip multiplier (S2), applied on load; NULL = G already scaled. */
static void qlm_gpu_update(Qlm*q,float*P,float*G,
                           float lr,float b1,float b2,float wd,
                           float bc1,float bc2,int t,const float*gs){
    float inv_bc1=1.0f/bc1, inv_sqrt_bc2=1.0f/sqrtf(bc2);
    qlm_step_launch(P,G,q->d,q->r,q->s,q->m8,q->m8s,q->v8,q->v8s,q->spanD,q->nspan,q->group,q->dmax,
        lr,b1,b2,wd,1.0f,gs,inv_bc1,inv_sqrt_bc2,q->lrmode,
        (unsigned long long)t,q->adam,getenv("CUDALM_QLM_DIAG")?q->statD:NULL,q->packed,0,(int)q->ngrp);
    kchk("qlm step",-1);
    /* float spans index the COMPACT moment arrays, not the full-size ones --
     * those no longer exist when qlm is on. */
    for(int i=0;i<q->nfs;i++){
        size_t off=(size_t)q->fs_off[i], n=(size_t)q->fs_n[i], cs=(size_t)q->fs_cs[i];
        KL(k_adamw,nblk(n,TPB),TPB)(P+off,G+off,q->Mc+cs,q->Vc+cs,n,
                                     lr,b1,b2,wdf_of(wd),bc1,bc2,NULL,0,0,0,MU_OFF,gs);
    }
    kchk("qlm float spans",-1);
    if(q->servo_every>0 && (t % q->servo_every)==0){
        KL(k_qlm_servo,(unsigned)q->ngrp,q->group,q->group)(q->d,q->r,q->s,q->spanD,
            q->nspan,q->group,q->dmax,q->f_hi,q->f_lo,q->packed);
        kchk("qlm servo",-1);
    }
}

/* Update exactly the parameters of layer l, whose gradients are live now.
 * Quantized spans go through k_qlm_step restricted to that layer's groups; the
 * norms and router are float spans and use stock AdamW against the compact
 * moment arrays. k_sqsum OVERWRITES its 64 partials, so each layer gets its own
 * slice and the total is summed once per step in qlm_fused_tail. */
static void qlm_fused_layer(float*P,float*G,const Off*o,const Off*og,int l){
    Qlm*q=&QQ;
    float b1=0.9f,b2=0.95f,wd=0.1f;
    { const char*e=getenv("CUDALM_QLM_WD"); if(e) wd=(float)atof(e); }
    int tt=g_fused_t;
    float bc1=1.0f-powf(b1,(float)tt), bc2=1.0f-powf(b2,(float)tt);
    float inv_bc1=1.0f/bc1, inv_sqrt_bc2=1.0f/sqrtf(bc2);
    float gscale = g_prev_gn>1.0f ? 1.0f/g_prev_gn : 1.0f;
    float lr=g_fused_lr;

    int diag_l = getenv("CUDALM_QLM_DIAG") && (g_fused_t%20)==1;
    if(diag_l) ck(cudaMemset(q->statD,0,4*8),"diag stat zero");
    KL(k_sqsum,64,TPB)(G+g_layer_g0,g_layer_gn,g_gnpart+64*l);

    int s0=7*l, gi0=q->spanH[s0].gi0, ng=0;
    for(int i=s0;i<s0+7;i++) ng+=q->spanH[i].ngrp;
    qlm_step_launch(P,G,q->d,q->r,q->s,q->m8,q->m8s,q->v8,q->v8s,q->spanD,q->nspan,q->group,q->dmax,
        lr,b1,b2,wd,gscale,NULL,inv_bc1,inv_sqrt_bc2,q->lrmode,
        (unsigned long long)tt,q->adam,diag_l?q->statD:NULL,q->packed,gi0,ng);

    size_t po[3],go2[3],nn[3]; int nf=0;
    po[nf]=o->an1[l]; go2[nf]=og->an1[l]; nn[nf]=gC.D; nf++;
    po[nf]=o->an2[l]; go2[nf]=og->an2[l]; nn[nf]=gC.D; nf++;
    if(gC.n_exp>0){ po[nf]=o->wr[l]; go2[nf]=og->wr[l]; nn[nf]=(size_t)gC.D*gC.n_exp; nf++; }
    for(int i=0;i<nf;i++){
        size_t cs=(size_t)qlm_fs_cs(q,po[i]);
        KL(k_adamw,nblk(nn[i],TPB),TPB)(P+po[i],G+go2[i],q->Mc+cs,q->Vc+cs,
                                         nn[i],lr,b1,b2,wdf_of(wd),bc1,bc2,NULL,0,0,0,MU_OFF,g_fgs);
    }
    kchk("qlm fused layer",l);
    if(diag_l){
        long long h[4]={0,0,0,0};
        ck(cudaDeviceSynchronize(),"diag sync");
        cudaMemcpy(h,q->statD,4*8,cudaMemcpyDeviceToHost);
        float ph[64]; cudaMemcpy(ph,g_gnpart+64*l,64*4,cudaMemcpyDeviceToHost);
        double ss=0; for(int i=0;i<64;i++) ss+=ph[i];
        int s0b=7*l; long long lnq=0;
        for(int i=s0b;i<s0b+7;i++) lnq+=(long long)q->spanH[i].n;
        printf("[diag] t=%d l=%d |g|=%.4g carry=%.3f%% max|idl|=%lld\n",
               g_fused_t,l,sqrt(ss),100.0*(double)h[0]/(double)lnq,h[3]);
    }
}

/* emb / nf / mtp keep private gradient slots and are updated once per step. */
static void qlm_fused_tail(float*P,float*G,const Off*o,const Off*og){
    Qlm*q=&QQ;
    float b1=0.9f,b2=0.95f,wd=0.1f;
    { const char*e=getenv("CUDALM_QLM_WD"); if(e) wd=(float)atof(e); }
    int tt=g_fused_t;
    float bc1=1.0f-powf(b1,(float)tt), bc2=1.0f-powf(b2,(float)tt);
    float lr=g_fused_lr;
    size_t po[8],go2[8],nn[8]; int nf=0;
    po[nf]=o->emb; go2[nf]=og->emb; nn[nf]=(size_t)gC.V*gC.D; nf++;
    po[nf]=o->nf;  go2[nf]=og->nf;  nn[nf]=gC.D; nf++;
    for(int k=0;k<gC.n_mtp && nf<8;k++){ po[nf]=o->mtp[k]; go2[nf]=og->mtp[k];
                                         nn[nf]=(size_t)gC.D*gC.D; nf++; }
    for(int i=0;i<nf;i++){
        size_t cs=(size_t)qlm_fs_cs(q,po[i]);
        KL(k_adamw,nblk(nn[i],TPB),TPB)(P+po[i],G+go2[i],q->Mc+cs,q->Vc+cs,
                                         nn[i],lr,b1,b2,i<2?wdf_of(wd):wd,bc1,bc2,NULL,0,0,0,MU_OFF,g_fgs);
    }
    kchk("qlm fused tail",-1);
    if(getenv("CUDALM_QLM_DIAG") && (g_fused_t%20)==1){
        long long h[4]={0,0,0,0}; cudaMemcpy(h,QQ.statD,4*8,cudaMemcpyDeviceToHost);
        printf("[diag] t=%d prev_gn=%.4g carry=%.3f%% max|idl|=%lld\n",
               g_fused_t,g_prev_gn,100.0*(double)h[0]/(double)QQ.nq,h[3]);
        cudaMemset(QQ.statD,0,4*8);
    }
    /* close out this step's grad norm; it clips the NEXT step. emb / nf / mtp live outside
     * the layer slices, so their squares are summed here (the non-fused norm has them too) */
    { static float*tp=NULL; if(!tp) tp=(float*)dmalloc((size_t)64*(2+16)*4);
      int nt=0; for(int i=0;i<nf;i++){ KL(k_sqsum,64,TPB)(G+go2[i],nn[i],tp+64*nt); nt++; }
      int nsl=64*gC.L; float*ph=(float*)malloc((size_t)(nsl+64*nt)*4);
      ck(cudaMemcpy(ph,g_gnpart,nsl*4,cudaMemcpyDeviceToHost),"fused sqsum d2h");
      ck(cudaMemcpy(ph+nsl,tp,(size_t)64*nt*4,cudaMemcpyDeviceToHost),"fused tail sqsum d2h");
      double ss=0; for(int i=0;i<nsl+64*nt;i++) ss+=ph[i]; free(ph);
      g_prev_gn=(float)sqrt(ss); }
}

/* ===================== Muon ============================================
 * Momentum -> orthogonalise via a quintic Newton-Schulz -> step. 2-D
 * tensors only; norm gains + tied embedding stay on AdamW. Ported from the
 * cudalm_ms lineage; RMS-matched so it runs at the same schedule. */
static float g_muon_mom=0.95f, g_muon_lrmul=5.0f;   /* g_muon is declared above */
static int g_muon_ns=5;
static float *gMuM=NULL;            /* fp32 momentum for Muon-managed tensors */
/* Y = alpha*A*op(B) + beta*Y over `batch` matrices with fixed strides. Row-major
 * is expressed as the transposed column-major problem, same trick as g_ab. */
static void gb_ab(const float*A,long long sA,const float*B,long long sB,
                  float*C,long long sC,int M,int K,int N,int batch,
                  float alpha,float beta,int transB){
    ckb(cublasSgemmStridedBatched(CB, transB?CUBLAS_OP_T:CUBLAS_OP_N, CUBLAS_OP_N,
        N,M,K,&alpha, B,transB?K:N,sB, A,K,sA, &beta, C,N,sC, batch),"gb_ab");
}
#ifndef CUDALM_CPU
__global__ void k_axpy(float*Y,const float*X,float alpha,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) Y[i]+=alpha*X[i];
}
#else
#define CPUK_PART 10
#include "cpu_kern.h"
#endif
/* Y[n,n] = alpha * X[m,n]^T @ X[m,n] + beta*Y, batched. Same row-major mapping
 * as g_atb. Used for the m>n branch of ns_batch. */
static void gb_atb(const float*X,long long sX,float*A,long long sA,
                   int m,int n,int batch,float alpha,float beta){
    ckb(cublasSgemmStridedBatched(CB, CUBLAS_OP_N, CUBLAS_OP_T,
        n,n,m,&alpha, X,n,sX, X,n,sX, &beta, A,n,sA, batch),"gb_atb");
}
/* One Newton-Schulz sweep over `batch` matrices of shape [m,n], either orientation.
 *
 * The iteration is  X <- a*X + (b*A + c*A^2)*X   with  A = X X^T,
 * which is the quintic from the Muon reference implementation. It is written
 * with explicit scratch rather than folded into fewer GEMMs because cuBLAS
 * cannot alias an operand with its output -- an earlier version of this that
 * passed X as both input and C was silently wrong.
 *
 * X must arrive scaled so ||X||_F <= 1, otherwise the iteration diverges.
 * Scratch: A,B are [batch,m,m]; T is [batch,m,n]. */
static void ns_batch(float*X,int m,int n,int batch,float*A,float*B,float*T){
    const float a=3.4445f,b=-4.7750f,c=2.0315f;
    long long sX=(long long)m*n;
    /* Gram side. With X = U S V^T,  p(XX^T) X == X p(X^T X)  for any polynomial
     * with no constant term -- both equal U p(S^2) S V^T -- so the two branches
     * compute the SAME thing and the only question is which Gram matrix is
     * cheaper. Always take the smaller: wk/wv are [D,KD] with D 5x KD here, and
     * forming the [D,D] Gram instead of the [KD,KD] one costs 16x the FLOPs for
     * an identical answer. Both branches are checked in the selftest. */
    int p = m<=n ? m : n;
    long long sA=(long long)p*p;
    size_t nX=(size_t)batch*m*n, nA=(size_t)batch*p*p;
    for(int it=0; it<g_muon_ns; it++){
        if(m<=n){
            gb_ab(X,sX, X,sX, A,sA, m,n,m, batch, 1.0f,0.0f, 1);  /* A = X X^T [m,m] */
            gb_ab(A,sA, A,sA, B,sA, m,m,m, batch, c,   0.0f, 0);  /* B = c A^2       */
            KL(k_axpy,nblk(nA,TPB),TPB)(B,A,b,nA);               /* B += b A        */
            gb_ab(B,sA, X,sX, T,sX, m,m,n, batch, 1.0f,0.0f, 0);  /* T = B X         */
        } else {
            gb_atb(X,sX, A,sA, m,n, batch, 1.0f,0.0f);            /* A = X^T X [n,n] */
            gb_ab(A,sA, A,sA, B,sA, n,n,n, batch, c,   0.0f, 0);  /* B = c A^2       */
            KL(k_axpy,nblk(nA,TPB),TPB)(B,A,b,nA);               /* B += b A        */
            gb_ab(X,sX, B,sA, T,sX, m,n,n, batch, 1.0f,0.0f, 0);  /* T = X B         */
        }
        KL(k_scale,nblk(nX,TPB),TPB)(X,nX,a);                    /* X = a X         */
        KL(k_axpy,nblk(nX,TPB),TPB)(X,T,1.0f,nX);                /* X += T          */
    }
}
/* ---- Muon tensor table -----------------------------------------------------
 * One entry is a GROUP: `batch` matrices of [m,n] at parameter offset `off`,
 * spaced `stride` apart. Grouping is what stops this being launch-bound -- a
 * fine-grained MoE has n_exp*3*L expert matrices and running one
 * Newton-Schulz chain each would spend its whole time in launch overhead.
 * wq/wk/wv/wo/wr batch over LAYERS (the per-layer block size is a constant
 * stride); expert matrices batch over EXPERTS within a layer, because their
 * two strides differ and cublas takes only one. */
typedef struct { size_t off; int m,n,batch; long long stride; } MuT;
static MuT g_mut[8+3*64]; static int g_nmut=0;
static size_t g_mu_np=0;
/* layer-block geometry, for telling AdamW which parameters Muon has taken */
static size_t g_mu_l0=0,g_mu_per=0,g_mu_a2r=0;
static float *gMuA=NULL,*gMuB=NULL,*gMuT=NULL,*gMuX=NULL,*gMuF=NULL;
/* momentum + Nesterov lookahead, gathering one group's strided slice of G */
#ifndef CUDALM_CPU
__global__ void k_mu_gather(const float*G,float*Mo,float*X,size_t off,long long stride,
                            int mn,size_t tot,float mom){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=tot)return;
    size_t b=i/(size_t)mn, j=i-b*(size_t)mn;
    size_t src=off+b*(size_t)stride+j;
    float g=G[src], m=mom*Mo[src]+g; Mo[src]=m;
    X[i]=g+mom*m;
}
/* scatter the orthogonalised step back, with decoupled weight decay */
__global__ void k_mu_apply(float*Pp,const float*X,size_t off,long long stride,
                           int mn,size_t tot,float lr,float scale,float wd){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=tot)return;
    size_t b=i/(size_t)mn, j=i-b*(size_t)mn;
    size_t dst=off+b*(size_t)stride+j;
    float p=Pp[dst];
    Pp[dst]=p-lr*(scale*X[i]+wd*p);
}
/* per-matrix 1/||X||_F, which is the scaling Newton-Schulz needs to converge */
__global__ void k_fro_inv(const float*X,int mn,float*inv){
    int b=blockIdx.x; const float*x=X+(size_t)b*mn;
    __shared__ float sh[32]; float s=0;
    for(int i=threadIdx.x;i<mn;i+=blockDim.x) s+=x[i]*x[i];
    for(int o=16;o>0;o>>=1) s+=__shfl_down_sync(0xffffffff,s,o);
    if((threadIdx.x&31)==0) sh[threadIdx.x>>5]=s;
    __syncthreads();
    if(threadIdx.x==0){ float t=0; int nw=(blockDim.x+31)/32; for(int i=0;i<nw;i++)t+=sh[i];
        inv[b]=1.0f/(sqrtf(t)+1e-7f); }
}
__global__ void k_scale_rows(float*X,int mn,const float*inv,size_t tot){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=tot)return;
    X[i]*=inv[i/(size_t)mn];
}
#else
#define CPUK_PART 11
#include "cpu_kern.h"
#endif
static void muon_build_table(const Cfg*c,const Off*o){
    g_nmut=0; g_mu_np=0;
    size_t per=(o->nf-o->an1[0])/(size_t)c->L;      /* exact for L>=1 */
    g_mu_l0=o->an1[0]; g_mu_per=per;
    g_mu_a2r = o->an2[0]-o->an1[0];
    int D=c->D,KD=c->KD,F=c->F,L=c->L,E=c->n_exp>0?c->n_exp:1;
    #define MUADD(O,M,N,B,S) do{ g_mut[g_nmut].off=(O); g_mut[g_nmut].m=(M); \
        g_mut[g_nmut].n=(N); g_mut[g_nmut].batch=(B); g_mut[g_nmut].stride=(S); \
        g_mu_np+=(size_t)(M)*(size_t)(N)*(size_t)(B); g_nmut++; }while(0)
    MUADD(o->wq[0],D,D, L,(long long)per);
    MUADD(o->wk[0],D,KD,L,(long long)per);
    MUADD(o->wv[0],D,KD,L,(long long)per);
    MUADD(o->wo[0],D,D, L,(long long)per);
    if(c->n_exp>0) MUADD(o->wr[0],D,c->n_exp,L,(long long)per);
    if(E==1){                                   /* dense FFN: batch over layers */
        MUADD(o->w1[0],D,F,L,(long long)per);
        MUADD(o->w3[0],D,F,L,(long long)per);
        MUADD(o->w2[0],F,D,L,(long long)per);
    } else {                                    /* MoE: batch over experts */
        for(int l=0;l<(c->pool?1:L);l++){
            MUADD(o->w1[l],D,F,E,(long long)D*F);
            MUADD(o->w3[l],D,F,E,(long long)D*F);
            MUADD(o->w2[l],F,D,E,(long long)F*D);
        }
    }
    #undef MUADD
}
static void muon_alloc(size_t np){
    /* looped: the table walks the PHYSICAL layers (repeat passes alias them and their grads already sum into them) */
    Cfg pc=gC; if(gC.lp){ pc.L=gC.lp; pc.lp=0; for(int v=gC.L-1;v>=0;v--){ int l=gC.lmap[v]; pc.lt[l]=gC.lt[v]; pc.lE[l]=gC.lE[v]; pc.lF[l]=gC.lF[v]; } }
    Off o; calc_offsets(&pc,&o);
    muon_build_table(&pc,&o);
    size_t mxA=0,mxX=0; int mxB=1;
    for(int i=0;i<g_nmut;i++){ MuT*t=&g_mut[i]; int p=t->m<t->n?t->m:t->n;
        size_t A=(size_t)t->batch*p*p, X=(size_t)t->batch*t->m*t->n;
        if(A>mxA)mxA=A; if(X>mxX)mxX=X; if(t->batch>mxB)mxB=t->batch; }
    gMuM=(float*)dmalloc(np*4); ck(cudaMemset(gMuM,0,np*4),"zero muon momentum");
    gMuA=(float*)dmalloc(mxA*4); gMuB=(float*)dmalloc(mxA*4);
    gMuT=(float*)dmalloc(mxX*4); gMuX=(float*)dmalloc(mxX*4);
    gMuF=(float*)dmalloc((size_t)mxB*4);
    printf("[muon] %d tensor groups, %.2fM of %.2fM params (%.0f%%) | momentum %.2f GB"
           " + %.0f MB scratch\n",
           g_nmut, g_mu_np/1e6, np/1e6, 100.0*g_mu_np/np, np*4/1e9,
           (2.0*mxA+2.0*mxX)*4/1e6);
}
static void gpu_muon(float*P,float*G,float lr){
    static float wd=-1; if(wd<0){ const char*e=getenv("CUDALM_WD"); wd=e?(float)atof(e):0.1f; }
    for(int i=0;i<g_nmut;i++){
        MuT*t=&g_mut[i];
        int m=t->m,n=t->n,B=t->batch,mn=m*n;
        size_t tot=(size_t)mn*(size_t)B;
        KL(k_mu_gather,nblk(tot,TPB),TPB)(G,gMuM,gMuX,t->off,t->stride,mn,tot,g_muon_mom);
        KL(k_fro_inv,B,TPB)(gMuX,mn,gMuF);
        KL(k_scale_rows,nblk(tot,TPB),TPB)(gMuX,mn,gMuF,tot);
        ns_batch(gMuX,m,n,B,gMuA,gMuB,gMuT);
        /* RMS-match to AdamW rather than the original sqrt(max(1,m/n)).
         * An orthogonal [m,n] matrix has RMS sqrt(min(m,n)/(m*n)) = 1/sqrt(max(m,n)),
         * so multiplying by 0.2*sqrt(max(m,n)) puts the update's RMS at ~0.2 --
         * about what AdamW's m/sqrt(v) delivers. That is what lets Muon run at
         * the SAME learning rate the existing WSD schedule already produces,
         * instead of needing a separately tuned rate an order of magnitude up. */
        float scale=0.2f*sqrtf((float)(m>n?m:n));
        KL(k_mu_apply,nblk(tot,TPB),TPB)(P,gMuX,t->off,t->stride,mn,tot,
                                          lr*g_muon_lrmul,scale,wd);
    }
    kchk("muon",-1);
}
/* sidecar <optpath>.mu: Muon's fp32 momentum. The .opt file holds only the AdamW moments,
 * so without it every resume of a CUDALM_MUON=1 run restarted the momentum at zero. */
static void mu_save(const char*optpath,int it,size_t np){
    if(!g_muon||g_qlm_on||!gMuM) return;
    char path[1200],tmp[1300]; snprintf(path,sizeof path,"%s.mu",optpath);
    float*h=(float*)malloc(np*4); if(!h){ if(g_sv_batch) g_sv_fail=1; fprintf(stderr,"[muon] momentum save FAILED: out of host memory\n"); return; }
    ck(cudaMemcpy(h,gMuM,np*4,cudaMemcpyDeviceToHost),"mu d2h");
    FILE*f=save_open(path,tmp,sizeof tmp); unsigned long long n=np;
    int ok=fwrite("MUM1",1,4,f)==4&&fwrite(&it,4,1,f)==1&&fwrite(&n,8,1,f)==1&&fwrite(h,4,np,f)==np;
    free(h);
    if(!ok){ save_abort(f,tmp); fprintf(stderr,"[muon] momentum save FAILED for %s (previous kept)\n",path); return; }
    if(!save_commit(f,tmp,path)) fprintf(stderr,"[muon] momentum save FAILED for %s (previous kept)\n",path); }
static void mu_load(const char*optpath,size_t np,int start_iter){
    if(!g_muon||g_qlm_on) return;
    char path[1200]; snprintf(path,sizeof path,"%s.mu",optpath);
    FILE*f=fopen(path,"rb"); if(!f){ printf("[muon] no %s - momentum starts at zero\n",path); return; }
    char m[4]; int it=0; unsigned long long n=0;
    if(fread(m,1,4,f)!=4||memcmp(m,"MUM1",4)||fread(&it,4,1,f)!=1||fread(&n,8,1,f)!=1||n>(unsigned long long)np){
        fclose(f); fprintf(stderr,"[muon] %s: bad header or size - momentum starts at zero\n",path); return; }
    if(it!=start_iter){ fclose(f);   /* e.g. a stretch trained without Muon, or a failed save */
        fprintf(stderr,"[muon] %s is from iter %d but the optimizer state is from %d - momentum starts at zero\n",path,it,start_iter); return; }
    float*h=(float*)calloc(np,4); int ok=h&&fread(h,4,(size_t)n,f)==(size_t)n; fclose(f);   /* n < np: looped model's extra norms */
    if(ok){ if(!gMuM) muon_alloc(np); ck(cudaMemcpy(gMuM,h,np*4,cudaMemcpyHostToDevice),"mu h2d");
            printf("[muon] loaded momentum %s (iter %d)\n",path,it); }
    else fprintf(stderr,"[muon] truncated %s - momentum starts at zero\n",path);
    free(h); }

/* returns 0 (nothing updated) when the grad norm is NaN/Inf: NaN>1 is false,
 * so the clip alone would pass it straight into the weights. */
static float g_gn=0;
/* Gradients of the parameters outside P/G: n-gram/engram table and projections, canon +
 * shared expert, attention gate. Their sum of squares joins the clip norm and their AdamW
 * steps take the same multiplier (g_side_gs), so one global norm covers every trainable
 * parameter. CUDALM_CLIP_SIDE=0: the norm covers G only, as before; the multiplier and the
 * NaN check still apply to them. */
#define SIDE_MAX 48
static int g_clip_side=1;
static int side_grads(const float**p,size_t*n){
    int k=0; size_t D=(size_t)gC.D;
#define SG(ptr,cnt) do{ if((ptr)&&(cnt)>0&&k<SIDE_MAX){ p[k]=(ptr); n[k]=(size_t)(cnt); k++; } }while(0)
    if(g_ngram_on||g_eg_on){
        if(g_ng_q4fold) SG(g_ngGc,(size_t)g_ng_Gt*g_ngq_grp); else SG(g_ngG,(size_t)g_ng_R*g_ng_De); }
    if(g_ngram_on){ SG(g_ngWG,(size_t)g_ng_De*D); if(g_ng_gate) SG(g_ngWkG,(size_t)g_ng_De*D); }
    if(g_eg_on){ size_t nw=(size_t)g_eg_S*g_ng_De*D;
        for(int i=0;i<g_eg_np;i++){ EgP*E=&g_egp[i]; SG(E->WvG,nw); if(g_eg_gate) SG(E->WkG,nw);
            if(g_eg_conv){ SG(E->nwG,D); SG(E->KG,D*4); } } }
    if(g_xn) SG(g_xG,g_xn);
    if(g_attg) SG(g_WgG,(size_t)gC.L*D*D);
    if(g_feE) SG(g_feG,(size_t)gC.V*g_feE+(size_t)g_feE*D);
#undef SG
    return k; }
/* fused QLM path: every update of a step (quantized layers, their float spans, the emb/nf
 * tail, the side parameters) is clipped with the PREVIOUS step's norm; call before the
 * backward, since the layers step inside it */
static void fused_side_clip(void){
    if(!g_fgs) g_fgs=(float*)dmalloc(2*4);
    KL(k_clipmul,1,1)(g_fgs,g_prev_gn); g_side_gs=g_fgs; }
/* fused path: this step's side-gradient sum of squares, folded into g_prev_gn by the caller */
static float *g_fsq=NULL;
static double fused_side_sq(void){
    const float*sp[SIDE_MAX]; size_t sn[SIDE_MAX]; int ns=side_grads(sp,sn); if(!ns) return 0.0;
    if(!g_fsq) g_fsq=(float*)dmalloc((size_t)64*SIDE_MAX*4);
    for(int i=0;i<ns;i++) KL(k_sqsum,64,TPB)(sp[i],sn[i],g_fsq+64*i);
    static float h[64*SIDE_MAX]; ck(cudaMemcpy(h,g_fsq,(size_t)64*ns*4,cudaMemcpyDeviceToHost),"side sq d2h");
    double sd=0; for(int i=0;i<64*ns;i++) sd+=h[i]; return sd; }
static int gpu_adam(float*P,float*G,bf16*M,bf16*V,size_t np,float lr,int t){
    if(!gPart) gPart=(float*)dmalloc(((size_t)64*(1+SIDE_MAX)+2)*4);
    float*gs=gPart+64*(1+SIDE_MAX);
    const float*sp[SIDE_MAX]; size_t sn[SIDE_MAX]; int ns=side_grads(sp,sn);
    KL(k_sqsum,64,TPB)(G,np,gPart);
    for(int i=0;i<ns;i++) KL(k_sqsum,64,TPB)(sp[i],sn[i],gPart+64*(1+i));
    KL(k_gnorm,1,1)(gPart,ns,g_clip_side,gs); kchk("adam sqsum",-1);
    g_side_gs=gs;
    ck(cudaMemcpy(&g_gn,gs+1,4,cudaMemcpyDeviceToHost),"gn d2h");
    if(!isfinite(g_gn)) return 0;
    /* Muon reads G itself, so it still needs the scaled copy */
    if(g_muon) KL(k_scale_d,nblk(np,TPB),TPB)(G,np,gs);
    static float b1=-1,b2,wd; const char*e;
    if(b1<0){ b1=(e=getenv("CUDALM_B1"))?(float)atof(e):0.9f; b2=(e=getenv("CUDALM_B2"))?(float)atof(e):g_qlm_on?0.95f:0.99f;
        wd=(e=getenv("CUDALM_WD"))||(e=getenv("CUDALM_QLM_WD"))?(float)atof(e):0.1f;
        printf("[adam] b1 %g b2 %g wd %g\n",b1,b2,wd); }
    float bc1=1.0f-powf(b1,(float)t),bc2=1.0f-powf(b2,(float)t);
    if(g_qlm_on){ qlm_gpu_update(&QQ,P,G,lr,b1,b2,wd,bc1,bc2,t,g_muon?NULL:gs); return 1; }
    MuSkip mu=MU_OFF;
    if(g_muon){
        if(!gMuM) muon_alloc(np);
        mu.on=1; mu.l0=g_mu_l0; mu.per=g_mu_per; mu.L=gC.lp?gC.lp:gC.L; mu.D=gC.D; mu.a2r=g_mu_a2r;
    }
    KL(k_adamw,nblk(np,TPB),TPB)(P,G,M,V,np,lr,b1,b2,wd,bc1,bc2,
                                  g_lmul,g_lay0,g_layper,gC.L,mu,g_muon?NULL:gs);
    if(g_muon) gpu_muon(P,G,lr);
    return 1;
}

/* CUDALM_LR_BOOST_LAYERS="2,5,8,11" with CUDALM_LR_BOOST=4 gives those layers
 * 4x the scheduled rate. Safety note: the multiplied rate stays BELOW the peak
 * LR the run already used successfully as long as boost*current <= peak, which
 * holds once the schedule has decayed past 1/boost of peak -- so this is not
 * asking the model to tolerate a rate it has not already survived. */
static void lr_boost_setup(const Cfg*c,const Off*o){
    const char*ls=getenv("CUDALM_LR_BOOST_LAYERS");
    float mul=1.0f;
    { const char*m=getenv("CUDALM_LR_BOOST"); if(m){ float v=(float)atof(m); if(v>0) mul=v; } }
    if(!ls||!*ls||mul==1.0f) return;
    if(c->L<2){ printf("[cudalm] LR boost ignored: needs L>=2\n"); return; }
    if(c->mod){ printf("[cudalm] LR boost ignored: per-layer layout\n"); return; }
    float*h=(float*)malloc((size_t)c->L*4);
    for(int l=0;l<c->L;l++) h[l]=1.0f;
    int nset=0;
    for(const char*p=ls;*p;){
        while(*p==' '||*p==',') p++;
        if(!*p) break;
        int l=atoi(p); if(l>=0&&l<c->L && h[l]==1.0f){ h[l]=mul; nset++; }
        while(*p && *p!=',') p++;
    }
    if(!nset){ free(h); printf("[cudalm] LR boost ignored: no valid layer index in '%s'\n",ls); return; }
    g_lay0=o->an1[0]; g_layper=o->an1[1]-o->an1[0];
    g_lmul=(float*)dmalloc((size_t)c->L*4);
    ck(cudaMemcpy(g_lmul,h,(size_t)c->L*4,cudaMemcpyHostToDevice),"lmul h2d");
    printf("[cudalm] LR boost %.2fx on %d of %d layers:",mul,nset,c->L);
    for(int l=0;l<c->L;l++) if(h[l]!=1.0f) printf(" %d",l);
    printf("\n");
    free(h);
}

/* Learning-rate schedule.
 *
 * cosine (CUDALM_SCHED=cosine): warmup, then cosine from lr down to 0.1*lr over `iters`.
 * wsd (default): warmup, hold at lr, then cool down over the last
 *   CUDALM_WSD_DECAY of the run with a 1-sqrt shape.
 *
 * Cosine requires the token budget to be fixed up front. This model gets grown
 * and its corpus extended repeatedly, and each time that happens cosine has
 * already decayed the LR toward its floor -- so new capacity arrives exactly
 * when there is no rate left to train it with. WSD holds the peak until a
 * single decay at the end, so a run can be extended or branched without the
 * schedule fighting it (MiniCPM's WSD; Hagele et al. on constant-LR + cooldown,
 * which reports parity with cosine at equal budget). */
static float lr_minf(void){ static float f=-1; if(f<0){ const char*e=getenv("CUDALM_LR_MIN"); f=e?(float)atof(e):0.1f; if(f<0) f=0; } return f; }
static float lr_at(int it,int iters,float lr,float warmup,int wsd,float dfrac){
    float base; int ita=it; it-=g_stage0; iters-=g_stage0;
    if(it<warmup) base=lr*(it+1)/warmup;
    else {
        float lo=lr_minf()*lr;
        if(!wsd){
            float span=(float)(iters>warmup?iters-warmup:1);
            base=lo+0.5f*(1+cosf(3.14159265f*(it-warmup)/span))*(lr-lo);
        } else {
            int d=(int)(iters*dfrac); if(d<1) d=1;
            int s=iters-d; if(s<(int)warmup) s=(int)warmup;
            if(it<s) base=lr;
            else { float x=(float)(it-s)/(float)(iters-s>0?iters-s:1); if(x>1.0f) x=1.0f;
                   base=lo+(lr-lo)*(1.0f-sqrtf(x)); }
        }
    }
    if(g_rewarm>0){                    /* ramp whatever the schedule asked for */
        int d=ita-g_start_iter;
        if(d<g_rewarm){ float w=(float)(d+1)/(float)g_rewarm; if(w<0.0f)w=0.0f; base*=w; }
    }
    return base;
}

/* CUDALM_LRS=1: automatic LR scaler. Every CUDALM_LRS_EVERY iters the full
 * trainable state is snapshotted to host, K iters are trained at lr*g_lrmul*m
 * for a few m, each arm is scored on fixed val batches, and g_lrmul moves to
 * the winner only if it beats m=1 by > eps. A baseline worse than the best
 * seen by 0.1 rolls back to the last good snapshot and halves g_lrmul. */
static float g_lrmul=1.0f;
typedef struct { void*d; size_t n; } LrsBuf;
static LrsBuf g_lrsb[128]; static int g_lrsn=0; static size_t g_lrs_mbn=0;
static void lrs_add(void*d,size_t n){ if(d&&n){ g_lrsb[g_lrsn].d=d; g_lrsb[g_lrsn++].n=n; } }
static size_t lrs_setup(const float*P,size_t np,bf16*M,bf16*V){
    g_lrsn=0; lrs_add((void*)P,(g_pscratch?g_psize:np)*4);   /* PSCRATCH: P is the compact layout */
    /* the trial arms run real optimizer steps: without QLM the full-size moments
     * (and Muon's momentum) are trainable state too, or every probe leaks 4 arms of
     * moment updates into the main run */
    if(!g_qlm_on){ lrs_add(M,np*2); lrs_add(V,np*2);
        if(g_muon){ if(!gMuM) muon_alloc(np); lrs_add(gMuM,np*4); } }
    if(g_qlm_on){ Qlm*q=&QQ; lrs_add(q->Mc,q->nfloat*2); lrs_add(q->Vc,q->nfloat*2); lrs_add(q->d,qd_bytes(q->nq,q->packed));
        lrs_add(q->r,q->nq); lrs_add(q->m8,q->nq); lrs_add(q->v8,q->adam?q->nq:0);
        lrs_add(q->s,q->ngrp*4); lrs_add(q->m8s,q->ngrp*4); lrs_add(q->v8s,q->adam?q->ngrp*4:0); }
    if(g_ngram_on||g_eg_on){ size_t nt=(size_t)g_ng_R*g_ng_De,nw=(size_t)g_ng_De*gC.D;
        if(g_ngram_on){ lrs_add(g_ngW,nw*4); lrs_add(g_ngWM,nw*2); lrs_add(g_ngWV,nw*2); }
        if(g_ng_q4){ size_t ng=(size_t)g_ngq_ngrp*4; lrs_add(g_ngq_d,(nt+1)/2); lrs_add(g_ngq_r,nt); lrs_add(g_ngq_m8,nt); lrs_add(g_ngq_v8,nt);
            lrs_add(g_ngq_s,ng); lrs_add(g_ngq_ms,ng); lrs_add(g_ngq_vs,ng); }
        else { lrs_add(g_ng,nt*4); lrs_add(g_ngM,nt*2); lrs_add(g_ngV,nt*2); }
        if(g_ngram_on&&g_ng_gate){ lrs_add(g_ngWk,nw*4); lrs_add(g_ngWkM,nw*2); lrs_add(g_ngWkV,nw*2); } }
    if(g_eg_on) eg_lrs();
    if(g_attg){ size_t n=(size_t)gC.L*gC.D*gC.D; lrs_add(g_Wg,n*4); lrs_add(g_WgM,n*2); lrs_add(g_WgV,n*2); }
    if(g_xn){ lrs_add(g_xP,g_xn*4); lrs_add(g_xM,g_xn*2); lrs_add(g_xV,g_xn*2); }
    g_lrs_mbn=0; if(mbias(0)){ g_lrs_mbn=(size_t)gC.L*mb_S()*gC.n_exp*4; lrs_add(g_mbiasD,g_lrs_mbn); }
    size_t t=sizeof(Qlm)+g_lrs_mbn+sizeof(float)*2; for(int i=0;i<g_lrsn;i++) t+=g_lrsb[i].n; return t;
}
static void lrs_snap(uint8_t*b){
    ck(cudaDeviceSynchronize(),"lrs sync");
    for(int i=0;i<g_lrsn;i++){ ck(cudaMemcpy(b,g_lrsb[i].d,g_lrsb[i].n,cudaMemcpyDeviceToHost),"lrs d2h"); b+=g_lrsb[i].n; }
    memcpy(b,&QQ,sizeof(Qlm)); b+=sizeof(Qlm); if(g_lrs_mbn) memcpy(b,g_mbiasH,g_lrs_mbn); b+=g_lrs_mbn;
#ifndef CUDALM_CPU
    ck(cudaMemcpyFromSymbol(b,d_opd,sizeof(float)*2),"lrs opd");   /* TOKW=2 auto-baseline accumulators */
#endif
}
static void lrs_restore(const uint8_t*b){
    ck(cudaDeviceSynchronize(),"lrs sync");
    for(int i=0;i<g_lrsn;i++){ ck(cudaMemcpy(g_lrsb[i].d,b,g_lrsb[i].n,cudaMemcpyHostToDevice),"lrs h2d"); b+=g_lrsb[i].n; }
    memcpy(&QQ,b,sizeof(Qlm)); b+=sizeof(Qlm); if(g_lrs_mbn) memcpy(g_mbiasH,b,g_lrs_mbn); b+=g_lrs_mbn;
#ifndef CUDALM_CPU
    ck(cudaMemcpyToSymbol(d_opd,b,sizeof(float)*2),"lrs opd");
#endif
    if((g_ngram_on||g_eg_on)&&g_ng_q4){ KL(k_qlm_render,(unsigned)g_ngq_ngrp,g_ngq_grp)(g_ng,g_ngq_d,g_ngq_s,g_ngq_spD,1,g_ngq_grp,1,0); kchk("lrs render",-1); }
    ck(cudaDeviceSynchronize(),"lrs sync");
}
static double gpu_loss(const float*P,const Off*o,GA*a,const int*tgtG,int N,float*perpos);

/* CUDALM_SAMPLER=perm: epoch-shuffled, non-overlapping T-token windows instead
 * of independent random offsets. With replacement sampling one "epoch" leaves
 * e^-1 = 37% of tokens unseen and spends ~26% of steps on repeats; a permutation
 * visits every window exactly once per epoch. Position is a pure function of the
 * iteration, so resume needs no extra state. CUDALM_PERM_BASE shifts position 0
 * to that iteration (a continuation of a run that sampled randomly). Each epoch
 * reshuffles with seed+epoch. */
typedef struct { uint32_t *perm; size_t nw; int T; long long epoch; uint64_t seed; } PermSampler;
static uint64_t pm_mix(uint64_t *s){ uint64_t z=(*s+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }
static void pm_shuffle(PermSampler*pm,long long ep){
    uint64_t s=pm->seed+(uint64_t)ep*0x632BE59BD9B4E019ULL;
    for(size_t i=0;i<pm->nw;i++) pm->perm[i]=(uint32_t)i;
    for(size_t i=pm->nw-1;i>0;i--){ size_t j=(size_t)(pm_mix(&s)%(uint64_t)(i+1));
        uint32_t t=pm->perm[i]; pm->perm[i]=pm->perm[j]; pm->perm[j]=t; }
    pm->epoch=ep;
}
static size_t pm_start(PermSampler*pm,size_t idx){
    long long ep=(long long)(idx/pm->nw);
    if(ep!=pm->epoch) pm_shuffle(pm,ep);
    return (size_t)pm->perm[idx%pm->nw]*(size_t)pm->T;
}

static void rope_tables(const Cfg*c){
    int mp=c->T; float*rc=(float*)malloc((size_t)mp*c->half*4),*rs=(float*)malloc((size_t)mp*c->half*4);
    for(int t=0;t<mp;t++)for(int i=0;i<c->half;i++){
        float inv=powf(c->rope_base,-(2.0f*i)/c->hd),aa=t*inv;
        rc[t*c->half+i]=cosf(aa); rs[t*c->half+i]=sinf(aa); }
    gRC=(float*)dmalloc((size_t)mp*c->half*4); gRS=(float*)dmalloc((size_t)mp*c->half*4);
    ck(cudaMemcpy(gRC,rc,(size_t)mp*c->half*4,cudaMemcpyHostToDevice),"rc");
    ck(cudaMemcpy(gRS,rs,(size_t)mp*c->half*4,cudaMemcpyHostToDevice),"rs");
    free(rc); free(rs);
}
static void cuda_init(void){
#ifdef CUDALM_CPU
    /* fp16/fp8/tensor-core/lane paths have no CPU kernels */
    g_fp8gemm=g_fp16=g_fp16gemm=g_h16=g_lane=g_fwd_tc=g_dq_tc=g_dkv_tc=g_moef8=g_tiled=g_sub=g_moe_sb=0;
#ifdef __AVX2__
    if(!cpu_has_avx2()){ fprintf(stderr,"[cpu] this CPU lacks AVX2/FMA: rebuild without -march=haswell\n"); exit(1); }
#endif
#endif
    int dev=0; ck(cudaSetDevice(dev),"setDevice");
    cudaDeviceProp pr; ck(cudaGetDeviceProperties(&pr,dev),"props");
    size_t fr=0,tt=0; cudaMemGetInfo(&fr,&tt);
    printf("[cuda] %s | sm_%d%d | %.1f/%.1f GB free | %d SMs\n",
           pr.name,pr.major,pr.minor,fr/1e9,tt/1e9,pr.multiProcessorCount);
    g_tdr=tdr_budget_s();
    { const char*e=getenv("CUDALM_ATTN_CHUNK"); if(e){ int v=atoi(e); if(v>=8) g_attn_chunk=v; } }
#if defined(_WIN32) && !defined(CUDALM_CPU)
    printf("[cuda] TDR watchdog %.0fs%s | attn chunk %d\n", g_tdr,
           g_tdr<=2.0?" (registry default - a long kernel resets the GPU)":" (raised)", g_attn_chunk);
#elif !defined(CUDALM_CPU)
    printf("[cuda] no WDDM watchdog (linux) | attn chunk %d\n", g_attn_chunk);
#endif
    ckb(cublasCreate(&CB),"cublasCreate");
    { /* TF32 tensor cores for the residual fp32 Sgemm paths (Muon Newton-Schulz,
       * any Sgemm fallback). Ampere+; a no-op on Turing (sm_75). The FP16-GEMM path
       * passes its own CUBLAS_COMPUTE_32F, so this only affects Sgemm callers.
       * CUDALM_TF32=0 forces strict IEEE fp32. */
      const char*t=getenv("CUDALM_TF32"); int on=!(t&&atoi(t)==0);
      if(on) cublasSetMathMode(CB,CUBLAS_TF32_TENSOR_OP_MATH);
      printf("[cuda] cublas math: %s\n", on?"TF32 tensor-op (Sgemm paths)":"IEEE fp32"); }
    if(g_fp8gemm){
      ckb((cublasStatus_t)cublasLtCreate(&LT),"cublasLtCreate");
      g_fp8wscap=32ull<<20; ck(cudaMalloc(&g_fp8ws,g_fp8wscap),"fp8 ws");
      ck(cudaMalloc((void**)&g_fp8sc,6*sizeof(float)),"fp8 scale buf");
      printf("[cuda] FP8 E4M3 GEMM: ON (TN dgrad matmuls g_abt/g_abt_b, per-tensor absmax, fp32 accum)\n");
    }
    ck(cudaMemcpyToSymbol(g_route_mode_d,&g_route_mode,sizeof(int)),"route mode");
    if(g_route_mode) printf("[cuda] ROUTE OVERRIDE: %s -- diagnostic only, never train with this set\n",
                            g_route_mode==1?"all tokens to expert 0":"all tokens to a pseudo-random expert");
    if(gC.hd>64){ fprintf(stderr,"head_dim %d > 64 (attention kernels use a 64-float register tile)\n",gC.hd); exit(1); }
    /* one lane owns a __half2, so an odd head dim has no valid mapping */
    if(g_fp16 && (gC.hd&1)){
        printf("[cuda] fp16 attention off: head_dim %d is odd\n",gC.hd); g_fp16=0; }
    printf("[cuda] attention operands: %s | thread map: %s\n",
           g_fp16?"fp16 (fp32 accumulate)":"fp32",
           (g_fp16&&g_lane)?"lane-per-query tile":"warp-per-output");
    if(g_muon) printf("[cuda] Muon on 2-D tensors: momentum %.2f, %d NS steps, lr x%.2f (AdamW elsewhere)\n",
                      g_muon_mom,g_muon_ns,g_muon_lrmul);
}

/* Time one fwd+bwd+adam with per-kernel sync and refuse to start a long run if
 * anything is close to the watchdog. Catches a slow kernel in ~2 seconds
 * instead of discovering it a thousand iterations in, with the GPU as the
 * casualty. CUDALM_NO_PREFLIGHT=1 skips it. */
static void preflight(float*P,float*G,const Off*o,GA*a,const int*tokG,const int*tgtG,
                      bf16*M,bf16*Vv,size_t np){
    if(getenv("CUDALM_NO_PREFLIGHT")) return;
    /* lr=0 keeps the weights fixed, but the probe still advances the moments (AdamW, QLM,
     * Muon), the router bias and g_prev_gn: snapshot and restore all of it, or every
     * session start (each resume) would add one stray optimizer step */
    size_t psz=lrs_setup(P,np,M,Vv); uint8_t*psnap=(uint8_t*)malloc(psz);
    if(!psnap){ fprintf(stderr,"[preflight] skipped: no host memory for the %.0f MB state snapshot\n",psz/1e6); return; }
    lrs_snap(psnap); float pgn=g_prev_gn;
    int save=g_debug; g_debug=1; g_maxms=0; strcpy(g_maxwhat,"(none)");
    cudaDeviceSynchronize(); g_tprev=wall();
    gpu_forward(P,o,a,tokG);
    g_fused_lr=0.0f; g_fused_t=1;            /* lr=0 so the probe does not train */
    gpu_backward(P,G,o,a,tokG,tgtG);
    if(g_fused) qlm_fused_tail(P,G,g_pscratch?&g_poff:o,&g_goff);
    else        gpu_adam(P,G,M,Vv,np,0.0f,1);
    cudaDeviceSynchronize();
    g_debug=save;
    lrs_restore(psnap); free(psnap); g_prev_gn=pgn;
    double frac=g_maxms/(g_tdr*1000.0);
    printf("[preflight] slowest kernel: %s %.1f ms (%.0f%% of the %.0fs TDR budget)\n",
           g_maxwhat,g_maxms,100.0*frac,g_tdr);
    if(frac>0.5){
        fprintf(stderr,
          "[preflight] ABORT: that is over half the watchdog budget - this run would\n"
          "  likely reset the GPU. Lower CUDALM_ATTN_CHUNK (now %d), lower CUDALM_BATCH,\n"
          "  or raise TdrDelay. Set CUDALM_NO_PREFLIGHT=1 to override.\n", g_attn_chunk);
        exit(1);
    }
    if(frac>0.25) fprintf(stderr,"[preflight] WARNING: over 25%% of the watchdog budget.\n");
}

/* ===================== model config + param stats ======================== */
/* Every env setting the code reads. rt=1: per-run/diagnostic, never saved with a model. */
typedef struct { const char*name,*def,*grp,*help; int rt; } EnvReg;
static const EnvReg g_envreg[]={
 {"CUDALM_NGRAM","1","memory","plain hashed n-gram memory (0 = off)",0},
 {"CUDALM_ENGRAM","0","memory","1 = Engram (implied by NGRAM_HEADS/ORDERS/LAYERS/CONV/CANON/LRMULT)",0},
 {"NGRAM_ROWS","655360","memory","table rows (engram: per sub-table, prime-rounded)",0},
 {"NGRAM_DIM","64","memory","row width De",0},
 {"NGRAM_Q4","1","memory","4-bit table with int moments",0},
 {"NGRAM_Q4_FOLD","=Q4","memory","compact table grads (needs 128 % De == 0)",0},
 {"NGRAM_GATE","0","memory","context gate on the memory output (+De*D per point)",0},
 {"NGRAM_ORDER","2","memory","ngram: n-gram order",0},
 {"NGRAM_TRI","0","memory","ngram: add an (order+1)-gram half table",0},
 {"NGRAM_LAYER","0","memory","ngram: injection layer (engram default point)",0},
 {"NGRAM_HEADS","1","memory","engram: hash heads per order",0},
 {"NGRAM_ORDERS","2","memory","engram: orders, e.g. 2,3 (1..4, max 4)",0},
 {"NGRAM_LAYERS","=LAYER","memory","engram: injection layers, e.g. 1,4 (max 8)",0},
 {"NGRAM_CONV","0","memory","engram: causal conv on the memory output",0},
 {"NGRAM_CANON","","memory","engram: canonical-token map file (V x uint16)",0},
 {"NGRAM_LRMULT","1","memory","engram: LR multiplier for memory params",0},
 {"CUDALM_NG_HOSTMAP","0","memory","1 = old host qsort for ngram keys (~15 ms/step)",0},
 {"CUDALM_ATTN_GATE","0","model","per-layer attention output gate (+L*D*D)",0},
 {"CUDALM_EMB_E","0","model","factorized embedding: V x E table + E x D projection (E < D; set at new)",0},
 {"CUDALM_VRES","0","model","value residual (ResFormer): layers >= 1 attend over (v_l + v_0)/2, no params",0},
 {"CUDALM_CANON","","model","Canon layers: A (after norm1), C (after norm2), AC (+L*4*D each)",0},
 {"CUDALM_SHARED_F","0","moe","shared always-on SwiGLU expert width (MoE only, <= F, +L*3*D*Fs)",0},
 {"CUDALM_D","","model","at new: override the preset width",1},
 {"CUDALM_L","","model","at new: override the preset depth",1},
 {"CUDALM_H","","model","at new: override the preset head count",1},
 {"CUDALM_KV","","model","at new: override the preset KV head count",1},
 {"CUDALM_F","","model","at new: override the preset FFN / expert width",1},
 {"CUDALM_LOOP","","model","looped layers a-bxk: run layers a..b k times, shared weights (e.g. 1-4x2)",0},
 {"CUDALM_DOCMASK","0","train","mask attention/canon across eot (pretraining; not for eot-separated chat turns)",1},
 {"CUDALM_UNIPOOL","0","model","at new: one expert pool shared by all layers",0},
 {"CUDALM_TOPK","2","moe","experts routed per token (1..8)",0},
 {"CUDALM_ROUTER_Z","0","moe","router z-loss coef (ST-MoE uses 1e-3)",0},
 {"CUDALM_CLIP_SIDE","1","optim","clip norm includes ngram/engram/canon/gate grads (0 = G only)",0},
 {"CUDALM_TOPK_L","","moe","per-layer top-k, e.g. 1,2,2,2",0},
 {"CUDALM_MOE_AUX","0.03","moe","load-balance aux loss weight",0},
 {"CUDALM_MOE_AUX_L","0","moe","per-layer aux on top of pool aux",0},
 {"CUDALM_MOE_AUX_LM","","moe","per-layer aux multipliers, e.g. 1,1,4,4 (missing = 1)",0},
 {"CUDALM_MOE_BIAS_LM","","moe","per-layer bias-step multipliers (missing = 1)",0},
 {"CUDALM_ROUTER_PRE","0","moe","router reads the residual minus engram injections (stop-grad)",0},
 {"CUDALM_MOE_BIAS","0.001","moe","aux-free balance bias step (0 = off)",0},
 {"CUDALM_MOE_BIAS_P","1","moe","proportional bias step",0},
 {"CUDALM_MOE_BIAS_SLOT","1","moe","per-slot bias",0},
 {"CUDALM_MOE_RCAP","0","moe","capacity reroute factor (>=1, train only)",0},
 {"CUDALM_MOE_SINK","0","moe","Sinkhorn selection (train only)",0},
 {"CUDALM_LR","auto","optim","peak LR (auto: by width, x6 Muon / x24 AdamW vs QLM; a new stage on a trained model gets 1/3)",0},
 {"CUDALM_WARMUP","100","optim","warmup iters",0},
 {"CUDALM_LR_MIN","0.1","optim","LR floor at the end of the decay, as a fraction of peak",0},
 {"CUDALM_B1","0.9","optim","AdamW beta1",0},
 {"CUDALM_B2","0.99","optim","AdamW beta2 (0.95 with QLM)",0},
 {"CUDALM_WD","0.1","optim","weight decay for AdamW, Muon and QLM (CUDALM_QLM_WD is the old name)",0},
 {"CUDALM_SCHED","wsd","optim","wsd | cosine",0},
 {"CUDALM_WSD_DECAY","0.15","optim","WSD decay fraction",1},
 {"CUDALM_REWARMUP","0","optim","re-warmup iters after resume",1},
 {"CUDALM_ITERS","auto","optim","pin ONE schedule over iters 0..N of the whole run (else: epochs of this corpus, as a new stage)",1},
 {"CUDALM_LR_BOOST","","optim","LR multiplier for CUDALM_LR_BOOST_LAYERS",0},
 {"CUDALM_LR_BOOST_LAYERS","","optim","layers to boost, e.g. 2,5,8",0},
 {"CUDALM_LRS","0","optim","LR scaler (periodic val-probed LR search)",0},
 {"CUDALM_LRS_EVERY","15000","optim","LRS probe interval (clamped 10k..20k)",0},
 {"CUDALM_LRS_ITERS","200","optim","LRS iters per probe",0},
 {"CUDALM_LRS_VB","8","optim","LRS val batches per probe",0},
 {"CUDALM_LRS_EPS","0.004","optim","LRS min gain to switch",0},
 {"CUDALM_MUON","1","optim","Muon on 2-D tensors (default on; off by default for QLM / TLM7 / looped models)",0},
 {"CUDALM_MUON_MOM","0.95","optim","Muon momentum",0},
 {"CUDALM_MUON_LRMUL","5","optim","Muon LR multiplier",0},
 {"CUDALM_MUON_NS","5","optim","Muon Newton-Schulz steps",0},
 {"CUDALM_QLM","0","qlm","integer weights + int AdamW (was default before 2026-10-24)",0},
 {"QLM_BITS","4","qlm","weight bits",0},
 {"QLM_GROUP","128","qlm","quant group size",0},
 {"QLM_OPT","qadam","qlm","qadam | qlion",0},
 {"QLM_LRMODE","rel","qlm","rel | abs",0},
 {"QLM_SERVO","100","qlm","scale servo interval",0},
 {"QLM_FSTAR","auto","qlm","servo target rail fraction",0},
 {"QLM_WDF","-1","qlm","float-span weight decay (<0 off)",0},
 {"CUDALM_QLM_WD","","qlm","weight decay override",0},
 {"QLM_V4","0","qlm","store v^(1/4) instead of sqrt(v)",0},
 {"QLM_STEP_OLD","0","qlm","old step kernel",0},
 {"CUDALM_QLM_FUSED","0","qlm","per-layer fused update (saves VRAM)",0},
 {"CUDALM_QLM_PSCRATCH","0","qlm","param scratch (needs FUSED)",0},
 {"CUDALM_QLM_NOALIAS","","qlm","disable grad aliasing",0},
 {"CUDALM_QLM_NOREUSE","","qlm","disable buffer reuse",0},
 {"CUDALM_BATCH","auto","train","batch size",0},
 {"CUDALM_SAVE_EVERY","auto","train","checkpoint interval (>=10)",0},
 {"CUDALM_BENCH_EVERY","10000","train","lmeval the checkpoint every N steps (0 = off) -> <model>.bench.tsv",0},
 {"CUDALM_BENCH_TASKS","all","train","bench tasks (default all: hellaswag,arc_easy,arc_challenge,piqa,lambada,arithmark)",0},
 {"CUDALM_BENCH_N","0","train","items per bench task (0 = full sets)",0},
 {"CUDALM_KEEP_EVERY","20000","train","keep *_itN snapshots every N (0 = off)",0},
 {"CUDALM_DIVSTOP","0.3","train","divergence stop threshold",0},
 {"CUDALM_MASK_PROMPT","1","train","chatml: loss on assistant tokens only",0},
 {"CUDALM_SAMPLER","perm","train","perm | rand",0},
 {"CUDALM_PERM_BASE","0","train","perm sampler start offset",1},
 {"CUDALM_KD","","train","teacher top-K cache (cudalm kdcache) for distillation",1},
 {"CUDALM_KD_ALPHA","0.5","train","weight of the teacher target in the CE target",1},
 {"CUDALM_KD_RENORM","0","train","1 = renormalise the teacher top-K to 1 (20n); 0 = raw top-K, missing mass on the true token",1},
 {"CUDALM_SEED","random","train","pins batch order / sampling",0},
 {"CUDALM_TOKW","0","train","per-token loss weights from <corpus>/train.w: 1 = weights (GRPO), 2 = q<0 is teacher log p*8 (on-policy distillation)",0},
 {"CUDALM_OPD_BIAS","auto","train","TOKW=2 baseline added to teacher log p; auto = running mean of the advantage, so it is centred; mask = drop teacher tokens (control)",0},
 {"CUDALM_NEG_PMIN","0.05","train","tokw: no push-down once a negative token p < this",0},
 {"CUDALM_STOP_MARK","1","run","chatml: stop generation at <|user|>",0},
 {"CUDALM_SERVE_SLOTS","64","run","serve: concurrent decode slots (continuous batching)",0},
 {"CUDALM_SERVE_MAXN","8","run","serve: max \"n\" samples per request",0},
 {"CUDALM_SERVE_RECOMPUTE","0","run","serve: full-context forward per token, no KV cache",0},
 {"CUDALM_VRAM_RESERVE_MB","auto","perf","VRAM kept free for the display",0},
 {"CUDALM_ATTN_CHUNK","128","perf","timesteps per attention-bwd launch (TDR guard)",0},
 {"CUDALM_FP16","1","perf","fp16 attention operands",0},
 {"CUDALM_FP8_GEMM","1","perf","FP8 GEMMs",0},
 {"CUDALM_FP8_MASK","3","perf","FP8 GEMM mask: 1 NN, 2 TN, 4 NT",0},
 {"CUDALM_FP16_GEMM","0","perf","fp16 GEMMs (trains worse)",0},
 {"CUDALM_TF32","1","perf","TF32 cuBLAS",0},
 {"CUDALM_H16","1","perf","fp16 logits in the FP8 head",0},
 {"CUDALM_HEAD_BF16","1","perf","bf16 dlogits in the FP8 head",0},
 {"CUDALM_HEAD_W8","0","perf","FP8 dEmb wgrad",0},
 {"CUDALM_RMS_WARP","1","perf","warp-per-row rmsnorm",0},
 {"CUDALM_LANE","1","perf","lane-per-query attention",0},
 {"CUDALM_TILE","0","perf","tiled attention",0},
 {"CUDALM_SUB","0","perf","sub-warp attention",0},
 {"CUDALM_HFMA2","0","perf","HFMA2 attention",0},
 {"CUDALM_DKV_SPLIT","1","perf","dKV split 1|2|4",0},
 {"CUDALM_DKV_TC","1","perf","tensor-core dKV",0},
 {"CUDALM_DQ_TC","1","perf","tensor-core dQ",0},
 {"CUDALM_FWD_TC","1","perf","tensor-core attention fwd",0},
 {"CUDALM_MOE_STASH","1","perf","stash expert fwd activations",0},
 {"CUDALM_MOE_GROUPED","7","perf","grouped MoE GEMM mask: 1 fwd, 2 dgrad, 4 wgrad",0},
 {"CUDALM_MOE_NTSER","0","perf","grouped-wgrad row cap",0},
 {"CUDALM_MOE_NOFP8","1","perf","16-aligned experts stay grouped",0},
 {"CUDALM_MOE_F8","0","perf","pad-F FP8 MoE (3 = on)",0},
 {"CUDALM_MOE_SB","0","perf","capacity-padded batched MoE",0},
 {"CUDALM_MOE_CAP","1.25","perf","capacity factor for MOE_SB",0},
 {"CUDALM_NOSYNC","0","perf","skip host syncs (default on for Linux)",0},
 {"CUDALM_TOP1_OLD","0","perf","top-1: old fp32 path instead of top-K K=1",0},
 {"CUDALM_MOE_GPERM","1","perf","MoE permutation on the GPU",0},
 {"CUDALM_MOE_MERGE","1","perf","top-2 passes as one grouped pass",0},
 {"CUDALM_FA_CP","1","perf","cp.async prefetch in FA2 attention",0},
 {"CUDALM_CH_ROWS","2048","perf","head row chunk (>= 2048)",0},
 {"CUDALM_LRS_KEEP","1","optim","LRS keeps the x1.0 arm instead of retraining it",0},
 {"CUDALM_MAX_SECONDS","","per-run","clean stop after N seconds",1},
 {"CUDALM_NO_PREFLIGHT","","per-run","skip the TDR preflight",1},
 {"CUDALM_ROUTE","0","per-run","eval diag: 1 = expert 0, 2 = pseudo-random",1},
 {"CUDALM_LRS_TEST","","per-run","LRS: allow any interval",1},
 {"CUDALM_DEBUG","0","per-run","per-kernel sync + fault attribution",1},
 {"CUDALM_PROF","0","per-run","profiling (2 = split fwd moe)",1},
 {"CUDALM_FP8_DBG","0","per-run","FP8 debug",1},
 {"CUDALM_QLM_DIAG","","per-run","QLM diagnostics",1},
 {"CUDALM_MASK_DUMP","","per-run","dump the loss mask",1},
 {"CUDALM_EVAL_AT","","per-run","eval: file of train.bin token offsets to score one window each",1},
 {"CUDALM_EVAL_OUT","","per-run","eval: output TSV for CUDALM_EVAL_AT (offset, mean loss) or CUDALM_EVAL_SPANS (doc, tokens, loss sum)",1},
 {"CUDALM_EVAL_SPANS","","per-run","eval: file of 'start len' train.bin spans to score per eot-delimited document",1},
 {"CUDALM_DATA_ROOTS","","per-run","data roots for bare corpus names (;-separated; default <exe>\\data;<exe>\\prep\\data + data_roots.txt)",1},
 {"CUDALM_PYTHON","","per-run","python used by `data fetch` (default: py -3)",1},
 {0,0,0,0,0}};
static const EnvReg* envreg_find(const char*k,size_t n){
    for(const EnvReg*r=g_envreg;r->name;r++) if(strlen(r->name)==n&&!strncmp(r->name,k,n)) return r; return 0; }
/* saved = any CUDALM_/QLM_/NGRAM_ var that is not per-run */
static int side_key(const char*kv){
    size_t n=strcspn(kv,"="); const EnvReg*r=envreg_find(kv,n);
    if(r) return !r->rt;
    return !strncmp(kv,"CUDALM_",7)||!strncmp(kv,"QLM_",4)||!strncmp(kv,"NGRAM_",6); }
static void lm_setenv(const char*k,const char*v){
#ifdef _WIN32
    _putenv_s(k,v?v:"");
#else
    if(v&&*v) setenv(k,v,1); else unsetenv(k);
#endif
}
static void model_resolve(const char*name,char*buf,size_t n){
    model_path(name,buf,n); FILE*f=fopen(buf,"rb"); if(f) fclose(f); else snprintf(buf,n,"%s",name); }
/* <model>.bin.env: written by new and by every train run; applied by every command
 * that names the model. Keys already set in the environment win. */
static void side_apply(const char*name){
    char p[1100]; model_resolve(name,p,sizeof p-4); strcat(p,".env");
    FILE*f=fopen(p,"rb"); if(!f) return; char l[1024]; int n=0;
    while(fgets(l,sizeof l,f)){ l[strcspn(l,"\r\n")]=0; char*eq=strchr(l,'='); if(!eq||eq==l||l[0]=='#') continue; *eq=0;
        const EnvReg*rg=envreg_find(l,strlen(l));
        if(rg&&rg->rt){ fprintf(stderr,"[config] %s=%s in %s ignored (per-run setting; set it in the environment for this run)\n",l,eq+1,p); continue; }
        const char*cur=getenv(l);
        if(cur){ if(strcmp(cur,eq+1)) fprintf(stderr,"[config] %s=%s from environment (model config has %s)\n",l,cur,eq+1); continue; }
        lm_setenv(l,eq+1); n++; }
    fclose(f); fprintf(stderr,"[config] %s: %d settings applied\n",p,n); }

typedef struct { int mode,R,De,order,q4,gate,tri,layer,H,no,ord[4],np,lay[8],conv,canon; } MemCfg;   /* mode 0 none, 1 ngram, 2 engram */
static const char* mem_name(int m){ return m==2?"engram":m==1?"ngram":"none"; }
static int env_i(const char*k,int d){ const char*e=getenv(k); return e&&*e?atoi(e):d; }
/* the memory config ngram_setup/engram_setup will build from the current environment */
static void mem_from_env(MemCfg*m,int L){
    memset(m,0,sizeof *m); const char*ng=getenv("CUDALM_NGRAM"),*e;
    m->gate=env_i("NGRAM_GATE",0)!=0; m->q4=env_i("NGRAM_Q4",1)!=0; m->De=env_i("NGRAM_DIM",64); m->R=env_i("NGRAM_ROWS",655360);
    m->layer=env_i("NGRAM_LAYER",0); if(m->layer<0) m->layer=0; if(m->layer>=L) m->layer=L-1;
    if(eg_wanted()){ m->mode=2; m->H=env_i("NGRAM_HEADS",1); m->ord[0]=2;
        m->no=(e=getenv("NGRAM_ORDERS"))?eg_ilist(e,m->ord,4):1;
        if((e=getenv("NGRAM_LAYERS"))) m->np=eg_ilist(e,m->lay,8); else { m->np=1; m->lay[0]=m->layer; }
        for(int i=0;i<m->np;i++){ int l=m->lay[i]; m->lay[i]=l<0?0:l>=L?L-1:l; }
        for(int i=1;i<m->np;i++) for(int j=i;j>0&&m->lay[j]<m->lay[j-1];j--){ int t=m->lay[j]; m->lay[j]=m->lay[j-1]; m->lay[j-1]=t; }
        { int n=0; for(int i=0;i<m->np;i++) if(!n||m->lay[i]!=m->lay[n-1]) m->lay[n++]=m->lay[i]; m->np=n; }
        m->conv=env_i("NGRAM_CONV",0)!=0; m->canon=getenv("NGRAM_CANON")!=0; }
    else if(!ng||atoi(ng)){ m->mode=1; m->order=env_i("NGRAM_ORDER",2); m->tri=env_i("NGRAM_TRI",0)!=0; }
}
/* the config train actually built (after ngram_setup; a resumed .ng header has decided by then) */
static void mem_live(MemCfg*m){
    memset(m,0,sizeof *m); m->De=g_ng_De; m->q4=g_ng_q4; m->layer=g_ng_layer;
    if(g_eg_on){ m->mode=2; m->H=g_eg_H; m->no=g_eg_no; memcpy(m->ord,g_eg_ord,16); m->R=g_eg_Rraw; m->np=g_eg_np;
        memcpy(m->lay,g_eg_lay,sizeof m->lay); m->conv=g_eg_conv!=0; m->gate=g_eg_gate; m->canon=g_eg_canon; }
    else if(g_ngram_on){ m->mode=1; m->R=g_ng_R; m->order=g_ng_order; m->gate=g_ng_gate; m->tri=g_ng_tri; } }
/* trained model: the .opt.ng header decides (gate/tri of plain ngram live in .opt.t1) */
static int mem_from_file(const char*mp,MemCfg*m){
    char p[1200]; snprintf(p,sizeof p,"%s.opt.ng",mp); FILE*f=fopen(p,"rb"); if(!f) return 0;
    char mg[4]; int h[24]={0}, th[9]; memset(m,0,sizeof *m); int ok=fread(mg,1,4,f)==4;
    if(ok&&!memcmp(mg,"NGR1",4)&&fread(h,4,5,f)==5){ m->mode=1; m->R=h[0]; m->De=h[1]; m->order=h[2]; m->q4=h[3];
        snprintf(p,sizeof p,"%s.opt.t1",mp); FILE*t=fopen(p,"rb");
        if(t){ if(fread(mg,1,4,t)==4&&!memcmp(mg,"T1S1",4)&&fread(th,4,9,t)==9){ m->gate=th[3]!=0; m->tri=th[6]!=0; m->layer=th[7]; } fclose(t); } }
    else if(ok&&!memcmp(mg,"NGR2",4)&&fread(h,4,24,f)==24){ m->mode=2; m->H=h[0]; m->no=h[1]; memcpy(m->ord,h+2,16);
        m->De=h[6]; m->R=h[7]; m->q4=h[8]; m->np=h[10]<8?h[10]:8; memcpy(m->lay,h+11,32); m->conv=h[19]!=0; m->canon=h[20]; m->gate=h[21]; }
    fclose(f); return m->mode!=0; }
static int t1_attg_file(const char*mp){
    char p[1200]; snprintf(p,sizeof p,"%s.opt.t1",mp); FILE*f=fopen(p,"rb"); if(!f) return -1;
    char mg[4]; int h[9], r=-1; if(fread(mg,1,4,f)==4&&!memcmp(mg,"T1S1",4)&&fread(h,4,9,f)==9) r=h[0]!=0; fclose(f); return r; }
static int topk_parse(const char*tk,const char*tl,int*kl){   /* mirrors main(); kl[64] */
    int k=2; memset(kl,0,64*4); if(tk&&*tk){ int v=atoi(tk); if(v>=1&&v<=8) k=v; }
    for(int i=0;tl&&*tl&&i<64;i++){ int v=atoi(tl); kl[i]=v; if(v>k&&v<=8) k=v; while(*tl&&*tl!=',') tl++; if(*tl) tl++; }
    return k; }

typedef struct { double bb,moe_off,attg,tbl,w,mact,tot,act; } PStats;
static void pstats(const Cfg*c,const MemCfg*m,int topk,const int*kl,int attg,PStats*s){
    memset(s,0,sizeof *s); s->bb=(double)param_count(c); double ef=3.0*c->D*c->F;
    if(c->n_exp>0){ double a=0; int PE=0;
        for(int l=0;l<c->L;l++){ int t=lyT(c,l), E=lyE(c,l), k=kl&&kl[l]>0?kl[l]:topk; if(!t) continue; if(k>E) k=E;
            if(t==2){ a+=k; PE=E; } else s->moe_off+=(E-k)*ef; }
        if(PE) s->moe_off+=(PE-(a<PE?a:PE))*ef; }
    { int fe=fe_dim(c->D); if(fe) s->bb-=(double)c->V*c->D-((double)c->V*fe+(double)fe*c->D); }
    s->attg=attg?(double)c->L*c->D*c->D:0;
    if(m->mode==1){ s->tbl=(double)m->R*m->De; s->w=(double)m->De*c->D*(1+m->gate); s->mact=(double)m->De*(1+m->tri)+s->w; }
    else if(m->mode==2&&m->De>0){ int S=m->no*m->H, al=128%m->De==0?128/m->De:1;
        s->tbl=(double)S*((m->R+al-1)/al*al)*m->De;
        s->w=m->np*((double)S*m->De*c->D*(1+m->gate)+(m->conv?5.0*c->D:0)); s->mact=(double)S*m->De+s->w; }
    s->tot=s->bb+s->attg+s->tbl+s->w; s->act=s->bb-s->moe_off+s->attg+s->mact; }
static const char*pcomma(double v){   /* exact count with thousands separators; 4 rotating buffers */
    static char b[4][32]; static int r; char*o=b[r++&3], t[32]; int n=snprintf(t,sizeof t,"%.0f",v), k=0;
    for(int i=0;i<n;i++){ if(i&&(n-i)%3==0) o[k++]=','; o[k++]=t[i]; } o[k]=0; return o; }
static void pstats_print(const Cfg*c,const MemCfg*m,int topk,const int*kl,int attg){
    PStats s; pstats(c,m,topk,kl,attg,&s);
    printf("  backbone    %9.2fM %12s",s.bb/1e6,pcomma(s.bb));
    if(c->mod){ double ef=3.0*c->D*c->F, dn=0, mo=0, po=0; int PE=0;
        for(int l=0;l<c->L;l++){ int t=lyT(c,l); if(!t) dn+=3.0*c->D*lyF(c,l); else if(t==1) mo+=lyE(c,l)*ef; else PE=lyE(c,l); }
        po=PE*ef; printf("   layout:");
        for(int l=0;l<c->L;l++){ int t=lyT(c,l); printf(t==0?" D%d":t==1?" M%d":" P%d",t?lyE(c,l):lyF(c,l)); }
        printf("  (expert F=%d, top-%d)\n  dense FFN %.2fM  MoE %.2fM  pool %.2fM  backbone-rest %.2fM",
               c->F,topk,dn/1e6,mo/1e6,po/1e6,(s.bb-dn-mo-po)/1e6); }
    else if(c->n_exp>0) printf("   %d experts%s, top-%d%s (%.2fM in experts)",c->n_exp,c->pool?" UniPool":"",topk,
                          kl&&kl[0]?" max, per-layer":"",(c->pool?1.0:c->L)*c->n_exp*3.0*c->D*c->F/1e6);
    printf("\n");
    if(attg) printf("  attn gate   %9.2fM %12s\n",s.attg/1e6,pcomma(s.attg));
    if(m->mode==1) printf("  ngram       %9.2fM %12s   table %.2fM (%d rows x %d, order %d%s, %s) + proj%s %.2fM at layer %d\n",
        (s.tbl+s.w)/1e6,pcomma(s.tbl+s.w),s.tbl/1e6,m->R,m->De,m->order,m->tri?"+tri":"",m->q4?"q4":"fp32",m->gate?"+gate":"",s.w/1e6,m->layer);
    else if(m->mode==2){ char o[32]="", ly[48]=""; for(int i=0;i<m->no&&i<4;i++) snprintf(o+strlen(o),sizeof o-strlen(o),"%s%d",i?",":"",m->ord[i]);
        for(int i=0;i<m->np;i++) snprintf(ly+strlen(ly),sizeof ly-strlen(ly),"%s%d",i?",":"",m->lay[i]);
        printf("  engram      %9.2fM %12s   table %.2fM (%d heads x orders %s = %d sub-tables x %d rows x %d, %s)\n"
               "                           + proj%s%s %.2fM at layers %s%s\n",
            (s.tbl+s.w)/1e6,pcomma(s.tbl+s.w),s.tbl/1e6,m->H,o,m->H*m->no,m->R,m->De,m->q4?"q4":"fp32",
            m->gate?"+gate":"",m->conv?"+conv":"",s.w/1e6,ly,m->canon?", canon map":""); }
    else printf("  ngram/engram off\n");
    printf("  total       %9.2fM %12s\n",s.tot/1e6,pcomma(s.tot));
    printf("  active      %9.2fM %12s / token  (backbone %s + memory %s; embedding %s included)\n",
           s.act/1e6,pcomma(s.act),pcomma(s.bb-s.moe_off+s.attg),pcomma(s.mact),pcomma(fe_dim(c->D)?((double)c->V+c->D)*fe_dim(c->D):(double)c->V*c->D)); }
/* model config: every saved env var that is set, with the memory section written from m
 * so it is always complete (a resumed model whose .ng decided still gets its full config) */
static void side_write(const char*mp,const MemCfg*m){
    static const char*memk[]={"CUDALM_NGRAM=","CUDALM_ENGRAM=","NGRAM_ROWS=","NGRAM_DIM=","NGRAM_ORDER=","NGRAM_TRI=","NGRAM_GATE=",
        "NGRAM_Q4=","NGRAM_LAYER=","NGRAM_HEADS=","NGRAM_ORDERS=","NGRAM_LAYERS=","NGRAM_CONV=",0};
    char p[1100], tmp[1200]; snprintf(p,sizeof p,"%s.env",mp); snprintf(tmp,sizeof tmp,"%s.tmp",p);
    FILE*f=fopen(tmp,"wb"); if(!f){ fprintf(stderr,"[config] cannot write %s\n",tmp); return; }
    fprintf(f,"# cudalm model config (written by new/train). KEY=VALUE; the environment overrides these.\n");
#ifdef _WIN32
    char**env=_environ;
#else
    extern char**environ; char**env=environ;
#endif
    for(;env&&*env;env++){ const char*kv=*env; int skip=!side_key(kv);
        for(int i=0;memk[i]&&!skip;i++) if(!strncmp(kv,memk[i],strlen(memk[i]))) skip=1;
        if(!strncmp(kv,"NGRAM_",6)&&(m->mode==0||(m->mode==1&&(!strncmp(kv,"NGRAM_CANON=",12)||!strncmp(kv,"NGRAM_LRMULT=",13))))) skip=1;
        if(!skip) fprintf(f,"%s\n",kv); }
    fprintf(f,"CUDALM_NGRAM=%d\nCUDALM_ENGRAM=%d\n",m->mode!=0,m->mode==2);
    if(m->mode) fprintf(f,"NGRAM_ROWS=%d\nNGRAM_DIM=%d\nNGRAM_Q4=%d\nNGRAM_GATE=%d\nNGRAM_LAYER=%d\n",m->R,m->De,m->q4,m->gate,m->layer);
    if(m->mode==1) fprintf(f,"NGRAM_ORDER=%d\nNGRAM_TRI=%d\n",m->order,m->tri);
    if(m->mode==2){ fprintf(f,"NGRAM_HEADS=%d\nNGRAM_ORDERS=",m->H); for(int i=0;i<m->no;i++) fprintf(f,"%s%d",i?",":"",m->ord[i]);
        fprintf(f,"\nNGRAM_LAYERS="); for(int i=0;i<m->np;i++) fprintf(f,"%s%d",i?",":"",m->lay[i]); fprintf(f,"\nNGRAM_CONV=%d\n",m->conv); }
    fclose(f); remove(p);
    if(rename(tmp,p)) fprintf(stderr,"[config] cannot write %s\n",p); else printf("[config] saved %s\n",p); }

/* ===================== KV-cache decode + batched GPU sampler ============== */
/* B independent sequences, one token each per step. Context-dependent ops get per-sequence state:
 * K/V cache per layer (post-rope), 4-slot rings for the canon A/C inputs, a 3*dil+1 ring for each engram conv,
 * the last 4 tokens for the engram keys, the position and the document start (CUDALM_DOCMASK). Everything else
 * (norms, GEMMs, MoE, shared expert, attention gate) is the training forward run on N=B rows. Prefill = gpu_forward
 * on the prompt, then its K/V and ring rows are copied into the slot. Legacy ngram and QLM scratch are not supported. */
#ifndef CUDALM_CPU
typedef struct { float temp,top_p; int top_k; unsigned long long seed; int step; } DSP;
typedef struct { int B,Tm,R; float **K,**V,**cA,**cC,*ey[EG_MAXP]; int *pos,*ds,*tokw,*tok,*otok; float *olp; DSP*sp;
                 int *hpos,*hds,*htokw,*htok,*hotok; float *holp; DSP*hsp; } Dec;
__global__ void k_dec_rope(float*buf,const float*cosb,const float*sinb,const int*pos,int B,int nh,int hd,int hf){
    size_t g=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(g>=(size_t)B*nh*hf) return;
    int i=(int)(g%hf); size_t r=g/hf; int h=(int)(r%nh), b=(int)(r/nh), t=pos[b];
    float*v=buf+(size_t)b*nh*hd+(size_t)h*hd; float c=cosb[t*hf+i],s=sinb[t*hf+i],x=v[i],y=v[i+hf];
    v[i]=x*c-y*s; v[i+hf]=y*c+x*s; }
__global__ void k_dec_kv(const float*k,const float*v,float*Kc,float*Vc,const int*pos,int B,int Tm,int KD){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(size_t)B*KD) return;
    int b=(int)(i/KD), j=(int)(i%KD); size_t o=((size_t)b*Tm+pos[b])*KD+j; Kc[o]=k[i]; Vc[o]=v[i]; }
/* one warp per (b,h): the training k_attn_fwd loop over cached keys, with the window and doc start */
__global__ void k_dec_attn(const float*Q,const float*Kc,const float*Vc,float*O,const int*pos,const int*ds,
                           int win,int B,int Tm,int H,int KV,int hd,int D,int KD){
    int g=blockIdx.x; if(g>=B*H) return;
    int lane=threadIdx.x, d0=lane, d1=lane+WARP, b=g/H, h=g%H, kvh=h/(H/KV), t=pos[b];
    float scale=1.0f/sqrtf((float)hd); const float*q=Q+(size_t)b*D+h*hd;
    float q0=d0<hd?q[d0]:0.f, q1=d1<hd?q[d1]:0.f, a0=0.f,a1=0.f,m=-1e30f,l=0.f;
    int lo=(win>0&&t>=win)?t-win+1:0; if(ds&&ds[b]>lo) lo=ds[b];
    for(int s=lo;s<=t;s++){
        const float*k=Kc+((size_t)b*Tm+s)*KD+kvh*hd;
        float dot=(d0<hd?q0*k[d0]:0.f)+(d1<hd?q1*k[d1]:0.f);
        for(int off=16;off>0;off>>=1) dot+=__shfl_xor_sync(0xffffffff,dot,off);
        dot*=scale; float nm=fmaxf(m,dot),corr=expf(m-nm),p=expf(dot-nm);
        const float*vv=Vc+((size_t)b*Tm+s)*KD+kvh*hd;
        l=l*corr+p; a0=a0*corr+(d0<hd?p*vv[d0]:0.f); a1=a1*corr+(d1<hd?p*vv[d1]:0.f); m=nm; }
    float inv=1.0f/l; float*o=O+(size_t)b*D+h*hd; if(d0<hd) o[d0]=a0*inv; if(d1<hd) o[d1]=a1*inv; }
/* canon: y = x + sum_j K[j] x[p-j] over taps inside [doc start, p]; ring[b][p&3] keeps the inputs */
__global__ void k_dec_canon(const float*x,const float*K,float*y,float*ring,const int*pos,const int*ds,int B,int D){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(size_t)B*D) return;
    int b=(int)(i/D), d=(int)(i%D), p=pos[b], lo=ds?ds[b]:0; float xv=x[i], s=xv+K[d]*xv; float*r=ring+(size_t)b*4*D;
    for(int j=1;j<4;j++) if(p-j>=lo) s+=K[j*D+d]*r[(size_t)((p-j)&3)*D+d];
    r[(size_t)(p&3)*D+d]=xv; y[i]=s; }
/* decoder: the same key from the slot's last tokens (tokw[b*4+3] = current), pos = position in the sequence */
__global__ void k_dec_ngkey(const int*tokw,int*key,const int*pos,int B,int R,int order,int base){
    int b=(int)(blockIdx.x*blockDim.x+threadIdx.x); if(b>=B)return;
    if(pos[b]<order-1){ key[b]=-1; return; }
    unsigned h=2166136261u;
    for(int j=order-1;j>=0;j--) h=(h^(unsigned)tokw[b*4+3-j])*16777619u;
    h^=h>>15;
    key[b]=base+(int)(h%(unsigned)R); }
__global__ void k_dec_egkey(const int*tokw,const unsigned short*cn,int*key,const int*pos,int B,int S,int Rp){
    int i=(int)(blockIdx.x*blockDim.x+threadIdx.x); if(i>=S*B) return;
    int s=i/B, b=i%B, o=c_eg_ord[s];
    if(pos[b]<o-1){ key[i]=-1; return; }
    unsigned long long m=0;
    for(int k=0;k<o;k++){ unsigned t=(unsigned)tokw[b*4+3-k]; if(cn) t=cn[t]; m^=(unsigned long long)t*c_eg_mult[s*4+k]; }
    key[i]=(int)(m%(unsigned long long)c_eg_prime[s])+s*Rp; }
__global__ void k_dec_egconv(float*h,const float*vg,const float*y,const float*K,float*ring,const int*pos,int B,int D,int dil,int R){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(size_t)B*D) return;
    int b=(int)(i/D), d=(int)(i%D), p=pos[b]; float*r=ring+(size_t)b*R*D, s=K[d*4]*y[i];
    for(int j=1;j<4;j++) if(p>=j*dil) s+=K[d*4+j]*r[(size_t)((p-j*dil)%R)*D+d];
    r[(size_t)(p%R)*D+d]=y[i]; h[i]+=vg[i]+s/(1.0f+expf(-s)); }
/* sampler (= mcpie k_sample without min_p): top_k -> top_p -> temperature; greedy if temp<=0. lp = raw log p of the pick */
#define DEC_SCAP 2048
__global__ void __launch_bounds__(1024) k_dec_sample(const float*logits,int V,const DSP*sp,int*out,float*lp){
    __shared__ float sv[DEC_SCAP]; __shared__ int si[DEC_SCAP]; __shared__ float rv[32]; __shared__ int ri[32];
    __shared__ int cnt; __shared__ float sth,ssum;
    int s=blockIdx.x,t=threadIdx.x,w=t>>5,ln=t&31; const float*lg=logits+(size_t)s*V; DSP p=sp[s];
    float mx=-INFINITY; int mi=0;
    for(int i=t;i<V;i+=1024){ float v=lg[i]; if(v>mx){ mx=v; mi=i; } }
    for(int o=16;o;o>>=1){ float ov=__shfl_xor_sync(~0u,mx,o); int oi=__shfl_xor_sync(~0u,mi,o); if(ov>mx||(ov==mx&&oi<mi)){ mx=ov; mi=oi; } }
    if(!ln){ rv[w]=mx; ri[w]=mi; } __syncthreads();
    if(!w){ mx=rv[ln]; mi=ri[ln];
        for(int o=16;o;o>>=1){ float ov=__shfl_xor_sync(~0u,mx,o); int oi=__shfl_xor_sync(~0u,mi,o); if(ov>mx||(ov==mx&&oi<mi)){ mx=ov; mi=oi; } }
        if(!ln){ rv[0]=mx; ri[0]=mi; sth=mx-30.f; ssum=0; } }
    __syncthreads();
    float M=rv[0], z=0; for(int i=t;i<V;i+=1024) z+=__expf(lg[i]-M);
    for(int o=16;o;o>>=1) z+=__shfl_xor_sync(~0u,z,o); if(!ln) atomicAdd(&ssum,z); __syncthreads();
    float ls=M+logf(ssum);
    if(p.temp<=0.f||p.top_k==1){ if(!t){ out[s]=ri[0]; lp[s]=M-ls; } return; }
    for(;;){
        float th=sth; if(!t) cnt=0; __syncthreads();
        int c=0; for(int i=t;i<V;i+=1024) c+=lg[i]>=th;
        for(int o=16;o;o>>=1) c+=__shfl_xor_sync(~0u,c,o);
        if(!ln) atomicAdd(&cnt,c); __syncthreads();
        if(cnt<=DEC_SCAP) break;
        __syncthreads(); if(!t) sth=0.5f*(th+M); __syncthreads(); }
    float th=sth; __syncthreads(); if(!t) cnt=0; __syncthreads();
    for(int i=t;i<V;i+=1024){ float v=lg[i]; if(v>=th){ int k=atomicAdd(&cnt,1); if(k<DEC_SCAP){ sv[k]=v; si[k]=i; } } }
    __syncthreads();
    int n=min(cnt,DEC_SCAP),P2=1; while(P2<n) P2<<=1;
    for(int i=n+t;i<P2;i+=1024){ sv[i]=-INFINITY; si[i]=0; }
    __syncthreads();
    for(int k=2;k<=P2;k<<=1) for(int j=k>>1;j>0;j>>=1){
        for(int i=t;i<P2;i+=1024){ int ij=i^j; if(ij>i){ float a=sv[i],b=sv[ij];
            if((i&k)==0?a<b:a>b){ sv[i]=b; sv[ij]=a; int x=si[i]; si[i]=si[ij]; si[ij]=x; } } }
        __syncthreads(); }
    if(t) return;
    int k=p.top_k>0?min(p.top_k,n):n;
    float zz=0; for(int i=0;i<k;i++) zz+=__expf(sv[i]-sv[0]);
    if(p.top_p<1.f){ float c=0; for(int i=0;i<k;i++){ c+=__expf(sv[i]-sv[0])/zz; if(c>=p.top_p){ k=i+1; break; } } }
    float it=1.f/p.temp; zz=0; for(int i=0;i<k;i++) zz+=__expf((sv[i]-sv[0])*it);
    unsigned long long h=p.seed*0x9E3779B97F4A7C15ull+(unsigned long long)(p.step+1)*0xD1B54A32D192ED03ull;
    h^=h>>30; h*=0xBF58476D1CE4E5B9ull; h^=h>>27; h*=0x94D049BB133111EBull; h^=h>>31;
    float u=(float)(h>>40)*(1.f/16777216.f)*zz; int pick=k-1;
    for(int i=0;i<k;i++){ u-=__expf((sv[i]-sv[0])*it); if(u<=0){ pick=i; break; } }
    out[s]=si[pick]; lp[s]=sv[pick]-ls; }

static int dec_ok(void){ return (!g_ngram_on||g_ng_order+g_ng_tri<=4) && !g_pscratch && gC.hd<=2*WARP && !vres_on(); }   /* tokw keeps 4 tokens */
static Dec* dec_new(int B,int Tm){
    Dec*d=(Dec*)calloc(1,sizeof(Dec)); int L=gC.L,D=gC.D,KD=gC.KD; d->B=B; d->Tm=Tm; d->R=3*g_eg_dil+1;
    d->K=(float**)calloc(L,sizeof(float*)); d->V=(float**)calloc(L,sizeof(float*)); d->cA=(float**)calloc(L,sizeof(float*)); d->cC=(float**)calloc(L,sizeof(float*));
    for(int l=0;l<L;l++){ d->K[l]=(float*)dmalloc((size_t)B*Tm*KD*4); d->V[l]=(float*)dmalloc((size_t)B*Tm*KD*4);
        if(g_canA) d->cA[l]=(float*)dmalloc((size_t)B*4*D*4); if(g_canC) d->cC[l]=(float*)dmalloc((size_t)B*4*D*4); }
    if(g_eg_on&&g_eg_conv) for(int i=0;i<g_eg_np;i++) d->ey[i]=(float*)dmalloc((size_t)B*d->R*D*4);
    d->pos=(int*)dmalloc(B*4); d->ds=(int*)dmalloc(B*4); d->tokw=(int*)dmalloc(B*16); d->tok=(int*)dmalloc(B*4);
    d->otok=(int*)dmalloc(B*4); d->olp=(float*)dmalloc(B*4); d->sp=(DSP*)dmalloc(B*sizeof(DSP));
    d->hpos=(int*)calloc(B,4); d->hds=(int*)calloc(B,4); d->htokw=(int*)calloc(B*4,4); d->htok=(int*)calloc(B,4);
    d->hotok=(int*)calloc(B,4); d->holp=(float*)calloc(B,4); d->hsp=(DSP*)calloc(B,sizeof(DSP));
    return d; }
static void dcpy(float*dst,const float*src,size_t n){ ck(cudaMemcpyAsync(dst,src,n*4,cudaMemcpyDeviceToDevice,0),"dec copy"); }
/* prefill ids[0..n) through gpu_forward, store the state in slots b0..b0+nb-1; logits of the last row -> a->logits row 0 */
static void dec_prefill(Dec*d,const float*P,const Off*o,GA*a,int*tokG,const int*ids,int n,int b0,int nb){
    int D=gC.D,KD=gC.KD,T0=gC.T,N0=a->N, Tr=(n+15)&~15; if(Tr>d->Tm) Tr=d->Tm;
    if(n<1||n>d->Tm){ fprintf(stderr,"dec_prefill: n=%d outside 1..%d\n",n,d->Tm); exit(1); }
    int*pad=(int*)calloc(Tr,4); memcpy(pad,ids,(size_t)n*4); ck(cudaMemcpy(tokG,pad,(size_t)Tr*4,cudaMemcpyHostToDevice),"dec tok"); free(pad);
    a->N=Tr; gC.T=Tr; gpu_forward(P,o,a,tokG); gC.T=T0;
    for(int b=b0;b<b0+nb;b++){
        for(int l=0;l<gC.L;l++){ dcpy(d->K[l]+(size_t)b*d->Tm*KD,a->kh[l],(size_t)n*KD); dcpy(d->V[l]+(size_t)b*d->Tm*KD,a->vh[l],(size_t)n*KD);
            for(int p=n>4?n-4:0;p<n;p++){
                if(g_canA) dcpy(d->cA[l]+((size_t)b*4+(p&3))*D,a->cna[l]+(size_t)p*D,D);
                if(g_canC) dcpy(d->cC[l]+((size_t)b*4+(p&3))*D,a->cnc[l]+(size_t)p*D,D); } }
        if(g_eg_on&&g_eg_conv) for(int i=0;i<g_eg_np;i++) for(int p=n>d->R?n-d->R:0;p<n;p++)
            dcpy(d->ey[i]+((size_t)b*d->R+p%d->R)*D,g_egp[i].y+(size_t)p*D,D);
        int st=0; for(int t=0;t<n;t++) if(ids[t]==gC.eot) st=t+1;
        d->hpos[b]=n; d->hds[b]=st; for(int k=0;k<3;k++) d->htokw[b*4+k]=n-3+k>=0?ids[n-3+k]:0; }
    g_abt(a->fn+(size_t)(n-1)*D,P+o->emb,a->logits,1,D,gC.V);
    a->N=N0; }
static void dec_eg(Dec*d,GA*a,int N,int l){
    EgP*Pp=eg_pt(l); if(!Pp) return;
    int S=g_eg_S,De=g_ng_De,D=gC.D,SD=S*De; size_t ND=(size_t)N*D; eg_ensure(N); float*h=a->a_in[l],*vg=Pp->v;
    if(g_rpre) e_cpy(g_rptmp,h,ND);
    if(Pp==g_egp){ KL(k_dec_egkey,nblk((size_t)S*N,TPB),TPB)(d->tokw,g_eg_canonD,g_ngkey,d->pos,N,S,g_eg_Rp);
        KL(k_eg_gather,nblk((size_t)N*SD,TPB),TPB)(g_ng,g_ngkey,g_ngbuf,N,S,De); }
    g_ab(g_ngbuf,Pp->Wv,Pp->v,N,SD,D);
    if(g_eg_gate){ g_ab(g_ngbuf,Pp->Wk,Pp->kn,N,SD,D); KL(k_eg_gate_f,N,128)(h,Pp->v,Pp->kn,Pp->hn,Pp->vg,Pp->g,Pp->rh,Pp->rk,D); vg=Pp->vg; }
    if(g_eg_conv){ e_rmsf(vg,Pp->nw,0,Pp->y,Pp->ri,N,D);
        KL(k_dec_egconv,nblk(ND,TPB),TPB)(h,vg,Pp->y,Pp->K,d->ey[Pp-g_egp],d->pos,N,D,g_eg_dil,d->R); }
    else e_add(h,vg,ND);
    if(g_rpre) KL(k_egacc,nblk(ND,TPB),TPB)(g_egc,h,g_rptmp,ND);
    kchk("dec eg",l); }
/* one token per slot 0..N-1 (d->htok at d->hpos); final-norm rows -> a->fn, logits -> a->logits, samples -> d->hotok/holp */
static void dec_step(Dec*d,const float*P,const Off*o,GA*a,int N,int sample){
    int D=gC.D,KD=gC.KD,N0=a->N; const Off*rd=g_prd?g_prd:o; const int*dsp=g_docmask?d->ds:NULL;
    for(int b=0;b<N;b++) d->htokw[b*4+3]=d->htok[b];
    ck(cudaMemcpyAsync(d->tok,d->htok,N*4,cudaMemcpyHostToDevice,0),"dec h2d"); ck(cudaMemcpyAsync(d->pos,d->hpos,N*4,cudaMemcpyHostToDevice,0),"dec h2d");
    ck(cudaMemcpyAsync(d->ds,d->hds,N*4,cudaMemcpyHostToDevice,0),"dec h2d"); ck(cudaMemcpyAsync(d->tokw,d->htokw,N*16,cudaMemcpyHostToDevice,0),"dec h2d");
    a->N=N;
    KL(k_embed,nblk((size_t)N*D,TPB),TPB)(P,rd->emb,d->tok,a->a_in[0],N,D);
    if(g_rpre){ rpre_ga(N); ck(cudaMemset(g_egc,0,(size_t)N*D*4),"egc0"); }
    if(g_eg_on) dec_eg(d,a,N,0);
    if(g_ngram_on&&!g_ng_layer) ngram_forward(a,NULL,N,d->tokw,d->pos);
    for(int l=0;l<gC.L;l++){
        if(g_eg_on&&l) dec_eg(d,a,N,l);
        if(g_ngram_on&&l&&l==g_ng_layer) ngram_forward(a,NULL,N,d->tokw,d->pos);
        e_rmsf(a->a_in[l],P,rd->an1[l],g_canA?a->cna[l]:a->a_norm[l],a->r1[l],N,D);
        if(g_canA) KL(k_dec_canon,nblk((size_t)N*D,TPB),TPB)(a->cna[l],XCAN(g_xP,l,0),a->a_norm[l],d->cA[l],d->pos,dsp,N,D);
        g_ab(a->a_norm[l],P+rd->wq[l],a->qh[l],N,D,D);
        g_ab(a->a_norm[l],P+rd->wk[l],a->kh[l],N,D,KD);
        g_ab(a->a_norm[l],P+rd->wv[l],a->vh[l],N,D,KD);
        KL(k_dec_rope,nblk((size_t)N*gC.H*gC.half,TPB),TPB)(a->qh[l],gRC,gRS,d->pos,N,gC.H,gC.hd,gC.half);
        KL(k_dec_rope,nblk((size_t)N*gC.KV*gC.half,TPB),TPB)(a->kh[l],gRC,gRS,d->pos,N,gC.KV,gC.hd,gC.half);
        KL(k_dec_kv,nblk((size_t)N*KD,TPB),TPB)(a->kh[l],a->vh[l],d->K[l],d->V[l],d->pos,N,d->Tm,KD);
        KL(k_dec_attn,N*gC.H,WARP)(a->qh[l],d->K[l],d->V[l],a->att[l],d->pos,dsp,lwin(l),N,d->Tm,gC.H,gC.KV,gC.hd,D,KD);
        const float*ao=a->att[l];
        if(g_attg){ g_ab(a->a_norm[l],g_Wg+(size_t)l*D*D,a->agz[l],N,D,D);
            KL(k_agate_f,nblk((size_t)N*D,TPB),TPB)(a->att[l],a->agz[l],a->attg,(size_t)N*D); ao=a->attg; }
        g_ab(ao,P+rd->wo[l],a->f_in[l],N,D,D);
        e_add(a->f_in[l],a->a_in[l],(size_t)N*D);
        e_rmsf(a->f_in[l],P,rd->an2[l],g_canC?a->cnc[l]:a->f_norm[l],a->r2[l],N,D);
        if(g_canC) KL(k_dec_canon,nblk((size_t)N*D,TPB),TPB)(a->cnc[l],XCAN(g_xP,l,1),a->f_norm[l],d->cC[l],d->pos,dsp,N,D);
        float*out=(l+1<gC.L)?a->a_in[l+1]:a->x_final;
        if(lyT(&gC,l)){ moe_forward(P,o,a,l,out,N); if(g_shF) shared_fwd(a,l,out,N); }
        else { int F=lyF(&gC,l); g_ab(a->f_norm[l],P+rd->w1[l],a->gg[l],N,D,F); g_ab(a->f_norm[l],P+rd->w3[l],a->uu[l],N,D,F);
            e_swig(a->gg[l],a->uu[l],a->acttmp,(size_t)N*F); g_ab(a->acttmp,P+rd->w2[l],out,N,F,D); e_add(out,a->f_in[l],(size_t)N*D); }
        layer_sync(); kchk("dec layer",l); }
    e_rmsf(a->x_final,P,rd->nf,a->fn,a->fnr,N,D);
    g_abt(a->fn,P+o->emb,a->logits,N,D,gC.V);
    if(sample){ ck(cudaMemcpyAsync(d->sp,d->hsp,N*sizeof(DSP),cudaMemcpyHostToDevice,0),"dec sp");
        k_dec_sample<<<N,1024>>>(a->logits,gC.V,d->sp,d->otok,d->olp); kchk("dec sample",-1);
        ck(cudaMemcpy(d->hotok,d->otok,N*4,cudaMemcpyDeviceToHost),"dec d2h"); ck(cudaMemcpy(d->holp,d->olp,N*4,cudaMemcpyDeviceToHost),"dec d2h"); }
    for(int b=0;b<N;b++){ int t=d->htok[b]; if(t==gC.eot) d->hds[b]=d->hpos[b]+1; d->hpos[b]++;
        d->htokw[b*4]=d->htokw[b*4+1]; d->htokw[b*4+1]=d->htokw[b*4+2]; d->htokw[b*4+2]=t; }
    a->N=N0; }
/* sample the prefill row (a->logits row 0) into slots b0..b0+nb-1 with each slot's own params */
static void dec_sample_rows(Dec*d,GA*a,int b0,int nb){
    for(int b=1;b<nb;b++) dcpy(a->logits+(size_t)b*gC.V,a->logits,gC.V);
    ck(cudaMemcpyAsync(d->sp+b0,d->hsp+b0,nb*sizeof(DSP),cudaMemcpyHostToDevice,0),"dec sp");
    k_dec_sample<<<nb,1024>>>(a->logits,gC.V,d->sp+b0,d->otok+b0,d->olp+b0); kchk("dec sample",-1);
    ck(cudaMemcpy(d->hotok+b0,d->otok+b0,nb*4,cudaMemcpyDeviceToHost),"dec d2h"); ck(cudaMemcpy(d->holp+b0,d->olp+b0,nb*4,cudaMemcpyDeviceToHost),"dec d2h"); }
/* move slot s into slot t (continuous batching keeps the live slots at 0..N-1) */
static void dec_move(Dec*d,int s,int t){
    int D=gC.D,KD=gC.KD; size_t n=(size_t)d->hpos[s]*KD, ss=(size_t)s*d->Tm*KD, tt=(size_t)t*d->Tm*KD;
    for(int l=0;l<gC.L;l++){ dcpy(d->K[l]+tt,d->K[l]+ss,n); dcpy(d->V[l]+tt,d->V[l]+ss,n);
        if(g_canA) dcpy(d->cA[l]+(size_t)t*4*D,d->cA[l]+(size_t)s*4*D,(size_t)4*D);
        if(g_canC) dcpy(d->cC[l]+(size_t)t*4*D,d->cC[l]+(size_t)s*4*D,(size_t)4*D); }
    if(g_eg_on&&g_eg_conv) for(int i=0;i<g_eg_np;i++) dcpy(d->ey[i]+(size_t)t*d->R*D,d->ey[i]+(size_t)s*d->R*D,(size_t)d->R*D);
    d->hpos[t]=d->hpos[s]; d->hds[t]=d->hds[s]; d->htok[t]=d->htok[s]; d->hsp[t]=d->hsp[s];
    memcpy(d->htokw+t*4,d->htokw+s*4,16); }
/* decheck <model> [n=200] [k=40]: decode logits vs the full forward, 2 slots with different prefixes */
static int cmd_decheck(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: cudalm decheck <model> [n=200] [k=40]\n"); return 1; }
    g_fp8gemm=0;    /* batch-wide FP8 scales would make the full-forward reference differ from the decoder by up to ~3.6 logits */
    char path[1024]; model_path(argv[2],path,sizeof(path)); FILE*e=fopen(path,"rb"); if(e) fclose(e); else snprintf(path,sizeof(path),"%s",argv[2]);
    Cfg c; Tok*tk=NULL; float*Ph=model_load_full(path,&c,&tk); gC=c; cuda_init(); Off o; calc_offsets(&c,&o); ngram_infer(path);
    size_t np=param_count(&c); float*P=(float*)dmalloc(np*4); ck(cudaMemcpy(P,Ph,np*4,cudaMemcpyHostToDevice),"P"); rope_tables(&c);
    if(!dec_ok()){ fprintf(stderr,"decheck: ngram order > 4 / QLM scratch not supported by the decoder\n"); return 1; }
    int n=argc>3?atoi(argv[3]):200, k=argc>4?atoi(argv[4]):40; if(n+k>c.T){ n=c.T-k; }
    GA*a=ga_new(c.T,0); int*tokG=(int*)dmalloc((size_t)c.T*4);
    int*ids=(int*)malloc(sizeof(int)*2*(n+k)); uint64_t r=12345;
    for(int i=0;i<2*(n+k);i++){ r=r*6364136223846793005ULL+1442695040888963407ULL; ids[i]=(int)((r>>33)%(uint64_t)c.V); }
    for(int i=37;i<2*(n+k);i+=97) ids[i]=c.eot;                  /* documents, for CUDALM_DOCMASK */
    int len[2]={n,n-7}; float*ref=(float*)malloc((size_t)2*(k+1)*c.V*4),*got=(float*)malloc((size_t)(k+1)*c.V*4);
    for(int s=0;s<2;s++){ int L=len[s]+k, Tr=(L+15)&~15; const int*q=ids+s*(n+k);
        int*pad=(int*)calloc(Tr,4); memcpy(pad,q,(size_t)L*4); ck(cudaMemcpy(tokG,pad,(size_t)Tr*4,cudaMemcpyHostToDevice),"tok"); free(pad);
        a->N=Tr; gC.T=Tr; gpu_forward(P,&o,a,tokG); gC.T=c.T; a->N=c.T;
        g_abt(a->fn+(size_t)(len[s]-1)*c.D,P+o.emb,a->logits,k+1,c.D,c.V);
        ck(cudaMemcpy(ref+(size_t)s*(k+1)*c.V,a->logits,(size_t)(k+1)*c.V*4,cudaMemcpyDeviceToHost),"ref"); }
    Dec*d=dec_new(2,c.T); double worst=0; int agree=0,tot=0;
    for(int s=0;s<2;s++){ dec_prefill(d,P,&o,a,tokG,ids+s*(n+k),len[s],s,1);
        ck(cudaMemcpy(got,a->logits,(size_t)c.V*4,cudaMemcpyDeviceToHost),"got");
        const float*rf=ref+(size_t)s*(k+1)*c.V; double m=0; int am=0,bm=0;
        for(int i=0;i<c.V;i++){ m=fmax(m,fabs(got[i]-rf[i])); if(got[i]>got[am]) am=i; if(rf[i]>rf[bm]) bm=i; }
        printf("  slot %d prefill(%d): max|dlogit| %.2e argmax %s\n",s,len[s],m,am==bm?"same":"DIFF"); worst=fmax(worst,m); agree+=am==bm; tot++; }
    for(int j=0;j<k;j++){
        for(int s=0;s<2;s++) d->htok[s]=ids[s*(n+k)+len[s]+j];
        dec_step(d,P,&o,a,2,0);
        ck(cudaMemcpy(got,a->logits,(size_t)2*c.V*4,cudaMemcpyDeviceToHost),"got");
        for(int s=0;s<2;s++){ const float*rf=ref+((size_t)s*(k+1)+j+1)*c.V,*g=got+(size_t)s*c.V; double m=0; int am=0,bm=0;
            for(int i=0;i<c.V;i++){ m=fmax(m,fabs(g[i]-rf[i])); if(g[i]>g[am]) am=i; if(rf[i]>rf[bm]) bm=i; }
            worst=fmax(worst,m); agree+=am==bm; tot++;
            if(j<2||j==k-1) printf("  slot %d step %d (pos %d): max|dlogit| %.2e argmax %s\n",s,j,len[s]+j,m,am==bm?"same":"DIFF"); } }
    printf("[decheck] worst max|dlogit| %.3e, argmax agreement %d/%d\n",worst,agree,tot);
    return worst<5e-2&&agree==tot?0:2; }
#endif

/* tokcheck <model> <tokens.u16|.bin> [maxdocs]: decode docs (eot-separated), re-encode with tok_encode, count mismatches */
static int* tok_encode_whole(const Tok*tk,const char*s,int*out){   /* pre-18b encoder: BPE over the whole string */
    int n=(int)strlen(s); TEnc e={tk,(int*)malloc(sizeof(int)*(n>0?n:1)),0}; te_bpe(&e,(const unsigned char*)s,n); *out=e.cnt; return e.ids; }
static int cmd_tokcheck(int argc,char**argv){
    if(argc<4){ fprintf(stderr,"usage: cudalm tokcheck <model> <tokens.u16> [maxdocs] [old]\n"); return 1; }
    int*(*enc)(const Tok*,const char*,int*)=argc>5&&!strcmp(argv[5],"old")?tok_encode_whole:tok_encode;
    char mp[1024]; model_resolve(argv[2],mp,sizeof mp);
    Cfg c; Tok*tk=NULL; float*Ph=model_load_full(mp,&c,&tk); free(Ph);
    size_t len; uint8_t*b=read_file(argv[3],&len); if(!b){ fprintf(stderr,"cannot open %s\n",argv[3]); return 1; }
    const uint16_t*t=(const uint16_t*)b; size_t n=len/2; int maxd=argc>4?atoi(argv[4]):1000000, docs=0, bad=0; size_t toks=0;
    printf("[tokcheck] %d special tokens\n",tk->nsp);
    for(size_t i=0;i<n&&docs<maxd;){
        size_t j=i; while(j<n&&t[j]!=tk->eot) j++;
        if(j>i){ size_t L=0; for(size_t k=i;k<j;k++) L+=tk->declen[t[k]];
            char*s=(char*)malloc(L+1); L=0; for(size_t k=i;k<j;k++){ memcpy(s+L,tk->dec[t[k]],tk->declen[t[k]]); L+=tk->declen[t[k]]; } s[L]=0;
            int m; int*ids=enc(tk,s,&m); int same=(size_t)m==j-i; for(int k=0;same&&k<m;k++) same=ids[k]==t[i+k];
            if(!same&&bad++<3){ int k=0; while(k<m&&(size_t)k<j-i&&ids[k]==t[i+k]) k++;
                printf("  doc %d differs at token %d: got %d want %d (\"",docs,k,k<m?ids[k]:-1,(size_t)k<j-i?t[i+k]:-1);
                for(int q=k>2?k-2:0;q<k+3&&(size_t)q<j-i;q++){ fwrite(tk->dec[t[i+q]],1,tk->declen[t[i+q]],stdout); putchar('|'); } printf("\")\n"); }
            toks+=j-i; docs++; free(ids); free(s); }
        i=j+1; }
    printf("[tokcheck] %d docs, %zu tokens, %d mismatched docs\n",docs,toks,bad); free(b); return bad?2:0; }
static int cmd_info(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: cudalm info <name|model.bin>  (config, saved settings, total/memory/active params)\n"); return 1; }
    char p[1100]; model_resolve(argv[2],p,sizeof p); FILE*f=fopen(p,"rb"); if(!f){ fprintf(stderr,"cannot open %s\n",p); return 1; }
    char mg[4]={0}; int h[256]={0}, ver=0, nh=0; size_t got=fread(mg,1,4,f);
    if(!memcmp(mg,"TLQ1",4)){ nh=11; ver=4; } else if(!memcmp(mg,"TLQ2",4)){ nh=13; ver=5; } else nh=hdr_ints(mg,&ver);
    if(got!=4||!nh||fread(h,4,nh,f)!=(size_t)nh){ fclose(f); fprintf(stderr,"%s: not a model file\n",p); return 1; }
    lm_fseek64(f,0,SEEK_END); double mb=lm_ftell64(f)/1048576.0; fclose(f);
    Cfg c; cfg_from_hdr(&c,h,ver);
    printf("%s  (%.1f MB, %.4s)\n  D=%d L=%d H=%d KV=%d F=%d T=%d V=%d experts=%d%s win=%d full_every=%d mtp=%d\n",
           p,mb,mg,c.D,c.L,c.H,c.KV,c.F,c.T,c.V,c.n_exp,c.pool?" (UniPool)":"",c.win,c.full_every,c.n_mtp);
    if(c.mod){ printf("  layout:"); for(int l=0;l<c.L;l++){ int t=lyT(&c,l); printf(t==0?" D%d":t==1?" M%d":" P%d",t?lyE(&c,l):lyF(&c,l)); } printf("\n"); }
    char sp[1200]; snprintf(sp,sizeof sp,"%s.env",p); FILE*e=fopen(sp,"rb");
    if(e){ char l[1024]; printf("  config %s:\n",sp); while(fgets(l,sizeof l,e)){ l[strcspn(l,"\r\n")]=0; if(*l&&*l!='#') printf("    %s\n",l); } fclose(e); }
    else printf("  no model config yet (train writes one; untrained memory settings come from the environment)\n");
    MemCfg m; int ff=mem_from_file(p,&m); if(!ff) mem_from_env(&m,c.L);
    int ag=t1_attg_file(p); if(ag<0) ag=env_i("CUDALM_ATTN_GATE",0)!=0;
    { snprintf(sp,sizeof sp,"%s.opt",p); FILE*q=fopen(sp,"rb"); if(q){ fclose(q); g_fe_file=0; strcat(sp,".fe");
        if((q=fopen(sp,"rb"))){ char m4[4]; int fh[4]; if(fread(m4,1,4,q)==4&&!memcmp(m4,"FES1",4)&&fread(fh,4,4,q)==4) g_fe_file=fh[0]; fclose(q); } } }
    if(fe_dim(c.D)) printf("  embedding: factorized E=%d (%s x %d + %d x %d)\n",fe_dim(c.D),pcomma(c.V),fe_dim(c.D),fe_dim(c.D),c.D);
    printf("  memory: %s (from %s)\n",mem_name(m.mode),ff?"trained .opt.ng":"config/environment");
    pstats_print(&c,&m,g_topk,g_topkL,ag); return 0; }

/* ===================== new (create an untrained model) =================== */
/* Mirrors tinylm.c init_params exactly so a model created here is
 * indistinguishable from one made by the CPU build: N(0, 0.02) for embeddings
 * and matrices, residual projections (wo, w2) scaled by 1/sqrt(2L), norm gains
 * at 1, MTP heads zeroed. Dense only -- no MoE router. */
static int cmd_new(int argc,char**argv){
    if(argc<4){ fprintf(stderr,
        "usage: cudalm new <name> <preset> [n_experts] [ctx] [window] [full_every]\n"
        "  presets: nano micro m1 m3 m6 m8 m16 m32 m70 m50 m64e m220e a8m a8m9 s8x s8w mod230\n"
        "           custom = D160 L11 H5 KV1 F432 T2048 base, meant for CUDALM_D/_L/_H/_KV/_F\n"
        "  n_experts: >=2 turns the FFN into top-1 routed MoE (more total capacity\n"
        "             at ~the same per-token compute). e.g. m64e 8 = 16.2M active / 63.0M total\n"
        "  ctx:       override the preset context length (e.g. 8192)\n"
        "  window:    sliding attention window; 0 = full causal. Attention then costs\n"
        "             O(window) per token instead of O(ctx), so long contexts stay cheap.\n"
        "             Receptive field = layers * window, so pick window >= ctx/layers.\n"
        "  full_every: every Nth layer ignores the window and attends to the whole\n"
        "             prefix (0 = never). Restores exact long-range retrieval.\n"
        "  CUDALM_D / _L / _H / _KV / _F: override the preset shape (e.g. D=160 L=15 H=5 KV=1 F=432)\n"
        "  CUDALM_UNIPOOL=1: n_experts becomes ONE pool shared by all layers\n"
        "  CUDALM_LAYOUT=D2560,M28,P,...: per-layer FFN (dense F / MoE E / UniPool), overrides L and n_experts;\n"
        "             with CUDALM_EXP_F (expert width) and CUDALM_POOL_E (pool size)\n"
        "  every CUDALM_*/QLM_*/NGRAM_* setting in the environment (memory mode, top-k, QLM, LR...)\n"
        "             is saved to models/<name>.bin.env; train re-saves it each run, and every\n"
        "             command naming the model applies it (the environment still overrides)\n"
        "             (per-layer routers, pool-level aux loss)\n"
        "  e.g. cudalm new big m64e 8 8192 1024 4\n"
        "  creates models/<name>.bin (untrained), then: cudalm train <name> <data_dir> <epochs>\n");
        return 1; }
    const char *name=argv[2], *ps=argv[3];
    Cfg c; if(!preset(ps,&c)){ fprintf(stderr,"unknown preset '%s'\n",ps); return 1; }
    { char err[200]; if(!shape_env(&c,err,sizeof err)){ fprintf(stderr,"new: %s\n",err); return 1; } }
    c.n_mtp=0;                                 /* MTP is not trained on the CUDA path */
    c.n_exp = argc>4 ? atoi(argv[4]) : 0; if(c.n_exp==1) c.n_exp=0;
    if(c.n_exp>256){ fprintf(stderr,"n_experts %d > 256\n",c.n_exp); return 1; }
    { const char*up=getenv("CUDALM_UNIPOOL"); c.pool=(up&&atoi(up)&&c.n_exp>1)?1:0; }
    if(argc>5){ int t=atoi(argv[5]); if(t>=16) c.T=t; }
    c.win        = argc>6 ? atoi(argv[6]) : 0; if(c.win<0) c.win=0;
    c.full_every = argc>7 ? atoi(argv[7]) : 0; if(c.full_every<0) c.full_every=0;
    if(c.win>=c.T) c.win=0;                    /* window covers everything -> just full */
    if(!layout_apply(&c)) return 1;
    cfg_derive(&c);
    char path[1024]; model_path(name,path,sizeof(path));
    { FILE*e=fopen(path,"rb"); if(e){ fclose(e); fprintf(stderr,"model '%s' already exists (%s)\n",name,path); return 1; } }
    Off o; calc_offsets(&c,&o);
    size_t np=param_count(&c); float*P=(float*)calloc(np,4);
    if(!P){ fprintf(stderr,"OOM %zu params\n",np); return 1; }
    int pinit=0;
    for(size_t i=0;i<(size_t)c.V*c.D;i++) P[o.emb+i]=0.02f*rnd_normal();
    float scaled=0.02f/sqrtf(2.0f*c.L);
    for(int l=0;l<c.L;l++){
        for(int i=0;i<c.D;i++){ P[o.an1[l]+i]=1.0f; P[o.an2[l]+i]=1.0f; }
        size_t dd=(size_t)c.D*c.D, dkd=(size_t)c.D*c.KD;
        int t=lyT(&c,l); size_t W=t?(size_t)lyE(&c,l)*c.F:(size_t)lyF(&c,l), edf=W*c.D, efd=edf;
        for(size_t i=0;i<dd; i++) P[o.wq[l]+i]=0.02f*rnd_normal();
        for(size_t i=0;i<dkd;i++) P[o.wk[l]+i]=0.02f*rnd_normal();
        for(size_t i=0;i<dkd;i++) P[o.wv[l]+i]=0.02f*rnd_normal();
        for(size_t i=0;i<dd; i++) P[o.wo[l]+i]=scaled*rnd_normal();
        if(t) for(size_t i=0;i<(size_t)c.D*lyE(&c,l);i++) P[o.wr[l]+i]=0.02f*rnd_normal();
        if(t==2&&pinit++) continue;               /* the pool is initialised once */
        for(size_t i=0;i<edf;i++) P[o.w1[l]+i]=0.02f*rnd_normal();
        for(size_t i=0;i<edf;i++) P[o.w3[l]+i]=0.02f*rnd_normal();
        for(size_t i=0;i<efd;i++) P[o.w2[l]+i]=scaled*rnd_normal();
    }
    for(int i=0;i<c.D;i++) P[o.nf+i]=1.0f;
    model_save_bundle(path,&c,P,np,NULL,0);     /* untrained: tokenizer adopted on first train */
    printf("[new] created %s  preset=%s params=%zu (%.2fM total)\n", path, ps, np, np/1e6);
    { MemCfg m; mem_from_env(&m,c.L); side_write(path,&m);
      printf("      memory: %s\n",mem_name(m.mode)); pstats_print(&c,&m,g_topk,g_topkL,env_i("CUDALM_ATTN_GATE",0)!=0); }
    printf("      D=%d L=%d H=%d KV=%d F=%d T=%d V=%d (V is replaced by the data's vocab on first train)\n",
           c.D,c.L,c.H,c.KV,c.F,c.T,c.V);
    if(c.win>0){
        int rf=c.L*c.win;
        printf("      sliding attention: window %d, receptive field %d x %d = %d %s\n",
               c.win,c.L,c.win,rf, rf>=c.T?"(>= ctx, OK)":"(< ctx - deep tokens cannot see the start)");
        if(c.full_every>0) printf("      every %dth layer uses full attention (%d of %d layers)\n",
                                  c.full_every,c.L/c.full_every,c.L);
    }
    printf("[next] cudalm train %s <data_dir> <epochs>\n", name);
    free(P); return 0;
}

/* ===================== run (inference) =================================== */
/* chatml end-of-turn: replies end with eot or "\n<|user|>"; stop on the marker's first
 * token "<|". CUDALM_STOP_MARK=0 disables. */
static int chat_stop_id(const Cfg*c,const Tok*tk){
    const char*e=getenv("CUDALM_STOP_MARK");
    if(c->tmpl!=2||(e&&!atoi(e))) return -1;
    int n=0,*ids=tok_encode(tk,"<|user|>",&n),r=n>1?ids[0]:-1; free(ids); return r;
}
static void cmd_run(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: cudalm run <name|model.bin> [\"prompt\"] [n] [temp] [topk]\n"); exit(1); }
    char path[1024]; model_path(argv[2],path,sizeof(path));
    FILE*e=fopen(path,"rb"); if(e)fclose(e); else snprintf(path,sizeof(path),"%s",argv[2]);
    Cfg c; Tok*tk=NULL; float*Ph=model_load_full(path,&c,&tk);
    if(!tk){
        /* Distinguish "never trained" from "checkpoint is damaged" -- an unclean
         * shutdown can leave a full-length file whose tail was never committed,
         * which reads as zero weights AND a zero-length tokenizer blob. */
        size_t np=param_count(&c), nz=0, probe=np<100000?np:100000;
        for(size_t i=np-probe;i<np;i++) if(Ph[i]!=0.0f){ nz=1; break; }
        if(!nz) fprintf(stderr,
            "%s looks CORRUPT: the tail of the parameter block is all zeros.\n"
            "  Most likely the checkpoint was never flushed to disk (unclean shutdown\n"
            "  or the process was killed mid-save). Retrain, or resume from .opt if a\n"
            "  good earlier copy exists.\n", path);
        else fprintf(stderr,"model has no tokenizer (untrained - train it first)\n");
        exit(1); }
    const char*prompt=argc>3?argv[3]:"Once upon a time"; int n=argc>4?atoi(argv[4]):120;
    float temp=argc>5?(float)atof(argv[5]):0.8f; int topk=argc>6?atoi(argv[6]):40;
    gC=c; cuda_init(); Off o; calc_offsets(&c,&o);
    ngram_infer(path);
    size_t np=param_count(&c);
    float*P=(float*)dmalloc(np*4); ck(cudaMemcpy(P,Ph,np*4,cudaMemcpyHostToDevice),"P h2d");
    rope_tables(&c);
    char fmt[16384];
    if(c.tmpl==1)      snprintf(fmt,sizeof(fmt),"### Instruction:\n%s\n\n### Response:\n",prompt);
    else if(c.tmpl==2) snprintf(fmt,sizeof(fmt),"<|user|>\n%s\n<|assistant|>\n",prompt);
    else               snprintf(fmt,sizeof(fmt),"%s",prompt);
    int pn; int*ids=tok_encode(tk,fmt,&pn);
    int cap=c.T; int*seq=(int*)malloc(sizeof(int)*cap); int slen=0;
    for(int i=0;i<pn&&slen<cap;i++) seq[slen++]=ids[i];
    GA*a=ga_new(c.T,0); int*tokG=(int*)dmalloc((size_t)c.T*4);
    float*logits=(float*)malloc((size_t)c.V*4); int*padded=(int*)malloc(sizeof(int)*c.T);
    printf("[cudalm run] %s | %s | temp=%.2f topk=%d\n", argv[2],
           c.tmpl==2?"chatml":(c.tmpl==1?"alpaca":"raw"), temp, topk);
    printf("------------------------------------------------------------\n%s",fmt); fflush(stdout);
    int greedy=(topk<=1||temp<=1e-4f), stopid=chat_stop_id(&c,tk);
    double t0=wall(); int produced=0;
    for(int step=0;step<n;step++){
        int ctx=slen<c.T?slen:c.T, off=slen-ctx;
        for(int j=0;j<c.T;j++) padded[j]= j<ctx? seq[off+j]:0;
        ck(cudaMemcpy(tokG,padded,(size_t)c.T*4,cudaMemcpyHostToDevice),"tok h2d");
#ifdef CUDALM_CPU
        /* causal: rows past ctx cannot reach row ctx-1, so only run ctx (rounded up) */
        { int Tr=(ctx+15)&~15; if(Tr>c.T) Tr=c.T; a->N=Tr; gC.T=Tr; }
        gpu_forward(P,&o,a,tokG); a->N=c.T; gC.T=c.T;
#else
        gpu_forward(P,&o,a,tokG);
#endif
        /* only the sampled row's logits are needed */
        g_abt(a->fn+(size_t)(ctx-1)*c.D,P+o.emb,a->logits,1,c.D,c.V);
        ck(cudaMemcpy(logits,a->logits,(size_t)c.V*4,cudaMemcpyDeviceToHost),"logits d2h");
        int nxt;
        if(greedy){ nxt=0; for(int i=1;i<c.V;i++) if(logits[i]>logits[nxt])nxt=i; }
        else { float it=temp>1e-6f?temp:1e-6f; for(int i=0;i<c.V;i++)logits[i]/=it;
            if(topk>0&&topk<c.V){ float*tmp=(float*)malloc((size_t)c.V*4); memcpy(tmp,logits,(size_t)c.V*4);
                for(int i=0;i<topk;i++){int mi=i;for(int j=i+1;j<c.V;j++)if(tmp[j]>tmp[mi])mi=j;float t=tmp[i];tmp[i]=tmp[mi];tmp[mi]=t;}
                float thr=tmp[topk-1]; for(int i=0;i<c.V;i++)if(logits[i]<thr)logits[i]=-1e30f; free(tmp); }
            float mx=-1e30f;for(int i=0;i<c.V;i++)if(logits[i]>mx)mx=logits[i]; float sum=0;
            for(int i=0;i<c.V;i++){logits[i]=expf(logits[i]-mx);sum+=logits[i];} float r=urand()*sum,ac=0; nxt=c.V-1;
            for(int i=0;i<c.V;i++){ac+=logits[i];if(ac>=r){nxt=i;break;}} }
        if(nxt==tk->eot||nxt==stopid) break;
        tok_print(tk,nxt); fflush(stdout); produced++;
        if(slen<cap) seq[slen++]=nxt; else { memmove(seq,seq+1,(cap-1)*sizeof(int)); seq[cap-1]=nxt; }
    }
    double dt=wall()-t0;
    printf("\n------------------------------------------------------------\n[cudalm] %d tok in %.2fs (%.1f tok/s)\n",
           produced,dt,dt>0?produced/dt:0.0);
}

/* ===================== serve ============================================ */
/* cudalm serve <model> [port=8090]: a llama-server-compatible subset for agentenv/evals.
 * POST /completion {"prompt","n_predict","temperature","top_k","top_p","seed","stop":[..],"num_first"} -> {"content",
 * "tokens_predicted","stop_type":"eos|word|limit","stopped_eos","nll"}; POST /logprob (scoring, see sv_logprob); GET /health; POST /shutdown (clean exit, never kill a CUDA process). Raw prompt (no template),
 * concurrent clients: continuous batching on the KV-cache decoder (CUDALM_SERVE_SLOTS, default 64), else one request at a
 * time with a full-context forward per token like `run`. Winsock is loaded at run time. */
#ifdef _WIN32
typedef uintptr_t SK;
static struct { int (WINAPI*startup)(WORD,void*); SK (WINAPI*sock)(int,int,int); int (WINAPI*bind_)(SK,const void*,int);
    int (WINAPI*listen_)(SK,int); SK (WINAPI*accept_)(SK,void*,int*); int (WINAPI*recv_)(SK,char*,int,int);
    int (WINAPI*send_)(SK,const char*,int,int); int (WINAPI*close_)(SK); } W;
static void ws_load(void){
    HMODULE h=LoadLibraryA("ws2_32.dll"); if(!h){ fprintf(stderr,"no ws2_32.dll\n"); exit(1); }
    *(FARPROC*)&W.startup=GetProcAddress(h,"WSAStartup"); *(FARPROC*)&W.sock=GetProcAddress(h,"socket");
    *(FARPROC*)&W.bind_=GetProcAddress(h,"bind"); *(FARPROC*)&W.listen_=GetProcAddress(h,"listen");
    *(FARPROC*)&W.accept_=GetProcAddress(h,"accept"); *(FARPROC*)&W.recv_=GetProcAddress(h,"recv");
    *(FARPROC*)&W.send_=GetProcAddress(h,"send"); *(FARPROC*)&W.close_=GetProcAddress(h,"closesocket");
    char wd[512]; W.startup(0x0202,wd);
}
typedef struct { char*p; size_t n,cap; } SBuf;
static void sb_put(SBuf*b,const char*s,size_t n){ if(b->n+n+1>b->cap){ b->cap=(b->n+n+1)*2; b->p=(char*)realloc(b->p,b->cap); }
    memcpy(b->p+b->n,s,n); b->n+=n; b->p[b->n]=0; }
static void sb_str(SBuf*b,const char*s){ sb_put(b,s,strlen(s)); }
static void sb_esc(SBuf*b,const char*s,size_t n){
    for(size_t i=0;i<n;i++){ unsigned char c=(unsigned char)s[i]; char t[8];
        if(c=='"'||c=='\\'){ t[0]='\\'; t[1]=(char)c; sb_put(b,t,2); }
        else if(c=='\n') sb_put(b,"\\n",2); else if(c=='\r') sb_put(b,"\\r",2); else if(c=='\t') sb_put(b,"\\t",2);
        else if(c<0x20) sb_put(b,t,(size_t)sprintf(t,"\\u%04x",c)); else sb_put(b,(const char*)&c,1); } }
/* JSON string at p (just past the opening quote) -> malloc'd UTF-8; *end = past the closing quote */
static char* js_str(const char*p,const char**end){
    SBuf b={0}; sb_put(&b,"",0);
    for(;*p&&*p!='"';p++){
        if(*p!='\\'){ sb_put(&b,p,1); continue; }
        char c=*++p;
        if(c=='u'){ char h[5]={p[1],p[2],p[3],p[4],0}; unsigned v=(unsigned)strtoul(h,0,16); p+=4;
            if(v>=0xD800&&v<0xDC00&&p[1]=='\\'&&p[2]=='u'){ char l[5]={p[3],p[4],p[5],p[6],0}; v=0x10000+((v-0xD800)<<10)+((unsigned)strtoul(l,0,16)-0xDC00); p+=6; }
            char t[4]; int k=0;
            if(v<0x80) t[k++]=(char)v; else if(v<0x800){ t[k++]=(char)(0xC0|v>>6); t[k++]=(char)(0x80|(v&63)); }
            else if(v<0x10000){ t[k++]=(char)(0xE0|v>>12); t[k++]=(char)(0x80|((v>>6)&63)); t[k++]=(char)(0x80|(v&63)); }
            else { t[k++]=(char)(0xF0|v>>18); t[k++]=(char)(0x80|((v>>12)&63)); t[k++]=(char)(0x80|((v>>6)&63)); t[k++]=(char)(0x80|(v&63)); }
            sb_put(&b,t,(size_t)k); continue; }
        c= c=='n'?'\n':c=='t'?'\t':c=='r'?'\r':c=='b'?'\b':c=='f'?'\f':c; sb_put(&b,&c,1); }
    if(end) *end=*p?p+1:p; return b.p; }
static const char* js_key(const char*j,const char*k){
    char q[80]; snprintf(q,sizeof q,"\"%s\"",k); const char*p=strstr(j,q); if(!p) return NULL;
    p+=strlen(q); while(*p==' '||*p==':'||*p=='\n'||*p=='\r'||*p=='\t') p++; return p; }
static double js_num(const char*j,const char*k,double d){ const char*p=js_key(j,k); return p&&(*p=='-'||(*p>='0'&&*p<='9'))?atof(p):d; }
typedef struct { char*prompt,*stops[16]; int npred,n,topk,ns,greedy,numf; float temp,topp; double sd; } SReq;
static void sreq_parse(const char*body,SReq*r,int T,int nmax){
    memset(r,0,sizeof *r); const char*pp=js_key(body,"prompt"); r->prompt=pp&&*pp=='"'?js_str(pp+1,NULL):strdup("");
    r->npred=(int)js_num(body,"n_predict",js_num(body,"max_tokens",128)); if(r->npred<0) r->npred=T;
    r->temp=(float)js_num(body,"temperature",0.8); r->topp=(float)js_num(body,"top_p",1.0); r->topk=(int)js_num(body,"top_k",40);
    r->sd=js_num(body,"seed",-1); r->greedy=r->temp<=1e-4f||r->topk==1;
    const char*sp=js_key(body,"stop");
    if(sp&&*sp=='"') r->stops[r->ns++]=js_str(sp+1,NULL);
    else if(sp&&*sp=='[') for(sp++;*sp&&*sp!=']'&&r->ns<16;){ if(*sp=='"'){ r->stops[r->ns]=js_str(sp+1,&sp); if(*r->stops[r->ns]) r->ns++; } else sp++; }
    r->n=(int)js_num(body,"n",1); if(r->n<1) r->n=1; if(r->n>nmax) r->n=nmax;
    r->numf=js_num(body,"num_first",0)!=0; }
/* "num_first":1 -> the first sampled token is a number token (" 42", "7"): -inf on every other logit of prefill row 0 */
static void num_mask(GA*a,const Tok*tk){
    static float*m=NULL; int V=gC.V;
    if(!m){ float*h=(float*)malloc((size_t)V*4);
        for(int i=0;i<V;i++){ const unsigned char*t=tk->dec[i]; int n=tk->declen[i], j=n&&t[0]==' ', ok=j<n;
            for(int k=j;k<n;k++) ok&=t[k]>='0'&&t[k]<='9'; h[i]=ok?0.0f:-1e30f; }
        m=(float*)dmalloc((size_t)V*4); ck(cudaMemcpy(m,h,(size_t)V*4,cudaMemcpyHostToDevice),"nmask"); free(h); }
    e_add(a->logits,m,(size_t)V); }
static void sreq_free(SReq*r){ free(r->prompt); for(int q=0;q<r->ns;q++) free(r->stops[q]); }
/* append token text; 1 if a stop string now appears (text cut before it) */
static int sv_emit(SBuf*g,const Tok*tk,int t,const SReq*r){
    size_t before=g->n; sb_put(g,(const char*)tk->dec[t],(size_t)tk->declen[t]);
    for(int q=0;q<r->ns;q++){ size_t L=strlen(r->stops[q]), from=before>=L?before-L+1:0;
        char*f=strstr(g->p+from,r->stops[q]); if(f){ g->n=(size_t)(f-g->p); g->p[g->n]=0; return 1; } }
    return 0; }
static void sv_results(SBuf*out,int B,SBuf*gen,const int*prod,const int*st,const double*nll,int pn){
    for(int b=0;b<B;b++){
        sb_str(out,B>1?(b?",{":"{\"results\":[{"):"{");
        sb_str(out,"\"content\":\""); sb_esc(out,gen[b].p,gen[b].n);
        char t[256]; snprintf(t,sizeof t,"\",\"tokens_predicted\":%d,\"tokens_evaluated\":%d,\"stop_type\":\"%s\",\"stopped_eos\":%s,\"nll\":%.5f}",
            prod[b],pn,st[b]==0?"eos":st[b]==1?"word":"limit",st[b]==0?"true":"false",prod[b]?nll[b]/(prod[b]+(st[b]==0)):0.0);
        sb_str(out,t); }
    if(B>1) sb_str(out,"]}"); }
static void sv_reply(SK cs,const char*code,const SBuf*out){
    char h[200]; int hl=snprintf(h,sizeof h,"HTTP/1.1 %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",code,out->n);
    W.send_(cs,h,hl,0); for(size_t q=0;q<out->n;){ int s=W.send_(cs,out->p+q,(int)(out->n-q),0); if(s<=0) break; q+=(size_t)s; }
    W.close_(cs); }
static void sv_replys(SK cs,const char*code,const char*s){ SBuf o={0}; sb_str(&o,s); sv_reply(cs,code,&o); free(o.p); }
/* accept thread: reads whole requests, answers /health itself, queues the rest for the GPU loop */
typedef struct SQ { SK cs; char*rq; const char*body; struct SQ*next; } SQ;
static CRITICAL_SECTION g_sqc; static SQ*g_sqh,*g_sqt; static HANDLE g_sqe; static SK g_sqls;
static DWORD WINAPI sv_accept(void*u){ (void)u;
    for(;;){
        SK cs=W.accept_(g_sqls,NULL,NULL); if(cs==(SK)~(uintptr_t)0){ Sleep(1); continue; }
        SBuf rq={0}; char buf[65536]; int k; size_t need=0; const char*body=NULL;
        while((k=W.recv_(cs,buf,sizeof buf,0))>0){ sb_put(&rq,buf,(size_t)k);
            char*h=strstr(rq.p,"\r\n\r\n");
            if(h){ const char*cl=strstr(rq.p,"Content-Length:"); if(!cl) cl=strstr(rq.p,"content-length:");
                need=(size_t)(h-rq.p)+4+(cl&&cl<h?(size_t)atol(cl+15):0); if(rq.n>=need){ body=h+4; break; } } }
        if(!body){ W.close_(cs); free(rq.p); continue; }
        if(!strncmp(rq.p,"GET /health",11)){ sv_replys(cs,"200 OK","{\"status\":\"ok\"}"); free(rq.p); continue; }
        SQ*q=(SQ*)calloc(1,sizeof(SQ)); q->cs=cs; q->rq=rq.p; q->body=body;
        EnterCriticalSection(&g_sqc); if(g_sqt) g_sqt->next=q; else g_sqh=q; g_sqt=q; LeaveCriticalSection(&g_sqc);
        SetEvent(g_sqe); } }
static SQ* sq_pop(void){ EnterCriticalSection(&g_sqc); SQ*q=g_sqh; if(q){ g_sqh=q->next; if(!g_sqh) g_sqt=NULL; } LeaveCriticalSection(&g_sqc); return q; }
static void sq_free(SQ*q){ free(q->rq); free(q); }
/* recompute path (CPU build, legacy ngram, CUDALM_SERVE_RECOMPUTE): full-context forward per token, "n" samples in lockstep */
static void sv_lockstep(const SReq*r,const float*P,const Off*o,GA*a,int*tokG,const Cfg*cp,const Tok*tk,int nmax,int*pad,float*lgs,int*ord,SBuf*out){
    Cfg c=*cp; int B=r->n, npred=r->npred, greedy=r->greedy; float temp=r->temp,topp=r->topp; int topk=r->topk;
    if(r->sd>=0) g_rng=(uint64_t)r->sd*0x9E3779B97F4A7C15ULL+1;
    int pn; int*ids=tok_encode((Tok*)tk,r->prompt,&pn); int cap=pn+npred+1, slen=pn;
    int*seq=(int*)malloc(sizeof(int)*(size_t)B*cap);
    for(int b=0;b<B;b++) memcpy(seq+(size_t)b*cap,ids,sizeof(int)*pn);
    SBuf*gen=(SBuf*)calloc(B,sizeof(SBuf)); int*prod=(int*)calloc(B,sizeof(int)),*st=(int*)malloc(sizeof(int)*B),*done=(int*)calloc(B,sizeof(int));
    double*nll=(double*)calloc(B,sizeof(double)); int ndone=0;
    for(int b=0;b<B;b++){ sb_put(&gen[b],"",0); st[b]=2; }
    for(int s=0;s<npred&&slen>0&&ndone<B;s++){
        int ctx=slen<c.T?slen:c.T, off=slen-ctx, Tr=(ctx+15)&~15; if(Tr>c.T) Tr=c.T;
        for(int b=0;b<B;b++) for(int j=0;j<Tr;j++) pad[b*Tr+j]=j<ctx?seq[(size_t)b*cap+off+j]:0;
        ck(cudaMemcpy(tokG,pad,(size_t)B*Tr*4,cudaMemcpyHostToDevice),"tok h2d");
        a->N=B*Tr; gC.T=Tr;
        gpu_forward(P,o,a,tokG);
        for(int b=0;b<B;b++) g_abt(a->fn+((size_t)b*Tr+ctx-1)*c.D,P+o->emb,a->logits+(size_t)b*c.V,1,c.D,c.V);
        a->N=nmax*c.T; gC.T=c.T;
        ck(cudaMemcpy(lgs,a->logits,(size_t)B*c.V*4,cudaMemcpyDeviceToHost),"logits d2h");
        for(int b=0;b<B;b++){
            float*lg=lgs+(size_t)b*c.V; int nxt=0;
            if(done[b]){ seq[(size_t)b*cap+slen]=0; continue; }        /* keeps the batch in lockstep */
            float it=greedy?1.0f:1.0f/temp, mx=-1e30f; for(int i=0;i<c.V;i++){ lg[i]*=it; if(lg[i]>mx) mx=lg[i]; }
            double z=0; for(int i=0;i<c.V;i++) z+=exp((double)lg[i]-mx);
            if(greedy){ for(int i=1;i<c.V;i++) if(lg[i]>lg[nxt]) nxt=i; }
            else {
                int K=topk>0&&topk<c.V?topk:c.V;
                for(int i=0;i<c.V;i++) ord[i]=i;
                for(int i=0;i<K;i++){ int m=i; for(int j=i+1;j<c.V;j++) if(lg[ord[j]]>lg[ord[m]]) m=j; int t=ord[i]; ord[i]=ord[m]; ord[m]=t;
                    if(K==c.V&&i>=255) { K=256; break; } }
                double tot=0; int kk=0; for(;kk<K;kk++){ tot+=exp((double)lg[ord[kk]]-mx); if(tot>=topp*z){ kk++; break; } }
                double rr=urand()*tot,ac=0; nxt=ord[kk-1];
                for(int i=0;i<kk;i++){ ac+=exp((double)lg[ord[i]]-mx); if(ac>=rr){ nxt=ord[i]; break; } } }
            nll[b]-=(double)lg[nxt]-mx-log(z);
            seq[(size_t)b*cap+slen]=nxt;
            if(nxt==tk->eot){ st[b]=0; done[b]=1; ndone++; continue; }
            prod[b]++; if(sv_emit(&gen[b],tk,nxt,r)){ st[b]=1; done[b]=1; ndone++; }
        }
        slen++;
        if(slen>=cap) break;
    }
    sv_results(out,B,gen,prod,st,nll,pn);
    for(int b=0;b<B;b++) free(gen[b].p);
    free(gen); free(prod); free(st); free(done); free(nll); free(ids); free(seq); }
/* POST /logprob {"context"|"prompt", "continuation":".." | "continuations":[..], "bos":1, "tokens":0, "top_logprobs":0}:
 * log P(continuation | context) per continuation, scored like lmeval (eot + context + continuation, encoded separately, context
 * cut from the left to fit T; FP8 off so scores do not depend on padding) -> {"context_tokens","results":[{"logprob","tokens",
 * "bytes","is_greedy"[,"token_logprobs":[{"token","logprob"[,"top":[["tok",lp],..]]}]]}]}. No continuation: every context token
 * is scored (rolling loglikelihood / perplexity). Handled between decode steps; ~1 forward per continuation. */
static void sv_logprob(const char*body,const float*P,const Off*o,GA*a,int*tokG,const Cfg*cp,const Tok*tk,int rows,SBuf*out){
    Cfg c=*cp; int V=c.V; const char*pp=js_key(body,"context"); if(!pp) pp=js_key(body,"prompt");
    char*ctx=pp&&*pp=='"'?js_str(pp+1,NULL):strdup(""), *cs[64]; int nk=0;
    const char*sp=js_key(body,"continuations"); if(!sp) sp=js_key(body,"continuation");
    if(sp&&*sp=='"') cs[nk++]=js_str(sp+1,NULL);
    else if(sp&&*sp=='[') for(sp++;*sp&&*sp!=']'&&nk<64;){ if(*sp=='"') cs[nk++]=js_str(sp+1,&sp); else sp++; }
    int bos=js_num(body,"bos",1)!=0, ntop=(int)js_num(body,"top_logprobs",0), det=js_num(body,"tokens",0)!=0||ntop>0, roll=!nk;
    if(ntop>20) ntop=20; if(roll) cs[nk++]=strdup("");
    int cn; int*ci=tok_encode((Tok*)tk,ctx,&cn), ch=rows<g_chrows?rows:g_chrows, sN=a->N, sT=gC.T, f8=g_fp8gemm; g_fp8gemm=0;
    float*lg=(float*)malloc((size_t)ch*V*4); int*pad=(int*)calloc(c.T,4); char t[200];
    snprintf(t,sizeof t,"{\"context_tokens\":%d,\"results\":[",cn); sb_str(out,t);
    for(int k=0;k<nk;k++){
        int kn; int*ki=tok_encode((Tok*)tk,cs[k],&kn), L=0, *s=(int*)malloc(4*(size_t)(1+cn+kn));
        if(bos) s[L++]=tk->eot; memcpy(s+L,ci,4*(size_t)cn); L+=cn; int s0=roll?1:L; memcpy(s+L,ki,4*(size_t)kn); L+=kn; free(ki);
        if(L-1>c.T){ int d=L-1-c.T; memmove(s,s+d,4*(size_t)(L-d)); L-=d; s0-=d; }
        if(s0<1) s0=1;
        double sum=0; int gr=1, n=L>s0?L-s0:0;
        sb_str(out,k?",{":"{");
        if(det) sb_str(out,"\"token_logprobs\":[");
        if(n){ int Tr=(L-1+15)&~15; if(Tr>c.T) Tr=c.T;
            for(int j=0;j<Tr;j++) pad[j]=j<L-1?s[j]:0;
            ck(cudaMemcpy(tokG,pad,(size_t)Tr*4,cudaMemcpyHostToDevice),"tok h2d");
            a->N=Tr; gC.T=Tr; gpu_forward(P,o,a,tokG);
            for(int j0=s0-1;j0<L-1;j0+=ch){ int m=L-1-j0<ch?L-1-j0:ch;     /* row j predicts s[j+1] */
                g_abt(a->fn+(size_t)j0*c.D,P+o->emb,a->logits,m,c.D,V);
                ck(cudaMemcpy(lg,a->logits,(size_t)m*V*4,cudaMemcpyDeviceToHost),"logits d2h");
                for(int r=0;r<m;r++){ const float*z=lg+(size_t)r*V; int y=s[j0+r+1], bi=0; double se=0;
                    for(int v=1;v<V;v++) if(z[v]>z[bi]) bi=v;
                    for(int v=0;v<V;v++) se+=exp((double)z[v]-z[bi]);
                    double lse=z[bi]+log(se), lp=z[y]-lse; sum+=lp; gr&=bi==y;
                    if(!det) continue;
                    sb_str(out,j0+r>s0-1?",{\"token\":\"":"{\"token\":\""); sb_esc(out,(const char*)tk->dec[y],(size_t)tk->declen[y]);
                    snprintf(t,sizeof t,"\",\"logprob\":%.6f",lp); sb_str(out,t);
                    if(ntop>0){ int tp[20]; sb_str(out,",\"top\":[");
                        for(int q=0;q<ntop;q++){ int b=-1; for(int v=0;v<V;v++){ int u=0; for(int w=0;w<q;w++) u|=tp[w]==v;
                                if(!u&&(b<0||z[v]>z[b])) b=v; }
                            tp[q]=b; sb_str(out,q?",[\"":"[\""); sb_esc(out,(const char*)tk->dec[b],(size_t)tk->declen[b]);
                            snprintf(t,sizeof t,"\",%.6f]",z[b]-lse); sb_str(out,t); }
                        sb_str(out,"]"); }
                    sb_str(out,"}"); } }
            a->N=sN; gC.T=sT; }
        snprintf(t,sizeof t,"%s\"logprob\":%.6f,\"tokens\":%d,\"bytes\":%d,\"is_greedy\":%s}",det?"],":"",sum,n,
                 (int)strlen(roll?ctx:cs[k]),n&&gr?"true":"false"); sb_str(out,t);
        free(s); free(cs[k]); }
    sb_str(out,"]}"); g_fp8gemm=f8; free(lg); free(pad); free(ci); free(ctx); }
#ifndef CUDALM_CPU
/* continuous batching: every request ("n" samples each) takes slots of one Dec; new requests are prefilled into
 * free slots between steps, finished slots are compacted away, each step decodes all live slots together */
typedef struct { SQ*q; SReq r; int pn,left; SBuf*gen; int*prod,*st; double*nll; } BReq;
typedef struct { BReq*r; int k,cnt,done; } Slot;
static void breq_done(BReq*R){
    SBuf out={0}; sv_results(&out,R->r.n,R->gen,R->prod,R->st,R->nll,R->pn); sv_reply(R->q->cs,"200 OK",&out); free(out.p);
    for(int b=0;b<R->r.n;b++) free(R->gen[b].p);
    free(R->gen); free(R->prod); free(R->st); free(R->nll); sreq_free(&R->r); sq_free(R->q); free(R); }
static void sv_batched(Dec*dc,const float*P,const Off*o,GA*a,int*tokG,const Cfg*cp,const Tok*tk,int ns){
    Cfg c=*cp; Slot*sl=(Slot*)calloc(ns,sizeof(Slot)); int N=0,quit=0; BReq*pr=NULL; SQ*bye=NULL;
    long long ntok=0,nreq=0; double t0=wall(),tl=t0;
    for(;;){
        while(!quit){                                     /* admit */
            if(!pr){ SQ*q=sq_pop(); if(!q) break;
                if(!strncmp(q->rq,"POST /shutdown",14)){ quit=1; bye=q; break; }
                if(!strncmp(q->rq,"POST /logprob",13)){ SBuf out={0}; sv_logprob(q->body,P,o,a,tokG,&c,tk,c.T,&out);
                    sv_reply(q->cs,"200 OK",&out); free(out.p); sq_free(q); continue; }
                if(strncmp(q->rq,"POST /completion",16)){ sv_replys(q->cs,"404 Not Found","{\"error\":\"unknown endpoint\"}"); sq_free(q); continue; }
                BReq*R=(BReq*)calloc(1,sizeof(BReq)); R->q=q; sreq_parse(q->body,&R->r,c.T,ns); int B=R->r.n;
                R->gen=(SBuf*)calloc(B,sizeof(SBuf)); R->prod=(int*)calloc(B,4); R->st=(int*)malloc(B*4); R->nll=(double*)calloc(B,8);
                for(int b=0;b<B;b++){ sb_put(&R->gen[b],"",0); R->st[b]=2; }
                pr=R; }
            if(N+pr->r.n>ns) break;
            BReq*R=pr; pr=NULL; int pn; int*ids=tok_encode((Tok*)tk,R->r.prompt,&pn);
            int keep=c.T-R->r.npred; if(keep<c.T/2) keep=c.T/2; if(pn>keep){ memmove(ids,ids+pn-keep,sizeof(int)*keep); pn=keep; }
            R->pn=pn; R->left=R->r.n; nreq++;
            if(pn<1||R->r.npred<1){ free(ids); breq_done(R); continue; }
            unsigned long long base=R->r.sd>=0?(unsigned long long)R->r.sd:(unsigned long long)urand64();
            for(int b=0;b<R->r.n;b++){ DSP q={R->r.greedy?0.f:R->r.temp,R->r.topp,R->r.topk,base+(unsigned long long)b*7919ULL,0};
                dc->hsp[N+b]=q; Slot s={R,b,0,0}; sl[N+b]=s; }
            dec_prefill(dc,P,o,a,tokG,ids,pn,N,R->r.n); if(R->r.numf) num_mask(a,tk); dec_sample_rows(dc,a,N,R->r.n); N+=R->r.n; free(ids); }
        if(!N){ if(quit) break; WaitForSingleObject(g_sqe,50); continue; }
        for(int b=0;b<N;b++){                             /* consume the sampled tokens */
            Slot*s=&sl[b]; BReq*R=s->r; int k=s->k, nxt=dc->hotok[b], fin=0; R->nll[k]-=dc->holp[b]; s->cnt++; ntok++;
            if(nxt==tk->eot){ R->st[k]=0; fin=1; }
            else { R->prod[k]++; if(sv_emit(&R->gen[k],tk,nxt,&R->r)){ R->st[k]=1; fin=1; } }
            if(!fin&&(s->cnt>=R->r.npred||dc->hpos[b]>=c.T)) fin=1;
            if(fin){ s->done=1; if(--R->left==0) breq_done(R); }
            else { dc->htok[b]=nxt; dc->hsp[b].step=s->cnt; } }
        int w=0; for(int b=0;b<N;b++) if(!sl[b].done){ if(w!=b){ dec_move(dc,b,w); sl[w]=sl[b]; } w++; }
        N=w;
        double tn=wall(); if(tn-tl>=30){ printf("[serve] %lld req, %lld tok, %.0f tok/s, %d live\n",nreq,ntok,ntok/(tn-t0),N); fflush(stdout); tl=tn; }
        if(N) dec_step(dc,P,o,a,N,1); }
    if(pr) breq_done(pr);
    sv_replys(bye->cs,"200 OK","{\"status\":\"bye\"}"); sq_free(bye); free(sl); }
#endif
static void cmd_serve(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: cudalm serve <name|model.bin> [port=8090]\n"); exit(1); }
    char path[1024]; model_path(argv[2],path,sizeof(path));
    FILE*e=fopen(path,"rb"); if(e)fclose(e); else snprintf(path,sizeof(path),"%s",argv[2]);
    int port=argc>3?atoi(argv[3]):8090;
    Cfg c; Tok*tk=NULL; float*Ph=model_load_full(path,&c,&tk);
    if(!tk){ fprintf(stderr,"model has no tokenizer\n"); exit(1); }
    gC=c; cuda_init(); Off o; calc_offsets(&c,&o); ngram_infer(path);
    size_t np=param_count(&c);
    float*P=(float*)dmalloc(np*4); ck(cudaMemcpy(P,Ph,np*4,cudaMemcpyHostToDevice),"P h2d");
    rope_tables(&c);
    int nmax=8; { const char*v=getenv("CUDALM_SERVE_MAXN"); if(v&&atoi(v)>0) nmax=atoi(v); }
    int dec=0;
#ifndef CUDALM_CPU
    dec=dec_ok()&&!getenv("CUDALM_SERVE_RECOMPUTE");
    int ns=64; { const char*v=getenv("CUDALM_SERVE_SLOTS"); if(v&&atoi(v)>0) ns=atoi(v); }
#endif
    int rows=dec?c.T:nmax*c.T;
    GA*a=ga_new(rows,0); int*tokG=(int*)dmalloc((size_t)rows*4);
    int*pad=(int*)calloc(rows,sizeof(int)); ck(cudaMemcpy(tokG,pad,(size_t)rows*4,cudaMemcpyHostToDevice),"tok0");
    gpu_forward(P,&o,a,tokG);                 /* warm-up at full size: lazily sized buffers get their max */
    ws_load();
    SK ls=W.sock(2,1,6); struct { short f; unsigned short p; unsigned a; char z[8]; } sa={2,(unsigned short)((port>>8)|((port&255)<<8)),0x0100007f,{0}};
    if(W.bind_(ls,&sa,16)||W.listen_(ls,256)){ fprintf(stderr,"[serve] cannot bind 127.0.0.1:%d\n",port); exit(1); }
    InitializeCriticalSection(&g_sqc); g_sqe=CreateEventA(NULL,FALSE,FALSE,NULL); g_sqls=ls;
    CreateThread(NULL,0,sv_accept,NULL,0,NULL);
#ifndef CUDALM_CPU
    if(dec){ if(ns>c.T) ns=c.T; Dec*dc=dec_new(ns,c.T);
        printf("[serve] %s on http://127.0.0.1:%d  (V=%d T=%d), KV-cache decoder, %d slots, continuous batching\n",path,port,c.V,c.T,ns); fflush(stdout);
        sv_batched(dc,P,&o,a,tokG,&c,tk,ns); printf("[serve] shutdown\n"); fflush(stdout); W.close_(ls); return; }
#endif
    printf("[serve] %s on http://127.0.0.1:%d  (V=%d T=%d), recompute, one request at a time\n",path,port,c.V,c.T); fflush(stdout);
    float*lgs=(float*)malloc((size_t)nmax*c.V*4); int*ord=(int*)malloc(sizeof(int)*c.V);
    for(;;){
        SQ*q=sq_pop(); if(!q){ WaitForSingleObject(g_sqe,50); continue; }
        if(!strncmp(q->rq,"POST /completion",16)){ SReq r; sreq_parse(q->body,&r,c.T,nmax); SBuf out={0};
            sv_lockstep(&r,P,&o,a,tokG,&c,tk,nmax,pad,lgs,ord,&out); sv_reply(q->cs,"200 OK",&out); free(out.p); sreq_free(&r); }
        else if(!strncmp(q->rq,"POST /logprob",13)){ SBuf out={0}; sv_logprob(q->body,P,&o,a,tokG,&c,tk,rows,&out);
            sv_reply(q->cs,"200 OK",&out); free(out.p); }
        else if(!strncmp(q->rq,"POST /shutdown",14)){ sv_replys(q->cs,"200 OK","{\"status\":\"bye\"}"); sq_free(q);
            printf("[serve] shutdown\n"); fflush(stdout); W.close_(ls); return; }
        else sv_replys(q->cs,"404 Not Found","{\"error\":\"unknown endpoint\"}");
        sq_free(q); }
}
#else
static void cmd_serve(int argc,char**argv){ (void)argc;(void)argv; fprintf(stderr,"serve: Windows only\n"); exit(1); }
#endif

/* ===================== train ============================================ */
/* ---- prompt-loss masking (CUDALM_MASK_PROMPT=1) --------------------------
 * By default every token in the window contributes loss, so the model is
 * explicitly trained to produce the USER's turns too. That is exactly why it
 * emits <|user|> and then answers its own question -- it is reproducing what it
 * was taught. Masking restricts the loss to assistant-generated spans.
 *
 * The corpus is a flat token stream with the chat markers stored as PLAIN TEXT,
 * so turn boundaries are recovered by matching the markers' token sequences
 * rather than by any stored structure. That means no corpus rebuild.
 *
 * The mix is ~43% raw prose (FineWeb, code) with no markers at all, and that is
 * pretraining data we DO want every token of. So the state machine starts
 * trainable and only masks after seeing a user-side marker; eot resets it. A
 * window straddling a prose/chat boundary is therefore handled correctly.
 *
 * Only TRAINING is masked. `eval` builds its own unmasked targets, so every val
 * loss stays comparable with the numbers measured before this existed. */
#define MASK_MARKS 7
typedef struct { int*ids[MASK_MARKS]; int len[MASK_MARKS]; int kind[MASK_MARKS]; int n; } MaskSpec;

static void mask_build(MaskSpec*ms,const Tok*tk){
    /* kind 0 : <|assistant|>, supplied by the prompt, a reply follows
       kind 1 : <|call|>, emitted by the model mid-reply
       kind 2 : user-side, supplied; content after it is never generated
       kind 4 : <|result|>, supplied by the tool, up to and including its <|end|> (kind 3); the reply then resumes */
    static const char*txt[MASK_MARKS] = {"<|assistant|>","<|call|>",
                                         "<|user|>","<|system|>","<|tools|>","<|result|>","<|end|>"};
    static const int  tr [MASK_MARKS] = { 0, 1, 2, 2, 2, 4, 3 };
    ms->n=0;
    for(int i=0;i<MASK_MARKS;i++){
        int n=0; int*ids=tok_encode(tk,txt[i],&n);
        if(n<=0){ free(ids); continue; }
        ms->ids[ms->n]=ids; ms->len[ms->n]=n; ms->kind[ms->n]=tr[i]; ms->n++;
    }
}
static int mask_match(const MaskSpec*ms,const int*tok,int i,int n){
    for(int m=0;m<ms->n;m++){ int L=ms->len[m];
        if(i+L<=n && !memcmp(tok+i,ms->ids[m],(size_t)L*sizeof(int))) return m; }
    return -1;
}
/* Mask tgt[] in place for one sequence of `n` tokens; returns how many positions
 * remain trainable. TARGET-aligned: tgt[j-1] (predicting tok[j]) is masked iff tok[j]
 * lies in a supplied span. A user-side marker right after a reply is trainable, so the
 * model learns "\n<|user|>" as end-of-turn (multi-turn replies end that way, not with
 * eot). The window prefix before the first marker/eot is user text iff that first
 * marker is <|assistant|>; otherwise it is prose or a reply and stays trainable. */
static int mask_apply(const MaskSpec*ms,const int*tok,int*tgt,int n,int eot){
    int st=0, inres=0, masked=0;                  /* st: 1 = inside a supplied span; inres: that span is a tool result */
    /* window prefix: inside a prompt iff the next <|assistant|>/user-side marker is <|assistant|> (tool markers
       can be quoted inside a system prompt); otherwise a lone <|end|> first closes a result unless a result follows */
    for(int i=0;i<n;i++){ if(tok[i]==eot) break;
        int m=mask_match(ms,tok,i,n); if(m<0) continue;
        int k=ms->kind[m];
        if(k==1||k==3||k==4){ int nx=-1;
            for(int j=i;j<n&&tok[j]!=eot;j++){ int q=mask_match(ms,tok,j,n); if(q>=0&&(ms->kind[q]==0||ms->kind[q]==2)){ nx=q; break; } }
            if(nx>=0&&ms->kind[nx]==0){ st=1; break; }
            if(k==3){ nx=-1; for(int j=i+ms->len[m];j<n&&tok[j]!=eot&&nx<0;j++) nx=mask_match(ms,tok,j,n);
                if(nx<0||ms->kind[nx]!=4) st=inres=1; } }
        else st=k==0;
        break; }
    for(int i=0;i<n;i++){
        int m, L=1, sm;                           /* sm: state of tok[i..i+L-1] */
        if(tok[i]==eot){ sm=0; st=0; inres=0; }
        else if((m=mask_match(ms,tok,i,n))>=0){ L=ms->len[m];
            switch(ms->kind[m]){ case 0: sm=1; st=0; inres=0; break;
                                 case 1: if(st) sm=1; else { sm=0; inres=0; } break;   /* quoted inside a prompt/result: ignored */
                                 case 3: sm=st; if(inres){ st=0; inres=0; } break;
                                 case 4: sm=st; if(!st){ st=1; inres=1; } break;
                                 default: sm=st; st=1; inres=0; } }
        else sm=st;
        if(sm) for(int j=i;j<i+L&&j<n;j++) if(j>0&&tgt[j-1]>=0){ tgt[j-1]=-1; masked++; }
        i+=L-1;
    }
    if(st&&tgt[n-1]>=0){ tgt[n-1]=-1; masked++; }
    return n-masked;
}

/* Peak LR for long runs. Short sweeps favour hotter peaks than long runs survive: D>=384 swept flat 5e-4..1e-3 at
 * 3k it, but s8w diverged at 1e-3/7e-4 (LRS settled at ~9e-5) and mod230 collapsed at 7e-4 (3.5e-4 stable);
 * LS-Micro (D192) at 3e-4 grew gn <1 -> 40 with NaNs late in its epoch. */
/* table = QLM rates. Without QLM the 2026-10-24 sweep (D=192, 262M tokens) put the optimum at 6x (Muon, 1.2e-3)
 * and 24x (fp32 AdamW, 4.8e-3); other widths are scaled by the same factor, untested. */
static float default_lr_for(const Cfg*c){ float m=g_qlm_on?1.0f:g_muon?6.0f:24.0f;
    if(c->D<=64)return 4e-4f*m; if(c->D<=96)return 3e-4f*m; if(c->D<=192)return 2e-4f*m;
    if(c->D<=256)return 1.75e-4f*m; return 3e-4f*m; }
/* A new stage on a trained model (CPT/SFT) re-warms out of a cooled-down minimum: LS-Micro-Base CPT at 1.5e-4
 * blew up (gn 30 -> 400+ in 5k it, loss rising), 7.5e-5 held. Default to a third of the from-scratch peak. */
#define LR_STAGE_MUL (1.0f/3.0f)
/* pick B so activations fit in the free VRAM (leaving room for P/G/M/V) */
static int auto_batch(const Cfg*c,size_t np){
    const char*e=getenv("CUDALM_BATCH"); if(e){ int v=atoi(e); if(v>=1)return v; }
    size_t fr=0,tt=0; cudaMemGetInfo(&fr,&tt);
    size_t state=opt_state_bytes(np);                 /* P + G fp32, m + v bf16 */
    /* Reserve headroom for the DESKTOP, not just for us. This card also drives
     * the display: a browser or a Parsec session can claim several hundred MB
     * at any moment, and once WDDM runs out it pages GPU allocations over PCIe,
     * which is slow enough to trip the TDR watchdog and take the driver down.
     * The old 512 MB reserve left ~1.1 GB spare on an *idle* desktop (81% VRAM
     * used) -- fine unattended, fatal while the machine was in use.
     * Override with CUDALM_VRAM_RESERVE_MB. */
    size_t resv=(size_t)1200e6;
    { const char*r=getenv("CUDALM_VRAM_RESERVE_MB"); if(r){ long v=atol(r); if(v>=0) resv=(size_t)v*1000000; } }
    if(resv < tt/8) resv = tt/8;                      /* at least 1/8 of the card */
    size_t avail=(fr>state+resv)?(fr-state-resv):(size_t)256e6;
    /* Size the batch as if the stash were off. It is an optional ~9% FLOP saving
     * and must never cost a batch halving, which is worth far more. */
    int save=g_moe_stash; g_moe_stash=0;
    int B=32; for(;B>=1;B--){ if(ga_bytes(B*c->T,1)<=avail) break; }
    g_moe_stash=save;
    return B<1?1:B;
}
/* Called once B is fixed: keep the stash only if it fits in what is left. */
static void moe_stash_decide(const Cfg*c,size_t np,int N){
    if(!g_moe_stash) return;
    if(c->n_exp<=0){ g_moe_stash=0; return; }
    size_t fr=0,tt=0; cudaMemGetInfo(&fr,&tt);
    size_t resv=(size_t)1200e6;
    { const char*r=getenv("CUDALM_VRAM_RESERVE_MB"); if(r){ long v=atol(r); if(v>=0) resv=(size_t)v*1000000; } }
    if(resv < tt/8) resv = tt/8;
    size_t state=opt_state_bytes(np), avail=(fr>state+resv)?(fr-state-resv):(size_t)256e6;
    size_t want=ga_bytes(N,1);
    if(want>avail){
        g_moe_stash=0;
        printf("[cudalm] MoE stash OFF: needs %.2f GB, %.2f GB available -- backward will recompute\n",
               want/1e9, avail/1e9);
    } else {
        size_t ND=(size_t)N*c->D, NF=(size_t)N*c->F;
        printf("[cudalm] MoE stash ON: %.2f GB buys back the w1/w3/w2 recompute (~9%% of step FLOPs)\n",
               (size_t)c->L*(2*NF+ND)*4.0*(g_topk<1?1:g_topk)/1e9);
    }
}
/* runs a helper command line (python scripts) in this console and returns its exit code */
static int run_cmd(const char*cmd){
    printf("> %s\n",cmd); fflush(stdout);
#ifdef _WIN32
    static char cl[8192]; snprintf(cl,sizeof cl,"%s",cmd);
    STARTUPINFOA si; PROCESS_INFORMATION pi; memset(&si,0,sizeof si); si.cb=sizeof si; DWORD code=1;
    if(!CreateProcessA(NULL,cl,NULL,NULL,TRUE,0,NULL,NULL,&si,&pi)){ fprintf(stderr,"cannot start: %s (%lu) - set CUDALM_PYTHON\n",cmd,GetLastError()); return 1; }
    WaitForSingleObject(pi.hProcess,INFINITE); GetExitCodeProcess(pi.hProcess,&code); CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
    return (int)code;
#else
    int r=system(cmd); return r==-1?1:WEXITSTATUS(r);
#endif
}
static const char* py_cmd(void){ const char*p=getenv("CUDALM_PYTHON"); return p&&*p?p:
#ifdef _WIN32
    "py -3"; }
#else
    "python3"; }
#endif
#include "data.cuh"
/* lmeval per-item dumps (CUDALM_LMEVAL_ITEMS): task \t item \t correct \t margin. Margin = logP/byte of the gold
 * continuation minus the best wrong one (MC), or logP/byte of the target (LAMBADA). le_vs pairs two dumps item by item. */
typedef struct { char t[32]; int n; float*m; signed char*c; } LeT;
static int le_load(const char*p,LeT*T){
    FILE*f=fopen(p,"rb"); if(!f) return -1; char l[256]; int nt=0;
    while(fgets(l,sizeof l,f)){ char tn[32]; int it,c; float m;
        if(sscanf(l,"%31[^\t]\t%d\t%d\t%f",tn,&it,&c,&m)!=4||it<0) continue;
        int k=0; while(k<nt&&strcmp(T[k].t,tn)) k++;
        if(k==nt){ if(nt==16) continue; memset(&T[nt],0,sizeof *T); snprintf(T[nt].t,32,"%s",tn); nt++; }
        LeT*x=&T[k]; if(it>=x->n){ int nn=it+1>2*x->n?it+1:2*x->n;
            x->m=(float*)realloc(x->m,4*(size_t)nn); x->c=(signed char*)realloc(x->c,nn); memset(x->c+x->n,-1,nn-x->n); x->n=nn; }
        x->m[it]=m; x->c[it]=(signed char)(c!=0); }
    fclose(f); return nt; }
/* prints the paired comparison B - A; returns z of the margin-index change (NAN if it has < 2 components) */
static double le_vs(const char*pa,const char*pb,const char*pre,double*dix){
    LeT A[16],B[16]; int na=le_load(pa,A),nb=le_load(pb,B); double z=NAN;
    if(na<0||nb<0){ printf("%scannot read %s\n",pre,na<0?pa:pb); goto out; }
    {   static const char*ixn[5]={"hellaswag","arc_easy","arc_challenge","piqa","arithmark"};
        double dm[5],ds[5]; int h[5]={0};
        for(int i=0;i<na;i++){ int j=0; while(j<nb&&strcmp(A[i].t,B[j].t)) j++; if(j==nb) continue;
            int n=0,up=0,dn=0,L=A[i].n<B[j].n?A[i].n:B[j].n; double s=0,q=0;
            for(int k=0;k<L;k++) if(A[i].c[k]>=0&&B[j].c[k]>=0){ double d=B[j].m[k]-A[i].m[k]; s+=d; q+=d*d; n++;
                up+=B[j].c[k]>A[i].c[k]; dn+=B[j].c[k]<A[i].c[k]; }
            if(!n) continue; double mu=s/n, se=n>1?sqrt((q/n-mu*mu)/(n-1)):0;
            printf("%s%-13s %6d   acc %+6.2f%% (+%d/-%d)   margin %+.4f +-%.4f   z %+5.1f\n",pre,A[i].t,n,100.0*(up-dn)/n,up,dn,mu,se,se>0?mu/se:0);
            for(int q2=0;q2<5;q2++) if(!strcmp(A[i].t,ixn[q2])){ dm[q2]=mu; ds[q2]=se; h[q2]=1; } }
        double w[4]={1,1,1,0.65},m4[4],s4[4],num=0,den=0,var=0; int h4[4]={h[0],h[1]||h[2],h[3],h[4]},nc=0;
        m4[0]=dm[0]; s4[0]=ds[0]; m4[2]=dm[3]; s4[2]=ds[3]; m4[3]=dm[4]; s4[3]=ds[4];
        if(h[1]&&h[2]){ m4[1]=(dm[1]+dm[2])/2; s4[1]=0.5*sqrt(ds[1]*ds[1]+ds[2]*ds[2]); }
        else if(h4[1]){ m4[1]=h[1]?dm[1]:dm[2]; s4[1]=h[1]?ds[1]:ds[2]; }
        for(int q2=0;q2<4;q2++) if(h4[q2]){ num+=w[q2]*m4[q2]; var+=w[q2]*w[q2]*s4[q2]*s4[q2]; den+=w[q2]; nc++; }
        if(nc>=2){ double d=100*num/den, se=100*sqrt(var)/den; z=se>0?d/se:0; if(dix) *dix=d;
            printf("%smargin index %+.3f +-%.3f   z %+.1f   -> %s\n",pre,d,se,z,fabs(z)<2?"no clear change (|z| < 2)":z>0?"better":"worse"); }
    }
out:for(int i=0;i<na;i++){ free(A[i].m); free(A[i].c); } for(int i=0;i<nb;i++){ free(B[i].m); free(B[i].c); }
    return z; }
static void cmd_lmeval_vs(int argc,char**argv){
    if(argc<4){ fprintf(stderr,"usage: cudalm lmeval-vs <A.items.tsv> <B.items.tsv>\n"
        "  paired comparison of two per-item dumps (CUDALM_LMEVAL_ITEMS=<file> on lmeval, or <model>.items/ from CUDALM_BENCH_EVERY):\n"
        "  per task the accuracy flips and the mean margin change B - A +- paired SE; then the margin-index change and its z\n"); exit(1); }
    printf("[lmeval-vs] B = %s\n           A = %s\n",argv[3],argv[2]); le_vs(argv[2],argv[3],"  ",NULL); }
/* CUDALM_BENCH_EVERY=N: every N steps of the stage, score the checkpoint just saved with `lmeval` in a
 * child process (training waits; the time is left out of tok/s) and append a row to <model>.bench.tsv.
 * CUDALM_BENCH_TASKS (default all tasks), CUDALM_BENCH_N items per task (default 0 = full sets). */
static double train_bench(const char*mpath,int it,double ep){
    const char*ts=getenv("CUDALM_BENCH_TASKS"),*nn=getenv("CUDALM_BENCH_N"); double t=wall(),ix=NAN;
    char cmd[3200],ln[600],res[800]="",tsv[1200],exe[1024],idir[1100],ipath[1200]; int lo=0,sum=0; double mi=NAN;
    static char prev[1200];      /* previous bench's per-item dump: each bench is compared against it */
    snprintf(idir,sizeof idir,"%s.items",mpath); lm_mkdir(idir);
    snprintf(ipath,sizeof ipath,"%s" PSEP "it%d.tsv",idir,it); lm_setenv("CUDALM_LMEVAL_ITEMS",ipath);
#ifdef _WIN32
    GetModuleFileNameA(0,exe,sizeof exe);
    snprintf(cmd,sizeof cmd,"\"\"%s\" lmeval \"%s\" %s %s 2>&1\"",exe,mpath,ts&&*ts?ts:"all",nn&&*nn?nn:"0");
    FILE*p=_popen(cmd,"r");
#else
    snprintf(cmd,sizeof cmd,"/proc/%d/exe lmeval \"%s\" %s %s 2>&1",(int)getpid(),mpath,ts&&*ts?ts:"all",nn&&*nn?nn:"0");
    FILE*p=popen(cmd,"r");
#endif
    (void)exe;
    printf("[bench] iter %d (ep %.2f): lmeval on %s ...\n",it,ep,mpath); fflush(stdout);
    if(!p){ printf("[bench] could not start lmeval - skipped\n"); return wall()-t; }
    while(fgets(ln,sizeof ln,p)){ char tn[64]; const char*a=strstr(ln,"acc_norm"),*x=strstr(ln,"intelligence index ");
        if(strstr(ln,"[lmeval] summary")) sum=1;
        else if(sum&&a&&sscanf(ln," %63s",tn)==1) lo+=snprintf(res+lo,sizeof res-lo,"%s%s %.2f",lo?"  ":"",tn,atof(a+8));
        else if(sum&&strstr(ln,"ppl")&&sscanf(ln," %63s",tn)==1) lo+=snprintf(res+lo,sizeof res-lo,"%s%s %.2f",lo?"  ":"",tn,atof(strstr(ln,"acc")+3));
        if(x) ix=atof(x+19);
        if((x=strstr(ln,"margin index "))) mi=atof(x+13); }
    lm_setenv("CUDALM_LMEVAL_ITEMS",NULL);
#ifdef _WIN32
    _pclose(p);
#else
    pclose(p);
#endif
    if(!lo){ printf("[bench] lmeval produced no scores (out of memory? missing bench files?) - skipped\n"); fflush(stdout); return wall()-t; }
    printf("[bench] iter %d (ep %.2f): index %.2f  margin index %.3f | %s  (%.0fs)\n",it,ep,ix,mi,res,wall()-t);
    if(!*prev){ /* resumed run: the newest older dump in <model>.items */
        int bi=-1; for(int k=it-1;k>0&&bi<0;k--){ char q[1200]; snprintf(q,sizeof q,"%s" PSEP "it%d.tsv",idir,k); if(lm_exists(q)){ bi=k; snprintf(prev,sizeof prev,"%s",q); } } }
    if(*prev&&lm_exists(ipath)){ printf("[bench] vs %s:\n",prev); le_vs(prev,ipath,"[bench]   ",NULL); }
    if(lm_exists(ipath)) snprintf(prev,sizeof prev,"%s",ipath);
    fflush(stdout);
    snprintf(tsv,sizeof tsv,"%s.bench.tsv",mpath); int nw=!lm_exists(tsv); FILE*f=fopen(tsv,"a");
    if(f){ if(nw) fprintf(f,"iter\tepoch\tindex\tscores (acc_norm %%; lambada acc %%)\tmargin_index\n");
        fprintf(f,"%d\t%.3f\t%.2f\t%s\t%.3f\n",it,ep,ix,res,mi); fclose(f); }
    return wall()-t; }
/* [nan] locator: per tensor, non-finite grad count, max|G| and max|P| (first few events only) */
static void nan_where(const float*P,const float*G,const Off*o,size_t np){
    static int calls; if(++calls>3) return;
    typedef struct { char n[16]; size_t off; } T; T t[64*10+20]; int k=0;
    const char*nm[10]={"an1","wq","wk","wv","wo","an2","wr","w1","w3","w2"};
    const size_t*ar[10]={o->an1,o->wq,o->wk,o->wv,o->wo,o->an2,o->wr,o->w1,o->w3,o->w2};
    snprintf(t[k].n,16,"emb"); t[k++].off=o->emb;
    for(int l=0;l<gC.L;l++) for(int j=0;j<10;j++) if(ar[j][l]){ snprintf(t[k].n,16,"L%d.%s",l,nm[j]); t[k++].off=ar[j][l]; }
    if(o->nf){ snprintf(t[k].n,16,"nf"); t[k++].off=o->nf; }
    for(int i=0;i<k;i++) for(int j=i+1;j<k;j++) if(t[j].off<t[i].off){ T x=t[i]; t[i]=t[j]; t[j]=x; }
    float*g=(float*)malloc(np*4),*p=(float*)malloc(np*4);
    ck(cudaMemcpy(g,G,np*4,cudaMemcpyDeviceToHost),"nan G"); ck(cudaMemcpy(p,P,np*4,cudaMemcpyDeviceToHost),"nan P");
    for(int i=0;i<k;i++){ size_t a=t[i].off,b=i+1<k?t[i+1].off:np; size_t bad=0; float mg=0,mp=0;
        for(size_t x=a;x<b;x++){ if(!isfinite(g[x])) bad++; else if(fabsf(g[x])>mg) mg=fabsf(g[x]); if(fabsf(p[x])>mp) mp=fabsf(p[x]); }
        if(bad||i==0||!strcmp(t[i].n,"nf")||strstr(t[i].n,"an")) printf("  [nan] %-9s n=%-8zu nonfinite=%-7zu max|G|=%-10.3g max|P|=%.3g\n",t[i].n,b-a,bad,mg,mp); }
    if(g_side_gs){ float h[64*(1+SIDE_MAX)]; ck(cudaMemcpy(h,g_side_gs-64*(1+SIDE_MAX),sizeof h,cudaMemcpyDeviceToHost),"nan side");
        for(int s=0;s<=SIDE_MAX;s++){ double q=0; for(int i=0;i<64;i++) q+=h[64*s+i]; if(q!=0) printf("  [nan] %s sq-sum %g\n",s?"side":"main G",q); } }
    fflush(stdout); free(g); free(p); }
static void cmd_train(int argc,char**argv){
    if(argc<4){ fprintf(stderr,"usage: cudalm train <name> <data_dir> [epochs]\n"
        "  <name> must already exist (create with: cudalm new <name> <preset>)\n"
        "  epochs = passes over THIS corpus from now on (default 1). On a model that has been trained before, this starts\n"
        "  a new stage: warmup and LR schedule restart over the stage, sampling starts at epoch 0 of the corpus.\n"
        "  <name>.bin.run records the stage; re-running the same command after a stop resumes it instead of adding more.\n"
        "  CUDALM_ITERS=N instead pins one schedule over iters 0..N of the model's whole life (old behaviour).\n"
        "  CUDALM_BENCH_EVERY=N scores the checkpoint with lmeval every N steps (default 10000, 0 = off; all tasks, full sets; CUDALM_BENCH_TASKS / CUDALM_BENCH_N narrow it)\n"
        "  and appends to <name>.bin.bench.tsv.\n"); exit(1); }
    const char*name=argv[2],*ddir=argv[3]; float epochs=argc>4?(float)atof(argv[4]):1.0f;
    char mpath[1024],optpath[1100]; model_path(name,mpath,sizeof(mpath)); snprintf(optpath,sizeof(optpath),"%s.opt",mpath);
    FILE*ex=fopen(mpath,"rb"); if(!ex){ fprintf(stderr,"model '%s' not found.\n  create it: cudalm new %s <preset>\n",mpath,name); exit(1);} fclose(ex);
    char p[1024]; snprintf(p,sizeof(p),"%s/tokenizer.bin",ddir); size_t tlz; uint8_t*tok_blob=read_file(p,&tlz);
    if(!tok_blob){fprintf(stderr,"no %s\n",p);exit(1);} int tok_len=(int)tlz;
    Tok*dtk=tok_parse(tok_blob,tlz,0); int dataV=dtk->V,dataEot=dtk->eot;
    snprintf(p,sizeof(p),"%s/template.txt",ddir); int tmpl=0;
    { size_t tl; uint8_t*tb=read_file(p,&tl);
      if(tb){ if(tl>=6 && !memcmp(tb,"alpaca",6)) tmpl=1;
              else if(tl>=6 && !memcmp(tb,"chatml",6)) tmpl=2;   /* multi-turn + tools */
              free(tb); } }
    snprintf(p,sizeof(p),"%s/train.bin",ddir); size_t ntr; uint16_t*tr=load_u16(p,&ntr);
    int8_t*trw=NULL; float*hw=NULL; size_t*wst=NULL;
    { const char*e=getenv("CUDALM_TOKW"); if(e&&atoi(e)){ size_t nw; g_tokw=atoi(e); snprintf(p,sizeof(p),"%s/train.w",ddir); trw=(int8_t*)read_file(p,&nw);
        if(!trw||nw!=ntr){ fprintf(stderr,"[tokw] %s missing or not %zu bytes\n",p,ntr); exit(1); }
        if((e=getenv("CUDALM_NEG_PMIN"))) g_negp=(float)atof(e);
        if((e=getenv("CUDALM_OPD_BIAS"))&&strcmp(e,"auto")){ g_opdb=(float)atof(e); g_opdauto=strcmp(e,"mask")?0:-1; }
        size_t neg=0,zero=0; for(size_t i=0;i<ntr;i++){ neg+=trw[i]<0; zero+=trw[i]==0; }
        printf("[tokw] per-token weights: %.1f%% negative, %.1f%% zero, neg p-min %.3f\n",100.0*neg/ntr,100.0*zero/ntr,g_negp); } }
    snprintf(p,sizeof(p),"%s/train.bin",ddir);
    Cfg c; Tok*mtk=NULL; float*Ph=model_load_full(mpath,&c,&mtk); int trained=(mtk!=NULL);
    /* same vocab size is not enough: another tokenizer maps the same ids to other text */
    if(trained && mtk->V==dataV){ int bad=-1;
        for(int i=0;i<dataV && bad<0;i++) if(mtk->declen[i]!=dtk->declen[i] || (dtk->declen[i]>0 && memcmp(mtk->dec[i],dtk->dec[i],dtk->declen[i]))) bad=i;
        if(bad>=0 || mtk->eot!=dataEot){ fprintf(stderr,"tokenizer mismatch: model '%s' was trained with a different tokenizer than %s\n"
            "  (first differing token id %d). Encode the data with the model's tokenizer.\n",name,ddir,bad); exit(1); } }
    /* Vocab adoption. A trained model can't change vocab. An untrained one can,
     * but Ph was sized from the FILE's V, so we must actually resize rather than
     * just raising c.V (that would read past the host buffer on the H2D copy).
     * The embedding is FIRST in the layout, so everything after it is unchanged
     * in size and can be carried over verbatim -- this lets any preset be used
     * with any dataset. */
    if(c.V!=dataV){
        if(trained){ fprintf(stderr,"vocab mismatch: model %d vs data %d\n",c.V,dataV); exit(1); }
        int oldV=c.V; size_t oldnp=param_count(&c);
        Cfg nc=c; nc.V=dataV; cfg_derive(&nc);
        size_t newnp=param_count(&nc);
        size_t tail=oldnp-(size_t)oldV*c.D;          /* == newnp - dataV*D */
        float*NP=(float*)malloc(newnp*4);
        if(!NP){ fprintf(stderr,"OOM resizing embedding to V=%d\n",dataV); exit(1); }
        for(size_t i=0;i<(size_t)dataV*c.D;i++) NP[i]=0.02f*rnd_normal();
        memcpy(NP+(size_t)dataV*c.D, Ph+(size_t)oldV*c.D, tail*4);
        free(Ph); Ph=NP; c.V=dataV;
        printf("[cudalm train] preset vocab %d -> data vocab %d (embedding re-initialised,\n"
               "               %zu -> %zu params)\n", oldV, dataV, oldnp, newnp);
    }
    c.eot=dataEot; c.tmpl=tmpl; cfg_derive(&c);
    /* Decided here, not with the rest of the masking setup below, because it
     * adds buffers to the arena and ga_new runs first. */
    { const char*e=getenv("CUDALM_MASK_PROMPT");
      int want = e ? (atoi(e)!=0) : 1;   /* default ON; CUDALM_MASK_PROMPT=0 disables */
      g_mask_head = want && c.tmpl==2; }
    gC=c; cuda_init(); Off o; calc_offsets(&c,&o);
    size_t np=param_count(&c);
    int B=auto_batch(&c,np); int N=B*c.T;
    moe_stash_decide(&c,np,N);
    { size_t fr=0,tt=0; cudaMemGetInfo(&fr,&tt);
      double use=(ga_bytes(N,1)+(double)opt_state_bytes(np))/1e9, head=(tt-(tt-fr))/1e9;
      printf("[cudalm train] act=%.2f GB state=%.2f GB (B=%d T=%d, bf16 moments) | ~%.2f GB of %.2f GB, %.2f GB left for the desktop\n",
             ga_bytes(N,1)/1e9, opt_state_bytes(np)/1e9, B, c.T, use, tt/1e9, head-use);
      if(head-use < 0.8) printf("[cudalm train] WARNING: thin VRAM headroom - this GPU also drives the display.\n"
                                "               If the driver resets, lower CUDALM_BATCH or raise CUDALM_VRAM_RESERVE_MB.\n"); }
    { const char*qe=getenv("CUDALM_QLM"); g_qlm_on = qe ? atoi(qe)!=0 : 0; }
    /* default Muon steps aside where it is unsupported or untested; an explicit CUDALM_MUON=1 keeps the old errors */
    if(g_muon && !g_muon_exp && (g_qlm_on||c.mod)){ g_muon=0; printf("[cudalm train] Muon off (default) for a %s model: AdamW\n",g_qlm_on?"QLM":"per-layer layout"); }
    lm_setenv("CUDALM_QLM",g_qlm_on?"1":"0"); lm_setenv("CUDALM_MUON",g_muon?"1":"0");   /* pin both in the model config */
    /* CUDALM_QLM_FUSED=1 consumes each layer's gradients during the backward
     * pass, so G holds ONE layer instead of the whole model. Needs qlm, because
     * the quantized update is what runs per layer. */
    { const char*fe=getenv("CUDALM_QLM_FUSED");
      g_fused = (fe && atoi(fe)!=0 && g_qlm_on && !c.pool);
      if(fe && atoi(fe)!=0 && c.pool) printf("[qlm] CUDALM_QLM_FUSED ignored with UniPool\n");
      if(fe && atoi(fe)!=0 && !g_qlm_on)
          printf("[qlm] CUDALM_QLM_FUSED ignored: needs CUDALM_QLM=1\n");
      if(c.mod && g_fused){ g_fused=0; printf("[qlm] CUDALM_QLM_FUSED ignored with a per-layer layout (TLM7)\n"); }
      if(c.mod && g_muon){ fprintf(stderr,"[cudalm] CUDALM_MUON does not support per-layer layouts (TLM7) yet\n"); exit(1); } }
    { const char*pe=getenv("CUDALM_QLM_PSCRATCH");
      g_pscratch = (pe && atoi(pe)!=0 && g_fused);
      if(pe && atoi(pe)!=0 && !g_fused)
          printf("[qlm] CUDALM_QLM_PSCRATCH ignored: needs CUDALM_QLM_FUSED=1\n"); }
    if(c.lp){
        if(g_fused||getenv("CUDALM_LR_BOOST_LAYERS")||env_i("CUDALM_ATTN_GATE",0)||env_i("CUDALM_SHARED_F",0)||(getenv("CUDALM_CANON")&&*getenv("CUDALM_CANON"))){
            fprintf(stderr,"[loop] CUDALM_LOOP does not combine with QLM_FUSED, LR_BOOST_LAYERS, ATTN_GATE, SHARED_F or CANON\n"); exit(1); }
        printf("[loop] depth %d from %d physical layers:",c.L,c.lp); for(int v=0;v<c.L;v++) printf(" %d",c.lmap[v]); printf("\n"); }
    ngram_setup(&c);
    { MemCfg m; mem_live(&m); side_write(mpath,&m); }
    if(g_ngram_on) ngram_load(optpath);
    if(g_eg_on) engram_load(optpath);
    t1_load(optpath);
    if(g_fused){ g_gsize=calc_goffsets(&c,&o,&g_goff);
                 g_gnpart=(float*)dmalloc((size_t)64*c.L*4);
                 ck(cudaMemset(g_gnpart,0,(size_t)64*c.L*4),"zero gnpart"); }
    float*P;
    if(g_pscratch){
        /* encode digits from a full real-layout P, then compact P to the float
         * spans + one shared quantized block and free the full copy. */
        g_psize=calc_poffsets(&c,&o,&g_poff);
        float*Pf=(float*)dmalloc(np*4); ck(cudaMemcpy(Pf,Ph,np*4,cudaMemcpyHostToDevice),"Pf h2d");
        qlm_gpu_init(&QQ,&c,&o,Pf,np);
        if(g_fused) qlm_set_gof(&QQ,&c,&g_goff);
        P=(float*)dmalloc(g_psize*4);
        qlm_scatter_float(&c,&o,&g_poff,Pf,P);
        qlm_set_pof(&QQ,&c,&g_poff);
        qlm_remap_fs(&QQ,&c,&g_poff);
        cudaFree(Pf); g_prd=&g_poff;
        printf("[qlm] PSCRATCH: P is %.3f GB (float spans + one layer) instead of %.3f GB\n",
               g_psize*4.0/1e9, np*4.0/1e9);
    } else {
        P=(float*)dmalloc(np*4); ck(cudaMemcpy(P,Ph,np*4,cudaMemcpyHostToDevice),"P h2d");
    }
    float*G=dzero(g_fused ? g_gsize : np);
    /* With qlm on the full-size bf16 moments are NOT allocated: the quantized
     * spans carry int8 m/v instead, and the float spans use compact arrays
     * inside the Qlm. Allocating both cost 15.5 B/param -- worse than stock. */
    size_t mvn = g_qlm_on ? 1 : np;
    bf16*M=(bf16*)dmalloc(mvn*2),*Vv=(bf16*)dmalloc(mvn*2);
    ck(cudaMemset(M,0,mvn*2),"zero M"); ck(cudaMemset(Vv,0,mvn*2),"zero V");
    int start_iter=0;
    if(!g_qlm_on){
      bf16*mh=(bf16*)calloc(np,2),*vh=(bf16*)calloc(np,2); start_iter=opt_load(optpath,mh,vh,np);
      if(start_iter>0){ ck(cudaMemcpy(M,mh,np*2,cudaMemcpyHostToDevice),"M h2d"); ck(cudaMemcpy(Vv,vh,np*2,cudaMemcpyHostToDevice),"V h2d");
                        mu_load(optpath,np,start_iter); }
      free(mh);free(vh); }
    /* CUDALM_QLM=1 replaces the fp32 master weights on every 2D weight matrix
     * with an int8 digit + int8 residual + per-group scale, updated by Integer
     * AdamW. P becomes a rendered view of s*d, so every GEMM downstream and the
     * whole of gpu_backward() are untouched. See qlm.cuh. */
    if(g_qlm_on){
        if(getenv("CUDALM_LR_BOOST_LAYERS"))
            printf("[qlm] WARNING: per-layer LR boost is not applied to quantized spans\n");
        if(!g_pscratch){
        qlm_gpu_init(&QQ,&c,&o,P,np);
        /* Point each span's GRADIENT read at the compact map. Without this gof
         * stays == pof, so the step kernel reads ABSOLUTE offsets (up to np) out
         * of a G that is only one layer long -- out-of-bounds reads returning
         * plausible-magnitude garbage, i.e. updates of the right size in random
         * directions. */
        if(g_fused) qlm_set_gof(&QQ,&c,&g_goff);
        }
        char qp[1200]; snprintf(qp,sizeof qp,"%s.q",optpath);
        int qit=0;
        if(qlm_load(&QQ,qp,&qit)){ start_iter=qit;
            printf("[qlm] resumed integer state from %s at iter %d\n",qp,qit); }
        else { FILE*hf=fopen(optpath,"rb"); char hm[4]; int hit=0;   /* float (Muon/AdamW) -> QLM handoff */
            if(hf&&fread(hm,1,4,hf)==4&&(!memcmp(hm,"TOP2",4)||!memcmp(hm,"TOPT",4))&&fread(&hit,4,1,hf)==1&&hit>0){
                start_iter=hit;
                printf("[qlm] handoff: no %s, quantizing the float weights and continuing at iter %d from %s (fresh int moments)\n",qp,hit,optpath); }
            if(hf) fclose(hf); }
        if(!g_pscratch) qlm_gpu_render(&QQ,P);
        printf("[qlm] %llu of %zu params quantized (%.1f%%) in %llu groups | bits=%d group=%d opt=%s\n",
               (unsigned long long)QQ.nq, np, 100.0*(double)QQ.nq/(double)np,
               (unsigned long long)QQ.ngrp, QQ.dmax==1?2:QQ.dmax==3?3:QQ.dmax==7?4:8,
               QQ.group, QQ.adam?"Integer AdamW":"Integer Lion");
        printf("[qlm] integer state %.2f B/param (%.3f GB) -- digits+residual+moments+scales\n",
               qlm_persist_bytes(&QQ)/(double)np, qlm_persist_bytes(&QQ)/1e9);
        double qtot = qlm_total_bytes(&QQ) - (g_pscratch ? ((double)np*4.0-(double)g_psize*4.0) : 0.0);
        printf("[qlm] TOTAL %.2f B/param (%.2f GB) vs stock %.2f B/param (%.2f GB) = %.0f%% of stock\n",
               qtot/(double)np, qtot/1e9,
               (double)opt_state_bytes(np)/(double)np, opt_state_bytes(np)/1e9,
               100.0*qtot/(double)opt_state_bytes(np));
        if(g_fused){
            double tot = qlm_total_bytes(&QQ)-(double)np*4.0+(double)g_gsize*4.0
                       - (g_pscratch ? ((double)np*4.0-(double)g_psize*4.0) : 0.0);
            printf("[qlm] FUSED: G is %.3f GB (one layer) instead of %.3f GB\n"
                   "      -> total %.2f B/param = %.0f%% of stock. Clipping uses the PREVIOUS\n"
                   "      step's grad norm -- a real change to the update.\n",
                   g_gsize*4.0/1e9, np*4.0/1e9, tot/(double)np,
                   100.0*tot/(double)opt_state_bytes(np));
        } else
            printf("[qlm] (rendered fp32 P + fp32 G are 8 of those bytes; CUDALM_QLM_FUSED=1\n"
                   "      removes the G half)\n");
    }
    rope_tables(&c);
    GA*a=ga_new(N,1); a->tr=1; int*tokG=(int*)dmalloc((size_t)N*4),*tgtG=(int*)dmalloc((size_t)N*4);
    /* tok/tgt double-buffered in pinned memory so the upload is async and the
     * host builds batch n+1 while step n runs; upev[b] guards reuse of buffer b */
    int*tokb,*tgtb; cudaEvent_t upev[2];
    ck(cudaHostAlloc((void**)&tokb,sizeof(int)*2*N,0),"tok pin"); ck(cudaHostAlloc((void**)&tgtb,sizeof(int)*2*N,0),"tgt pin");
    for(int i=0;i<2;i++) ck(cudaEventCreateWithFlags(&upev[i],cudaEventDisableTiming),"upev");
    int*tok=tokb,*tgt=tgtb; float*lossh=(float*)malloc((size_t)N*4);
    int spe=(int)(ntr/((size_t)N)); if(spe<1)spe=1;
    int iters=(int)(epochs*spe+0.5f);
    /* CUDALM_ITERS pins the TOTAL iteration count instead of deriving it from
     * epochs*steps-per-epoch. The WSD cooldown window is a fraction of `iters`,
     * so a phase that continues an earlier run on a DIFFERENT corpus would
     * otherwise have its schedule silently rescaled by the new corpus's size --
     * the cooldown would start in the wrong place. Pin it and the schedule is
     * a property of the run, not of whichever data directory it points at. */
    /* Stages. [epochs] = passes over THIS corpus from now on. A trained model starts a new stage at its current iter
     * (fresh warmup + schedule over the stage, sampler at epoch 0 of this corpus); <model>.bin.run records the stage
     * (start, end, corpus) so re-running after a stop/crash finishes it instead of adding more.
     * CUDALM_ITERS keeps the old absolute meaning (total iters counted from 0, one schedule over the whole run). */
    char runp[1200]; snprintf(runp,sizeof runp,"%s.run",mpath); int pin_iters=0, run_note=0;
    { const char*e=getenv("CUDALM_ITERS"); if(e){ int v=atoi(e); if(v>0){ iters=v; pin_iters=1; } } }
    int stage_n=iters>0?iters:1;
    if(!pin_iters&&start_iter>0){
        int rs=-1,re=-1; long long rn=-1; char rd[1100]={0}; FILE*rf=fopen(runp,"r");
        if(rf){ if(fscanf(rf,"start=%d end=%d tokens=%lld corpus=%1099[^\n]",&rs,&re,&rn,rd)!=4) rs=-1; fclose(rf); }
        if(rs>=0&&re>start_iter&&rn==(long long)ntr&&!strcmp(rd,ddir)){ g_stage0=rs; iters=re; run_note=1; }
        else { if(rs>=0&&re>start_iter) printf("[cudalm train] note: the stage on %s stopped at iter %d of %d; starting a new stage on this corpus instead\n",rd,start_iter,re);
               g_stage0=start_iter; iters=start_iter+stage_n; run_note=2; }
    }
    if(iters<1)iters=1; if(iters<start_iter)iters=start_iter;
    if(!pin_iters){ FILE*rf=fopen(runp,"w"); if(rf){ fprintf(rf,"start=%d end=%d tokens=%lld corpus=%s\n",g_stage0,iters,(long long)ntr,ddir); fclose(rf); } }
    int save_every=spe<400?spe:200; if(save_every<50)save_every=50;
    /* On a flaky box, checkpoint interval is how much work a hang costs.
     * Saves are atomic+fsynced so a frequent cadence is safe, it just costs
     * ~1.5 s of I/O each time. CUDALM_SAVE_EVERY overrides. */
    { const char*e=getenv("CUDALM_SAVE_EVERY"); if(e){ int v=atoi(e); if(v>=10) save_every=v; } }
    float lr=default_lr_for(&c)*(g_stage0>0?LR_STAGE_MUL:1.0f), warmup=100.0f;
    /* CUDALM_LR overrides the peak. Needed after `grow`: a grown checkpoint has
     * no optimizer state, so training restarts at iter 0 and would otherwise ramp
     * back to the full from-scratch peak (2e-4 here) and wash out the weights the
     * growth just preserved. Continued training wants a fraction of that -- the
     * warmup still applies, which is what lets the new identity layers settle. */
    { const char*e=getenv("CUDALM_LR"); if(e){ float v=(float)atof(e); if(v>0) lr=v; } }
    { const char*e=getenv("CUDALM_WARMUP"); if(e){ int v=atoi(e); if(v>=1) warmup=(float)v; } }
    { const char*e=getenv("QLM_WDF"); if(e) g_wdf=(float)atof(e); }
    printf("[cudalm train] warmup %d iters\n",(int)warmup);
    if(g_wdf>=0.0f) printf("[cudalm train] float-span weight decay %g\n",g_wdf);
    /* Prompt-loss masking. Opt-in: it changes what the training loss MEANS
     * (averaged over assistant tokens only), so a masked run's training loss is
     * not comparable with an unmasked one. `eval` is untouched, so val loss
     * remains comparable across both. */
    MaskSpec ms; memset(&ms,0,sizeof(ms));
    int mask_on = g_mask_head;                 /* already validated against tmpl */
    { const char*e=getenv("CUDALM_MASK_PROMPT");
      if(e && atoi(e)!=0 && c.tmpl!=2)
        printf("[cudalm train] CUDALM_MASK_PROMPT ignored: template is %s, not chatml\n",
               c.tmpl==1?"alpaca":"raw"); }
    if(mask_on){
        /* dtk was parsed with full=0, so it carries no BPE merge tables and
         * tok_encode would dereference a null rank[] -- an instant access
         * violation. Build a full tokenizer just to encode the six markers,
         * then release the two V*V tables (268 MB each at V=8192). */
        Tok*mtk=tok_parse(tok_blob,tlz,1);
        mask_build(&ms,mtk);
        free(mtk->rank); free(mtk->newid);
        for(int i=0;i<mtk->V;i++) free(mtk->dec[i]);
        free(mtk->dec); free(mtk->declen); free(mtk);
        printf("[cudalm train] prompt-loss masking ON: loss only on assistant spans\n");
        printf("               markers matched as token sequences:");
        for(int i=0;i<ms.n;i++) printf(" %s(%d)", i<2?(i?"call":"asst"):(i==2?"user":(i==3?"sys":(i==4?"tools":"result"))), ms.len[i]);
        printf("\n               raw prose (no markers) stays fully trainable.\n");
    }
    printf("[cudalm train] %s %s params=%zu B=%d V=%d lr=%g | corpus %d steps/epoch\n",
           name, trained?"(resume)":"(fresh)", np, B, c.V, lr, spe);
    if(pin_iters) printf("[cudalm train] CUDALM_ITERS=%d: one schedule over iters 0..%d (absolute), now at %d\n",iters,iters,start_iter);
    else if(run_note==1) printf("[cudalm train] continuing the unfinished stage: iters %d..%d (%.2f epochs of this corpus), now at %d, %d left\n",
                                g_stage0,iters,(double)(iters-g_stage0)/spe,start_iter,iters-start_iter);
    else printf("[cudalm train] %s: %.2f epochs of this corpus = %d steps, iters %d..%d%s\n",
                run_note==2?"new stage":"stage 1",epochs,iters-g_stage0,g_stage0,iters,
                run_note==2?" (warmup + schedule restart; CUDALM_LR sets the peak, 1e-4 suits chat SFT)":"");
    g_start_iter=start_iter;
    if(fe_dim(c.D)&&g_qlm_on){ fprintf(stderr,"[emb] CUDALM_EMB_E does not combine with QLM\n"); exit(1); }
    if(!g_qlm_on) fe_init(optpath,start_iter,P+o.emb);
    { int S=iters-g_stage0, d=(int)(S*g_wsd_decay);
      if(g_wsd) printf("[cudalm train] schedule WSD: warmup %d, hold %g, cooldown over the last %.0f%% of the stage (%d iters, from iter %d) to %g\n",
                       (int)warmup, lr, g_wsd_decay*100.0f, d, iters-d, lr_minf()*lr);
      else      printf("[cudalm train] schedule cosine: %g -> %g over iters %d..%d\n", lr, lr_minf()*lr, g_stage0, iters); }
    if(g_rewarm>0) printf("[cudalm train] re-warmup: ramping to the scheduled rate over %d iters from iter %d\n",
                          g_rewarm, start_iter);
    lr_boost_setup(&c,&o);
    if(c.n_mtp>0) printf("[cudalm train] note: MTP heads are not trained on the CUDA path (main model only)\n");
    fflush(stdout);
    /* probe with a real batch before committing to the loop */
    for(int i=0;i<B;i++){ size_t st=(size_t)(urand64()%(uint64_t)(ntr-(size_t)c.T-2));
        for(int j=0;j<c.T;j++){ int r=i*c.T+j; tok[r]=tr[st+j]; tgt[r]=tr[st+j+1]; } }
    ck(cudaMemcpy(tokG,tok,(size_t)N*4,cudaMemcpyHostToDevice),"tok h2d");
    ck(cudaMemcpy(tgtG,tgt,(size_t)N*4,cudaMemcpyHostToDevice),"tgt h2d");
    preflight(P,G,&o,a,tokG,tgtG,M,Vv,np);
    fflush(stdout);
    /* Session-limited compute (a Kaggle session is capped at ~12 h): stop
     * cleanly after CUDALM_MAX_SECONDS so a run always ends on a committed
     * checkpoint rather than being killed mid-step. stop_it is what the
     * optimizer records as "done up to" -- the post-loop save used to hardcode
     * `iters`, which after an early exit would tell the next session the run had
     * finished and it would do nothing. */
    double tmax=0; { const char*e=getenv("CUDALM_MAX_SECONDS"); if(e){ double v=atof(e); if(v>0) tmax=v; } }
    double t0=wall();   /* tok/s reporting */
    double tstart=wall();                       /* session wall clock */
    int bench_every=10000; { const char*e=getenv("CUDALM_BENCH_EVERY"); if(e) bench_every=atoi(e); }   /* default on since 20f */
    int next_bench=bench_every>0?((start_iter-g_stage0)/bench_every+1)*bench_every:0;
    if(bench_every>0) printf("[cudalm train] lmeval every %d steps at the next checkpoint (CUDALM_BENCH_EVERY) -> %s.bench.tsv\n",bench_every,mpath);
    int stop_it=iters;
    if(tmax>0) printf("[cudalm train] will stop cleanly after %.0fs\n",tmax);
    PermSampler pm; memset(&pm,0,sizeof(pm)); int perm_on=1, perm_base=0;
    { const char*e=getenv("CUDALM_SAMPLER"); if(e && strcmp(e,"perm")) perm_on=0; }   /* default perm; CUDALM_SAMPLER=rand = old urand64 */
    if(perm_on){
        const char*b=getenv("CUDALM_PERM_BASE"); perm_base=b?atoi(b):g_stage0;   /* a stage starts at epoch 0 of its corpus */
        const char*sd=getenv("CUDALM_SEED");
        pm.seed = sd ? (uint64_t)lm_strtou64(sd) : 0x5EEDULL;
        pm.T=c.T; pm.nw=(ntr-2)/(size_t)c.T; pm.epoch=-1;
        if(pm.nw<1 || pm.nw>0xFFFFFFFFULL){ fprintf(stderr,"[sampler] bad window count %zu\n",pm.nw); exit(1); }
        pm.perm=(uint32_t*)malloc(pm.nw*4);
        size_t first=(size_t)(start_iter>perm_base?start_iter-perm_base:0)*(size_t)B;
        pm_start(&pm,first);
        printf("[sampler] perm: %zu windows of %d (%.1f MB), base iter %d, resuming at window %zu of epoch %lld (%.2f ep left in run)\n",
               pm.nw,c.T,pm.nw*4/1e6,perm_base,first%pm.nw,pm.epoch,
               (double)(iters-start_iter)*B/(double)pm.nw);
    }
    /* CUDALM_KD=<file.kd> (cudalm kdcache over THIS corpus): CE against (1-a) onehot + a teacher top-K, a = CUDALM_KD_ALPHA */
    FILE*kdf=NULL; unsigned short*kdh=NULL,*kdI=NULL,*kdP=NULL; size_t kdrec=0;
    { const char*e=getenv("CUDALM_KD"); if(e&&*e){ char m[4]; int kh[2]={0,0}; unsigned long long kn=0; kdf=fopen(e,"rb");
        if(!kdf||fread(m,1,4,kdf)!=4||memcmp(m,"KDC1",4)||fread(kh,4,2,kdf)!=2||fread(&kn,8,1,kdf)!=1){ fprintf(stderr,"[kd] cannot read %s\n",e); exit(1); }
        if(kh[1]!=c.V||kn!=(unsigned long long)ntr){ fprintf(stderr,"[kd] %s is V %d / %llu tokens, model V %d / corpus %zu tokens\n",e,kh[1],kn,c.V,ntr); exit(1); }
        if(mask_on||trw||g_tokw){ fprintf(stderr,"[kd] not with prompt masking or token weights\n"); exit(1); }
        g_kdK=kh[0]; kdrec=(size_t)g_kdK*4; g_kda=0.5f; { const char*x=getenv("CUDALM_KD_ALPHA"); if(x) g_kda=(float)atof(x); x=getenv("CUDALM_KD_RENORM"); g_kdrn=x&&atoi(x); }
        kdh=(unsigned short*)malloc((size_t)c.T*kdrec); kdI=(unsigned short*)malloc((size_t)N*g_kdK*2); kdP=(unsigned short*)malloc((size_t)N*g_kdK*2);
#ifdef CUDALM_CPU
        fprintf(stderr,"[kd] CUDALM_KD needs the CUDA build\n"); exit(1);
#endif
        g_kdI=(unsigned short*)dmalloc((size_t)N*g_kdK*2); g_kdP=(unsigned short*)dmalloc((size_t)N*g_kdK*2);
        printf("[kd] %s: teacher top-%d targets (%s), alpha %.2f\n",e,g_kdK,g_kdrn?"renormalised, 20n":"raw + missing mass on the true token",g_kda); } }
    auto build=[&](int it,int*tok,int*tgt)->int{
        size_t span=(size_t)c.T+2;
        for(int i=0;i<B;i++){ size_t st=perm_on ? pm_start(&pm,(size_t)(it-perm_base)*(size_t)B+(size_t)i)
                                             : (size_t)(urand64()%(uint64_t)(ntr-span));
            if(trw){ if(!wst) wst=(size_t*)malloc(sizeof(size_t)*B); wst[i]=st; }
            for(int j=0;j<c.T;j++){ int r=i*c.T+j; tok[r]=tr[st+j]; tgt[r]=tr[st+j+1]; }
            if(kdf){ lm_fseek64(kdf,20+(long long)st*(long long)kdrec,SEEK_SET);
                if(fread(kdh,kdrec,(size_t)c.T,kdf)!=(size_t)c.T){ fprintf(stderr,"[kd] short read at %zu\n",st); exit(1); }
                for(int j=0;j<c.T;j++){ size_t r=(size_t)i*c.T+j; memcpy(kdI+r*g_kdK,kdh+(size_t)j*2*g_kdK,(size_t)g_kdK*2);
                    memcpy(kdP+r*g_kdK,kdh+(size_t)j*2*g_kdK+g_kdK,(size_t)g_kdK*2); } } }
        if(kdf){ ck(cudaMemcpy(g_kdI,kdI,(size_t)N*g_kdK*2,cudaMemcpyHostToDevice),"kd I"); ck(cudaMemcpy(g_kdP,kdP,(size_t)N*g_kdK*2,cudaMemcpyHostToDevice),"kd P"); }
        int ntrain=N;
        if(mask_on){ ntrain=0;
            for(int i=0;i<B;i++) ntrain+=mask_apply(&ms,tok+(size_t)i*c.T,tgt+(size_t)i*c.T,c.T,c.eot);
            if(ntrain<1) ntrain=1;               /* a window with no assistant span */
        }
        if(trw){ if(!hw){ hw=(float*)malloc((size_t)N*4); g_wG=(float*)dmalloc((size_t)N*4); g_wGk=(float*)dmalloc((size_t)N*4); }
            ntrain=0;
            for(int i=0;i<B;i++){ size_t st=wst[i];
                for(int j=0;j<c.T;j++){ int r=i*c.T+j, q=trw[st+j+1]; float w=g_tokw==2?(q>0?(q>=32?1.f:q/32.f):q&&g_opdauto>=0?-100.f+q/8.f+g_opdb:0.f):q/32.f;
                    hw[r]=w<-50.f?w:w<-1.f?-1.f:w>1.f?1.f:w; if(w==0.f) tgt[r]=-1; ntrain+=tgt[r]>=0; } }
            if(ntrain<1) ntrain=1;
            ck(cudaMemcpy(g_wG,hw,(size_t)N*4,cudaMemcpyHostToDevice),"tokw h2d"); }
        g_loss_denom=ntrain;
        /* Trainable row list for the head. Built here because tgt[] is already
         * on the host; N ints is nothing against a 1.3 s step. */
        g_nkeep=0;
        if(mask_on && a->keep){
            int k=0; for(int i=0;i<N;i++) if(tgt[i]>=0) a->h_keep[k++]=i;
            if(k>0 && k<N){ g_nkeep=k;
                ck(cudaMemcpy(a->keep,a->h_keep,(size_t)k*4,cudaMemcpyHostToDevice),"keep h2d"); }
        }
        return ntrain;
    };
    /* ---- LR scaler setup (see g_lrmul) ---- */
    int lrs_on=0,lrs_every=15000,lrs_K=200,lrs_vb=8; float lrs_eps=0.004f; double lrs_best=1e30;
    { const char*e=getenv("CUDALM_LRS"); lrs_on=e&&atoi(e)!=0; }
    if(lrs_on&&g_feE){ printf("[lrs] off: not supported with CUDALM_EMB_E\n"); lrs_on=0; }
    if(lrs_on&&(!perm_on||g_fused||g_pscratch)){ printf("[lrs] off: needs the perm sampler and no QLM_FUSED/PSCRATCH\n"); lrs_on=0; }
    uint16_t*lv=NULL; size_t nlv=0,*lvst=NULL,lsz=0; uint8_t*lsnap=NULL,*lgood=NULL; int lgood_ok=0;
    char lrsp[1200]; snprintf(lrsp,sizeof lrsp,"%s.lrs",optpath);
    if(lrs_on){
        const char*e;
        if((e=getenv("CUDALM_LRS_EVERY"))) lrs_every=atoi(e);
        if(!getenv("CUDALM_LRS_TEST")) lrs_every=lrs_every<10000?10000:lrs_every>20000?20000:lrs_every;
        if(lrs_every<1) lrs_every=1;
        if((e=getenv("CUDALM_LRS_ITERS"))&&atoi(e)>0) lrs_K=atoi(e);
        if((e=getenv("CUDALM_LRS_VB"))&&atoi(e)>0) lrs_vb=atoi(e);
        if((e=getenv("CUDALM_LRS_EPS"))) lrs_eps=(float)atof(e);
        snprintf(p,sizeof p,"%s/val.bin",ddir);
        { size_t s; uint8_t*b=read_file(p,&s); if(b){ lv=(uint16_t*)b; nlv=s/2; } else { lv=tr; nlv=ntr; printf("[lrs] no val.bin, scoring on train.bin\n"); } }
        lvst=(size_t*)malloc(sizeof(size_t)*lrs_vb*B);
        { uint64_t s=0xA11CE5ULL; for(int i=0;i<lrs_vb*B;i++) lvst[i]=(size_t)(pm_mix(&s)%(uint64_t)(nlv-(size_t)c.T-2)); }
        lsz=lrs_setup(P,np,M,Vv); lsnap=(uint8_t*)malloc(lsz); lgood=(uint8_t*)malloc(lsz);
        if(!lsnap||!lgood){ fprintf(stderr,"[lrs] OOM for 2 x %.2f GB host snapshots\n",lsz/1e9); exit(1); }
        FILE*f=fopen(lrsp,"rb"); if(f){ if(fscanf(f,"%f %lf",&g_lrmul,&lrs_best)!=2){ g_lrmul=1; lrs_best=1e30; } fclose(f); }
        printf("[lrs] on: every %d iters, %d-iter arms, %d val batches, eps %.4f | snapshot %.2f GB | lrmul %.3f best %.4f\n",
               lrs_every,lrs_K,lrs_vb,lrs_eps,lsz/1e9,g_lrmul,lrs_best>1e29?0.0:lrs_best);
    }
    auto lrs_save=[&](){ if(!lrs_on) return; FILE*f=fopen(lrsp,"wb"); if(f){ fprintf(f,"%.6f %.6f\n",g_lrmul,lrs_best); fclose(f); } };
    auto lrs_eval=[&]()->double{
        int sm=g_moe_train; g_moe_train=0; double s=0;
        for(int b=0;b<lrs_vb;b++){
            for(int i=0;i<B;i++){ size_t st=lvst[b*B+i];
                for(int j=0;j<c.T;j++){ int r=i*c.T+j; tokb[r]=lv[st+j]; tgtb[r]=lv[st+j+1]; } }
            g_nkeep=0;
            ck(cudaMemcpy(tokG,tokb,(size_t)N*4,cudaMemcpyHostToDevice),"lrs tok");
            ck(cudaMemcpy(tgtG,tgtb,(size_t)N*4,cudaMemcpyHostToDevice),"lrs tgt");
            gpu_forward(P,&o,a,tokG); s+=gpu_loss(P,&o,a,tgtG,N,NULL);
        }
        g_moe_train=sm; return s/lrs_vb;
    };
    auto lrs_arm=[&](int it0,float m)->double{
        lrs_restore(lsnap); int bad=0;
        for(int k=it0;k<it0+lrs_K&&k<iters;k++){
            float cl=lr_at(k,iters,lr,warmup,g_wsd,g_wsd_decay)*g_lrmul*m;
            build(k,tokb,tgtb);
            ck(cudaMemcpy(tokG,tokb,(size_t)N*4,cudaMemcpyHostToDevice),"lrs tok");
            ck(cudaMemcpy(tgtG,tgtb,(size_t)N*4,cudaMemcpyHostToDevice),"lrs tgt");
            gpu_forward(P,&o,a,tokG);
            g_fused_lr=cl; g_fused_t=k+1;
            gpu_backward(P,G,&o,a,tokG,tgtG);
            if(gpu_adam(P,G,M,Vv,np,cl,k+1)){ if(g_ngram_on) ngram_update(cl,k+1); if(g_eg_on) engram_update(cl,k+1); t1_update(cl,k+1); } else bad++;
        }
        double v=lrs_eval(); return (bad>lrs_K/10||!isfinite(v))?1e30:v;
    };
    /* O7: the x1.0 arm runs last; if the multiplier stays put, its trained state is kept and the
     * main loop skips those lrs_K steps instead of retraining them (CUDALM_LRS_KEEP=0 = old) */
    int lrs_keep=1; { const char*e=getenv("CUDALM_LRS_KEEP"); if(e) lrs_keep=atoi(e); }
    auto lrs_probe=[&](int it)->int{
        double tw=wall(); lrs_snap(lsnap); double v0=lrs_eval();
        if(lgood_ok&&v0>lrs_best+0.1){
            printf("[lrs] iter %d: val %.4f > best %.4f + 0.1 - rolling back to the last good snapshot, lrmul %.3f -> %.3f\n",
                   it,v0,lrs_best,g_lrmul,fmaxf(g_lrmul*0.5f,0.125f));
            memcpy(lsnap,lgood,lsz); g_lrmul=fmaxf(g_lrmul*0.5f,0.125f);
        } else { memcpy(lgood,lsnap,lsz); lgood_ok=1; if(v0<lrs_best) lrs_best=v0; }
        static const float ms_[4]={0.5f,0.70710678f,1.0f,1.41421356f}; double v[4]; int bi=2;
        static const int ord_[4]={0,1,3,2};
        for(int j=0;j<4;j++){ int i=ord_[j]; v[i]=lrs_arm(it,ms_[i]); }
        for(int i=0;i<4;i++) if(v[i]<v[bi]) bi=i;
        float old=g_lrmul; int skip=0;
        if(bi!=2&&v[bi]<v[2]-lrs_eps){ g_lrmul*=ms_[bi]; g_lrmul=g_lrmul<0.125f?0.125f:g_lrmul>2.0f?2.0f:g_lrmul; }
        if(lrs_keep&&g_lrmul==old&&v[2]<1e29&&it+lrs_K<=iters) skip=lrs_K;
        else lrs_restore(lsnap);
        printf("[lrs] iter %d: val %.4f | arms x0.50 %.4f x0.71 %.4f x1.00 %.4f x1.41 %.4f | lrmul %.3f -> %.3f (lr %.2e) | %.0fs\n",
               it,v0,v[0],v[1],v[2],v[3],old,g_lrmul,lr_at(it,iters,lr,warmup,g_wsd,g_wsd_decay)*g_lrmul,wall()-tw);
        if(skip) printf("[lrs] kept the x1.00 arm: resuming at iter %d\n",it+skip);
        fflush(stdout); lrs_save(); return skip;
    };
    /* one checkpoint set (model + every optimizer sidecar), committed all-or-nothing */
    auto save_all=[&](int done)->int{
        ck(cudaDeviceSynchronize(),"sync");
        if(g_pscratch) qlm_gather_full(&QQ,&c,&o,&g_poff,P,Ph);
        else ck(cudaMemcpy(Ph,P,np*4,cudaMemcpyDeviceToHost),"P d2h");
        save_begin();
        model_save_bundle(mpath,&c,Ph,np,tok_blob,tok_len);
        if(!g_qlm_on){
            bf16*mh=(bf16*)malloc(np*2),*vh=(bf16*)malloc(np*2);
            ck(cudaMemcpy(mh,M,np*2,cudaMemcpyDeviceToHost),"M d2h");
            ck(cudaMemcpy(vh,Vv,np*2,cudaMemcpyDeviceToHost),"V d2h");
            opt_save(optpath,done,mh,vh,np); mu_save(optpath,done,np);
            free(mh);free(vh);
        } else { char qp[1200]; snprintf(qp,sizeof qp,"%s.q",optpath); qlm_save(&QQ,qp,done); }
        if(g_ngram_on) ngram_save(optpath,done);
        if(g_eg_on) engram_save(optpath,done);
        t1_save(optpath,done); fe_save(optpath,done);
        int ok=save_end();
        if(ok) lrs_save();
        else { printf("[ckpt] save FAILED - the previous checkpoint set is kept\n"); fflush(stdout); }
        return ok; };
    /* divergence auto-stop: 50 logged losses (1000 iters) window */
    float dv_thr=0.3f; { const char*e=getenv("CUDALM_DIVSTOP"); if(e) dv_thr=(float)atof(e); }
    double dvw[50]; int dvn=0,nanw[50]={0},nanc=0; double dv_best=1e30;
    int keep_every=20000; { const char*e=getenv("CUDALM_KEEP_EVERY"); if(e) keep_every=atoi(e); }
    char stopp[1100]; snprintf(stopp,sizeof stopp,"%s.stop",mpath);
    int lrs_first=1, nskip=0, save_due=0;
    for(int it=start_iter; it<iters; it++){
        if(lrs_on&&it>=(int)warmup&&(lrs_first||it%lrs_every==0)){ lrs_first=0; int sk=lrs_probe(it);
            if(sk){ if((it+sk-1)/save_every>(it-1)/save_every) save_due=1; it+=sk-1; continue; } }
        float cur=lr_at(it,iters,lr,warmup,g_wsd,g_wsd_decay)*g_lrmul;
        tok=tokb+(size_t)(it&1)*N; tgt=tgtb+(size_t)(it&1)*N; ck(cudaEventSynchronize(upev[it&1]),"upev");
        int ntrain=build(it,tok,tgt);
        ck(cudaMemcpyAsync(tokG,tok,(size_t)N*4,cudaMemcpyHostToDevice,0),"tok h2d");
        ck(cudaMemcpyAsync(tgtG,tgt,(size_t)N*4,cudaMemcpyHostToDevice,0),"tgt h2d");
        ck(cudaEventRecord(upev[it&1],0),"upev");
        gpu_forward(P,&o,a,tokG);
        g_fused_lr=cur; g_fused_t=it+1;
        if(g_fused) fused_side_clip();       /* the layers step inside the backward */
        gpu_backward(P,G,&o,a,tokG,tgtG);
        int ok=1;
        if(g_fused){ double sd=fused_side_sq();
            qlm_fused_tail(P,G,g_pscratch?&g_poff:&o,&g_goff);
            if(g_clip_side&&isfinite(sd)) g_prev_gn=(float)sqrt((double)g_prev_gn*g_prev_gn+sd);
            g_gn=g_prev_gn; ok=isfinite(sd)&&isfinite(g_gn); }   /* the layers are already applied: !ok skips the side updates */
        else        ok=gpu_adam(P,G,M,Vv,np,cur,it+1);
        if(ok){ nskip=0;
            if(g_ngram_on) ngram_update(cur,it+1);
            if(g_eg_on) engram_update(cur,it+1);
            t1_update(cur,it+1);
        } else {
            printf("[nan] iter %d grad norm %g - %s (%d in a row)\n",it,g_gn,
                   g_fused?"side-parameter updates skipped (fused layers already stepped)":"step skipped",++nskip); fflush(stdout);
            if(!g_fused) nan_where(P,G,&o,np);
            if(nskip>=20){ printf("[nan] %d non-finite steps in a row - aborting, last ckpt is intact\n",nskip); fflush(stdout); exit(3); }
        }
        if(it%20==0){ ck(cudaDeviceSynchronize(),"sync");
            ck(cudaMemcpy(lossh,a->lossb,(size_t)N*4,cudaMemcpyDeviceToHost),"loss d2h");
            /* masked positions carry loss 0, so divide by what actually counted */
            double L=0; for(int i=0;i<N;i++)L+=lossh[i]; L/=ntrain;
            double dt=wall()-t0;
            printf("iter %5d (ep %.2f) | loss %.4f | lr %.2e | gn %.2f | %.0fs | %.0f tok/s",
                   it,(double)(it-g_stage0)/spe,L,cur,g_gn,dt, it>start_iter?(double)N*(it-start_iter)/dt:0.0);
            /* trainable fraction: the one number that shows masking is doing
             * something sane. ~100% means no chat data in the window (all prose);
             * near 0% would mean the markers are not matching at all. */
            if(g_zl>0) printf(" | zl %.4f", zl_value(N));
            if(mask_on) printf(" | train %.0f%%", 100.0*ntrain/N);
#ifndef CUDALM_CPU
            if(g_tokw==2){ float o[2]={0,0},z[2]={0,0}; cudaMemcpyFromSymbol(o,d_opd,8); cudaMemcpyToSymbol(d_opd,z,8);
                if(o[1]>0){ float raw=o[0]/o[1]-g_opdb; printf(" | opd adv %.3f raw %.3f",o[0]/o[1],raw);
                    if(g_opdauto) g_opdb=g_opdb?0.8f*g_opdb-0.2f*raw:-raw; } }
#endif
            printf("\n"); fflush(stdout);
            if(dv_thr>0&&it>=(int)warmup+2000){
                int nf=!isfinite(L); nanc+=nf-nanw[dvn%50]; nanw[dvn%50]=nf; dvw[dvn%50]=nf?0:L; dvn++;
                if(dvn>=50){ double w=0; for(int i=0;i<50;i++) w+=dvw[i]; w/=(50-nanc>0?50-nanc:1);
                    if(nanc<5&&w<dv_best) dv_best=w;
                    if(nanc>=10||w>dv_best+dv_thr){
                        printf("[divstop] iter %d: 1000-iter avg loss %.4f vs best %.4f (+%.2f), %d NaN - stopping, ckpt not overwritten (see *_it<N> snapshots)\n",it,w,dv_best,dv_thr,nanc);
                        fflush(stdout); exit(5); } } } }
        if(it>start_iter && (it%save_every==0||save_due) && !nskip && disk_low(mpath)){ save_due=0;
            printf("[ckpt] SKIPPED @ iter %d: under 3 GB free\n",it); fflush(stdout); }
        else if(it>start_iter && (it%save_every==0||save_due) && !nskip){ save_due=0;
            /* step `it` is already applied: record it+1 as "done up to", exactly as the
             * post-loop save records stop_it, or a resume would train iteration `it` twice */
            int done=it+1;
            if(save_all(done)){
            printf("[ckpt] saved %s @ iter %d (resumes at %d)\n",mpath,it,done); fflush(stdout);
            if(bench_every>0 && it-g_stage0>=next_bench){ t0+=train_bench(mpath,it,(double)(it-g_stage0)/spe);
                next_bench=((it-g_stage0)/bench_every+1)*bench_every; }
            if(keep_every>0&&it%keep_every==0){
                /* ".opt" = the AdamW moments of a CUDALM_QLM=0 run, ".opt.mu" its Muon momentum */
                const char*sfx[10]={"",".opt",".opt.mu",".opt.q",".opt.ng",".opt.t1",".opt.lrs",".env",".opt.x1",".opt.fe"}; char s[1200],d[1300]; int nk=0;
                for(int i=0;i<10;i++){ snprintf(s,sizeof s,"%s%s",mpath,sfx[i]);
                    const char*dot=strrchr(mpath,'.'); int bl=dot?(int)(dot-mpath):(int)strlen(mpath);
                    snprintf(d,sizeof d,"%.*s_it%d%s%s",bl,mpath,it,dot?dot:"",sfx[i]);
                    if(lm_copyfile(s,d)) nk++; }
                printf("[keep] %d files copied as *_it%d\n",nk,it); fflush(stdout); } } }
        if(tmax>0 && (wall()-tstart)>=tmax){          /* checked after the save */
            stop_it=it+1;
            printf("[cudalm train] CUDALM_MAX_SECONDS (%.0fs) reached at iter %d - stopping cleanly\n",tmax,it);
            fflush(stdout); break; }
        if(it%20==0&&lm_exists(stopp)){
            stop_it=it+1; remove(stopp);
            printf("[cudalm train] %s found at iter %d - stopping cleanly\n",stopp,it); fflush(stdout); break; }
    }
    /* wait for space; a .stop file or CUDALM_MAX_SECONDS ends the wait. The checkpoint set is
     * committed all-or-nothing, so trying it on a nearly full disk keeps the previous set if it fails */
    { int why=0;
      for(int s=0;disk_low(mpath);s++){
          if(lm_exists(stopp)){ why=1; remove(stopp); break; }
          if(tmax>0&&(wall()-tstart)>=tmax){ why=2; break; }
          if(s%600==0){ printf("[ckpt] under 3 GB free - waiting to save %s (create %s to try anyway)\n",mpath,stopp); fflush(stdout); }
          lm_sleep_ms(1000); }
      if(why){ printf("[ckpt] under 3 GB free - trying the save anyway (%s)\n",why==1?"stop file":"CUDALM_MAX_SECONDS"); fflush(stdout); } }
    /* fused QLM applies each layer's update inside the backward, so a non-finite step is already in
     * the weights: never let it overwrite the last good checkpoint */
    if(g_fused&&nskip) printf("[ckpt] NOT saved: the last %d step(s) were non-finite and fused QLM had already applied them - previous checkpoint kept\n",nskip);
    else if(save_all(stop_it) && bench_every>0) train_bench(mpath,stop_it,(double)(stop_it-g_stage0)/spe);
    if(stop_it<iters) printf("[cudalm train] paused at iter %d of %d -> %s (resume by re-running)\n",stop_it,iters,mpath);
    else              printf("[cudalm train] done -> %s\n", mpath);
    if(g_sb_n) printf("[cudalm] MoE load max/avg per pass: mean %.2f recent %.2f max %.2f (%ld passes, %ld SB fallbacks)\n",g_sb_r/g_sb_n,g_sb_ema,g_sb_rmax,g_sb_n,g_sb_fb);
    if(g_sk_n[0]&&g_sk_n[1]) printf("[cudalm] MoE load max/avg by slot: slot0 %.2f  slot1+ %.2f\n",g_sk_r[0]/g_sk_n[0],g_sk_r[1]/g_sk_n[1]);
    if(g_rb_tot) printf("[cudalm] MoE rebalance moved %.2f%% of routed slots\n",100.0*g_rb_mv/g_rb_tot);
    prof_report();
}

/* ===================== probe (batch generation benchmark) ================
 * Loads the model ONCE and generates for every prompt in a file. Two reasons
 * not to just call `run` in a loop: it would re-upload 252 MB per prompt, and
 * `run` recomputes the full T=4096 context for every single token. Generation
 * here only needs prompt+reply, so the context is cut to PROBE_CTX, making each
 * forward ~8x cheaper and far kinder to a GPU that is busy training. */
#define PROBE_CTX 512
/* rep: CTRL-style penalty over hist (logit>0 ? /rep : *rep), then temp,
 * then top-k and nucleus (top-p) truncation over the sorted distribution. */
static const float*g_sp;
static int cmp_pdesc(const void*x,const void*y){ float a=g_sp[*(const int*)x],b=g_sp[*(const int*)y]; return a<b?1:(a>b?-1:0); }
static int sample_ex(float*lg,int V,float temp,int topk,float topp,float rep,const int*hist,int nh,int greedy){
    if(rep>1.0f&&nh>0){ static int*stamp=NULL,sv=0,gen=0; if(sv<V){ free(stamp); stamp=(int*)calloc(V,4); sv=V; } gen++;
        for(int i=0;i<nh;i++){ int t=hist[i]; if(t<0||t>=V||stamp[t]==gen) continue; stamp[t]=gen;
            lg[t]= lg[t]>0? lg[t]/rep : lg[t]*rep; } }
    if(greedy){ int nx=0; for(int i=1;i<V;i++) if(lg[i]>lg[nx])nx=i; return nx; }
    float it=temp>1e-6f?temp:1e-6f, mx=-1e30f;
    for(int i=0;i<V;i++){ lg[i]/=it; if(lg[i]>mx)mx=lg[i]; }
    double sum=0; for(int i=0;i<V;i++){ lg[i]=expf(lg[i]-mx); sum+=lg[i]; }
    static int*idx=NULL,iv=0; if(iv<V){ free(idx); idx=(int*)malloc((size_t)V*4); iv=V; }
    for(int i=0;i<V;i++) idx[i]=i;
    g_sp=lg; qsort(idx,V,sizeof(int),cmp_pdesc);
    int cut=V; if(topk>0&&topk<cut) cut=topk;
    if(topp>0&&topp<1){ double ac=0; for(int i=0;i<cut;i++){ ac+=lg[idx[i]]/sum; if(ac>=topp){ cut=i+1; break; } } }
    double tot=0; for(int i=0;i<cut;i++) tot+=lg[idx[i]];
    double r=urand()*tot,ac=0;
    for(int i=0;i<cut;i++){ ac+=lg[idx[i]]; if(ac>=r) return idx[i]; }
    return idx[cut-1];
}
static void cmd_probe(int argc,char**argv){
    if(argc<4){ fprintf(stderr,
        "usage: cudalm probe <name|model.bin> <prompts.txt> [ntok] [temp] [topk] [topp] [reppen]\n"
        "  prompts.txt: one prompt per line; lines starting with # are echoed as headings,\n"
        "  blank lines ignored. Loads the model once and generates for each.\n"); exit(1); }
    char path[1024]; model_path(argv[2],path,sizeof(path));
    FILE*e=fopen(path,"rb"); if(e)fclose(e); else snprintf(path,sizeof(path),"%s",argv[2]);
    Cfg c; Tok*tk=NULL; float*Ph=model_load_full(path,&c,&tk);
    if(!tk){ fprintf(stderr,"model has no tokenizer (untrained?)\n"); exit(1); }
    int ntok=argc>4?atoi(argv[4]):80;
    float temp=argc>5?(float)atof(argv[5]):0.0f;
    int topk=argc>6?atoi(argv[6]):1;
    float topp=argc>7?(float)atof(argv[7]):0.0f, rep=argc>8?(float)atof(argv[8]):1.0f;
    int greedy=(topk==1||temp<=1e-4f), stopid=chat_stop_id(&c,tk);
    size_t flen; uint8_t*fbuf=read_file(argv[3],&flen);
    if(!fbuf){ fprintf(stderr,"cannot read %s\n",argv[3]); exit(1); }
    int modelT=c.T;
    if(c.T>PROBE_CTX) c.T=PROBE_CTX;          /* short context: prompt+reply only */
    cfg_derive(&c); gC=c; cuda_init(); Off o; calc_offsets(&c,&o);
    ngram_infer(path);
    size_t np=param_count(&c);
    float*P=(float*)dmalloc(np*4); ck(cudaMemcpy(P,Ph,np*4,cudaMemcpyHostToDevice),"P h2d");
    rope_tables(&c);
    GA*a=ga_new(c.T,0); int*tokG=(int*)dmalloc((size_t)c.T*4);
    float*logits=(float*)malloc((size_t)c.V*4); int*padded=(int*)malloc(sizeof(int)*c.T);
    int*seq=(int*)malloc(sizeof(int)*c.T);
    printf("model %s | trained ctx %d, probing at %d | %s | temp=%.2f topk=%d topp=%.2f rep=%.2f\n",
           argv[2],modelT,c.T, c.tmpl==2?"chatml":(c.tmpl==1?"alpaca":"raw"),temp,topk,topp,rep);
    printf("================================================================\n");
    char*p=(char*)fbuf, *end=(char*)fbuf+flen;
    int nq=0; double t0=wall(); int gentot=0;
    while(p<end){
        char*nl=p; while(nl<end && *nl!='\n') nl++;
        int len=(int)(nl-p); while(len>0 && (p[len-1]=='\r'||p[len-1]==' ')) len--;
        char line[4096]; if(len>4000) len=4000;
        memcpy(line,p,len); line[len]=0; p=nl+1;
        if(len==0){ continue; }
        if(line[0]=='#'){ printf("\n%s\n", line); fflush(stdout); continue; }
        nq++;
        char fmt[8192];
        if(c.tmpl==2)      snprintf(fmt,sizeof(fmt),"<|user|>\n%s\n<|assistant|>\n",line);
        else if(c.tmpl==1) snprintf(fmt,sizeof(fmt),"### Instruction:\n%s\n\n### Response:\n",line);
        else               snprintf(fmt,sizeof(fmt),"%s",line);
        int pn; int*ids=tok_encode(tk,fmt,&pn);
        int slen=0; for(int i=0;i<pn&&slen<c.T;i++) seq[slen++]=ids[i]; free(ids);
        int pn0=slen;
        printf("\nQ: %s\nA: ",line); fflush(stdout);
        for(int step=0; step<ntok && slen<c.T; step++){
            int ctx=slen, off=0;
            for(int j=0;j<c.T;j++) padded[j]= j<ctx? seq[off+j]:0;
            ck(cudaMemcpy(tokG,padded,(size_t)c.T*4,cudaMemcpyHostToDevice),"tok h2d");
            gpu_forward(P,&o,a,tokG);
            g_abt(a->fn+(size_t)(ctx-1)*c.D,P+o.emb,a->logits,1,c.D,c.V);
            ck(cudaMemcpy(logits,a->logits,(size_t)c.V*4,cudaMemcpyDeviceToHost),"logits d2h");
            int nxt=sample_ex(logits,c.V,temp,topk,topp,rep,seq+pn0,slen-pn0,greedy);
            if(nxt==tk->eot||nxt==stopid) break;
            tok_print(tk,nxt); gentot++;
            seq[slen++]=nxt;
        }
        printf("\n"); fflush(stdout);
    }
    double dt=wall()-t0;
    printf("\n================================================================\n");
    printf("%d prompts, %d tokens generated in %.1fs (%.1f tok/s)\n", nq, gentot, dt, dt>0?gentot/dt:0.0);
}

/* ===================== eval (val loss + expert balance) ==================
 * Forward-only, so it can run against a live checkpoint while training
 * continues -- the atomic save means a reader sees either the old file or the
 * new one, never a torn one. Validation loss over many batches has far lower
 * variance than a single 2-document training batch, which is what makes it
 * possible to tell a real plateau from B=2 noise.
 * Expert counts come free: moe_perm already builds per-layer offsets on the
 * host, so off[e+1]-off[e] is the token count for expert e. */
/* perpos (optional, N floats) receives the per-position loss, so a caller can
 * re-average over a SUBSET without a second forward pass. */
static double gpu_loss(const float*P,const Off*o,GA*a,const int*tgtG,int N,float*perpos){
    int D=gC.D,V=gC.V;
    for(int r0=0;r0<N;r0+=g_chrows){
        int rows=(N-r0<g_chrows)?(N-r0):g_chrows;
        g_abt(a->fn+(size_t)r0*D,P+o->emb,a->logits,rows,D,V);
        KL(k_softmax_ce,rows,TPB)(a->logits,tgtG+r0,a->lossb+r0,a->dlogits,NULL,rows,V,N,(const float*)NULL,0.f);
    }
    ck(cudaDeviceSynchronize(),"eval loss");
    float*h = perpos ? perpos : (float*)malloc((size_t)N*4);
    ck(cudaMemcpy(h,a->lossb,(size_t)N*4,cudaMemcpyDeviceToHost),"loss d2h");
    double s=0; for(int i=0;i<N;i++) s+=h[i];
    if(!perpos) free(h);
    return s/N;
}
/* kdcache (20n): the teacher's top-K next-token probabilities for every train.bin position, for CUDALM_KD.
 * Windows of T advance by T/2 and keep only positions not yet written, so every target past the first T/2
 * sees >= T/2 tokens of context. File: "KDC1", K, V, ntok (u64), then per input position p (predicting
 * token p+1) K uint16 ids + K fp16 probabilities; the last position is a zero record (no gradient). */
static void cmd_kdcache(int argc,char**argv){
    if(argc<5){ fprintf(stderr,"usage: cudalm kdcache <teacher> <data_dir> <out.kd> [K=8] [B=8]\n"
        "  teacher top-K (<= 16) next-token probabilities for every train.bin position; train with CUDALM_KD=<out.kd>\n"); exit(1); }
#ifdef CUDALM_CPU
    fprintf(stderr,"kdcache needs the CUDA build\n"); exit(1);
#else
    char path[1024],p[1100]; model_resolve(argv[2],path,sizeof path);
    Cfg c; Tok*tk=NULL; float*Ph=model_load_full(path,&c,&tk);
    int K=argc>5?atoi(argv[5]):8, B=argc>6?atoi(argv[6]):8; if(K<1)K=1; if(K>16)K=16; if(B<1)B=1; if(B>64)B=64;
    snprintf(p,sizeof p,"%s/train.bin",argv[3]); size_t ntr; uint16_t*tr=(uint16_t*)read_file(p,&ntr);
    if(!tr){ fprintf(stderr,"cannot read %s\n",p); exit(1); } ntr/=2;
    int T=c.T, H=T/2, N=B*T, D=c.D, V=c.V; if(ntr<(size_t)T+2){ fprintf(stderr,"[kdcache] corpus shorter than T\n"); exit(1); }
    gC=c; cuda_init(); Off o; calc_offsets(&c,&o); ngram_infer(path);
    size_t np=param_count(&c); float*P=(float*)dmalloc(np*4); ck(cudaMemcpy(P,Ph,np*4,cudaMemcpyHostToDevice),"P h2d");
    rope_tables(&c); GA*a=ga_new(N,1);
    int*tokG=(int*)dmalloc((size_t)N*4),*tok=(int*)malloc((size_t)N*4);
    unsigned short*oiG=(unsigned short*)dmalloc((size_t)g_chrows*K*2),*hI=(unsigned short*)malloc((size_t)N*K*2),*hP=(unsigned short*)malloc((size_t)N*K*2);
    __half*opG=(__half*)dmalloc((size_t)g_chrows*K*2);
    unsigned short*rec=(unsigned short*)calloc((size_t)T,(size_t)K*4);
    FILE*f=fopen(argv[4],"wb"); if(!f){ perror(argv[4]); exit(1); }
    int hv[2]={K,V}; unsigned long long nn=ntr; fwrite("KDC1",1,4,f); fwrite(hv,4,2,f); fwrite(&nn,8,1,f);
    size_t nwin=(ntr-T-1)/H+1, done=0; int last=(nwin-1)*(size_t)H+T<ntr-1; nwin+=last;   /* + one window flush with the end */
    printf("[kdcache] %s -> %s | %zu tokens, %zu windows of %d (stride %d), top-%d\n",argv[2],argv[4],ntr,nwin,T,H,K); fflush(stdout);
    double t0=wall();
    for(size_t w0=0;w0<nwin;w0+=B){ int nb=(int)(nwin-w0<(size_t)B?nwin-w0:B); size_t sw[64];
        memset(tok,0,(size_t)N*4);
        for(int i=0;i<nb;i++){ size_t w=w0+i; sw[i]=last&&w==nwin-1?ntr-1-T:w*H; for(int j=0;j<T;j++) tok[i*T+j]=tr[sw[i]+j]; }
        ck(cudaMemcpy(tokG,tok,(size_t)N*4,cudaMemcpyHostToDevice),"tok h2d");
        gpu_forward(P,&o,a,tokG);
        for(int r0=0;r0<nb*T;r0+=g_chrows){ int rows=nb*T-r0<g_chrows?nb*T-r0:g_chrows;
            g_abt(a->fn+(size_t)r0*D,P+o.emb,a->logits,rows,D,V);
            KL(k_topk_row,rows,256)(a->logits,V,K,oiG,opG); kchk("kdcache topk",r0);
            ck(cudaMemcpy(hI+(size_t)r0*K,oiG,(size_t)rows*K*2,cudaMemcpyDeviceToHost),"kd I d2h");
            ck(cudaMemcpy(hP+(size_t)r0*K,opG,(size_t)rows*K*2,cudaMemcpyDeviceToHost),"kd P d2h"); }
        for(int i=0;i<nb;i++){ int j0=sw[i]+T>done?(int)(done>sw[i]?done-sw[i]:0):T;
            for(int j=j0;j<T;j++){ memcpy(rec+(size_t)(j-j0)*2*K,hI+((size_t)i*T+j)*K,(size_t)K*2); memcpy(rec+(size_t)(j-j0)*2*K+K,hP+((size_t)i*T+j)*K,(size_t)K*2); }
            if(j0<T){ fwrite(rec,(size_t)K*4,(size_t)(T-j0),f); done=sw[i]+T; } }
        if((w0/B)%2000==0){ printf("  %zu/%zu windows, %.0fs\n",w0+nb,nwin,wall()-t0); fflush(stdout); } }
    memset(rec,0,(size_t)K*4); for(;done<ntr;done++) fwrite(rec,(size_t)K*4,1,f);
    if(fclose(f)){ fprintf(stderr,"[kdcache] write failed\n"); exit(1); }
    printf("[kdcache] done: %zu positions, %.2f GB, %.0fs\n",ntr,(20+(double)ntr*K*4)/1e9,wall()-t0);
#endif
}
/* vrescheck (20n): central finite differences vs gpu_backward on a random 3-layer dense model (Wv of layers 0-2,
 * Wq of layer 1), with CUDALM_VRES off and on. Exact in the CPU build (fp32); GPU fp16/FP8 paths are switched off. */
static void cmd_vrescheck(void){
    Cfg c; preset("nano",&c); c.L=3; c.n_exp=0; c.n_mtp=0; c.T=32; c.win=0; c.full_every=0; cfg_derive(&c);
    gC=c; cuda_init(); g_fp8gemm=0; g_fp16=0; Off o; calc_offsets(&c,&o); t1_setup(&c);
    size_t np=param_count(&c); float*Ph=(float*)malloc(np*4); for(size_t i=0;i<np;i++) Ph[i]=0.05f*rnd_normal();
    for(int l=0;l<c.L;l++) for(int i=0;i<c.D;i++){ Ph[o.an1[l]+i]=1; Ph[o.an2[l]+i]=1; }
    for(int i=0;i<c.D;i++) Ph[o.nf+i]=1;
    int B=2,N=B*c.T; float*P=(float*)dmalloc(np*4),*G=(float*)dmalloc(np*4),*Gh=(float*)malloc(np*4);
    rope_tables(&c); GA*a=ga_new(N,1); a->tr=1;
    int*tok=(int*)malloc(N*4),*tgt=(int*)malloc(N*4),*tokG=(int*)dmalloc(N*4),*tgtG=(int*)dmalloc(N*4);
    for(int i=0;i<N;i++){ tok[i]=(int)(urand64()%(uint64_t)c.V); tgt[i]=(int)(urand64()%(uint64_t)c.V); }
    ck(cudaMemcpy(tokG,tok,N*4,cudaMemcpyHostToDevice),"t"); ck(cudaMemcpy(tgtG,tgt,N*4,cudaMemcpyHostToDevice),"t");
    auto loss=[&](){ ck(cudaMemcpy(P,Ph,np*4,cudaMemcpyHostToDevice),"P"); gpu_forward(P,&o,a,tokG); return gpu_loss(P,&o,a,tgtG,N,NULL); };
    int okall=1;
    for(int vr=0;vr<2;vr++){ g_vres=vr; printf("[vrescheck] VRES=%d loss %.6f\n",vr,loss()); gpu_backward(P,G,&o,a,tokG,tgtG); ck(cudaMemcpy(Gh,G,np*4,cudaMemcpyDeviceToHost),"G");
        double num=0,den=0; size_t blk[4]={o.wv[0],o.wv[1],o.wv[2],o.wq[1]};
        for(int b=0;b<4;b++) for(int k=0;k<6;k++){ size_t i=blk[b]+(size_t)(urand64()%(uint64_t)(c.D*c.KD)); float s=Ph[i]; double e=1e-2;
            Ph[i]=s+(float)e; double lp=loss(); Ph[i]=s-(float)e; double lm=loss(); Ph[i]=s;
            double fd=(lp-lm)/(2*e); num+=(fd-Gh[i])*(fd-Gh[i]); den+=fd*fd; }
        double re=sqrt(num/(den>0?den:1)); printf("[vrescheck] VRES=%d: rel err %.2e %s\n",vr,re,re<2e-2?"ok":"FAIL"); okall&=re<2e-2; }
    printf("[vrescheck] %s\n",okall?"PASS":"FAIL"); }
/* avg (20n): uniform average of checkpoints of one architecture (seeds of one recipe, the last checkpoints of
 * a decay): weights, the rendered n-gram table (re-encoded to q4), Wng and the .t1/.x1 side weights. The
 * optimizer state is not averaged (the first input's sidecar moments are written): eval and export only. */
static void cmd_avg(int argc,char**argv){
    if(argc<5){ fprintf(stderr,"usage: cudalm avg <out> <in1> <in2> [...]   (names or .bin paths, one architecture)\n"); exit(1); }
    side_apply(argv[3]); int nin=argc-3; Cfg c; Tok*tk=NULL; size_t np=0; double*acc=NULL; char p0[1024],pk[1024],op[1100];
    for(int k=0;k<nin;k++){ model_resolve(argv[3+k],pk,sizeof pk); Cfg ck2; float*Pk=model_load_full(pk,&ck2,&tk);
        if(!k){ c=ck2; np=param_count(&c); acc=(double*)calloc(np,8); snprintf(p0,sizeof p0,"%s",pk); }
        else if(param_count(&ck2)!=np||ck2.D!=c.D||ck2.L!=c.L||ck2.V!=c.V){ fprintf(stderr,"[avg] %s: different architecture\n",pk); exit(1); }
        for(size_t i=0;i<np;i++) acc[i]+=Pk[i]; free(Pk); }
    float*P=(float*)malloc(np*4); for(size_t i=0;i<np;i++) P[i]=(float)(acc[i]/nin); free(acc);
    gC=c; cuda_init();
    float*dv[5]={0}; size_t dn[5]={0}; double*hs[5]={0}; float*tmp=NULL; size_t tcap=0;
    for(int k=0;k<nin;k++){ model_resolve(argv[3+k],pk,sizeof pk); snprintf(op,sizeof op,"%s.opt",pk);
        if(!k){ ngram_infer(pk); if(g_eg_on){ fprintf(stderr,"[avg] engram tables are not supported\n"); exit(1); }
            size_t nt=(size_t)g_ng_R*g_ng_De, nw=(size_t)g_ng_De*c.D;
            if(g_ngram_on){ dv[0]=g_ng; dn[0]=nt; dv[1]=g_ngW; dn[1]=nw; if(g_ng_gate){ dv[2]=g_ngWk; dn[2]=nw; } }
            if(g_attg){ dv[3]=g_Wg; dn[3]=(size_t)c.L*c.D*c.D; }
            if(g_xn){ dv[4]=g_xP; dn[4]=g_xn; }
            for(int i=0;i<5;i++) if(dv[i]){ hs[i]=(double*)calloc(dn[i],8); if(dn[i]>tcap) tcap=dn[i]; }
            tmp=(float*)malloc(tcap*4+4); }
        else { if(g_ngram_on&&!ngram_load(op)){ fprintf(stderr,"[avg] no n-gram table at %s.ng\n",op); exit(1); } t1_load(op); }
        for(int i=0;i<5;i++) if(dv[i]){ ck(cudaMemcpy(tmp,dv[i],dn[i]*4,cudaMemcpyDeviceToHost),"avg d2h"); for(size_t j=0;j<dn[i];j++) hs[i][j]+=tmp[j]; } }
    for(int i=0;i<5;i++) if(dv[i]){ for(size_t j=0;j<dn[i];j++) tmp[j]=(float)(hs[i][j]/nin); ck(cudaMemcpy(dv[i],tmp,dn[i]*4,cudaMemcpyHostToDevice),"avg h2d"); free(hs[i]); }
    if(g_ngram_on&&g_ng_q4){ size_t nt=dn[0];   /* re-encode the averaged table: fresh digits/residual/scales, zero moments */
        KL(k_qlm_encode,(unsigned)g_ngq_ngrp,g_ngq_grp,g_ngq_grp)(g_ng,g_ngq_d,g_ngq_r,g_ngq_s,g_ngq_spD,1,g_ngq_grp,g_ngq_dmax,1); kchk("avg ngq encode",-1);
        ck(cudaMemset(g_ngq_m8,0,nt),"avg m8"); ck(cudaMemset(g_ngq_v8,0,nt),"avg v8");
        ck(cudaMemset(g_ngq_ms,0,(size_t)g_ngq_ngrp*4),"avg ms"); ck(cudaMemset(g_ngq_vs,0,(size_t)g_ngq_ngrp*4),"avg vs"); }
    char out[1024]; model_path(argv[2],out,sizeof out); snprintf(op,sizeof op,"%s.opt",out);
    model_save_bundle(out,&c,P,np,g_tokblob,g_tokblobl);
    if(g_ngram_on) ngram_save(op,0);
    t1_save(op,0);
    { char a1[1100],b1[1100]; size_t sl; snprintf(a1,sizeof a1,"%s.env",p0); snprintf(b1,sizeof b1,"%s.env",out);
      uint8_t*sb=read_file(a1,&sl); if(sb){ FILE*g=fopen(b1,"wb"); if(g){ fwrite(sb,1,sl,g); fclose(g); } free(sb); } }
    printf("[avg] %d checkpoints -> %s (%zu params%s%s)\n",nin,out,np,g_ngram_on?", n-gram table re-encoded":"",g_attg||g_xn?", side weights":""); free(P); free(tmp); }
/* cudalm ngfold <out> <in> <rows> <data_dir> [tokens]: shrink the hashed n-gram table to <rows> = R/f rows with no retrain.
 * Keys are h % R, so h % (R/f) == (h % R) % (R/f): rows r, r+R/f, ... collide in the smaller table exactly as they merge here.
 * Each merged row is the usage-weighted mean (row counts over the first [tokens] of <data_dir>/train.bin, +1 smoothing).
 * Table moments restart at zero (re-encoded q4, old sref kept); weights, Wng and side weights are copied unchanged. */
static void cmd_ngfold(int argc,char**argv){
    if(argc<6){ fprintf(stderr,"usage: cudalm ngfold <out> <in> <rows> <data_dir> [tokens=2e8]\n"); exit(1); }
    side_apply(argv[3]); char pk[1024],op[1100],out[1024]; model_resolve(argv[3],pk,sizeof pk);
    Cfg c; Tok*tk=NULL; float*P=model_load_full(pk,&c,&tk); size_t np=param_count(&c);
    gC=c; cuda_init(); ngram_infer(pk);
    if(!g_ngram_on||g_eg_on||g_ng_tri){ fprintf(stderr,"[ngfold] needs a plain n-gram table (not engram, not NGRAM_TRI)\n"); exit(1); }
    int R=g_ng_R, R2=atoi(argv[4]), De=g_ng_De, O=g_ng_order;
    if(R2<=0||R2>=R||R%R2||O<1||O>8){ fprintf(stderr,"[ngfold] rows %d must divide %d (and be smaller)\n",R2,R); exit(1); }
    size_t nt=(size_t)R*De; float*t=(float*)malloc(nt*4); ck(cudaMemcpy(t,g_ng,nt*4,cudaMemcpyDeviceToHost),"fold d2h");
    double*cnt=(double*)calloc(R,8);
    { char p[1100]; snprintf(p,sizeof p,"%s/train.bin",argv[5]); FILE*f=fopen(p,"rb"); if(!f){ perror(p); exit(1); }
      size_t want=argc>6?(size_t)atof(argv[6]):200000000, got=0, nb=1<<20; uint16_t*b=(uint16_t*)malloc(nb*2); unsigned hs[8]; int nh=0;
      while(got<want){ size_t n=fread(b,2,nb,f); if(!n) break;
          for(size_t i=0;i<n;i++){ memmove(hs+1,hs,7*sizeof(unsigned)); hs[0]=b[i]; if(nh<O){ nh++; if(nh<O) continue; }
              unsigned h=2166136261u; for(int j=O-1;j>=0;j--) h=(h^hs[j])*16777619u;   /* = k_ngram_key */
              h^=h>>15; cnt[h%(unsigned)R]+=1; }
          got+=n; }
      fclose(f); free(b); printf("[ngfold] row usage from %zu tokens of %s\n",got,p); }
    int fo=R/R2; float*t2=(float*)malloc((size_t)R2*De*4); double tot=0,mix=0;
    for(int r=0;r<R2;r++){ double w=0,mx=0;
        for(int k=0;k<fo;k++){ double x=cnt[r+(size_t)k*R2]; w+=x+1; tot+=x; if(x>mx) mx=x; }
        mix+=w-fo-mx;                                  /* usage landing on a row it now shares with a busier one */
        for(int d=0;d<De;d++){ double s=0; for(int k=0;k<fo;k++){ size_t rr=r+(size_t)k*R2; s+=(cnt[rr]+1)*t[rr*De+d]; }
            t2[(size_t)r*De+d]=(float)(s/w); } }
    float sref=g_ngq_spH.sref; g_ng_R=R2; g_ng=(float*)dmalloc((size_t)R2*De*4);
    ck(cudaMemcpy(g_ng,t2,(size_t)R2*De*4,cudaMemcpyHostToDevice),"fold h2d");
    if(g_ng_q4){ ngram_q4_setup(); g_ngq_spH.sref=sref; ck(cudaMemcpy(g_ngq_spD,&g_ngq_spH,sizeof(QSpanD),cudaMemcpyHostToDevice),"fold sref"); }
    model_path(argv[2],out,sizeof out); snprintf(op,sizeof op,"%s.opt",out);
    model_save_bundle(out,&c,P,np,g_tokblob,g_tokblobl); ngram_save(op,0); t1_save(op,0);
    { char a1[1100],b1[1100]; snprintf(a1,sizeof a1,"%s.env",pk); snprintf(b1,sizeof b1,"%s.env",out);   /* env with the new row count */
      FILE*fi=fopen(a1,"r"),*fw=fopen(b1,"w"); char l[1024]; int seen=0;
      if(fw){ if(fi) while(fgets(l,sizeof l,fi)){ if(!strncmp(l,"NGRAM_ROWS=",11)){ fprintf(fw,"NGRAM_ROWS=%d\n",R2); seen=1; } else fputs(l,fw); }
          if(!seen) fprintf(fw,"NGRAM_ROWS=%d\n",R2); fclose(fw); }
      if(fi) fclose(fi); }
    printf("[ngfold] %s: n-gram table %d -> %d rows (-%zu params); %.1f%% of row usage now shares a row with a busier one -> %s\n",
           pk,R,R2,(size_t)(R-R2)*De,100.0*mix/(tot>0?tot:1),out);
    free(t); free(t2); free(cnt); free(P); }
static void cmd_eval(int argc,char**argv){
    if(argc<4){ fprintf(stderr,
        "usage: cudalm eval <name|model.bin> <data_dir> [batches] [batch_size]\n"
        "  forward-only: mean val loss + per-layer MoE expert utilisation.\n"
        "  Safe to run while training is in progress.\n"
        "  CUDALM_EVAL_AT=<file of token offsets> scores those T-token windows of train.bin instead\n"
        "  (batches = all of them) and writes 'offset<TAB>mean loss' rows to CUDALM_EVAL_OUT.\n"
        "  CUDALM_EVAL_SPANS=<file of 'start len' train.bin spans> scores every target in each span (same\n"
        "  coverage for any T; spans shorter than 4097 are skipped) and writes one row per eot-delimited\n"
        "  document: 'doc start<TAB>tokens<TAB>summed loss'.\n"); exit(1); }
    char path[1024]; model_path(argv[2],path,sizeof(path));
    FILE*e=fopen(path,"rb"); if(e)fclose(e); else snprintf(path,sizeof(path),"%s",argv[2]);
    Cfg c; Tok*tk=NULL; float*Ph=model_load_full(path,&c,&tk);
    if(!tk){ fprintf(stderr,"model has no tokenizer (untrained?)\n"); exit(1); }
    int nb=argc>4?atoi(argv[4]):20; if(nb<1)nb=1;
    int B =argc>5?atoi(argv[5]):1;  if(B<1)B=1;
    char p[1024]; size_t nval=0; uint16_t*vl=NULL;
    /* CUDALM_EVAL_AT: per-window scoring of fixed train.bin offsets (data filtering probes) */
    size_t nat=0,*at=NULL; FILE*atf=NULL,*ato=NULL; int*wc=NULL; char*wf=NULL;
    if(getenv("CUDALM_EVAL_SPANS")){ FILE*f=fopen(getenv("CUDALM_EVAL_SPANS"),"r"); if(!f){perror(getenv("CUDALM_EVAL_SPANS"));exit(1);}
        size_t cap=1<<16; at=(size_t*)malloc(cap*8); wc=(int*)malloc(cap*4); wf=(char*)malloc(cap); unsigned long long s0,sl;
        while(fscanf(f,"%llu %llu",&s0,&sl)==2){ if(sl<4097) continue;
            for(size_t k=0;;k++){ size_t o=(size_t)s0+k*c.T; int cf=0, last=o+c.T+1>s0+sl;
                if(last){ size_t e=o; o=(size_t)(s0+sl)-c.T-1; cf=(int)(e-o); }
                if(nat+1>=cap){ cap*=2; at=(size_t*)realloc(at,cap*8); wc=(int*)realloc(wc,cap*4); wf=(char*)realloc(wf,cap); }
                at[nat]=o; wc[nat]=cf; wf[nat]=k==0; nat++;
                if(last||o+c.T+1==s0+sl) break; } }
        fclose(f); printf("[eval] spans -> %zu windows\n",nat);
        snprintf(p,sizeof(p),"%s/train.bin",argv[3]); atf=fopen(p,"rb");
        const char*op=getenv("CUDALM_EVAL_OUT"); ato=fopen(op?op:"evaldocs.tsv","w");
        if(!nat||!atf||!ato){ fprintf(stderr,"[eval] CUDALM_EVAL_SPANS: no spans, no %s or bad CUDALM_EVAL_OUT\n",p); exit(1); } }
    else if(getenv("CUDALM_EVAL_AT")){ FILE*f=fopen(getenv("CUDALM_EVAL_AT"),"r"); if(!f){perror(getenv("CUDALM_EVAL_AT"));exit(1);}
        size_t cap=1<<16; at=(size_t*)malloc(cap*8); unsigned long long x;
        while(fscanf(f,"%llu",&x)==1){ if(nat==cap) at=(size_t*)realloc(at,(cap*=2)*8); at[nat++]=(size_t)x; } fclose(f);
        snprintf(p,sizeof(p),"%s/train.bin",argv[3]); atf=fopen(p,"rb");
        const char*op=getenv("CUDALM_EVAL_OUT"); ato=fopen(op?op:"evalat.tsv","w");
        if(!nat||!atf||!ato){ fprintf(stderr,"[eval] CUDALM_EVAL_AT: no offsets, no %s or bad CUDALM_EVAL_OUT\n",p); exit(1); } }
    if(atf){ vl=(uint16_t*)malloc(((size_t)c.T+1)*2); nval=0; } else {
    snprintf(p,sizeof(p),"%s/val.bin",argv[3]);
    { size_t s; uint8_t*b=read_file(p,&s); if(b){ vl=(uint16_t*)b; nval=s/2; } }
    if(!vl){ snprintf(p,sizeof(p),"%s/train.bin",argv[3]);
             size_t s; uint8_t*b=read_file(p,&s); if(!b){fprintf(stderr,"no val.bin or train.bin in %s\n",argv[3]);exit(1);}
             vl=(uint16_t*)b; nval=s/2; printf("[eval] val.bin missing, using train.bin\n"); } }
    if(atf) nb=(int)((nat+B-1)/B);
    gC=c; cuda_init(); Off o; calc_offsets(&c,&o);
    ngram_infer(path);
    size_t np=param_count(&c);
    int N=B*c.T;
    printf("[eval] %s | %d batches of B=%d T=%d (%d tokens each) | val pool %.1fM tokens\n",
           argv[2],nb,B,c.T,N,nval/1e6);
    float*P=(float*)dmalloc(np*4); ck(cudaMemcpy(P,Ph,np*4,cudaMemcpyHostToDevice),"P h2d");
    rope_tables(&c);
    GA*a=ga_new(N,1);                       /* train=1 only for the loss buffers */
    int*tokG=(int*)dmalloc((size_t)N*4),*tgtG=(int*)dmalloc((size_t)N*4);
    int*tok=(int*)malloc(sizeof(int)*N),*tgt=(int*)malloc(sizeof(int)*N);
    int E=c.n_exp;
    double *ecount=NULL, *e0=NULL; if(E>0){ ecount=(double*)calloc((size_t)c.L*E,sizeof(double)); e0=(double*)calloc((size_t)c.L*E,sizeof(double)); }
    int Km=g_topk<=1?1:(g_topk>E?E:g_topk);
    double tot=0; size_t span=(size_t)c.T+2;
    /* ---- assistant-only loss, reported alongside the all-token number -------
     * Once training masks the prompt, the model is deliberately no longer
     * learning to predict user/system/tool tokens -- and the all-token val loss
     * counts exactly those, so it rises for a reason that has nothing to do
     * with quality. Measured on this model: after masking, a chat-heavy slice
     * went 1.5404 -> 1.6502 while raw prose went 2.3575 -> 2.1840.
     * The assistant-only figure is the one that stays comparable across the
     * masking boundary. Both are printed: the all-token number keeps continuity
     * with every measurement taken before masking existed.
     * Costs one host-side pass over the per-position losses -- no extra
     * forward, so the all-token number is bit-identical to before. */
    MaskSpec ms; memset(&ms,0,sizeof(ms));
    int can_mask = (c.tmpl==2);
    if(can_mask) mask_build(&ms,tk);
    float*lp=(float*)malloc((size_t)N*4);
    int  *tm=(int*)malloc((size_t)N*4);
    double tot_a=0; size_t n_a=0;
    g_rng=12345;                            /* fixed seed: comparable across runs */
    for(int b=0;b<nb;b++){
        for(int i=0;i<B;i++){ size_t st;
            if(atf){ size_t k=(size_t)b*B+i; if(k>=nat) k=nat-1; lm_fseek64(atf,(long long)at[k]*2,SEEK_SET);
                if(fread(vl,2,(size_t)c.T+1,atf)!=(size_t)c.T+1){ fprintf(stderr,"[eval] offset %zu past the end\n",at[k]); exit(1); } st=0; }
            else st=((size_t)(urand()*4294967296.0))%(nval-span);
            for(int j=0;j<c.T;j++){ int r=i*c.T+j; tok[r]=vl[st+j]; tgt[r]=vl[st+j+1]; } }
        ck(cudaMemcpy(tokG,tok,(size_t)N*4,cudaMemcpyHostToDevice),"tok h2d");
        ck(cudaMemcpy(tgtG,tgt,(size_t)N*4,cudaMemcpyHostToDevice),"tgt h2d");
        gpu_forward(P,&o,a,tokG);
        tot += gpu_loss(P,&o,a,tgtG,N,lp);
        if(ato&&wc){ static size_t ds; static long long dn; static double dl;   /* per-document accumulation, rows in span order */
            for(int i=0;i<B&&(size_t)b*B+i<nat;i++){ size_t k=(size_t)b*B+i;
                if(wf[k]&&dn){ fprintf(ato,"%zu\t%lld\t%.4f\n",ds,dn,dl); dn=0; }
                for(int j=wc[k];j<c.T;j++){ if(!dn){ ds=at[k]+j+1; dl=0; } dl+=lp[i*c.T+j]; dn++;
                    if(tgt[i*c.T+j]==c.eot){ fprintf(ato,"%zu\t%lld\t%.4f\n",ds,dn,dl); dn=0; } } }
            if((size_t)(b+1)*B>=nat&&dn){ fprintf(ato,"%zu\t%lld\t%.4f\n",ds,dn,dl); dn=0; } }
        else if(ato) for(int i=0;i<B&&(size_t)b*B+i<nat;i++){ double r=0; for(int j=0;j<c.T;j++) r+=lp[i*c.T+j];
            fprintf(ato,"%zu\t%.5f\n",at[(size_t)b*B+i],r/c.T); }
        if(can_mask){
            memcpy(tm,tgt,(size_t)N*4);
            for(int i=0;i<B;i++) mask_apply(&ms,tok+(size_t)i*c.T,tm+(size_t)i*c.T,c.T,c.eot);
            for(int i=0;i<N;i++) if(tm[i]>=0){ tot_a+=lp[i]; n_a++; }
            if(b==0&&getenv("CUDALM_MASK_DUMP")){ int nm=0,sq=-1;
                for(int i=0;i<N;i++) if(tm[i]<0){ nm++; if(sq<0) sq=i/c.T; }
                printf("[mask] batch 0: %.1f%% of targets masked\n",100.0*nm/N);
                if(sq>=0){ printf("[mask] seq %d, targets; masked in [[ ]]:\n",sq);
                    for(int j=0,in=0;j<c.T;j++){ int r=sq*c.T+j, mk=tm[r]<0;
                        if(mk!=in){ printf(mk?"[[":"]]"); in=mk; } tok_print(tk,tgt[r]); }
                    printf("\n"); } fflush(stdout); }
        }
        if(E>0) for(int l=0;l<c.L;l++){ int Kl=(Km>1&&g_topkL[l]>0&&g_topkL[l]<Km)?g_topkL[l]:Km, El=lyE(&c,l);
            if(!lyT(&c,l)) continue; if(Kl>El) Kl=El;
            for(int k=0;k<Kl;k++){ int*off=a->hoff+((size_t)l*Km+k)*(E+1);
                for(int x=0;x<El;x++){ ecount[(size_t)l*E+x] += off[x+1]-off[x]; if(!k) e0[(size_t)l*E+x] += off[x+1]-off[x]; } } }
        if((b+1)%10==0){ printf("  %d/%d batches\n",b+1,nb); fflush(stdout); }
    }
    if(ato){ fclose(ato); fclose(atf); printf("[eval] %zu windows scored -> %s\n",nat,getenv("CUDALM_EVAL_OUT")?getenv("CUDALM_EVAL_OUT"):wc?"evaldocs.tsv":"evalat.tsv"); }
    double mean=tot/nb;
    printf("\n[eval] val loss %.4f   perplexity %.2f   (%d batches x %d tokens)\n",
           mean, exp(mean), nb, N);
    if(can_mask && n_a>0){
        double ma=tot_a/(double)n_a;
        printf("[eval] assistant-only %.4f   perplexity %.2f   (%.1f%% of tokens)\n",
               ma, exp(ma), 100.0*(double)n_a/((double)N*nb));
        printf("       ^ the metric that stays comparable once prompt masking is on\n");
    }
    free(lp); free(tm);
    if(E>0){
        printf("\n[experts] tokens routed per expert, per layer (%% of layer's tokens)\n");
        printf("  ideal = 100/E %% each; 'dead' = <10%% of ideal, 'cold' = <50%% of ideal\n");
        double worst=0; int deadtot=0, np_=0; double *pc=(double*)calloc(E,sizeof(double)); int *alive=(int*)calloc(E,sizeof(int));
        for(int l=0;l<c.L;l++){ int El=lyE(&c,l), P=lyT(&c,l)==2; if(!lyT(&c,l)) continue;
            double s=0; for(int x=0;x<El;x++) s+=ecount[(size_t)l*E+x];
            printf("  L%-2d%c ",l,P?'P':'M');
            double mx=0, id=100.0/El; int dead=0, cold=0;
            for(int x=0;x<El;x++){ double f=s>0?100.0*ecount[(size_t)l*E+x]/s:0;
                printf("%5.1f ",f); if(f>mx)mx=f; if(f<0.1*id) dead++; else if(P) alive[x]++; if(f<0.5*id) cold++;
                if(P) pc[x]+=ecount[(size_t)l*E+x]; }
            /* max/ideal: 1.0 is perfectly balanced, E means total collapse */
            double imb = mx/id;
            printf("  | max %.1f%% imbalance %.2fx dead %d cold %d\n", mx, imb, dead, cold);
            if(imb>worst) worst=imb;
            if(P) np_++; else deadtot+=dead;
        }
        if(np_){ int PE=0; for(int l=0;l<c.L;l++) if(lyT(&c,l)==2) PE=lyE(&c,l);
            double s=0,mx=0; for(int x=0;x<PE;x++){ s+=pc[x]; if(pc[x]>mx) mx=pc[x]; }
            int pd=0, pcold=0, spec=0; double al=0;
            for(int x=0;x<PE;x++){ double r=s>0?pc[x]*PE/s:0; if(r<0.1) pd++; if(r<0.5) pcold++;
                if(r>=0.1&&alive[x]<np_) spec++; al+=alive[x]; }
            printf("  pool (%d experts summed over %d P layers): imbalance %.2fx, dead %d, cold %d | %d alive pool-wide but\n"
                   "       unused by >=1 layer (specialised); each expert serves %.2f of %d P layers on average\n",
                   PE,np_,s>0?mx*PE/s:0,pd,pcold,spec,al/PE,np_);
            deadtot+=pd; }
        free(pc); free(alive);
        printf("\n  worst-layer imbalance %.2fx (1.00 = perfect), %d dead (MoE layers + pool-wide)\n", worst, deadtot);
        if(Km>1){ double w0=0; int wl=0;
            for(int l=0;l<c.L;l++){ int El=lyE(&c,l); double s=0,mx=0; if(!lyT(&c,l)) continue;
                for(int x=0;x<El;x++){ double v=e0[(size_t)l*E+x]; s+=v; if(v>mx)mx=v; }
                if(s>0&&mx*El/s>w0){ w0=mx*El/s; wl=l; } }
            printf("  top-1 slot only: worst imbalance %.2fx (L%d)\n",w0,wl); }
        free(e0);
        if(worst>2.5 || deadtot>0)
            printf("  >> ROUTER IS UNBALANCED: raise MOE_AUX (currently %.3f) and restart from a checkpoint\n", MOE_AUX);
        else printf("  >> routing looks healthy\n");
    }
}

/* ===================== chatbench (coherence / sensibility) ===============
 * bench file, '|' separated, '#' comments, "\n" in text = newline:
 *   M|cat|prompt|correct|wrong1|wrong2...   likelihood multiple choice
 *   G|cat|prompt|kw1,kw2,...                sampled generation, scored on
 *     termination, repetition, keyword hits, and prompt-relevance
 *     (logP(reply|own prompt) - logP(reply|other prompts), per token).
 * Everything is batched B sequences x T tokens through one forward. */
typedef struct { char type; char cat[32]; char*prompt; char*f[9]; int nf; } CBItem;
typedef struct { const float*P; const Off*o; GA*a; int*tokG,*tgtG,*tok,*tgt; float*lp; int B,T; } CBCtx;
static char cb_lc(char c){ return c>='A'&&c<='Z'?(char)(c+32):c; }
static void cb_unescape(char*s){ char*w=s; for(char*r=s;*r;r++){ if(r[0]=='\\'&&r[1]=='n'){*w++='\n';r++;} else *w++=*r; } *w=0; }
static int* cb_fmt(const Tok*tk,int tmpl,const char*prompt,const char*reply,int*n){
    size_t L=strlen(prompt)+(reply?strlen(reply):0)+64; char*s=(char*)malloc(L);
    if(tmpl==2)      snprintf(s,L,"<|user|>\n%s\n<|assistant|>\n%s",prompt,reply?reply:"");
    else if(tmpl==1) snprintf(s,L,"### Instruction:\n%s\n\n### Response:\n%s",prompt,reply?reply:"");
    else             snprintf(s,L,"%s\n%s",prompt,reply?reply:"");
    int*ids=tok_encode(tk,s,n); free(s); return ids;
}
/* lp[k] = sum log P(seq[k][st[k]..len[k]-1] | prefix) */
static void cb_score(CBCtx*x,int n,int**seq,const int*len,const int*st,double*lp){
    int T=x->T,N=x->B*T;
    for(int k0=0;k0<n;k0+=x->B){
        int nb=n-k0<x->B?n-k0:x->B;
        memset(x->tok,0,(size_t)N*4); memset(x->tgt,0,(size_t)N*4);
        for(int i=0;i<nb;i++){ const int*s=seq[k0+i]; int L=len[k0+i]<T+1?len[k0+i]:T+1;
            for(int j=0;j<L-1;j++){ x->tok[i*T+j]=s[j]; x->tgt[i*T+j]=s[j+1]; } }
        ck(cudaMemcpy(x->tokG,x->tok,(size_t)N*4,cudaMemcpyHostToDevice),"tok h2d");
        ck(cudaMemcpy(x->tgtG,x->tgt,(size_t)N*4,cudaMemcpyHostToDevice),"tgt h2d");
        gpu_forward(x->P,x->o,x->a,x->tokG); gpu_loss(x->P,x->o,x->a,x->tgtG,N,x->lp);
        for(int i=0;i<nb;i++){ int k=k0+i, L=len[k]<T+1?len[k]:T+1; double s=0;
            for(int j=st[k]-1;j<L-1;j++) s-=x->lp[i*T+j]; lp[k]=s; }
    }
}
static int cb_lcp(const int*a,int na,const int*b,int nb){ int i=0; while(i<na&&i<nb&&a[i]==b[i]) i++; return i; }
static void cmd_chatbench(int argc,char**argv){
    if(argc<4){ fprintf(stderr,
        "usage: cudalm chatbench <name|model.bin> <bench.txt> [ngen=128] [temp=0.7] [topp=0.9] [reppen=1.15] [nsamp=4] [T=256]\n"); exit(1); }
    char path[1024]; model_path(argv[2],path,sizeof(path));
    FILE*e=fopen(path,"rb"); if(e)fclose(e); else snprintf(path,sizeof(path),"%s",argv[2]);
    Cfg c; Tok*tk=NULL; float*Ph=model_load_full(path,&c,&tk);
    if(!tk){ fprintf(stderr,"model has no tokenizer (untrained?)\n"); exit(1); }
    int ngen=argc>4?atoi(argv[4]):128; float temp=argc>5?(float)atof(argv[5]):0.7f;
    float topp=argc>6?(float)atof(argv[6]):0.9f, rep=argc>7?(float)atof(argv[7]):1.15f;
    int nsamp=argc>8?atoi(argv[8]):4, T=argc>9?atoi(argv[9]):256, B=16;
    int greedy=temp<=1e-4f, stopid=chat_stop_id(&c,tk); if(greedy) nsamp=1;
    size_t flen; char*fb=(char*)read_file(argv[3],&flen); if(!fb){ fprintf(stderr,"cannot read %s\n",argv[3]); exit(1); }
    fb=(char*)realloc(fb,flen+1); fb[flen]=0;
    CBItem*it=(CBItem*)calloc(4096,sizeof(CBItem)); int ni=0;
    for(char*ln=strtok(fb,"\n"); ln&&ni<4096; ln=strtok(NULL,"\n")){
        int l=(int)strlen(ln); while(l>0&&(ln[l-1]=='\r'||ln[l-1]==' ')) ln[--l]=0;
        if(l<3||ln[0]=='#'||ln[1]!='|'||(ln[0]!='M'&&ln[0]!='G')) continue;
        CBItem*q=&it[ni]; q->type=ln[0]; char*f[12]; int nf=0; char*s=ln+2;
        while(nf<12){ f[nf++]=s; char*b=strchr(s,'|'); if(!b) break; *b=0; s=b+1; }
        if(nf<3||(q->type=='M'&&nf<5)) continue;
        snprintf(q->cat,sizeof(q->cat),"%s",f[0]); q->prompt=f[1]; cb_unescape(q->prompt);
        for(int i=2;i<nf&&q->nf<9;i++){ cb_unescape(f[i]); q->f[q->nf++]=f[i]; }
        ni++;
    }
    c.T=T; cfg_derive(&c); gC=c; cuda_init(); Off o; calc_offsets(&c,&o); ngram_infer(path);
    size_t np=param_count(&c);
    float*P=(float*)dmalloc(np*4); ck(cudaMemcpy(P,Ph,np*4,cudaMemcpyHostToDevice),"P h2d");
    rope_tables(&c);
    int N=B*T; CBCtx x={P,&o,ga_new(N,1),(int*)dmalloc((size_t)N*4),(int*)dmalloc((size_t)N*4),
        (int*)malloc((size_t)N*4),(int*)malloc((size_t)N*4),(float*)malloc((size_t)N*4),B,T};
    g_rng=12345;
    printf("[chatbench] %s | %d items | T=%d B=%d | gen %d tok, %s temp=%.2f topp=%.2f rep=%.2f nsamp=%d stop=%d\n",
           argv[2],ni,T,B,ngen,greedy?"greedy":"sample",temp,topp,rep,nsamp,stopid);
    double t0=wall();
    /* ---------------- M: likelihood multiple choice ---------------- */
    int nseq=0; for(int i=0;i<ni;i++) if(it[i].type=='M') nseq+=it[i].nf;
    if(nseq){
        int**sq=(int**)malloc(sizeof(int*)*nseq),*ln=(int*)malloc(4*nseq),*st=(int*)malloc(4*nseq);
        double*lp=(double*)malloc(8*nseq); int k=0;
        for(int i=0;i<ni;i++) if(it[i].type=='M'){
            int pn; int*pi=cb_fmt(tk,c.tmpl,it[i].prompt,NULL,&pn);
            for(int j=0;j<it[i].nf;j++){ int n; int*s=cb_fmt(tk,c.tmpl,it[i].prompt,it[i].f[j],&n);
                s=(int*)realloc(s,sizeof(int)*(n+1)); s[n++]=tk->eot;
                int b=cb_lcp(pi,pn,s,n); st[k]=b>0?b:1; sq[k]=s; ln[k]=n; k++; }
            free(pi); }
        cb_score(&x,nseq,sq,ln,st,lp);
        char cats[64][32]; int nc=0; double ca[64][4]={{0}};    /* n, acc, accnorm, chance */
        printf("\n=== multiple choice (correct option listed first in file) ===\n");
        k=0; double A=0,AN=0,CH=0; int NM=0;
        for(int i=0;i<ni;i++) if(it[i].type=='M'){
            int bs=0,bn=0; double bv=-1e30,bnv=-1e30;
            for(int j=0;j<it[i].nf;j++){ double v=lp[k+j], vn=v/(ln[k+j]-st[k+j]);
                if(v>bv){bv=v;bs=j;} if(vn>bnv){bnv=vn;bn=j;} }
            int ci=0; while(ci<nc&&strcmp(cats[ci],it[i].cat)) ci++;
            if(ci==nc&&nc<64) strcpy(cats[nc++],it[i].cat);
            ca[ci][0]++; ca[ci][1]+=bs==0; ca[ci][2]+=bn==0; ca[ci][3]+=1.0/it[i].nf;
            A+=bs==0; AN+=bn==0; CH+=1.0/it[i].nf; NM++;
            printf("  %s%s %-12s %s  -> %s\n",bs==0?"ok":"--",bn==0?"ok":"--",it[i].cat,it[i].prompt,it[i].f[bn]);
            k+=it[i].nf; }
        printf("\n  %-14s %5s %8s %8s %8s\n","category","n","acc","accnorm","chance");
        for(int ci=0;ci<nc;ci++) printf("  %-14s %5.0f %7.1f%% %7.1f%% %7.1f%%\n",cats[ci],ca[ci][0],
            100*ca[ci][1]/ca[ci][0],100*ca[ci][2]/ca[ci][0],100*ca[ci][3]/ca[ci][0]);
        printf("  %-14s %5d %7.1f%% %7.1f%% %7.1f%%\n","ALL",NM,100*A/NM,100*AN/NM,100*CH/NM);
        for(int i=0;i<nseq;i++) free(sq[i]); free(sq); free(ln); free(st); free(lp);
    }
    /* ---------------- G: sampled generation ---------------- */
    int*gi=(int*)malloc(4*ni),ng=0; for(int i=0;i<ni;i++) if(it[i].type=='G') gi[ng++]=i;
    if(ng){
        int R=ng*nsamp; int**gs=(int**)malloc(sizeof(int*)*R),*gp=(int*)malloc(4*R),*gl=(int*)malloc(4*R),*ge=(int*)calloc(R,4);
        float*lg=(float*)malloc((size_t)B*c.V*4); int gentot=0;
        for(int r=0;r<R;r++){ int n; int*s=cb_fmt(tk,c.tmpl,it[gi[r/nsamp]].prompt,NULL,&n);
            gs[r]=(int*)malloc(4*T); if(n>T-8) n=T-8; memcpy(gs[r],s,4*n); free(s); gp[r]=gl[r]=n; }
        for(int r0=0;r0<R;r0+=B){
            int nb=R-r0<B?R-r0:B;
            for(int step=0;step<ngen;step++){
                int live=0; for(int i=0;i<nb;i++) if(!ge[r0+i]&&gl[r0+i]<T) live++;
                if(!live) break;
                memset(x.tok,0,(size_t)N*4);
                for(int i=0;i<nb;i++) memcpy(x.tok+i*T,gs[r0+i],4*gl[r0+i]);
                ck(cudaMemcpy(x.tokG,x.tok,(size_t)N*4,cudaMemcpyHostToDevice),"tok h2d");
                gpu_forward(P,&o,x.a,x.tokG);
                for(int i=0;i<nb;i++) g_abt(x.a->fn+((size_t)i*T+gl[r0+i]-1)*c.D,P+o.emb,x.a->logits+(size_t)i*c.V,1,c.D,c.V);
                ck(cudaMemcpy(lg,x.a->logits,(size_t)nb*c.V*4,cudaMemcpyDeviceToHost),"logits d2h");
                for(int i=0;i<nb;i++){ int r=r0+i; if(ge[r]||gl[r]>=T) continue;
                    int t=sample_ex(lg+(size_t)i*c.V,c.V,temp,0,topp,rep,gs[r]+gp[r],gl[r]-gp[r],greedy);
                    if(t==tk->eot||t==stopid){ ge[r]=1; continue; }
                    gs[r][gl[r]++]=t; gentot++; }
            }
            printf("  generated %d/%d\n",r0+nb,R); fflush(stdout);
        }
        /* relevance: re-score each reply under its own prompt and 3 others */
        int K=ng>=4?4:ng, M=R*K; int**sq=(int**)malloc(sizeof(int*)*M),*ln=(int*)malloc(4*M),*st=(int*)malloc(4*M);
        double*lp=(double*)malloc(8*M);
        for(int r=0;r<R;r++) for(int q=0;q<K;q++){ int g=(r/nsamp+q*(ng/K>0?ng/K:1))%ng, n, m=r*K+q;
            int*s=cb_fmt(tk,c.tmpl,it[gi[g]].prompt,NULL,&n); int rl=gl[r]-gp[r]; if(n+rl>T+1) rl=T+1-n; if(rl<0) rl=0;
            s=(int*)realloc(s,4*(n+rl+1)); memcpy(s+n,gs[r]+gp[r],4*rl);
            sq[m]=s; st[m]=n; ln[m]=n+rl; }
        cb_score(&x,M,sq,ln,st,lp);
        printf("\n=== generation ===\n");
        double s_eot=0,s_len=0,s_d2=0,s_r4=0,s_deg=0,s_kw=0,s_rel=0,s_win=0; int nrel=0;
        char cats[64][32]; int nc=0; double ca[64][6]={{0}};           /* n, eot, kw, deg, rel, win */
        char*txt=(char*)malloc(65536),*low=(char*)malloc(65536);
        for(int r=0;r<R;r++){
            CBItem*q=&it[gi[r/nsamp]]; int*g=gs[r]+gp[r], n=gl[r]-gp[r];
            int tl=0; for(int j=0;j<n&&tl<60000;j++){ memcpy(txt+tl,tk->dec[g[j]],tk->declen[g[j]]); tl+=tk->declen[g[j]]; } txt[tl]=0;
            int ub=0,r4=0; for(int j=0;j+1<n;j++){ int d=1; for(int m=0;m<j;m++) if(g[m]==g[j]&&g[m+1]==g[j+1]){d=0;break;} ub+=d; }
            for(int j=1;j+3<n;j++) for(int m=0;m<j;m++) if(!memcmp(g+m,g+j,16)){ r4++; break; }
            double d2=n>1?(double)ub/(n-1):1, rr=n>3?(double)r4/(n-3):0; int deg=rr>0.3;
            for(int j=0;j<=tl;j++) low[j]=cb_lc(txt[j]);
            int kw=0; if(q->nf>0){ char kb[1024]; snprintf(kb,sizeof(kb),"%s",q->f[0]);
                for(char*w=strtok(kb,",");w&&!kw;w=strtok(NULL,",")){ while(*w==' ')w++; char lw[256]; int k=0;
                    for(;w[k]&&k<255;k++) lw[k]=cb_lc(w[k]); lw[k]=0; if(k&&strstr(low,lw)) kw=1; } }
            double rel=0; int win=0, ok=n>0;
            if(ok&&K>1){ double own=lp[r*K]/n, oth=0; win=1;
                for(int k=1;k<K;k++){ double v=lp[r*K+k]/n; oth+=v; if(v>=own) win=0; }
                rel=own-oth/(K-1); s_rel+=rel; s_win+=win; nrel++; }
            s_eot+=ge[r]; s_len+=n; s_d2+=d2; s_r4+=rr; s_deg+=deg; s_kw+=kw;
            int ci=0; while(ci<nc&&strcmp(cats[ci],q->cat)) ci++;
            if(ci==nc&&nc<64) strcpy(cats[nc++],q->cat);
            ca[ci][0]++; ca[ci][1]+=ge[r]; ca[ci][2]+=kw; ca[ci][3]+=deg; ca[ci][4]+=rel; ca[ci][5]+=win;
            if(r%nsamp==0) printf("\n[%s] Q: %s\n",q->cat,q->prompt);
            for(int j=0;j<tl;j++) if(txt[j]=='\n') txt[j]='~';
            printf("  #%d %s kw=%d d2=%.2f rep4=%.2f rel=%+.2f%s | %s\n",r%nsamp,ge[r]?"eot":"CUT",kw,d2,rr,rel,deg?" DEGEN":"",txt);
        }
        printf("\n  %-14s %5s %7s %7s %7s %8s %7s\n","category","n","eot","kw","degen","rel/tok","relwin");
        for(int ci=0;ci<nc;ci++){ double n=ca[ci][0]; printf("  %-14s %5.0f %6.1f%% %6.1f%% %6.1f%% %+8.3f %6.1f%%\n",cats[ci],n,
            100*ca[ci][1]/n,100*ca[ci][2]/n,100*ca[ci][3]/n,ca[ci][4]/n,100*ca[ci][5]/n); }
        printf("  %-14s %5d %6.1f%% %6.1f%% %6.1f%% %+8.3f %6.1f%%\n","ALL",R,100*s_eot/R,100*s_kw/R,100*s_deg/R,
            nrel?s_rel/nrel:0,nrel?100*s_win/nrel:0);
        printf("  mean len %.1f tok | distinct-2 %.3f | rep-4 %.3f | relevance chance win = %.0f%%\n",
            s_len/R,s_d2/R,s_r4/R,100.0/K);
        printf("[chatbench] %d tokens generated\n",gentot);
    }
    printf("[chatbench] done in %.1fs\n",wall()-t0);
}

/* ===================== lmeval (HellaSwag / ARC-Easy / PIQA / LAMBADA) =====
 * Zero-shot, lm-eval-harness prompt formats; lmeval_prep.py writes <dir>/<task>.tsv:
 *   gold \t context \t cont0 \t cont1 ...      (\n \t \\ escaped)
 * >1 continuation: acc = argmax sum logP, acc_norm = argmax logP/bytes.
 * 1 continuation (LAMBADA): acc = every continuation token is the greedy argmax,
 * ppl = exp(-mean logP). Sequences are [eot] context continuation, left-truncated
 * to T; T = per-task pow2 >= longest, rows per forward = N/T (one GA, fixed N). */
static void le_unescape(char*s){ char*w=s; for(char*r=s;*r;r++){ if(*r=='\\'&&r[1]){ r++; *w++=*r=='n'?'\n':*r=='t'?'\t':*r; } else *w++=*r; } *w=0; }
static void cmd_lmeval(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: cudalm lmeval <name|model.bin> [tasks|all] [n=all] [Tmax=1024]\n"
        "  tasks: comma list of hellaswag,arc_easy,arc_challenge,piqa,lambada,arithmark (files $CUDALM_BENCH_DIR/<task>.tsv, default dir bench)\n"
        "         or .tsv paths; n>0 scores only the first n items of each task\n"
        "  ends with the Open SLM intelligence index (acc_norm, chance-normalised:\n"
        "  (HellaSwag + mean(ARC-E,ARC-C) + PIQA + 0.65 ArithMark-3) / 3.65; missing tasks drop out, >=2 needed)\n"
        "  also a margin index (logP/byte gold - best wrong, same weights); CUDALM_LMEVAL_ITEMS=<file> dumps per-item margins for lmeval-vs\n"); exit(1); }
    g_fp8gemm=0;    /* per-tensor FP8 scales see the whole batch (padding included): scores would depend on batching */
    char path[1024]; model_path(argv[2],path,sizeof(path));
    FILE*e=fopen(path,"rb"); if(e)fclose(e); else snprintf(path,sizeof(path),"%s",argv[2]);
    Cfg c; Tok*tk=NULL; float*Ph=model_load_full(path,&c,&tk);
    if(!tk){ fprintf(stderr,"model has no tokenizer (untrained?)\n"); exit(1); }
    char tl[1024]; snprintf(tl,sizeof tl,"%s",argc>3&&strcmp(argv[3],"all")?argv[3]:"hellaswag,arc_easy,arc_challenge,piqa,lambada,arithmark");
    int nmax=argc>4?atoi(argv[4]):0, Tm=64, Tq=argc>5?atoi(argv[5]):1024; if(Tq>c.T) Tq=c.T;
    while(Tm*2<=Tq) Tm*=2;
    char bdd[1100]; const char*bd=getenv("CUDALM_BENCH_DIR");
    if(!bd||!*bd){ snprintf(bdd,sizeof bdd,"%s" PSEP "bench",exe_dir()); bd=bdd; }
    for(char*tn=tl;*tn;){ size_t k=strcspn(tn,","); char fp[1200];       /* missing task files: download + convert once */
        snprintf(fp,sizeof fp,"%s" PSEP "%.*s.tsv",bd,(int)k,tn);
        if(!strstr(fp+strlen(bd),".tsv.tsv") && !d_exists(fp)){ char sp[1200],cl[3000];
            snprintf(sp,sizeof sp,"%s" PSEP "bench" PSEP "lmeval_prep.py",exe_dir());
            if(!d_exists(sp)) snprintf(sp,sizeof sp,"%s" PSEP "lmeval_prep.py",exe_dir());
            snprintf(cl,sizeof cl,"%s -u \"%s\" \"%s\"",py_cmd(),sp,bd);
            printf("[lmeval] %s missing - fetching the benchmark files\n",fp); fflush(stdout);
            if(run_cmd(cl)) fprintf(stderr,"[lmeval] lmeval_prep.py failed (needs python + pyarrow; CUDALM_PYTHON picks the interpreter)\n");
            break; }
        tn+=k; if(*tn) tn++; }
    int N=Tm>16384?Tm:16384, D=c.D, V=c.V;
    c.T=Tm; cfg_derive(&c); gC=c; cuda_init(); Off o; calc_offsets(&c,&o); ngram_infer(path);
    size_t np=param_count(&c);
    float*P=(float*)dmalloc(np*4); ck(cudaMemcpy(P,Ph,np*4,cudaMemcpyHostToDevice),"P h2d");
    rope_tables(&c); GA*a=ga_new(N,1);
    int*tokG=(int*)dmalloc((size_t)N*4),*tgtG=(int*)dmalloc((size_t)N*4),*tok=(int*)malloc((size_t)N*4),*tgt=(int*)malloc((size_t)N*4);
    float*lp=(float*)malloc((size_t)N*4),*lg=(float*)malloc((size_t)CH_ROWS*V*4);
    printf("[lmeval] %s | Tmax %d | %d tokens per forward | tasks %s\n",argv[2],Tm,N,tl);
    char sum[16][260]; int ns=0; double t0=wall();
    /* Open SLM intelligence index inputs (acc_norm %): hellaswag, arc_easy, arc_challenge, piqa, arithmark */
    static const char*ixn[5]={"hellaswag","arc_easy","arc_challenge","piqa","arithmark"}; double ixs[5],ixm[5],ixe[5]; int ixh[5]={0};
    const char*ip=getenv("CUDALM_LMEVAL_ITEMS"); FILE*itf=ip&&*ip?fopen(ip,"w"):NULL;
    if(itf) fprintf(itf,"task\titem\tcorrect\tmargin\n");
    for(char*tn=tl,*tc; tn&&*tn; tn=tc){
        tc=strchr(tn,','); if(tc) *tc++=0;
        char fp[1200]; if(strstr(tn,".tsv")) snprintf(fp,sizeof fp,"%s",tn); else snprintf(fp,sizeof fp,"%s/%s.tsv",bd,tn);
        size_t fl; char*fb=(char*)read_file(fp,&fl); if(!fb){ printf("[lmeval] %s: cannot read %s\n",tn,fp); continue; }
        fb=(char*)realloc(fb,fl+1); fb[fl]=0;
        int ni=0,ic=0,nq=0,qc=0,*gold=NULL,*nc=NULL,*q0=NULL,**sq=NULL,*ln=NULL,*st=NULL,*by=NULL,maxl=0,trunc=0;
        for(char*l=fb,*nx; *l&&(nmax<=0||ni<nmax); l=nx){
            char*nl=strchr(l,'\n'); nx=nl?nl+1:l+strlen(l); if(nl) *nl=0;
            int k=(int)strlen(l); while(k>0&&l[k-1]=='\r') l[--k]=0;
            char*f[18]; int nf=0; for(char*s=l; nf<18; ){ f[nf++]=s; char*t=strchr(s,'\t'); if(!t) break; *t=0; s=t+1; }
            if(nf<3) continue;
            for(int j=1;j<nf;j++) le_unescape(f[j]);
            if(ni==ic){ ic=ic?2*ic:1024; gold=(int*)realloc(gold,4*ic); nc=(int*)realloc(nc,4*ic); q0=(int*)realloc(q0,4*ic); }
            gold[ni]=atoi(f[0]); nc[ni]=nf-2; q0[ni]=nq;
            int cn; int*ci=tok_encode(tk,f[1],&cn);
            for(int j=2;j<nf;j++){
                if(nq==qc){ qc=qc?2*qc:4096; sq=(int**)realloc(sq,sizeof(int*)*qc); ln=(int*)realloc(ln,4*qc); st=(int*)realloc(st,4*qc); by=(int*)realloc(by,4*qc); }
                int kn; int*ki=tok_encode(tk,f[j],&kn); int L=1+cn+kn, s0=1+cn;
                int*s=(int*)malloc(4*(size_t)L); s[0]=tk->eot; memcpy(s+1,ci,4*(size_t)cn); memcpy(s+s0,ki,4*(size_t)kn); free(ki);
                if(L>Tm+1){ int d=L-(Tm+1); memmove(s,s+d,4*(size_t)(Tm+1)); L-=d; s0-=d; trunc++; if(s0<1) s0=1; }
                sq[nq]=s; ln[nq]=L; st[nq]=s0; by[nq]=(int)strlen(f[j]); if(L>maxl) maxl=L; nq++; }
            free(ci); ni++;
        }
        free(fb);
        if(!ni){ printf("[lmeval] %s: no items\n",tn); continue; }
        const char*ft=getenv("CUDALM_LMEVAL_T"); int T=ft&&atoi(ft)>64?atoi(ft):64;   /* floor, for padding checks */
        while(T<maxl-1) T*=2; if(T>Tm) T=Tm; gC.T=T; int B=N/T; double tt=wall();
        double*ll=(double*)malloc(8*nq); char*gr=(char*)calloc(nq,1); int greedy=1;
        for(int i=0;i<ni;i++) if(nc[i]!=1){ greedy=0; break; }
        for(int k0=0;k0<nq;k0+=B){
            int nb=nq-k0<B?nq-k0:B;
            memset(tok,0,(size_t)N*4); memset(tgt,0,(size_t)N*4);
            for(int i=0;i<nb;i++){ const int*s=sq[k0+i]; for(int j=0;j<ln[k0+i]-1;j++){ tok[i*T+j]=s[j]; tgt[i*T+j]=s[j+1]; } }
            ck(cudaMemcpy(tokG,tok,(size_t)N*4,cudaMemcpyHostToDevice),"tok h2d");
            ck(cudaMemcpy(tgtG,tgt,(size_t)N*4,cudaMemcpyHostToDevice),"tgt h2d");
            gpu_forward(P,&o,a,tokG); gpu_loss(P,&o,a,tgtG,N,lp);
            for(int i=0;i<nb;i++){ int k=k0+i; double s=0; for(int j=st[k]-1;j<ln[k]-1;j++) s-=lp[i*T+j]; ll[k]=s; }
            if(!greedy) continue;
            for(int i0=0;i0<nb;){                   /* logits of continuation rows, <= CH_ROWS per pass */
                int r=0,i1=i0; while(i1<nb&&r+ln[k0+i1]-st[k0+i1]<=CH_ROWS){ int k=k0+i1, m=ln[k]-st[k];
                    g_abt(a->fn+((size_t)i1*T+st[k]-1)*D,P+o.emb,a->logits+(size_t)r*V,m,D,V); r+=m; i1++; }
                if(i1==i0){ fprintf(stderr,"[lmeval] continuation longer than %d tokens\n",CH_ROWS); exit(1); }
                ck(cudaMemcpy(lg,a->logits,(size_t)r*V*4,cudaMemcpyDeviceToHost),"logits d2h");
                r=0; for(int i=i0;i<i1;i++){ int k=k0+i, ok=1;
                    for(int j=st[k];j<ln[k];j++,r++){ const float*z=lg+(size_t)r*V; int bi=0; for(int v=1;v<V;v++) if(z[v]>z[bi]) bi=v; ok&=bi==sq[k][j]; }
                    gr[k]=(char)ok; }
                i0=i1; }
        }
        double A=0,AN=0,CH=0,LL=0,MS=0,MQ=0;
        const char*b=tn+strlen(tn); while(b>tn&&b[-1]!='/'&&b[-1]!='\\') b--; int bl=(int)strcspn(b,".");
        for(int i=0;i<ni;i++){ int k=q0[i],ok; double m;
            if(greedy){ A+=gr[k]; LL+=ll[k]; ok=gr[k]; m=ll[k]/(by[k]>0?by[k]:1); }
            else{ int bs=0,bn=0; double bw=-1e30;    /* margin: gold logP/byte minus the best wrong one */
                for(int j=0;j<nc[i];j++){ if(ll[k+j]>ll[k+bs]) bs=j; if(ll[k+j]/by[k+j]>ll[k+bn]/by[k+bn]) bn=j;
                    if(j!=gold[i]&&ll[k+j]/by[k+j]>bw) bw=ll[k+j]/by[k+j]; }
                A+=bs==gold[i]; AN+=bn==gold[i]; CH+=1.0/nc[i]; ok=bn==gold[i]; m=ll[k+gold[i]]/by[k+gold[i]]-bw; }
            MS+=m; MQ+=m*m; if(itf) fprintf(itf,"%.*s\t%d\t%d\t%.6f\n",bl,b,i,ok,m); }
        double p=A/ni,pn=AN/ni,mm=MS/ni,me=ni>1?sqrt(fmax(MQ/ni-mm*mm,0)/(ni-1)):0;
        if(!greedy) for(int q=0;q<5;q++) if((int)strlen(ixn[q])==bl&&!strncmp(b,ixn[q],bl)){ ixs[q]=100*pn; ixm[q]=mm; ixe[q]=me; ixh[q]=1; }
        if(greedy) snprintf(sum[ns],260,"  %-13s %6d   acc %6.2f%% +-%.2f   ppl %9.2f          logP/B %+.4f +-%.4f   (T %d, %.1fs%s)",tn,ni,100*p,100*sqrt(p*(1-p)/ni),exp(-LL/ni),mm,me,T,wall()-tt,trunc?", truncated":"");
        else snprintf(sum[ns],260,"  %-13s %6d   acc %6.2f%% +-%.2f   acc_norm %6.2f%% +-%.2f   chance %4.1f%%   margin %+.4f +-%.4f   (T %d, %.1fs)",tn,ni,
                      100*p,100*sqrt(p*(1-p)/ni),100*pn,100*sqrt(pn*(1-pn)/ni),100*CH/ni,mm,me,T,wall()-tt);
        printf("%s\n",sum[ns]); fflush(stdout); if(ns<15) ns++;
        for(int k=0;k<nq;k++) free(sq[k]); free(sq); free(ln); free(st); free(by); free(gold); free(nc); free(q0); free(ll); free(gr);
    }
    printf("\n[lmeval] summary %s (%.1fs)\n",argv[2],wall()-t0); for(int i=0;i<ns;i++) printf("%s\n",sum[i]);
    {   /* Open SLM intelligence index: N(s,c)=100(s-c)/(100-c); (HS + ARC + PIQA + 0.65 ArithMark) / 3.65,
         * ARC = mean(ARC-E, ARC-C) before normalising; missing components drop out of both sums (>=2 needed) */
        double cs[4]={0},cc[4]={25,25,50,25},cw[4]={1,1,1,0.65},num=0,den=0; int ch[4]={ixh[0],ixh[1]||ixh[2],ixh[3],ixh[4]},nc=0;
        static const char*cn[4]={"HellaSwag","ARC","PIQA","ArithMark-3"}; char line[300]; int lo=0;
        cs[0]=ixs[0]; cs[2]=ixs[3]; cs[3]=ixs[4];
        if(ch[1]) cs[1]=ixh[1]&&ixh[2]?(ixs[1]+ixs[2])/2:ixh[1]?ixs[1]:ixs[2];
        for(int q=0;q<4;q++) if(ch[q]){ double v=100*(cs[q]-cc[q])/(100-cc[q]); num+=cw[q]*v; den+=cw[q]; nc++;
            lo+=snprintf(line+lo,sizeof line-lo,"  %s %.2f",cn[q],v); }
        if(nc>=2) printf("\n[lmeval] Open SLM intelligence index %.2f   (normalised:%s%s%s)\n",num/den,line,
                         ch[1]&&!(ixh[1]&&ixh[2])?"; ARC from one of ARC-E/ARC-C only":"",nc<4?"; partial":"");
        else printf("\n[lmeval] Open SLM intelligence index: needs at least 2 of HellaSwag, ARC, PIQA, ArithMark-3\n");
        /* margin index: the same weights over mean margins (100 x nats/byte); far less noisy than accuracy near chance */
        double mn=0,mv=0,md=0,m4[4],e4[4];
        m4[0]=ixm[0]; e4[0]=ixe[0]; m4[2]=ixm[3]; e4[2]=ixe[3]; m4[3]=ixm[4]; e4[3]=ixe[4];
        if(ixh[1]&&ixh[2]){ m4[1]=(ixm[1]+ixm[2])/2; e4[1]=0.5*sqrt(ixe[1]*ixe[1]+ixe[2]*ixe[2]); }
        else if(ch[1]){ m4[1]=ixh[1]?ixm[1]:ixm[2]; e4[1]=ixh[1]?ixe[1]:ixe[2]; }
        for(int q=0;q<4;q++) if(ch[q]){ mn+=cw[q]*m4[q]; mv+=cw[q]*cw[q]*e4[q]*e4[q]; md+=cw[q]; }
        if(nc>=2) printf("[lmeval] margin index %.3f +-%.3f   (100 x mean logP/byte gold - best wrong; same weights)\n",100*mn/md,100*sqrt(mv)/md);
    }
    if(itf){ fclose(itf); printf("[lmeval] per-item margins -> %s\n",ip); }
}

/* ===================== lens ============================================= */
/* cudalm lens <model> <data_dir|"text"> [tokens=16384] [T=1024]: logit lens. The residual stream after the embedding
 * (+ layer-0 n-gram), after each block's attention (Ln.att) and after each block (Ln) is read out through the final norm
 * and the tied unembedding, as if the model stopped there. Per stage: CE of the true next token, top-1 accuracy, agreement
 * with the full model's top-1, logP of that token, entropy. If <model>.bin.tlens exists (cudalm lens-tune), the tuned lens
 * is read out too. data_dir: windows spread over val.bin (else train.bin); CUDALM_LENS_OUT=<tsv> also writes the table.
 * text: adds a grid of each stage's top-1 guess per position (CUDALM_LENS_COLS positions, default 12; * = the true next
 * token, _ = space, | = newline). */
static void lens_cell(const Tok*tk,int t,char*o,int w){
    int k=0; for(int i=0;i<tk->declen[t]&&k<w;i++){ unsigned char ch=tk->dec[t][i]; o[k++]=ch==' '?'_':ch=='\n'?'|':ch<32||ch>126?'~':(char)ch; }
    o[k]=0; }
static float* lens_open(const char*name,char*path,Cfg*c,Tok**tk){
    g_fp8gemm=0; model_path(name,path,1024);
    FILE*e=fopen(path,"rb"); if(e)fclose(e); else snprintf(path,1024,"%s",name);
    return model_load_full(path,c,tk); }
static GA* lens_gpu(float*Ph,Cfg*c,int T,int N,const char*path,float**P,Off*o){
    c->T=T; cfg_derive(c); gC=*c; cuda_init(); calc_offsets(c,o); ngram_infer(path);
    size_t np=param_count(c); *P=(float*)dmalloc(np*4); ck(cudaMemcpy(*P,Ph,np*4,cudaMemcpyHostToDevice),"P h2d");
    rope_tables(c); return ga_new(N,0); }
/* stage s: 0 = embedding, 2l+1 = after block l's attention, 2l+2 = after block l */
static const float* lens_src(GA*a,int s,int L){ return s==0?a->a_in[0]:s&1?a->f_in[s/2]:s/2<L?a->a_in[s/2]:a->x_final; }
/* tuned-lens translator: Y = H + H A + b */
static const float* tl_apply(const float*H,const float*A,const float*b,float*Y,int N,int D){
    g_ab(H,A,Y,N,D,D); KL(k_tl_add,nblk((size_t)N*D,TPB),TPB)(Y,H,b,(size_t)N*D,D); return Y; }
/* <model>.tlens: "TLN1", stages, D, then every stage's A [D x D], then every stage's b [D] */
static int tl_load(const char*path,int D,int M,float**TA,float**TB){
    char p[1100]; snprintf(p,sizeof p,"%s.tlens",path); size_t n; int*h=(int*)read_file(p,&n);
    if(!h) return 0;
    if(n<12||memcmp(h,"TLN1",4)||h[1]!=M||h[2]!=D||n!=12+4*((size_t)M*D*D+(size_t)M*D)){ fprintf(stderr,"[lens] %s does not match this model, ignored\n",p); free(h); return 0; }
    const float*f=(const float*)(h+3); size_t nA=(size_t)M*D*D;
    *TA=(float*)dmalloc(nA*4); *TB=(float*)dmalloc((size_t)M*D*4);
    ck(cudaMemcpy(*TA,f,nA*4,cudaMemcpyHostToDevice),"tl A"); ck(cudaMemcpy(*TB,f+nA,(size_t)M*D*4,cudaMemcpyHostToDevice),"tl b");
    free(h); printf("[lens] tuned lens <- %s\n",p); return 1; }
static void lens_name(int s,char*nm){ if(!s) snprintf(nm,16,"emb"); else snprintf(nm,16,"L%d%s",(s-1)/2,s&1?".att":""); }
static void cmd_lens(int argc,char**argv){
    if(argc<4){ fprintf(stderr,"usage: cudalm lens <name|model.bin> <data_dir|\"text\"> [tokens=16384] [T=1024]\n"); exit(1); }
    char path[1024]; Cfg c; Tok*tk=NULL; float*Ph=lens_open(argv[2],path,&c,&tk);
    char vp[1200]; size_t vs=0; snprintf(vp,sizeof vp,"%s" PSEP "val.bin",argv[3]); uint16_t*vl=(uint16_t*)read_file(vp,&vs);
    if(!vl){ snprintf(vp,sizeof vp,"%s" PSEP "train.bin",argv[3]); vl=(uint16_t*)read_file(vp,&vs); }
    int txt=!vl, n=0, *ids=NULL, T, N, R, B=1;
    if(txt){ if(!tk){ fprintf(stderr,"model has no tokenizer\n"); exit(1); }
        int k; int*t=tok_encode(tk,argv[3],&k); n=k+1; if(n>c.T+1) n=c.T+1;
        ids=(int*)malloc(4*(size_t)n); ids[0]=tk->eot; memcpy(ids+1,t,4*(size_t)(n-1)); free(t);
        if(n<2){ fprintf(stderr,"[lens] empty text\n"); exit(1); }
        T=64; while(T<n-1) T*=2; if(T>c.T) T=c.T; N=T; R=n-1; }
    else { T=argc>5?atoi(argv[5]):1024; if(T>c.T) T=c.T; if(T<16) T=16;
        long long want=argc>4?atoll(argv[4]):16384; size_t nv=vs/2; B=(int)((want+T-1)/T); if(B<1) B=1;
        if((size_t)B*T+1>nv) B=(int)((nv-1)/T); if(B<1){ fprintf(stderr,"[lens] %s too small\n",vp); exit(1); }
        N=R=B*T; }
    float*P; Off o; GA*a=lens_gpu(Ph,&c,T,N,path,&P,&o);
    int D=c.D, V=c.V, L=c.L, S=2*L+1, ch=N<g_chrows?N:g_chrows;
    float*TA=NULL,*TB=NULL; int K=1+tl_load(path,D,S-1,&TA,&TB); float*Y=K>1?(float*)dmalloc((size_t)N*D*4):NULL;
    int*tok=(int*)calloc(N,4),*tgt=(int*)calloc(N,4),*fin=(int*)malloc(4*(size_t)R);
    int*tokG=(int*)dmalloc((size_t)N*4),*tgtG=(int*)dmalloc((size_t)N*4),*finG=(int*)dmalloc((size_t)N*4);
    float*outG=(float*)dmalloc((size_t)ch*16),*ho=(float*)malloc((size_t)ch*16);
    if(txt) for(int j=0;j<R;j++){ tok[j]=ids[j]; tgt[j]=ids[j+1]; }
    else { size_t nv=vs/2; for(int b=0;b<B;b++){ size_t off=(size_t)b*(nv-T-1)/B; for(int j=0;j<T;j++){ tok[b*T+j]=vl[off+j]; tgt[b*T+j]=vl[off+j+1]; } } }
    ck(cudaMemcpy(tokG,tok,(size_t)N*4,cudaMemcpyHostToDevice),"tok h2d"); ck(cudaMemcpy(tgtG,tgt,(size_t)N*4,cudaMemcpyHostToDevice),"tgt h2d");
    gpu_forward(P,&o,a,tokG);
    const Off*rd=g_prd?g_prd:&o; int cols=12; { const char*v=getenv("CUDALM_LENS_COLS"); if(v&&atoi(v)>0) cols=atoi(v); }
    if(cols>R) cols=R;
    double*st=(double*)calloc((size_t)2*S*5,8); int*grid=(int*)malloc(4*(size_t)2*S*cols);
    for(int k=0;k<K;k++) for(int q=0;q<S;q++){ int s=(q+S-1)%S;   /* the final stage first: its top-1 is the reference */
        double*z=st+((size_t)k*S+s)*5;
        if(k&&s==S-1){ memcpy(z,st+(size_t)(S-1)*5,40); memcpy(grid+(size_t)(S+s)*cols,grid+(size_t)s*cols,4*(size_t)cols); continue; }
        const float*src=lens_src(a,s,L); if(k) src=tl_apply(src,TA+(size_t)s*D*D,TB+(size_t)s*D,Y,N,D);
        e_rmsf(src,P,rd->nf,a->fn,a->fnr,N,D);
        for(int r0=0;r0<R;r0+=ch){ int m=R-r0<ch?R-r0:ch;
            g_abt(a->fn+(size_t)r0*D,P+o.emb,a->logits,m,D,V);
            KL(k_lens,m,TPB)(a->logits,tgtG+r0,s==S-1?NULL:finG+r0,outG,m,V);
            ck(cudaMemcpy(ho,outG,(size_t)m*16,cudaMemcpyDeviceToHost),"lens d2h");
            for(int i=0;i<m;i++){ const float*h=ho+4*(size_t)i; int r=r0+i, am=(int)h[1];
                if(s==S-1) fin[r]=am;
                z[0]-=h[0]; z[1]+=am==tgt[r]; z[2]+=am==fin[r]; z[3]+=h[3]; z[4]+=h[2];
                if(r<cols) grid[((size_t)k*S+s)*cols+r]=am; } }
        if(s==S-1) ck(cudaMemcpy(finG,fin,4*(size_t)R,cudaMemcpyHostToDevice),"fin h2d"); }
    printf("[lens] %s | %s | %d positions%s\n",argv[2],txt?"text":vp,R,txt?"":" (val windows)");
    if(txt){ char cell[16],nm[16];
        for(int k=0;k<K;k++){
            printf("\n  %s lens\n  %-8s",k?"tuned":"logit","input"); for(int j=0;j<cols;j++){ lens_cell(tk,tok[j],cell,7); printf(" %-8s",cell); }
            printf("\n  %-8s","next");  for(int j=0;j<cols;j++){ lens_cell(tk,tgt[j],cell,7); printf(" %-8s",cell); } printf("\n");
            for(int s=0;s<S;s++){ lens_name(s,nm); printf("  %-8s",nm);
                for(int j=0;j<cols;j++){ int t=grid[((size_t)k*S+s)*cols+j]; lens_cell(tk,t,cell,6); printf(" %c%-7s",t==tgt[j]?'*':' ',cell); }
                printf("\n"); } }
        if(cols<R) printf("  (first %d of %d positions; CUDALM_LENS_COLS for more)\n",cols,R); }
    const char*op=getenv("CUDALM_LENS_OUT"); FILE*of=op&&*op?fopen(op,"w"):NULL;
    if(of) fprintf(of,"stage\tce\ttop1\tagree\tlogp_final_top1\tentropy%s\n",K>1?"\ttce\tttop1\ttagree\ttlogp_final_top1\ttentropy":"");
    printf("\n  %-8s %8s %7s %7s %10s %8s%s\n","stage","CE","top1","=final","logP(fin1)","entropy",K>1?"   | tuned CE    top1  =final":"");
    for(int s=0;s<S;s++){ char nm[16]; lens_name(s,nm); const double*z=st+(size_t)s*5,*u=st+(size_t)(S+s)*5;
        printf("  %-8s %8.4f %6.2f%% %6.2f%% %10.4f %8.4f",nm,z[0]/R,100*z[1]/R,100*z[2]/R,z[3]/R,z[4]/R);
        if(K>1) printf("   | %8.4f %6.2f%% %6.2f%%",u[0]/R,100*u[1]/R,100*u[2]/R);
        printf("\n");
        if(of){ fprintf(of,"%s\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f",nm,z[0]/R,z[1]/R,z[2]/R,z[3]/R,z[4]/R);
            if(K>1) fprintf(of,"\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f",u[0]/R,u[1]/R,u[2]/R,u[3]/R,u[4]/R);
            fprintf(of,"\n"); } }
    if(of){ fclose(of); printf("[lens] table -> %s\n",op); }
}
/* cudalm lens-tune <model> <data_dir> [steps=500] [tokens/step=8192] [T=1024]: tuned lens (Belrose et al. 2023). One affine
 * translator per stage, y = h + hA + b (A, b start at 0 = the logit lens), trained on random train.bin windows to minimise
 * KL(full model || readout through the final norm + head) with the model frozen; Adam, lr 1e-3 decaying linearly to 0.
 * Writes <model>.bin.tlens, which `cudalm lens` then reads out next to the logit lens. */
static void cmd_lens_tune(int argc,char**argv){
    if(argc<4){ fprintf(stderr,"usage: cudalm lens-tune <name|model.bin> <data_dir> [steps=500] [tokens/step=8192] [T=1024]\n"); exit(1); }
    char path[1024]; Cfg c; Tok*tk=NULL; float*Ph=lens_open(argv[2],path,&c,&tk);
    int steps=argc>4?atoi(argv[4]):500, T=argc>6?atoi(argv[6]):1024; if(T>c.T) T=c.T; if(T<16) T=16; if(steps<1) steps=1;
    long long want=argc>5?atoll(argv[5]):8192; int B=(int)((want+T-1)/T); if(B<1) B=1; int N=B*T;
    char tp[1200]; snprintf(tp,sizeof tp,"%s" PSEP "train.bin",argv[3]); FILE*tf=fopen(tp,"rb"); size_t nt=0;
    if(tf){ lm_fseek64(tf,0,SEEK_END); nt=(size_t)lm_ftell64(tf)/2; }
    if(nt<(size_t)T+2){ fprintf(stderr,"[lens-tune] no usable %s\n",tp); exit(1); }
    uint16_t*win=(uint16_t*)malloc(2*(size_t)T);
    float*P; Off o; GA*a=lens_gpu(Ph,&c,T,N,path,&P,&o);
    int D=c.D, V=c.V, L=c.L, M=2*L, ch=N<g_chrows?N:g_chrows; const Off*rd=g_prd?g_prd:&o; size_t nA=(size_t)D*D;
    float*hA=(float*)calloc((size_t)M*nA,4),*hb=(float*)calloc((size_t)M*D,4),*mA=(float*)calloc((size_t)M*nA,4),*vA=(float*)calloc((size_t)M*nA,4);
    float*mb=(float*)calloc((size_t)M*D,4),*vb=(float*)calloc((size_t)M*D,4),*gA=(float*)malloc(nA*4),*gb=(float*)malloc((size_t)D*4),*hl=(float*)malloc((size_t)N*4);
    float*TA=(float*)dmalloc((size_t)M*nA*4),*TB=(float*)dmalloc((size_t)M*D*4),*PF=(float*)dmalloc((size_t)N*V*4);
    float*Y=(float*)dmalloc((size_t)N*D*4),*dF=(float*)dmalloc((size_t)N*D*4),*dY=(float*)dmalloc((size_t)N*D*4),*RI=(float*)dmalloc((size_t)N*4);
    float*lossG=(float*)dmalloc((size_t)N*4),*dA=(float*)dmalloc(nA*4),*db=(float*)dmalloc((size_t)D*4),*one=(float*)dmalloc((size_t)N*4),*Gs=(float*)dmalloc((size_t)D*4);
    ck(cudaMemcpy(TA,hA,(size_t)M*nA*4,cudaMemcpyHostToDevice),"TA0"); ck(cudaMemcpy(TB,hb,(size_t)M*D*4,cudaMemcpyHostToDevice),"TB0");
    for(int i=0;i<N;i++) hl[i]=1.f; ck(cudaMemcpy(one,hl,(size_t)N*4,cudaMemcpyHostToDevice),"ones");
    int*tok=(int*)malloc((size_t)N*4),*tokG=(int*)dmalloc((size_t)N*4); double*kl=(double*)calloc(M,8),*kl0=(double*)calloc(M,8);
    int rep=steps/10>0?steps/10:1, nrep=0; g_rng=0x7E115ULL; double t0=wall();
    printf("[lens-tune] %s | %d stages | %d steps x %d tokens (B=%d T=%d) from %s\n",argv[2],M,steps,N,B,T,tp);
    for(int it=0;it<steps;it++){
        for(int b=0;b<B;b++){ size_t off=(size_t)(urand()*(double)(nt-T-1)); lm_fseek64(tf,(long long)off*2,SEEK_SET);
            if(fread(win,2,(size_t)T,tf)!=(size_t)T){ fprintf(stderr,"[lens-tune] read error\n"); exit(1); }
            for(int j=0;j<T;j++) tok[b*T+j]=win[j]; }
        ck(cudaMemcpy(tokG,tok,(size_t)N*4,cudaMemcpyHostToDevice),"tok h2d"); gpu_forward(P,&o,a,tokG);
        e_rmsf(a->x_final,P,rd->nf,a->fn,a->fnr,N,D);
        for(int r0=0;r0<N;r0+=ch){ int m=N-r0<ch?N-r0:ch;
            g_abt(a->fn+(size_t)r0*D,P+o.emb,a->logits,m,D,V); KL(k_logsm,m,TPB)(a->logits,PF+(size_t)r0*V,m,V); }
        float lr=1e-3f*(1.0f-(float)it/steps); double c1=1-pow(0.9,it+1), c2=1-pow(0.999,it+1);
        for(int s=0;s<M;s++){
            const float*h=lens_src(a,s,L); tl_apply(h,TA+(size_t)s*nA,TB+(size_t)s*D,Y,N,D);
            e_rmsf(Y,P,rd->nf,a->fn,RI,N,D);
            for(int r0=0;r0<N;r0+=ch){ int m=N-r0<ch?N-r0:ch;
                g_abt(a->fn+(size_t)r0*D,P+o.emb,a->logits,m,D,V);
                KL(k_kld,m,TPB)(a->logits,PF+(size_t)r0*V,lossG+r0,m,V,1.0f/N);
                g_ab(a->logits,P+o.emb,dF+(size_t)r0*D,m,V,D); }
            ck(cudaMemset(dY,0,(size_t)N*D*4),"dY0"); e_rmsb(Y,P,rd->nf,RI,dF,dY,Gs,0,N,D);
            ck(cudaMemset(dA,0,nA*4),"dA0"); ck(cudaMemset(db,0,(size_t)D*4),"db0");
            g_atb(h,dY,dA,N,D,D); g_atb(one,dY,db,N,1,D);
            ck(cudaMemcpy(gA,dA,nA*4,cudaMemcpyDeviceToHost),"dA d2h"); ck(cudaMemcpy(gb,db,(size_t)D*4,cudaMemcpyDeviceToHost),"db d2h");
            ck(cudaMemcpy(hl,lossG,(size_t)N*4,cudaMemcpyDeviceToHost),"kl d2h");
            double sk=0; for(int i=0;i<N;i++) sk+=hl[i]; kl[s]+=sk/N; if(!it) kl0[s]=sk/N;
            float*pa=hA+(size_t)s*nA,*ma=mA+(size_t)s*nA,*va=vA+(size_t)s*nA,*pb=hb+(size_t)s*D,*mbs=mb+(size_t)s*D,*vbs=vb+(size_t)s*D;
            for(size_t i=0;i<nA+(size_t)D;i++){ int w=i<nA; size_t j=w?i:i-nA; float g=w?gA[j]:gb[j],*pm=w?ma+j:mbs+j,*pv=w?va+j:vbs+j;
                *pm=0.9f**pm+0.1f*g; *pv=0.999f**pv+0.001f*g*g;
                (w?pa:pb)[j]-=lr*(float)((*pm/c1)/(sqrt(*pv/c2)+1e-8)); }
            ck(cudaMemcpy(TA+(size_t)s*nA,pa,nA*4,cudaMemcpyHostToDevice),"A h2d"); ck(cudaMemcpy(TB+(size_t)s*D,pb,(size_t)D*4,cudaMemcpyHostToDevice),"b h2d"); }
        nrep++;
        if((it+1)%rep==0||it+1==steps){ char nm[16]; printf("[lens-tune] step %4d  KL:",it+1);
            for(int s=0;s<M;s++) if(s%4==0||s==M-1){ lens_name(s,nm); printf(" %s %.3f",nm,kl[s]/nrep); }
            printf("  (%.0fs)\n",wall()-t0); fflush(stdout); memset(kl,0,8*(size_t)M); nrep=0; } }
    char p[1100]; snprintf(p,sizeof p,"%s.tlens",path); FILE*f=fopen(p,"wb"); int h[3]={0,M,D}; memcpy(h,"TLN1",4);
    if(!f||fwrite(h,4,3,f)!=3||fwrite(hA,4,(size_t)M*nA,f)!=(size_t)M*nA||fwrite(hb,4,(size_t)M*D,f)!=(size_t)M*D){ fprintf(stderr,"[lens-tune] cannot write %s\n",p); exit(1); }
    fclose(f); printf("[lens-tune] translators -> %s  (first-step KL emb %.3f, L0 %.3f)\n",p,kl0[0],kl0[2]);
}

/* ===================== bench ============================================ */
static void cmd_bench(int argc,char**argv){
    if(argc<3){ fprintf(stderr,
        "usage: cudalm bench <preset> [iters] [batch] [n_experts] [ctx] [window]\n"
        "  n_experts matters: a preset alone gives a DENSE model, and on the real\n"
        "  220M the MoE is 38.7%% of the step -- benchmarking without it measures\n"
        "  the wrong 60%%.\n"); exit(1); }
    Cfg c; if(!preset(argv[2],&c)){ fprintf(stderr,"unknown preset '%s'\n",argv[2]); exit(1); }
    { char err[200]; if(!shape_env(&c,err,sizeof err)){ fprintf(stderr,"bench: %s\n",err); exit(1); } }
    int iters=argc>3?atoi(argv[3]):20;
    if(argc>5){ c.n_exp=atoi(argv[5]); if(c.n_exp==1) c.n_exp=0; }
    if(argc>6){ int t=atoi(argv[6]); if(t>0) c.T=t; }
    if(argc>7){ c.win=atoi(argv[7]); if(c.win<0) c.win=0; }
    if(!layout_apply(&c)) exit(1);
    cfg_derive(&c);
    gC=c; cuda_init(); Off o; calc_offsets(&c,&o);
    size_t np=param_count(&c);
    int B=(argc>4)?atoi(argv[4]):auto_batch(&c,np); int N=B*c.T;
    moe_stash_decide(&c,np,N);
    printf("[bench] %s params=%.2fM D=%d L=%d H=%d KV=%d F=%d T=%d B=%d V=%d\n",
           argv[2],np/1e6,c.D,c.L,c.H,c.KV,c.F,c.T,B,c.V);
    printf("[bench] act=%.2f GB state=%.2f GB\n", ga_bytes(N,1)/1e9, opt_state_bytes(np)/1e9);
    float*Ph=(float*)malloc(np*4); for(size_t i=0;i<np;i++)Ph[i]=rnd_normal()*0.02f;
    float*P=(float*)dmalloc(np*4); ck(cudaMemcpy(P,Ph,np*4,cudaMemcpyHostToDevice),"P h2d");
    float*G=dzero(np);
    bf16*M=(bf16*)dmalloc(np*2),*Vv=(bf16*)dmalloc(np*2);
    ck(cudaMemset(M,0,np*2),"zero M"); ck(cudaMemset(Vv,0,np*2),"zero V");
    rope_tables(&c);
    GA*a=ga_new(N,1); a->tr=1; int*tokG=(int*)dmalloc((size_t)N*4),*tgtG=(int*)dmalloc((size_t)N*4);
    int*tok=(int*)malloc(sizeof(int)*N),*tgt=(int*)malloc(sizeof(int)*N);
    for(int i=0;i<N;i++){ tok[i]=(int)(urand()*c.V); tgt[i]=(int)(urand()*c.V); }
    ck(cudaMemcpy(tokG,tok,(size_t)N*4,cudaMemcpyHostToDevice),"h2d");
    ck(cudaMemcpy(tgtG,tgt,(size_t)N*4,cudaMemcpyHostToDevice),"h2d");
    for(int w=0;w<3;w++){ gpu_forward(P,&o,a,tokG); gpu_backward(P,G,&o,a,tokG,tgtG); gpu_adam(P,G,M,Vv,np,1e-4f,w+1); }
    ck(cudaDeviceSynchronize(),"warmup");
    double t0=wall();
    for(int i=0;i<iters;i++){ gpu_forward(P,&o,a,tokG); gpu_backward(P,G,&o,a,tokG,tgtG); gpu_adam(P,G,M,Vv,np,1e-4f,i+1); }
    ck(cudaDeviceSynchronize(),"bench");
    double dt=wall()-t0;
    double toks=(double)N*iters;
    double flops=6.0*np*toks;
    size_t fr=0,tt=0; cudaMemGetInfo(&fr,&tt);
    printf("[bench] %d iters in %.2fs -> %.1f ms/step | %.0f tok/s | ~%.2f TFLOP/s | %s used %.2f GB\n",
#ifdef CUDALM_CPU
           iters,dt,dt*1000/iters,toks/dt,flops/dt/1e12,"system RAM",(tt-fr)/1e9);
#else
           iters,dt,dt*1000/iters,toks/dt,flops/dt/1e12,"VRAM",(tt-fr)/1e9);
#endif
    prof_report();
}

/* ===================== grow (model expansion) ===========================
 * Take a trained checkpoint and enlarge it, then keep training, instead of
 * starting from scratch. Two axes:
 *
 *   --layers N   insert new layers. EXACTLY function-preserving: a new layer is
 *                a clone of the one before it with `wo` and `w2` zeroed, and
 *                both sit on residual branches, so the layer is an identity at
 *                step 0. No scaling, no noise. It does not stay stuck either --
 *                dL/dwo = att^T . d(f_in) is non-zero, so the branch unfreezes
 *                on the first update (the ReZero / LoRA-B=0 trick).
 *                Cloning rather than randomising means the new layer starts
 *                from an attention pattern that already works.
 *
 *   --ffn F      widen every expert's FFN. Duplicate columns of w1/w3, then
 *                DIVIDE the matching rows of w2 by how many copies each unit
 *                got, so the summed output is unchanged. Function-preserving
 *                only at --noise 0; but with zero noise the duplicated units
 *                receive identical gradients forever and the added capacity is
 *                wasted, so a little symmetry-breaking is the point.
 *
 * The correctness gate is exact and cheap: run `eval` before and after. For a
 * layers-only grow at noise 0 the val loss must come out IDENTICAL. If it does
 * not, the surgery is wrong -- do not spend GPU hours finding that out later. */
/* grow: carry the model config and the sidecars eval/run need (.env, .opt.ng, .opt.t1). Per-layer blocks of
 * the new layer l come from its source layer src[l] (inserted layers are zero-output, so any copy is exact);
 * layer indices (ngram/engram injection points) move to the source layer's new index. */
static void grow_sidecars(const char*in,const char*out,const int*src,const int*ident,int Lold,int Lnew,int D){
    int map[64]; for(int l=0;l<Lnew;l++) if(!ident[l]) map[src[l]]=l;
    char a[1200],b[1200]; size_t n; uint8_t*d;
    snprintf(a,sizeof a,"%s.env",in); snprintf(b,sizeof b,"%s.env",out);
    if((d=read_file(a,&n))){ FILE*f=fopen(b,"wb"); d=(uint8_t*)realloc(d,n+1); d[n]=0;
        for(char*l=(char*)d,*e;l&&*l;l=e){ e=strchr(l,'\n'); if(e) *e++=0; l[strcspn(l,"\r")]=0;
            if(!strncmp(l,"NGRAM_LAYER=",12)){ int v=atoi(l+12); fprintf(f,"NGRAM_LAYER=%d\n",v>=0&&v<Lold?map[v]:v); }
            else if(!strncmp(l,"NGRAM_LAYERS=",13)){ fprintf(f,"NGRAM_LAYERS="); int k=0;
                for(char*t=l+13;*t;){ int v=atoi(t); fprintf(f,"%s%d",k++?",":"",v>=0&&v<Lold?map[v]:v); t+=strcspn(t,","); if(*t) t++; }
                fprintf(f,"\n"); }
            else fprintf(f,"%s\n",l); }
        fclose(f); free(d); printf("[grow] config %s -> %s\n",a,b); }
    snprintf(a,sizeof a,"%s.opt.ng",in); snprintf(b,sizeof b,"%s.opt.ng",out);
    if((d=read_file(a,&n))){ if(n>=4+24*4&&!memcmp(d,"NGR2",4)){ int*h=(int*)(d+4); for(int i=0;i<h[10]&&i<8;i++) if(h[11+i]<Lold) h[11+i]=map[h[11+i]]; }
        FILE*f=fopen(b,"wb"); fwrite(d,1,n,f); fclose(f); free(d); printf("[grow] memory table %s -> %s\n",a,b); }
    snprintf(a,sizeof a,"%s.opt.t1",in); snprintf(b,sizeof b,"%s.opt.t1",out);
    if((d=read_file(a,&n))){ int*h=(int*)(d+4); size_t o=4+9*4; FILE*f=fopen(b,"wb");
        int hn[9]; memcpy(hn,h,36); hn[1]=Lnew; if(h[7]>=0&&h[7]<Lold) hn[7]=map[h[7]];
        fwrite(d,1,4,f); fwrite(hn,4,9,f);
        if(h[0]) for(int part=0;part<3;part++){ size_t per=(size_t)D*D*(part?2:4);   /* Wg fp32, then M, V 16-bit */
            for(int l=0;l<Lnew;l++) fwrite(d+o+(size_t)src[l]*per,1,per,f); o+=(size_t)Lold*per; }
        if(h[3]){ size_t w=(size_t)h[4]*D*8; fwrite(d+o,1,w,f); o+=w; }                /* ngram Wk(+M,V): no layers */
        if(h[5]){ size_t per=(size_t)h[5]*4; for(int l=0;l<Lnew;l++) fwrite(d+o+(size_t)src[l]*per,1,per,f); o+=(size_t)Lold*per; }
        fclose(f); printf("[grow] side weights %s -> %s%s\n",a,b,o==n?"":"  (SIZE MISMATCH - check it)"); free(d); }
    snprintf(a,sizeof a,"%s.opt.fe",in); snprintf(b,sizeof b,"%s.opt.fe",out);
    if(lm_copyfile(a,b)) printf("[grow] factorized embedding %s -> %s\n",a,b);
    snprintf(a,sizeof a,"%s.opt.x1",in);
    if(d_exists(a)) printf("[grow] note: %s (canon/shared side params) is not carried over\n",a);
}
static int cmd_grow(int argc,char**argv){
    if(argc<4){ fprintf(stderr,
        "usage: cudalm grow <in|name> <out.bin> [--layers N] [--ffn F] [--noise x]\n"
        "  --layers N   grow to N layers (new layers are zero-init identities)\n"
        "  --ffn F      widen each expert FFN to F (duplicate + rescale)\n"
        "  --noise x    relative symmetry-breaking noise on duplicated FFN units\n"
        "               (default 0.01; 0 = exact function preservation)\n"
        "  Verify with: cudalm eval <in> <data> 20 1  and the same on <out>.\n"
        "  Layers-only growth MUST give an identical val loss.\n"); return 1; }
    const char*in=argv[2],*out=argv[3];
    int newL=0,newF=0; float noise=0.01f;
    for(int i=4;i<argc;i++){
        if(!strcmp(argv[i],"--layers")&&i+1<argc) newL=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--ffn")&&i+1<argc) newF=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--noise")&&i+1<argc) noise=(float)atof(argv[++i]);
        else { fprintf(stderr,"unknown option %s\n",argv[i]); return 1; }
    }
    char inpath[1024];
    if(strstr(in,".bin")) snprintf(inpath,sizeof(inpath),"%s",in);
    else model_path(in,inpath,sizeof(inpath));

    Cfg c0; Tok*tk=NULL; float*P0=model_load_full(inpath,&c0,&tk);
    if(P0&&(c0.pool||c0.mod)){ fprintf(stderr,"grow: UniPool / per-layer-layout models not supported\n"); return 1; }
    if(!tk){ fprintf(stderr,"%s has no bundled tokenizer\n",inpath); return 1; }
    Cfg c1=c0;
    if(newL) c1.L=newL;
    if(newF) c1.F=newF;
    cfg_derive(&c1);
    if(c1.L<c0.L){ fprintf(stderr,"cannot shrink layers (%d -> %d)\n",c0.L,c1.L); return 1; }
    if(c1.F<c0.F){ fprintf(stderr,"cannot shrink ffn (%d -> %d)\n",c0.F,c1.F); return 1; }
    if(c1.L>64){ fprintf(stderr,"layer offsets are sized for 64 layers\n"); return 1; }
    if(c1.L==c0.L && c1.F==c0.F){ fprintf(stderr,"nothing to grow\n"); return 1; }

    Off o0,o1; calc_offsets(&c0,&o0); calc_offsets(&c1,&o1);
    size_t n0=param_count(&c0), n1=param_count(&c1);
    float*P1=(float*)calloc(n1,4);
    if(!P1){ fprintf(stderr,"OOM allocating %.1f MB\n",n1*4/1e6); return 1; }

    /* Which new layers are copies and which are inserted identities. Spread the
     * insertions evenly rather than appending them all at the end. */
    int Lold=c0.L,Lnew=c1.L,add=Lnew-Lold;
    int src[64],ident[64],ni=0;
    for(int k=0;k<Lold;k++){
        src[ni]=k; ident[ni]=0; ni++;
        int want=(int)(((long long)(k+1)*add)/Lold), have=(int)(((long long)k*add)/Lold);
        for(int j=have;j<want && ni<Lnew;j++){ src[ni]=k; ident[ni]=1; ni++; }
    }
    while(ni<Lnew){ src[ni]=Lold-1; ident[ni]=1; ni++; }   /* rounding remainder */

    int D=c0.D,KD=c0.KD,E=c0.n_exp>0?c0.n_exp:1,F0=c0.F,F1=c1.F;
    memcpy(P1+o1.emb,P0+o0.emb,(size_t)c0.V*D*4);
    P1[o1.nf]=0; memcpy(P1+o1.nf,P0+o0.nf,(size_t)D*4);

    /* how many times FFN unit j is duplicated, so w2 can be divided by it */
    int*cnt=(int*)calloc(F0,sizeof(int));
    for(int f=0;f<F1;f++) cnt[f%F0]++;

    for(int l=0;l<Lnew;l++){
        int s=src[l];
        memcpy(P1+o1.an1[l],P0+o0.an1[s],(size_t)D*4);
        memcpy(P1+o1.wq[l], P0+o0.wq[s], (size_t)D*D*4);
        memcpy(P1+o1.wk[l], P0+o0.wk[s], (size_t)D*KD*4);
        memcpy(P1+o1.wv[l], P0+o0.wv[s], (size_t)D*KD*4);
        memcpy(P1+o1.an2[l],P0+o0.an2[s],(size_t)D*4);
        if(c0.n_exp>0) memcpy(P1+o1.wr[l],P0+o0.wr[s],(size_t)D*c0.n_exp*4);
        /* wo: zeroed for an inserted layer -> attention branch contributes 0 */
        if(!ident[l]) memcpy(P1+o1.wo[l],P0+o0.wo[s],(size_t)D*D*4);

        for(int e=0;e<E;e++){
            const float*s1=P0+o0.w1[s]+(size_t)e*D*F0, *s3=P0+o0.w3[s]+(size_t)e*D*F0;
            const float*s2=P0+o0.w2[s]+(size_t)e*F0*D;
            float*d1=P1+o1.w1[l]+(size_t)e*D*F1, *d3=P1+o1.w3[l]+(size_t)e*D*F1;
            float*d2=P1+o1.w2[l]+(size_t)e*F1*D;
            for(int d=0;d<D;d++) for(int f=0;f<F1;f++){
                int j=f%F0; float g=1.0f;
                if(f>=F0 && noise>0) g=1.0f+noise*rnd_normal();   /* only the copies */
                d1[(size_t)d*F1+f]=s1[(size_t)d*F0+j]*g;
                d3[(size_t)d*F1+f]=s3[(size_t)d*F0+j];
            }
            /* w2: divide by the duplication count so the summed output matches */
            if(!ident[l]) for(int f=0;f<F1;f++){ int j=f%F0; float inv=1.0f/(float)cnt[j];
                for(int d=0;d<D;d++) d2[(size_t)f*D+d]=s2[(size_t)j*D+d]*inv; }
        }
    }
    free(cnt);

    model_save_bundle(out,&c1,P1,n1,NULL,0);
    /* model_save_bundle takes the blob; re-emit with the tokenizer attached */
    {   size_t tl_len; uint8_t*raw=read_file(inpath,&tl_len);
        Rd r={raw,raw+tl_len}; char m[4]; rd(&r,m,4);
        int ver,nh=hdr_ints(m,&ver); int hdr[256]; rd(&r,hdr,4*nh);
        r.p += (size_t)n0*4;                       /* skip the old params */
        int tl=0; if(ver>=2 && rd(&r,&tl,4) && tl>0)
            model_save_bundle(out,&c1,P1,n1,r.p,tl);
        free(raw);
    }

    grow_sidecars(inpath,out,src,ident,Lold,Lnew,D);
    int addF = F1-F0;
    printf("[grow] %s -> %s\n", inpath, out);
    printf("[grow] layers %d -> %d  (%d inserted, zero-init identity)\n", Lold,Lnew,add);
    if(addF) printf("[grow] expert ffn %d -> %d  (noise %.3g on duplicated units)\n",F0,F1,noise);
    printf("[grow] params %.2fM -> %.2fM  (%.2fx)\n", n0/1e6, n1/1e6, (double)n1/n0);
    { /* active = everything except the 7 experts a token never routes to */
      size_t a0=n0, a1=n1;
      if(c0.n_exp>1){ a0-=(size_t)Lold*(c0.n_exp-1)*(2*(size_t)D*F0+(size_t)F0*D);
                      a1-=(size_t)Lnew*(c0.n_exp-1)*(2*(size_t)D*F1+(size_t)F1*D); }
      printf("[grow] active %.2fM -> %.2fM  (%.2fx)\n", a0/1e6, a1/1e6, (double)a1/a0);
    }
    printf("[grow] VERIFY NOW -- layers-only growth must give an identical val loss:\n");
    printf("         cudalm eval %s <data> 20 1\n", in);
    printf("         cudalm eval %s <data> 20 1\n", out);
    if(addF && noise>0) printf("       (--ffn with noise>0 is only approximately preserving; a tiny\n"
                               "        val-loss change is expected, a large one is a bug)\n");
    free(P0); free(P1);
    return 0;
}

/* ===================== quantize ========================================= */
static int cmd_quantize(int argc,char**argv){
    if(argc<4){ fprintf(stderr,
        "usage: cudalm quantize <in.bin|name> <out.bin> [q8|q4|q4n|fp16]\n"
        "  q4n = exact export of a QLM 4-bit model (fp32 scale per 128, lossless; non-grid groups kept fp32).\n"
        "  Shrinks a checkpoint ON DISK (q8 ~3.8x, fp16 ~2x). Loading dequantizes\n"
        "  back to fp32, so run-time speed is unchanged and quality is whatever\n"
        "  survived the rounding. The embedded tokenizer is carried over, so the\n"
        "  quantized file still runs standalone.\n"
        "  Windowed (TLM5) models are written as TLQ2; full-attention ones as TLQ1,\n"
        "  which the CPU tinylm build can also read.\n"); return 1; }
    const char*in=argv[2],*out=argv[3];
    const char*qa=argc>4?argv[4]:"q8";
    int qtype = !strcmp(qa,"fp16") ? 0 : !strcmp(qa,"q4") ? 2 : !strcmp(qa,"q4n") ? 3 : 1;
    const int group = qtype==3 ? 128 : qtype==2 ? 32 : 64;
    char inpath[1024];
    if(strstr(in,".bin")) snprintf(inpath,sizeof(inpath),"%s",in);
    else model_path(in,inpath,sizeof(inpath));

    size_t flen; uint8_t*buf=read_file(inpath,&flen);
    if(!buf){ fprintf(stderr,"cannot open %s\n",inpath); return 1; }
    Rd r={buf,buf+flen}; char m[4]; rd(&r,m,4);
    if(!memcmp(m,"TLQ1",4)||!memcmp(m,"TLQ2",4)){
        fprintf(stderr,"%s is already quantized\n",inpath); free(buf); return 1; }
    int ver,nh=hdr_ints(m,&ver);
    if(!nh){ fprintf(stderr,"%s: not a tinylm/cudalm model\n",inpath); free(buf); return 1; }
    int hdr[256]; if(!rd(&r,hdr,4*nh)){ fprintf(stderr,"bad header\n"); free(buf); return 1; }
    Cfg c; cfg_from_hdr(&c,hdr,ver); loop_apply(&c);
    if(c.pool||c.mod||c.lp){ fprintf(stderr,"quantize: UniPool / per-layer-layout (TLM7) / looped models not supported yet\n"); free(buf); return 1; }
    size_t n=param_count(&c); float*P=(float*)malloc(n*4);
    if(!rd(&r,P,n*4)){ fprintf(stderr,"bad params\n"); free(buf); return 1; }
    int tl=0; const uint8_t*blob=NULL;
    if(ver>=2 && rd(&r,&tl,4) && tl>0) blob=r.p;      /* points into buf, still alive */

    size_t qlen; uint8_t*q=quantize_params(P,n,qtype,group,&qlen);
    /* Round-trip and report the ACTUAL error rather than asserting it is small. */
    float*P2=(float*)malloc(n*4); Rd rq={q,q+qlen};
    dequantize_params(&rq,P2,n,qtype,group);
    double maxe=0,sume=0,sumx=0;
    for(size_t i=0;i<n;i++){ double e=fabs((double)P[i]-(double)P2[i]);
        if(e>maxe)maxe=e; sume+=e; sumx+=fabs((double)P[i]); }

    int windowed = (c.win>0);
    FILE*f=fopen(out,"wb"); if(!f){ fprintf(stderr,"cannot write %s\n",out); return 1; }
    if(windowed){
        int oh[13]={c.V,c.D,c.L,c.H,c.KV,c.F,c.T,c.eot,c.tmpl,c.n_mtp,c.n_exp,c.win,c.full_every};
        fwrite("TLQ2",1,4,f); fwrite(oh,4,13,f);
    } else {
        int oh[11]={c.V,c.D,c.L,c.H,c.KV,c.F,c.T,c.eot,c.tmpl,c.n_mtp,c.n_exp};
        fwrite("TLQ1",1,4,f); fwrite(oh,4,11,f);
    }
    fwrite(&qtype,4,1,f); fwrite(&group,4,1,f);
    fwrite(q,1,qlen,f);
    fwrite(&tl,4,1,f); if(tl>0&&blob) fwrite(blob,1,tl,f);
    fclose(f);
    for(int k=0;k<3;k++){ const char*sx=k==2?".env":k?".opt.t1":".opt.ng"; char si[1100],so[1100]; size_t sl;   /* .env: the model config */
        snprintf(si,sizeof si,"%s%s",inpath,sx); snprintf(so,sizeof so,"%s%s",out,sx);
        uint8_t*sb=read_file(si,&sl); if(!sb) continue;
        FILE*g=fopen(so,"wb"); if(g){ fwrite(sb,1,sl,g); fclose(g); printf("[quantize] sidecar %s -> %s (%.1f MB)\n",si,so,sl/1e6); }
        free(sb); }

    double insz=(double)flen, outsz=4+4.0*(windowed?13:11)+8+(double)qlen+4+tl;
    printf("[quantize] %s -> %s  (%s, %zu params, %s)\n", inpath, out,
           qtype==3?"q4n":qtype==2?"q4":(qtype?"q8":"fp16"), n, windowed?"TLQ2 windowed":"TLQ1");
    printf("[quantize] %.2f MB -> %.2f MB  (%.2fx smaller) | max|err| %.2e  mean-rel %.2e | tokenizer %s\n",
           insz/1e6, outsz/1e6, insz/outsz, maxe, sumx>0?sume/sumx:0.0, tl>0?"kept":"NONE");
    if(windowed) printf("[quantize] note: TLQ2 carries win=%d full_every=%d; the CPU tinylm\n"
                        "           build does not know TLM5/TLQ2 and cannot read this file.\n",
                        c.win,c.full_every);
    free(P); free(P2); free(q); free(buf);
    return 0;
}

/* gguf: runs export/bin2gguf.py. q4n/q4/q8 go through this exe's `quantize` first, so the GGUF holds
 * exactly what quantize writes; the model's .env (applied in main) supplies CUDALM_TOPK. */
static int cmd_gguf(int argc,char**argv){
    if(argc<3){ fprintf(stderr,
        "usage: cudalm gguf <name|model.bin> [q4n|q4|q8|fp16|f32] [tinylm|llama] [out.gguf] [--ctx N] [--ng-f32]\n"
        "  tinylm (default): exact GGUF for the tinylm C runtime (MoE top-k, router bias, ngram/engram kept).\n"
        "  llama: stock llama.cpp GGUF, q8|fp16|f32 only; drops ngram/engram and routes top-1 (lossy).\n"
        "  type: same modes as quantize (q4n = lossless for QLM models), default q4n (tinylm) / q8 (llama).\n"
        "  Needs python + numpy (CUDALM_PYTHON overrides the interpreter).\n"); return 1; }
    char mp[1024],self[1024],cl[8192]; const char*pos[3]={"","",""}; int np=0,i=3;
    model_resolve(argv[2],mp,sizeof mp);
    for(;i<argc&&strncmp(argv[i],"--",2)&&np<3;i++) pos[np++]=argv[i];
#ifdef _WIN32
    GetModuleFileNameA(0,self,sizeof self);
#else
    ssize_t w=readlink("/proc/self/exe",self,sizeof self-1); self[w>0?w:0]=0;
#endif
    const char*tg=*pos[1]?pos[1]:"tinylm", *ty=*pos[0]?pos[0]:strcmp(tg,"llama")?"q4n":"q8";
    int n=snprintf(cl,sizeof cl,"%s -u \"%s" PSEP "export" PSEP "bin2gguf.py\" \"%s\"",py_cmd(),exe_dir(),mp);
    if(*pos[2]) n+=snprintf(cl+n,sizeof cl-n," \"%s\"",pos[2]);
    n+=snprintf(cl+n,sizeof cl-n," --type %s --target %s --cudalm \"%s\"",ty,tg,self);
    for(;i<argc;i++) n+=snprintf(cl+n,sizeof cl-n," \"%s\"",argv[i]);
    return run_cmd(cl);
}

/* ===================== selftest ========================================= */
/* Validates the row-major/column-major cuBLAS mapping and the hand-written
 * kernels against naive CPU references. This is the piece most likely to be
 * silently wrong, so it is checked first and explicitly. */
static float relerr(const float*a,const float*b,size_t n){
    double num=0,den=0; for(size_t i=0;i<n;i++){ double d=(double)a[i]-b[i]; num+=d*d; den+=(double)b[i]*b[i]; }
    return (float)sqrt(num/(den>0?den:1)); }
static int check(const char*name,float e,float tol){
    printf("  %-22s rel-err %.3e  %s\n",name,e,e<=tol?"OK":"FAIL"); return e<=tol; }

/* Verify that a checkpoint trained with CUDALM_QLM=1 really contains low-bit
 * weights, rather than a model that merely ran quantized kernels. Counts the
 * DISTINCT values in each scale group of every quantized tensor: a 4-bit model
 * must show at most 2*dmax+1 = 15, ternary at most 3. Embeddings and norms are
 * excluded because they are deliberately left in fp32.
 *
 * This is the GPU counterpart of the CPU trial's check_quant.py, and it is the
 * check that distinguishes "the forward pass saw 4-bit weights" from "we
 * believe the forward pass saw 4-bit weights". */
static int cmd_qlmcheck(int argc,char**argv){
    if(argc<3){ fprintf(stderr,
        "usage: cudalm qlmcheck <name|model.bin> [group]\n"
        "  reports max distinct values per group in wq/wk/wv/wo/w1/w3/w2\n"); return 1; }
    int group = argc>3 ? atoi(argv[3]) : 128; if(group<8) group=128;
    char path[1024]; model_path(argv[2],path,sizeof path);
    { FILE*e=fopen(path,"rb"); if(e) fclose(e); else snprintf(path,sizeof path,"%s",argv[2]); }
    Cfg c; Tok*qtk=NULL; float*P=model_load_full(path,&c,&qtk);
    if(!P){ fprintf(stderr,"cannot load %s\n",path); return 1; }
    if(c.lp){ fprintf(stderr,"looped models not supported here\n"); return 1; }
    Off o; calc_offsets(&c,&o); size_t np=param_count(&c);
    int E = c.n_exp>0 ? c.n_exp : 1;
    struct { const char*nm; size_t off, n; } sp[16*7]; int ns=0;
    for(int l=0;l<c.L;l++){
        sp[ns].nm="wq"; sp[ns].off=o.wq[l]; sp[ns].n=(size_t)c.D*c.D; ns++;
        sp[ns].nm="wk"; sp[ns].off=o.wk[l]; sp[ns].n=(size_t)c.D*c.KD; ns++;
        sp[ns].nm="wv"; sp[ns].off=o.wv[l]; sp[ns].n=(size_t)c.D*c.KD; ns++;
        sp[ns].nm="wo"; sp[ns].off=o.wo[l]; sp[ns].n=(size_t)c.D*c.D; ns++;
        sp[ns].nm="w1"; sp[ns].off=o.w1[l]; sp[ns].n=(size_t)E*c.D*c.F; ns++;
        sp[ns].nm="w3"; sp[ns].off=o.w3[l]; sp[ns].n=(size_t)E*c.D*c.F; ns++;
        sp[ns].nm="w2"; sp[ns].off=o.w2[l]; sp[ns].n=(size_t)E*c.F*c.D; ns++;
        if(ns>=16*7-7) break;
    }
    printf("[qlmcheck] %s  V=%d D=%d L=%d F=%d E=%d params=%zu  group=%d\n",
           path,c.V,c.D,c.L,c.F,E,np,group);
    const char*names[7]={"wq","wk","wv","wo","w1","w3","w2"};
    int best[7]={0,0,0,0,0,0,0}; int worst=0;
    float*lv=(float*)malloc(sizeof(float)*group);
    for(int i=0;i<ns;i++){
        int slot=i%7;
        for(size_t g0=0; g0<sp[i].n; g0+=group){
            size_t n = (sp[i].n-g0 < (size_t)group) ? sp[i].n-g0 : (size_t)group;
            int nl=0;
            for(size_t j=0;j<n;j++){
                float v=P[sp[i].off+g0+j]; int seen=0;
                for(int q=0;q<nl;q++) if(lv[q]==v){ seen=1; break; }
                if(!seen) lv[nl++]=v;
            }
            if(nl>best[slot]) best[slot]=nl;
            if(nl>worst) worst=nl;
        }
    }
    free(lv);
    for(int k=0;k<7;k++)
        printf("  %-3s max distinct values per group of %d: %d\n",names[k],group,best[k]);
    { int nl=0; float*e=(float*)malloc(sizeof(float)*group);
      for(int j=0;j<group && j<(int)((size_t)c.V*c.D);j++){
        int seen=0; for(int q=0;q<nl;q++) if(e[q]==P[o.emb+j]){seen=1;break;}
        if(!seen) e[nl++]=P[o.emb+j]; }
      printf("  emb (NOT quantized, expect ~%d): %d\n",group,nl); free(e); }
    int bits=0; while((1<<bits)<worst) bits++;
    printf("[qlmcheck] worst case %d levels = %d bits%s\n", worst, bits,
           worst<=15 ? "  -> low-bit weights CONFIRMED" : "  -> NOT a low-bit checkpoint");
    free(P);
    return worst<=15?0:1;
}

static void cmd_selftest(void){
    Cfg c; preset("m1",&c); gC=c; cuda_init();
    setvbuf(stdout,NULL,_IONBF,0);     /* a crash must not swallow the last checks */
    int ok=1;
    /* ---- GEMM mappings ---- */
    { int M=67,K=53,N=91;
      float*A=(float*)malloc((size_t)M*K*4),*Bm=(float*)malloc((size_t)K*N*4),*Bt=(float*)malloc((size_t)N*K*4);
      float*Cr=(float*)malloc((size_t)M*N*4),*Cg=(float*)malloc((size_t)M*N*4);
      for(int i=0;i<M*K;i++)A[i]=rnd_normal(); for(int i=0;i<K*N;i++)Bm[i]=rnd_normal();
      for(int i=0;i<N*K;i++)Bt[i]=rnd_normal();
      float*dA=(float*)dmalloc((size_t)M*K*4),*dB=(float*)dmalloc((size_t)K*N*4),*dBt=(float*)dmalloc((size_t)N*K*4);
      float*dC=(float*)dmalloc((size_t)M*N*4);
      cudaMemcpy(dA,A,(size_t)M*K*4,cudaMemcpyHostToDevice);
      cudaMemcpy(dB,Bm,(size_t)K*N*4,cudaMemcpyHostToDevice);
      cudaMemcpy(dBt,Bt,(size_t)N*K*4,cudaMemcpyHostToDevice);
      /* ab */
      for(int i=0;i<M;i++)for(int j=0;j<N;j++){ double s=0; for(int k=0;k<K;k++)s+=A[i*K+k]*Bm[k*N+j]; Cr[i*N+j]=(float)s; }
      g_ab(dA,dB,dC,M,K,N); cudaMemcpy(Cg,dC,(size_t)M*N*4,cudaMemcpyDeviceToHost);
      ok &= check("gemm A@B",relerr(Cg,Cr,(size_t)M*N),1e-3f);
      /* abt */
      for(int i=0;i<M;i++)for(int j=0;j<N;j++){ double s=0; for(int k=0;k<K;k++)s+=A[i*K+k]*Bt[j*K+k]; Cr[i*N+j]=(float)s; }
      g_abt(dA,dBt,dC,M,K,N); cudaMemcpy(Cg,dC,(size_t)M*N*4,cudaMemcpyDeviceToHost);
      ok &= check("gemm A@B^T",relerr(Cg,Cr,(size_t)M*N),1e-3f);
      /* atb: C[K,N] += A[M,K]^T @ B[M,N] */
      { int MM=M; float*Bmn=(float*)malloc((size_t)MM*N*4); for(int i=0;i<MM*N;i++)Bmn[i]=rnd_normal();
        float*dBmn=(float*)dmalloc((size_t)MM*N*4); cudaMemcpy(dBmn,Bmn,(size_t)MM*N*4,cudaMemcpyHostToDevice);
        float*Ck=(float*)malloc((size_t)K*N*4),*Cgk=(float*)malloc((size_t)K*N*4);
        for(int i=0;i<K*N;i++)Ck[i]=0.f;
        float*dCk=dzero((size_t)K*N);
        for(int k=0;k<K;k++)for(int j=0;j<N;j++){ double s=0; for(int m=0;m<MM;m++)s+=A[m*K+k]*Bmn[m*N+j]; Ck[k*N+j]=(float)s; }
        g_atb(dA,dBmn,dCk,MM,K,N); cudaMemcpy(Cgk,dCk,(size_t)K*N*4,cudaMemcpyDeviceToHost);
        ok &= check("gemm A^T@B (accum)",relerr(Cgk,Ck,(size_t)K*N),1e-3f); }
    }
    /* ---- FP8 E4M3 isolation: aligned dims (all divisible by 16) so fp8_ok fires.
     * Reports rel-err AND norm ratio ||out||/||ref|| vs a CPU double reference.
     * A mis-scaled dequant -> ratio != 1; pure E4M3 rounding -> ratio~1, err ~few%. */
#ifndef CUDALM_CPU
    { int M=64,K=384,N=512;
      int sf8=g_fp8gemm, sf16=g_fp16gemm;
      if(!LT){ ckb((cublasStatus_t)cublasLtCreate(&LT),"cublasLtCreate");
               g_fp8wscap=32ull<<20; ck(cudaMalloc(&g_fp8ws,g_fp8wscap),"fp8 ws");
               ck(cudaMalloc((void**)&g_fp8sc,6*sizeof(float)),"fp8 scale buf"); }
      float*A=(float*)malloc((size_t)M*K*4),*Bm=(float*)malloc((size_t)K*N*4),*Bt=(float*)malloc((size_t)N*K*4);
      float*Cr=(float*)malloc((size_t)M*N*4),*Cg=(float*)malloc((size_t)M*N*4);
      for(int i=0;i<M*K;i++)A[i]=rnd_normal(); for(int i=0;i<K*N;i++)Bm[i]=rnd_normal();
      for(int i=0;i<N*K;i++)Bt[i]=rnd_normal();
      float*dA=(float*)dmalloc((size_t)M*K*4),*dB=(float*)dmalloc((size_t)K*N*4),*dBt=(float*)dmalloc((size_t)N*K*4);
      float*dC=(float*)dmalloc((size_t)M*N*4);
      cudaMemcpy(dA,A,(size_t)M*K*4,cudaMemcpyHostToDevice);
      cudaMemcpy(dB,Bm,(size_t)K*N*4,cudaMemcpyHostToDevice);
      cudaMemcpy(dBt,Bt,(size_t)N*K*4,cudaMemcpyHostToDevice);
      #define NRAT(g,r,n) do{ double gg=0,rr=0; for(size_t _i=0;_i<(size_t)(n);_i++){gg+=(double)(g)[_i]*(g)[_i]; rr+=(double)(r)[_i]*(r)[_i];} \
          printf("  %-24s rel-err %.3e  ratio %.4f\n",(g==Cg?curlbl:curlbl),relerr((g),(r),(size_t)(n)), rr>0?sqrt(gg/rr):0.0); }while(0)
      const char*curlbl;
      /* g_ab  C[M,N]=A@B */
      for(int i=0;i<M;i++)for(int j=0;j<N;j++){ double s=0; for(int k=0;k<K;k++)s+=A[i*K+k]*Bm[k*N+j]; Cr[i*N+j]=(float)s; }
      g_fp8gemm=1; g_fp16gemm=1; g_ab(dA,dB,dC,M,K,N); cudaMemcpy(Cg,dC,(size_t)M*N*4,cudaMemcpyDeviceToHost);
      curlbl="FP8 g_ab (fwd NN)"; NRAT(Cg,Cr,(size_t)M*N);
      /* g_abt C[M,N]=A@B^T */
      for(int i=0;i<M;i++)for(int j=0;j<N;j++){ double s=0; for(int k=0;k<K;k++)s+=A[i*K+k]*Bt[j*K+k]; Cr[i*N+j]=(float)s; }
      g_abt(dA,dBt,dC,M,K,N); cudaMemcpy(Cg,dC,(size_t)M*N*4,cudaMemcpyDeviceToHost);
      curlbl="FP8 g_abt (dgrad TN)"; NRAT(Cg,Cr,(size_t)M*N);
      /* g_atb C[K,N]+=A^T@B, B is [M,N] */
      { float*Bmn=(float*)malloc((size_t)M*N*4); for(int i=0;i<M*N;i++)Bmn[i]=rnd_normal();
        float*dBmn=(float*)dmalloc((size_t)M*N*4); cudaMemcpy(dBmn,Bmn,(size_t)M*N*4,cudaMemcpyHostToDevice);
        float*Ck=(float*)malloc((size_t)K*N*4),*Cgk=(float*)malloc((size_t)K*N*4);
        float*dCk=dzero((size_t)K*N);
        for(int k=0;k<K;k++)for(int j=0;j<N;j++){ double s=0; for(int m=0;m<M;m++)s+=A[m*K+k]*Bmn[m*N+j]; Ck[k*N+j]=(float)s; }
        g_atb(dA,dBmn,dCk,M,K,N); cudaMemcpy(Cgk,dCk,(size_t)K*N*4,cudaMemcpyDeviceToHost);
        curlbl="FP8 g_atb (wgrad NT)"; NRAT(Cgk,Ck,(size_t)K*N);
        free(Bmn);cudaFree(dBmn);free(Ck);free(Cgk);cudaFree(dCk); }
      /* S6: with an unchanged amax the delayed path must reproduce the exact one */
      { float*C2=(float*)malloc((size_t)M*N*4); int on0=g_ds_on; g_ds_on=1;
        for(int r=0;r<3;r++){ g_ds_step++; g_ab(dA,dB,dC,M,K,N); }
        cudaMemcpy(C2,dC,(size_t)M*N*4,cudaMemcpyDeviceToHost);
        g_ds_on=0; g_ab(dA,dB,dC,M,K,N); cudaMemcpy(Cg,dC,(size_t)M*N*4,cudaMemcpyDeviceToHost);
        ok &= check("FP8 delayed scale == exact",relerr(C2,Cg,(size_t)M*N),1e-6f);
        g_ds_on=on0; free(C2); }
      #undef NRAT
      g_fp8gemm=sf8; g_fp16gemm=sf16;
      free(A);free(Bm);free(Bt);free(Cr);free(Cg);
      cudaFree(dA);cudaFree(dB);cudaFree(dBt);cudaFree(dC);
    }
#endif
    /* ---- rmsnorm ---- */
    { int N=37,D=128; float*X=(float*)malloc((size_t)N*D*4),*g=(float*)malloc((size_t)D*4);
      float*R=(float*)malloc((size_t)N*D*4),*O=(float*)malloc((size_t)N*D*4);
      for(int i=0;i<N*D;i++)X[i]=rnd_normal(); for(int i=0;i<D;i++)g[i]=1.0f+0.1f*rnd_normal();
      for(int n=0;n<N;n++){ double ms=0; for(int i=0;i<D;i++)ms+=(double)X[n*D+i]*X[n*D+i];
        float s=1.0f/sqrtf((float)(ms/D)+1e-5f); for(int i=0;i<D;i++)R[n*D+i]=X[n*D+i]*s*g[i]; }
      float*dX=(float*)dmalloc((size_t)N*D*4),*dG=(float*)dmalloc((size_t)D*4),*dO=(float*)dmalloc((size_t)N*D*4),*dr=(float*)dmalloc((size_t)N*4);
      cudaMemcpy(dX,X,(size_t)N*D*4,cudaMemcpyHostToDevice); cudaMemcpy(dG,g,(size_t)D*4,cudaMemcpyHostToDevice);
      KL(k_rmsnorm,N,TPB)(dX,dG,0,dO,dr,N,D); cudaMemcpy(O,dO,(size_t)N*D*4,cudaMemcpyDeviceToHost);
      ok &= check("rmsnorm",relerr(O,R,(size_t)N*D),1e-6f); }
    /* ---- warp rmsnorm fwd/bwd vs block kernels; h8 softmax_ce vs fp32 one ---- */
    { int N=37,D=384; size_t nd=(size_t)N*D; float*X=(float*)malloc(nd*4),*g=(float*)malloc(D*4),*Y=(float*)malloc(nd*4),*O1=(float*)malloc(nd*4),*O2=(float*)malloc(nd*4);
      for(size_t i=0;i<nd;i++){X[i]=rnd_normal();Y[i]=rnd_normal();} for(int i=0;i<D;i++)g[i]=1.0f+0.1f*rnd_normal();
      float*dX=(float*)dmalloc(nd*4),*dG=(float*)dmalloc(D*4),*dY=(float*)dmalloc(nd*4),*dO=(float*)dmalloc(nd*4),*dr=(float*)dmalloc(N*4),*dr2=(float*)dmalloc(N*4);
      cudaMemcpy(dX,X,nd*4,cudaMemcpyHostToDevice); cudaMemcpy(dG,g,D*4,cudaMemcpyHostToDevice); cudaMemcpy(dY,Y,nd*4,cudaMemcpyHostToDevice);
      KL(k_rmsnorm,N,TPB)(dX,dG,0,dO,dr,N,D); cudaMemcpy(O1,dO,nd*4,cudaMemcpyDeviceToHost);
      KL(k_rmsnorm_w,nblk(N,TPB/32),TPB)(dX,dG,0,dO,dr2,N,D); cudaMemcpy(O2,dO,nd*4,cudaMemcpyDeviceToHost);
      ok &= check("rmsnorm warp",relerr(O2,O1,nd),1e-6f);
      cudaMemcpy(dO,Y,nd*4,cudaMemcpyHostToDevice); KL(k_rmsnorm_bwd,N,TPB)(dX,dG,0,dr,dY,dO,N,D); cudaMemcpy(O1,dO,nd*4,cudaMemcpyDeviceToHost);
      cudaMemcpy(dO,Y,nd*4,cudaMemcpyHostToDevice); KL(k_rmsnorm_bwd_w,nblk(N,TPB/32),TPB)(dX,dG,0,dr,dY,dO,N,D); cudaMemcpy(O2,dO,nd*4,cudaMemcpyDeviceToHost);
      ok &= check("rmsnorm bwd warp",relerr(O2,O1,nd),1e-6f);
      free(X);free(g);free(Y);free(O1);free(O2); cudaFree(dX);cudaFree(dG);cudaFree(dY);cudaFree(dO);cudaFree(dr);cudaFree(dr2); }
#ifndef CUDALM_CPU
    { int R=5,V=4096,nA=R; size_t nv=(size_t)R*V; int tg[5]={7,-1,4095,0,1234};
      float*L=(float*)malloc(nv*4),*D1=(float*)malloc(nv*4),*D2=(float*)malloc(nv*4),l1[5],l2[5]; bf16*Db=(bf16*)malloc(nv*2);
      for(size_t i=0;i<nv;i++) L[i]=__half2float(__float2half(3.0f*rnd_normal()));
      float*dL=(float*)dmalloc(nv*4),*dD=(float*)dmalloc(nv*4),*dl=(float*)dmalloc(R*4); int*dt=(int*)dmalloc(R*4);
      __half*dh=(__half*)dmalloc(nv*2); __nv_fp8_e4m3*q=(__nv_fp8_e4m3*)dmalloc(nv);
      cudaMemcpy(dL,L,nv*4,cudaMemcpyHostToDevice); cudaMemcpy(dt,tg,R*4,cudaMemcpyHostToDevice); KL(k_f2h,nblk(nv,TPB),TPB)(dL,dh,nv,1.0f);
      KL(k_softmax_ce,R,TPB)(dL,dt,dl,dD,q,R,V,nA,(const float*)NULL,0.f); cudaMemcpy(D1,dD,nv*4,cudaMemcpyDeviceToHost); cudaMemcpy(l1,dl,R*4,cudaMemcpyDeviceToHost);
      KL(k_sce_h8,R,TPB)(dh,dt,dl,(bf16*)dD,q,V,nA,(const float*)NULL,0.f); cudaMemcpy(Db,dD,nv*2,cudaMemcpyDeviceToHost); cudaMemcpy(l2,dl,R*4,cudaMemcpyDeviceToHost);
      for(size_t i=0;i<nv;i++) D2[i]=h_bf2f(Db[i]);
      ok &= check("softmax_ce h8/bf16 dl",relerr(D2,D1,nv),5e-3f);
      ok &= check("softmax_ce h8 loss",relerr(l2,l1,R),1e-4f);
      free(L);free(D1);free(D2);free(Db); cudaFree(dL);cudaFree(dD);cudaFree(dl);cudaFree(dt);cudaFree(dh);cudaFree(q); }
#endif
    /* ---- swiglu ---- */
    { int n=5000; float*A=(float*)malloc((size_t)n*4),*Bv=(float*)malloc((size_t)n*4),*R=(float*)malloc((size_t)n*4),*O=(float*)malloc((size_t)n*4);
      for(int i=0;i<n;i++){A[i]=rnd_normal();Bv[i]=rnd_normal();}
      for(int i=0;i<n;i++){ float z=A[i]; R[i]=(z/(1.0f+expf(-z)))*Bv[i]; }
      float*dA=(float*)dmalloc((size_t)n*4),*dB=(float*)dmalloc((size_t)n*4),*dO=(float*)dmalloc((size_t)n*4);
      cudaMemcpy(dA,A,(size_t)n*4,cudaMemcpyHostToDevice); cudaMemcpy(dB,Bv,(size_t)n*4,cudaMemcpyHostToDevice);
      KL(k_swiglu,nblk(n,TPB),TPB)(dA,dB,dO,n); cudaMemcpy(O,dO,(size_t)n*4,cudaMemcpyDeviceToHost);
      ok &= check("swiglu",relerr(O,R,n),1e-6f); }
    /* ---- causal GQA attention, run twice: full then sliding-window ----------
     * WIN=6 over T=16 means most queries drop keys off the back, so the window
     * bounds in all four kernels are genuinely exercised (and CH=4 forces the
     * chunked accumulation rather than one full-range call). */
    for(int wi=0; wi<2; wi++){
      const int WIN = wi ? 6 : 0;
      const char *tag = wi ? "sliding W=6" : "full";
      int B=2,T=16,H=4,KV=2,hd=8,D=H*hd,KD=KV*hd,N=B*T;
      float*Q=(float*)malloc((size_t)N*D*4),*K=(float*)malloc((size_t)N*KD*4),*V=(float*)malloc((size_t)N*KD*4);
      float*R=(float*)malloc((size_t)N*D*4),*O=(float*)malloc((size_t)N*D*4);
      for(int i=0;i<N*D;i++)Q[i]=rnd_normal(); for(int i=0;i<N*KD;i++){K[i]=rnd_normal();V[i]=rnd_normal();}
      float sc=1.0f/sqrtf((float)hd);
      #define SLO(t) ((WIN>0 && (t)>=WIN) ? (t)-WIN+1 : 0)
      for(int b=0;b<B;b++)for(int h=0;h<H;h++){ int kvh=h/(H/KV);
        for(int t=0;t<T;t++){ float*pr=(float*)malloc((size_t)(t+1)*4); float mx=-1e30f; int lo=SLO(t);
          for(int s=lo;s<=t;s++){ double d=0; for(int i=0;i<hd;i++)d+=Q[(size_t)(b*T+t)*D+h*hd+i]*K[(size_t)(b*T+s)*KD+kvh*hd+i];
            pr[s]=(float)d*sc; if(pr[s]>mx)mx=pr[s]; }
          float sm=0; for(int s=lo;s<=t;s++){pr[s]=expf(pr[s]-mx);sm+=pr[s];}
          for(int i=0;i<hd;i++){ double acc=0; for(int s=lo;s<=t;s++)acc+=pr[s]*V[(size_t)(b*T+s)*KD+kvh*hd+i];
            R[(size_t)(b*T+t)*D+h*hd+i]=(float)(acc/sm); } free(pr); } }
      float*dQ=(float*)dmalloc((size_t)N*D*4),*dK=(float*)dmalloc((size_t)N*KD*4),*dV=(float*)dmalloc((size_t)N*KD*4);
      float*dO=(float*)dmalloc((size_t)N*D*4),*dl=(float*)dmalloc((size_t)B*H*T*4);
      cudaMemcpy(dQ,Q,(size_t)N*D*4,cudaMemcpyHostToDevice); cudaMemcpy(dK,K,(size_t)N*KD*4,cudaMemcpyHostToDevice);
      cudaMemcpy(dV,V,(size_t)N*KD*4,cudaMemcpyHostToDevice);
      KL(k_attn_fwd,B*H*T,WARP)(dQ,dK,dV,dO,dl,WIN,B,T,H,KV,hd,D,KD);
      cudaMemcpy(O,dO,(size_t)N*D*4,cudaMemcpyDeviceToHost);
      char lbl[64];
      snprintf(lbl,sizeof(lbl),"attn fwd (%s)",tag);
      ok &= check(lbl,relerr(O,R,(size_t)N*D),1e-6f);
      /* tiled forward must match the same CPU reference */
      { size_t shm=(size_t)2*KTILE*hd*sizeof(float);
        cudaMemset(dO,0,(size_t)N*D*4);
        KL(k_attn_fwd_tiled,(B*H*T)/QPB,QPB*WARP,shm)(dQ,dK,dV,dO,dl,WIN,B,T,H,KV,hd,D,KD);
        cudaMemcpy(O,dO,(size_t)N*D*4,cudaMemcpyDeviceToHost);
        snprintf(lbl,sizeof(lbl),"attn fwd TILED (%s)",tag);
        ok &= check(lbl,relerr(O,R,(size_t)N*D),1e-6f); }
      /* sub-warp mapping must match the same CPU reference (hd=8 here, SUBL=8
       * so EPL=1 -- the degenerate one-dim-per-lane case, still a valid check) */
      if(hd%SUBL==0){ cudaMemset(dO,0,(size_t)N*D*4);
        size_t thr=(size_t)B*H*T*SUBL;
        KL(k_attn_fwd_sub,nblk(thr,TPB),TPB)(dQ,dK,dV,dO,dl,WIN,B,T,H,KV,hd,D,KD);
        cudaMemcpy(O,dO,(size_t)N*D*4,cudaMemcpyDeviceToHost);
        snprintf(lbl,sizeof(lbl),"attn fwd SUBWARP (%s)",tag);
        ok &= check(lbl,relerr(O,R,(size_t)N*D),1e-6f); }

      float*dAh=(float*)malloc((size_t)N*D*4);
      for(int i=0;i<N*D;i++)dAh[i]=rnd_normal();
      float *rQ=(float*)calloc((size_t)N*D,4),*rK=(float*)calloc((size_t)N*KD,4),*rV=(float*)calloc((size_t)N*KD,4);
      for(int b=0;b<B;b++)for(int h=0;h<H;h++){ int kvh=h/(H/KV);
        for(int t=0;t<T;t++){ float*pr=(float*)malloc((size_t)(t+1)*4),*dpr=(float*)malloc((size_t)(t+1)*4);
          float mx=-1e30f; int lo=SLO(t);
          for(int s=lo;s<=t;s++){ double d=0; for(int i=0;i<hd;i++)d+=Q[(size_t)(b*T+t)*D+h*hd+i]*K[(size_t)(b*T+s)*KD+kvh*hd+i];
            pr[s]=(float)d*sc; if(pr[s]>mx)mx=pr[s]; }
          float sm=0; for(int s=lo;s<=t;s++){pr[s]=expf(pr[s]-mx);sm+=pr[s];}
          for(int s=lo;s<=t;s++)pr[s]/=sm;
          for(int s=lo;s<=t;s++){ double d=0; for(int i=0;i<hd;i++)d+=dAh[(size_t)(b*T+t)*D+h*hd+i]*V[(size_t)(b*T+s)*KD+kvh*hd+i];
            dpr[s]=(float)d;
            for(int i=0;i<hd;i++) rV[(size_t)(b*T+s)*KD+kvh*hd+i]+=pr[s]*dAh[(size_t)(b*T+t)*D+h*hd+i]; }
          double dotpd=0; for(int s=lo;s<=t;s++)dotpd+=pr[s]*dpr[s];
          for(int s=lo;s<=t;s++){ float ds=(float)(pr[s]*(dpr[s]-dotpd))*sc;
            for(int i=0;i<hd;i++){ rQ[(size_t)(b*T+t)*D+h*hd+i]+=ds*K[(size_t)(b*T+s)*KD+kvh*hd+i];
                                   rK[(size_t)(b*T+s)*KD+kvh*hd+i]+=ds*Q[(size_t)(b*T+t)*D+h*hd+i]; } }
          free(pr); free(dpr); } }
      #undef SLO
      float*ddA=(float*)dmalloc((size_t)N*D*4); cudaMemcpy(ddA,dAh,(size_t)N*D*4,cudaMemcpyHostToDevice);
      float*gQ=dzero((size_t)N*D),*gK=dzero((size_t)N*KD),*gV=dzero((size_t)N*KD),*gR=dzero((size_t)B*H*T);
      const int CH=4, RG=(WIN>0?WIN:T);   /* offsets when windowed, else abs time */
      /* The fused Drow is what production runs; k_attn_bwd_dr is the independent
       * window-walk it was derived from. Checking them against each other is the
       * whole proof that dA.O == sum_s a_s (dA.v_s). */
      { float*gRw=dzero((size_t)B*H*T);
        for(int s0=0;s0<RG;s0+=CH){ int s1=(s0+CH<RG)?s0+CH:RG;
          KL(k_attn_bwd_dr,B*H*T,WARP)(dQ,dK,dV,ddA,dl,gRw,s0,s1,WIN,B,T,H,KV,hd,D,KD); }
        KL(k_attn_drow,B*H*T,WARP)(ddA,dO,gR,B,T,H,hd,D,1.0f);
        float*hw=(float*)malloc((size_t)B*H*T*4),*hf=(float*)malloc((size_t)B*H*T*4);
        cudaMemcpy(hw,gRw,(size_t)B*H*T*4,cudaMemcpyDeviceToHost);
        cudaMemcpy(hf,gR ,(size_t)B*H*T*4,cudaMemcpyDeviceToHost);
        snprintf(lbl,sizeof(lbl),"attn Drow fused (%s)",tag);
        ok &= check(lbl,relerr(hf,hw,(size_t)B*H*T),1e-5f);
        free(hw); free(hf); cudaFree(gRw); }
      for(int s0=0;s0<RG;s0+=CH){ int s1=(s0+CH<RG)?s0+CH:RG;
        KL(k_attn_bwd_dq,B*H*T,WARP)(dQ,dK,dV,ddA,dl,gQ,gR,s0,s1,WIN,B,T,H,KV,hd,D,KD); }
      for(int t0=0;t0<RG;t0+=CH){ int t1=(t0+CH<RG)?t0+CH:RG;
        KL(k_attn_bwd_dkv,B*KV*T,WARP)(dQ,dK,dV,ddA,dl,gR,gK,gV,t0,t1,WIN,B,T,H,KV,hd,D,KD); }
      float*oQ=(float*)malloc((size_t)N*D*4),*oK=(float*)malloc((size_t)N*KD*4),*oV=(float*)malloc((size_t)N*KD*4);
      cudaMemcpy(oQ,gQ,(size_t)N*D*4,cudaMemcpyDeviceToHost);
      cudaMemcpy(oK,gK,(size_t)N*KD*4,cudaMemcpyDeviceToHost);
      cudaMemcpy(oV,gV,(size_t)N*KD*4,cudaMemcpyDeviceToHost);
      snprintf(lbl,sizeof(lbl),"attn bwd dQ (%s)",tag); ok &= check(lbl,relerr(oQ,rQ,(size_t)N*D),1e-5f);
      snprintf(lbl,sizeof(lbl),"attn bwd dK (%s)",tag); ok &= check(lbl,relerr(oK,rK,(size_t)N*KD),1e-5f);
      snprintf(lbl,sizeof(lbl),"attn bwd dV (%s)",tag); ok &= check(lbl,relerr(oV,rV,(size_t)N*KD),1e-5f);
      /* ---- fp16 operands, same CPU reference, looser tolerance --------------
       * Half carries ~3 decimal digits, so rounding the inputs costs ~1e-3
       * relative on a dot product. That is the operands, not drift: every
       * accumulator, the softmax and the lse in these kernels are still fp32.
       * dQ/dK tolerate more because ds = p*(dp-dr) cancels. */
#ifndef CUDALM_CPU
      { __half *hQ=(__half*)dmalloc((size_t)N*D*2),*hK=(__half*)dmalloc((size_t)N*KD*2),
               *hV=(__half*)dmalloc((size_t)N*KD*2),*hA=(__half*)dmalloc((size_t)N*D*2);
        e_f2h(dQ ,hQ,(size_t)N*D ,1.0f);  e_f2h(dK,hK,(size_t)N*KD,1.0f);
        e_f2h(dV ,hV,(size_t)N*KD,1.0f);  e_f2h(ddA,hA,(size_t)N*D,DSCALE);
        float*fO=dzero((size_t)N*D),*fl=dzero((size_t)B*H*T);
        KL(k_attn_fwd_h,B*H*T,WARP)(hQ,hK,hV,fO,fl,WIN,B,T,H,KV,hd,D,KD);
        cudaMemcpy(O,fO,(size_t)N*D*4,cudaMemcpyDeviceToHost);
        /* measured 2.6e-4 - 5.5e-4 across both window modes; tolerances sit ~3x
         * above that, since selftest inputs are re-randomised every run */
        snprintf(lbl,sizeof(lbl),"attn fwd FP16 (%s)",tag);
        ok &= check(lbl,relerr(O,R,(size_t)N*D),1e-3f);
        /* lane-per-query tile: different thread map, same operands, same answer.
         * lse is checked too -- the backward reads it, so a wrong one is silent. */
        { float*lO=dzero((size_t)N*D),*ll=dzero((size_t)B*H*T);
          if(attn_fwd_lane(hQ,hK,hV,lO,ll,WIN,B,T,H,KV,hd,D,KD)){
            cudaMemcpy(O,lO,(size_t)N*D*4,cudaMemcpyDeviceToHost);
            snprintf(lbl,sizeof(lbl),"attn fwd LANE (%s)",tag);
            ok &= check(lbl,relerr(O,R,(size_t)N*D),1e-3f);
            float*hl=(float*)malloc((size_t)B*H*T*4),*hr=(float*)malloc((size_t)B*H*T*4);
            cudaMemcpy(hl,ll,(size_t)B*H*T*4,cudaMemcpyDeviceToHost);
            cudaMemcpy(hr,fl,(size_t)B*H*T*4,cudaMemcpyDeviceToHost);
            snprintf(lbl,sizeof(lbl),"attn lse LANE (%s)",tag);
            ok &= check(lbl,relerr(hl,hr,(size_t)B*H*T),1e-3f);
            free(hl); free(hr);
          } else printf("  attn fwd LANE (%s): no instantiation for hd=%d\n",tag,hd);
          cudaFree(lO); cudaFree(ll); }
        float*fQ=dzero((size_t)N*D),*fK=dzero((size_t)N*KD),*fV=dzero((size_t)N*KD),*fR=dzero((size_t)B*H*T);
        KL(k_attn_drow,B*H*T,WARP)(ddA,fO,fR,B,T,H,hd,D,DSCALE);
        for(int s0=0;s0<RG;s0+=CH){ int s1=(s0+CH<RG)?s0+CH:RG;
          KL(k_attn_bwd_dq_h,B*H*T,WARP)(hQ,hK,hV,hA,fl,fQ,fR,s0,s1,WIN,B,T,H,KV,hd,D,KD); }
        for(int t0=0;t0<RG;t0+=CH){ int t1=(t0+CH<RG)?t0+CH:RG;
          KL(k_attn_bwd_dkv_h,B*KV*T,WARP)(hQ,hK,hV,hA,fl,fR,fK,fV,t0,t1,WIN,B,T,H,KV,hd,D,KD); }
        cudaMemcpy(oQ,fQ,(size_t)N*D*4,cudaMemcpyDeviceToHost);
        cudaMemcpy(oK,fK,(size_t)N*KD*4,cudaMemcpyDeviceToHost);
        cudaMemcpy(oV,fV,(size_t)N*KD*4,cudaMemcpyDeviceToHost);
        /* lane dQ: single launch over the whole window, vs the chunked warp path */
        /* both dot forms: fp32-accumulate (production default) and packed
         * __hfma2. The packed error is the price of Pascal's 2x fp16 rate --
         * printed so the trade is visible, with a looser bound. */
        for(int pk=0;pk<2;pk++){ float*lQ=dzero((size_t)N*D);
          if(attn_bwd_dq_lane(hQ,hK,hV,hA,fl,lQ,fR,WIN,B,T,H,KV,hd,D,KD,pk)){
            float*oL=(float*)malloc((size_t)N*D*4);
            cudaMemcpy(oL,lQ,(size_t)N*D*4,cudaMemcpyDeviceToHost);
            snprintf(lbl,sizeof(lbl),"attn bwd dQ LANE%s (%s)",pk?" HFMA2":"",tag);
            ok &= check(lbl,relerr(oL,rQ,(size_t)N*D),pk?2e-2f:2e-3f);
            free(oL);
          } else printf("  attn bwd dQ LANE (%s): no instantiation for hd=%d\n",tag,hd);
          cudaFree(lQ); }
        /* every (packed, split) form is checked against the same CPU reference --
         * SPL only changes which thread owns which dim, so all of them must land
         * on the SAME answer to the SAME tolerance */
        { static const int SPLS[3]={1,2,4};
        for(int si=0;si<3;si++) for(int pk=0;pk<2;pk++){
          int spl=SPLS[si]; float*lK=dzero((size_t)N*KD),*lV=dzero((size_t)N*KD);
          if(attn_bwd_dkv_lane(hQ,hK,hV,hA,fl,fR,lK,lV,WIN,B,T,H,KV,hd,D,KD,pk,spl)){
            float*oL=(float*)malloc((size_t)N*KD*4);
            cudaMemcpy(oL,lK,(size_t)N*KD*4,cudaMemcpyDeviceToHost);
            snprintf(lbl,sizeof(lbl),"attn bwd dK LANE/%d%s (%s)",spl,pk?" HFMA2":"",tag);
            ok &= check(lbl,relerr(oL,rK,(size_t)N*KD),pk?2e-2f:2e-3f);
            cudaMemcpy(oL,lV,(size_t)N*KD*4,cudaMemcpyDeviceToHost);
            snprintf(lbl,sizeof(lbl),"attn bwd dV LANE/%d%s (%s)",spl,pk?" HFMA2":"",tag);
            ok &= check(lbl,relerr(oL,rV,(size_t)N*KD),pk?1e-2f:1e-3f);
            free(oL);
          } else printf("  attn bwd dKV LANE/%d (%s): no instantiation for hd=%d\n",spl,tag,hd);
          cudaFree(lK); cudaFree(lV); } }
        snprintf(lbl,sizeof(lbl),"attn bwd dQ FP16 (%s)",tag); ok &= check(lbl,relerr(oQ,rQ,(size_t)N*D),2e-3f);
        snprintf(lbl,sizeof(lbl),"attn bwd dK FP16 (%s)",tag); ok &= check(lbl,relerr(oK,rK,(size_t)N*KD),2e-3f);
        snprintf(lbl,sizeof(lbl),"attn bwd dV FP16 (%s)",tag); ok &= check(lbl,relerr(oV,rV,(size_t)N*KD),1e-3f);
        cudaFree(hQ);cudaFree(hK);cudaFree(hV);cudaFree(hA);
        cudaFree(fO);cudaFree(fl);cudaFree(fQ);cudaFree(fK);cudaFree(fV);cudaFree(fR); }
#else
      /* fused CPU backward (the training path) vs the same reference */
      { cudaMemset(gQ,0,(size_t)N*D*4); cudaMemset(gK,0,(size_t)N*KD*4); cudaMemset(gV,0,(size_t)N*KD*4);
        cpu_attn_bwd(dQ,dK,dV,ddA,dl,gR,gQ,gK,gV,WIN,B,T,H,KV,hd,D,KD);
        snprintf(lbl,sizeof(lbl),"attn bwd dQ FUSED (%s)",tag); ok &= check(lbl,relerr(gQ,rQ,(size_t)N*D),1e-5f);
        snprintf(lbl,sizeof(lbl),"attn bwd dK FUSED (%s)",tag); ok &= check(lbl,relerr(gK,rK,(size_t)N*KD),1e-5f);
        snprintf(lbl,sizeof(lbl),"attn bwd dV FUSED (%s)",tag); ok &= check(lbl,relerr(gV,rV,(size_t)N*KD),1e-5f); }
#endif
    }
    /* ---- tensor-core dKV vs the (CPU-validated) lane kernel ---------------
     * hd=64 with a ragged T (not a multiple of the 64-key / 32-query tiles),
     * full and sliding window, GQA grp=2. Pure kernel equivalence on random
     * fp16 operands; tolerance covers the fp16 rounding of P and dS. */
#ifndef CUDALM_CPU
    for(int wi=0;wi<2;wi++){
      int WIN=wi?37:0, B=2,T=200,H=4,KV=2,hd=64,D=H*hd,KD=KV*hd,N=B*T;
      size_t nq=(size_t)N*D, nk=(size_t)N*KD, nl=(size_t)B*H*T;
      __half*hq=(__half*)malloc(nq*2),*hg=(__half*)malloc(nq*2),*hk=(__half*)malloc(nk*2),*hv=(__half*)malloc(nk*2);
      float*hl=(float*)malloc(nl*4),*hr=(float*)malloc(nl*4);
      for(size_t i=0;i<nq;i++){ hq[i]=__float2half(rnd_normal()); hg[i]=__float2half(rnd_normal()); }
      for(size_t i=0;i<nk;i++){ hk[i]=__float2half(rnd_normal()); hv[i]=__float2half(rnd_normal()); }
      for(size_t i=0;i<nl;i++){ hl[i]=3.f+0.5f*rnd_normal(); hr[i]=rnd_normal(); }
      __half*dq_,*dg_,*dk_,*dv_; float*dl_,*dr_;
      cudaMalloc(&dq_,nq*2); cudaMalloc(&dg_,nq*2); cudaMalloc(&dk_,nk*2); cudaMalloc(&dv_,nk*2);
      cudaMalloc(&dl_,nl*4); cudaMalloc(&dr_,nl*4);
      cudaMemcpy(dq_,hq,nq*2,cudaMemcpyHostToDevice); cudaMemcpy(dg_,hg,nq*2,cudaMemcpyHostToDevice);
      cudaMemcpy(dk_,hk,nk*2,cudaMemcpyHostToDevice); cudaMemcpy(dv_,hv,nk*2,cudaMemcpyHostToDevice);
      cudaMemcpy(dl_,hl,nl*4,cudaMemcpyHostToDevice); cudaMemcpy(dr_,hr,nl*4,cudaMemcpyHostToDevice);
      float*aK=dzero(nk),*aV=dzero(nk),*bK=dzero(nk),*bV=dzero(nk);
      attn_bwd_dkv_lane(dq_,dk_,dv_,dg_,dl_,dr_,aK,aV,WIN,B,T,H,KV,hd,D,KD,0,1);
      if(attn_bwd_dkv_tc(dq_,dk_,dv_,dg_,dl_,dr_,bK,bV,WIN,B,T,H,KV,hd,D,KD)){
        float*ra=(float*)malloc(nk*4),*rb=(float*)malloc(nk*4); char lb[96];
        cudaMemcpy(ra,aK,nk*4,cudaMemcpyDeviceToHost); cudaMemcpy(rb,bK,nk*4,cudaMemcpyDeviceToHost);
        snprintf(lb,sizeof(lb),"attn bwd dK TC vs lane (W=%d)",WIN); ok &= check(lb,relerr(rb,ra,nk),5e-3f);
        cudaMemcpy(ra,aV,nk*4,cudaMemcpyDeviceToHost); cudaMemcpy(rb,bV,nk*4,cudaMemcpyDeviceToHost);
        snprintf(lb,sizeof(lb),"attn bwd dV TC vs lane (W=%d)",WIN); ok &= check(lb,relerr(rb,ra,nk),5e-3f);
        free(ra); free(rb);
      } else printf("  attn bwd dKV TC: no instantiation for hd=%d\n",hd);
      { float*aO=dzero(nq),*bO=dzero(nq),*aL=dzero(nl),*bL=dzero(nl);
        attn_fwd_lane(dq_,dk_,dv_,aO,aL,WIN,B,T,H,KV,hd,D,KD);
        if(attn_fwd_tc(dq_,dk_,dv_,bO,bL,WIN,B,T,H,KV,hd,D,KD)){
          float*ra=(float*)malloc(nq*4),*rb=(float*)malloc(nq*4); char lb[96];
          cudaMemcpy(ra,aO,nq*4,cudaMemcpyDeviceToHost); cudaMemcpy(rb,bO,nq*4,cudaMemcpyDeviceToHost);
          snprintf(lb,sizeof(lb),"attn fwd O TC vs lane (W=%d)",WIN); ok &= check(lb,relerr(rb,ra,nq),5e-3f);
          cudaMemcpy(ra,aL,nl*4,cudaMemcpyDeviceToHost); cudaMemcpy(rb,bL,nl*4,cudaMemcpyDeviceToHost);
          snprintf(lb,sizeof(lb),"attn fwd lse TC vs lane (W=%d)",WIN); ok &= check(lb,relerr(rb,ra,nl),1e-3f);
          free(ra); free(rb);
        } else printf("  attn fwd TC: no instantiation for hd=%d\n",hd);
        cudaFree(aO); cudaFree(bO); cudaFree(aL); cudaFree(bL); }      { float*aQ=dzero(nq),*bQ=dzero(nq);
        attn_bwd_dq_lane(dq_,dk_,dv_,dg_,dl_,aQ,dr_,WIN,B,T,H,KV,hd,D,KD,0);
        if(attn_bwd_dq_tc(dq_,dk_,dv_,dg_,dl_,dr_,bQ,WIN,B,T,H,KV,hd,D,KD)){
          float*ra=(float*)malloc(nq*4),*rb=(float*)malloc(nq*4); char lb[96];
          cudaMemcpy(ra,aQ,nq*4,cudaMemcpyDeviceToHost); cudaMemcpy(rb,bQ,nq*4,cudaMemcpyDeviceToHost);
          snprintf(lb,sizeof(lb),"attn bwd dQ TC vs lane (W=%d)",WIN); ok &= check(lb,relerr(rb,ra,nq),5e-3f);
          free(ra); free(rb);
        } else printf("  attn bwd dQ TC: no instantiation for hd=%d\n",hd);
        cudaFree(aQ); cudaFree(bQ); }
      { float*aO=dzero(nq),*aL=dzero(nl),*aQ=dzero(nq),*bO=dzero(nq),*bL=dzero(nl),*bQ=dzero(nq),*cK=dzero(nk),*cV=dzero(nk);
        attn_fwd_lane(dq_,dk_,dv_,aO,aL,WIN,B,T,H,KV,hd,D,KD);
        attn_bwd_dq_lane(dq_,dk_,dv_,dg_,dl_,aQ,dr_,WIN,B,T,H,KV,hd,D,KD,0);
        int fa0=g_fa; g_fa=1;
        if(attn_fwd_fa(dq_,dk_,dv_,bO,bL,WIN,B,T,H,KV,hd,D,KD)){
          attn_bwd_dq_fa(dq_,dk_,dv_,dg_,dl_,dr_,bQ,WIN,B,T,H,KV,hd,D,KD);
          attn_bwd_dkv_fa(dq_,dk_,dv_,dg_,dl_,dr_,cK,cV,WIN,B,T,H,KV,hd,D,KD);
          float*ra=(float*)malloc(nq*4),*rb=(float*)malloc(nq*4); char lb[96];
          cudaMemcpy(ra,aO,nq*4,cudaMemcpyDeviceToHost); cudaMemcpy(rb,bO,nq*4,cudaMemcpyDeviceToHost);
          snprintf(lb,sizeof(lb),"attn fwd O FA vs lane (W=%d)",WIN); ok &= check(lb,relerr(rb,ra,nq),5e-3f);
          cudaMemcpy(ra,aL,nl*4,cudaMemcpyDeviceToHost); cudaMemcpy(rb,bL,nl*4,cudaMemcpyDeviceToHost);
          snprintf(lb,sizeof(lb),"attn fwd lse FA vs lane (W=%d)",WIN); ok &= check(lb,relerr(rb,ra,nl),1e-3f);
          cudaMemcpy(ra,aQ,nq*4,cudaMemcpyDeviceToHost); cudaMemcpy(rb,bQ,nq*4,cudaMemcpyDeviceToHost);
          snprintf(lb,sizeof(lb),"attn bwd dQ FA vs lane (W=%d)",WIN); ok &= check(lb,relerr(rb,ra,nq),5e-3f);
          cudaMemcpy(ra,aK,nk*4,cudaMemcpyDeviceToHost); cudaMemcpy(rb,cK,nk*4,cudaMemcpyDeviceToHost);
          snprintf(lb,sizeof(lb),"attn bwd dK FA vs lane (W=%d)",WIN); ok &= check(lb,relerr(rb,ra,nk),5e-3f);
          cudaMemcpy(ra,aV,nk*4,cudaMemcpyDeviceToHost); cudaMemcpy(rb,cV,nk*4,cudaMemcpyDeviceToHost);
          snprintf(lb,sizeof(lb),"attn bwd dV FA vs lane (W=%d)",WIN); ok &= check(lb,relerr(rb,ra,nk),5e-3f);
          free(ra); free(rb);
        } else printf("  attn FA: no instantiation for hd=%d\n",hd);
        g_fa=fa0;
        cudaFree(aO);cudaFree(aL);cudaFree(aQ);cudaFree(bO);cudaFree(bL);cudaFree(bQ);cudaFree(cK);cudaFree(cV); }
      cudaFree(aK);cudaFree(aV);cudaFree(bK);cudaFree(bV);
      cudaFree(dq_);cudaFree(dg_);cudaFree(dk_);cudaFree(dv_);cudaFree(dl_);cudaFree(dr_);
      free(hq);free(hg);free(hk);free(hv);free(hl);free(hr);
    }
    /* doc mask: one sequence with a document boundary at S must equal the two
     * documents run as separate unmasked sequences (fwd O/lse, dQ, dK, dV) */
    { const int T=200,S=77,H=4,KV=2,hd=64,D=H*hd,KD=KV*hd;
      size_t nq=(size_t)T*D, nk=(size_t)T*KD, nl=(size_t)H*T;
      __half*hq=(__half*)malloc(nq*2),*hg=(__half*)malloc(nq*2),*hk=(__half*)malloc(nk*2),*hv=(__half*)malloc(nk*2);
      float*hl=(float*)malloc(nl*4),*hr=(float*)malloc(nl*4);
      for(size_t i=0;i<nq;i++){ hq[i]=__float2half(rnd_normal()); hg[i]=__float2half(rnd_normal()); }
      for(size_t i=0;i<nk;i++){ hk[i]=__float2half(rnd_normal()); hv[i]=__float2half(rnd_normal()); }
      for(size_t i=0;i<nl;i++){ hl[i]=3.f+0.5f*rnd_normal(); hr[i]=rnd_normal(); }
      /* out: O[nq] lse[nl] dQ[nq] dK[nk] dV[nk] for rows t0..t0+Tn-1 written at their full-sequence places */
      float*res[2]; for(int r=0;r<2;r++) res[r]=(float*)calloc(2*nq+nl+2*nk,4);
      auto run=[&](float*out,int t0,int Tn){
          size_t q=(size_t)Tn*D,k=(size_t)Tn*KD,l=(size_t)H*Tn;
          __half*dq_,*dg_,*dk_,*dv_; float*dl_,*dr_; cudaMalloc(&dq_,q*2); cudaMalloc(&dg_,q*2); cudaMalloc(&dk_,k*2); cudaMalloc(&dv_,k*2);
          cudaMalloc(&dl_,l*4); cudaMalloc(&dr_,l*4);
          cudaMemcpy(dq_,hq+(size_t)t0*D,q*2,cudaMemcpyHostToDevice); cudaMemcpy(dg_,hg+(size_t)t0*D,q*2,cudaMemcpyHostToDevice);
          cudaMemcpy(dk_,hk+(size_t)t0*KD,k*2,cudaMemcpyHostToDevice); cudaMemcpy(dv_,hv+(size_t)t0*KD,k*2,cudaMemcpyHostToDevice);
          for(int h=0;h<H;h++){ cudaMemcpy(dl_+(size_t)h*Tn,hl+(size_t)h*T+t0,(size_t)Tn*4,cudaMemcpyHostToDevice);
                                cudaMemcpy(dr_+(size_t)h*Tn,hr+(size_t)h*T+t0,(size_t)Tn*4,cudaMemcpyHostToDevice); }
          float*O=dzero(q),*L=dzero(l),*Q=dzero(q),*K=dzero(k),*V=dzero(k);
          attn_fwd_fa(dq_,dk_,dv_,O,L,0,1,Tn,H,KV,hd,D,KD) || attn_fwd_tc(dq_,dk_,dv_,O,L,0,1,Tn,H,KV,hd,D,KD);
          attn_bwd_dq_fa(dq_,dk_,dv_,dg_,dl_,dr_,Q,0,1,Tn,H,KV,hd,D,KD) || attn_bwd_dq_tc(dq_,dk_,dv_,dg_,dl_,dr_,Q,0,1,Tn,H,KV,hd,D,KD);
          attn_bwd_dkv_fa(dq_,dk_,dv_,dg_,dl_,dr_,K,V,0,1,Tn,H,KV,hd,D,KD) || attn_bwd_dkv_tc(dq_,dk_,dv_,dg_,dl_,dr_,K,V,0,1,Tn,H,KV,hd,D,KD);
          cudaMemcpy(out+(size_t)t0*D,O,q*4,cudaMemcpyDeviceToHost);
          for(int h=0;h<H;h++) cudaMemcpy(out+nq+(size_t)h*T+t0,L+(size_t)h*Tn,(size_t)Tn*4,cudaMemcpyDeviceToHost);
          cudaMemcpy(out+nq+nl+(size_t)t0*D,Q,q*4,cudaMemcpyDeviceToHost);
          cudaMemcpy(out+2*nq+nl+(size_t)t0*KD,K,k*4,cudaMemcpyDeviceToHost);
          cudaMemcpy(out+2*nq+nl+nk+(size_t)t0*KD,V,k*4,cudaMemcpyDeviceToHost);
          cudaFree(dq_);cudaFree(dg_);cudaFree(dk_);cudaFree(dv_);cudaFree(dl_);cudaFree(dr_);
          cudaFree(O);cudaFree(L);cudaFree(Q);cudaFree(K);cudaFree(V); };
      int fa0=g_fa;
      for(int fv=0;fv<2;fv++){ g_fa=fv; memset(res[0],0,(2*nq+nl+2*nk)*4); memset(res[1],0,(2*nq+nl+2*nk)*4);
      int dsh[T]; for(int t=0;t<T;t++) dsh[t]=t<S?0:S;
      int*dsd; cudaMalloc(&dsd,T*4); cudaMemcpy(dsd,dsh,T*4,cudaMemcpyHostToDevice);
      cudaMemcpyToSymbol(d_dstart,&dsd,sizeof(int*)); run(res[0],0,T);
      const int*nul=NULL; cudaMemcpyToSymbol(d_dstart,&nul,sizeof(int*));
      run(res[1],0,S); run(res[1],S,T-S);
      ok &= check("attn docmask fwd O == split docs",relerr(res[0],res[1],nq),1e-3f);
      ok &= check("attn docmask fwd lse == split docs",relerr(res[0]+nq,res[1]+nq,nl),1e-4f);
      ok &= check("attn docmask dQ == split docs",relerr(res[0]+nq+nl,res[1]+nq+nl,nq),1e-3f);
      ok &= check("attn docmask dK == split docs",relerr(res[0]+2*nq+nl,res[1]+2*nq+nl,nk),1e-3f);
      ok &= check(fv?"attn docmask dV == split docs (FA)":"attn docmask dV == split docs",relerr(res[0]+2*nq+nl+nk,res[1]+2*nq+nl+nk,nk),1e-3f);
      cudaFree(dsd); }
      g_fa=fa0; free(res[0]); free(res[1]); free(hq);free(hg);free(hk);free(hv);free(hl);free(hr);
      if(g_ds) cudaMemcpyToSymbol(d_dstart,&g_ds,sizeof(int*)); }
#endif
    /* ---- MoE top-1 forward + backward vs a naive CPU reference -------------
     * The routed FFN is the newest code here and its backward touches the
     * router, the gate path and the load-balance aux term, none of which is
     * covered by the dense checks above. */
    { Cfg m; memset(&m,0,sizeof(m));
      m.V=64; m.D=32; m.L=1; m.H=4; m.KV=2; m.F=48; m.T=8; m.n_exp=4; m.rope_base=10000.f;
      cfg_derive(&m); gC=m;
      int N=64,D=m.D,F=m.F,E=m.n_exp;
      Off mo; calc_offsets(&m,&mo);
      size_t np=param_count(&m);
      float*Ph=(float*)calloc(np,4);
      for(size_t i=0;i<np;i++) Ph[i]=0.15f*rnd_normal();
      float*fn=(float*)malloc((size_t)N*D*4),*fi=(float*)malloc((size_t)N*D*4),*dO=(float*)malloc((size_t)N*D*4);
      for(int i=0;i<N*D;i++){ fn[i]=rnd_normal(); fi[i]=rnd_normal(); dO[i]=0.3f*rnd_normal(); }
      /* ---- CPU reference ---- */
      const float *wr=Ph+mo.wr[0],*W1=Ph+mo.w1[0],*W3=Ph+mo.w3[0],*W2=Ph+mo.w2[0];
      int *as=(int*)malloc(N*4); float *gt=(float*)malloc(N*4),*pr=(float*)malloc((size_t)N*E*4);
      float *rout=(float*)malloc((size_t)N*D*4);
      memcpy(rout,fi,(size_t)N*D*4);
      float *ry=(float*)calloc((size_t)N*D,4);
      for(int n=0;n<N;n++){
          float r[64]; for(int e=0;e<E;e++){ double s=0; for(int d=0;d<D;d++) s+=fn[n*D+d]*wr[d*E+e]; r[e]=(float)s; }
          float mx=-1e30f; for(int e=0;e<E;e++) if(r[e]>mx)mx=r[e];
          float s=0; for(int e=0;e<E;e++){ pr[n*E+e]=expf(r[e]-mx); s+=pr[n*E+e]; }
          int best=0; for(int e=0;e<E;e++){ pr[n*E+e]/=s; if(pr[n*E+e]>pr[n*E+best])best=e; }
          as[n]=best; gt[n]=pr[n*E+best];
      }
      float *rg=(float*)malloc((size_t)F*4),*ru=(float*)malloc((size_t)F*4),*ra=(float*)malloc((size_t)F*4);
      for(int n=0;n<N;n++){ int e=as[n];
          const float*w1=W1+(size_t)e*D*F,*w3=W3+(size_t)e*D*F,*w2=W2+(size_t)e*F*D;
          for(int f=0;f<F;f++){ double a=0,b=0; for(int d=0;d<D;d++){ a+=fn[n*D+d]*w1[d*F+f]; b+=fn[n*D+d]*w3[d*F+f]; }
              rg[f]=(float)a; ru[f]=(float)b; ra[f]=(rg[f]/(1.0f+expf(-rg[f])))*ru[f]; }
          for(int d=0;d<D;d++){ double y=0; for(int f=0;f<F;f++) y+=ra[f]*w2[f*D+d];
              ry[n*D+d]=(float)y; rout[n*D+d]+=gt[n]*(float)y; } }
      /* CPU backward */
      float *rdf=(float*)calloc((size_t)N*D,4),*rG=(float*)calloc(np,4),*dgate=(float*)calloc(N,4);
      for(int n=0;n<N;n++){ double s=0; for(int d=0;d<D;d++) s+=dO[n*D+d]*ry[n*D+d]; dgate[n]=(float)s; }
      float *rda=(float*)malloc((size_t)F*4),*rdg=(float*)malloc((size_t)F*4),*rdu=(float*)malloc((size_t)F*4);
      for(int n=0;n<N;n++){ int e=as[n];
          const float*w1=W1+(size_t)e*D*F,*w3=W3+(size_t)e*D*F,*w2=W2+(size_t)e*F*D;
          float*g1=rG+mo.w1[0]+(size_t)e*D*F,*g3=rG+mo.w3[0]+(size_t)e*D*F,*g2=rG+mo.w2[0]+(size_t)e*F*D;
          for(int f=0;f<F;f++){ double a=0,b=0; for(int d=0;d<D;d++){ a+=fn[n*D+d]*w1[d*F+f]; b+=fn[n*D+d]*w3[d*F+f]; }
              rg[f]=(float)a; ru[f]=(float)b; ra[f]=(rg[f]/(1.0f+expf(-rg[f])))*ru[f]; }
          for(int f=0;f<F;f++){ double s=0; for(int d=0;d<D;d++){ float dy=gt[n]*dO[n*D+d];
                  g2[f*D+d]+=ra[f]*dy; s+=dy*w2[f*D+d]; } rda[f]=(float)s; }
          for(int f=0;f<F;f++){ float z=rg[f],sg=1.0f/(1.0f+expf(-z));
              rdg[f]=rda[f]*ru[f]*(sg+z*sg*(1.0f-sg)); rdu[f]=rda[f]*(z*sg); }
          for(int d=0;d<D;d++){ double s=0; for(int f=0;f<F;f++){ g1[d*F+f]+=fn[n*D+d]*rdg[f];
                  g3[d*F+f]+=fn[n*D+d]*rdu[f]; s+=rdg[f]*w1[d*F+f]+rdu[f]*w3[d*F+f]; }
              rdf[n*D+d]+=(float)s; } }
      { int cnt[64]={0}; for(int n=0;n<N;n++) cnt[as[n]]++;
        float invN=1.0f/N; float *gwr=rG+mo.wr[0];
        for(int n=0;n<N;n++){ float dp[64],dot=0;
            for(int e=0;e<E;e++){ dp[e]=MOE_AUX*E*(cnt[e]*invN)*invN; if(e==as[n]) dp[e]+=dgate[n]; }
            for(int e=0;e<E;e++) dot+=pr[n*E+e]*dp[e];
            float dr[64]; for(int e=0;e<E;e++) dr[e]=pr[n*E+e]*(dp[e]-dot);
            for(int d=0;d<D;d++){ double s=0; for(int e=0;e<E;e++){ s+=dr[e]*wr[d*E+e]; gwr[d*E+e]+=fn[n*D+d]*dr[e]; }
                rdf[n*D+d]+=(float)s; } } }
      /* ---- GPU ---- */
      GA*ga=ga_new(N,1);
      float*Pd=(float*)dmalloc(np*4); cudaMemcpy(Pd,Ph,np*4,cudaMemcpyHostToDevice);
      float*Gd=dzero(np);
      cudaMemcpy(ga->f_norm[0],fn,(size_t)N*D*4,cudaMemcpyHostToDevice);
      cudaMemcpy(ga->f_in[0],  fi,(size_t)N*D*4,cudaMemcpyHostToDevice);
      cudaMemcpy(ga->dout,     dO,(size_t)N*D*4,cudaMemcpyHostToDevice);
      float*outd=(float*)dmalloc((size_t)N*D*4);
      int tk0=g_topk, t1o=g_t1old; g_topk=1; g_t1old=1; /* CPU ref is top-1 */
      moe_forward(Pd,&mo,ga,0,outd,N);
      moe_backward(Pd,Gd,&mo,ga,0,N);
      g_topk=tk0; g_t1old=t1o;
      ck(cudaDeviceSynchronize(),"moe");
      float*go=(float*)malloc((size_t)N*D*4),*gdf=(float*)malloc((size_t)N*D*4),*gG=(float*)malloc(np*4);
      cudaMemcpy(go, outd,      (size_t)N*D*4,cudaMemcpyDeviceToHost);
      cudaMemcpy(gdf,ga->dfnorm,(size_t)N*D*4,cudaMemcpyDeviceToHost);
      cudaMemcpy(gG, Gd,        np*4,         cudaMemcpyDeviceToHost);
      ok &= check("moe fwd out",   relerr(go, rout,(size_t)N*D),2e-3f);
      ok &= check("moe bwd dfnorm",relerr(gdf,rdf, (size_t)N*D),2e-3f);
      ok &= check("moe bwd dW1",   relerr(gG+mo.w1[0],rG+mo.w1[0],(size_t)E*D*F),2e-3f);
      ok &= check("moe bwd dW2",   relerr(gG+mo.w2[0],rG+mo.w2[0],(size_t)E*F*D),2e-3f);
      ok &= check("moe bwd dWr",   relerr(gG+mo.wr[0],rG+mo.wr[0],(size_t)D*E),2e-3f);
#ifndef CUDALM_CPU
      /* S3b: top-2 BF16 grouped experts vs the fp32 path, stash on and off */
      { int tk1=g_topk, st0=g_moe_stash; float mu=g_mbias_u; g_topk=2; g_mbias_u=0;
        size_t no=(size_t)N*D, nr=2*no+np; float*res[4]; int mg0=g_moemrg;
        for(int r=0;r<4;r++){ g_moe_stash=r!=2; g_moemrg=r==3?0:mg0; GA*gb=ga_new(N,1); gb->tr=r>0; g_fwd_id++;
          cudaMemcpy(gb->f_norm[0],fn,no*4,cudaMemcpyHostToDevice); cudaMemcpy(gb->f_in[0],fi,no*4,cudaMemcpyHostToDevice);
          cudaMemcpy(gb->dout,dO,no*4,cudaMemcpyHostToDevice); ck(cudaMemset(Gd,0,np*4),"moe bf G");
          moe_forward(Pd,&mo,gb,0,outd,N); moe_backward(Pd,Gd,&mo,gb,0,N); ck(cudaDeviceSynchronize(),"moe bf");
          res[r]=(float*)malloc(nr*4);
          cudaMemcpy(res[r],outd,no*4,cudaMemcpyDeviceToHost); cudaMemcpy(res[r]+no,gb->dfnorm,no*4,cudaMemcpyDeviceToHost);
          cudaMemcpy(res[r]+2*no,Gd,np*4,cudaMemcpyDeviceToHost); }
        float*g0=res[0]+2*no,*g1=res[1]+2*no; size_t nw=(size_t)E*D*F;
        ok &= check("moe top2 BF16 fwd vs fp32",   relerr(res[1],res[0],no),3e-2f);
        ok &= check("moe top2 BF16 dfnorm vs fp32",relerr(res[1]+no,res[0]+no,no),3e-2f);
        ok &= check("moe top2 BF16 dW1 vs fp32",   relerr(g1+mo.w1[0],g0+mo.w1[0],nw),3e-2f);
        ok &= check("moe top2 BF16 dW3 vs fp32",   relerr(g1+mo.w3[0],g0+mo.w3[0],nw),3e-2f);
        ok &= check("moe top2 BF16 dW2 vs fp32",   relerr(g1+mo.w2[0],g0+mo.w2[0],nw),3e-2f);
        ok &= check("moe top2 BF16 no-stash == stash",relerr(res[2],res[3],nr),1e-3f);
        ok &= check("moe top2 BF16 merged vs per-pass",relerr(res[1],res[3],nr),1e-2f);
        g_moemrg=mg0; for(int r=0;r<4;r++) free(res[r]);
        g_topk=tk1; g_moe_stash=st0; g_mbias_u=mu; }
      /* O1: GPU counting-sort permutation == the host build, per pass and merged (with inverse) */
      { const int Nn=5000,Ee=7,Kk=2,KN=Kk*Nn,NC=(KN+GP_T-1)/GP_T; int *ha=(int*)malloc(KN*4),*hp=(int*)malloc(KN*4),*hr=(int*)malloc(KN*4),*hi=(int*)malloc(KN*4);
        uint64_t sd=77; for(int i=0;i<KN;i++){ uint64_t x=pm_mix(&sd); ha[i]=(int)(x%3?x%2:x%Ee); }
        int *d=(int*)dmalloc(((size_t)3*KN+(size_t)Kk*Ee*NC+Kk*(Ee+1)+1)*4), *da=d, *dp=d+KN, *di=d+2*KN, *ch=d+3*KN, *off=ch+(size_t)Kk*Ee*NC, *bad=off+Kk*(Ee+1);
        cudaMemcpy(da,ha,KN*4,cudaMemcpyHostToDevice); cudaMemset(bad,0,4);
        long mis=0;
        for(int mode=0;mode<2;mode++){ int n=mode?KN:Nn, k=mode?1:Kk, nc=(n+GP_T-1)/GP_T;
          KL(k_gp_hist,dim3(nc,k),GP_T,Ee*4)(da,n,Ee,nc,ch,bad); KL(k_gp_scan,k,256,Ee*4)(ch,Ee,nc,off);
          KL(k_gp_scatter,dim3(nc,k),GP_T,32*Ee*4)(da,n,Ee,nc,ch,dp,(size_t)n,mode?di:(int*)NULL);
          cudaMemcpy(hp,dp,KN*4,cudaMemcpyDeviceToHost); cudaMemcpy(hi,di,KN*4,cudaMemcpyDeviceToHost);
          for(int kk=0;kk<k;kk++){ int cur[8]={0},o[9]={0}; const int*as=ha+(size_t)kk*n;
            for(int i=0;i<n;i++) o[as[i]+1]++; for(int e=0;e<Ee;e++) o[e+1]+=o[e]; for(int e=0;e<Ee;e++) cur[e]=o[e];
            for(int i=0;i<n;i++) hr[(size_t)kk*n+cur[as[i]]++]=i; }
          for(int i=0;i<k*n;i++){ if(hp[i]!=hr[i]) mis++; if(mode&&hi[hr[i]]!=i) mis++; } }
        ok &= check("moe GPU perm == host perm",(float)mis,0.5f);
        cudaFree(d); free(ha); free(hp); free(hr); free(hi); }
#endif
    }
    /* ---- router z-loss: drt += 2c/N*lse*softmax against central differences of c*mean_n lse_n^2 ---- */
    { int N=37,E=8; float c=0.01f; size_t ne=(size_t)N*E;
      float*hr=(float*)malloc(ne*4),*hp=(float*)malloc(ne*4),*hg=(float*)malloc(ne*4),*fd=(float*)malloc(ne*4);
      for(size_t i=0;i<ne;i++) hr[i]=2.0f*rnd_normal();
      for(int n=0;n<N;n++){ double mx=-1e30,s=0; for(int e=0;e<E;e++) mx=fmax(mx,(double)hr[n*E+e]);
          for(int e=0;e<E;e++) s+=exp(hr[n*E+e]-mx);
          for(int e=0;e<E;e++) hp[n*E+e]=(float)(exp(hr[n*E+e]-mx)/s); }
      float*dr=(float*)dmalloc(ne*4),*dp=(float*)dmalloc(ne*4),*dl=(float*)dmalloc((size_t)N*4),*dg=dzero(ne);
      cudaMemcpy(dr,hr,ne*4,cudaMemcpyHostToDevice); cudaMemcpy(dp,hp,ne*4,cudaMemcpyHostToDevice);
      KL(k_rlse,nblk(N,TPB),TPB)(dr,dl,N,E); KL(k_zl_bwd,nblk(ne,TPB),TPB)(dg,dp,dl,N,E,2.f*c/N);
      ck(cudaDeviceSynchronize(),"zl selftest"); cudaMemcpy(hg,dg,ne*4,cudaMemcpyDeviceToHost);
      auto zf=[&](size_t i,double h)->double{ double z=0;
          for(int n=0;n<N;n++){ double mx=-1e30,s=0;
              for(int e=0;e<E;e++){ size_t k=(size_t)n*E+e; double v=hr[k]+(k==i?h:0.0); mx=v>mx?v:mx; }
              for(int e=0;e<E;e++){ size_t k=(size_t)n*E+e; s+=exp(hr[k]+(k==i?h:0.0)-mx); }
              double l=mx+log(s); z+=l*l; }
          return c*z/N; };
      for(size_t i=0;i<ne;i++) fd[i]=(float)((zf(i,1e-3)-zf(i,-1e-3))/2e-3);
      ok &= check("router z-loss grad",relerr(hg,fd,ne),1e-3f);
      cudaFree(dr); cudaFree(dp); cudaFree(dl); cudaFree(dg); free(hr); free(hp); free(hg); free(fd); }
    /* ---- qlm: integer weights + Integer AdamW ----------------------------
     * Every other kernel here is checked against a CPU reference; these must be
     * too, because the CPU/GPU agreement invariant is what keeps this project
     * debuggable. qsri() is __host__ __device__ and keyed on (index, step), so
     * the reference can reproduce the stochastic rounding exactly and the
     * digits should agree bit-for-bit apart from fp tie-breaks. */
    {
        Cfg qc; preset("m1",&qc); Off qo; calc_offsets(&qc,&qo);
        size_t qnp=param_count(&qc);
        float*Ph=(float*)malloc(qnp*4), *Gh=(float*)malloc(qnp*4);
        for(size_t i=0;i<qnp;i++){ Ph[i]=0.02f*rnd_normal(); Gh[i]=0.01f*rnd_normal(); }
        float*Pd=(float*)dmalloc(qnp*4), *Gd=(float*)dmalloc(qnp*4);
        cudaMemcpy(Pd,Ph,qnp*4,cudaMemcpyHostToDevice);
        cudaMemcpy(Gd,Gh,qnp*4,cudaMemcpyHostToDevice);

        Qlm qt; qlm_gpu_init(&qt,&qc,&qo,Pd,qnp);

        /* (a) encode+render must land within half a quantisation step */
        { float*Pr=(float*)malloc(qnp*4);
          cudaMemcpy(Pr,Pd,qnp*4,cudaMemcpyDeviceToHost);
          float*sh=(float*)malloc((size_t)qt.ngrp*4);
          cudaMemcpy(sh,qt.s,(size_t)qt.ngrp*4,cudaMemcpyDeviceToHost);
          double worst=0; int lvlmax=0;
          for(int si=0;si<qt.nspan;si++){
            for(int g=0;g<qt.spanH[si].ngrp;g++){
              int gi=qt.spanH[si].gi0+g;
              unsigned long long j0=(unsigned long long)g*qt.group;
              int n=(int)((qt.spanH[si].n-j0 < (unsigned long long)qt.group)
                          ? qt.spanH[si].n-j0 : (unsigned long long)qt.group);
              float sg=sh[gi];
              float lv[512]; int nl=0;
              for(int j=0;j<n;j++){
                size_t k=(size_t)qt.spanH[si].pof+j0+j;
                double e=fabs((double)Pr[k]-(double)Ph[k]);
                if(e/sg>worst) worst=e/sg;
                int seen=0; for(int q2=0;q2<nl;q2++) if(lv[q2]==Pr[k]){seen=1;break;}
                if(!seen && nl<512) lv[nl++]=Pr[k];
              }
              if(nl>lvlmax) lvlmax=nl;
            }
          }
          ok &= check("qlm encode err/step",(float)worst,0.51f);
          printf("  qlm distinct levels per group: %d (max allowed %d)  %s\n",
                 lvlmax, 2*qt.dmax+1, lvlmax<=2*qt.dmax+1?"OK":"FAIL");
          if(lvlmax>2*qt.dmax+1) ok=0;
          free(Pr); free(sh);
        }

        /* (b) one Integer AdamW step, GPU vs scalar CPU reference */
        int   T=3;
        float lr=3e-4f,b1=0.9f,b2=0.95f,wd=0.1f;
        float bc1=1.0f-powf(b1,(float)T), bc2=1.0f-powf(b2,(float)T);
        unsigned long long nq=qt.nq, ng=qt.ngrp;
        unsigned long long ndb=qd_bytes(nq,qt.packed);
        signed char*hd=(signed char*)malloc(ndb),*hr=(signed char*)malloc(nq),
                   *hm=(signed char*)malloc(nq);
        unsigned char*hv=(unsigned char*)malloc(nq);
        float*hs=(float*)malloc(ng*4),*hms=(float*)malloc(ng*4),*hvs=(float*)malloc(ng*4);
        cudaMemcpy(hd,qt.d,ndb,cudaMemcpyDeviceToHost);
        cudaMemcpy(hr,qt.r,nq,cudaMemcpyDeviceToHost);
        cudaMemcpy(hm,qt.m8,nq,cudaMemcpyDeviceToHost);
        if(qt.adam) cudaMemcpy(hv,qt.v8,nq,cudaMemcpyDeviceToHost);
        cudaMemcpy(hs,qt.s,ng*4,cudaMemcpyDeviceToHost);
        cudaMemcpy(hms,qt.m8s,ng*4,cudaMemcpyDeviceToHost);
        if(qt.adam) cudaMemcpy(hvs,qt.v8s,ng*4,cudaMemcpyDeviceToHost);

        signed char*rd=(signed char*)malloc(ndb),*rr_=(signed char*)malloc(nq);
        float*rP=(float*)malloc(qnp*4); memcpy(rP,Ph,qnp*4);
        float inv_bc1=1.0f/bc1, inv_sqrt_bc2=1.0f/sqrtf(bc2);
        for(int si=0;si<qt.nspan;si++){
          for(int g=0;g<qt.spanH[si].ngrp;g++){
            int gi=qt.spanH[si].gi0+g;
            unsigned long long j0=(unsigned long long)g*qt.group;
            int n=(int)((qt.spanH[si].n-j0 < (unsigned long long)qt.group)
                        ? qt.spanH[si].n-j0 : (unsigned long long)qt.group);
            unsigned long long pof=qt.spanH[si].pof+j0, qi0=qt.spanH[si].qi0+j0;
            float sg=hs[gi];
            float L=lr*128.0f/(qt.lrmode?sg:qt.spanH[si].sref), dec=lr*wd*128.0f;
            float ms_old=hms[gi], vs_old=hvs[gi];
            float mns[512], rvs[512], amax=0.0f, vmax=0.0f;
            for(int j=0;j<n;j++){
              unsigned long long qi=qi0+j, k=pof+j;
              float gr=Gh[k];
              float mv=(float)hm[qi]*ms_old, u, mn;
              if(qt.adam){
                float sv=(float)hv[qi]*vs_old, vv=sv*sv;
                mn=b1*mv+(1.0f-b1)*gr;
                vv=b2*vv+(1.0f-b2)*gr*gr;
                float rv=sqrtf(vv);
                float veps=fmaxf(vs_old,1e-8f); u=(mn*inv_bc1)/(rv*inv_sqrt_bc2+veps); u=fmaxf(-8.0f,fminf(8.0f,u));
                rvs[j]=rv; if(rv>vmax)vmax=rv;
              } else {
                float cc=b1*mv+(1.0f-b1)*gr;
                u=(float)((cc>0.0f)-(cc<0.0f));
                mn=b2*mv+(1.0f-b2)*gr; rvs[j]=0.0f;
              }
              mns[j]=mn; { float a=fabsf(mn); if(a>amax)amax=a; }
              int dd=qd_get(hd,qi,qt.packed);
              int idl=qsri(-(L*u+dec*(float)dd), qi, (unsigned long long)T);
              int rr2=(int)hr[qi]+idl;
              int cc2=(rr2>64)?(((rr2-65)>>7)+1):((rr2<-64)?-((((-rr2)-65)>>7)+1):0);
              rr2-=cc2*128; dd+=cc2;
              dd=qclmp(dd,-qt.dmax,qt.dmax);
              qd_set_host(rd,qi,dd,qt.packed); rr_[qi]=(signed char)rr2;
              rP[k]=sg*(float)dd;
            }
            (void)mns; (void)rvs; (void)amax; (void)vmax;
          }
        }

        /* (b0) S1/S2: warp kernel with a device gscale must match the block
         * kernel bit-for-bit (same state, same step, gscale 0.37) */
        { void*buf[7]={Pd,qt.d,qt.r,qt.m8,qt.v8,qt.m8s,qt.v8s};
          size_t bn[7]={qnp*4,ndb,nq,nq,qt.adam?nq:0,ng*4,qt.adam?ng*4:0};
          for(int i=0;i<7;i++) if(!bn[i]) buf[i]=NULL;
          void*s0[7],*sa[7]; char*ha=NULL,*hb=NULL; size_t tot=0;
          for(int i=0;i<7;i++){ if(!bn[i]){ s0[i]=sa[i]=NULL; continue; } s0[i]=dmalloc(bn[i]); sa[i]=dmalloc(bn[i]); tot+=bn[i];
                                cudaMemcpy(s0[i],buf[i],bn[i],cudaMemcpyDeviceToDevice); }
          float*gsd=(float*)dmalloc(4), gsv=0.37f; cudaMemcpy(gsd,&gsv,4,cudaMemcpyHostToDevice);
          KL(k_qlm_step,(unsigned)qt.ngrp,qt.group,qt.group)(Pd,Gd,qt.d,qt.r,qt.s,qt.m8,qt.m8s,
              qt.v8,qt.v8s,qt.spanD,qt.nspan,qt.group,qt.dmax,lr,b1,b2,wd,gsv,inv_bc1,inv_sqrt_bc2,
              qt.lrmode,(unsigned long long)T,qt.adam,NULL,qt.packed,0);
          for(int i=0;i<7;i++) if(bn[i]){ cudaMemcpy(sa[i],buf[i],bn[i],cudaMemcpyDeviceToDevice);
                                cudaMemcpy(buf[i],s0[i],bn[i],cudaMemcpyDeviceToDevice); }
          qlm_step_launch(Pd,Gd,qt.d,qt.r,qt.s,qt.m8,qt.m8s,qt.v8,qt.v8s,qt.spanD,qt.nspan,qt.group,
              qt.dmax,lr,b1,b2,wd,0.0f,gsd,inv_bc1,inv_sqrt_bc2,qt.lrmode,(unsigned long long)T,
              qt.adam,NULL,qt.packed,0,(int)qt.ngrp);
          ck(cudaDeviceSynchronize(),"qlm stepw selftest");
          ha=(char*)malloc(tot); hb=(char*)malloc(tot);
          size_t o=0; int bad=0;
          for(int i=0;i<7;i++) if(bn[i]){ cudaMemcpy(ha+o,sa[i],bn[i],cudaMemcpyDeviceToHost);
                                cudaMemcpy(hb+o,buf[i],bn[i],cudaMemcpyDeviceToHost);
                                if(memcmp(ha+o,hb+o,bn[i])) bad|=1<<i; o+=bn[i];
                                cudaMemcpy(buf[i],s0[i],bn[i],cudaMemcpyDeviceToDevice);
                                cudaFree(s0[i]); cudaFree(sa[i]); }
          printf("  qlm warp step vs block step (group %d): %s (mask %d)\n",qt.group,bad?"FAIL":"bit-exact OK",bad);
          if(bad) ok=0;
          free(ha); free(hb); cudaFree(gsd);
        }
        cudaMemset(qt.statD,0,4*8);
        qlm_step_launch(Pd,Gd,qt.d,qt.r,qt.s,qt.m8,qt.m8s,qt.v8,qt.v8s,qt.spanD,qt.nspan,qt.group,
            qt.dmax,lr,b1,b2,wd,1.0f,NULL,inv_bc1,inv_sqrt_bc2,qt.lrmode,(unsigned long long)T,
            qt.adam,qt.statD,qt.packed,0,(int)qt.ngrp);
        ck(cudaDeviceSynchronize(),"qlm selftest step");

        signed char*gd=(signed char*)malloc(ndb),*gr2=(signed char*)malloc(nq);
        float*gP=(float*)malloc(qnp*4);
        cudaMemcpy(gd,qt.d,ndb,cudaMemcpyDeviceToHost);
        cudaMemcpy(gr2,qt.r,nq,cudaMemcpyDeviceToHost);
        cudaMemcpy(gP,Pd,qnp*4,cudaMemcpyDeviceToHost);
        unsigned long long dmiss=0,rmiss=0; int dworst=0;
        for(unsigned long long i=0;i<nq;i++){
          int dv=qd_get(gd,i,qt.packed)-qd_get(rd,i,qt.packed);
          if(dv){ dmiss++; if(abs(dv)>dworst)dworst=abs(dv); }
          if(gr2[i]!=rr_[i]) rmiss++;
        }
        double dfrac=100.0*(double)dmiss/(double)nq, rfrac=100.0*(double)rmiss/(double)nq;
        /* fp tie-breaks near a rounding boundary can differ between host and
         * device, so a handful of disagreements is expected; a systematic bug
         * shows up as percent-level, not 0.0x-level. */
        printf("  qlm step digits differ %.4f%% (max |d| diff %d), residual differ %.4f%%  %s\n",
               dfrac, dworst, rfrac, (dfrac<0.05 && dworst<=1 && rfrac<0.05)?"OK":"FAIL");
        if(!(dfrac<0.05 && dworst<=1 && rfrac<0.05)) ok=0;
        ok &= check("qlm step P",relerr(gP,rP,qnp),2e-3f);

        free(Ph);free(Gh);free(hd);free(hr);free(hm);free(hv);free(hs);free(hms);free(hvs);
        free(rd);free(rr_);free(rP);free(gd);free(gr2);free(gP);
        cudaFree(Pd);cudaFree(Gd);
        cudaFree(qt.d);cudaFree(qt.r);cudaFree(qt.m8);cudaFree(qt.v8);
        cudaFree(qt.s);cudaFree(qt.m8s);cudaFree(qt.v8s);cudaFree(qt.statD);cudaFree(qt.spanD);
        cudaFree(qt.Mc);cudaFree(qt.Vc);
        free(qt.spanH);
    }
    /* ---- Newton-Schulz orthogonalisation (Muon) ---------------------------
     * NS has a property worth testing directly: for m<=n the result should
     * satisfy X X^T ~= I. That is checkable without a reference implementation,
     * which matters because there is no CPU Muon here to diff against -- and a
     * subtly wrong orthogonalisation would not crash, it would just make the
     * optimiser quietly worse than AdamW.
     *
     * The tolerance is loose (5 quintic steps do not converge to machine
     * precision, and are not meant to) but it separates "approximately
     * orthogonal" from "diverged" or "silently the identity". */
    for(int oi=0;oi<2;oi++){
        /* oi=0 is the m<=n branch (Gram = X X^T), oi=1 the m>n branch
         * (Gram = X^T X). Both must orthogonalise; they are also checked
         * against EACH OTHER further down, which is the stronger test. */
        int m = oi?80:48, n = oi?48:80, batch=3, p=(m<n?m:n);
        const char*obr = oi?"m>n":"m<n";
        size_t nX=(size_t)batch*m*n, nA=(size_t)batch*p*p;
        float*Xh=(float*)malloc(nX*4);
        for(size_t i=0;i<nX;i++) Xh[i]=rnd_normal();
        /* scale each slice so ||X||_F <= 1, which NS requires to converge */
        for(int b=0;b<batch;b++){
            double s=0; for(int i=0;i<m*n;i++){ float v=Xh[(size_t)b*m*n+i]; s+=(double)v*v; }
            float inv=(float)(1.0/(sqrt(s)+1e-12));
            for(int i=0;i<m*n;i++) Xh[(size_t)b*m*n+i]*=inv;
        }
        float*X=(float*)dmalloc(nX*4),*A=(float*)dmalloc(nA*4),
             *B=(float*)dmalloc(nA*4),*T=(float*)dmalloc(nX*4);
        ck(cudaMemcpy(X,Xh,nX*4,cudaMemcpyHostToDevice),"ns X h2d");
        float*Ah=(float*)malloc(nA*4), *Ih=(float*)calloc(nA,4);
        for(int b=0;b<batch;b++) for(int i=0;i<p;i++) Ih[(size_t)b*p*p+(size_t)i*p+i]=1.0f;
        /* distance from orthogonality BEFORE the sweep, measured on the SMALLER
         * Gram matrix (the larger one is rank-deficient and can never be I) */
        #define NSGRAM() do{ if(m<=n) gb_ab(X,(long long)m*n, X,(long long)m*n, A,(long long)p*p, \
                                            m,n,m, batch, 1.0f,0.0f, 1); \
                             else gb_atb(X,(long long)m*n, A,(long long)p*p, m,n, batch, 1.0f,0.0f); }while(0)
        NSGRAM();
        ck(cudaMemcpy(Ah,A,nA*4,cudaMemcpyDeviceToHost),"ns A0 d2h");
        float before=relerr(Ah,Ih,nA);
        int save_ns=g_muon_ns; g_muon_ns=5;
        ns_batch(X,m,n,batch,A,B,T);
        g_muon_ns=save_ns;
        NSGRAM();
        #undef NSGRAM
        ck(cudaMemcpy(Ah,A,nA*4,cudaMemcpyDeviceToHost),"ns A1 d2h");
        float after=relerr(Ah,Ih,nA);
        /* The quintic is deliberately inexact -- 5 steps leave the singular
         * values spread around 1 rather than equal to it, so "after" lands near
         * 0.3, not 1e-6. What must hold is that it moved a long way TOWARD
         * orthogonality: a broken iteration either diverges (after >> before) or
         * does nothing (after ~= before). */
        char nl[48]; snprintf(nl,sizeof nl,"muon ns [%s] Gram-I",obr);
        printf("  %-22s %.3f -> %.3f\n",nl,before,after);
        snprintf(nl,sizeof nl,"muon ns [%s] orthog.",obr);
        ok &= check(nl, after/(before>0?before:1.0f), 0.5f);
        free(Xh);free(Ah);free(Ih);
        cudaFree(X);cudaFree(A);cudaFree(B);cudaFree(T);
    }
    /* The two branches are supposed to be the SAME function of X -- both equal
     * U p(S^2) S V^T. Feeding one X and its transpose through the two paths and
     * checking the results are transposes of each other tests that claim
     * directly, which orthogonality alone does not: a branch that quietly
     * computed a different polynomial could still look orthogonal. */
    {   int m=48,n=80,batch=2;
        size_t nX=(size_t)batch*m*n, nA=(size_t)batch*m*m;
        float*Xh=(float*)malloc(nX*4),*Th=(float*)malloc(nX*4);
        for(size_t i=0;i<nX;i++) Xh[i]=rnd_normal();
        for(int b=0;b<batch;b++){                       /* scale, then transpose */
            double s=0; for(int i=0;i<m*n;i++){ float v=Xh[(size_t)b*m*n+i]; s+=(double)v*v; }
            float inv=(float)(1.0/(sqrt(s)+1e-12));
            for(int i=0;i<m*n;i++) Xh[(size_t)b*m*n+i]*=inv;
            for(int i=0;i<m;i++) for(int j=0;j<n;j++)
                Th[(size_t)b*m*n+(size_t)j*m+i]=Xh[(size_t)b*m*n+(size_t)i*n+j];
        }
        float*X=(float*)dmalloc(nX*4),*A=(float*)dmalloc(nA*4),
             *B=(float*)dmalloc(nA*4),*T=(float*)dmalloc(nX*4);
        int save_ns=g_muon_ns; g_muon_ns=5;
        ck(cudaMemcpy(X,Xh,nX*4,cudaMemcpyHostToDevice),"nsT X h2d");
        ns_batch(X,m,n,batch,A,B,T);                    /* m<n branch */
        float*R1=(float*)malloc(nX*4); ck(cudaMemcpy(R1,X,nX*4,cudaMemcpyDeviceToHost),"nsT r1");
        ck(cudaMemcpy(X,Th,nX*4,cudaMemcpyHostToDevice),"nsT XT h2d");
        ns_batch(X,n,m,batch,A,B,T);                    /* m>n branch, same data */
        g_muon_ns=save_ns;
        float*R2=(float*)malloc(nX*4); ck(cudaMemcpy(R2,X,nX*4,cudaMemcpyDeviceToHost),"nsT r2");
        float*R2t=(float*)malloc(nX*4);
        for(int b=0;b<batch;b++) for(int i=0;i<n;i++) for(int j=0;j<m;j++)
            R2t[(size_t)b*m*n+(size_t)j*n+i]=R2[(size_t)b*m*n+(size_t)i*m+j];
        ok &= check("muon ns  branches agree",relerr(R2t,R1,nX),1e-4f);
        free(Xh);free(Th);free(R1);free(R2);free(R2t);
        cudaFree(X);cudaFree(A);cudaFree(B);cudaFree(T);
    }
    /* ---- sparse engram ---- */
    ok &= eg_selftest();
    ok &= x1_selftest();
    ck(cudaGetLastError(),"selftest kernels");
    printf("%s\n", ok?"[selftest] ALL PASS":"[selftest] FAILURES ABOVE");
    if(!ok) exit(1);
}

#include "tui.cuh"

/* the folders cudalm looks in beside the exe (models, data root, bench tasks, TUI logs); mkdir is a no-op if present */
static void ensure_dirs(void){ static const char*sub[]={"models","data","bench","logs"}; char p[800];
    for(int i=0;i<4;i++){ snprintf(p,sizeof p,"%.700s" PSEP "%s",exe_dir(),sub[i]); lm_mkdir(p); } }
int main(int argc,char**argv){
    ensure_dirs();
    if(getenv("LMTUI_CHILD")) setvbuf(stdout,NULL,_IONBF,0);     /* the TUI pipes our output: keep it live */
    if(argc>2 && strcmp(argv[1],"new") && strcmp(argv[1],"bench") && strcmp(argv[1],"data")) side_apply(argv[2]);   /* before any env is read */
#ifdef _WIN32
    if(argc>1 && !strcmp(argv[1],"data")) return cmd_data(argc,argv);
#endif
    if(argc>3 && (!strcmp(argv[1],"train") || !strcmp(argv[1],"eval"))){ static char dd[1200]; argv[3]=(char*)data_resolve(argv[3],dd,sizeof dd); }
    /* CUDALM_SEED pins the batch order so two runs see identical data -- the only
     * way to compare loss curves between two numeric paths (fp32 vs fp16). */
    { const char*sd=getenv("CUDALM_SEED");
      if(sd) g_rng=(uint64_t)lm_strtou64(sd)|1ULL;
      else   g_rng^=(uint64_t)time(NULL); }
    { const char*d=getenv("CUDALM_DEBUG"); g_debug=(d && atoi(d)!=0);
      const char*p=getenv("CUDALM_PROF");  g_prof =p?atoi(p):0;   /* 2 = split fwd moe */
      const char*n=getenv("CUDALM_TILE"); if(n && atoi(n)!=0) g_tiled=1;
      const char*s=getenv("CUDALM_SUB");  if(s && atoi(s)!=0) g_sub=1;
      const char*L=getenv("CUDALM_LANE"); if(L) g_lane=(atoi(L)!=0);
      const char*hf=getenv("CUDALM_HFMA2"); if(hf) g_hfma2=(atoi(hf)!=0);
      const char*ks=getenv("CUDALM_DKV_SPLIT"); if(ks){ int v=atoi(ks); if(v==1||v==2||v==4) g_dkv_split=v; }
      const char*kt=getenv("CUDALM_DKV_TC"); if(kt) g_dkv_tc=(atoi(kt)!=0);
      const char*qt=getenv("CUDALM_DQ_TC"); if(qt) g_dq_tc=(atoi(qt)!=0);
      const char*fa_=getenv("CUDALM_FA"); if(fa_) g_fa=(atoi(fa_)!=0);
      const char*ft=getenv("CUDALM_FWD_TC"); if(ft) g_fwd_tc=(atoi(ft)!=0);
      const char*ms=getenv("CUDALM_MOE_STASH"); if(ms) g_moe_stash=(atoi(ms)!=0);
    { const char*mgs=getenv("CUDALM_MOE_GROUPED"); if(mgs) g_moe_grp=atoi(mgs); }
    { const char*e=getenv("CUDALM_MOE_NTSER"); if(e) g_moe_ntser=atoi(e); }
    { const char*e=getenv("CUDALM_MOE_NOFP8"); if(e) g_moe_nof8=atoi(e); }
    { const char*v=getenv("CUDALM_MOE_SB"); if(v) g_moe_sb=atoi(v);
      v=getenv("CUDALM_MOE_CAP"); if(v&&atof(v)>=1.0) g_moe_capf=(float)atof(v); }
{ const char*mf=getenv("CUDALM_MOE_F8"); if(mf) g_moef8=atoi(mf); }
    { const char*hw=getenv("CUDALM_HEAD_W8"); if(hw) g_headw8=atoi(hw); }
    { const char*h=getenv("CUDALM_H16"); if(h) g_h16=atoi(h); }
    { const char*h=getenv("CUDALM_HEAD_BF16"); if(h) g_hbf=atoi(h); }
    { const char*h=getenv("CUDALM_RMS_WARP"); if(h) g_rmsw=atoi(h); }
      { const char*tk=getenv("CUDALM_TOPK"); if(tk){ int v=atoi(tk); if(v>=1&&v<=8) g_topk=v; }
        const char*tl=getenv("CUDALM_TOPK_L"); if(tl){ int i=0; for(const char*q=tl;*q&&i<64;i++){ int v=atoi(q); g_topkL[i]=v; if(v>g_topk&&v<=8) g_topk=v;
            while(*q&&*q!=',')q++; if(*q)q++; } printf("[moe] per-layer top-k %s (max %d)\n",tl,g_topk); }
        const char*mb=getenv("CUDALM_MOE_BIAS"); if(mb){ float v=(float)atof(mb); g_mbias_u=v>0?v:0; }
        if(g_topk>1) printf("[moe] top-%d routing (CUDALM_TOPK) -- stash keeps one slice per pass\n",g_topk); }
      const char*fg=getenv("CUDALM_FP16_GEMM"); if(fg) g_fp16gemm=(atoi(fg)!=0);
      const char*f8=getenv("CUDALM_FP8_GEMM"); if(f8) g_fp8gemm=(atoi(f8)!=0); { const char*fm=getenv("CUDALM_FP8_MASK"); if(fm) g_fp8mask=atoi(fm); }
      { const char*fh=getenv("CUDALM_F8_HOLD"); if(fh) g_f8hold=atoi(fh); }
      { const char*mo=getenv("CUDALM_MOE_OVL"); if(mo) g_moe_ovl=atoi(mo); mo=getenv("CUDALM_TOP1_OLD"); if(mo) g_t1old=atoi(mo); mo=getenv("CUDALM_MOE_GPERM"); if(mo) g_gperm=atoi(mo); mo=getenv("CUDALM_FA_CP"); if(mo) g_facp=atoi(mo); mo=getenv("CUDALM_MOE_MERGE"); if(mo) g_moemrg=atoi(mo); }
      { const char*re=getenv("CUDALM_RESEPI"); if(re) g_resepi=atoi(re); }
      { const char*rh=getenv("CUDALM_ROPEH"); if(rh) g_ropeh=atoi(rh); }
#ifndef CUDALM_CPU
      { const char*bf=getenv("CUDALM_BFFN"); if(bf) g_bffn=atoi(bf); bf=getenv("CUDALM_BFMOE"); if(bf) g_bfmoe=atoi(bf); }
      { const char*v=getenv("CUDALM_F8_DELAY"); if(v) g_dscale=atoi(v); }
#endif
      const char*f8d=getenv("CUDALM_FP8_DBG"); if(f8d) g_fp8dbg=(atoi(f8d)!=0);
      const char*sc=getenv("CUDALM_SCHED"); if(sc) g_wsd=strcmp(sc,"cosine")!=0;
      const char*wd=getenv("CUDALM_WSD_DECAY");
      if(wd){ float v=(float)atof(wd); if(v>0.0f && v<1.0f) g_wsd_decay=v; }
      const char*rw=getenv("CUDALM_REWARMUP"); if(rw){ int v=atoi(rw); if(v>0) g_rewarm=v; }
      { const char*ma=getenv("CUDALM_MOE_AUX"); if(ma){ float v=(float)atof(ma); if(v>=0.0f) g_moe_aux=v; printf("[moe] aux loss %.3f (CUDALM_MOE_AUX)\n",g_moe_aux); } }
      { const char*v;
        if((v=getenv("CUDALM_MOE_BIAS_P"))) g_mbias_p=atoi(v);
        if((v=getenv("CUDALM_MOE_BIAS_SLOT"))) g_mbias_slot=atoi(v);
        if((v=getenv("CUDALM_MOE_RCAP"))&&atof(v)>=1.0) g_moe_rcap=(float)atof(v);
        if(g_moe_rcap>0&&g_moe_capf<g_moe_rcap+0.05f) g_moe_capf=g_moe_rcap+0.05f; /* SB pads each expert to 16 */
        if((v=getenv("CUDALM_MOE_SINK"))) g_moe_sink=atoi(v);
        if((v=getenv("CUDALM_MOE_AUX_L"))&&atof(v)>0) g_moe_auxL=(float)atof(v);
        if((v=getenv("CUDALM_ROUTER_Z"))) g_zl=(float)atof(v);
        if((v=getenv("CUDALM_CLIP_SIDE"))) g_clip_side=atoi(v)!=0;
        if((v=getenv("CUDALM_MOE_AUX_LM"))) g_auxlm_n=parse_flist(v,g_auxlm,64);
        if((v=getenv("CUDALM_MOE_BIAS_LM"))) g_biaslm_n=parse_flist(v,g_biaslm,64);
        if((v=getenv("CUDALM_ROUTER_PRE"))&&atoi(v)){ g_rpre=1; printf("[moe] router reads the pre-injection residual\n"); }
        if(g_auxlm_n||g_biaslm_n) printf("[moe] per-layer multipliers: aux %d set, bias %d set\n",g_auxlm_n,g_biaslm_n);
        if(g_topk>1)
            printf("[moe] aux %.3f bias %g balance: bias_p %d bias_slot %d rcap %.2f sink %d aux_l %.3f\n",g_moe_aux,g_mbias_u,g_mbias_p,g_mbias_slot,g_moe_rcap,g_moe_sink,g_moe_auxL); }
      { const char*rm=getenv("CUDALM_ROUTE"); int v=rm?atoi(rm):0;
        g_route_mode = (v<0||v>2) ? 0 : v; }   /* uploaded in cuda_init */
      const char*y=getenv("CUDALM_NOSYNC"); if(y) g_sync=atoi(y)==0;
      y=getenv("CUDALM_CH_ROWS"); if(y&&atoi(y)>=CH_ROWS) g_chrows=atoi(y)&~15;
      const char*f=getenv("CUDALM_FP16"); if(f) g_fp16=(atoi(f)!=0);
      const char*mu=getenv("CUDALM_MUON"); g_muon=mu?*mu!='0':1; g_muon_exp=mu!=NULL;
      const char*mm=getenv("CUDALM_MUON_MOM"); if(mm){ float v=(float)atof(mm); if(v>=0&&v<1) g_muon_mom=v; }
      const char*ml=getenv("CUDALM_MUON_LRMUL"); if(ml){ float v=(float)atof(ml); if(v>0) g_muon_lrmul=v; }
      const char*mns=getenv("CUDALM_MUON_NS"); if(mns){ int v=atoi(mns); if(v>=1&&v<=20) g_muon_ns=v; } }
#ifdef _WIN32
    if(argc<2 && _isatty(0) && _isatty(1)){ int r=tui_main(); if(r>=0) return r; }
#endif
    if(argc<2 || !strcmp(argv[1],"help")){ fprintf(stderr,
        "cudalm - CUDA trainer/runner for tinylm models (no arguments = interactive menu)\n"
        "  cudalm new      <name> <preset>\n"
        "  cudalm run      <name|model.bin> [\"prompt\"] [n] [temp] [topk]\n"
        "  cudalm serve    <name|model.bin> [port=8090]   (HTTP /completion, /logprob for external benches)\n"
        "  cudalm train    <name> <data_dir> [epochs]\n"
        "  cudalm eval     <name> <data_dir> [batches] [batch_size]\n"
        "  cudalm probe    <name> <prompts.txt> [ntok] [temp] [topk]\n"
        "  cudalm bench    <preset> [iters] [batch]\n"
        "  cudalm chatbench <name> <bench.txt> [ngen] [temp] [topp] [reppen] [nsamp] [T]\n"
        "  cudalm lmeval   <name> [hellaswag,arc_easy,arc_challenge,piqa,lambada,arithmark|all] [n] [Tmax]\n"
        "  cudalm lmeval-vs <A.items.tsv> <B.items.tsv>   paired per-item comparison of two lmeval dumps\n"
        "  cudalm lens     <name> <data_dir|\"text\"> [tokens=16384] [T=1024]   logit lens: every layer read out through the head\n"
        "  cudalm lens-tune <name> <data_dir> [steps=500] [tokens/step=8192] [T=1024]   train the tuned lens (<model>.tlens)\n"
        "  cudalm avg      <out> <in1> <in2> [...]   (checkpoint average / soup; eval + export)\n"
        "  cudalm ngfold   <out> <in> <rows> <data_dir> [tokens]   (fold the n-gram table to rows = R/f, usage-weighted)\n"
        "  cudalm kdcache  <teacher> <data_dir> <out.kd> [K=8] [B=8]   (teacher top-K for CUDALM_KD)\n"
        "  cudalm grow    <in> <out.bin> [--layers N] [--ffn F] [--noise x]\n"
        "  cudalm quantize <in> <out.bin> [q8|q4|q4n|fp16]\n"
        "  cudalm gguf     <name> [q4n|q4|q8|fp16|f32] [tinylm|llama] [out.gguf]\n"
        "  cudalm qlmcheck <name> [group]\n"
        "  cudalm decheck  <name> [n=200] [k=40]   (decoder vs full forward)\n"
        "  cudalm tokcheck <name> <tokens.u16> [maxdocs] [old]\n"
        "  cudalm info     <name>   (config + total / ngram-engram / active params)\n"
        "  cudalm selftest\n"
        "  cudalm data     <fetch|tokenizer|encode|mix|info|canon> ...  (tokenizer + corpus tools)\n"
        "presets: nano micro m1 m3 m6 m8 m16 m32 m70 m50 m64e m220e a8m a8m9 s8x s8w\n"); return argc<2; }
    if(!strcmp(argv[1],"new"))      { return cmd_new(argc,argv); }
    if(!strcmp(argv[1],"run"))      { cmd_run(argc,argv);   return 0; }
    if(!strcmp(argv[1],"serve"))    { cmd_serve(argc,argv); return 0; }
    if(!strcmp(argv[1],"train"))    { g_moe_train=1; cmd_train(argc,argv); return 0; }
    if(!strcmp(argv[1],"eval"))     { cmd_eval(argc,argv);  return 0; }
    if(!strcmp(argv[1],"probe"))    { cmd_probe(argc,argv); return 0; }
    if(!strcmp(argv[1],"chatbench")){ cmd_chatbench(argc,argv); return 0; }
    if(!strcmp(argv[1],"lmeval"))   { cmd_lmeval(argc,argv); return 0; }
    if(!strcmp(argv[1],"lens"))     { cmd_lens(argc,argv); return 0; }
    if(!strcmp(argv[1],"lens-tune")){ cmd_lens_tune(argc,argv); return 0; }
    if(!strcmp(argv[1],"lmeval-vs")){ cmd_lmeval_vs(argc,argv); return 0; }
    if(!strcmp(argv[1],"avg"))      { cmd_avg(argc,argv); return 0; }
    if(!strcmp(argv[1],"ngfold"))   { cmd_ngfold(argc,argv); return 0; }
    if(!strcmp(argv[1],"kdcache"))  { cmd_kdcache(argc,argv); return 0; }
    if(!strcmp(argv[1],"vrescheck")){ cmd_vrescheck(); return 0; }
    if(!strcmp(argv[1],"bench"))    { cmd_bench(argc,argv); return 0; }
    if(!strcmp(argv[1],"grow"))     { return cmd_grow(argc,argv); }
    if(!strcmp(argv[1],"quantize")) { return cmd_quantize(argc,argv); }
    if(!strcmp(argv[1],"gguf"))     { return cmd_gguf(argc,argv); }
    if(!strcmp(argv[1],"selftest")) { cmd_selftest();       return 0; }
    if(!strcmp(argv[1],"qlmcheck")) { return cmd_qlmcheck(argc,argv); }
    if(!strcmp(argv[1],"info"))     { return cmd_info(argc,argv); }
    if(!strcmp(argv[1],"tokcheck")) { return cmd_tokcheck(argc,argv); }
#ifdef _WIN32
    if(!strcmp(argv[1],"tuicheck")) { return tui_check(argc,argv); }
#endif
#ifndef CUDALM_CPU
    if(!strcmp(argv[1],"decheck"))  { return cmd_decheck(argc,argv); }
#else
    if(!strcmp(argv[1],"decheck"))  { fprintf(stderr,"decheck: GPU build only (it tests the KV-cache decoder; the CPU build recomputes the full forward)\n"); return 1; }
#endif
    fprintf(stderr,"unknown command '%s'\n",argv[1]); return 1;
}
