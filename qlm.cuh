/* ===================== qlm: all-integer training ==========================
 * CUDA port of the method validated on CPU in tinylm (see QLM_TRIAL.md).
 *
 * For every 2D weight (wq/wk/wv/wo and each expert w1/w3/w2) the persistent
 * state is integer only:
 *     d[i]  int8   digit in [-dmax,+dmax]   (dmax 7 = 4-bit, the headline)
 *     r[i]  int8   sub-digit residual, an error-feedback accumulator that
 *                  carries into d at +-64 -- the job fp32 master weights do
 *     s[g]  fp32   one scale per group of QLM_GROUP elements
 * P is a RENDERED VIEW of s*d, rewritten after each step and never itself
 * updated, so every cuBLAS GEMM downstream is unchanged. Straight-through
 * estimation means gpu_backward() needs no changes at all.
 *
 * Update is Integer AdamW: the per-group scale s converts Adam's real-valued
 * step into LSBs,  dr = -(lr*128/s)*m_hat/(sqrt(v_hat)+eps) - lr*wd*128*d.
 * Measured on CPU at m1: +0.038 nats of fp32 AdamW (two seeds).
 *
 * THREAD MAP: one block per scale group, one thread per element. Deliberately
 * no per-thread arrays -- `float acc[hd]` in local memory is what put this
 * project's attention kernel over the WDDM watchdog and cost two reboots.
 * Check `-Xptxas -v` for spill stores before running this unattended.
 */
#ifndef QLM_CUH
#define QLM_CUH

#define QLM_MAXSPAN 4096

/* One quantized span (a whole weight matrix). Groups are contiguous chunks of
 * `group` elements over the span, matching how cudalm's q8 already groups. */
typedef struct {
    unsigned long long pof;   /* offset into P                            */
    unsigned long long gof;   /* offset into G -- differs from pof only when
                               * the fused per-layer path compacts G          */
    unsigned long long qi0;   /* offset into d/r/m8/v8                    */
    unsigned long long n;     /* elements in the span                     */
    int   gi0, ngrp;          /* first group index, group count           */
    float sref;               /* reference scale, frozen at init (rel LR) */
} QSpanD;

typedef struct {
    int on, adam, dmax, group, lrmode, servo_every, packed;
    float fstar, f_hi, f_lo;
    int nspan;
    QSpanD *spanH, *spanD;
    unsigned long long nq, ngrp, np;
    signed char  *d, *r, *m8;
    unsigned char *v8;
    float *s, *m8s, *v8s;
    /* float spans (embeddings, norms, router, MTP) keep the stock bf16 AdamW.
     * fs_cs is their COMPACT offset: the full-size bf16 M/V are replaced by
     * arrays holding only these params, which at a real model size is ~3% of
     * them. Leaving M/V full-size while also carrying int8 m/v made qlm cost
     * MORE than stock -- 15.5 vs 12.0 B/param. */
    unsigned long long fs_off[QLM_MAXSPAN], fs_n[QLM_MAXSPAN], fs_cs[QLM_MAXSPAN];
    int nfs; unsigned long long nfloat;
    bf16 *Mc, *Vc;
    long long *statD; long long statH[4];   /* carry, rail, zero, |r| sum */
} Qlm;

static Qlm QQ;
static int  g_qlm_on = 0;
/* QLM_V4 (E1): v8 holds v^(1/4) instead of sqrt(v). Linear uint8 over sqrt(v)
 * spans only 255:1 of |g| per group, so small-gradient elements sit on the
 * floor; the fourth root spans 65025:1. qv_dec returns sqrt(v). */
__device__ int d_qv4 = 0;
__device__ __forceinline__ float qv_dec(float q, float vs){ float x=q*vs; return d_qv4 ? x*x : x; }
__device__ __forceinline__ float qv_enc(float rv){ return d_qv4 ? sqrtf(rv) : rv; }
static void qlm_v4_init(void){
    const char*e=getenv("QLM_V4"); int v=e&&atoi(e);
    ck(cudaMemcpyToSymbol(d_qv4,&v,sizeof(int)),"qv4");
    if(v) printf("[qlm] QLM_V4: second moment stored as v^(1/4)\n");
}

/* ---- stochastic rounding, identical to the CPU build ---------------------
 * Keyed on (index, step) rather than a per-thread RNG so it is deterministic,
 * reproducible across CPU and GPU, and needs no state. That determinism is
 * what makes an A/B at a pinned seed meaningful. */
__host__ __device__ __forceinline__ unsigned int qhashu(unsigned long long x){
    x += 0x9E3779B97F4A7C15ULL; x ^= x>>30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x>>27; x *= 0x94D049BB133111EBULL; x ^= x>>31;
    return (unsigned int)x;
}
__host__ __device__ __forceinline__ int qsri(float v, unsigned long long i, unsigned long long t){
    float f = floorf(v);
    unsigned int h = qhashu(i*0x2545F4914F6CDD1DULL + t*0x9E3779B97F4A7C15ULL);
    return (int)f + (((h>>8)*(1.0f/16777216.0f) < (v-f)) ? 1 : 0);
}
__host__ __device__ __forceinline__ int qclmp(int v,int lo,int hi){ return v<lo?lo:(v>hi?hi:v); }

/* ---- A0: digits packed two per byte -------------------------------------
 * dmax<=7 fits a signed nibble, so 4-bit/3-bit/ternary all store two digits
 * per byte. int8 digits (dmax=127) cannot pack and fall back to one per byte.
 * WRITES ARE THE CATCH: two threads sharing a byte would race, so the step
 * kernel stages digits in shared memory and one thread per BYTE writes the
 * packed pair (see k_qlm_step). Reads are race-free and unpack in place. */
__host__ __device__ __forceinline__ int qd_get(const signed char*d, unsigned long long i, int packed){
    if(!packed) return (int)d[i];
    unsigned char b=(unsigned char)d[i>>1];
    int v = (i&1ULL) ? (b>>4) : (b&0xF);
    return v>=8 ? v-16 : v;                   /* sign-extend the nibble */
}
__host__ __device__ __forceinline__ void qd_set_host(signed char*d, unsigned long long i, int v, int packed){
    if(!packed){ d[i]=(signed char)v; return; }
    unsigned char*b=(unsigned char*)&d[i>>1];
    unsigned char nib=(unsigned char)(v&0xF);
    *b = (i&1ULL) ? (unsigned char)((*b & 0x0F) | (nib<<4))
                  : (unsigned char)((*b & 0xF0) | nib);
}
static unsigned long long qd_bytes(unsigned long long nq,int packed){
    return packed ? (nq+1)/2 : nq;
}

/* locate the span owning group gi (few dozen spans, linear scan is free) */
__device__ __forceinline__ int qspan_of(const QSpanD*sp,int nspan,int gi){
    for(int i=0;i<nspan;i++) if(gi>=sp[i].gi0 && gi<sp[i].gi0+sp[i].ngrp) return i;
    return 0;
}

/* Commit a group's digits from a shared-memory staging array. With packing on,
 * ONE thread owns each byte, which is what makes the nibble writes race-free.
 * qi0 is always a multiple of the group size, and the group size is even, so a
 * group never starts mid-byte and groups never share a byte. */
__device__ __forceinline__ void qd_write_group(signed char*d, unsigned long long qi0,
                                               const signed char*shd, int n, int j, int packed){
    if(!packed){ if(j<n) d[qi0+j]=shd[j]; return; }
    int nb=(n+1)>>1;
    if(j<nb){
        int lo=shd[2*j] & 0xF;
        int hi=(2*j+1<n) ? (shd[2*j+1] & 0xF) : 0;
        d[(qi0>>1)+j]=(signed char)((hi<<4)|lo);
    }
}

/* ---- init: quantize P in place into (d,r,s) ---------------------------- */
__global__ void k_qlm_encode(const float*P, signed char*d, signed char*r, float*s,
                             const QSpanD*sp, int nspan, int group, int dmax, int packed){
    int gi = blockIdx.x;
    int si = qspan_of(sp,nspan,gi);
    unsigned long long j0 = (unsigned long long)(gi - sp[si].gi0)*group;
    int n = (int)min((unsigned long long)group, sp[si].n - j0);
    unsigned long long pof = sp[si].pof + j0, qi0 = sp[si].qi0 + j0;
    int j = threadIdx.x;
    /* group absmax -> scale = absmax/dmax (ternary uses absmean; 4-bit is the
     * headline config so absmax is what matters here) */
    float a = (j<n) ? fabsf(P[pof+j]) : 0.0f;
    __shared__ float sh[4];
    for(int o=16;o>0;o>>=1) a = fmaxf(a, __shfl_down_sync(0xffffffff,a,o));
    if((j&31)==0) sh[j>>5]=a;
    __syncthreads();
    if(j==0){
        float m=sh[0]; int nw=(blockDim.x+31)/32;
        for(int i=1;i<nw;i++) m=fmaxf(m,sh[i]);
        float sg = m/(float)dmax; if(!(sg>0.0f)) sg=1e-8f;
        s[gi]=sg;
    }
    __syncthreads();
    extern __shared__ signed char shd[];
    if(j<n){
        float sg=s[gi], t=P[pof+j]/sg;
        int dd=qclmp((int)rintf(t),-dmax,dmax);
        int rr=qclmp((int)rintf((t-(float)dd)*128.0f),-64,64);
        shd[j]=(signed char)dd; r[qi0+j]=(signed char)rr;
    } else if(j<blockDim.x) shd[j]=0;
    __syncthreads();
    qd_write_group(d,qi0,shd,n,j,packed);
}

/* ---- render: P = s*d  (the only way P is ever written) ----------------- */
__global__ void k_qlm_render(float*P, const signed char*d, const float*s,
                             const QSpanD*sp, int nspan, int group, int packed, int gi_base){
    int gi = gi_base + blockIdx.x;
    int si = qspan_of(sp,nspan,gi);
    unsigned long long j0 = (unsigned long long)(gi - sp[si].gi0)*group;
    int n = (int)min((unsigned long long)group, sp[si].n - j0);
    int j = threadIdx.x; if(j>=n) return;
    P[sp[si].pof + j0 + j] = s[gi]*(float)qd_get(d, sp[si].qi0 + j0 + j, packed);
}

/* ---- Integer AdamW + render, one block per group ----------------------- */
__global__ void k_qlm_step(float*P, const float*G,
                           signed char*d, signed char*r, float*s,
                           signed char*m8, float*m8s,
                           unsigned char*v8, float*v8s,
                           const QSpanD*sp, int nspan, int group, int dmax,
                           float lr, float b1, float b2, float wd, float gscale,
                           float inv_bc1, float inv_sqrt_bc2, int lrmode,
                           unsigned long long t, int adam, long long*stat, int packed,
                           int gi_base){
    int gi = gi_base + blockIdx.x;
    int si = qspan_of(sp,nspan,gi);
    unsigned long long j0 = (unsigned long long)(gi - sp[si].gi0)*group;
    int n = (int)min((unsigned long long)group, sp[si].n - j0);
    int j = threadIdx.x;
    unsigned long long pof = sp[si].pof + j0, qi0 = sp[si].qi0 + j0;

    extern __shared__ signed char shd[];
    if(j<blockDim.x) shd[j]=0;
    float sg = s[gi];
    /* `rel` (default): a fixed number of LSBs per tensor, so digits move at a
     * uniform rate and the effective fp LR follows each group's own scale. */
    float L = lr*128.0f/(lrmode ? sg : sp[si].sref);
    float dec = lr*wd*128.0f;
    float ms_old = m8s[gi], vs_old = v8s ? v8s[gi] : 0.0f;

    float mn=0.0f, rv=0.0f;
    if(j<n){
        unsigned long long qi=qi0+j, k=pof+j;
        float gr = G[sp[si].gof + j0 + j]*gscale;
        float mv = (float)m8[qi]*ms_old;
        float u;
        if(adam){
            float sv=qv_dec((float)v8[qi],vs_old), vv=sv*sv;
            mn = b1*mv + (1.0f-b1)*gr;
            vv = b2*vv + (1.0f-b2)*gr*gr;
            rv = sqrtf(vv);                       /* ONE sqrt: sqrt(vv/bc2) = rv*inv_sqrt_bc2 */
            /* eps floored at one quantisation step of sqrt(v): an absolute
             * 1e-8 is meaningless against a uint8-quantised second moment */
            float veps = fmaxf(qv_dec(1.0f,vs_old), 1e-8f);
            u  = (mn*inv_bc1)/(rv*inv_sqrt_bc2 + veps);
            /* belt and braces: a real Adam step is |u| ~ 1, never 1e6. Without
             * this a single bad element overflows qsri() to INT_MIN. */
            u  = fmaxf(-8.0f, fminf(8.0f, u));
            rv = qv_enc(rv);
        } else {
            float cc = b1*mv + (1.0f-b1)*gr;
            u  = (float)((cc>0.0f)-(cc<0.0f));
            mn = b2*mv + (1.0f-b2)*gr;
        }
        int dd = qd_get(d,qi,packed);
        int idl = qsri(-(L*u + dec*(float)dd), qi, t);
        int rr  = (int)r[qi] + idl;
        /* exact multi-carry, no loop and no clamp: the unique c with
         * rr-128c in [-64,64], matching while(rr>64){rr-=128;} at the boundary
         * (rr==64 must NOT carry). Steps here run ~48 LSBs, so a single-carry
         * assumption is NOT safe -- that bug cost a wrong CPU result. */
        int c = (rr > 64) ?  (((rr - 65) >> 7) + 1)
              : (rr < -64) ? -((((-rr) - 65) >> 7) + 1) : 0;
        rr -= c*128; dd += c;
        dd = qclmp(dd,-dmax,dmax);
        shd[j]=(signed char)dd; r[qi]=(signed char)rr;
        P[k] = sg*(float)dd;
        if(stat){
            /* max |idl| tells us directly whether the digit step is exploding,
             * instead of inferring it from an accumulated carry count */
            atomicMax((unsigned long long*)&stat[3], (unsigned long long)(idl<0?-idl:idl));
            atomicAdd((unsigned long long*)&stat[0], (unsigned long long)(c<0?-c:c));
            atomicAdd((unsigned long long*)&stat[1], (unsigned long long)((dd==dmax)|(dd==-dmax)));
            atomicAdd((unsigned long long*)&stat[2], (unsigned long long)(dd==0));
        }
    }
    __syncthreads();
    qd_write_group(d,qi0,shd,n,j,packed);
    /* block-reduce the new moment magnitudes, then requantize them */
    __shared__ float sha[4], shv[4];
    float am = (j<n)? fabsf(mn) : 0.0f, vm = (j<n)? rv : 0.0f;
    for(int o=16;o>0;o>>=1){
        am = fmaxf(am, __shfl_down_sync(0xffffffff,am,o));
        vm = fmaxf(vm, __shfl_down_sync(0xffffffff,vm,o));
    }
    if((j&31)==0){ sha[j>>5]=am; shv[j>>5]=vm; }
    __syncthreads();
    if(j==0){
        int nw=(blockDim.x+31)/32; float a=sha[0], v=shv[0];
        for(int i=1;i<nw;i++){ a=fmaxf(a,sha[i]); v=fmaxf(v,shv[i]); }
        m8s[gi] = a>0.0f ? a/127.0f : 1e-12f;
        if(adam) v8s[gi] = v>0.0f ? v/255.0f : 1e-12f;
    }
    __syncthreads();
    if(j<n){
        unsigned long long qi=qi0+j;
        m8[qi] = (signed char)qclmp(qsri(mn/m8s[gi], qi, t+0x5EEDULL), -127,127);
        if(adam){
            int vq = qclmp(qsri(rv/v8s[gi], qi, t+0xBEEFULL), 0,255);
            if(vq==0 && rv>0.0f) vq=1;      /* keep v alive; 0 is absorbing */
            v8[qi] = (unsigned char)vq;
        }
    }
}

/* ---- S1: same math as k_qlm_step, one WARP per group (8 per block) --------
 * Element j = k*32+lane keeps every load coalesced; no shared memory and no
 * __syncthreads. Packed nibbles: the even lane pulls its odd neighbour's digit
 * by shuffle and writes the byte. Reductions are fmax (order-free) and qsri is
 * keyed on (qi,t), so the result is bit-identical to k_qlm_step.
 * gsp != NULL: gscale is read from the device (S2: no host round-trip). */
template<int E>
__global__ void __launch_bounds__(256) k_qlm_stepw(float*P, const float*G,
                           signed char*d, signed char*r, float*s,
                           signed char*m8, float*m8s,
                           unsigned char*v8, float*v8s,
                           const QSpanD*sp, int nspan, int dmax,
                           float lr, float b1, float b2, float wd, float gscale, const float*gsp,
                           float inv_bc1, float inv_sqrt_bc2, int lrmode,
                           unsigned long long t, int adam, long long*stat, int packed,
                           int gi_base, int ng){
    const int group=E*32, lane=threadIdx.x&31;
    int w=blockIdx.x*8+(threadIdx.x>>5); if(w>=ng) return;
    int gi = gi_base + w;
    int si = qspan_of(sp,nspan,gi);
    unsigned long long j0 = (unsigned long long)(gi - sp[si].gi0)*group;
    int n = (int)min((unsigned long long)group, sp[si].n - j0);
    unsigned long long pof = sp[si].pof + j0, qi0 = sp[si].qi0 + j0, gof = sp[si].gof + j0;
    float gs = gsp ? *gsp : gscale;
    float sg = s[gi];
    float L = lr*128.0f/(lrmode ? sg : sp[si].sref);
    float dec = lr*wd*128.0f;
    float ms_old = m8s[gi], vs_old = v8s ? v8s[gi] : 0.0f;
    float mn[E], rv[E], am=0.0f, vm=0.0f;
#pragma unroll
    for(int k=0;k<E;k++){
        int j=k*32+lane; mn[k]=0.0f; rv[k]=0.0f; int dd=0;
        if(j<n){
            unsigned long long qi=qi0+j;
            float gr = G[gof + j]*gs;
            float mv = (float)m8[qi]*ms_old, u;
            if(adam){
                float sv=qv_dec((float)v8[qi],vs_old), vv=sv*sv;
                mn[k] = b1*mv + (1.0f-b1)*gr;
                vv = b2*vv + (1.0f-b2)*gr*gr;
                rv[k] = sqrtf(vv);
                float veps = fmaxf(qv_dec(1.0f,vs_old), 1e-8f);
                u = (mn[k]*inv_bc1)/(rv[k]*inv_sqrt_bc2 + veps);
                u = fmaxf(-8.0f, fminf(8.0f, u));
                rv[k] = qv_enc(rv[k]);
            } else {
                float cc = b1*mv + (1.0f-b1)*gr;
                u = (float)((cc>0.0f)-(cc<0.0f));
                mn[k] = b2*mv + (1.0f-b2)*gr;
            }
            dd = qd_get(d,qi,packed);
            int idl = qsri(-(L*u + dec*(float)dd), qi, t);
            int rr  = (int)r[qi] + idl;
            int c = (rr > 64) ?  (((rr - 65) >> 7) + 1)
                  : (rr < -64) ? -((((-rr) - 65) >> 7) + 1) : 0;
            rr -= c*128; dd += c;
            dd = qclmp(dd,-dmax,dmax);
            r[qi]=(signed char)rr;
            P[pof+j] = sg*(float)dd;
            if(!packed) d[qi]=(signed char)dd;
            if(stat){
                atomicMax((unsigned long long*)&stat[3], (unsigned long long)(idl<0?-idl:idl));
                atomicAdd((unsigned long long*)&stat[0], (unsigned long long)(c<0?-c:c));
                atomicAdd((unsigned long long*)&stat[1], (unsigned long long)((dd==dmax)|(dd==-dmax)));
                atomicAdd((unsigned long long*)&stat[2], (unsigned long long)(dd==0));
            }
            am=fmaxf(am,fabsf(mn[k])); vm=fmaxf(vm,rv[k]);
        }
        if(packed){
            int hi=__shfl_down_sync(0xffffffff,dd,1);
            if(!(lane&1) && j<n) d[(qi0+j)>>1]=(signed char)(((hi&0xF)<<4)|(dd&0xF));
        }
    }
    for(int o=16;o>0;o>>=1){
        am = fmaxf(am, __shfl_xor_sync(0xffffffff,am,o));
        vm = fmaxf(vm, __shfl_xor_sync(0xffffffff,vm,o));
    }
    float ms = am>0.0f ? am/127.0f : 1e-12f, vs = vm>0.0f ? vm/255.0f : 1e-12f;
    if(lane==0){ m8s[gi]=ms; if(adam) v8s[gi]=vs; }
#pragma unroll
    for(int k=0;k<E;k++){
        int j=k*32+lane; if(j>=n) break;
        unsigned long long qi=qi0+j;
        m8[qi] = (signed char)qclmp(qsri(mn[k]/ms, qi, t+0x5EEDULL), -127,127);
        if(adam){
            int vq = qclmp(qsri(rv[k]/vs, qi, t+0xBEEFULL), 0,255);
            if(vq==0 && rv[k]>0.0f) vq=1;
            v8[qi] = (unsigned char)vq;
        }
    }
}
/* Dispatch: warp kernel for group 64/128/256, else the block kernel. */
static void qlm_step_launch(float*P, const float*G, signed char*d, signed char*r, float*s,
                            signed char*m8, float*m8s, unsigned char*v8, float*v8s,
                            const QSpanD*sp, int nspan, int group, int dmax,
                            float lr, float b1, float b2, float wd, float gscale, const float*gsp,
                            float inv_bc1, float inv_sqrt_bc2, int lrmode,
                            unsigned long long t, int adam, long long*stat, int packed,
                            int gi_base, int ng){
    static int old=-1; if(old<0){ const char*e=getenv("QLM_STEP_OLD"); old=e&&atoi(e); }
    unsigned nb=(unsigned)((ng+7)/8);
#define QSW(E_) k_qlm_stepw<E_><<<nb,256>>>(P,G,d,r,s,m8,m8s,v8,v8s,sp,nspan,dmax,lr,b1,b2,wd,gscale,gsp,\
        inv_bc1,inv_sqrt_bc2,lrmode,t,adam,stat,packed,gi_base,ng)
    if(!old && group==128) QSW(4);
    else if(!old && group==64) QSW(2);
    else if(!old && group==256) QSW(8);
    else {
        if(gsp){ float h; cudaMemcpy(&h,gsp,4,cudaMemcpyDeviceToHost); gscale=h; }
        k_qlm_step<<<(unsigned)ng,group,group>>>(P,G,d,r,s,m8,m8s,v8,v8s,sp,nspan,group,dmax,
            lr,b1,b2,wd,gscale,inv_bc1,inv_sqrt_bc2,lrmode,t,adam,stat,packed,gi_base);
    }
#undef QSW
}

/* ---- scale servo: with no latent weights there is nothing to recompute
 * mean|W| from, so each group's scale is steered by how often it rails.
 * Measured on CPU to be NOT load-bearing at 4-bit -- kept because it costs
 * nothing and guards against drift over a long run. */
__global__ void k_qlm_servo(signed char*d, signed char*r, float*s,
                            const QSpanD*sp, int nspan, int group, int dmax,
                            float f_hi, float f_lo, int packed){
    int gi = blockIdx.x;
    int si = qspan_of(sp,nspan,gi);
    unsigned long long j0 = (unsigned long long)(gi - sp[si].gi0)*group;
    int n = (int)min((unsigned long long)group, sp[si].n - j0);
    unsigned long long qi0 = sp[si].qi0 + j0;
    int j = threadIdx.x;
    int rail = 0;
    if(j<n){ int dd=qd_get(d,qi0+j,packed); rail = (dd==dmax)|(dd==-dmax); }
    for(int o=16;o>0;o>>=1) rail += __shfl_down_sync(0xffffffff,rail,o);
    __shared__ int sh[4]; if((j&31)==0) sh[j>>5]=rail;
    __syncthreads();
    __shared__ float conv; __shared__ int act;
    if(j==0){
        int nw=(blockDim.x+31)/32, tot=sh[0];
        for(int i=1;i<nw;i++) tot+=sh[i];
        float f=(float)tot/(float)n, sg=s[gi], sn=sg;
        act=0;
        if(f>f_hi){ sn=sg*1.09051f; act=1; }        /* 2^(1/8) */
        else if(f<f_lo){ sn=sg/1.09051f; act=1; }
        conv = sg/sn;
        if(act) s[gi]=sn;
    }
    __syncthreads();
    extern __shared__ signed char shd2[];
    if(j<blockDim.x) shd2[j]=0;
    if(act && j<n){
        unsigned long long qi=qi0+j;
        float t=((float)qd_get(d,qi,packed) + (float)r[qi]*(1.0f/128.0f))*conv;
        int dd=qclmp((int)rintf(t),-dmax,dmax);
        int rr=qclmp((int)rintf((t-(float)dd)*128.0f),-64,64);
        shd2[j]=(signed char)dd; r[qi]=(signed char)rr;
    }
    __syncthreads();
    if(act) qd_write_group(d,qi0,shd2,n,j,packed);
}

/* ---- host side --------------------------------------------------------- */
static void qlm_add_span(Qlm*q, unsigned long long pof, unsigned long long n){
    if(q->nspan>=QLM_MAXSPAN) return;
    QSpanD*sp=&q->spanH[q->nspan];
    sp->pof=pof; sp->gof=pof; sp->qi0=q->nq; sp->n=n;
    sp->gi0=(int)q->ngrp; sp->ngrp=(int)((n + q->group - 1)/q->group);
    sp->sref=1.0f;
    q->nq += n; q->ngrp += sp->ngrp; q->nspan++;
}

/* build the span table, allocate, quantize P on the device */
static void qlm_gpu_init(Qlm*q, const Cfg*c, const Off*o, float*P, size_t np){
    memset(q,0,sizeof(*q));
    const char*e;
    int bits = (e=getenv("QLM_BITS")) ? atoi(e) : 4;
    q->dmax  = bits<=2?1 : bits==3?3 : bits==4?7 : 127;
    q->group = (e=getenv("QLM_GROUP")) ? atoi(e) : 128;
    if(q->group<32) q->group=32; if(q->group>256) q->group=256;
    q->lrmode = (e=getenv("QLM_LRMODE")) ? (!strcmp(e,"abs")?1:0) : 0;
    q->servo_every = (e=getenv("QLM_SERVO")) ? atoi(e) : 100;
    q->adam = 1;
    qlm_v4_init();
    { const char*op=getenv("QLM_OPT"); if(op && !strcmp(op,"qlion")) q->adam=0; }
    /* init uses absmax/dmax, which rails exactly one element per group, so the
     * servo target has to be ~1/group or it shrinks every scale on tick one */
    q->fstar = (e=getenv("QLM_FSTAR")) ? (float)atof(e)
             : (q->dmax==1 ? 0.60f : 1.5f/(float)q->group);
    q->f_hi=q->fstar*1.5f; q->f_lo=q->fstar*0.5f;
    q->np = np;

    q->spanH=(QSpanD*)calloc(QLM_MAXSPAN,sizeof(QSpanD));
    int E = c->n_exp>0 ? c->n_exp : 1;
    for(int l=0;l<c->L;l++){
        qlm_add_span(q,o->wq[l],(unsigned long long)c->D*c->D);
        qlm_add_span(q,o->wk[l],(unsigned long long)c->D*c->KD);
        qlm_add_span(q,o->wv[l],(unsigned long long)c->D*c->KD);
        qlm_add_span(q,o->wo[l],(unsigned long long)c->D*c->D);
        if(!c->pool){                                 /* UniPool: pool span added once, last (spans ascend) */
        qlm_add_span(q,o->w1[l],(unsigned long long)E*c->D*c->F);
        qlm_add_span(q,o->w3[l],(unsigned long long)E*c->D*c->F);
        qlm_add_span(q,o->w2[l],(unsigned long long)E*c->F*c->D); }
    }
    if(c->pool){ qlm_add_span(q,o->w1[0],(unsigned long long)3*E*c->D*c->F); }
    /* float spans = the gaps (embeddings, both norms, router, final norm, MTP);
     * they keep the stock bf16 AdamW so only the representation is under test */
    { unsigned long long cur=0;
      for(int i=0;i<q->nspan;i++){
        if(q->spanH[i].pof>cur){ q->fs_off[q->nfs]=cur; q->fs_n[q->nfs]=q->spanH[i].pof-cur; q->nfs++; }
        cur=q->spanH[i].pof+q->spanH[i].n;
      }
      if(cur<np){ q->fs_off[q->nfs]=cur; q->fs_n[q->nfs]=np-cur; q->nfs++; }
    }
    { unsigned long long acc=0;
      for(int i=0;i<q->nfs;i++){ q->fs_cs[i]=acc; acc+=q->fs_n[i]; }
      q->nfloat=acc; }

    q->packed = (q->dmax<=7);
    q->Mc=(bf16*)dmalloc(q->nfloat*2); q->Vc=(bf16*)dmalloc(q->nfloat*2);
    ck(cudaMemset(q->Mc,0,q->nfloat*2),"qlm Mc zero");
    ck(cudaMemset(q->Vc,0,q->nfloat*2),"qlm Vc zero");
    q->d  =(signed char*)dmalloc(qd_bytes(q->nq,q->packed));
    q->r  =(signed char*)dmalloc(q->nq);
    q->m8 =(signed char*)dmalloc(q->nq);
    q->v8 = q->adam ? (unsigned char*)dmalloc(q->nq) : NULL;
    ck(cudaMemset(q->m8,0,q->nq),"qlm m8 zero");
    if(q->adam) ck(cudaMemset(q->v8,0,q->nq),"qlm v8 zero");
    q->s  =(float*)dmalloc(q->ngrp*4);
    q->m8s=(float*)dmalloc(q->ngrp*4);
    q->v8s= q->adam ? (float*)dmalloc(q->ngrp*4) : NULL;
    ck(cudaMemset(q->m8s,0,q->ngrp*4),"qlm m8s zero");
    if(q->adam) ck(cudaMemset(q->v8s,0,q->ngrp*4),"qlm v8s zero");
    q->statD=(long long*)dmalloc(4*8);
    q->spanD=(QSpanD*)dmalloc((size_t)q->nspan*sizeof(QSpanD));
    ck(cudaMemcpy(q->spanD,q->spanH,(size_t)q->nspan*sizeof(QSpanD),cudaMemcpyHostToDevice),"qlm span h2d");

    cudaDeviceSynchronize(); g_tprev=wall();
    k_qlm_encode<<<(unsigned)q->ngrp,q->group,q->group>>>(P,q->d,q->r,q->s,q->spanD,q->nspan,q->group,q->dmax,q->packed);
    kchk("qlm encode",-1);

    /* freeze each span's reference scale (mean of its group scales at init) */
    { float*sh=(float*)malloc(q->ngrp*4);
      ck(cudaMemcpy(sh,q->s,q->ngrp*4,cudaMemcpyDeviceToHost),"qlm s d2h");
      for(int i=0;i<q->nspan;i++){
        double acc=0; for(int g=0;g<q->spanH[i].ngrp;g++) acc+=sh[q->spanH[i].gi0+g];
        q->spanH[i].sref=(float)(acc/(double)q->spanH[i].ngrp);
      }
      free(sh);
      ck(cudaMemcpy(q->spanD,q->spanH,(size_t)q->nspan*sizeof(QSpanD),cudaMemcpyHostToDevice),"qlm span h2d 2");
    }
    cudaDeviceSynchronize(); g_tprev=wall();
    k_qlm_render<<<(unsigned)q->ngrp,q->group>>>(P,q->d,q->s,q->spanD,q->nspan,q->group,q->packed,0);
    kchk("qlm render",-1);
    q->on=1;
}

/* absolute P offset -> index into the compact float-span moment arrays */
static unsigned long long qlm_fs_cs(const Qlm*q, unsigned long long off){
    for(int i=0;i<q->nfs;i++)
        if(off>=q->fs_off[i] && off<q->fs_off[i]+q->fs_n[i])
            return q->fs_cs[i] + (off - q->fs_off[i]);
    return 0;
}

/* Point each span's gradient read at the COMPACT G layout. Without this the
 * step kernel reads gradients at the absolute offset while the fused backward
 * wrote them at the aliased per-layer offset, so every layer but the first
 * trains on the wrong memory (observed as loss 6.39 vs 4.93). */
static void qlm_set_gof(Qlm*q, const Cfg*c, const Off*go){
    if(getenv("CUDALM_QLM_NOALIAS")) return;   /* gof already == pof */
    for(int l=0;l<c->L;l++){
        int s0=7*l;
        q->spanH[s0+0].gof=go->wq[l]; q->spanH[s0+1].gof=go->wk[l];
        q->spanH[s0+2].gof=go->wv[l]; q->spanH[s0+3].gof=go->wo[l];
        q->spanH[s0+4].gof=go->w1[l]; q->spanH[s0+5].gof=go->w3[l];
        q->spanH[s0+6].gof=go->w2[l];
    }
    if(getenv("CUDALM_QLM_DIAG")){
        const char*nm[7]={"wq","wk","wv","wo","w1","w3","w2"};
        for(int l=0;l<2 && l<c->L;l++) for(int k=0;k<7;k++){
            const QSpanD*sp=&q->spanH[7*l+k];
            printf("[map] l=%d %-2s pof=%llu gof=%llu n=%llu\n", l, nm[k],
                   (unsigned long long)sp->pof,(unsigned long long)sp->gof,
                   (unsigned long long)sp->n);
        }
    }
    ck(cudaMemcpy(q->spanD,q->spanH,(size_t)q->nspan*sizeof(QSpanD),
                  cudaMemcpyHostToDevice),"qlm gof h2d");
}

static void qlm_gpu_render(Qlm*q, float*P){
    k_qlm_render<<<(unsigned)q->ngrp,q->group>>>(P,q->d,q->s,q->spanD,q->nspan,q->group,q->packed,0);
    kchk("qlm render",-1);
}

/* ---- Phase 2: per-layer P scratch helpers --------------------------------
 * With the scratch path on, P holds only the float spans plus ONE shared
 * layer's quantized weights. Each span's pof already points into that shared
 * block; a layer is rendered from its digits just before it is read. */
static void qlm_render_layer(Qlm*q, float*P, int l){
    int s0=7*l, gi0=q->spanH[s0].gi0, ng=0;
    for(int i=s0;i<s0+7;i++) ng+=q->spanH[i].ngrp;
    k_qlm_render<<<(unsigned)ng,q->group>>>(P,q->d,q->s,q->spanD,q->nspan,q->group,q->packed,gi0);
    kchk("qlm render layer",l);
}

/* Repoint each quantized span's P offset at the compact shared block. pof is
 * used by render (write P) and the step kernel (write P[k]); after this every
 * layer aliases the same block, rendered just before it is read. Call AFTER
 * qlm_gpu_init has encoded the digits from the full real-layout P. */
static void qlm_set_pof(Qlm*q, const Cfg*c, const Off*po){
    for(int l=0;l<c->L;l++){
        int s0=7*l;
        q->spanH[s0+0].pof=po->wq[l]; q->spanH[s0+1].pof=po->wk[l];
        q->spanH[s0+2].pof=po->wv[l]; q->spanH[s0+3].pof=po->wo[l];
        q->spanH[s0+4].pof=po->w1[l]; q->spanH[s0+5].pof=po->w3[l];
        q->spanH[s0+6].pof=po->w2[l];
    }
    ck(cudaMemcpy(q->spanD,q->spanH,(size_t)q->nspan*sizeof(QSpanD),
                  cudaMemcpyHostToDevice),"qlm pof h2d");
}

/* Rebuild the float-span moment map for the compact P layout. Same set of
 * float params (so nfloat and Mc/Vc are unchanged), relocated to po-> offsets;
 * only fs_off/fs_cs change. */
static void qlm_remap_fs(Qlm*q, const Cfg*c, const Off*po){
    unsigned long long acc=0; int k=0;
    #define QFS(off,nn) do{ q->fs_off[k]=(off); q->fs_n[k]=(unsigned long long)(nn); \
                            q->fs_cs[k]=acc; acc+=(unsigned long long)(nn); k++; }while(0)
    QFS(po->emb,(unsigned long long)c->V*c->D);
    for(int l=0;l<c->L;l++){
        QFS(po->an1[l],(unsigned long long)c->D);
        QFS(po->an2[l],(unsigned long long)c->D);
        if(c->n_exp>0) QFS(po->wr[l],(unsigned long long)c->D*c->n_exp);
    }
    QFS(po->nf,(unsigned long long)c->D);
    for(int kk=0;kk<c->n_mtp;kk++) QFS(po->mtp[kk],(unsigned long long)c->D*c->D);
    #undef QFS
    q->nfs=k;
    if(acc!=q->nfloat) fprintf(stderr,"[qlm] remap_fs nfloat %llu vs %llu\n",acc,q->nfloat);
}

/* Copy the float spans from a full real-layout P (src) into the compact P
 * (dst) at their po-> offsets. Device to device. */
static void qlm_scatter_float(const Cfg*c, const Off*o, const Off*po,
                              const float*src, float*dst){
    #define QCP(so,dof,nn) ck(cudaMemcpy(dst+(dof),src+(so),(size_t)(nn)*4,\
                              cudaMemcpyDeviceToDevice),"qlm scatter")
    QCP(o->emb,po->emb,(size_t)c->V*c->D);
    for(int l=0;l<c->L;l++){
        QCP(o->an1[l],po->an1[l],c->D);
        QCP(o->an2[l],po->an2[l],c->D);
        if(c->n_exp>0) QCP(o->wr[l],po->wr[l],(size_t)c->D*c->n_exp);
    }
    QCP(o->nf,po->nf,c->D);
    for(int k=0;k<c->n_mtp;k++) QCP(o->mtp[k],po->mtp[k],(size_t)c->D*c->D);
    #undef QCP
}

/* Reconstruct a full real-layout host weight buffer from the compact P: float
 * spans copied straight out, quantized weights rendered layer-by-layer. Used
 * only when saving a model checkpoint under the scratch path. */
static void qlm_gather_full(Qlm*q, const Cfg*c, const Off*o, const Off*po,
                            float*P, float*Ph){
    #define QG(dof,so,nn) ck(cudaMemcpy(Ph+(so),P+(dof),(size_t)(nn)*4,\
                             cudaMemcpyDeviceToHost),"qlm gather f")
    QG(po->emb,o->emb,(size_t)c->V*c->D);
    for(int l=0;l<c->L;l++){
        QG(po->an1[l],o->an1[l],c->D);
        QG(po->an2[l],o->an2[l],c->D);
        if(c->n_exp>0) QG(po->wr[l],o->wr[l],(size_t)c->D*c->n_exp);
    }
    QG(po->nf,o->nf,c->D);
    for(int k=0;k<c->n_mtp;k++) QG(po->mtp[k],o->mtp[k],(size_t)c->D*c->D);
    #undef QG
    for(int l=0;l<c->L;l++){
        qlm_render_layer(q,P,l);
        int s0=7*l;
        size_t so[7]={o->wq[l],o->wk[l],o->wv[l],o->wo[l],o->w1[l],o->w3[l],o->w2[l]};
        for(int k=0;k<7;k++){
            const QSpanD*sp=&q->spanH[s0+k];
            ck(cudaMemcpy(Ph+so[k],P+sp->pof,(size_t)sp->n*4,
                          cudaMemcpyDeviceToHost),"qlm gather q");
        }
    }
    ck(cudaDeviceSynchronize(),"qlm gather sync");
}

/* Bytes of PERSISTENT integer state -- the digits, residual, moments and
 * scales. This is NOT comparable to opt_state_bytes(), which also counts the
 * fp32 P and G. Reporting one against the other overstates the saving ~4x. */
static double qlm_persist_bytes(const Qlm*q){
    return (double)qd_bytes(q->nq,q->packed)   /* digits, as ALLOCATED */
         + (double)q->nq*1.0                   /* residual */
         + (double)q->nq*(q->adam?2.0:1.0)     /* int8 m (+ uint8 v) */
         + (double)q->ngrp*4.0*(q->adam?3.0:2.0);  /* s, m8s (, v8s) */
}

/* Total training state, on the SAME basis as opt_state_bytes(): includes the
 * rendered fp32 P (which every cuBLAS GEMM reads) and the fp32 gradient.
 * In this stage those two dominate at 8 B/param, so the headline saving is
 * small -- the digits only pay off once P goes fp16 and G is consumed
 * per-layer inside the backward pass. */
static double qlm_total_bytes(const Qlm*q){
    return qlm_persist_bytes(q)
         + (double)q->np*4.0                /* rendered P, fp32 */
         + (double)q->np*4.0                /* gradient, fp32 */
         + (double)q->nfloat*4.0;           /* COMPACT bf16 m + v, float spans only */
}
static double qlm_state_bytes(const Qlm*q){ return qlm_total_bytes(q); }

/* ---- durable checkpoint of the integer state ---------------------------
 * Same write-tmp -> commit -> atomic-replace discipline the rest of this
 * project uses: an unclean shutdown once left a full-length checkpoint whose
 * tail was zeros, because fclose() only flushes to the OS, not to disk. */
static int qlm_save(const Qlm*q, const char*path, int iter){
    char tmp[1300]; snprintf(tmp,sizeof tmp,"%s.tmp",path);
    FILE*f=fopen(tmp,"wb"); if(!f) return 0;
    int hdr[6]={q->dmax,q->group,q->adam,q->nspan,q->packed,iter};
    unsigned long long nq=q->nq, ng=q->ngrp, nf=q->nfloat;
    unsigned long long nd=qd_bytes(nq,q->packed);
    signed char *hd=(signed char*)malloc(nd), *hr=(signed char*)malloc(nq),
                *hm=(signed char*)malloc(nq);
    unsigned char *hv= q->adam ? (unsigned char*)malloc(nq) : NULL;
    float *hs=(float*)malloc(ng*4), *hms=(float*)malloc(ng*4), *hvs= q->adam ? (float*)malloc(ng*4) : NULL;
    bf16 *hM=(bf16*)malloc(nf*2), *hV=(bf16*)malloc(nf*2);
    cudaMemcpy(hM,q->Mc,nf*2,cudaMemcpyDeviceToHost);
    cudaMemcpy(hV,q->Vc,nf*2,cudaMemcpyDeviceToHost);
    cudaMemcpy(hd,q->d,nd,cudaMemcpyDeviceToHost);
    cudaMemcpy(hr,q->r,nq,cudaMemcpyDeviceToHost);
    cudaMemcpy(hm,q->m8,nq,cudaMemcpyDeviceToHost);
    if(q->adam) cudaMemcpy(hv,q->v8,nq,cudaMemcpyDeviceToHost);
    cudaMemcpy(hs,q->s,ng*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(hms,q->m8s,ng*4,cudaMemcpyDeviceToHost);
    if(q->adam) cudaMemcpy(hvs,q->v8s,ng*4,cudaMemcpyDeviceToHost);
    int ok = fwrite("QLMH",1,4,f)==4 && fwrite(hdr,4,6,f)==6
          && fwrite(&nq,8,1,f)==1 && fwrite(&ng,8,1,f)==1 && fwrite(&nf,8,1,f)==1
          && fwrite(hM,2,nf,f)==nf && fwrite(hV,2,nf,f)==nf
          && fwrite(hd,1,nd,f)==nd && fwrite(hr,1,nq,f)==nq
          && fwrite(hm,1,nq,f)==nq && (q->adam ? fwrite(hv,1,nq,f)==nq : 1)
          && fwrite(hs,4,ng,f)==ng && fwrite(hms,4,ng,f)==ng && (q->adam ? fwrite(hvs,4,ng,f)==ng : 1);
    if(ok){ fflush(f); _commit(_fileno(f)); }
    fclose(f);
    free(hd);free(hr);free(hm);free(hv);free(hs);free(hms);free(hvs);free(hM);free(hV);
    if(!ok){ remove(tmp); return 0; }
    return MoveFileExA(tmp,path,MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)?1:0;
}

static int qlm_load(Qlm*q, const char*path, int*iter){
    FILE*f=fopen(path,"rb"); if(!f) return 0;
    char mg[4]; int hdr[6]; unsigned long long nq,ng,nf;
    if(fread(mg,1,4,f)!=4 || memcmp(mg,"QLMH",4)){ fclose(f); return 0; }
    if(fread(hdr,4,6,f)!=6 || fread(&nq,8,1,f)!=1 || fread(&ng,8,1,f)!=1
       || fread(&nf,8,1,f)!=1){ fclose(f); return 0; }
    if(hdr[0]!=q->dmax || hdr[1]!=q->group || hdr[2]!=q->adam || hdr[4]!=q->packed
       || nq!=q->nq || ng!=q->ngrp || nf!=q->nfloat){
        fprintf(stderr,"[qlm] %s has a different shape (bits/group/opt) -- ignoring\n",path);
        fclose(f); return 0;
    }
    if(iter) *iter=hdr[5];
    unsigned long long nd=qd_bytes(nq,q->packed);
    signed char *hd=(signed char*)malloc(nd), *hr=(signed char*)malloc(nq),
                *hm=(signed char*)malloc(nq);
    unsigned char *hv= q->adam ? (unsigned char*)malloc(nq) : NULL;
    float *hs=(float*)malloc(ng*4), *hms=(float*)malloc(ng*4), *hvs= q->adam ? (float*)malloc(ng*4) : NULL;
    bf16 *hM=(bf16*)malloc(nf*2), *hV=(bf16*)malloc(nf*2);
    int ok = fread(hM,2,nf,f)==nf && fread(hV,2,nf,f)==nf
          && fread(hd,1,nd,f)==nd && fread(hr,1,nq,f)==nq
          && fread(hm,1,nq,f)==nq && (q->adam ? fread(hv,1,nq,f)==nq : 1)
          && fread(hs,4,ng,f)==ng && fread(hms,4,ng,f)==ng && (q->adam ? fread(hvs,4,ng,f)==ng : 1);
    fclose(f);
    if(ok){
        cudaMemcpy(q->Mc,hM,nf*2,cudaMemcpyHostToDevice);
        cudaMemcpy(q->Vc,hV,nf*2,cudaMemcpyHostToDevice);
        cudaMemcpy(q->d,hd,nd,cudaMemcpyHostToDevice);
        cudaMemcpy(q->r,hr,nq,cudaMemcpyHostToDevice);
        cudaMemcpy(q->m8,hm,nq,cudaMemcpyHostToDevice);
        if(q->adam) cudaMemcpy(q->v8,hv,nq,cudaMemcpyHostToDevice);
        cudaMemcpy(q->s,hs,ng*4,cudaMemcpyHostToDevice);
        cudaMemcpy(q->m8s,hms,ng*4,cudaMemcpyHostToDevice);
        if(q->adam) cudaMemcpy(q->v8s,hvs,ng*4,cudaMemcpyHostToDevice);
    }
    free(hd);free(hr);free(hm);free(hv);free(hs);free(hms);free(hvs);free(hM);free(hV);
    return ok;
}

static void qlm_stats(Qlm*q){
    long long h[4]={0,0,0,0};
    cudaMemcpy(h,q->statD,4*8,cudaMemcpyDeviceToHost);
    double nq=(double)q->nq;
    printf("[qlm] carry %.2f%%/step | rail %.1f%% | zero %.1f%%\n",
           100.0*(double)h[0]/nq, 100.0*(double)h[1]/nq, 100.0*(double)h[2]/nq);
}
#endif
