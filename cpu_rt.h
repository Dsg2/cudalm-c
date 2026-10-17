/* cpu_rt.h - CUDA runtime + cuBLAS shim for the CPU build of cudalm (-DCUDALM_CPU).
 *
 * "Device" memory is host memory, every kernel launch KL(k,grid,block[,shm])(args)
 * calls the same-named OpenMP/AVX2 function from cpu_kern.h, and all cuBLAS SGEMM
 * variants land in one packed GEMM (6x16 AVX2+FMA micro-kernel, OpenMP over tiles).
 * GPU-only features (FP8/cublasLt, fp16 GEMM, tensor-core attention) report
 * CUBLAS_STATUS_NOT_SUPPORTED; main() switches them off before they are reached.
 *
 * Build (MinGW-w64 g++):
 *   g++ -O3 -march=haswell -fopenmp -DCUDALM_CPU -x c++ cudalm.cu -o cudalm_cpu.exe -ladvapi32
 * -march=x86-64 (no AVX2) also builds: every AVX2 path has a scalar fallback. */
#ifndef CPU_RT_H
#define CPU_RT_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <omp.h>
#ifdef __AVX2__
  #include <immintrin.h>
#endif
#ifdef _WIN32
  #include <windows.h>
  #include <malloc.h>
  #include <intrin.h>
  #define cpu_amalloc(n) _aligned_malloc((n)?(n):64,64)
  #define cpu_afree(p)   _aligned_free(p)
#else
  #include <cpuid.h>
  #include <unistd.h>
  static inline void*cpu_amalloc(size_t n){ void*p=NULL; return posix_memalign(&p,64,n?n:64)?NULL:p; }
  #define cpu_afree(p)   free(p)
#endif

/* ---- CUDA qualifiers / builtin types ---- */
#define __global__
#define __device__
#define __host__
#define __constant__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __align__(n) __attribute__((aligned(n)))
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
struct float2 { float x,y; };
struct __align__(16) float4 { float x,y,z,w; };
struct __align__(8)  uint2 { unsigned x,y; };
struct __align__(16) uint4 { unsigned x,y,z,w; };
static inline float2 make_float2(float x,float y){ float2 r={x,y}; return r; }
static inline float4 make_float4(float x,float y,float z,float w){ float4 r={x,y,z,w}; return r; }
static inline uint2 make_uint2(unsigned x,unsigned y){ uint2 r={x,y}; return r; }
static inline uint4 make_uint4(unsigned x,unsigned y,unsigned z,unsigned w){ uint4 r={x,y,z,w}; return r; }
static inline unsigned __float_as_uint(float f){ unsigned u; memcpy(&u,&f,4); return u; }
static inline int      __float_as_int(float f){ int u; memcpy(&u,&f,4); return u; }
static inline float    __uint_as_float(unsigned u){ float f; memcpy(&f,&u,4); return f; }

/* fp16 / E4M3 storage types: host code only moves them around, so plain
 * round-to-nearest-even conversions are all that is needed */
static inline unsigned short cpu_f2h(float f){
    unsigned x=__float_as_uint(f), s=(x>>16)&0x8000u; int e=(int)((x>>23)&0xff)-127+15; unsigned m=x&0x7fffffu;
    if(((x>>23)&0xff)==0xff) return (unsigned short)(s|0x7c00u|(m?0x200u:0));
    if(e>=31) return (unsigned short)(s|0x7c00u);
    if(e<=0){ if(e<-10) return (unsigned short)s; m|=0x800000u; int sh=14-e; unsigned h=m>>sh, r=m&((1u<<sh)-1), hf=1u<<(sh-1);
              if(r>hf||(r==hf&&(h&1))) h++; return (unsigned short)(s|h); }
    unsigned h=((unsigned)e<<10)|(m>>13), r=m&0x1fffu; if(r>0x1000u||(r==0x1000u&&(h&1))) h++;
    return (unsigned short)(s|h);
}
static inline float cpu_h2f(unsigned short h){
    unsigned s=(unsigned)(h&0x8000u)<<16, e=(h>>10)&0x1f, m=h&0x3ffu;
    if(!e){ if(!m) return __uint_as_float(s); float f=(float)m*(1.0f/16777216.0f); return s?-f:f; }
    if(e==31) return __uint_as_float(s|0x7f800000u|(m<<13));
    return __uint_as_float(s|((e+112)<<23)|(m<<13));
}
struct __half { unsigned short x; __half(){} __half(float f):x(cpu_f2h(f)){} operator float() const { return cpu_h2f(x); } };
struct __half2 { __half x,y; };
static inline __half __float2half(float f){ return __half(f); }
static inline float  __half2float(__half h){ return (float)h; }
struct __nv_fp8_e4m3 {
    unsigned char __x;
    __nv_fp8_e4m3(){}
    __nv_fp8_e4m3(float f){                       /* saturating RNE, max 448 */
        unsigned s=f<0?0x80u:0; float a=fabsf(f);
        if(!(a==a)){ __x=0x7f; return; } if(a>448.0f) a=448.0f;
        if(a<0.0009765625f){ __x=(unsigned char)(s|(unsigned)lrintf(a*512.0f)); return; }   /* subnormal, step 2^-9 */
        int e; float m=frexpf(a,&e); int q=(int)lrintf(m*16.0f); if(q==16){ q=8; e++; }
        int be=e-1+7; if(be<1){ __x=(unsigned char)(s|(unsigned)lrintf(a*512.0f)); return; }
        __x=(unsigned char)(s|((unsigned)be<<3)|(unsigned)(q-8)); }
    operator float() const { int e=(__x>>3)&15, m=__x&7; float v=e?ldexpf(1.0f+m/8.0f,e-7):ldexpf((float)m,-9); return (__x&0x80)?-v:v; }
};

/* ---- runtime API ---- */
typedef enum { cudaSuccess=0, cudaErrorMemoryAllocation=2, cudaErrorNotSupported=801 } cudaError_t;
typedef enum { cudaMemcpyHostToHost=0, cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2, cudaMemcpyDeviceToDevice=3, cudaMemcpyDefault=4 } cudaMemcpyKind;
typedef void* cudaStream_t;
typedef int   cudaEvent_t;
#define cudaEventDisableTiming 2
typedef struct { char name[256]; int major,minor,multiProcessorCount; size_t totalGlobalMem; } cudaDeviceProp;
typedef enum { CUDA_R_32F=0, CUDA_R_16F=2, CUDA_R_16BF=14, CUDA_R_8F_E4M3=28 } cudaDataType_t;
static inline const char* cudaGetErrorString(cudaError_t e){
    return e==cudaSuccess?"no error":e==cudaErrorMemoryAllocation?"out of memory (host RAM)":"not supported on the CPU backend"; }
static inline cudaError_t cudaGetLastError(void){ return cudaSuccess; }
static inline cudaError_t cudaDeviceSynchronize(void){ return cudaSuccess; }
static inline cudaError_t cudaStreamSynchronize(cudaStream_t){ return cudaSuccess; }
static inline cudaError_t cudaSetDevice(int){ return cudaSuccess; }
static inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t*e,unsigned){ *e=0; return cudaSuccess; }
static inline cudaError_t cudaEventRecord(cudaEvent_t,cudaStream_t=0){ return cudaSuccess; }
static inline cudaError_t cudaEventSynchronize(cudaEvent_t){ return cudaSuccess; }
static inline cudaError_t cudaEventCreate(cudaEvent_t*e){ *e=0; return cudaSuccess; }
static inline cudaError_t cudaEventQuery(cudaEvent_t){ return cudaSuccess; }
static inline cudaError_t cudaEventElapsedTime(float*ms,cudaEvent_t,cudaEvent_t){ *ms=0.f; return cudaSuccess; }
static inline cudaError_t cudaEventDestroy(cudaEvent_t){ return cudaSuccess; }
/* big copies/fills split across threads: a cold 100 MB memset is ~4x faster */
static __attribute__((noinline)) void cpu_memcpy(void*d,const void*s,size_t n){
    if(n<(4u<<20)||d==s){ if(d!=s) memmove(d,s,n); return; }
    if((const char*)d<(const char*)s+n && (const char*)s<(const char*)d+n){ memmove(d,s,n); return; }
    long nb=(long)((n+(1u<<20)-1)>>20);
    #pragma omp parallel for schedule(static)
    for(long b=0;b<nb;b++){ size_t o=(size_t)b<<20, k=n-o<(1u<<20)?n-o:(1u<<20); memcpy((char*)d+o,(const char*)s+o,k); }
}
static inline void cpu_memset(void*d,int v,size_t n){
    if(n<(4u<<20)){ memset(d,v,n); return; }
    long nb=(long)((n+(1u<<20)-1)>>20);
    #pragma omp parallel for schedule(static)
    for(long b=0;b<nb;b++){ size_t o=(size_t)b<<20, k=n-o<(1u<<20)?n-o:(1u<<20); memset((char*)d+o,v,k); }
}
template<class T> static inline cudaError_t cudaMalloc(T**p,size_t n){ *p=(T*)cpu_amalloc(n); return *p?cudaSuccess:cudaErrorMemoryAllocation; }
template<class T> static inline cudaError_t cudaHostAlloc(T**p,size_t n,unsigned){ return cudaMalloc(p,n); }
static inline cudaError_t cudaFree(void*p){ if(p) cpu_afree(p); return cudaSuccess; }
static inline cudaError_t cudaFreeHost(void*p){ return cudaFree(p); }
static inline cudaError_t cudaMemcpy(void*d,const void*s,size_t n,cudaMemcpyKind){ cpu_memcpy(d,s,n); return cudaSuccess; }
static inline cudaError_t cudaMemcpyAsync(void*d,const void*s,size_t n,cudaMemcpyKind,cudaStream_t=0){ cpu_memcpy(d,s,n); return cudaSuccess; }
static inline cudaError_t cudaMemset(void*d,int v,size_t n){ cpu_memset(d,v,n); return cudaSuccess; }
static inline cudaError_t cudaMemsetAsync(void*d,int v,size_t n,cudaStream_t=0){ cpu_memset(d,v,n); return cudaSuccess; }
template<class T> static inline cudaError_t cudaMemcpyToSymbol(T&sym,const void*s,size_t n,size_t off=0,cudaMemcpyKind=cudaMemcpyHostToDevice){
    memcpy((char*)&sym+off,s,n); return cudaSuccess; }
static inline cudaError_t cudaMemGetInfo(size_t*fr,size_t*tt){
#ifdef _WIN32
    MEMORYSTATUSEX m; m.dwLength=sizeof m; GlobalMemoryStatusEx(&m); *fr=(size_t)m.ullAvailPhys; *tt=(size_t)m.ullTotalPhys;
#else
    long pg=sysconf(_SC_PAGESIZE); *fr=(size_t)sysconf(_SC_AVPHYS_PAGES)*pg; *tt=(size_t)sysconf(_SC_PHYS_PAGES)*pg;
#endif
    return cudaSuccess; }
static inline int cpu_has_avx2(void){
    unsigned r[4];
#ifdef _WIN32
    int c[4]; __cpuidex(c,7,0); memcpy(r,c,16);
#else
    __cpuid_count(7,0,r[0],r[1],r[2],r[3]);
#endif
    return (r[1]>>5)&1; }
static inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp*p,int){
    memset(p,0,sizeof *p); unsigned b[12]; int c[4];
#ifdef _WIN32
    for(int i=0;i<3;i++){ __cpuid(c,(int)(0x80000002u+i)); memcpy(b+4*i,c,16); }
#else
    for(int i=0;i<3;i++){ __cpuid(0x80000002u+i,b[4*i],b[4*i+1],b[4*i+2],b[4*i+3]); } (void)c;
#endif
    char nm[49]; memcpy(nm,b,48); nm[48]=0; const char*s=nm; while(*s==' ') s++;
    snprintf(p->name,sizeof p->name,"CPU %s [%s, %d threads]",s,
#ifdef __AVX2__
             "AVX2+FMA",
#else
             "scalar/SSE",
#endif
             omp_get_max_threads());
    p->multiProcessorCount=omp_get_max_threads();
    size_t fr,tt; cudaMemGetInfo(&fr,&tt); p->totalGlobalMem=tt;
    return cudaSuccess; }

/* ---- kernel launch: KL(k,grid,block[,shm[,stream]])(args) ----
 * The grid is kept because a few kernels size their work from it
 * (one block per row with no row-count argument). */
static dim3 g_kgrid, g_kblk;
static inline int cpu_cfg(dim3 g,dim3 b,size_t=0,cudaStream_t=0){ g_kgrid=g; g_kblk=b; return 1; }
#define KL(k,...) for(int kl_=cpu_cfg(__VA_ARGS__);kl_;kl_=0) k

/* ---- SGEMM ------------------------------------------------------------------
 * Row-major core: O[R,C] = alpha * X[R,K] @ Y[K,C] + beta*O, where X/Y are read
 * through arbitrary (row,col) strides so all four transpose cases share one path.
 * A column-major cuBLAS call C(m,n)=op(A)op(B) is the row-major problem
 * C^T(n,m) = op(B)^T op(A)^T, which is how the wrappers below map onto it.
 * beta==0 never reads O (cuBLAS semantics). */
#define SG_MR 6
#define SG_NR 16
#define SG_KC 256
#define SG_MC 96
#define SG_NB 256
typedef ptrdiff_t sgi;
static inline void sg_store(float*o,sgi ldo,const float*acc,int mr,int nr,float alpha,float beta,int first){
    for(int i=0;i<mr;i++){ float*r=o+(sgi)i*ldo; const float*a=acc+i*SG_NR;
        if(!first)       for(int j=0;j<nr;j++) r[j]+=alpha*a[j];
        else if(beta==0) for(int j=0;j<nr;j++) r[j]=alpha*a[j];
        else             for(int j=0;j<nr;j++) r[j]=beta*r[j]+alpha*a[j]; }
}
/* acc[6][16] = sum_p a[p][0..5] (x) b[p][0..15] over kc packed steps */
static inline void sg_ukr(int kc,const float*a,const float*b,float*o,sgi ldo,int mr,int nr,float alpha,float beta,int first){
#ifdef __AVX2__
    __m256 c00=_mm256_setzero_ps(),c01=c00,c10=c00,c11=c00,c20=c00,c21=c00,c30=c00,c31=c00,c40=c00,c41=c00,c50=c00,c51=c00;
    for(int p=0;p<kc;p++,a+=SG_MR,b+=SG_NR){
        __m256 b0=_mm256_load_ps(b), b1=_mm256_load_ps(b+8), x;
        x=_mm256_broadcast_ss(a+0); c00=_mm256_fmadd_ps(x,b0,c00); c01=_mm256_fmadd_ps(x,b1,c01);
        x=_mm256_broadcast_ss(a+1); c10=_mm256_fmadd_ps(x,b0,c10); c11=_mm256_fmadd_ps(x,b1,c11);
        x=_mm256_broadcast_ss(a+2); c20=_mm256_fmadd_ps(x,b0,c20); c21=_mm256_fmadd_ps(x,b1,c21);
        x=_mm256_broadcast_ss(a+3); c30=_mm256_fmadd_ps(x,b0,c30); c31=_mm256_fmadd_ps(x,b1,c31);
        x=_mm256_broadcast_ss(a+4); c40=_mm256_fmadd_ps(x,b0,c40); c41=_mm256_fmadd_ps(x,b1,c41);
        x=_mm256_broadcast_ss(a+5); c50=_mm256_fmadd_ps(x,b0,c50); c51=_mm256_fmadd_ps(x,b1,c51);
    }
    if(mr==SG_MR && nr==SG_NR){
        __m256 al=_mm256_set1_ps(alpha), be=_mm256_set1_ps(beta);
        #define SG_ST(i,v0,v1) { float*r=o+(sgi)(i)*ldo; \
            if(!first){ _mm256_storeu_ps(r,_mm256_fmadd_ps(al,v0,_mm256_loadu_ps(r))); _mm256_storeu_ps(r+8,_mm256_fmadd_ps(al,v1,_mm256_loadu_ps(r+8))); } \
            else if(beta==0){ _mm256_storeu_ps(r,_mm256_mul_ps(al,v0)); _mm256_storeu_ps(r+8,_mm256_mul_ps(al,v1)); } \
            else { _mm256_storeu_ps(r,_mm256_fmadd_ps(al,v0,_mm256_mul_ps(be,_mm256_loadu_ps(r)))); \
                   _mm256_storeu_ps(r+8,_mm256_fmadd_ps(al,v1,_mm256_mul_ps(be,_mm256_loadu_ps(r+8)))); } }
        SG_ST(0,c00,c01) SG_ST(1,c10,c11) SG_ST(2,c20,c21) SG_ST(3,c30,c31) SG_ST(4,c40,c41) SG_ST(5,c50,c51)
        #undef SG_ST
        return;
    }
    float __attribute__((aligned(32))) acc[SG_MR*SG_NR];
    _mm256_store_ps(acc+0,c00); _mm256_store_ps(acc+8,c01); _mm256_store_ps(acc+16,c10); _mm256_store_ps(acc+24,c11);
    _mm256_store_ps(acc+32,c20); _mm256_store_ps(acc+40,c21); _mm256_store_ps(acc+48,c30); _mm256_store_ps(acc+56,c31);
    _mm256_store_ps(acc+64,c40); _mm256_store_ps(acc+72,c41); _mm256_store_ps(acc+80,c50); _mm256_store_ps(acc+88,c51);
    sg_store(o,ldo,acc,mr,nr,alpha,beta,first);
#else
    float acc[SG_MR*SG_NR]={0};
    for(int p=0;p<kc;p++,a+=SG_MR,b+=SG_NR)
        for(int i=0;i<SG_MR;i++){ float x=a[i]; for(int j=0;j<SG_NR;j++) acc[i*SG_NR+j]+=x*b[j]; }
    sg_store(o,ldo,acc,mr,nr,alpha,beta,first);
#endif
}
/* small-R path (inference lm_head row, tiny batches): no packing, stream Y once */
static void sg_small(int R,int C,int K,float alpha,const float*X,sgi xr,sgi xc,const float*Y,sgi yr,sgi yc,
                     float beta,float*O,sgi ldo){
    float xs[4][1024]; int kk=K<=1024;
    if(kk) for(int r=0;r<R;r++) for(int p=0;p<K;p++) xs[r][p]=X[r*xr+p*xc];
    #pragma omp parallel for schedule(static)
    for(int c0=0;c0<C;c0+=64){
        int c1=c0+64<C?c0+64:C; float acc[4][64];
        for(int r=0;r<R;r++) for(int c=c0;c<c1;c++) acc[r][c-c0]=0;
        if(yr==1 && kk){                     /* Y columns contiguous along p: dot products */
            for(int c=c0;c<c1;c++){ const float*y=Y+(sgi)c*yc;
                for(int r=0;r<R;r++){ const float*x=xs[r]; float s=0;
                    #pragma omp simd reduction(+:s)
                    for(int p=0;p<K;p++) s+=x[p]*y[p];
                    acc[r][c-c0]=s; } }
        } else {                             /* axpy over rows of Y */
            for(int p=0;p<K;p++){ const float*y=Y+(sgi)p*yr;
                for(int r=0;r<R;r++){ float x=kk?xs[r][p]:X[r*xr+p*xc]; float*a=acc[r];
                    if(yc==1){
                        #pragma omp simd
                        for(int c=c0;c<c1;c++) a[c-c0]+=x*y[c];
                    } else for(int c=c0;c<c1;c++) a[c-c0]+=x*y[(sgi)c*yc]; } }
        }
        for(int r=0;r<R;r++){ float*o=O+(sgi)r*ldo;
            for(int c=c0;c<c1;c++) o[c]= beta==0 ? alpha*acc[r][c-c0] : beta*o[c]+alpha*acc[r][c-c0]; }
    }
}
struct SgBuf { float*a,*b; size_t ca,cb; };
static SgBuf* sg_buf(void){ static thread_local SgBuf t={0,0,0,0}; return &t; }
static float* sg_grow(float**p,size_t*cap,size_t n){
    if(n>*cap){ cpu_afree(*p); *p=(float*)cpu_amalloc(n*4); *cap=n; if(!*p){ fprintf(stderr,"[cpu] sgemm scratch OOM\n"); exit(1); } }
    return *p; }
static void sg_core(int R,int C,int K,float alpha,const float*X,sgi xr,sgi xc,const float*Y,sgi yr,sgi yc,
                    float beta,float*O,sgi ldo){
    if(R<=0||C<=0) return;
    if(K<=0||alpha==0){ for(int r=0;r<R;r++){ float*o=O+(sgi)r*ldo; for(int c=0;c<C;c++) o[c]=beta==0?0.f:beta*o[c]; } return; }
    if(R<=4){ sg_small(R,C,K,alpha,X,xr,xc,Y,yr,yc,beta,O,ldo); return; }
    int nmp=(R+SG_MR-1)/SG_MR, nnp=(C+SG_NR-1)/SG_NR;
    SgBuf*sb=sg_buf();
    float*Ap=sg_grow(&sb->a,&sb->ca,(size_t)nmp*SG_MR*SG_KC);
    float*Bp=sg_grow(&sb->b,&sb->cb,(size_t)nnp*SG_NR*SG_KC);
    int nic=(R+SG_MC-1)/SG_MC, njb=(C+SG_NB-1)/SG_NB, ntile=nic*njb;
    for(int pc=0;pc<K;pc+=SG_KC){
        int kc=K-pc<SG_KC?K-pc:SG_KC, first=(pc==0);
        #pragma omp parallel
        {
            #pragma omp for schedule(static) nowait
            for(int ip=0;ip<nmp;ip++){            /* pack X: [panel][p][6] */
                float*d=Ap+(size_t)ip*SG_MR*kc; int r0=ip*SG_MR, mr=R-r0<SG_MR?R-r0:SG_MR;
                const float*x=X+(sgi)r0*xr+(sgi)pc*xc;
                if(xc==1){ for(int i=0;i<mr;i++){ const float*xi=x+(sgi)i*xr; for(int p=0;p<kc;p++) d[p*SG_MR+i]=xi[p]; } }
                else for(int p=0;p<kc;p++){ const float*xp=x+(sgi)p*xc; for(int i=0;i<mr;i++) d[p*SG_MR+i]=xp[(sgi)i*xr]; }
                if(mr<SG_MR) for(int p=0;p<kc;p++) for(int i=mr;i<SG_MR;i++) d[p*SG_MR+i]=0; }
            #pragma omp for schedule(static)
            for(int jp=0;jp<nnp;jp++){            /* pack Y: [panel][p][16] */
                float*d=Bp+(size_t)jp*SG_NR*kc; int c0=jp*SG_NR, nr=C-c0<SG_NR?C-c0:SG_NR;
                const float*y=Y+(sgi)pc*yr+(sgi)c0*yc;
                if(yc==1 && nr==SG_NR){ for(int p=0;p<kc;p++) memcpy(d+p*SG_NR,y+(sgi)p*yr,SG_NR*4); }
                else if(yr==1){ for(int j=0;j<nr;j++){ const float*yj=y+(sgi)j*yc; for(int p=0;p<kc;p++) d[p*SG_NR+j]=yj[p]; }
                                if(nr<SG_NR) for(int p=0;p<kc;p++) for(int j=nr;j<SG_NR;j++) d[p*SG_NR+j]=0; }
                else for(int p=0;p<kc;p++) for(int j=0;j<SG_NR;j++) d[p*SG_NR+j]= j<nr ? y[(sgi)p*yr+(sgi)j*yc] : 0.f; }
            #pragma omp for schedule(dynamic,1)
            for(int t=0;t<ntile;t++){
                int jb=t/nic, ic=t-jb*nic;
                int r0=ic*SG_MC, r1=r0+SG_MC<R?r0+SG_MC:R, c0=jb*SG_NB, c1=c0+SG_NB<C?c0+SG_NB:C;
                for(int c=c0;c<c1;c+=SG_NR){ const float*bp=Bp+(size_t)(c/SG_NR)*SG_NR*kc; int nr=c1-c<SG_NR?c1-c:SG_NR;
                    for(int r=r0;r<r1;r+=SG_MR){ int mr=r1-r<SG_MR?r1-r:SG_MR;
                        sg_ukr(kc,Ap+(size_t)(r/SG_MR)*SG_MR*kc,bp,O+(sgi)r*ldo+c,ldo,mr,nr,alpha,beta,first); } }
            }
        }
    }
}

/* ---- cuBLAS ---- */
typedef int* cublasHandle_t;
typedef enum { CUBLAS_STATUS_SUCCESS=0, CUBLAS_STATUS_NOT_SUPPORTED=15 } cublasStatus_t;
typedef enum { CUBLAS_OP_N=0, CUBLAS_OP_T=1 } cublasOperation_t;
typedef enum { CUBLAS_DEFAULT_MATH=0, CUBLAS_TF32_TENSOR_OP_MATH=3 } cublasMath_t;
typedef enum { CUBLAS_COMPUTE_16F=64, CUBLAS_COMPUTE_32F=68, CUBLAS_COMPUTE_32F_FAST_TF32=77 } cublasComputeType_t;
typedef enum { CUBLAS_GEMM_DEFAULT=-1, CUBLAS_GEMM_DEFAULT_TENSOR_OP=99 } cublasGemmAlgo_t;
static inline cublasStatus_t cublasCreate(cublasHandle_t*h){ static int x; *h=&x; return CUBLAS_STATUS_SUCCESS; }
static inline cublasStatus_t cublasSetMathMode(cublasHandle_t,cublasMath_t){ return CUBLAS_STATUS_SUCCESS; }
/* column-major C(m,n) = alpha op(A) op(B) + beta C */
static inline void cpu_sgemm_cm(cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,float alpha,
                                const float*A,int lda,const float*B,int ldb,float beta,float*C,int ldc){
    sgi xr=tb?1:ldb, xc=tb?ldb:1, yr=ta?1:lda, yc=ta?lda:1;
    sg_core(n,m,k,alpha,B,xr,xc,A,yr,yc,beta,C,ldc);
}
static inline cublasStatus_t cublasSgemm(cublasHandle_t,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,
        const float*alpha,const float*A,int lda,const float*B,int ldb,const float*beta,float*C,int ldc){
    cpu_sgemm_cm(ta,tb,m,n,k,*alpha,A,lda,B,ldb,*beta,C,ldc); return CUBLAS_STATUS_SUCCESS; }
/* many small independent problems: one thread each; few large: thread the GEMM */
static inline int cpu_par_batch(int count,double flops){ return count>=2*omp_get_max_threads() || (count>1 && flops<2e6); }
static inline cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,
        const float*alpha,const float*A,int lda,long long sA,const float*B,int ldb,long long sB,const float*beta,
        float*C,int ldc,long long sC,int batch){
    if(cpu_par_batch(batch,2.0*m*n*k)){
        #pragma omp parallel for schedule(dynamic,1)
        for(int b=0;b<batch;b++) cpu_sgemm_cm(ta,tb,m,n,k,*alpha,A+b*sA,lda,B+b*sB,ldb,*beta,C+b*sC,ldc);
    } else for(int b=0;b<batch;b++) cpu_sgemm_cm(ta,tb,m,n,k,*alpha,A+b*sA,lda,B+b*sB,ldb,*beta,C+b*sC,ldc);
    return CUBLAS_STATUS_SUCCESS; }
static inline cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,
        const void*alpha,const void*A,cudaDataType_t at,int lda,long long sA,const void*B,cudaDataType_t bt,int ldb,long long sB,
        const void*beta,void*C,cudaDataType_t ct,int ldc,long long sC,int batch,cublasComputeType_t,cublasGemmAlgo_t){
    if(at!=CUDA_R_32F||bt!=CUDA_R_32F||ct!=CUDA_R_32F) return CUBLAS_STATUS_NOT_SUPPORTED;
    return cublasSgemmStridedBatched(h,ta,tb,m,n,k,(const float*)alpha,(const float*)A,lda,sA,(const float*)B,ldb,sB,
                                     (const float*)beta,(float*)C,ldc,sC,batch); }
static inline cublasStatus_t cublasSgemmGroupedBatched(cublasHandle_t,const cublasOperation_t*ta,const cublasOperation_t*tb,
        const int*m,const int*n,const int*k,const float*alpha,const float*const*A,const int*lda,const float*const*B,const int*ldb,
        const float*beta,float*const*C,const int*ldc,int gc,const int*gs){
    int np=0; double fl=0; for(int g=0;g<gc;g++){ np+=gs[g]; fl+=2.0*m[g]*n[g]*k[g]*gs[g]; }
    int*gi=(int*)malloc((size_t)np*4*2), *bi=gi+np;
    for(int g=0,i=0;g<gc;g++) for(int j=0;j<gs[g];j++,i++){ gi[i]=g; bi[i]=i; }
    #define GB1(i) { int g=gi[i]; cpu_sgemm_cm(ta[g],tb[g],m[g],n[g],k[g],alpha[g],A[bi[i]],lda[g],B[bi[i]],ldb[g],beta[g],C[bi[i]],ldc[g]); }
    if(cpu_par_batch(np,fl/(np?np:1))){
        #pragma omp parallel for schedule(dynamic,1)
        for(int i=0;i<np;i++) GB1(i)
    } else for(int i=0;i<np;i++) GB1(i)
    #undef GB1
    free(gi); return CUBLAS_STATUS_SUCCESS; }
template<class...A> static inline cublasStatus_t cublasGemmEx(A...){ return CUBLAS_STATUS_NOT_SUPPORTED; }

/* ---- cublasLt (FP8 only, never reached on the CPU backend) ---- */
typedef void* cublasLtHandle_t;
typedef void* cublasLtMatmulDesc_t;
typedef void* cublasLtMatrixLayout_t;
typedef void* cublasLtMatmulPreference_t;
typedef struct { uint64_t data[8]; } cublasLtMatmulAlgo_t;
typedef struct { cublasLtMatmulAlgo_t algo; size_t workspaceSize; cublasStatus_t state; float wavesCount; int reserved[4]; } cublasLtMatmulHeuristicResult_t;
enum { CUBLASLT_MATMUL_DESC_TRANSA, CUBLASLT_MATMUL_DESC_TRANSB, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
       CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, CUBLASLT_MATMUL_DESC_FAST_ACCUM, CUBLASLT_MATRIX_LAYOUT_COLS,
       CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES };
#define CPU_LT(f) template<class...A> static inline cublasStatus_t f(A...){ return CUBLAS_STATUS_NOT_SUPPORTED; }
CPU_LT(cublasLtCreate) CPU_LT(cublasLtMatmulDescCreate) CPU_LT(cublasLtMatmulDescSetAttribute)
CPU_LT(cublasLtMatrixLayoutCreate) CPU_LT(cublasLtMatrixLayoutSetAttribute) CPU_LT(cublasLtMatrixLayoutDestroy)
CPU_LT(cublasLtMatmulPreferenceCreate) CPU_LT(cublasLtMatmulPreferenceSetAttribute) CPU_LT(cublasLtMatmulPreferenceDestroy)
CPU_LT(cublasLtMatmulAlgoGetHeuristic) CPU_LT(cublasLtMatmul)
#undef CPU_LT
#endif
