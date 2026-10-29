/* cpu_kern.h - OpenMP + AVX2 implementations of the cudalm kernels (CPU build).
 *
 * Included in pieces from the places in cudalm.cu / qlm.cuh where the CUDA
 * kernels live (CPUK_PART picks the piece), so every piece sees exactly the
 * declarations its CUDA counterpart does. Signatures match the __global__
 * kernels; the few kernels that take no row count read it from the launch
 * grid (g_kgrid, set by KL). Math mirrors the CUDA code op for op -- only the
 * summation order differs. Attention is a blocked flash-style rewrite; the
 * chunked per-slot kernels are kept as plain references for the selftest. */
#ifndef CPUK_PART
#error "define CPUK_PART before including cpu_kern.h"
#endif

/* =========================== part 1: core kernels ========================= */
#if CPUK_PART==1
#define GG_CH 128
#define WARP 32
#define QPB 8
#define KTILE 32
#define SUBL 8
#define EPLMAX 8
#define DSCALE 1024.0f
#define DSCALE_INV (1.0f/DSCALE)
#define CPU_PMIN 32768            /* loops shorter than this stay on one thread */
static void cpu_unreachable(const char*k){ fprintf(stderr,"[cpu] %s is GPU-only; no CPU path\n",k); exit(1); }
#define CPU_STUB(k) template<class...A> static inline void k(A...){ cpu_unreachable(#k); }

/* exp: Cody-Waite reduction + degree-6 polynomial, rel. err < 2e-7 */
static inline float fexp(float x){
    x=x<-87.0f?-87.0f:(x>88.0f?88.0f:x);
    float fi=rintf(x*1.44269504089f), r=x-fi*0.693145751953125f; r-=fi*1.428606765330187e-06f;
    float p=1.0f+r*(1.0f+r*(0.5f+r*(1.0f/6+r*(1.0f/24+r*(1.0f/120+r*(1.0f/720))))));
    return p*__uint_as_float((unsigned)((int)fi+127)<<23);
}
static inline float fsig(float x){ return 1.0f/(1.0f+fexp(-x)); }
#ifdef __AVX2__
static inline __m256 fexp8(__m256 x){
    x=_mm256_min_ps(_mm256_max_ps(x,_mm256_set1_ps(-87.0f)),_mm256_set1_ps(88.0f));
    __m256 fi=_mm256_round_ps(_mm256_mul_ps(x,_mm256_set1_ps(1.44269504089f)),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256 r=_mm256_fnmadd_ps(fi,_mm256_set1_ps(0.693145751953125f),x);
    r=_mm256_fnmadd_ps(fi,_mm256_set1_ps(1.428606765330187e-06f),r);
    __m256 p=_mm256_set1_ps(1.0f/720);
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f/120)); p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f/24));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f/6));   p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(0.5f));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f));     p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f));
    __m256i e=_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fi),_mm256_set1_epi32(127)),23);
    return _mm256_mul_ps(p,_mm256_castsi256_ps(e));
}
static inline __m256 fsig8(__m256 x){ __m256 o=_mm256_set1_ps(1.0f);
    return _mm256_div_ps(o,_mm256_add_ps(o,fexp8(_mm256_sub_ps(_mm256_setzero_ps(),x)))); }
static inline float hsum8(__m256 v){ __m128 a=_mm_add_ps(_mm256_castps256_ps128(v),_mm256_extractf128_ps(v,1));
    a=_mm_add_ps(a,_mm_movehl_ps(a,a)); a=_mm_add_ss(a,_mm_movehdup_ps(a)); return _mm_cvtss_f32(a); }
#endif
/* x[i] = exp(x[i]-m) over [0,n), returns the sum */
static inline float vexp_sub(float*x,int n,float m){
    int i=0; float s=0;
#ifdef __AVX2__
    __m256 acc=_mm256_setzero_ps(), mm=_mm256_set1_ps(m);
    for(;i+8<=n;i+=8){ __m256 e=fexp8(_mm256_sub_ps(_mm256_loadu_ps(x+i),mm)); _mm256_storeu_ps(x+i,e); acc=_mm256_add_ps(acc,e); }
    s=hsum8(acc);
#endif
    for(;i<n;i++){ x[i]=fexp(x[i]-m); s+=x[i]; }
    return s;
}

/* 8-wide vector type: __m256 with AVX2, a plain struct the compiler vectorizes otherwise */
#ifdef __AVX2__
typedef __m256 v8;
static inline v8 v8z(void){ return _mm256_setzero_ps(); }
static inline v8 v8ld(const float*p){ return _mm256_loadu_ps(p); }
static inline void v8st(float*p,v8 a){ _mm256_storeu_ps(p,a); }
static inline v8 v8bc(float x){ return _mm256_set1_ps(x); }
static inline v8 v8fma(v8 a,v8 b,v8 c){ return _mm256_fmadd_ps(a,b,c); }
static inline v8 v8mul(v8 a,v8 b){ return _mm256_mul_ps(a,b); }
static inline v8 v8add(v8 a,v8 b){ return _mm256_add_ps(a,b); }
#else
struct v8 { float f[8]; };
static inline v8 v8z(void){ v8 r; for(int i=0;i<8;i++) r.f[i]=0; return r; }
static inline v8 v8ld(const float*p){ v8 r; memcpy(r.f,p,32); return r; }
static inline void v8st(float*p,v8 a){ memcpy(p,a.f,32); }
static inline v8 v8bc(float x){ v8 r; for(int i=0;i<8;i++) r.f[i]=x; return r; }
static inline v8 v8fma(v8 a,v8 b,v8 c){ for(int i=0;i<8;i++) c.f[i]+=a.f[i]*b.f[i]; return c; }
static inline v8 v8mul(v8 a,v8 b){ for(int i=0;i<8;i++) a.f[i]*=b.f[i]; return a; }
static inline v8 v8add(v8 a,v8 b){ for(int i=0;i<8;i++) a.f[i]+=b.f[i]; return a; }
#endif

__global__ void k_embed(const float*P,size_t eo,const int*tok,float*X,int N,int D){
    #pragma omp parallel for schedule(static) if((size_t)N*D>=CPU_PMIN)
    for(int n=0;n<N;n++) memcpy(X+(size_t)n*D,P+eo+(size_t)tok[n]*D,(size_t)D*4);
}
/* scatter-add by column strip: rows sharing a token never race, order is fixed */
__global__ void k_embed_bwd(const int*tok,const float*dout,float*G,size_t eo,int N,int D){
    int ns=(D+15)/16;
    #pragma omp parallel for schedule(static)
    for(int c=0;c<ns;c++){ int d0=c*16, d1=d0+16<D?d0+16:D;
        for(int n=0;n<N;n++){ float*g=G+eo+(size_t)tok[n]*D; const float*s=dout+(size_t)n*D;
            for(int d=d0;d<d1;d++) g[d]+=s[d]; } }
}
static inline float cpu_dot(const float*a,const float*b,int n){ float s=0;
    #pragma omp simd reduction(+:s)
    for(int i=0;i<n;i++) s+=a[i]*b[i];
    return s; }
__global__ void k_rmsnorm(const float*X,const float*P,size_t go,float*O,float*rinv,int N,int D){
    #pragma omp parallel for schedule(static) if((size_t)N*D>=CPU_PMIN)
    for(int n=0;n<N;n++){ const float*x=X+(size_t)n*D,*g=P+go; float*o=O+(size_t)n*D;
        float s=1.0f/sqrtf(cpu_dot(x,x,D)/D+1e-5f); rinv[n]=s;
        #pragma omp simd
        for(int i=0;i<D;i++) o[i]=x[i]*s*g[i]; }
}
__global__ void k_rmsnorm_bwd(const float*X,const float*P,size_t go,const float*rinv,const float*dy,float*dx,int N,int D){
    #pragma omp parallel for schedule(static) if((size_t)N*D>=CPU_PMIN)
    for(int n=0;n<N;n++){ const float*x=X+(size_t)n*D,*d=dy+(size_t)n*D,*g=P+go; float*r=dx+(size_t)n*D;
        float s=rinv[n], c=0;
        #pragma omp simd reduction(+:c)
        for(int i=0;i<D;i++) c+=x[i]*g[i]*d[i];
        float cf=s*s*s/D*c;
        #pragma omp simd
        for(int i=0;i<D;i++) r[i]+=s*g[i]*d[i]-cf*x[i]; }
}
__global__ void k_rmsnorm_w(const float*X,const float*P,size_t go,float*O,float*rinv,int N,int D){ k_rmsnorm(X,P,go,O,rinv,N,D); }
__global__ void k_rmsnorm_bwd_w(const float*X,const float*P,size_t go,const float*rinv,const float*dy,float*dx,int N,int D){
    k_rmsnorm_bwd(X,P,go,rinv,dy,dx,N,D); }
__global__ void k_gamma_grad(const float*X,const float*rinv,const float*dy,float*G,size_t ggo,int N,int D){
    for(int n=0;n<N;n++){ const float*x=X+(size_t)n*D,*d=dy+(size_t)n*D; float r=rinv[n]; float*g=G+ggo;
        for(int i=0;i<D;i++) g[i]+=x[i]*r*d[i]; }
}
/* GG_CH fixed row chunks, summed in order: deterministic like the CUDA pair */
__global__ void k_gamma_part(const float*X,const float*rinv,const float*dy,float*part,int N,int D){
    #pragma omp parallel for schedule(static)
    for(int c=0;c<GG_CH;c++){ int r0=(int)((long long)N*c/GG_CH), r1=(int)((long long)N*(c+1)/GG_CH);
        float*p=part+(size_t)c*D; for(int i=0;i<D;i++) p[i]=0;
        for(int n=r0;n<r1;n++){ const float*x=X+(size_t)n*D,*d=dy+(size_t)n*D; float r=rinv[n];
            #pragma omp simd
            for(int i=0;i<D;i++) p[i]+=x[i]*r*d[i]; } }
}
__global__ void k_gamma_sum(const float*part,float*G,size_t ggo,int D){
    for(int d=0;d<D;d++){ float a=0; for(int c=0;c<GG_CH;c++) a+=part[(size_t)c*D+d]; G[ggo+d]+=a; }
}
__global__ void k_rope(float*buf,const float*cosb,const float*sinb,int N,int nh,int hd,int hf,int T,int back){
    #pragma omp parallel for schedule(static) if((size_t)N*nh*hd>=CPU_PMIN)
    for(int n=0;n<N;n++){ const float*cs=cosb+(size_t)(n%T)*hf,*sn=sinb+(size_t)(n%T)*hf;
        for(int h=0;h<nh;h++){ float*v=buf+(size_t)n*nh*hd+(size_t)h*hd;
            if(back){
                #pragma omp simd
                for(int i=0;i<hf;i++){ float a=v[i],b=v[i+hf]; v[i]=a*cs[i]+b*sn[i]; v[i+hf]=-a*sn[i]+b*cs[i]; } }
            else {
                #pragma omp simd
                for(int i=0;i<hf;i++){ float a=v[i],b=v[i+hf]; v[i]=a*cs[i]-b*sn[i]; v[i+hf]=b*cs[i]+a*sn[i]; } } } }
}

/* ---------------------------- attention ------------------------------------
 * Blocked flash attention. A task owns AQ queries (forward) or one (b,h)
 * (backward); keys come in AK=64 blocks transposed into Kt[d][j], so a query's
 * 64 scores are 64 broadcast-FMAs per head dim with no horizontal reductions.
 * Head dims that are a multiple of 8 use the vector path, others the scalar one. */
enum { AQ=32, AK=64 };
template<int NV> static inline void kt_dot(const float*q,const float*Kt,float*s){
    v8 a[8]; for(int v=0;v<8;v++) a[v]=v8z();
    for(int d=0;d<NV*8;d++){ v8 x=v8bc(q[d]); const float*k=Kt+d*AK; for(int v=0;v<8;v++) a[v]=v8fma(x,v8ld(k+8*v),a[v]); }
    for(int v=0;v<8;v++) v8st(s+8*v,a[v]);
}
static inline void kt_load(float*Kt,const float*K,size_t r0,int KD,int off,int hd,int nk){
    for(int j=0;j<AK;j++){ if(j<nk){ const float*kr=K+(r0+j)*KD+off; for(int d=0;d<hd;d++) Kt[d*AK+j]=kr[d]; }
                           else for(int d=0;d<hd;d++) Kt[d*AK+j]=0; }
}
template<int NV> static void attn_fwd_v(const float*Q,const float*K,const float*V,float*O,float*lse,
                                        int win,int B,int T,int H,int KV,int D,int KD){
    const int hd=NV*8, grp=H/KV, nqb=(T+AQ-1)/AQ; const float scale=1.0f/sqrtf((float)hd);
    const long ntask=(long)B*H*nqb;
    #pragma omp parallel
    {
        float __attribute__((aligned(32))) Kt[NV*8*AK], s[AK]; float m[AQ], l[AQ]; v8 o[AQ][NV];
        #pragma omp for schedule(dynamic,1)
        for(long tk=0;tk<ntask;tk++){
            int qb=nqb-1-(int)(tk%nqb); long bh=tk/nqb; int h=(int)(bh%H), b=(int)(bh/H), kvh=h/grp;
            int t0=qb*AQ, t1=t0+AQ<T?t0+AQ:T, nq=t1-t0;
            for(int i=0;i<nq;i++){ m[i]=-1e30f; l[i]=0; for(int v=0;v<NV;v++) o[i][v]=v8z(); }
            int klo=win>0?t0-win+1:0; if(klo<0) klo=0;
            for(int k0=klo;k0<t1;k0+=AK){
                int nk=t1-k0<AK?t1-k0:AK;
                kt_load(Kt,K,(size_t)b*T+k0,KD,kvh*hd,hd,nk);
                for(int i=0;i<nq;i++){
                    int t=t0+i, jlo=(win>0?t-win+1:0)-k0, jhi=t-k0; if(jlo<0) jlo=0; if(jhi>nk-1) jhi=nk-1; if(jhi<jlo) continue;
                    kt_dot<NV>(Q+((size_t)b*T+t)*D+(size_t)h*hd,Kt,s);
                    float mb=-1e30f; for(int j=jlo;j<=jhi;j++){ s[j]*=scale; mb=s[j]>mb?s[j]:mb; }
                    float nm=m[i]>mb?m[i]:mb, corr=fexp(m[i]-nm);
                    l[i]=l[i]*corr+vexp_sub(s+jlo,jhi-jlo+1,nm); m[i]=nm;
                    v8 r[NV], c=v8bc(corr); for(int v=0;v<NV;v++) r[v]=v8mul(o[i][v],c);
                    for(int j=jlo;j<=jhi;j++){ v8 p=v8bc(s[j]); const float*vr=V+((size_t)b*T+k0+j)*KD+(size_t)kvh*hd;
                        for(int v=0;v<NV;v++) r[v]=v8fma(p,v8ld(vr+8*v),r[v]); }
                    for(int v=0;v<NV;v++) o[i][v]=r[v];
                }
            }
            for(int i=0;i<nq;i++){ int t=t0+i; float*orow=O+((size_t)b*T+t)*D+(size_t)h*hd; v8 c=v8bc(1.0f/l[i]);
                for(int v=0;v<NV;v++) v8st(orow+8*v,v8mul(o[i][v],c));
                lse[((size_t)b*H+h)*T+t]=m[i]+logf(l[i]); }
        }
    }
}
/* scalar path (head dims not a multiple of 8): two-pass softmax per query */
static void attn_fwd_g(const float*Q,const float*K,const float*V,float*O,float*lse,
                       int win,int B,int T,int H,int KV,int hd,int D,int KD){
    const int grp=H/KV; const float scale=1.0f/sqrtf((float)hd);
    #pragma omp parallel
    {   float*s=(float*)malloc((size_t)T*4);
        #pragma omp for schedule(dynamic,16)
        for(long g=0;g<(long)B*H*T;g++){
            int t=(int)(g%T), h=(int)((g/T)%H), b=(int)(g/((long)T*H)), kvh=h/grp, lo=(win>0&&t>=win)?t-win+1:0;
            const float*q=Q+((size_t)b*T+t)*D+(size_t)h*hd; float mx=-1e30f;
            for(int j=lo;j<=t;j++){ s[j]=cpu_dot(q,K+((size_t)b*T+j)*KD+(size_t)kvh*hd,hd)*scale; if(s[j]>mx) mx=s[j]; }
            float l=vexp_sub(s+lo,t-lo+1,mx), inv=1.0f/l; float*o=O+((size_t)b*T+t)*D+(size_t)h*hd;
            for(int d=0;d<hd;d++) o[d]=0;
            for(int j=lo;j<=t;j++){ const float*v=V+((size_t)b*T+j)*KD+(size_t)kvh*hd; float p=s[j]*inv; for(int d=0;d<hd;d++) o[d]+=p*v[d]; }
            lse[((size_t)b*H+h)*T+t]=mx+logf(l);
        }
        free(s); }
}
#define ATTN_DISPATCH(F,...) switch(hd%8?0:hd/8){ case 1:F<1>(__VA_ARGS__);break; case 2:F<2>(__VA_ARGS__);break; \
    case 3:F<3>(__VA_ARGS__);break; case 4:F<4>(__VA_ARGS__);break; case 5:F<5>(__VA_ARGS__);break; \
    case 6:F<6>(__VA_ARGS__);break; case 7:F<7>(__VA_ARGS__);break; case 8:F<8>(__VA_ARGS__);break; default: goto scalar; }
__global__ void k_attn_fwd(const float*Q,const float*K,const float*V,float*O,float*lse,
                           int win,int B,int T,int H,int KV,int hd,int D,int KD){
    ATTN_DISPATCH(attn_fwd_v,Q,K,V,O,lse,win,B,T,H,KV,D,KD) return;
scalar: attn_fwd_g(Q,K,V,O,lse,win,B,T,H,KV,hd,D,KD);
}
__global__ void k_attn_fwd_tiled(const float*Q,const float*K,const float*V,float*O,float*lse,
                                 int win,int B,int T,int H,int KV,int hd,int D,int KD){ k_attn_fwd(Q,K,V,O,lse,win,B,T,H,KV,hd,D,KD); }
__global__ void k_attn_fwd_sub(const float*Q,const float*K,const float*V,float*O,float*lse,
                               int win,int B,int T,int H,int KV,int hd,int D,int KD){ k_attn_fwd(Q,K,V,O,lse,win,B,T,H,KV,hd,D,KD); }

/* p = exp(s*scale-L), ds = p*(dp-dr)*scale over [0,n); p into s, ds into dp */
static inline void attn_pds(float*s,float*dp,int n,float scale,float L,float dr){
    int j=0;
#ifdef __AVX2__
    __m256 sc=_mm256_set1_ps(scale), lv=_mm256_set1_ps(L), dv=_mm256_set1_ps(dr);
    for(;j+8<=n;j+=8){ __m256 p=fexp8(_mm256_fmsub_ps(_mm256_loadu_ps(s+j),sc,lv));
        _mm256_storeu_ps(s+j,p); _mm256_storeu_ps(dp+j,_mm256_mul_ps(_mm256_mul_ps(p,_mm256_sub_ps(_mm256_loadu_ps(dp+j),dv)),sc)); }
#endif
    for(;j<n;j++){ float p=fexp(s[j]*scale-L); s[j]=p; dp[j]=p*(dp[j]-dr)*scale; }
}
/* Fused backward: one task per (b,h) walks key blocks once, producing dQ for
 * its head directly and dK/dV partials per head, which a second pass sums over
 * each GQA group. Same math as k_attn_bwd_dq + k_attn_bwd_dkv, one S/dP pass. */
static float*g_attn_scr=NULL; static size_t g_attn_scrn=0;
static float* attn_scr(size_t n){ if(n>g_attn_scrn){ cpu_afree(g_attn_scr); g_attn_scr=(float*)cpu_amalloc(n*4); g_attn_scrn=n;
    if(!g_attn_scr){ fprintf(stderr,"[cpu] attention scratch OOM\n"); exit(1); } } return g_attn_scr; }
static void attn_bwd_reduce(const float*scr,float*dK,float*dV,int B,int T,int H,int KV,int hd,int KD){
    const int grp=H/KV;
    #pragma omp parallel for schedule(static)
    for(long r=0;r<(long)B*KV*T;r++){ int t=(int)(r%T), kvh=(int)((r/T)%KV), b=(int)(r/((long)T*KV));
        float*dk=dK+((size_t)b*T+t)*KD+(size_t)kvh*hd,*dv=dV+((size_t)b*T+t)*KD+(size_t)kvh*hd;
        for(int hh=0;hh<grp;hh++){ const float*pk=scr+(((size_t)b*H+kvh*grp+hh)*T*2)*hd+(size_t)t*hd, *pv=pk+(size_t)T*hd;
            for(int d=0;d<hd;d++){ dk[d]+=pk[d]; dv[d]+=pv[d]; } } }
}
template<int NV> static void attn_bwd_v(const float*Q,const float*K,const float*V,const float*dA,const float*lse,const float*Drow,
                                        float*dQ,float*dK,float*dV,int win,int B,int T,int H,int KV,int D,int KD){
    const int hd=NV*8, grp=H/KV; const float scale=1.0f/sqrtf((float)hd);
    float*scr=attn_scr((size_t)B*H*T*hd*2);
    #pragma omp parallel
    {
        float __attribute__((aligned(32))) Kt[NV*8*AK], Vt[NV*8*AK], s[AK], dp[AK]; v8 dk[AK][NV], dv[AK][NV];
        #pragma omp for schedule(dynamic,1)
        for(long tk=0;tk<(long)B*H;tk++){
            int h=(int)(tk%H), b=(int)(tk/H), kvh=h/grp; size_t r0=(size_t)b*T;
            float*pk=scr+(size_t)tk*T*2*hd, *pv=pk+(size_t)T*hd;
            const float*Lr=lse+(size_t)tk*T, *Dr=Drow+(size_t)tk*T;
            for(int k0=0;k0<T;k0+=AK){
                int nk=T-k0<AK?T-k0:AK, tq1=win>0?k0+nk-1+win:T; if(tq1>T) tq1=T;
                kt_load(Kt,K,r0+k0,KD,kvh*hd,hd,nk); kt_load(Vt,V,r0+k0,KD,kvh*hd,hd,nk);
                for(int j=0;j<nk;j++) for(int v=0;v<NV;v++){ dk[j][v]=v8z(); dv[j][v]=v8z(); }
                for(int t=k0;t<tq1;t++){
                    int jlo=(win>0?t-win+1:0)-k0, jhi=t-k0; if(jlo<0) jlo=0; if(jhi>nk-1) jhi=nk-1; if(jhi<jlo) continue;
                    const float*q=Q+(r0+t)*D+(size_t)h*hd, *g=dA+(r0+t)*D+(size_t)h*hd;
                    kt_dot<NV>(q,Kt,s); kt_dot<NV>(g,Vt,dp);
                    attn_pds(s+jlo,dp+jlo,jhi-jlo+1,scale,Lr[t],Dr[t]);
                    v8 qv[NV], gv[NV], a[NV];
                    for(int v=0;v<NV;v++){ qv[v]=v8ld(q+8*v); gv[v]=v8ld(g+8*v); a[v]=v8z(); }
                    for(int j=jlo;j<=jhi;j++){ v8 ds=v8bc(dp[j]), p=v8bc(s[j]); const float*kr=K+(r0+k0+j)*KD+(size_t)kvh*hd;
                        for(int v=0;v<NV;v++){ a[v]=v8fma(ds,v8ld(kr+8*v),a[v]); dk[j][v]=v8fma(ds,qv[v],dk[j][v]); dv[j][v]=v8fma(p,gv[v],dv[j][v]); } }
                    float*dq=dQ+(r0+t)*D+(size_t)h*hd;
                    for(int v=0;v<NV;v++) v8st(dq+8*v,v8add(v8ld(dq+8*v),a[v]));
                }
                for(int j=0;j<nk;j++) for(int v=0;v<NV;v++){ v8st(pk+(size_t)(k0+j)*hd+8*v,dk[j][v]); v8st(pv+(size_t)(k0+j)*hd+8*v,dv[j][v]); }
            }
        }
    }
    attn_bwd_reduce(scr,dK,dV,B,T,H,KV,hd,KD);
}
static void attn_bwd_g(const float*Q,const float*K,const float*V,const float*dA,const float*lse,const float*Drow,
                       float*dQ,float*dK,float*dV,int win,int B,int T,int H,int KV,int hd,int D,int KD){
    const int grp=H/KV; const float scale=1.0f/sqrtf((float)hd);
    float*scr=attn_scr((size_t)B*H*T*hd*2);
    #pragma omp parallel for schedule(dynamic,1)
    for(long tk=0;tk<(long)B*H;tk++){
        int h=(int)(tk%H), b=(int)(tk/H), kvh=h/grp; size_t r0=(size_t)b*T;
        float*pk=scr+(size_t)tk*T*2*hd, *pv=pk+(size_t)T*hd; memset(pk,0,(size_t)T*2*hd*4);
        for(int t=0;t<T;t++){
            int lo=(win>0&&t>=win)?t-win+1:0; const float*q=Q+(r0+t)*D+(size_t)h*hd,*g=dA+(r0+t)*D+(size_t)h*hd;
            float L=lse[(size_t)tk*T+t], dr=Drow[(size_t)tk*T+t]; float*dq=dQ+(r0+t)*D+(size_t)h*hd;
            for(int j=lo;j<=t;j++){ const float*kr=K+(r0+j)*KD+(size_t)kvh*hd,*vr=V+(r0+j)*KD+(size_t)kvh*hd;
                float p=fexp(cpu_dot(q,kr,hd)*scale-L), ds=p*(cpu_dot(g,vr,hd)-dr)*scale;
                float*a=pk+(size_t)j*hd,*c=pv+(size_t)j*hd;
                for(int d=0;d<hd;d++){ dq[d]+=ds*kr[d]; a[d]+=ds*q[d]; c[d]+=p*g[d]; } } }
    }
    attn_bwd_reduce(scr,dK,dV,B,T,H,KV,hd,KD);
}
static void cpu_attn_bwd(const float*Q,const float*K,const float*V,const float*dA,const float*lse,const float*Drow,
                         float*dQ,float*dK,float*dV,int win,int B,int T,int H,int KV,int hd,int D,int KD){
    ATTN_DISPATCH(attn_bwd_v,Q,K,V,dA,lse,Drow,dQ,dK,dV,win,B,T,H,KV,D,KD) return;
scalar: attn_bwd_g(Q,K,V,dA,lse,Drow,dQ,dK,dV,win,B,T,H,KV,hd,D,KD);
}
__global__ void k_attn_drow(const float*dA,const float*O,float*Drow,int B,int T,int H,int hd,int D,float s){
    #pragma omp parallel for schedule(static)
    for(long g=0;g<(long)B*H*T;g++){ int t=(int)(g%T), h=(int)((g/T)%H), b=(int)(g/((long)T*H));
        size_t o=((size_t)b*T+t)*D+(size_t)h*hd; Drow[((size_t)b*H+h)*T+t]=cpu_dot(dA+o,O+o,hd)*s; }
}
/* chunked per-slot references (selftest only; training uses cpu_attn_bwd) */
#define ATTN_SLOT int t=(int)(g%T), h=(int)((g/T)%H), b=(int)(g/((long)T*H)), kvh=h/(H/KV); \
    int lo,hi; if(win>0){ lo=t-(s1-1); hi=t-s0; if(hi<0) continue; if(lo<0) lo=0; } \
    else { lo=s0; hi=(t<s1-1)?t:(s1-1); if(lo>hi) continue; } \
    float scale=1.0f/sqrtf((float)hd); const float*q=Q+((size_t)b*T+t)*D+(size_t)h*hd,*da=dA+((size_t)b*T+t)*D+(size_t)h*hd; \
    float L=lse[((size_t)b*H+h)*T+t];
__global__ void k_attn_bwd_dr(const float*Q,const float*K,const float*V,const float*dA,const float*lse,float*Drow,
                              int s0,int s1,int win,int B,int T,int H,int KV,int hd,int D,int KD){
    #pragma omp parallel for schedule(static)
    for(long g=0;g<(long)B*H*T;g++){ ATTN_SLOT float dr=0;
        for(int s=lo;s<=hi;s++){ size_t o=((size_t)b*T+s)*KD+(size_t)kvh*hd;
            dr+=fexp(cpu_dot(q,K+o,hd)*scale-L)*cpu_dot(da,V+o,hd); }
        Drow[((size_t)b*H+h)*T+t]+=dr; }
}
__global__ void k_attn_bwd_dq(const float*Q,const float*K,const float*V,const float*dA,const float*lse,float*dQ,const float*Drow,
                              int s0,int s1,int win,int B,int T,int H,int KV,int hd,int D,int KD){
    #pragma omp parallel for schedule(static)
    for(long g=0;g<(long)B*H*T;g++){ ATTN_SLOT float dr=Drow[((size_t)b*H+h)*T+t]; float*dq=dQ+((size_t)b*T+t)*D+(size_t)h*hd;
        for(int s=lo;s<=hi;s++){ size_t o=((size_t)b*T+s)*KD+(size_t)kvh*hd;
            float p=fexp(cpu_dot(q,K+o,hd)*scale-L), ds=p*(cpu_dot(da,V+o,hd)-dr)*scale;
            for(int d=0;d<hd;d++) dq[d]+=ds*K[o+d]; } }
}
#undef ATTN_SLOT
__global__ void k_attn_bwd_dkv(const float*Q,const float*K,const float*V,const float*dA,const float*lse,const float*Drow,
                               float*dK,float*dV,int t0,int t1,int win,int B,int T,int H,int KV,int hd,int D,int KD){
    #pragma omp parallel for schedule(static)
    for(long g=0;g<(long)B*KV*T;g++){
        int s=(int)(g%T), kvh=(int)((g/T)%KV), b=(int)(g/((long)T*KV)), grp=H/KV, lo,hi;
        if(win>0){ lo=s+t0; hi=s+t1; if(hi>s+win) hi=s+win; if(hi>T) hi=T; } else { lo=s>t0?s:t0; hi=t1; }
        if(lo>=hi) continue;
        float scale=1.0f/sqrtf((float)hd); size_t o=((size_t)b*T+s)*KD+(size_t)kvh*hd;
        for(int hh=0;hh<grp;hh++){ int h=kvh*grp+hh;
            for(int t=lo;t<hi;t++){ size_t qo=((size_t)b*T+t)*D+(size_t)h*hd;
                float L=lse[((size_t)b*H+h)*T+t], dr=Drow[((size_t)b*H+h)*T+t];
                float p=fexp(cpu_dot(Q+qo,K+o,hd)*scale-L), ds=p*(cpu_dot(dA+qo,V+o,hd)-dr)*scale;
                for(int d=0;d<hd;d++){ dV[o+d]+=p*dA[qo+d]; dK[o+d]+=ds*Q[qo+d]; } } } }
}
__global__ void k_f2h(const float*src,__half*dst,size_t n,float s){
    #pragma omp parallel for schedule(static) if(n>=CPU_PMIN)
    for(long i=0;i<(long)n;i++) dst[i]=__half(src[i]*s);
}
__global__ void k_rope_h(float*buf,__half*dst,const float*cosb,const float*sinb,int N,int nh,int hd,int hf,int T,int wb){
    #pragma omp parallel for schedule(static) if((size_t)N*nh*hf>=CPU_PMIN)
    for(long r=0;r<(long)N*nh;r++){ int h=(int)(r%nh), n=(int)(r/nh), t=n%T; size_t o=(size_t)n*nh*hd+(size_t)h*hd;
        for(int i=0;i<hf;i++){ float c=cosb[t*hf+i],s=sinb[t*hf+i],a=buf[o+i],b=buf[o+i+hf],x=a*c-b*s,y=b*c+a*s;
            dst[o+i]=__half(x); dst[o+i+hf]=__half(y); if(wb){ buf[o+i]=x; buf[o+i+hf]=y; } } }
}
CPU_STUB(k_attn_fwd_h) CPU_STUB(k_attn_bwd_dq_h) CPU_STUB(k_attn_bwd_dkv_h)
template<class...A> static int attn_bwd_dkv_tc(A...){ return 0; }
template<class...A> static int attn_bwd_dq_tc(A...){ return 0; }
template<class...A> static int attn_fwd_tc(A...){ return 0; }
template<class...A> static int attn_fwd_lane(A...){ return 0; }
template<class...A> static int attn_fwd_fa(A...){ return 0; }
template<class...A> static int attn_bwd_dq_fa(A...){ return 0; }
template<class...A> static int attn_bwd_dkv_fa(A...){ return 0; }
static int g_fa=0;
template<class...A> static int attn_bwd_dq_lane(A...){ return 0; }
template<class...A> static int attn_bwd_dkv_lane(A...){ return 0; }

__global__ void k_swiglu(const float*G,const float*U,float*O,size_t n){
    #pragma omp parallel for schedule(static) if(n>=CPU_PMIN)
    for(long c=0;c<(long)((n+4095)/4096);c++){ size_t i=(size_t)c*4096, e=i+4096<n?i+4096:n;
#ifdef __AVX2__
        for(;i+8<=e;i+=8){ __m256 z=_mm256_loadu_ps(G+i); _mm256_storeu_ps(O+i,_mm256_mul_ps(_mm256_mul_ps(z,fsig8(z)),_mm256_loadu_ps(U+i))); }
#endif
        for(;i<e;i++){ float z=G[i]; O[i]=z*fsig(z)*U[i]; } }
}
__global__ void k_swiglu_bwd(const float*Gg,const float*U,const float*dact,float*dg,float*du,size_t n){
    #pragma omp parallel for schedule(static) if(n>=CPU_PMIN)
    for(long c=0;c<(long)((n+4095)/4096);c++){ size_t i=(size_t)c*4096, e=i+4096<n?i+4096:n;
#ifdef __AVX2__
        __m256 one=_mm256_set1_ps(1.0f);
        for(;i+8<=e;i+=8){ __m256 z=_mm256_loadu_ps(Gg+i), sg=fsig8(z), a=_mm256_loadu_ps(dact+i), silu=_mm256_mul_ps(z,sg);
            __m256 ds=_mm256_fmadd_ps(silu,_mm256_sub_ps(one,sg),sg);
            _mm256_storeu_ps(dg+i,_mm256_mul_ps(_mm256_mul_ps(a,_mm256_loadu_ps(U+i)),ds)); _mm256_storeu_ps(du+i,_mm256_mul_ps(a,silu)); }
#endif
        for(;i<e;i++){ float z=Gg[i], sg=fsig(z), silu=z*sg, ds=sg+z*sg*(1.0f-sg); dg[i]=dact[i]*U[i]*ds; du[i]=dact[i]*silu; } }
}
__global__ void k_swiglu4(const float4*G,const float4*U,float4*O,size_t n4){ k_swiglu((const float*)G,(const float*)U,(float*)O,n4*4); }
__global__ void k_swiglu_bwd4(const float4*Gg,const float4*U,const float4*dact,float4*dg,float4*du,size_t n4){
    k_swiglu_bwd((const float*)Gg,(const float*)U,(const float*)dact,(float*)dg,(float*)du,n4*4); }
#endif /* part 1 */

/* ================ part 2: MoE, elementwise, head, grad norm =============== */
#if CPUK_PART==2
__global__ void k_router_top1(const float*rt,float*rprobs,int*assign,float*gate,int N,int E,const float*bias){
    #pragma omp parallel for schedule(static) if(N*E>=CPU_PMIN)
    for(int n=0;n<N;n++){ const float*r=rt+(size_t)n*E; float*pr=rprobs+(size_t)n*E;
        float mx=-1e30f; for(int e=0;e<E;e++) if(r[e]>mx) mx=r[e];
        float s=0; for(int e=0;e<E;e++){ pr[e]=fexp(r[e]-mx); s+=pr[e]; }
        float inv=1.0f/s; int best=0;
        for(int e=0;e<E;e++){ pr[e]*=inv; if(pr[e]+(bias?bias[e]:0.f)>pr[best]+(bias?bias[best]:0.f)) best=e; }
        if(g_route_mode_d==1){ assign[n]=0; gate[n]=1.0f; continue; }
        if(g_route_mode_d==2){ unsigned hh=(unsigned)n*2654435761u; hh^=hh>>15; assign[n]=(int)(hh%(unsigned)E); gate[n]=1.0f; continue; }
        assign[n]=best; gate[n]=pr[best]; }
}
__global__ void k_router_topk(const float*rt,float*rprobs,int*assign,float*gate,int N,int E,int K,const float*bias,int bs){
    #pragma omp parallel for schedule(static) if(N*E>=CPU_PMIN)
    for(int n=0;n<N;n++){ const float*r=rt+(size_t)n*E; float*pr=rprobs+(size_t)n*E;
        float mx=-1e30f; for(int e=0;e<E;e++) if(r[e]>mx) mx=r[e];
        float s=0; for(int e=0;e<E;e++){ pr[e]=fexp(r[e]-mx); s+=pr[e]; }
        float inv=1.0f/s; for(int e=0;e<E;e++) pr[e]*=inv;
        int chosen[8];
        for(int k=0;k<K;k++){ int best=-1; float bp=-1e30f;
            for(int e=0;e<E;e++){ int taken=0; for(int j=0;j<k;j++) if(chosen[j]==e){ taken=1; break; }
                if(taken) continue; float sc=pr[e]+(bias?bias[k*bs+e]:0.f); if(sc>bp){ bp=sc; best=e; } }
            if(best<0) best=k;
            chosen[k]=best; assign[(size_t)k*N+n]=best; gate[(size_t)k*N+n]=pr[best]; } }
}
__global__ void k_gather(const float*src,const int*perm,float*Xp,int N,int D){
    #pragma omp parallel for schedule(static) if((size_t)N*D>=CPU_PMIN)
    for(int j=0;j<N;j++){ int n=perm[j]; float*o=Xp+(size_t)j*D;
        if(n<0) memset(o,0,(size_t)D*4); else memcpy(o,src+(size_t)n*D,(size_t)D*4); }
}
__global__ void k_scatter_gate(const float*yp,const int*perm,const float*gate,float*out,int N,int D){
    #pragma omp parallel for schedule(static) if((size_t)N*D>=CPU_PMIN)
    for(int j=0;j<N;j++){ int n=perm[j]; if(n<0) continue; float g=gate[n]; float*o=out+(size_t)n*D; const float*y=yp+(size_t)j*D;
        #pragma omp simd
        for(int d=0;d<D;d++) o[d]+=g*y[d]; }
}
__global__ void k_gather_i(const int*src,const int*idx,int*dst,int n){ for(int j=0;j<n;j++) dst[j]=src[idx[j]]; }
__global__ void k_scatter_rows(const float*src,const int*idx,float*dst,int n,int D){
    #pragma omp parallel for schedule(static) if((size_t)n*D>=CPU_PMIN)
    for(int j=0;j<n;j++) memcpy(dst+(size_t)idx[j]*D,src+(size_t)j*D,(size_t)D*4);
}
#define RB4(D4) ((((D4)+31)&~31)>256?256:(((D4)+31)&~31))
__global__ void k_gather4(const float4*src,const int*perm,float4*Xp,int D4){
    k_gather((const float*)src,perm,(float*)Xp,(int)g_kgrid.x,D4*4); }
__global__ void k_scatter_add(const float*dXp,const int*perm,float*dst,int N,int D){
    #pragma omp parallel for schedule(static) if((size_t)N*D>=CPU_PMIN)
    for(int j=0;j<N;j++){ int n=perm[j]; if(n<0) continue; float*o=dst+(size_t)n*D; const float*s=dXp+(size_t)j*D;
        #pragma omp simd
        for(int d=0;d<D;d++) o[d]+=s[d]; }
}
__global__ void k_scatter_add4(const float4*dXp,const int*perm,float4*dst,int D4){
    k_scatter_add((const float*)dXp,perm,(float*)dst,(int)g_kgrid.x,D4*4); }
/* d_gate[n] = dout[n].yp[j]; dyp[j] = gate[n]*dout[n]; optionally Xp[j] = src[n] */
static void cpu_moe_dgate(const float*dout,const float*yp,const int*perm,const float*gate,float*d_gate,float*dyp,
                          const float*src,float*Xp,int R,int D){
    #pragma omp parallel for schedule(static) if((size_t)R*D>=CPU_PMIN)
    for(int j=0;j<R;j++){ int n=perm[j]; float*dy=dyp+(size_t)j*D;
        if(n<0){ memset(dy,0,(size_t)D*4); if(src) memset(Xp+(size_t)j*D,0,(size_t)D*4); continue; }
        const float*dr=dout+(size_t)n*D; float g=gate[n];
        d_gate[n]=cpu_dot(dr,yp+(size_t)j*D,D);
        for(int d=0;d<D;d++) dy[d]=g*dr[d];
        if(src) memcpy(Xp+(size_t)j*D,src+(size_t)n*D,(size_t)D*4); }
}
__global__ void k_moe_dgate4(const float4*dout,const float4*yp,const int*perm,const float*gate,
                             float*d_gate,float4*dyp,const float4*src,float4*Xp,int D4){
    cpu_moe_dgate((const float*)dout,(const float*)yp,perm,gate,d_gate,(float*)dyp,(const float*)src,(float*)Xp,(int)g_kgrid.x,D4*4); }
__global__ void k_moe_dgate(const float*dout,const float*yp,const int*perm,const float*gate,
                            float*d_gate,float*dyp,int N,int D){ cpu_moe_dgate(dout,yp,perm,gate,d_gate,dyp,NULL,NULL,N,D); }
__global__ void k_router_bwd(const float*rprobs,const int*assign,const float*d_gate,const int*cnt,float*drt,int N,int E,float aux){
    #pragma omp parallel for schedule(static) if(N*E>=CPU_PMIN)
    for(int n=0;n<N;n++){ const float*pr=rprobs+(size_t)n*E; float*d=drt+(size_t)n*E; float invN=1.0f/N, dot=0;
        for(int e=0;e<E;e++){ float dp=aux*E*((float)cnt[e]*invN)*invN; if(e==assign[n]) dp+=d_gate[n]; d[e]=dp; dot+=pr[e]*dp; }
        for(int e=0;e<E;e++) d[e]=pr[e]*(d[e]-dot); }
}
__global__ void k_router_bwd_topk(const float*rprobs,const int*assign,const float*d_gate,const float*w,float*drt,int N,int E,int K){
    #pragma omp parallel for schedule(static) if(N*E>=CPU_PMIN)
    for(int n=0;n<N;n++){ const float*pr=rprobs+(size_t)n*E; float*d=drt+(size_t)n*E; float dot=0;
        for(int e=0;e<E;e++) d[e]=w[e];
        for(int k=0;k<K;k++) d[assign[(size_t)k*N+n]]+=d_gate[(size_t)k*N+n];
        for(int e=0;e<E;e++) dot+=pr[e]*d[e];
        for(int e=0;e<E;e++) d[e]=pr[e]*(d[e]-dot); }
}
#define CPU_EW(name,args,...) __global__ void name args { \
    _Pragma("omp parallel for simd schedule(static) if(n>=CPU_PMIN)") for(long i=0;i<(long)n;i++){ __VA_ARGS__; } }
CPU_EW(k_add,(float*A,const float*B,size_t n),A[i]+=B[i])
CPU_EW(k_copy,(float*A,const float*B,size_t n),A[i]=B[i])
CPU_EW(k_scale,(float*G,size_t n,float s),G[i]*=s)
CPU_EW(k_scale_d,(float*G,size_t n,const float*s),G[i]*=*s)
/* softmax + CE per row; dl = (p - onehot)/nAll, masked rows (tgt<0) zeroed */
template<class TL> __global__ void k_softmax_ce(const TL*lg,const int*tgt,float*loss,float*dl,__nv_fp8_e4m3*q8,int rows,int V,int nAll,const float*wt,float negp){
    #pragma omp parallel for schedule(static)
    for(int n=0;n<rows;n++){
        const TL*l=lg+(size_t)n*V; float*d=dl+(size_t)n*V; __nv_fp8_e4m3*q=q8?q8+(size_t)n*V:NULL; float qs=448.0f*nAll;
        if(tgt[n]<0){ memset(d,0,(size_t)V*4); if(q) for(int i=0;i<V;i++) q[i]=__nv_fp8_e4m3(0.f); loss[n]=0.f; continue; }
        float mx=-1e30f; for(int i=0;i<V;i++){ float v=(float)l[i]; d[i]=v; mx=v>mx?v:mx; }
        float inv=1.0f/vexp_sub(d,V,mx); int tg=tgt[n];
        float ptg=d[tg]*inv, w=wt?wt[n]:1.f; if(w<=-50.f) w=fminf(1.f,fmaxf(-1.f,w+100.f-logf(ptg+1e-12f))); if(w<0.f&&ptg<negp) w=0.f; float iN=w/nAll, sc=inv*iN;
        #pragma omp simd
        for(int i=0;i<V;i++) d[i]*=sc;
        d[tg]-=iN; loss[n]=-logf(ptg+1e-12f);
        if(q) for(int i=0;i<V;i++) q[i]=__nv_fp8_e4m3(d[i]*qs); }
}
/* logit lens: out[4n..] = logP(tgt), argmax, entropy, logP(fin) */
__global__ void k_lens(const float*lg,const int*tgt,const int*fin,float*out,int rows,int V){
    #pragma omp parallel for schedule(static)
    for(int n=0;n<rows;n++){ const float*l=lg+(size_t)n*V; float mx=l[0]; int mi=0;
        for(int i=1;i<V;i++) if(l[i]>mx){ mx=l[i]; mi=i; }
        double s=0,e=0; for(int i=0;i<V;i++){ float x=l[i]-mx, p=fexp(x); s+=p; e+=(double)p*x; }
        float lse=mx+(float)log(s); float*o=out+4*(size_t)n;
        o[0]=l[tgt[n]]-lse; o[1]=(float)mi; o[2]=(float)(log(s)-e/s); o[3]=fin?l[fin[n]]-lse:-(float)log(s); }
}
CPU_STUB(k_sce_h8)
CPU_EW(k_f2bf,(const float*src,bf16*dst,size_t n),dst[i]=d_f2bf(src[i]))
/* sum of squares -> one partial per launch block (64 in every caller) */
__global__ void k_sqsum(const float*G,size_t n,float*part){
    int nb=(int)g_kgrid.x;
    #pragma omp parallel for schedule(static)
    for(int c=0;c<nb;c++){ size_t i0=n*(size_t)c/nb, i1=n*(size_t)(c+1)/nb; double s=0;
        #pragma omp simd reduction(+:s)
        for(size_t i=i0;i<i1;i++) s+=(double)G[i]*G[i];
        part[c]=(float)s; }
}
__global__ void k_gnorm(const float*part,int nside,int clip_side,float*gs){
    double ss=0,sd=0; for(int i=0;i<64;i++) ss+=part[i];
    for(int i=64;i<64*(1+nside);i++) sd+=part[i];
    float gn=(float)sqrt(clip_side?ss+sd:ss); if(!isfinite(sd)) gn=NAN;
    gs[0]=gn>1.0f?1.0f/gn:1.0f; gs[1]=gn;
}
__global__ void k_clipmul(float*gs,float gn){ gs[0]=gn>1.0f?1.0f/gn:1.0f; gs[1]=gn; }
#endif /* part 2 */

/* ===================== part 3: AdamW (bf16 moments) ======================== */
#if CPUK_PART==3
__global__ void k_adamw(float*P,const float*G,bf16*M,bf16*V,size_t n,float lr,float b1,float b2,float wd,float bc1,float bc2,
                        const float*lmul,size_t l0,size_t per,int L,MuSkip mu,const float*gs=NULL){
    #pragma omp parallel for schedule(static) if(n>=CPU_PMIN)
    for(long j=0;j<(long)n;j++){ size_t i=(size_t)j;
        if(mu.on && muon_owns(i,mu.l0,mu.per,mu.L,mu.D,mu.a2r)) continue;
        float lp=lr; if(lmul && i>=l0){ size_t d=i-l0; if(d<(size_t)L*per) lp=lr*lmul[d/per]; }
        float g=gs?G[i]*gs[0]:G[i], m=b1*d_bf2f(M[i])+(1.0f-b1)*g, v=b2*d_bf2f(V[i])+(1.0f-b2)*g*g;
        M[i]=d_f2bf(m); V[i]=d_f2bf(v); P[i]-=lp*((m/bc1)/(sqrtf(v/bc2)+1e-8f)+wd*P[i]); }
}
#endif /* part 3 */

/* ============== part 4: FP8 GEMM helpers (GPU-only, never reached) ========= */
#if CPUK_PART==4
CPU_STUB(k_absmax) CPU_STUB(k_setf) CPU_STUB(k_mkscale) CPU_STUB(k_quant_e4m3) CPU_STUB(k_qtr_e4m3)
#endif

/* ============== part 5: attention gate, n-gram context gate ================= */
#if CPUK_PART==5
CPU_EW(k_agate_f,(const float*att,float*s,float*o,size_t n),float g=2.0f*fsig(s[i]); s[i]=g; o[i]=att[i]*g)
CPU_EW(k_agate_b,(const float*att,const float*s,float*datt,float*dz,size_t n),float g=s[i],d=datt[i]; dz[i]=d*att[i]*g*(1.0f-0.5f*g); datt[i]=d*g)
CPU_EW(k_mul,(const float*a,const float*b,float*o,size_t n),o[i]=a[i]*b[i])
__global__ void k_nggate_f(float*h,const float*v,float*k,float*hn,float*g,float*rh,float*rk,int D){
    int N=(int)g_kgrid.x;
    #pragma omp parallel for schedule(static) if((size_t)N*D>=CPU_PMIN)
    for(int r=0;r<N;r++){ size_t o=(size_t)r*D; float a=cpu_dot(h+o,h+o,D), b=cpu_dot(k+o,k+o,D), c=cpu_dot(h+o,k+o,D);
        float r1=1.0f/sqrtf(a/D+1e-6f), r2=1.0f/sqrtf(b/D+1e-6f), gg=fsig(c*r1*r2/sqrtf((float)D));
        for(int i=0;i<D;i++){ hn[o+i]=h[o+i]*r1; k[o+i]*=r2; h[o+i]+=gg*v[o+i]; }
        g[r]=gg; rh[r]=r1; rk[r]=r2; }
}
__global__ void k_nggate_b(float*dy,const float*v,const float*hn,const float*kn,const float*g,
                           const float*rh,const float*rk,float*dv,float*dk,int D){
    int N=(int)g_kgrid.x;
    #pragma omp parallel for schedule(static) if((size_t)N*D>=CPU_PMIN)
    for(int r=0;r<N;r++){ size_t o=(size_t)r*D; float a=cpu_dot(dy+o,v+o,D), b=cpu_dot(hn+o,kn+o,D);
        float gg=g[r], cz=a*gg*(1.0f-gg)/sqrtf((float)D), m=b/D, r1=rh[r], r2=rk[r];
        for(int i=0;i<D;i++){ float d=dy[o+i],x=hn[o+i],y=kn[o+i]; dv[o+i]=gg*d; dk[o+i]=cz*(x-y*m)*r2; dy[o+i]=d+cz*(y-x*m)*r1; } }
}
#endif /* part 5 */

/* ============== part 6: per-expert FP8 casts (GPU-only) ===================== */
#if CPUK_PART==6
CPU_STUB(k_quant_pad) CPU_STUB(k_qtrp_b)
#endif

/* ============== part 7: n-gram table (keys, gather, scatter, q4 step) ====== */
#if CPUK_PART==7
__global__ void k_ngram_key(const int*tok,int*key,int N,int T,int R,int order,int base){
    #pragma omp parallel for schedule(static) if(N>=CPU_PMIN)
    for(int n=0;n<N;n++){ if(n%T<order-1){ key[n]=-1; continue; }
        unsigned h=2166136261u; for(int j=order-1;j>=0;j--) h=(h^(unsigned)tok[n-j])*16777619u;
        h^=h>>15; key[n]=base+(int)(h%(unsigned)R); }
}
__global__ void k_ngram_gather(const float*tbl,const int*key,float*buf,int N,int De,int nk){
    #pragma omp parallel for schedule(static) if((size_t)N*De>=CPU_PMIN)
    for(int n=0;n<N;n++){ float*o=buf+(size_t)n*De; int k=key[n];
        for(int e=0;e<De;e++) o[e]= k>=0 ? tbl[(size_t)k*De+e] : 0.0f;
        for(int j=1;j<nk;j++){ k=key[(size_t)j*N+n]; if(k>=0) for(int e=0;e<De;e++) o[e]+=tbl[(size_t)k*De+e]; } }
}
/* scatter-adds run serially: positions sharing a row would race, and it is ~N*De adds */
__global__ void k_ngram_scatter(float*grad,const int*key,const float*dbuf,int N,int De,int Nd){
    for(int n=0;n<N;n++){ int k=key[n]; if(k<0) continue; float*g=grad+(size_t)k*De; const float*s=dbuf+(size_t)(n%Nd)*De;
        for(int e=0;e<De;e++) g[e]+=s[e]; }
}
__global__ void k_ng_scatter_compact(float*Gc,const int*key,const int*cidx,const float*dbuf,int N,int De,int group,int Nd){
    for(int n=0;n<N;n++){ int r=key[n]; if(r<0) continue; int c=cidx[n]; if(c<0) continue;
        const float*s=dbuf+(size_t)(n%Nd)*De;
        for(int e=0;e<De;e++) Gc[(size_t)c*group+(int)(((long long)r*De+e)%group)]+=s[e]; }
}
__global__ void k_ng_q4_step(float*P,const float*Gc,signed char*d,signed char*r,float*s,signed char*m8,float*m8s,
                             unsigned char*v8,float*v8s,const int*touched_gi,int group,int dmax,
                             float lr,float b1,float b2,float wd,float sref,float inv_bc1,float inv_sqrt_bc2,
                             unsigned long long t,unsigned long long nt,const float*gs){
    int ng=(int)g_kgrid.x;
    #pragma omp parallel for schedule(static)
    for(int w=0;w<ng;w++){
        int gi=touched_gi[w]; unsigned long long j0=(unsigned long long)gi*group, qi0=j0;
        int n=(int)((unsigned long long)group<nt-j0?(unsigned long long)group:nt-j0);
        float sg=s[gi], L=lr*128.0f/sref, dec=lr*wd*128.0f, ms_old=m8s[gi], vs_old=v8s[gi];
        float mn[256], rv[256], am=0, vm=0; signed char shd[256];
        for(int j=0;j<group;j++) shd[j]=0;
        for(int j=0;j<n;j++){ unsigned long long qi=qi0+j;
            float gr=Gc[(size_t)w*group+j]*(gs?gs[0]:1.0f), mv=(float)m8[qi]*ms_old, sv=qv_dec((float)v8[qi],vs_old), vv=sv*sv;
            mn[j]=b1*mv+(1.0f-b1)*gr; vv=b2*vv+(1.0f-b2)*gr*gr; rv[j]=sqrtf(vv);
            float veps=fmaxf(qv_dec(1.0f,vs_old),1e-8f), u=(mn[j]*inv_bc1)/(rv[j]*inv_sqrt_bc2+veps);
            u=fmaxf(-8.0f,fminf(8.0f,u)); rv[j]=qv_enc(rv[j]);
            int dd=qd_get(d,qi,1), idl=qsri(-(L*u+dec*(float)dd),qi,t), rr=(int)r[qi]+idl;
            int c=(rr>64)?(((rr-65)>>7)+1):(rr<-64)?-((((-rr)-65)>>7)+1):0;
            rr-=c*128; dd+=c; dd=qclmp(dd,-dmax,dmax);
            shd[j]=(signed char)dd; r[qi]=(signed char)rr; P[qi]=sg*(float)dd;
            am=fmaxf(am,fabsf(mn[j])); vm=fmaxf(vm,rv[j]); }
        for(int j=0;j<group;j++) qd_write_group(d,qi0,shd,n,j,1);
        float ms=am>0.0f?am/127.0f:1e-12f, vs=vm>0.0f?vm/255.0f:1e-12f; m8s[gi]=ms; v8s[gi]=vs;
        for(int j=0;j<n;j++){ unsigned long long qi=qi0+j;
            m8[qi]=(signed char)qclmp(qsri(mn[j]/ms,qi,t+0x5EEDULL),-127,127);
            int vq=qclmp(qsri(rv[j]/vs,qi,t+0xBEEFULL),0,255); if(vq==0&&rv[j]>0.0f) vq=1; v8[qi]=(unsigned char)vq; }
    }
}
#endif /* part 7 */

/* ============== part 8: touched-group map (serial: slot order is free) ===== */
#if CPUK_PART==8
__global__ void k_ng_claim(const int*key,int*slot,int*tgi,int*cnt,int N,int De,int grp){
    for(int n=0;n<N;n++){ int r=key[n]; if(r<0) continue; int gi=(int)(((long long)r*De)/grp);
        if(slot[gi]==-1){ int c=(*cnt)++; tgi[c]=gi; slot[gi]=c; } }
}
__global__ void k_ng_cidx(const int*key,const int*slot,int*cidx,int N,int De,int grp){
    for(int n=0;n<N;n++){ int r=key[n]; cidx[n]= r<0 ? -1 : slot[(int)(((long long)r*De)/grp)]; }
}
__global__ void k_ng_unclaim(int*slot,const int*tgi,int n){ for(int i=0;i<n;i++) slot[tgi[i]]=-1; }
#endif /* part 8 */

/* ============== part 9: sparse engram ===================================== */
#if CPUK_PART==9
__global__ void k_eg_key(const int*tok,const unsigned short*cn,int*key,int N,int T,int S,int Rp){
    #pragma omp parallel for schedule(static) if(S*N>=CPU_PMIN)
    for(int i=0;i<S*N;i++){ int s=i/N, n=i%N, o=c_eg_ord[s];
        if(n%T<o-1){ key[i]=-1; continue; }
        unsigned long long m=0;
        for(int k=0;k<o;k++){ unsigned t=(unsigned)tok[n-k]; if(cn) t=cn[t]; m^=(unsigned long long)t*c_eg_mult[s*4+k]; }
        key[i]=(int)(m%(unsigned long long)c_eg_prime[s])+s*Rp; }
}
__global__ void k_eg_gather(const float*tbl,const int*key,float*buf,int N,int S,int De){
    #pragma omp parallel for schedule(static) if((size_t)N*S*De>=CPU_PMIN)
    for(int n=0;n<N;n++) for(int s=0;s<S;s++){ int k=key[(size_t)s*N+n]; float*o=buf+((size_t)n*S+s)*De;
        if(k>=0) memcpy(o,tbl+(size_t)k*De,(size_t)De*4); else memset(o,0,(size_t)De*4); }
}
/* sub-tables own disjoint rows (and q4 groups), so they scatter in parallel */
__global__ void k_eg_scatter(float*G,const int*key,const float*db,int N,int S,int De){
    #pragma omp parallel for schedule(static)
    for(int s=0;s<S;s++) for(int n=0;n<N;n++){ int k=key[(size_t)s*N+n]; if(k<0) continue;
        float*g=G+(size_t)k*De; const float*d=db+((size_t)n*S+s)*De; for(int e=0;e<De;e++) g[e]+=d[e]; }
}
__global__ void k_eg_scatter_c(float*Gc,const int*key,const int*cidx,const float*db,int N,int S,int De,int grp){
    #pragma omp parallel for schedule(static)
    for(int s=0;s<S;s++) for(int n=0;n<N;n++){ size_t ki=(size_t)s*N+n; int k=key[ki]; if(k<0) continue; int c=cidx[ki]; if(c<0) continue;
        const float*d=db+((size_t)n*S+s)*De; for(int e=0;e<De;e++) Gc[(size_t)c*grp+(int)(((long long)k*De+e)%grp)]+=d[e]; }
}
__global__ void k_eg_gate_f(const float*h,const float*v,float*k,float*hn,float*vg,float*g,float*rh,float*rk,int D){
    int N=(int)g_kgrid.x;
    #pragma omp parallel for schedule(static) if((size_t)N*D>=CPU_PMIN)
    for(int r=0;r<N;r++){ size_t o=(size_t)r*D; float a=cpu_dot(h+o,h+o,D), b=cpu_dot(k+o,k+o,D), c=cpu_dot(h+o,k+o,D);
        float r1=1.0f/sqrtf(a/D+1e-6f), r2=1.0f/sqrtf(b/D+1e-6f), z=c*r1*r2/sqrtf((float)D);
        float gg=fsig(copysignf(sqrtf(fmaxf(fabsf(z),1e-6f)),z));
        for(int i=0;i<D;i++){ hn[o+i]=h[o+i]*r1; k[o+i]*=r2; vg[o+i]=gg*v[o+i]; }
        g[r]=gg; rh[r]=r1; rk[r]=r2; }
}
__global__ void k_eg_gate_b(const float*dvg,float*dh,const float*v,const float*hn,const float*kn,const float*g,
                            const float*rh,const float*rk,float*dv,float*dk,int D){
    int N=(int)g_kgrid.x;
    #pragma omp parallel for schedule(static) if((size_t)N*D>=CPU_PMIN)
    for(int r=0;r<N;r++){ size_t o=(size_t)r*D; float a=cpu_dot(dvg+o,v+o,D), b=cpu_dot(hn+o,kn+o,D);
        float gg=g[r], m=b/D, z=m*sqrtf((float)D), r1=rh[r], r2=rk[r];
        float cz=a*gg*(1.0f-gg)*0.5f/sqrtf(fmaxf(fabsf(z),1e-6f))/sqrtf((float)D);
        for(int i=0;i<D;i++){ float d=dvg[o+i],x=hn[o+i],y=kn[o+i]; dv[o+i]=gg*d; dk[o+i]=cz*(x-y*m)*r2; dh[o+i]+=cz*(y-x*m)*r1; } }
}
__global__ void k_eg_conv_f(float*h,const float*vg,const float*y,const float*K,float*c,int N,int D,int T,int dil){
    #pragma omp parallel for schedule(static) if((size_t)N*D>=CPU_PMIN)
    for(int n=0;n<N;n++){ int p=n%T; size_t o=(size_t)n*D;
        for(int d=0;d<D;d++){ float s=0; for(int j=0;j<4;j++) if(p>=j*dil) s+=K[d*4+j]*y[o+d-(size_t)j*dil*D];
            c[o+d]=s; h[o+d]+=vg[o+d]+s*fsig(s); } }
}
CPU_EW(k_eg_conv_dc,(const float*dout,const float*c,float*dc,size_t n),float x=c[i],s=fsig(x); dc[i]=dout[i]*s*(1.0f+x*(1.0f-s)))
__global__ void k_eg_conv_dy(const float*dc,const float*K,float*dy,int N,int D,int T,int dil){
    #pragma omp parallel for schedule(static) if((size_t)N*D>=CPU_PMIN)
    for(int n=0;n<N;n++){ int p=n%T; size_t o=(size_t)n*D;
        for(int d=0;d<D;d++){ float s=0; for(int j=0;j<4;j++) if(p+j*dil<T&&n+j*dil<N) s+=K[d*4+j]*dc[o+d+(size_t)j*dil*D]; dy[o+d]=s; } }
}
__global__ void k_eg_conv_dK(const float*dc,const float*y,float*KG,int N,int D,int T,int dil){
    #pragma omp parallel for schedule(static)
    for(int d0=0;d0<D;d0+=16){ int d1=d0+16<D?d0+16:D; float s[16][4]={{0}};
        for(int n=0;n<N;n++){ int p=n%T; const float*g=dc+(size_t)n*D;
            for(int j=0;j<4;j++) if(p>=j*dil){ const float*yy=y+(size_t)(n-j*dil)*D; for(int d=d0;d<d1;d++) s[d-d0][j]+=g[d]*yy[d]; } }
        for(int d=d0;d<d1;d++) for(int j=0;j<4;j++) KG[d*4+j]=s[d-d0][j]; }
}
#endif /* part 9 */

/* ============== part 12: router z-loss ===================================== */
#if CPUK_PART==12
__global__ void k_rlse(const float*rt,float*lse,int N,int E){
    #pragma omp parallel for schedule(static) if(N*E>=CPU_PMIN)
    for(int n=0;n<N;n++){ const float*r=rt+(size_t)n*E; float mx=-1e30f; for(int e=0;e<E;e++) mx=fmaxf(mx,r[e]);
        float s=0; for(int e=0;e<E;e++) s+=fexp(r[e]-mx); lse[n]=mx+logf(s); }
}
__global__ void k_zl_bwd(float*drt,const float*pr,const float*lse,int N,int E,float c){
    size_t n=(size_t)N*E;
    #pragma omp parallel for simd schedule(static) if(n>=CPU_PMIN)
    for(long i=0;i<(long)n;i++) drt[i]+=c*lse[(size_t)i/E]*pr[i];
}
#endif /* part 12 */

/* ============== part 10: Muon helpers ===================================== */
#if CPUK_PART==10
CPU_EW(k_axpy,(float*Y,const float*X,float alpha,size_t n),Y[i]+=alpha*X[i])
#endif
#if CPUK_PART==11
__global__ void k_mu_gather(const float*G,float*Mo,float*X,size_t off,long long stride,int mn,size_t tot,float mom){
    #pragma omp parallel for schedule(static) if(tot>=CPU_PMIN)
    for(long i=0;i<(long)tot;i++){ size_t b=(size_t)i/(size_t)mn, j=(size_t)i-b*(size_t)mn, src=off+b*(size_t)stride+j;
        float g=G[src], m=mom*Mo[src]+g; Mo[src]=m; X[i]=g+mom*m; }
}
__global__ void k_mu_apply(float*Pp,const float*X,size_t off,long long stride,int mn,size_t tot,float lr,float scale,float wd){
    #pragma omp parallel for schedule(static) if(tot>=CPU_PMIN)
    for(long i=0;i<(long)tot;i++){ size_t b=(size_t)i/(size_t)mn, j=(size_t)i-b*(size_t)mn, dst=off+b*(size_t)stride+j;
        float p=Pp[dst]; Pp[dst]=p-lr*(scale*X[i]+wd*p); }
}
__global__ void k_fro_inv(const float*X,int mn,float*inv){
    int nb=(int)g_kgrid.x;
    #pragma omp parallel for schedule(static)
    for(int b=0;b<nb;b++){ const float*x=X+(size_t)b*mn; double s=0;
        #pragma omp simd reduction(+:s)
        for(int i=0;i<mn;i++) s+=(double)x[i]*x[i];
        inv[b]=1.0f/(sqrtf((float)s)+1e-7f); }
}
__global__ void k_scale_rows(float*X,int mn,const float*inv,size_t tot){
    #pragma omp parallel for simd schedule(static) if(tot>=CPU_PMIN)
    for(long i=0;i<(long)tot;i++) X[i]*=inv[(size_t)i/(size_t)mn];
}
#endif

/* ============== qlm.cuh part A: encode / render / Integer AdamW ============= */
#if CPUK_PART==21
static inline void cpu_qlm_span(const QSpanD*sp,int nspan,int gi,int group,int*n,unsigned long long*j0,int*si){
    *si=qspan_of(sp,nspan,gi); *j0=(unsigned long long)(gi-sp[*si].gi0)*group;
    unsigned long long left=sp[*si].n-*j0; *n=(int)((unsigned long long)group<left?(unsigned long long)group:left);
}
__global__ void k_qlm_encode(const float*P,signed char*d,signed char*r,float*s,const QSpanD*sp,int nspan,int group,int dmax,int packed){
    int ng=(int)g_kgrid.x;
    #pragma omp parallel for schedule(static)
    for(int gi=0;gi<ng;gi++){ int n,si; unsigned long long j0; cpu_qlm_span(sp,nspan,gi,group,&n,&j0,&si);
        unsigned long long pof=sp[si].pof+j0, qi0=sp[si].qi0+j0;
        float m=0; for(int j=0;j<n;j++) m=fmaxf(m,fabsf(P[pof+j]));
        float sg=m/(float)dmax; if(!(sg>0.0f)) sg=1e-8f; s[gi]=sg;
        signed char shd[256];
        for(int j=0;j<group;j++){ if(j>=n){ shd[j]=0; continue; }
            float t=P[pof+j]/sg; int dd=qclmp((int)rintf(t),-dmax,dmax), rr=qclmp((int)rintf((t-(float)dd)*128.0f),-64,64);
            shd[j]=(signed char)dd; r[qi0+j]=(signed char)rr; }
        for(int j=0;j<group;j++) qd_write_group(d,qi0,shd,n,j,packed); }
}
__global__ void k_qlm_render(float*P,const signed char*d,const float*s,const QSpanD*sp,int nspan,int group,int packed,int gi_base){
    int ng=(int)g_kgrid.x;
    #pragma omp parallel for schedule(static)
    for(int w=0;w<ng;w++){ int gi=gi_base+w, n,si; unsigned long long j0; cpu_qlm_span(sp,nspan,gi,group,&n,&j0,&si);
        float*p=P+sp[si].pof+j0; unsigned long long qi0=sp[si].qi0+j0; float sg=s[gi];
        for(int j=0;j<n;j++) p[j]=sg*(float)qd_get(d,qi0+j,packed); }
}
/* one group of Integer AdamW, op-for-op the CUDA k_qlm_step */
static inline void cpu_qlm_grp(int gi,float*P,const float*G,signed char*d,signed char*r,float*s,signed char*m8,float*m8s,
                               unsigned char*v8,float*v8s,const QSpanD*sp,int nspan,int group,int dmax,
                               float lr,float b1,float b2,float wd,float gs,float inv_bc1,float inv_sqrt_bc2,int lrmode,
                               unsigned long long t,int adam,int packed,long long*st){
    int n,si; unsigned long long j0; cpu_qlm_span(sp,nspan,gi,group,&n,&j0,&si);
    unsigned long long pof=sp[si].pof+j0, qi0=sp[si].qi0+j0, gof=sp[si].gof+j0;
    float sg=s[gi], L=lr*128.0f/(lrmode?sg:sp[si].sref), dec=lr*wd*128.0f;
    float ms_old=m8s[gi], vs_old=v8s?v8s[gi]:0.0f, mn[256], rv[256], am=0, vm=0; signed char shd[256];
    for(int j=0;j<group;j++) shd[j]=0;
    for(int j=0;j<n;j++){
        unsigned long long qi=qi0+j; float gr=G[gof+j]*gs, mv=(float)m8[qi]*ms_old, u; rv[j]=0;
        if(adam){ float sv=qv_dec((float)v8[qi],vs_old), vv=sv*sv;
            mn[j]=b1*mv+(1.0f-b1)*gr; vv=b2*vv+(1.0f-b2)*gr*gr; rv[j]=sqrtf(vv);
            float veps=fmaxf(qv_dec(1.0f,vs_old),1e-8f); u=(mn[j]*inv_bc1)/(rv[j]*inv_sqrt_bc2+veps);
            u=fmaxf(-8.0f,fminf(8.0f,u)); rv[j]=qv_enc(rv[j]);
        } else { float cc=b1*mv+(1.0f-b1)*gr; u=(float)((cc>0.0f)-(cc<0.0f)); mn[j]=b2*mv+(1.0f-b2)*gr; }
        int dd=qd_get(d,qi,packed), idl=qsri(-(L*u+dec*(float)dd),qi,t), rr=(int)r[qi]+idl;
        int c=(rr>64)?(((rr-65)>>7)+1):(rr<-64)?-((((-rr)-65)>>7)+1):0;
        rr-=c*128; dd+=c; dd=qclmp(dd,-dmax,dmax);
        shd[j]=(signed char)dd; r[qi]=(signed char)rr; P[pof+j]=sg*(float)dd;
        if(st){ int ai=idl<0?-idl:idl; if(ai>st[3]) st[3]=ai; st[0]+=c<0?-c:c; st[1]+=(dd==dmax)|(dd==-dmax); st[2]+=(dd==0); }
        am=fmaxf(am,fabsf(mn[j])); vm=fmaxf(vm,rv[j]); }
    for(int j=0;j<group;j++) qd_write_group(d,qi0,shd,n,j,packed);
    float ms=am>0.0f?am/127.0f:1e-12f, vs=vm>0.0f?vm/255.0f:1e-12f;
    m8s[gi]=ms; if(adam) v8s[gi]=vs;
    for(int j=0;j<n;j++){ unsigned long long qi=qi0+j;
        m8[qi]=(signed char)qclmp(qsri(mn[j]/ms,qi,t+0x5EEDULL),-127,127);
        if(adam){ int vq=qclmp(qsri(rv[j]/vs,qi,t+0xBEEFULL),0,255); if(vq==0&&rv[j]>0.0f) vq=1; v8[qi]=(unsigned char)vq; } }
}
static void cpu_qlm_step(float*P,const float*G,signed char*d,signed char*r,float*s,signed char*m8,float*m8s,
                         unsigned char*v8,float*v8s,const QSpanD*sp,int nspan,int group,int dmax,
                         float lr,float b1,float b2,float wd,float gs,float inv_bc1,float inv_sqrt_bc2,int lrmode,
                         unsigned long long t,int adam,long long*stat,int packed,int gi_base,int ng){
    long long s0=0,s1=0,s2=0,s3=0;
    #pragma omp parallel for schedule(static) reduction(+:s0,s1,s2) reduction(max:s3)
    for(int w=0;w<ng;w++){ long long st[4]={0,0,0,0};
        cpu_qlm_grp(gi_base+w,P,G,d,r,s,m8,m8s,v8,v8s,sp,nspan,group,dmax,lr,b1,b2,wd,gs,inv_bc1,inv_sqrt_bc2,lrmode,t,adam,packed,stat?st:NULL);
        s0+=st[0]; s1+=st[1]; s2+=st[2]; if(st[3]>s3) s3=st[3]; }
    if(stat){ stat[0]+=s0; stat[1]+=s1; stat[2]+=s2; if(s3>stat[3]) stat[3]=s3; }
}
__global__ void k_qlm_step(float*P,const float*G,signed char*d,signed char*r,float*s,signed char*m8,float*m8s,
                           unsigned char*v8,float*v8s,const QSpanD*sp,int nspan,int group,int dmax,
                           float lr,float b1,float b2,float wd,float gscale,float inv_bc1,float inv_sqrt_bc2,int lrmode,
                           unsigned long long t,int adam,long long*stat,int packed,int gi_base){
    cpu_qlm_step(P,G,d,r,s,m8,m8s,v8,v8s,sp,nspan,group,dmax,lr,b1,b2,wd,gscale,inv_bc1,inv_sqrt_bc2,lrmode,t,adam,stat,packed,
                 gi_base,(int)g_kgrid.x);
}
template<int E> __global__ void k_qlm_stepw(float*P,const float*G,signed char*d,signed char*r,float*s,signed char*m8,float*m8s,
                           unsigned char*v8,float*v8s,const QSpanD*sp,int nspan,int dmax,
                           float lr,float b1,float b2,float wd,float gscale,const float*gsp,
                           float inv_bc1,float inv_sqrt_bc2,int lrmode,unsigned long long t,int adam,long long*stat,int packed,
                           int gi_base,int ng){
    cpu_qlm_step(P,G,d,r,s,m8,m8s,v8,v8s,sp,nspan,E*32,dmax,lr,b1,b2,wd,gsp?*gsp:gscale,inv_bc1,inv_sqrt_bc2,lrmode,t,adam,stat,packed,
                 gi_base,ng);
}
#endif /* qlm part A */

/* ============== qlm.cuh part B: scale servo ================================ */
#if CPUK_PART==22
__global__ void k_qlm_servo(signed char*d,signed char*r,float*s,const QSpanD*sp,int nspan,int group,int dmax,
                            float f_hi,float f_lo,int packed){
    int ng=(int)g_kgrid.x;
    #pragma omp parallel for schedule(static)
    for(int gi=0;gi<ng;gi++){ int n,si; unsigned long long j0; cpu_qlm_span(sp,nspan,gi,group,&n,&j0,&si);
        unsigned long long qi0=sp[si].qi0+j0; int tot=0;
        for(int j=0;j<n;j++){ int dd=qd_get(d,qi0+j,packed); tot+=(dd==dmax)|(dd==-dmax); }
        float f=(float)tot/(float)n, sg=s[gi], sn=sg; int act=0;
        if(f>f_hi){ sn=sg*1.09051f; act=1; } else if(f<f_lo){ sn=sg/1.09051f; act=1; }
        if(!act) continue;
        float conv=sg/sn; s[gi]=sn; signed char shd[256];
        for(int j=0;j<group;j++){ if(j>=n){ shd[j]=0; continue; } unsigned long long qi=qi0+j;
            float t=((float)qd_get(d,qi,packed)+(float)r[qi]*(1.0f/128.0f))*conv;
            int dd=qclmp((int)rintf(t),-dmax,dmax), rr=qclmp((int)rintf((t-(float)dd)*128.0f),-64,64);
            shd[j]=(signed char)dd; r[qi]=(signed char)rr; }
        for(int j=0;j<group;j++) qd_write_group(d,qi0,shd,n,j,packed); }
}
#endif /* qlm part B */
#undef CPUK_PART
