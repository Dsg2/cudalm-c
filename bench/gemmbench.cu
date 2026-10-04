/* MoE expert-GEMM microbench: how much do the small per-expert GEMMs lose against
 * bigger / uniform / batched shapes?  s8w: N=16384 rows per pass, D=F=384, E=32.
 * ms are per call set (one pass, all experts), TF32 like the trainer. */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#define CK(x) do{ int e_=(int)(x); if(e_){ printf("err %d line %d\n",e_,__LINE__); exit(1);} }while(0)
static double wall(){ LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart/f.QuadPart; }
enum { N=16384, D=384, F=384, E=32, REP=30, MAXP=2048 };
static cublasHandle_t H; static float *X,*W,*Y,*G,ONE=1.0f,ZERO=0.0f;
static const float **dA,**dB; static float **dC;
static int cmpd(const void*a,const void*b){ return *(const int*)b-*(const int*)a; }
/* op 0 fwd Y=X@W, 1 dgrad X=Y@W^T, 2 wgrad G+=X^T@Y.  w[i] = weight index of problem i */
static double grouped(int op,const int*sz,const int*w,int n,int merge){
    static cublasOperation_t ta[MAXP],tb[MAXP]; static int m[MAXP],nn[MAXP],k[MAXP],la[MAXP],lb[MAXP],lc[MAXP],gs[MAXP];
    static float al[MAXP],be[MAXP]; static const float*A[MAXP],*B[MAXP]; static float*C[MAXP];
    int g=0,off=0;
    for(int i=0;i<n;i++){ int ne=sz[i]; const float*We=W+(size_t)w[i]*D*F;
        if(merge&&g&&nn[g-1]==(op==2?D:ne)&&k[g-1]==(op==0?D:op==1?F:ne)) gs[g-1]++;
        else { ta[g]=op==1?CUBLAS_OP_T:CUBLAS_OP_N; tb[g]=op==2?CUBLAS_OP_T:CUBLAS_OP_N;
            m[g]=op==1?D:F; nn[g]=op==2?D:ne; k[g]=op==0?D:op==1?F:ne;
            la[g]=F; lb[g]=op==1?F:D; lc[g]=op==1?D:F; al[g]=1; be[g]=op==2?1.0f:0.0f; gs[g]=1; g++; }
        A[i]=op==2?Y+(size_t)off*F:We; B[i]=op==1?Y+(size_t)off*F:X+(size_t)off*D;
        C[i]=op==0?Y+(size_t)off*F:op==1?X+(size_t)off*D:G+(size_t)w[i]*D*F; off+=ne; }
    CK(cudaMemcpy(dA,A,n*sizeof(void*),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB,B,n*sizeof(void*),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dC,C,n*sizeof(void*),cudaMemcpyHostToDevice));
    cudaDeviceSynchronize(); double t=wall();
    for(int r=0;r<REP;r++) CK(cublasSgemmGroupedBatched(H,ta,tb,m,nn,k,al,dA,la,dB,lb,be,dC,lc,g,gs));
    cudaDeviceSynchronize(); return (wall()-t)*1000/REP;
}
static double serial(int op,const int*sz,int n){
    cudaDeviceSynchronize(); double t=wall();
    for(int r=0;r<REP;r++){ int off=0; for(int i=0;i<n;i++){ int ne=sz[i]; const float*We=W+(size_t)(i%E)*D*F;
        if(op==0) CK(cublasSgemm(H,CUBLAS_OP_N,CUBLAS_OP_N,F,ne,D,&ONE,We,F,X+(size_t)off*D,D,&ZERO,Y+(size_t)off*F,F));
        else if(op==1) CK(cublasSgemm(H,CUBLAS_OP_T,CUBLAS_OP_N,D,ne,F,&ONE,We,F,Y+(size_t)off*F,F,&ZERO,X+(size_t)off*D,D));
        else CK(cublasSgemm(H,CUBLAS_OP_N,CUBLAS_OP_T,F,D,ne,&ONE,Y+(size_t)off*F,F,X+(size_t)off*D,D,&ONE,G+(size_t)(i%E)*D*F,F));
        off+=ne; } }
    cudaDeviceSynchronize(); return (wall()-t)*1000/REP;
}
static double strided(int op,int rows,int n){
    cudaDeviceSynchronize(); double t=wall();
    for(int r=0;r<REP;r++){
        if(op==0) CK(cublasSgemmStridedBatched(H,CUBLAS_OP_N,CUBLAS_OP_N,F,rows,D,&ONE,W,F,(long long)D*F,X,D,(long long)rows*D,&ZERO,Y,F,(long long)rows*F,n));
        else if(op==1) CK(cublasSgemmStridedBatched(H,CUBLAS_OP_T,CUBLAS_OP_N,D,rows,F,&ONE,W,F,(long long)D*F,Y,F,(long long)rows*F,&ZERO,X,D,(long long)rows*D,n));
        else CK(cublasSgemmStridedBatched(H,CUBLAS_OP_N,CUBLAS_OP_T,F,D,rows,&ONE,Y,F,(long long)rows*F,X,D,(long long)rows*D,&ONE,G,F,(long long)D*F,n)); }
    cudaDeviceSynchronize(); return (wall()-t)*1000/REP;
}
/* serial wgrad launched from T host threads, each its own handle + stream; fwd=1: the
 * main thread meanwhile runs REP grouped fwd sets on the default stream (overlap test) */
typedef struct { int t,T; const int*sz; cublasHandle_t h; } TA;
static DWORD WINAPI wg_thr(LPVOID p){ TA*a=(TA*)p;
    for(int r=0;r<REP;r++){ int off=0; for(int i=0;i<E;i++){ int ne=a->sz[i];
        if(i%a->T==a->t) CK(cublasSgemm(a->h,CUBLAS_OP_N,CUBLAS_OP_T,F,D,ne,&ONE,Y+(size_t)off*F,F,X+(size_t)off*D,D,&ONE,G+(size_t)i*D*F,F));
        off+=ne; } }
    return 0; }
static double threaded(const int*sz,int T,int fwd,int nb){        /* nb: non-blocking streams (no implicit sync with stream 0) */
    static TA aa[2][8]; static cudaStream_t st[2][8]; HANDLE th[8]; TA*a=aa[nb];
    for(int t=0;t<T;t++) if(!a[t].h){ CK(cublasCreate(&a[t].h)); CK(cublasSetMathMode(a[t].h,CUBLAS_TF32_TENSOR_OP_MATH));
        CK(cudaStreamCreateWithFlags(&st[nb][t],nb?cudaStreamNonBlocking:cudaStreamDefault)); CK(cublasSetStream(a[t].h,st[nb][t])); }
    cudaDeviceSynchronize(); double t0=wall();
    for(int t=0;t<T;t++){ a[t].t=t; a[t].T=T; a[t].sz=sz; th[t]=CreateThread(0,0,wg_thr,&a[t],0,0); }
    if(fwd) for(int r=0;r<REP;r++){ int off=0; for(int i=0;i<E;i++){ int ne=sz[i];
        CK(cublasSgemm(H,CUBLAS_OP_N,CUBLAS_OP_N,F,ne,D,&ONE,W+(size_t)i*D*F,F,X+(size_t)(N+off)*D,D,&ZERO,Y+(size_t)(N+off)*F,F)); off+=ne; } }
    WaitForMultipleObjects(T,th,TRUE,INFINITE); for(int t=0;t<T;t++) CloseHandle(th[t]);
    cudaDeviceSynchronize(); return (wall()-t0)*1000/REP;
}
static void mksz(int*sz,int n,int tot,double cv,unsigned seed){      /* sums to tot, multiples of 1 */
    double w[MAXP],s=0; srand(seed);
    for(int i=0;i<n;i++){ double u=0; for(int j=0;j<12;j++) u+=rand()/(double)RAND_MAX; u-=6; w[i]=exp(cv*u); s+=w[i]; }
    int a=0; for(int i=0;i<n;i++){ sz[i]=(int)(tot*w[i]/s); a+=sz[i]; } sz[0]+=tot-a;
}
int main(){
    CK(cublasCreate(&H)); CK(cublasSetMathMode(H,CUBLAS_TF32_TENSOR_OP_MATH));
    size_t nx=(size_t)2*N*D*5/4, nw=(size_t)E*D*F;
    CK(cudaMalloc(&X,nx*4)); CK(cudaMalloc(&Y,nx*4)); CK(cudaMalloc(&W,nw*4)); CK(cudaMalloc(&G,nw*4));
    CK(cudaMalloc(&dA,MAXP*sizeof(void*))); CK(cudaMalloc(&dB,MAXP*sizeof(void*))); CK(cudaMalloc(&dC,MAXP*sizeof(void*)));
    float*h=(float*)malloc(nx*4); for(size_t i=0;i<nx;i++) h[i]=(rand()/(float)RAND_MAX-0.5f)*0.1f;
    CK(cudaMemcpy(X,h,nx*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(Y,h,nx*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(W,h,nw*4,cudaMemcpyHostToDevice)); CK(cudaMemset(G,0,nw*4));
    int var[MAXP],skew[MAXP],eq[MAXP],one[1]={N},mg[MAXP],id[MAXP],pad[MAXP],tile[MAXP],tw[MAXP],srt[MAXP];
    for(int i=0;i<MAXP;i++) id[i]=i%E;
    mksz(var,E,N,0.35,1); mksz(skew,E,N,1.0,2); mksz(mg,E,2*N,0.35,3); for(int i=0;i<E;i++) eq[i]=N/E;
    int mx=0,mxs=0; for(int i=0;i<E;i++){ if(var[i]>mx)mx=var[i]; if(skew[i]>mxs)mxs=skew[i]; }
    printf("N=%d D=%d F=%d E=%d | var max/avg %.2f  skew max/avg %.2f | GFLOP per set %.1f\n",N,D,F,E,mx/(N/(double)E),mxs/(N/(double)E),2.0*N*D*F/1e9);
    const char*on[3]={"fwd  NN","dgrad TN","wgrad NT"};
    for(int w=0;w<2;w++) for(int op=0;op<3;op++){                 /* w=0 warms up */
        double r[16]; int q=0;
        r[q++]=serial(op,one,1);                                  /* dense: upper bound */
        r[q++]=serial(op,var,E);
        r[q++]=grouped(op,var,id,E,0);
        r[q++]=grouped(op,skew,id,E,0);
        r[q++]=grouped(op,eq,id,E,0);
        r[q++]=grouped(op,eq,id,E,1);
        r[q++]=strided(op,N/E,E);
        r[q++]=grouped(op,mg,id,E,0)/2;                           /* merged top-2 passes, per pass */
        r[q++]=serial(op,mg,E)/2;
        for(int t=64;t<=256;t*=2){                                /* pad to a tile, group equal sizes */
            int tot=0; for(int i=0;i<E;i++){ pad[i]=(var[i]+t-1)/t*t; tot+=pad[i]; }
            memcpy(srt,pad,E*4); qsort(srt,E,4,cmpd); r[q++]=grouped(op,srt,id,E,1);
            if(w&&op==0) printf("   pad%-3d rows %.3fx\n",t,tot/(double)N); }
        if(op<2) for(int t=64;t<=256;t*=2){                       /* uniform tiles, one batched call */
            int nt=0; for(int i=0;i<E;i++) for(int j=0;j<(var[i]+t-1)/t;j++){ tile[nt]=t; tw[nt++]=i; }
            r[q++]=grouped(op,tile,tw,nt,1); }
        if(!w) continue;
        printf("%s | dense %.3f | serial %.3f | grouped %.3f (skew %.3f) | equal %.3f, 1 group %.3f | strided %.3f | merged/2: grouped %.3f serial %.3f |"
               " pad+group 64/128/256: %.3f %.3f %.3f",on[op],r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9],r[10],r[11]);
        if(op<2) printf(" | tiles 64/128/256: %.3f %.3f %.3f",r[12],r[13],r[14]);
        printf("  ms\n");
    }
    int real[MAXP]; mksz(real,E,N,0.6,5); int mr=0; for(int i=0;i<E;i++) if(real[i]>mr) mr=real[i];
    for(int w=0;w<2;w++){ double r[12]; int q=0;
        r[q++]=serial(2,real,E); r[q++]=grouped(2,real,id,E,0);
        r[q++]=threaded(real,1,0,1); r[q++]=threaded(real,2,0,1); r[q++]=threaded(real,3,0,1); r[q++]=threaded(real,4,0,1);
        r[q++]=serial(0,real,E); r[q++]=threaded(real,1,1,1); r[q++]=threaded(real,2,1,1); r[q++]=threaded(real,1,1,0); r[q++]=threaded(real,2,1,0);
        if(w) printf("wgrad, max/avg %.2f | serial %.3f | grouped %.3f | threads 1/2/3/4: %.3f %.3f %.3f %.3f |"
            " serial fwd alone %.3f, + wgrad on 1/2 threads: non-blocking %.3f %.3f, blocking streams %.3f %.3f  ms\n",
            mr/(N/(double)E),r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9],r[10]); }
    return 0;
}
