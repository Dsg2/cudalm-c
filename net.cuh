/* net.cuh: the small HTTP/JSON toolkit behind `cudalm serve` and `cudalm monitor`, Windows + POSIX.
 * Windows loads ws2_32 at run time (no import lib); POSIX uses BSD sockets + pthreads.
 *   sockets   net_init, net_listen(ip,port), net_accept, net_timeout, net_close, net_send_all
 *   requests  net_read_req[_t]: whole request (headers + Content-Length body) into an SBuf, optional total deadline
 *   replies   net_reply[_x](cs,code,content-type,[extra headers,]data,n)  (HTTP/1.1, Connection: close; closes cs)
 *   threads   nt_thread, nt_lock/nt_unlock (one global mutex), nt_signal/nt_wait(ms) (one auto-reset event)
 *   JSON      SBuf + sb_put/sb_str/sb_esc writers, js_str/js_key/js_num readers (flat, forgiving) */
#ifndef CUDALM_NET_CUH
#define CUDALM_NET_CUH
#ifdef _WIN32
typedef uintptr_t SK;
#define NET_BAD ((SK)~(uintptr_t)0)
static struct { int (WINAPI*startup)(WORD,void*); SK (WINAPI*sock)(int,int,int); int (WINAPI*bind_)(SK,const void*,int);
    int (WINAPI*listen_)(SK,int); SK (WINAPI*accept_)(SK,void*,int*); int (WINAPI*recv_)(SK,char*,int,int);
    int (WINAPI*send_)(SK,const char*,int,int); int (WINAPI*close_)(SK); int (WINAPI*sopt)(SK,int,int,const char*,int); } W;
static void net_init(void){
    if(W.sock) return;
    HMODULE h=LoadLibraryA("ws2_32.dll"); if(!h){ fprintf(stderr,"no ws2_32.dll\n"); exit(1); }
    *(FARPROC*)&W.startup=GetProcAddress(h,"WSAStartup"); *(FARPROC*)&W.sock=GetProcAddress(h,"socket");
    *(FARPROC*)&W.bind_=GetProcAddress(h,"bind"); *(FARPROC*)&W.listen_=GetProcAddress(h,"listen");
    *(FARPROC*)&W.accept_=GetProcAddress(h,"accept"); *(FARPROC*)&W.recv_=GetProcAddress(h,"recv");
    *(FARPROC*)&W.send_=GetProcAddress(h,"send"); *(FARPROC*)&W.close_=GetProcAddress(h,"closesocket");
    *(FARPROC*)&W.sopt=GetProcAddress(h,"setsockopt");
    char wd[512]; W.startup(0x0202,wd);
}
static SK  net_sock(void){ return W.sock(2,1,6); }
static int net_bind(SK s,const void*sa){ return W.bind_(s,sa,16)||W.listen_(s,256); }
static SK  net_accept(SK s){ return W.accept_(s,NULL,NULL); }
static int net_recv(SK s,char*b,int n){ return W.recv_(s,b,n,0); }
static int net_send(SK s,const char*b,int n){ return W.send_(s,b,n,0); }
static void net_close(SK s){ W.close_(s); }
static void net_timeout(SK s,int ms){ DWORD v=(DWORD)ms; W.sopt(s,0xffff,0x1006,(const char*)&v,4); W.sopt(s,0xffff,0x1005,(const char*)&v,4); }
static CRITICAL_SECTION g_ntc; static HANDLE g_nte;
static void nt_init(void){ InitializeCriticalSection(&g_ntc); g_nte=CreateEventA(NULL,FALSE,FALSE,NULL); }
static void nt_lock(void){ EnterCriticalSection(&g_ntc); }
static void nt_unlock(void){ LeaveCriticalSection(&g_ntc); }
static void nt_signal(void){ SetEvent(g_nte); }
static void nt_wait(int ms){ WaitForSingleObject(g_nte,(DWORD)ms); }
typedef struct { void(*f)(void*); void*u; } NtArg;
static DWORD WINAPI nt_tramp(void*p){ NtArg a=*(NtArg*)p; free(p); a.f(a.u); return 0; }
static void nt_thread(void(*f)(void*),void*u){ NtArg*a=(NtArg*)malloc(sizeof *a); a->f=f; a->u=u; CloseHandle(CreateThread(NULL,0,nt_tramp,a,0,NULL)); }
#else
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
typedef int SK;
#define NET_BAD (-1)
static void net_init(void){ signal(SIGPIPE,SIG_IGN); }
static SK  net_sock(void){ SK s=socket(AF_INET,SOCK_STREAM,IPPROTO_TCP); int one=1; if(s>=0) setsockopt(s,SOL_SOCKET,SO_REUSEADDR,&one,sizeof one); return s; }
static int net_bind(SK s,const void*sa){ return bind(s,(const struct sockaddr*)sa,16)||listen(s,256); }
static SK  net_accept(SK s){ return accept(s,NULL,NULL); }
static int net_recv(SK s,char*b,int n){ return (int)recv(s,b,(size_t)n,0); }
static int net_send(SK s,const char*b,int n){ return (int)send(s,b,(size_t)n,MSG_NOSIGNAL); }
static void net_close(SK s){ close(s); }
static void net_timeout(SK s,int ms){ struct timeval v={ms/1000,(ms%1000)*1000};
    setsockopt(s,SOL_SOCKET,SO_RCVTIMEO,&v,sizeof v); setsockopt(s,SOL_SOCKET,SO_SNDTIMEO,&v,sizeof v); }
static pthread_mutex_t g_ntc=PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_ntv=PTHREAD_COND_INITIALIZER; static int g_ntf;
static void nt_init(void){}
static void nt_lock(void){ pthread_mutex_lock(&g_ntc); }
static void nt_unlock(void){ pthread_mutex_unlock(&g_ntc); }
static void nt_signal(void){ pthread_mutex_lock(&g_ntc); g_ntf=1; pthread_cond_signal(&g_ntv); pthread_mutex_unlock(&g_ntc); }
static void nt_wait(int ms){ struct timespec t; clock_gettime(CLOCK_REALTIME,&t); t.tv_nsec+=(long)(ms%1000)*1000000L; t.tv_sec+=ms/1000+t.tv_nsec/1000000000L; t.tv_nsec%=1000000000L;
    pthread_mutex_lock(&g_ntc); while(!g_ntf) if(pthread_cond_timedwait(&g_ntv,&g_ntc,&t)) break; g_ntf=0; pthread_mutex_unlock(&g_ntc); }
typedef struct { void(*f)(void*); void*u; } NtArg;
static void* nt_tramp(void*p){ NtArg a=*(NtArg*)p; free(p); a.f(a.u); return NULL; }
static void nt_thread(void(*f)(void*),void*u){ NtArg*a=(NtArg*)malloc(sizeof *a); a->f=f; a->u=u; pthread_t t; pthread_create(&t,NULL,nt_tramp,a); pthread_detach(t); }
#endif
/* "a.b.c.d" (or NULL = 127.0.0.1) + port -> listening socket, NET_BAD on failure */
static SK net_listen(const char*ip,int port){
    unsigned q[4]={127,0,0,1}; if(ip&&sscanf(ip,"%u.%u.%u.%u",q,q+1,q+2,q+3)!=4) return NET_BAD;
    if(q[0]>255||q[1]>255||q[2]>255||q[3]>255||port<1||port>65535) return NET_BAD;
    struct { unsigned short f, p; unsigned a; char z[8]; } sa={2,(unsigned short)((port>>8)|((port&255)<<8)),0,{0}};   /* sockaddr_in, network order */
    unsigned char*b=(unsigned char*)&sa.a; b[0]=(unsigned char)q[0]; b[1]=(unsigned char)q[1]; b[2]=(unsigned char)q[2]; b[3]=(unsigned char)q[3];
    net_init(); SK s=net_sock(); if(s==NET_BAD) return NET_BAD;
    if(net_bind(s,&sa)){ net_close(s); return NET_BAD; }
    return s; }

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

static long long net_ms(void){
#ifdef _WIN32
    return (long long)GetTickCount64();
#else
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (long long)t.tv_sec*1000+t.tv_nsec/1000000;
#endif
}
/* read one request: headers + Content-Length bytes. Returns the body pointer inside rq->p, NULL if the peer closed early,
 * timed out, or sent more than maxb bytes (0 = no cap). ms > 0 bounds the whole read (a trickling client cannot hold it). */
static const char* net_read_req_t(SK cs,SBuf*rq,size_t maxb,int ms){
    char buf[65536]; int k; size_t need=0; long long t0=net_ms();
    for(;;){
        if(ms){ long long left=ms-(net_ms()-t0); if(left<=0) return NULL; net_timeout(cs,(int)left); }
        if((k=net_recv(cs,buf,sizeof buf))<=0) return NULL;
        sb_put(rq,buf,(size_t)k);
        if(maxb&&rq->n>maxb) return NULL;
        char*h=strstr(rq->p,"\r\n\r\n");
        if(h){ const char*cl=strstr(rq->p,"Content-Length:"); if(!cl) cl=strstr(rq->p,"content-length:");
            need=(size_t)(h-rq->p)+4+(cl&&cl<h?(size_t)atol(cl+15):0);
            if(maxb&&need>maxb) return NULL;
            if(rq->n>=need) return h+4; } } }
static const char* net_read_req(SK cs,SBuf*rq,size_t maxb){ return net_read_req_t(cs,rq,maxb,0); }
static void net_send_all(SK cs,const char*p,size_t n){ for(size_t q=0;q<n;){ int s=net_send(cs,p+q,(int)(n-q)); if(s<=0) break; q+=(size_t)s; } }
/* extra = more header lines, each ending in \r\n ("" for none) */
static void net_reply_x(SK cs,const char*code,const char*ctype,const char*extra,const char*p,size_t n){
    char h[1024]; int hl=snprintf(h,sizeof h,"HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n%sConnection: close\r\n\r\n",code,ctype,n,extra);
    if(hl<0||hl>=(int)sizeof h) hl=(int)strlen(h);
    net_send(cs,h,hl); net_send_all(cs,p,n); net_close(cs); }
static void net_reply(SK cs,const char*code,const char*ctype,const char*p,size_t n){ net_reply_x(cs,code,ctype,"",p,n); }
#endif
