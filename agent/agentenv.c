/* agentenv: tool harness + E-code / E-tool evals + teacher task/trajectory generation for cudalm.
 * Talks to any llama-server-style POST /completion (cudalm serve, mcpie, llama-server) on 127.0.0.1.
 *
 *   agentenv try   "<task>"            [opts]   one tool episode, printed
 *   agentenv tool  <tasks.jsonl> <out.jsonl> [opts]   E-tool: success, valid-call %, hallucinated-result %,
 *                                                     unnecessary-call %, missing-call %, per category
 *   agentenv code  <tasks.jsonl> <out.jsonl> [opts]   E-code: pass@1, pass@5, syntax-valid %
 *   agentenv gen-code <out.jsonl> <n>  [opts]   teacher writes tasks; kept if its solution passes and a stub fails
 *   agentenv gen-tool <out.jsonl> <n>  [opts]   lookup (needs -store), compute, file/multi, none, recover
 *   agentenv gen-steps <out.jsonl> <n> [-seed S]   procedural step-by-step calc trajectories for SFT (no server)
 *   agentenv traj  <tasks.jsonl> <out.jsonl> [opts]   teacher rollouts -> verified trajectories (student format)
 *   agentenv mkstore <in.jsonl|txt> <store.tsv> [words=120]  "text" docs -> title\tpassage lines for search
 * opts: -port N (8090) -teacher (wrap with the server's /apply-template chat template) -temp T -n samples
 *       -seed S -store store.tsv -steps N (6) -max N tokens per step (384) -v
 *       -student (traj: roll out the student format against a cudalm serve, for rejection-sampling SFT)
 *       -traj (gen-tool: also store the gating rollout as "traj", student format, for SFT)
 * Env: AGENT_PYTHON (python). Tool-call text format (the same for student and teacher):
 *   <|call|>NAME\nINPUT\n<|end|>   ->   <|result|>\nOUTPUT\n<|end|>
 * Build: gcc -O2 -march=haswell -static -o agentenv.exe agentenv.c -lws2_32 */
#include <winsock2.h>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>

/* ---------------- buffers, JSON ---------------- */
typedef struct { char *p; size_t n, cap; } Buf;
static void bput(Buf *b, const char *s, size_t n){
    if(b->n+n+1>b->cap){ b->cap=(b->n+n+1)*2; b->p=realloc(b->p,b->cap); }
    memcpy(b->p+b->n,s,n); b->n+=n; b->p[b->n]=0;
}
static void bstr(Buf *b, const char *s){ bput(b,s,strlen(s)); }
static void bfmt(Buf *b, const char *f, ...){ char t[4096]; va_list a; va_start(a,f); int n=vsnprintf(t,sizeof t,f,a); va_end(a); bput(b,t,n<(int)sizeof t?n:(int)sizeof t-1); }
static void besc(Buf *b, const char *s){
    for(const unsigned char *c=(const unsigned char*)s; *c; c++){ char t[8];
        if(*c=='"'||*c=='\\'){ t[0]='\\'; t[1]=*c; bput(b,t,2); }
        else if(*c=='\n') bput(b,"\\n",2); else if(*c=='\r') bput(b,"\\r",2); else if(*c=='\t') bput(b,"\\t",2);
        else if(*c<0x20) bput(b,t,sprintf(t,"\\u%04x",*c)); else bput(b,(const char*)c,1);
    }
}
typedef struct J { char t; char *s; double n; int len; struct J **v; char **k; } J;   /* t: o a s n b z */
static void jws(const char **p){ while(**p==' '||**p=='\n'||**p=='\r'||**p=='\t') (*p)++; }
static void utf8(Buf *b, unsigned v){
    char t[4]; int k=0;
    if(v<0x80) t[k++]=(char)v; else if(v<0x800){ t[k++]=(char)(0xC0|v>>6); t[k++]=(char)(0x80|(v&63)); }
    else if(v<0x10000){ t[k++]=(char)(0xE0|v>>12); t[k++]=(char)(0x80|((v>>6)&63)); t[k++]=(char)(0x80|(v&63)); }
    else { t[k++]=(char)(0xF0|v>>18); t[k++]=(char)(0x80|((v>>12)&63)); t[k++]=(char)(0x80|((v>>6)&63)); t[k++]=(char)(0x80|(v&63)); }
    bput(b,t,k);
}
static char *jrawstr(const char **p){            /* *p at the opening quote */
    Buf b={0}; bput(&b,"",0); (*p)++;
    for(; **p && **p!='"'; (*p)++){
        if(**p!='\\'){ bput(&b,*p,1); continue; }
        char c=*++(*p);
        if(c=='u'){ char h[5]={(*p)[1],(*p)[2],(*p)[3],(*p)[4],0}; unsigned v=(unsigned)strtoul(h,0,16); *p+=4;
            if(v>=0xD800&&v<0xDC00&&(*p)[1]=='\\'&&(*p)[2]=='u'){ char l[5]={(*p)[3],(*p)[4],(*p)[5],(*p)[6],0};
                v=0x10000+((v-0xD800)<<10)+((unsigned)strtoul(l,0,16)-0xDC00); *p+=6; }
            utf8(&b,v); continue; }
        c= c=='n'?'\n':c=='t'?'\t':c=='r'?'\r':c=='b'?'\b':c=='f'?'\f':c; bput(&b,&c,1);
    }
    if(**p) (*p)++; return b.p;
}
static J *jparse(const char **p){
    jws(p); J *j=calloc(1,sizeof(J));
    if(**p=='{'||**p=='['){ int o=**p=='{'; j->t=o?'o':'a'; (*p)++;
        for(;;){ jws(p); if(**p=='}'||**p==']'){ (*p)++; break; } if(!**p) break;
            if(**p==','){ (*p)++; continue; }
            j->v=realloc(j->v,sizeof(J*)*(j->len+1)); j->k=realloc(j->k,sizeof(char*)*(j->len+1)); j->k[j->len]=NULL;
            if(o){ j->k[j->len]=jrawstr(p); jws(p); if(**p==':') (*p)++; }
            j->v[j->len++]=jparse(p); }
    } else if(**p=='"'){ j->t='s'; j->s=jrawstr(p); }
    else if(!strncmp(*p,"true",4)){ j->t='b'; j->n=1; *p+=4; } else if(!strncmp(*p,"false",5)){ j->t='b'; *p+=5; }
    else if(!strncmp(*p,"null",4)){ j->t='z'; *p+=4; }
    else { j->t='n'; char *e; j->n=strtod(*p,&e); if(e==*p) e++; *p=e; }
    return j;
}
static J *jload(const char *s){ const char *p=s; return jparse(&p); }
static void jfree(J *j){ if(!j) return; for(int i=0;i<j->len;i++){ jfree(j->v[i]); if(j->k) free(j->k[i]); } free(j->v); free(j->k); free(j->s); free(j); }
static J *jget(J *j, const char *k){ if(!j||j->t!='o') return NULL; for(int i=0;i<j->len;i++) if(!strcmp(j->k[i],k)) return j->v[i]; return NULL; }
static const char *jgs(J *j, const char *k, const char *d){ J *v=jget(j,k); return v&&v->t=='s'?v->s:d; }
static double jgn(J *j, const char *k, double d){ J *v=jget(j,k); return v&&(v->t=='n'||v->t=='b')?v->n:d; }

/* ---------------- options ---------------- */
static int o_port=8090, o_teacher=0, o_n=1, o_steps=6, o_max=384, o_v=0, o_trajout=0, o_student=0, o_all=0, o_mix[5]={70,60,70,50,50}; static double o_temp=-1; static unsigned long long o_seed=1;
static const char *o_store=NULL;
static const char *PY(void){ const char *e=getenv("AGENT_PYTHON"); return e&&*e?e:"python"; }

/* ---------------- HTTP ---------------- */
static char *post(const char *path, const char *body){
    for(int tries=0; tries<60; tries++){
        SOCKET s=socket(AF_INET,SOCK_STREAM,0); struct sockaddr_in a={0};
        a.sin_family=AF_INET; a.sin_port=htons((u_short)o_port); a.sin_addr.s_addr=inet_addr("127.0.0.1");
        if(connect(s,(struct sockaddr*)&a,sizeof a)){ closesocket(s); Sleep(1000); continue; }
        char h[256]; size_t n=strlen(body);
        int hl=snprintf(h,sizeof h,"POST %s HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                        "Content-Length: %zu\r\nConnection: close\r\n\r\n",path,n);
        send(s,h,hl,0);
        for(size_t o=0;o<n;){ int k=send(s,body+o,(int)(n-o),0); if(k<=0) break; o+=k; }
        Buf r={0}; char tmp[65536]; int k;
        while((k=recv(s,tmp,sizeof tmp,0))>0) bput(&r,tmp,k);
        closesocket(s);
        char *b=r.p?strstr(r.p,"\r\n\r\n"):NULL;
        if(!b){ free(r.p); Sleep(1000); continue; }
        if(strncmp(r.p+9,"200",3)){ fprintf(stderr,"%s: HTTP %.3s: %.300s\n",path,r.p+9,b+4); free(r.p); return NULL; }
        char *o=strdup(b+4); free(r.p); return o;
    }
    fprintf(stderr,"server on port %d unreachable\n",o_port); exit(1);
}
/* continuation of a raw prompt; *stop: 0 eos, 1 stop word (which in *word), 2 length */
static char *complete(const char *prompt, int npred, double temp, unsigned long long seed, const char **stops, int ns, int *stop, char **word){
    Buf b={0}; bstr(&b,"{\"prompt\":\""); besc(&b,prompt);
    bfmt(&b,"\",\"n_predict\":%d,\"temperature\":%.3f,\"top_p\":0.95,\"top_k\":40,\"seed\":%llu,\"stop\":[",npred,temp,seed%2000000000ULL);
    for(int i=0;i<ns;i++){ bstr(&b,i?",\"":"\""); besc(&b,stops[i]); bstr(&b,"\""); }
    bstr(&b,o_teacher?"],\"special\":true}":"]}");
    char *r=post("/completion",b.p); if(o_v>1) fprintf(stderr,"[req] %s\n[resp] %s\n",b.p,r?r:"(null)"); free(b.p);
    J *j=r?jload(r):NULL; char *o=strdup(jgs(j,"content",""));
    const char *st=jgs(j,"stop_type",""); J *sw=jget(j,"stopping_word");
    *stop = !strcmp(st,"eos")||jgn(j,"stopped_eos",0) ? 0 : (!strcmp(st,"word")||jgn(j,"stopped_word",0)) ? 1 : 2;
    if(word) *word = sw&&sw->t=='s'&&*sw->s ? strdup(sw->s) : NULL;
    jfree(j); free(r); return o;
}
/* n samples of one prompt: one batched request when the server returns "results" (cudalm serve), else n requests */
static void complete_n(const char *prompt, int npred, double temp, unsigned long long seed, const char **stops, int ns, int n, char **outs){
    Buf b={0}; bstr(&b,"{\"prompt\":\""); besc(&b,prompt);
    bfmt(&b,"\",\"n\":%d,\"n_predict\":%d,\"temperature\":%.3f,\"top_p\":0.95,\"top_k\":40,\"seed\":%llu,\"stop\":[",n,npred,temp,seed%2000000000ULL);
    for(int i=0;i<ns;i++){ bstr(&b,i?",\"":"\""); besc(&b,stops[i]); bstr(&b,"\""); }
    bstr(&b,o_teacher?"],\"special\":true}":"]}");
    char *r=post("/completion",b.p); free(b.p);
    J *j=r?jload(r):NULL, *res=jget(j,"results"); int got=0;
    if(res&&res->t=='a') for(int i=0;i<res->len&&got<n;i++) outs[got++]=strdup(jgs(res->v[i],"content",""));
    else if(j) outs[got++]=strdup(jgs(j,"content",""));
    jfree(j); free(r);
    for(;got<n;got++){ int st; outs[got]=complete(prompt,npred,temp,seed+got*7919,stops,ns,&st,NULL); }
}
/* chat wrapper: student = corpus markers; teacher = the server's own template (thinking off) */
static char *t_spre,*t_smid,*t_post,*t_pre,*t_tool;
static void get_templates(void){
    char *t=post("/apply-template","{\"messages\":[{\"role\":\"system\",\"content\":\"@@S@@\"},{\"role\":\"user\",\"content\":\"@@P@@\"}],"
                                   "\"chat_template_kwargs\":{\"enable_thinking\":false}}");
    J *j=t?jload(t):NULL; const char *tp=jgs(j,"prompt",NULL); char *s=tp?strdup(tp):NULL;
    char *as=s?strstr(s,"@@S@@"):NULL, *at=s?strstr(s,"@@P@@"):NULL;
    if(!as||!at){ fprintf(stderr,"apply-template failed (is this a chat server?)\n"); exit(1); }
    *as=0; *at=0; t_spre=s; t_smid=as+5; t_post=at+5; jfree(j); free(t);
    t=post("/apply-template","{\"messages\":[{\"role\":\"user\",\"content\":\"@@P@@\"}],\"chat_template_kwargs\":{\"enable_thinking\":false}}");
    j=t?jload(t):NULL; tp=jgs(j,"prompt",""); s=strdup(tp); at=strstr(s,"@@P@@"); if(!at){ fprintf(stderr,"apply-template failed\n"); exit(1); }
    *at=0; t_pre=s; jfree(j); free(t);
    /* tool turn = end-of-turn + role header with "tool" in place of "assistant" (ChatML-style templates) */
    const char *a=strstr(t_post,"assistant"); if(!a){ fprintf(stderr,"template: no assistant header\n"); exit(1); }
    Buf tt={0}; bput(&tt,t_post,a-t_post); bstr(&tt,"tool\n"); t_tool=tt.p;
}
static void wrap(Buf *b, const char *sys, const char *user){
    if(o_teacher){ if(sys){ bstr(b,t_spre); bstr(b,sys); bstr(b,t_smid); } else bstr(b,t_pre); bstr(b,user); bstr(b,t_post); return; }
    if(sys){ bstr(b,"<|system|>\n"); bstr(b,sys); bstr(b,"\n"); }
    bstr(b,"<|user|>\n"); bstr(b,user); bstr(b,"\n<|assistant|>\n");
}

/* ---------------- process + python ---------------- */
static unsigned g_tmpc=0;
static void mktmp(char *dir, size_t n){
    char t[MAX_PATH]; GetTempPathA(sizeof t,t);
    snprintf(dir,n,"%sagentenv_%lu_%u",t,(unsigned long)GetCurrentProcessId(),(unsigned)InterlockedIncrement((LONG*)&g_tmpc)); CreateDirectoryA(dir,NULL);
}
static void rmtree(const char *dir){
    char pat[MAX_PATH+4]; snprintf(pat,sizeof pat,"%s\\*",dir); WIN32_FIND_DATAA fd; HANDLE h=FindFirstFileA(pat,&fd);
    if(h!=INVALID_HANDLE_VALUE){ do{ if(fd.cFileName[0]=='.') continue; char f[MAX_PATH*2]; snprintf(f,sizeof f,"%s\\%s",dir,fd.cFileName);
        if(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY) rmtree(f); else DeleteFileA(f); } while(FindNextFileA(h,&fd)); FindClose(h); }
    RemoveDirectoryA(dir);
}
/* run cmd in cwd, merged stdout+stderr into out (capped), returns exit code, -1 timeout, -2 spawn failure */
static int run_proc(const char *cmd, const char *cwd, int ms, Buf *out, size_t cap){
    SECURITY_ATTRIBUTES sa={sizeof sa,NULL,TRUE}; HANDLE rd,wr; CreatePipe(&rd,&wr,&sa,0); SetHandleInformation(rd,HANDLE_FLAG_INHERIT,0);
    HANDLE nul=CreateFileA("NUL",GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,&sa,OPEN_EXISTING,0,NULL);
    STARTUPINFOA si={sizeof si}; si.dwFlags=STARTF_USESTDHANDLES; si.hStdInput=nul; si.hStdOutput=wr; si.hStdError=wr;
    PROCESS_INFORMATION pi; char *c=strdup(cmd);
    BOOL ok=CreateProcessA(NULL,c,NULL,NULL,TRUE,CREATE_NO_WINDOW,NULL,cwd,&si,&pi); free(c); CloseHandle(wr); CloseHandle(nul);
    if(!ok){ CloseHandle(rd); return -2; }
    DWORD t0=GetTickCount(), code=0; int to=0; char tmp[4096];
    for(;;){ DWORD av=0; PeekNamedPipe(rd,NULL,0,NULL,&av,NULL);
        if(av){ DWORD k=0; if(ReadFile(rd,tmp,av<sizeof tmp?av:sizeof tmp,&k,NULL)&&k&&out->n<cap) bput(out,tmp,out->n+k>cap?cap-out->n:k); continue; }
        if(WaitForSingleObject(pi.hProcess,20)==WAIT_OBJECT_0){ DWORD k; while(ReadFile(rd,tmp,sizeof tmp,&k,NULL)&&k) if(out->n<cap) bput(out,tmp,out->n+k>cap?cap-out->n:k); break; }
        if(GetTickCount()-t0>(DWORD)ms){ TerminateProcess(pi.hProcess,9); WaitForSingleObject(pi.hProcess,2000); to=1; break; } }
    GetExitCodeProcess(pi.hProcess,&code); CloseHandle(pi.hProcess); CloseHandle(pi.hThread); CloseHandle(rd);
    if(!out->p) bput(out,"",0);
    return to?-1:(int)code;
}
static const char *DENY[]={"import os","from os","import sys","from sys","subprocess","import shutil","from shutil","socket","ctypes",
    "multiprocessing","threading","urllib","http.","import http","pathlib","import signal","__import__","eval(","exec(","open(","input(",0};
static const char *py_denied(const char *code){ for(int i=0;DENY[i];i++) if(strstr(code,DENY[i])) return DENY[i]; return NULL; }
static void wfile(const char *dir, const char *name, const char *s){
    char f[MAX_PATH*2]; snprintf(f,sizeof f,"%s\\%s",dir,name); FILE *o=fopen(f,"wb"); if(o){ fwrite(s,1,strlen(s),o); fclose(o); } }
/* run python source; returns exit code (-1 timeout); out = merged output */
static int py_run(const char *code, int ms, Buf *out, size_t cap){
    char dir[MAX_PATH]; mktmp(dir,sizeof dir); wfile(dir,"main.py",code);
    char cmd[MAX_PATH*2]; snprintf(cmd,sizeof cmd,"\"%s\" -I main.py",PY());
    int rc=run_proc(cmd,dir,ms,out,cap); rmtree(dir);
    if(rc==-2){ fprintf(stderr,"cannot run %s (set AGENT_PYTHON)\n",PY()); exit(1); }
    return rc;
}

/* ---------------- calc ---------------- */
static const char *cp; static int cerr;
static double cexpr(void);
static void csp(void){ while(*cp==' '||*cp=='\t'||*cp=='\n') cp++; }
static double cprim(void){
    csp(); double v;
    if(*cp=='('){ cp++; v=cexpr(); csp(); if(*cp==')') cp++; else cerr=1; return v; }
    if(*cp=='-'){ cp++; return -cprim(); } if(*cp=='+'){ cp++; return cprim(); }
    if(isdigit((unsigned char)*cp)||*cp=='.'){ char *e; v=strtod(cp,&e); cp=e; while(*cp==','&&isdigit((unsigned char)cp[1])){ /* 1,000 */
            char *e2; double w=strtod(cp+1,&e2); int d=(int)(e2-cp-1); v=v*pow(10,d)+w; cp=e2; } return v; }
    if(isalpha((unsigned char)*cp)){ char f[16]; int n=0; while(isalpha((unsigned char)*cp)&&n<15) f[n++]=(char)tolower(*cp++); f[n]=0;
        if(!strcmp(f,"pi")) return 3.14159265358979323846; if(!strcmp(f,"e")) return 2.71828182845904523536;
        csp(); if(*cp!='('){ cerr=1; return 0; } cp++; double a=cexpr(), b=0; csp(); if(*cp==','){ cp++; b=cexpr(); csp(); } if(*cp==')') cp++; else cerr=1;
        if(!strcmp(f,"sqrt")) return sqrt(a); if(!strcmp(f,"abs")) return fabs(a); if(!strcmp(f,"floor")) return floor(a);
        if(!strcmp(f,"ceil")) return ceil(a); if(!strcmp(f,"round")) return b?round(a*pow(10,b))/pow(10,b):round(a);
        if(!strcmp(f,"log")||!strcmp(f,"ln")) return log(a); if(!strcmp(f,"log10")) return log10(a); if(!strcmp(f,"exp")) return exp(a);
        if(!strcmp(f,"sin")) return sin(a); if(!strcmp(f,"cos")) return cos(a); if(!strcmp(f,"tan")) return tan(a);
        if(!strcmp(f,"min")) return a<b?a:b; if(!strcmp(f,"max")) return a>b?a:b; if(!strcmp(f,"pow")) return pow(a,b);
        cerr=1; return 0; }
    cerr=1; return 0;
}
static double cpowr(void){ double a=cprim(); csp(); if(*cp=='^'||(cp[0]=='*'&&cp[1]=='*')){ cp+=*cp=='^'?1:2; return pow(a,cpowr()); } return a; }
static double cterm(void){ double a=cpowr(); for(;;){ csp();
    if(*cp=='*'&&cp[1]!='*'){ cp++; a*=cpowr(); } else if(*cp=='/'&&cp[1]=='/'){ cp+=2; a=floor(a/cpowr()); }
    else if(*cp=='/'){ cp++; a/=cpowr(); } else if(*cp=='%'){ cp++; a=fmod(a,cpowr()); } else return a; } }
static double cexpr(void){ double a=cterm(); for(;;){ csp(); if(*cp=='+'){ cp++; a+=cterm(); } else if(*cp=='-'){ cp++; a-=cterm(); } else return a; } }
static void tool_calc(const char *in, Buf *out){
    cp=in; cerr=0; double v=cexpr(); csp();
    if(cerr||*cp||!isfinite(v)) bstr(out,"Error: could not evaluate the expression");
    else if(fabs(v-round(v))<1e-9&&fabs(v)<1e15) bfmt(out,"%.0f",v); else bfmt(out,"%.10g",v);
}

/* ---------------- search: BM25 over title\tpassage lines ---------------- */
typedef struct { char *title,*text; int len; } Doc;
static Doc *g_docs; static int g_nd; static double g_avgl;
typedef struct { unsigned h; int *d, *tf, n, cap; } Post;
static Post *g_post; static unsigned g_pcap;
static const char *STOPW[]={"the","a","an","of","and","or","to","in","on","is","are","was","were","be","for","by","with","as","at","it","its",
    "that","this","what","which","who","how","when","where","why","from","does","did","do",0};
static int stopw(const char *w){ for(int i=0;STOPW[i];i++) if(!strcmp(w,STOPW[i])) return 1; return 0; }
static unsigned fnv(const char *s){ unsigned h=2166136261u; for(;*s;s++){ h^=(unsigned char)*s; h*=16777619u; } return h|1; }
static unsigned g_pused;
static Post *pget(unsigned h, int add);
static void pgrow(void){
    Post *old=g_post; unsigned oc=g_pcap; g_pcap=oc?oc*2:1u<<16; g_post=calloc(g_pcap,sizeof(Post)); g_pused=0;
    for(unsigned i=0;i<oc;i++) if(old[i].h){ Post *p=pget(old[i].h,1); *p=old[i]; } free(old); }
static Post *pget(unsigned h, int add){
    if(add&&(g_pused+1)*2>g_pcap) pgrow();
    for(unsigned i=h&(g_pcap-1);;i=(i+1)&(g_pcap-1)){ if(g_post[i].h==h) return &g_post[i]; if(!g_post[i].h){ if(!add) return NULL; g_post[i].h=h; g_pused++; return &g_post[i]; } } }
/* lowercase alnum words; calls f(word) */
static int words(const char *s, char w[][32], int max){
    int n=0; while(*s&&n<max){ while(*s&&!isalnum((unsigned char)*s)) s++; int k=0;
        while(isalnum((unsigned char)*s)){ if(k<31) w[n][k++]=(char)tolower(*s); s++; }
        if(k){ w[n][k]=0; if(!stopw(w[n])) n++; } } return n; }
static void store_load(const char *path){
    FILE *f=fopen(path,"rb"); if(!f){ fprintf(stderr,"no store %s\n",path); exit(1); }
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET); char *all=malloc(sz+1); fread(all,1,sz,f); all[sz]=0; fclose(f);
    int cap=1024; g_docs=malloc(sizeof(Doc)*cap);
    for(char *p=all;*p;){ char *e=strchr(p,'\n'); if(e) *e=0; if(e&&e>p&&e[-1]=='\r') e[-1]=0;
        char *t=strchr(p,'\t'); if(t){ *t=0; if(g_nd==cap){ cap*=2; g_docs=realloc(g_docs,sizeof(Doc)*cap); } g_docs[g_nd].title=p; g_docs[g_nd].text=t+1; g_nd++; }
        if(!e) break; p=e+1; }
    pgrow();
    static char w[4096][32]; double tl=0;
    for(int d=0;d<g_nd;d++){ char tmp[8192]; snprintf(tmp,sizeof tmp,"%s %s",g_docs[d].title,g_docs[d].text);
        int n=words(tmp,w,4096); g_docs[d].len=n; tl+=n;
        for(int i=0;i<n;i++){ Post *p=pget(fnv(w[i]),1);
            if(p->n&&p->d[p->n-1]==d){ p->tf[p->n-1]++; continue; }
            if(p->n==p->cap){ p->cap=p->cap?p->cap*2:4; p->d=realloc(p->d,sizeof(int)*p->cap); p->tf=realloc(p->tf,sizeof(int)*p->cap); }
            p->d[p->n]=d; p->tf[p->n++]=1; } }
    g_avgl=g_nd?tl/g_nd:1; fprintf(stderr,"[store] %d passages from %s\n",g_nd,path);
}
static int search_top(const char *q, int *top, double *ts, int k){
    static double *sc; static int *touch; if(!sc){ sc=calloc(g_nd,sizeof(double)); touch=malloc(sizeof(int)*g_nd); }
    char w[64][32]; int n=words(q,w,64), nt=0;
    for(int i=0;i<n;i++){ int dup=0; for(int j=0;j<i;j++) if(!strcmp(w[i],w[j])) dup=1; if(dup) continue;
        Post *p=g_pcap?pget(fnv(w[i]),0):NULL; if(!p) continue; double idf=log(1+(g_nd-p->n+0.5)/(p->n+0.5));
        for(int m=0;m<p->n;m++){ int d=p->d[m]; double tf=p->tf[m]; if(sc[d]==0) touch[nt++]=d;
            sc[d]+=idf*tf*2.2/(tf+1.2*(0.25+0.75*g_docs[d].len/g_avgl)); } }
    int got=0;
    for(int i=0;i<nt;i++){ int d=touch[i]; double s=sc[d]; int j=got<k?got++:k;
        if(j==k&&s<=ts[k-1]) continue; if(j==k) j=k-1;
        while(j>0&&ts[j-1]<s){ ts[j]=ts[j-1]; top[j]=top[j-1]; j--; } ts[j]=s; top[j]=d; }
    for(int i=0;i<nt;i++) sc[touch[i]]=0;
    return got;
}
static void tool_search(const char *q, Buf *out){
    if(!g_nd){ bstr(out,"Error: search is not available"); return; }
    int top[3]; double ts[3]; int n=search_top(q,top,ts,3);
    if(!n){ bstr(out,"No results."); return; }
    for(int i=0;i<n;i++){ const Doc *d=&g_docs[top[i]]; int L=(int)strlen(d->text), dup=!strncmp(d->text,d->title,strlen(d->title));
        bfmt(out,"%s[%d] %s%s%.*s%s",i?"\n":"",i+1,dup?"":d->title,dup?"":": ",L>400?400:L,d->text,L>400?"...":""); }
}

/* ---------------- episode ---------------- */
#define MAXF 32
typedef struct { char *name[MAXF], *data[MAXF]; int n; } VFS;
static void vfs_set(VFS *v, const char *n, const char *d){
    for(int i=0;i<v->n;i++) if(!strcmp(v->name[i],n)){ free(v->data[i]); v->data[i]=strdup(d); return; }
    if(v->n<MAXF){ v->name[v->n]=strdup(n); v->data[v->n++]=strdup(d); } }
static const char *vfs_get(VFS *v, const char *n){ for(int i=0;i<v->n;i++) if(!strcmp(v->name[i],n)) return v->data[i]; return NULL; }
static void vfs_free(VFS *v){ for(int i=0;i<v->n;i++){ free(v->name[i]); free(v->data[i]); } v->n=0; }
static char *trim(char *s){ while(*s==' '||*s=='\n'||*s=='\r'||*s=='\t') s++; char *e=s+strlen(s); while(e>s&&(e[-1]==' '||e[-1]=='\n'||e[-1]=='\r'||e[-1]=='\t')) *--e=0; return s; }
/* teacher call noise: CDATA wrappers, "- " / ": " before the argument, quoted one-line arguments */
static char *unwrap(char *s){
    s=trim(s); if(!strncmp(s,"<![CDATA[",9)){ s+=9; char *e=strstr(s,"]]>"); if(e) *e=0; s=trim(s); }
    for(char *e; (e=strstr(s,"</|call"))||(e=strstr(s,"</function>")); ) *e=0;
    s=trim(s);
    if((*s=='-'||*s==':')&&(s[1]==' '||s[1]=='\n'||!s[1])) s=trim(s+1);
    if(!_strnicmp(s,"query",5)||!_strnicmp(s,"input",5)){ char *q=s+5; while(*q==' ') q++; if(*q==':'||*q=='"') s=trim(*q==':'?q+1:q); }
    size_t L=strlen(s); if(L>=2&&(*s=='"'||*s=='\'')&&s[L-1]==*s&&!memchr(s+1,*s,L-2)){ s[L-1]=0; s++; }
    return s; }

static const char *SYS =
 "You can use tools. To call a tool, write <|call|> followed by the tool name on the same line, then its input on the next "
 "lines, then <|end|>. The result comes back between <|result|> and <|end|>. Tools:\n"
 "python - runs Python code; print() what you need\n"
 "calc - evaluates a math expression, e.g. 3*(4+5)/2\n"
 "search - searches a reference library; input is a short query\n"
 "read_file - input is a file name\n"
 "write_file - first line is the file name, the following lines are the content\n"
 "list_dir - lists the files (no input)\n"
 "Only call a tool when it helps. Never write a result yourself. When you have what you need, give the final answer.";

/* teacher's native call <function name="N"><param name="k">v</param>...</function> -> <|call|>N\nv1\nv2 */
static char *native_call(char *g){
    char *f=strstr(g,"<function name=\""); if(!f) return g;
    Buf b={0}; bput(&b,g,f-g); bstr(&b,"<|call|>"); char *p=f+16, *q=strchr(p,'"'); if(!q) { free(b.p); return g; }
    bput(&b,p,q-p); bstr(&b,"\n");
    for(char *e=strstr(q,"</function>"), *v=strstr(q,"<param name=\""); v&&(!e||v<e); v=strstr(v,"<param name=\"")){
        v=strchr(v+13,'>'); if(!v) break; v++; char *ve=strstr(v,"</param>"); if(!ve) ve=v+strlen(v);
        bput(&b,v,ve-v); bstr(&b,"\n"); v=ve; }
    free(g); return b.p;
}
typedef struct { int calls, valid, halluc, nerr, steps; char *answer; Buf text; } Ep;
/* runs one tool episode; task fields: prompt, files{}, inject_error */
static void episode(J *task, VFS *vfs, Ep *ep, unsigned long long seed, double temp){
    memset(ep,0,sizeof *ep);
    J *files=jget(task,"files"); if(files&&files->t=='o') for(int i=0;i<files->len;i++) if(files->v[i]->t=='s') vfs_set(vfs,files->k[i],files->v[i]->s);
    int inject=(int)jgn(task,"inject_error",0);
    Buf ctx={0}; wrap(&ctx,SYS,jgs(task,"prompt",""));
    Buf stu={0}, last={0}; bput(&stu,"",0); bput(&last,"",0);   /* stu: student-format transcript; ctx: what the server sees */
    static const char *ST[]={"<|end|>","<|result|>","<|user|>","<|system|>","</|call>"};   /* teacher closes calls with </|call> */
    for(int s=0;s<o_steps;s++){
        ep->steps++; int stop; char *word=NULL;
        char *g=complete(ctx.p,o_max,temp,seed+s*7919,ST,5,&stop,&word);
        /* servers that ignore "stop": cut at the first marker ourselves */
        const char *hitw=NULL; char *cut=NULL;
        for(int i=0;i<5;i++){ char *f=strstr(g,ST[i]); if(f&&(!cut||f<cut)){ cut=f; hitw=ST[i]; } }
        if(cut){ *cut=0; stop=1; } else if(stop==1&&word) hitw=word; else if(stop==1) hitw="<|end|>";
        if(o_v>1) fprintf(stderr,"[gen] %s\n",g);
        char *raw=o_teacher?strdup(g):NULL; if(o_teacher) g=native_call(g);
        bstr(&ctx,raw?raw:g); free(raw);
        char *call=strstr(g,"<|call|>"), *nx;
        while(call&&(nx=strstr(call+8,"<|call|>"))) call=nx;           /* the last call in this chunk */
        if(call) bput(&stu,g,call-g); else bstr(&stu,g);
        int is_result=hitw&&!strcmp(hitw,"<|result|>");
        if(!call){
            if(is_result){ ep->halluc=1; free(g); free(word); break; }  /* invented a tool result */
            free(last.p); last=(Buf){0}; bstr(&last,g); free(g); free(word); break;
        }
        /* parse the call: NAME [arg-on-line]\nINPUT */
        ep->calls++;
        char *c=call+8; while(*c==' ') c++;
        char name[32]={0}; int k=0; while(*c&&!isspace((unsigned char)*c)&&k<31) name[k++]=*c++;
        char *line1=c; while(*c==' ') c++; char *nl=strchr(c,'\n'); char a1[256]={0};
        if(nl){ int L=(int)(nl-c); if(L>255) L=255; memcpy(a1,c,L); } else snprintf(a1,sizeof a1,"%s",c);
        char *input=strdup(nl?nl+1:""); (void)line1;
        char *in=unwrap(input); { char *u=unwrap(a1); memmove(a1,u,strlen(u)+1); } Buf out={0};
        bfmt(&stu,"<|call|>%s\n",name);                                  /* canonical call for the transcript */
        if(!strcmp(name,"write_file")) bfmt(&stu,"%s\n%s\n",a1,in);
        else if(!strcmp(name,"read_file")){ const char *f=*a1?a1:in; if(*f) bfmt(&stu,"%s\n",f); }
        else if(strcmp(name,"list_dir")){ const char *x=*in?in:a1; if(*x) bfmt(&stu,"%s\n",x); } bput(&out,"",0); int valid=1;
        if(inject&&ep->calls==1) { bstr(&out,"Error: the tool service is temporarily unavailable. Please try again."); ep->nerr++; }
        else if(!strcmp(name,"python")){ const char *code=*in?in:trim(a1);
            if(!*code){ bstr(&out,"Error: no code"); valid=0; }
            else { const char *d=py_denied(code);
                if(d) bfmt(&out,"Error: '%s' is not allowed here",d);
                else { int rc=py_run(code,5000,&out,4000); if(rc==-1) bstr(&out,"\nError: timed out after 5 s"); if(!out.n) bstr(&out,"(no output - use print())"); } } }
        else if(!strcmp(name,"calc")) tool_calc(*in?in:trim(a1),&out);
        else if(!strcmp(name,"search")) tool_search(*in?in:trim(a1),&out);
        else if(!strcmp(name,"read_file")){ char *fn=*trim(a1)?trim(a1):in; const char *d=vfs_get(vfs,fn);
            if(d) bstr(&out,d); else { bfmt(&out,"Error: no file named '%s'",fn); } }
        else if(!strcmp(name,"write_file")){ char *fn=trim(a1); if(!*fn){ bstr(&out,"Error: missing file name"); valid=0; } else { vfs_set(vfs,fn,in); bfmt(&out,"Wrote %d bytes to %s",(int)strlen(in),fn); } }
        else if(!strcmp(name,"list_dir")){ for(int i=0;i<vfs->n;i++) bfmt(&out,"%s%s",i?"\n":"",vfs->name[i]); if(!vfs->n) bstr(&out,"(empty)"); }
        else { bfmt(&out,"Error: unknown tool '%s'",name); valid=0; }
        ep->valid+=valid;
        { size_t w=0; for(size_t r=0;r<out.n;r++) if(out.p[r]!='\r') out.p[w++]=out.p[r];       /* CRLF -> LF, trim tail */
          while(w&&(out.p[w-1]=='\n'||out.p[w-1]==' ')) w--; out.n=w; out.p[w]=0; }
        if(out.n>4000){ out.n=4000; out.p[4000]=0; bstr(&out,"\n...(truncated)"); }
        bstr(&stu,"<|end|>\n<|result|>\n"); bstr(&stu,out.p); bstr(&stu,"\n<|end|>\n");
        if(o_teacher){ bstr(&ctx,t_tool); bstr(&ctx,out.p); bstr(&ctx,t_post); } else { bstr(&ctx,"<|end|>\n<|result|>\n"); bstr(&ctx,out.p); bstr(&ctx,"\n<|end|>\n"); }
        free(out.p); free(input); free(g); free(word);
    }
    ep->answer=strdup(trim(last.p)); free(last.p);
    ep->text=stu; free(ctx.p);
}
static int ci_contains(const char *h, const char *n){
    size_t L=strlen(n); if(!L) return 1;
    for(;*h;h++){ size_t i=0; while(i<L&&h[i]&&tolower((unsigned char)h[i])==tolower((unsigned char)n[i])) i++; if(i==L) return 1; } return 0; }
static int has_number(const char *s, double v, double tol, double rel){
    for(const char *p=s;*p;p++){ if(!(isdigit((unsigned char)*p)||((*p=='-'||*p=='.')&&isdigit((unsigned char)p[1])))) continue;
        if(p>s&&(isdigit((unsigned char)p[-1])||p[-1]=='.')) continue;
        char t[64]; int k=0; const char *q=p; while((isdigit((unsigned char)*q)||*q=='.'||*q==','||(*q=='-'&&q==p))&&k<63){ if(*q!=',') t[k++]=*q; q++; } t[k]=0;
        while(k&&t[k-1]=='.') t[--k]=0; double x=atof(t);
        if(fabs(x-v)<=fmax(tol,rel*fabs(v))) return 1; }
    return 0; }
/* check: {"type":"contains","any":[..]} | {"type":"number","value":x,"tol":t} | {"type":"file","path":p,"contains":s}
 *        | {"type":"nonempty"} */
static int check_ep(J *task, Ep *ep, VFS *vfs){
    J *c=jget(task,"check"); const char *t=jgs(c,"type","nonempty"); const char *ans=ep->answer?ep->answer:"";
    if(ep->halluc) return 0;
    if(!strcmp(t,"contains")){ J *a=jget(c,"any"); if(a&&a->t=='a'){ for(int i=0;i<a->len;i++) if(a->v[i]->t=='s'&&ci_contains(ans,a->v[i]->s)) return 1; return 0; }
        return ci_contains(ans,jgs(c,"value","")); }
    if(!strcmp(t,"number")) return has_number(ans,jgn(c,"value",NAN),jgn(c,"tol",1e-9),jgn(c,"rel",1e-6));
    if(!strcmp(t,"file")){ const char *d=vfs_get(vfs,jgs(c,"path","")); return d&&ci_contains(d,jgs(c,"contains","")); }
    return strlen(ans)>0;
}
/* correct answer AND correct tool policy, all calls valid */
static int ep_ok(J *t, Ep *ep, VFS *v){
    const char *tool=jgs(t,"tool","any");
    return check_ep(t,ep,v)&&!(!strcmp(tool,"none")&&ep->calls)&&!(!strcmp(tool,"required")&&!ep->calls)&&ep->calls==ep->valid;
}

/* ---------------- E-tool ---------------- */
static char **read_lines(const char *path, int *n){
    FILE *f=fopen(path,"rb"); if(!f){ fprintf(stderr,"cannot open %s\n",path); exit(1); }
    char **v=NULL; *n=0; Buf b={0}; int c;
    for(;;){ c=fgetc(f); if(c==EOF||c=='\n'){ if(b.n){ char *s=trim(b.p); if(*s){ v=realloc(v,sizeof(char*)*(*n+1)); v[(*n)++]=strdup(s); } b.n=0; b.p[0]=0; } if(c==EOF) break; }
        else { char ch=(char)c; bput(&b,&ch,1); } }
    fclose(f); free(b.p); return v;
}
static void cmd_tool(const char *in, const char *outp){
    int n; char **L=read_lines(in,&n); FILE *o=fopen(outp,"wb");
    double temp=o_temp<0?0:o_temp;
    typedef struct { char cat[24]; int n,ok; } Cat; Cat cats[16]; int nc=0;
    int ok=0,calls=0,valid=0,hall=0,unn=0,nnone=0,miss=0,nreq=0;
    for(int i=0;i<n;i++){
        J *t=jload(L[i]); VFS v={0}; Ep ep; episode(t,&v,&ep,o_seed+(unsigned long long)i*104729,temp);
        int s=check_ep(t,&ep,&v); const char *tool=jgs(t,"tool","any"), *cat=jgs(t,"cat","?");
        if(!strcmp(tool,"none")){ nnone++; if(ep.calls){ unn++; s=0; } }
        if(!strcmp(tool,"required")){ nreq++; if(!ep.calls){ miss++; s=0; } }
        ok+=s; calls+=ep.calls; valid+=ep.valid; hall+=ep.halluc;
        int ci=0; while(ci<nc&&strcmp(cats[ci].cat,cat)) ci++;
        if(ci==nc&&nc<16){ snprintf(cats[nc].cat,24,"%s",cat); cats[nc].n=cats[nc].ok=0; nc++; }
        if(ci<nc){ cats[ci].n++; cats[ci].ok+=s; }
        Buf b={0}; bfmt(&b,"{\"id\":\"%s\",\"cat\":\"%s\",\"ok\":%d,\"calls\":%d,\"valid\":%d,\"halluc\":%d,\"answer\":\"",jgs(t,"id",""),cat,s,ep.calls,ep.valid,ep.halluc);
        besc(&b,ep.answer); bstr(&b,"\",\"text\":\""); besc(&b,ep.text.p?ep.text.p:""); bstr(&b,"\"}\n"); fwrite(b.p,1,b.n,o); fflush(o);
        if(o_v) printf("---- %s [%s] ok=%d calls=%d\n%s\n",jgs(t,"id",""),cat,s,ep.calls,ep.text.p?ep.text.p:"");
        fprintf(stderr,"\r[tool] %d/%d  success %.1f%%   ",i+1,n,100.0*ok/(i+1));
        free(b.p); free(ep.answer); free(ep.text.p); vfs_free(&v); jfree(t);
    }
    fclose(o);
    printf("\n[E-tool] %s: n=%d  success %.1f%%  valid-call %.1f%% (%d calls)  hallucinated-result %.1f%%  unnecessary-call %.1f%% (of %d)  missing-call %.1f%% (of %d)\n",
           in,n,100.0*ok/n,calls?100.0*valid/calls:0,calls,100.0*hall/n,nnone?100.0*unn/nnone:0,nnone,nreq?100.0*miss/nreq:0,nreq);
    for(int c=0;c<nc;c++) printf("  %-10s %4d  %5.1f%%\n",cats[c].cat,cats[c].n,100.0*cats[c].ok/cats[c].n);
}

/* ---------------- E-code ---------------- */
/* task: {"id","prompt": "def f(..):\n    \"\"\"doc\"\"\"\n", "tests": "assert ...\n"} */
/* natural chat style (no prefill: small models close a prefilled fence at once); the answer's first code block
 * is the solution, with the stub prepended when it holds only a body */
static char *code_prompt(J *t, Buf *pre){
    Buf u={0}; bstr(&u,"Write this Python function so that it works as described.\n```python\n"); bstr(&u,jgs(t,"prompt","")); bstr(&u,"```");
    wrap(pre,NULL,u.p); free(u.p);
    return pre->p;
}
static void code_extract(const char *g, J *t, Buf *code){
    const char *a=strstr(g,"```"), *s=g, *e=NULL;
    if(a){ s=strchr(a,'\n'); s=s?s+1:a+3; e=strstr(s,"```"); }
    size_t n=e?(size_t)(e-s):strlen(s);
    char fn[64]; const char *d=strstr(jgs(t,"prompt",""),"def "); int k=0;
    if(d){ d+=4; while(d[k]&&d[k]!='('&&k<60){ fn[k]=d[k]; k++; } } fn[k]=0;
    char key[80]; snprintf(key,sizeof key,"def %s",fn);
    char *body=malloc(n+1); memcpy(body,s,n); body[n]=0;
    if(!k||!strstr(body,key)) bstr(code,jgs(t,"prompt",""));
    bstr(code,body); free(body);
}
static int code_check(const char *code, const char *tests, int *syntax){
    if(py_denied(code)){ *syntax=1; return 0; }
    Buf src={0}; bstr(&src,code); bstr(&src,"\n\n"); bstr(&src,tests); bstr(&src,"\nprint('PASS')\n");
    Buf out={0}; int rc=py_run(src.p,5000,&out,4000);
    *syntax = !strstr(out.p,"SyntaxError")&&!strstr(out.p,"IndentationError");
    int ok = rc==0&&strstr(out.p,"PASS");
    free(src.p); free(out.p); return ok;
}
static double comb(int n, int k){ if(k>n) return 0; double r=1; for(int i=0;i<k;i++) r=r*(n-i)/(i+1); return r; }
static void cmd_code(const char *in, const char *outp){
    int n; char **L=read_lines(in,&n); FILE *o=fopen(outp,"wb");
    double temp=o_temp<0?0.2:o_temp; int S=o_n<1?1:o_n;
    double p1=0,p5=0; int syn=0,tot=0;
    static const char *ST[]={"<|user|>","<|system|>"};
    for(int i=0;i<n;i++){
        J *t=jload(L[i]); Buf pre={0}; code_prompt(t,&pre); int c=0;
        Buf rec={0}; bfmt(&rec,"{\"id\":\"%s\",\"samples\":[",jgs(t,"id",""));
        char **gs=malloc(sizeof(char*)*S); complete_n(pre.p,o_max,temp,o_seed+(unsigned long long)i*1000003,ST,2,S,gs);
        for(int s=0;s<S;s++){ char *g=gs[s];
            for(int k=0;k<2;k++){ char *f=strstr(g,ST[k]); if(f) *f=0; }
            Buf code={0}; code_extract(g,t,&code);
            int sx, ok=code_check(code.p,jgs(t,"tests",""),&sx); c+=ok; syn+=sx; tot++;
            bfmt(&rec,"%s{\"ok\":%d,\"syntax\":%d,\"code\":\"",s?",":"",ok,sx); besc(&rec,code.p); bstr(&rec,"\"}");
            if(o_v) printf("---- %s #%d ok=%d\n%s\n",jgs(t,"id",""),s,ok,code.p);
            free(code.p); free(g); }
        free(gs);
        p1+=(double)c/S; if(S>=5) p5+=1.0-comb(S-c,5)/comb(S,5);
        bfmt(&rec,"],\"pass\":%d}\n",c); fwrite(rec.p,1,rec.n,o); fflush(o); free(rec.p); free(pre.p); jfree(t);
        fprintf(stderr,"\r[code] %d/%d  pass@1 %.1f%%   ",i+1,n,100.0*p1/(i+1));
    }
    fclose(o);
    printf("\n[E-code] %s: n=%d samples=%d T=%.2f  pass@1 %.1f%%",in,n,S,temp,100.0*p1/n);
    if(S>=5) printf("  pass@5 %.1f%%",100.0*p5/n);
    printf("  syntax-valid %.1f%%\n",100.0*syn/tot);
}

/* ---------------- teacher generation ---------------- */
static unsigned long long rs=88172645463325252ULL;
static unsigned rnd(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (unsigned)(rs>>11); }
static char *ask(const char *sys, const char *user, int npred, double temp, unsigned long long seed){
    Buf p={0}; wrap(&p,sys,user); int stop; char *g=complete(p.p,npred,temp,seed,NULL,0,&stop,NULL); free(p.p);
    if(stop==2){ free(g); return NULL; } return g; }
/* text between "### NAME" and the next "###", with a ```fence``` stripped */
static char *section(const char *s, const char *name){
    char k[64]; snprintf(k,sizeof k,"### %s",name); const char *a=strstr(s,k); if(!a) return NULL; a+=strlen(k);
    const char *e=strstr(a,"\n###"); size_t n=e?(size_t)(e-a):strlen(a); char *r=malloc(n+1); memcpy(r,a,n); r[n]=0;
    char *f=strstr(r,"```"); if(f){ char *b=strchr(f,'\n'); char *g=b?strstr(b+1,"```"):NULL; if(b&&g){ *g=0; memmove(r,b+1,strlen(b+1)+1); } }
    return r; }
static const char *CODE_TOPICS[]={"string manipulation","list processing","dictionaries and counting","sorting and ranking","number theory",
    "simple math formulas","dates given as integers","parsing simple text formats","recursion","two pointers","sets","matrices as nested lists",
    "prefix sums","binary search","greedy choices","simulation of a simple process","validation of input strings","run-length encoding",
    "stack-based parsing","word statistics","grids and coordinates","intervals","frequency analysis","string formatting","bit manipulation",
    "geometry with integers","unit conversion","sequences and series","searching in lists","character classification",0};
static void cmd_gencode(const char *outp, int N){
    FILE *o=fopen(outp,"ab"); int nt=0; while(CODE_TOPICS[nt]) nt++;
    rs^=o_seed*0x9E3779B97F4A7C15ULL; int kept=0, tried=0; char seen[4096][40]; int nseen=0;
    while(kept<N&&tried<N*6){
        tried++; const char *topic=CODE_TOPICS[rnd()%nt]; int diff=rnd()%3;
        Buf u={0}; bfmt(&u,"Invent one new, self-contained Python programming exercise about %s, %s difficulty, solvable in at most 20 lines "
            "with the standard library only (no imports needed ideally, no input(), no files, no randomness). Use this exact format:\n"
            "### PROMPT\n```python\ndef function_name(args):\n    \"\"\"Clear description of what it returns, with one example.\"\"\"\n```\n"
            "### SOLUTION\n```python\n<the complete function>\n```\n### TESTS\n```python\n<4 or 5 lines, each one assert, covering edge cases>\n```",
            topic,diff==0?"easy":diff==1?"medium":"slightly harder");
        char *g=ask(NULL,u.p,900,0.9,o_seed*7777+tried); free(u.p); if(!g) continue;
        char *pr=section(g,"PROMPT"), *so=section(g,"SOLUTION"), *te=section(g,"TESTS");
        int good=pr&&so&&te&&strstr(pr,"def ")&&strstr(te,"assert")&&!py_denied(so)&&!py_denied(te);
        char fn[40]={0};
        if(good){ const char *d=strstr(pr,"def ")+4; int k=0; while(d[k]&&d[k]!='('&&k<39){ fn[k]=d[k]; k++; } fn[k]=0;
            for(int i=0;i<nseen;i++) if(!strcmp(seen[i],fn)) good=0; }
        if(good){ int sx; good=code_check(so,te,&sx);                        /* the solution passes */
            if(good){ Buf stub={0}; bstr(&stub,pr); if(!strstr(pr,"\n    pass")) bstr(&stub,"\n    pass\n");
                good=!code_check(stub.p,te,&sx); free(stub.p); } }          /* and the bare stub fails */
        if(good){ char *p=trim(pr); Buf b={0}; bfmt(&b,"{\"id\":\"code%05d\",\"topic\":\"%s\",\"entry\":\"%s\",\"prompt\":\"",kept,topic,fn);
            besc(&b,p); besc(&b,"\n"); bstr(&b,"\",\"tests\":\""); besc(&b,trim(te)); bstr(&b,"\",\"solution\":\""); besc(&b,trim(so)); bstr(&b,"\"}\n");
            fwrite(b.p,1,b.n,o); fflush(o); free(b.p); if(nseen<4096) snprintf(seen[nseen++],40,"%s",fn); kept++; }
        fprintf(stderr,"\r[gen-code] kept %d / tried %d   ",kept,tried);
        free(pr); free(so); free(te); free(g);
    }
    fclose(o); fprintf(stderr,"\n");
}
/* gate: keep a generated task only if the teacher solves it in-format (greedy) - drops bad labels / unanswerable items */
static int emit_task(FILE *o, int id, const char *cat, const char *tool, const char *prompt, VFS *files, const char *check, int inject){
    Buf b={0}; bfmt(&b,"{\"id\":\"tool%05d\",\"cat\":\"%s\",\"tool\":\"%s\",\"prompt\":\"",id,cat,tool); besc(&b,prompt); bstr(&b,"\"");
    if(files&&files->n){ bstr(&b,",\"files\":{"); for(int i=0;i<files->n;i++){ bstr(&b,i?",\"":"\""); besc(&b,files->name[i]); bstr(&b,"\":\""); besc(&b,files->data[i]); bstr(&b,"\""); } bstr(&b,"}"); }
    if(inject) bstr(&b,",\"inject_error\":1");
    bfmt(&b,",\"check\":%s}\n",check);
    J *t=jload(b.p); VFS v={0}; Ep ep; episode(t,&v,&ep,o_seed^(unsigned long long)id*2654435761u,0);
    int ok=ep_ok(t,&ep,&v);
    if(ok){ if(o_trajout){ Buf x={0}; o_teacher=0; wrap(&x,SYS,prompt); o_teacher=1; bstr(&x,ep.text.p);   /* the gate rollout = a trajectory */
            b.n-=2; b.p[b.n]=0; bstr(&b,",\"traj\":\""); besc(&b,x.p); bstr(&b,"\"}\n"); free(x.p); }
        fwrite(b.p,1,b.n,o); fflush(o); }
    free(ep.answer); free(ep.text.p); vfs_free(&v); jfree(t);
    free(b.p); return ok;
}
/* procedural file tasks: ground truth computed here */
static int gen_file_task(FILE *o, int id){
    static const char *NAMES[]={"alice","bob","carol","dave","erin","frank","grace","heidi","ivan","judy","mallory","nina","oscar","peggy"};
    VFS v={0}; Buf d={0}, pr={0}; char chk[256]; int kind=rnd()%5, rows=5+rnd()%12;
    if(kind<=2){ double sum=0,mx=-1; char best[16]=""; int cnt=0, thr=50+rnd()%40;
        bstr(&d,"name,score\n");
        for(int r=0;r<rows;r++){ const char *nm=NAMES[rnd()%14]; int sc=10+rnd()%90; bfmt(&d,"%s,%d\n",nm,sc); sum+=sc; if(sc>mx){ mx=sc; snprintf(best,16,"%s",nm); } cnt+=sc>thr; }
        vfs_set(&v,"scores.csv",d.p);
        if(kind==0){ bstr(&pr,"The file scores.csv has a name and a score on each row. What is the total of all scores?"); snprintf(chk,sizeof chk,"{\"type\":\"number\",\"value\":%.0f,\"tol\":1e-9}",sum); }
        else if(kind==1){ bstr(&pr,"Look at scores.csv. What is the average score? Give it to two decimal places."); snprintf(chk,sizeof chk,"{\"type\":\"number\",\"value\":%.2f,\"tol\":0.006}",sum/rows); }
        else { bfmt(&pr,"How many rows in scores.csv have a score above %d?",thr); snprintf(chk,sizeof chk,"{\"type\":\"number\",\"value\":%d,\"tol\":1e-9}",cnt); }
    } else if(kind==3){ static const char *W[]={"apple","river","stone","cloud","light","paper","green","music"}; const char *tw=W[rnd()%8]; int c=0;
        for(int r=0;r<rows;r++){ int nw=3+rnd()%6; for(int k=0;k<nw;k++){ const char *w=W[rnd()%8]; c+=w==tw; bfmt(&d,"%s%s",k?" ":"",w); } bstr(&d,"\n"); }
        vfs_set(&v,"notes.txt",d.p); bfmt(&pr,"How many times does the word \"%s\" appear in notes.txt?",tw); snprintf(chk,sizeof chk,"{\"type\":\"number\",\"value\":%d,\"tol\":1e-9}",c);
    } else { int a=100+rnd()%900, b=100+rnd()%900;
        bfmt(&d,"first=%d\nsecond=%d\n",a,b); vfs_set(&v,"input.txt",d.p);
        bstr(&pr,"Read input.txt, multiply the two numbers in it, and write the product to a new file called result.txt.");
        snprintf(chk,sizeof chk,"{\"type\":\"file\",\"path\":\"result.txt\",\"contains\":\"%d\"}",a*b); }
    int ok=emit_task(o,id,"file","required",pr.p,&v,chk,0); vfs_free(&v); free(d.p); free(pr.p); return ok;
}
static const char *TOPICS[]={"shopping discounts","travel distances","cooking recipes","home renovation","bank interest","a science lab",
 "sports statistics","fuel costs","hourly pay","floor tiles","gardening","a bakery","a school trip","electricity bills","currency exchange",
 "a road trip","painting a room","a library","a factory","a bike shop","water tanks","a concert","phone plans","a farm market","rent sharing",
 "a marathon","a pizza party","shipping boxes","a bookstore","taxes","a coffee shop","a swimming pool","a train timetable","saving money"};
static const char *NTOP[]={"animals","colors","food","the weather","greetings","word meanings","the human body","holidays","music","sports",
 "household objects","manners","geography","plants","time and calendars","simple advice","school subjects","transport","clothing","jobs"};
static int nwords(const char *s){ int n=0,in=0; for(;*s;s++){ int w=!isspace((unsigned char)*s); n+=w&&!in; in=w; } return n; }
/* ---------------- procedural step-by-step compute trajectories (no teacher) ----------------
 * One calc call per step, operands copied from the question or from earlier results (verbatim tool output).
 * ~30% open with a typical bad call (thousands comma, unit word, "x", "%") -> error -> corrected call. */
static const char *GN[]={"Maria","Tom","Aisha","Ken","Lena","Omar","Sofia","Raj","Emma","Luis","Mei","Jonas","Priya","Sam"};
static const char *GI[]={"notebooks","shirts","tickets","plants","lamps","chairs","books","mugs","bags of rice","boxes of tea"};
static double pick(double lo, double hi, int dec){ double v=lo+(hi-lo)*(rnd()%100000)/100000.0, s=dec?pow(10,dec):1; return round(v*s)/s; }
static void numf(char *b, double v, int comma){        /* question form; comma groups thousands */
    char t[64]; if(fabs(v-round(v))<1e-9) snprintf(t,sizeof t,"%.0f",v); else snprintf(t,sizeof t,comma&2?"%.2f":"%g",v);
    if(!comma||fabs(v)<1000||strchr(t,'.')&&strchr(t,'.')-t<4){ strcpy(b,t); return; }
    char *d=strchr(t,'.'); int il=d?(int)(d-t):(int)strlen(t), k=0;
    for(int i=0;i<il;i++){ if(i&&(il-i)%3==0) b[k++]=','; b[k++]=t[i]; } b[k]=0; if(d) strcat(b,d); }
static void ansf(char *b, double v){ if(fabs(v-round(v))<1e-9) snprintf(b,64,"%.0f",v); else snprintf(b,64,"%.2f",v); }
typedef struct { Buf q; char ex[4][160]; char r[4][64]; int ns; char fin[256]; } GS;
static int gs_run(GS *g, int s){             /* evaluates step s with the real tool; 0 on error */
    Buf o={0}; bput(&o,"",0); tool_calc(g->ex[s],&o); int ok=strncmp(o.p,"Error",5)!=0; snprintf(g->r[s],64,"%s",o.p); free(o.p); return ok; }
static int gen_steps_one(Buf *out){
    GS g; memset(&g,0,sizeof g); bput(&g.q,"",0); int k=rnd()%10, cm=rnd()%2; char a1[64],a2[64],a3[64],a4[64];
    const char *nm=GN[rnd()%14], *it=GI[rnd()%10]; double A,B,C,D;
    #define R(i) g.r[i]
    switch(k){
    case 0: A=pick(2,40,0); B=pick(1,60,rnd()%2?2:0); C=pick(3,12,0); numf(a1,A,cm); numf(a2,B,cm|2); numf(a3,C,0);
        bfmt(&g.q,"%s buys %s %s at $%s each and pays %s%% sales tax. How much does %s pay in total?",nm,a1,it,a2,a3,nm);
        snprintf(g.ex[0],160,"%g * %g",A,B); g.ns=1; if(!gs_run(&g,0)) return 0;
        snprintf(g.ex[1],160,"%s * %g / 100",R(0),C); g.ns=2; if(!gs_run(&g,1)) return 0;
        snprintf(g.ex[2],160,"%s + %s",R(0),R(1)); g.ns=3; if(!gs_run(&g,2)) return 0;
        ansf(a4,atof(R(2))); snprintf(g.fin,256,"%s pays $%s in total.",nm,a4); break;
    case 1: A=pick(20,2500,rnd()%2?2:0); B=pick(5,60,0); numf(a1,A,cm|2); numf(a2,B,0);
        bfmt(&g.q,"A set of %s costs $%s and is on sale for %s%% off. What is the sale price?",it,a1,a2);
        snprintf(g.ex[0],160,"%g * %g / 100",A,B); g.ns=1; if(!gs_run(&g,0)) return 0;
        snprintf(g.ex[1],160,"%g - %s",A,R(0)); g.ns=2; if(!gs_run(&g,1)) return 0;
        ansf(a4,atof(R(1))); snprintf(g.fin,256,"The sale price is $%s.",a4); break;
    case 2: A=pick(40,3000,2); B=pick(10,25,0); C=pick(2,9,0); numf(a1,A,cm|2); numf(a2,B,0); numf(a3,C,0);
        bfmt(&g.q,"%s friends share a restaurant bill of $%s and add a %s%% tip. How much does each friend pay?",a3,a1,a2);
        snprintf(g.ex[0],160,"%g * %g / 100",A,B); g.ns=1; if(!gs_run(&g,0)) return 0;
        snprintf(g.ex[1],160,"%g + %s",A,R(0)); g.ns=2; if(!gs_run(&g,1)) return 0;
        snprintf(g.ex[2],160,"%s / %g",R(1),C); g.ns=3; if(!gs_run(&g,2)) return 0;
        ansf(a4,atof(R(2))); snprintf(g.fin,256,"Each friend pays $%s.",a4); break;
    case 3: A=pick(30,120,0); B=pick(1,9,rnd()%2?1:0); C=pick(5,400,0); numf(a1,A,0); numf(a2,B,0); numf(a3,C,cm);
        bfmt(&g.q,"A bus drives at %s km/h for %s hours and then another %s km. How far does it travel in total?",a1,a2,a3);
        snprintf(g.ex[0],160,"%g * %g",A,B); g.ns=1; if(!gs_run(&g,0)) return 0;
        snprintf(g.ex[1],160,"%s + %g",R(0),C); g.ns=2; if(!gs_run(&g,1)) return 0;
        ansf(a4,atof(R(1))); snprintf(g.fin,256,"The bus travels %s km in total.",a4); break;
    case 4: { int n=3+rnd()%4; double v[6],s=0; Buf e={0}; bput(&e,"",0); bfmt(&g.q,"%s's test scores were ",nm);
        for(int i=0;i<n;i++){ v[i]=pick(40,100,0); s+=v[i]; bfmt(&g.q,"%s%.0f",i?(i==n-1?" and ":", "):"",v[i]); bfmt(&e,"%s%.0f",i?" + ":"",v[i]); }
        bstr(&g.q,". What is the average score?"); snprintf(g.ex[0],160,"%s",e.p); free(e.p); g.ns=1; if(!gs_run(&g,0)) return 0;
        snprintf(g.ex[1],160,"%s / %d",R(0),n); g.ns=2; if(!gs_run(&g,1)) return 0;
        ansf(a4,atof(R(1))); snprintf(g.fin,256,"The average score is %s.",a4); break; }
    case 5: A=pick(3,15,1); B=pick(2,12,1); C=pick(8,90,2); numf(a1,A,0); numf(a2,B,0); numf(a3,C,2);
        bfmt(&g.q,"A room is %s m long and %s m wide. Flooring costs $%s per square meter. How much does the flooring cost?",a1,a2,a3);
        snprintf(g.ex[0],160,"%g * %g",A,B); g.ns=1; if(!gs_run(&g,0)) return 0;
        snprintf(g.ex[1],160,"%s * %g",R(0),C); g.ns=2; if(!gs_run(&g,1)) return 0;
        ansf(a4,atof(R(1))); snprintf(g.fin,256,"The flooring costs $%s.",a4); break;
    case 6: A=pick(1000,20000,0); B=pick(50,A/3,0); C=pick(50,A/3,0); numf(a1,A,cm); numf(a2,B,cm); numf(a3,C,cm);
        bfmt(&g.q,"A water tank holds %s liters. %s liters are used on Monday and %s liters on Tuesday. How many liters are left?",a1,a2,a3);
        snprintf(g.ex[0],160,"%g + %g",B,C); g.ns=1; if(!gs_run(&g,0)) return 0;
        snprintf(g.ex[1],160,"%g - %s",A,R(0)); g.ns=2; if(!gs_run(&g,1)) return 0;
        ansf(a4,atof(R(1))); snprintf(g.fin,256,"%s liters are left.",a4); break;
    case 7: A=pick(500,50000,0); B=pick(1,9,1); C=pick(2,10,0); numf(a1,A,cm|2); numf(a2,B,0); numf(a3,C,0);
        bfmt(&g.q,"%s deposits $%s at %s%% simple interest per year. What is the total amount after %s years?",nm,a1,a2,a3);
        snprintf(g.ex[0],160,"%g * %g / 100",A,B); g.ns=1; if(!gs_run(&g,0)) return 0;
        snprintf(g.ex[1],160,"%s * %g",R(0),C); g.ns=2; if(!gs_run(&g,1)) return 0;
        snprintf(g.ex[2],160,"%g + %s",A,R(1)); g.ns=3; if(!gs_run(&g,2)) return 0;
        ansf(a4,atof(R(2))); snprintf(g.fin,256,"The total amount after %s years is $%s.",a3,a4); break;
    case 8: A=pick(12,40,2); B=pick(4,10,1); C=pick(3,7,0); numf(a1,A,2); numf(a2,B,0); numf(a3,C,0);
        bfmt(&g.q,"%s earns $%s per hour and works %s hours a day for %s days. How much does %s earn?",nm,a1,a2,a3,nm);
        snprintf(g.ex[0],160,"%g * %g",A,B); g.ns=1; if(!gs_run(&g,0)) return 0;
        snprintf(g.ex[1],160,"%s * %g",R(0),C); g.ns=2; if(!gs_run(&g,1)) return 0;
        ansf(a4,atof(R(1))); snprintf(g.fin,256,"%s earns $%s.",nm,a4); break;
    default: A=20*pick(10,450,0); B=5*pick(1,19,0); numf(a1,A,cm); numf(a2,B,0);
        bfmt(&g.q,"A school has %s students and %s%% of them take the bus. How many students take the bus?",a1,a2);
        snprintf(g.ex[0],160,"%g * %g / 100",A,B); g.ns=1; if(!gs_run(&g,0)) return 0;
        ansf(a4,atof(R(0))); snprintf(g.fin,256,"%s students take the bus.",a4); break; }
    #undef R
    if(rnd()%2) bstr(&g.q," Round to two decimal places if needed.");
    Buf bad={0}; bput(&bad,"",0);
    if(rnd()%10<4){                                    /* typical first mistake, kept only if the tool rejects it */
        const char *e=g.ex[0], *p; int m=rnd()%4, L=0; while(isdigit((unsigned char)e[L])||e[L]=='.') L++;
        if(m==0&&L>=4&&!memchr(e,'.',L)){ char c[64],t[64]; snprintf(t,sizeof t,"%.*s",L,e); numf(c,atof(t),1); bfmt(&bad,"%s%s",c,e+L); }
        else if(m==1&&L) bfmt(&bad,"%.*s%s%s",L,e,rnd()%2?" dollars":" units",e+L);
        else if(m==2&&(p=strstr(e," * "))) bfmt(&bad,"%.*s x %s",(int)(p-e),e,p+3);
        else if(m==3&&(p=strstr(e," / 100"))&&!p[6]) bfmt(&bad,"%.*s%%",(int)(p-e),e);
        if(bad.n){ Buf o={0}; bput(&o,"",0); tool_calc(bad.p,&o); if(strncmp(o.p,"Error",5)){ bad.n=0; bad.p[0]=0; } free(o.p); } }
    o_teacher=0; wrap(out,SYS,g.q.p);
    if(bad.n) bfmt(out,"<|call|>calc\n%s\n<|end|>\n<|result|>\nError: could not evaluate the expression\n<|end|>\n",bad.p);
    for(int s=0;s<g.ns;s++) bfmt(out,"<|call|>calc\n%s\n<|end|>\n<|result|>\n%s\n<|end|>\n",g.ex[s],g.r[s]);
    bstr(out,g.fin); free(bad.p); free(g.q.p); return 1; }
static void cmd_gensteps(const char *outp, int N){
    FILE *o=fopen(outp,"wb"); int k=0; rs^=o_seed*0x9E3779B97F4A7C15ULL;
    while(k<N){ Buf t={0}; if(gen_steps_one(&t)){ Buf b={0}; bstr(&b,"{\"kind\":\"agent\",\"cat\":\"steps\",\"text\":\""); besc(&b,t.p); bstr(&b,"\"}\n");
            fwrite(b.p,1,b.n,o); free(b.p); k++; } free(t.p); }
    fclose(o); printf("[gen-steps] %d trajectories -> %s\n",k,outp); }
static void cmd_gentool(const char *outp, int N){
    FILE *o=fopen(outp,"ab"); rs^=o_seed*0x9E3779B97F4A7C15ULL;
    /* category quotas, -mix L,C,F,N,R (default 70,60,70,50,50 = the 300 eval) */
    int ms=o_mix[0]+o_mix[1]+o_mix[2]+o_mix[3]+o_mix[4], quota[5]; for(int c=0;c<4;c++) quota[c]=N*o_mix[c]/ms; quota[4]=N*o_mix[4]/ms;
    if(!g_nd){ quota[1]+=quota[0]; quota[0]=0; fprintf(stderr,"[gen-tool] no -store: lookup tasks skipped\n"); }
    int id=0, made[5]={0}, tries=0;
    for(int cat=0;cat<5;cat++) while(made[cat]<quota[cat]&&tries<N*20){ tries++;
        char chk[1024]; Buf pr={0}; int ok=0, rec=cat==4, sub=rec?(g_nd?(int)(rnd()%2):1):cat;
        if(sub==2&&!rec){ ok=gen_file_task(o,id); if(ok){ id++; made[cat]++; } free(pr.p); continue; }
        if(sub==0&&g_nd){ const Doc *d=&g_docs[rnd()%g_nd]; if(strlen(d->text)<200){ free(pr.p); continue; }
            Buf u={0}; bfmt(&u,"Passage (from \"%s\"):\n%s\n\nWrite one question that can be answered only with a specific fact from this passage "
                "(the question must make sense to someone who has NOT seen the passage, so name the subject), and its short answer copied "
                "exactly from the passage (1 to 5 words). Format:\nQ: <question>\nA: <answer>",d->title,d->text);
            char *g=ask(NULL,u.p,200,0.7,o_seed*31+tries); free(u.p);
            char *q=g?strstr(g,"Q:"):NULL, *a=g?strstr(g,"A:"):NULL;
            if(q&&a&&a>q){ *a=0; char *qq=trim(q+2), *aa=trim(a+2); char *nl=strchr(aa,'\n'); if(nl) *nl=0; aa=trim(aa);
                size_t al=strlen(aa); while(al&&(aa[al-1]=='.')) aa[--al]=0;
                if(al&&al<60&&nwords(qq)>=6&&ci_contains(d->text,aa)&&!ci_contains(qq,aa)){ Buf c={0}; bstr(&c,"{\"type\":\"contains\",\"any\":[\""); besc(&c,aa); bstr(&c,"\"]}");
                    snprintf(chk,sizeof chk,"%s",c.p); free(c.p); bstr(&pr,"Look this up in the reference library: "); bstr(&pr,qq); ok=1; } }
            free(g); }
        else if(sub==1||(sub==0&&!g_nd)){
            Buf u={0}; bfmt(&u,"Write one short word problem about %s that needs exact arithmetic with awkward numbers (e.g. multi-digit "
                "multiplication, division with decimals, percentages, or compound growth) so that a calculator is genuinely needed. "
                "Then give a single Python expression that computes the answer. Format:\nPROBLEM: <problem>\nEXPR: <python expression>",TOPICS[rnd()%(sizeof TOPICS/sizeof *TOPICS)]);
            char *g=ask(NULL,u.p,250,0.9,o_seed*37+tries); free(u.p);
            char *p=g?strstr(g,"PROBLEM:"):NULL, *e=g?strstr(g,"EXPR:"):NULL;
            if(p&&e&&e>p){ *e=0; char *pp=trim(p+8), *ee=trim(e+5); char *nl=strchr(ee,'\n'); if(nl) *nl=0;
                if(!py_denied(ee)){ Buf src={0},out={0}; bfmt(&src,"print(float(%s))",ee);
                    if(py_run(src.p,5000,&out,200)==0){ double v=atof(out.p); if(isfinite(v)&&v!=0){
                        snprintf(chk,sizeof chk,"{\"type\":\"number\",\"value\":%.10g,\"tol\":0.006,\"rel\":1e-4}",v);
                        bstr(&pr,pp); bstr(&pr,fabs(v-round(v))>1e-9?" Round to two decimal places if needed.":""); ok=1; } }
                    free(src.p); free(out.p); } }
            free(g); }
        else if(cat==3){
            Buf u={0}; bfmt(&u,"Write one simple, unusual question about %s that any adult can answer from common knowledge or by simple reasoning, "
                "with no calculation tools or lookup needed (e.g. everyday facts, word meanings, simple advice, greetings). Then the key word "
                "that a correct answer must contain. Avoid famous trivia like capital cities. Format:\nQ: <question>\nKEY: <one word>",NTOP[rnd()%(sizeof NTOP/sizeof *NTOP)]);
            char *g=ask(NULL,u.p,120,0.9,o_seed*41+tries); free(u.p);
            char *q=g?strstr(g,"Q:"):NULL, *k=g?strstr(g,"KEY:"):NULL;
            if(q&&k&&k>q){ *k=0; char *qq=trim(q+2), *kk=trim(k+4); char *nl=strchr(kk,'\n'); if(nl) *nl=0; kk=trim(kk);
                size_t kl=strlen(kk); while(kl&&(kk[kl-1]=='.')) kk[--kl]=0;
                if(kl&&kl<30&&!strchr(kk,' ')){ Buf c={0}; bstr(&c,"{\"type\":\"contains\",\"any\":[\""); besc(&c,kk); bstr(&c,"\"]}");
                    snprintf(chk,sizeof chk,"%s",c.p); free(c.p); bstr(&pr,qq); ok=1; } }
            free(g); }
        if(ok&&emit_task(o,id,rec?"recover":cat==0?"lookup":cat==1?"compute":"none",cat==3?"none":sub==0?"required":"any",pr.p,NULL,chk,rec)){ id++; made[cat]++; }
        free(pr.p);
        fprintf(stderr,"\r[gen-tool] lookup %d compute %d file %d none %d recover %d   ",made[0],made[1],made[2],made[3],made[4]);
    }
    fclose(o); fprintf(stderr,"\n");
}
/* teacher rollouts -> verified trajectories in the student's text format */
static void cmd_traj(const char *in, const char *outp){
    int n; char **L=read_lines(in,&n); FILE *o=fopen(outp,"ab"); int kept=0;
    double temp=o_temp<0?0.7:o_temp;
    for(int i=0;i<n;i++){
        J *t=jload(L[i]); int got=0;
        for(int s=0;s<o_n&&!got;s++){ VFS v={0}; Ep ep; episode(t,&v,&ep,o_seed+(unsigned long long)i*104729+s,temp);
            if(ep_ok(t,&ep,&v)){ Buf txt={0}; int ot=o_teacher; o_teacher=0; wrap(&txt,SYS,jgs(t,"prompt","")); o_teacher=ot; bstr(&txt,ep.text.p);
                Buf b={0}; bfmt(&b,"{\"i\":%d,\"kind\":\"agent\",\"cat\":\"%s\",\"text\":\"",i,jgs(t,"cat","")); besc(&b,txt.p); bstr(&b,"\"}\n");
                fwrite(b.p,1,b.n,o); fflush(o); free(b.p); free(txt.p); got=1; kept++; }
            free(ep.answer); free(ep.text.p); vfs_free(&v); }
        jfree(t); fprintf(stderr,"\r[traj] %d/%d kept %d (%.0f%%)   ",i+1,n,kept,100.0*kept/(i+1));
    }
    fclose(o); fprintf(stderr,"\n");
}
/* jsonl docs with "text" (or plain text, blank-line separated) -> title\tpassage, ~W words per passage */
static void cmd_mkstore(const char *in, const char *outp, int W){
    int n; char **L=read_lines(in,&n); FILE *o=fopen(outp,"wb"); int np=0;
    for(int i=0;i<n;i++){
        J *j=L[i][0]=='{'?jload(L[i]):NULL; const char *t=j?jgs(j,"text",NULL):L[i]; if(!t){ jfree(j); continue; }
        const char *ti=j?jgs(j,"title",NULL):NULL; char title[96];
        if(ti) snprintf(title,sizeof title,"%s",ti);
        else { int k=0; while(t[k]&&t[k]!='\n'&&t[k]!='.'&&k<80){ title[k]=t[k]; k++; } title[k]=0; }
        for(char *c=title;*c;c++) if(*c=='\t') *c=' ';
        const char *p=t;
        while(*p){ Buf b={0}; int w=0; bput(&b,"",0);
            while(*p&&w<W){ while(*p&&isspace((unsigned char)*p)) p++; const char *s=p; while(*p&&!isspace((unsigned char)*p)) p++; if(p>s){ if(w) bstr(&b," "); bput(&b,s,p-s); w++; } }
            if(w>=20){ fprintf(o,"%s\t%s\n",title,b.p); np++; } free(b.p); }
        jfree(j);
    }
    fclose(o); printf("[mkstore] %d docs -> %d passages in %s\n",n,np,outp);
}

/* GRPO rollouts: -n G sampled episodes per task; reward 1 = ep_ok, else -0.5 invented result, -0.25 invalid call, 0;
 * rows of groups with mixed rewards get the group-relative advantage "w" = r - mean (Dr. GRPO, no std scaling);
 * -all keeps every row (for teacher scoring). "a0" = byte offset of the transcript in "text" */
static void cmd_group(const char *in, const char *outp){
    int n; char **L=read_lines(in,&n); FILE *o=fopen(outp,"ab"); double temp=o_temp<0?0.8:o_temp; int G=o_n<2?4:o_n>64?64:o_n;
    long rows=0, mixed=0, okn=0;
    for(int i=0;i<n;i++){
        J *t=jload(L[i]); double r[64], mu=0; char *tx[64]; size_t a0=0;
        for(int g=0;g<G;g++){ VFS v={0}; Ep ep; episode(t,&v,&ep,o_seed+(unsigned long long)i*104729+g*7,temp);
            int ok=ep_ok(t,&ep,&v); okn+=ok; r[g]=ok?1.0:ep.halluc?-0.5:ep.calls>ep.valid?-0.25:0.0; mu+=r[g];
            Buf x={0}; wrap(&x,SYS,jgs(t,"prompt","")); a0=x.n; bstr(&x,ep.text.p); tx[g]=x.p;
            free(ep.answer); free(ep.text.p); vfs_free(&v); }
        mu/=G; int mix=0; for(int g=0;g<G;g++) if(fabs(r[g]-mu)>1e-9) mix=1;
        mixed+=mix;
        for(int g=0;g<G;g++){
            if(mix||o_all){ Buf b={0}; bfmt(&b,"{\"i\":%d,\"g\":%d,\"r\":%.2f,\"w\":%.4f,\"a0\":%zu,\"kind\":\"agent\",\"cat\":\"%s\",\"prompt\":\"",i,g,r[g],r[g]-mu,a0,jgs(t,"cat",""));
                besc(&b,jgs(t,"prompt","")); bstr(&b,"\",\"text\":\"");
                besc(&b,tx[g]); bstr(&b,"\"}\n"); fwrite(b.p,1,b.n,o); fflush(o); free(b.p); rows++; }
            free(tx[g]); }
        jfree(t); fprintf(stderr,"\r[group] %d/%d  pass %.1f%%  mixed groups %ld  rows %ld   ",i+1,n,100.0*okn/((double)(i+1)*G),mixed,rows);
    }
    fclose(o); fprintf(stderr,"\n");
}
/* teacher scoring for on-policy distillation: each row's transcript (text from a0) is scored by the teacher's /score
 * under its own chat template (system SYS + user prompt); adds "tl" (teacher token byte lengths) and "tp" (log p) */
static void cmd_tscore(const char *in, const char *outp){
    int n; char **L=read_lines(in,&n); FILE *o=fopen(outp,"ab"); int ok=0, bad=0; double nll=0; long nt=0;
    for(int i=0;i<n;i++){
        J *t=jload(L[i]); const char *tx=jgs(t,"text",""); size_t a0=(size_t)jgn(t,"a0",0);
        if(a0>=strlen(tx)){ bad++; jfree(t); continue; }
        Buf p={0},b={0}; wrap(&p,SYS,jgs(t,"prompt","")); bstr(&b,"{\"prompt\":\""); besc(&b,p.p); bstr(&b,"\",\"completion\":\""); besc(&b,tx+a0); bstr(&b,"\"}");
        char *r=post("/score",b.p); J *j=r?jload(r):NULL, *pl=jget(j,"piece_len"), *lp=jget(j,"logprobs");
        long sum=0; if(pl&&pl->t=='a') for(int k=0;k<pl->len;k++) sum+=(long)pl->v[k]->n;
        if(!pl||!lp||pl->t!='a'||lp->t!='a'||pl->len!=lp->len||sum!=(long)strlen(tx+a0)){ bad++; }
        else { size_t L0=strlen(L[i]); while(L0&&L[i][L0-1]!='}') L0--;
            Buf x={0}; bput(&x,L[i],L0-1); bstr(&x,",\"tl\":[");
            for(int k=0;k<pl->len;k++) bfmt(&x,"%s%d",k?",":"",(int)pl->v[k]->n);
            bstr(&x,"],\"tp\":["); for(int k=0;k<lp->len;k++){ bfmt(&x,"%s%.3f",k?",":"",lp->v[k]->n); nll-=lp->v[k]->n; nt++; }
            bstr(&x,"]}\n"); fwrite(x.p,1,x.n,o); fflush(o); free(x.p); ok++; }
        jfree(j); free(r); free(p.p); free(b.p); jfree(t);
        fprintf(stderr,"\r[tscore] %d/%d  scored %d  misaligned %d  teacher nll %.3f   ",i+1,n,ok,bad,nt?nll/nt:0.0);
    }
    fclose(o); fprintf(stderr,"\n");
}
int main(int argc, char **argv){
    WSADATA wd; WSAStartup(MAKEWORD(2,2),&wd);
    if(argc<3){ fprintf(stderr,
        "usage: agentenv try \"<task>\" | tool <tasks> <out> | code <tasks> <out> | gen-code <out> <n> | gen-tool <out> <n> [-mix L,C,F,N,R] | gen-steps <out> <n>\n"
        "       | traj <tasks> <out> | group <tasks> <out> [-all] | tscore <rows> <out> | mkstore <in> <store.tsv> [words]\n"
        "opts: -port N -teacher -temp T -n samples -seed S -store file -steps N -max N -v\n"); return 1; }
    char *pos[8]; int np=0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-port")&&i+1<argc) o_port=atoi(argv[++i]); else if(!strcmp(argv[i],"-teacher")) o_teacher=1;
        else if(!strcmp(argv[i],"-temp")&&i+1<argc) o_temp=atof(argv[++i]); else if(!strcmp(argv[i],"-n")&&i+1<argc) o_n=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-seed")&&i+1<argc) o_seed=strtoull(argv[++i],0,10); else if(!strcmp(argv[i],"-store")&&i+1<argc) o_store=argv[++i];
        else if(!strcmp(argv[i],"-steps")&&i+1<argc) o_steps=atoi(argv[++i]); else if(!strcmp(argv[i],"-max")&&i+1<argc) o_max=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-traj")) o_trajout=1; else if(!strcmp(argv[i],"-student")) o_student=1; else if(!strcmp(argv[i],"-all")) o_all=1; else if(!strcmp(argv[i],"-mix")&&i+1<argc) sscanf(argv[++i],"%d,%d,%d,%d,%d",o_mix,o_mix+1,o_mix+2,o_mix+3,o_mix+4); else if(!strcmp(argv[i],"-v")) o_v=1; else if(!strcmp(argv[i],"-vv")) o_v=2; else if(np<8) pos[np++]=argv[i]; }
    const char *cmd=pos[0];
    if(!strcmp(cmd,"mkstore")&&np>=3){ cmd_mkstore(pos[1],pos[2],np>3?atoi(pos[3]):120); return 0; }
    if(o_store) store_load(o_store);
    if(!strcmp(cmd,"search")&&np>=2){ Buf b={0}; DWORD t=GetTickCount(); tool_search(pos[1],&b); printf("%s\n(%lu ms)\n",b.p?b.p:"",(unsigned long)(GetTickCount()-t)); return 0; }
    int gen=(!strncmp(cmd,"gen-",4)&&strcmp(cmd,"gen-steps"))||!strcmp(cmd,"traj")||!strcmp(cmd,"tscore");
    if(gen&&!o_student) o_teacher=1;
    if(o_teacher) get_templates();
    if(!strcmp(cmd,"try")){ J *t=calloc(1,sizeof(J)); t->t='o'; Buf b={0}; bstr(&b,"{\"prompt\":\""); besc(&b,pos[1]); bstr(&b,"\"}"); jfree(t); t=jload(b.p);
        VFS v={0}; Ep ep; episode(t,&v,&ep,o_seed,o_temp<0?0:o_temp); printf("%s\n---- calls %d valid %d halluc %d\n",ep.text.p?ep.text.p:"",ep.calls,ep.valid,ep.halluc); return 0; }
    if(!strcmp(cmd,"tool")&&np>=3){ cmd_tool(pos[1],pos[2]); return 0; }
    if(!strcmp(cmd,"code")&&np>=3){ cmd_code(pos[1],pos[2]); return 0; }
    if(!strcmp(cmd,"gen-code")&&np>=3){ cmd_gencode(pos[1],atoi(pos[2])); return 0; }
    if(!strcmp(cmd,"gen-tool")&&np>=3){ cmd_gentool(pos[1],atoi(pos[2])); return 0; }
    if(!strcmp(cmd,"gen-steps")&&np>=3){ cmd_gensteps(pos[1],atoi(pos[2])); return 0; }
    if(!strcmp(cmd,"traj")&&np>=3){ if(o_n<1) o_n=1; cmd_traj(pos[1],pos[2]); return 0; }
    if(!strcmp(cmd,"group")&&np>=3){ cmd_group(pos[1],pos[2]); return 0; }
    if(!strcmp(cmd,"tscore")&&np>=3){ cmd_tscore(pos[1],pos[2]); return 0; }
    fprintf(stderr,"bad command\n"); return 1;
}
