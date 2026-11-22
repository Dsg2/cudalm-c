/* monitor.cuh: `cudalm monitor` - a read-only HTTP view of training runs. No CUDA: it only reads files under the
 * cudalm folder (models/, logs/), so it runs next to training without touching the GPU's memory.
 *   GET /api/runs[?days=14]   runs with a status file or a log written in the last N days
 *   GET /api/run/<m>          one run (status + checkpoints)       GET /api/loss/<m>[?pts=400]  loss/lr/tok/s curve
 *   GET /api/bench/<m>        <m>.bin.bench.tsv rows               GET /api/log/<m>[?tail=80]   newest log, last lines
 *   GET /api/res[/<name>]     chain .res files / one tail          GET /api/sys                 GPU (NVML), disk, time
 *   POST /api/stop/<m>        writes <m>.bin.stop (only with --allow-stop and a token)
 * Run state comes from <m>.bin.status.json (written by 22c+ trainers; a dead pid = "dead"), else from the train log
 * (train-<m>-YYYYMMDD-HHMMSS.log or <m>.train-*.log in logs/ or one folder below): running / bench / stale / ended.
 * Limits: GET/POST only, 8 KB requests read within 5 s, 16 connections at once (more get 503), one thread each.
 * Security: non-loopback binds need --token-file (Authorization: Bearer, constant-time); without a token only
 * Host: localhost / IP literals are served (DNS rebinding); no-store + nosniff on every reply, a CSP on the page. */
#ifndef CUDALM_MONITOR_CUH
#define CUDALM_MONITOR_CUH
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
typedef struct _stat64 MStat;
#define m_stat _stat64
#define m_seek _fseeki64
#define m_ncase _strnicmp
#define M_OS "windows"
#else
#define M_OS "linux"
#include <dlfcn.h>
#include <strings.h>
typedef struct stat MStat;
#define m_stat stat
#define m_seek fseeko
#define m_ncase strncasecmp
#endif

/* ---- files ---- */
static FILE* m_open(const char*p){   /* read without blocking the writer's rename/replace (FILE_SHARE_DELETE) */
#ifdef _WIN32
    HANDLE h=CreateFileA(p,GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,NULL,OPEN_EXISTING,0,NULL);
    if(h==INVALID_HANDLE_VALUE) return NULL;
    int fd=_open_osfhandle((intptr_t)h,_O_RDONLY|_O_BINARY); if(fd<0){ CloseHandle(h); return NULL; }
    return _fdopen(fd,"rb");
#else
    return fopen(p,"rb");
#endif
}
static char* m_slurp(const char*p,size_t cap,size_t*n){ FILE*f=m_open(p); if(!f) return NULL;
    char*b=(char*)malloc(cap+1); size_t k=fread(b,1,cap,f); fclose(f); b[k]=0; if(n) *n=k; return b; }
static long long m_mtime(const char*p,long long*sz){ MStat s; if(m_stat(p,&s)) return 0; if(sz) *sz=(long long)s.st_size; return (long long)s.st_mtime; }
static void m_iso(long long t,char*b,size_t n){ time_t x=(time_t)t; struct tm m;
#ifdef _WIN32
    localtime_s(&m,&x);
#else
    localtime_r(&x,&m);
#endif
    strftime(b,n,"%Y-%m-%dT%H:%M:%S",&m); }
static int m_name_ok(const char*s,size_t n){ if(!n||n>96||s[0]=='.') return 0;
    for(size_t i=0;i<n;i++){ char c=s[i]; if(!((c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='.'||c=='_'||c=='-')) return 0; } return 1; }
static int m_pid_alive(int pid){
    if(pid<=0) return 0;
#ifdef _WIN32
    HANDLE h=OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION|SYNCHRONIZE,FALSE,(DWORD)pid); if(!h) return 0;
    int alive=WaitForSingleObject(h,0)==WAIT_TIMEOUT; char e[1024]; DWORD n=sizeof e;
    if(alive&&QueryFullProcessImageNameA(h,0,e,&n)) alive=strstr(e,"cudalm")!=NULL;   /* a reused pid is not ours */
    CloseHandle(h); return alive;
#else
    char p[64],e[1024]; snprintf(p,sizeof p,"/proc/%d/exe",pid); ssize_t n=readlink(p,e,sizeof e-1);
    if(n<=0) return kill(pid,0)==0;
    e[n]=0; return strstr(e,"cudalm")!=NULL;
#endif
}

/* ---- runs: incremental log parse ---- */
typedef struct { char path[1200]; long long off,mt; int u16; } MLog;   /* u16: PowerShell `*>` logs are UTF-16LE + BOM */
/* UTF-16LE bytes -> malloc'd UTF-8 (n even); *used = bytes consumed through the last complete line ('\n' unit), or all if !lines */
static char* m_u16(const unsigned char*b,size_t n,size_t*outn,size_t*used,int lines){
    char*o=(char*)malloc(n/2*3+1); size_t k=0,last=0,lk=0;
    for(size_t i=0;i+1<n;i+=2){ unsigned v=b[i]|(unsigned)b[i+1]<<8;
        if(v>=0xD800&&v<0xDC00&&i+3<n){ unsigned w=b[i+2]|(unsigned)b[i+3]<<8; if(w>=0xDC00&&w<0xE000){ v=0x10000+((v-0xD800)<<10)+(w-0xDC00); i+=2; } }
        if(v==0xFEFF) continue;
        if(v<0x80) o[k++]=(char)v; else if(v<0x800){ o[k++]=(char)(0xC0|v>>6); o[k++]=(char)(0x80|(v&63)); }
        else if(v<0x10000){ o[k++]=(char)(0xE0|v>>12); o[k++]=(char)(0x80|((v>>6)&63)); o[k++]=(char)(0x80|(v&63)); }
        else { o[k++]=(char)(0xF0|v>>18); o[k++]=(char)(0x80|((v>>12)&63)); o[k++]=(char)(0x80|((v>>6)&63)); o[k++]=(char)(0x80|(v&63)); }
        if(v=='\n'){ last=i+2; lk=k; } }
    if(lines){ k=lk; *used=last; } else *used=n&~(size_t)1;
    o[k]=0; *outn=k; return o; }
static int m_isu16(const char*path){ FILE*f=m_open(path); if(!f) return 0; unsigned char b[2]={0,0}; size_t k=fread(b,1,2,f); fclose(f); return k==2&&b[0]==0xFF&&b[1]==0xFE; }
typedef struct { float it,loss,lr,tps; } MPt;
typedef struct {
    char name[100]; MLog lg[32]; int nl;
    MPt*pt; int np,cap;
    double ema; int iters,stage0,ckpt,ended,inbench,s_it0; double s_dt0,ep,gn,trf,s_rate; char endst[16];
} MRun;
static MRun g_mr[256]; static int g_nmr;
static char g_mroot[1024];
static MRun* m_run(const char*name){
    for(int i=0;i<g_nmr;i++) if(!strcmp(g_mr[i].name,name)) return &g_mr[i];
    if(g_nmr>=256) return NULL;
    MRun*r=&g_mr[g_nmr++]; memset(r,0,sizeof *r); snprintf(r->name,sizeof r->name,"%s",name); r->ema=NAN; r->ep=r->gn=r->trf=NAN; r->s_it0=-1; return r; }
static void m_line(MRun*r,const char*s){
    int it; double ep,L,lr,gn,dt,tps; const char*q;
    if(sscanf(s,"iter %d (ep %lf) | loss %lf | lr %lf | gn %lf | %lfs | %lf tok/s",&it,&ep,&L,&lr,&gn,&dt,&tps)==7){
        while(r->np&&r->pt[r->np-1].it>=(float)it) r->np--;                  /* a resume re-runs iters after its checkpoint */
        if(r->np==r->cap){ r->cap=r->cap?r->cap*2:4096; r->pt=(MPt*)realloc(r->pt,sizeof(MPt)*(size_t)r->cap); }
        MPt p={(float)it,(float)L,(float)lr,(float)tps}; r->pt[r->np++]=p;
        if(isfinite(L)) r->ema=isfinite(r->ema)?0.98*r->ema+0.02*L:L;
        r->ep=ep; r->gn=gn; r->inbench=0; r->ended=0;
        q=strstr(s,"| train "); r->trf=q?atof(q+8)/100.0:1.0;
        if(r->s_it0<0){ r->s_it0=it; r->s_dt0=dt; } else if(dt>r->s_dt0) r->s_rate=(it-r->s_it0)/(dt-r->s_dt0);
        return; }
    if(!strncmp(s,"[ckpt] saved ",13)&&(q=strstr(s,"(resumes at "))){ r->ckpt=atoi(q+12); return; }
    if(!strncmp(s,"[bench] iter ",13)){ if(strstr(s,": lmeval on ")) r->inbench=1; else if(strstr(s,": index ")) r->inbench=0; return; }
    if(!strncmp(s,"[cudalm train] ",15)){
        const char*t=s+15; int a,b;
        if(!strncmp(t,r->name,strlen(r->name))&&(strstr(t," (resume) params=")||strstr(t," (fresh) params="))){ r->ended=0; r->s_it0=-1; r->s_rate=0; r->inbench=0; return; }
        if((q=strstr(t,"iters "))&&sscanf(q,"iters %d..%d",&a,&b)==2){ r->stage0=a; r->iters=b; return; }
        if(strstr(t,"stopping cleanly")){ r->ended=1; snprintf(r->endst,sizeof r->endst,"stopping"); return; }
        if(!strncmp(t,"paused at iter",14)){ r->ended=1; snprintf(r->endst,sizeof r->endst,"paused"); return; }
        if(!strncmp(t,"done -> ",8)){ r->ended=1; snprintf(r->endst,sizeof r->endst,"finished"); return; }
        return; }
    if(!strncmp(s,"[divstop]",9)){ r->ended=1; snprintf(r->endst,sizeof r->endst,"divstop"); return; }
    if(!strncmp(s,"[nan]",5)&&strstr(s,"aborting")){ r->ended=1; snprintf(r->endst,sizeof r->endst,"nan-abort"); }
}
static void m_parse(MRun*r){
    int w=0, gone=0;                                   /* a deleted log takes its points with it: re-read the rest */
    for(int i=0;i<r->nl;i++){ if(m_mtime(r->lg[i].path,NULL)) r->lg[w++]=r->lg[i]; else gone=1; }
    r->nl=w;
    if(gone){ for(int i=0;i<r->nl;i++) r->lg[i].off=0;
        r->np=0; r->ema=NAN; r->s_it0=-1; r->s_rate=0; r->ended=r->inbench=r->ckpt=r->iters=r->stage0=0; }
    for(int i=0;i<r->nl;i++){ MLog*l=&r->lg[i]; long long sz=0, mt=m_mtime(l->path,&sz); if(!mt) continue;
        if(sz<l->off){ l->off=0; }                                       /* rewritten */
        if(sz==l->off){ l->mt=mt; continue; }
        if(!l->off) l->u16=m_isu16(l->path);
        FILE*f=m_open(l->path); if(!f) continue; m_seek(f,l->off,SEEK_SET);
        size_t want=(size_t)(sz-l->off); char*b=(char*)malloc(want+1); size_t k=fread(b,1,want,f); fclose(f); b[k]=0;
        size_t used=0, u16used=0;
        if(l->u16){ size_t n8; char*t=m_u16((const unsigned char*)b,k,&n8,&u16used,1); free(b); b=t; k=n8; }
        char*p=b;
        for(char*nl;(nl=(char*)memchr(p,'\n',k-(size_t)(p-b)));p=nl+1){ *nl=0; if(nl>p&&nl[-1]=='\r') nl[-1]=0; m_line(r,p); used=(size_t)(nl+1-b); }
        l->off+=(long long)(l->u16?u16used:used); l->mt=mt; free(b); }
}
/* model name from a log file name, NULL if it is not a train log */
static int m_logname(const char*fn,char*out,size_t n){
    size_t L=strlen(fn); if(L<5||strcmp(fn+L-4,".log")) return 0;
    if(!strncmp(fn,"train-",6)&&L>6+17+4){ const char*t=fn+L-4-15;   /* YYYYMMDD-HHMMSS */
        int ok=t[-1]=='-'&&t[8]=='-'; for(int i=0;i<15&&ok;i++) if(i!=8) ok=t[i]>='0'&&t[i]<='9';
        if(ok){ size_t m=(size_t)(t-1-(fn+6)); if(m<n&&m_name_ok(fn+6,m)){ memcpy(out,fn+6,m); out[m]=0; return 1; } } }
    const char*d=strstr(fn,".train-"); if(d){ size_t m=(size_t)(d-fn); if(m<n&&m_name_ok(fn,m)){ memcpy(out,fn,m); out[m]=0; return 1; } }
    return 0; }
static void m_addlog(const char*name,const char*path){
    char mp[1200]; snprintf(mp,sizeof mp,"%s" PSEP "models" PSEP "%s.bin",g_mroot,name); if(!lm_exists(mp)) return;
    MRun*r=m_run(name); if(!r) return;
    for(int i=0;i<r->nl;i++) if(!strcmp(r->lg[i].path,path)) return;
    if(r->nl==32){ memmove(r->lg,r->lg+1,sizeof(MLog)*31); r->nl--; }
    long long mt=m_mtime(path,NULL); int at=r->nl;                       /* keep the list in mtime order */
    while(at>0&&r->lg[at-1].mt>mt) at--;
    if(at<r->nl){ /* an older log turned up late: re-read this run from scratch */
        for(int i=0;i<r->nl;i++) r->lg[i].off=0; r->np=0; r->ema=NAN; r->s_it0=-1; r->ended=0; }
    memmove(r->lg+at+1,r->lg+at,sizeof(MLog)*(size_t)(r->nl-at)); r->nl++;
    MLog*l=&r->lg[at]; memset(l,0,sizeof *l); snprintf(l->path,sizeof l->path,"%s",path); l->mt=mt; }
static void m_scandir(const char*dir,int depth){
    char pat[1200]; snprintf(pat,sizeof pat,"%s" PSEP "*",dir); WIN32_FIND_DATAA fd; HANDLE h=FindFirstFileA(pat,&fd);
    if(h==INVALID_HANDLE_VALUE) return;
    do { const char*fn=fd.cFileName; if(fn[0]=='.') continue; char p[1200],nm[100]; snprintf(p,sizeof p,"%s" PSEP "%s",dir,fn);
         if(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY){ if(depth>0) m_scandir(p,depth-1); continue; }
         if(m_logname(fn,nm,sizeof nm)) m_addlog(nm,p);
         else { size_t L=strlen(fn); if(L>17&&!strcmp(fn+L-16,".bin.status.json")&&L-16<sizeof nm&&m_name_ok(fn,L-16)){ memcpy(nm,fn,L-16); nm[L-16]=0; m_run(nm); } }
    } while(FindNextFileA(h,&fd));
    FindClose(h); }
static void m_scan(void){ char d[1100]; snprintf(d,sizeof d,"%s" PSEP "logs",g_mroot); m_scandir(d,1);
    snprintf(d,sizeof d,"%s" PSEP "models",g_mroot); m_scandir(d,0); }   /* runs are parsed on demand (m_parse) */

/* ---- JSON out ---- */
static void m_num(SBuf*o,const char*k,double v,const char*fmt){ char t[96]; int n=snprintf(t,sizeof t,"\"%s\":",k); sb_put(o,t,(size_t)n);
    if(isfinite(v)){ n=snprintf(t,sizeof t,fmt,v); sb_put(o,t,(size_t)n); } else sb_str(o,"null"); sb_str(o,","); }
static void m_s(SBuf*o,const char*k,const char*v){ sb_str(o,"\""); sb_str(o,k); sb_str(o,"\":\""); sb_esc(o,v,strlen(v)); sb_str(o,"\","); }
static void m_end(SBuf*o){ if(o->n&&o->p[o->n-1]==','){ o->n--; o->p[o->n]=0; } }
/* last data row of the bench tsv -> iter, ii, margin; returns rows */
static int m_bench_last(const char*name,int*it,double*ii,double*mi){
    char p[1200]; snprintf(p,sizeof p,"%s" PSEP "models" PSEP "%s.bin.bench.tsv",g_mroot,name);
    FILE*f=m_open(p); if(!f) return 0; char ln[2048]; int n=0;
    while(fgets(ln,sizeof ln,f)){ double ep; if(ln[0]<'0'||ln[0]>'9') continue;
        if(sscanf(ln,"%d\t%lf\t%lf",it,&ep,ii)==3){ const char*t=strrchr(ln,'\t'); char*e; double v=t?strtod(t+1,&e):NAN; *mi=t&&e>t+1?v:NAN; n++; } }
    fclose(f); return n; }
static void m_summary(SBuf*o,MRun*r,long long now){
    char sp[1200],t[32]; snprintf(sp,sizeof sp,"%s" PSEP "models" PSEP "%s.bin.status.json",g_mroot,r->name);
    long long smt=m_mtime(sp,NULL), lmt=r->nl?r->lg[r->nl-1].mt:0;
    char*sj=smt&&smt+300>=lmt?m_slurp(sp,8192,NULL):NULL;               /* the status file of an older session loses to a newer log */
    sb_str(o,"{"); m_s(o,"name",r->name);
    double eta=NAN; int iter=r->np?(int)r->pt[r->np-1].it:0, iters=r->iters;
    if(sj){ const char*q=js_key(sj,"state"); char*st=q&&*q=='"'?js_str(q+1,NULL):strdup("?");
        int pid=(int)js_num(sj,"pid",0), term=!strcmp(st,"paused")||!strcmp(st,"finished")||!strcmp(st,"divstop")||!strcmp(st,"nan-abort");
        q=js_key(sj,"os"); int same=!q||!strncmp(q,"\"" M_OS "\"",strlen(M_OS)+2);   /* a pid means nothing to another OS (WSL watching E:) */
        const char*state=st;
        if(!term){ int slow=now-smt>600&&(!strcmp(st,"running")||!strcmp(st,"starting"));
            if(!same) state=slow?"stale":st; else if(!m_pid_alive(pid)) state="dead"; else if(slow) state="stalled";
            if(state!=st&&strcmp(state,"stalled")) term=1; }   /* no ETA for a dead/stale run */
        m_s(o,"state",state); m_s(o,"source","status"); m_num(o,"pid",pid,"%.0f");
        q=js_key(sj,"exe"); if(q&&*q=='"'){ char*e=js_str(q+1,NULL); m_s(o,"exe",e); free(e); }
        iter=(int)js_num(sj,"iter",iter); iters=(int)js_num(sj,"iters",iters);
        m_num(o,"iter",iter,"%.0f"); m_num(o,"iters",iters,"%.0f"); m_num(o,"stage0",js_num(sj,"stage0",0),"%.0f");
        m_num(o,"ep",js_num(sj,"ep",NAN),"%.4f"); m_num(o,"loss",js_num(sj,"loss",NAN),"%.4f"); m_num(o,"loss_ema",js_num(sj,"loss_ema",NAN),"%.4f");
        m_num(o,"lr",js_num(sj,"lr",NAN),"%.3e"); m_num(o,"gn",js_num(sj,"gn",NAN),"%.3f"); m_num(o,"tok_s",js_num(sj,"tok_s",NAN),"%.0f");
        m_num(o,"train_frac",js_num(sj,"train_frac",NAN),"%.4f"); eta=term?NAN:js_num(sj,"eta_s",NAN);
        static const char*ks[3]={"corpus","schedule","started"};
        for(int k=0;k<3;k++){ const char*k2=ks[k]; q=js_key(sj,k2); if(q&&*q=='"'){ char*e=js_str(q+1,NULL); m_s(o,k2,e); free(e); } }
        if((q=strstr(sj,"\"last_ckpt\":{"))){ m_num(o,"ckpt",js_num(q,"iter",0),"%.0f"); }
        m_iso(smt,t,sizeof t); m_s(o,"updated",t); m_num(o,"age_s",(double)(now-smt),"%.0f");
        free(st); free(sj); }
    else {
        const char*state=r->ended?r->endst:r->inbench?(now-lmt<4*3600?"bench":"stale"):(now-lmt<600?"running":"stale");
        if(!r->nl) state="unknown";
        m_s(o,"state",state); m_s(o,"source","log"); m_num(o,"iter",iter,"%.0f"); m_num(o,"iters",iters,"%.0f"); m_num(o,"stage0",r->stage0,"%.0f");
        MPt*p=r->np?&r->pt[r->np-1]:NULL;
        m_num(o,"ep",r->ep,"%.4f"); m_num(o,"loss",p?p->loss:NAN,"%.4f"); m_num(o,"loss_ema",r->ema,"%.4f"); m_num(o,"lr",p?p->lr:NAN,"%.3e");
        m_num(o,"gn",r->gn,"%.3f"); m_num(o,"tok_s",p?p->tps:NAN,"%.0f"); m_num(o,"train_frac",r->trf,"%.4f");
        if(!r->ended&&r->s_rate>0&&iters>iter) eta=(iters-iter)/r->s_rate;
        if(r->ckpt) m_num(o,"ckpt",r->ckpt,"%.0f");
        if(lmt){ m_iso(lmt,t,sizeof t); m_s(o,"updated",t); m_num(o,"age_s",(double)(now-lmt),"%.0f"); } }
    m_num(o,"eta_s",eta,"%.0f");
    if(r->nl){ const char*lp=r->lg[r->nl-1].path; size_t rl=strlen(g_mroot); m_s(o,"log",strncmp(lp,g_mroot,rl)?lp:lp+rl+1); }
    int bi; double ii,mi; if(m_bench_last(r->name,&bi,&ii,&mi)){ char b[160]; snprintf(b,sizeof b,"\"bench\":{\"iter\":%d,",bi); sb_str(o,b);
        m_num(o,"ii",ii,"%.2f"); m_num(o,"margin",mi,"%.3f"); m_end(o); sb_str(o,"},"); }
    m_end(o); sb_str(o,"}"); }
static long long m_newest(MRun*r){ char sp[1200]; snprintf(sp,sizeof sp,"%s" PSEP "models" PSEP "%s.bin.status.json",g_mroot,r->name);
    long long a=m_mtime(sp,NULL), b=r->nl?m_mtime(r->lg[r->nl-1].path,NULL):0; return a>b?a:b; }

/* ---- NVML (loaded at run time; absent = no GPU fields) ---- */
static struct { int ok,tried; int (*init)(void); int (*count)(unsigned*); int (*byidx)(unsigned,void**); int (*name)(void*,char*,unsigned);
    int (*mem)(void*,unsigned long long*); int (*util)(void*,unsigned*); int (*temp)(void*,int,unsigned*); int (*power)(void*,unsigned*); int (*fan)(void*,unsigned*); } g_nv;
static void m_nvml(void){
    if(g_nv.tried) return; g_nv.tried=1; void*h;
#ifdef _WIN32
    h=(void*)LoadLibraryA("nvml.dll"); if(!h){ char p[600]; const char*pf=getenv("ProgramW6432"); snprintf(p,sizeof p,"%s\\NVIDIA Corporation\\NVSMI\\nvml.dll",pf?pf:"C:\\Program Files"); h=(void*)LoadLibraryA(p); }
    #define NVS(f,n) *(FARPROC*)&g_nv.f=GetProcAddress((HMODULE)h,n)
#else
    h=dlopen("libnvidia-ml.so.1",RTLD_NOW); if(!h) h=dlopen("/usr/lib/wsl/lib/libnvidia-ml.so.1",RTLD_NOW);
    #define NVS(f,n) *(void**)&g_nv.f=dlsym(h,n)
#endif
    if(!h) return;
    NVS(init,"nvmlInit_v2"); NVS(count,"nvmlDeviceGetCount_v2"); NVS(byidx,"nvmlDeviceGetHandleByIndex_v2"); NVS(name,"nvmlDeviceGetName");
    NVS(mem,"nvmlDeviceGetMemoryInfo"); NVS(util,"nvmlDeviceGetUtilizationRates"); NVS(temp,"nvmlDeviceGetTemperature");
    NVS(power,"nvmlDeviceGetPowerUsage"); NVS(fan,"nvmlDeviceGetFanSpeed");
    #undef NVS
    g_nv.ok=g_nv.init&&g_nv.count&&g_nv.byidx&&g_nv.mem&&g_nv.init()==0; }
static void m_sys(SBuf*o,long long now){
    char t[32]; m_iso(now,t,sizeof t); sb_str(o,"{"); m_s(o,"now",t);
    char d[1100]; snprintf(d,sizeof d,"%s" PSEP "models",g_mroot); ULARGE_INTEGER fr;
    if(GetDiskFreeSpaceExA(d,&fr,NULL,NULL)) m_num(o,"disk_free_gb",fr.QuadPart/1e9,"%.1f");
    m_nvml(); sb_str(o,"\"gpus\":[");
    unsigned n=0; if(g_nv.ok) g_nv.count(&n);
    for(unsigned i=0;i<n&&i<8;i++){ void*dv; if(g_nv.byidx(i,&dv)) continue; sb_str(o,i?",{":"{");
        char nm[96]=""; if(g_nv.name) g_nv.name(dv,nm,sizeof nm); m_s(o,"name",nm);
        unsigned long long mm[3]; if(!g_nv.mem(dv,mm)){ m_num(o,"mem_used_mb",mm[2]/1048576.0,"%.0f"); m_num(o,"mem_total_mb",mm[0]/1048576.0,"%.0f"); }
        unsigned u[2],v; if(g_nv.util&&!g_nv.util(dv,u)) m_num(o,"util",u[0],"%.0f");
        if(g_nv.temp&&!g_nv.temp(dv,0,&v)) m_num(o,"temp_c",v,"%.0f");
        if(g_nv.power&&!g_nv.power(dv,&v)) m_num(o,"power_w",v/1000.0,"%.1f");
        if(g_nv.fan&&!g_nv.fan(dv,&v)) m_num(o,"fan",v,"%.0f");
        m_end(o); sb_str(o,"}"); }
    sb_str(o,"]}"); }

/* ---- tails ---- */
static void m_tail(SBuf*o,const char*path,int n){   /* JSON array of the last n lines */
    long long sz=0; m_mtime(path,&sz); int u16=m_isu16(path); FILE*f=m_open(path); sb_str(o,"[");
    if(f){ long long from=sz>262144?(sz-262144)&~1LL:0; m_seek(f,from,SEEK_SET); size_t want=(size_t)(sz-from);
        char*b=(char*)malloc(want+1); size_t k=fread(b,1,want,f); fclose(f); b[k]=0;
        if(u16){ size_t n8,u; char*t=m_u16((const unsigned char*)b,k,&n8,&u,0); free(b); b=t; k=n8; }
        int cnt=0; size_t e=k; while(e>0&&(b[e-1]=='\n'||b[e-1]=='\r')) e--;
        size_t s=e; while(s>0&&cnt<n){ s--; if(b[s]=='\n'){ cnt++; if(cnt==n){ s++; break; } } }
        if(from>0&&s==0){ while(s<e&&b[s]!='\n') s++; if(s<e) s++; }      /* drop a partial first line */
        int first=1; for(size_t a=s;a<e;){ size_t z=a; while(z<e&&b[z]!='\n') z++; size_t m=z; if(m>a&&b[m-1]=='\r') m--;
            sb_str(o,first?"\"":",\""); first=0; sb_esc(o,b+a,m-a); sb_str(o,"\""); a=z+1; }
        free(b); }
    sb_str(o,"]"); }

/* ---- HTTP ---- */
static char g_mtok[80]; static int g_mstop;
static int m_qint(const char*q,const char*k,int d,int lo,int hi){ if(!q) return d; char key[32]; snprintf(key,sizeof key,"%s=",k);
    for(const char*p=q;(p=strstr(p,key));p++) if(p==q||p[-1]=='&'||p[-1]=='?'){ int v=atoi(p+strlen(key)); return v<lo?lo:v>hi?hi:v; }
    return d; }
static int m_authed(const char*rq){
    if(!g_mtok[0]) return 1;
    for(const char*l=strstr(rq,"\r\n");l&&l[2]&&!(l[2]=='\r'&&l[3]=='\n');l=strstr(l+2,"\r\n")){
        if(m_ncase(l+2,"authorization:",14)) continue;
        const char*v=l+16; while(*v==' '||*v=='\t') v++; if(m_ncase(v,"bearer ",7)) return 0; v+=7; while(*v==' ') v++;
        size_t vl=strcspn(v,"\r\n"); while(vl&&(v[vl-1]==' '||v[vl-1]=='\t')) vl--;
        size_t n=strlen(g_mtok); unsigned d=vl!=n;                   /* constant time over the token, never reads past the header */
        for(size_t i=0;i<n;i++) d|=(unsigned)(unsigned char)(i<vl?v[i]:0)^(unsigned char)g_mtok[i];
        return !d; }
    return 0; }
/* without a token, only accept Host: localhost / an IP literal, so a web page cannot reach a loopback monitor
 * through DNS rebinding (a name of its own that resolves to 127.0.0.1) */
static int m_host_ok(const char*rq){
    if(g_mtok[0]) return 1;
    for(const char*l=strstr(rq,"\r\n");l&&l[2]&&!(l[2]=='\r'&&l[3]=='\n');l=strstr(l+2,"\r\n")){
        if(m_ncase(l+2,"host:",5)) continue;
        const char*v=l+7; while(*v==' '||*v=='\t') v++;
        if(*v=='[') return 1;                                        /* IPv6 literal */
        size_t n=strcspn(v,":\r\n "); if(n==9&&!m_ncase(v,"localhost",9)) return 1;
        for(size_t i=0;i<n;i++) if(!(v[i]=='.'||(v[i]>='0'&&v[i]<='9'))) return 0;
        return n>0; }
    return 1; }                                                      /* HTTP/1.0 without Host */
#include "monitor_html.h"
static const char*g_mhfile;   /* --html <file>: serve the page from disk (dashboard development) */
typedef struct { const char*code,*ctype,*extra; char*p; size_t n; } MRep;
static void m_rep(MRep*r,const char*code,const char*ctype,const char*p,size_t n){
    r->code=code; r->ctype=ctype; r->p=(char*)malloc(n+1); memcpy(r->p,p,n); r->n=n; }
#define M_HDR "Cache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\n"
#define M_CSP "Content-Security-Policy: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; " \
              "base-uri 'none'; form-action 'none'; frame-ancestors 'none'\r\nX-Frame-Options: DENY\r\n"
static void m_handle(const char*rq,MRep*rp){
    rp->extra=M_HDR;
    const char*sp=strchr(rq,' '); if(!sp){ m_rep(rp,"400 Bad Request","text/plain","bad request",11); return; }
    int get=!strncmp(rq,"GET ",4), post=!strncmp(rq,"POST ",5);
    #define M_J(c,s) m_rep(rp,c,"application/json",s,sizeof(s)-1)
    if(!get&&!post){ M_J("405 Method Not Allowed","{\"error\":\"GET only\"}"); return; }
    if(!m_host_ok(rq)){ M_J("403 Forbidden","{\"error\":\"Host must be localhost or an IP address (or use a token)\"}"); return; }
    char path[256]; size_t pl=strcspn(sp+1," ?\r\n"); if(pl>=sizeof path) pl=sizeof path-1; memcpy(path,sp+1,pl); path[pl]=0;
    const char*qs=sp[1+pl]=='?'?sp+2+pl:NULL; char qb[200]=""; if(qs){ size_t ql=strcspn(qs," \r\n"); if(ql>=sizeof qb) ql=sizeof qb-1; memcpy(qb,qs,ql); qb[ql]=0; qs=qb; }
    if(get&&(!strcmp(path,"/")||!strcmp(path,"/index.html"))){ size_t n; char*h=g_mhfile?m_slurp(g_mhfile,1<<20,&n):NULL;
        rp->extra=M_HDR M_CSP; m_rep(rp,"200 OK","text/html; charset=utf-8",h?h:g_mhtml,h?n:sizeof g_mhtml-1); free(h); return; }
    if(strncmp(path,"/api/",5)){ M_J("404 Not Found","{\"error\":\"not found\"}"); return; }
    if(!m_authed(rq)){ M_J("401 Unauthorized","{\"error\":\"token required\"}"); return; }
    SBuf o={0}; sb_put(&o,"",0); long long now=(long long)time(NULL); const char*code="200 OK";
    const char*a=path+5, *sl=strchr(a,'/'); char arg[100]=""; size_t al=sl?strlen(sl+1):0;
    if(sl&&!m_name_ok(sl+1,al)){ M_J("404 Not Found","{\"error\":\"bad name\"}"); free(o.p); return; }
    if(sl) memcpy(arg,sl+1,al+1);
    #define IS(x) (!strncmp(a,x,strlen(x))&&(a[strlen(x)]==0||a[strlen(x)]=='/'))
    MRun*r=NULL;
    if(sl&&!IS("res")){ m_scan(); for(int i=0;i<g_nmr;i++) if(!strcmp(g_mr[i].name,arg)) r=&g_mr[i]; if(r) m_parse(r); }
    if(get&&IS("runs")&&!sl){ m_scan(); int days=m_qint(qs,"days",14,0,100000); sb_str(&o,"{\"runs\":[");
        int*ix=(int*)malloc(sizeof(int)*(g_nmr+1)); long long*nw=(long long*)malloc(sizeof(long long)*(g_nmr+1)); int k=0;
        for(int i=0;i<g_nmr;i++){ long long t=m_newest(&g_mr[i]); if(days&&now-t>days*86400LL) continue; int j=k++; while(j>0&&nw[j-1]<t){ ix[j]=ix[j-1]; nw[j]=nw[j-1]; j--; } ix[j]=i; nw[j]=t; }
        for(int i=0;i<k;i++){ if(i) sb_str(&o,","); m_parse(&g_mr[ix[i]]); m_summary(&o,&g_mr[ix[i]],now); }
        free(ix); free(nw); sb_str(&o,"]}"); }
    else if(get&&IS("sys")&&!sl){ static SBuf sc; static long long st; if(!sc.n||now-st>=2){ sc.n=0; m_sys(&sc,now); st=now; } sb_put(&o,sc.p,sc.n); }
    else if(get&&IS("res")&&!sl){ char d[1100]; snprintf(d,sizeof d,"%s" PSEP "logs" PSEP "*.res",g_mroot); WIN32_FIND_DATAA fd; HANDLE h=FindFirstFileA(d,&fd);
        sb_str(&o,"{\"res\":["); int n=0;
        if(h!=INVALID_HANDLE_VALUE){ do { char p[1200],t[32]; size_t L=strlen(fd.cFileName); if(L<5||strcmp(fd.cFileName+L-4,".res")) continue;
                snprintf(p,sizeof p,"%s" PSEP "logs" PSEP "%s",g_mroot,fd.cFileName); long long sz=0, mt=m_mtime(p,&sz); m_iso(mt,t,sizeof t);
                sb_str(&o,n++?",{":"{"); fd.cFileName[L-4]=0; m_s(&o,"name",fd.cFileName); m_num(&o,"bytes",(double)sz,"%.0f"); m_s(&o,"updated",t); m_num(&o,"age_s",(double)(now-mt),"%.0f"); m_end(&o); sb_str(&o,"}");
            } while(FindNextFileA(h,&fd)); FindClose(h); }
        sb_str(&o,"]}"); }
    else if(get&&IS("res")){ char p[1200]; snprintf(p,sizeof p,"%s" PSEP "logs" PSEP "%s.res",g_mroot,arg);
        if(!lm_exists(p)){ code="404 Not Found"; sb_str(&o,"{\"error\":\"no such res\"}"); }
        else { sb_str(&o,"{"); m_s(&o,"name",arg); sb_str(&o,"\"lines\":"); m_tail(&o,p,m_qint(qs,"tail",40,1,1000)); sb_str(&o,"}"); } }
    else if(sl&&!r&&!(post&&IS("stop")&&!(g_mstop&&g_mtok[0]))){ code="404 Not Found"; sb_str(&o,"{\"error\":\"no such run\"}"); }
    else if(get&&IS("run")){ m_summary(&o,r,now); }
    else if(get&&IS("loss")){ int pts=m_qint(qs,"pts",400,10,20000); sb_str(&o,"{"); m_s(&o,"name",r->name); m_num(&o,"iters",r->iters,"%.0f");
        m_num(&o,"points_total",r->np,"%.0f"); sb_str(&o,"\"points\":[");    /* [iter, mean loss, min loss, lr, tok/s] per bucket */
        int B=r->np<pts?r->np:pts; char t[160];
        for(int b=0;b<B;b++){ int s=(int)((long long)r->np*b/B), e=(int)((long long)r->np*(b+1)/B); double sm=0,mn=1e30; int c=0;
            for(int i=s;i<e;i++){ double L=r->pt[i].loss; if(isfinite(L)){ sm+=L; c++; if(L<mn) mn=L; } }
            MPt*z=&r->pt[e-1]; snprintf(t,sizeof t,"%s[%.0f,%.4f,%.4f,%.3e,%.0f]",b?",":"",z->it,c?sm/c:0.0,c?mn:0.0,z->lr,z->tps); sb_str(&o,t); }
        sb_str(&o,"]}"); }
    else if(get&&IS("bench")){ char p[1200],ln[2048]; snprintf(p,sizeof p,"%s" PSEP "models" PSEP "%s.bin.bench.tsv",g_mroot,arg);
        FILE*f=m_open(p); sb_str(&o,"{"); m_s(&o,"name",arg); sb_str(&o,"\"rows\":["); int n=0;
        if(f){ while(fgets(ln,sizeof ln,f)){ int it; double ep,ii; if(ln[0]<'0'||ln[0]>'9'||sscanf(ln,"%d\t%lf\t%lf",&it,&ep,&ii)!=3) continue;
                char*f3=strchr(strchr(strchr(ln,'\t')+1,'\t')+1,'\t'); char*f4=f3?strchr(f3+1,'\t'):NULL; char*e; double mi=f4?strtod(f4+1,&e):NAN; if(f4&&e==f4+1) mi=NAN;
                sb_str(&o,n++?",{":"{"); m_num(&o,"iter",it,"%.0f"); m_num(&o,"ep",ep,"%.4f"); m_num(&o,"ii",ii,"%.2f"); m_num(&o,"margin",mi,"%.3f");
                if(f3){ size_t L=f4?(size_t)(f4-f3-1):strcspn(f3+1,"\r\n"); sb_str(&o,"\"scores\":\""); sb_esc(&o,f3+1,L); sb_str(&o,"\","); }
                m_end(&o); sb_str(&o,"}"); }
            fclose(f); }
        sb_str(&o,"]}"); }
    else if(get&&IS("log")){ sb_str(&o,"{"); m_s(&o,"name",r->name);
        if(r->nl){ const char*lp=r->lg[r->nl-1].path; size_t rl=strlen(g_mroot); m_s(&o,"log",strncmp(lp,g_mroot,rl)?lp:lp+rl+1); sb_str(&o,"\"lines\":"); m_tail(&o,r->lg[r->nl-1].path,m_qint(qs,"tail",80,1,2000)); }
        else sb_str(&o,"\"lines\":[]"); sb_str(&o,"}"); }
    else if(post&&IS("stop")){
        if(!g_mstop||!g_mtok[0]){ code="403 Forbidden"; sb_str(&o,"{\"error\":\"stop is off (start with --allow-stop and --token-file)\"}"); }
        else { char p[1200]; snprintf(p,sizeof p,"%s" PSEP "models" PSEP "%s.bin.stop",g_mroot,arg); FILE*f=fopen(p,"wb");
            if(f){ fclose(f); printf("[monitor] stop requested for %s\n",arg); fflush(stdout); sb_str(&o,"{\"ok\":true}"); } else { code="500 Internal Server Error"; sb_str(&o,"{\"error\":\"cannot write stop file\"}"); } } }
    else { code="404 Not Found"; sb_str(&o,"{\"error\":\"not found\"}"); }
    #undef IS
    #undef M_J
    m_rep(rp,code,"application/json",o.p,o.n); free(o.p); }

/* one connection per thread: the read (5 s total, 8 KB cap) runs unlocked, so a slow client cannot stall the others;
 * the handler runs under the global lock (the run cache is shared), the reply is sent after it */
static int g_mconn;
#define M_MAXCONN 16
static void m_conn(void*u){
    SK cs=(SK)(uintptr_t)u; SBuf rq={0}; const char*body=net_read_req_t(cs,&rq,8192,5000);
    if(!body){ if(rq.n) net_reply_x(cs,"400 Bad Request","text/plain",M_HDR,"bad request",11); else net_close(cs); }
    else { MRep rp={0}; net_timeout(cs,10000); nt_lock(); m_handle(rq.p,&rp); nt_unlock();
        net_reply_x(cs,rp.code,rp.ctype,rp.extra,rp.p,rp.n); free(rp.p); }
    free(rq.p); nt_lock(); g_mconn--; nt_unlock(); }

static void cmd_monitor(int argc,char**argv){
    int port=8091, newtok=0; const char*bind="127.0.0.1",*tokf=NULL;
    snprintf(g_mroot,sizeof g_mroot,"%s",exe_dir());
    for(int i=2;i<argc;i++){ const char*a=argv[i], *v=i+1<argc?argv[i+1]:NULL;
        if(!strcmp(a,"--port")&&v){ port=atoi(v); i++; } else if(!strcmp(a,"--bind")&&v){ bind=v; i++; }
        else if(!strcmp(a,"--root")&&v){ snprintf(g_mroot,sizeof g_mroot,"%s",v); i++; }
        else if(!strcmp(a,"--token-file")&&v){ tokf=v; i++; } else if(!strcmp(a,"--new-token")&&v){ tokf=v; newtok=1; i++; }
        else if(!strcmp(a,"--allow-stop")) g_mstop=1;
        else if(!strcmp(a,"--html")&&v){ g_mhfile=v; i++; }
        else { fprintf(stderr,"usage: cudalm monitor [--port 8091] [--bind 127.0.0.1|0.0.0.0|<ip>] [--root <cudalm folder>]\n"
                              "                      [--token-file <f>] [--allow-stop]\n"
                              "       cudalm monitor --new-token <f>     write a random token file and exit\n"
                              "  Read-only HTTP view of training runs (models/*.status.json, logs/, *.bench.tsv, logs/*.res, NVML).\n"
                              "  A non-loopback --bind needs --token-file; clients send  Authorization: Bearer <token>.\n"); exit(1); } }
    if(newtok){ if(lm_exists(tokf)){ fprintf(stderr,"[monitor] %s exists - not overwritten\n",tokf); exit(1); }
        unsigned char r[16]={0}; int ok=0;
#ifdef _WIN32
        { HMODULE h=LoadLibraryA("advapi32.dll"); BOOLEAN (WINAPI*rg)(void*,ULONG)=NULL; if(h) *(FARPROC*)&rg=GetProcAddress(h,"SystemFunction036"); ok=rg&&rg(r,16); }
#else
        { FILE*u=fopen("/dev/urandom","rb"); ok=u&&fread(r,1,16,u)==16; if(u) fclose(u); }
#endif
        if(!ok){ fprintf(stderr,"[monitor] no system random source\n"); exit(1); }
        FILE*f=fopen(tokf,"wb"); if(!f){ perror(tokf); exit(1); } for(int i=0;i<16;i++) fprintf(f,"%02x",r[i]); fprintf(f,"\n"); fclose(f);
        printf("[monitor] wrote a new token to %s\n",tokf); return; }
    if(tokf){ size_t n; char*t=m_slurp(tokf,200,&n); if(!t){ fprintf(stderr,"[monitor] cannot read %s\n",tokf); exit(1); }
        while(n&&(t[n-1]=='\n'||t[n-1]=='\r'||t[n-1]==' ')) t[--n]=0;
        if(n<16||n>=sizeof g_mtok){ fprintf(stderr,"[monitor] token in %s must be 16..79 characters\n",tokf); exit(1); }
        for(size_t i=0;i<n;i++) if(t[i]<=' '){ fprintf(stderr,"[monitor] token has whitespace\n"); exit(1); }
        memcpy(g_mtok,t,n+1); free(t); }
    int loop=!strncmp(bind,"127.",4);
    if(!loop&&!g_mtok[0]){ fprintf(stderr,"[monitor] --bind %s is reachable from other machines: add --token-file (make one with --new-token)\n",bind); exit(1); }
    char md[1100]; snprintf(md,sizeof md,"%s" PSEP "models",g_mroot); if(!lm_exists(md)){ fprintf(stderr,"[monitor] no models folder under %s (--root)\n",g_mroot); exit(1); }
    SK ls=net_listen(bind,port); if(ls==NET_BAD){ fprintf(stderr,"[monitor] cannot listen on %s:%d\n",bind,port); exit(1); }
    nt_init(); m_scan(); m_nvml();
    printf("[monitor] http://%s:%d  root %s | %d runs known | token %s | stop %s | NVML %s\n",bind,port,g_mroot,g_nmr,
           g_mtok[0]?"required":"off",g_mstop&&g_mtok[0]?"on":"off",g_nv.ok?"ok":"not found"); fflush(stdout);
    for(;;){ SK cs=net_accept(ls); if(cs==NET_BAD){ lm_sleep_ms(10); continue; }
        nt_lock(); int busy=g_mconn>=M_MAXCONN; if(!busy) g_mconn++; nt_unlock();
        if(busy){ SBuf d={0}; net_read_req_t(cs,&d,8192,300); free(d.p);   /* read it first: closing with unread data resets the reply */
            net_timeout(cs,1000); net_reply_x(cs,"503 Service Unavailable","application/json",M_HDR,"{\"error\":\"busy\"}",16); continue; }
        nt_thread(m_conn,(void*)(uintptr_t)cs); }
}
#endif
