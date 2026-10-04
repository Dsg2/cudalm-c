/* data.cuh -- tokenizer + corpus toolchain: `cudalm data <sub>` and the TUI's Data tools (CPU only, no CUDA).
 * Corpus dir    = train.bin/val.bin (uint16, every doc ends in eot) + tokenizer.bin (TLTK) + tokenizer.json
 *                 + template.txt + meta.json (+ canon.bin). A tokenizer dir is the same without the .bin token files.
 * Text input    = .jsonl/.json ({"text":..} or chat {"messages"|"conversations":[{role|from, content|value}],
 *                 "system", "tools"} rendered as chatml markers) and .txt (one doc, <|endoftext|> splits).
 *                 Parquet / Hugging Face downloads go through prep\fetch.py (`data fetch`).
 * Encoding is token-exact with the HF ByteLevel BPE (GPT-2 regex split, then merge ranks) -- the teachtok port.
 * Bare names resolve against the data roots: CUDALM_DATA_ROOTS, default <exe>\data;<exe>\prep\data plus the lines
 * of <exe>\data_roots.txt; new outputs go to the root with the most free space. */
#ifdef _WIN32
#include "uniclass.h"
#pragma comment(lib,"normaliz.lib")

static void data_roots(char*o,size_t cap){ const char*e=getenv("CUDALM_DATA_ROOTS");
    if(e&&*e){ snprintf(o,cap,"%s",e); return; }
    snprintf(o,cap,"%s\\data;%s\\prep\\data",exe_dir(),exe_dir());
    char p[1100],l[1024]; snprintf(p,sizeof p,"%s\\data_roots.txt",exe_dir()); FILE*f=fopen(p,"r"); if(!f) return;
    while(fgets(l,sizeof l,f)){ l[strcspn(l,"\r\n")]=0; size_t n=strlen(o);
        if(*l && *l!='#' && n+strlen(l)+2<cap) snprintf(o+n,cap-n,";%s",l); }
    fclose(f); }
static int d_root(int i,char*o,size_t cap){ char r[2048]; data_roots(r,sizeof r); char*s=r;
    for(;;){ char*e=strchr(s,';'); if(e) *e=0; if(!i){ snprintf(o,cap,"%s",s); return 1; } if(!e) return 0; s=e+1; i--; } }
static int d_isdir(const char*p){ DWORD a=GetFileAttributesA(p); return a!=INVALID_FILE_ATTRIBUTES && (a&FILE_ATTRIBUTE_DIRECTORY); }
static int d_exists(const char*p){ return GetFileAttributesA(p)!=INVALID_FILE_ATTRIBUTES; }
static int d_bare(const char*n){ return !strpbrk(n,"\\/:"); }
static const char*d_base(const char*p){ for(const char*q=p;*q;q++) if(*q=='\\'||*q=='/') p=q+1; return p; }
static long long d_fsize(const char*p){ WIN32_FILE_ATTRIBUTE_DATA a; if(!GetFileAttributesExA(p,GetFileExInfoStandard,&a)) return -1;
    return ((long long)a.nFileSizeHigh<<32)|a.nFileSizeLow; }
static int d_ends(const char*s,const char*e){ size_t a=strlen(s),b=strlen(e); return a>=b && !_stricmp(s+a-b,e); }
static const char* data_resolve(const char*n,char*o,size_t cap){
    snprintf(o,cap,"%s",n); if(!*n||!d_bare(n)||d_isdir(n)) return o;
    char r[1024],p[1200]; for(int i=0;d_root(i,r,sizeof r);i++){ snprintf(p,sizeof p,"%s\\%s",r,n); if(d_isdir(p)){ snprintf(o,cap,"%s",p); break; } }
    return o; }
static const char* data_outdir(const char*n,char*o,size_t cap){
    if(!d_bare(n)){ snprintf(o,cap,"%s",n); return o; }
    char r[1024],best[1024]=""; unsigned long long bf=0;
    for(int i=0;d_root(i,r,sizeof r);i++){ ULARGE_INTEGER fr; if(d_isdir(r)&&GetDiskFreeSpaceExA(r,&fr,NULL,NULL)&&fr.QuadPart>bf){ bf=fr.QuadPart; snprintf(best,sizeof best,"%s",r); } }
    if(!best[0]){ d_root(0,best,sizeof best); lm_mkdir(best); }
    snprintf(o,cap,"%s\\%s",best,n); return o; }

/* ---------------- growable buffer + minimal JSON reader ---------------- */
typedef struct { char*p; size_t n,cap; } DBuf;
static uint64_t d_fnv(const char*s,size_t n){ uint64_t h=1469598103934665603ULL; for(size_t i=0;i<n;i++) h=(h^(unsigned char)s[i])*1099511628211ULL; return h?h:1; }
static void db_put(DBuf*b,const void*s,size_t n){ if(b->n+n+1>b->cap){ b->cap=(b->n+n+1)*2; b->p=(char*)realloc(b->p,b->cap); }
    memcpy(b->p+b->n,s,n); b->n+=n; b->p[b->n]=0; }
static void db_s(DBuf*b,const char*s){ db_put(b,s,strlen(s)); }
static const char* jws(const char*p){ while(*p==' '||*p=='\t'||*p=='\r'||*p=='\n') p++; return p; }
static const char* jstr_end(const char*p){ for(p++;*p&&*p!='"';p++) if(*p=='\\'&&p[1]) p++; return *p?p+1:NULL; }
static const char* jskip(const char*p){ p=jws(p);
    if(*p=='"') return jstr_end(p);
    if(*p=='{'||*p=='['){ int d=0; while(*p){ if(*p=='"'){ p=jstr_end(p); if(!p) return NULL; continue; }
            if(*p=='{'||*p=='[') d++; else if((*p=='}'||*p==']')&&--d==0) return p+1; p++; } return NULL; }
    while(*p&&*p!=','&&*p!='}'&&*p!=']'&&*p!=' '&&*p!='\r'&&*p!='\n'&&*p!='\t') p++; return p; }
static const char* jget(const char*o,const char*key){          /* value of a top-level key, or NULL */
    o=jws(o); if(*o!='{') return NULL; size_t kl=strlen(key); const char*p=jws(o+1);
    while(*p=='"'){ const char*ke=jstr_end(p); if(!ke) return NULL; int hit=(size_t)(ke-p-2)==kl && !memcmp(p+1,key,kl);
        p=jws(ke); if(*p!=':') return NULL; p=jws(p+1); if(hit) return p;
        p=jskip(p); if(!p) return NULL; p=jws(p); if(*p!=',') return NULL; p=jws(p+1); }
    return NULL; }
static const char* jfirst(const char*a){ a=jws(a); if(*a!='['&&*a!='{') return NULL; a=jws(a+1); return *a==']'||*a=='}'||!*a?NULL:a; }
static const char* jnext(const char*e){ e=jskip(e); if(!e) return NULL; e=jws(e); return *e==','?jws(e+1):NULL; }
static const char* jobj_val(const char*k){ const char*p=jstr_end(k); if(!p) return NULL; p=jws(p); return *p==':'?jws(p+1):NULL; }
static const char* jobj_next(const char*k){ const char*v=jobj_val(k); return v?jnext(v):NULL; }
static void d_utf8(DBuf*b,unsigned v){ char o[4]; int n;
    if(v<0x80){ o[0]=(char)v; n=1; } else if(v<0x800){ o[0]=(char)(0xC0|v>>6); o[1]=(char)(0x80|(v&63)); n=2; }
    else if(v<0x10000){ o[0]=(char)(0xE0|v>>12); o[1]=(char)(0x80|((v>>6)&63)); o[2]=(char)(0x80|(v&63)); n=3; }
    else { o[0]=(char)(0xF0|v>>18); o[1]=(char)(0x80|((v>>12)&63)); o[2]=(char)(0x80|((v>>6)&63)); o[3]=(char)(0x80|(v&63)); n=4; }
    db_put(b,o,n); }
static int jdec(const char*p,DBuf*b){                          /* JSON string at p -> UTF-8 appended to b */
    if(*p!='"') return 0; const char*s=++p;
    for(;*p&&*p!='"';p++){
        if(*p!='\\') continue;
        if(p>s) db_put(b,s,p-s);
        char c=*++p; if(!c) break;
        if(c=='u'){ unsigned v=0; sscanf(p+1,"%4x",&v); p+=4;
            if(v>=0xD800&&v<0xDC00&&p[1]=='\\'&&p[2]=='u'){ unsigned w=0; sscanf(p+3,"%4x",&w);
                if(w>=0xDC00&&w<0xE000){ p+=6; v=0x10000+((v-0xD800)<<10)+(w-0xDC00); } }
            if(v>=0xD800&&v<0xE000) v=0xFFFD;
            d_utf8(b,v); }
        else { char o=c=='n'?'\n':c=='t'?'\t':c=='r'?'\r':c=='b'?'\b':c=='f'?'\f':c; db_put(b,&o,1); }
        s=p+1; }
    if(p>s) db_put(b,s,p-s);
    if(!b->p) db_put(b,"",0);
    return 1; }
static void jesc(FILE*f,const char*s){ fputc('"',f);
    for(;*s;s++){ unsigned char c=(unsigned char)*s; if(c=='"'||c=='\\') fprintf(f,"\\%c",c); else if(c<32) fprintf(f,"\\u%04x",c); else fputc(c,f); }
    fputc('"',f); }

/* ---------------- docs: JSON object -> text (chat rendered with the chatml markers) ---------------- */
typedef struct { DBuf o,t,m,a; } DWs;
static const char* d_role(const char*r,size_t l){
    static const char*m[][2]={{"system","system"},{"human","user"},{"user","user"},{"prompter","user"},{"gpt","assistant"},
        {"assistant","assistant"},{"chatgpt","assistant"},{"tool","result"},{"observation","result"},{"function_response","result"},
        {"function_call","call"},{"tool_call","call"}};
    for(size_t i=0;i<sizeof m/sizeof m[0];i++) if(strlen(m[i][0])==l && !_strnicmp(r,m[i][0],l)) return m[i][1];
    return NULL; }
static int d_ws(char c){ return c==' '||c=='\n'||c=='\t'||c=='\r'||c=='\f'||c=='\v'; }
static void d_trim(const char**s,size_t*n){ while(*n&&d_ws(**s)){ (*s)++; (*n)--; } while(*n&&d_ws((*s)[*n-1])) (*n)--; }
/* first non-blank string among keys -> b (decoded, trimmed); 0 = none */
static int d_field(const char*obj,const char*const*k,int nk,DBuf*b){
    for(int i=0;i<nk;i++){ const char*v=jget(obj,k[i]); if(!v||*v!='"') continue;
        b->n=0; jdec(v,b); const char*s=b->p; size_t n=b->n; d_trim(&s,&n);
        if(n){ memmove(b->p,s,n); b->n=n; return 1; } }
    return 0; }
static void d_part(DBuf*o,int*np,const char*role,const char*s,size_t n){
    d_trim(&s,&n); if(!n) return; if(*np) db_put(o,"\n",1);
    db_s(o,"<|"); db_s(o,role); db_s(o,"|>\n"); db_put(o,s,n); (*np)++; }
/* JSON value -> text the way Python's json.dumps(v, ensure_ascii=False) prints it (strings decoded) */
static void jval(const char*v,DBuf*o){
    if(*v=='"'){ jdec(v,o); return; }
    const char*e=jskip(v); if(!e) return;
    for(const char*p=v;p<e;){
        if(*p=='"'){ DBuf t={0}; jdec(p,&t); db_put(o,"\"",1);
            for(size_t i=0;i<t.n;i++){ unsigned char c=(unsigned char)t.p[i]; char x[8];
                if(c=='"'||c=='\\'){ x[0]='\\'; x[1]=(char)c; db_put(o,x,2); }
                else if(c<32){ const char*m=c=='\n'?"\\n":c=='\r'?"\\r":c=='\t'?"\\t":c=='\b'?"\\b":c=='\f'?"\\f":NULL;
                    if(m) db_s(o,m); else { snprintf(x,sizeof x,"\\u%04x",c); db_s(o,x); } }
                else db_put(o,&t.p[i],1); }
            db_put(o,"\"",1); free(t.p); p=jstr_end(p); if(!p) return; continue; }
        if(*p==',') db_s(o,", "); else if(*p==':') db_s(o,": "); else if(!(*p==' '||*p=='\n'||*p=='\r'||*p=='\t')) db_put(o,p,1);
        p++; } }
static int d_render(const char*obj,DWs*w){
    int np=0; const char*v; w->o.n=0;
    if((v=jget(obj,"system"))&&*v=='"'){ w->t.n=0; jdec(v,&w->t); d_part(&w->o,&np,"system",w->t.p,w->t.n); }
    if((v=jget(obj,"tools"))&&*v!='n'){ w->t.n=0; jval(v,&w->t); d_part(&w->o,&np,"tools",w->t.p,w->t.n); }
    const char*a=jget(obj,"messages"); if(!a) a=jget(obj,"conversations"); if(!a) a=jget(obj,"conversation");
    if(!a){ static const char*qk[]={"instruction","prompt","question"},*ak[]={"output","response","answer"},*ik[]={"input","context"};
        if(!d_field(obj,qk,3,&w->t)||!d_field(obj,ak,3,&w->a)) return 0;      /* alpaca-style pair */
        if(d_field(obj,ik,2,&w->m)){ db_s(&w->t,"\n\n"); db_put(&w->t,w->m.p,w->m.n); }
        d_part(&w->o,&np,"user",w->t.p,w->t.n); d_part(&w->o,&np,"assistant",w->a.p,w->a.n); return np>=2; }
    if(*a=='"'){ w->a.n=0; jdec(a,&w->a); a=w->a.p; }           /* parquet exports store the list as a JSON string */
    for(const char*e=jfirst(a);e;e=jnext(e)){ if(*e!='{') continue;
        const char*r=jget(e,"from"); if(!r) r=jget(e,"role");
        const char*c=jget(e,"value"); if(!c) c=jget(e,"content");
        if(!r||*r!='"'||!c||*c=='n') continue;
        w->t.n=0; jdec(r,&w->t); const char*role=d_role(w->t.p,w->t.n); if(!role) continue;
        w->t.n=0; jval(c,&w->t); d_part(&w->o,&np,role,w->t.p,w->t.n); }
    return np>=2; }
/* text of one JSON object + its category ("kind/cat" or "cat"); 0 = no usable doc */
static int d_doc(const char*obj,DWs*w,const char**txt,size_t*n,char*cat,size_t ccap){
    static const char*tk[]={"text"},*ck[]={"content","code","markdown","body"};
    const char*v=jget(obj,"text");
    if(v&&*v=='"'){ if(!d_field(obj,tk,1,&w->o)) return 0; }
    else if(!d_render(obj,w)&&!d_field(obj,ck,4,&w->o)) return 0;
    *txt=w->o.p; *n=w->o.n; cat[0]=0;
    const char*k=jget(obj,"kind"),*c=jget(obj,"cat");
    if(k&&*k=='"'&&c&&*c=='"'){ w->m.n=0; jdec(k,&w->m); db_put(&w->m,"/",1); jdec(c,&w->m); snprintf(cat,ccap,"%s",w->m.p); }
    else if(c&&*c=='"'){ w->m.n=0; jdec(c,&w->m); snprintf(cat,ccap,"%s",w->m.p); }
    return *n>0; }

/* ---------------- input files ---------------- */
typedef struct { char**f; int n,cap,npq; } DFiles;
static void d_addf(DFiles*l,const char*p){ if(l->n==l->cap){ l->cap=l->cap?l->cap*2:64; l->f=(char**)realloc(l->f,sizeof(char*)*l->cap); } l->f[l->n++]=_strdup(p); }
static int d_textfile(const char*n){ return d_ends(n,".jsonl")||d_ends(n,".json")||d_ends(n,".txt"); }
static void d_walk(DFiles*l,const char*dir){
    char pat[1200]; snprintf(pat,sizeof pat,"%s\\*",dir); WIN32_FIND_DATAA fd; HANDLE h=FindFirstFileA(pat,&fd);
    if(h==INVALID_HANDLE_VALUE) return;
    do{ if(fd.cFileName[0]=='.'||fd.cFileName[0]=='_') continue;
        char p[1200]; snprintf(p,sizeof p,"%s\\%s",dir,fd.cFileName);
        if(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY) d_walk(l,p);
        else if(d_textfile(fd.cFileName)) d_addf(l,p); else if(d_ends(fd.cFileName,".parquet")) l->npq++;
    }while(FindNextFileA(h,&fd)); FindClose(h); }
static int d_inputs(const char*spec,DFiles*l){                   /* ';'-separated files / folders */
    char s[4096]; snprintf(s,sizeof s,"%s",spec); memset(l,0,sizeof *l);
    for(char*p=s,*e;p;p=e){ e=strchr(p,';'); if(e) *e++=0; while(*p==' ') p++; if(!*p) continue;
        char r[1200]; data_resolve(p,r,sizeof r);
        if(d_isdir(r)) d_walk(l,r); else if(d_exists(r)){ if(d_ends(r,".parquet")) l->npq++; else d_addf(l,r); }
        else { fprintf(stderr,"[data] input not found: %s\n",p); return 0; } }
    if(l->npq) fprintf(stderr,"[data] %d parquet file(s) skipped - convert them first (Data tools > Get text, or: cudalm data fetch <file|folder> <out>)\n",l->npq);
    if(!l->n) fprintf(stderr,"[data] no .jsonl/.json/.txt files in: %s\n",spec);
    return l->n; }

/* reader: calls fn per doc. fn returns 0 go on, 1 stop, 2 skip the rest of this file.
 * follow=1: a .jsonl is tailed until <file>.done exists (idle() is called while waiting). */
typedef int (*DDocFn)(void*ctx,int fi,const char*txt,size_t n,const char*cat);
typedef void (*DIdleFn)(void*ctx);
static long long g_dbad;
static int d_line(const char*s,DWs*w,DDocFn fn,void*ctx,int fi,const char*stem){
    s=jws(s); if(!*s) return 0; if(*s!='{'){ g_dbad++; return 0; }
    const char*t; size_t n; char cat[96]; if(!d_doc(s,w,&t,&n,cat,sizeof cat)){ g_dbad++; return 0; }
    return fn(ctx,fi,t,n,cat[0]?cat:stem); }
static int d_read(DFiles*fl,int follow,DDocFn fn,DIdleFn idle,void*ctx){
    DWs w; memset(&w,0,sizeof w); int r=0;
    for(int fi=0;fi<fl->n && r!=1;fi++){
        const char*path=fl->f[fi]; char stem[96]; const char*b=d_base(path);
        snprintf(stem,sizeof stem,"%s",b); char*x=strrchr(stem,'.'); if(x) *x=0; x=strstr(stem,"__"); if(x) *x=0;
        if(d_ends(path,".txt")){ size_t len; uint8_t*t=read_file(path,&len); if(t){ r=fn(ctx,fi,(const char*)t,len,stem); free(t); } continue; }
        FILE*f=fopen(path,"rb"); if(!f){ fprintf(stderr,"[data] cannot open %s\n",path); continue; }
        size_t cap=1<<24, have=0, st=0; char*buf=(char*)malloc(cap); int first=1, arr=-1, fin=0;
        char done[1200]; snprintf(done,sizeof done,"%s.done",path);
        if(d_ends(path,".json")){ int c; while((c=fgetc(f))==' '||c=='\n'||c=='\r'||c=='\t'||c==0xEF||c==0xBB||c==0xBF); arr=c=='['; rewind(f); }
        if(arr==1){ fclose(f); free(buf); size_t len; char*t=(char*)read_file(path,&len); if(!t) continue;
            t=(char*)realloc(t,len+1); t[len]=0;
            for(const char*e=jfirst(t);e&&r==0;e=jnext(e)) if(*e=='{'){ const char*q; size_t n; char cat[96];
                if(d_doc(e,&w,&q,&n,cat,sizeof cat)) r=fn(ctx,fi,q,n,cat[0]?cat:stem); else g_dbad++; }
            free(t); continue; }
        for(;;){
            char*nl=(char*)memchr(buf+st,'\n',have-st);
            if(nl){ *nl=0; char*s=buf+st; st=nl-buf+1;
                if(first){ first=0; if(!memcmp(s,"\xEF\xBB\xBF",3)) s+=3; }
                if((r=d_line(s,&w,fn,ctx,fi,stem))) break; continue; }
            memmove(buf,buf+st,have-st); have-=st; st=0;
            if(have+(1<<22)>cap){ cap*=2; buf=(char*)realloc(buf,cap); }
            size_t got=fread(buf+have,1,cap-have-1,f); if(got){ have+=got; continue; }
            if(follow && !fin){ if(d_exists(done)) fin=1; else { if(idle) idle(ctx); Sleep(5000); } clearerr(f); continue; }
            if(have){ buf[have]=0; r=d_line(buf,&w,fn,ctx,fi,stem); }
            break; }
        fclose(f); free(buf);
        if(r==2) r=0; }
    free(w.o.p); free(w.t.p); free(w.m.p); free(w.a.p);
    return r; }

/* ---------------- tokenizer (TLTK) + exact encoder ---------------- */
typedef struct { int V,eot,nm,base[256]; size_t mcap; uint64_t*mkey; int*mnew,*mrank; uint8_t*blob; size_t blen; Tok*tk; } DTok;
static size_t dt_slot(const DTok*t,uint64_t k){ size_t i=(size_t)((k*0x9E3779B97F4A7C15ULL)>>20)&(t->mcap-1);
    while(t->mrank[i]>=0 && t->mkey[i]!=k) i=(i+1)&(t->mcap-1); return i; }
static DTok* dtok_load(const char*dir){
    char p[1200]; snprintf(p,sizeof p,"%s\\tokenizer.bin",dir); size_t len; uint8_t*b=read_file(p,&len);
    if(!b){ fprintf(stderr,"[data] no tokenizer.bin in %s\n",dir); return NULL; }
    DTok*t=(DTok*)calloc(1,sizeof(DTok)); t->blob=b; t->blen=len; t->tk=tok_parse(b,len,0);
    Rd r={b,b+len}; r.p+=4; rd(&r,&t->V,4); rd(&r,&t->eot,4); rd(&r,&t->nm,4);
    for(int i=0;i<t->V;i++){ int l=0; rd(&r,&l,4); r.p+=l; }
    rd(&r,t->base,4*256);
    t->mcap=16; while(t->mcap<(size_t)t->nm*2) t->mcap<<=1;
    t->mkey=(uint64_t*)malloc(8*t->mcap); t->mnew=(int*)malloc(4*t->mcap); t->mrank=(int*)malloc(4*t->mcap); memset(t->mrank,-1,4*t->mcap);
    for(int m=0;m<t->nm;m++){ int a,c,nw; if(!rd(&r,&a,4)||!rd(&r,&c,4)||!rd(&r,&nw,4)) break;
        uint64_t k=(uint64_t)a*t->V+c; size_t i=dt_slot(t,k); if(t->mrank[i]<0){ t->mkey[i]=k; t->mrank[i]=m; t->mnew[i]=nw; } }
    return t; }
static int d_inr(const unsigned (*r)[2],int n,unsigned c){ int lo=0,hi=n-1;
    while(lo<=hi){ int m=(lo+hi)/2; if(c<r[m][0]) hi=m-1; else if(c>r[m][1]) lo=m+1; else return 1; } return 0; }
static int d_isW(unsigned c){ return (c>=9&&c<=13)||c==32||c==0x85||c==0xA0||c==0x1680||(c>=0x2000&&c<=0x200A)||c==0x2028||c==0x2029||c==0x202F||c==0x205F||c==0x3000; }
static int d_cls(unsigned c){ if(d_isW(c)) return 0;
    if(c<128) return ((c|32)>='a'&&(c|32)<='z')?1:(c>='0'&&c<='9')?2:3;
    return d_inr(UNI_L,sizeof UNI_L/8,c)?1:d_inr(UNI_N,sizeof UNI_N/8,c)?2:3; }
/* GPT-2 ByteLevel regex split ('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+); cp/off hold n+1 */
typedef void (*DPieceFn)(void*,const unsigned char*,int);
static void d_split(const unsigned char*s,int n,unsigned*cp,int*off,DPieceFn f,void*ctx){
    int k=0;
    for(int i=0;i<n;){ unsigned c=s[i]; int l=c<0x80?1:c<0xE0?2:c<0xF0?3:4; if(i+l>n) l=n-i;
        unsigned v=l==1?c:l==2?c&31:l==3?c&15:c&7; for(int j=1;j<l;j++) v=v<<6|(s[i+j]&63);
        off[k]=i; cp[k++]=v; i+=l; }
    off[k]=n;
    for(int i=0;i<k;){ int j=i+1; unsigned c=cp[i];
        if(c=='\''&&i+1<k){ unsigned a=cp[i+1],b=i+2<k?cp[i+2]:0;
            if(a=='s'||a=='t'||a=='m'||a=='d') j=i+2; else if((a=='r'&&b=='e')||(a=='v'&&b=='e')||(a=='l'&&b=='l')) j=i+3; }
        if(j==i+1){ int st=(c==' '&&i+1<k&&!d_isW(cp[i+1]))?i+1:i, t=d_cls(cp[st]);
            if(t){ j=st+1; while(j<k&&d_cls(cp[j])==t) j++; }
            else { j=i+1; while(j<k&&d_isW(cp[j])) j++; if(j<k&&j-i>=2) j--; } }
        f(ctx,s+off[i],off[j]-off[i]); i=j; } }
static const char* d_memstr(const char*h,size_t n,const char*nd,size_t m){
    if(n<m) return NULL; const char*e=h+n-m;
    for(const char*p=h;p<=e;p++){ p=(const char*)memchr(p,nd[0],e-p+1); if(!p) return NULL; if(!memcmp(p,nd,m)) return p; }
    return NULL; }
typedef struct { const DTok*t; int*ids; size_t n,cap; unsigned*cp; int*off; size_t cb; int*w; size_t cw; } DEnc;
static void de_put(DEnc*e,int v){ if(e->n==e->cap){ e->cap=e->cap?e->cap*2:1<<16; e->ids=(int*)realloc(e->ids,4*e->cap); } e->ids[e->n++]=v; }
static void de_bpe(void*ctx,const unsigned char*s,int n){
    DEnc*e=(DEnc*)ctx; const DTok*t=e->t;
    if((size_t)n>e->cw){ e->cw=n; e->w=(int*)realloc(e->w,4*e->cw); }
    int*w=e->w,c=n; for(int i=0;i<n;i++) w[i]=t->base[s[i]];
    while(c>1){ int best=-1,bi=-1; size_t bs=0;
        for(int i=0;i+1<c;i++){ size_t sl=dt_slot(t,(uint64_t)w[i]*t->V+w[i+1]); int r=t->mrank[sl]; if(r>=0&&(best<0||r<best)){ best=r; bi=i; bs=sl; } }
        if(bi<0) break; w[bi]=t->mnew[bs]; memmove(w+bi+1,w+bi+2,4*(c-bi-2)); c--; }
    for(int i=0;i<c;i++) de_put(e,w[i]); }
static void de_seg(DEnc*e,const char*s,size_t n){ if(!n) return;
    if(n+1>e->cb){ e->cb=n+1; e->cp=(unsigned*)realloc(e->cp,4*e->cb); e->off=(int*)realloc(e->off,4*e->cb); }
    d_split((const unsigned char*)s,(int)n,e->cp,e->off,de_bpe,e); }
static void de_text(DEnc*e,const char*s,size_t n){ const char*end=s+n;
    for(;;){ const char*p=d_memstr(s,end-s,"<|endoftext|>",13); if(!p){ de_seg(e,s,end-s); return; }
        de_seg(e,s,p-s); de_put(e,e->t->eot); s=p+13; } }
static int d_samefile(const char*a,const char*b){ size_t la,lb; uint8_t*x=read_file(a,&la),*y=read_file(b,&lb);
    int same=x&&y&&la==lb&&!memcmp(x,y,la); free(x); free(y); return same; }
static void d_copy(const char*sd,const char*dd,const char*nm){ char a[1200],b[1200];
    snprintf(a,sizeof a,"%s\\%s",sd,nm); snprintf(b,sizeof b,"%s\\%s",dd,nm); if(d_exists(a)) CopyFileA(a,b,FALSE); }
static void d_template(const char*dir){ char p[1200]; snprintf(p,sizeof p,"%s\\template.txt",dir);
    if(!d_exists(p)){ FILE*f=fopen(p,"wb"); if(f){ fputs("chatml",f); fclose(f); } } }
static void d_now(char*o,size_t cap){ SYSTEMTIME s; GetLocalTime(&s); snprintf(o,cap,"%04d-%02d-%02d %02d:%02d",s.wYear,s.wMonth,s.wDay,s.wHour,s.wMinute); }

/* ---------------- canon map: token -> smallest id with the same NFKC + lowercase + leading-space-stripped text ---------------- */
static int d_canon(const DTok*t,const char*outp){
    int V=t->V; uint16_t*cn=(uint16_t*)malloc(2*V); size_t hc=1; while(hc<(size_t)V*4) hc<<=1;
    uint64_t*hk=(uint64_t*)calloc(hc,8); int*hv=(int*)malloc(4*hc); int merged=0;
    for(int i=0;i<V;i++){ cn[i]=(uint16_t)i; if(i==t->eot||t->tk->declen[i]<=0) continue;
        wchar_t w[600],nz[2400],lw[2400]; char key[4800];
        int wl=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,(const char*)t->tk->dec[i],t->tk->declen[i],w,600); if(wl<=0) continue;
        int nl=NormalizeString(NormalizationKC,w,wl,nz,2400); if(nl<=0) continue;
        int ll=LCMapStringEx(LOCALE_NAME_INVARIANT,LCMAP_LOWERCASE,nz,nl,lw,2400,NULL,NULL,0); if(ll<=0) continue;
        int s=0; while(s<ll&&lw[s]==L' ') s++; if(s==ll) continue;
        int kl=WideCharToMultiByte(CP_UTF8,0,lw+s,ll-s,key,sizeof key,NULL,NULL); if(kl<=0) continue;
        uint64_t h=1469598103934665603ULL; for(int j=0;j<kl;j++) h=(h^(unsigned char)key[j])*1099511628211ULL; if(!h) h=1;
        size_t j=h&(hc-1); while(hk[j]&&hk[j]!=h) j=(j+1)&(hc-1);
        if(hk[j]){ cn[i]=(uint16_t)hv[j]; merged++; } else { hk[j]=h; hv[j]=i; } }
    FILE*f=fopen(outp,"wb"); if(f){ fwrite(cn,2,V,f); fclose(f); }
    printf("[canon] %s: V=%d, %d tokens merged into %d classes\n",outp,V,merged,V-merged);
    free(cn); free(hk); free(hv); return f?0:1; }

/* ---------------- encode: text -> corpus ---------------- */
typedef struct { size_t off,len,tok; int cat,ntok; char val; } DDoc;
typedef struct { DEnc e; const char*ar; DDoc*d; int d0,d1; } DJob;
typedef struct { char k[64]; long long docs,tok,odocs,otok; } DCat;
#define D_NCAT 256
static DWORD WINAPI d_job(LPVOID p){ DJob*j=(DJob*)p; j->e.n=0;
    for(int i=j->d0;i<j->d1;i++){ DDoc*d=j->d+i; size_t s=j->e.n; de_text(&j->e,j->ar+d->off,d->len); de_put(&j->e,j->e.t->eot); d->tok=s; d->ntok=(int)(j->e.n-s); }
    return 0; }
static int d_nthreads(void){ const char*e=getenv("CUDALM_DATA_THREADS"); int n=e?atoi(e):(int)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    return n<1?1:n>64?64:n; }
static int dcat_id(DCat*c,int*n,const char*k){ for(int i=0;i<*n;i++) if(!strcmp(c[i].k,k)) return i;
    if(*n>=D_NCAT) return *n-1; memset(&c[*n],0,sizeof(DCat)); snprintf(c[*n].k,64,"%s",k); return (*n)++; }
typedef struct {
    DTok*t; DJob jb[64]; int T; DBuf ar; DDoc*d; int nd,cd; FILE*ft,*fv; int vN,dedup,append;
    long long kept,dups,ntr,nva; DCat cat[D_NCAT]; int ncat; uint64_t*seen; size_t scap,sn; uint16_t*u; size_t cu;
    DWORD t0,tl; long long bytes; const char*out;
} DEc;
static int d_seen(DEc*c,const char*s,size_t n){            /* 1 = new text */
    uint64_t h=1469598103934665603ULL; for(size_t i=0;i<n;i++) h=(h^(unsigned char)s[i])*1099511628211ULL; if(!h) h=1;
    if(c->sn*2>=c->scap){ uint64_t*o=c->seen; size_t oc=c->scap; c->scap=c->scap?c->scap*2:1<<20; c->seen=(uint64_t*)calloc(c->scap,8);
        for(size_t i=0;i<oc;i++) if(o[i]){ size_t j=o[i]&(c->scap-1); while(c->seen[j]) j=(j+1)&(c->scap-1); c->seen[j]=o[i]; } free(o); }
    size_t j=h&(c->scap-1); while(c->seen[j]){ if(c->seen[j]==h) return 0; j=(j+1)&(c->scap-1); }
    c->seen[j]=h; c->sn++; return 1; }
static void d_flush(DEc*c){
    if(!c->nd) return;
    size_t per=c->ar.n/c->T+1, acc=0; int t=0, s=0; HANDLE h[64]; int nj=0;
    for(int i=0;i<=c->nd;i++){
        if(i==c->nd || (acc>=per*(t+1) && t<c->T-1)){ if(i>s){ DJob*j=&c->jb[nj]; j->ar=c->ar.p; j->d=c->d; j->d0=s; j->d1=i;
                h[nj]=CreateThread(NULL,0,d_job,j,0,NULL); nj++; } s=i; t++; }
        if(i<c->nd) acc+=c->d[i].len; }
    WaitForMultipleObjects(nj,h,TRUE,INFINITE); for(int i=0;i<nj;i++) CloseHandle(h[i]);
    for(int j=0;j<nj;j++){ DJob*jb=&c->jb[j];
        for(int i=jb->d0;i<jb->d1;i++){ DDoc*d=&c->d[i]; int*ids=jb->e.ids+d->tok;
            if((size_t)d->ntok>c->cu){ c->cu=d->ntok*2; c->u=(uint16_t*)realloc(c->u,2*c->cu); }
            for(int k=0;k<d->ntok;k++) c->u[k]=(uint16_t)ids[k];
            fwrite(c->u,2,d->ntok,d->val?c->fv:c->ft); if(d->val) c->nva+=d->ntok; else c->ntr+=d->ntok;
            c->cat[d->cat].docs++; c->cat[d->cat].tok+=d->ntok; } }
    c->nd=0; c->ar.n=0; fflush(c->ft); if(c->fv) fflush(c->fv);
    DWORD now=GetTickCount(); if(now-c->tl>10000){ c->tl=now; double s2=(now-c->t0)/1000.0;
        printf("[encode] %6.0fs  %lld docs (%lld dup)  train %.1fM  val %.2fM tok  %.0f MB text  %.1f MB/s\n",
               s2,c->kept,c->dups,c->ntr/1e6,c->nva/1e6,c->bytes/1e6,c->bytes/1e6/(s2>0?s2:1)); fflush(stdout); } }
static int d_enc_doc(void*ctx,int fi,const char*txt,size_t n,const char*cat){
    DEc*c=(DEc*)ctx; (void)fi;
    if(c->dedup && !d_seen(c,txt,n)){ c->dups++; return 0; }
    if(c->nd==c->cd){ c->cd=c->cd?c->cd*2:65536; c->d=(DDoc*)realloc(c->d,sizeof(DDoc)*c->cd); }
    DDoc*d=&c->d[c->nd++]; d->off=c->ar.n; d->len=n; d->cat=dcat_id(c->cat,&c->ncat,cat);
    d->val=(char)(!c->append && c->vN>0 && c->kept%c->vN==0); c->kept++; c->bytes+=n;
    db_put(&c->ar,txt,n);
    if(c->ar.n>=(64u<<20) || c->nd>=200000) d_flush(c);
    return 0; }
static void d_enc_idle(void*ctx){ d_flush((DEc*)ctx); }
static void d_meta_old(const char*path,DEc*c,DBuf*inputs,DBuf*hist){   /* carry an existing meta.json forward */
    size_t len; char*b=(char*)read_file(path,&len); if(!b) return; b=(char*)realloc(b,len+1); b[len]=0;
    const char*v=jget(b,"categories");
    if(v) for(const char*k=jfirst(v);k&&*k=='"';k=jobj_next(k)){ const char*o=jobj_val(k); if(!o) break;
        DBuf nm={0}; jdec(k,&nm); int i=dcat_id(c->cat,&c->ncat,nm.p); free(nm.p);
        const char*x=jget(o,"docs"); c->cat[i].odocs+=x?_atoi64(x):0;
        x=jget(o,"tokens"); if(!x) x=jget(o,"emitted"); c->cat[i].otok+=x?_atoi64(x):0; }
    for(int pass=0;pass<2;pass++){ v=jget(b,pass?"history":"inputs"); DBuf*o=pass?hist:inputs;
        if(v) for(const char*e=jfirst(v);e;e=jnext(e)){ const char*q=jskip(e); if(!q) break; if(o->n) db_s(o,", "); db_put(o,e,q-e); } }
    free(b); }
static void d_meta_write(const char*out,const DTok*t,const char*tokdir,DEc*c,DBuf*inputs,DBuf*hist){
    char p[1200],q[1200]; snprintf(q,sizeof q,"%s\\meta.json.tmp",out); FILE*f=fopen(q,"wb"); if(!f) return;
    snprintf(p,sizeof p,"%s\\train.bin",out); long long tr=d_fsize(p)/2; snprintf(p,sizeof p,"%s\\val.bin",out); long long va=d_fsize(p); va=va>0?va/2:0;
    const char*nm=d_base(out);
    fprintf(f,"{\n  \"profile\": "); jesc(f,nm);
    fprintf(f,",\n  \"vocab_size\": %d,\n  \"eot_id\": %d,\n  \"total_tokens\": %lld,\n  \"train_tokens\": %lld,\n  \"val_tokens\": %lld,\n",t->V,t->eot,tr+va,tr,va);
    long long docs=0; for(int i=0;i<c->ncat;i++) docs+=c->cat[i].docs+c->cat[i].odocs;
    fprintf(f,"  \"docs\": %lld,\n  \"tokenizer\": ",docs); jesc(f,tokdir);
    fprintf(f,",\n  \"built_by\": \"cudalm data encode\",\n  \"inputs\": [%s],\n  \"history\": [%s],\n  \"categories\": {\n",inputs->p?inputs->p:"",hist->p?hist->p:"");
    long long tt=0; for(int i=0;i<c->ncat;i++) tt+=c->cat[i].tok+c->cat[i].otok;
    for(int i=0;i<c->ncat;i++){ DCat*x=&c->cat[i]; fprintf(f,"    "); jesc(f,x->k);
        fprintf(f,": {\"docs\": %lld, \"tokens\": %lld, \"pct\": %.2f}%s\n",x->docs+x->odocs,x->tok+x->otok,100.0*(x->tok+x->otok)/(tt?tt:1),i+1<c->ncat?",":""); }
    fprintf(f,"  }\n}\n"); fclose(f);
    snprintf(p,sizeof p,"%s\\meta.json",out); MoveFileExA(q,p,MOVEFILE_REPLACE_EXISTING); }
static int cmd_data_encode(int argc,char**argv){
    if(argc<7){ fprintf(stderr,"usage: cudalm data encode <new|append> <out> <inputs> <tokenizer_dir> [val%%=1] [dedup=1] [follow=0]\n"); return 1; }
    int append=!strcmp(argv[3],"append"); if(!append && strcmp(argv[3],"new")){ fprintf(stderr,"mode must be new or append\n"); return 1; }
    char out[1200],tokd[1200],p[1200],q[1200]; DFiles fl;
    if(append) data_resolve(argv[4],out,sizeof out); else data_outdir(argv[4],out,sizeof out);
    data_resolve(argv[6],tokd,sizeof tokd);
    snprintf(p,sizeof p,"%s\\train.bin",out);
    if(append && !d_exists(p)){ fprintf(stderr,"[encode] append: no corpus at %s\n",out); return 1; }
    if(!append && d_exists(p)){ fprintf(stderr,"[encode] %s already holds a corpus - pick a new name, or mode=append to add to it\n",out); return 1; }
    DTok*t=dtok_load(tokd); if(!t) return 1;
    if(t->V>65536){ fprintf(stderr,"[encode] vocab %d does not fit uint16 tokens\n",t->V); return 1; }
    if(append){ snprintf(q,sizeof q,"%s\\tokenizer.bin",tokd); snprintf(p,sizeof p,"%s\\tokenizer.bin",out);
        if(!d_samefile(p,q)){ fprintf(stderr,"[encode] REFUSED: %s uses a different tokenizer than %s\n"
            "         (appending would mix two token->text mappings in one train.bin)\n",out,tokd); return 1; } }
    if(!d_inputs(argv[5],&fl)) return 1;
    if(!append){ if(!CreateDirectoryA(out,NULL) && !d_isdir(out)){ fprintf(stderr,"[encode] cannot create %s\n",out); return 1; }
        const char*cp[]={"tokenizer.bin","tokenizer.json","template.txt","canon.bin"}; for(int i=0;i<4;i++) d_copy(tokd,out,cp[i]); d_template(out); }
    static DEc c; memset(&c,0,sizeof c); c.t=t; c.append=append; c.out=out;
    double vp=argc>7?atof(argv[7]):1.0; c.vN=vp>0?(int)(100.0/vp+0.5):0; if(vp>0&&c.vN<2) c.vN=2;
    c.dedup=argc>8?atoi(argv[8]):1; int follow=argc>9?atoi(argv[9]):0;
    c.T=d_nthreads(); for(int i=0;i<c.T;i++) c.jb[i].e.t=t;
    snprintf(p,sizeof p,"%s\\train.bin",out); c.ft=fopen(p,append?"ab":"wb");
    snprintf(p,sizeof p,"%s\\val.bin",out); c.fv=append?NULL:fopen(p,"wb");
    if(!c.ft||(!append&&!c.fv)){ fprintf(stderr,"[encode] cannot write in %s\n",out); return 1; }
    DBuf inputs={0},hist={0}; snprintf(p,sizeof p,"%s\\meta.json",out); long long tr0=0;
    if(append){ d_meta_old(p,&c,&inputs,&hist); snprintf(q,sizeof q,"%s\\train.bin",out); tr0=d_fsize(q)/2; }
    printf("[encode] %s %s <- %d file(s), tokenizer %s (V=%d), %d threads, val %s, dedup %s%s\n",append?"append to":"new",out,fl.n,tokd,t->V,c.T,
           append?"unchanged (append adds to train.bin only)":vp>0?"every Nth doc":"none",c.dedup?"on":"off",follow?", follow":"");
    if(!append&&vp>0) printf("         val = every %d-th doc (%.2f%%)\n",c.vN,100.0/c.vN);
    c.t0=c.tl=GetTickCount(); g_dbad=0;
    d_read(&fl,follow,d_enc_doc,d_enc_idle,&c); d_flush(&c);
    fclose(c.ft); if(c.fv) fclose(c.fv);
    { if(inputs.n) db_s(&inputs,", "); DBuf e={0}; db_s(&e,"\"");
      for(const char*s=argv[5];*s;s++){ if(*s=='"'||*s=='\\') db_put(&e,"\\",1); db_put(&e,s,1); } db_s(&e,"\""); db_s(&inputs,e.p); free(e.p);
      char now[40],h[300]; d_now(now,sizeof now);
      snprintf(h,sizeof h,"\"%s %s: %lld docs, +%lld train tokens (train.bin was %lld tokens)\"",now,append?"append":"new",c.kept,c.ntr,tr0);
      if(hist.n) db_s(&hist,", "); db_s(&hist,h); }
    d_meta_write(out,t,tokd,&c,&inputs,&hist);
    printf("[encode] done in %.1f s: %lld docs (%lld duplicates skipped, %lld unreadable lines), +%lld train, +%lld val tokens -> %s\n",
           (GetTickCount()-c.t0)/1000.0,c.kept,c.dups,g_dbad,c.ntr,c.nva,out);
    for(int i=0;i<c.ncat;i++) if(c.cat[i].docs) printf("  %-28s %9lld docs %12lld tok %7.0f tok/doc\n",c.cat[i].k,c.cat[i].docs,c.cat[i].tok,(double)c.cat[i].tok/c.cat[i].docs);
    return 0; }

/* ---------------- tokenizer training: byte-level BPE (HF-compatible output) ---------------- */
typedef struct { uint64_t h; uint32_t off,len; int64_t cnt; } DWord;
typedef struct { DWord*t; size_t cap,n; DBuf ar; long long budget,used,fbud,fused; int fi,nf; DEnc e; } DTr;
static void tr_piece(void*ctx,const unsigned char*s,int n){
    DTr*x=(DTr*)ctx; if(n>256) return;
    uint64_t h=1469598103934665603ULL; for(int i=0;i<n;i++) h=(h^s[i])*1099511628211ULL; if(!h) h=1;
    if(x->n*2>=x->cap){ size_t oc=x->cap; DWord*o=x->t; x->cap=oc?oc*2:1<<20; x->t=(DWord*)calloc(x->cap,sizeof(DWord));
        for(size_t i=0;i<oc;i++) if(o[i].h){ size_t j=o[i].h&(x->cap-1); while(x->t[j].h) j=(j+1)&(x->cap-1); x->t[j]=o[i]; } free(o); }
    size_t j=h&(x->cap-1);
    while(x->t[j].h){ DWord*w=&x->t[j]; if(w->h==h&&w->len==(uint32_t)n&&!memcmp(x->ar.p+w->off,s,n)){ w->cnt++; return; } j=(j+1)&(x->cap-1); }
    x->t[j].h=h; x->t[j].off=(uint32_t)x->ar.n; x->t[j].len=n; x->t[j].cnt=1; db_put(&x->ar,s,n); x->n++; }
static int tr_doc(void*ctx,int fi,const char*t,size_t n,const char*cat){
    DTr*x=(DTr*)ctx; (void)cat;
    if(fi!=x->fi){ x->fi=fi; x->fbud=(x->budget-x->used)/(x->nf-fi); x->fused=0; }
    const char*end=t+n;
    for(const char*s=t;;){ const char*p=d_memstr(s,end-s,"<|endoftext|>",13); size_t l=p?(size_t)(p-s):(size_t)(end-s);
        if(l){ if(l+1>x->e.cb){ x->e.cb=l+1; x->e.cp=(unsigned*)realloc(x->e.cp,4*x->e.cb); x->e.off=(int*)realloc(x->e.off,4*x->e.cb); }
            d_split((const unsigned char*)s,(int)l,x->e.cp,x->e.off,tr_piece,x); }
        if(!p) break; s=p+13; }
    x->used+=n; x->fused+=n;
    if(x->used>=x->budget) return 1;
    return x->fused>=x->fbud?2:0; }
typedef struct { int64_t c; uint64_t k; } DHe;
static DHe*g_hp; static size_t g_hn,g_hc;
static int he_gt(const DHe*a,const DHe*b){ return a->c>b->c||(a->c==b->c&&a->k<b->k); }
static void he_push(int64_t c,uint64_t k){ if(g_hn==g_hc){ g_hc=g_hc?g_hc*2:1<<20; g_hp=(DHe*)realloc(g_hp,sizeof(DHe)*g_hc); }
    size_t i=g_hn++; DHe e={c,k}; while(i){ size_t p=(i-1)/2; if(!he_gt(&e,&g_hp[p])) break; g_hp[i]=g_hp[p]; i=p; } g_hp[i]=e; }
static DHe he_pop(void){ DHe top=g_hp[0], e=g_hp[--g_hn]; size_t i=0;
    for(;;){ size_t l=2*i+1; if(l>=g_hn) break; if(l+1<g_hn&&he_gt(&g_hp[l+1],&g_hp[l])) l++; if(!he_gt(&g_hp[l],&e)) break; g_hp[i]=g_hp[l]; i=l; }
    if(g_hn) g_hp[i]=e; return top; }
/* pair table: key (a<<32|b) -> index; count + list of words holding it */
static uint64_t*g_pk; static int64_t*g_pc; static int**g_po; static int*g_pn,*g_pcap,*g_pst; static int g_np,g_npc; static int*g_ph; static size_t g_phc;
static int pr_get(uint64_t k,int make){
    if(make && (size_t)g_np*2>=g_phc){ size_t oc=g_phc; g_phc=oc?oc*2:1<<22; free(g_ph); g_ph=(int*)malloc(4*g_phc); memset(g_ph,-1,4*g_phc);
        for(int i=0;i<g_np;i++){ size_t j=(size_t)((g_pk[i]*0x9E3779B97F4A7C15ULL)>>17)&(g_phc-1); while(g_ph[j]>=0) j=(j+1)&(g_phc-1); g_ph[j]=i; } }
    size_t j=(size_t)((k*0x9E3779B97F4A7C15ULL)>>17)&(g_phc-1);
    while(g_ph[j]>=0){ if(g_pk[g_ph[j]]==k) return g_ph[j]; j=(j+1)&(g_phc-1); }
    if(!make) return -1;
    if(g_np==g_npc){ g_npc=g_npc?g_npc*2:1<<20; g_pk=(uint64_t*)realloc(g_pk,8*g_npc); g_pc=(int64_t*)realloc(g_pc,8*g_npc);
        g_po=(int**)realloc(g_po,sizeof(int*)*g_npc); g_pn=(int*)realloc(g_pn,4*g_npc); g_pcap=(int*)realloc(g_pcap,4*g_npc); g_pst=(int*)realloc(g_pst,4*g_npc); }
    int i=g_np++; g_pk[i]=k; g_pc[i]=0; g_po[i]=NULL; g_pn[i]=g_pcap[i]=0; g_pst[i]=0; g_ph[j]=i; return i; }
static void pr_occ(int q,int w){ if(g_pn[q]==g_pcap[q]){ g_pcap[q]=g_pcap[q]?g_pcap[q]*2:4; g_po[q]=(int*)realloc(g_po[q],4*g_pcap[q]); } g_po[q][g_pn[q]++]=w; }
#define PK(a,b) (((uint64_t)(uint32_t)(a)<<32)|(uint32_t)(b))
static void d_b2u(unsigned*u){ int n=0; for(int b=0;b<256;b++) u[b]=((b>='!'&&b<='~')||(b>=0xA1&&b<=0xAC)||(b>=0xAE&&b<=0xFF))?(unsigned)b:256u+n++; }
static void d_jtok(FILE*f,const unsigned*u,const uint8_t*s,int n){       /* token bytes -> HF ByteLevel string, JSON-quoted */
    fputc('"',f); for(int i=0;i<n;i++){ unsigned c=u[s[i]]; if(c=='"'||c=='\\') fprintf(f,"\\%c",c); else if(c<0x80) fputc(c,f); else { fputc(0xC0|c>>6,f); fputc(0x80|(c&63),f); } }
    fputc('"',f); }
static int cmd_data_tokenizer(int argc,char**argv){
    if(argc<5){ fprintf(stderr,"usage: cudalm data tokenizer <out> <inputs> [vocab=16384] [sample_mb=300]\n"); return 1; }
    char out[1200],p[1200]; data_outdir(argv[3],out,sizeof out); DFiles fl;
    int V=argc>5?atoi(argv[5]):16384; double mb=argc>6?atof(argv[6]):300;
    if(V<512||V>65535){ fprintf(stderr,"[tokenizer] vocab must be 512..65535 (tokens are stored as uint16)\n"); return 1; }
    snprintf(p,sizeof p,"%s\\tokenizer.bin",out); if(d_exists(p)){ fprintf(stderr,"[tokenizer] %s already has a tokenizer - pick a new name\n",out); return 1; }
    if(!d_inputs(argv[4],&fl)) return 1;
    static DTr x; memset(&x,0,sizeof x); x.budget=(long long)(mb*1e6); x.fi=-1; x.nf=fl.n;
    DWORD t0=GetTickCount(); g_dbad=0;
    printf("[tokenizer] counting words in up to %.0f MB of text from %d file(s) (an equal share per file)\n",mb,fl.n); fflush(stdout);
    d_read(&fl,0,tr_doc,NULL,&x);
    printf("[tokenizer] %.1f MB read, %zu distinct words, %.1f s\n",x.used/1e6,x.n,(GetTickCount()-t0)/1000.0); fflush(stdout);
    unsigned u[256]; d_b2u(u); int base[256], rk[256]; for(int b=0;b<256;b++) rk[b]=b;
    for(int i=1;i<256;i++){ int v=rk[i],j=i; while(j>0&&u[rk[j-1]]>u[v]){ rk[j]=rk[j-1]; j--; } rk[j]=v; }
    for(int i=0;i<256;i++) base[rk[i]]=1+i;
    size_t nw=0, ns=0; for(size_t i=0;i<x.cap;i++) if(x.t[i].h&&x.t[i].len>=2){ nw++; ns+=x.t[i].len; }
    int*sym=(int*)malloc(4*(ns?ns:1)); size_t*ws=(size_t*)malloc(sizeof(size_t)*(nw?nw:1)); int*wl=(int*)malloc(4*(nw?nw:1)); int64_t*wc=(int64_t*)malloc(8*(nw?nw:1)); int*wst=(int*)malloc(4*(nw?nw:1));
    { size_t w=0,o=0; for(size_t i=0;i<x.cap;i++){ DWord*d=&x.t[i]; if(!d->h||d->len<2) continue;
        ws[w]=o; wl[w]=d->len; wc[w]=d->cnt; wst[w]=-1; for(uint32_t k=0;k<d->len;k++) sym[o++]=base[(unsigned char)x.ar.p[d->off+k]]; w++; } }
    free(x.t); free(x.ar.p);
    for(size_t w=0;w<nw;w++){ int*s=sym+ws[w]; for(int i=0;i+1<wl[w];i++){ int q=pr_get(PK(s[i],s[i+1]),1); g_pc[q]+=wc[w]; pr_occ(q,(int)w); } }
    for(int i=0;i<g_np;i++) he_push(g_pc[i],g_pk[i]);
    int mcap2=V,nv=257; int*ma=(int*)malloc(4*mcap2),*mb2=(int*)malloc(4*mcap2),*mz=(int*)malloc(4*mcap2); int*touch=(int*)malloc(4*1024); int tcap=1024;
    DBuf tb={0}; size_t*toff=(size_t*)malloc(sizeof(size_t)*V); int*tlen=(int*)malloc(4*V);
    size_t shc=1; while(shc<(size_t)V*4) shc<<=1; int*sh=(int*)calloc(shc,4);       /* token bytes -> id+1 (two merges can spell the same token) */
    toff[0]=0; tlen[0]=0; for(int b=0;b<256;b++){ unsigned char c=(unsigned char)b; toff[base[b]]=tb.n; tlen[base[b]]=1; db_put(&tb,&c,1); }
    for(int i=1;i<257;i++){ size_t j=(d_fnv(tb.p+toff[i],1)*0x9E3779B97F4A7C15ULL>>20)&(shc-1); while(sh[j]) j=(j+1)&(shc-1); sh[j]=i+1; }
    int m=0; DWORD tl=GetTickCount();
    while(nv<V){
        int pi=-1;
        while(g_hn){ DHe h=he_pop(); int i=pr_get(h.k,0); if(g_pc[i]!=h.c){ if(g_pc[i]>0) he_push(g_pc[i],h.k); continue; } pi=i; break; }
        if(pi<0||g_pc[pi]<2) break;
        int A=(int)(g_pk[pi]>>32), B=(int)(uint32_t)g_pk[pi], Z=nv;
        { DBuf*b=&tb; size_t oa=toff[A],ob=toff[B]; int la=tlen[A],lb=tlen[B];
          if(b->n+la+lb+1>b->cap){ b->cap=(b->n+la+lb+1)*2; b->p=(char*)realloc(b->p,b->cap); } memcpy(b->p+b->n,b->p+oa,la); memcpy(b->p+b->n+la,b->p+ob,lb);
          const char*s=b->p+b->n; size_t j=(d_fnv(s,la+lb)*0x9E3779B97F4A7C15ULL>>20)&(shc-1);
          while(sh[j] && !(tlen[sh[j]-1]==la+lb && !memcmp(b->p+toff[sh[j]-1],s,la+lb))) j=(j+1)&(shc-1);
          if(sh[j]) Z=sh[j]-1; else { toff[Z]=b->n; tlen[Z]=la+lb; b->n+=la+lb; sh[j]=Z+1; nv++; } }
        if(m==mcap2){ mcap2*=2; ma=(int*)realloc(ma,4*mcap2); mb2=(int*)realloc(mb2,4*mcap2); mz=(int*)realloc(mz,4*mcap2); }
        ma[m]=A; mb2[m]=B; mz[m]=Z;
        int*ol=g_po[pi], on=g_pn[pi], nt=0; g_po[pi]=NULL; g_pn[pi]=g_pcap[pi]=0;
        for(int j=0;j<on;j++){ int w=ol[j]; if(wst[w]==m) continue; wst[w]=m;
            int*s=sym+ws[w], L=wl[w], has=0; for(int i=0;i+1<L;i++) if(s[i]==A&&s[i+1]==B){ has=1; break; } if(!has) continue;
            int64_t c=wc[w];
            for(int i=0;i+1<L;i++) g_pc[pr_get(PK(s[i],s[i+1]),0)]-=c;
            int o=0; for(int i=0;i<L;){ if(i+1<L&&s[i]==A&&s[i+1]==B){ s[o++]=Z; i+=2; } else s[o++]=s[i++]; } wl[w]=L=o;
            for(int i=0;i+1<L;i++){ int q=pr_get(PK(s[i],s[i+1]),1); g_pc[q]+=c;
                if(s[i]==Z||s[i+1]==Z){ pr_occ(q,w); if(g_pst[q]!=m+1){ g_pst[q]=m+1; if(nt==tcap){ tcap*=2; touch=(int*)realloc(touch,4*tcap); } touch[nt++]=q; } } } }
        free(ol); m++;
        for(int i=0;i<nt;i++) if(g_pc[touch[i]]>0) he_push(g_pc[touch[i]],g_pk[touch[i]]);
        if(GetTickCount()-tl>10000){ tl=GetTickCount(); printf("[tokenizer] %d / %d tokens (last pair count %lld)\n",nv,V,(long long)g_pc[pi]); fflush(stdout); }
    }
    int Vwant=V; V=nv;
    if(!CreateDirectoryA(out,NULL)&&!d_isdir(out)){ fprintf(stderr,"[tokenizer] cannot create %s\n",out); return 1; }
    FILE*f=fopen(p,"wb"); int eot=0; fwrite("TLTK",1,4,f); fwrite(&V,4,1,f); fwrite(&eot,4,1,f); fwrite(&m,4,1,f);
    for(int i=0;i<V;i++){ fwrite(&tlen[i],4,1,f); fwrite(tb.p+toff[i],1,tlen[i],f); }
    fwrite(base,4,256,f); for(int i=0;i<m;i++){ fwrite(&ma[i],4,1,f); fwrite(&mb2[i],4,1,f); fwrite(&mz[i],4,1,f); }
    fclose(f);
    snprintf(p,sizeof p,"%s\\tokenizer.json",out); f=fopen(p,"wb");
    fprintf(f,"{\n  \"version\": \"1.0\",\n  \"truncation\": null,\n  \"padding\": null,\n  \"added_tokens\": [\n    {\n      \"id\": 0,\n"
              "      \"content\": \"<|endoftext|>\",\n      \"single_word\": false,\n      \"lstrip\": false,\n      \"rstrip\": false,\n"
              "      \"normalized\": false,\n      \"special\": true\n    }\n  ],\n  \"normalizer\": null,\n"
              "  \"pre_tokenizer\": {\n    \"type\": \"ByteLevel\",\n    \"add_prefix_space\": false,\n    \"trim_offsets\": true,\n    \"use_regex\": true\n  },\n"
              "  \"post_processor\": null,\n"
              "  \"decoder\": {\n    \"type\": \"ByteLevel\",\n    \"add_prefix_space\": true,\n    \"trim_offsets\": true,\n    \"use_regex\": true\n  },\n"
              "  \"model\": {\n    \"type\": \"BPE\",\n    \"dropout\": null,\n    \"unk_token\": null,\n    \"continuing_subword_prefix\": null,\n"
              "    \"end_of_word_suffix\": null,\n    \"fuse_unk\": false,\n    \"byte_fallback\": false,\n    \"ignore_merges\": false,\n    \"vocab\": {\n"
              "      \"<|endoftext|>\": 0");
    for(int i=1;i<V;i++){ fprintf(f,",\n      "); d_jtok(f,u,(const uint8_t*)tb.p+toff[i],tlen[i]); fprintf(f,": %d",i); }
    fprintf(f,"\n    },\n    \"merges\": [");
    for(int i=0;i<m;i++){ fprintf(f,"%s\n      [\n        ",i?",":""); d_jtok(f,u,(const uint8_t*)tb.p+toff[ma[i]],tlen[ma[i]]);
        fprintf(f,",\n        "); d_jtok(f,u,(const uint8_t*)tb.p+toff[mb2[i]],tlen[mb2[i]]); fprintf(f,"\n      ]"); }
    fprintf(f,"\n    ]\n  }\n}"); fclose(f);
    d_template(out);
    snprintf(p,sizeof p,"%s\\meta.json",out); f=fopen(p,"wb");
    fprintf(f,"{\n  \"vocab_size\": %d,\n  \"eot_id\": 0,\n  \"built_by\": \"cudalm data tokenizer\",\n  \"sample_bytes\": %lld,\n  \"inputs\": [",V,x.used);
    jesc(f,argv[4]); fprintf(f,"]\n}\n"); fclose(f);
    DTok*t=dtok_load(out); snprintf(p,sizeof p,"%s\\canon.bin",out); d_canon(t,p);
    printf("[tokenizer] %s: V=%d (%d merges%s), %.1f s\n",out,V,m,V<Vwant?" - ran out of pairs seen twice":"",(GetTickCount()-t0)/1000.0);
    return 0; }

/* ---------------- mix: weighted blend of corpora (doc-level, shuffled) ---------------- */
typedef struct { HANDLE f,m; const uint16_t*p; size_t n; } DMap;
static int dmap(DMap*d,const char*path){ memset(d,0,sizeof *d);
    d->f=CreateFileA(path,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_FLAG_SEQUENTIAL_SCAN,NULL); if(d->f==INVALID_HANDLE_VALUE){ d->f=NULL; return 0; }
    LARGE_INTEGER s; GetFileSizeEx(d->f,&s); d->n=(size_t)s.QuadPart/2; if(!d->n) return 1;
    d->m=CreateFileMappingA(d->f,NULL,PAGE_READONLY,0,0,NULL); d->p=d->m?(const uint16_t*)MapViewOfFile(d->m,FILE_MAP_READ,0,0,0):NULL; return d->p!=NULL; }
static void dunmap(DMap*d){ if(d->p) UnmapViewOfFile(d->p); if(d->m) CloseHandle(d->m); if(d->f) CloseHandle(d->f); memset(d,0,sizeof *d); }
typedef struct { uint64_t s; uint32_t n; } DSpan;
static DSpan* d_docs(const DMap*m,int eot,size_t*nd){         /* doc spans (each ends in eot) */
    size_t cap=1<<16,n=0,s=0; DSpan*d=(DSpan*)malloc(sizeof(DSpan)*cap);
    for(size_t i=0;i<m->n;i++) if(m->p[i]==eot||i+1==m->n){ if(n==cap){ cap*=2; d=(DSpan*)realloc(d,sizeof(DSpan)*cap); }
        size_t l=i+1-s; d[n].s=s; d[n].n=(uint32_t)(l>0xFFFFFFFFu?0xFFFFFFFFu:l); n++; s=i+1; }
    *nd=n; return d; }
static uint64_t g_mx=0x9E3779B97F4A7C15ULL;
static uint64_t d_rnd(void){ g_mx^=g_mx<<13; g_mx^=g_mx>>7; g_mx^=g_mx<<17; return g_mx; }
static int cmd_data_mix(int argc,char**argv){
    if(argc<9||(argc-7)%2){ fprintf(stderr,"usage: cudalm data mix <out> <train_tokens|0> <max_repeat> <val_tokens> <corpus> <weight> [<corpus> <weight> ...]\n"
        "  train_tokens 0 = as many as the weights allow without exceeding max_repeat on any corpus\n"); return 1; }
    char out[1200],p[1200]; data_outdir(argv[3],out,sizeof out);
    snprintf(p,sizeof p,"%s\\train.bin",out); if(d_exists(p)){ fprintf(stderr,"[mix] %s already holds a corpus - pick a new name\n",out); return 1; }
    double total=atof(argv[4]), rep=atof(argv[5]), vtok=atof(argv[6]); if(rep<=0) rep=1;
    int ns=(argc-7)/2; if(ns>16){ fprintf(stderr,"[mix] at most 16 corpora\n"); return 1; }
    char dir[16][1200]; double w[16],ws=0; DMap tr[16],va[16]; DSpan*dd[16]; size_t nd[16]; DTok*t0=NULL;
    for(int i=0;i<ns;i++){ data_resolve(argv[7+2*i],dir[i],sizeof dir[i]); w[i]=atof(argv[8+2*i]); ws+=w[i];
        DTok*t=dtok_load(dir[i]); if(!t) return 1;
        if(!t0) t0=t; else if(t->blen!=t0->blen||memcmp(t->blob,t0->blob,t->blen)){ fprintf(stderr,"[mix] REFUSED: %s uses a different tokenizer than %s\n",dir[i],dir[0]); return 1; }
        snprintf(p,sizeof p,"%s\\train.bin",dir[i]); if(!dmap(&tr[i],p)||!tr[i].n){ fprintf(stderr,"[mix] no train.bin in %s\n",dir[i]); return 1; }
        snprintf(p,sizeof p,"%s\\val.bin",dir[i]); dmap(&va[i],p);
        dd[i]=d_docs(&tr[i],t0->eot,&nd[i]);
        printf("[mix] %-24s %10.1fM tok %9zu docs  weight %g\n",d_base(dir[i]),tr[i].n/1e6,nd[i],w[i]); fflush(stdout); }
    if(ws<=0){ fprintf(stderr,"[mix] weights must be > 0\n"); return 1; }
    if(total<=0){ total=1e300; for(int i=0;i<ns;i++) if(w[i]>0){ double c=tr[i].n*rep/(w[i]/ws); if(c<total) total=c; } }
    printf("[mix] target %.1fM train tokens, max repeat %.2f, val %.1fM\n",total/1e6,rep,vtok/1e6);
    size_t cap=1<<20,n=0; uint64_t*sel=(uint64_t*)malloc(8*cap); double got[16];
    for(int i=0;i<ns;i++){ double want=total*w[i]/ws, lim=tr[i].n*rep; got[i]=0;
        uint32_t*perm=(uint32_t*)malloc(4*nd[i]); for(size_t k=0;k<nd[i];k++) perm[k]=(uint32_t)k;
        for(size_t k=0;got[i]<want && got[i]<lim;k++){ size_t r=k%nd[i];
            if(r==0) for(size_t a=nd[i]-1;a>0;a--){ size_t b=d_rnd()%(a+1); uint32_t x=perm[a]; perm[a]=perm[b]; perm[b]=x; }
            if(n==cap){ cap*=2; sel=(uint64_t*)realloc(sel,8*cap); } sel[n++]=((uint64_t)i<<40)|perm[r]; got[i]+=dd[i][perm[r]].n; }
        free(perm);
        if(got[i]<want*0.98) printf("[mix] note: %s gives only %.1fM of the %.1fM wanted (max repeat %.2f)\n",dir[i],got[i]/1e6,want/1e6,rep); }
    for(size_t a=n?n-1:0;a>0;a--){ size_t b=d_rnd()%(a+1); uint64_t x=sel[a]; sel[a]=sel[b]; sel[b]=x; }
    if(!CreateDirectoryA(out,NULL)&&!d_isdir(out)){ fprintf(stderr,"[mix] cannot create %s\n",out); return 1; }
    const char*cp[]={"tokenizer.bin","tokenizer.json","template.txt","canon.bin"}; for(int i=0;i<4;i++) d_copy(dir[0],out,cp[i]); d_template(out);
    snprintf(p,sizeof p,"%s\\train.bin",out); FILE*ft=fopen(p,"wb"); snprintf(p,sizeof p,"%s\\val.bin",out); FILE*fv=fopen(p,"wb");
    long long ntr=0,nva=0; DWORD t0s=GetTickCount(),tl=t0s;
    for(size_t k=0;k<n;k++){ int i=(int)(sel[k]>>40); DSpan*s=&dd[i][sel[k]&0xFFFFFFFFFFull]; fwrite(tr[i].p+s->s,2,s->n,ft); ntr+=s->n;
        if(GetTickCount()-tl>10000){ tl=GetTickCount(); printf("[mix] %.1fM / %.1fM tokens written\n",ntr/1e6,total/1e6); fflush(stdout); } }
    double vgot[16];
    for(int i=0;i<ns;i++){ vgot[i]=0; if(!va[i].n) continue; double want=vtok*w[i]/ws; size_t s=0;
        for(size_t j=0;j<va[i].n && vgot[i]<want;j++) if(va[i].p[j]==t0->eot){ fwrite(va[i].p+s,2,j+1-s,fv); vgot[i]+=j+1-s; s=j+1; }
        nva+=(long long)vgot[i]; }
    fclose(ft); fclose(fv);
    snprintf(p,sizeof p,"%s\\meta.json",out); FILE*f=fopen(p,"wb"); const char*nm=d_base(out);
    fprintf(f,"{\n  \"profile\": "); jesc(f,nm);
    fprintf(f,",\n  \"vocab_size\": %d,\n  \"eot_id\": %d,\n  \"total_tokens\": %lld,\n  \"train_tokens\": %lld,\n  \"val_tokens\": %lld,\n  \"max_repeat\": %g,\n"
              "  \"built_by\": \"cudalm data mix\",\n  \"categories\": {\n",t0->V,t0->eot,ntr+nva,ntr,nva,rep);
    for(int i=0;i<ns;i++){ fprintf(f,"    "); jesc(f,d_base(dir[i]));
        fprintf(f,": {\"dir\": "); jesc(f,dir[i]); fprintf(f,", \"weight\": %g, \"unique\": %zu, \"emitted\": %.0f, \"tokens\": %.0f, \"repeat\": %.2f, \"val\": %.0f}%s\n",
                w[i]/ws,tr[i].n,got[i],got[i],got[i]/tr[i].n,vgot[i],i+1<ns?",":""); }
    fprintf(f,"  }\n}\n"); fclose(f);
    printf("[mix] done in %.1f s: %lld train + %lld val tokens -> %s\n",(GetTickCount()-t0s)/1000.0,ntr,nva,out);
    for(int i=0;i<ns;i++) printf("  %-24s %5.1f%%  repeat %.2f\n",d_base(dir[i]),100.0*got[i]/(ntr?ntr:1),got[i]/tr[i].n);
    for(int i=0;i<ns;i++){ dunmap(&tr[i]); dunmap(&va[i]); }
    return 0; }

/* ---------------- info ---------------- */
static void d_show(const DTok*t,const uint16_t*s,size_t n,size_t maxb){
    size_t b=0; printf("    \x1b[36m");
    for(size_t i=0;i<n&&b<maxb;i++){ int id=s[i]; if(id==t->eot){ printf("\x1b[33m<eot>\x1b[36m"); continue; }
        if(id>=t->V) continue; fwrite(t->tk->dec[id],1,t->tk->declen[id],stdout); b+=t->tk->declen[id];
        if(memchr(t->tk->dec[id],'\n',t->tk->declen[id])) printf("    "); }
    printf("\x1b[0m%s\n",b>=maxb?" ...":""); }
static int u32cmp(const void*a,const void*b){ uint32_t x=*(const uint32_t*)a,y=*(const uint32_t*)b; return x<y?-1:x>y; }
static int cmd_data_info(int argc,char**argv){
    if(argc<4){ fprintf(stderr,"usage: cudalm data info <corpus|tokenizer dir>\n"); return 1; }
    char dir[1200],p[1200]; data_resolve(argv[3],dir,sizeof dir); DTok*t=dtok_load(dir); if(!t) return 1;
    snprintf(p,sizeof p,"%s\\template.txt",dir); size_t tl; char*tp=(char*)read_file(p,&tl);
    snprintf(p,sizeof p,"%s\\canon.bin",dir); long long cs=d_fsize(p);
    printf("\x1b[1m%s\x1b[0m\n  tokenizer  V=%d  eot=%d  merges=%d  template=%.*s  canon.bin %s\n",dir,t->V,t->eot,t->nm,tp?(int)(tl>16?16:tl):4,tp?tp:"none",
           cs==2LL*t->V?"yes":cs>0?"WRONG SIZE":"no");
    { int lo=1,li=0; for(int i=0;i<t->V;i++) if(t->tk->declen[i]>lo){ lo=t->tk->declen[i]; li=i; }
      printf("  longest token #%d (%d bytes): \"%.*s\"\n",li,lo,lo>60?60:lo,(const char*)t->tk->dec[li]); }
    snprintf(p,sizeof p,"%s\\meta.json",dir); size_t ml; char*mj=(char*)read_file(p,&ml);
    if(mj){ mj=(char*)realloc(mj,ml+1); mj[ml]=0; const char*v=jget(mj,"categories");
        if(v&&jfirst(v)){ printf("  categories (meta.json):\n");
            for(const char*k=jfirst(v);k&&*k=='"';k=jobj_next(k)){ const char*o=jobj_val(k); if(!o) break; DBuf nm={0}; jdec(k,&nm);
                const char*x=jget(o,"tokens"); if(!x) x=jget(o,"emitted"); const char*d=jget(o,"docs"); const char*pc=jget(o,"pct");
                printf("    %-28s %10.1fM tok",nm.p,x?_atoi64(x)/1e6:0.0); if(d) printf(" %9lld docs",_atoi64(d)); if(pc) printf("  %5.1f%%",atof(pc)); printf("\n"); free(nm.p); } }
        v=jget(mj,"history"); if(v) for(const char*e=jfirst(v);e;e=jnext(e)){ DBuf h={0}; jdec(e,&h); printf("  history: %s\n",h.p?h.p:""); free(h.p); }
        free(mj); }
    for(int f=0;f<2;f++){ DMap m; snprintf(p,sizeof p,"%s\\%s",dir,f?"val.bin":"train.bin"); if(!dmap(&m,p)){ if(!f) printf("  (no train.bin - a tokenizer-only folder)\n"); continue; }
        size_t nd=0; DSpan*d=d_docs(&m,t->eot,&nd); size_t bad=0; uint16_t mx=0; for(size_t i=0;i<m.n;i++){ if(m.p[i]>=t->V) bad++; if(m.p[i]>mx) mx=m.p[i]; }
        uint32_t*l=(uint32_t*)malloc(4*(nd?nd:1)); for(size_t i=0;i<nd;i++) l[i]=d[i].n; qsort(l,nd,4,u32cmp);
        printf("  %s  %10.2fM tok  %9zu docs  doc len mean %.0f p50 %u p90 %u max %u  | max id %u%s\n",f?"val.bin  ":"train.bin",m.n/1e6,nd,
               nd?(double)m.n/nd:0.0,nd?l[nd/2]:0,nd?l[nd*9/10]:0,nd?l[nd-1]:0,mx,bad?"  \x1b[31mIDS >= V!\x1b[0m":"  (all < V)");
        if(bad) printf("    \x1b[31m%zu tokens are >= V=%d: this data was built with a different tokenizer\x1b[0m\n",bad,t->V);
        if(m.n&&m.p[m.n-1]!=t->eot) printf("    note: does not end with eot (cut mid-document)\n");
        if(nd){ size_t k=(size_t)(GetTickCount()%nd); printf("  sample %s doc #%zu (%u tokens):\n",f?"val":"train",k,d[k].n); d_show(t,m.p+d[k].s,d[k].n,600); }
        free(l); free(d); dunmap(&m); }
    return 0; }

/* ---------------- fetch: prep\fetch.py (Hugging Face profiles, parquet/json conversion) ---------------- */
static int cmd_data_fetch(int argc,char**argv){
    if(argc<5){ fprintf(stderr,"usage: cudalm data fetch <profile|url> <out> [target_tokens]   download\n"
        "       cudalm data fetch <file|folder> <out> [category]       convert parquet/json/jsonl\n  profiles: py -3 prep\\fetch.py list\n"); return 1; }
    char out[1200],src[1200],cl[4096]; data_outdir(argv[4],out,sizeof out); data_resolve(argv[3],src,sizeof src);
    const char*py=getenv("CUDALM_PYTHON"); if(!py||!*py) py="py -3";
    int conv=_strnicmp(argv[3],"http",4) && (!d_bare(argv[3])||strchr(argv[3],'.')) && d_exists(src);   /* bare word = profile */
    snprintf(cl,sizeof cl,"%s -u \"%s\\prep\\fetch.py\" %s \"%s\" \"%s\" %s",py,exe_dir(),conv?"convert":"get",conv?src:argv[3],out,argc>5?argv[5]:"");
    printf("> %s\n",cl); fflush(stdout);
    STARTUPINFOA si; PROCESS_INFORMATION pi; memset(&si,0,sizeof si); si.cb=sizeof si; DWORD code=1;
    if(!CreateProcessA(NULL,cl,NULL,NULL,FALSE,0,NULL,NULL,&si,&pi)){ fprintf(stderr,"[fetch] cannot start python (%lu) - set CUDALM_PYTHON\n",GetLastError()); return 1; }
    WaitForSingleObject(pi.hProcess,INFINITE); GetExitCodeProcess(pi.hProcess,&code); CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
    return (int)code; }

static int cmd_data_canon(int argc,char**argv){
    if(argc<4){ fprintf(stderr,"usage: cudalm data canon <tokenizer|corpus dir> [out.bin]\n"); return 1; }
    char dir[1200],p[1200]; data_resolve(argv[3],dir,sizeof dir); DTok*t=dtok_load(dir); if(!t) return 1;
    if(argc>4) snprintf(p,sizeof p,"%s",argv[4]); else snprintf(p,sizeof p,"%s\\canon.bin",dir);
    return d_canon(t,p); }

static int cmd_data(int argc,char**argv){
    const char*s=argc>2?argv[2]:"";
    if(!strcmp(s,"encode"))    return cmd_data_encode(argc,argv);
    if(!strcmp(s,"tokenizer")) return cmd_data_tokenizer(argc,argv);
    if(!strcmp(s,"mix"))       return cmd_data_mix(argc,argv);
    if(!strcmp(s,"info"))      return cmd_data_info(argc,argv);
    if(!strcmp(s,"canon"))     return cmd_data_canon(argc,argv);
    if(!strcmp(s,"fetch"))     return cmd_data_fetch(argc,argv);
    char roots[2048]; data_roots(roots,sizeof roots);
    fprintf(stderr,
        "cudalm data - tokenizer and corpus tools (bare names resolve under the data roots)\n"
        "  cudalm data fetch     <profile|url|file|folder> <out> [tokens|category]  download / convert to .jsonl\n"
        "  cudalm data tokenizer <out> <inputs> [vocab=16384] [sample_mb=300]  train a byte-level BPE (+ canon.bin)\n"
        "  cudalm data encode    <new|append> <out> <inputs> <tokenizer> [val%%=1] [dedup=1] [follow=0]\n"
        "  cudalm data mix       <out> <train_tokens|0> <max_repeat> <val_tokens> <corpus> <weight> ...\n"
        "  cudalm data info      <corpus|tokenizer>\n"
        "  cudalm data canon     <tokenizer> [out.bin]\n"
        "  inputs: files or folders (.jsonl .json .txt), ';'-separated\n"
        "  data roots: %s  (CUDALM_DATA_ROOTS)\n",roots);
    return 1; }
#else
static const char* data_resolve(const char*n,char*o,size_t cap){ snprintf(o,cap,"%s",n); return o; }
#endif
