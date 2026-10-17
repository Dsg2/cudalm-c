/* teachtok: teachgen jsonl -> cudalm corpus dir (uint16 train.bin/val.bin + tokenizer + meta.json).
 * Token-exact port of the v3 HF tokenizer: GPT-2 ByteLevel regex split, then BPE ranks from tokenizer.bin (TLTK).
 * Streams: follow=1 keeps tailing a growing jsonl until <in>.done exists. Dedups by text; val = docs with i%100==0.
 * usage: teachtok <tok_dir> <in.jsonl> <out_dir> [follow=0] */
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "uniclass.h"

static int V, EOT, NM, base[256], nsp, spid[64]; static char sps[64][33]; static int spl[64];   /* specials: <|name|> vocab entries (v4) */
static uint64_t *mkey; static int *mnew, *mrank; static size_t mcap;
static size_t slot(uint64_t k){ size_t i=(size_t)((k*0x9E3779B97F4A7C15ULL)>>20)&(mcap-1);
    while(mrank[i]>=0 && mkey[i]!=k) i=(i+1)&(mcap-1); return i; }
static void load_tok(const char *p){
    FILE *f=fopen(p,"rb"); if(!f){ fprintf(stderr,"cannot open %s\n",p); exit(1); }
    char m[4]; fread(m,1,4,f); if(memcmp(m,"TLTK",4)){ fprintf(stderr,"bad tok\n"); exit(1); }
    fread(&V,4,1,f); fread(&EOT,4,1,f); fread(&NM,4,1,f);
    for(int i=0;i<V;i++){ int l; char d[33]; fread(&l,4,1,f);
        if(l>=5&&l<=32){ fread(d,1,l,f); int ok=i!=EOT&&d[0]=='<'&&d[1]=='|'&&d[l-2]=='|'&&d[l-1]=='>';
            for(int k=2;ok&&k<l-2;k++){ int c=d[k]; ok=(c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='_'||c=='/'; }
            if(ok&&nsp<64){ memcpy(sps[nsp],d,l); spl[nsp]=l; spid[nsp++]=i; } }
        else fseek(f,l,SEEK_CUR); }
    fread(base,4,256,f);
    mcap=16; while(mcap<(size_t)NM*2) mcap<<=1;
    mkey=malloc(8*mcap); mnew=malloc(4*mcap); mrank=malloc(4*mcap); memset(mrank,-1,4*mcap);
    for(int m=0;m<NM;m++){ int a,b,nw; fread(&a,4,1,f); fread(&b,4,1,f); fread(&nw,4,1,f);
        uint64_t k=(uint64_t)a*V+b; size_t i=slot(k); if(mrank[i]<0){ mkey[i]=k; mrank[i]=m; mnew[i]=nw; } }
    fclose(f);
}

static int inr(const unsigned (*r)[2], int n, unsigned c){
    int lo=0, hi=n-1; while(lo<=hi){ int m=(lo+hi)/2; if(c<r[m][0]) hi=m-1; else if(c>r[m][1]) lo=m+1; else return 1; } return 0; }
static int isL(unsigned c){ return c<128 ? ((c|32)>='a'&&(c|32)<='z') : inr(UNI_L,sizeof UNI_L/8,c); }
static int isN(unsigned c){ return c<128 ? (c>='0'&&c<='9') : inr(UNI_N,sizeof UNI_N/8,c); }
static int isW(unsigned c){ return (c>=9&&c<=13)||c==32||c==0x85||c==0xA0||c==0x1680||(c>=0x2000&&c<=0x200A)||
                            c==0x2028||c==0x2029||c==0x202F||c==0x205F||c==0x3000; }
static int cls(unsigned c){ return isW(c)?0 : isL(c)?1 : isN(c)?2 : 3; }

/* decode UTF-8 into code points + byte offsets */
static int ucs(const unsigned char *s, int n, unsigned *cp, int *off){
    int k=0, i=0;
    while(i<n){ unsigned c=s[i]; int l=c<0x80?1:c<0xE0?2:c<0xF0?3:4; if(i+l>n) l=n-i;
        unsigned v=l==1?c:l==2?c&31:l==3?c&15:c&7; for(int j=1;j<l;j++) v=v<<6|(s[i+j]&63);
        off[k]=i; cp[k++]=v; i+=l; }
    off[k]=n; return k;
}

static int *ids; static size_t nids, capids;
static void put(int t){ if(nids==capids){ capids=capids?capids*2:1<<16; ids=realloc(ids,4*capids); } ids[nids++]=t; }
static int wb[1<<16];
static void bpe(const unsigned char *s, int n){
    int *w=n<=(int)(sizeof wb/4)?wb:malloc(4*n), c=n;
    for(int i=0;i<n;i++) w[i]=base[s[i]];
    while(c>1){ int best=-1, bi=-1;
        for(int i=0;i+1<c;i++){ int r=mrank[slot((uint64_t)w[i]*V+w[i+1])]; if(r>=0&&(best<0||r<best)){ best=r; bi=i; } }
        if(bi<0) break;
        w[bi]=mnew[slot((uint64_t)w[bi]*V+w[bi+1])]; memmove(w+bi+1,w+bi+2,4*(c-bi-2)); c--; }
    for(int i=0;i<c;i++) put(w[i]);
    if(w!=wb) free(w);
}
static unsigned *cpb; static int *offb, capb;
static void encode_seg(const unsigned char *s, int n){
    if(n+1>capb){ capb=n+1; cpb=realloc(cpb,4*capb); offb=realloc(offb,4*capb); }
    unsigned *cp=cpb; int *off=offb, k=ucs(s,n,cp,off), i=0;
    while(i<k){ int j=i+1; unsigned c=cp[i];
        if(c=='\'' && i+1<k){ unsigned a=cp[i+1], b=i+2<k?cp[i+2]:0;
            if(a=='s'||a=='t'||a=='m'||a=='d') j=i+2;
            else if((a=='r'&&b=='e')||(a=='v'&&b=='e')||(a=='l'&&b=='l')) j=i+3; }
        if(j==i+1){ int st=(c==' '&&i+1<k&&!isW(cp[i+1]))?i+1:i, t=cls(cp[st]);
            if(t){ j=st+1; while(j<k&&cls(cp[j])==t) j++; }
            else { j=i+1; while(j<k&&isW(cp[j])) j++; if(j<k&&j-i>=2) j--; } }
        bpe(s+off[i],off[j]-off[i]); i=j; }
}
static void encode(const char *s){   /* <|endoftext|> and specials matched literally (= cudalm tok_encode) */
    const char *end=s+strlen(s);
    while(s<end){ const char *q=s; int id=-1, l=0;
        for(;(q=strchr(q,'<'));q++){
            if(!strncmp(q,"<|endoftext|>",13)){ id=EOT; l=13; break; }
            for(int k=0;k<nsp;k++) if(!strncmp(q,sps[k],spl[k])){ id=spid[k]; l=spl[k]; break; }
            if(id>=0) break; }
        if(!q) q=end;
        encode_seg((const unsigned char*)s,(int)(q-s)); if(id>=0){ put(id); q+=l; } s=q; }
}

/* JSON string field -> UTF-8 (in place into dst) */
static char *jstr(const char *j, const char *key, char *dst){
    char k[32]; snprintf(k,sizeof k,"\"%s\":\"",key); const char *p=strstr(j,k); if(!p) return NULL;
    char *o=dst; p+=strlen(k);
    for(; *p && *p!='"'; p++){
        if(*p!='\\'){ *o++=*p; continue; }
        char c=*++p;
        if(c=='u'){ unsigned v=0; sscanf(p+1,"%4x",&v); p+=4;
            if(v>=0xD800&&v<0xDC00&&p[1]=='\\'&&p[2]=='u'){ unsigned w=0; sscanf(p+3,"%4x",&w); p+=6; v=0x10000+((v-0xD800)<<10)+(w-0xDC00); }
            if(v<0x80) *o++=(char)v;
            else if(v<0x800){ *o++=(char)(0xC0|v>>6); *o++=(char)(0x80|(v&63)); }
            else if(v<0x10000){ *o++=(char)(0xE0|v>>12); *o++=(char)(0x80|((v>>6)&63)); *o++=(char)(0x80|(v&63)); }
            else { *o++=(char)(0xF0|v>>18); *o++=(char)(0x80|((v>>12)&63)); *o++=(char)(0x80|((v>>6)&63)); *o++=(char)(0x80|(v&63)); }
            continue; }
        *o++ = c=='n'?'\n':c=='t'?'\t':c=='r'?'\r':c=='b'?'\b':c=='f'?'\f':c;
    }
    *o=0; return dst;
}

static uint64_t *seen; static size_t scap=1<<20, sn;
static int dedup(const char *s){   /* 1 if new */
    uint64_t h=1469598103934665603ULL; for(; *s; s++) h=(h^(unsigned char)*s)*1099511628211ULL; if(!h) h=1;
    if(sn*2>=scap){ uint64_t *o=seen; size_t oc=scap; scap*=2; seen=calloc(scap,8);
        for(size_t i=0;i<oc;i++) if(o[i]){ size_t j=o[i]&(scap-1); while(seen[j]) j=(j+1)&(scap-1); seen[j]=o[i]; } free(o); }
    size_t j=h&(scap-1); while(seen[j]){ if(seen[j]==h) return 0; j=(j+1)&(scap-1); }
    seen[j]=h; sn++; return 1;
}

typedef struct { char k[48]; long long docs, tok; } Cat;
static Cat cats[64]; static int ncat;
static long long ndoc, ndup, nbad, ntr, nva;
static void write_meta(const char *out){
    char p[512]; snprintf(p,sizeof p,"%s\\meta.json",out); FILE *f=fopen(p,"wb"); if(!f) return;
    const char *nm=strrchr(out,'\\'); nm=nm?nm+1:out;
    fprintf(f,"{\n  \"profile\": \"%s\",\n  \"vocab_size\": %d,\n  \"eot_id\": %d,\n  \"total_tokens\": %lld,\n"
              "  \"train_tokens\": %lld,\n  \"val_tokens\": %lld,\n  \"docs\": %lld,\n  \"dups\": %lld,\n"
              "  \"sources\": {\"teachgen\": {\"tokenizer\": \"teachtok (HF v3tok exact)\"}},\n  \"categories\": {\n",
              nm,V,EOT,ntr+nva,ntr,nva,ndoc,ndup);
    for(int i=0;i<ncat;i++) fprintf(f,"    \"%s\": {\"docs\": %lld, \"tokens\": %lld}%s\n",cats[i].k,cats[i].docs,cats[i].tok,i+1<ncat?",":"");
    fprintf(f,"  }\n}\n"); fclose(f);
}

int main(int argc, char **argv){
    setvbuf(stderr,NULL,_IONBF,0);
    if(argc<4){ fprintf(stderr,"usage: teachtok <tok_dir> <in.jsonl> <out_dir> [follow=0]\n"); return 1; }
    const char *td=argv[1], *in=argv[2], *out=argv[3]; int follow=argc>4?atoi(argv[4]):0;
    char p[512], q[512], done[512]; snprintf(done,sizeof done,"%s.done",in);
    snprintf(p,sizeof p,"%s\\tokenizer.bin",td); load_tok(p);
    CreateDirectoryA(out,NULL);
    const char *cp[]={"tokenizer.json","tokenizer.bin","template.txt"};
    for(int i=0;i<3;i++){ snprintf(p,sizeof p,"%s\\%s",td,cp[i]); snprintf(q,sizeof q,"%s\\%s",out,cp[i]); CopyFileA(p,q,FALSE); }
    snprintf(p,sizeof p,"%s\\train.bin",out); FILE *ft=fopen(p,"wb");
    snprintf(p,sizeof p,"%s\\val.bin",out); FILE *fv=fopen(p,"wb");
    FILE *fi=NULL; while(!(fi=fopen(in,"rb"))){ if(!follow){ fprintf(stderr,"cannot open %s\n",in); return 1; } Sleep(5000); }
    seen=calloc(scap,8);
    size_t cap=1<<20, n=0; char *line=malloc(cap), *txt=malloc(cap); uint16_t *u16=NULL; size_t cu=0;
    long long pos=0; DWORD t0=GetTickCount(), tl=t0; int first=1, last=0;
    for(;;){
        int c; n=0;
        while((c=fgetc(fi))!=EOF){ if(n+1>=cap){ cap*=2; line=realloc(line,cap); txt=realloc(txt,cap); } line[n++]=(char)c; if(c=='\n') break; }
        if(n==0 || line[n-1]!='\n'){                  /* EOF or partial line: wait for more */
            int fin=!follow || GetFileAttributesA(done)!=INVALID_FILE_ATTRIBUTES;
            if(fin && last){ if(n) nbad++; break; }
            last=fin; fflush(ft); fflush(fv); write_meta(out);
            fseeko64(fi,pos,SEEK_SET); clearerr(fi); if(!fin) Sleep(5000); continue;
        }
        pos+=n; line[n]=0;
        char *s=line; if(first){ first=0; if(!memcmp(s,"\xEF\xBB\xBF",3)) s+=3; }
        char *ip=strstr(s,"\"i\":"), kind[16]="?", cat[24]="?", tmp[64];
        if(!ip || !jstr(s,"text",txt)){ nbad++; continue; }
        long di=atol(ip+4);
        if(jstr(s,"kind",tmp)) snprintf(kind,sizeof kind,"%s",tmp);
        if(jstr(s,"cat",tmp)) snprintf(cat,sizeof cat,"%s",tmp);
        if(!dedup(txt)){ ndup++; continue; }
        nids=0; encode(txt); put(EOT);
        if(nids>cu){ cu=nids; u16=realloc(u16,2*cu); }
        for(size_t i=0;i<nids;i++) u16[i]=(uint16_t)ids[i];
        int isv=di%100==0; fwrite(u16,2,nids,isv?fv:ft); if(isv) nva+=nids; else ntr+=nids;
        snprintf(tmp,sizeof tmp,"%s/%s",kind,cat); int ci=0; while(ci<ncat && strcmp(cats[ci].k,tmp)) ci++;
        if(ci==ncat && ncat<64) snprintf(cats[ncat++].k,48,"%s",tmp);
        if(ci<64){ cats[ci].docs++; cats[ci].tok+=nids; }
        ndoc++;
        DWORD t=GetTickCount();
        if(t-tl>30000){ tl=t; fprintf(stderr,"%lus docs %lld dup %lld | train %lld val %lld tok | in %.1f MB\n",
                                     (t-t0)/1000,ndoc,ndup,ntr,nva,pos/1e6); }
    }
    fclose(ft); fclose(fv); fclose(fi); write_meta(out);
    fprintf(stderr,"[teachtok] %lld docs (%lld dup, %lld bad), %lld train + %lld val tokens, %.1f s -> %s\n",
            ndoc,ndup,nbad,ntr,nva,(GetTickCount()-t0)/1000.0,out);
    for(int i=0;i<ncat;i++) fprintf(stderr,"  %-18s %8lld docs %11lld tok %6.0f/doc\n",cats[i].k,cats[i].docs,cats[i].tok,(double)cats[i].tok/cats[i].docs);
    return 0;
}
