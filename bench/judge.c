/* judge: teacher-scored coherence of chatbench replies, via a running llama-server.
 *   judge <chatbench_out.txt> <out.tsv> [mode=student|shuffle|teacher] [port=8099]
 * Per reply: (a) teacher NLL of the reply given the prompt, in nats/byte (reply forced with a
 * grammar; the server reports raw-logit log-probs), (b) expected 0-9 score from the teacher's
 * digit probabilities. shuffle = replies paired with the wrong prompt (sanity floor);
 * teacher = the teacher's own sampled replies to the same prompts (ceiling).
 * Build: gcc -O2 -static -o judge.exe judge.c -lws2_32 */
#include <winsock2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char *p; size_t n, cap; } Buf;
static void bput(Buf *b, const char *s, size_t n){
    if(b->n+n+1>b->cap){ b->cap=(b->n+n+1)*2; b->p=realloc(b->p,b->cap); }
    memcpy(b->p+b->n,s,n); b->n+=n; b->p[b->n]=0;
}
static void bstr(Buf *b, const char *s){ bput(b,s,strlen(s)); }
static void besc(Buf *b, const char *s, int gbnf){   /* JSON (gbnf=0) or GBNF-literal escaping */
    for(const unsigned char *c=(const unsigned char*)s; *c; c++){ char t[8];
        if(*c=='"'||*c=='\\'){ t[0]='\\'; t[1]=*c; bput(b,t,2); }
        else if(*c=='\n') bput(b,"\\n",2); else if(*c=='\r') bput(b,"\\r",2); else if(*c=='\t') bput(b,"\\t",2);
        else if(*c<0x20){ int k=gbnf?sprintf(t,"\\x%02X",*c):sprintf(t,"\\u%04x",*c); bput(b,t,k); }
        else bput(b,(const char*)c,1);
    }
}

static int g_port=8099;
static char *post(const char *path, const char *body){
    SOCKET s=socket(AF_INET,SOCK_STREAM,0); struct sockaddr_in a={0};
    a.sin_family=AF_INET; a.sin_port=htons((u_short)g_port); a.sin_addr.s_addr=inet_addr("127.0.0.1");
    if(connect(s,(struct sockaddr*)&a,sizeof a)){ fprintf(stderr,"cannot connect to llama-server on port %d\n",g_port); exit(1); }
    char h[256]; size_t n=strlen(body);
    int hl=snprintf(h,sizeof h,"POST %s HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                    "Content-Length: %zu\r\nConnection: close\r\n\r\n",path,n);
    send(s,h,hl,0);
    for(size_t o=0;o<n;){ int k=send(s,body+o,(int)(n-o),0); if(k<=0) break; o+=k; }
    Buf r={0}; char tmp[65536]; int k;
    while((k=recv(s,tmp,sizeof tmp,0))>0) bput(&r,tmp,k);
    closesocket(s);
    char *b=r.p?strstr(r.p,"\r\n\r\n"):NULL;
    if(!b){ fprintf(stderr,"bad HTTP response\n"); exit(1); }
    if(strncmp(r.p+9,"200",3)){ fprintf(stderr,"%s: HTTP %.3s: %.400s\n",path,r.p+9,b+4); free(r.p); return NULL; }
    char *o=strdup(b+4); free(r.p); return o;
}
static void utf8(Buf *b, unsigned v){
    char t[3]; if(v<0x80){ t[0]=(char)v; bput(b,t,1); }
    else if(v<0x800){ t[0]=(char)(0xC0|v>>6); t[1]=(char)(0x80|(v&63)); bput(b,t,2); }
    else { t[0]=(char)(0xE0|v>>12); t[1]=(char)(0x80|((v>>6)&63)); t[2]=(char)(0x80|(v&63)); bput(b,t,3); }
}
static char *jstr(const char *j, const char *key){   /* first "key":"..." value, unescaped */
    char k[64]; snprintf(k,sizeof k,"\"%s\":\"",key);
    const char *p=j?strstr(j,k):NULL; if(!p) return NULL; p+=strlen(k);
    Buf b={0}; bput(&b,"",0);
    for(; *p && *p!='"'; p++){
        if(*p!='\\'){ bput(&b,p,1); continue; }
        char c=*++p;
        if(c=='u'){ char h[5]={p[1],p[2],p[3],p[4],0}; utf8(&b,(unsigned)strtoul(h,0,16)); p+=4; continue; }
        c = c=='n'?'\n' : c=='t'?'\t' : c=='r'?'\r' : c; bput(&b,&c,1);
    }
    return b.p;
}
static const char *skip_group(const char *p){       /* p at '[' or '{' -> just past its match */
    int d=0, str=0;
    for(; *p; p++){
        if(str){ if(*p=='\\') p++; else if(*p=='"') str=0; continue; }
        if(*p=='"') str=1; else if(*p=='['||*p=='{') d++;
        else if((*p==']'||*p=='}') && --d==0) return p+1;
    }
    return p;
}

static char *g_pre, *g_post;                          /* chat template around the user message */
static char *wrap(const char *msg){ Buf b={0}; bstr(&b,g_pre); bstr(&b,msg); bstr(&b,g_post); return b.p; }

/* teacher log-prob of `reply` after `ctx`; tokens counted until they cover the reply's bytes */
static double force_nll(const char *ctx, const char *reply, int *ntok, int *ok){
    Buf g={0}, b={0}; int L=(int)strlen(reply);
    bstr(&g,"root ::= \""); besc(&g,reply,1); bstr(&g,"\"");
    bstr(&b,"{\"prompt\":\""); besc(&b,ctx,0); bstr(&b,"\",\"grammar\":\""); besc(&b,g.p,0);
    char t[160]; snprintf(t,sizeof t,"\",\"n_predict\":%d,\"temperature\":0,\"n_probs\":1,\"cache_prompt\":false}",L+8);
    bstr(&b,t);
    char *r=post("/completion",b.p); free(g.p); free(b.p);
    double s=0; int nb=0, nt=0;
    const char *p=r?strstr(r,"\"completion_probabilities\""):NULL;
    while(p && nb<L){
        const char *q=strstr(p,"\"bytes\":["); if(!q) break; q+=9;
        int cnt=0; if(*q!=']'){ cnt=1; for(const char *z=q; *z && *z!=']'; z++) cnt+=*z==','; }
        const char *lp=strstr(q,"\"logprob\":"); if(!lp) break;
        s+=atof(lp+10); nb+=cnt; nt++;
        p=strstr(lp,"\"top_logprobs\":["); if(p) p=skip_group(p+15);
    }
    *ntok=nt; *ok = r && nb==L; free(r);
    return -s;
}
/* expected 0-9 rating from the digit probabilities of the teacher's first answer token */
static double judge_score(const char *q, const char *reply){
    Buf m={0}, b={0};
    bstr(&m,"Rate the assistant reply below for coherence and relevance to the user's message, from 0 "
            "(nonsense or off-topic) to 9 (fully coherent, relevant and correct). Answer with a single digit.\n\n"
            "User message:\n"); bstr(&m,q); bstr(&m,"\n\nAssistant reply:\n"); bstr(&m,reply);
    char *ctx=wrap(m.p);
    bstr(&b,"{\"prompt\":\""); besc(&b,ctx,0);
    bstr(&b,"\",\"n_predict\":1,\"temperature\":0,\"n_probs\":20,\"cache_prompt\":false}");
    char *r=post("/completion",b.p); free(m.p); free(b.p); free(ctx);
    double pw=0, ev=0;
    const char *p=r?strstr(r,"\"top_logprobs\":["):NULL;
    if(p){ const char *e=skip_group(p+15);
        for(const char *x=strstr(p,"\"token\":\""); x && x<e; x=strstr(x+1,"\"token\":\"")){
            const char *v=x+9; while(*v==' ') v++;
            const char *lp=strstr(x,"\"logprob\":");
            if(v[0]>='0' && v[0]<='9' && v[1]=='"' && lp){ double pr=exp(atof(lp+10)); pw+=pr; ev+=pr*(v[0]-'0'); }
        } }
    free(r);
    return pw>0 ? ev/pw : -1;
}
static char *teacher_reply(const char *q){
    Buf b={0}; char *ctx=wrap(q);
    bstr(&b,"{\"prompt\":\""); besc(&b,ctx,0);
    bstr(&b,"\",\"n_predict\":128,\"temperature\":0.7,\"top_p\":0.9,\"cache_prompt\":false}");
    char *r=post("/completion",b.p); free(b.p); free(ctx);
    char *c=jstr(r,"content"); free(r); return c?c:strdup("");
}

typedef struct { char cat[32]; char *q, *r; int eot; } Item;
static void fix_utf8(char *s){                       /* invalid bytes -> '?' (JSON must be UTF-8) */
    unsigned char *u=(unsigned char*)s;
    while(*u){
        int n = *u<0x80?0 : (*u>>5)==6?1 : (*u>>4)==14?2 : (*u>>3)==30?3 : -1, k=0;
        if(n>0) for(k=1;k<=n && (u[k]>>6)==2;k++);
        if(n<0 || (n>0 && k<=n)){ *u++='?'; continue; }
        u+=n+1;
    }
}
int main(int argc, char **argv){
    if(argc<3){ fprintf(stderr,"usage: judge <chatbench_out.txt> <out.tsv> [student|shuffle|teacher] [port]\n"); return 1; }
    const char *mode=argc>3?argv[3]:"student"; if(argc>4) g_port=atoi(argv[4]);
    WSADATA wd; WSAStartup(MAKEWORD(2,2),&wd);
    FILE *f=fopen(argv[1],"rb"); if(!f){ fprintf(stderr,"cannot open %s\n",argv[1]); return 1; }
    Item *it=NULL; int n=0, cap=0, gen=0; char line[1<<16], cat[32]="", *q=NULL;
    while(fgets(line,sizeof line,f)){
        line[strcspn(line,"\r\n")]=0;
        if(!gen){ gen=strstr(line,"=== generation ===")!=NULL; continue; }
        char *qq;
        if(line[0]=='[' && (qq=strstr(line,"] Q: "))){
            int cl=(int)(qq-line-1); if(cl>31) cl=31; memcpy(cat,line+1,cl); cat[cl]=0;
            free(q); q=strdup(qq+5); fix_utf8(q); continue;
        }
        char *bar;
        if(q && !strncmp(line,"  #",3) && (bar=strstr(line," | "))){
            if(n==cap){ cap=cap?cap*2:128; it=realloc(it,sizeof(Item)*cap); }
            strcpy(it[n].cat,cat); it[n].q=strdup(q); it[n].r=strdup(bar+3);
            it[n].eot = strstr(line," eot ")!=NULL && strstr(line," eot ")<bar;
            for(char *c=it[n].r; *c; c++) if(*c=='~') *c='\n';
            fix_utf8(it[n].r); n++;
        }
    }
    fclose(f);
    if(!n){ fprintf(stderr,"no generation items in %s\n",argv[1]); return 1; }

    char *t=post("/apply-template","{\"messages\":[{\"role\":\"user\",\"content\":\"@@P@@\"}],"
                                   "\"chat_template_kwargs\":{\"enable_thinking\":false}}");
    char *tp=jstr(t,"prompt"), *at=tp?strstr(tp,"@@P@@"):NULL;
    if(!at){ fprintf(stderr,"apply-template failed\n"); return 1; }
    *at=0; g_pre=tp+(strncmp(tp,"<s>",3)?0:3); g_post=at+5; free(t);

    FILE *o=fopen(argv[2],"w"); if(!o){ fprintf(stderr,"cannot write %s\n",argv[2]); return 1; }
    fprintf(o,"cat\tidx\teot\tbytes\tntok\tnll\tnats_per_byte\tjudge\tok\n");
    enum{ NC=16 }; char cn[NC][32]; int cc[NC]={0}, ncat=0, bad=0; double cb[NC]={0}, cj[NC]={0};
    double tn=0, tb=0;
    for(int i=0;i<n;i++){
        const char *qi = !strcmp(mode,"shuffle") ? it[(i+n/2)%n].q : it[i].q;
        char *r = !strcmp(mode,"teacher") ? teacher_reply(it[i].q) : strdup(it[i].r);
        if(!strcmp(mode,"teacher")) fix_utf8(r);
        int nb=(int)strlen(r), nt=0, ok=0;
        char *ctx=wrap(qi);
        double nll = nb ? force_nll(ctx,r,&nt,&ok) : 0, jd=judge_score(qi,r);
        free(ctx);
        if(!ok || jd<0) bad++;
        fprintf(o,"%s\t%d\t%d\t%d\t%d\t%.4f\t%.5f\t%.3f\t%d\n",it[i].cat,i,it[i].eot,nb,nt,nll,nb?nll/nb:0,jd,ok);
        int c=0; while(c<ncat && strcmp(cn[c],it[i].cat)) c++;
        if(c==ncat && ncat<NC) strcpy(cn[ncat++],it[i].cat);
        if(c<ncat && ok && jd>=0){ cc[c]++; cb[c]+=nll/nb; cj[c]+=jd; tn+=nll; tb+=nb; }
        if((i+1)%16==0) fprintf(stderr,"  %d/%d\n",i+1,n);
        free(r);
    }
    fclose(o);
    printf("[judge] %s | mode %s | %d replies (%d unscored)\n  %-10s %4s %10s %7s\n",argv[1],mode,n,bad,"category","n","nats/byte","judge");
    int an=0; double ab=0, aj=0;
    for(int c=0;c<ncat;c++){ if(!cc[c]) continue;
        printf("  %-10s %4d %10.4f %7.2f\n",cn[c],cc[c],cb[c]/cc[c],cj[c]/cc[c]); an+=cc[c]; ab+=cb[c]; aj+=cj[c]; }
    if(an) printf("  %-10s %4d %10.4f %7.2f   (pooled %.4f nats/byte)\n","ALL",an,ab/an,aj/an,tb>0?tn/tb:0);
    return 0;
}
