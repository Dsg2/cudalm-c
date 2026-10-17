/* teachgen: Cosmopedia-style teacher data from a running llama-server (MiniCPM5 or any chat model).
 *   teachgen <out.jsonl> <ndocs> [threads=16] [ports=8099[,8100,..]] [chat_frac=0.5] [seed=1]
 * Several ports = several server instances (llama-server's main loop is single-threaded and caps one
 * instance at ~800 tok/s on a 6-core desktop CPU); threads are spread round-robin over them.
 * Stage 1 (cached in <out>.topics): the teacher lists subtopics for each built-in field.
 * Stage 2: doc i (deterministic from seed+i) is either a text document (topic x audience x format x length)
 * or a chat pair (teacher writes a user message, then answers it under a hidden "be concise" system prompt).
 * Only finished (eos) generations are kept. Output lines: {"i":N,"kind":"text|chat","cat":..,"tok":N,"text":..};
 * chat text uses the corpus markers "<|user|>\n..\n<|assistant|>\n..". Re-running resumes (skips done i).
 * Build: gcc -O2 -static -o teachgen.exe teachgen.c -lws2_32 */
#include <winsock2.h>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#ifdef _MSC_VER
#define __thread __declspec(thread)
#endif

typedef struct { char *p; size_t n, cap; } Buf;
static void bput(Buf *b, const char *s, size_t n){
    if(b->n+n+1>b->cap){ b->cap=(b->n+n+1)*2; b->p=realloc(b->p,b->cap); }
    memcpy(b->p+b->n,s,n); b->n+=n; b->p[b->n]=0;
}
static void bstr(Buf *b, const char *s){ bput(b,s,strlen(s)); }
static void besc(Buf *b, const char *s){
    for(const unsigned char *c=(const unsigned char*)s; *c; c++){ char t[8];
        if(*c=='"'||*c=='\\'){ t[0]='\\'; t[1]=*c; bput(b,t,2); }
        else if(*c=='\n') bput(b,"\\n",2); else if(*c=='\r') bput(b,"\\r",2); else if(*c=='\t') bput(b,"\\t",2);
        else if(*c<0x20){ bput(b,t,sprintf(t,"\\u%04x",*c)); }
        else bput(b,(const char*)c,1);
    }
}
static int g_ports[16]={8099}, g_nport=1; static __thread int t_port=8099;   /* one server per port; threads round-robin */
static char *post(const char *path, const char *body){
    for(int tries=0; tries<120; tries++){
        SOCKET s=socket(AF_INET,SOCK_STREAM,0); struct sockaddr_in a={0};
        a.sin_family=AF_INET; a.sin_port=htons((u_short)t_port); a.sin_addr.s_addr=inet_addr("127.0.0.1");
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
    fprintf(stderr,"llama-server on port %d unreachable\n",t_port); exit(1);
}
static void utf8(Buf *b, unsigned v){
    char t[3]; if(v<0x80){ t[0]=(char)v; bput(b,t,1); }
    else if(v<0x800){ t[0]=(char)(0xC0|v>>6); t[1]=(char)(0x80|(v&63)); bput(b,t,2); }
    else { t[0]=(char)(0xE0|v>>12); t[1]=(char)(0x80|((v>>6)&63)); t[2]=(char)(0x80|(v&63)); bput(b,t,3); }
}
static char *jstr(const char *j, const char *key){
    char k[64]; snprintf(k,sizeof k,"\"%s\":\"",key);
    const char *p=j?strstr(j,k):NULL; if(!p) return NULL; p+=strlen(k);
    Buf b={0}; bput(&b,"",0);
    for(; *p && *p!='"'; p++){
        if(*p!='\\'){ bput(&b,p,1); continue; }
        char c=*++p;
        if(c=='u'){ char h[5]={p[1],p[2],p[3],p[4],0}; unsigned v=(unsigned)strtoul(h,0,16); p+=4;
            if(v>=0xD800 && v<0xDC00 && p[1]=='\\' && p[2]=='u'){   /* surrogate pair -> 4-byte UTF-8 */
                char l[5]={p[3],p[4],p[5],p[6],0}; unsigned w=(unsigned)strtoul(l,0,16); p+=6;
                v=0x10000+((v-0xD800)<<10)+(w-0xDC00);
                char t[4]={(char)(0xF0|v>>18),(char)(0x80|((v>>12)&63)),(char)(0x80|((v>>6)&63)),(char)(0x80|(v&63))};
                bput(&b,t,4); continue; }
            utf8(&b,v); continue; }
        c = c=='n'?'\n' : c=='t'?'\t' : c=='r'?'\r' : c; bput(&b,&c,1);
    }
    return b.p;
}
static long jint(const char *j, const char *key){
    char k[64]; snprintf(k,sizeof k,"\"%s\":",key); const char *p=j?strstr(j,k):NULL;
    return p?atol(p+strlen(k)):0;
}
static char *trim(char *s){
    while(*s==' '||*s=='\n'||*s=='\r'||*s=='\t') s++;
    char *e=s+strlen(s); while(e>s && (e[-1]==' '||e[-1]=='\n'||e[-1]=='\r'||e[-1]=='\t')) *--e=0;
    return s;
}

/* chat template pieces (thinking off), from /apply-template */
static char *t_pre, *t_post, *t_spre, *t_smid;
static void get_templates(void){
    char *t=post("/apply-template","{\"messages\":[{\"role\":\"user\",\"content\":\"@@P@@\"}],"
                                   "\"chat_template_kwargs\":{\"enable_thinking\":false}}");
    char *tp=jstr(t,"prompt"), *at=tp?strstr(tp,"@@P@@"):NULL;
    if(!at){ fprintf(stderr,"apply-template failed\n"); exit(1); }
    *at=0; t_pre=tp+(strncmp(tp,"<s>",3)?0:3); t_post=at+5; free(t);
    t=post("/apply-template","{\"messages\":[{\"role\":\"system\",\"content\":\"@@S@@\"},{\"role\":\"user\",\"content\":\"@@P@@\"}],"
                             "\"chat_template_kwargs\":{\"enable_thinking\":false}}");
    tp=jstr(t,"prompt"); char *as=tp?strstr(tp,"@@S@@"):NULL; at=tp?strstr(tp,"@@P@@"):NULL;
    if(!as||!at){ fprintf(stderr,"apply-template (system) failed\n"); exit(1); }
    *as=0; *at=0; t_spre=tp+(strncmp(tp,"<s>",3)?0:3); t_smid=as+5; free(t);
}
/* one completion; returns content (NULL on failure), *eos = finished naturally, *ntok = tokens generated */
static char *complete(const char *sys, const char *msg, int npred, double temp, unsigned long long seed, int *eos, long *ntok){
    Buf b={0}, c={0};
    if(sys){ bstr(&c,t_spre); bstr(&c,sys); bstr(&c,t_smid); } else bstr(&c,t_pre);
    bstr(&c,msg); bstr(&c,t_post);
    bstr(&b,"{\"prompt\":\""); besc(&b,c.p);
    char t[200]; snprintf(t,sizeof t,"\",\"n_predict\":%d,\"temperature\":%.2f,\"top_p\":0.95,\"seed\":%llu}",npred,temp,seed%2000000000ULL);
    bstr(&b,t);
    char *r=post("/completion",b.p); free(b.p); free(c.p);
    char *o=jstr(r,"content");
    *eos = r && (strstr(r,"\"stop_type\":\"eos\"") || strstr(r,"\"stopped_eos\":true"));
    *ntok = jint(r,"tokens_predicted");
    free(r); return o;
}
/* multi-turn chat via /v1/chat/completions; turns alternate user/assistant starting with user */
static char *chatc(const char *sys, char **tu, int n, int npred, double temp, unsigned long long seed, int *eos, long *ntok){
    Buf b={0}; bstr(&b,"{\"messages\":[");
    if(sys){ bstr(&b,"{\"role\":\"system\",\"content\":\""); besc(&b,sys); bstr(&b,"\"},"); }
    for(int k=0;k<n;k++){ bstr(&b,k&1?"{\"role\":\"assistant\",\"content\":\"":"{\"role\":\"user\",\"content\":\"");
        besc(&b,tu[k]); bstr(&b,k+1<n?"\"},":"\"}"); }
    char t[256]; snprintf(t,sizeof t,"],\"max_tokens\":%d,\"temperature\":%.2f,\"top_p\":0.95,\"seed\":%llu,"
        "\"chat_template_kwargs\":{\"enable_thinking\":false}}",npred,temp,seed%2000000000ULL);
    bstr(&b,t);
    char *r=post("/v1/chat/completions",b.p); free(b.p);
    char *o=jstr(r,"content");
    *eos = r && strstr(r,"\"finish_reason\":\"stop\""); *ntok = jint(r,"completion_tokens");
    free(r); return o;
}

static const char *FIELDS[]={
 "arithmetic and number sense","algebra","geometry","probability and statistics","calculus","personal finance",
 "physics","chemistry","biology","human anatomy","ecology","astronomy","geology and earth science","weather and climate",
 "oceans and marine life","botany and gardening","zoology and animal behaviour","microbiology","genetics","medicine and common illnesses",
 "nutrition","fitness and exercise","sleep and mental health","first aid and safety","psychology","sociology","economics",
 "world history","ancient civilizations","medieval history","modern history","history of science","geography","world cultures and traditions",
 "languages and linguistics","English grammar and writing","literature and famous books","poetry","philosophy","ethics","logic and reasoning puzzles",
 "law and civics","government and politics","religion and mythology","art history","painting and drawing","music theory","musical instruments",
 "film and television","photography","architecture","design and typography","fashion and clothing","cooking techniques","baking","world cuisines",
 "beverages and coffee","home cleaning and organization","home repair and DIY","woodworking","sewing and knitting","pets and pet care",
 "parenting","childhood education","study skills","career advice","job interviews","workplace communication","entrepreneurship and small business",
 "marketing","management and leadership","productivity and time management","relationships and friendship","etiquette and manners",
 "travel and tourism","transportation and cars","aviation","trains and public transit","cycling","outdoor survival and camping","hiking",
 "sports rules and strategy","football (soccer)","basketball","chess and board games","video games","card games and puzzles","hobbies and collecting",
 "computers and how they work","programming basics","Python programming","web development","databases","computer networks and the internet",
 "cybersecurity basics","artificial intelligence","smartphones and apps","electronics and circuits","robotics","energy and electricity",
 "renewable energy","engineering and inventions","materials and manufacturing","agriculture and farming","food science","environment and recycling",
 "space exploration","dinosaurs and prehistoric life","natural disasters","cities and urban life","shopping and consumer advice","holidays and celebrations",
 "letters, emails and everyday writing","public speaking","creative writing","humor and jokes","everyday science questions","household budgeting",
 "health insurance and paperwork","emotions and self-care","volunteering and community","famous people and biographies"
};
#define NFIELD (int)(sizeof FIELDS/sizeof *FIELDS)
static const char *AUD[]={"young children (age 6-9)","middle school students","high school students","college students",
                          "curious adults with no background in the subject","professionals who want a quick refresher"};
static const char *FMT[]={   /* %s = topic */
 "a clear textbook section explaining %s",
 "an engaging blog post about %s",
 "a short story in which the characters learn something about %s",
 "a step-by-step practical guide related to %s",
 "a FAQ with several questions and answers about %s",
 "an encyclopedia-style article on %s",
 "a dialogue between a teacher and a student discussing %s",
 "a short lesson with a worked example about %s",
 "a persuasive essay on a question related to %s",
 "a descriptive explanation of how %s works in everyday life"};
static const char *LEN[]={"about 150 words","about 300 words","about 500 words"};
static const char *STYLE[][2]={   /* chatbench-like category, request style */
 {"greet","a friendly greeting or small talk, possibly with a light question about"},
 {"explain","a request to explain a concept from"},
 {"explain","a factual question about"},
 {"advice","a request for practical advice about a personal situation involving"},
 {"howto","a request for step-by-step instructions for a task involving"},
 {"list","a request for a list of ideas, tips or examples about"},
 {"creative","a request for a short poem, story or description about"},
 {"opinion","a request for an opinion or recommendation about"},
 {"math","a simple math or reasoning word problem set in the context of"}};
#define NSTYLE (int)(sizeof STYLE/sizeof *STYLE)
static const char *PERSONA[]={"a teenager","a busy parent","a retiree","a university student","a software developer","a teacher",
                              "a small business owner","a non-native English speaker","a curious child","a nurse"};
static const char *SYS="You are a helpful assistant. Give accurate, clear and well-organized answers. Keep replies concise: "
                       "usually under 150 words, longer only when the task truly needs it.";

typedef struct { char *t, *f; } Topic;
static Topic *g_top; static int g_ntop;
static unsigned long long rng(unsigned long long *s){ unsigned long long z=(*s+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }

static char *g_fres[NFIELD]; static volatile LONG g_fnext; static unsigned long long g_tseed;
static DWORD WINAPI topic_worker(void *arg){
    t_port=g_ports[(int)(INT_PTR)arg%g_nport];
    for(int i; (i=(int)InterlockedIncrement(&g_fnext)-1)<NFIELD; ){
        char m[512]; int eos; long nt;
        snprintf(m,sizeof m,"List 50 distinct, specific subtopics within the field of %s, ranging from basic to advanced. "
                 "Output one subtopic per line, with no numbering, bullets or explanations.",FIELDS[i]);
        g_fres[i]=complete(NULL,m,900,0.8,g_tseed*1000+i,&eos,&nt);
    }
    return 0;
}
static void load_topics(const char *path, int nth){
    FILE *f=fopen(path,"rb"); char line[1024];
    if(!f){
        fprintf(stderr,"[topics] generating for %d fields\n",NFIELD);
        HANDLE th[64]; if(nth>64) nth=64;
        for(int k=0;k<nth;k++) th[k]=CreateThread(NULL,0,topic_worker,(void*)(INT_PTR)k,0,NULL);
        WaitForMultipleObjects(nth,th,TRUE,INFINITE);
        FILE *o=fopen(path,"wb"); int tot=0;
        for(int i=0;i<NFIELD;i++){
            char *r=g_fres[i];
            for(char *l=r?strtok(r,"\n"):NULL; l; l=strtok(NULL,"\n")){
                while(*l=='-'||*l=='*'||*l==' '||(*l>='0'&&*l<='9')||*l=='.'||*l==')') l++;
                l=trim(l); size_t n=strlen(l);
                if(n<3||n>120||strchr(l,'\t')||strchr(l,':')) continue;
                fprintf(o,"%s\t%s\n",FIELDS[i],l); tot++;
            }
            free(r);
        }
        fclose(o); f=fopen(path,"rb");
        fprintf(stderr,"[topics] wrote %d\n",tot);
    }
    int cap=0;
    while(fgets(line,sizeof line,f)){
        line[strcspn(line,"\r\n")]=0; char *tab=strchr(line,'\t'); if(!tab) continue; *tab=0;
        if(g_ntop==cap){ cap=cap?cap*2:4096; g_top=realloc(g_top,sizeof(Topic)*cap); }
        g_top[g_ntop].f=strdup(line); g_top[g_ntop].t=strdup(tab+1); g_ntop++;
    }
    fclose(f);
    fprintf(stderr,"[topics] %d topics\n",g_ntop);
}

static CRITICAL_SECTION g_cs; static FILE *g_out; static unsigned char *g_done;
static volatile LONG g_next, g_kept, g_drop; static volatile LONG64 g_tok, g_keptbytes;
static int g_n; static unsigned long long g_seed; static double g_chat=0.5;

static void emit(int i, const char *kind, const char *cat, long nt, const char *text){
    Buf b={0}; char h[160];
    snprintf(h,sizeof h,"{\"i\":%d,\"kind\":\"%s\",\"cat\":\"%s\",\"tok\":%ld,\"text\":\"",i,kind,cat,nt);
    bstr(&b,h); besc(&b,text); bstr(&b,"\"}\n");
    EnterCriticalSection(&g_cs); fwrite(b.p,1,b.n,g_out); fflush(g_out); LeaveCriticalSection(&g_cs);
    InterlockedIncrement(&g_kept); InterlockedAdd64(&g_keptbytes,(LONG64)strlen(text)); free(b.p);
}
static DWORD WINAPI worker(void *arg){
    t_port=g_ports[(int)(INT_PTR)arg%g_nport];
    for(;;){
        int i=(int)InterlockedIncrement(&g_next)-1; if(i>=g_n) return 0;
        if(g_done[i]) continue;
        unsigned long long s=g_seed*0x100000001B3ULL+i; rng(&s);
        Topic *tp=&g_top[rng(&s)%g_ntop]; char m[1024]; int eos; long nt, nt2;
        if((rng(&s)%1000)/1000.0 < g_chat){
            const char **st=STYLE[rng(&s)%NSTYLE];
            snprintf(m,sizeof m,"Write one realistic message that %s might send to an AI assistant: %s \"%s\" (%s). "
                     "Output only the message itself, without quotes or any introduction.",
                     PERSONA[rng(&s)%10],st[1],tp->t,tp->f);
            char *q=complete(NULL,m,160,1.0,rng(&s),&eos,&nt);
            char *qt=q?trim(q):NULL;
            if(!qt || !eos || strlen(qt)<8 || strstr(qt,"<think>")){ free(q); InterlockedIncrement(&g_drop); continue; }
            if(qt[0]=='"' && qt[strlen(qt)-1]=='"' && strlen(qt)>2){ qt[strlen(qt)-1]=0; qt++; }
            char *a=complete(SYS,qt,600,0.7,rng(&s),&eos,&nt2);
            char *at=a?trim(a):NULL;
            InterlockedAdd64(&g_tok,nt+nt2);
            if(at && eos && strlen(at)>=2 && !strstr(at,"<think>")){
                Buf t={0}; bstr(&t,"<|user|>\n"); bstr(&t,qt); bstr(&t,"\n<|assistant|>\n"); bstr(&t,at);
                emit(i,"chat",st[0],nt2,t.p); free(t.p);
            } else InterlockedIncrement(&g_drop);
            free(q); free(a);
        } else {
            char f[256]; snprintf(f,sizeof f,FMT[rng(&s)%10],tp->t);
            const char *aud=AUD[rng(&s)%6], *len=LEN[rng(&s)%3];
            snprintf(m,sizeof m,"Write %s (a subject within %s) for %s, %s long. Be accurate, clear and engaging. "
                     "Start directly with the content; no meta commentary.",f,tp->f,aud,len);
            char *r=complete(NULL,m,1100,0.8,rng(&s),&eos,&nt);
            char *rt=r?trim(r):NULL;
            InterlockedAdd64(&g_tok,nt);
            if(rt && eos && strlen(rt)>200 && !strstr(rt,"<think>")) emit(i,"text","text",nt,rt);
            else InterlockedIncrement(&g_drop);
            free(r);
        }
    }
}
/* ---- mode "chat": assistant-style conversations (1-3 user turns) for a small chat SFT ---- */
static const char *SYS2="You are LS-Tiny, a friendly and helpful AI assistant made by an independent developer. "
 "Answer accurately and directly. Keep replies short: one or two sentences for greetings, small talk and simple "
 "questions, and at most about 120 words otherwise. Write plain English text without headings, bold text, tables or "
 "emoji; use a short numbered or bulleted list only for steps or several items. Do not end every reply with an offer "
 "of more help. You cannot browse the internet, see images or know the current date. If a message is unclear, briefly "
 "ask what the user means.";
static const char *GREET[]={"hi","hello","hey","hi there","hello!","hey there","hiya","yo","good morning","good afternoon",
 "good evening","morning!","hi, how are you?","how are you?","how are you doing today?","what's up?","sup","howdy","greetings",
 "hey, how's it going?","hello, anyone there?","hi :)","heyy","helo","hii","good night","nice to meet you","hello friend",
 "hey buddy","hi again","i'm back","how's your day going?","how was your day?","whats up","hey whats up","hello how are you",
 "hi! i'm new here","hey, i just wanted to chat","can we talk?","i'm bored, let's chat","hi, can you hear me?","hello?"};
static const char *IDENT[]={"who are you?","what is your name?","what's your name","what are you?","are you a robot?",
 "are you human?","are you an AI?","what can you do?","what can you help me with?","how can you help me?","who made you?",
 "who created you?","are you real?","do you have feelings?","are you smart?","can you think?","do you sleep?","how old are you?",
 "where do you live?","what do you know?","what are you good at?","what can't you do?","can you remember what I said?",
 "do you like me?","are you my friend?","what languages do you speak?","can you browse the internet?","tell me about yourself",
 "introduce yourself","what should i call you?","are you better than other AIs?","what time is it?","can you see me?"};
static const char *VAGUE[]={"help","help me","i need help","can you help me?","i have a question","question","?","hmm","ok",
 "okay","yes","no","idk","i don't know","what?","huh?","test","testing","asdf","...","can i ask you something?","are you there?",
 "hello??","so","um","tell me something","say something","anything interesting?","i'm confused","what should i do?",
 "thanks","thank you","thx","bye","goodbye","lol","haha","nice","cool","got it"};
#define NS(a) (int)(sizeof a/sizeof *a)
/* topic: 0 = subtopic, 1 = field, 2 = TASK (req is a format), 3 = subtopic or field, 4 = none.
 * ex: three '|'-separated example requests (the 2B teacher writes far better users when shown the shape) */
typedef struct { const char *cat; int w; const char **seed; int nseed; const char *req; int topic; double temp; const char *ex; } Scen;
static const Scen SC[]={
 {"greet",70,GREET,NS(GREET),0,0,0.6,0},{"identity",35,IDENT,NS(IDENT),0,0,0.4,0},{"vague",30,VAGUE,NS(VAGUE),0,0,0.6,0},
 {"smalltalk",50,0,0,"to chat casually: share how their day went, a feeling or a hobby, or ask the assistant a light question",1,0.6,
  "i just got back from a long run and i'm exhausted|Do you like rainy days?|My cat knocked my coffee off the table this morning lol"},
 {"fact",150,0,0,"to ask one simple factual question (only the question, not its answer)",3,0.2,
  "What is the capital of Australia?|How many bones are in the human body?|why is the ocean salty"},
 {"explain",120,0,0,"a simple explanation of a concept",0,0.4,
  "Can you explain how vaccines work in simple terms?|what is inflation?|Explain photosynthesis like I'm 10."},
 {"advice",90,0,0,"practical advice about a personal situation",0,0.5,
  "My roommate keeps eating my food. How do I bring it up without a fight?|i get nervous before job interviews, any tips?|What should I do if my plants keep dying?"},
 {"howto",80,0,0,"step-by-step instructions for a practical task",0,0.4,
  "How do I boil an egg?|What are the steps to change a flat tire?|how do i make a monthly budget"},
 {"list",70,0,0,"a short list of ideas, tips or examples",0,0.5,
  "Give me 5 ideas for a rainy weekend.|List some healthy snacks I can take to work.|What are a few good habits for studying?"},
 {"creative",50,0,0,"a very short poem, story, joke or description",3,0.8,
  "Write a short poem about autumn leaves.|tell me a joke about cats|Write a two-sentence story about a lost robot."},
 {"opinion",35,0,0,"an opinion or a recommendation",0,0.5,
  "Is it better to rent or buy a house?|What's a good first programming language?|Do cats or dogs make better pets?"},
 {"math",55,0,0,"help with one simple arithmetic or everyday math word problem with small whole numbers, stated completely (only the problem, not its answer)",4,0.1,
  "What is 48 plus 27?|If I buy 3 notebooks for $4 each, how much do I spend?|A movie starts at 7:15 and lasts 2 hours. When does it end?"},
 {"rewrite",45,0,0,"the assistant to rewrite, summarize, shorten, fix the grammar of, or change the tone of a short text of their own; the text (one to four sentences) must be included in quotes in the message",3,0.4,
  "Can you make this sound more polite? \"Send me the report now.\"|Fix the grammar: \"She don't like going to school on mondays.\"|Summarize in one sentence: \"Our team met on Monday about the new schedule. We agreed to start earlier on Fridays so everyone can leave by 3 pm.\""},
 {"task",60,0,0,"a quick answer, %s",2,0.2,
  "What's another word for happy?|How do you say thank you in French?|Name three animals that live in the ocean."},
 {"personal",30,0,0,"to share a personal detail (their name, where they live, a pet, a preference, a plan, or a limit such as an allergy or a budget) and then ask an everyday question that depends on that detail",1,0.5,
  "I'm vegetarian and friends are coming over Saturday. What could I cook?|My name is Priya and I'm 15. What summer jobs could I do?|i only have $20 for groceries this week, what should i buy?"},
 {"support",30,0,0,"some support because they feel worried, frustrated or sad",1,0.6,
  "I failed my driving test today and I feel really stupid.|i've been feeling lonely since i moved to a new city|I'm so stressed about my exams next week."}};
#define NSC NS(SC)
static const char *TASK[]={"such as a synonym of a common English word","such as the opposite of a common word",
 "such as how to say a common everyday phrase in Spanish, French or German","such as a few examples of a simple category like animals, fruits, colors, countries or jobs",
 "such as yes or no to a simple question about the everyday world","such as the end of a well-known saying or proverb",
 "such as words that rhyme with a simple word","such as a few words sorted alphabetically or a few numbers sorted by size",
 "such as a one-word answer to a simple question","such as the plural or past tense of a common word",
 "such as a short sentence turned into a question or into the past tense","such as what a common English word means",
 "such as a common word used in a sentence","such as which of two everyday things is bigger, heavier or faster"};
static const char *USTY[]={"casual","polite, in complete sentences","short, one sentence","with a little background about their situation",
 "simple English, like a non-native speaker","friendly and enthusiastic"};
static const char *FOLLOW[]={"asks a natural follow-up question about the same subject","asks for part of the answer to be explained more simply",
 "asks for a concrete example","reacts briefly (agreement or surprise) and asks a related question","asks for a shorter answer",
 "asks for more detail on one point","adds a detail about their own situation and asks the assistant to adjust its answer",
 "changes the subject and asks a question about","says thanks, possibly with a goodbye"};
static double g_maxb;

static int bad(const char *s){
    for(const unsigned char *c=(const unsigned char*)s; *c; c++) if(*c>=0xE3 && *c<=0xE9) return 1;   /* CJK */
    return strstr(s,"<|")||strstr(s,"<think>")||strstr(s,"MiniCPM")||strstr(s,"OpenBMB")||strstr(s,"ModelBest")||
           strstr(s,"\n#")||s[0]=='#'||strstr(s,"```");
}
static int replines(const char *s){   /* some non-trivial line occurs twice */
    const char *ln[64]; int len[64], n=0;
    for(const char *p=s; *p && n<64; ){ const char *e=strchr(p,'\n'); int l=e?(int)(e-p):(int)strlen(p);
        if(l>3){ for(int k=0;k<n;k++) if(len[k]==l&&!memcmp(ln[k],p,l)) return 1; ln[n]=p; len[n++]=l; }
        if(!e) break; else p=e+1; }
    return 0;
}
/* drop bold markers, emoji (4-byte UTF-8, VS16, ZWJ) and the spaces they leave behind; curly quotes -> ASCII */
static char *clean(char *s){
    unsigned char *r=(unsigned char*)s, *w=r, *ls=r;
    while(*r){
        if(r[0]=='*'&&r[1]=='*'){ r+=2; continue; }
        if(*r>=0xF0){ r++; while((*r&0xC0)==0x80) r++; continue; }
        if((r[0]==0xEF&&r[1]==0xB8&&r[2]==0x8F)||(r[0]==0xE2&&r[1]==0x80&&r[2]==0x8D)){ r+=3; continue; }
        if(r[0]==0xE2&&r[1]==0x80&&r[2]>=0x98&&r[2]<=0x9D&&r[2]!=0x9A&&r[2]!=0x9B){ *w++=r[2]<0x9C?'\'':'"'; r+=3; continue; }
        if(*r=='\n'){ while(w>ls&&w[-1]==' ') w--; *w++=*r++; ls=w; continue; }
        if(*r==' '&&w>ls&&w[-1]==' '){ int ind=1; for(unsigned char *p=ls;p<w;p++) if(*p!=' '){ind=0;break;} if(!ind){ r++; continue; } }
        *w++=*r++;
    }
    *w=0; return trim(s);
}
/* cut a closing "let me know / feel free to ask" sentence (callers apply it to ~3/4 of replies) */
static void uncloser(char *s){
    static const char *cl[]={"let me know","feel free to","just ask","happy to help","if you need more","if you have more","if you have any","reach out"};
    size_t n=strlen(s); if(n<2) return;
    char *st=s; for(char *p=s; p<s+n-2; p++) if(*p=='\n'||((p[0]=='.'||p[0]=='!'||p[0]=='?')&&p[1]==' ')) st=p+1;
    char low[400]; size_t ln=strlen(st); if(ln>=sizeof low) return;
    for(size_t k=0;k<=ln;k++) low[k]=(char)tolower((unsigned char)st[k]);
    int hit=0; for(int k=0;k<NS(cl);k++) if(strstr(low,cl[k])) hit=1;
    if(!hit||st-s<15) return;
    *st=0; trim(s); char *e=s+strlen(s); while(e>s&&(e[-1]==' '||e[-1]=='\n')) *--e=0;
}
static char *umsg(char *q, unsigned long long *s){   /* teacher-written user message -> bare text, maybe made sloppier */
    char *t=trim(q);
    if(!_strnicmp(t,"User:",5)) t=trim(t+5); else if(!_strnicmp(t,"Message:",8)) t=trim(t+8);
    size_t n=strlen(t); if(n>2&&t[0]=='"'&&t[n-1]=='"'&&!memchr(t+1,'"',n-2)){ t[n-1]=0; t++; }
    t=clean(t);
    static const char *no[]={"Assistant","Sure","Certainly","Of course","Here","I would be happy","I'd be happy","Absolutely"};
    for(int k=0;k<NS(no);k++) if(!_strnicmp(t,no[k],strlen(no[k]))) return NULL;
    static const char *inv[]={"AI assistant","as an AI","I'm an AI","I am an AI","help you","assist you","typo"};
    for(int k=0;k<NS(inv);k++) if(strstr(t,inv[k])) return NULL;
    n=strlen(t); if(n<2||n>=700||bad(t)) return NULL;
    t=strdup(t); int r=(int)(rng(s)%100);
    if(r<15){ for(char *p=t;*p;p++) *p=(char)tolower((unsigned char)*p); n=strlen(t); while(n>1&&strchr(".!",t[n-1])) t[--n]=0; }
    else if(r<22){ int c=0; for(char *p=t;*p;p++) if(isalpha((unsigned char)p[0])&&isalpha((unsigned char)p[1])&&p[1]!=p[0]) c++;   /* swap two letters */
        if(c){ int k=(int)(rng(s)%c); for(char *p=t;*p;p++) if(isalpha((unsigned char)p[0])&&isalpha((unsigned char)p[1])&&p[1]!=p[0]&&!k--){ char x=p[0]; p[0]=p[1]; p[1]=x; break; } } }
    return t;
}
static int isreq(const char *u){   /* a question or a request, not a bare statement */
    static const char *v[]={"write","give","tell","list","explain","name","make","fix","summarize","help","rewrite","describe","suggest",
     "show","translate","create","share","recommend","please","i need","i want","i'd like","i would like","how","what","can","could","compare"};
    if(strchr(u,'?')) return 1;
    for(int k=0;k<NS(v);k++) if(!_strnicmp(u,v[k],strlen(v[k]))) return 1;
    return 0;
}
static const Scen *pick(unsigned long long *s, int teacher_only){
    for(;;){ int w=(int)(rng(s)%1000); const Scen *c=SC; while(w>=c->w){ w-=c->w; c++; }
             if(!teacher_only||c->req) return c; }
}
/* prompt asking the teacher to play the user for scenario c */
static void user_prompt(const Scen *c, unsigned long long *s, char *m, size_t n){
    Topic *tp=&g_top[rng(s)%g_ntop]; char req[512], sub[300]="", ex[1024]; int t=c->topic==3?(int)(rng(s)%2):c->topic;
    if(c->topic==2) snprintf(req,sizeof req,c->req,TASK[rng(s)%NS(TASK)]); else snprintf(req,sizeof req,"%s",c->req);
    if(t==0) snprintf(sub,sizeof sub," The subject is \"%s\" (%s).",tp->t,tp->f);
    else if(t==1) snprintf(sub,sizeof sub," The subject is %s.",tp->f);
    snprintf(ex,sizeof ex,"- %s",c->ex); for(char *p=ex;*p;p++) if(*p=='|'){ memmove(p+3,p+1,strlen(p+1)+1); p[0]='\n'; p[1]='-'; p[2]=' '; }
    snprintf(m,n,"Write a realistic message that a human user (%s) would type to an AI chatbot. The user wants %s.%s\n"
             "Examples of this kind of message:\n%s\nWrite one new, different message. Style: %s. "
             "Output only the message itself, without quotes or any introduction.",
             PERSONA[rng(s)%10],req,sub,ex,USTY[rng(s)%NS(USTY)]);
}
static char *ask_user(const char *m, unsigned long long *s, long *tok){
    char *mm=(char*)m; int eos; long n; char *q=chatc(NULL,&mm,1,200,0.9,rng(s),&eos,&n); *tok+=n;
    char *u=q&&eos?umsg(q,s):NULL; free(q); return u;
}
static DWORD WINAPI chat_worker(void *arg){
    t_port=g_ports[(int)(INT_PTR)arg%g_nport];
    for(;;){
        if(g_maxb>0 && g_keptbytes>=g_maxb) return 0;
        int i=(int)InterlockedIncrement(&g_next)-1; if(i>=g_n) return 0;
        if(g_done[i]) continue;
        unsigned long long s=g_seed*0x100000001B3ULL+i; rng(&s);
        const Scen *sc=pick(&s,0); char *tu[8], m[8192]; int nt=0, end=0, eos; long tok=0, n; size_t chars=0;
        char *u;
        if(sc->seed){ u=strdup(sc->seed[rng(&s)%sc->nseed]); int r=(int)(rng(&s)%10);
            if(r<3) u[0]=(char)toupper((unsigned char)u[0]);
            else if(r<4){ size_t l=strlen(u); while(l>1&&strchr(".!?",u[l-1])) u[--l]=0; } }
        else { user_prompt(sc,&s,m,sizeof m); u=ask_user(m,&s,&tok);
            if(u&&((!strcmp(sc->cat,"rewrite")&&!strchr(u,'"'))||(sc->topic!=1&&!isreq(u)))){ free(u); u=NULL; } }
        while(u){
            tu[nt++]=u; chars+=strlen(u);
            char *a=chatc(SYS2,tu,nt,400,sc->temp,rng(&s),&eos,&n); tok+=n;
            char *at=a&&eos?clean(a):NULL;
            if(at&&rng(&s)%4) uncloser(at);
            if(!at||strlen(at)<2||strlen(at)>1600||bad(at)||replines(at)||(nt>2&&!strcmp(at,tu[nt-2]))){ free(a); free(tu[--nt]); break; }
            tu[nt++]=strdup(at); chars+=strlen(at); free(a);
            double p = nt==2 ? (sc->seed?0.7:0.4) : 0.35;
            if(end||nt>=6||chars>2400||(rng(&s)%1000)/1000.0>=p) break;
            Buf tr={0}; for(int k=0;k<nt;k++){ bstr(&tr,k&1?"Assistant: ":"User: "); bstr(&tr,tu[k]); bstr(&tr,"\n\n"); }
            if(nt==2&&sc->seed){ char f[4096]; user_prompt(pick(&s,1),&s,f,sizeof f);   /* after a greeting: the real request */
                snprintf(m,sizeof m,"Below is the start of a conversation between a user and an AI chatbot.\n\n%s"
                         "The user now makes their real request. %s",tr.p,f); }
            else { char g[600]; int k=(int)(rng(&s)%NS(FOLLOW)); end = k==NS(FOLLOW)-1;
                if(k==7) snprintf(g,sizeof g,"%s %s",FOLLOW[k],g_top[rng(&s)%g_ntop].t); else snprintf(g,sizeof g,"%s",FOLLOW[k]);
                snprintf(m,sizeof m,"Below is a conversation between a user and an AI chatbot.\n\n%s"
                         "Write the human user's next message, in which the user %s. Style: %s. "
                         "Output only the message itself, without quotes, labels or any introduction.",tr.p,g,USTY[rng(&s)%NS(USTY)]); }
            free(tr.p); u=ask_user(m,&s,&tok);
            if(u&&!end&&!isreq(u)){ free(u); u=NULL; }
        }
        if(nt>=2){ Buf t={0};
            for(int k=0;k<nt;k++){ bstr(&t,k&1?"\n<|assistant|>\n":(k?"\n<|user|>\n":"<|user|>\n")); bstr(&t,tu[k]); }
            emit(i,"chat",sc->cat,tok,t.p); free(t.p);
        } else InterlockedIncrement(&g_drop);
        InterlockedAdd64(&g_tok,tok);
        for(int k=0;k<nt;k++) free(tu[k]);
    }
}

int main(int argc, char **argv){
    if(argc<3){ fprintf(stderr,"usage: teachgen <out.jsonl> <ndocs> [threads=16] [port=8099] [chat_frac=0.5] [seed=1] [mode=mix|chat]\n"
                               "  mode chat: assistant conversations only (chat_frac ignored); env TEACH_MAXMB stops at that much text\n"); return 1; }
    g_n=atoi(argv[2]); int nth=argc>3?atoi(argv[3]):16;
    if(argc>4){ g_nport=0; for(char *p=argv[4]; *p && g_nport<16; ){ g_ports[g_nport++]=(int)strtol(p,&p,10); if(*p==',') p++; else break; } }
    t_port=g_ports[0];
    if(argc>5) g_chat=atof(argv[5]);
    g_seed=argc>6?strtoull(argv[6],0,10):1;
    int chat=argc>7&&!strcmp(argv[7],"chat"); { const char *e=getenv("TEACH_MAXMB"); g_maxb=e?atof(e)*1048576.0:0; }
    setvbuf(stderr,NULL,_IONBF,0);
    WSADATA wd; WSAStartup(MAKEWORD(2,2),&wd);
    get_templates();
    char tpath[1024]; snprintf(tpath,sizeof tpath,"%s.topics",argv[1]); g_tseed=g_seed; load_topics(tpath,nth);
    if(!g_ntop){ fprintf(stderr,"no topics\n"); return 1; }
    g_done=calloc(g_n,1); int prev=0;
    FILE *f=fopen(argv[1],"rb");
    if(f){ char *line=malloc(1<<20);
        while(fgets(line,1<<20,f)) if(!strncmp(line,"{\"i\":",5)){ int i=atoi(line+5); if(i>=0&&i<g_n&&!g_done[i]){ g_done[i]=1; prev++; } }
        fclose(f); free(line); }
    g_out=fopen(argv[1],"ab"); if(!g_out){ fprintf(stderr,"cannot write %s\n",argv[1]); return 1; }
    InitializeCriticalSection(&g_cs);
    fprintf(stderr,"[gen] %d docs (%d already done), %d threads, %s %.2f, seed %llu, max %.0f MB\n",g_n,prev,nth,
            chat?"mode chat":"chat",g_chat,g_seed,g_maxb/1048576.0);
    DWORD t0=GetTickCount(); HANDLE *th=malloc(sizeof(HANDLE)*nth);
    for(int k=0;k<nth;k++) th[k]=CreateThread(NULL,0,chat?chat_worker:worker,(void*)(INT_PTR)k,0,NULL);
    for(int live=nth; live; ){
        Sleep(30000); live=0;
        for(int k=0;k<nth;k++) live += WaitForSingleObject(th[k],0)==WAIT_TIMEOUT;
        double sec=(GetTickCount()-t0)/1000.0;
        fprintf(stderr,"  %.0fs kept %ld drop %ld | teacher %lld tok, %.0f tok/s | kept %.1f MB\n",sec,g_kept,g_drop,
                (long long)g_tok,g_tok/(sec>0?sec:1),g_keptbytes/1048576.0);
    }
    fclose(g_out);
    return 0;
}
