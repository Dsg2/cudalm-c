# Extra decay sources for docsel / d1build (+<name>N arms), filtered like v3pre's HW recipe.
#   xgen.py gen <name> <tokenizer.json> <bench dir> <cache dir> <cand dir> [tokens=160e6]
#       download -> English / length / exact-dup / 13-gram bench decontam -> <cand dir>/<name>/train.bin (+ spans.txt)
#   xgen.py sel <name> <cand dir> <docs.tsv> <out .bin> [frac=0.5]
#       keep the higher-loss <frac> of the tokens (scores from `cudalm eval` with CUDALM_EVAL_SPANS=spans.txt)
#   xgen.py list
import sys, os, re, json, time, random, hashlib, urllib.request, urllib.error
import numpy as np
HF = 'https://huggingface.co/datasets'
UM = f'{HF}/openbmb/UltraData-Math/resolve/main/data'
def sh(fmt, n, start=0): return [fmt.format(i) for i in range(start, start + n)]
def um3(k): return sh(f'{UM}/UltraData-Math-L3/{k}/UltraData-Math-L3-{k}-part-{{:04d}}-of-50.parquet', 50)
def cos(k, n): return sh(f'{HF}/HuggingFaceTB/cosmopedia/resolve/main/data/{k}/train-{{:05d}}-of-{n:05d}.parquet', n)
col = lambda c: lambda r: r[c]
X = {  # name: (shard urls, row -> text, licence)
  'umqa':   (um3('QA-Synthetic'), col('content'), 'openbmb/UltraData-Math L3 QA (Apache-2.0)'),
  'umtb':   (um3('Textbook-Exercise-Synthetic'), col('content'), 'openbmb/UltraData-Math L3 textbook/exercise (Apache-2.0)'),
  'umms':   (um3('Multi-Style-Synthetic'), col('content'), 'openbmb/UltraData-Math L3 multi-style (Apache-2.0)'),
  'umcv':   (um3('Conversation-Synthetic'), col('content'), 'openbmb/UltraData-Math L3 conversation (Apache-2.0)'),
  'uml':    (sh(f'{UM}/UltraData-Math-L2-preview/UltraData-Math-L2-part-{{:05d}}-of-00138.parquet', 138, 1), col('content'),
             'openbmb/UltraData-Math L2-preview web math (Apache-2.0)'),
  'fwedu':  (sh(f'{HF}/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/{{:03d}}_00000.parquet', 14), col('text'),
             'HuggingFaceFW/fineweb-edu sample-10BT (ODC-By-1.0)'),
  'cwiki':  (cos('wikihow', 2), col('text'), 'HuggingFaceTB/cosmopedia wikihow (Apache-2.0)'),
  'cstory': (cos('stories', 43), col('text'), 'HuggingFaceTB/cosmopedia stories (Apache-2.0)'),
  'ctext':  ([u for z in zip(cos('stanford', 13), cos('openstax', 2) + [None] * 11, cos('khanacademy', 1) + [None] * 12) for u in z if u],
             col('text'), 'HuggingFaceTB/cosmopedia stanford + openstax + khanacademy (Apache-2.0)'),
  'orca':   ([f'{HF}/microsoft/orca-math-word-problems-200k/resolve/main/data/train-00000-of-00001.parquet'],
             lambda r: r['question'].strip() + '\n\n' + r['answer'].strip(), 'microsoft/orca-math-word-problems-200k (MIT)'),
}
W = re.compile(r'\w+')
def grams(t): w = W.findall(t.lower()); return (hash(tuple(w[i:i + 13])) for i in range(max(0, len(w) - 12)))
def bench13(bd):
    g = set()
    for fn in ('hellaswag', 'arc_easy', 'arc_challenge', 'piqa', 'arithmark', 'lambada'):
        p = os.path.join(bd, fn + '.tsv')
        if not os.path.exists(p): continue
        for l in open(p, encoding='utf-8'):
            f = l.rstrip('\n').split('\t')
            if len(f) > 2: g.update(grams((f[1] + ' ' + ' '.join(f[2:])).replace('\\n', ' ')))
    return g
def english(t):
    if len(t) < 200: return False
    n = len(t); a = sum(1 for c in t if c < '\x80')
    return a / n >= 0.97 and sum(c.isalpha() for c in t) / n >= 0.4
def get(url, p):   # resumable (Range), 60 s read timeout: a stalled connection is retried instead of hanging
    if os.path.exists(p): return p
    print('  download', url.rsplit('/', 1)[-1], flush=True); q = p + '.part'
    for k in range(20):
        have = os.path.getsize(q) if os.path.exists(q) else 0
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers={'Range': f'bytes={have}-', 'User-Agent': 'Mozilla/5.0'}), timeout=60)
            if have and getattr(r, 'status', 206) != 206: have = 0
            with open(q, 'ab' if have else 'wb') as f:
                while c := r.read(1 << 20): f.write(c)
            os.replace(q, p); return p
        except urllib.error.HTTPError as e:
            if e.code == 416 and have: os.replace(q, p); return p
            print(f'  retry {k + 1}: HTTP {e.code}', flush=True); time.sleep(3)
        except Exception as e: print(f'  retry {k + 1}: {type(e).__name__} {e}', flush=True); time.sleep(3)
    return None
def gen(name, tk, bd, cache, cand, ntok):
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    urls, fn, lic = X[name]; g13 = bench13(bd); tok = Tokenizer.from_file(tk)
    d = os.path.join(cand, name); os.makedirs(d, exist_ok=True); os.makedirs(cache, exist_ok=True)
    st = dict(rows=0, short_or_foreign=0, dup=0, decontam=0, docs=0, tokens=0); seen = set(); rng = random.Random(name)
    with open(os.path.join(d, 'train.bin'), 'wb', buffering=1 << 24) as out:
        for u in urls:
            p = get(u, os.path.join(cache, name + '_' + u.rsplit('/', 1)[-1]))
            if not p: continue
            pf = pq.ParquetFile(p)
            for b in pf.iter_batches(batch_size=4096):
                ok = []
                for r in b.to_pylist():
                    st['rows'] += 1
                    try: t = fn(r)
                    except Exception: t = None
                    if not isinstance(t, str) or not english(t): st['short_or_foreign'] += 1; continue
                    h = hashlib.md5(' '.join(t.split()).lower().encode()).digest()[:8]
                    if h in seen: st['dup'] += 1; continue
                    seen.add(h)
                    if any(x in g13 for x in grams(t)): st['decontam'] += 1; continue
                    ok.append(t)
                rng.shuffle(ok)
                for e in tok.encode_batch(ok):
                    a = np.array(e.ids + [0], np.uint16); out.write(a.tobytes()); st['tokens'] += len(a); st['docs'] += 1
                if st['tokens'] >= ntok: break
            print(f"  {name} {os.path.basename(p)}: {st['docs']} docs {st['tokens']} tok | drop foreign/short {st['short_or_foreign']} dup {st['dup']} decontam {st['decontam']}", flush=True)
            if st['tokens'] >= ntok: break
    open(os.path.join(d, 'spans.txt'), 'w').write(f"0 {st['tokens']}\n")
    st.update(source=lic); json.dump(st, open(os.path.join(d, 'gen.json'), 'w'), indent=1)
    print(f"{name}: {st['docs']} docs {st['tokens']} tokens -> {d}", flush=True)
def sel(name, cand, docs, outp, frac):
    tb = np.memmap(os.path.join(cand, name, 'train.bin'), dtype=np.uint16, mode='r')
    v = []
    for l in open(docs):
        ds, n, s = l.split('\t'); ds, n = int(ds), int(n); v.append((float(s) / n, ds, n))
    v.sort(reverse=True); tot = sum(x[2] for x in v); acc = 0; k = []
    for x in v:
        if acc >= tot * frac: break
        k.append(x); acc += x[2]
    random.Random(name).shuffle(k); nt = 0
    with open(outp, 'wb', buffering=1 << 24) as f:
        for _, ds, n in k:
            seg = tb[ds:ds + n]; f.write(seg.tobytes()); nt += n
            if seg[-1] != 0: f.write(b'\0\0'); nt += 1
    ls = [x[0] for x in v]
    r = dict(name=name, docs=len(v), tokens=tot, kept_docs=len(k), kept_tokens=nt, loss_mean=float(np.mean(ls)),
             loss_min_kept=round(k and min(x[0] for x in k) or 0, 4))
    json.dump(r, open(os.path.join(cand, name, 'sel.json'), 'w'), indent=1); print(json.dumps(r), flush=True)
if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'list':
        for k, (u, _, l) in X.items(): print(f'{k:7s} {len(u):3d} shards  {l}')
    elif a[0] == 'gen': gen(a[1], a[2], a[3], a[4], a[5], float(a[6]) if len(a) > 6 else 160e6)
    elif a[0] == 'sel': sel(a[1], a[2], a[3], a[4], float(a[5]) if len(a) > 5 else 0.5)
