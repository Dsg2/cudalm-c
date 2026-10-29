# Multiple-choice scoring against `cudalm serve` POST /logprob (reference client for external benches).
# Input: lmeval-format TSV (gold \t context \t choice1 \t choice2 ...; \n \t \\ escaped) or JSONL {"context","choices","gold"}.
# Prints acc (sum logP), acc_norm (logP/byte, what lmeval and the intelligence index use) and the mean margin.
# Optional: an lmeval items TSV (CUDALM_LMEVAL_ITEMS) to check per-item agreement with `cudalm lmeval`.
#   py -3 logprob_bench.py <file.tsv|.jsonl> [n=all] [url=http://127.0.0.1:8090] [items.tsv]
import json, sys, urllib.request

def unesc(s):
    import re
    return re.sub(r'\\(.)', lambda m: {'n': '\n', 't': '\t'}.get(m.group(1), m.group(1)), s)

def items(path):
    with open(path, encoding='utf-8') as f:
        for l in f:
            l = l.rstrip('\r\n')
            if not l: continue
            if path.endswith('.jsonl'):
                d = json.loads(l); yield int(d['gold']), d['context'], d['choices']
            else:
                f_ = l.split('\t')
                if len(f_) >= 3: yield int(f_[0]), unesc(f_[1]), [unesc(x) for x in f_[2:]]

def post(url, body):
    rq = urllib.request.Request(url + '/logprob', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(rq, timeout=600))

def main():
    path = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    url = sys.argv[3] if len(sys.argv) > 3 else 'http://127.0.0.1:8090'
    ref = None
    if len(sys.argv) > 4:
        task = path.replace('\\', '/').split('/')[-1].split('.')[0]
        ref = {int(r[1]): (int(r[2]), float(r[3])) for r in (l.rstrip('\n').split('\t') for l in open(sys.argv[4], encoding='utf-8')) if r[0] == task}
    k = a = an = 0; ms = 0.0; agree = 0; md = 0.0
    for i, (g, ctx, ch) in enumerate(items(path)):
        if n and i >= n: break
        r = post(url, {'context': ctx, 'continuations': ch})['results']
        lp = [x['logprob'] for x in r]; pb = [x['logprob'] / max(x['bytes'], 1) for x in r]
        a += max(range(len(ch)), key=lp.__getitem__) == g
        ok = max(range(len(ch)), key=pb.__getitem__) == g; an += ok
        m = pb[g] - max(p for j, p in enumerate(pb) if j != g); ms += m; k += 1
        if ref and i in ref: agree += ref[i][0] == ok; md = max(md, abs(ref[i][1] - m))
    print(f'{path}: {k} items  acc {100*a/k:.2f}%  acc_norm {100*an/k:.2f}%  margin {ms/k:+.4f}')
    if ref: print(f'vs lmeval items: acc_norm agreement {agree}/{k}, max |margin diff| {md:.2e}')

main()
