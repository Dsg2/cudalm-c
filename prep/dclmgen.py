# E6: DCLM-baseline-1.0 slice (already the fastText top-10% of DCLM-pool) as an extra corpus for docsel (+dclmN).
#   dclmgen.py <tokenizer.json> <out .bin> <bench dir> <cache dir> [tokens=150e6]
# Docs sharing a 13-word span with any bench test file are dropped (HellaSwag/WikiHow text is common on the web).
import sys, os, re, random, urllib.request
import numpy as np, pyarrow.parquet as pq
from tokenizers import Tokenizer
tk, outp, bd, cd = sys.argv[1:5]; NTOK = float(sys.argv[5]) if len(sys.argv) > 5 else 150e6; os.makedirs(cd, exist_ok=True)
B = ('https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet/resolve/main/filtered/OH_eli5_vs_rw_v2_bigram_200k_train/'
     'fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data/')
SH = ['global-shard_03_of_10/local-shard_0_of_10/shard_00000001_processed.parquet',
      'global-shard_07_of_10/local-shard_5_of_10/shard_00000001_processed.parquet',
      'global-shard_09_of_10/local-shard_2_of_10/shard_00000002_processed.parquet']
G13 = set()
for fn in ('hellaswag', 'arc_easy', 'arc_challenge', 'piqa', 'arithmark', 'lambada'):
    for l in open(os.path.join(bd, fn + '.tsv'), encoding='utf-8'):
        f = l.rstrip('\n').split('\t')
        if len(f) > 2:
            w = re.findall(r'\w+', (f[1] + ' ' + ' '.join(f[2:])).replace('\\n', ' ').lower()); G13.update(hash(tuple(w[i:i + 13])) for i in range(max(0, len(w) - 12)))
tok = Tokenizer.from_file(tk); out = open(outp, 'wb'); ntok = nd = drop = 0; r = random.Random(5)
for s in SH:
    p = os.path.join(cd, 'dclm_' + s.replace('/', '_'))
    if not os.path.exists(p): print('download', s, flush=True); urllib.request.urlretrieve(B + s, p + '.part'); os.replace(p + '.part', p)
    texts = pq.read_table(p, columns=['text']).column('text').to_pylist(); r.shuffle(texts)
    for i in range(0, len(texts), 4000):
        ok = []
        for t in texts[i:i + 4000]:
            w = re.findall(r'\w+', t.lower())
            if any(hash(tuple(w[j:j + 13])) in G13 for j in range(max(0, len(w) - 12))): drop += 1
            else: ok.append(t)
        for e in tok.encode_batch(ok):
            a = np.array(e.ids + [0], np.uint16); out.write(a.tobytes()); ntok += len(a); nd += 1
        if ntok >= NTOK: break
    print(f'{s}: {nd} docs {ntok} tokens, {drop} dropped', flush=True)
    if ntok >= NTOK: break
out.close(); print(f'dclm -> {outp}: {nd} docs {ntok} tokens, {drop} dropped by decontam')
