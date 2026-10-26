# E2: benchmark-format train splits as cloze text (gold answer written as the continuation) + a 10% dev carve-out.
#   clozegen.py <tokenizer.json> <out .bin> <dev .tsv> <bench dir> <cache dir>
# Train splits only: HellaSwag, ARC-E/C, PIQA, OpenBookQA, SciQ, CommonsenseQA, WinoGrande-xl. Prompts follow lmeval_prep.py
# (ARC/PIQA "Question: ...\nAnswer: ...", HellaSwag "activity: ctx ending"). Every item sharing a 13-word or 50-char span
# with any bench test file (hellaswag arc_easy arc_challenge piqa arithmark lambada) is dropped. Dev (10% of HS, ARC, PIQA,
# OBQA, CSQA items, MC in the same prompt format) never enters the train text.
import sys, os, re, json, random, urllib.request
import numpy as np, pyarrow.parquet as pq
from tokenizers import Tokenizer
tk, outp, devp, bd, cd = sys.argv[1:6]; os.makedirs(cd, exist_ok=True)
def hf(ds, cfg, split):
    p = os.path.join(cd, f'{ds.replace("/", "_")}_{cfg}_{split}.parquet')
    if not os.path.exists(p):
        j = json.load(urllib.request.urlopen(f'https://datasets-server.huggingface.co/parquet?dataset={ds}', timeout=60))
        u = [f['url'] for f in j['parquet_files'] if f['config'] == cfg and f['split'] == split]
        import pyarrow as pa; ts = []
        for i, x in enumerate(u): urllib.request.urlretrieve(x, p + f'.{i}'); ts.append(pq.read_table(p + f'.{i}'))
        pq.write_table(pa.concat_tables(ts), p); [os.remove(p + f'.{i}') for i in range(len(u))]
    return pq.read_table(p).to_pylist()
def raw(url, fn):
    p = os.path.join(cd, fn)
    if not os.path.exists(p): urllib.request.urlretrieve(url, p)
    return open(p, encoding='utf-8').read()
def hs_pre(t):
    t = t.strip().replace(' [title]', '. '); return re.sub(r'\[.*?\]', '', t).replace('  ', ' ')
items = []  # (source, context, choices, gold, prefix text for train only)
for r in hf('Rowan/hellaswag', 'default', 'train'):
    items.append(('hs', hs_pre(r['activity_label'] + ': ' + r['ctx_a'] + ' ' + r['ctx_b'].capitalize()), [' ' + hs_pre(e) for e in r['endings']], int(r['label']), ''))
for cfg in ('ARC-Easy', 'ARC-Challenge'):
    for r in hf('allenai/ai2_arc', cfg, 'train'):
        lab = r['choices']['label']
        if r['answerKey'] in lab: items.append(('arc', 'Question: ' + r['question'] + '\nAnswer:', [' ' + t for t in r['choices']['text']], lab.index(r['answerKey']), ''))
G = 'https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/'
for ln, g in zip(raw(G + 'train.jsonl', 'piqa_train.jsonl').split('\n'), raw(G + 'train-labels.lst', 'piqa_train-labels.lst').split()):
    r = json.loads(ln); items.append(('piqa', 'Question: ' + r['goal'] + '\nAnswer:', [' ' + r['sol1'], ' ' + r['sol2']], int(g), ''))
for r in hf('allenai/openbookqa', 'additional', 'train'):
    lab = r['choices']['label']
    items.append(('obqa', 'Question: ' + r['question_stem'] + '\nAnswer:', [' ' + t for t in r['choices']['text']], lab.index(r['answerKey']), r.get('fact1', '') or ''))
for r in hf('allenai/sciq', 'default', 'train'):
    items.append(('sciq', 'Question: ' + r['question'] + '\nAnswer:', [' ' + r['correct_answer']], 0, r['support'] or ''))
for r in hf('tau/commonsense_qa', 'default', 'train'):
    lab = r['choices']['label']
    if r['answerKey'] in lab: items.append(('csqa', 'Question: ' + r['question'] + '\nAnswer:', [' ' + t for t in r['choices']['text']], lab.index(r['answerKey']), ''))
for r in hf('allenai/winogrande', 'winogrande_xl', 'train'):
    a, b = r['sentence'].split('_', 1) if '_' in r['sentence'] else (r['sentence'], '')
    items.append(('wg', a.rstrip(), [' ' + r['option1'] + b, ' ' + r['option2'] + b], int(r['answer']) - 1, ''))
print(len(items), 'items', {s: sum(1 for i in items if i[0] == s) for s in sorted({i[0] for i in items})})
# decontamination
def bench_texts():
    for fn in ('hellaswag', 'arc_easy', 'arc_challenge', 'piqa', 'arithmark', 'lambada'):
        for l in open(os.path.join(bd, fn + '.tsv'), encoding='utf-8'):
            f = l.rstrip('\n').split('\t')
            if len(f) > 2: yield (f[1] + ' ' + ' '.join(f[2:])).replace('\\n', ' ').lower()
G13, C50 = set(), set()
for s in bench_texts():
    w = re.findall(r'\w+', s); G13.update(tuple(w[i:i + 13]) for i in range(max(0, len(w) - 12)))
    C50.update(s[i:i + 50] for i in range(0, max(1, len(s) - 49)))
def clean(s):
    s = s.lower(); w = re.findall(r'\w+', s)
    return not any(tuple(w[i:i + 13]) in G13 for i in range(max(0, len(w) - 12))) and not any(s[i:i + 50] in C50 for i in range(0, max(1, len(s) - 49), 5))
r = random.Random(77); keep, drop, dev = [], {}, []
for it in items:
    s, ctx, ch, g, pre = it
    if not clean(pre + ' ' + ctx + ' ' + ' '.join(ch)): drop[s] = drop.get(s, 0) + 1; continue
    if s in ('hs', 'arc', 'piqa', 'obqa', 'csqa') and r.random() < 0.1: dev.append(it)
    else: keep.append(it)
esc = lambda x: x.replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n')
with open(devp, 'w', encoding='utf-8', newline='\n') as f:
    for s, ctx, ch, g, pre in dev: f.write('\t'.join([str(g), esc(ctx)] + [esc(x) for x in ch]) + '\n')
# train docs: 4-16 same-source items per doc, blank-line separated; OBQA/SciQ put their fact/support passage first
by = {}
for it in keep: by.setdefault(it[0], []).append(it)
texts = []
for s, v in by.items():
    r.shuffle(v); i = 0
    while i < len(v):
        k = r.randint(4, 16); texts.append('\n\n'.join(((pre.strip() + '\n') if pre.strip() else '') + ctx + ch[g] for _, ctx, ch, g, pre in v[i:i + k])); i += k
r.shuffle(texts); tok = Tokenizer.from_file(tk); docs = [np.array(e.ids + [0], np.uint16) for e in tok.encode_batch(texts)]
np.concatenate(docs).tofile(outp)
with open(outp + '.sample.txt', 'w', encoding='utf-8') as f: f.write('\n\n=====\n\n'.join(texts[:6]))
print(f'dropped by decontam {drop}; dev {len(dev)} items -> {devp}; train {len(keep)} items, {len(docs)} docs, {sum(map(len, docs))} tokens -> {outp}')
