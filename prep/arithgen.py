# E3: grade-school arithmetic as answer-as-continuation text + a held-out dev benchmark (AM-dev).
#   arithgen.py <tokenizer.json> <out .bin> <dev .tsv> <bench arithmark.tsv> [train tokens=40e6] [dev items=5000]
# Own templates over ArithMark-3's 16 topics (none copied from it). Family 0 of every topic is DEV ONLY: AM-dev is drawn
# from it, train text never uses it. Train docs drop any problem sharing a 13-word or 50-char span with ArithMark.
# Train doc = 4-10 problems, each an answer sentence (65%) or answer + ", because <equation>." (35%); 15% of docs number
# them "Problem N:"; 12% of lines are bare equation drills.
import sys, random, re
import numpy as np
from tokenizers import Tokenizer
tk, outp, devp, amp = sys.argv[1:5]; NTOK = float(sys.argv[5]) if len(sys.argv) > 5 else 40e6
NDEV = int(sys.argv[6]) if len(sys.argv) > 6 else 5000
NAMES = ('Ava Ben Cora Dev Ella Finn Gia Hana Ivan Jada Kai Lena Milo Nora Omar Pia Quinn Rosa Sami Tara Uma Vic Wren Yara '
         'Zane Aria Bo Cleo Dario Esme Felix Greta Hugo Iris Joel Kira Leo Maya Nico Olga Paz Rhea Sora Theo Una Vera Will Xia').split()
OBJ = [('shell', 'shells'), ('marble', 'marbles'), ('sticker', 'stickers'), ('apple', 'apples'), ('card', 'cards'),
       ('pencil', 'pencils'), ('coin', 'coins'), ('bead', 'beads'), ('cookie', 'cookies'), ('stamp', 'stamps'),
       ('book', 'books'), ('balloon', 'balloons'), ('crayon', 'crayons'), ('leaf', 'leaves'), ('rock', 'rocks'),
       ('toy car', 'toy cars'), ('cupcake', 'cupcakes'), ('flower', 'flowers'), ('button', 'buttons'), ('ticket', 'tickets'),
       ('peach', 'peaches'), ('box of juice', 'boxes of juice'), ('feather', 'feathers'), ('block', 'blocks')]
def U(n, u): return f'{n} {u[0] if n == 1 else u[1]}'
def rng_n(r, d, lo, hi_e, hi_m): return r.randint(lo, hi_e if d == 'easy' else hi_m)
# each family: f(r, d, n, m, o) -> (context, answer, unit pair or None, equation)
def add(r, d):
    a, b = rng_n(r, d, 2, 40, 120), rng_n(r, d, 2, 40, 90); return a, b, a + b
def sub(r, d):
    a = rng_n(r, d, 10, 50, 160); b = r.randint(1, a - 1); return a, b, a - b
FAM = {
 'addition': [
  lambda r, d, n, m, o: (lambda a, b, c: (f'{n} counted {U(a, o)} in the morning and {U(b, o)} in the afternoon. In total, {n} counted', c, o, f'{a} + {b} = {c}'))(*add(r, d)),
  lambda r, d, n, m, o: (lambda a, b, c: (f'{n} had {U(a, o)}. Then {n} found {b} more. Now {n} has', c, o, f'{a} + {b} = {c}'))(*add(r, d)),
  lambda r, d, n, m, o: (lambda a, b, c: (f'There were {U(a, o)} on the table and {U(b, o)} on the shelf. Altogether there were', c, o, f'{a} + {b} = {c}'))(*add(r, d)),
  lambda r, d, n, m, o: (lambda a, b, c: (f'{n} collected {U(a, o)} on Monday and {b} on Tuesday. Over the two days, {n} collected', c, o, f'{a} + {b} = {c}'))(*add(r, d))],
 'subtraction': [
  lambda r, d, n, m, o: (lambda a, b, c: (f'{n} started with {U(a, o)} and gave {b} to a neighbor. After that, {n} had', c, o, f'{a} - {b} = {c}'))(*sub(r, d)),
  lambda r, d, n, m, o: (lambda a, b, c: (f'A jar held {U(a, o)}. {n} removed {b} of them. The jar now held', c, o, f'{a} - {b} = {c}'))(*sub(r, d)),
  lambda r, d, n, m, o: (lambda a, b, c: (f'{n} had {U(a, o)}, but {b} got lost on the bus. {n} still had', c, o, f'{a} - {b} = {c}'))(*sub(r, d)),
  lambda r, d, n, m, o: (lambda a, b, c: (f'There were {U(a, o)} in the basket. After {n} took {b}, the basket had', c, o, f'{a} - {b} = {c}'))(*sub(r, d))],
 'multiplication': [
  lambda r, d, n, m, o: (lambda a, b: (f'{n} bought {a} packs with {U(b, o)} in each pack. In all, {n} bought', a * b, o, f'{a} x {b} = {a * b}'))(r.randint(2, 9), rng_n(r, d, 2, 10, 15)),
  lambda r, d, n, m, o: (lambda a, b: (f'There are {a} tables with {U(b, o)} on each table. All together there are', a * b, o, f'{a} x {b} = {a * b}'))(r.randint(2, 9), rng_n(r, d, 2, 10, 15)),
  lambda r, d, n, m, o: (lambda a, b: (f'{n} read {b} pages every day for {a} days. In those days, {n} read', a * b, ('page', 'pages'), f'{a} x {b} = {a * b}'))(r.randint(2, 9), rng_n(r, d, 2, 10, 25)),
  lambda r, d, n, m, o: (lambda a, b: (f'A garden has {a} rows of plants with {b} plants in each row. The garden has', a * b, ('plant', 'plants'), f'{a} x {b} = {a * b}'))(r.randint(2, 9), rng_n(r, d, 2, 10, 15))],
 'division': [
  lambda r, d, n, m, o: (lambda a, b: (f'{n} shared {U(a * b, o)} equally among {a} friends. Each friend got', b, o, f'{a * b} / {a} = {b}'))(r.randint(2, 9), rng_n(r, d, 1, 10, 15)),
  lambda r, d, n, m, o: (lambda a, b: (f'{U(a * b, o).capitalize()} were put into {a} equal piles. Each pile had', b, o, f'{a * b} / {a} = {b}'))(r.randint(2, 9), rng_n(r, d, 1, 10, 15)),
  lambda r, d, n, m, o: (lambda a, b: (f'A teacher split {U(a * b, o)} evenly between {a} tables. Every table received', b, o, f'{a * b} / {a} = {b}'))(r.randint(2, 9), rng_n(r, d, 1, 10, 15))],
 'comparison': [
  lambda r, d, n, m, o: (lambda a, b: (f'{n} has {U(a, o)}. {m} has {b} more than {n}. Together they have', 2 * a + b, o, f'{a} + ({a} + {b}) = {2 * a + b}'))(rng_n(r, d, 2, 30, 70), r.randint(1, 20)),
  lambda r, d, n, m, o: (lambda a, b: (f'{n} picked {U(a, o)}, and {m} picked {b} fewer than {n}. The two of them picked', 2 * a - b, o, f'{a} + ({a} - {b}) = {2 * a - b}'))(rng_n(r, d, 12, 30, 70), r.randint(1, 10)),
  lambda r, d, n, m, o: (lambda a, b: (f'A red bag holds {U(a, o)}. A blue bag holds {b} more than the red bag. Both bags hold', 2 * a + b, o, f'{a} + ({a} + {b}) = {2 * a + b}'))(rng_n(r, d, 2, 30, 70), r.randint(1, 20))],
 'comparison_difference': [
  lambda r, d, n, m, o: (lambda a, b: (f'{n} scored {a} points and {m} scored {b} points. {n} scored more than {m} by', a - b, ('point', 'points'), f'{a} - {b} = {a - b}'))(*sorted(r.sample(range(3, 60 if d == 'easy' else 150), 2), reverse=True)),
  lambda r, d, n, m, o: (lambda a, b: (f'{n} owns {U(a, o)} and {m} owns {U(b, o)}. The difference between them is', a - b, o, f'{a} - {b} = {a - b}'))(*sorted(r.sample(range(3, 60 if d == 'easy' else 150), 2), reverse=True)),
  lambda r, d, n, m, o: (lambda a, b: (f'A tall stack has {U(a, o)} and a short stack has {U(b, o)}. The short stack would need this many more to match:', a - b, o, f'{a} - {b} = {a - b}'))(*sorted(r.sample(range(3, 60 if d == 'easy' else 150), 2), reverse=True))],
 'data': [
  lambda r, d, n, m, o: (lambda a, b: (f'In a survey, {a} students chose apples and {b} chose bananas. The number of students who answered was', a + b, ('student', 'students'), f'{a} + {b} = {a + b}'))(rng_n(r, d, 2, 30, 80), rng_n(r, d, 2, 30, 80)),
  lambda r, d, n, m, o: (lambda a, b: (f'A tally chart shows {a} votes for soccer and {b} votes for tennis. Soccer got more votes than tennis by', a - b, ('vote', 'votes'), f'{a} - {b} = {a - b}'))(*sorted(r.sample(range(2, 50 if d == 'easy' else 120), 2), reverse=True)),
  lambda r, d, n, m, o: (lambda a, b: (f'A bar graph shows {a} sunny days and {b} rainy days. Sunny and rainy days together made', a + b, ('day', 'days'), f'{a} + {b} = {a + b}'))(rng_n(r, d, 2, 20, 60), rng_n(r, d, 2, 20, 60)),
  lambda r, d, n, m, o: (lambda a, b: (f'The class picture graph had {a} cats and {b} dogs as pets. There were more cats than dogs by', a - b, ('pet', 'pets'), f'{a} - {b} = {a - b}'))(*sorted(r.sample(range(2, 40 if d == 'easy' else 90), 2), reverse=True))],
 'fractions_counting': [
  lambda r, d, n, m, o: (lambda t: (f'{n} had {U(2 * t, o)} and gave away half of them. {n} kept', t, o, f'{2 * t} / 2 = {t}'))(rng_n(r, d, 1, 20, 50)),
  lambda r, d, n, m, o: (lambda t: (f'There were {U(3 * t, o)}, and one third of them were red. The number of red ones was', t, o, f'{3 * t} / 3 = {t}'))(rng_n(r, d, 1, 12, 30)),
  lambda r, d, n, m, o: (lambda a, b: (f'{n} cut a pizza into {a} slices and ate {b}. The slices left numbered', a - b, ('slice', 'slices'), f'{a} - {b} = {a - b}'))(*(lambda a: (a, r.randint(1, a - 1)))(r.choice([4, 6, 8, 10, 12, 16]))),
  lambda r, d, n, m, o: (lambda a, b: (f'Two plates had {U(a, o)} each. {n} ate {b}. The plates still had', 2 * a - b, o, f'{a} + {a} - {b} = {2 * a - b}'))(*(lambda a: (a, r.randint(1, 2 * a - 1)))(rng_n(r, d, 2, 20, 45)))],
 'geometry_area': [
  lambda r, d, n, m, o: (lambda a, b: (f'A rug is {a} feet long and {b} feet wide. Its area is', a * b, ('square foot', 'square feet'), f'{a} x {b} = {a * b}'))(r.randint(2, 12), r.randint(2, 12)),
  lambda r, d, n, m, o: (lambda a, b: (f'A rectangle measures {a} units by {b} units. The area of the rectangle is', a * b, ('square unit', 'square units'), f'{a} x {b} = {a * b}'))(r.randint(2, 12), r.randint(2, 15)),
  lambda r, d, n, m, o: (lambda a, b: (f'{n} covered a floor with {a} rows of {b} tiles. The number of tiles used was', a * b, ('tile', 'tiles'), f'{a} x {b} = {a * b}'))(r.randint(2, 10), r.randint(2, 12))],
 'geometry_perimeter': [
  lambda r, d, n, m, o: (lambda a, b: (f'A rectangular garden is {a} meters long and {b} meters wide. The distance around the garden is', 2 * (a + b), ('meter', 'meters'), f'2 x ({a} + {b}) = {2 * (a + b)}'))(r.randint(2, 30), r.randint(2, 30)),
  lambda r, d, n, m, o: (lambda a: (f'A square has sides of {a} centimeters. Its perimeter is', 4 * a, ('centimeter', 'centimeters'), f'4 x {a} = {4 * a}'))(r.randint(2, 25)),
  lambda r, d, n, m, o: (lambda a, b, c: (f'A triangle has sides of {a}, {b}, and {c} inches. Its perimeter is', a + b + c, ('inch', 'inches'), f'{a} + {b} + {c} = {a + b + c}'))(r.randint(3, 20), r.randint(3, 20), r.randint(3, 20))],
 'measurement': [
  lambda r, d, n, m, o: (lambda a, b, c: (f'A rope was {a} meters long. {n} cut off {b} meters. The rope left was', c, ('meter', 'meters'), f'{a} - {b} = {c}'))(*sub(r, d)),
  lambda r, d, n, m, o: (lambda a, b, c: (f'{n} walked {a} meters and then {b} more meters. In total, {n} walked', c, ('meter', 'meters'), f'{a} + {b} = {c}'))(*add(r, d)),
  lambda r, d, n, m, o: (lambda a, b, c: (f'One ribbon is {a} centimeters long and another is {b} centimeters long. The longer ribbon is longer by', c, ('centimeter', 'centimeters'), f'{a} - {b} = {c}'))(*sub(r, d))],
 'money': [
  lambda r, d, n, m, o: (lambda p, c: (f'{n} paid with a {p}-dollar bill for a toy that cost {c} dollars. The change was', p - c, ('dollar', 'dollars'), f'{p} - {c} = {p - c}'))(*(lambda p: (p, r.randint(1, p - 1)))(r.choice([10, 20, 50, 100]))),
  lambda r, d, n, m, o: (lambda a, b, c: (f'{n} had {a} dollars and spent {b} dollars on a book. {n} had left', c, ('dollar', 'dollars'), f'{a} - {b} = {c}'))(*sub(r, d)),
  lambda r, d, n, m, o: (lambda a, b, c: (f'A pen costs {a} cents and a notebook costs {b} cents. Together they cost', c, ('cent', 'cents'), f'{a} + {b} = {c}'))(*add(r, d))],
 'patterns': [
  lambda r, d, n, m, o: (lambda x, s: (f'The sequence goes {x}, {x + s}, {x + 2 * s}, {x + 3 * s}. The next number is', x + 4 * s, None, f'{x + 3 * s} + {s} = {x + 4 * s}'))(r.randint(0, 40), r.randint(2, 12)),
  lambda r, d, n, m, o: (lambda x, s: (f'Counting backward by {s}: {x}, {x - s}, {x - 2 * s}. The next number is', x - 3 * s, None, f'{x - 2 * s} - {s} = {x - 3 * s}'))(r.randint(40, 120), r.randint(2, 10)),
  lambda r, d, n, m, o: (lambda x, s: (f'{n} counts by {s}s starting at {x}: {x}, {x + s}, {x + 2 * s}. The number after that is', x + 3 * s, None, f'{x + 2 * s} + {s} = {x + 3 * s}'))(r.randint(0, 50), r.choice([2, 3, 4, 5, 10]))],
 'time': [
  lambda r, d, n, m, o: (lambda a, b, c: (f'A movie lasted {a} minutes. {n} had watched {b} minutes. The time remaining was', c, ('minute', 'minutes'), f'{a} - {b} = {c}'))(*sub(r, d)),
  lambda r, d, n, m, o: (lambda a, b, c: (f'{n} practiced piano for {a} minutes and then guitar for {b} minutes. In all, {n} practiced for', c, ('minute', 'minutes'), f'{a} + {b} = {c}'))(*add(r, d)),
  lambda r, d, n, m, o: (lambda a, b, c: (f'{n} had {a} minutes for homework and used {b} of them. The minutes left came to', c, ('minute', 'minutes'), f'{a} - {b} = {c}'))(*sub(r, d))],
 'two_step': [
  lambda r, d, n, m, o: (lambda a, b, c: (f'{n} had {U(a, o)}, got {b} more, then gave away {c}. Now {n} has', a + b - c, o, f'{a} + {b} - {c} = {a + b - c}'))(*(lambda a, b: (a, b, r.randint(1, a + b - 1)))(rng_n(r, d, 2, 30, 90), rng_n(r, d, 2, 30, 60))),
  lambda r, d, n, m, o: (lambda a, b, c: (f'A shelf had {U(a, o)}. {b} were added and later {c} were taken away. The shelf ended with', a + b - c, o, f'{a} + {b} - {c} = {a + b - c}'))(*(lambda a, b: (a, b, r.randint(1, a + b - 1)))(rng_n(r, d, 2, 30, 90), rng_n(r, d, 2, 30, 60))),
  lambda r, d, n, m, o: (lambda a, b, c: (f'{n} collected {U(a, o)} on Friday, {b} on Saturday, and {c} on Sunday. Over the weekend, {n} collected', a + b + c, o, f'{a} + {b} + {c} = {a + b + c}'))(rng_n(r, d, 2, 25, 60), rng_n(r, d, 2, 25, 60), rng_n(r, d, 2, 25, 60)),
  lambda r, d, n, m, o: (lambda a, b, c: (f'A box had {U(a, o)}. {n} took out {b} and then {c} more. The box had', a - b - c, o, f'{a} - {b} - {c} = {a - b - c}'))(*(lambda a: (lambda b: (a, b, r.randint(1, a - b)))(r.randint(1, a // 2)))(rng_n(r, d, 10, 50, 150)))],
}
def ans_text(c, u): return f' {c}.' if u is None else f' {U(c, u)}.'
def gen(r, fam):
    t = r.choice(list(FAM)); fs = FAM[t]; k = 0 if fam == 'dev' else r.randrange(1, len(fs))
    n, m = r.sample(NAMES, 2); o = r.choice(OBJ); d = 'easy' if r.random() < 0.42 else 'medium'
    ctx, c, u, eq = fs[k](r, d, n, m, o)
    return t, k, ctx, c, u, eq
# decontamination against ArithMark contexts
am = [l.split('\t') for l in open(amp, encoding='utf-8').read().split('\n') if l]
amtxt = [(f[1] + f[1 + 1 + int(f[0])]).lower() for f in am]
G13 = {tuple(w[i:i + 13]) for s in amtxt for w in [re.findall(r"\w+", s)] for i in range(max(0, len(w) - 12))}
C50 = {s[i:i + 50] for s in amtxt for i in range(0, max(1, len(s) - 49))}
def clean(s):
    s = s.lower(); w = re.findall(r"\w+", s)
    return not any(tuple(w[i:i + 13]) in G13 for i in range(max(0, len(w) - 12))) and not any(s[i:i + 50] in C50 for i in range(0, max(1, len(s) - 49), 7))
r = random.Random(4242)
# AM-dev: family 0 only, AM's format (4 consecutive candidates containing the answer, shuffled), unique contexts
seen, dev = set(), []
while len(dev) < NDEV:
    t, k, ctx, c, u, eq = gen(r, 'dev')
    if ctx in seen or c < 0: continue
    lo = max(0, c - r.randint(0, 3)); cand = [lo + i for i in range(4)]; r.shuffle(cand)
    seen.add(ctx); dev.append((cand.index(c), ctx, [ans_text(x, u) for x in cand]))
with open(devp, 'w', encoding='utf-8', newline='\n') as f:
    for g, ctx, e in dev: f.write('\t'.join([str(g), ctx] + e) + '\n')
tok = Tokenizer.from_file(tk); EOT = 0
docs, ntok, drop, nprob = [], 0, 0, 0
while ntok < NTOK:
    texts = []
    for _ in range(2000):
        lines = []; num = r.random() < 0.15
        if r.random() < 0.3: lines.append(r.choice(['Practice problems:', 'Word problems with answers.', 'Math practice', 'Solve each problem.', 'Story problems']))
        for i in range(r.randint(4, 10)):
            if r.random() < 0.12:  # bare equation drill
                a, b = r.randint(1, 99), r.randint(1, 99); op = r.choice('+-x')
                if op == '-' and b > a: a, b = b, a
                lines.append(f'{a} {op} {b} = {a + b if op == "+" else a - b if op == "-" else a * b}'); continue
            t, k, ctx, c, u, eq = gen(r, 'train'); nprob += 1
            txt = ctx + ans_text(c, u) if r.random() < 0.65 else ctx + ans_text(c, u)[:-1] + f', because {eq}.'
            if not clean(txt): drop += 1; continue
            lines.append(f'Problem {i + 1}: {txt}' if num else txt)
        texts.append('\n'.join(lines))
    for e in tok.encode_batch(texts):
        docs.append(np.array(e.ids + [EOT], np.uint16)); ntok += len(e.ids) + 1
np.concatenate(docs).tofile(outp)
with open(outp + '.sample.txt', 'w', encoding='utf-8') as f: f.write('\n\n'.join(tok.decode(d[:-1].tolist()) for d in docs[:8]))
print(f'AM-dev {len(dev)} items -> {devp}; train {len(docs)} docs {ntok} tokens {nprob} problems, {drop} dropped by decontam -> {outp}')
