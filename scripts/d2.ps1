# D2 = D1 recipe + umcv (UltraData-Math L3 conversation, 4x larger hard-half pool) 10% + orca 5% (xdata probes: +0.063 / +0.081
# at 8 reps, ARC vs ArithMark). LS-Micro3_it60000 decayed on D2 with the default LR floor and LR_MIN 0.02, then averaged;
# compared with LS-Micro3-D1 / -D1lr / -D1soup. Results: logs\master\d2.res
$E = 'E:\cudalm'; $X = "$E\cudalm20o.exe"; $M = "$E\models"; $V = "$E\prep\data\v3pre"; $L = "$E\logs\master"; $RES = "$L\d2.res"
$C2 = 'F:\cudalm_stage\xcand2'; $XD = "$E\logs\fp\extra"; $R = 'HWC+arith1+cloze6+umcvx10+orca5'; $D2 = 'D:\cudalm_data\d2_HWC_arith1_cloze6_umcvx10_orca5'
Set-Location $E
function Log($s) { "$(Get-Date -Format 'MM-dd HH:mm') $s" | Out-File -Append -Encoding ascii $RES }
function Clean { Get-ChildItem env: | ? { $_.Name -like 'CUDALM_*' -or $_.Name -like 'QLM_*' -or $_.Name -like 'NGRAM_*' } | % { Remove-Item "env:$($_.Name)" } }
$BASE = @{ CUDALM_NGRAM='1'; CUDALM_ENGRAM='0'; NGRAM_DIM='32'; NGRAM_Q4='1'; NGRAM_Q4_FOLD='1'; NGRAM_GATE='1'; NGRAM_LAYER='0';
           NGRAM_ORDER='2'; NGRAM_TRI='0'; CUDALM_TOPK='1'; CUDALM_ATTN_GATE='1'; CUDALM_VRES='1'; CUDALM_UNIPOOL='0'; CUDALM_QLM='0';
           CUDALM_WARMUP='1000'; CUDALM_WSD_DECAY='0.2'; CUDALM_MASK_PROMPT='0'; CUDALM_BATCH='32';
           NGRAM_ROWS='65536'; CUDALM_D='160'; CUDALM_L='11'; CUDALM_H='5'; CUDALM_KV='1'; CUDALM_F='432' }
function Score($nm) {
    Clean; & $X eval $nm $V 100 4 *> "$L\$nm.eval.log"
    $vl = (Select-String "$L\$nm.eval.log" -Pattern 'val loss ([0-9.]+)').Matches[0].Groups[1].Value
    $env:CUDALM_LMEVAL_ITEMS = "$L\$nm.items.tsv"; & $X lmeval $nm all *> "$L\$nm.lm.log"; Clean
    $ix = (Select-String "$L\$nm.lm.log" -Pattern 'intelligence index ([0-9.]+)').Matches[0].Groups[1].Value
    $mg = (Select-String "$L\$nm.lm.log" -Pattern 'margin index (\S+)').Matches[0].Groups[1].Value
    Log "$nm | val $vl index $ix margin $mg" }
function Vs($a, $b) { Log "== $a -> $b"; & $X lmeval-vs "$L\$a.items.tsv" "$L\$b.items.tsv" 2>&1 | ? { $_ -match 'hellaswag|arc_|piqa|arithmark|margin index' } | % { Log "  $_" } }
if (!(Test-Path $RES)) { Log "start D2 = $R" }
# 1. larger umcv pool (640M candidate tokens -> ~320M hard half), so 10% of 1.8B is < 1 pass
if (!(Test-Path "$XD\umcvx.bin")) {
    if (!(Test-Path "$C2\umcv\gen.json")) { py -3 "$E\prep\xgen.py" gen umcv "$V\tokenizer.json" "$E\bench" F:\cudalm_stage\xsrc $C2 640e6 *> "$L\d2.gen.log"
        Get-ChildItem F:\cudalm_stage\xsrc\umcv_* -EA 0 | Remove-Item -Force; Log "gen $(Get-Content "$L\d2.gen.log" -Tail 1)" }
    Clean; $env:CUDALM_EVAL_SPANS = "$C2\umcv\spans.txt"; $env:CUDALM_EVAL_OUT = "$C2\umcv\docs.tsv"
    & $X eval LS-Micro3 "$C2\umcv" 1 16 *> "$L\d2.score.log"; Clean
    py -3 "$E\prep\xgen.py" sel umcv $C2 "$C2\umcv\docs.tsv" "$XD\umcvx.bin" *> "$L\d2.sel.log"; Log "sel $(Get-Content "$L\d2.sel.log" -Tail 1)" }
# 2. full-scale corpus
if (!(Test-Path "$D2\meta.json")) {
    py -3 "$E\prep\d1build.py" $V "$E\logs\fdec" $XD $D2 $R 1.8e9 *> "$L\d2build.log"; Get-Content "$L\d2build.log" -Tail 4 | % { Log "  $_" } }
# 3. two decays of LS-Micro3_it60000, then their average
foreach ($v in @(@('LS-Micro3-D2', ''), @('LS-Micro3-D2lr', '0.02'))) { $nm = $v[0]
    if (Test-Path "$L\$nm.done") { continue }
    Get-ChildItem "$M\LS-Micro3_it60000.bin*" | % { Copy-Item $_.FullName ("$M\$nm" + $_.Name.Substring('LS-Micro3_it60000'.Length)) -Force }
    Clean; foreach ($k in $BASE.Keys) { Set-Item "env:$k" $BASE[$k] }
    $env:CUDALM_ITERS = '87036'; $env:CUDALM_LR = '1.2e-3'; $env:CUDALM_SEED = '9013'; $env:CUDALM_BENCH_EVERY = '0'; $env:CUDALM_KEEP_EVERY = '0'
    if ($v[1]) { $env:CUDALM_LR_MIN = $v[1] }
    $t0 = Get-Date; & $X train $nm $D2 1 *>> "$L\$nm.train.log"; $ex = $LASTEXITCODE; Clean
    Log "$nm train exit $ex $([int]((Get-Date)-$t0).TotalMinutes) min"; Score $nm; "ok" | Out-File -Encoding ascii "$L\$nm.done" }
$nm = 'LS-Micro3-D2soup'
if (!(Test-Path "$L\$nm.done")) { Clean; & $X avg $nm LS-Micro3-D2 LS-Micro3-D2lr *> "$L\$nm.avg.log"; Score $nm; "ok" | Out-File -Encoding ascii "$L\$nm.done" }
Vs 'LS-Micro3-D1' 'LS-Micro3-D2'; Vs 'LS-Micro3-D1lr' 'LS-Micro3-D2lr'; Vs 'LS-Micro3-D1soup' 'LS-Micro3-D2soup'
Log 'd2 done'
