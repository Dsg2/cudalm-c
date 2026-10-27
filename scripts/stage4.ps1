# II plan follow-up, runs after stage3.ps1 (waits for its final line). Results: logs\master\stage4.res
#  1. loop arm: stage 3's from-scratch proxy (18k steps, seeds 9301/9302) with CUDALM_LOOP=3-7x2 (depth 11 -> 16, same
#     params, ~1.3x compute), vs S3_base_<seed> (equal steps; train minutes logged for the compute read).
#  2. LS-Micro3-D1lr: if stage 3 adopted changes, LS-Micro3_it60000 decayed on D1 again with CUDALM_LR_MIN=0.02
#     (D1A-lrmin beat D1A by +0.274 margin, II 9.05); compared with D1A-lrmin and LS-Micro3-D1.
$E = 'E:\cudalm'; $X = "$E\cudalm20o.exe"; $M = "$E\models"; $V = "$E\prep\data\v3pre"; $L = "$E\logs\master"; $RES = "$L\stage4.res"
$S3 = "$L\stage3.res"; Set-Location $E
function Log($s) { "$(Get-Date -Format 'MM-dd HH:mm') $s" | Out-File -Append -Encoding ascii $RES }
function Clean { Get-ChildItem env: | ? { $_.Name -like 'CUDALM_*' -or $_.Name -like 'QLM_*' -or $_.Name -like 'NGRAM_*' } | % { Remove-Item "env:$($_.Name)" } }
if (!(Test-Path $RES)) { Log 'stage 4 queued: waiting for stage 3' }
while (!(Test-Path $S3) -or ((Get-Content $S3 -Tail 1) -notmatch 'stage 3 done|- done$')) { Start-Sleep 60 }
$BASE = @{ CUDALM_NGRAM='1'; CUDALM_ENGRAM='0'; NGRAM_DIM='32'; NGRAM_Q4='1'; NGRAM_Q4_FOLD='1'; NGRAM_GATE='1'; NGRAM_LAYER='0';
           NGRAM_ORDER='2'; NGRAM_TRI='0'; CUDALM_TOPK='1'; CUDALM_ATTN_GATE='0'; CUDALM_UNIPOOL='0'; CUDALM_QLM='0';
           CUDALM_WARMUP='1000'; CUDALM_WSD_DECAY='0.2'; CUDALM_MASK_PROMPT='0'; CUDALM_BATCH='32';
           NGRAM_ROWS='65536'; CUDALM_D='160'; CUDALM_L='11'; CUDALM_H='5'; CUDALM_KV='1'; CUDALM_F='432' }
function Score($nm, $lp) {
    Clean; if ($lp) { $env:CUDALM_LOOP = $lp }; & $X eval $nm $V 100 4 *> "$L\$nm.eval.log"
    $vl = (Select-String "$L\$nm.eval.log" -Pattern 'val loss ([0-9.]+)').Matches[0].Groups[1].Value
    $env:CUDALM_LMEVAL_ITEMS = "$L\$nm.items.tsv"; & $X lmeval $nm all *> "$L\$nm.lm.log"; Clean
    $ix = (Select-String "$L\$nm.lm.log" -Pattern 'intelligence index ([0-9.]+)').Matches[0].Groups[1].Value
    $mg = (Select-String "$L\$nm.lm.log" -Pattern 'margin index (\S+)').Matches[0].Groups[1].Value
    Log "$nm | val $vl index $ix margin $mg" }
function Vs($a, $b) { Log "== $a -> $b"; & $X lmeval-vs $a $b 2>&1 | ? { $_ -match 'hellaswag|arc_|piqa|arithmark|margin index' } | % { Log "  $_" } }
# 1. loop arm
foreach ($seed in 9301, 9302) { $nm = "S3_loop_$seed"
    if (Test-Path "$L\$nm.done") { continue }
    Clean; foreach ($k in $BASE.Keys) { Set-Item "env:$k" $BASE[$k] }; $env:CUDALM_LOOP = '3-7x2'
    $env:CUDALM_SEED = "$seed"; $env:CUDALM_BENCH_EVERY = '0'; $env:CUDALM_KEEP_EVERY = '0'
    if (!(Test-Path "$M\$nm.bin")) { & $X new $nm m3 2 2048 1024 *> "$L\$nm.train.log" }
    $env:CUDALM_ITERS = '18000'; $t0 = Get-Date; & $X train $nm $V 1 *>> "$L\$nm.train.log"; $ex = $LASTEXITCODE; Clean
    Log "$nm train exit $ex $([int]((Get-Date)-$t0).TotalMinutes) min"; Score $nm '3-7x2'
    Vs "$L\S3_base_$seed.items.tsv" "$L\$nm.items.tsv"; "ok" | Out-File -Encoding ascii "$L\$nm.done" }
# 2. LS-Micro3-D1 with LR_MIN 0.02
$D1 = Get-ChildItem 'D:\cudalm_data' -Directory | ? { $_.Name -like 'd1_*' } | Select-Object -First 1
$ARMS = @{ vres = @{ CUDALM_VRES='1' }; gate = @{ CUDALM_ATTN_GATE='1' }; ng13 = @{ NGRAM_LAYER='1'; NGRAM_TRI='1' }; dense = @{ CUDALM_F='864'; X_EXP='0' } }
$adopt = @(Get-Content $S3 | ? { $_ -match ' (\w+): mean d_margin \S+ -> ADOPT' } | % { $Matches[1] })
$nm = 'LS-Micro3-D1lr'
if ($adopt -and $D1 -and (Test-Path "$M\LS-Micro3_it60000.bin") -and !(Test-Path "$L\$nm.done")) {
    Get-ChildItem "$M\LS-Micro3_it60000.bin*" | % { Copy-Item $_.FullName ("$M\$nm" + $_.Name.Substring('LS-Micro3_it60000'.Length)) -Force }
    Clean; foreach ($k in $BASE.Keys) { Set-Item "env:$k" $BASE[$k] }
    foreach ($a in $adopt) { foreach ($k in $ARMS[$a].Keys) { if ($k -ne 'X_EXP') { Set-Item "env:$k" $ARMS[$a][$k] } } }
    $env:CUDALM_ITERS = '87036'; $env:CUDALM_LR = '1.2e-3'; $env:CUDALM_LR_MIN = '0.02'; $env:CUDALM_SEED = '9013'
    $env:CUDALM_BENCH_EVERY = '0'; $env:CUDALM_KEEP_EVERY = '0'
    $t0 = Get-Date; & $X train $nm $D1.FullName 1 *>> "$L\$nm.train.log"; Clean
    Log "$nm [$($adopt -join ' ')] train $([int]((Get-Date)-$t0).TotalMinutes) min"; Score $nm ''
    Vs "$L\D1A-lrmin.items.tsv" "$L\$nm.items.tsv"; if (Test-Path "$L\LS-Micro3-D1.items.tsv") { Vs "$L\LS-Micro3-D1.items.tsv" "$L\$nm.items.tsv" }
    "ok" | Out-File -Encoding ascii "$L\$nm.done" }
elseif (!$adopt) { Log 'no stage-3 change adopted: LS-Micro3-D1lr skipped' }
Log 'stage 4 done'
