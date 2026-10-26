# II plan stage 3: architecture. LS-Micro2's config from scratch for 18k steps on v3pre (CUDALM_ITERS=18000, WSD 0.2:
# hold then decay, 1.18B tokens), seeds 9301/9302, one change per arm vs the plain control:
#   vres  CUDALM_VRES=1 (value residual)       gate  CUDALM_ATTN_GATE=1 (sigmoid output gate, +L*D*D)
#   ng13  NGRAM_LAYER=1 NGRAM_TRI=1 (bigram+trigram at layer 1)    dense  1 dense FFN of F=864 instead of 2 experts x 432
# Adopt: mean paired d_margin over both seeds >= +0.10 and both seeds > 0. Then ONE full pretrain (87,036 steps, v3pre)
# with every adopted change (LS-Micro3), and its it60000 decayed on the stage-2 D1 corpus like D1A (LS-Micro3-D1).
# Results: logs\master\stage3.res
$E = 'E:\cudalm'; $X = "$E\cudalm20n.exe"; $M = "$E\models"; $V = "$E\prep\data\v3pre"; $L = "$E\logs\master"; $RES = "$L\stage3.res"
Set-Location $E
function Log($s) { "$(Get-Date -Format 'MM-dd HH:mm') $s" | Out-File -Append -Encoding ascii $RES }
function Clean { Get-ChildItem env: | ? { $_.Name -like 'CUDALM_*' -or $_.Name -like 'QLM_*' -or $_.Name -like 'NGRAM_*' } | % { Remove-Item "env:$($_.Name)" } }
$BASE = @{ CUDALM_NGRAM='1'; CUDALM_ENGRAM='0'; NGRAM_DIM='32'; NGRAM_Q4='1'; NGRAM_Q4_FOLD='1'; NGRAM_GATE='1'; NGRAM_LAYER='0';
           NGRAM_ORDER='2'; NGRAM_TRI='0'; CUDALM_TOPK='1'; CUDALM_ATTN_GATE='0'; CUDALM_UNIPOOL='0'; CUDALM_QLM='0';
           CUDALM_WARMUP='1000'; CUDALM_WSD_DECAY='0.2'; CUDALM_MASK_PROMPT='0'; CUDALM_BATCH='32';
           NGRAM_ROWS='65536'; CUDALM_D='160'; CUDALM_L='11'; CUDALM_H='5'; CUDALM_KV='1'; CUDALM_F='432' }
$ARMS = [ordered]@{ base = @{}; vres = @{ CUDALM_VRES='1' }; gate = @{ CUDALM_ATTN_GATE='1' }; ng13 = @{ NGRAM_LAYER='1'; NGRAM_TRI='1' }; dense = @{ CUDALM_F='864'; X_EXP='0' } }
function Score($nm) {
    Clean; & $X eval $nm $V 100 4 *> "$L\$nm.eval.log"
    $vl = (Select-String "$L\$nm.eval.log" -Pattern 'val loss ([0-9.]+)').Matches[0].Groups[1].Value
    $env:CUDALM_LMEVAL_ITEMS = "$L\$nm.items.tsv"; & $X lmeval $nm all *> "$L\$nm.lm.log"; Clean
    $ix = (Select-String "$L\$nm.lm.log" -Pattern 'intelligence index ([0-9.]+)').Matches[0].Groups[1].Value
    $mg = (Select-String "$L\$nm.lm.log" -Pattern 'margin index (\S+)').Matches[0].Groups[1].Value
    Log "$nm | val $vl index $ix margin $mg" }
function Run($nm, $cfg, $iters, $data, $seed) {
    if (Test-Path "$L\$nm.done") { return }
    Clean; foreach ($k in $BASE.Keys) { Set-Item "env:$k" $BASE[$k] }; $ne = '2'
    foreach ($k in $cfg.Keys) { if ($k -eq 'X_EXP') { $ne = $cfg[$k] } else { Set-Item "env:$k" $cfg[$k] } }
    $env:CUDALM_SEED = "$seed"; $env:CUDALM_BENCH_EVERY = '0'
    if (!(Test-Path "$M\$nm.bin")) { & $X new $nm m3 $ne 2048 1024 *> "$L\$nm.train.log" }
    $env:CUDALM_ITERS = "$iters"; $t0 = Get-Date; & $X train $nm $data 1 *>> "$L\$nm.train.log"; $ex = $LASTEXITCODE; Clean
    Log "$nm train exit $ex $([int]((Get-Date)-$t0).TotalMinutes) min"; Score $nm; "ok" | Out-File -Encoding ascii "$L\$nm.done" }
if (!(Test-Path $RES)) { Log 'start stage 3' }
foreach ($seed in 9301, 9302) { foreach ($a in $ARMS.Keys) { $env:CUDALM_KEEP_EVERY = '0'; Run "S3_$($a)_$seed" $ARMS[$a] 18000 $V $seed } }
$adopt = @()
foreach ($a in @($ARMS.Keys | ? { $_ -ne 'base' })) {
    $d = foreach ($seed in 9301, 9302) {
        $o = & $X lmeval-vs "$L\S3_base_$seed.items.tsv" "$L\S3_$($a)_$seed.items.tsv" 2>&1 | ? { $_ -match '^margin index' }
        Log "  $a seed $seed : $o"; [double](($o -split '\s+')[2]) }
    $mean = ($d | Measure-Object -Average).Average; $ok = $mean -ge 0.10 -and ($d | ? { $_ -le 0 }).Count -eq 0
    Log "$($a): mean d_margin $([math]::Round($mean,3)) -> $(if ($ok) { 'ADOPT' } else { 'reject' })"; if ($ok) { $adopt += $a } }
if (!$adopt) { Log 'stage 3: no architecture change adopted - done'; exit }
$cfg = @{}; foreach ($a in $adopt) { foreach ($k in $ARMS[$a].Keys) { $cfg[$k] = $ARMS[$a][$k] } }
Log "stage 3: full pretrain LS-Micro3 with [$($adopt -join ' ')]"
$env:CUDALM_KEEP_EVERY = '20000'; Run 'LS-Micro3' $cfg 87036 $V 9013
$D1 = Get-ChildItem 'D:\cudalm_data' -Directory | ? { $_.Name -like 'd1_*' } | Select-Object -First 1
if ($D1 -and (Test-Path "$M\LS-Micro3_it60000.bin") -and !(Test-Path "$L\LS-Micro3-D1.done")) {
    Get-ChildItem "$M\LS-Micro3_it60000.bin*" | % { Copy-Item $_.FullName ("$M\LS-Micro3-D1" + $_.Name.Substring('LS-Micro3_it60000'.Length)) }
    Clean; foreach ($k in $BASE.Keys) { Set-Item "env:$k" $BASE[$k] }; foreach ($k in $cfg.Keys) { if ($k -ne 'X_EXP') { Set-Item "env:$k" $cfg[$k] } }
    $env:CUDALM_ITERS = '87036'; $env:CUDALM_LR = '1.2e-3'; $env:CUDALM_SEED = '9013'; $env:CUDALM_BENCH_EVERY = '0'; $env:CUDALM_KEEP_EVERY = '0'
    $t0 = Get-Date; & $X train 'LS-Micro3-D1' $D1.FullName 1 *>> "$L\LS-Micro3-D1.train.log"; Clean
    Log "LS-Micro3-D1 train $([int]((Get-Date)-$t0).TotalMinutes) min"; Score 'LS-Micro3-D1'; "ok" | Out-File -Encoding ascii "$L\LS-Micro3-D1.done"
    & $X lmeval-vs "$E\logs\hwrun\LS-Micro2-HW.items.tsv" "$L\LS-Micro3-D1.items.tsv" 2>&1 | ? { $_ -match 'hellaswag|arc_|piqa|arithmark|margin index' } | % { Log "  $_" } }
Log 'stage 3 done'
