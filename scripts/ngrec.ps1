# Recovery after ngfold: a short low-LR run of the folded model on the D1 corpus (fresh optimizer, LR -> LR/10), scored vs the
# unfolded and the raw-folded model. Results: logs\master\ngfold.res
param([string]$In = 'LS-Micro3-D1soup-F32', [string]$Ref = 'LS-Micro3-D1soup', [int]$Iters = 3000, [string]$Lr = '1.2e-4')
$E = 'E:\cudalm'; $X = "$E\cudalm20p.exe"; $M = "$E\models"; $V = "$E\prep\data\v3pre"; $L = "$E\logs\master"; $RES = "$L\ngfold.res"; Set-Location $E
function Log($s) { "$(Get-Date -Format 'MM-dd HH:mm') $s" | Out-File -Append -Encoding ascii $RES }
function Clean { Get-ChildItem env: | ? { $_.Name -like 'CUDALM_*' -or $_.Name -like 'QLM_*' -or $_.Name -like 'NGRAM_*' } | % { Remove-Item "env:$($_.Name)" } }
$D1 = (Get-ChildItem 'D:\cudalm_data' -Directory | ? { $_.Name -like 'd1_*' } | Select-Object -First 1).FullName
$nm = "$In-R$($Iters / 1000)k"
Get-ChildItem "$M\$In.bin*" | % { Copy-Item $_.FullName ("$M\$nm" + $_.Name.Substring($In.Length)) -Force }
Clean; $env:CUDALM_ITERS = "$Iters"; $env:CUDALM_LR = $Lr; $env:CUDALM_WARMUP = '100'; $env:CUDALM_WSD_DECAY = '0.6'; $env:CUDALM_SEED = '9014'
$env:CUDALM_BENCH_EVERY = '0'; $env:CUDALM_KEEP_EVERY = '0'
$t0 = Get-Date; & $X train $nm $D1 1 *> "$L\$nm.train.log"; $ex = $LASTEXITCODE; Clean
Log "$nm train exit $ex $([int]((Get-Date)-$t0).TotalMinutes) min ($Iters iters, LR $Lr, D1)"
& $X eval $nm $V 100 4 *> "$L\$nm.eval.log"
$vl = (Select-String "$L\$nm.eval.log" -Pattern 'val loss ([0-9.]+)').Matches[0].Groups[1].Value
$env:CUDALM_LMEVAL_ITEMS = "$L\$nm.items.tsv"; & $X lmeval $nm all *> "$L\$nm.lm.log"; Clean
$ix = (Select-String "$L\$nm.lm.log" -Pattern 'intelligence index ([0-9.]+)').Matches[0].Groups[1].Value
$mg = (Select-String "$L\$nm.lm.log" -Pattern 'margin index (\S+)').Matches[0].Groups[1].Value
Log "$nm | val $vl index $ix margin $mg"
foreach ($b in $Ref, $In) { Log "== $b -> $nm"; & $X lmeval-vs "$L\$b.items.tsv" "$L\$nm.items.tsv" 2>&1 | ? { $_ -match 'hellaswag|arc_|piqa|arithmark|margin index' } | % { Log "  $_" } }
Log 'ngrec done'
