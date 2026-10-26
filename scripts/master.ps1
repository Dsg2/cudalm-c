# II plan master queue (reports\Tiny LM intelligence index plan.md). Restartable: every step leaves a .done marker.
# Stage 1: 500-step proxy anneals of LS-Micro2_it60000 (fdoc.ps1), 4 reps vs control HWK, gated by prep\gate.py:
#   ADOPT d >= +0.10 & t >= 3 | EXTEND to 8 reps if +0.05 <= d < +0.10 (or t < 3) | KILL d <= 0 | else HOLD
#   arms: HWM (FineMath unfiltered), HWC (code 21% -> 10%), HW+arith1/3, HW+cloze3/6, HWW+dclm12 (DCLM for 1/3 of web),
#         HWK5/HWK9 (online KD from LS2.5 = g3aux top-8, alpha 0.5 / 0.9)
# Gate 1: stop unless the adopted gains sum to >= +0.30 margin or AM-dev gains >= 2 points.
# Stage 2: the adopted recipe (D1) as the full decay of it60000 (CUDALM_ITERS=87036, LS-Micro2-HW's schedule) at seeds 9013
#   and 9014, controls LS-Micro2-HW (9013) and HWB (v3pre-hw, 9014); LR_MIN 0.02 variant; soup = avg of the seeds;
#   last-k average of the 9013 run. D1 adopted if d >= +0.20 & z >= 2.5, or >= +0.12 on both seeds.
# Results: logs\master\master.res
$E = 'E:\cudalm'; $X = "$E\cudalm20n.exe"; $M = "$E\models"; $V = "$E\prep\data\v3pre"; $FP = "$E\logs\fp"; $DOC = "$FP\doc"
$L = "$E\logs\master"; $RES = "$L\master.res"; New-Item -ItemType Directory -Force $L | Out-Null; Set-Location $E
$env:FDOC_EXE = $X
function Log($s) { "$(Get-Date -Format 'MM-dd HH:mm') $s" | Out-File -Append -Encoding ascii $RES }
function Clean { Get-ChildItem env: | ? { $_.Name -like 'CUDALM_*' -or $_.Name -like 'QLM_*' -or $_.Name -like 'NGRAM_*' } | % { Remove-Item "env:$($_.Name)" } }
function Fdoc($arms, $reps, $envs, $kd) {
    $env:FDOC_ENV = $envs; $env:FDOC_KD = $kd
    powershell -ExecutionPolicy Bypass -Command "& $E\fdoc.ps1 -Reps $reps -Arms $($arms -join ',')" *>> "$L\fdoc.log"
    Remove-Item env:FDOC_ENV, env:FDOC_KD -EA 0 }
function Gate($arms) {
    $o = py -3 "$E\prep\gate.py" $DOC HWK @arms 2>&1; $o | % { Log "  $_" }
    $d = @{}; $o | ? { $_ -match '^DECISION (\S+) (\S+) (\S+)' } | % { $d[$Matches[1]] = @($Matches[2], [double]$Matches[3]) }; $d }
if (!(Test-Path $RES)) { Log "start | build $X" }
# ---------------- Stage 1 ----------------
$data = @('HWM', 'HWC', 'HW+arith1', 'HW+arith3', 'HW+cloze3', 'HW+cloze6', 'HWW+dclm12')
if (!(Test-Path "$L\s1.done")) {
    Log 'stage 1: data arms + control'; Fdoc (@('HWK') + $data) 4 '' ''
    Log 'stage 1: KD arms'; Fdoc @('HWK5') 4 'CUDALM_KD_ALPHA=0.5' 'g3aux'; Fdoc @('HWK9') 4 'CUDALM_KD_ALPHA=0.9' 'g3aux'
    $all = $data + @('HWK5', 'HWK9'); $g = Gate $all
    $ext = @($all | ? { $g[$_] -and $g[$_][0] -eq 'EXTEND' })
    if ($ext) { Log "stage 1: extending to 8 reps: $($ext -join ' ')"
        Fdoc (@('HWK') + @($ext | ? { $_ -notlike 'HWK*' })) 8 '' ''
        if ($ext -contains 'HWK5') { Fdoc @('HWK5') 8 'CUDALM_KD_ALPHA=0.5' 'g3aux' }
        if ($ext -contains 'HWK9') { Fdoc @('HWK9') 8 'CUDALM_KD_ALPHA=0.9' 'g3aux' }
        $g = Gate $all }
    $g.Keys | % { "$_ $($g[$_][0]) $($g[$_][1])" } | Set-Content -Encoding ascii "$L\s1.decisions"
    "ok" | Out-File -Encoding ascii "$L\s1.done" }
$dec = @{}; Get-Content "$L\s1.decisions" | % { $p = $_.Split(' '); $dec[$p[0]] = @($p[1], [double]$p[2]) }
$ad = @($dec.Keys | ? { $dec[$_][0] -eq 'ADOPT' })
# D1 recipe: base from HWM/HWC, the better-scoring fraction of each extra, KD with the better alpha
$base = if ($ad -contains 'HWM' -and $ad -contains 'HWC') { 'HWMC' } elseif ($ad -contains 'HWM') { 'HWM' } elseif ($ad -contains 'HWC') { 'HWC' } else { 'HW' }
$ex = @(); $sum = 0.0; foreach ($k in @('HWM', 'HWC')) { if ($ad -contains $k) { $sum += $dec[$k][1] } }
foreach ($fam in @('arith', 'cloze', 'dclm')) {
    $best = $ad | ? { $_ -match "\+$fam\d+$" } | Sort-Object { -$dec[$_][1] } | Select-Object -First 1
    if ($best) { $ex += ($best -split '\+')[1]; $sum += $dec[$best][1]; if ($fam -eq 'dclm') { $base = $base -replace '^HW', 'HWW' } } }
if ($base -eq 'HWWMC' -or $base -eq 'HWWM' -or $base -eq 'HWWC') { $base = $base.Substring(0, 2) + $base.Substring(3); Log "note: DCLM swap + $($base): web multiplier dropped (no combined HWV entry)" }
$kdA = $ad | ? { $_ -like 'HWK?' } | Sort-Object { -$dec[$_][1] } | Select-Object -First 1
if ($kdA) { $sum += $dec[$kdA][1] }
$recipe = (@($base) + $ex) -join '+'
# AM-dev gain of the adopted arith arm (dev items, same reps)
$amdev = 0.0; $ar = $ad | ? { $_ -match '\+arith' } | Select-Object -First 1
if ($ar) { $o = py -3 "$E\prep\gate.py" $DOC HWK $ar 2>&1 | ? { $_ -match 'dev amdev: d_acc (\S+)' }; if ($o) { $amdev = [double]($o -replace '.*d_acc (\S+).*', '$1') } }
Log "gate 1: adopted [$($ad -join ' ')] | recipe $recipe | KD $kdA | summed d_margin $([math]::Round($sum,3)) | AM-dev d_acc $amdev"
if ($sum -lt 0.30 -and $amdev -lt 2) { Log 'gate 1: FAIL (gains < +0.30 margin and AM-dev < 2 points) - no stage 2; going to stage 3 (architecture)'
    powershell -ExecutionPolicy Bypass -File "$E\stage3.ps1"; exit }
# ---------------- Stage 2 ----------------
$D1 = "D:\cudalm_data\d1_$($recipe.Replace('+','_'))"
if (!(Test-Path "$D1\train.bin")) { Log "stage 2: building $D1"
    py -3 "$E\prep\d1build.py" $V "$E\logs\fdec" "$FP\extra" $D1 $recipe 1.8e9 *> "$L\d1build.log"; Get-Content "$L\d1build.log" -Tail 3 | % { Log "  $_" } }
$kdf = ''
if ($kdA) { $kdf = "F:\cudalm_kd\d1.kd"; $alpha = if ($kdA -eq 'HWK5') { '0.5' } else { '0.9' }
    if (!(Test-Path "$L\kd.done")) { Log 'stage 2: teacher cache over the D1 corpus'; Clean
        & $X kdcache g3aux $D1 $kdf 8 8 *> "$L\kdcache.log"; "ok" | Out-File -Encoding ascii "$L\kd.done" } }
function Decay($nm, $data, $seed, $extra) {
    if (Test-Path "$L\$nm.done") { return }
    if (!(Test-Path "$M\$nm.bin")) { Get-ChildItem "$M\LS-Micro2_it60000.bin*" | % { Copy-Item $_.FullName ("$M\$nm" + $_.Name.Substring('LS-Micro2_it60000'.Length)) } }
    Clean; $env:CUDALM_ITERS = '87036'; $env:CUDALM_LR = '1.2e-3'; $env:CUDALM_WARMUP = '1000'; $env:CUDALM_WSD_DECAY = '0.2'
    $env:CUDALM_SEED = "$seed"; $env:CUDALM_BENCH_EVERY = '0'; $env:CUDALM_KEEP_EVERY = '0'
    foreach ($kv in $extra) { $p = $kv.Split('=', 2); Set-Item "env:$($p[0])" $p[1] }
    # last-k: snapshots at 80k and 84k for the 9013 D1 run (CUDALM_KEEP_EVERY copies whole sets)
    $t0 = Get-Date; & $X train $nm $data 1 *>> "$L\$nm.train.log"; $ex2 = $LASTEXITCODE; Clean
    Log "$nm train exit $ex2 $([int]((Get-Date)-$t0).TotalMinutes) min"; Score $nm; "ok" | Out-File -Encoding ascii "$L\$nm.done" }
function Score($nm) {
    Clean; & $X eval $nm $V 100 4 *> "$L\$nm.eval.log"
    $vl = (Select-String "$L\$nm.eval.log" -Pattern 'val loss ([0-9.]+)').Matches[0].Groups[1].Value
    $env:CUDALM_LMEVAL_ITEMS = "$L\$nm.items.tsv"; & $X lmeval $nm all *> "$L\$nm.lm.log"
    $env:CUDALM_LMEVAL_ITEMS = "$L\$nm.dev.items.tsv"; & $X lmeval $nm amdev,clozedev *> "$L\$nm.dev.log"; Clean
    $ix = (Select-String "$L\$nm.lm.log" -Pattern 'intelligence index ([0-9.]+)').Matches[0].Groups[1].Value
    $mg = (Select-String "$L\$nm.lm.log" -Pattern 'margin index (\S+)').Matches[0].Groups[1].Value
    Log "$nm | val $vl index $ix margin $mg" }
function Vs($a, $b) { Log "== $a -> $b"; & $X lmeval-vs $a $b 2>&1 | ? { $_ -match 'hellaswag|arc_|piqa|arithmark|margin index' } | % { Log "  $_" }
    $z = & $X lmeval-vs $a $b 2>&1 | ? { $_ -match '^margin index' }; return $z }
$kdenv = if ($kdf) { @("CUDALM_KD=$kdf", "CUDALM_KD_ALPHA=$alpha") } else { @() }
$CA = "$E\logs\hwrun\LS-Micro2-HW.items.tsv"
Decay 'D1A' $D1 9013 ($kdenv + @('CUDALM_KEEP_EVERY=4000'))
Decay 'HWB' 'D:\cudalm_data\v3pre-hw' 9014 @()
Decay 'D1B' $D1 9014 $kdenv
$za = Vs $CA "$L\D1A.items.tsv"; $zb = Vs "$L\HWB.items.tsv" "$L\D1B.items.tsv"
foreach ($z in @($za, $zb)) { if ($z) { Log "  $z" } }
Decay 'D1A-lrmin' $D1 9013 ($kdenv + @('CUDALM_LR_MIN=0.02'))
Vs "$L\D1A.items.tsv" "$L\D1A-lrmin.items.tsv" | Out-Null
if (!(Test-Path "$L\soup.done")) {
    & $X avg D1-soup D1A D1B *> "$L\soup.log"; Score 'D1-soup'; Vs "$L\D1A.items.tsv" "$L\D1-soup.items.tsv" | Out-Null
    $k = @(Get-ChildItem "$M\D1A_it*.bin" | ? { $_.Name -match '^D1A_it(\d+)\.bin$' } | Sort-Object { [int]($_.Name -replace '\D', '') } | Select-Object -Last 2 | % { $_.BaseName })
    if ($k) { & $X avg D1A-last3 @k D1A *> "$L\last3.log"; Score 'D1A-last3'; Vs "$L\D1A.items.tsv" "$L\D1A-last3.items.tsv" | Out-Null }
    "ok" | Out-File -Encoding ascii "$L\soup.done" }
Log 'stage 2 done: compare D1A / D1B / soup / last3 against LS-Micro2-HW (II 8.54, margin -7.736); starting stage 3'
powershell -ExecutionPolicy Bypass -File "$E\stage3.ps1"
