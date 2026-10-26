# Doc-level v3pre selection probes with replicates. Pool of $NP random blocks scored per document by LS-Micro2 and
# LS2.5 (g3aux); arms (see prep\docsel.py) x $Reps replicates (own docs + own seed), each a 500-step
# anneal of LS-Micro2_it60000 (LR 1.2e-3 -> 1.2e-4). Results: logs\fp\fdoc.res, summary logs\fp\fdoc_sum.txt
# Dev sets (bench\amdev.tsv, bench\clozedev.tsv when present) are scored into <job>.dev.items.tsv; prep\gate.py decides.
param([int]$NP = 4000, [int]$Reps = 4, [string[]]$Arms = @('R', 'L50', 'H50', 'X50'))
$E = 'E:\cudalm'; $X = "$E\cudalm20m.exe"; $M = "$E\models"; $V = "$E\prep\data\v3pre"; $FP = "$E\logs\fp"
$DD = 'D:\cudalm_data\fdoc'; $L = "$FP\doc"; $RES = "$FP\fdoc.res"; $SRC = 'LS-Micro2_it60000'; $TOK = 32768000
if ($env:FDOC_EXE) { $X = $env:FDOC_EXE }
New-Item -ItemType Directory -Force $L, $DD | Out-Null; Set-Location $E
function Clean { Get-ChildItem env: | ? { $_.Name -like 'CUDALM_*' -or $_.Name -like 'QLM_*' -or $_.Name -like 'NGRAM_*' } | % { Remove-Item "env:$($_.Name)" } }
function CopyModel($a, $b) { Get-ChildItem "$M\$a.bin*" | % { Copy-Item $_.FullName ("$M\$b" + $_.Name.Substring($a.Length)) -Force } }
function Log($s) { $s | Out-File -Append -Encoding ascii $RES }
$DEV = (@('amdev', 'clozedev') | ? { Test-Path "$E\bench\$_.tsv" }) -join ','
function DevScore($j) {
    if (!$DEV -or (Test-Path "$L\$j.dev.items.tsv") -or !(Test-Path "$M\FD_$j.bin")) { return }
    Clean; $env:CUDALM_LMEVAL_ITEMS = "$L\$j.dev.items.tsv"; & $X lmeval "FD_$j" $DEV *> "$L\$j.dev.log"; Clean }
if (!(Test-Path $RES)) { "start $(Get-Date) | pool $NP blocks | reps $Reps | arms $($Arms -join ' ') | base $SRC" | Out-File -Encoding ascii $RES }
if (!(Test-Path "$FP\pool_spans.txt")) { py -3 "$E\prep\docsel.py" pool $FP $NP *>> "$FP\docsel.log" }
foreach ($sc in @(@('micro', 'LS-Micro2'), @('ref', 'g3aux'))) {
    $out = "$FP\docs_$($sc[0]).tsv"; if (Test-Path "$out.done") { continue }
    Clean; $env:CUDALM_EVAL_SPANS = "$FP\pool_spans.txt"; $env:CUDALM_EVAL_OUT = $out; $t0 = Get-Date
    & $X eval $sc[1] $V 1 16 *> "$FP\score_$($sc[0]).log"; Clean
    Log "scored $($sc[1]) -> docs_$($sc[0]).tsv $((Get-Content $out | Measure-Object -Line).Lines) docs $([int]((Get-Date)-$t0).TotalSeconds)s"
    "ok" | Out-File -Encoding ascii "$out.done" }
$jobs = foreach ($rp in 1..$Reps) { foreach ($a in $Arms) { "$($a)_$rp" } }
$need = $jobs | ? { !(Test-Path "$DD\$_\train.bin") } | % { $p = $_.Split('_'); $p[0]; $p[1] }
if ($need) { py -3 "$E\prep\docsel.py" build $V $FP $DD $TOK @need *>> "$FP\docsel.log" }
foreach ($j in $jobs) {
    $nm = "FD_$j"; if (Test-Path "$L\$j.done") { DevScore $j; continue }
    CopyModel $SRC $nm; Clean
    $env:CUDALM_LR = '1.2e-3'; $env:CUDALM_WARMUP = '10'; $env:CUDALM_WSD_DECAY = '0.99'; $env:CUDALM_SEED = "$(9200 + [int]$j.Split('_')[1])"
    $env:CUDALM_BENCH_EVERY = '0'; $env:CUDALM_KEEP_EVERY = '0'
    if ($env:FDOC_ENV) { foreach ($kv in $env:FDOC_ENV.Split(';')) { $p = $kv.Split('='); Set-Item "env:$($p[0])" $p[1] } }
    if ($env:FDOC_KD) {   # teacher name: top-8 cache of this job's corpus on F:, then train against it
        $kd = "F:\cudalm_kd\$j.kd"; New-Item -ItemType Directory -Force F:\cudalm_kd | Out-Null
        if (!(Test-Path $kd)) { $sv = Get-ChildItem env: | ? { $_.Name -like 'CUDALM_*' } | % { "$($_.Name)=$($_.Value)" }; Clean
            & $X kdcache $env:FDOC_KD "$DD\$j" $kd 8 8 *> "$L\$j.kd.log"
            foreach ($kv in $sv) { $p = $kv.Split('=', 2); Set-Item "env:$($p[0])" $p[1] } }
        $env:CUDALM_KD = $kd }
    $t0 = Get-Date; & $X train $nm "$DD\$j" 1 *> "$L\$j.train.log"; $ex = $LASTEXITCODE; Clean
    $it = (Select-String "$L\$j.train.log" -Pattern 'new stage: \S+ epochs of this corpus = (\d+) steps').Matches[0].Groups[1].Value
    & $X eval $nm $V 100 4 *> "$L\$j.eval.log"
    $vl = (Select-String "$L\$j.eval.log" -Pattern 'val loss ([0-9.]+)').Matches[0].Groups[1].Value
    $env:CUDALM_LMEVAL_ITEMS = "$L\$j.items.tsv"; & $X lmeval $nm all *> "$L\$j.lm.log"; Clean
    $ix = (Select-String "$L\$j.lm.log" -Pattern 'intelligence index ([0-9.]+)').Matches[0].Groups[1].Value
    $mg = (Select-String "$L\$j.lm.log" -Pattern 'margin index (\S+)').Matches[0].Groups[1].Value
    Log "$j iters $it exit $ex $([int]((Get-Date)-$t0).TotalSeconds)s | val $vl index $ix margin $mg"
    DevScore $j; "ok" | Out-File -Encoding ascii "$L\$j.done" }
py -3 "$E\prep\fdocsum.py" $RES *> "$FP\fdoc_sum.txt"; Get-Content "$FP\fdoc_sum.txt" | % { Log $_ }
Log "done $(Get-Date)"
