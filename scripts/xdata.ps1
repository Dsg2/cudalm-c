# Extra decay sources (prep\xgen.py), filtered like HW, then proxy-probed against the D1 recipe. Results: logs\xdata\xdata.res
#   -Phase gen   : download + English/dup/bench-13-gram filter + tokenize, F:\cudalm_stage\xcand\<name> (CPU, run alongside)
#   -Phase score : waits for each gen, scores every doc with LS-Micro3, keeps the higher-loss half -> logs\fp\extra\<name>.bin,
#                  then fdoc.ps1 arms D1 vs D1+<name>10 (4 reps, 500-step anneals of LS-Micro3_it60000) and gate.py
param([string]$Phase = 'score', [string[]]$Names = @('umqa', 'umtb', 'umms', 'umcv', 'uml', 'fwedu', 'cwiki', 'cstory', 'ctext', 'orca'),
      [double]$Tok = 160e6, [int]$Pct = 10, [int]$Reps = 4)
$E = 'E:\cudalm'; $X = "$E\cudalm20o.exe"; $C = 'F:\cudalm_stage\xcand'; $K = 'F:\cudalm_stage\xsrc'; $XD = "$E\logs\fp\extra"
$L = "$E\logs\xdata"; $RES = "$L\xdata.res"; $TK = "$E\prep\data\v3pre\tokenizer.json"; $D1 = 'HWC+arith1+cloze6'
New-Item -ItemType Directory -Force $L, $C, $K | Out-Null; Set-Location $E
function Log($s) { "$(Get-Date -Format 'MM-dd HH:mm') $s" | Out-File -Append -Encoding ascii $RES }
function Clean { Get-ChildItem env: | ? { $_.Name -like 'CUDALM_*' -or $_.Name -like 'NGRAM_*' } | % { Remove-Item "env:$($_.Name)" } }
if ($Phase -eq 'gen') {
    foreach ($n in $Names) { if (Test-Path "$C\$n\gen.json") { continue }
        py -3 "$E\prep\xgen.py" gen $n $TK "$E\bench" $K $C $Tok *> "$L\$n.gen.log"
        Log "gen $($n): $(Get-Content "$L\$n.gen.log" -Tail 1)"
        Get-ChildItem "$K\$($n)_*" -EA 0 | Remove-Item -Force }   # parquet cache: re-downloadable
    exit }
foreach ($n in $Names) {
    while (!(Test-Path "$C\$n\gen.json")) { Start-Sleep 60 }
    if (!(Test-Path "$C\$n\docs.tsv.done")) {
        Clean; $env:CUDALM_EVAL_SPANS = "$C\$n\spans.txt"; $env:CUDALM_EVAL_OUT = "$C\$n\docs.tsv"; $t0 = Get-Date
        & $X eval LS-Micro3 "$C\$n" 1 16 *> "$L\$n.score.log"; Clean
        Log "scored $($n): $((Get-Content "$C\$n\docs.tsv" | Measure-Object -Line).Lines) docs $([int]((Get-Date)-$t0).TotalSeconds)s"
        "ok" | Out-File -Encoding ascii "$C\$n\docs.tsv.done" }
    if (!(Test-Path "$XD\$n.bin")) { py -3 "$E\prep\xgen.py" sel $n $C "$C\$n\docs.tsv" "$XD\$n.bin" *> "$L\$n.sel.log"; Log "sel $(Get-Content "$L\$n.sel.log" -Tail 1)" } }
$arms = @($D1) + @($Names | % { "$D1+$_$Pct" })
Log "probe: $($arms -join ' ') x $Reps reps from LS-Micro3_it60000"
powershell -ExecutionPolicy Bypass -Command "& $E\fdoc.ps1 -Reps $Reps -Arms $($arms -join ',') -Src LS-Micro3_it60000 -Tag x" *>> "$L\fdoc.log"
py -3 "$E\prep\gate.py" "$E\logs\fp\doc_x" $D1 @($arms | Select -Skip 1) 2>&1 | % { Log "  $_" }
Log 'xdata done'
