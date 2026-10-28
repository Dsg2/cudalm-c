# LS-Micro3-D1soup = mean of LS-Micro3-D1 and LS-Micro3-D1lr (both decayed from LS-Micro3_it60000). Results: logs\master\soup3.res
$E = 'E:\cudalm'; $X = "$E\cudalm20o.exe"; $V = "$E\prep\data\v3pre"; $L = "$E\logs\master"; $RES = "$L\soup3.res"; $nm = 'LS-Micro3-D1soup'
Set-Location $E
function Log($s) { "$(Get-Date -Format 'MM-dd HH:mm') $s" | Out-File -Append -Encoding ascii $RES }
Get-ChildItem env: | ? { $_.Name -like 'CUDALM_*' -or $_.Name -like 'NGRAM_*' } | % { Remove-Item "env:$($_.Name)" }
& $X avg $nm LS-Micro3-D1 LS-Micro3-D1lr *> "$L\$nm.avg.log"; Log (Get-Content "$L\$nm.avg.log" | Select-String '\[avg\]' | Select -Last 1)
& $X eval $nm $V 100 4 *> "$L\$nm.eval.log"
$vl = (Select-String "$L\$nm.eval.log" -Pattern 'val loss ([0-9.]+)').Matches[0].Groups[1].Value
$env:CUDALM_LMEVAL_ITEMS = "$L\$nm.items.tsv"; & $X lmeval $nm all *> "$L\$nm.lm.log"; Remove-Item env:CUDALM_LMEVAL_ITEMS
$ix = (Select-String "$L\$nm.lm.log" -Pattern 'intelligence index ([0-9.]+)').Matches[0].Groups[1].Value
$mg = (Select-String "$L\$nm.lm.log" -Pattern 'margin index (\S+)').Matches[0].Groups[1].Value
Log "$nm | val $vl index $ix margin $mg"
foreach ($b in 'LS-Micro3-D1', 'LS-Micro3-D1lr') { Log "== $b -> $nm"
    & $X lmeval-vs "$L\$b.items.tsv" "$L\$nm.items.tsv" 2>&1 | ? { $_ -match 'hellaswag|arc_|piqa|arithmark|margin index' } | % { Log "  $_" } }
Log 'soup done'
