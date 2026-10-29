# Fold LS-Micro3-D1soup's n-gram table 65536 -> 32768 rows (10.26M -> 9.21M params, under the 10M cap) and score it
# against the unfolded soup. Results: logs\master\ngfold.res
param([string]$In = 'LS-Micro3-D1soup', [string]$Out = 'LS-Micro3-D1soup-F32', [int]$Rows = 32768)
$E = 'E:\cudalm'; $X = "$E\cudalm20p.exe"; $V = "$E\prep\data\v3pre"; $L = "$E\logs\master"; $RES = "$L\ngfold.res"; Set-Location $E
function Log($s) { "$(Get-Date -Format 'MM-dd HH:mm') $s" | Out-File -Append -Encoding ascii $RES }
function Clean { Get-ChildItem env: | ? { $_.Name -like 'CUDALM_*' -or $_.Name -like 'QLM_*' -or $_.Name -like 'NGRAM_*' } | % { Remove-Item "env:$($_.Name)" } }
Clean; & $X ngfold $Out $In $Rows $V 2e8 *> "$L\$Out.fold.log"; Log ((Get-Content "$L\$Out.fold.log" | Select-String '\[ngfold\]') -join ' | ')
& $X info $Out *> "$L\$Out.info.log"; Log ((Get-Content "$L\$Out.info.log" | Select-String '^\s+(ngram|total|active)') -join ' | ')
Clean; & $X eval $Out $V 100 4 *> "$L\$Out.eval.log"
$vl = (Select-String "$L\$Out.eval.log" -Pattern 'val loss ([0-9.]+)').Matches[0].Groups[1].Value
$env:CUDALM_LMEVAL_ITEMS = "$L\$Out.items.tsv"; & $X lmeval $Out all *> "$L\$Out.lm.log"; Clean
$ix = (Select-String "$L\$Out.lm.log" -Pattern 'intelligence index ([0-9.]+)').Matches[0].Groups[1].Value
$mg = (Select-String "$L\$Out.lm.log" -Pattern 'margin index (\S+)').Matches[0].Groups[1].Value
Log "$Out | val $vl index $ix margin $mg"
Log "== $In -> $Out"; & $X lmeval-vs "$L\$In.items.tsv" "$L\$Out.items.tsv" 2>&1 | ? { $_ -match 'hellaswag|arc_|piqa|arithmark|lambada|margin index' } | % { Log "  $_" }
Log 'ngfold done'
