# xdata gate EXTEND: control, +orca10 (EXTEND) and +umcv10 (ADOPT, confirm) to 8 reps, then gate again. -> logs\xdata\xdata.res
$E = 'E:\cudalm'; $L = "$E\logs\xdata"; $RES = "$L\xdata.res"; $D1 = 'HWC+arith1+cloze6'; Set-Location $E
function Log($s) { "$(Get-Date -Format 'MM-dd HH:mm') $s" | Out-File -Append -Encoding ascii $RES }
$arms = @($D1, "$D1+orca10", "$D1+umcv10"); Log "extend to 8 reps: $($arms -join ' ')"
powershell -ExecutionPolicy Bypass -Command "& $E\fdoc.ps1 -Reps 8 -Arms $($arms -join ',') -Src LS-Micro3_it60000 -Tag x" *>> "$L\fdoc.log"
py -3 "$E\prep\gate.py" "$E\logs\fp\doc_x" $D1 "$D1+orca10" "$D1+umcv10" 2>&1 | % { Log "  $_" }
Log 'xext done'
