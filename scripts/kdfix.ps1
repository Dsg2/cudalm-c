# KD rerun with build 20o: teacher top-8 kept raw, its missing mass on the true token (20n renormalised the top-8).
# Arms HWR5 / HWR9 (alpha 0.5 / 0.9, teacher g3aux), 4 reps of the stage-1 proxy, gated against HWK like stage 1.
# Not part of master.ps1: run it while the GPU is free. Results: logs\master\kdfix.res
$E = 'E:\cudalm'; $L = "$E\logs\master"; $RES = "$L\kdfix.res"; Set-Location $E
function Log($s) { "$(Get-Date -Format 'MM-dd HH:mm') $s" | Out-File -Append -Encoding ascii $RES }
$env:FDOC_EXE = "$E\cudalm20o.exe"
foreach ($a in @(@('HWR5', '0.5'), @('HWR9', '0.9'))) {
    Log "kdfix: $($a[0]) alpha $($a[1])"; $env:FDOC_ENV = "CUDALM_KD_ALPHA=$($a[1])"; $env:FDOC_KD = 'g3aux'
    powershell -ExecutionPolicy Bypass -Command "& $E\fdoc.ps1 -Reps 4 -Arms $($a[0])" *>> "$L\kdfix.log" }
Remove-Item env:FDOC_ENV, env:FDOC_KD, env:FDOC_EXE -EA 0
py -3 "$E\prep\gate.py" "$E\logs\fp\doc" HWK HWK5 HWK9 HWR5 HWR9 2>&1 | % { Log "  $_" }
Log 'kdfix done'
