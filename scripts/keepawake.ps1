# Holds ES_SYSTEM_REQUIRED (no idle sleep; the display may still turn off; a manual sleep still works) until the last line
# of -Res matches -Done.   keepawake.ps1 -Res E:\cudalm\logs\xdata\xdata.res -Done 'xdata done'
param([string]$Res = 'E:\cudalm\logs\master\stage4.res', [string]$Done = 'stage 4 done')
Add-Type -Name P -Namespace W -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$t0 = Get-Date; "keepawake on $t0 until '$Done' in $Res"
while ($true) {
    [W.P]::SetThreadExecutionState([uint32]2147483649) | Out-Null; Start-Sleep 300
    if ((Test-Path $Res) -and (Get-Item $Res).LastWriteTime -gt $t0 -and (Get-Content $Res -Tail 1) -match $Done) { break } }
[W.P]::SetThreadExecutionState([uint32]2147483648) | Out-Null
"keepawake off $(Get-Date)"
