# Holds ES_SYSTEM_REQUIRED (no system sleep, display may still turn off) until stage 4 finishes or no cudalm process runs.
Add-Type -Name P -Namespace W -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$res = 'E:\cudalm\logs\master\stage4.res'
[W.P]::SetThreadExecutionState([uint32]2147483649) | Out-Null
"keepawake on $(Get-Date)"
while ($true) {
    Start-Sleep 300; [W.P]::SetThreadExecutionState([uint32]2147483649) | Out-Null
    $s3 = 'E:\cudalm\logs\master\stage3.res'
    $done = (Get-Content $s3 -Tail 1) -match 'stage 3 done' -and (Get-Content $res -Tail 1) -match 'stage 4 done' -and
            (Get-Item $res).LastWriteTime -gt (Get-Item $s3).LastWriteTime
    if ($done -and !(Get-Process cudalm* -EA 0)) { break } }
[W.P]::SetThreadExecutionState([uint32]2147483648) | Out-Null
"keepawake off $(Get-Date)"
