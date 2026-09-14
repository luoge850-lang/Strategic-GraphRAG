[CmdletBinding()]
param(
    [int]$Port = 8000,
    [switch]$Restart
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Missing project Python runtime: $pythonPath"
}

function Get-ListeningPids {
    $pattern = "(?:127\.0\.0\.1|0\.0\.0\.0):$Port\s+\S+\s+LISTENING\s+(\d+)"
    @(
        netstat -ano | Select-String -Pattern $pattern | ForEach-Object {
            $match = [regex]::Match($_.Line, $pattern)
            if ($match.Success) { [int]$match.Groups[1].Value }
        }
    ) | Sort-Object -Unique
}

function Get-ControlledProcessIds([int]$listenerPid) {
    $projectPythonPath = [System.IO.Path]::GetFullPath($pythonPath)
    $ids = @($listenerPid)
    $current = Get-Process -Id $listenerPid -ErrorAction SilentlyContinue
    while ($null -ne $current) {
        $parent = $current.Parent
        if ($null -eq $parent) { break }
        if ($parent.ProcessName -notin @("python", "python3")) { break }
        if (-not $parent.Path -or [System.IO.Path]::GetFullPath($parent.Path) -ne $projectPythonPath) { break }
        $ids += $parent.Id
        $current = $parent
    }
    $ids | Sort-Object -Unique
}

$existingPids = @(Get-ListeningPids)
if ($existingPids.Count -gt 0) {
    if (-not $Restart) {
        Write-Output "Demo already listens on http://127.0.0.1:$Port (PID $($existingPids -join ', '))."
        Write-Output "Use -Restart only when a controlled restart is required."
        exit 0
    }

    $controlledPids = @(
        foreach ($existingPid in $existingPids) {
            Get-ControlledProcessIds $existingPid
        }
    ) | Sort-Object -Unique
    foreach ($controlledPid in $controlledPids) {
        $existingProcess = Get-Process -Id $controlledPid -ErrorAction Stop
        if ($existingProcess.ProcessName -notin @("python", "python3")) {
            throw "Refusing to stop non-Python process $controlledPid ($($existingProcess.ProcessName)) on port $Port."
        }
        Stop-Process -Id $controlledPid -Force
    }
    Start-Sleep -Seconds 1
}

$stdoutPath = Join-Path ([System.IO.Path]::GetTempPath()) "nvidia-graphrag-uvicorn-$Port.stdout.log"
$stderrPath = Join-Path ([System.IO.Path]::GetTempPath()) "nvidia-graphrag-uvicorn-$Port.stderr.log"
$process = Start-Process `
    -FilePath $pythonPath `
    -ArgumentList @("-m", "uvicorn", "strategic_graphrag.api.server:app", "--host", "127.0.0.1", "--port", "$Port") `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -PassThru `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath

$deadline = (Get-Date).AddSeconds(30)
$live = $null
do {
    Start-Sleep -Milliseconds 750
    try {
        $live = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health/live" -TimeoutSec 3
    } catch {
        $live = $null
    }
} while ($null -eq $live -and (Get-Date) -lt $deadline)

if ($null -eq $live) {
    $errorTail = if (Test-Path -LiteralPath $stderrPath) {
        (Get-Content -LiteralPath $stderrPath -Tail 20) -join [Environment]::NewLine
    } else {
        "No stderr log was created."
    }
    throw "Demo failed to become live within 30 seconds. $errorTail"
}

# Python may expose a child process as the actual listener on Windows. Report
# the listener PID because that is the process controlled by -Restart.
$listenerPids = @(Get-ListeningPids)
$ready = $null
try {
    $ready = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health/ready" -TimeoutSec 3
} catch {
    $ready = $null
}

[PSCustomObject]@{
    url = "http://127.0.0.1:$Port/"
    pid = if ($listenerPids.Count -eq 1) { $listenerPids[0] } else { $listenerPids }
    launcher_pid = $process.Id
    status = $live.status
    readiness = if ($ready) { $ready.status } else { "unknown" }
    version = $live.version
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
} | ConvertTo-Json -Compress
