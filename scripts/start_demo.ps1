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

$existingPids = @(Get-ListeningPids)
if ($existingPids.Count -gt 0) {
    if (-not $Restart) {
        Write-Output "Demo already listens on http://127.0.0.1:$Port (PID $($existingPids -join ', '))."
        Write-Output "Use -Restart only when a controlled restart is required."
        exit 0
    }

    foreach ($existingPid in $existingPids) {
        $existingProcess = Get-Process -Id $existingPid -ErrorAction Stop
        if ($existingProcess.ProcessName -notin @("python", "python3")) {
            throw "Refusing to stop non-Python process $existingPid ($($existingProcess.ProcessName)) on port $Port."
        }
        Stop-Process -Id $existingPid -Force
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

[PSCustomObject]@{
    url = "http://127.0.0.1:$Port/"
    pid = $process.Id
    status = $live.status
    version = $live.version
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
} | ConvertTo-Json -Compress
