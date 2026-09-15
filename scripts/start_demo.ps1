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

function Get-ReadinessDependencyError($payload) {
    $parts = @()
    if ($null -ne $payload -and $null -ne $payload.dependencies) {
        foreach ($dependency in $payload.dependencies.PSObject.Properties) {
            $value = $dependency.Value
            if (-not [bool]$value.ready) {
                $detail = $value.error
                if (-not $detail) { $detail = $value.status }
                if (-not $detail) { $detail = "not ready" }
                $parts += "$($dependency.Name): $detail"
            }
        }
    }
    if ($parts.Count -eq 0) {
        if ($null -ne $payload -and $payload.status) {
            return "readiness status: $($payload.status)"
        }
        return "No readiness payload received."
    }
    return ($parts -join "; ")
}

function Get-LogTail([string]$path) {
    if (Test-Path -LiteralPath $path) {
        return (Get-Content -LiteralPath $path -Tail 30) -join [Environment]::NewLine
    }
    return "No log was created."
}

# /health/ready may return 503 while the cloud Neo4j dependency is waking up.
# Keep retrying the dependency-aware probe within a bounded startup window.
$readinessTimeoutSeconds = 120
$readinessRequestTimeoutSeconds = 5
$readinessPollMilliseconds = 1000
$deadline = (Get-Date).AddSeconds($readinessTimeoutSeconds)
$ready = $null
$lastDependencyError = "No readiness response received."
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds $readinessPollMilliseconds
    try {
        $candidate = Invoke-RestMethod `
            -Uri "http://127.0.0.1:$Port/health/ready" `
            -TimeoutSec $readinessRequestTimeoutSeconds `
            -ErrorAction Stop
        $ready = $candidate
        if ($ready.status -eq "ready") { break }
        $lastDependencyError = Get-ReadinessDependencyError $ready
    } catch {
        $lastDependencyError = $_.Exception.Message
        $errorBody = $_.ErrorDetails.Message
        if (-not $errorBody -and $null -ne $_.Exception.Response) {
            try {
                $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
                $errorBody = $reader.ReadToEnd()
                $reader.Dispose()
            } catch {
                # Keep the transport error when the response body is unavailable.
            }
        }
        if ($errorBody) {
            try {
                $errorPayload = $errorBody | ConvertFrom-Json
                $lastDependencyError = Get-ReadinessDependencyError $errorPayload
            } catch {
                # Keep the transport error when the response is not JSON.
            }
        }
    }
}

if ($null -eq $ready -or $ready.status -ne "ready") {
    $stderrTail = Get-LogTail $stderrPath
    $stdoutTail = Get-LogTail $stdoutPath
    throw @"
Demo failed to become ready within $readinessTimeoutSeconds seconds.
Last dependency error: $lastDependencyError
Stderr log tail:
$stderrTail
Stdout log tail:
$stdoutTail
"@
}

# Python may expose a child process as the actual listener on Windows. Report
# the listener PID because that is the process controlled by -Restart.
$listenerPids = @(Get-ListeningPids)
[PSCustomObject]@{
    url = "http://127.0.0.1:$Port/"
    pid = if ($listenerPids.Count -eq 1) { $listenerPids[0] } else { $listenerPids }
    launcher_pid = $process.Id
    status = $ready.status
    readiness = $ready.status
    dependencies = $ready.dependencies
    version = $ready.version
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
} | ConvertTo-Json -Compress
