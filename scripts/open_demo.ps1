[CmdletBinding()]
param(
    [int]$Port = 8000,
    [switch]$Restart
)

$ErrorActionPreference = "Stop"

$launcher = Join-Path $PSScriptRoot "start_demo.ps1"
if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Missing Demo launcher: $launcher"
}

# start_demo.ps1 starts a stopped API and waits for readiness. When an API is
# already listening it exits early, so this wrapper performs a second readiness
# wait in both cases before handing the URL to the browser.
if ($Restart) {
    & $launcher -Port $Port -Restart
} else {
    & $launcher -Port $Port
}

$deadline = (Get-Date).AddSeconds(120)
$lastError = "No readiness response received."
$ready = $null
while ((Get-Date) -lt $deadline) {
    try {
        $ready = Invoke-RestMethod `
            -Uri "http://127.0.0.1:$Port/health/ready" `
            -TimeoutSec 5 `
            -ErrorAction Stop
        if ($ready.status -eq "ready") {
            break
        }
        $lastError = "Readiness status: $($ready.status)"
    } catch {
        $lastError = $_.Exception.Message
    }
    Start-Sleep -Seconds 1
}

if ($null -eq $ready -or $ready.status -ne "ready") {
    throw "Demo did not become ready within 120 seconds. $lastError"
}

$url = "http://127.0.0.1:$Port/"
Start-Process $url | Out-Null
Write-Output "Demo is ready and opened: $url"
Write-Output "Neo4j ready: $($ready.dependencies.neo4j.ready); vector count: $($ready.dependencies.vector.count)"
