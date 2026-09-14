[CmdletBinding()]
param(
    [switch]$Apply
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Missing project Python runtime: $pythonPath"
}

$stamp = Get-Date -Format "yyyy-MM-dd"
[string[]]$modeArgs = @()
if ($Apply) { $modeArgs = @("--apply") }
$mode = if ($Apply) { "apply" } else { "dry-run" }
$steps = @(
    @("build_temporal_disclosure_index.py", "temporal_disclosure_index_post_rebuild_$stamp.json"),
    @("migrate_financial_observations.py", "financial_observations_post_rebuild_$stamp.json"),
    @("build_temporal_change_model.py", "temporal_change_post_rebuild_$stamp.json")
)

foreach ($step in $steps) {
    $scriptPath = Join-Path $PSScriptRoot $step[0]
    $outputPath = Join-Path $projectRoot "reports\$($step[1])"
    Write-Output "Running $($step[0]) ($mode)"
    & $pythonPath $scriptPath @modeArgs --output $outputPath
    if ($LASTEXITCODE -ne 0) {
        throw "$($step[0]) failed with exit code $LASTEXITCODE"
    }
}

if (-not $Apply) {
    Write-Output "Dry-run complete. Review the three reports, then rerun with -Apply to replace only derived temporal artifacts."
}
