param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string]$Dataset = "experiments/financial-evidence-qa-2026-09-28/financial_qa_dev_source_review_20260924_v3.jsonl",
    [string]$TableAudit = "experiments/financial-evidence-qa-2026-09-28/table_quality_ai_visual_diagnostic_2026-09-24_v2.jsonl",
    [string]$BuildId = "build_f74bb1dfbf96b8a2",
    [string]$RunTag = (Get-Date -Format "yyyyMMdd-HHmmss")
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$runTagPattern = '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$'
if ($RunTag -notmatch $runTagPattern) {
    throw "RunTag must contain only letters, numbers, dot, underscore, or hyphen."
}

Push-Location $repoRoot
try {
    $pythonPath = (Get-Command $PythonExe -ErrorAction Stop).Source
    foreach ($requiredPath in @($Dataset, $TableAudit, "requirements-lock-2026-09-19.txt")) {
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "Required input is missing: $requiredPath"
        }
    }

    Write-Host "[1/4] Full test suite using $pythonPath"
    & $pythonPath -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Full test suite failed with exit code $LASTEXITCODE" }

    Write-Host "[2/4] Verify immutable candidate package $BuildId"
    $verifyOutput = & $pythonPath scripts/run_isolated_staging.py --verify-build-id $BuildId
    if ($LASTEXITCODE -ne 0) { throw "Candidate package verification failed with exit code $LASTEXITCODE" }
    $verification = ($verifyOutput -join [Environment]::NewLine) | ConvertFrom-Json
    if ($verification.status -ne "PASS") { throw "Candidate package did not verify: $($verification | ConvertTo-Json -Compress -Depth 8)" }
    if ($verification.build_id -ne $BuildId) { throw "Verified package identity does not match requested BuildId $BuildId" }

    $buildDir = Join-Path $repoRoot "reports/isolated_staging/$BuildId"
    $outputDir = Join-Path $repoRoot "reports/evaluation"
    $raw = Join-Path $outputDir "financial_retrieval_raw_$RunTag.jsonl"
    $summary = Join-Path $outputDir "financial_qa_summary_$RunTag.json"
    $chart = Join-Path $outputDir "financial_qa_summary_$RunTag.png"

    Write-Host "[3/4] Run the complete frozen six-method retrieval matrix on build $BuildId"
    & $pythonPath scripts/run_financial_retrieval_matrix.py --build-dir $buildDir --dataset $Dataset --output $raw
    if ($LASTEXITCODE -ne 0) { throw "Retrieval matrix failed with exit code $LASTEXITCODE" }

    Write-Host "[4/4] Recompute summary and chart from raw outputs"
    & $pythonPath scripts/summarize_financial_candidate_run.py --raw $raw --dataset $Dataset --table-audit $TableAudit --output $summary --chart $chart
    if ($LASTEXITCODE -ne 0) { throw "Summary generation failed with exit code $LASTEXITCODE" }

    $manifest = [IO.Path]::ChangeExtension($raw, ".manifest.json")
    [pscustomobject]@{
        status = "PASS"
        build_id = $BuildId
        tests = "PASS"
        package_verification = "PASS"
        retrieval_raw = $raw
        retrieval_manifest = $manifest
        summary = $summary
        chart = $chart
        quality_metrics = "AI_PDF_SOURCE_DIAGNOSTIC_NOT_GOLD"
    } | ConvertTo-Json -Depth 5
}
finally {
    Pop-Location
}
