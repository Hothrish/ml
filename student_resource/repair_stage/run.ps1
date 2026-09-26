param(
    [ValidateSet('doctor','tests','validate','audit','test','train')]
    [string]$Stage = 'doctor',
    [string]$DataRoot = 'C:\student_resource\student_resource',
    [string]$RunRoot = '',
    [string]$Config = 'default',
    [int]$SampleSize = 1000,
    [int]$Workers = 4,
    [string]$PythonPath = ''
)
$ErrorActionPreference = 'Stop'
$codePath = Join-Path $PSScriptRoot 'src\blocking.py'
if (!$RunRoot) { $RunRoot = Join-Path $PSScriptRoot 'runs' }
$normPath = Join-Path $DataRoot 'data\normalized'
$truthPath = Join-Path $DataRoot 'dataset\train\train_ground_truth.tsv'
$configPath = Join-Path $PSScriptRoot "config\$Config.json"
if (!(Test-Path -LiteralPath $configPath)) { throw "Configuration not found: $configPath" }
$pythonPrefix = @()
if ($PythonPath) {
    $pythonExe = $PythonPath
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonExe = (Get-Command py).Source
    $pythonPrefix = @('-3')
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExe = (Get-Command python).Source
} else {
    $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (!(Test-Path -LiteralPath $bundledPython)) { throw 'Python not found. Install Python 3.10+ or pass -PythonPath.' }
    $pythonExe = $bundledPython
}
function Invoke-Blocking {
    & $pythonExe @pythonPrefix $codePath @args
    if ($LASTEXITCODE -ne 0) { throw "Blocking command failed (exit $LASTEXITCODE). See error above." }
}
function Ensure-Index([string]$split) {
    $indexPath = Join-Path $RunRoot "$split.sqlite"
    if (!(Test-Path -LiteralPath $indexPath)) {
        Invoke-Blocking build --normalized-dir $normPath --split $split --index $indexPath | Out-Host
    } else {
        Invoke-Blocking resume --index $indexPath --normalized-dir $normPath --split $split | Out-Host
    }
    # Reused indexes are checked for completeness and source-file changes.
    Invoke-Blocking check-index --index $indexPath --normalized-dir $normPath --split $split | Out-Host
    return $indexPath
}
if ($Stage -eq 'doctor') {
    Invoke-Blocking doctor
} elseif ($Stage -eq 'tests') {
    & $pythonExe @pythonPrefix -m unittest discover -s (Join-Path $PSScriptRoot 'tests') -v
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
} elseif ($Stage -eq 'validate' -or $Stage -eq 'audit') {
    $indexPath = Ensure-Index 'train'
    $sampleName = if ($Stage -eq 'audit') { 'audit_sample' } else { 'tune_sample' }
    $samplePath = Join-Path $RunRoot $sampleName
    $sampleManifest = Join-Path $samplePath 'sample_manifest.json'
    if (!(Test-Path -LiteralPath $sampleManifest)) {
        if ($Stage -eq 'audit') {
            $exclude = Join-Path $RunRoot 'tune_sample\queries.tsv'
            if (!(Test-Path -LiteralPath $exclude)) { throw 'Run validate first to create the tuning sample.' }
            Invoke-Blocking sample --normalized-dir $normPath --ground-truth $truthPath --out $samplePath --size $SampleSize --seed 29 --exclude-ids $exclude
        } else {
            Invoke-Blocking sample --normalized-dir $normPath --ground-truth $truthPath --out $samplePath --size $SampleSize --seed 17
        }
    }
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $outPath = Join-Path $RunRoot "$Stage-$Config-$stamp"
    $queryPath = Join-Path $samplePath 'queries.tsv'
    Invoke-Blocking generate --index $indexPath --queries $queryPath --split train --out $outPath --config $configPath --workers $Workers
    Invoke-Blocking evaluate --index $indexPath --queries $queryPath --ground-truth (Join-Path $samplePath 'ground_truth.tsv') --candidates (Join-Path $outPath 'candidate_pairs.tsv') --evidence (Join-Path $outPath 'candidate_evidence.tsv.gz') --report (Join-Path $outPath 'blocking_report.json')
    Write-Host "Results: $outPath"
} else {
    $indexPath = Ensure-Index $Stage
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $outPath = Join-Path $RunRoot "$Stage-$Config-$stamp"
    Invoke-Blocking generate --index $indexPath --normalized-dir $normPath --split $Stage --out $outPath --config $configPath --workers $Workers
    Write-Host "Results: $outPath"
}
