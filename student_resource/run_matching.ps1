param(
    [ValidateSet('tests','train','setup-test','benchmark','predict','validate-output')]
    [string]$Stage = 'tests',
    [string]$DataRoot = 'C:\student_resource\student_resource',
    [string]$RunRoot = '',
    [string]$OutputRoot = '',
    [int]$Workers = 4,
    [string]$PythonPath = ''
)
$ErrorActionPreference = 'Stop'
if (!$RunRoot) { $RunRoot = Join-Path $DataRoot 'runs' }
if (!$OutputRoot) { $OutputRoot = Join-Path $DataRoot 'output' }
$normPath = Join-Path $DataRoot 'data\normalized'
$matchScript = Join-Path $PSScriptRoot 'src\matching.py'
$blockScript = Join-Path $PSScriptRoot 'src\blocking.py'
$modelPath = Join-Path $PSScriptRoot 'models\model.json'
$indexPath = Join-Path $RunRoot 'test_match.sqlite'
$lshPath = Join-Path $RunRoot 'test_match_lsh_b8.sqlite'
$pythonPrefix = @()
$bundled = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if ($PythonPath) { $pythonExe = $PythonPath }
elseif (Test-Path -LiteralPath $bundled) { $pythonExe = $bundled }
elseif (Get-Command py -ErrorAction SilentlyContinue) { $pythonExe = (Get-Command py).Source; $pythonPrefix = @('-3') }
elseif (Get-Command python -ErrorAction SilentlyContinue) { $pythonExe = (Get-Command python).Source }
else { throw 'Install Python and the packages in matching_requirements.txt, or supply -PythonPath.' }
function Invoke-Matcher {
    & $pythonExe @pythonPrefix $matchScript @args
    if ($LASTEXITCODE -ne 0) { throw "Matcher failed (exit $LASTEXITCODE)." }
}
function Invoke-Blocker {
    & $pythonExe @pythonPrefix $blockScript @args
    if ($LASTEXITCODE -ne 0) { throw "Blocker failed (exit $LASTEXITCODE)." }
}
if ($Stage -eq 'tests') {
    & $pythonExe @pythonPrefix -m unittest discover -s (Join-Path $PSScriptRoot 'tests') -p 'test_matching.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Matcher tests failed.' }
} elseif ($Stage -eq 'train') {
    $fitRoot = Join-Path $RunRoot 'matcher_training_v1'
    $samplePath = Join-Path $fitRoot 'sample'
    $candidatePath = Join-Path $fitRoot 'candidates'
    $fieldPath = Join-Path $fitRoot 'fields.sqlite'
    $trainedPath = Join-Path $fitRoot 'model'
    $trainIndex = Join-Path $RunRoot 'train.sqlite'
    $trainLsh = Join-Path $RunRoot 'train_lsh_b8.sqlite'
    if (!(Test-Path (Join-Path $samplePath 'sample_manifest.json'))) {
        Invoke-Blocker sample --normalized-dir $normPath --ground-truth (Join-Path $DataRoot 'dataset\train\train_ground_truth.tsv') --out $samplePath --size 3000 --seed 41 --exclude-ids (Join-Path $RunRoot 'tune_sample\queries.tsv')
    }
    if (!(Test-Path (Join-Path $candidatePath 'blocking_stats.json'))) {
        Invoke-Blocker generate --index $trainIndex --lsh-index $trainLsh --queries (Join-Path $samplePath 'queries.tsv') --split train --out $candidatePath --config (Join-Path $PSScriptRoot 'config\default.json') --workers $Workers
    }
    if (!(Test-Path $fieldPath)) {
        Invoke-Matcher build-fields --normalized-dir $normPath --split train --candidates (Join-Path $candidatePath 'candidate_pairs.tsv') --out $fieldPath
    }
    if (!(Test-Path (Join-Path $trainedPath 'model.json'))) {
        Invoke-Matcher train --index $trainIndex --queries (Join-Path $samplePath 'queries.tsv') --ground-truth (Join-Path $samplePath 'ground_truth.tsv') --evidence (Join-Path $candidatePath 'candidate_evidence.tsv.gz') --fields $fieldPath --out $trainedPath
    }
    New-Item -ItemType Directory -Path (Split-Path $modelPath) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $trainedPath 'model.json') -Destination $modelPath -Force
    Copy-Item -LiteralPath (Join-Path $trainedPath 'training_report.json') -Destination (Join-Path (Split-Path $modelPath) 'training_report.json') -Force
    Write-Host "Model: $modelPath"
} elseif ($Stage -eq 'setup-test') {
    Invoke-Matcher build-records --normalized-dir $normPath --split test --index $indexPath
    if (!(Test-Path -LiteralPath $lshPath)) {
        Invoke-Blocker build-lsh --normalized-dir $normPath --split test --index $indexPath --lsh-index $lshPath --bands 8
    }
    Invoke-Blocker check-lsh --index $indexPath --lsh-index $lshPath
    Write-Host 'Test indexes ready. Run -Stage benchmark next.'
} elseif ($Stage -eq 'benchmark' -or $Stage -eq 'predict') {
    if (!(Test-Path $indexPath) -or !(Test-Path $lshPath)) { throw 'Run -Stage setup-test first.' }
    if (!(Test-Path $modelPath)) { throw 'Trained model is missing. Extract the supplied models folder or run -Stage train.' }
    if ($Stage -eq 'benchmark') {
        $benchmarkPath = Join-Path $RunRoot ('matching_benchmark_' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
        Invoke-Matcher predict --index $indexPath --lsh-index $lshPath --normalized-dir $normPath --split test --model $modelPath --out $benchmarkPath --limit 1000 --workers $Workers
        Write-Host "Benchmark only (not a submission): $benchmarkPath"
    } else {
        Invoke-Matcher predict --index $indexPath --lsh-index $lshPath --normalized-dir $normPath --split test --model $modelPath --out $OutputRoot --workers $Workers
        Write-Host "Full prediction files: $OutputRoot"
    }
} else {
    $validatorPath = Join-Path $DataRoot 'utils\validate_submission.py'
    & $pythonExe @pythonPrefix $validatorPath --matching (Join-Path $OutputRoot 'matching_results.tsv') --candidate (Join-Path $OutputRoot 'candidate_pairs.tsv') --test-dir (Join-Path $DataRoot 'dataset\test')
    if ($LASTEXITCODE -ne 0) { throw 'Submission validation failed.' }
}
