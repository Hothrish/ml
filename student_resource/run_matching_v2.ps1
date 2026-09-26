param(
    [ValidateSet('tests','train','benchmark','predict','validate-output')]
    [string]$Stage = 'tests',
    [string]$DataRoot = 'C:\student_resource\student_resource',
    [int]$Workers = 8,
    [string]$OutputRoot = '',
    [string]$ModelPath = '',
    [string]$PythonPath = ''
)
$ErrorActionPreference = 'Stop'
if (!$PythonPath) { $PythonPath = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' }
if (!(Test-Path -LiteralPath $PythonPath)) { throw 'Supply -PythonPath pointing to Python with matching_requirements_v2.txt installed.' }
if (!$OutputRoot) { $OutputRoot = Join-Path $DataRoot 'output' }
$deps = Join-Path $DataRoot '.matching_deps'
if (Test-Path -LiteralPath $deps) { $env:PYTHONPATH = $deps }
$norm = Join-Path $DataRoot 'data\normalized'
$runs = Join-Path $DataRoot 'runs'
$script = Join-Path $PSScriptRoot 'src\matching_v2.py'
$model = if ($ModelPath) { $ModelPath } else { Join-Path $PSScriptRoot 'models\model_v2.json' }
function Invoke-Python {
    & $PythonPath @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE)." }
}
if ($Stage -eq 'tests') {
    Invoke-Python -m unittest discover -s (Join-Path $PSScriptRoot 'tests') -p 'test_matching_v2.py' -v
} elseif ($Stage -eq 'train') {
    $fitRoot = Join-Path $runs 'matcher_training_v1'
    $sample = Join-Path $fitRoot 'sample'
    $candidates = Join-Path $fitRoot 'candidates'
    $fields = Join-Path $fitRoot 'fields.sqlite'
    $out = Join-Path $runs ('matcher_training_v2_' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
    $blocker = Join-Path $PSScriptRoot 'src\blocking.py'
    $baseMatcher = Join-Path $PSScriptRoot 'src\matching.py'
    $index = Join-Path $runs 'train.sqlite'
    $lsh = Join-Path $runs 'train_lsh_b8.sqlite'
    if (!(Test-Path (Join-Path $sample 'sample_manifest.json'))) {
        Invoke-Python $blocker sample --normalized-dir $norm --ground-truth (Join-Path $DataRoot 'dataset\train\train_ground_truth.tsv') --out $sample --size 3000 --seed 41 --exclude-ids (Join-Path $runs 'tune_sample\queries.tsv')
    }
    if (!(Test-Path (Join-Path $candidates 'blocking_stats.json'))) {
        Invoke-Python $blocker generate --index $index --lsh-index $lsh --queries (Join-Path $sample 'queries.tsv') --split train --out $candidates --config (Join-Path $PSScriptRoot 'config\default.json') --workers $Workers
    }
    if (!(Test-Path $fields)) {
        Invoke-Python $baseMatcher build-fields --normalized-dir $norm --split train --candidates (Join-Path $candidates 'candidate_pairs.tsv') --out $fields
    }
    Invoke-Python $script train --index $index --queries (Join-Path $sample 'queries.tsv') --ground-truth (Join-Path $sample 'ground_truth.tsv') --evidence (Join-Path $candidates 'candidate_evidence.tsv.gz') --fields $fields --out $out --workers $Workers
    Write-Host "Training complete: $out. Use its model.json explicitly after reviewing its training_report.json."
} elseif ($Stage -eq 'benchmark' -or $Stage -eq 'predict') {
    $index = Join-Path $runs 'test_match.sqlite'
    $lsh = Join-Path $runs 'test_match_lsh_b8.sqlite'
    if (!(Test-Path $index) -or !(Test-Path $lsh)) { throw 'Complete run_matching.ps1 -Stage setup-test first.' }
    if (!(Test-Path $model)) { throw 'Extract the supplied models\model_v2.json.' }
    $countries = Join-Path $runs 'test_match_countries_v2'
    if (!(Test-Path ($countries + '.json')) -or !(Test-Path ($countries + '.bin'))) {
        Invoke-Python $script build-country-cache --index $index --out $countries
    }
    $extra = @()
    if ($Stage -eq 'benchmark') {
        $OutputRoot = Join-Path $runs ('matching_v2_benchmark_' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
        $extra = @('--limit','1000')
    }
    Invoke-Python $script predict --index $index --lsh-index $lsh --normalized-dir $norm --split test --model $model --country-cache $countries --out $OutputRoot --workers $Workers @extra
    Write-Host "Results: $OutputRoot"
} else {
    Invoke-Python (Join-Path $PSScriptRoot 'src\validate_matching_v2.py') --matching (Join-Path $OutputRoot 'matching_results.tsv') --candidate (Join-Path $OutputRoot 'candidate_pairs.tsv') --test-dir (Join-Path $DataRoot 'dataset\test') --validator (Join-Path $DataRoot 'utils\validate_submission.py')
}
