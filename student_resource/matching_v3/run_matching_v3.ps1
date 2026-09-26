param(
    [ValidateSet('tests','benchmark','predict','validate-output')][string]$Stage = 'tests',
    [string]$DataRoot = 'C:\student_resource\student_resource',
    [int]$Workers = 8,
    [string]$OutputRoot = '',
    [string]$PythonPath = '',
    [string]$DependencyPath = ''
)
$ErrorActionPreference = 'Stop'
if (!$PythonPath) { $PythonPath = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' }
if (!(Test-Path -LiteralPath $PythonPath)) { throw 'Supply -PythonPath for Python with matching_requirements_v3.txt installed.' }
if (!$DependencyPath) { $DependencyPath = Join-Path $DataRoot '.matching_deps' }
if (Test-Path -LiteralPath $DependencyPath) { $env:PYTHONPATH = $DependencyPath }
if (!$OutputRoot) { $OutputRoot = Join-Path $DataRoot 'output_v3' }
$runs = Join-Path $DataRoot 'runs'
$norm = Join-Path $DataRoot 'data\normalized'
$model = Join-Path $PSScriptRoot 'models\model_v3.json'
$matcher = Join-Path $PSScriptRoot 'src\matching_v3.py'
function Invoke-Python {
    & $PythonPath @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE)." }
}
if ($Stage -eq 'tests') {
    Invoke-Python -m unittest discover -s (Join-Path $PSScriptRoot 'tests') -p 'test_matching_v3.py' -v
} elseif ($Stage -eq 'benchmark' -or $Stage -eq 'predict') {
    $compact = Join-Path $runs 'test_compact_v3'
    $records = Join-Path $runs 'test_records_v3'
    $countries = Join-Path $runs 'test_countries_v3'
    foreach ($required in @($model,($compact+'.json'),($records+'.json'),($countries+'.json'))) {
        if (!(Test-Path -LiteralPath $required)) { throw "Missing file: $required. Run start_matching_v3.ps1 to install the supplied caches." }
    }
    $extra = @()
    if ($Stage -eq 'benchmark') {
        $OutputRoot = Join-Path $runs ('matching_v3_benchmark_' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
        $extra = @('--limit','5000')
    }
    Invoke-Python $matcher predict --index (Join-Path $runs 'test_match.sqlite') --lsh-index (Join-Path $runs 'test_match_lsh_b8.sqlite') --normalized-dir $norm --split test --model $model --compact-index $compact --compact-records $records --country-cache $countries --out $OutputRoot --workers $Workers @extra
    Write-Host "Results: $OutputRoot"
} else {
    Invoke-Python (Join-Path $PSScriptRoot 'src\validate_matching_v3.py') --matching (Join-Path $OutputRoot 'matching_results.tsv') --candidate (Join-Path $OutputRoot 'candidate_pairs.tsv') --test-dir (Join-Path $DataRoot 'dataset\test') --validator (Join-Path $DataRoot 'utils\validate_submission.py')
}
