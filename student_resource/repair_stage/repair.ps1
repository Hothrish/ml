param(
    [string]$ProjectRoot = 'C:\student_resource\student_resource',
    [switch]$SkipValidate
)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath $ProjectRoot).Path
$normalized = Join-Path $project 'data\normalized'
$rawTest = Join-Path $project 'dataset\test\test_source3.tsv'
foreach ($path in @($normalized,$rawTest,(Join-Path $project 'src\script_classifier.py'))) {
    if (!(Test-Path -LiteralPath $path)) { throw "Expected project input is missing: $path" }
}
foreach ($stem in @('train_source1','train_source2','train_source3','test_source1','test_source2','test_source3')) {
    foreach ($i in 0..5) {
        $name = '{0}.part-{1:00}.tsv.gz' -f $stem,$i
        if (!(Test-Path -LiteralPath (Join-Path $normalized $name))) { throw "Missing fresh part: $name" }
    }
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
foreach ($mapping in @(
    @('normalization.py','src\normalization.py'),
    @('blocking.py','src\blocking.py'),
    @('run.ps1','run.ps1')
)) {
    $source = Join-Path $PSScriptRoot $mapping[0]
    $destination = Join-Path $project $mapping[1]
    if (!(Test-Path -LiteralPath $source)) { throw "Missing repair package file: $source" }
    if (Test-Path -LiteralPath $destination) {
        Copy-Item -LiteralPath $destination -Destination "$destination.before-repair-$stamp" -ErrorAction Stop
    }
    Copy-Item -LiteralPath $source -Destination $destination -Force -ErrorAction Stop
    Write-Host "Installed $($mapping[1])"
}
Push-Location -LiteralPath $project
try {
    Write-Host 'Regenerating only test_source3 with the corrected byte-boundary handling...'
    & py -3 src\normalization.py --workers 6 --out data\normalized --in dataset\test\test_source3.tsv
    if ($LASTEXITCODE -ne 0) { throw 'test_source3 normalization failed; no cleanup was attempted.' }
    Write-Host 'Checking all six normalized sources and removing verified stale tail parts...'
    & py -3 (Join-Path $PSScriptRoot 'verify_parts.py') $normalized --remove-stale
    if ($LASTEXITCODE -ne 0) { throw 'Normalized row verification failed; blocking was not resumed.' }
    if (!$SkipValidate) {
        Write-Host 'Resuming the interrupted training index and validation...'
        & powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage validate -DataRoot $project
        if ($LASTEXITCODE -ne 0) { throw 'Blocking validation failed. See the error above.' }
    }
} finally {
    Pop-Location
}
