param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

& $Python -m compileall -q (Join-Path $ProjectRoot "src") (Join-Path $ProjectRoot "tests")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -m unittest discover -s (Join-Path $ProjectRoot "tests") -v
exit $LASTEXITCODE
