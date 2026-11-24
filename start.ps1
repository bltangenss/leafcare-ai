param([switch]$SkipBuild)

$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not $SkipBuild) {
        Push-Location (Join-Path $PSScriptRoot 'frontend')
        try {
            if (-not (Test-Path -LiteralPath 'node_modules')) {
                npm ci
                if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
            }
            npm run build
            if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
        } finally { Pop-Location }
    }
    if (-not (Test-Path -LiteralPath 'frontend/dist/index.html')) {
        throw 'Build the frontend first: cd frontend; npm ci; npm run build'
    }
    Write-Host 'LeafCare AI: http://127.0.0.1:5000  (Ctrl+C to stop)'
    python -m app.app
    if ($LASTEXITCODE -ne 0) { throw 'LeafCare AI stopped with an error.' }
} finally { Pop-Location }
