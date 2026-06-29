$ErrorActionPreference = "Stop"

# Resolve absolute paths
$BaseDir = Resolve-Path ".."
$ArtifactsDir = Join-Path $BaseDir "artifacts"

Write-Host "Creating robust directory structure..."

if (-not (Test-Path $ArtifactsDir)) {
    New-Item -ItemType Directory -Force -Path $ArtifactsDir | Out-Null
    Write-Host "[OK] Created artifacts directory."
} else {
    Write-Host "[OK] Artifacts directory already exists."
}

# Verify candidates.jsonl
$CandidatesPath = Join-Path $BaseDir "candidates.jsonl"
if (-not (Test-Path $CandidatesPath)) {
    Write-Host "[WARN] candidates.jsonl not found in parent directory!" -ForegroundColor Yellow
} else {
    Write-Host "[OK] candidates.jsonl located."
}

Write-Host "Directory setup complete."
