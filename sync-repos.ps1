# ============================================================================
#  sync-repos.ps1  -  push HIL-Testbed to GitLab (uni-oldenburg) + GitHub
#  Excludes __pycache__, .vscode, and the heavy outputs/runs folders.
#  No LFS needed (all >100 MB files live in the excluded folders).
# ============================================================================

$ErrorActionPreference = 'Stop'

# --- Config -----------------------------------------------------------------
$ProjectRoot = 'D:\My Files\Personal Projects\HIL-Testbed'
$GitLabUrl   = 'https://gitlab.uni-oldenburg.de/sebi0767/hil-tesbed.git'
$GitHubUrl   = 'https://github.com/KKhan02/HIL-Testbed.git'
$Branch      = 'main'
$CommitMsg   = 'Initial sync (code, docs, configs; run outputs excluded)'

# --- 0. Sanity: right place? ------------------------------------------------
if (-not (Test-Path $ProjectRoot)) { throw "Project root not found: $ProjectRoot" }
Set-Location $ProjectRoot
Write-Host "==> Working in: $ProjectRoot" -ForegroundColor Cyan

# --- 1. Write .gitignore (array -> lines, ASCII, no here-string) ------------
$gitignoreLines = @(
    '# Editor / cache'
    '__pycache__/'
    '.vscode/'
    ''
    '# Heavy, regenerable run artifacts (excluded)'
    '/outputs/'
    '/runs/'
    '/outputs (RPi)/'
)
Set-Content -Path (Join-Path $ProjectRoot '.gitignore') -Value $gitignoreLines -Encoding ASCII
Write-Host "==> Wrote .gitignore" -ForegroundColor Green

# --- 2. Abort if any >100 MB file exists OUTSIDE the excluded folders -------
Write-Host "==> Scanning for oversized files outside excluded folders..." -ForegroundColor Cyan
$big = Get-ChildItem -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Length -gt 100MB -and $_.FullName -notmatch '\\(outputs|runs|outputs \(RPi\))\\' }
if ($big) {
    Write-Host "ABORTING - these files are >100 MB and would be rejected by both hosts:" -ForegroundColor Red
    $big | Select-Object @{N='MB';E={[math]::Round($_.Length/1MB,1)}}, FullName | Format-Table -AutoSize
    Write-Host "Add their folder to .gitignore or set up Git LFS, then re-run." -ForegroundColor Yellow
    exit 1
}
Write-Host "    OK - nothing oversized outside the excluded folders." -ForegroundColor Green

# --- 3. Warn on likely credentials ------------------------------------------
Write-Host "==> Scanning for credential-like files..." -ForegroundColor Cyan
$secrets = Get-ChildItem -Recurse -File -Force -Include '*.env','.cdsapirc','*.key','*.pem','id_rsa*' -ErrorAction SilentlyContinue
if ($secrets) {
    Write-Host "    WARNING - possible secrets found:" -ForegroundColor Yellow
    $secrets | Select-Object FullName | Format-Table -AutoSize
    if ((Read-Host "Continue anyway? (y/N)") -ne 'y') { Write-Host "Cancelled."; exit 1 }
} else {
    Write-Host "    None found." -ForegroundColor Green
}

# --- 4. Init repo (idempotent) ----------------------------------------------
if (-not (Test-Path (Join-Path $ProjectRoot '.git'))) {
    git init | Out-Null
    Write-Host "==> git init" -ForegroundColor Green
} else {
    Write-Host "==> Existing git repo detected - reusing it." -ForegroundColor Green
}

# --- 5. Stage everything; ensure excluded folders are not tracked -----------
git add -A
git rm -r --cached --ignore-unmatch "outputs" "runs" "outputs (RPi)" 2>$null | Out-Null

# --- 6. Commit (skip cleanly if nothing changed) ----------------------------
if (git status --porcelain) {
    git commit -m $CommitMsg | Out-Null
    Write-Host "==> Committed." -ForegroundColor Green
} else {
    Write-Host "==> Nothing to commit (working tree already clean)." -ForegroundColor Yellow
}
git branch -M $Branch

# --- 7. Configure remotes (idempotent) --------------------------------------
git remote remove origin 2>$null | Out-Null ; git remote add origin $GitLabUrl
git remote remove github 2>$null | Out-Null ; git remote add github $GitHubUrl
Write-Host "==> Remotes set:  origin -> GitLab   github -> GitHub" -ForegroundColor Green

# --- 8. Confirm + push ------------------------------------------------------
Write-Host ""
Write-Host "About to push branch '$Branch':" -ForegroundColor Cyan
Write-Host "  origin (GitLab): normal push"
Write-Host "  github (GitHub): FORCE push (overwrites remote)" -ForegroundColor Yellow
if ((Read-Host "Proceed? (y/N)") -ne 'y') { Write-Host "Cancelled."; exit 0 }

Write-Host "`n==> Pushing to GitLab..." -ForegroundColor Cyan
git push -u origin $Branch
if ($LASTEXITCODE -ne 0) {
    Write-Host "GitLab push was rejected (the remote likely has commits)." -ForegroundColor Yellow
    Write-Host "If your local copy is authoritative, force it with:" -ForegroundColor Yellow
    Write-Host "    git push --force origin $Branch" -ForegroundColor Yellow
}

Write-Host "`n==> Pushing to GitHub (force)..." -ForegroundColor Cyan
git push --force github $Branch
if ($LASTEXITCODE -eq 0) { Write-Host "`n==> Done." -ForegroundColor Green }