New-Item -ItemType Directory -Force handoff | Out-Null

# file tree with sizes (no venv/models/cache)
Get-ChildItem -Recurse -File | Where-Object { $_.FullName -notmatch '\\(venv|\.git|models|__pycache__|\.pytest_cache)\\' } |
  ForEach-Object { "{0,12}  {1}" -f $_.Length, $_.FullName.Substring((Get-Location).Path.Length+1) } | Out-File handoff\file_tree.txt

# git history (skip if not a git repo)
git log --date=short --format="%h %ad %s" -n 100 2>&1 | Out-File handoff\git_log.txt
git status --short 2>&1 | Out-File handoff\git_status.txt

# tests and benchmarks (using venv python)
$py = ".\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

& $py -m pytest -q 2>&1 | Out-File handoff\pytest.txt
& $py bench\harness.py 2>&1 | Out-File handoff\bench_harness.txt
& $py bench\headtohead.py 2>&1 | Out-File handoff\bench_headtohead.txt
& $py bench\ablation_drift.py 2>&1 | Out-File handoff\bench_ablation.txt
& $py bench\calibration_report.py 2>&1 | Out-File handoff\bench_calibration.txt

# docs and config (no .env)
foreach ($f in "compliance_matrix.md","pyproject.toml","requirements.txt","Makefile",".env.example","schema\capture_v1.json","fixloop\declaration.md","fixloop\diff.patch","protocol\capture_protocol.md","reports\technical_report.md","reports\device_matrix.md") {
  if (Test-Path $f) { Copy-Item $f handoff\ -Force }
}
if (Test-Path Data\ground_truth) { Copy-Item Data\ground_truth handoff\ground_truth -Recurse -Force }
elseif (Test-Path data\ground_truth) { Copy-Item data\ground_truth handoff\ground_truth -Recurse -Force }

# source code
Copy-Item src handoff\src -Recurse -Force
Copy-Item bench handoff\bench -Recurse -Force
Get-ChildItem handoff -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force

# latest outputs
Get-ChildItem out -Recurse -Include plan.json,run_log.json,plan.svg | ForEach-Object {
  $dest = Join-Path handoff ("out\" + $_.FullName.Substring((Resolve-Path out).Path.Length+1))
  New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
  Copy-Item $_.FullName $dest
}

# secret check: this should print nothing
Get-ChildItem handoff -Recurse -File | Select-String -Pattern "hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{30,}"

Compress-Archive -Path handoff\* -DestinationPath handoff.zip -Force
Write-Host "Handoff collection completed successfully."
