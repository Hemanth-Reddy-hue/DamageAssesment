# 1. Clean the workspace of any corrupted state
if (Test-Path venv) {
    Write-Host "Removing old venv..."
    Remove-Item -Recurse -Force venv -ErrorAction SilentlyContinue
}
if (Test-Path .setup_done) {
    Remove-Item -Force .setup_done -ErrorAction SilentlyContinue
}

# 2. Re-create the Python Virtual Environment
Write-Host "Creating fresh virtual environment..."
python -m venv venv

# 3. Test on House1 (Photos)
Write-Host "========================================"
Write-Host "Running Photo Pipeline on House1"
Write-Host "========================================"
.\venv\Scripts\python main.py "C:\Users\Chetana\Downloads\WhatsApp Unknown 2026-10-04 at 5.04.44 PM\House1"

# 4. Test on House2 (Video)
Write-Host "========================================"
Write-Host "Running Video Pipeline on House2"
Write-Host "========================================"
.\venv\Scripts\python main.py "C:\Users\Chetana\Downloads\WhatsApp Unknown 2026-10-04 at 5.04.44 PM\House2"

Write-Host "========================================"
Write-Host "Done! Outputs are stored in the 'out/' folder."
Write-Host "========================================"
