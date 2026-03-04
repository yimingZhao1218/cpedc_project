# Wait for ablation to finish, then run UQ ensemble
$logFile = "C:\Users\Administrator\Desktop\cpedc_project\logs\Ablation_20260228_210553.log"
$projectRoot = "C:\Users\Administrator\Desktop\cpedc_project"

Write-Host "[Monitor] Waiting for ablation pinn_no_fourier to finish..."

while ($true) {
    $content = Get-Content $logFile -Raw -ErrorAction SilentlyContinue
    if ($content -match "pinn_no_fourier.*RMSE=") {
        Write-Host "[Monitor] Ablation suite DONE! Starting UQ ensemble..."
        break
    }
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "$ts - Still waiting..."
    Start-Sleep -Seconds 1800
}

# Run UQ ensemble
Set-Location $projectRoot
& python src/m6/run_uq_ensemble.py --n 10
