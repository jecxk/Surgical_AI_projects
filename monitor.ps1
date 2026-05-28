# ===============================================================
# Surgical_AI — Training monitor
# Theo dõi GPU temp/util + tail log realtime
# Usage:  .\monitor.ps1
# ===============================================================

$log = "results/efficientnet_b0_lstm/training.log"

if (-not (Test-Path $log)) {
    Write-Host "[!] Log file chưa tồn tại: $log" -ForegroundColor Yellow
    Write-Host "    Train đã bắt đầu chưa?" -ForegroundColor Yellow
    exit 1
}

Write-Host "Monitoring training... (Ctrl+C để thoát monitor, không ảnh hưởng training)" -ForegroundColor Cyan
Write-Host ""

# Run nvidia-smi every 30s in background loop
$smiJob = Start-Job -ScriptBlock {
    while ($true) {
        $line = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw `
                            --format=csv,noheader,nounits 2>$null
        $time = Get-Date -Format "HH:mm:ss"
        Write-Host "[$time GPU] $line" -ForegroundColor Magenta
        Start-Sleep -Seconds 30
    }
}

try {
    # Tail training.log
    Get-Content $log -Wait -Tail 20
} finally {
    Stop-Job $smiJob -ErrorAction SilentlyContinue
    Remove-Job $smiJob -ErrorAction SilentlyContinue
}
