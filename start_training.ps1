# ===============================================================
# Surgical_AI — One-button training startup for RTX 2050
# Usage:  .\start_training.ps1
# ===============================================================

$ErrorActionPreference = "Stop"

function Section($msg) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
}

Set-Location $PSScriptRoot

# 1. Verify data extract complete
Section "STEP 1 / 4  — Verify dataset"
$progress = Get-Content "data/prepare_progress.json" | ConvertFrom-Json
$done = $progress.completed.Count
$failed = $progress.failed.Count

Write-Host "Completed videos: $done / 80"
Write-Host "Failed videos  : $failed"

if (Test-Path "D:/cholec80.zip") {
    Write-Host "[!] Zip file vẫn còn = pipeline chưa xong." -ForegroundColor Yellow
    Write-Host "    Chạy lại:  python prepare_dataset.py" -ForegroundColor Yellow
    Write-Host "    Sau khi xong rồi run script này lại." -ForegroundColor Yellow
    exit 1
}

if ($done -lt 80) {
    Write-Host "[!] Mới có $done/80 — pipeline chưa xong hoặc có failed." -ForegroundColor Yellow
    Write-Host "    Chạy:  python prepare_dataset.py  (resume)" -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Tất cả 80 video đã extract." -ForegroundColor Green

# 2. Smoke test on real data
Section "STEP 2 / 4  — Smoke test (forward pass with real data)"
Write-Host "Test 1 batch qua model, xác nhận không OOM trước khi train thật..."
python smoke_test.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Smoke test thất bại — kiểm tra log phía trên." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Smoke test pass." -ForegroundColor Green

# 3. Disable sleep / hibernate while training
Section "STEP 3 / 4  — Disable sleep mode"
Write-Host "Tắt sleep + hibernate khi cắm điện (để khôi phục sau train)..."
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 30   # màn hình vẫn tắt được sau 30 phút
Write-Host "[OK] Sleep disabled. Sau khi train xong nhớ chạy: powercfg /change standby-timeout-ac 30" -ForegroundColor Green

# 4. Launch training
Section "STEP 4 / 4  — Bắt đầu training"
Write-Host "Config: configs/local_2050.yaml"
Write-Host "ETA   : ~12-15 giờ"
Write-Host "Log   : results/efficientnet_b0_lstm/training.log"
Write-Host ""
Write-Host "Để monitor song song:  .\monitor.ps1" -ForegroundColor Yellow
Write-Host "Để dừng:               Ctrl+C  (checkpoint sẽ giữ, resume sau)" -ForegroundColor Yellow
Write-Host ""

# Find latest checkpoint if any (resume mode)
$ckpt = Get-ChildItem "results/efficientnet_b0_lstm/checkpoints/*.pt" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($ckpt) {
    Write-Host "[i] Tìm thấy checkpoint cũ: $($ckpt.Name)" -ForegroundColor Yellow
    $resume = Read-Host "Resume từ checkpoint này? [Y/n]"
    if ($resume -ne 'n') {
        python scripts/train.py --config configs/local_2050.yaml --resume $ckpt.FullName
    } else {
        python scripts/train.py --config configs/local_2050.yaml
    }
} else {
    python scripts/train.py --config configs/local_2050.yaml
}
