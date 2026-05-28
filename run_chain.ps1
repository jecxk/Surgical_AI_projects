# =====================================================
# Surgical_AI - M1 -> M2 -> M3 chain (ASCII-only)
# Bypass run_experiments.ps1 parser issue on PS 5.1.
# Each stage skips itself if best.pt already exists,
# resumes from latest checkpoint otherwise.
# =====================================================

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

function RunStage([string]$tag, [string]$cfg, [string]$outName) {
    Write-Host ""
    Write-Host "===================================================="
    Write-Host "  [$tag] Config: $cfg"
    Write-Host "  Output: results/$outName"
    Write-Host "  Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "===================================================="

    $ckptDir = "results/$outName/checkpoints"
    # Only skip if final_model.pth exists — that's saved at the END of trainer.train(), guaranteeing completion.
    # best_model.pth can exist mid-training (saved on every val_f1 improvement) and must NOT count as done.
    if (Test-Path "$ckptDir/final_model.pth") {
        Write-Host "[SKIP] $tag already complete (final_model.pth exists)"
        return 0
    }

    $pyArgs = @("scripts/train.py", "--config", $cfg)

    if (Test-Path $ckptDir) {
        # Prefer best_model.pth over latest epoch_X.pth (best weights, not just latest)
        $resumeFile = $null
        if (Test-Path "$ckptDir/best_model.pth") {
            $resumeFile = Get-Item "$ckptDir/best_model.pth"
        } else {
            $resumeFile = Get-ChildItem "$ckptDir/*.pth" -ErrorAction SilentlyContinue |
                          Sort-Object LastWriteTime -Descending |
                          Select-Object -First 1
        }
        if ($resumeFile) {
            Write-Host "[RESUME] from $($resumeFile.Name)"
            $pyArgs += @("--resume", $resumeFile.FullName)
        }
    }

    $start = Get-Date
    & python @pyArgs
    $rc = $LASTEXITCODE
    $dur = (Get-Date) - $start

    if ($rc -eq 0) {
        Write-Host "[OK] $tag finished in $($dur.TotalHours.ToString('F2'))h"
    } else {
        Write-Host "[FAIL] $tag exit code $rc after $($dur.TotalHours.ToString('F2'))h"
    }
    return $rc
}

# Verify data ready
if (-not (Test-Path "data/cholec80/video80/frames")) {
    Write-Host "[!] Data not ready (video80 missing)."
    exit 1
}

$startAll = Get-Date
Write-Host "CHAIN START: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

$rc1 = RunStage "M1" "configs/m1_resnet_lstm.yaml"       "resnet50_lstm"
if ($rc1 -ne 0) {
    Write-Host "[!] Stopping chain - M1 failed. Re-run script to resume."
    exit $rc1
}

$rc2 = RunStage "M2" "configs/m2_efficientnet_tcn.yaml"  "efficientnet_b3_tcn"
if ($rc2 -ne 0) {
    Write-Host "[!] Stopping chain - M2 failed. Re-run script to resume."
    exit $rc2
}

$rc3 = RunStage "M3" "configs/m3_swin_transformer.yaml"  "swin_tiny_transformer"
if ($rc3 -ne 0) {
    Write-Host "[!] Stopping chain - M3 failed. Re-run script to resume."
    exit $rc3
}

$totalDur = (Get-Date) - $startAll
Write-Host ""
Write-Host "===================================================="
Write-Host "  ALL 3 MAIN MODELS DONE"
Write-Host "  Total time: $($totalDur.TotalHours.ToString('F2'))h"
Write-Host "  Finish: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "===================================================="
