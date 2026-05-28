# ===============================================================
# Surgical_AI — Full experiment runner (M1 -> M2 -> M3 -> Ablations)
#
# Runs experiments sequentially with auto-resume if interrupted.
# Logs separate per experiment. Safe to Ctrl+C and re-run.
# ===============================================================

param(
    [string]$Only = "",        # run only specific stage: m1 / m2 / m3 / abl / ensemble
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Section($msg) {
    Write-Host "`n============================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
}

function RunOne([string]$tag, [string]$cfg, [string]$expectDir) {
    Section "[$tag] Config: $cfg"

    # Detect existing checkpoint to resume
    $ckptDir = "results/$expectDir/checkpoints"
    $latest = $null
    if (Test-Path $ckptDir) {
        $latest = Get-ChildItem "$ckptDir/*.pt" -ErrorAction SilentlyContinue |
                  Sort-Object LastWriteTime -Descending |
                  Select-Object -First 1
    }

    # Skip if already trained (best.pt or final marker exists)
    if (Test-Path "$ckptDir/best.pt") {
        $bestF1 = "?"
        try {
            $info = python -c "import torch; ck=torch.load('$ckptDir/best.pt', map_location='cpu', weights_only=False); print(ck.get('best_val_f1', '?'))" 2>$null
            $bestF1 = $info.Trim()
        } catch {}
        Write-Host "[SKIP] $tag already complete (best.pt exists, val_f1=$bestF1)" -ForegroundColor Green
        return
    }

    $cmd = "python scripts/train.py --config $cfg"
    if ($latest) {
        Write-Host "[RESUME] Found checkpoint: $($latest.Name)" -ForegroundColor Yellow
        $cmd += " --resume `"$($latest.FullName)`""
    }

    Write-Host "Command: $cmd"
    if ($DryRun) {
        Write-Host "[DRY RUN] would execute above" -ForegroundColor Yellow
        return
    }

    $start = Get-Date
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] $tag failed (exit code $LASTEXITCODE)" -ForegroundColor Red
        Write-Host "    Re-run script to resume from last checkpoint" -ForegroundColor Yellow
        exit 1
    }
    $dur = (Get-Date) - $start
    Write-Host "[OK] $tag completed in $($dur.TotalHours.ToString('F1'))h" -ForegroundColor Green
}

# Verify data ready
if (-not (Test-Path "data/cholec80/video80/frames")) {
    Write-Host "[!] Data extraction not complete (video80 missing). Run prepare_dataset.py first." -ForegroundColor Red
    exit 1
}

$startAll = Get-Date

# ===== MAIN MODELS =====
if (!$Only -or $Only -eq "m1") { RunOne "M1" "configs/m1_resnet_lstm.yaml" "resnet50_lstm" }
if (!$Only -or $Only -eq "m2") { RunOne "M2" "configs/m2_efficientnet_tcn.yaml" "efficientnet_b3_tcn" }
if (!$Only -or $Only -eq "m3") { RunOne "M3" "configs/m3_swin_transformer.yaml" "swin_tiny_transformer" }

# ===== ABLATIONS =====
if (!$Only -or $Only -eq "abl") {
    RunOne "Abl-NoMT"      "configs/abl_no_multitask.yaml"      "resnet50_lstm_no_multitask"
    RunOne "Abl-NoCW"      "configs/abl_no_class_weights.yaml"  "resnet50_lstm_no_class_weights"
    RunOne "Abl-Seq16"     "configs/abl_seqlen16.yaml"          "resnet50_lstm_seq16"
}

# ===== ENSEMBLE =====
if (!$Only -or $Only -eq "ensemble") {
    Section "ENSEMBLE evaluation"
    $m1 = "results/resnet50_lstm/checkpoints/best.pt"
    $m2 = "results/efficientnet_b3_tcn/checkpoints/best.pt"
    $m3 = "results/swin_tiny_transformer/checkpoints/best.pt"

    foreach ($p in @($m1, $m2, $m3)) {
        if (-not (Test-Path $p)) {
            Write-Host "[!] Missing: $p — skipping ensemble" -ForegroundColor Yellow
            exit 0
        }
    }

    python scripts/ensemble.py --m1 $m1 --m2 $m2 --m3 $m3 --output results/ensemble
}

$totalDur = (Get-Date) - $startAll
Write-Host "`n============================================" -ForegroundColor Green
Write-Host " ALL EXPERIMENTS DONE — total time: $($totalDur.TotalHours.ToString('F1'))h" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
