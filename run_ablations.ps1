# =====================================================
# Surgical_AI - Ablation chain (A1 -> A2 -> A3)
# A1: no_multitask        | A2: no_class_weights | A3: seqlen16
# Skip rule: final_model.pth in target dir = done.
# Resume rule: prefer best_model.pth, else latest .pth.
# =====================================================

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

function RunAblation([string]$tag, [string]$cfg, [string]$outDir) {
    Write-Host ""
    Write-Host "===================================================="
    Write-Host "  [$tag] Config: $cfg"
    Write-Host "  Output: $outDir"
    Write-Host "  Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "===================================================="

    # train.py appends '{backbone}_{temporal_model}' onto --output, so final dir is "$outDir/resnet50_lstm"
    $ckptDir = "$outDir/resnet50_lstm/checkpoints"

    if (Test-Path "$ckptDir/final_model.pth") {
        Write-Host "[SKIP] $tag already complete (final_model.pth exists)"
        return 0
    }

    $pyArgs = @("scripts/train.py", "--config", $cfg, "--output", $outDir)

    if (Test-Path $ckptDir) {
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
    # Pipe python output to Out-Host so it stays visible (Start-Transcript still captures it)
    # WITHOUT contaminating this function's pipeline return value.
    & python @pyArgs *>&1 | Out-Host
    $rc = $LASTEXITCODE
    $dur = (Get-Date) - $start

    # Exit code 1 from cosmetic print bug is benign if final_model.pth was saved.
    if (Test-Path "$ckptDir/final_model.pth") {
        Write-Host "[OK] $tag finished in $($dur.TotalHours.ToString('F2'))h (rc=$rc; final_model.pth saved so treating as success)"
        return 0
    }

    Write-Host "[FAIL] $tag exit code $rc after $($dur.TotalHours.ToString('F2'))h"
    return $rc
}

if (-not (Test-Path "data/cholec80/video80/frames")) {
    Write-Host "[!] Data not ready."
    exit 1
}

$startAll = Get-Date
Write-Host "ABLATION CHAIN START: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

$rc1 = RunAblation "A1-NoMultiTask" "configs/abl_no_multitask.yaml"     "results/abl_no_multitask"
if ($rc1 -ne 0) { Write-Host "[!] Chain stopped at A1"; exit $rc1 }

$rc2 = RunAblation "A2-NoClassWeights" "configs/abl_no_class_weights.yaml" "results/abl_no_class_weights"
if ($rc2 -ne 0) { Write-Host "[!] Chain stopped at A2"; exit $rc2 }

$rc3 = RunAblation "A3-SeqLen16" "configs/abl_seqlen16.yaml" "results/abl_seqlen16"
if ($rc3 -ne 0) { Write-Host "[!] Chain stopped at A3"; exit $rc3 }

$totalDur = (Get-Date) - $startAll
Write-Host ""
Write-Host "===================================================="
Write-Host "  ALL 3 ABLATIONS DONE"
Write-Host "  Total time: $($totalDur.TotalHours.ToString('F2'))h"
Write-Host "  Finish: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "===================================================="
