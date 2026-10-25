# Script PowerShell untuk menjalankan semua eksperimen Studio 2
# Jalankan dari folder repository: .\run_experiments.ps1

$ErrorActionPreference = "Stop"
$DataPath = "Q:\Pembelajaran Mendalam\Studio 2\train"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "STUDIO 2 - EKSPERIMEN FINAL" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Backup model baseline jika ada
if (Test-Path .\submission\model.pth) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    Copy-Item .\submission\model.pth ".\submission\model_baseline_$timestamp.pth"
    Write-Host "[INFO] Backup baseline dibuat" -ForegroundColor Yellow
}

# Eksperimen 1: Weighted Loss + Augmentasi Agresif
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "EKSPERIMEN 1: Weighted Loss" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
python .\train_exp1_weighted.py --data "$DataPath" --epochs 12 --batch-size 16 --lr 1e-3
python .\evaluate.py --submission .\submission --data .\local_data\my_val --out .\exp1_result.json
$exp1 = Get-Content .\exp1_result.json | ConvertFrom-Json
Write-Host "[HASIL] Exp1 Macro-F1: $($exp1.macro_f1)" -ForegroundColor Magenta
Copy-Item .\submission\model.pth .\submission\model_exp1.pth

# Eksperimen 2: ResNet-34
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "EKSPERIMEN 2: ResNet-34" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
python .\train_exp2_resnet34.py --data "$DataPath" --epochs 12 --batch-size 16 --lr 1e-3
python .\evaluate.py --submission .\submission --data .\local_data\my_val --out .\exp2_result.json
$exp2 = Get-Content .\exp2_result.json | ConvertFrom-Json
Write-Host "[HASIL] Exp2 Macro-F1: $($exp2.macro_f1)" -ForegroundColor Magenta
Copy-Item .\submission\model.pth .\submission\model_exp2.pth

# Eksperimen 3: Progressive Unfreezing
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "EKSPERIMEN 3: Progressive Unfreezing" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
python .\train_exp3_progressive.py --data "$DataPath" --epochs 15 --batch-size 16 --lr-head 1e-3 --lr-backbone 1e-4 --unfreeze-epoch 5
python .\evaluate.py --submission .\submission --data .\local_data\my_val --out .\exp3_result.json
$exp3 = Get-Content .\exp3_result.json | ConvertFrom-Json
Write-Host "[HASIL] Exp3 Macro-F1: $($exp3.macro_f1)" -ForegroundColor Magenta
Copy-Item .\submission\model.pth .\submission\model_exp3.pth

# Bandingkan hasil
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "RINGKASAN HASIL" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Baseline (Preview 1): 0.9363"
Write-Host "Exp1 Weighted Loss:   $($exp1.macro_f1)"
Write-Host "Exp2 ResNet-34:       $($exp2.macro_f1)"
Write-Host "Exp3 Progressive:     $($exp3.macro_f1)"

# Pilih model terbaik
$best = "baseline"
$bestScore = 0.9363
if ($exp1.macro_f1 -gt $bestScore) { $best = "exp1"; $bestScore = $exp1.macro_f1 }
if ($exp2.macro_f1 -gt $bestScore) { $best = "exp2"; $bestScore = $exp2.macro_f1 }
if ($exp3.macro_f1 -gt $bestScore) { $best = "exp3"; $bestScore = $exp3.macro_f1 }

Write-Host ""
Write-Host "[BEST] Model terbaik: $best dengan Macro-F1: $bestScore" -ForegroundColor Green

# Copy model terbaik ke submission
switch ($best) {
    "exp1" { Copy-Item .\submission\model_exp1.pth .\submission\model.pth }
    "exp2" { Copy-Item .\submission\model_exp2.pth .\submission\model.pth }
    "exp3" { Copy-Item .\submission\model_exp3.pth .\submission\model.pth }
}

Write-Host ""
Write-Host "[INFO] Model terbaik sudah disalin ke submission/model.pth" -ForegroundColor Green
Write-Host "[INFO] Jalankan: python .\evaluate.py --submission .\submission --data .\local_data\my_val" -ForegroundColor Yellow
