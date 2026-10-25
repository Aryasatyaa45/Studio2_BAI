# Panduan Eksperimen Final Studio 2

## File yang tersedia

| File | Fungsi |
|---|---|
| `train.py` | Baseline ResNet-18 (Preview 1, Macro-F1 0.9363) |
| `train_exp1_weighted.py` | Weighted loss + augmentasi agresif + oversampling |
| `train_exp2_resnet34.py` | ResNet-34 dengan augmentasi agresif |
| `train_exp3_progressive.py` | Progressive unfreezing + differential LR |
| `run_experiments.ps1` | Script otomatis menjalankan semua eksperimen |

## Cara menjalankan

### Opsi A: Satu per satu (manual)

```powershell
# Eksperimen 1: Weighted Loss
python train_exp1_weighted.py --data "Q:/Pembelajaran Mendalam/Studio 2/train" --epochs 12 --batch-size 16

# Evaluasi
python evaluate.py --submission ./submission --data ./local_data/my_val --out ./exp1.json

# Eksperimen 2: ResNet-34
python train_exp2_resnet34.py --data "Q:/Pembelajaran Mendalam/Studio 2/train" --epochs 12 --batch-size 16

# Evaluasi
python evaluate.py --submission ./submission --data ./local_data/my_val --out ./exp2.json

# Eksperimen 3: Progressive
python train_exp3_progressive.py --data "Q:/Pembelajaran Mendalam/Studio 2/train" --epochs 15 --batch-size 16

# Evaluasi
python evaluate.py --submission ./submission --data ./local_data/my_val --out ./exp3.json
```

### Opsi B: Otomatis semua (PowerShell)

```powershell
.\run_experiments.ps1
```

Script ini akan:
1. Backup model baseline
2. Jalankan 3 eksperimen secara berurutan
3. Evaluasi masing-masing
4. Pilih model terbaik otomatis
5. Salin ke `submission/model.pth`

**Catatan:** Total waktu bisa 3-6 jam di CPU. Jalankan saat laptop tidak digunakan.

## Parameter yang bisa diubah

| Parameter | Default | Keterangan |
|---|---|---|
| `--epochs` | 12-15 | Jumlah epoch training |
| `--batch-size` | 16 | Ukuran batch (kecilkan jika RAM terbatas) |
| `--lr` / `--lr-head` | 1e-3 | Learning rate |
| `--lr-backbone` | 1e-4 | Learning rate backbone (exp3) |
| `--patience` | 4-5 | Early stopping patience |
| `--unfreeze-epoch` | 5 | Epoch unfreeze backbone (exp3) |

## Tips

- **Jangan hapus** folder `local_data/` agar split validasi tetap sama
- **Backup model** sebelum setiap eksperimen: `Copy-Item .\submission\model.pth .\submission\model_backup.pth`
- **Monitor** F1 per kelas, fokus pada `kertas_kardus`
- **Catat hasil** di tabel untuk perbandingan

## Tabel hasil (isi setelah running)

| Eksperimen | Macro-F1 | Accuracy | F1 kertas_kardus | Catatan |
|---|---|---|---|---|
| Baseline | 0.9363 | 0.9390 | 0.8932 | Preview 1 |
| Exp1 Weighted | ? | ? | ? | |
| Exp2 ResNet-34 | ? | ? | ? | |
| Exp3 Progressive | ? | ? | ? | |

## Setelah mendapat model terbaik

1. Verifikasi evaluator lolos:
```powershell
python evaluate.py --submission ./submission --data ./local_data/my_val
```

2. Update README dengan hasil Final

3. Commit dan push:
```powershell
git add .
git commit -m "Final: [nama eksperimen terbaik], Macro-F1 [nilai]"
git push origin main
```
