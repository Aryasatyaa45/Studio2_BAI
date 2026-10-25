# Studio 2 — Klasifikasi Sampah · Kelompok BAI

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Muhammad Nurikhsan | 123140057 | Sebagai ketua kelompok serta membantu membangun kode |
| 2  | Aryasatya Widyatna Akbar | 123140164 | Merancang pendekatan serta membantu membangun kode |
| 3  | Muhammad Bimastiar | 123140211 | Merancang strategi serta membantu membangun kode |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: ResNet-18 pretrained ImageNet dengan classification head untuk 5 kelas.
- **Strategi pelatihan**: full fine-tuning, AdamW, learning rate `3e-4`, weight decay `1e-4`, CosineAnnealingLR, label smoothing `0.1`, 8 epoch, dan seed `42`.
- **Augmentasi**: RandomResizedCrop 224×224, RandomHorizontalFlip, dan ColorJitter.
- **Data tambahan**: Tidak ada.
- **Ukuran input & preprocessing**: Gambar RGB diubah menjadi 224×224, tensor, lalu dinormalisasi menggunakan mean dan standard deviation ImageNet.
- **Metrik pemilihan model**: Macro-F1 pada split validasi lokal 15% per kelas.

Dataset dan `train_sources.csv` disimpan di lingkungan lokal dan tidak diunggah ke repository publik sesuai aturan kompetisi.

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | `21ce85a` | `0.9363` | - |
| Preview 2 (21 Okt) | - | - | - |
| Final (25 Okt)     | `TERISI_SETELAH_PUSH` | `0.9356` | - |

### Hasil Evaluasi Final (Google Colab GPU)

| Metrik | Nilai |
|--------|-------|
| **Macro-F1** | **0.9356** |
| **Akurasi** | **0.9390** |

**F1 per kelas:**

| Kelas | F1 |
|-------|-----|
| `botol_plastik` | 0.9536 |
| `gelas_plastik` | 0.9434 |
| `kertas_kardus` | 0.8909 |
| `saset_kemasan` | 0.9300 |
| `styrofoam` | 0.9600 |

**Catatan eksperimen:**

Tiga eksperimen dilakukan pada Google Colab GPU (Tesla T4) untuk Final:

1. **Baseline ResNet-18** (8 epoch): best Macro-F1 `0.9356` (epoch 4)
2. **ResNet-18 + Weighted Loss + Augmentasi Agresif** (15 epoch): best Macro-F1 `0.9325` (epoch 14)
3. **ResNet-34 + Weighted Loss + Augmentasi Agresif** (12 epoch): best Macro-F1 `0.9319` (epoch 12)

Model terpilih: **Baseline ResNet-18** karena mencapai Macro-F1 tertinggi.

- Kelas dengan performa terendah tetap `kertas_kardus` (F1 0.8909), diikuti `saset_kemasan` (F1 0.9300).
- Eksperimen lanjutan dengan weighted loss dan augmentasi agresif tidak meningkatkan performa dibanding baseline, kemungkinan karena dataset relatif kecil dan baseline sudah cukup optimal.
- Hasil di atas berasal dari split validasi lokal (15%, seed 42) dan bukan skor validation set tersembunyi resmi.

## Cara Reproduksi

### Lokal (CPU)

Install dependensi:

```bash
pip install -r requirements.txt
```

Training baseline 8 epoch:

```bash
python train.py --data "Q:/Pembelajaran Mendalam/Studio 2/train" --epochs 8 --batch-size 16
```

### Google Colab (GPU)

1. Upload folder `train/` ke Google Drive (`MyDrive/Studio2/train/`)
2. Buka `Studio2_Colab_Training.ipynb` di [Google Colab](https://colab.research.google.com)
3. Aktifkan GPU: **Runtime → Change runtime type → GPU**
4. Jalankan cell satu per satu

Training otomatis membuat split validasi lokal 15% per kelas di `local_data/` menggunakan seed `42`. Dataset harus berada di luar repository dengan struktur berikut:

```text
train/
├── botol_plastik/
├── gelas_plastik/
├── kertas_kardus/
├── saset_kemasan/
└── styrofoam/
```

Uji submission menggunakan evaluator resmi:

```bash
python evaluate.py --submission ./submission --data ./local_data/my_val --out ./local_metrics.json
```

## Struktur Repositori

```text
submission/
  model.py              # get_model() dan get_transform() (ResNet-18)
  model_resnet34.py     # alternatif arsitektur ResNet-34
  model.pth             # state_dict model terbaik

evaluate.py             # salinan skrip evaluasi resmi
train.py                # kode training baseline
train_exp1_weighted.py  # eksperimen weighted loss + augmentasi agresif
train_exp2_resnet34.py  # eksperimen ResNet-34
Studio2_Colab_Training.ipynb  # notebook Google Colab GPU
requirements.txt        # dependensi Python
README.md               # dokumentasi kelompok
```

`submission/model.py` harus menghasilkan logits berukuran `[batch, 5]` dengan urutan kelas berikut:

```text
0 botol_plastik
1 gelas_plastik
2 kertas_kardus
3 saset_kemasan
4 styrofoam
```
