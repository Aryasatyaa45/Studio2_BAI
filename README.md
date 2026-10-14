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
| Final (25 Okt)     | - | - | - |

Catatan eksperimen singkat:

- Baseline 1 epoch menghasilkan Macro-F1 lokal `0.8337` dan akurasi `0.8366`.
- Training 8 epoch menghasilkan Macro-F1 lokal terbaik `0.9363` dan akurasi `0.9390` pada epoch ke-8.
- F1 per kelas pada model terbaik: `botol_plastik` `0.9605`, `gelas_plastik` `0.9554`, `kertas_kardus` `0.8932`, `saset_kemasan` `0.9300`, dan `styrofoam` `0.9423`.
- Kelas dengan performa terendah adalah `kertas_kardus`, sehingga menjadi fokus eksperimen berikutnya.
- Hasil di atas berasal dari split validasi lokal dan bukan skor validation set tersembunyi resmi.

## Cara Reproduksi

Install dependensi dari root repository:

```bash
pip install -r requirements.txt
```

Jalankan training baseline 8 epoch:

```bash
python train.py --data "Q:/Pembelajaran Mendalam/Studio 2/train" --epochs 8 --batch-size 16
```

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
  model.py          # get_model() dan get_transform()
  model.pth         # state_dict model terbaik

evaluate.py         # salinan skrip evaluasi resmi
train.py            # kode training baseline dan pembuatan split lokal
requirements.txt    # dependensi Python
README.md           # dokumentasi kelompok
```

`submission/model.py` harus menghasilkan logits berukuran `[batch, 5]` dengan urutan kelas berikut:

```text
0 botol_plastik
1 gelas_plastik
2 kertas_kardus
3 saset_kemasan
4 styrofoam
```
