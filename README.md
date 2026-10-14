# Studio 2 — Klasifikasi Sampah

**Nama kelompok:** BAI  
**Mata kuliah:** Deep Learning (IF25-40401)  
**Institusi:** Teknik Informatika ITERA  
**Semester:** Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|---:|---|---|---|
| 1 | Muhammad Nurikhsan | 123140057 | Sebagai Ketua kelompok serta membantu membangun kode |
| 2 | Aryasatya Widyatna Akbar | 123140164 | Merancang pendekatan serta membantu membangun kode  |
| 3 | Muhammad Bimastiar | 123140211 | Memrancang strategi serta membantu membangun kode |

## Ringkasan Pendekatan

- **Backbone awal:** ResNet-18 pretrained ImageNet.
- **Jumlah kelas:** 5.
- **Ukuran input:** 224×224.
- **Strategi awal:** RandomResizedCrop, horizontal flip, ColorJitter, normalisasi ImageNet, AdamW, CosineAnnealingLR, dan label smoothing.
- **Metrik pemilihan model:** Macro-F1 pada split validasi lokal.
- **Data tambahan:** Belum digunakan.

Dataset train disimpan lokal dan tidak diunggah ke repository sesuai aturan kompetisi. Metadata sumber gambar juga tidak dipublikasikan ulang.

## Urutan Kelas

```text
0 botol_plastik
1 gelas_plastik
2 kertas_kardus
3 saset_kemasan
4 styrofoam
```

## Setup

```bash
pip install -r requirements.txt
```

Dataset harus tetap berada di luar repository dalam struktur:

```text
train/
├── botol_plastik/
├── gelas_plastik/
├── kertas_kardus/
├── saset_kemasan/
└── styrofoam/
```

## Training Lokal

Jalankan smoke test satu epoch terlebih dahulu:

```bash
python train.py --data "Q:/Pembelajaran Mendalam/Studio 2/train" --epochs 1
```

Training baseline penuh:

```bash
python train.py --data "Q:/Pembelajaran Mendalam/Studio 2/train" --epochs 8
```

Jika bobot ImageNet belum tersedia di cache, torchvision akan mengunduhnya saat training pertama. Untuk uji tanpa unduhan bobot:

```bash
python train.py --data "Q:/Pembelajaran Mendalam/Studio 2/train" --epochs 1 --scratch
```

Split validasi lokal dibuat otomatis di `local_data/` dengan proporsi 15% per kelas dan seed 42. Folder tersebut diabaikan oleh Git.

## Evaluasi Submission

Setelah training menghasilkan `submission/model.pth`:

```bash
python evaluate.py --submission ./submission --data ./local_data/my_val --out local_metrics.json
```

Evaluator resmi memeriksa output logits `[batch, 5]` dan menghitung Macro-F1, akurasi, F1 per kelas, serta confusion matrix.

## Struktur Submission

```text
submission/
├── model.py
└── model.pth
```

`submission/model.py` menyediakan `get_model()` dan `get_transform()` sesuai kontrak evaluator resmi. `model.pth` berisi `state_dict` model terbaik berdasarkan Macro-F1 validasi lokal.

## Progress Check

| Putaran | Tenggat | Status |
|---|---|---|
| Preview 1 | 14 Oktober 2026, 23:59 WIB | Belum dikirim |
| Preview 2 | 21 Oktober 2026, 23:59 WIB | Belum dikirim |
| Final | 25 Oktober 2026, 23:59 WIB | Belum dikirim |

Commit harus dibuat sebelum setiap tenggat. Skor Final ditentukan dari Macro-F1 pada validation set tersembunyi.

## Catatan Aturan

- Repository kelompok harus publik.
- README final wajib memuat nama lengkap dan NIM seluruh anggota.
- `model.pth` maksimal 500 MB.
- Validation set resmi tidak dibagikan kepada kelompok.
- Jangan mengunggah dataset atau `train_sources.csv` ke repository publik.
- Berbagi kode, bobot, atau prediksi dengan kelompok lain tidak diperbolehkan.
