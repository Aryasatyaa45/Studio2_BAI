# Panduan Google Colab GPU — Studio 2 Final

**Deadline:** Minggu, 25 Oktober 2026, 23:59 WIB  
**Repo:** https://github.com/Aryasatyaa45/Studio2_BAI

---

## Persiapan Sebelum Buka Colab

### 1. Upload Dataset ke Google Drive

1. Buka [Google Drive](https://drive.google.com)
2. Buat folder: `Studio2/train/`
3. Upload seluruh isi folder `Q:\Pembelajaran Mendalam\Studio 2\train\` ke Google Drive  
   Struktur di Drive harus seperti ini:
   ```
   MyDrive/
   └── Studio2/
       └── train/
           ├── botol_plastik/
           ├── gelas_plastik/
           ├── kertas_kardus/
           ├── saset_kemasan/
           └── styrofoam/
   ```
4. Pastikan semua 2.753 gambar JPG terupload

### 2. Upload Notebook ke Colab

1. Buka [Google Colab](https://colab.research.google.com)
2. Klik **File → Upload notebook**
3. Pilih file: `Studio2_Colab_Training.ipynb` (dari repo lokal)
4. Atau drag & drop file `.ipynb` ke tab browser Colab

---

## Konfigurasi Runtime GPU

**WAJIB dilakukan sebelum menjalankan cell:**

1. Di menu Colab, klik **Runtime → Change runtime type**
2. Pada dropdown "Hardware accelerator", pilih **GPU**
3. Klik **Save**
4. Tunggu runtime restart jika diminta

**Verifikasi GPU:** Jalankan cell ke-4 ("Verifikasi GPU"), pastikan output:
```
CUDA available: True
GPU name: Tesla T4
```

---

## Urutan Menjalankan Cell

| Urutan | Cell | Keterangan |
|--------|------|------------|
| 1 | Mount Google Drive | Autentikasi Google Drive |
| 2 | Clone Repository | Download repo dari GitHub |
| 3 | Verifikasi GPU | Pastikan GPU aktif |
| 4 | Siapkan Dataset | Copy dataset ke runtime Colab |
| 5 | Install Dependencies | Install torch, torchvision, dll |
| 6 | **Eksperimen 1** — Baseline | ResNet-18, 8 epoch (~10-15 menit) |
| 7 | Backup Model | Simpan model baseline |
| 8 | **Eksperimen 2** — Weighted+Augmentasi | ResNet-18 agresif, 15 epoch (~20-30 menit) |
| 9 | Backup Model | Simpan model exp2 |
| 10 | **Eksperimen 3** — ResNet-34 | Backbone lebih dalam, 12 epoch (~25-35 menit) |
| 11 | Backup Model | Simpan model exp3 |
| 12 | Pilih Model Terbaik | Bandingkan hasil, copy model terbaik |
| 13 | Evaluasi Final | Jalankan evaluator resmi |
| 14 | Lihat Hasil | Cetak Macro-F1 per kelas |
| 15 | **Push ke GitHub** | Commit & push model terbaik |

**Total waktu estimasi:** 1-2 jam untuk ketiga eksperimen.

---

## Catatan Penting

### Jika Eksperimen 3 (ResNet-34) Terpilih

ResNet-34 memiliki arsitektur berbeda dari ResNet-18. Jika model terbaik berasal dari eksperimen 3, **WAJIB** mengganti `submission/model.py`:

```python
# Di cell "Pilih Model Terbaik", uncomment:
shutil.copy('submission/model_resnet34.py', 'submission/model.py')
```

Jika tidak mengganti `model.py`, evaluator akan gagal memuat bobot!

### Jika Eksperimen 1 atau 2 Terpilih

`submission/model.py` tetap menggunakan ResNet-18 (default), tidak perlu diubah.

### Early Stopping

Eksperimen 2 dan 3 menggunakan early stopping patience=5. Jika Macro-F1 tidak meningkat selama 5 epoch berturut-turut, training otomatis berhenti. Ini normal dan menghemat waktu.

### Batch Size

Batch size 64 diatur untuk GPU T4 (16 GB VRAM). Jika out-of-memory error, turunkan ke 32:
```python
--batch-size 32
```

### Jika Runtime Diskonnect

Colab kadang disconnect setelah idle. Jika terjadi:
1. Klik **Runtime → Run all** (atau jalankan ulang dari cell Mount Drive)
2. Dataset perlu di-copy ulang ke runtime
3. Training harus diulang dari awal

**Tips:** Jangan biarkan tab Colab idle terlalu lama. Sesekali klik/tab ke tab Colab.

---

## Push ke GitHub dari Colab

### Cara 1: Password Biasa (Jika tidak pakai 2FA)

Saat `git push` meminta password, masukkan **Personal Access Token** (bukan password Google):

1. Buka GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token → centang **repo**
3. Copy token, paste sebagai password di Colab

### Cara 2: Token di URL (Lebih Mudah)

Ganti remote URL sebelum push:
```python
!git remote set-url origin https://TOKEN@github.com/Aryasatyaa45/Studio2_BAI.git
!git push origin main
```
Ganti `TOKEN` dengan Personal Access Token Anda.

---

## Checklist Sebelum Deadline

- [ ] Dataset terupload ke Google Drive
- [ ] Runtime Colab di-set ke GPU
- [ ] Ketiga eksperimen selesai dijalankan
- [ ] Model terbaik dipilih dan disalin ke `submission/model.pth`
- [ ] `model.py` sesuai dengan arsitektur model terbaik
- [ ] Evaluator lokal lolos dengan Macro-F1 terbaik
- [ ] Push ke GitHub berhasil
- [ ] Commit terakhir sebelum 25 Okt 2026, 23:59 WIB
- [ ] README.md diperbarui dengan hasil Final
- [ ] Repository publik dan bisa diakses

---

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `CUDA out of memory` | Turunkan `--batch-size` ke 32 atau 16 |
| `Dataset tidak ditemukan` | Cek path di Google Drive, harus `/content/drive/MyDrive/Studio2/train` |
| `git push gagal` | Gunakan Personal Access Token sebagai password |
| `Runtime disconnect` | Jalankan ulang, copy dataset, training dari awal |
| `Model.pth tidak valid` | Pastikan `model.py` dan `model.pth` berasal dari eksperimen yang sama |
| `Evaluator gagal` | Cek `model.py` mendefinisikan `get_model()` dan `get_transform()` |

---

## Perbandingan Eksperimen (Isi Setelah Training)

| Eksperimen | Macro-F1 | Accuracy | Catatan |
|------------|----------|----------|---------|
| 1 Baseline ResNet-18 | ? | ? | 8 epoch |
| 2 Weighted + Augmentasi | ? | ? | 15 epoch |
| 3 ResNet-34 | ? | ? | 12 epoch |

**Model terbaik:** Eksperimen ___ dengan Macro-F1 = ___
