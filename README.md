# SignLink

Penerjemah isyarat BISINDO (Bahasa Isyarat Indonesia) dari webcam secara real-time.
Kamera membaca gerakan tubuh dan tangan lewat MediaPipe Holistic. Urutan gerakan itu diklasifikasi
oleh model Conv1D + BiGRU menjadi salah satu dari 32 kata, lalu kata-kata tersusun jadi kalimat.

## Kosakata (32 kata)

Air, Belajar, Cari, Hari, Ingat, Lagi, Maaf, Makan, Motor, Saya, Terima kasih, Tuli,
Apa, Siapa, Kapan, Di mana, Mengapa, Bagaimana, Merah, Kuning, Hijau, Hitam, Dengar,
Berangkat, Datang, Teman, Keluarga, Rumah, Pagi, Siang, Sore, Malam.

Contoh gerakan tiap kata ada di [`panduan/index.html`](panduan/index.html) (dua peraga per kata, ada mode cermin).

## Struktur folder

```
SignLink/
├── training/
│   ├── landmarks.py          # fitur per frame, normalisasi, resample (dipakai semua skrip)
│   ├── extract_landmarks.py  # video dataset -> data_landmarks/*.npy
│   ├── train_lstm.py         # latih model / uji leave-one-signer-out
│   ├── webcam_lstm.py        # aplikasi webcam real-time
│   └── buat_panduan.py       # buat halaman panduan/ dari video dataset
├── models/
│   ├── signlink_wl.keras        # model terlatih
│   └── signlink_wl_labels.json  # urutan label
├── panduan/                  # halaman panduan gerakan + video contoh
├── new_dataset/              # video WL-BISINDO (tidak ikut di git)
└── data_landmarks/           # hasil ekstraksi landmark (tidak ikut di git)
```

## Instalasi

Butuh Python 3.10. Versi `mediapipe` harus 0.10.14 karena versi yang lebih baru menghapus `mp.solutions.holistic`.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

`ffmpeg` di PATH hanya dibutuhkan untuk `training/buat_panduan.py`.

## Menjalankan aplikasi webcam

Model sudah ada di `models/`, jadi aplikasi bisa langsung dijalankan tanpa dataset.

```bash
.venv/Scripts/python.exe training/webcam_lstm.py
```

Opsi:

| Opsi | Keterangan |
|---|---|
| `--camera N` | nomor kamera; default otomatis memilih kamera pertama yang ada gambarnya |
| `--video file.mp4` | pakai file video, bukan webcam (untuk uji) |
| `--workers N` | jumlah proses MediaPipe, default 2 (lebih banyak = deteksi lebih rapat) |

Cara pakai: angkat tangan, peragakan satu kata, lalu turunkan tangan. Begitu tangan turun,
gerakan tadi diprediksi dan kata yang yakin masuk ke kalimat di panel kanan.

| Tombol | Fungsi |
|---|---|
| `1`–`3` | pilih/ganti tebakan |
| `H` | daftar kata |
| `P` | buka video panduan |
| `S` | salin kalimat |
| `Backspace` | hapus kata terakhir |
| `C` | hapus kalimat |
| `Z` | batalkan hapus |
| `Q` / `Esc` | keluar |

Font Satoshi opsional: taruh `Satoshi-*.otf` di `training/fonts/`. Tanpa itu dipakai Segoe UI.

## Melatih ulang model

Semua perintah dijalankan dari folder `SignLink`.

1. Unduh dataset [WL-BISINDO](https://github.com/AceKinnn/WL-BISINDO) dan taruh videonya di `new_dataset/`
   dengan nama `signer{s}_label{l}_sample{n}.mp4`.
2. Ekstrak landmark. Proses bisa dihentikan dan dilanjutkan; video yang sudah punya `.npy` dilewati.
   ```bash
   .venv/Scripts/python.exe training/extract_landmarks.py --workers 6
   ```
3. Uji akurasi dengan leave-one-signer-out (model tidak disimpan):
   ```bash
   .venv/Scripts/python.exe training/train_lstm.py --cv
   ```
4. Latih model final dari semua signer. Hasilnya menimpa `models/signlink_wl.keras`.
   ```bash
   .venv/Scripts/python.exe training/train_lstm.py --epochs 60
   ```
5. (Opsional) Buat ulang halaman panduan:
   ```bash
   .venv/Scripts/python.exe training/buat_panduan.py
   ```

## Cara kerja

- **Fitur:** tiap frame jadi vektor 201 angka: 25 titik pose tubuh atas + 21 titik tangan kiri + 21 titik tangan kanan, masing-masing (x, y, z). Tangan yang tidak terlihat diisi nol.
- **Normalisasi:** koordinat dibuat relatif ke tengah bahu dan diskalakan dengan lebar bahu. Posisi orang di frame, jarak ke kamera, dan resolusi jadi tidak berpengaruh. Tangan yang istirahat di samping pinggul dianggap tidak ada.
- **Panjang urutan:** setiap isyarat di-resample jadi 32 frame.
- **Augmentasi saat training:** potong waktu, cermin (untuk orang kidal), putar ±10°, skala, geser, dan noise.
- **Model:** Conv1D(128) → BiGRU(96) → BiGRU(64) → Dense(64) → softmax 32 kelas.

## Akurasi

Uji leave-one-signer-out: model dilatih dari 4 peraga, lalu diuji pada peraga ke-5 yang belum pernah dilihat.

| Peraga uji | Top-1 | Top-3 |
|---|---|---|
| signer0 | 85.0% | 95.0% |
| signer1 | 95.7% | 99.8% |
| signer2 | 86.7% | 95.0% |
| signer3 | 72.5% | 91.9% |
| signer4 | 74.7% | 89.4% |
| **Rata-rata** | **82.9%** | |

Pasangan yang paling sering tertukar antara lain *Tuli → Dengar* dan *Keluarga → Apa*.

## Lisensi dataset

Video dan landmark berasal dari dataset [WL-BISINDO](https://github.com/AceKinnn/WL-BISINDO), berlisensi
CC BY-NC 4.0. Model dan video panduan di repo ini turunan dari dataset tersebut, jadi tidak boleh dipakai untuk tujuan komersial.
