# 🛣️ PotholeEye

PotholeEye adalah sistem deteksi lubang jalan berbasis **Artificial Intelligence (AI)** yang dibangun menggunakan **YOLOv8** dan **Streamlit**. Aplikasi ini dirancang untuk membantu proses identifikasi kerusakan jalan secara otomatis melalui analisis citra, sehingga dapat mendukung pemantauan kondisi jalan dengan lebih cepat, akurat, dan efisien.

Sistem memungkinkan pengguna untuk mengunggah gambar jalan, melakukan deteksi lubang jalan secara otomatis, menampilkan hasil deteksi dalam bentuk bounding box, memberikan analisis statistik deteksi, serta mengklasifikasikan tingkat bahaya kondisi jalan berdasarkan jumlah lubang yang ditemukan.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)

---

# ✨ Fitur Utama

### 🔍 Deteksi Lubang Jalan Otomatis
- Mendeteksi lubang jalan menggunakan model **YOLOv8**.
- Menampilkan posisi lubang jalan dalam bentuk **bounding box**.
- Menampilkan confidence score untuk setiap objek yang terdeteksi.

### 📤 Upload Gambar
- Mendukung upload gambar dengan format:
  - JPG
  - JPEG
  - PNG
- Tersedia fitur **Demo Image** untuk mencoba sistem tanpa harus mengunggah gambar sendiri.

### 📊 Dashboard Analisis
- Menampilkan jumlah lubang jalan yang terdeteksi.
- Menampilkan statistik confidence score.
- Menampilkan ukuran (bounding box area) setiap lubang jalan.
- Menampilkan tabel detail hasil deteksi.

### ⚠️ Analisis Tingkat Bahaya Jalan
Sistem secara otomatis mengklasifikasikan kondisi jalan menjadi:

- ✅ **Aman**
- ⚠️ **Hati-hati**
- 🚨 **Bahaya**

berdasarkan jumlah lubang jalan yang berhasil dideteksi.

### 📈 Visualisasi Data
Menggunakan **Plotly** untuk menghasilkan visualisasi interaktif berupa:

- Histogram Confidence Score
- Grafik Luas Lubang Jalan
- Statistik menghasilkan hasil deteksi yang interaktif

### 💾 Download Hasil Deteksi
Pengguna dapat mengunduh gambar hasil deteksi yang telah diberi bounding box.

### 🎨 Antarmuka Modern
- Responsive Web Interface
- Dark Theme Dashboard
- Glassmorphism UI
- Interactive Visualization

---

# 🛠️ Teknologi yang Digunakan

Project ini dibangun menggunakan teknologi berikut:

### Artificial Intelligence

- YOLOv8 (Ultralytics)
- PyTorch

### Backend

- Python

### Frontend

- Streamlit

### Computer Vision

- OpenCV

### Data Processing

- NumPy
- Pandas

### Visualization

- Plotly

### Image Processing

- Pillow

### Configuration

- PyYAML

### Package Manager

- pip

---

# 📋 Prasyarat

Sebelum menjalankan aplikasi, pastikan sistem Anda memiliki:

- Python >= 3.11
- pip
- Git

Disarankan menggunakan virtual environment.

---

# 🚀 Instalasi

## 1. Clone Repository

```bash
git clone https://github.com/Krisnaaxz/PotholeEye.git

cd PotholeEye
```

---

## 2. Buat Virtual Environment

```bash
python -m venv venv
```

---

## 3. Aktifkan Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Jalankan Aplikasi

```bash
streamlit run app.py
```

Secara default aplikasi dapat diakses melalui

```
http://localhost:8501
```

---

# 📁 Struktur Project

```
PotholeEye/
│
├── app.py                     # Main Streamlit Application
├── requirements.txt           # Python Dependencies
├── runtime.txt                # Streamlit Runtime Version
├── packages.txt               # Linux Packages (Deployment)
│
├── data/
│   ├── train/
│   ├── valid/
│   └── data.yaml
│
├── runs/
│   └── detect/
│       └── pothole_detection/
│           └── yolov8n_ep30_tuned/
│               ├── results.csv
│               ├── train_batch0.jpg
│               ├── labels.jpg
│               └── weights/
│                   └── best.pt
│
├── assets/                    # Static Assets
│
└── README.md
```

---

# 💻 Contoh Penggunaan

## Melakukan Deteksi Lubang Jalan

1. Jalankan aplikasi

```bash
streamlit run app.py
```

2. Buka browser

```
http://localhost:8501
```

3. Atur parameter model

- Confidence Threshold
- IoU Threshold

4. Upload gambar jalan

atau

Klik tombol

```
Gunakan Gambar Demo
```

5. Tunggu proses inferensi selesai.

6. Sistem akan menampilkan:

- Gambar asli
- Gambar hasil deteksi
- Bounding Box
- Confidence Score
- Jumlah Lubang Jalan
- Tingkat Bahaya Jalan
- Statistik Confidence
- Grafik Luas Lubang Jalan

7. Klik

```
Unduh Hasil Deteksi
```

untuk menyimpan hasil deteksi.

---

# 📊 Output Sistem

Aplikasi menghasilkan beberapa informasi penting, antara lain:

- Jumlah lubang jalan
- Confidence setiap deteksi
- Posisi Bounding Box
- Ukuran Bounding Box
- Histogram Confidence Score
- Grafik Luas Lubang Jalan
- Tingkat Bahaya Jalan
- Hasil gambar yang telah dianotasi

---

# 🎯 Target Pengguna

PotholeEye dirancang untuk digunakan oleh:

- Dinas Pekerjaan Umum
- Pemerintah Daerah
- Instansi Pengelola Jalan
- Peneliti Computer Vision
- Mahasiswa
- Developer AI
- Pengguna umum yang ingin melakukan analisis kondisi jalan

---

# 🤝 Kontribusi

Kontribusi sangat kami apresiasi.

Cara berkontribusi:

1. Fork repository ini

2. Clone repository

```bash
git clone https://github.com/username/PotholeEye.git
```

3. Buat branch baru

```bash
git checkout -b fitur-baru
```

4. Commit perubahan

```bash
git commit -m "Menambahkan fitur baru"
```

5. Push

```bash
git push origin fitur-baru
```

6. Buat Pull Request

---

### Panduan Kontribusi

- Gunakan commit message yang jelas.
- Update dokumentasi jika ada perubahan.
- Pastikan aplikasi dapat dijalankan tanpa error.
- Satu Pull Request hanya untuk satu fitur atau perbaikan.

---

# 🐛 Bug Report

Jika menemukan bug, silakan membuat Issue pada GitHub dengan menyertakan:

- Deskripsi bug
- Langkah reproduksi
- Expected behavior
- Screenshot
- Environment

---

# 👥 Kontributor

| Nama | NIM |
|------|------|
| Komang Krisna Jaya Nova Antara | 2308561029 |
| Anak Agung Gde Agung Pranandita | 2308561106 |
| I Putu Chandra Ananda Putra.S | 2308561126 |
| BENEDIKTUS SILABAN | 2308561139 |

---

# 📄 Lisensi

Project ini menggunakan lisensi **MIT License**.

Silakan menggunakan, memodifikasi, dan mendistribusikan project ini sesuai ketentuan lisensi MIT.

---

# 🙏 Acknowledgments

- Ultralytics YOLOv8
- Streamlit
- PyTorch
- OpenCV
- Plotly
- NumPy
- Pandas
- Pillow

---

⭐ Jika project ini bermanfaat, jangan lupa memberikan **Star** pada repository ini!
