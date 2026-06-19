# 🧠 Brain Tumor Detection using Deep Learning (CNNs)

Proyek ini mengimplementasikan **Convolutional Neural Network (CNN)** untuk mendeteksi dan mengklasifikasikan tumor otak dari citra MRI ke dalam empat kategori:

| Kelas | Deskripsi |
|-------|-----------|
| `glioma` | Tumor yang berasal dari sel glial otak |
| `meningioma` | Tumor yang tumbuh dari lapisan meninges |
| `notumor` | Tidak ditemukan tumor |
| `pituitary` | Tumor pada kelenjar pituitari |

**Akurasi model:** ~94% pada data uji (15 epoch)

---

## 📁 Struktur Proyek

```
brain-tumor-detection/
├── Brain_Tumor_Detection_CNN.ipynb   # Notebook utama
├── README.md                         # Dokumentasi proyek
└── brain_tumor_cnn_model.h5          # Model tersimpan (setelah training)
```

---

## ⚙️ Instalasi

### 1. Prasyarat

Pastikan sistem kamu sudah terinstal:
- Python **3.8 – 3.10**
- pip (package manager Python)
- Git (opsional)

### 2. Clone atau Unduh Proyek

```bash
git clone https://github.com/username/brain-tumor-detection.git
cd brain-tumor-detection
```

Atau unduh dan ekstrak file ZIP secara manual.

### 3. Buat Virtual Environment (Disarankan)

```bash
# Buat environment baru
python -m venv venv

# Aktifkan di Windows
venv\Scripts\activate

# Aktifkan di macOS/Linux
source venv/bin/activate
```

### 4. Instal Dependensi

```bash
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn tensorflow tqdm
```

Atau gunakan perintah berikut untuk instalasi lengkap sekaligus:

```bash
pip install numpy==1.24.3 \
            pandas==2.0.3 \
            matplotlib==3.7.2 \
            seaborn==0.12.2 \
            opencv-python==4.8.0.76 \
            scikit-learn==1.3.0 \
            tensorflow==2.13.0 \
            keras==2.13.1 \
            tqdm==4.66.1
```

> **Catatan:** Jika menggunakan GPU, instal `tensorflow-gpu` dan pastikan driver CUDA sudah terinstal.

### 5. Verifikasi Instalasi

```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU tersedia:", tf.config.list_physical_devices('GPU'))
```

---

## 📦 Dataset

### Sumber Dataset

Dataset yang digunakan adalah **Brain Tumor MRI Dataset** dari Kaggle:

🔗 [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

### Informasi Dataset

| Kelas | Jumlah Gambar (Training) |
|-------|--------------------------|
| Glioma | 1.321 |
| Meningioma | 1.339 |
| No Tumor | 1.595 |
| Pituitary | 1.457 |
| **Total** | **5.712** |

### Cara Mengunduh Dataset

#### Opsi A — Lewat Kaggle CLI (Direkomendasikan)

1. Instal Kaggle CLI:
   ```bash
   pip install kaggle
   ```

2. Unduh API token dari akun Kaggle kamu:
   - Buka [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
   - Klik **Create New Token** → file `kaggle.json` akan terunduh

3. Letakkan file token di direktori yang benar:
   ```bash
   # macOS/Linux
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json

   # Windows
   mkdir %USERPROFILE%\.kaggle
   copy kaggle.json %USERPROFILE%\.kaggle\
   ```

4. Unduh dataset:
   ```bash
   kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
   unzip brain-tumor-mri-dataset.zip -d dataset/
   ```

#### Opsi B — Unduh Manual

1. Buka halaman dataset: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
2. Klik tombol **Download** (memerlukan akun Kaggle)
3. Ekstrak file ZIP ke folder `dataset/`

### Struktur Folder Dataset

Setelah diekstrak, pastikan struktur folder seperti ini:

```
dataset/
├── Training/
│   ├── glioma/          # 1.321 gambar
│   ├── meningioma/      # 1.339 gambar
│   ├── notumor/         # 1.595 gambar
│   └── pituitary/       # 1.457 gambar
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

### Sesuaikan Path Dataset

Buka notebook `Brain_Tumor_Detection_CNN.ipynb` dan ubah path dataset di bagian awal:

```python
# Jika menjalankan di Kaggle (default)
data = '/kaggle/input/brain-tumor-mri-dataset/Training'

# Jika menjalankan secara lokal
data = 'dataset/Training'
```

---

## 🚀 Cara Menjalankan

### Di Lokal (Jupyter Notebook)

1. Jalankan Jupyter:
   ```bash
   jupyter notebook
   ```

2. Buka file `Brain_Tumor_Detection_CNN.ipynb`

3. Jalankan semua cell secara berurutan: **Kernel → Restart & Run All**

### Di Kaggle

1. Buka [kaggle.com](https://www.kaggle.com) dan login
2. Klik **+ Create → New Notebook**
3. Upload file `.ipynb` atau copy-paste kode
4. Tambahkan dataset lewat **Add Data → Brain Tumor MRI Dataset**
5. Klik **Run All**

### Di Google Colab

1. Buka [colab.research.google.com](https://colab.research.google.com)
2. Upload file `.ipynb` atau buka dari Google Drive
3. Mount Google Drive dan sesuaikan path dataset:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   data = '/content/drive/MyDrive/dataset/Training'
   ```
4. Jalankan semua cell

---

## 🔄 Alur Notebook

```
1. Import Libraries
       ↓
2. Setup Dataset → Eksplorasi & Visualisasi
       ↓
3. Preprocessing → Resize (128×128) → Normalisasi → One-Hot Encoding
       ↓
4. Train/Test Split (80:20)
       ↓
5. Bangun Model CNN
       ↓
6. Compile & Training (15 epoch, batch size 32)
       ↓
7. Evaluasi → Classification Report → Confusion Matrix
       ↓
8. Simpan Model (.h5)
```

---

## 📊 Hasil Pelatihan

| Epoch | Train Accuracy | Val Accuracy |
|-------|---------------|--------------|
| 1 | 54.55% | 73.49% |
| 5 | 90.58% | 90.55% |
| 10 | 97.09% | 93.26% |
| 15 | 98.87% | 94.05% |

### Classification Report

| Kelas | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Pituitary | 0.96 | 0.99 | 0.97 |
| No Tumor | 0.97 | 0.97 | 0.97 |
| Meningioma | 0.87 | 0.90 | 0.89 |
| Glioma | 0.96 | 0.89 | 0.92 |
| **Rata-rata** | **0.94** | **0.94** | **0.94** |

---

## 🏗️ Arsitektur Model CNN

```
Input (128×128×3)
      ↓
Conv2D (32 filter, 5×5, stride 2) + ReLU
      ↓
Conv2D (32 filter, 5×5, stride 2) + ReLU
      ↓
MaxPooling2D (2×2)
      ↓
Dropout (0.25)
      ↓
Flatten → 7.200 neuron
      ↓
Dense (512) + ReLU
      ↓
Dropout (0.50)
      ↓
Dense (4) + Softmax → Output
```

**Total parameter:** 3.717.028 (~14.18 MB)

---

## 🛠️ Teknologi yang Digunakan

- **TensorFlow / Keras** — framework deep learning
- **OpenCV** — pemrosesan gambar
- **NumPy & Pandas** — manipulasi data
- **Matplotlib & Seaborn** — visualisasi
- **Scikit-learn** — evaluasi model & split data

---

## 📄 Lisensi

Proyek ini menggunakan lisensi **MIT**. Dataset dari Kaggle tunduk pada [Kaggle Dataset License](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
