# ✋✊✂️ Paper, Rock, and Scissors Detection ✋✊✂️

![Example](/static/example1.png)
![Example2](/static/example2.png)

## Overview Project
Project ini adalah tugas Ujian Akhir Praktikum (UAP) yang ditujukan untuk deploy model machine laerning dengan task prediksi image RPS (rock, paper, scissors).

## Dataset 

Project ini hanya menggunakan 1 buah dataset
- Rock, Paper, and Scissors (RPS) Dataset

Splitting Dataset : Training = 70%, Validation = 25%, Testing = 5%

**Preprocessing** menggunakan bawaan library resnet **preprocess_input** dari keras.

## Model

Model yang digunakan :
- VGG-19

Model summary :

![Example3](/static/example3.png)

Accuracy and loss plot model :
![Example4](/static/example4.png)

Confusion Matrix :
![Example5](/static/example5.png)

Classification Report :

![Example6](/static/example6.png)

Predict 10 random picture :
![Example7](/static/example7.png)

## Usage

1. Lakukan Clone repository berikut:
   ```bash
   git clone https://github.com/212DevaPutra/Modul6Deploy
   ```
2. Navigasikan ke direktori project:
   ```bash
   cd Modul6Deploy
   ```
3. Buat virtual evironment dengan perintah :
   ```bash
   conda create -n modul6 python=3.9
   ```
4. Aktivasi evironment dengan perintah :
   ```bash
   activate modul6
   ```
5. Install requirement file yang diperlukan :
   ```bash
   pip install -r requirements.txt
   ```
6. Buat model dan latih model lalu simpan ke format `.h5`, dan salin ke `model/`. Atau anda bisa mendownload model pada link dibawah!
   ```bash
   https://drive.google.com/file/d/1-DIBGfhhqzpD6pQ8iLPwBFCRMk4K-zVc/view?usp=drive_link
   ```
7. Jalankan dengan perintah :
   ```bash
   python main.py
   ```
8. Buka browser dan masukkan alamat :
   ```bash
   127.0.0.1:2000
   ```
9. Input gambar paper, rock atau scissors random dalam input form kemudian click execute.
10. Hasil prediksi menampilkan nilai akurasi prediksi, label kelas, dan lama waktu prediksi.

## File/Project Structure
- `model/`: Berisi model yang digunakan dengan format `.h5`.
- `app.py`: Aplikasi labu untuk menyajikan model dan perutean.
- `templates/`: Berisi template HTML untuk antarmuka web.
- `statis/`: Direktori untuk file statis (CSS, gambar, dll.).
- `requirements.txt`: File yang berisi semua paket dan versi Python yang diperlukan.

## 👩‍💻👩‍💻 Authors

- [@212DevaPutra](https://github.com/212DevaPutra)

<a href="">
  <img alt="Language Python" src="https://img.shields.io/badge/Language-Python-blue" />
</a>
