# Praktikum Kecerdasan Buatan - Pertemuan 6 (Neural Network)

Repository ini berisi implementasi sederhana dari algoritma **Perceptron** dan **Backpropagation** menggunakan Python. Proyek ini mendemonstrasikan bagaimana Neural Network dasar dapat digunakan untuk menyelesaikan masalah klasifikasi pada gerbang logika (OR dan XOR).

## 🚀 Daftar File
*   **`Perceptron.py`**: Kelas utama yang berisi logika algoritma Perceptron, termasuk fungsi pelatihan (*fit*), prediksi, dan visualisasi garis pemisah (*decision boundary*).
*   **`Perceptron_or.py`**: Script implementasi untuk menguji algoritma Perceptron pada masalah gerbang logika **OR**.
*   **`Backpropagation.py`**: Kelas utama untuk algoritma Backpropagation (Neural Network multi-layer) dengan fungsi aktivasi Sigmoid Bipolar (Tanh).
*   **`Backpropagation_xor.py`**: Script implementasi untuk menyelesaikan masalah gerbang logika **XOR** yang tidak bisa diselesaikan oleh Perceptron biasa.

## 🛠️ Prasyarat
Sebelum menjalankan kode, pastikan kamu sudah menginstal library yang diperlukan:
```bash
pip install numpy matplotlib
```

## 💻 Cara Menjalankan
Kamu bisa menjalankan script pengujian langsung dari terminal:

**1. Menjalankan Perceptron (Masalah OR):**
```bash
python Perceptron_or.py
```

**2. Menjalankan Backpropagation (Masalah XOR):**
```bash
python Backpropagation_xor.py
```

## 📊 Hasil Output
1.  **Visualisasi**: Program akan menampilkan grafik `matplotlib` yang menunjukkan proses pemisahan data atau grafik penurunan *Error (SSE)* setiap epoch.
2.  **File Log**: Hasil detail setiap iterasi (bobot, bias, dan error) akan disimpan secara otomatis ke dalam file teks:
    *   `HasilPerceptron.txt`
    *   `hasilBackpropagation.txt`

---
*Dibuat untuk keperluan praktikum Kecerdasan Buatan.*
