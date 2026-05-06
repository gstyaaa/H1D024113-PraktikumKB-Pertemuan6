# Praktikum Kecerdasan Buatan - Pertemuan 6 (Neural Network)

Repository ini berisi implementasi sederhana dari algoritma Backpropagation dan Perceptron menggunakan Python. Proyek ini mendemonstrasikan bagaimana Neural Network dasar dapat digunakan untuk menyelesaikan masalah klasifikasi pada gerbang logika (OR dan XOR).

## Daftar File
*   **Perceptron.py**: Kelas utama yang berisi logika algoritma Perceptron, termasuk fungsi pelatihan (fit), prediksi, dan visualisasi garis pemisah (decision boundary).
*   **Perceptron_or.py**: Script implementasi untuk menguji algoritma Perceptron pada masalah gerbang logika OR.
*   **Backpropagation.py**: Kelas utama untuk algoritma Backpropagation (Neural Network multi-layer) dengan fungsi aktivasi Sigmoid Bipolar (Tanh).
*   **Backpropagation_xor.py**: Script implementasi untuk menyelesaikan masalah gerbang logika XOR yang tidak bisa diselesaikan oleh Perceptron biasa.

## Prasyarat
Sebelum menjalankan kode, pastikan library berikut sudah terinstal:
```bash
pip install numpy matplotlib
```

## Cara Menjalankan
Script pengujian dapat dijalankan langsung melalui terminal:

**1. Menjalankan Perceptron (Masalah OR):**
```bash
python Perceptron_or.py
```

**2. Menjalankan Backpropagation (Masalah XOR):**
```bash
python Backpropagation_xor.py
```

## Hasil Output

### 1. Visualisasi Grafik
Program akan menghasilkan grafik yang disimpan secara otomatis dalam format PNG:

*   **Perceptron Decision Boundary (OR):**
    ![Perceptron Output](perceptron_output.png)

*   **Backpropagation Error Improvement (XOR):**
    ![Backpropagation Output](backpropagation_output.png)

### 2. File Log
Detail hasil setiap iterasi (bobot, bias, dan error) disimpan dalam file teks berikut:
*   `HasilPerceptron.txt`
*   `hasilBackpropagation.txt`

---