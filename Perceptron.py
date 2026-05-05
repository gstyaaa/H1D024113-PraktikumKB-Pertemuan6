import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """
    Kelas Perceptron untuk klasifikasi biner.
    """
    def __init__(self, alpha=0.1, epoch=10):
        self.alpha = alpha
        self.epoch = epoch
        self.w_ = None

    def weighted_sum(self, X):
        """Menghitung nilai y_in atau net."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Menerapkan fungsi aktivasi bipolar."""
        return np.where(self.weighted_sum(X) >= 0.0, 1, -1)

    def plot_decision_boundary(self, X, t, epoch):
        """Membuat simulasi garis pemisah data."""
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=t.ravel(), marker='o',
                    edgecolors='k', cmap=plt.cm.RdYlBu)
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        x_vals = np.linspace(x_min, x_max, 100)
        # Menghindari pembagian dengan nol jika w[2] adalah 0
        if self.w_[2] != 0:
            y_vals = -(self.w_[0] + self.w_[1] * x_vals) / self.w_[2]
            plt.plot(x_vals, y_vals, 'b-', label=f'Decision boundary (Epoch {epoch+1})')
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(f"Decision Boundary Pada Epoch {epoch+1}")
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.show()

    def fit(self, X, t):
        """Fungsi utama pelatihan Perceptron."""
        # Inisialisasi bobot dan bias awal = 0
        self.w_ = np.zeros(1 + X.shape[1])
        
        # Menyimpan hasil pada HasilPerceptron.txt
        with open("HasilPerceptron.txt", "w") as f:
            f.write("Masalah OR dengan Perceptron\n")
            f.write("----------------------------\n")
            f.write(f"Input :\n{X}\n")
            f.write(f"Target:\n{t}\n")
            f.write(f"Bobot awal : {self.w_[1:]}\n")
            f.write(f"Bias awal : {self.w_[0]}\n")
            f.write(f"Learning rate : {self.alpha}\n")
            f.write(f"Max Epoch : {self.epoch}\n")

            for epoch in range(self.epoch):
                f.write(f"\nEpoch {epoch + 1}/{self.epoch}\n")
                f.write("----------\n")
                error_list = []
                
                # Iterasi setiap pasang matriks input dengan targetnya
                for xi, target in zip(X, t):
                    # Periksa input dengan model Perceptron
                    y_pred = self.predict(xi)
                    
                    # Hitung error
                    error = target[0] - y_pred
                    error_list.append(error)
                    
                    # Modifikasi bobot dengan Delta Rule
                    update = self.alpha * error
                    self.w_[1:] += update * xi
                    self.w_[0] += update
                    
                    # Menuliskan hasil tiap iterasi input pada satu epoch
                    f.write(f"Input: {xi}, Target: {target[0]}, Predict: {y_pred}, "
                            f"Error: {error}, Bobot: {self.w_[1:]}, Bias: {self.w_[0]}\n")
                
                # Simulasikan garis pemisah model Perceptron
                self.plot_decision_boundary(X, t, epoch)
                
                # Menuliskan penjumlahan kuadrat error setiap input
                sse = sum(np.array(error_list) ** 2)
                f.write(f"Sum Square Error(SSE): {sse}\n")
                
                # Periksa kondisi berhenti
                if sse == 0 or epoch + 1 == self.epoch:
                    f.write("-" * 80 + "\n")
                    reason = "Sum Square Error(SSE) mencapai target." if sse == 0 else "max epoch tercapai."
                    f.write(f"Pelatihan berhenti pada epoch ke-{epoch + 1} karena {reason}\n")
                    f.write(f"\nBobot akhir :{self.w_[1:]}\n")
                    f.write(f"Bias akhir :{self.w_[0]}")
                    break

