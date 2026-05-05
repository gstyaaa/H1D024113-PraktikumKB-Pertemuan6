import numpy as np
import matplotlib.pyplot as plt

class Backpropagation:
    """
    Kelas Backpropagation untuk Neural Network sederhana.
    """
    def __init__(self, alpha, epoch, target_error):
        self.alpha = alpha
        self.epoch = epoch
        self.target_error = target_error
        self.n_input = 2
        self.n_hidden = 2
        self.n_output = 1
        
        # Inisialisasi bobot dan bias awal secara acak
        self.w_hidden = np.random.rand(self.n_input, self.n_hidden)
        self.b_hidden = np.random.rand(1, self.n_hidden)
        self.w_output = np.random.rand(self.n_hidden, self.n_output)
        self.b_output = np.random.rand(1, self.n_output)

    def bi_sigmoid(self, x):
        """Fungsi aktivasi sigmoid bipolar (tanh)."""
        return np.tanh(x)

    def deriv_bi_sigmoid(self, x):
        """Turunan fungsi aktivasi sigmoid bipolar (1 - tanh^2)."""
        return 1 - x**2

    def plot_error(self, errors, epoch):
        """Membuat grafik perbaikan error setiap epoch."""
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, epoch + 1), errors, linestyle='-', color='b', label='Error')
        
        final_error = errors[-1]
        plt.annotate(f'Epoch {epoch}, Error: {final_error:.4f}',
                    xy=(epoch, final_error),
                    xytext=(epoch * 0.7, final_error + 0.05),
                    arrowprops=dict(facecolor='black', arrowstyle="->"),
                    fontsize=10, color='red')
        
        plt.title('Perbaikan Error Setiap Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Sum Square Error (SSE)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def fit(self, X, t):
        """Fungsi utama pelatihan Backpropagation."""
        errors_per_epoch = []
        
        # Menyimpan hasil pada hasilBackpropagation.txt
        with open("hasilBackpropagation.txt", "w") as f:
            f.write("Masalah XOR dengan Backpropagation\n")
            f.write("-" * 35 + "\n")
            f.write(f"Input :\n{X}\n")
            f.write(f"Target :\n{t}\n\n")
            f.write(f"Bobot awal hidden layer :\n{self.w_hidden}\n\n")
            f.write(f"Bias awal hidden layer :\n{self.b_hidden}\n\n")
            f.write(f"Bobot awal output layer :\n{self.w_output}\n\n")
            f.write(f"Bias awal output layer :\n{self.b_output}\n\n")
            f.write(f"Learning rate : {self.alpha}\n")
            f.write(f"Max Epoch : {self.epoch}\n\n")

            for epoch in range(self.epoch):
                f.write("-" * 45 + "\n")
                f.write(f"Epoch {epoch + 1}/{self.epoch}\n")
                f.write("-" * 12 + "\n")
                
                total_error = 0
                outputs = []
                
                # Iterasi setiap pasang matriks input dengan targetnya
                for i, (xi, target) in enumerate(zip(X, t), 1):
                    f.write(f"Data ke-{i}\n")
                    f.write("-" * 10 + "\n")
                    f.write("------------ Forward Propagation ------------\n")
                    
                    # Forward ke hidden layer
                    h_in = np.dot(xi, self.w_hidden) + self.b_hidden
                    f.write(f"Operasi input ke hidden layer:\n{h_in}\n\n")
                    h = self.bi_sigmoid(h_in)
                    f.write(f"Aktivasi hidden layer:\n{h}\n\n")
                    
                    # Forward ke output layer
                    y_in = np.dot(h, self.w_output) + self.b_output
                    f.write(f"Operasi hidden ke output layer:\n{y_in}\n\n")
                    y = self.bi_sigmoid(y_in)
                    outputs.append(y[0])
                    f.write(f"Aktivasi output layer:\n{y}\n")
                    
                    f.write("------------ Backward Propagation ------------\n")
                    
                    # Hitung error output layer
                    error = target - y
                    total_error += np.sum(error**2)
                    f.write(f"Error:\n{error}\n\n")
                    
                    # Delta output layer
                    d_y = error * self.deriv_bi_sigmoid(y)
                    f.write(f"Delta output (d_y):\n{d_y}\n\n")
                    
                    # Hitung error hidden layer
                    error_h = np.dot(d_y, self.w_output.T)
                    f.write(f"Error hidden layer (error_h):\n{error_h}\n\n")
                    
                    # Delta hidden layer
                    d_h = error_h * self.deriv_bi_sigmoid(h)
                    f.write(f"Delta hidden layer (d_h):\n{d_h}\n\n")
                    
                    # Perbaiki bobot dan bias output layer
                    self.w_output += np.dot(h.T, d_y) * self.alpha
                    f.write(f"Bobot output layer baru (w_output):\n{self.w_output}\n\n")
                    self.b_output += d_y * self.alpha
                    f.write(f"Bias output layer baru (b_output):\n{self.b_output}\n\n")
                    
                    # Perbaiki bobot dan bias hidden layer
                    self.w_hidden += np.dot(xi.reshape(-1, 1), d_h) * self.alpha
                    f.write(f"Bobot hidden layer baru (w_hidden):\n{self.w_hidden}\n\n")
                    self.b_hidden += d_h * self.alpha
                    f.write(f"Bias hidden layer baru (b_hidden):\n{self.b_hidden}\n")
                    f.write("-" * 45 + "\n")

                average_error = total_error / len(X)
                errors_per_epoch.append(average_error)
                
                f.write(f"Output : {np.array(outputs).flatten()}\n")
                f.write(f"Sum Square Error(SSE) epoch ke-{epoch + 1}: {average_error}\n")
                
                # Cek kondisi berhenti
                if average_error < self.target_error or epoch + 1 == self.epoch:
                    f.write("-" * 80 + "\n")
                    reason = "Sum Square Error(SSE) mencapai target." if average_error < self.target_error else "max epoch tercapai."
                    f.write(f"Pelatihan berhenti pada epoch ke-{epoch + 1} karena {reason}\n")
                    f.write(f"Bobot akhir hidden layer :\n{self.w_hidden}\n\n")
                    f.write(f"Bias akhir hidden layer :\n{self.b_hidden}\n\n")
                    f.write(f"Bobot akhir output layer :\n{self.w_output}\n\n")
                    f.write(f"Bias akhir output layer :\n{self.b_output}")
                    
                    self.plot_error(errors_per_epoch, epoch + 1)
                    break
