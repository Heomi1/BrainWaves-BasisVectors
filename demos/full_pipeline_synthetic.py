"""Complete pipeline on synthetic data: PCA spatial filtering + FFT.

A 10 Hz alpha wave is injected into the O1 channel of correlated
3-channel noise. We remove the dominant (blink-like) principal component,
then compare the O1 power spectrum before and after cleaning.
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. Covariance matrix (channels: Fp1, Fp2, O1).
#    O1 is weakly correlated with the frontal channels. Must be positive
#    semi-definite to be a valid covariance matrix.
cov = np.array([[100, 95, 25],
                [95, 100, 20],
                [25, 20, 20]])
mean = [0, 0, 0]
eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending order

# 2. Time axis
fs = 100                     # sampling rate [Hz]
t = np.arange(1000) / fs     # 10 seconds

# 3. Correlated noise + 10 Hz alpha rhythm on O1 (index 2)
rng = np.random.default_rng(42)
data = rng.multivariate_normal(mean, cov, len(t))
data[:, 2] += 1.5 * np.cos(2 * np.pi * 10 * t)

# 4. Spatial filtering: change basis, zero the principal component, go back
data_pca = data @ eigenvectors
data_pca[:, 2] = 0
data_clean = data_pca @ eigenvectors.T


# 5. Fourier transform -> power spectral density
def psd(signal, fs):
    """One-sided power spectrum |X_k|^2 for positive frequencies."""
    values = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1 / fs)
    return freqs[1:], np.abs(values[1:]) ** 2


freqs, psd_raw = psd(data[:, 2], fs)
_, psd_clean = psd(data_clean[:, 2], fs)

# 6. Visualisation
fig = plt.figure(figsize=(18, 5))
colors = ["green", "orange", "red"]
names = ["PC3", "PC2", "PC1 (blink)"]

for k, (points, color, title) in enumerate([
        (data, "purple", "Raw EEG"),
        (data_clean, "blue", "Cleaned EEG (PC1 removed)")]):
    ax = fig.add_subplot(1, 4, k + 1, projection="3d")
    ax.scatter(*points.T, alpha=0.3, color=color)
    for i in range(3):
        vec = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 2
        ax.quiver(0, 0, 0, *vec, color=colors[i], linewidth=4,
                  label=f"{names[i]}: λ={eigenvalues[i]:.1f}")
    ax.set_title(title)
    ax.set_xlabel("Fp1"); ax.set_ylabel("Fp2"); ax.set_zlabel("O1")
    ax.legend(fontsize=7)

for k, (spectrum, title) in enumerate([
        (psd_raw, "O1 spectrum before cleaning"),
        (psd_clean, "O1 spectrum after cleaning")]):
    ax = fig.add_subplot(1, 4, k + 3)
    ax.plot(freqs, spectrum, color="darkred")
    ax.axvline(10, color="gray", linestyle="--", alpha=0.6, label="10 Hz alpha")
    ax.set_xlim(0, 30)
    ax.set_title(title)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power")
    ax.legend()

plt.tight_layout()
plt.show()
