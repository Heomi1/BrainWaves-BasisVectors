"""3D geometric view of PCA on a toy 3-electrode EEG covariance matrix.

Fp1 and Fp2 are strongly correlated (an eye blink seen by both frontal
electrodes), O1 is nearly independent. The eigenvector with the largest
eigenvalue is the "blink" direction; zeroing that component and projecting
back removes the artifact while keeping the rest of the signal.
"""

import numpy as np
import matplotlib.pyplot as plt

# Toy covariance matrix (channels: Fp1, Fp2, O1)
cov = np.array([[100, 95, 0],
                [95, 100, 0],
                [0, 0, 2]])
mean = [0, 0, 0]

# eigh returns eigenvalues in ascending order -> index 2 is the largest
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# 1000 time samples drawn with exactly this covariance geometry
rng = np.random.default_rng(42)
data = rng.multivariate_normal(mean, cov, 1000)

# Change of basis -> remove the principal (blink) component -> change back
data_pca = data @ eigenvectors
data_pca[:, 2] = 0
data_clean = data_pca @ eigenvectors.T

colors = ["green", "orange", "red"]
names = ["PC3 (brain waves)", "PC2 (local noise)", "PC1 (eye blink)"]
labels = [f"{n}: λ={v:.0f}" for n, v in zip(names, eigenvalues)]

fig = plt.figure(figsize=(14, 6))
for k, (points, color, title) in enumerate([
        (data, "purple", "Raw EEG samples"),
        (data_clean, "blue", "After removing PC1 (blink)")]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    ax.scatter(*points.T, alpha=0.3, color=color, label="Time samples")
    for i in range(3):
        vec = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 2  # scaled for visibility
        ax.quiver(0, 0, 0, *vec, color=colors[i], linewidth=4, label=labels[i])
    ax.set_xlabel("Fp1 voltage")
    ax.set_ylabel("Fp2 voltage")
    ax.set_zlabel("O1 voltage")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

plt.suptitle("3D geometric view of EEG covariance and its eigenvectors")
plt.tight_layout()
plt.show()
