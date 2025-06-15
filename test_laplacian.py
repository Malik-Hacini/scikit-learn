from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_circles, make_moons
import numpy as np
import os
os.environ['MPLBACKEND'] = 'Qt5Agg'
import matplotlib.pyplot as plt
from time import time

 
# Generate circles dataset
X_circles, y_circles = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)

# Generate moons dataset
X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)

# Test spectral clustering on circles dataset
print("Testing Spectral Clustering on Circles Dataset")
time_start = time()
clustering_circles = SpectralClustering(n_clusters=2,
                                       standard=True,
                                       affinity='rbf',
                                       laplacian_method='unnorm',
                                       assign_labels='discretize',
                                       random_state=42).fit(X_circles)
time_end = time()
print("Time taken for clustering circles:", time_end - time_start)
print("Circles clustering labels:", clustering_circles.labels_)

# Test spectral clustering on moons dataset
print("\nTesting Spectral Clustering on Moons Dataset")
time_start = time()
clustering_moons = SpectralClustering(n_clusters=2,
                                     laplacian_method='random_walk',
                                     assign_labels='discretize',
                                     random_state=42).fit(X_moons)
time_end = time()
print("Time taken for clustering moons:", time_end - time_start)
print("Moons clustering labels:", clustering_moons.labels_)

# Plot the results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot circles - original data
axes[0, 0].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='viridis', alpha=0.7)
axes[0, 0].set_title('Circles Dataset - True Labels')
axes[0, 0].set_xlabel('Feature 1')
axes[0, 0].set_ylabel('Feature 2')

# Plot circles - spectral clustering results
axes[0, 1].scatter(X_circles[:, 0], X_circles[:, 1], c=clustering_circles.labels_, cmap='viridis', alpha=0.7)
axes[0, 1].set_title('Circles Dataset - Spectral Clustering')
axes[0, 1].set_xlabel('Feature 1')
axes[0, 1].set_ylabel('Feature 2')

# Plot moons - original data
axes[1, 0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis', alpha=0.7)
axes[1, 0].set_title('Moons Dataset - True Labels')
axes[1, 0].set_xlabel('Feature 1')
axes[1, 0].set_ylabel('Feature 2')

# Plot moons - spectral clustering results
axes[1, 1].scatter(X_moons[:, 0], X_moons[:, 1], c=clustering_moons.labels_, cmap='viridis', alpha=0.7)
axes[1, 1].set_title('Moons Dataset - Spectral Clustering')
axes[1, 1].set_xlabel('Feature 1')
axes[1, 1].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()