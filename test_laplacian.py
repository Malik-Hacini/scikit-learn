from sklearn.cluster import SpectralClustering
import numpy as np
from time import time



X = np.array([[1, 1], [2, 1], [1, 0],
                  [4, 7], [3, 5], [3, 6]])

time_start = time()

clustering = SpectralClustering(n_clusters=2,
             laplacian_method='random_walk',
             assign_labels='discretize',
             random_state=0).fit(X)
time_end = time()
print("Time taken for clustering:", time_end - time_start)
print(clustering.labels_)