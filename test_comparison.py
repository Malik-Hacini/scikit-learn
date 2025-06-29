# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import time
import warnings
from itertools import cycle, islice

import matplotlib.pyplot as plt
import os
os.environ['MPLBACKEND'] = 'Qt5Agg'
import numpy as np

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 500
seed = 30
noisy_circles = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
rng = np.random.RandomState(seed)
no_structure = rng.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(9 * 2 + 3, 13))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
)

plot_num = 1

default_base = {
    "n_neighbors": 6,
    "n_clusters": 3,
    "random_state": 42,
}
datasets = [
    (
        noisy_circles,
        {
            "n_clusters": 2,
        },
    ),
    (
        noisy_moons,
        {
            "n_clusters": 2,
            "n_neighbors": 7,
        },
    ),
    (
        varied,{},
    ),
    (
       aniso,{},
    ),
    (blobs, {}),
    (no_structure, {}),
]

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # ============
    # Create cluster objects
    # ============
    
    standard_spectral = cluster.SpectralClustering(
        n_clusters=params["n_clusters"],
        standard=True,
        affinity="nearest_neighbors",
        laplacian_method="norm",
        random_state=params["random_state"],
    )
    
    gsc = cluster.SpectralClustering(
        n_clusters=params["n_clusters"],
        n_neighbors=params["n_neighbors"],
        affinity="nearest_neighbors",       
        laplacian_method="unnorm",
        random_state=params["random_state"])

    normalized_gsc= cluster.SpectralClustering(
        n_clusters=params["n_clusters"],
        n_neighbors=params["n_neighbors"],
        affinity="nearest_neighbors",
        laplacian_method="norm",
        random_state=params["random_state"])
    
    random_walk_gsc = cluster.SpectralClustering(
        n_clusters=params["n_clusters"],
        n_neighbors=params["n_neighbors"],
        affinity="nearest_neighbors",
        laplacian_method="random_walk",
        random_state=params["random_state"])

    clustering_algorithms = (
        ("Normalized SC", standard_spectral),
        ("Unnormalized GSC", gsc),
        ("Normalized GSC", normalized_gsc),
        ("Random Walk GSC", random_walk_gsc),
    )

    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the "
                "connectivity matrix is [0-9]{1,2}"
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding"
                " may not work as expected.",
                category=UserWarning,
            )
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(
            0.99,
            0.01,
            ("%.2fs" % (t1 - t0)).lstrip("0"),
            transform=plt.gca().transAxes,
            size=15,
            horizontalalignment="right",
        )
        plot_num += 1

plt.show()