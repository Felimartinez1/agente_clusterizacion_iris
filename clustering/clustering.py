import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os

def choose_k(X, output_prefix, k_max=10):
    scores, distortions = [], []

    for k in range(2, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        scores.append(silhouette_score(X, labels))
        distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)))

    best_k = np.argmax(scores) + 2

    plt.figure()
    plt.plot(range(2, 2 + len(scores)), scores, marker='o')
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.savefig(os.path.join(output_prefix, "silhouette_scores.png"))
    plt.close()

    plt.figure()
    plt.plot(range(2, 2 + len(distortions)), distortions, marker='o')
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Distorsión")
    plt.grid(True)
    plt.savefig(os.path.join(output_prefix, "elbow_method.png"))
    plt.close()

    return best_k

def cluster_data(X, algorithm="kmeans", k=None):
    if algorithm == "kmeans":
        model = KMeans(n_clusters=k, random_state=42)
    elif algorithm == "dbscan":
        model = DBSCAN(eps=0.5, min_samples=2)
    else:
        raise ValueError(f"Algoritmo no soportado: {algorithm}")
    labels = model.fit_predict(X)
    return labels, model
