import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
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

def estimate_eps(X, output_prefix, min_samples=2):
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X)
    distances, _ = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, -1])  # Tomamos la distancia al k-ésimo vecino

    # Graficar
    plt.figure()
    plt.plot(distances)
    plt.xlabel("Puntos ordenados")
    plt.ylabel(f"Distancia al {min_samples}º vecino más cercano")
    plt.grid(True)
    plt.savefig(os.path.join(output_prefix, "k_distance_plot.png"))
    plt.close()

    # Heurística: codo en la curva
    # Buscamos el punto con mayor pendiente relativa (dif máxima)
    diffs = np.diff(distances)
    eps_estimate = distances[np.argmax(diffs)]

    return eps_estimate

def cluster_data(X, output_prefix, algorithm="kmeans", k=None):
    if algorithm == "kmeans":
        model = KMeans(n_clusters=k, random_state=42)
    elif algorithm == "dbscan":
        min_samples = 2
        eps = estimate_eps(X, output_prefix, min_samples=min_samples)
        model = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        raise ValueError(f"Algoritmo no soportado: {algorithm}")
    labels = model.fit_predict(X)
    return labels, model
