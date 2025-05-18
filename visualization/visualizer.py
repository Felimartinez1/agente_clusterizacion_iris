import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import os

def visualize_clusters(X_reduced, labels, mask_missing, output_prefix, is_3d=False):
    if is_3d and X_reduced.shape[1] >= 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=labels, cmap='viridis', s=50)
        for i in range(X_reduced.shape[0]):
            if mask_missing[i].any():
                ax.scatter(X_reduced[i, 0], X_reduced[i, 1], X_reduced[i, 2], facecolors='none', edgecolors='red', s=100)
        ax.set_title("Visualización de Clusters (3D)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
    else:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', s=50, label="Original")
        plt.scatter(X_reduced[mask_missing.any(axis=1), 0],
                    X_reduced[mask_missing.any(axis=1), 1],
                    facecolors='none', edgecolors='red', s=100, linewidths=1.5, label="Imputados")
        plt.title("Visualización de Clusters (2D) con imputación KNN")
        plt.xlabel("Componente principal 1")
        plt.ylabel("Componente principal 2")
        plt.grid(True)
        plt.legend(loc='upper right')
    plt.savefig(os.path.join(output_prefix, "clusters_visual.png"))
    plt.close()

def plot_cluster_centers(model, feature_names, output_prefix):
    if not hasattr(model, "cluster_centers_"):
        print("[AVISO] El modelo no tiene centroides (no es KMeans).")
        return

    centers_df = pd.DataFrame(model.cluster_centers_, columns=feature_names)
    centers_df.index = [f"Cluster {i}" for i in range(centers_df.shape[0])]

    plt.figure(figsize=(10, 6))
    sns.heatmap(centers_df, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Valores promedio de cada feature por cluster (centroides)")
    plt.ylabel("Clusters")
    plt.xlabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(output_prefix, "cluster_centers_heatmap.png"))
    plt.close()


def plot_anova_boxplots(df, labels, anova_results, output_prefix):
    df_plot = df.copy()
    df_plot['cluster'] = labels
    variables = list(anova_results.keys())

    plt.figure(figsize=(16, 12))
    for i, var in enumerate(variables, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x='cluster', y=var, data=df_plot, palette='Set2')
        plt.title(f"{var}\nF={anova_results[var]['F']:.3f}, p={anova_results[var]['p']:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_prefix, "anova_boxplots_clusters.png"))
    plt.close()
