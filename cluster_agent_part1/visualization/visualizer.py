import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_clusters(X_reduced, labels, mask_missing, output_path, is_3d=False):
    if is_3d and X_reduced.shape[1] >= 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=labels, cmap='viridis', s=50)
        for i in range(X_reduced.shape[0]):
            if mask_missing[i].any():
                ax.scatter(X_reduced[i, 0], X_reduced[i, 1], X_reduced[i, 2], facecolors='none', edgecolors='red', s=100)
    else:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', s=50)
        plt.scatter(X_reduced[mask_missing.any(axis=1), 0],
                    X_reduced[mask_missing.any(axis=1), 1],
                    facecolors='none', edgecolors='red', s=100)
    plt.savefig(output_path)
    plt.close()

def plot_cluster_centers(model, feature_names):
    if not hasattr(model, "cluster_centers_"):
        print("[AVISO] El modelo no tiene centroides.")
        return
    centers_df = pd.DataFrame(model.cluster_centers_, columns=feature_names)
    centers_df.index = [f"Cluster {i}" for i in range(centers_df.shape[0])]
    plt.figure(figsize=(10, 6))
    sns.heatmap(centers_df, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Valores promedio por cluster")
    plt.savefig("outputs/cluster_centers_heatmap.png")
    plt.close()

def plot_anova_boxplots(df, labels, anova_results):
    df_plot = df.copy()
    df_plot['cluster'] = labels
    variables = list(anova_results.keys())

    plt.figure(figsize=(16, 12))
    for i, var in enumerate(variables, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x='cluster', y=var, data=df_plot, palette='Set2')
        plt.title(f"{var}\nF={anova_results[var]['F']:.3f}, p={anova_results[var]['p']:.4f}")
    plt.tight_layout()
    plt.savefig("outputs/anova_boxplots_clusters.png")
    plt.close()
