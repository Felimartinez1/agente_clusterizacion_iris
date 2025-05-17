from clustering.preprocess import load_data, preprocess_data
from clustering.clustering import choose_k, cluster_data
from clustering.dimensionality import reduce_dimensions
from clustering.analysis import analyze_feature_importance, save_result
from visualization.visualizer import (
    visualize_clusters,
    plot_cluster_centers,
    plot_anova_boxplots
)
import argparse


def main(path_csv, is_3d, algorithm):
    print(f"[INFO] Cargando datos desde {path_csv}")
    df = load_data(path_csv)

    print("[INFO] Preprocesando datos...")
    X_processed, mask_missing, feature_cols, df_imputed = preprocess_data(df)

    if algorithm == "kmeans":
        k = choose_k(X_processed)
        print(f"[INFO] Clusters óptimos encontrados: {k}")
    else:
        k = None

    print("[INFO] Ejecutando clustering...")
    labels, model = cluster_data(X_processed, algorithm=algorithm, k=k)

    anova_results = analyze_feature_importance(df_imputed, labels, feature_cols)
    plot_anova_boxplots(df_imputed, labels, anova_results)

    save_result(df_imputed, labels, "outputs/resultado_cluster.csv")

    if X_processed.shape[1] >= 2:
        X_reduced = reduce_dimensions(X_processed, n_components=3 if is_3d else 2)
        visualize_clusters(X_reduced, labels, mask_missing, "outputs/clusters_visual.png", is_3d)

    if algorithm == "kmeans":
        plot_cluster_centers(model, feature_cols)

    print("[✔] Proceso finalizado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_csv", type=str)
    parser.add_argument("--is_3d", action='store_true')
    parser.add_argument("--algorithm", choices=["kmeans", "dbscan"], default="kmeans")
    args = parser.parse_args()
    main(args.path_csv, is_3d=args.is_3d, algorithm=args.algorithm)
