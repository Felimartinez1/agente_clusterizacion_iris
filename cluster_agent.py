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


def main(path_csv, output_prefix, is_3d, algorithm, k_clusters=None):
    print(f"[INFO] Cargando datos desde {path_csv}")
    df = load_data(path_csv)

    print("[INFO] Preprocesando datos...")
    X_processed, mask_missing, feature_cols, df_imputed = preprocess_data(df)

    if algorithm == "kmeans":
        if k_clusters is None:
            k = choose_k(X_processed, output_prefix)
            print(f"[INFO] Clusters óptimos encontrados automáticamente: {k}")
        else:
            k = k_clusters
            print(f"[INFO] Usando número de clusters proporcionado: {k}")
    else:
        k = None

    print("[INFO] Ejecutando clustering...")
    labels, model = cluster_data(X_processed, output_prefix, algorithm=algorithm, k=k)
    
    anova_results = analyze_feature_importance(df_imputed, labels, feature_cols)
    plot_anova_boxplots(df_imputed, labels, anova_results, output_prefix)

    output_path = save_result(df_imputed, labels, output_prefix)

    if X_processed.shape[1] >= 2:
        X_reduced = reduce_dimensions(X_processed, n_components=3 if is_3d else 2)
        visualize_clusters(X_reduced, labels, mask_missing, output_prefix, is_3d)

    if algorithm == "kmeans":
        plot_cluster_centers(model, feature_cols, output_prefix)

    print("[INFO] Proceso finalizado.")

    # Crear resumen simple del proceso
    resumen = f"""
    Clustering realizado con {algorithm.upper()}
    Cantidad de puntos: {len(df_imputed)}
    Clusters generados: {len(set(labels)) if algorithm != 'dbscan' else len(set(labels)) - (1 if -1 in labels else 0)}
    Columnas utilizadas: {', '.join(feature_cols)}
    Archivo de salida: {output_path}
    """

    return resumen.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_csv", type=str)
    parser.add_argument("--output_prefix", type=str, default="outputs/normal_agent/")
    parser.add_argument("--is_3d", action='store_true')
    parser.add_argument("--algorithm", choices=["kmeans", "dbscan"], default="kmeans")
    parser.add_argument("--k_clusters", type=int, default=None, help="Número de clusters (solo para KMeans)")
    args = parser.parse_args()
    main(args.path_csv, args.output_prefix, is_3d=args.is_3d, algorithm=args.algorithm, k_clusters=args.k_clusters)
