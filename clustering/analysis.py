import pandas as pd
from scipy.stats import f_oneway
import os

def analyze_feature_importance(df, labels, feature_cols):
    df_clustered = df.copy()
    df_clustered['cluster'] = labels
    anova_results = {}
    for feature in feature_cols:
        groups = [group[feature].values for name, group in df_clustered.groupby('cluster')]
        f_stat, p_val = f_oneway(*groups)
        anova_results[feature] = {"F": f_stat, "p": p_val}
    return anova_results

def save_result(df_original, labels, output_prefix):
    df_result = df_original.copy()
    df_result['cluster'] = labels
    output_path = os.path.join(output_prefix, "resultado_cluster.csv")
    df_result.to_csv(output_path, index=False)
    return output_path
