import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    numeric_df = df.select_dtypes(include=[np.number])
    mask_missing = numeric_df.isnull()

    imputer = KNNImputer(n_neighbors=5)
    imputed = imputer.fit_transform(numeric_df)
    imputed = np.round(imputed, 1)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)

    df_imputed = df.copy()
    df_imputed[numeric_df.columns] = imputed

    return scaled, mask_missing.values, numeric_df.columns.tolist(), df_imputed
