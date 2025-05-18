# Agente Autónomo de Clusterización

Este script implementa un pipeline completo de clustering no supervisado con análisis exploratorio e imputación de datos, diseñado para funcionar de forma autónoma sobre un archivo `.csv`.

---

## Cómo correrlo
Desde un entorno con Python (>=3.8), instalá las dependencias necesarias:
```bash
python3.10 -m pip install -r requirements.txt
```

Luego ejecutá:
```bash
python3.10 cluster_agent.py data/iris_data_challenge.csv
```

Si quisieras una visualización de los clusters en 3D:
```bash
python3.10 cluster_agent.py data/iris_data_challenge.csv --is_3d
```

Si quisieras usar otro tipo de algoritmo como DBSCAN:
```bash
python3.10 cluster_agent.py data/iris_data_challenge.csv --algorithm dbscan
```

Si quisieras asignar un numero de clusters específico (ignorando la automatización reflejada en el código para elegir la mejor cantidad de clusters):
```bash
python3.10 cluster_agent.py data/iris_data_challenge.csv --k_clusters 4
```

## Decisiones de diseño principales
- Pipeline autónomo y modular: automatiza carga de raw dataset(pandas), selección de cantidad óptima de clusters (kmeans), imputación (KNN), normalización (StandardScaler), reducción de dimensionalidad (PCA) y análisis estadístico (ANOVA).

- Visualización interpretativa: los puntos originalmente faltantes se resaltan en rojo para facilitar su interpretación dentro de los clusters y debuggear KNN.

- Evaluación de calidad: incluye visualización de métodos Silhouette y Elbow para evaluar la elección de la cantidad de clusters.

- Análisis explicativo: genera mapas de calor con centroides (kmeans) y boxplots por feature con resultados estadísticos (ANOVA).

## Outputs
Todos los resultados del agente normal (sin Langchain) se guardan en la carpeta outputs/normal_agent/:

* resultado_cluster.csv: archivo original con columna cluster añadida.

* clusters_visual.png: visualización PCA 2D o 3D con puntos imputados marcados.

* silhouette_scores.png: gráfico de lineas con silhouette score por cantidad de clusters.

* elbow_method.png: gráfico de lineas con distorsión por cantidad de clusters.

* cluster_centers_heatmap.png: mapa de calor de los centroides.

* anova_boxplots_clusters.png: boxplots de cada variable según los clusters.

## Supuestos Asumidos

* Se utiliza python 3.10

* La imputación KNN es razonable para los datos faltantes, bajo la suposición de similitud local entre ejemplos, además no hay filas completamente nulas por lo tanto puede hacer bien su trabajo basandose en sus vecinos.

* Se asume que kmeans es adecuado cuando los clusters son aproximadamente convexos y bien separados.

* No se necesita columna objetivo o etiquetas: el análisis es puramente no supervisado.

## Principales hallazgos

* La imputación de valores faltantes con KNN no distorsionó significativamente la distribución general de los datos, lo cual puede observarse en la visualización 2D reducida con PCA.

* El agente logró identificar dos clusters bien diferenciados, como se evidencia en:

    * La separación visual clara en el espacio reducido de características (PCA).

    * Las diferencias marcadas en los centroides promedio por feature.

    * Los resultados del test ANOVA, que indicaron diferencias altamente significativas (p < 0.0001) en todas las variables según el cluster asignado.

* La principal distinción entre los clusters está en la longitud y el ancho de los pétalos:

    * Cluster 0 agrupa flores con pétalos cortos y estrechos.

    * Cluster 1 contiene flores con pétalos más largos y anchos.

* Las herramientas de visualización como boxplots y mapas de calor resultaron clave para validar la interpretación del modelo.
* Hay dos diferencias clave entre KMeans y DBSCAN:
    * DBSCAN resultó ser más efectivo a la hora de encontrar Outliers.
    * KMeans resultó performar mejor cuando se quiere trabajar con más clusters.