# NLP Cluster Agent

Este módulo implementa un **agente de clusterización controlado por lenguaje natural**. El objetivo es permitir que un usuario describa con palabras una instrucción de clustering, y el agente interprete esa instrucción, ejecute el análisis correspondiente y devuelva los resultados.

## Cómo correrlo
Desde un entorno con Python (>=3.8), instalá las dependencias necesarias:
```bash
python3.10 -m pip install -r requirements.txt
```

Luego ejecutá:
```bash
python3.10 nlp_cluster_agent.py data/iris_data_challenge.csv
```

Una vez ejecutado se te pedirá un prompt en el cual debes indicar instrucciones para el cluster como por ejemplo:
```bash
-Clusterizá el dataset con KMeans en 3D
-Clusterizá los datos usando DBSCAN
-Clusterizá usando 4 clusters con KMeans y mostrá en 3D
```
El agente va a interpretar la instrucción, correr el análisis, y mostrarte los resultados en el directorio outputs/nlp_agent/.

## Estructura
- `nlp_cluster_agent.py`: Script principal que contiene el agente.
- `cluster_agent.py`: Contiene la función `main()` que ejecuta el pipeline de clusterización (desde la Parte 1).
- `FakeSimpleLLM`: Clase que simula un modelo de lenguaje interpretando instrucciones con reglas básicas.
- `tools`: Herramientas expuestas al agente, incluyendo la ejecución del pipeline de clustering.

## Outputs
Todos los resultados del agente de clusterización con insturcciones con nlp se guardan en la carpeta outputs/nlp_agent/:

* resultado_cluster.csv: archivo original con columna cluster añadida.

* clusters_visual.png: visualización PCA 2D o 3D con puntos imputados marcados.

* silhouette_scores.png: gráfico de lineas con silhouette score por cantidad de clusters.

* elbow_method.png: gráfico de lineas con distorsión por cantidad de clusters.

* cluster_centers_heatmap.png: mapa de calor de los centroides.

* anova_boxplots_clusters.png: boxplots de cada variable según los clusters.

* k_distance_plot.png: gráfico de líneas que muestra el mejor eps posible.

## Supuestos Asumidos
* Se utiliza python 3.10
* Se deben mencionar ciertas palabras como "clusteriza" para que el modelo entienda que debe usar los parametros que se le piden.

## Diagrama de Flujo

```text
+---------+            +-------------------+              
| Usuario |            |     Agente        |              
|         | ───────▶  | (LangChain Agent) |     
+---------+            |   + LLM (Fake)    |              
       ▲               +-------------------+               +-------------------------+ 
       │                        │                          |       Herramienta       |         
       │                        ▼                SI        | ejecutar_cluster_agente |        
       │                ¿Instrucción válida? ───────────▶ |  (main clustering func) |  
       │                (interpreta el texto)              +-------------------------+   
       │                        │                                   │
       │                     NO │                                   │
       │                        │                                   │ 
       │                        ▼                                   │
       └────────────── Muestra respuesta final al usuario ◀────────┘
```
       
1. Usuario escribe una instrucción (ej. "Clusterizá el dataset en 3D con DBSCAN").

2. El Agente recibe el input y lo pasa al LLM simulado (FakeSimpleLLM).

3. El LLM interpreta el texto y:
    * Si detecta términos como “cluster”, “dbscan”, “kmeans”, etc., construye una llamada a herramienta (tool_calls).

    * Si no entiende, devuelve un mensaje como "No entiendo la instrucción.".

4. Si hay una llamada válida, el Agente ejecuta la herramienta ejecutar_cluster_agente.

5. La herramienta corre la función main() que hace clustering.

6. El output (logs o resumen del clustering) se devuelve al agente.

7. El agente muestra la salida al Usuario mediante el directorio outputs/nlp_agent/.

## Posibles mejoras - Next steps
- Reemplazar FakeSimpleLLM por un modelo real (como GPT-4, LLaMA, Mistral, etc.) y así hacerlo más flexible y natural, sin que el modelo dependa de la mención de una palabra hardcodeada.
- En el algoritmo DBSCAN, encontrar el numero óptimo de min_samples con un grid search.
- Desarrollar una API.
- Hacer un diseño con buena UX.
- Desarrollar un frontend.
- Dockerizar.
- CI/CD.
- Deploy en cloud.
- Administrar containers una vez el proyecto sea lo suficientemente grande.