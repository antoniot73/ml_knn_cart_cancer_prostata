# Análisis, implementación y evaluación de KNN y CART para la clasificación de cáncer de próstata

**Repositorio:** https://github.com/antoniot73/ml_knn_cart_cancer_prostata  

**GitHub Pages:** https://antoniot73.github.io/ml_knn_cart_cancer_prostata/notebooks/practica_knn_cart_prostata.html  

**Binder:** https://mybinder.org/v2/gh/antoniot73/ml_knn_cart_cancer_prostata/main?filepath=notebooks/practica_knn_cart_prostata.ipynb

---

## Instituto Internacional de Aguascalientes

**Maestría en Inteligencia Artificial para la Transformación Digital**  
**Programa:** Aprendizaje Inteligente  
**Alumno:** Antonio Nicolás Toro González  
**Tutor:** Dr. Francisco Javier Luna Rosas

---

## Descripción

Este repositorio contiene la práctica **“Análisis, implementación y prueba de máquinas de aprendizaje supervisado”**, desarrollada con dos modelos de clasificación supervisada:

- **K-Nearest Neighbors (KNN)**
- **Classification and Regression Tree (CART)**

Ambos modelos se aplican al mismo problema de clasificación binaria: predecir si un registro corresponde a diagnóstico **benigno (B)** o **maligno (M)** de cáncer de próstata.

La práctica incluye marco teórico, carga del dataset, validación de datos, exploración visual, preparación de variables, entrenamiento de modelos, evaluación mediante métricas de clasificación, matrices de confusión, curva ROC, AUC, importancia de variables y visualización del árbol CART.

---

## Objetivo general

Implementar, evaluar y comparar dos algoritmos de aprendizaje supervisado, **KNN y CART**, sobre un dataset de cáncer de próstata, utilizando la curva ROC y el AUC como criterios principales de comparación.

---

## Objetivos específicos

- Analizar el funcionamiento de KNN y CART.
- Cargar y validar un dataset real de clasificación binaria.
- Preparar variables predictoras y variable objetivo.
- Entrenar ambos modelos bajo el mismo conjunto de prueba.
- Evaluar resultados con accuracy, precision, recall, F1-score y AUC.
- Comparar ambos modelos mediante curva ROC.
- Interpretar el modelo CART mediante importancia de variables y árbol de decisión.
- Generar un reporte HTML reproducible desde el notebook.

---

## Dataset

**Nombre:** Prostate Cancer Dataset  
**Autor:** Sajid Saifi  
**Fuente:** https://www.kaggle.com/datasets/sajidsaifi/prostate-cancer  

El dataset contiene variables morfológicas numéricas asociadas a registros de cáncer de próstata.

### Variable objetivo

- `diagnosis_result`
  - `B`: benigno
  - `M`: maligno

### Variables predictoras utilizadas

- `radius`
- `texture`
- `perimeter`
- `area`
- `smoothness`
- `compactness`
- `symmetry`
- `fractal_dimension`

---

## Tecnologías utilizadas

- Python 3.12+
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- pathlib
- logging

---

## Modelos implementados

### KNN

El modelo **K-Nearest Neighbors** clasifica registros según la cercanía geométrica entre observaciones. Debido a que depende de distancias, se aplicó escalamiento de variables mediante `StandardScaler`.

Componentes utilizados:

- `StandardScaler`
- `KNeighborsClassifier`
- `fit()`
- `predict()`
- `predict_proba()`

---

### CART

El modelo **Classification and Regression Tree** genera reglas jerárquicas de decisión. Su principal ventaja es la interpretabilidad, ya que permite observar nodos, condiciones, importancia de variables y estructura completa del árbol.

Componentes utilizados:

- `DecisionTreeClassifier`
- `plot_tree`
- importancia de variables
- visualización del árbol CART

---

## Evaluación

Los modelos se comparan con:

- Accuracy
- Precision
- Recall
- F1-score
- Matriz de confusión
- Curva ROC
- AUC

La curva ROC permite comparar los modelos sin depender de un único umbral de decisión. El AUC resume la capacidad global de discriminación entre clases.

---

## Resultados principales

| Modelo | Accuracy | Precision | Recall | F1-score | AUC |
|---|---:|---:|---:|---:|---:|
| KNN | 0.8500 | 0.8462 | 0.9167 | 0.8800 | 0.8542 |
| CART | 0.8000 | 0.8333 | 0.8333 | 0.8333 | 0.8125 |

### Interpretación general

KNN obtuvo el mejor rendimiento predictivo, con mayor AUC y recall. Esto indica mejor capacidad para discriminar entre casos benignos y malignos.

CART obtuvo un desempeño competitivo, pero su principal aporte fue la interpretabilidad mediante reglas de decisión, visualización de nodos e importancia de variables.

Variables destacadas en CART:

- `compactness`
- `perimeter`
- `area`

---

## Archivos generados

El notebook genera salidas en:

```text
outputs/
├── graficas/
└── tablas/
```

Incluye:

- distribución de clases
- matriz de correlación
- matriz de confusión KNN
- matriz de confusión CART
- curva ROC comparativa
- importancia de variables CART
- árbol CART con nodos
- métricas comparativas en CSV

---

## Estructura del repositorio

```text
ml_knn_cart_cancer_prostata/
│
├── data/
│   └── Prostate_Cancer.csv
│
├── notebooks/
│   ├── practica_knn_cart_prostata.ipynb
│   └── practica_knn_cart_prostata.html
│
├── outputs/
│   ├── graficas/
│   │   ├── arbol_cart_nodos.png
│   │   ├── curva_roc_knn_cart.png
│   │   ├── distribucion_clases_dataset.png
│   │   ├── importancia_variables_cart.png
│   │   ├── matriz_confusion_cart.png
│   │   ├── matriz_confusion_knn.png
│   │   └── matriz_correlacion.png
│   │
│   └── tablas/
│       ├── importancia_variables_cart.csv
│       ├── metricas_comparativas_knn_cart.csv
│       └── resumen_dataset.csv
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── modeling.py
│   ├── reporting.py
│   └── visualization.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Instalación

Crear entorno virtual:

```bash
python -m venv .venv
```

Activar entorno en Windows:

```bash
.venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Ejecución local

Abrir Jupyter:

```bash
jupyter notebook
```

Ejecutar:

```bash
practica_knn_cart_prostata.ipynb
```

Generar HTML:

```bash
jupyter nbconvert --to html --execute practica_knn_cart_prostata.ipynb
```

---

## Ejecución en Binder

Abrir:

https://hub.gesis.mybinder.org/user/antoniot73-ml_k-cancer_prostata-lu4sh758/notebooks/notebooks/practica_knn_cart_prostata.ipynb

---

## Publicación en GitHub Pages

El reporte HTML puede consultarse en:

https://antoniot73.github.io/ml_knn_cart_cancer_prostata/notebooks/practica_knn_cart_prostata.html

---

## Reproducibilidad

La práctica utiliza:

- semilla aleatoria fija
- separación entrenamiento/prueba
- validación de columnas
- revisión de valores faltantes
- salidas en rutas relativas
- exportación de gráficas y tablas

Esto permite ejecutar el proyecto tanto en entorno local como en cloud.

---

## Referencias

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., VanderPlas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). *Scikit-learn: Machine Learning in Python*. *Journal of Machine Learning Research, 12*, 2825–2830.

Saifi, S. (2018). *Prostate Cancer Dataset*. Kaggle. https://www.kaggle.com/datasets/sajidsaifi/prostate-cancer