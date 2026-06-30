# Análisis, implementación y evaluación de KNN y CART para la clasificación de cáncer de próstata

**Universidad Internacional de Aguascalientes**  
**Maestría en Inteligencia Artificial para la Transformación Digital**  

**Asignatura:** Aprendizaje Inteligente  
**Unidad 2 – Aprendizaje Supervisado**

---

## Descripción

Este repositorio contiene el desarrollo de la práctica **"Análisis, implementación y prueba de máquinas de aprendizaje supervisado"**, cuyo objetivo es implementar y comparar dos algoritmos de clasificación supervisada:

- **K-Nearest Neighbors (KNN)**
- **Classification and Regression Tree (CART)**

Ambos modelos fueron entrenados y evaluados sobre un conjunto de datos real de **cáncer de próstata**, utilizando métricas de clasificación, matrices de confusión y la **Curva ROC (Receiver Operating Characteristic)** como principal criterio de comparación.

La implementación fue desarrollada completamente en **Python** utilizando la biblioteca **Scikit-learn**, siguiendo un flujo reproducible de ciencia de datos que incluye preparación del conjunto de datos, entrenamiento, evaluación e interpretación de resultados.

---

# Objetivos

- Analizar el funcionamiento de KNN y CART.
- Implementar ambos algoritmos mediante Scikit-learn.
- Comparar su desempeño utilizando métricas de clasificación.
- Evaluar la capacidad discriminativa mediante Curvas ROC y AUC.
- Interpretar cuantitativamente y visualmente los resultados obtenidos.

---

# Dataset

**Nombre:** Prostate Cancer Dataset

**Autor:** Sajid Saifi

**Fuente oficial**

https://www.kaggle.com/datasets/sajidsaifi/prostate-cancer

### Características

- Clasificación binaria.
- Diagnóstico:
  - Benigno (B)
  - Maligno (M)
- Variables morfológicas numéricas.
- Adecuado para algoritmos supervisados de clasificación.

---

# Tecnologías utilizadas

- Python 3.12+
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

# Algoritmos implementados

## K-Nearest Neighbors (KNN)

Implementado mediante:

- StandardScaler
- KNeighborsClassifier
- Matriz de confusión
- Curva ROC
- Área Bajo la Curva (AUC)

### Características

- Clasificación basada en vecinos más cercanos.
- Requiere escalamiento de variables.
- Muy adecuado para variables continuas.
- Excelente capacidad de discriminación en este problema.

---

## Árbol de Decisión (CART)

Implementado mediante:

- DecisionTreeClassifier
- Matriz de confusión
- Curva ROC
- Área Bajo la Curva (AUC)
- Importancia de variables
- Visualización completa del árbol

### Características

- Modelo interpretable.
- Genera reglas jerárquicas.
- Permite visualizar nodos de decisión.
- Calcula importancia relativa de cada variable.

---

# Flujo de trabajo

1. Introducción.
2. Marco teórico.
3. Carga del dataset.
4. Exploración inicial del dataset.
5. Visualización del dataset mediante ventana con scroll.
6. Preparación de datos.
7. División entrenamiento/prueba.
8. Escalamiento de variables.
9. Entrenamiento del modelo KNN.
10. Entrenamiento del modelo CART.
11. Evaluación mediante:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Matriz de confusión
   - Curva ROC
   - Área Bajo la Curva (AUC)
12. Comparación de resultados.
13. Discusión.
14. Conclusiones.

---

# Resultados generados

El notebook genera automáticamente:

- Exploración interactiva del dataset.
- Estadísticas descriptivas.
- Tabla comparativa de métricas.
- Matriz de confusión de KNN.
- Matriz de confusión de CART.
- Curva ROC comparativa.
- Cálculo del AUC.
- Importancia de variables.
- Árbol CART con todos sus nodos.
- Exportación automática de tablas y figuras.

---

# Estructura del proyecto

```text
ml_knn_cart_cancer_prostata/
│
├── practica_knn_cart_prostata.ipynb
├── practica_knn_cart_prostata.html
├── README.md
├── requirements.txt
├── data/
│   └── prostate_cancer.csv
│
├── outputs/
│   ├── graficas/
│   ├── tablas/
│   └── modelos/
│
└── figures/
```

---

# Instalación

Crear entorno virtual (opcional)

```bash
python -m venv .venv
```

Activar entorno

Windows

```bash
.venv\Scripts\activate
```

Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# Ejecución

Abrir el notebook:

```bash
jupyter notebook
```

o

```bash
jupyter lab
```

Para generar automáticamente el reporte HTML:

```bash
jupyter nbconvert --to html --execute practica_knn_cart_prostata.ipynb
```

---

# Repositorio

**GitHub**

https://github.com/antoniot73/ml_knn_cart_cancer_prostata

**GitHub Pages**

https://antoniot73.github.io/ml_knn_cart_cancer_prostata/practica_knn_cart_prostata.html

**Binder**

https://mybinder.org/v2/gh/antoniot73/ml_knn_cart_cancer_prostata/main?filepath=practica_knn_cart_prostata.ipynb

---

# Resultados principales

La comparación mostró que:

- **KNN obtuvo el mejor desempeño global**, alcanzando un **AUC = 0.8542**, lo que indica mayor capacidad para discriminar entre casos benignos y malignos.

- **CART obtuvo un AUC = 0.8125**, ofreciendo un rendimiento competitivo y una mayor interpretabilidad mediante reglas de decisión y visualización del árbol.

- La **Curva ROC** permitió comparar ambos modelos bajo distintos umbrales de clasificación, confirmando la superioridad de KNN en rendimiento predictivo y la utilidad de CART como modelo explicativo.

---

# Bibliotecas principales

- pandas
- numpy
- matplotlib
- scikit-learn

---

# Referencias

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., VanderPlas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). *Scikit-learn: Machine Learning in Python*. *Journal of Machine Learning Research, 12*, 2825–2830.

Saifi, S. (2018). *Prostate Cancer Dataset*. Kaggle. https://www.kaggle.com/datasets/sajidsaifi/prostate-cancer