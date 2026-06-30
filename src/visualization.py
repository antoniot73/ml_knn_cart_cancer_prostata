"""Funciones de visualización para la práctica KNN/CART."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree

from src.dataset import FEATURE_COLUMNS
from src.modeling import ModelResult


def ensure_output_dir(output_dir: Path) -> None:
    """Crea el directorio de salida si no existe."""
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Grafica la distribución de clases benignas y malignas."""
    ensure_output_dir(output_path.parent)
    counts = df["diagnosis_result"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("Clase diagnóstica")
    plt.ylabel("Número de registros")
    plt.title("Distribución de clases: cáncer de próstata")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, output_path: Path) -> None:
    """Grafica la matriz de correlación entre variables predictoras."""
    ensure_output_dir(output_path.parent)
    corr = df[list(FEATURE_COLUMNS)].corr()
    plt.figure(figsize=(8, 6))
    image = plt.imshow(corr, interpolation="nearest")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS, rotation=45, ha="right")
    plt.yticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS)
    plt.title("Matriz de correlación de variables predictoras")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_importance(feature_table: pd.DataFrame, output_path: Path) -> None:
    """Grafica la importancia de variables calculada por CART."""
    ensure_output_dir(output_path.parent)
    plt.figure(figsize=(8, 5))
    plt.barh(feature_table["variable"], feature_table["importancia"])
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.title("Importancia de variables según CART")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix(result: ModelResult, output_path: Path) -> None:
    """Grafica la matriz de confusión de un modelo."""
    ensure_output_dir(output_path.parent)
    display = ConfusionMatrixDisplay(
        confusion_matrix=result.confusion,
        display_labels=["B", "M"],
    )
    display.plot(values_format="d")
    plt.title(f"Matriz de confusión: {result.name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_curves(results: list[ModelResult], output_path: Path) -> None:
    """Grafica curvas ROC comparativas para varios modelos."""
    ensure_output_dir(output_path.parent)
    plt.figure(figsize=(7, 5))
    for result in results:
        plt.plot(result.fpr, result.tpr, label=f"{result.name} (AUC = {result.roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Clasificador aleatorio (AUC = 0.500)")
    plt.xlabel("Tasa de falsos positivos (FPR)")
    plt.ylabel("Tasa de verdaderos positivos (TPR)")
    plt.title("Curva ROC comparativa: KNN vs CART")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_cart_tree(cart_model: object, output_path: Path) -> None:
    """Grafica el árbol de decisión entrenado."""
    ensure_output_dir(output_path.parent)
    plt.figure(figsize=(14, 8))
    plot_tree(
        cart_model,
        feature_names=list(FEATURE_COLUMNS),
        class_names=["B", "M"],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title("Árbol de decisión CART")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
