"""Modelado, entrenamiento y evaluación de KNN y CART."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.dataset import FEATURE_COLUMNS, TARGET_COLUMN

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelResult:
    """Contenedor inmutable para resultados de evaluación de un modelo."""

    name: str
    y_true: object
    y_pred: object
    y_score: object
    fpr: object
    tpr: object
    roc_auc: float
    confusion: object
    metrics: dict[str, float]


def prepare_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, object, LabelEncoder]:
    """Separa variables predictoras y codifica la clase objetivo.

    Args:
        df: Dataset validado.

    Returns:
        Tupla con X, y codificada y codificador de etiquetas.
    """
    x_data = df.loc[:, FEATURE_COLUMNS].copy()
    encoder = LabelEncoder()
    y_data = encoder.fit_transform(df[TARGET_COLUMN].astype(str).str.upper())
    # LabelEncoder ordena alfabéticamente: B=0, M=1.
    return x_data, y_data, encoder


def build_train_test_split(
    x_data: pd.DataFrame,
    y_data: object,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[object, object, object, object]:
    """Divide los datos en entrenamiento y prueba de forma estratificada."""
    if not 0 < test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1.")

    return train_test_split(
        x_data,
        y_data,
        test_size=test_size,
        random_state=random_state,
        stratify=y_data,
    )


def build_knn_model(n_neighbors: int = 5) -> Pipeline:
    """Construye el pipeline de KNN con estandarización previa."""
    if n_neighbors < 1:
        raise ValueError("n_neighbors debe ser mayor o igual que 1.")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ]
    )


def build_cart_model(random_state: int = 42, max_depth: int = 4) -> DecisionTreeClassifier:
    """Construye el clasificador CART con control de profundidad."""
    if max_depth < 1:
        raise ValueError("max_depth debe ser mayor o igual que 1.")

    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        random_state=random_state,
    )


def evaluate_model(name: str, model: object, x_test: object, y_test: object) -> ModelResult:
    """Evalúa un modelo entrenado mediante métricas y curva ROC.

    Args:
        name: Nombre del modelo.
        model: Modelo entrenado con métodos predict y predict_proba.
        x_test: Variables predictoras de prueba.
        y_test: Etiquetas reales de prueba.

    Returns:
        Resultados estructurados del modelo.
    """
    if not hasattr(model, "predict"):
        raise TypeError("El modelo debe implementar predict().")
    if not hasattr(model, "predict_proba"):
        raise TypeError("El modelo debe implementar predict_proba().")

    LOGGER.info("Evaluando modelo: %s", name)
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    confusion = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc,
    }

    return ModelResult(
        name=name,
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc,
        confusion=confusion,
        metrics=metrics,
    )


def build_metrics_table(results: list[ModelResult]) -> pd.DataFrame:
    """Construye una tabla comparativa de métricas para varios modelos."""
    records = []
    for result in results:
        row = {"modelo": result.name}
        row.update(result.metrics)
        records.append(row)
    return pd.DataFrame(records)


def build_feature_importance_table(cart_model: DecisionTreeClassifier) -> pd.DataFrame:
    """Obtiene la importancia de variables calculada por CART."""
    importances = cart_model.feature_importances_
    table = pd.DataFrame(
        {
            "variable": list(FEATURE_COLUMNS),
            "importancia": importances,
        }
    ).sort_values("importancia", ascending=False, ignore_index=True)
    return table
