"""Carga, validación y visualización del dataset de cáncer de próstata."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "id",
    "diagnosis_result",
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "symmetry",
    "fractal_dimension",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "symmetry",
    "fractal_dimension",
)

TARGET_COLUMN = "diagnosis_result"


def configure_logging() -> None:
    """Configura una bitácora básica para reportar el flujo de ejecución."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Carga el dataset desde un archivo CSV local.

    Args:
        csv_path: Ruta del archivo CSV.

    Returns:
        DataFrame con los registros cargados.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo está vacío.
    """
    if not isinstance(csv_path, Path):
        raise TypeError("csv_path debe ser un objeto pathlib.Path.")
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {csv_path}")

    LOGGER.info("Cargando dataset: %s", csv_path)
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("El dataset está vacío.")

    validate_dataset(df)
    LOGGER.info("Dataset validado: %d registros, %d columnas.", df.shape[0], df.shape[1])
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Valida columnas requeridas, valores faltantes y clases esperadas.

    Args:
        df: DataFrame a validar.

    Raises:
        TypeError: Si df no es un DataFrame.
        ValueError: Si faltan columnas, hay nulos o clases no esperadas.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df debe ser un pandas.DataFrame.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")

    null_counts = df[list(REQUIRED_COLUMNS)].isna().sum()
    if int(null_counts.sum()) > 0:
        raise ValueError(f"Existen valores faltantes: {null_counts[null_counts > 0].to_dict()}")

    classes = set(df[TARGET_COLUMN].astype(str).str.upper().unique())
    expected_classes = {"M", "B"}
    if classes != expected_classes:
        raise ValueError(f"Clases esperadas {expected_classes}; clases encontradas: {classes}")

    for column in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"La columna predictora {column} debe ser numérica.")


def split_dataframe_blocks(df: pd.DataFrame, block_size: int = 10) -> list[pd.DataFrame]:
    """Divide el DataFrame en bloques de registros para visualización controlada.

    Args:
        df: DataFrame original.
        block_size: Número de registros por bloque.

    Returns:
        Lista de DataFrames con bloques consecutivos.

    Raises:
        ValueError: Si block_size es menor que 1.
    """
    if block_size < 1:
        raise ValueError("block_size debe ser mayor o igual que 1.")

    blocks: list[pd.DataFrame] = []
    for start in range(0, len(df), block_size):
        blocks.append(df.iloc[start : start + block_size].copy())
    return blocks


def build_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Construye una tabla resumen del dataset.

    Args:
        df: Dataset validado.

    Returns:
        DataFrame con características generales del dataset.
    """
    return pd.DataFrame(
        {
            "Característica": [
                "Dominio",
                "Tipo de aprendizaje",
                "Tipo de problema",
                "Número de registros",
                "Número de columnas",
                "Variable objetivo",
                "Clases",
                "Variables predictoras",
            ],
            "Descripción": [
                "Diagnóstico médico: cáncer de próstata",
                "Supervisado",
                "Clasificación binaria",
                str(df.shape[0]),
                str(df.shape[1]),
                TARGET_COLUMN,
                "M = maligno; B = benigno",
                ", ".join(FEATURE_COLUMNS),
            ],
        }
    )
