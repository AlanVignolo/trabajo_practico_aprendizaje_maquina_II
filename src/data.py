"""Carga del CSV crudo y limpieza estructural previa al split.

Estos pasos se aplican sobre TODO el dataset porque no dependen de
estadísticos derivados de los datos (no hay leakage).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Lee el CSV crudo del dataset principal."""
    if path is None:
        path = config.DATASETS_DIR / config.RAW_DATASET_FILENAME
    path = Path(path)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def clean_structural(
    df: pd.DataFrame,
    *,
    current_year: int | None = None,
) -> pd.DataFrame:
    """Aplica los pasos deterministas previos al split.

    - Crea `Age = current_year - Year` y dropea `Year`.
    - Castea columnas categóricas a `category`.
    - Extrae el número de las columnas con unidad textual
      (Engine, Max Power, Max Torque).
    """
    df = df.copy()

    year = current_year if current_year is not None else datetime.now().year
    df["Age"] = year - df["Year"]
    df.drop(columns=["Year"], inplace=True)

    for col in config.CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")

    for col, pattern in config.REGEX_NUMERIC_COLUMNS.items():
        df[col] = df[col].str.extract(pattern).astype(float)

    return df


def split(
    df: pd.DataFrame,
    *,
    test_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el dataset en train/test usando los defaults de `config`."""
    return train_test_split(
        df,
        test_size=(
            test_size
            if test_size is not None
            else config.TEST_SIZE
        ),
        random_state=(
            random_state
            if random_state is not None
            else config.RANDOM_STATE
        ),
    )
