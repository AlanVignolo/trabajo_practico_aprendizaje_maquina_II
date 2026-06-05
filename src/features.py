"""Feature engineering con separación fit/transform.

`fit_transform_train` ajusta sobre train y devuelve el dataframe procesado
junto con un dict de `artifacts` (mediana, moda, mapas de frecuencia,
medias de target, OrdinalEncoder, StandardScaler, lista de columnas
finales, etc). `transform_test` consume esos `artifacts` para aplicar
exactamente las mismas transformaciones al test set y a datos nuevos
en inferencia.

El PCA se ajusta sobre el resultado de `fit_transform_train` (sin la
columna target) y se aplica al test set con `apply_pca`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src import config


# ---------------------------------------------------------------------------
# Fit + transform sobre el conjunto de entrenamiento
# ---------------------------------------------------------------------------


def fit_transform_train(
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ajusta todos los transformadores sobre train y devuelve (df, artifacts).

    `artifacts` contiene todos los objetos necesarios para reproducir
    la transformación en test/inferencia.
    """
    df = train_df.copy()
    artifacts: dict[str, Any] = {}

    # 1. Imputación: dropear filas y rellenar con mediana/moda
    df.dropna(subset=config.DROPNA_SUBSET, inplace=True)

    fuel_tank_median = float(df["Fuel Tank Capacity"].median())
    df["Fuel Tank Capacity"] = df["Fuel Tank Capacity"].fillna(
        fuel_tank_median
    )
    artifacts["fuel_tank_median"] = fuel_tank_median

    drivetrain_mode = df["Drivetrain"].mode()[0]
    df["Drivetrain"] = df["Drivetrain"].fillna(drivetrain_mode)
    artifacts["drivetrain_mode"] = drivetrain_mode

    # 2. Transformación logarítmica
    for col in config.LOG1P_COLUMNS:
        df[col] = np.log1p(df[col])

    # 3. Eliminación por IQR para `Kilometer`
    q1 = df[config.IQR_COLUMN].quantile(0.25)
    q3 = df[config.IQR_COLUMN].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    in_range = (df[config.IQR_COLUMN] >= lower) & (
        df[config.IQR_COLUMN] <= upper
    )
    df = df[in_range]

    # 4. Agrupación de categorías raras (frecuencia < umbral) bajo "Other"
    rare_kept: dict[str, list[str]] = {}
    for col in config.RARE_CATEGORY_COLUMNS:
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < config.RARE_CATEGORY_THRESHOLD].index

        df[col] = df[col].cat.add_categories("Other")
        df[col] = df[col].replace(rare, "Other")
        df[col] = df[col].cat.remove_unused_categories()

        # Categorías a CONSERVAR (todo lo que no sea "Other"), usadas en test.
        kept = (
            df[col]
            .value_counts()
            .drop("Other", errors="ignore")
            .index.tolist()
        )
        rare_kept[col] = kept
    artifacts["rare_kept"] = rare_kept

    # 5. Combinación de categorías en `Owner`
    for src_val, dst_val in config.OWNER_REPLACEMENTS.items():
        df["Owner"] = df["Owner"].replace(src_val, dst_val)
    df["Owner"] = df["Owner"].cat.remove_unused_categories()

    # 6. Frequency Encoding (Make, Location)
    frequency_maps: dict[str, pd.Series] = {}
    for col in config.FREQUENCY_ENCODING_COLUMNS:
        freq = df[col].value_counts(normalize=True)
        df[f"{col}_freq"] = df[col].map(freq)
        frequency_maps[col] = freq
    df.drop(columns=config.FREQUENCY_ENCODING_COLUMNS, inplace=True)
    artifacts["frequency_maps"] = frequency_maps

    # 7. One-Hot Encoding
    df = pd.get_dummies(df, columns=config.ONE_HOT_COLUMNS, drop_first=True)

    # 8. Target Encoding (Color)
    price_mean_by_color = (
        df.groupby(config.TARGET_ENCODING_COLUMN, observed=True)[
            config.TARGET_COLUMN
        ].mean()
    )
    df[f"{config.TARGET_ENCODING_COLUMN}_target"] = (
        df[config.TARGET_ENCODING_COLUMN].map(price_mean_by_color)
    )
    df.drop(columns=[config.TARGET_ENCODING_COLUMN], inplace=True)
    artifacts["price_mean_by_color"] = price_mean_by_color

    # 9. Filtrado de Owner inválido + OrdinalEncoder
    df = df[~df["Owner"].isin(config.OWNER_DROP_VALUES)]
    ordinal_encoder = OrdinalEncoder(categories=[config.OWNER_ORDER])
    df["Owner_encoded"] = ordinal_encoder.fit_transform(df[["Owner"]])
    df.drop(columns=["Owner"], inplace=True)
    artifacts["owner_encoder"] = ordinal_encoder

    # 10. Eliminación de `Model` (alta cardinalidad)
    df.drop(columns=["Model"], inplace=True)

    # 11. Booleanos a int (resultado del OHE)
    bool_cols = df.select_dtypes(include="bool").columns
    df = df.astype({col: int for col in bool_cols})

    # 12. Standardización (Z-score). Guardamos también el orden de columnas
    # para alinear test con `reindex(columns=train_cols)`.
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    train_cols = df.columns.tolist()
    artifacts["scaler"] = scaler
    artifacts["train_cols"] = train_cols

    # 13. Eliminación de columnas correlacionadas
    drop_cols = [c for c in config.CORRELATED_DROP_COLUMNS if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    artifacts["correlated_drop_columns"] = drop_cols

    return df, artifacts


# ---------------------------------------------------------------------------
# Transformación del conjunto de test usando los artefactos del fit
# ---------------------------------------------------------------------------


def transform_test(
    test_df: pd.DataFrame,
    artifacts: dict[str, Any],
) -> pd.DataFrame:
    """Aplica al test set las mismas transformaciones que train.

    Reusa los objetos ajustados en `artifacts`.
    """
    df = test_df.copy()

    # 1. Imputación
    df.dropna(subset=config.DROPNA_SUBSET, inplace=True)
    fuel_median = artifacts["fuel_tank_median"]
    df["Fuel Tank Capacity"] = df["Fuel Tank Capacity"].fillna(fuel_median)
    df["Drivetrain"] = df["Drivetrain"].fillna(artifacts["drivetrain_mode"])

    # 2. log1p
    for col in config.LOG1P_COLUMNS:
        df[col] = np.log1p(df[col])

    # NOTA: el filtrado por IQR (paso 3) NO se aplica a test (eliminaría filas
    # válidas), respetando el comportamiento del notebook original.

    # 4. Agrupación de raras: todo lo que no aparece en `rare_kept` -> "Other"
    for col, kept in artifacts["rare_kept"].items():
        df[col] = df[col].apply(
            lambda x, kept=kept: "Other" if x not in kept else x
        )

    # 5. Combinación de categorías en `Owner`
    df["Owner"] = df["Owner"].astype(object)
    for src_val, dst_val in config.OWNER_REPLACEMENTS.items():
        df["Owner"] = df["Owner"].replace(src_val, dst_val)

    # 6. Frequency Encoding usando los mapas de train (categorías nuevas -> 0)
    frequency_maps: dict[str, pd.Series] = artifacts["frequency_maps"]
    for col, freq in frequency_maps.items():
        df[f"{col}_freq"] = df[col].astype(object).map(freq).fillna(0)
    df.drop(columns=list(frequency_maps.keys()), inplace=True)

    # 7. One-Hot Encoding
    for col in config.ONE_HOT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(object)
    df = pd.get_dummies(df, columns=config.ONE_HOT_COLUMNS, drop_first=False)

    # 8. Target Encoding (categorías nuevas -> media global de train)
    price_mean_by_color: pd.Series = artifacts["price_mean_by_color"]
    df[f"{config.TARGET_ENCODING_COLUMN}_target"] = (
        df[config.TARGET_ENCODING_COLUMN]
        .astype(object)
        .map(price_mean_by_color)
        .fillna(price_mean_by_color.mean())
    )
    df.drop(columns=[config.TARGET_ENCODING_COLUMN], inplace=True)

    # 9. Filtrado de Owner inválido + OrdinalEncoder ya ajustado
    df = df[~df["Owner"].isin(config.OWNER_DROP_VALUES)].copy()
    df["Owner_encoded"] = artifacts["owner_encoder"].transform(df[["Owner"]])
    df.drop(columns=["Owner"], inplace=True)

    # 10. Drop de `Model`
    df.drop(columns=["Model"], inplace=True)

    # 11. Booleanos a int
    bool_cols = df.select_dtypes(include="bool").columns
    df = df.astype({col: int for col in bool_cols})

    # 12. Reindex al orden de columnas de train + scaler.transform
    df = df.reindex(columns=artifacts["train_cols"], fill_value=0)
    df = pd.DataFrame(
        artifacts["scaler"].transform(df),
        columns=df.columns,
    )

    # 13. Drop de columnas correlacionadas (mismas que train)
    drop_cols = [
        c for c in artifacts["correlated_drop_columns"] if c in df.columns
    ]
    df.drop(columns=drop_cols, inplace=True)

    return df


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def fit_pca(
    df: pd.DataFrame,
    *,
    n_components: int | None = None,
    target_column: str | None = None,
) -> tuple[pd.DataFrame, PCA]:
    """Ajusta PCA sobre `df` (excluyendo el target) y devuelve (df_pca, pca).

    El dataframe resultante tiene columnas `PC1..PCn + target_column`.
    """
    if n_components is None:
        n_components = config.PCA_N_COMPONENTS
    if target_column is None:
        target = config.TARGET_COLUMN
    else:
        target = target_column

    X = df.drop(columns=[target])
    y = df[target]

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    pc_cols = [f"PC{i + 1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=pc_cols)
    df_pca[target] = y.values
    return df_pca, pca


def apply_pca(
    df: pd.DataFrame,
    pca: PCA,
    *,
    target_column: str | None = None,
) -> pd.DataFrame:
    """Aplica un PCA ya ajustado a `df`, preservando la columna target."""
    if target_column is None:
        target = config.TARGET_COLUMN
    else:
        target = target_column

    X = df.drop(columns=[target])
    y = df[target].values

    X_pca = pca.transform(X)
    n_comp = pca.n_components_
    pc_cols = [f"PC{i + 1}" for i in range(n_comp)]
    df_pca = pd.DataFrame(X_pca, columns=pc_cols)
    df_pca[target] = y
    return df_pca


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------


def save_artifacts(
    artifacts: dict[str, Any],
    artifacts_dir: Path | str | None = None,
    *,
    filename: str = "preprocessing_artifacts.joblib",
) -> Path:
    """Serializa el dict de artefactos con joblib."""
    artifacts_dir = (
        Path(artifacts_dir) if artifacts_dir else config.ARTIFACTS_DIR
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / filename
    joblib.dump(artifacts, path)
    return path


def load_artifacts(
    artifacts_dir: Path | str | None = None,
    *,
    filename: str = "preprocessing_artifacts.joblib",
) -> dict[str, Any]:
    """Carga el dict de artefactos previamente persistido."""
    artifacts_dir = (
        Path(artifacts_dir) if artifacts_dir else config.ARTIFACTS_DIR
    )
    return joblib.load(artifacts_dir / filename)


def save_pca(
    pca: PCA,
    artifacts_dir: Path | str | None = None,
    *,
    filename: str = "pca.joblib",
) -> Path:
    """Serializa el objeto PCA."""
    artifacts_dir = (
        Path(artifacts_dir) if artifacts_dir else config.ARTIFACTS_DIR
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / filename
    joblib.dump(pca, path)
    return path


def load_pca(
    artifacts_dir: Path | str | None = None,
    *,
    filename: str = "pca.joblib",
) -> PCA:
    """Carga el PCA previamente persistido."""
    artifacts_dir = (
        Path(artifacts_dir) if artifacts_dir else config.ARTIFACTS_DIR
    )
    return joblib.load(artifacts_dir / filename)


def save_datasets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_pca: pd.DataFrame,
    test_pca: pd.DataFrame,
    datasets_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Escribe los 4 CSVs (train/test crudos y PCA) y devuelve los paths."""
    datasets_dir = Path(datasets_dir) if datasets_dir else config.DATASETS_DIR
    datasets_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": datasets_dir / "train_data.csv",
        "test": datasets_dir / "test_data.csv",
        "train_pca": datasets_dir / "train_data_pca.csv",
        "test_pca": datasets_dir / "test_data_pca.csv",
    }
    train.to_csv(paths["train"], index=False)
    test.to_csv(paths["test"], index=False)
    train_pca.to_csv(paths["train_pca"], index=False)
    test_pca.to_csv(paths["test_pca"], index=False)
    return paths
