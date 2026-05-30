"""Configuración compartida: rutas, semilla y listas de columnas.

Centralizar acá facilita invocar los módulos desde MLflow/Airflow
sin tener que pasar paths/parametros por todos lados.
"""

from __future__ import annotations

from pathlib import Path

# Raíz del repo (este archivo vive en src/, subimos un nivel).
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Datos crudos y procesados.
DATASETS_DIR: Path = REPO_ROOT / "modelo_ml" / "datasets"
RAW_DATASET_FILENAME: str = "car_details_v4.csv"

# Artefactos del preprocesado (scaler, encoders, mapas, PCA, etc.).
# Usado por features.save_artifacts / load_artifacts.
ARTIFACTS_DIR: Path = REPO_ROOT / "modelo_ml" / "artifacts"

# Reproducibilidad.
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

# --- Columnas ---------------------------------------------------------------

# Columnas a castear como `category` en clean_structural.
CATEGORICAL_COLUMNS: list[str] = [
    "Make",
    "Fuel Type",
    "Transmission",
    "Location",
    "Color",
    "Owner",
    "Seller Type",
    "Drivetrain",
]

# Columnas con unidad textual (ej: "1498 cc") de las que extraemos el número.
REGEX_NUMERIC_COLUMNS: dict[str, str] = {
    "Engine": r"(\d+)",
    "Max Power": r"(\d+\.?\d*)",
    "Max Torque": r"(\d+\.?\d*)",
}

# Subset con el que dropeamos NaN en features.fit_transform_train.
DROPNA_SUBSET: list[str] = [
    "Length",
    "Width",
    "Height",
    "Seating Capacity",
    "Engine",
    "Max Power",
    "Max Torque",
]

# Columnas a las que aplicamos log1p (target + variables de cola larga).
LOG1P_COLUMNS: list[str] = ["Price", "Engine", "Max Power"]

# Columna sobre la que recortamos outliers por IQR.
IQR_COLUMN: str = "Kilometer"

# Categóricas a las que aplicamos agrupación de raras + el umbral.
RARE_CATEGORY_COLUMNS: list[str] = ["Fuel Type", "Color"]
RARE_CATEGORY_THRESHOLD: float = 0.01

# Encoders.
FREQUENCY_ENCODING_COLUMNS: list[str] = ["Make", "Location"]
ONE_HOT_COLUMNS: list[str] = [
  "Fuel Type",
  "Transmission",
  "Seller Type",
  "Drivetrain",
]
TARGET_ENCODING_COLUMN: str = "Color"
TARGET_COLUMN: str = "Price"

# Owner: combinamos categorías y aplicamos OrdinalEncoder.
OWNER_REPLACEMENTS: dict[str, str] = {"Fourth": "4 or More"}
OWNER_DROP_VALUES: list[str] = ["UnRegistered Car"]
OWNER_ORDER: list[str] = ["First", "Second", "Third", "Fourth", "4 or More"]

# Columnas que se eliminan después de mirar la matriz de correlación.
CORRELATED_DROP_COLUMNS: list[str] = [
    "Fuel Type_Other",
    "Engine",
    "Max Torque",
    "Length",
    "Width",
    "Fuel Tank Capacity",
    "Fuel Type_Diesel",
    "Seller Type_Corporate",
]

# PCA.
PCA_N_COMPONENTS: int = 10
