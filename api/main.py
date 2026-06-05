"""API REST para predicción de precios de autos.

Carga el modelo 'champion' del MLflow Model Registry y los artefactos
de preprocesamiento generados por el DAG data_pipeline.

Endpoints:
  POST /predict  — devuelve el precio estimado en INR
  GET  /health   — estado del servicio y del modelo
"""

from __future__ import annotations

import sys

# Permite importar `src` desde el volumen montado en /app/src
sys.path.insert(0, "/app")

import os
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
S3_ENDPOINT = os.environ.get("AWS_ENDPOINT_URL_S3", "http://s3:9000")
ARTIFACTS_S3_KEY = "data/artifacts/preprocessing_artifacts.joblib"

# Referencias globales al modelo y artefactos cargados al iniciar el servidor.
_model = None
_artifacts = None


def _load_resources() -> None:
    global _model, _artifacts
    import s3fs
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _model = mlflow.sklearn.load_model("models:/CarPriceModel@champion")
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT})
    with fs.open(ARTIFACTS_S3_KEY, "rb") as f:
        _artifacts = joblib.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_resources()
        print("Modelo y artefactos cargados correctamente.")
    except Exception as exc:
        # El servidor arranca igual; /predict devuelve 503 si el modelo no está.
        print(f"ADVERTENCIA: No se pudo cargar el modelo al iniciar: {exc}")
    yield


app = FastAPI(
    title="Car Price Prediction API",
    description=(
        "Predice el precio de un auto a partir de sus características. "
        "El modelo es cargado desde el MLflow Model Registry (alias 'champion')."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Esquemas
# ---------------------------------------------------------------------------


class CarInput(BaseModel):
    make: str = Field(..., examples=["Maruti"])
    year: int = Field(..., examples=[2019], ge=1990, le=2030)
    fuel_type: str = Field(..., examples=["Petrol"])
    transmission: str = Field(..., examples=["Manual"])
    location: str = Field(..., examples=["Mumbai"])
    color: str = Field(..., examples=["White"])
    owner: str = Field(..., examples=["First"])
    seller_type: str = Field(..., examples=["Individual"])
    drivetrain: Optional[str] = Field(None, examples=["FWD"])
    engine_cc: float = Field(..., examples=[1197.0], gt=0)
    max_power_bhp: float = Field(..., examples=[82.0], gt=0)
    max_torque_nm: float = Field(..., examples=[113.0], gt=0)
    kilometer: float = Field(..., examples=[45000.0], ge=0)
    length: float = Field(..., examples=[3845.0], gt=0)
    width: float = Field(..., examples=[1695.0], gt=0)
    height: float = Field(..., examples=[1520.0], gt=0)
    seating_capacity: float = Field(..., examples=[5.0], gt=0)
    fuel_tank_capacity: Optional[float] = Field(None, examples=[37.0])


class PredictionResponse(BaseModel):
    predicted_price_inr: float
    model_alias: str = "champion"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=PredictionResponse)
def predict(car: CarInput):
    """Predice el precio del auto en rupias indias (INR)."""
    global _model, _artifacts
    if _model is None or _artifacts is None:
        try:
            _load_resources()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Modelo no disponible ({exc}). "
                    "Ejecute primero el DAG training_pipeline en Airflow."
                ),
            )

    from src import config
    from src.features import transform_test

    current_year = pd.Timestamp.now().year

    # Construir DataFrame con la misma estructura que produce clean_structural.
    # 'Model' es requerido por transform_test (lo descarta internamente).
    # 'Price' es dummy: el scaler fue ajustado con esa columna presente.
    data = {
        "Make": [car.make],
        "Model": ["Unknown"],
        "Age": [current_year - car.year],
        "Fuel Type": [car.fuel_type],
        "Transmission": [car.transmission],
        "Location": [car.location],
        "Color": [car.color],
        "Owner": [car.owner],
        "Seller Type": [car.seller_type],
        "Drivetrain": [car.drivetrain],
        "Engine": [car.engine_cc],
        "Max Power": [car.max_power_bhp],
        "Max Torque": [car.max_torque_nm],
        "Kilometer": [car.kilometer],
        "Length": [car.length],
        "Width": [car.width],
        "Height": [car.height],
        "Seating Capacity": [car.seating_capacity],
        "Fuel Tank Capacity": [car.fuel_tank_capacity],
        "Price": [0.0],
    }
    df = pd.DataFrame(data)

    # Castear categóricas igual que hace clean_structural, para que
    # transform_test reciba el mismo tipo de datos que durante el training.
    from src import config as cfg
    for col in cfg.CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    df_transformed = transform_test(df, _artifacts)

    if df_transformed.empty:
        raise HTTPException(
            status_code=422,
            detail="El input no pasó la validación del preprocesador (campos requeridos con NaN o Owner inválido).",
        )

    X = df_transformed.drop(columns=["Price"], errors="ignore")

    # El modelo predice Price en espacio escalado (StandardScaler + log1p).
    # Invertir el scaler sobre Price y luego aplicar expm1.
    price_scaled = float(_model.predict(X)[0])
    train_cols = _artifacts["train_cols"]
    price_idx = train_cols.index("Price")
    price_mean = _artifacts["scaler"].mean_[price_idx]
    price_std = _artifacts["scaler"].scale_[price_idx]
    log_price = price_scaled * price_std + price_mean
    predicted_price = float(np.expm1(log_price))

    return PredictionResponse(predicted_price_inr=round(predicted_price, 2))


@app.get("/health")
def health():
    """Verifica el estado del servicio y si el modelo está cargado."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "artifacts_loaded": _artifacts is not None,
    }
