"""Métricas y comparativa de modelos.

Las métricas se devuelven como `dict` para poder loguearlas con
`mlflow.log_metrics(...)` sin transformaciones extra.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)


def compute_metrics(
    model: BaseEstimator,
    X_train,
    y_train,
    X_test,
    y_test,
    name: str,
) -> dict[str, Any]:
    """Ajusta `model` y devuelve un dict con MAE_training, MAE, RMSE, MAPE, R2.

    Coincide con la salida de `evaluate()` del notebook, pero sin la capa
    de cache: el caller decide cómo persistir/loguear el resultado.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "name": name,
        "MAE_training": mean_absolute_error(y_train, model.predict(X_train)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }


def results_to_dataframe(
    results: dict[str, dict[str, Any]],
    *,
    sort_by: str = "R2",
    ascending: bool = False,
) -> pd.DataFrame:
    """Convierte el dict de resultados en un DataFrame ordenado."""
    return pd.DataFrame(results).T.sort_values(sort_by, ascending=ascending)
