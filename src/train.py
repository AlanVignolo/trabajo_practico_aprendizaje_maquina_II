"""Entrenamiento: baseline, factory de modelos, GridSearch y Optuna.

Toda la lógica de hiperparametrización (espacios de búsqueda, grids)
vive acá como constantes para que pueda reutilizarse desde MLflow/Airflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src import config


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


class BrandAverageBaseline(BaseEstimator, RegressorMixin):
    """Predice el precio promedio del conjunto de entrenamiento para cada
    "marca" (brand).

    Regresa al promedio general del conjunto de entrenamiento para marcas
    no vistas, o cuando la columna `marca` esté ausente (p. ej. después de
    la transformación PCA).
    """

    def fit(self, X: pd.DataFrame, y) -> "BrandAverageBaseline":
        self.overall_avg_ = float(np.mean(y))
        if "marca" in X.columns:
            df = pd.DataFrame(
                {"marca": X["marca"].values, "_y": np.asarray(y)}
            )
            self.brand_avg_ = df.groupby("marca")["_y"].mean().to_dict()
        else:
            self.brand_avg_ = {}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.brand_avg_ or "marca" not in X.columns:
            return np.full(len(X), self.overall_avg_)
        return (
            pd.Series(X["marca"].values)
            .map(self.brand_avg_)
            .fillna(self.overall_avg_)
            .to_numpy()
        )


# ---------------------------------------------------------------------------
# Factory de modelos
# ---------------------------------------------------------------------------


def make_model(name: str, **params: Any) -> BaseEstimator:
    """Devuelve un estimador (o pipeline) listo para `.fit`.

    Modelos soportados:
        - "baseline": BrandAverageBaseline
        - "linear_regression": Pipeline(StandardScaler -> LinearRegression)
        - "svr": Pipeline(SVR)
        - "decision_tree": DecisionTreeRegressor
        - "random_forest": RandomForestRegressor
        - "adaboost": AdaBoostRegressor
        - "xgboost": xgb.XGBRegressor
        - "lightgbm": lgb.LGBMRegressor
    """
    name = name.lower()
    if name == "baseline":
        return BrandAverageBaseline(**params)
    if name == "linear_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression(**params)),
            ]
        )
    if name == "svr":
        return Pipeline([("model", SVR(**params))])
    if name == "decision_tree":
        params.setdefault("random_state", config.RANDOM_STATE)
        return DecisionTreeRegressor(**params)
    if name == "random_forest":
        params.setdefault("random_state", config.RANDOM_STATE)
        return RandomForestRegressor(**params)
    if name == "adaboost":
        params.setdefault("random_state", config.RANDOM_STATE)
        return AdaBoostRegressor(**params)
    if name == "xgboost":
        params.setdefault("random_state", config.RANDOM_STATE)
        params.setdefault("n_jobs", -1)
        return xgb.XGBRegressor(**params)
    if name == "lightgbm":
        params.setdefault("random_state", config.RANDOM_STATE)
        params.setdefault("n_jobs", -1)
        params.setdefault("verbose", -1)
        return lgb.LGBMRegressor(**params)
    raise ValueError(f"Modelo desconocido: {name!r}")


def train_model(model: BaseEstimator, X_train, y_train) -> BaseEstimator:
    """Llama a `model.fit(X_train, y_train)` y devuelve el modelo entrenado."""
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# GridSearch
# ---------------------------------------------------------------------------


def run_grid_search(
    X,
    y,
    estimator: BaseEstimator,
    param_grid: dict[str, Any],
    *,
    cv: int = 5,
    scoring: str = "neg_mean_absolute_error",
    n_jobs: int = -1,
    verbose: int = 1,
) -> GridSearchCV:
    """Ejecuta GridSearchCV con los defaults del notebook."""
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
    )
    gs.fit(X, y)
    return gs


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------


def tune_with_optuna(
    model_cls: type,
    search_space: dict[str, Callable[[optuna.trial.Trial], Any]],
    X,
    y,
    *,
    study_name: str,
    static_params: dict[str, Any] | None = None,
    n_trials: int = 500,
    cv: int | None = 5,
    val_size: float = 0.2,
    storage: str | None = None,
    random_state: int = 42,
    show_progress_bar: bool = True,
) -> optuna.study.Study:
    """Ejecuta un estudio Optuna minimizando MAE.

    - `search_space`: dict `{param: lambda trial: trial.suggest_*(...)}`.
    - `cv=int`  -> score = media de cross_val_score con `cv` folds.
    - `cv=None` -> score = MAE sobre un holdout (`val_size`).
    - `storage`: cadena SQLAlchemy (ej. `sqlite:///optuna.db`). Si es
      `None`, el estudio vive en memoria (no se reusan trials previos).
    """
    static_params = static_params or {}

    if cv is None:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=val_size, random_state=random_state
        )

    def objective(trial: optuna.trial.Trial) -> float:
        trial_params = {k: fn(trial) for k, fn in search_space.items()}
        params = {**static_params, **trial_params}
        model = model_cls(**params)
        if cv is None:
            model.fit(X_tr, y_tr)
            return mean_absolute_error(y_val, model.predict(X_val))
        scores = cross_val_score(
            model, X, y, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1
        )
        return -scores.mean()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    remaining = max(0, n_trials - len(study.trials))
    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            show_progress_bar=show_progress_bar,
        )
    return study


# ---------------------------------------------------------------------------
# Cache opcional (extraido del notebook). Útil hasta que MLflow tome control.
# ---------------------------------------------------------------------------


def cached(
    key: str,
    builder: Callable[[], Any],
    cache_dir: Path | str,
    *,
    force: bool = False,
) -> Any:
    """Memoiza el resultado de `builder()` en disco con joblib."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.joblib"
    if path.exists() and not force:
        return joblib.load(path)
    value = builder()
    joblib.dump(value, path)
    return value


# ---------------------------------------------------------------------------
# Espacios de búsqueda (constantes reutilizables)
# ---------------------------------------------------------------------------

RF_GRID: dict[str, Any] = {
    "max_depth": list(range(1, 11)),
    "min_samples_split": list(range(2, 20, 4)),
    "min_samples_leaf": list(range(1, 10, 5)),
    "n_estimators": list(range(10, 300, 100)),
    "n_jobs": [-1],
    "criterion": ["squared_error"],
}

XGB_GRID: dict[str, Any] = {
    "subsample": list(np.linspace(0.4, 1.0, 10)),
    "learning_rate": list(np.linspace(0.01, 0.3, 10)),
    "alpha": list(np.linspace(0.01, 10, 10)),
}

LGBM_GRID: dict[str, Any] = {
    "subsample": list(np.linspace(0.4, 1.0, 10)),
    "learning_rate": list(np.linspace(0.01, 0.3, 10)),
    "reg_alpha": list(np.linspace(0.01, 10, 10)),
}

RF_SEARCH_SPACE: dict[str, Callable[[optuna.trial.Trial], Any]] = {
    "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 1000),
    "max_depth": lambda t: t.suggest_int("max_depth", 3, 30),
    "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 20),
    "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 10),
    "max_features": lambda t: t.suggest_float("max_features", 0.2, 1.0),
    "max_samples": lambda t: t.suggest_float("max_samples", 0.5, 1.0),
}
RF_STATIC_PARAMS: dict[str, Any] = {
    "random_state": config.RANDOM_STATE,
    "n_jobs": -1,
    "bootstrap": True,
}

XGB_SEARCH_SPACE: dict[str, Callable[[optuna.trial.Trial], Any]] = {
    "subsample": lambda t: t.suggest_float("subsample", 0.4, 1.0),
    "learning_rate": (
        lambda t: t.suggest_float("learning_rate", 0.01, 0.3, log=True)
    ),
    "alpha": lambda t: t.suggest_float("alpha", 0.01, 10.0, log=True),
    "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 1000),
    "max_depth": lambda t: t.suggest_int("max_depth", 3, 10),
    "min_child_weight": lambda t: t.suggest_int("min_child_weight", 1, 10),
    "gamma": lambda t: t.suggest_float("gamma", 0.0, 5.0),
    "colsample_bytree": (
        lambda t: t.suggest_float("colsample_bytree", 0.5, 1.0)
    ),
    "reg_lambda": (
        lambda t: t.suggest_float("reg_lambda", 0.01, 10.0, log=True)
    ),
}
XGB_STATIC_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "random_state": config.RANDOM_STATE,
    "n_jobs": -1,
}

LGB_SEARCH_SPACE: dict[str, Callable[[optuna.trial.Trial], Any]] = {
    "num_leaves": lambda t: t.suggest_int("num_leaves", 20, 300),
    "max_depth": lambda t: t.suggest_int("max_depth", 3, 12),
    "learning_rate": (
        lambda t: t.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    ),
    "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 1000),
    "subsample": lambda t: t.suggest_float("subsample", 0.4, 1.0),
    "colsample_bytree": (
        lambda t: t.suggest_float("colsample_bytree", 0.4, 1.0)
    ),
    "reg_alpha": (
        lambda t: t.suggest_float("reg_alpha", 1e-3, 10.0, log=True)
    ),
    "reg_lambda": (
        lambda t: t.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
    ),
    "min_child_samples": lambda t: t.suggest_int("min_child_samples", 5, 100),
}
LGB_STATIC_PARAMS: dict[str, Any] = {
    "random_state": config.RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}
