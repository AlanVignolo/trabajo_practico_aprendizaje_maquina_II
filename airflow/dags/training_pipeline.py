"""DAG de entrenamiento de modelos.

Lee los datasets procesados de MinIO, entrena los 8 modelos (con y sin
PCA), loguea cada run en MLflow y registra el mejor como 'champion' en
el Model Registry.

Disparado automáticamente por data_pipeline, o manualmente desde la UI.
"""

import datetime
import os
import sys

sys.path.insert(0, "/opt/airflow")

from airflow.decorators import dag, task

S3_OPTS = {"endpoint_url": os.environ.get("AWS_ENDPOINT_URL_S3", "http://s3:9000")}

PROCESSED = {
    "train":     "s3://data/processed/train.parquet",
    "test":      "s3://data/processed/test.parquet",
    "train_pca": "s3://data/processed/train_pca.parquet",
    "test_pca":  "s3://data/processed/test_pca.parquet",
}

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "car-price-prediction"

MODELS = [
    "baseline",
    "linear_regression",
    "svr",
    "decision_tree",
    "random_forest",
    "adaboost",
    "xgboost",
    "lightgbm",
]

default_args = {
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}


@dag(
    dag_id="training_pipeline",
    description="Entrena modelos de regresión y registra el mejor en MLflow",
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=["training", "mlflow"],
)
def training_pipeline():

    @task()
    def train_and_log() -> str:
        """Entrena los 8 modelos (con y sin PCA) y loguea cada run en MLflow.

        Devuelve el run_id del run con menor MAE en test.
        """
        import mlflow
        import mlflow.sklearn
        import pandas as pd
        from src.evaluate import compute_metrics
        from src.train import make_model

        train_proc = pd.read_parquet(PROCESSED["train"],     storage_options=S3_OPTS)
        test_proc  = pd.read_parquet(PROCESSED["test"],      storage_options=S3_OPTS)
        train_pca  = pd.read_parquet(PROCESSED["train_pca"], storage_options=S3_OPTS)
        test_pca   = pd.read_parquet(PROCESSED["test_pca"],  storage_options=S3_OPTS)

        X_train     = train_proc.drop(columns=["Price"])
        y_train     = train_proc["Price"]
        X_test      = test_proc.drop(columns=["Price"])
        y_test      = test_proc["Price"]
        X_train_pca = train_pca.drop(columns=["Price"])
        y_train_pca = train_pca["Price"]
        X_test_pca  = test_pca.drop(columns=["Price"])
        y_test_pca  = test_pca["Price"]

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        mlflow.sklearn.autolog(log_models=True, silent=True)

        for model_name in MODELS:
            print(f"Entrenando {model_name}...")
            with mlflow.start_run(run_name=model_name):
                metrics = compute_metrics(
                    make_model(model_name),
                    X_train, y_train, X_test, y_test,
                    name=model_name,
                )
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})

            print(f"Entrenando {model_name} + PCA...")
            with mlflow.start_run(run_name=f"{model_name}_pca"):
                metrics = compute_metrics(
                    make_model(model_name),
                    X_train_pca, y_train_pca, X_test_pca, y_test_pca,
                    name=f"{model_name}_pca",
                )
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})

        runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
        best_run = runs.sort_values("metrics.MAE").iloc[0]
        best_run_id = best_run["run_id"]
        print(f"Mejor modelo: {best_run['tags.mlflow.runName']}  MAE={best_run['metrics.MAE']:.4f}")
        return best_run_id

    @task()
    def register_champion(best_run_id: str) -> None:
        """Registra el mejor run en el Model Registry con alias 'champion'."""
        import mlflow
        from mlflow import MlflowClient
        from mlflow.exceptions import MlflowException

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        run = client.get_run(best_run_id)
        artifact_uri = run.info.artifact_uri

        try:
            client.create_registered_model("CarPriceModel")
        except MlflowException as e:
            if "RESOURCE_ALREADY_EXISTS" not in str(e):
                raise

        version = client.create_model_version(
            name="CarPriceModel",
            source=f"{artifact_uri}/model",
            run_id=best_run_id,
        )
        client.set_registered_model_alias("CarPriceModel", "champion", version.version)
        print(f"CarPriceModel v{version.version} registrada con alias 'champion'.")

    best_run_id = train_and_log()
    register_champion(best_run_id)


dag = training_pipeline()