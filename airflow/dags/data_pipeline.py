"""DAG de preprocesamiento (ETL).

Lee el Parquet crudo de MinIO, aplica limpieza estructural, divide en
train/test, aplica feature engineering y PCA. Persiste los datasets
procesados y los artefactos de preprocesamiento en MinIO.

Al finalizar dispara training_pipeline.

Requiere que seed_raw_data haya corrido al menos una vez.
"""

import datetime
import os
import sys

sys.path.insert(0, "/opt/airflow")

from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

S3_OPTS = {"endpoint_url": os.environ.get("AWS_ENDPOINT_URL_S3", "http://s3:9000")}

RAW_KEY = "s3://data/raw/car_details_v4.parquet"
CLEAN_KEY = "s3://data/clean/car_details.parquet"
TRAIN_KEY = "s3://data/interim/train.parquet"
TEST_KEY = "s3://data/interim/test.parquet"
PROCESSED = {
    "train":     "s3://data/processed/train.parquet",
    "test":      "s3://data/processed/test.parquet",
    "train_pca": "s3://data/processed/train_pca.parquet",
    "test_pca":  "s3://data/processed/test_pca.parquet",
}
ARTIFACTS = {
    "preprocessing": "data/artifacts/preprocessing_artifacts.joblib",
    "pca":           "data/artifacts/pca.joblib",
}

default_args = {
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}


@dag(
    dag_id="data_pipeline",
    description="ETL: limpieza, split y feature engineering del dataset de autos",
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=["data", "preprocessing", "ETL"],
)
def data_pipeline():

    @task(multiple_outputs=True)
    def clean_structural_data() -> dict:
        """Aplica limpieza estructural y guarda el resultado en MinIO."""
        import pandas as pd
        from src.data import clean_structural

        df = pd.read_parquet(RAW_KEY, storage_options=S3_OPTS)
        df = clean_structural(df)
        df.to_parquet(CLEAN_KEY, index=False, storage_options=S3_OPTS)
        print(f"Limpio: {df.shape[0]} filas, {df.shape[1]} columnas → {CLEAN_KEY}")
        return {"path": CLEAN_KEY, "rows": df.shape[0], "cols": df.shape[1]}

    @task(multiple_outputs=True)
    def split_dataset(path: str, rows: int, cols: int) -> dict:
        """Divide en train/test y guarda ambos en MinIO."""
        import pandas as pd
        from src.data import split

        df = pd.read_parquet(path, storage_options=S3_OPTS)
        assert df.shape == (rows, cols), "Shape del dataset no coincide."

        train, test = split(df)
        train.to_parquet(TRAIN_KEY, index=False, storage_options=S3_OPTS)
        test.to_parquet(TEST_KEY, index=False, storage_options=S3_OPTS)
        print(f"Split: train={len(train)}  test={len(test)}")
        return {"train": TRAIN_KEY, "test": TEST_KEY}

    @task()
    def transform_features(train_path: str, test_path: str) -> dict:
        """Ajusta transformadores y PCA sobre train, transforma test.

        Guarda los 4 datasets procesados y los 2 artefactos en MinIO.
        Los artefactos (encoders, scaler, PCA) se guardan en MinIO para
        que la FastAPI los pueda descargar al momento de inferencia.
        """
        import joblib
        import pandas as pd
        import s3fs
        from src.features import apply_pca, fit_pca, fit_transform_train, transform_test

        train = pd.read_parquet(train_path, storage_options=S3_OPTS)
        test = pd.read_parquet(test_path, storage_options=S3_OPTS)

        train_proc, artifacts = fit_transform_train(train)
        test_proc = transform_test(test, artifacts)

        train_pca, pca = fit_pca(train_proc)
        test_pca = apply_pca(test_proc, pca)

        # Guardar datasets procesados
        for key, df in [
            (PROCESSED["train"],     train_proc),
            (PROCESSED["test"],      test_proc),
            (PROCESSED["train_pca"], train_pca),
            (PROCESSED["test_pca"],  test_pca),
        ]:
            df.to_parquet(key, index=False, storage_options=S3_OPTS)

        # Guardar artefactos usando s3fs (joblib no soporta URIs s3://)
        fs = s3fs.S3FileSystem(**S3_OPTS)
        for s3_key, obj in [
            (ARTIFACTS["preprocessing"], artifacts),
            (ARTIFACTS["pca"],           pca),
        ]:
            with fs.open(s3_key, "wb") as f:
                joblib.dump(obj, f)

        print(f"Datasets procesados y artefactos guardados en MinIO.")
        return PROCESSED

    trigger = TriggerDagRunOperator(
        task_id="trigger_training_pipeline",
        trigger_dag_id="training_pipeline",
        wait_for_completion=False,
    )

    clean = clean_structural_data()
    split = split_dataset(clean["path"], clean["rows"], clean["cols"])
    transform_features(split["train"], split["test"]) >> trigger


dag = data_pipeline()