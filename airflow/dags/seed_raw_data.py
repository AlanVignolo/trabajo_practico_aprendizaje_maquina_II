"""DAG de setup (one-off): carga el CSV crudo en MinIO.

Trigger manual (`schedule=None`).
"""

import datetime
import os

from airflow.decorators import dag, task

S3_OPTS = {"endpoint_url": os.environ.get("AWS_ENDPOINT_URL_S3", "http://s3:9000")}

default_args = {
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}


@dag(
    dag_id="seed_raw_data",
    description="Setup one-off: carga el CSV crudo en MinIO",
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=["setup", "bootstrap"],
)
def seed_raw_data():
    @task(task_id="upload_raw_to_minio")
    def upload_raw_to_minio() -> str:
        """Lee el CSV crudo local y lo guarda en MinIO como Parquet."""
        from src import data

        df = data.load_raw()  # /opt/project/modelo_ml/datasets/car_details_v4.csv
        key = "s3://data/raw/car_details_v4.parquet"
        df.to_parquet(key, index=False, storage_options=S3_OPTS)
        print(f"Subidas {df.shape[0]} filas, {df.shape[1]} columnas -> {key}")
        return key

    upload_raw_to_minio()


dag = seed_raw_data()
