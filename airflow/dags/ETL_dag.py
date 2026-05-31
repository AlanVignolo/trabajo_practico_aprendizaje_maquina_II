import datetime

from airflow.decorators import dag, task

RAW_KEY = "s3://data/raw/car_details_v4.parquet"
CLEAN_KEY = "s3://data/clean/car_details_v4.parquet"
TRAIN_KEY = "s3://data/interim/train.parquet"
TEST_KEY = "s3://data/interim/test.parquet"
PROCESSED = {
    "train": "s3://data/processed/train.parquet",
    "test": "s3://data/processed/test.parquet",
    "train_pca": "s3://data/processed/train_pca.parquet",
    "test_pca": "s3://data/processed/test_pca.parquet",
}

default_args = {
    "depends_on_past": False,
    "schedule_interval": None,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
    "dagrun_timeout": datetime.timedelta(minutes=15),
}


@dag(
    dag_id="ETL_dag",
    description="Proceso ETL del dataset de autos usando TaskFlow + src",
    default_args=default_args,
    catchup=False,
    tags=["ETL", "TaskFlow"],
)
def process_etl_taskflow():

    @task(task_id="obtain_original_data")
    def obtain_original_data():
        """
        Carga el parquet crudo desde MinIO (dejado por seed_raw_data)
        """
        import pandas as pd

        print("Leyendo parquet crudo desde MinIO...")
        df = pd.read_parquet(RAW_KEY)
        print(f"Crudo: {df.shape[0]} filas, {df.shape[1]} columnas")
        return RAW_KEY

    @task(task_id="clean_structural_data", multiple_outputs=True)
    def clean_structural_data(path_input):
        """
        Limpieza estructural previa al split (src.data.clean_structural)
        """
        import pandas as pd

        from src import data

        print(f"Leyendo: {path_input}")
        df = pd.read_parquet(path_input)

        print("Aplicando clean_structural...")
        df = data.clean_structural(df)
        df.to_parquet(CLEAN_KEY, index=False)
        print(f"Limpio guardado en: {CLEAN_KEY}")

        return {
            "path": CLEAN_KEY,
            "observations": df.shape[0],
            "columns": df.shape[1],
        }

    @task(task_id="split_dataset", multiple_outputs=True)
    def split_dataset(file_path, obs, col):
        """
        Separa en train/test (src.data.split)
        """
        import pandas as pd

        from src import data

        df = pd.read_parquet(file_path)
        assert df.shape == (obs, col), (
            "La forma del dataset no coincide con lo esperado."
        )

        print("Separando dataset en entrenamiento y prueba...")
        train, test = data.split(df)

        train.to_parquet(TRAIN_KEY, index=False)
        test.to_parquet(TEST_KEY, index=False)
        print(f"train={train.shape[0]} | test={test.shape[0]}")

        return {"train_file_path": TRAIN_KEY, "test_file_path": TEST_KEY}

    @task(task_id="transform_features", multiple_outputs=True)
    def transform_features(train_path, test_path):
        """
        Feature engineering + PCA (src.features). Los artifacts del fit
        se quedan en memoria de esta task; solo persistimos los datasets.
        """
        import pandas as pd

        from src import features

        print("Leyendo train/test...")
        train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)

        print("fit_transform_train + transform_test...")
        train_proc, artifacts = features.fit_transform_train(train)
        test_proc = features.transform_test(test, artifacts)

        print("fit_pca + apply_pca...")
        train_pca, pca = features.fit_pca(train_proc)
        test_pca = features.apply_pca(test_proc, pca)

        train_proc.to_parquet(PROCESSED["train"], index=False)
        test_proc.to_parquet(PROCESSED["test"], index=False)
        train_pca.to_parquet(PROCESSED["train_pca"], index=False)
        test_pca.to_parquet(PROCESSED["test_pca"], index=False)
        print("Datasets procesados guardados en MinIO.")

        return PROCESSED

    # Encadenamiento
    raw = obtain_original_data()
    clean = clean_structural_data(raw)
    files = split_dataset(clean["path"], clean["observations"], clean["columns"])
    transform_features(files["train_file_path"], files["test_file_path"])


dag = process_etl_taskflow()
