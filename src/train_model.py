import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.data import load_raw, clean_structural, split
from src.features import fit_transform_train, transform_test, fit_pca, apply_pca
from src.train import make_model
from src.evaluate import compute_metrics

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("car-price-prediction")

# Cargar y preprocesar
print("Cargando datos...")
raw = load_raw()
clean = clean_structural(raw)
train_df, test_df = split(clean)

print("Transformando features...")
train_df, artifacts = fit_transform_train(train_df)
test_df = transform_test(test_df, artifacts)

X_train = train_df.drop(columns=["Price"])
y_train = train_df["Price"]
X_test = test_df.drop(columns=["Price"])
y_test = test_df["Price"]

print("Calculando PCA...")
train_pca, pca = fit_pca(train_df)
test_pca = apply_pca(test_df, pca)

X_train_pca = train_pca.drop(columns=["Price"])
y_train_pca = train_pca["Price"]
X_test_pca = test_pca.drop(columns=["Price"])
y_test_pca = test_pca["Price"]

print("Datos listos. Iniciando runs de MLflow...")

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

for model_name in MODELS:
    print(f"Entrenando {model_name}...")
    mlflow.sklearn.autolog(log_models=True, silent=True)
    with mlflow.start_run(run_name=model_name):
        model = make_model(model_name)
        metrics = compute_metrics(model, X_train, y_train, X_test, y_test, name=model_name)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})

    print(f"Entrenando {model_name} + PCA...")
    mlflow.sklearn.autolog(log_models=True, silent=True)
    with mlflow.start_run(run_name=f"{model_name}_pca"):
        model = make_model(model_name)
        metrics = compute_metrics(model, X_train_pca, y_train_pca, X_test_pca, y_test_pca, name=f"{model_name}_pca")
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})

print("Todos los runs completados.")

print("Buscando el mejor modelo...")
runs = mlflow.search_runs(experiment_names=["car-price-prediction"])
best_run = runs.sort_values("metrics.MAE").iloc[0]

print(f"Mejor modelo: {best_run['tags.mlflow.runName']} con MAE={best_run['metrics.MAE']:.4f}")

client = MlflowClient()

# Crear el modelo en el registry (si ya existe, ignorar el error)
try:
    client.create_registered_model("CarPriceModel")
except Exception:
    pass

# Registrar la version del mejor run
version = client.create_model_version(
    name="CarPriceModel",
    source=f"{best_run['artifact_uri']}/model",
    run_id=best_run["run_id"],
)

# Asignar alias champion
client.set_registered_model_alias("CarPriceModel", "champion", version.version)

print(f"Modelo registrado como CarPriceModel v{version.version} con alias 'champion'.")
