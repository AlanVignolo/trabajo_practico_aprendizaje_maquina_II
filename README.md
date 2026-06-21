# Trabajo Práctico - Aprendizaje de Máquina II

Pipeline de MLOps para predecir el precio de autos usados (en rupias indias, INR) a partir de sus características (marca, año, motor, kilometraje, etc.). El dataset base es [Car Details v4](modelo_ml/datasets/car_details_v4.csv), que veníamos usando desde la materia anterior.

La idea del TP fue tomar ese trabajo y armarle alrededor toda la infraestructura: orquestación con Airflow, tracking y registry de modelos con MLflow, almacenamiento de datos y artefactos en MinIO, una API en FastAPI para servir las predicciones y un front mínimo en Streamlit. Todo se levanta con un `docker compose up`.

## Integrantes
- Agustín Biancardi (a2403)
- Marcos Lund (a2408)
- Alan Lihuen Vignolo (a2418)
- Tomás Agustín Mc Nally (a2420)

## Arquitectura

Los tres DAGs forman una cadena: el primero carga el dataset crudo, el segundo lo procesa y deja los datasets listos para entrenar, y el tercero entrena y registra el mejor modelo. La API después consume ese modelo desde el Registry de MLflow.

```
CSV crudo -> [seed_raw_data] -> MinIO: data/raw/
                                       |
                                       v
              [data_pipeline]  -> limpieza, split, features, PCA
                                  MinIO: data/processed/ + artefactos
                                       |
                                       v  (TriggerDagRun automático)
              [training_pipeline] -> entrena 8 modelos x {con PCA, sin PCA}
                                     MLflow Tracking + Registry (alias "champion")
                                       |
                                       v
              FastAPI /predict  -> "models:/CarPriceModel@champion"
                                       |
                                       v
              Streamlit (UI)    -> consume la API
```

### Componentes y puertos

| Servicio   | URL                     | Credenciales         |
|------------|-------------------------|----------------------|
| Airflow    | http://localhost:8080   | airflow / airflow    |
| MLflow     | http://localhost:5001   | -                    |
| MinIO (UI) | http://localhost:9001   | minio / minio123     |
| FastAPI    | http://localhost:8000   | - (docs en /docs)    |
| Streamlit  | http://localhost:8501   | -                    |

## Cómo levantarlo

Hace falta Docker y Docker Compose. La primera vez que se buildean las imágenes tarda un rato.

```bash
cp .env.example .env
docker compose up --build -d
docker compose ps
```

Para bajar todo:

```bash
docker compose down                       # conserva datos
docker compose down -v                    # borra Postgres y MinIO (reset total)
docker compose down --rmi all --volumes   # además elimina las imágenes
```

## Orden de ejecución de los DAGs

Los tres DAGs están con `schedule=None`, así que se disparan a mano desde la UI de Airflow. La primera vez que se levanta el proyecto hay que correrlos en este orden:

1. `seed_raw_data`. Es un setup inicial: lee el CSV crudo de `modelo_ml/datasets/car_details_v4.csv` y lo sube a MinIO como Parquet en `s3://data/raw/`. Solo hace falta volver a correrlo si se hizo `docker compose down -v` y se borró MinIO.

2. `data_pipeline`. Es el ETL. Lee el Parquet crudo, hace la limpieza estructural, divide en train/test, aplica el feature engineering y entrena un PCA paralelo. Deja los cuatro datasets (`train`, `test`, `train_pca`, `test_pca`) en `s3://data/processed/` junto con los artefactos (`preprocessing_artifacts.joblib`, `pca.joblib`). Cuando termina dispara automáticamente el siguiente DAG, así que con triggerar este alcanza.

3. `training_pipeline`. Entrena los 8 modelos (baseline, linear_regression, svr, decision_tree, random_forest, adaboost, xgboost, lightgbm) en sus dos variantes (con y sin PCA), loguea cada run en MLflow bajo el experimento `car-price-prediction`, y registra el mejor como `CarPriceModel` con alias `champion`.

Pasos concretos en la UI:

1. Entrar a http://localhost:8080 (airflow / airflow).
2. Despausar los tres DAGs.
3. Disparar `seed_raw_data` y esperar a que termine (es rápido, unos 10 segundos).
4. Disparar `data_pipeline`. Cuando termina arranca solo `training_pipeline`.
5. Para verificar que salió todo bien, entrar a http://localhost:5001 y revisar que el experimento `car-price-prediction` tenga 16 runs (8 modelos por 2 variantes) y que en el Registry esté el modelo `CarPriceModel` con el alias `champion`.

## Uso de la API

La API se levanta junto con el resto del compose, pero para que `/predict` funcione tiene que haber un modelo registrado como `champion` en MLflow. Si todavía no se corrió el training, devuelve 503.

- Docs interactivas: http://localhost:8000/docs
- Healthcheck: `GET http://localhost:8000/health`

Ejemplo de request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "make": "Maruti",
    "year": 2019,
    "fuel_type": "Petrol",
    "transmission": "Manual",
    "location": "Mumbai",
    "color": "White",
    "owner": "First",
    "seller_type": "Individual",
    "drivetrain": "FWD",
    "engine_cc": 1197,
    "max_power_bhp": 82,
    "max_torque_nm": 113,
    "kilometer": 45000,
    "length": 3845,
    "width": 1695,
    "height": 1520,
    "seating_capacity": 5,
    "fuel_tank_capacity": 37
  }'
```

Respuesta:

```json
{ "predicted_price_inr": 512345.67, "model_alias": "champion" }
```

Para quien no quiera tirar curl, en http://localhost:8501 hay un formulario en Streamlit que arma el request y muestra la predicción.

## Decisiones del modelo

El target es `Price` en INR. Se le aplica `log1p` para estabilizar la varianza y después se reescala con `StandardScaler` junto al resto de las features numéricas. La inversión (deshacer el scaler y aplicar `expm1`) se hace adentro de la API antes de devolver el valor.

El feature engineering vive en [src/features.py](src/features.py). Lo principal: `Age = año_actual - Year`, parsing de unidades en `Engine` (cc), `Max Power` (bhp) y `Max Torque` (Nm), imputación de faltantes y encoding de las categóricas. Todo eso queda serializado en `preprocessing_artifacts.joblib` para que la API lo reaplique exactamente igual que durante el entrenamiento (si no, las predicciones quedan corridas).

Sobre el PCA, decidimos no reemplazar las features originales sino generar un dataset paralelo y entrenar cada modelo dos veces, una con cada versión. De ahí salen las 16 runs por corrida. El champion se elige por menor RMSE en test.

Para reproducibilidad, la `seed` está fijada en [src/config.py](src/config.py), y tanto el split train/test como el orden de columnas (`train_cols`) quedan congelados en los artefactos.

La API carga el modelo por alias (`models:/CarPriceModel@champion`) en vez de por número de versión. La gracia es que para promover un nuevo modelo alcanza con reasignar el alias desde la UI de MLflow, sin tocar el código de la API.

## Estructura del repo

```
.
├── airflow/dags/         seed_raw_data, data_pipeline, training_pipeline
├── api/                  FastAPI (main.py) + Dockerfile
├── src/                  código del pipeline (data, features, train, evaluate, config)
├── streamlit_app/        frontend
├── modelo_ml/            dataset crudo + notebooks exploratorios originales
├── dockerfiles/          imágenes custom (airflow, mlflow, api, streamlit, postgres)
├── docker-compose.yml
└── .env.example
```

## Troubleshooting

Si `/predict` devuelve 503, probablemente todavía no se corrió `training_pipeline` y no hay un champion en el Registry.

Si `data_pipeline` falla intentando leer `s3://data/raw/...`, falta haber corrido `seed_raw_data` antes.

Si Airflow no levanta, lo más común es que el puerto 8080 esté ocupado o que Docker no tenga suficiente RAM asignada.

Si en algún momento se hace `docker compose down -v`, se borran Postgres y MinIO, así que cuando se vuelva a levantar el stack hay que repetir los tres DAGs en orden.

## Hoja de ruta

- [X] Paso 1. Armar el repo en GitHub con la estructura de carpetas (`dags/`, `src/`, `api/`, `notebooks/`) y el `.gitignore` con `.env` excluido.
- [X] Paso 2. Levantar el stack con Docker Compose (Airflow en :8080, MLflow en :5001, MinIO en :9001).
- [X] Paso 3. Modularizar el código del notebook en archivos `.py` dentro de `src/` (data, features, train, evaluate).
- [X] Paso 4. Registrar experimentos con MLflow: loguear parámetros, métricas y modelo. Subir el mejor al Model Registry en `Staging`.
- [X] Paso 5. DAG de datos en Airflow: extraer CSV, validar, preprocesar y guardar en MinIO como Parquet.
- [X] Paso 6. DAG de entrenamiento en Airflow: leer datos, entrenar, evaluar, registrar y promover a `Production` en MLflow.
- [X] Paso 7. API con FastAPI: endpoint `/predict` que carga el modelo desde MLflow Registry y devuelve la predicción.
- [X] Paso 8. Containerizar la API y agregarla al `docker-compose.yml`.
- [X] Paso 9. Probar todo de punta a punta desde cero (clonar, levantar, disparar DAGs, llamar a la API).
- [X] Paso 10. Documentar el README final con instrucciones de instalación, uso de la API y decisiones del modelo (entrega final).
