# trabajo_practico_aprendizaje_maquina_II

## Integrantes
- Agustín Biancardi (a2403)
- Marcos Lund (a2408)
- Alan Lihuen Vignolo (a2418)
- Tomás Agustín Mc Nally (a2420)

## Hoja de ruta 

- [X] **Paso 1** — Armar el repo en GitHub con la estructura de carpetas (`dags/`, `src/`, `api/`, `notebooks/`) y el `.gitignore` con `.env` excluido
- [X] **Paso 2** — Levantar el stack con Docker Compose (Airflow en :8080, MLflow en :5001, MinIO en :9001)
- [X] **Paso 3** — Modularizar el código del notebook en archivos `.py` dentro de `src/` (data, features, train, evaluate)
- [X] **Paso 4** — Registrar experimentos con MLflow: loguear parámetros, métricas y modelo. Subir el mejor al Model Registry en `Staging`
- [X] **Paso 5** — DAG de datos en Airflow: extraer CSV → validar → preprocesar y guardar en MinIO como Parquet
- [X] **Paso 6** — DAG de entrenamiento en Airflow: leer datos → entrenar → evaluar → registrar → promover a `Production` en MLflow
- [ ] **Paso 7** — API con FastAPI: endpoint `/predict` que carga el modelo desde MLflow Registry y devuelve la predicción
- [ ] **Paso 8** — Containerizar la API y agregarla al `docker-compose.yml`
- [ ] **Paso 9** — Probar todo de punta a punta desde cero (clonar → levantar → disparar DAGs → llamar a la API)
- [ ] **Paso 10** — Documentar el README final con instrucciones de instalación, uso de la API y decisiones del modelo*entrega final*


## Levantar el entorno con Docker

  Requisitos: Docker + Docker Compose y ~4 GB de RAM libres.

  ```bash
  cp .env.example .env          # valores por defecto ya sirven para desarrollo local
  docker compose up --build -d  # primera vez tarda (build de imágenes)
  docker compose ps             # esperar a que estén "healthy"

  Servicios:

  ┌─────────────────┬───────────────────────┬──────────────────────┐
  │    Servicio     │          URL          │ Usuario / contraseña │
  ├─────────────────┼───────────────────────┼──────────────────────┤
  │ Airflow         │ http://localhost:8080 │ airflow / airflow    │
  ├─────────────────┼───────────────────────┼──────────────────────┤
  │ MLflow          │ http://localhost:5001 │ —                    │
  ├─────────────────┼───────────────────────┼──────────────────────┤
  │ MinIO           │ http://localhost:9001 │ minio / minio123     │
  └─────────────────┴───────────────────────┴──────────────────────┘

  Para bajar todo: docker compose down (agregar -v para borrar también los datos de Postgres y MinIO).