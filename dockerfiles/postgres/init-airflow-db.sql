-- Runs once, automatically, the FIRST time the postgres volume is created.
-- (Files in /docker-entrypoint-initdb.d/ are skipped if db_data already exists.)
--
-- Purpose: this postgres already creates the MLflow database (POSTGRES_DB=mlflow_db).
-- Here we add a SECOND database + a dedicated role for Airflow, inside the same server.

-- Create the airflow role only if it doesn't exist yet.
DO
$$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'airflow') THEN
      CREATE ROLE airflow WITH LOGIN PASSWORD 'airflow';
   END IF;
END
$$;

-- Create the airflow database owned by that role.
-- CREATE DATABASE can't run inside the DO block above, so it's a plain statement.
-- This file only runs on a fresh volume, so a guard isn't strictly needed here.
CREATE DATABASE airflow OWNER airflow;

GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;
