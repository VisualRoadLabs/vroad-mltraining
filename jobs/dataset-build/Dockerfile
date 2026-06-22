# Imagen del job `job-dev-dataset-build-usc1` (materializa shards + manifiesto).
#
# El build context es la RAÍZ del repo: el cloudbuild hace
#   docker build -f jobs/dataset-build/Dockerfile .
# así que las rutas COPY son relativas a la raíz.
FROM python:3.12-slim

WORKDIR /app

# Paquete vroad-mlt + extra `gcp` (storage + bigquery) + numpy (dedup de CULane).
# NO necesita torch/opencv/scipy (eso es del trainer / la métrica).
COPY pyproject.toml ./
COPY libs/ ./libs/
RUN pip install --no-cache-dir ".[gcp]" "numpy>=2"

# Código del job + .env.example como config por defecto de dev (fallback de get_settings
# si el Cloud Run Job no trae las variables de entorno).
COPY .env.example ./
COPY jobs/dataset-build/ ./jobs/dataset-build/

# La spec llega como argumento en la ejecución del job:
#   gcloud run jobs execute job-dev-dataset-build-usc1 \
#     --args="--spec,gs://bkt-dev-datasets-usc1/_specs/culane.json"
ENTRYPOINT ["python", "jobs/dataset-build/run.py"]
