# Imagen del Vertex Custom Job de entrenamiento (`trainer`).
#
# Build context = RAÍZ del repo:
#   docker build -f trainer/Dockerfile -t <img> .
# torch + CUDA vienen en la base de PyTorch; encima va vroad-mlt + lanetr (pineado por SHA).
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# git -> instalar lanetr desde su repo por SHA. libglib2.0-0/libgl1 -> OpenCV (la métrica).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Paquete vroad-mlt + extras: gcp (gcs/bq), metric (F1 CULane: numpy/scipy/opencv), trainer
# (aiplatform + pyyaml). torch NO se reinstala (ya está en la base).
COPY pyproject.toml ./
COPY libs/ ./libs/
RUN pip install --no-cache-dir ".[gcp,metric,trainer]"

# Código del trainer (incluye trainer/lanetr_pin.txt).
COPY trainer/ ./trainer/

# Modelo PINEADO por SHA (debe coincidir con el schema de _model/lanetr@<sha>/ que lee el dashboard).
# El repo es privado: el build (Cloud Build) aporta las credenciales de git.
RUN pip install --no-cache-dir \
        "git+https://github.com/VisualRoadLabs/vroad-lanetr@$(cat trainer/lanetr_pin.txt)"

# La configuración (proyecto, buckets, RUN_ID, VERTEX_EXPERIMENT, ...) llega como VARIABLES DE
# ENTORNO en el Vertex Custom Job; los args (study/variant/manifest/--set) en `containerSpec.args`.
ENTRYPOINT ["python", "-m", "trainer.run"]
