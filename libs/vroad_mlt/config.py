"""vroad_mlt.config — la única fuente de verdad de configuración.

Lee `.env` / variables de entorno y entrega un objeto `Settings` validado e
inmutable (proyectos, buckets, dataset de BigQuery, SHA de lanetr, aceleradores,
concurrencia, service accounts...). Lo importan los demás módulos para no leer
`os.environ` a mano por todas partes.

Diseño:
- Lógica pura: NO toca GCP, solo lee configuración y valida.
- Núcleo ligero: parser `.env` propio (stdlib), sin dependencias.
- Precedencia: las variables de entorno reales GANAN al fichero `.env`
  (en Cloud Run/Vertex el entorno trae los valores; `.env` es para local).
- Los valores `__rellenar__` del `.env.example` se tratan como NO definidos.
- Errores agrupados: si falta/está mal más de una clave, se listan todas juntas.

Uso típico:
    from vroad_mlt.config import get_settings
    cfg = get_settings()
    cfg.bucket_datasets          # 'bkt-dev-datasets-usc1'
    cfg.lanetr_pip_url           # 'git+https://github.com/.../vroad-lanetr@dd2f8ab...'
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional

__all__ = [
    "PLACEHOLDER",
    "ConfigError",
    "Settings",
    "parse_env_file",
    "get_settings",
]

# Sentinela del `.env.example`: un valor así = "no definido".
PLACEHOLDER = "__rellenar__"

_SHA_RE = re.compile(r"[0-9a-f]{7,40}")


class ConfigError(ValueError):
    """La configuración es incompleta o inválida."""


def parse_env_file(path: os.PathLike | str) -> dict[str, str]:
    """Parsea un fichero estilo `.env` -> dict (sin mutar `os.environ`).

    Soporta: comentarios (`#`), líneas en blanco, `export KEY=val`, comillas
    alrededor del valor y `=` dentro del valor (URLs). No interpreta comentarios
    en línea ni expande variables.
    """
    values: dict[str, str] = {}
    text = Path(path).read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if (len(val) >= 2) and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key:
            values[key] = val
    return values


@dataclass(frozen=True)
class Settings:
    """Configuración resuelta y validada (inmutable)."""

    # Región y proyectos
    region: str
    project_training: str
    project_cicd: str
    project_datalake: str
    project_validation: Optional[str]
    training_project_number: Optional[str]
    cicd_project_number: Optional[str]
    datalake_project_number: Optional[str]

    # Artifact Registry / modelo
    ar_host: str
    ar_repo: str
    lanetr_repo: str
    lanetr_sha: Optional[str]

    # Buckets propios
    bucket_datasets: str
    bucket_workdirs: str
    bucket_handoff: str
    bucket_vertex_staging: str

    # Data Lake (solo lectura)
    datalake_images_public: str
    datalake_images_user: str

    # BigQuery
    bq_dataset_experiments: str

    # Vertex
    vertex_tensorboard: str
    accel_iter: str
    machine_iter: str
    accel_final: str
    machine_final: str

    # Concurrencia (locks de GPU)
    lock_prefix: str
    max_a100: int
    max_l4: int

    # Service accounts (derivadas del proyecto si no se fijan)
    sa_trainer: str
    sa_dataset: str
    sa_dashboard: str
    sa_workflow: str

    # ------------------------------------------------------------------ helpers

    @property
    def lanetr_pip_url(self) -> str:
        """URL `pip install` del modelo pineado por SHA."""
        if not self.lanetr_sha:
            raise ConfigError("LANETR_SHA no está definido; no hay pin de lanetr")
        return f"git+{self.lanetr_repo}@{self.lanetr_sha}"

    def image_uri(self, name: str, tag: str) -> str:
        """URI de una imagen del repo `ml-training`, p. ej. `trainer:<sha>`."""
        return f"{self.ar_host}/{self.project_cicd}/{self.ar_repo}/{name}:{tag}"

    def as_dict(self) -> dict:
        """Vista serializable (para el CLI / logging)."""
        return asdict(self)

    # ----------------------------------------------------------------- fábrica

    @classmethod
    def from_env(
        cls,
        *,
        env_file: Optional[os.PathLike | str] = ".env",
        environ: Optional[Mapping[str, str]] = None,
        require_lanetr: bool = True,
    ) -> "Settings":
        """Construye `Settings` desde entorno + `.env` (entorno gana).

        Lanza `ConfigError` listando TODO lo que falte o esté mal.
        """
        environ = os.environ if environ is None else environ
        file_values: dict[str, str] = {}
        if env_file is not None and Path(env_file).exists():
            file_values = parse_env_file(env_file)

        errors: list[str] = []

        def get(key: str, default: Optional[str] = None) -> Optional[str]:
            v = environ.get(key)
            if v is None or v == "":
                v = file_values.get(key)
            if v is None or v == "" or v == PLACEHOLDER:
                return default
            return v

        def required(key: str) -> Optional[str]:
            v = get(key)
            if v is None:
                errors.append(key)
            return v

        def get_int(key: str, default: int) -> int:
            raw = get(key)
            if raw is None:
                return default
            try:
                iv = int(raw)
            except ValueError:
                errors.append(f"{key} (no es un entero: {raw!r})")
                return default
            if iv < 1:
                errors.append(f"{key} (debe ser >= 1: {iv})")
            return iv

        # Requeridos
        region = get("GCP_REGION", "us-central1") or "us-central1"
        project_training = required("PROJECT_TRAINING")
        project_cicd = required("PROJECT_CICD")
        project_datalake = required("PROJECT_DATALAKE")
        ar_host = required("AR_HOST")
        ar_repo = required("AR_REPO")
        lanetr_repo = required("LANETR_REPO")
        bucket_datasets = required("BUCKET_DATASETS")
        bucket_workdirs = required("BUCKET_WORKDIRS")
        bucket_handoff = required("BUCKET_HANDOFF")
        bucket_vertex_staging = required("BUCKET_VERTEX_STAGING")
        datalake_images_public = required("DATALAKE_IMAGES_PUBLIC")
        datalake_images_user = required("DATALAKE_IMAGES_USER")
        bq_dataset_experiments = required("BQ_DATASET_EXPERIMENTS")
        vertex_tensorboard = required("VERTEX_TENSORBOARD")

        # SHA de lanetr (requerido salvo que se desactive explícitamente)
        lanetr_sha = get("LANETR_SHA")
        if lanetr_sha is not None and not _SHA_RE.fullmatch(lanetr_sha):
            errors.append(f"LANETR_SHA (no es un SHA hex de 7-40: {lanetr_sha!r})")
        elif lanetr_sha is None and require_lanetr:
            errors.append("LANETR_SHA")

        # Opcionales / con default
        project_validation = get("PROJECT_VALIDATION")
        training_project_number = get("TRAINING_PROJECT_NUMBER")
        cicd_project_number = get("CICD_PROJECT_NUMBER")
        datalake_project_number = get("DATALAKE_PROJECT_NUMBER")
        accel_iter = get("ACCEL_ITER", "NVIDIA_L4")
        machine_iter = get("MACHINE_ITER", "g2-standard-12")
        accel_final = get("ACCEL_FINAL", "NVIDIA_TESLA_A100")
        machine_final = get("MACHINE_FINAL", "a2-highgpu-1g")
        max_a100 = get_int("MAX_A100", 1)
        max_l4 = get_int("MAX_L4", 3)

        if errors:
            raise ConfigError(
                "configuración incompleta o inválida -> "
                + ", ".join(errors)
                + (f"  (env_file={env_file})" if env_file else "")
            )

        # Derivaciones (seguras: ya no hay errores, los requeridos existen)
        lock_prefix = get("LOCK_PREFIX") or f"gs://{bucket_vertex_staging}/locks"

        def sa_default(fn: str, override_key: str) -> str:
            return get(override_key) or f"sa-mlt-{fn}@{project_training}.iam.gserviceaccount.com"

        return cls(
            region=region,
            project_training=project_training,  # type: ignore[arg-type]
            project_cicd=project_cicd,  # type: ignore[arg-type]
            project_datalake=project_datalake,  # type: ignore[arg-type]
            project_validation=project_validation,
            training_project_number=training_project_number,
            cicd_project_number=cicd_project_number,
            datalake_project_number=datalake_project_number,
            ar_host=ar_host,  # type: ignore[arg-type]
            ar_repo=ar_repo,  # type: ignore[arg-type]
            lanetr_repo=lanetr_repo,  # type: ignore[arg-type]
            lanetr_sha=lanetr_sha,
            bucket_datasets=bucket_datasets,  # type: ignore[arg-type]
            bucket_workdirs=bucket_workdirs,  # type: ignore[arg-type]
            bucket_handoff=bucket_handoff,  # type: ignore[arg-type]
            bucket_vertex_staging=bucket_vertex_staging,  # type: ignore[arg-type]
            datalake_images_public=datalake_images_public,  # type: ignore[arg-type]
            datalake_images_user=datalake_images_user,  # type: ignore[arg-type]
            bq_dataset_experiments=bq_dataset_experiments,  # type: ignore[arg-type]
            vertex_tensorboard=vertex_tensorboard,  # type: ignore[arg-type]
            accel_iter=accel_iter,  # type: ignore[arg-type]
            machine_iter=machine_iter,  # type: ignore[arg-type]
            accel_final=accel_final,  # type: ignore[arg-type]
            machine_final=machine_final,  # type: ignore[arg-type]
            lock_prefix=lock_prefix,
            max_a100=max_a100,
            max_l4=max_l4,
            sa_trainer=sa_default("trainer", "SA_TRAINER"),
            sa_dataset=sa_default("dataset", "SA_DATASET"),
            sa_dashboard=sa_default("dashboard", "SA_DASHBOARD"),
            sa_workflow=sa_default("workflow", "SA_WORKFLOW"),
        )


_cached: Optional[Settings] = None


def get_settings(
    *,
    reload: bool = False,
    env_file: Optional[os.PathLike | str] = ".env",
    environ: Optional[Mapping[str, str]] = None,
    require_lanetr: bool = True,
) -> Settings:
    """Devuelve la configuración (cacheada). `reload=True` la recalcula.

    Comodidad de dev: si el `.env` no existe pero sí `.env.example`, usa ese (en
    Cloud Run/Vertex mandan las variables de entorno reales, no el fichero).
    """
    global _cached
    if _cached is None or reload:
        if (
            env_file is not None
            and not Path(env_file).exists()
            and Path(".env.example").exists()
        ):
            env_file = ".env.example"
        _cached = Settings.from_env(
            env_file=env_file, environ=environ, require_lanetr=require_lanetr
        )
    return _cached


# --------------------------------------------------------------------------- CLI


def _main(argv: Optional[list[str]] = None) -> int:
    """Comprobación manual: resuelve e imprime la configuración.

        python -m vroad_mlt.config [--env-file PATH] [--json]

    Sale 0 si la config es válida, 2 si falta/está mal algo.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    env_file: Optional[str] = ".env"
    if "--env-file" in args:
        i = args.index("--env-file")
        try:
            env_file = args[i + 1]
        except IndexError:
            print("ERROR: --env-file necesita una ruta", file=sys.stderr)
            return 2
        del args[i : i + 2]

    try:
        cfg = Settings.from_env(env_file=env_file)
    except ConfigError as e:
        print(f"INVÁLIDO: {e}", file=sys.stderr)
        return 2

    if as_json:
        import json

        print(json.dumps(cfg.as_dict(), indent=2, ensure_ascii=False))
        return 0

    print(f"OK  (env_file={env_file})")
    for k, v in cfg.as_dict().items():
        print(f"  {k:<26}: {v}")
    print(f"  lanetr_pip_url            : {cfg.lanetr_pip_url}")
    print(f"  image_uri(trainer)        : {cfg.image_uri('trainer', cfg.lanetr_sha or 'latest')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
