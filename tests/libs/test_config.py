"""Tests del cargador de configuración (vroad_mlt.config)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from vroad_mlt import config as cfgmod
from vroad_mlt.config import ConfigError, Settings, parse_env_file

REPO_ROOT = Path(__file__).parents[2]  # tests/libs/ -> raíz del repo

# Entorno mínimo válido (solo lo requerido); el resto usa defaults/derivaciones.
MINIMAL = {
    "PROJECT_TRAINING": "vr-prj-dev-training-v1",
    "PROJECT_CICD": "vr-prj-dev-cicd-v1",
    "PROJECT_DATALAKE": "vr-prj-prod-data-v1",
    "AR_HOST": "us-central1-docker.pkg.dev",
    "AR_REPO": "ml-training",
    "LANETR_REPO": "https://github.com/VisualRoadLabs/vroad-lanetr",
    "LANETR_SHA": "dd2f8ab87a8071e34fd3511fc82e5c71def85e07",
    "BUCKET_DATASETS": "bkt-dev-datasets-usc1",
    "BUCKET_WORKDIRS": "bkt-dev-training-workdirs-usc1",
    "BUCKET_HANDOFF": "bkt-dev-training-handoff-usc1",
    "BUCKET_VERTEX_STAGING": "bkt-dev-vertex-staging-usc1",
    "DATALAKE_IMAGES_PUBLIC": "bkt-prod-public-usc1",
    "DATALAKE_IMAGES_USER": "bkt-prod-user-usc1",
    "BQ_DATASET_EXPERIMENTS": "ds_experiments",
    "VERTEX_TENSORBOARD": "tb-dev-lanetr-usc1",
}


def _settings(environ, **kw):
    # env_file=None para no leer el `.env`/`.env.example` reales del repo.
    return Settings.from_env(env_file=None, environ=environ, **kw)


# ------------------------------------------------------------ parse_env_file

def test_parse_env_file_handles_comments_quotes_export_and_urls(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "\n".join(
            [
                "# comentario",
                "",
                "GCP_REGION=us-central1",
                'AR_REPO="ml-training"',
                "export PROJECT_TRAINING=vr-prj-dev-training-v1",
                "LANETR_REPO=https://github.com/VisualRoadLabs/vroad-lanetr",
                "SIN_IGUAL_SE_IGNORA",
                "TRAINING_PROJECT_NUMBER=__rellenar__",
            ]
        ),
        encoding="utf-8",
    )
    vals = parse_env_file(p)
    assert vals["GCP_REGION"] == "us-central1"
    assert vals["AR_REPO"] == "ml-training"  # comillas eliminadas
    assert vals["PROJECT_TRAINING"] == "vr-prj-dev-training-v1"  # export eliminado
    assert vals["LANETR_REPO"].endswith("vroad-lanetr")  # '=' en valor preservado
    assert "SIN_IGUAL_SE_IGNORA" not in vals
    assert vals["TRAINING_PROJECT_NUMBER"] == "__rellenar__"  # placeholder crudo aquí


# ----------------------------------------------------------------- from_env

def test_minimal_environ_builds_valid_settings():
    s = _settings(MINIMAL)
    assert s.project_training == "vr-prj-dev-training-v1"
    assert s.region == "us-central1"  # default
    assert s.max_a100 == 1 and s.max_l4 == 3  # defaults
    assert s.accel_iter == "NVIDIA_L4"


def test_lock_prefix_derived_from_staging_bucket():
    s = _settings(MINIMAL)
    assert s.lock_prefix == "gs://bkt-dev-vertex-staging-usc1/locks"


def test_lock_prefix_can_be_overridden():
    s = _settings({**MINIMAL, "LOCK_PREFIX": "gs://otro/locks"})
    assert s.lock_prefix == "gs://otro/locks"


def test_service_accounts_derived_from_project():
    s = _settings(MINIMAL)
    assert s.sa_trainer == "sa-mlt-trainer@vr-prj-dev-training-v1.iam.gserviceaccount.com"
    assert s.sa_workflow == "sa-mlt-workflow@vr-prj-dev-training-v1.iam.gserviceaccount.com"


def test_service_account_override():
    s = _settings({**MINIMAL, "SA_TRAINER": "custom@x.iam.gserviceaccount.com"})
    assert s.sa_trainer == "custom@x.iam.gserviceaccount.com"


def test_lanetr_pip_url_and_image_uri():
    s = _settings(MINIMAL)
    assert s.lanetr_pip_url == (
        "git+https://github.com/VisualRoadLabs/vroad-lanetr@"
        "dd2f8ab87a8071e34fd3511fc82e5c71def85e07"
    )
    assert s.image_uri("trainer", s.lanetr_sha) == (
        "us-central1-docker.pkg.dev/vr-prj-dev-cicd-v1/ml-training/trainer:"
        "dd2f8ab87a8071e34fd3511fc82e5c71def85e07"
    )


def test_settings_is_frozen():
    s = _settings(MINIMAL)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.region = "europe-west1"  # type: ignore[misc]


# ----------------------------------------------- precedencia entorno vs fichero

def test_environ_overrides_env_file(tmp_path):
    p = tmp_path / ".env"
    lines = [f"{k}={v}" for k, v in MINIMAL.items()]
    p.write_text("\n".join(lines), encoding="utf-8")
    # El fichero dice us-central1; el entorno fuerza otra región -> gana el entorno.
    s = Settings.from_env(env_file=p, environ={"GCP_REGION": "europe-west1"})
    assert s.region == "europe-west1"
    assert s.project_training == "vr-prj-dev-training-v1"  # viene del fichero


# ----------------------------------------------------------------- inválidos

def test_missing_required_lists_all_at_once():
    with pytest.raises(ConfigError) as exc:
        _settings({"PROJECT_TRAINING": "x"})  # falta casi todo
    msg = str(exc.value)
    assert "PROJECT_CICD" in msg
    assert "BUCKET_DATASETS" in msg
    assert "LANETR_SHA" in msg


def test_placeholder_is_treated_as_unset():
    bad = {**MINIMAL, "PROJECT_CICD": "__rellenar__"}
    with pytest.raises(ConfigError, match="PROJECT_CICD"):
        _settings(bad)


def test_optional_placeholder_becomes_none():
    s = _settings({**MINIMAL, "TRAINING_PROJECT_NUMBER": "__rellenar__"})
    assert s.training_project_number is None


def test_bad_int_raises():
    with pytest.raises(ConfigError, match="MAX_L4"):
        _settings({**MINIMAL, "MAX_L4": "tres"})


def test_int_below_one_raises():
    with pytest.raises(ConfigError, match="MAX_A100"):
        _settings({**MINIMAL, "MAX_A100": "0"})


def test_bad_sha_raises():
    with pytest.raises(ConfigError, match="LANETR_SHA"):
        _settings({**MINIMAL, "LANETR_SHA": "no-es-un-sha"})


def test_lanetr_optional_when_not_required():
    env = {k: v for k, v in MINIMAL.items() if k != "LANETR_SHA"}
    s = _settings(env, require_lanetr=False)
    assert s.lanetr_sha is None
    with pytest.raises(ConfigError, match="no hay pin"):
        _ = s.lanetr_pip_url


# -------------------------------------------- el .env.example del repo es válido

def test_repo_env_example_is_complete_and_valid():
    example = REPO_ROOT / ".env.example"
    # Sin entorno que interfiera: solo el fichero de ejemplo.
    s = Settings.from_env(env_file=example, environ={})
    assert s.project_training == "vr-prj-dev-training-v1"
    assert s.bucket_datasets == "bkt-dev-datasets-usc1"
    assert s.lanetr_sha == "dd2f8ab87a8071e34fd3511fc82e5c71def85e07"
