"""vroad_mlt.naming — nombres y rutas según la convención del proyecto.

Centraliza la construcción de TODOS los identificadores y rutas para no escribir
strings a mano por ahí (y que un cambio de convención sea de un solo sitio):

- identidad de una corrida: `run_id`, nombre de experimento `<study>/<variant>`,
- árbol del work_dir (`config.yaml`, logs, `checkpoints/`, `viz/`, `predictions/`),
- claves de los locks de GPU (`locks/a100.lock`, `locks/l4/slot-N.lock`),
- shards WebDataset, manifiestos, benchmark y artefactos `_model/` del bucket de datos,
- staging de Vertex (`jobs/`, `tensorboard/`) y handoff (`candidates/`).

Diseño:
- Lógica pura: NO toca GCP y NO depende de `config`. Las funciones que devuelven
  un `gs://...` reciben el nombre del bucket como argumento (lo pasa el llamante,
  típicamente desde `Settings`).
- Devuelve CLAVES relativas (sin `gs://` ni `/` inicial); `gs_uri()` compone la URI.
- Las claves de tiempo usan `YYYYMMDDTHHMMSSZ` (UTC, ISO básico sin separadores).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

__all__ = [
    "NamingError",
    "TS_FORMAT",
    "RUN_ID_SEP",
    "SHARD_DIGITS",
    "SPLITS",
    "format_ts",
    "parse_ts",
    "utc_timestamp",
    "validate_segment",
    "validate_split",
    "validate_ts",
    "run_id",
    "parse_run_id",
    "experiment_name",
    "Workdir",
    "gs_uri",
    "split_gs_uri",
    "shards_prefix",
    "shard_name",
    "shard_key",
    "manifest_key",
    "curve_keys_key",
    "benchmark_prefix",
    "benchmark_shard_key",
    "benchmark_categories_key",
    "model_artifacts_prefix",
    "config_schema_key",
    "metrics_spec_key",
    "model_info_key",
    "vertex_job_dir",
    "tensorboard_dir",
    "a100_lock_key",
    "l4_lock_key",
    "l4_lock_keys",
    "candidate_prefix",
    "candidate_best_key",
    "candidate_model_card_key",
]

TS_FORMAT = "%Y%m%dT%H%M%SZ"  # 20260620T101500Z
RUN_ID_SEP = "__"
SHARD_DIGITS = 5  # train-00000.tar
SPLITS = ("train", "val", "test")

# Un segmento de ruta seguro: empieza por alfanumérico; luego alfanum/._-; sin '/'
# ni espacios. Además se prohíbe el separador de run_id ('__') para poder parsearlo.
_SEGMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_TS_RE = re.compile(r"\d{8}T\d{6}Z")


class NamingError(ValueError):
    """Un identificador o ruta no cumple la convención."""


# --------------------------------------------------------------------- tiempo


def format_ts(dt: datetime) -> str:
    """Formatea un datetime como `YYYYMMDDTHHMMSSZ` en UTC.

    Si `dt` es naive se asume UTC; si lleva tz se convierte a UTC.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime(TS_FORMAT)


def parse_ts(s: str) -> datetime:
    """Inversa de `format_ts`: devuelve un datetime tz-aware en UTC."""
    if not _TS_RE.fullmatch(s):
        raise NamingError(f"timestamp con formato inválido (esperado YYYYMMDDTHHMMSSZ): {s!r}")
    return datetime.strptime(s, TS_FORMAT).replace(tzinfo=timezone.utc)


def utc_timestamp() -> str:
    """Timestamp compacto del momento actual (UTC). Para generar `ts` nuevos."""
    return format_ts(datetime.now(timezone.utc))


# ----------------------------------------------------------------- validación


def validate_segment(value: str, *, field: str = "segmento") -> str:
    """Valida un segmento de ruta (study, variant, dataset, viz_name...)."""
    if not isinstance(value, str) or not value:
        raise NamingError(f"{field} vacío o no es texto")
    if not _SEGMENT_RE.fullmatch(value):
        raise NamingError(
            f"{field} inválido {value!r}: usa alfanumérico/._- y no empieces por símbolo"
        )
    if RUN_ID_SEP in value:
        raise NamingError(f"{field} no puede contener {RUN_ID_SEP!r}: {value!r}")
    return value


def validate_split(split: str) -> str:
    if split not in SPLITS:
        raise NamingError(f"split inválido {split!r}; esperado uno de {SPLITS}")
    return split


def validate_ts(ts: str) -> str:
    if not isinstance(ts, str) or not _TS_RE.fullmatch(ts):
        raise NamingError(f"timestamp inválido (esperado YYYYMMDDTHHMMSSZ): {ts!r}")
    return ts


# ------------------------------------------------------------- identidad de run


def run_id(study: str, variant: str, ts: str) -> str:
    """Clave única de una corrida (sin `/`, apta para `jobs/<run_id>/` y BigQuery)."""
    validate_segment(study, field="study")
    validate_segment(variant, field="variant")
    validate_ts(ts)
    return f"{study}{RUN_ID_SEP}{variant}{RUN_ID_SEP}{ts}"


def parse_run_id(rid: str) -> tuple[str, str, str]:
    """Inversa de `run_id` -> (study, variant, ts)."""
    parts = rid.split(RUN_ID_SEP)
    if len(parts) != 3:
        raise NamingError(f"run_id mal formado (esperado study{RUN_ID_SEP}variant{RUN_ID_SEP}ts): {rid!r}")
    study, variant, ts = parts
    validate_segment(study, field="study")
    validate_segment(variant, field="variant")
    validate_ts(ts)
    return study, variant, ts


def experiment_name(study: str, variant: str) -> str:
    """Nombre que viaja por Vertex Experiments/Run: `<study>/<variant>`."""
    validate_segment(study, field="study")
    validate_segment(variant, field="variant")
    return f"{study}/{variant}"


# ------------------------------------------------------------------- work_dir


@dataclass(frozen=True)
class Workdir:
    """El árbol de salida de una corrida en `bkt-dev-training-workdirs-usc1`.

    Construido una vez con (study, variant, ts), expone las claves relativas de
    todos sus ficheros. Para URIs completas, pasa el bucket a `uri()`/`file_uri()`.
    """

    study: str
    variant: str
    ts: str

    def __post_init__(self) -> None:
        validate_segment(self.study, field="study")
        validate_segment(self.variant, field="variant")
        validate_ts(self.ts)

    # --- identidad ---
    @property
    def key(self) -> str:
        """Clave del directorio del run: `<study>/<variant>_<ts>`."""
        return f"{self.study}/{self.variant}_{self.ts}"

    @property
    def run_id(self) -> str:
        return run_id(self.study, self.variant, self.ts)

    @property
    def experiment_name(self) -> str:
        return experiment_name(self.study, self.variant)

    # --- composición ---
    def file(self, *parts: str) -> str:
        """Clave relativa de un fichero dentro del work_dir."""
        return _join(self.key, *parts)

    def uri(self, bucket: str) -> str:
        """URI `gs://` del directorio del run."""
        return gs_uri(bucket, self.key)

    def file_uri(self, bucket: str, *parts: str) -> str:
        return gs_uri(bucket, self.file(*parts))

    # --- ficheros con nombre fijo ---
    @property
    def config_yaml(self) -> str:
        return self.file("config.yaml")

    @property
    def manifest_lock(self) -> str:
        return self.file("manifest.lock.json")

    @property
    def train_log(self) -> str:
        return self.file("train.log")

    @property
    def eval_log(self) -> str:
        return self.file("eval.log")

    @property
    def gpu_log(self) -> str:
        return self.file("gpu.log")

    @property
    def results_json(self) -> str:
        return self.file("results.json")

    # --- checkpoints / viz / predicciones ---
    def checkpoint(self, which: str = "best") -> str:
        if which not in ("best", "last"):
            raise NamingError(f"checkpoint con nombre inválido {which!r} (best|last)")
        return self.file("checkpoints", f"{which}.pth")

    def epoch_checkpoint(self, epoch: int) -> str:
        return self.file("checkpoints", f"epoch_{int(epoch):03d}.pth")

    def viz(self, viz_name: str, epoch: int, img_index: int) -> str:
        validate_segment(viz_name, field="viz_name")
        if int(img_index) < 0:
            raise NamingError(f"img_index debe ser >= 0: {img_index}")
        return self.file("viz", f"epoch_{int(epoch):03d}", viz_name, f"img{int(img_index)}.png")

    def predictions(self, split: str) -> str:
        validate_split(split)
        return self.file("predictions", f"{split}.jsonl")


# --------------------------------------------------------------------- URIs gs


def _join(*parts: str) -> str:
    """Une partes en una clave relativa, normalizando `/` sobrantes."""
    out: list[str] = []
    for p in parts:
        if p is None:
            continue
        out.extend(seg for seg in str(p).strip("/").split("/") if seg)
    return "/".join(out)


def gs_uri(bucket: str, *parts: str) -> str:
    """Compone una URI `gs://bucket/parte1/parte2`. Acepta bucket con o sin `gs://`."""
    if not bucket:
        raise NamingError("bucket vacío")
    bucket = bucket[len("gs://"):] if bucket.startswith("gs://") else bucket
    bucket = bucket.strip("/")
    key = _join(*parts)
    return f"gs://{bucket}/{key}" if key else f"gs://{bucket}"


def split_gs_uri(uri: str) -> tuple[str, str]:
    """Inversa de `gs_uri` -> (bucket, clave). La clave puede ser ''."""
    if not uri.startswith("gs://"):
        raise NamingError(f"no es una URI gs://: {uri!r}")
    rest = uri[len("gs://"):]
    bucket, _, key = rest.partition("/")
    if not bucket:
        raise NamingError(f"URI gs:// sin bucket: {uri!r}")
    return bucket, key


# ------------------------------------------------ bucket de datasets (datasets)


def shards_prefix(dataset: str, version: str) -> str:
    validate_segment(dataset, field="dataset")
    validate_segment(version, field="version")
    return f"shards/{dataset}@{version}"


def shard_name(split: str, index: int, *, digits: int = SHARD_DIGITS) -> str:
    validate_split(split)
    if int(index) < 0:
        raise NamingError(f"índice de shard negativo: {index}")
    return f"{split}-{int(index):0{digits}d}.tar"


def shard_key(dataset: str, version: str, split: str, index: int, *, digits: int = SHARD_DIGITS) -> str:
    return _join(shards_prefix(dataset, version), shard_name(split, index, digits=digits))


def manifest_key(manifest_id: str, version: str) -> str:
    validate_segment(manifest_id, field="manifest_id")
    validate_segment(version, field="version")
    return f"manifests/{manifest_id}@{version}.json"


def curve_keys_key(manifest_id: str, version: str) -> str:
    validate_segment(manifest_id, field="manifest_id")
    validate_segment(version, field="version")
    return f"manifests/{manifest_id}@{version}.curve_keys.json"


def benchmark_prefix(benchmark: str = "culane@v1") -> str:
    # `benchmark` viene como un token tipo 'culane@v1'.
    return f"benchmarks/{benchmark}"


def benchmark_shard_key(split: str, index: int, *, benchmark: str = "culane@v1", digits: int = SHARD_DIGITS) -> str:
    return _join(benchmark_prefix(benchmark), shard_name(split, index, digits=digits))


def benchmark_categories_key(benchmark: str = "culane@v1") -> str:
    return _join(benchmark_prefix(benchmark), "categories.json")


def model_artifacts_prefix(model_name: str, sha: str) -> str:
    validate_segment(model_name, field="model_name")
    validate_segment(sha, field="sha")
    return f"_model/{model_name}@{sha}"


def config_schema_key(model_name: str, sha: str) -> str:
    return _join(model_artifacts_prefix(model_name, sha), "config_schema.json")


def metrics_spec_key(model_name: str, sha: str) -> str:
    return _join(model_artifacts_prefix(model_name, sha), "metrics_spec.json")


def model_info_key(model_name: str, sha: str) -> str:
    return _join(model_artifacts_prefix(model_name, sha), "model_info.json")


# --------------------------------------------- bucket de staging de Vertex


def vertex_job_dir(rid: str) -> str:
    """`baseOutputDirectory` del Custom Job: `jobs/<run_id>`."""
    return f"jobs/{rid}"


def tensorboard_dir(rid: str) -> str:
    return f"tensorboard/{rid}"


def a100_lock_key() -> str:
    return "locks/a100.lock"


def l4_lock_key(slot: int) -> str:
    if int(slot) < 0:
        raise NamingError(f"slot L4 negativo: {slot}")
    return f"locks/l4/slot-{int(slot)}.lock"


def l4_lock_keys(count: int) -> list[str]:
    """Las claves de los `count` slots L4 (0..count-1)."""
    return [l4_lock_key(i) for i in range(int(count))]


# ----------------------------------------------------- bucket de handoff


def candidate_prefix(study: str, variant: str) -> str:
    validate_segment(study, field="study")
    validate_segment(variant, field="variant")
    return f"candidates/{study}/{variant}"


def candidate_best_key(study: str, variant: str) -> str:
    return _join(candidate_prefix(study, variant), "best.pth")


def candidate_model_card_key(study: str, variant: str) -> str:
    return _join(candidate_prefix(study, variant), "model_card.json")


# --------------------------------------------------------------------------- CLI


def _main(argv: Optional[list[str]] = None) -> int:
    """Comprobación manual: imprime un árbol de nombres de ejemplo.

        python -m vroad_mlt.naming
    """
    study, variant, ts = "lr-sweep", "lr1e-4", "20260620T101500Z"
    wd = Workdir(study, variant, ts)
    datasets = "bkt-dev-datasets-usc1"
    workdirs = "bkt-dev-training-workdirs-usc1"
    staging = "bkt-dev-vertex-staging-usc1"
    handoff = "bkt-dev-training-handoff-usc1"

    print(f"timestamp (now)   : {utc_timestamp()}")
    print(f"run_id            : {wd.run_id}")
    print(f"experiment_name   : {wd.experiment_name}")
    print(f"workdir uri       : {wd.uri(workdirs)}")
    print(f"  config.yaml     : {wd.file_uri(workdirs, 'config.yaml')}")
    print(f"  train.log       : {wd.train_log}")
    print(f"  best.pth        : {wd.checkpoint('best')}")
    print(f"  epoch 5 ckpt    : {wd.epoch_checkpoint(5)}")
    print(f"  viz gt_vs_pred  : {wd.viz('gt_vs_pred', 5, 0)}")
    print(f"  predictions val : {wd.predictions('val')}")
    print(f"shard (culane@1)  : {gs_uri(datasets, shard_key('culane', '1', 'train', 0))}")
    print(f"manifest          : {gs_uri(datasets, manifest_key('culane-mix-curves', '3'))}")
    print(f"curve_keys        : {manifest_key('culane-mix-curves', '3').replace('.json', '.curve_keys.json')}")
    print(f"benchmark shard   : {gs_uri(datasets, benchmark_shard_key('val', 0))}")
    print(f"config_schema     : {gs_uri(datasets, config_schema_key('lanetr', 'dd2f8ab'))}")
    print(f"vertex job dir    : {gs_uri(staging, vertex_job_dir(wd.run_id))}")
    print(f"tensorboard dir   : {gs_uri(staging, tensorboard_dir(wd.run_id))}")
    print(f"lock a100         : {gs_uri(staging, a100_lock_key())}")
    print(f"locks l4 (x3)     : {[gs_uri(staging, k) for k in l4_lock_keys(3)]}")
    print(f"handoff best.pth  : {gs_uri(handoff, candidate_best_key(study, variant))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
