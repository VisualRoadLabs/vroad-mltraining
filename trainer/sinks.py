"""trainer.sinks — dónde aterriza lo que produce el entrenamiento (BigQuery, workdir, tracker).

Sub-pieza T5a (esta): las dos tablas de BigQuery.
- `tbl_train_metrics`: SERIE TEMPORAL en FORMATO LARGO (1 fila por `(run_id, metric, step)`). El loop
  loguea lo que quiera (`loss/*`, `lr`, `grad_norm`, `f1/*`, `gpu/*`); el dashboard plotea toda
  clave agrupada por prefijo. MERGE upsert → reintento del Vertex job no duplica puntos.
- `tbl_experiments`: la fila final del run, por MERGE PARCIAL sobre `run_id` (el workflow ya creó la
  fila con el estado; el trainer solo añade sus columnas: `f1_global`, umbral, 9 categorías, etc.).

`bq` se INYECTA (la lógica se prueba con un FakeBq; sin GCP en los tests).
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional

from vroad_mlt.logging import fmt_decimal
from vroad_mlt.naming import Workdir, gs_uri

__all__ = [
    "DS_EXPERIMENTS",
    "TBL_TRAIN_METRICS",
    "TBL_EXPERIMENTS",
    "TRAIN_METRICS_KEYS",
    "TRAIN_METRICS_TYPES",
    "metric_rows",
    "BqMetricsSink",
    "format_train_line",
    "format_eval_line",
    "format_gpu_line",
    "WorkdirSink",
    "NoopTracker",
    "VertexTracker",
    "make_tracker",
]

DS_EXPERIMENTS = "ds_experiments"
TBL_TRAIN_METRICS = "tbl_train_metrics"
TBL_EXPERIMENTS = "tbl_experiments"

# Clave de upsert de la serie: cada (run, métrica, iteración global) es un punto único.
TRAIN_METRICS_KEYS = ("run_id", "metric", "step")
# Tipos explícitos (la columna `epoch` puede ser None → BigQuery necesita el tipo igualmente).
TRAIN_METRICS_TYPES = {
    "run_id": "STRING", "epoch": "INT64", "step": "INT64",
    "metric": "STRING", "value": "FLOAT64", "logged_at": "TIMESTAMP",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def metric_rows(
    run_id: str,
    metrics: Mapping[str, Any],
    *,
    step: int,
    epoch: Optional[int] = None,
    logged_at: Optional[datetime] = None,
) -> list[dict]:
    """Aplana `{metric: value}` a filas largas de `tbl_train_metrics` (una por métrica).

    `step` es la iteración GLOBAL (siempre puesta → la clave de upsert no tiene NULLs). `epoch` es
    informativo (puede ser None). El valor se castea a float (la tabla es FLOAT).
    """
    ts = logged_at or _utcnow()
    ep = None if epoch is None else int(epoch)
    return [
        {"run_id": str(run_id), "epoch": ep, "step": int(step),
         "metric": str(name), "value": float(value), "logged_at": ts}
        for name, value in metrics.items()
    ]


class BqMetricsSink:
    """Escribe métricas a BigQuery: la serie (`tbl_train_metrics`) y la fila final (`tbl_experiments`)."""

    def __init__(self, bq: Any, *, dataset: str = DS_EXPERIMENTS) -> None:
        self._bq = bq
        self._dataset = dataset

    def log_metrics(
        self,
        run_id: str,
        metrics: Mapping[str, Any],
        *,
        step: int,
        epoch: Optional[int] = None,
        logged_at: Optional[datetime] = None,
    ) -> int:
        """Upsert de un evento de métricas (formato largo). Devuelve nº de filas; 0 si no hay métricas."""
        rows = metric_rows(run_id, metrics, step=step, epoch=epoch, logged_at=logged_at)
        if not rows:
            return 0
        return self._bq.merge_upsert(
            self._dataset, TBL_TRAIN_METRICS, rows, TRAIN_METRICS_KEYS, types=TRAIN_METRICS_TYPES
        )

    def finalize_experiment(
        self,
        run_id: str,
        fields: Mapping[str, Any],
        *,
        types: Optional[Mapping[str, str]] = None,
    ) -> int:
        """MERGE PARCIAL de la fila del run en `tbl_experiments` (clave `run_id`).

        `fields` son solo las columnas que pone el trainer (p.ej. `f1_global`, `threshold`,
        `metrics_by_category_json`, `params_m`, `best_ckpt_uri`…); las JSON pueden ir como dict/list
        (se serializan solas). Pasa `types` solo para columnas con valor None.
        """
        row = {"run_id": str(run_id), **dict(fields)}
        return self._bq.merge_upsert(self._dataset, TBL_EXPERIMENTS, [row], ("run_id",), types=types)


# ===================================================================== T5b: workdir
#
# Artefactos HUMANOS de la corrida (bkt-dev-training-workdirs): logs legibles, results.json,
# config.yaml, manifest.lock.json y checkpoints best/last. Los `loss_*` van en DECIMAL (sin
# notación científica) vía `fmt_decimal`. `gcs` se inyecta (testeable con un FakeGcs).


def _fmt_eta(seconds: float) -> str:
    """Segundos -> `H:MM:SS` (legible en el log)."""
    return str(timedelta(seconds=int(seconds)))


def format_train_line(
    epoch: int, step: int, metrics: Mapping[str, Any], *, eta_s: Optional[float] = None
) -> str:
    """Línea de `train.log`: época/iteración + `loss_*`/lr/grad_norm (DECIMAL) + ETA opcional."""
    parts = [f"{name}={fmt_decimal(value)}" for name, value in metrics.items()]
    if eta_s is not None:
        parts.append(f"eta={_fmt_eta(eta_s)}")
    return f"[epoch {int(epoch):03d} step {int(step):06d}] " + " ".join(parts)


def format_eval_line(epoch: int, results: Mapping[str, Any]) -> str:
    """Línea de `eval.log`: F1 global + F1 por categoría (de `results.json`)."""
    parts = [f"f1/global={fmt_decimal(results['f1_global'])}"]
    parts += [f"{cat}={fmt_decimal(v)}" for cat, v in results.get("metrics_by_category", {}).items()]
    return f"[epoch {int(epoch):03d}] " + " ".join(parts)


def format_gpu_line(epoch: int, *, mem_peak_mb: float, util_mean: float, util_max: float) -> str:
    """Línea de `gpu.log`: memoria pico (MB) + utilización media/máx por época."""
    return (f"[epoch {int(epoch):03d}] mem_peak_mb={int(mem_peak_mb)} "
            f"util_mean={fmt_decimal(util_mean)} util_max={fmt_decimal(util_max)}")


class WorkdirSink:
    """Escribe los artefactos del workdir en GCS: logs, JSONs y checkpoints best/last.

    Los logs se ACUMULAN en memoria (texto pequeño) y se vuelcan enteros en cada `flush` (sobrescribe
    el objeto). Los checkpoints se serializan con `torch.save` a un buffer y se suben (sin disco
    local: vale en el tmpfs de Vertex). `save_epoch` guarda `last.pth` siempre y `best.pth` cuando
    mejora la F1.
    """

    def __init__(self, gcs: Any, bucket: str, workdir: Workdir) -> None:
        self._gcs = gcs
        self._bucket = bucket
        self._wd = workdir
        self._buffers: dict[str, list[str]] = {"train": [], "eval": [], "gpu": []}
        self._best_f1: Optional[float] = None

    def _uri(self, key: str) -> str:
        return gs_uri(self._bucket, key)

    # ----------------------------------------------------------------- logs
    def log_train(self, epoch: int, step: int, metrics: Mapping[str, Any], *,
                  eta_s: Optional[float] = None, flush: bool = False) -> None:
        self._buffers["train"].append(format_train_line(epoch, step, metrics, eta_s=eta_s))
        if flush:
            self.flush()

    def log_eval(self, epoch: int, results: Mapping[str, Any], *, flush: bool = True) -> None:
        self._buffers["eval"].append(format_eval_line(epoch, results))
        if flush:
            self.flush()

    def log_gpu(self, epoch: int, *, mem_peak_mb: float, util_mean: float, util_max: float,
                flush: bool = True) -> None:
        self._buffers["gpu"].append(format_gpu_line(epoch, mem_peak_mb=mem_peak_mb,
                                                     util_mean=util_mean, util_max=util_max))
        if flush:
            self.flush()

    def flush(self) -> None:
        """Vuelca los 3 logs acumulados a GCS (sobrescribe el objeto con todo el contenido)."""
        for name, key in (("train", self._wd.train_log), ("eval", self._wd.eval_log),
                          ("gpu", self._wd.gpu_log)):
            lines = self._buffers[name]
            if lines:
                self._gcs.write_text(self._uri(key), "\n".join(lines) + "\n")

    # ------------------------------------------------------------- JSON/config
    def write_results(self, results: Mapping[str, Any]) -> str:
        uri = self._uri(self._wd.results_json)
        self._gcs.write_json(uri, dict(results), indent=2)
        return uri

    def write_config(self, config_yaml: str) -> str:
        uri = self._uri(self._wd.config_yaml)
        self._gcs.write_text(uri, config_yaml)
        return uri

    def write_manifest_lock(self, lock: Mapping[str, Any]) -> str:
        uri = self._uri(self._wd.manifest_lock)
        self._gcs.write_json(uri, dict(lock), indent=2)
        return uri

    # ------------------------------------------------------------- checkpoints
    def save_checkpoint(self, state_dict: Any, which: str = "last") -> str:
        """Serializa `state_dict` (torch.save) a un buffer y lo sube a `checkpoints/<which>.pth`."""
        import torch  # noqa: PLC0415 - perezoso (el sink de BQ no necesita torch)

        buf = io.BytesIO()
        torch.save(state_dict, buf)
        uri = self._uri(self._wd.checkpoint(which))
        self._gcs.write_bytes(uri, buf.getvalue())
        return uri

    def save_epoch(self, state_dict: Any, f1: float) -> tuple[Optional[str], str]:
        """Guarda `last.pth` SIEMPRE; `best.pth` solo si `f1` mejora. Devuelve `(best_uri|None, last_uri)`."""
        last_uri = self.save_checkpoint(state_dict, "last")
        best_uri: Optional[str] = None
        if self._best_f1 is None or f1 > self._best_f1:
            self._best_f1 = f1
            best_uri = self.save_checkpoint(state_dict, "best")
        return best_uri, last_uri

    @property
    def best_f1(self) -> Optional[float]:
        return self._best_f1


# ===================================================================== T5c: tracker
#
# Vertex AI Experiments (un experimento por `study`, un run por entrenamiento) + TensorBoard.
# Es telemetría SECUNDARIA: la fuente de verdad de las curvas es `tbl_train_metrics` (BQ, T5a). Por
# eso el tracker NUNCA debe romper el entrenamiento -> si aiplatform no está o falla, cae a no-op.


def _as_param(value: Any) -> Any:
    """Vertex `log_params` admite escalares; lo demás (listas/dicts) se pasa como str."""
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def _as_float(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {str(name): float(value) for name, value in metrics.items()}


class NoopTracker:
    """Tracker que no hace nada (local, tests, o si Vertex no está disponible)."""

    def log_params(self, params: Mapping[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Mapping[str, Any], *, step: Optional[int] = None) -> None:
        pass

    def log_summary(self, metrics: Mapping[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


class VertexTracker:
    """Adaptador fino sobre Vertex AI Experiments (+ TensorBoard).

    `backend` es el módulo `aiplatform` (INYECTABLE → testeable sin GCP). Hace `init` + `start_run`
    al construirse (`resume=True` para que un REINTENTO del job continúe el mismo run). Las métricas
    con `step` van como SERIE TEMPORAL (las pinta TensorBoard); las de resumen, como escalares.
    """

    def __init__(self, *, project: str, location: str, experiment: str, run_name: str,
                 tensorboard: Optional[str] = None, backend: Any = None) -> None:
        self._backend = backend if backend is not None else _import_aiplatform()
        self._backend.init(project=project, location=location, experiment=experiment,
                           experiment_tensorboard=tensorboard)
        self._backend.start_run(run_name, resume=True)

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._backend.log_params({str(k): _as_param(v) for k, v in params.items()})

    def log_metrics(self, metrics: Mapping[str, Any], *, step: Optional[int] = None) -> None:
        m = _as_float(metrics)
        if not m:
            return
        if step is None:
            self._backend.log_metrics(m)
        else:
            self._backend.log_time_series_metrics(m, step=int(step))

    def log_summary(self, metrics: Mapping[str, Any]) -> None:
        m = _as_float(metrics)
        if m:
            self._backend.log_metrics(m)

    def close(self) -> None:
        self._backend.end_run()


def _import_aiplatform() -> Any:
    from google.cloud import aiplatform  # noqa: PLC0415 - perezoso (no romper imports sin GCP)

    return aiplatform


def make_tracker(
    *,
    project: Optional[str] = None,
    location: Optional[str] = None,
    experiment: Optional[str] = None,
    run_name: Optional[str] = None,
    tensorboard: Optional[str] = None,
    backend: Any = None,
    enabled: bool = True,
    log: Any = None,
) -> Any:
    """`VertexTracker` si está habilitado y se puede crear; si no, `NoopTracker` (no rompe el run)."""
    if not enabled or not (project and experiment and run_name):
        return NoopTracker()
    try:
        return VertexTracker(project=project, location=location, experiment=experiment,
                             run_name=run_name, tensorboard=tensorboard, backend=backend)
    except Exception as e:  # noqa: BLE001 - el tracker es secundario; nunca tumba el entrenamiento
        if log:
            log.warning("tracker deshabilitado (Vertex no disponible)", extra={"error": str(e)})
        return NoopTracker()
