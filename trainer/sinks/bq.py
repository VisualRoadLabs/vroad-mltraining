"""trainer.sinks.bq — métricas a BigQuery: la serie y la fila final.

- `tbl_train_metrics`: SERIE TEMPORAL en FORMATO LARGO (1 fila por `(run_id, metric, step)`). El loop
  loguea lo que quiera (`loss/*`, `lr`, `grad_norm`, `f1/*`, `gpu/*`); el dashboard plotea toda clave
  agrupada por prefijo. MERGE upsert → un reintento del Vertex job no duplica puntos.
- `tbl_experiments`: la fila final del run, por MERGE PARCIAL sobre `run_id` (el workflow ya creó la
  fila con el estado; el trainer solo añade sus columnas: `f1_global`, umbral, 9 categorías, etc.).

`bq` se INYECTA (la lógica se prueba con un FakeBq; sin GCP en los tests).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional

__all__ = [
    "DS_EXPERIMENTS",
    "TBL_TRAIN_METRICS",
    "TBL_EXPERIMENTS",
    "TRAIN_METRICS_KEYS",
    "TRAIN_METRICS_TYPES",
    "metric_rows",
    "BqMetricsSink",
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
