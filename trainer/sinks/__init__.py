"""trainer.sinks — dónde aterriza lo que produce el entrenamiento (BigQuery, workdir, tracker)."""

from trainer.sinks.bq import (
    DS_EXPERIMENTS,
    TBL_EXPERIMENTS,
    TBL_TRAIN_METRICS,
    TRAIN_METRICS_KEYS,
    TRAIN_METRICS_TYPES,
    BqMetricsSink,
    metric_rows,
)
from trainer.sinks.tracker import NoopTracker, VertexTracker, make_tracker
from trainer.sinks.workdir import WorkdirSink, format_eval_line, format_gpu_line, format_train_line

__all__ = [
    "DS_EXPERIMENTS", "TBL_TRAIN_METRICS", "TBL_EXPERIMENTS", "TRAIN_METRICS_KEYS",
    "TRAIN_METRICS_TYPES", "metric_rows", "BqMetricsSink",
    "format_train_line", "format_eval_line", "format_gpu_line", "WorkdirSink",
    "NoopTracker", "VertexTracker", "make_tracker",
]
