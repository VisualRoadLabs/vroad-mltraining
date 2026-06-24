"""Tests de los sinks del trainer (trainer.sinks).

T5a: las tablas de BigQuery. Usa un FakeBq que captura las llamadas a merge_upsert (sin GCP):
verifica el aplanado a formato largo, la clave de upsert, los tipos y el MERGE parcial final.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone

import torch

from trainer import sinks
from vroad_mlt.naming import Workdir

DT = datetime(2026, 6, 23, 10, 0, 0, tzinfo=timezone.utc)


class FakeBq:
    def __init__(self):
        self.calls = []

    def merge_upsert(self, dataset, table, rows, key_columns, *, types=None):
        self.calls.append({"dataset": dataset, "table": table,
                           "rows": [dict(r) for r in rows], "keys": tuple(key_columns), "types": types})
        return len(rows)


class FakeGcs:
    def __init__(self):
        self.text: dict[str, str] = {}
        self.blobs: dict[str, bytes] = {}
        self.jsons: dict = {}

    def write_text(self, uri, text, **kw):
        self.text[uri] = text

    def write_json(self, uri, obj, **kw):
        self.jsons[uri] = obj

    def write_bytes(self, uri, data, **kw):
        self.blobs[uri] = data


# ----------------------------------------------------------------- metric_rows

def test_metric_rows_flattens_long_format():
    rows = sinks.metric_rows("run1", {"loss/total": 0.5, "lr": 3e-4}, step=10, epoch=2, logged_at=DT)
    assert rows == [
        {"run_id": "run1", "epoch": 2, "step": 10, "metric": "loss/total", "value": 0.5, "logged_at": DT},
        {"run_id": "run1", "epoch": 2, "step": 10, "metric": "lr", "value": 0.0003, "logged_at": DT},
    ]


def test_metric_rows_epoch_optional_and_value_is_float():
    (row,) = sinks.metric_rows("r", {"gpu/mem_mb": 8192}, step=5, logged_at=DT)
    assert row["epoch"] is None                    # epoch opcional (informativo)
    assert isinstance(row["value"], float) and row["value"] == 8192.0
    assert row["step"] == 5


def test_metric_rows_defaults_logged_at_to_now():
    (row,) = sinks.metric_rows("r", {"lr": 1.0}, step=1)
    assert isinstance(row["logged_at"], datetime)  # se rellena con utcnow si no se pasa


# ------------------------------------------------------------- log_metrics

def test_log_metrics_upserts_train_metrics():
    bq = FakeBq()
    n = sinks.BqMetricsSink(bq).log_metrics("run1", {"loss/total": 0.5, "f1/global": 0.7},
                                            step=100, epoch=3, logged_at=DT)
    assert n == 2
    (call,) = bq.calls
    assert call["dataset"] == "ds_experiments"
    assert call["table"] == "tbl_train_metrics"
    assert call["keys"] == ("run_id", "metric", "step")
    assert call["types"] == sinks.TRAIN_METRICS_TYPES
    assert {r["metric"] for r in call["rows"]} == {"loss/total", "f1/global"}
    assert all(r["step"] == 100 and r["epoch"] == 3 for r in call["rows"])


def test_log_metrics_empty_is_noop():
    bq = FakeBq()
    assert sinks.BqMetricsSink(bq).log_metrics("run1", {}, step=1) == 0
    assert bq.calls == []                          # sin métricas -> no toca BigQuery


# ------------------------------------------------------- finalize_experiment

def test_finalize_experiment_partial_merge_on_run_id():
    bq = FakeBq()
    n = sinks.BqMetricsSink(bq).finalize_experiment("run1", {
        "f1_global": 0.76, "threshold": 0.4,
        "metrics_by_category_json": {"normal": 0.91, "curve": 0.64},  # dict -> JSON (lo serializa bq)
        "params_m": 25.1,
    })
    assert n == 1
    (call,) = bq.calls
    assert call["table"] == "tbl_experiments"
    assert call["keys"] == ("run_id",)             # MERGE parcial por run_id
    (row,) = call["rows"]
    assert row["run_id"] == "run1"
    assert row["f1_global"] == 0.76
    assert row["metrics_by_category_json"] == {"normal": 0.91, "curve": 0.64}


def test_finalize_experiment_passes_types_through():
    bq = FakeBq()
    sinks.BqMetricsSink(bq).finalize_experiment("r", {"f1_official_cpp": None},
                                                types={"f1_official_cpp": "FLOAT64"})
    assert bq.calls[0]["types"] == {"f1_official_cpp": "FLOAT64"}


# ============================================================ T5b: formateadores

def test_format_train_line_decimal_no_sci_notation():
    line = sinks.format_train_line(3, 1234, {"loss/total": 0.5432, "lr": 3e-4}, eta_s=754)
    assert line == "[epoch 003 step 001234] loss/total=0.5432 lr=0.0003 eta=0:12:34"


def test_format_eval_line_global_and_categories():
    line = sinks.format_eval_line(3, {"f1_global": 0.76, "metrics_by_category": {"normal": 0.91, "curve": 0.64}})
    assert line == "[epoch 003] f1/global=0.76 normal=0.91 curve=0.64"


def test_format_gpu_line():
    line = sinks.format_gpu_line(3, mem_peak_mb=8192, util_mean=85.0, util_max=99.0)
    assert line == "[epoch 003] mem_peak_mb=8192 util_mean=85.0 util_max=99.0"


# ============================================================ T5b: WorkdirSink

WD = Workdir(study="lr-sweep", variant="lr1e-4", ts="20260623T100000Z")
BUCKET = "bkt-dev-training-workdirs-usc1"
PREFIX = f"gs://{BUCKET}/lr-sweep/lr1e-4_20260623T100000Z"


def test_workdir_sink_flushes_logs():
    gcs = FakeGcs()
    sink = sinks.WorkdirSink(gcs, BUCKET, WD)
    sink.log_train(1, 10, {"loss/total": 0.5}, flush=False)
    sink.log_train(1, 20, {"loss/total": 0.4}, flush=False)
    sink.log_eval(1, {"f1_global": 0.7, "metrics_by_category": {"normal": 0.9}})  # flush=True por defecto
    assert gcs.text[f"{PREFIX}/train.log"] == (
        "[epoch 001 step 000010] loss/total=0.5\n[epoch 001 step 000020] loss/total=0.4\n"
    )
    assert gcs.text[f"{PREFIX}/eval.log"] == "[epoch 001] f1/global=0.7 normal=0.9\n"


def test_workdir_sink_writes_json_artifacts():
    gcs = FakeGcs()
    sink = sinks.WorkdirSink(gcs, BUCKET, WD)
    sink.write_results({"f1_global": 0.76})
    sink.write_manifest_lock({"manifest_id": "culane", "version": "1"})
    sink.write_config("optim:\n  lr: 0.0003\n")
    assert gcs.jsons[f"{PREFIX}/results.json"] == {"f1_global": 0.76}
    assert gcs.jsons[f"{PREFIX}/manifest.lock.json"]["manifest_id"] == "culane"
    assert gcs.text[f"{PREFIX}/config.yaml"].startswith("optim:")


def test_workdir_sink_saves_last_always_and_best_on_improvement():
    gcs = FakeGcs()
    sink = sinks.WorkdirSink(gcs, BUCKET, WD)
    sd = {"w": torch.zeros(2)}

    best, last = sink.save_epoch(sd, f1=0.5)
    assert last == f"{PREFIX}/checkpoints/last.pth"
    assert best == f"{PREFIX}/checkpoints/best.pth"   # primera época -> best
    assert sink.best_f1 == 0.5

    best2, _ = sink.save_epoch(sd, f1=0.4)            # empeora -> no best
    assert best2 is None and sink.best_f1 == 0.5

    best3, _ = sink.save_epoch(sd, f1=0.8)            # mejora -> best
    assert best3 == f"{PREFIX}/checkpoints/best.pth" and sink.best_f1 == 0.8

    # el checkpoint subido es un state_dict cargable con torch.load
    loaded = torch.load(io.BytesIO(gcs.blobs[f"{PREFIX}/checkpoints/best.pth"]), weights_only=True)
    assert torch.equal(loaded["w"], torch.zeros(2))


# ============================================================ T5c: tracker

class FakeBackend:
    """Imita el módulo `aiplatform` (registra llamadas)."""

    def __init__(self, fail_init=False):
        self.calls = []
        self._fail_init = fail_init

    def init(self, **kw):
        if self._fail_init:
            raise RuntimeError("aiplatform no disponible")
        self.calls.append(("init", kw))

    def start_run(self, run, resume=False):
        self.calls.append(("start_run", run, resume))

    def log_params(self, params):
        self.calls.append(("log_params", params))

    def log_metrics(self, metrics):
        self.calls.append(("log_metrics", metrics))

    def log_time_series_metrics(self, metrics, step):
        self.calls.append(("log_time_series_metrics", metrics, step))

    def end_run(self):
        self.calls.append(("end_run",))


def test_noop_tracker_does_nothing():
    t = sinks.NoopTracker()
    assert t.log_params({"a": 1}) is None
    assert t.log_metrics({"loss": 0.5}, step=3) is None
    assert t.log_summary({"f1": 0.7}) is None
    assert t.close() is None


def test_vertex_tracker_inits_and_starts_run():
    be = FakeBackend()
    sinks.VertexTracker(project="p", location="us-central1", experiment="lr-sweep",
                        run_name="lr-sweep/lr1e-4", tensorboard="tb", backend=be)
    assert ("init", {"project": "p", "location": "us-central1",
                     "experiment": "lr-sweep", "experiment_tensorboard": "tb"}) in be.calls
    assert ("start_run", "lr-sweep/lr1e-4", True) in be.calls  # resume=True (continúa en reintento)


def test_vertex_tracker_metrics_params_and_summary():
    be = FakeBackend()
    t = sinks.VertexTracker(project="p", location="l", experiment="e", run_name="r", backend=be)
    t.log_params({"lr": 3e-4, "overrides": ["optim.lr=1e-4"]})  # lista -> str
    t.log_metrics({"loss/total": 0.5}, step=10)                 # con step -> serie temporal
    t.log_metrics({"f1/global": 0.7})                           # sin step -> escalares
    t.log_metrics({}, step=1)                                   # vacío -> no llama
    t.log_summary({"f1_global": 0.76})
    t.close()
    calls = be.calls
    assert ("log_params", {"lr": 3e-4, "overrides": "['optim.lr=1e-4']"}) in calls
    assert ("log_time_series_metrics", {"loss/total": 0.5}, 10) in calls
    assert ("log_metrics", {"f1/global": 0.7}) in calls
    assert ("log_metrics", {"f1_global": 0.76}) in calls
    assert ("end_run",) in calls
    assert not any(c[0] == "log_time_series_metrics" and c[1] == {} for c in calls)  # vacío no loguea


def test_make_tracker_falls_back_to_noop():
    # sin datos requeridos -> Noop
    assert isinstance(sinks.make_tracker(enabled=True), sinks.NoopTracker)
    # deshabilitado -> Noop
    assert isinstance(sinks.make_tracker(project="p", experiment="e", run_name="r", enabled=False),
                      sinks.NoopTracker)
    # init de Vertex falla -> Noop (no rompe el entrenamiento)
    be = FakeBackend(fail_init=True)
    t = sinks.make_tracker(project="p", location="l", experiment="e", run_name="r", backend=be)
    assert isinstance(t, sinks.NoopTracker)


def test_make_tracker_returns_vertex_when_ok():
    be = FakeBackend()
    t = sinks.make_tracker(project="p", location="l", experiment="e", run_name="r", backend=be)
    assert isinstance(t, sinks.VertexTracker)
    assert any(c[0] == "start_run" for c in be.calls)
