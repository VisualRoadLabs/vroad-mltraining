"""Tests del orquestador del trainer (trainer.run).

Helpers puros + la orquestación COMPLETA con fakes: las piezas pesadas (lanetr/torch) se inyectan
como costuras, y gcs/bq son fakes en memoria. Así se prueba el cableado (config->dataset->loop->
eval->sinks->cierre) sin lanetr, GPU ni GCP. El schema sí es el REAL (fixture exportado de lanetr).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from trainer import run
from vroad_mlt import naming

SHA = "dd2f8ab"
DATASETS = "bkt-dev-datasets-usc1"
WORKDIRS = "bkt-dev-training-workdirs-usc1"
SCHEMA_JSON = (Path(__file__).parents[1] / "fixtures" / "_model" / f"lanetr@{SHA}" / "config_schema.json").read_text(
    encoding="utf-8"
)

# Config base mínima (lo que devolvería lanetr.DEFAULT_CONFIG); el test inyecta esto.
BASE = {
    "arch": {"num_queries": 12, "num_decoder_layers": 6, "n_ref_points": 4, "load_strict": True},
    "optim": {"grad_clip": 0.1},
    "schedule": {"epochs": 2},
    "data": {"batch_size": 32},
    "train": {"amp": False, "channels_last": False, "eval_conf_thresh": 0.5},
}


# ------------------------------------------------------------------ helpers puros

def test_split_manifest_ref():
    assert run.split_manifest_ref("culane@1") == ("culane", "1")
    with pytest.raises(ValueError):
        run.split_manifest_ref("culane")


def test_step_event_prefixes_losses():
    ev = run.step_event({"step": 1, "lr": 3e-4, "grad_norm": 0.1, "losses": {"total": 0.5, "iou": 0.2}})
    assert ev == {"loss/total": 0.5, "loss/iou": 0.2, "lr": 3e-4, "grad_norm": 0.1}


def test_f1_event_global_and_categories():
    ev = run.f1_event({"f1_global": 0.7, "metrics_by_category": {"normal": 0.9, "curve": 0.6}})
    assert ev == {"f1/global": 0.7, "f1/normal": 0.9, "f1/curve": 0.6}


def test_experiment_fields_omits_none():
    spec = _spec()
    manifest = _manifest_obj()
    results = {"f1_global": 0.76, "threshold": 0.4, "metrics_by_category": {"normal": 0.9}}
    fields = run.experiment_fields(spec, manifest, results, arch_hash="abc", params_m=25.1,
                                   best_ckpt_uri="gs://b/best.pth", workdir_uri="gs://b/wd", config_uri="gs://b/c.yaml")
    assert fields["f1_global"] == 0.76 and fields["arch_hash"] == "abc"
    assert fields["dataset_manifest_id"] == "culane" and fields["num_train"] == 64
    assert "base_ckpt_uri" not in fields                # no es finetune
    assert all(v is not None for v in fields.values())  # sin None (MERGE parcial)


def test_gpu_stats_cpu_is_zero():
    assert run.gpu_stats("cpu") == {"mem_peak_mb": 0.0, "util_mean": 0.0, "util_max": 0.0}


def _manifest_dict():
    return {
        "manifest_id": "culane", "version": "1", "content_hash": "sha256:abc",
        "sources": [{"dataset": "culane", "version": "1",
                     "split_map": {"train": "train", "test": "test"}, "filters": {}}],
        "counts": {"train": 64, "val": 0, "test": 34},
    }


def _manifest_obj():
    from vroad_mlt.manifest import Manifest
    return Manifest.from_dict(_manifest_dict())


def _spec(**kw):
    base = dict(run_id="lr-sweep__lr1e-4__20260623T100000Z", manifest_ref="culane@1", run_kind="scratch",
               set_items=[], datasets_bucket=DATASETS, workdir_bucket=WORKDIRS, project="proj",
               region="us-central1", lanetr_sha=SHA)
    base.update(kw)
    return run.RunSpec(**base)


# --------------------------------------------------------------- resolve_shards

class _ListGcs:
    def __init__(self, listing):
        self.listing = list(listing)

    def list_uris(self, prefix):
        return sorted(u for u in self.listing if u.startswith(prefix))


def test_resolve_shards_sums_sources():
    gcs = _ListGcs([
        f"gs://{DATASETS}/shards/culane@1/train-00000.tar",
        f"gs://{DATASETS}/shards/culane@1/train-00001.tar",
        f"gs://{DATASETS}/shards/culane@1/val-00000.tar",   # otro split, se ignora
    ])
    uris = run.resolve_shards(gcs, DATASETS, _manifest_obj(), "train")
    assert uris == [f"gs://{DATASETS}/shards/culane@1/train-00000.tar",
                    f"gs://{DATASETS}/shards/culane@1/train-00001.tar"]


# ---------------------------------------------------------- orquestación (e2e fakes)

class FakeGcs:
    """GCS en memoria: reads desde `store`, listing desde `listing`, captura writes."""

    def __init__(self, store, listing):
        self.store = store
        self.listing = list(listing)
        self.wtext: dict[str, str] = {}
        self.wjson: dict = {}
        self.wblob: dict[str, bytes] = {}

    def read_text(self, uri):
        return self.store[uri]

    def read_json(self, uri):
        return json.loads(self.store[uri]) if isinstance(self.store[uri], str) else self.store[uri]

    def read_bytes(self, uri):
        return self.store[uri]

    def list_uris(self, prefix):
        return sorted(u for u in self.listing if u.startswith(prefix))

    def write_text(self, uri, text, **kw):
        self.wtext[uri] = text

    def write_json(self, uri, obj, **kw):
        self.wjson[uri] = obj

    def write_bytes(self, uri, data, **kw):
        self.wblob[uri] = data


class FakeBq:
    def __init__(self):
        self.calls = []

    def merge_upsert(self, dataset, table, rows, key_columns, *, types=None):
        self.calls.append({"table": table, "rows": [dict(r) for r in rows]})
        return len(rows)


class FakeTracker:
    def __init__(self):
        self.params = None
        self.metrics = []
        self.summary = None
        self.closed = False

    def log_params(self, p):
        self.params = p

    def log_metrics(self, m, *, step=None):
        self.metrics.append((m, step))

    def log_summary(self, m):
        self.summary = m

    def close(self):
        self.closed = True


def _fake_build_trainables(cfg, device, *, iters_per_epoch, epochs):
    model = nn.Linear(2, 2)
    ema = SimpleNamespace(ema=nn.Linear(2, 2))
    return model, "crit", "opt", "sched", ema, "prep"


def _fake_train_epoch(model, crit, opt, sched, loader, device, *, ema=None, prepare_targets=None,
                      amp=False, channels_last=False, grad_clip=0.1, on_step=None):
    if on_step:
        on_step({"step": 1, "lr": 3e-4, "grad_norm": 0.1, "losses": {"total": 0.5, "iou": 0.2}})
        on_step({"step": 2, "lr": 3e-4, "grad_norm": 0.1, "losses": {"total": 0.4, "iou": 0.18}})
    return {"total": 0.45}


def _make_env():
    """gcs+bq+seams cableados para una corrida de 2 épocas (F1 0.6 luego 0.5)."""
    schema_uri = naming.gs_uri(DATASETS, naming.model_artifacts_prefix("lanetr", SHA), "config_schema.json")
    manifest_uri = naming.gs_uri(DATASETS, naming.manifest_key("culane", "1"))
    cats_uri = naming.gs_uri(DATASETS, naming.benchmark_categories_key("culane@v1"))
    store = {
        schema_uri: SCHEMA_JSON,
        manifest_uri: json.dumps(_manifest_dict()),
        cats_uri: {"00000000": "normal"},
    }
    listing = [f"gs://{DATASETS}/shards/culane@1/train-00000.tar",
               f"gs://{DATASETS}/benchmarks/culane@v1/test-00000.tar"]
    gcs = FakeGcs(store, listing)
    bq = FakeBq()
    tracker = FakeTracker()

    f1s = iter([0.6, 0.5])  # 1ª época mejora -> best; 2ª empeora -> se mantiene best=0.6

    def fake_run_eval(model, loader, device, *, categories_map=None, threshold=0.5, **kw):
        f1 = next(f1s)
        return ({"benchmark": "culane@v1", "f1_global": f1, "threshold": threshold,
                 "metrics_by_category": {"normal": f1}}, [], [], [])

    seams = dict(
        default_config=lambda: BASE,
        make_loader=lambda *a, **k: [],
        make_eval_loader=lambda *a, **k: [],
        build_trainables=_fake_build_trainables,
        train_epoch=_fake_train_epoch,
        run_eval=fake_run_eval,
        make_tracker=lambda **kw: tracker,
    )
    return gcs, bq, tracker, seams


def test_run_training_end_to_end_with_fakes():
    gcs, bq, tracker, seams = _make_env()
    spec = _spec(log_every=1)  # log_every=1 -> los pasos del fake (1,2) sí loguean

    results = run.run_training(spec, gcs=gcs, bq=bq, device="cpu", **seams)

    prefix = f"gs://{WORKDIRS}/lr-sweep/lr1e-4_20260623T100000Z"

    # devuelve la MEJOR época (F1=0.6, no la última 0.5)
    assert results["f1_global"] == 0.6

    # artefactos del workdir
    assert gcs.wtext[f"{prefix}/config.yaml"].startswith(("arch", "{"))  # YAML de la config efectiva
    assert gcs.wjson[f"{prefix}/manifest.lock.json"]["manifest_id"] == "culane"
    assert gcs.wjson[f"{prefix}/results.json"]["f1_global"] == 0.6
    assert "[epoch 000 step 000001] loss/total=0.5" in gcs.wtext[f"{prefix}/train.log"]
    assert "f1/global=0.6" in gcs.wtext[f"{prefix}/eval.log"]
    assert f"{prefix}/gpu.log" in gcs.wtext
    # checkpoints best+last (state_dicts cargables)
    assert f"{prefix}/checkpoints/last.pth" in gcs.wblob
    sd = torch.load(io.BytesIO(gcs.wblob[f"{prefix}/checkpoints/best.pth"]), weights_only=True)
    assert "weight" in sd

    # BigQuery: serie (tbl_train_metrics) varias veces + fila final (tbl_experiments) 1 vez
    tables = [c["table"] for c in bq.calls]
    assert tables.count("tbl_experiments") == 1
    assert "tbl_train_metrics" in tables
    (final,) = [c for c in bq.calls if c["table"] == "tbl_experiments"]
    (frow,) = final["rows"]
    # params_m=0.0 con el modelo fake (nn.Linear); el real da ~25.1. Solo comprobamos que se calcula.
    assert frow["f1_global"] == 0.6 and frow["arch_hash"] and isinstance(frow["params_m"], float)
    assert frow["dataset_manifest_id"] == "culane" and frow["num_train"] == 64
    assert frow["model_sha"] == SHA

    # tracker: params + métricas + resumen + cierre
    assert tracker.params["manifest"] == "culane@1"
    assert tracker.summary == {"f1_global": 0.6}
    assert tracker.closed is True
    assert any(step is not None for _, step in tracker.metrics)  # serie temporal (con step)


def test_run_training_finetune_loads_parent():
    gcs, bq, tracker, seams = _make_env()
    parent_uri = f"gs://{WORKDIRS}/parent/best.pth"
    buf = io.BytesIO(); torch.save(nn.Linear(2, 2).state_dict(), buf)
    gcs.store[parent_uri] = buf.getvalue()
    spec = _spec(run_kind="finetune", parent_ckpt=parent_uri, log_every=1)

    results = run.run_training(spec, gcs=gcs, bq=bq, device="cpu", **seams)

    assert results["f1_global"] == 0.6
    (final,) = [c for c in bq.calls if c["table"] == "tbl_experiments"]
    assert final["rows"][0]["base_ckpt_uri"] == parent_uri   # linaje del padre en la fila final


def test_run_training_caps_train_steps_and_eval_batches():
    gcs, bq, tracker, seams = _make_env()
    seams["make_loader"] = lambda *a, **k: list(range(10))       # 10 batches
    seams["make_eval_loader"] = lambda *a, **k: list(range(10))
    seen = {"train": [], "eval": []}

    def counting_train_epoch(model, crit, opt, sched, loader, device, **kw):
        seen["train"].append(len(list(loader)))
        return {"total": 0.5}

    def counting_run_eval(model, loader, device, **kw):
        seen["eval"].append(len(list(loader)))
        return ({"f1_global": 0.6, "threshold": 0.5, "metrics_by_category": {}}, [], [], [])

    seams["train_epoch"] = counting_train_epoch
    seams["run_eval"] = counting_run_eval
    spec = _spec(max_train_steps=3, max_eval_batches=2)

    run.run_training(spec, gcs=gcs, bq=bq, device="cpu", **seams)
    assert seen["train"] == [3, 3]   # 2 épocas, cada train cortado a 3 batches
    assert seen["eval"] == [2, 2]    # cada eval cortada a 2 batches


def test_run_training_standalone_manages_experiment_row():
    gcs, bq, tracker, seams = _make_env()
    spec = _spec(standalone=True, log_every=1)

    run.run_training(spec, gcs=gcs, bq=bq, device="cpu", **seams)

    exp = [c for c in bq.calls if c["table"] == "tbl_experiments"]
    assert len(exp) == 2                                   # fila base (RUNNING) + finalize (SUCCEEDED)
    assert exp[0]["rows"][0]["status"] == "RUNNING" and exp[0]["rows"][0]["study"] == "lr-sweep"
    assert exp[-1]["rows"][0]["status"] == "SUCCEEDED" and "ended_at" in exp[-1]["rows"][0]
