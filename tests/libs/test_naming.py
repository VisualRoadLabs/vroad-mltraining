"""Tests de la convención de nombres y rutas (vroad_mlt.naming)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from vroad_mlt import naming as n
from vroad_mlt.naming import NamingError, Workdir

TS = "20260620T101500Z"


# --------------------------------------------------------------------- tiempo

def test_format_ts_from_aware_utc():
    dt = datetime(2026, 6, 20, 10, 15, 0, tzinfo=timezone.utc)
    assert n.format_ts(dt) == TS


def test_format_ts_naive_is_assumed_utc():
    assert n.format_ts(datetime(2026, 6, 20, 10, 15, 0)) == TS


def test_format_ts_converts_other_tz_to_utc():
    from datetime import timedelta

    plus2 = timezone(timedelta(hours=2))
    dt = datetime(2026, 6, 20, 12, 15, 0, tzinfo=plus2)  # 10:15 UTC
    assert n.format_ts(dt) == TS


def test_parse_ts_roundtrip():
    assert n.format_ts(n.parse_ts(TS)) == TS
    assert n.parse_ts(TS).tzinfo == timezone.utc


def test_parse_ts_rejects_bad():
    with pytest.raises(NamingError):
        n.parse_ts("2026-06-20 10:15")


def test_utc_timestamp_shape():
    ts = n.utc_timestamp()
    assert n.parse_ts(ts)  # parseable -> formato correcto


# ----------------------------------------------------------------- validación

@pytest.mark.parametrize("bad", ["", "a/b", "a b", "_x", "x__y", "/x"])
def test_validate_segment_rejects(bad):
    with pytest.raises(NamingError):
        n.validate_segment(bad)


@pytest.mark.parametrize("ok", ["lr-sweep", "lr1e-4", "culane", "user-curves", "v1", "a.b_c"])
def test_validate_segment_accepts(ok):
    assert n.validate_segment(ok) == ok


def test_validate_split():
    assert n.validate_split("val") == "val"
    with pytest.raises(NamingError):
        n.validate_split("holdout")


# ------------------------------------------------------------- identidad run

def test_run_id_and_parse_roundtrip():
    rid = n.run_id("lr-sweep", "lr1e-4", TS)
    assert rid == "lr-sweep__lr1e-4__20260620T101500Z"
    assert n.parse_run_id(rid) == ("lr-sweep", "lr1e-4", TS)


def test_run_id_rejects_bad_ts():
    with pytest.raises(NamingError):
        n.run_id("s", "v", "ayer")


def test_parse_run_id_rejects_malformed():
    with pytest.raises(NamingError):
        n.parse_run_id("solo-dos__partes")


def test_experiment_name():
    assert n.experiment_name("lr-sweep", "lr1e-4") == "lr-sweep/lr1e-4"


# ------------------------------------------------------------------- Workdir

def test_workdir_identity():
    wd = Workdir("lr-sweep", "lr1e-4", TS)
    assert wd.key == "lr-sweep/lr1e-4_20260620T101500Z"
    assert wd.run_id == "lr-sweep__lr1e-4__20260620T101500Z"
    assert wd.experiment_name == "lr-sweep/lr1e-4"


def test_workdir_named_files():
    wd = Workdir("s", "v", TS)
    base = "s/v_20260620T101500Z"
    assert wd.config_yaml == f"{base}/config.yaml"
    assert wd.manifest_lock == f"{base}/manifest.lock.json"
    assert wd.train_log == f"{base}/train.log"
    assert wd.eval_log == f"{base}/eval.log"
    assert wd.gpu_log == f"{base}/gpu.log"
    assert wd.results_json == f"{base}/results.json"


def test_workdir_checkpoints_viz_predictions():
    wd = Workdir("s", "v", TS)
    base = "s/v_20260620T101500Z"
    assert wd.checkpoint("best") == f"{base}/checkpoints/best.pth"
    assert wd.checkpoint("last") == f"{base}/checkpoints/last.pth"
    assert wd.epoch_checkpoint(5) == f"{base}/checkpoints/epoch_005.pth"
    assert wd.viz("gt_vs_pred", 5, 0) == f"{base}/viz/epoch_005/gt_vs_pred/img0.png"
    assert wd.predictions("val") == f"{base}/predictions/val.jsonl"


def test_workdir_checkpoint_rejects_bad_name():
    with pytest.raises(NamingError):
        Workdir("s", "v", TS).checkpoint("intermediate")


def test_workdir_validates_on_construction():
    with pytest.raises(NamingError):
        Workdir("bad/study", "v", TS)
    with pytest.raises(NamingError):
        Workdir("s", "v", "no-es-ts")


def test_workdir_uri_builders():
    wd = Workdir("s", "v", TS)
    bkt = "bkt-dev-training-workdirs-usc1"
    assert wd.uri(bkt) == "gs://bkt-dev-training-workdirs-usc1/s/v_20260620T101500Z"
    assert wd.file_uri(bkt, "config.yaml") == (
        "gs://bkt-dev-training-workdirs-usc1/s/v_20260620T101500Z/config.yaml"
    )


# ------------------------------------------------------------------- gs URIs

def test_gs_uri_joins_and_strips():
    assert n.gs_uri("bkt", "a", "b") == "gs://bkt/a/b"
    assert n.gs_uri("bkt", "/a/", "/b/") == "gs://bkt/a/b"
    assert n.gs_uri("gs://bkt", "a") == "gs://bkt/a"  # acepta bucket con gs://
    assert n.gs_uri("bkt") == "gs://bkt"  # sin partes


def test_split_gs_uri_roundtrip():
    assert n.split_gs_uri("gs://bkt/a/b") == ("bkt", "a/b")
    assert n.split_gs_uri("gs://bkt") == ("bkt", "")
    with pytest.raises(NamingError):
        n.split_gs_uri("http://bkt/a")


# -------------------------------------------------------------- bucket datasets

def test_shard_keys():
    assert n.shards_prefix("culane", "1") == "shards/culane@1"
    assert n.shard_name("train", 0) == "train-00000.tar"
    assert n.shard_name("val", 42) == "val-00042.tar"
    assert n.shard_key("culane", "1", "test", 7) == "shards/culane@1/test-00007.tar"


def test_manifest_and_curve_keys():
    assert n.manifest_key("culane-mix-curves", "3") == "manifests/culane-mix-curves@3.json"
    assert n.curve_keys_key("culane-mix-curves", "3") == (
        "manifests/culane-mix-curves@3.curve_keys.json"
    )


def test_benchmark_keys():
    assert n.benchmark_prefix() == "benchmarks/culane@v1"
    assert n.benchmark_shard_key("val", 0) == "benchmarks/culane@v1/val-00000.tar"
    assert n.benchmark_categories_key() == "benchmarks/culane@v1/categories.json"


def test_model_artifact_keys():
    assert n.model_artifacts_prefix("lanetr", "dd2f8ab") == "_model/lanetr@dd2f8ab"
    assert n.config_schema_key("lanetr", "dd2f8ab") == "_model/lanetr@dd2f8ab/config_schema.json"
    assert n.metrics_spec_key("lanetr", "dd2f8ab") == "_model/lanetr@dd2f8ab/metrics_spec.json"
    assert n.model_info_key("lanetr", "dd2f8ab") == "_model/lanetr@dd2f8ab/model_info.json"


# -------------------------------------------------------------- bucket staging

def test_staging_and_locks():
    rid = "lr-sweep__lr1e-4__20260620T101500Z"
    assert n.vertex_job_dir(rid) == f"jobs/{rid}"
    assert n.tensorboard_dir(rid) == f"tensorboard/{rid}"
    assert n.a100_lock_key() == "locks/a100.lock"
    assert n.l4_lock_key(2) == "locks/l4/slot-2.lock"
    assert n.l4_lock_keys(3) == [
        "locks/l4/slot-0.lock",
        "locks/l4/slot-1.lock",
        "locks/l4/slot-2.lock",
    ]


# -------------------------------------------------------------- bucket handoff

def test_handoff_keys():
    assert n.candidate_prefix("lr-sweep", "lr1e-4") == "candidates/lr-sweep/lr1e-4"
    assert n.candidate_best_key("lr-sweep", "lr1e-4") == "candidates/lr-sweep/lr1e-4/best.pth"
    assert n.candidate_model_card_key("lr-sweep", "lr1e-4") == (
        "candidates/lr-sweep/lr1e-4/model_card.json"
    )
