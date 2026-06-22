"""Tests de la lógica PURA de BigQuery (vroad_mlt.bq): SQL del MERGE, tipos, params.

No tocan GCP: la ejecución real se verifica a mano con `python -m vroad_mlt.bq`.
"""

from __future__ import annotations

import datetime as dt

import pytest

from vroad_mlt import bq
from vroad_mlt.bq import BqError, build_merge_sql, columns_of, param_name, plan_merge_params


# ------------------------------------------------------------------ param_name

def test_param_name():
    assert param_name(0, "run_id") == "p0_run_id"
    assert param_name(3, "f1_global") == "p3_f1_global"


# ------------------------------------------------------------------ bq_type_for

@pytest.mark.parametrize(
    "value, expected",
    [
        (True, "BOOL"),
        (5, "INT64"),
        (1.5, "FLOAT64"),
        ("x", "STRING"),
        (dt.datetime(2026, 6, 20, 10, 15), "TIMESTAMP"),
        (dt.date(2026, 6, 20), "DATE"),
        ({"a": 1}, "JSON"),
        ([1, 2], "JSON"),
    ],
)
def test_bq_type_for(value, expected):
    assert bq.bq_type_for(value) == expected


def test_bq_type_for_unsupported():
    with pytest.raises(BqError):
        bq.bq_type_for(object())


# ------------------------------------------------------------------- columns_of

def test_columns_of_keeps_first_row_order():
    rows = [{"run_id": "r", "status": "RUNNING"}, {"status": "OK", "run_id": "r2"}]
    assert columns_of(rows) == ["run_id", "status"]


def test_columns_of_rejects_inconsistent():
    with pytest.raises(BqError, match="mismas columnas"):
        columns_of([{"a": 1}, {"a": 1, "b": 2}])


def test_columns_of_empty():
    with pytest.raises(BqError):
        columns_of([])


# --------------------------------------------------------------- build_merge_sql

def test_build_merge_sql_single_row():
    sql = build_merge_sql("p.ds.t", ["run_id", "status"], ["run_id"], 1)
    assert "MERGE `p.ds.t` T USING (" in sql
    assert "SELECT @p0_run_id AS run_id, @p0_status AS status" in sql
    assert "ON T.run_id = S.run_id" in sql
    assert "WHEN MATCHED THEN UPDATE SET status = S.status" in sql
    assert "WHEN NOT MATCHED THEN INSERT (run_id, status) VALUES (S.run_id, S.status)" in sql
    assert "UNION ALL" not in sql


def test_build_merge_sql_multi_row_has_union_all():
    sql = build_merge_sql("p.ds.t", ["run_id", "metric", "value"], ["run_id", "metric"], 2)
    assert " UNION ALL " in sql
    assert "@p0_run_id" in sql and "@p1_run_id" in sql
    assert "ON T.run_id = S.run_id AND T.metric = S.metric" in sql


def test_build_merge_sql_only_keys_has_no_when_matched():
    sql = build_merge_sql("p.ds.t", ["run_id"], ["run_id"], 1)
    assert "WHEN MATCHED" not in sql
    assert "WHEN NOT MATCHED THEN INSERT (run_id) VALUES (S.run_id)" in sql


def test_build_merge_sql_key_not_in_columns():
    with pytest.raises(BqError, match="no presentes"):
        build_merge_sql("p.ds.t", ["a", "b"], ["c"], 1)


def test_build_merge_sql_requires_keys_and_rows():
    with pytest.raises(BqError):
        build_merge_sql("p.ds.t", ["a"], [], 1)
    with pytest.raises(BqError):
        build_merge_sql("p.ds.t", ["a"], ["a"], 0)


# ------------------------------------------------------------- plan_merge_params

def test_plan_merge_params_infers_types_and_names():
    rows = [{"run_id": "r1", "epoch": 3, "value": 0.5}]
    planned = plan_merge_params(rows, ["run_id", "epoch", "value"])
    assert planned == [
        ("p0_run_id", "STRING", "r1"),
        ("p0_epoch", "INT64", 3),
        ("p0_value", "FLOAT64", 0.5),
    ]


def test_plan_merge_params_serializes_json():
    rows = [{"run_id": "r", "overrides_json": {"optim": {"lr": 1e-4}}}]
    planned = plan_merge_params(rows, ["run_id", "overrides_json"])
    name, bqtype, value = planned[1]
    assert name == "p0_overrides_json" and bqtype == "JSON"
    assert value == '{"optim": {"lr": 0.0001}}'  # serializado a string


def test_plan_merge_params_type_override_for_timestamp_string():
    rows = [{"run_id": "r", "started_at": "2026-06-20T10:15:00Z"}]
    planned = plan_merge_params(rows, ["run_id", "started_at"], types={"started_at": "TIMESTAMP"})
    assert planned[1] == ("p0_started_at", "TIMESTAMP", "2026-06-20T10:15:00Z")


def test_plan_merge_params_none_requires_type():
    with pytest.raises(BqError, match="None sin tipo"):
        plan_merge_params([{"run_id": "r", "ended_at": None}], ["run_id", "ended_at"])
    # con tipo, None es válido (NULL)
    planned = plan_merge_params(
        [{"run_id": "r", "ended_at": None}], ["run_id", "ended_at"], types={"ended_at": "TIMESTAMP"}
    )
    assert planned[1] == ("p0_ended_at", "TIMESTAMP", None)


def test_plan_merge_params_multi_row_indices():
    rows = [{"run_id": "r0", "v": 1}, {"run_id": "r1", "v": 2}]
    planned = plan_merge_params(rows, ["run_id", "v"])
    names = [p[0] for p in planned]
    assert names == ["p0_run_id", "p0_v", "p1_run_id", "p1_v"]
