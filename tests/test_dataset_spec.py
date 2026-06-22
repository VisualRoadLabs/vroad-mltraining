"""Tests de la spec de dataset-build (vroad_mlt.dataset_spec). Lógica pura."""

from __future__ import annotations

import pytest

from vroad_mlt.dataset_spec import DEFAULT_SHARD_MAXCOUNT, Spec, SourceSpec, SpecError

SPEC = {
    "manifest_id": "culane-mix-curves",
    "version": "3",
    "shard_maxcount": 5000,
    "sources": [
        {"source": "public", "dataset": "culane", "version": "1",
         "splits": ["train", "val", "test"], "filters": {}, "dedup": True},
        {"source": "user", "dataset": "user", "version": "2",
         "splits": ["train"], "filters": {"road_geometry": "curve"}},
    ],
}


def test_spec_from_dict():
    s = Spec.from_dict(SPEC)
    assert s.manifest_id == "culane-mix-curves" and s.version == "3"
    assert s.shard_maxcount == 5000
    assert len(s.sources) == 2
    assert s.sources[0].dedup is True
    assert s.sources[1].filters == {"road_geometry": "curve"}


def test_spec_shard_maxcount_default():
    s = Spec.from_dict({k: v for k, v in SPEC.items() if k != "shard_maxcount"})
    assert s.shard_maxcount == DEFAULT_SHARD_MAXCOUNT


def test_source_spec_defaults():
    ss = SourceSpec.from_dict({"source": "public", "dataset": "culane", "version": "1", "splits": ["train"]})
    assert ss.filters == {} and ss.dedup is False


def test_to_manifest_source_identity_split_map():
    ss = SourceSpec.from_dict(
        {"source": "public", "dataset": "culane", "version": "1", "splits": ["train", "val"]}
    )
    src = ss.to_manifest_source()
    assert src.split_map == {"train": "train", "val": "val"}
    assert src.dataset == "culane" and src.version == "1"


def test_to_manifest_sources():
    s = Spec.from_dict(SPEC)
    srcs = s.to_manifest_sources()
    assert [x.dataset for x in srcs] == ["culane", "user"]


def test_from_json_roundtrip():
    import json

    s = Spec.from_json(json.dumps(SPEC))
    assert s.manifest_id == "culane-mix-curves"


# ----------------------------------------------------------------- inválidos

def test_invalid_source():
    with pytest.raises(SpecError, match="source inválido"):
        SourceSpec.from_dict({"source": "edge", "dataset": "x", "version": "1", "splits": ["train"]})


def test_invalid_split():
    with pytest.raises(Exception):
        SourceSpec.from_dict({"source": "public", "dataset": "x", "version": "1", "splits": ["holdout"]})


def test_empty_splits():
    with pytest.raises(SpecError, match="splits"):
        SourceSpec.from_dict({"source": "public", "dataset": "x", "version": "1", "splits": []})


def test_missing_sources():
    with pytest.raises(SpecError):
        Spec.from_dict({"manifest_id": "m", "version": "1", "sources": []})
