"""Tests del manifiesto de dataset (vroad_mlt.manifest). Lógica pura."""

from __future__ import annotations

from dataclasses import replace

import pytest

from vroad_mlt.manifest import (
    DEFAULT_SHARD_PATTERN,
    Manifest,
    ManifestError,
    ManifestLock,
    Source,
)


def _sources():
    return [
        Source("culane", "1", {"train": "train", "val": "val", "test": "test"}, {}),
        Source("user-curves", "2", {"train": "train"}, {"road_geometry": "curve"}),
    ]


def _manifest(created_at="20260620T101500Z"):
    return Manifest.build(
        "culane-mix-curves", "3", _sources(),
        {"train": 61234, "val": 9675, "test": 34680},
        created_at=created_at,
    )


# ----------------------------------------------------------------- Source

def test_source_roundtrip():
    s = Source("culane", "1", {"train": "train"}, {"road_geometry": "curve"})
    assert Source.from_dict(s.to_dict()) == s


def test_source_missing_field():
    with pytest.raises(ManifestError, match="split_map"):
        Source.from_dict({"dataset": "culane", "version": "1"})


# ----------------------------------------------------------------- build/hash

def test_build_sets_content_hash():
    m = _manifest()
    assert m.content_hash is not None and m.content_hash.startswith("sha256:")
    assert m.verify_hash()
    assert m.shard_pattern == DEFAULT_SHARD_PATTERN


def test_hash_is_deterministic():
    assert _manifest().content_hash == _manifest().content_hash


def test_hash_ignores_created_at():
    a = _manifest(created_at="20260620T101500Z")
    b = _manifest(created_at="20990101T000000Z")
    assert a.content_hash == b.content_hash  # created_at NO entra en el hash


def test_hash_changes_with_counts():
    base = _manifest()
    other = Manifest.build(
        "culane-mix-curves", "3", _sources(),
        {"train": 61235, "val": 9675, "test": 34680},  # +1
        created_at="20260620T101500Z",
    )
    assert base.content_hash != other.content_hash


def test_hash_changes_with_sources():
    base = _manifest()
    srcs = _sources()
    srcs[1] = replace(srcs[1], filters={})  # cambia el filtro
    other = Manifest.build("culane-mix-curves", "3", srcs, base.counts, created_at=base.created_at)
    assert base.content_hash != other.content_hash


def test_verify_hash_detects_tampering():
    m = _manifest()
    tampered = replace(m, counts={"train": 1, "val": 1, "test": 1})  # cambia counts, hash viejo
    assert tampered.verify_hash() is False


# ----------------------------------------------------------------- serializar

def test_to_from_dict_roundtrip():
    m = _manifest()
    assert Manifest.from_dict(m.to_dict()) == m


def test_to_from_json_roundtrip_preserves_hash():
    m = _manifest()
    back = Manifest.from_json(m.to_json())
    assert back == m and back.content_hash == m.content_hash and back.verify_hash()


def test_to_dict_key_order():
    m = _manifest()
    assert list(m.to_dict().keys())[:4] == ["manifest_id", "version", "created_at", "content_hash"]


def test_from_dict_validation():
    with pytest.raises(ManifestError):
        Manifest.from_dict({"manifest_id": "x", "version": "1", "counts": {}})  # sin sources
    with pytest.raises(ManifestError):
        Manifest.from_dict({"manifest_id": "x", "version": "1", "sources": {}, "counts": {}})


def test_curve_keys_uri_optional_and_in_hash():
    m1 = Manifest.build("m", "1", _sources(), {"train": 1}, created_at="t")
    m2 = Manifest.build("m", "1", _sources(), {"train": 1}, created_at="t",
                        curve_keys_uri="gs://b/x.curve_keys.json")
    assert "curve_keys_uri" not in m1.to_dict()
    assert m1.content_hash != m2.content_hash  # el URI sí cuenta en el contenido


# ----------------------------------------------------------------- integración

def test_to_experiment_fields():
    m = _manifest()
    f = m.to_experiment_fields()
    assert f["dataset_manifest_id"] == "culane-mix-curves"
    assert f["dataset_manifest_version"] == "3"
    assert f["dataset_manifest_hash"] == m.content_hash
    assert f["num_train"] == 61234 and f["num_val"] == 9675 and f["num_test"] == 34680
    assert isinstance(f["dataset_sources_json"], list) and len(f["dataset_sources_json"]) == 2


# ----------------------------------------------------------------- lock

def test_lock_from_manifest_and_verify():
    m = _manifest()
    lock = ManifestLock.from_manifest(m, resolved_at="20260620T101800Z")
    assert lock.manifest_id == m.manifest_id and lock.content_hash == m.content_hash
    assert lock.verify(m) is True


def test_lock_verify_rejects_different_content():
    m = _manifest()
    lock = ManifestLock.from_manifest(m)
    other = Manifest.build("culane-mix-curves", "3", _sources(),
                           {"train": 1, "val": 1, "test": 1}, created_at=m.created_at)
    assert lock.verify(other) is False  # mismo id@version, distinto hash


def test_lock_json_roundtrip():
    m = _manifest()
    lock = ManifestLock.from_manifest(m, resolved_at="20260620T101800Z")
    assert ManifestLock.from_json(lock.to_json()) == lock
