"""Tests de la resolución de overrides + arch_hash (vroad_mlt.config_resolve)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vroad_mlt import config_resolve as cr
from vroad_mlt.config_resolve import ConfigResolveError
from vroad_mlt.contract import ConfigSchema, ContractError

FIX = Path(__file__).parents[1] / "fixtures" / "_model" / "lanetr@dd2f8ab"


@pytest.fixture
def schema() -> ConfigSchema:
    return ConfigSchema.load(FIX / "config_schema.json")


# ------------------------------------------------------------------ parse_set

def test_parse_set_basic():
    assert cr.parse_set(["optim.lr=1e-4", "arch.num_queries=20"]) == {
        "optim.lr": "1e-4",
        "arch.num_queries": "20",
    }


def test_parse_set_value_with_equals():
    assert cr.parse_set(["data.dataset_manifest=culane@1=x"]) == {
        "data.dataset_manifest": "culane@1=x"
    }


def test_parse_set_last_wins():
    assert cr.parse_set(["optim.lr=1e-4", "optim.lr=2e-4"]) == {"optim.lr": "2e-4"}


def test_parse_set_missing_equals_raises():
    with pytest.raises(ConfigResolveError, match="falta '='"):
        cr.parse_set(["optim.lr"])


# ------------------------------------------------------------------ to_nested

def test_to_nested_simple():
    assert cr.to_nested({"optim.lr": 0.0001, "arch.num_queries": 20}) == {
        "optim": {"lr": 0.0001},
        "arch": {"num_queries": 20},
    }


def test_to_nested_three_levels():
    assert cr.to_nested({"data.aug.hflip_prob": 0.5}) == {
        "data": {"aug": {"hflip_prob": 0.5}}
    }


def test_to_nested_conflict_leaf_then_branch():
    with pytest.raises(ConfigResolveError, match="conflicto"):
        cr.to_nested({"a": 1, "a.b": 2})


def test_to_nested_conflict_branch_then_leaf():
    with pytest.raises(ConfigResolveError, match="conflicto"):
        cr.to_nested({"a.b": 2, "a": 1})


def test_flatten_roundtrip():
    flat = {"optim.lr": 0.0001, "data.aug.hflip_prob": 0.5, "arch.num_queries": 20}
    assert cr.flatten(cr.to_nested(flat)) == flat


# ------------------------------------------------------------------ deep_merge

def test_deep_merge_does_not_mutate_base():
    base = {"arch": {"num_queries": 12, "d_model": 256}, "optim": {"lr": 3e-4}}
    merged = cr.deep_merge(base, {"arch": {"num_queries": 20}})
    assert merged["arch"] == {"num_queries": 20, "d_model": 256}  # fusiona, no reemplaza el bloque
    assert merged["optim"] == {"lr": 3e-4}
    assert base["arch"]["num_queries"] == 12  # base intacta


# ------------------------------------------------------------- resolve_overrides

def test_resolve_overrides_types_and_nests(schema):
    r = cr.resolve_overrides(schema, ["optim.lr=1e-4", "arch.num_queries=20"])
    assert r.flat == {"optim.lr": 0.0001, "arch.num_queries": 20}  # coercionado
    assert r.nested == {"optim": {"lr": 0.0001}, "arch": {"num_queries": 20}}


def test_resolve_overrides_empty(schema):
    r = cr.resolve_overrides(schema, [])
    assert r.flat == {} and r.nested == {}


def test_resolve_overrides_invalid_value_raises(schema):
    with pytest.raises(ContractError):
        cr.resolve_overrides(schema, ["optim.lr=9"])  # > max


def test_resolve_overrides_unknown_path_raises(schema):
    with pytest.raises(ContractError, match="desconocido"):
        cr.resolve_overrides(schema, ["optim.no_existe=1"])


def test_resolve_overrides_batches_errors(schema):
    with pytest.raises(ContractError) as exc:
        cr.resolve_overrides(schema, ["optim.lr=9", "arch.num_queries=7"])
    msg = str(exc.value)
    assert "optim.lr" in msg and "arch.num_queries" in msg


# --------------------------------------------------------------------- arch_hash

def test_arch_hash_is_stable_and_keyorder_independent():
    a = cr.arch_hash({"x": 1, "y": 2})
    b = cr.arch_hash({"y": 2, "x": 1})
    assert a == b
    assert len(a) == cr.ARCH_HASH_LEN
    assert a != cr.arch_hash({"x": 1, "y": 3})


def test_effective_arch_values_uses_defaults(schema):
    vals = cr.effective_arch_values(schema, {})
    assert vals == {
        "arch.num_queries": 12,
        "arch.num_decoder_layers": 6,
        "arch.n_ref_points": 4,
    }


def test_effective_arch_values_applies_override(schema):
    vals = cr.effective_arch_values(schema, {"arch.num_queries": 20})
    assert vals["arch.num_queries"] == 20
    assert vals["arch.num_decoder_layers"] == 6  # sigue el default


def test_arch_hash_changes_when_arch_key_changes(schema):
    base = cr.arch_hash_for(schema, {})
    changed = cr.arch_hash_for(schema, {"arch.num_queries": 20})
    assert base != changed


def test_arch_hash_unaffected_by_non_arch_hash_params(schema):
    base = cr.arch_hash_for(schema, {})
    # ref_refine y load_strict son grupo 'arch' pero NO entran en el hash;
    # optim.lr tampoco. Ninguno debe cambiar el arch_hash.
    r1 = cr.resolve_overrides(schema, ["arch.ref_refine=xs"])
    r2 = cr.resolve_overrides(schema, ["optim.lr=1e-4"])
    assert cr.arch_hash_for(schema, r1.flat) == base
    assert cr.arch_hash_for(schema, r2.flat) == base


def test_arch_hash_for_decoder_layers(schema):
    base = cr.arch_hash_for(schema, {})
    r = cr.resolve_overrides(schema, ["arch.num_decoder_layers=3"])
    assert cr.arch_hash_for(schema, r.flat) != base
