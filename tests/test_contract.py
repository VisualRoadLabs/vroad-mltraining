"""Tests del lector ligero del contrato del modelo (vroad_mlt.contract).

Usa los artefactos REALES exportados por lanetr@dd2f8ab (fixtures descargadas).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vroad_mlt import contract as ct
from vroad_mlt.contract import (
    ARCH_HASH_KEYS,
    ConfigSchema,
    ContractError,
    MetricsSpec,
    ModelContract,
    ModelInfo,
    ParamSpec,
)

FIX = Path(__file__).parent / "fixtures" / "_model" / "lanetr@dd2f8ab"


@pytest.fixture
def schema() -> ConfigSchema:
    return ConfigSchema.load(FIX / "config_schema.json")


# --------------------------------------------------------------- ConfigSchema

def test_schema_loads_real_artifact(schema):
    assert len(schema) == 32
    assert schema.groups() == ["arch", "optim", "schedule", "loss", "data"]
    assert "optim.lr" in schema
    assert "no.existe" not in schema


def test_schema_get_and_paths(schema):
    p = schema.get("arch.num_queries")
    assert p.type == "choice" and p.choices == (4, 12, 20) and p.default == 12
    assert "data.dataset_manifest" in schema.paths()


def test_arch_params_and_hash_keys(schema):
    arch = {p.path for p in schema.arch_params()}
    assert arch == {
        "arch.num_queries",
        "arch.num_decoder_layers",
        "arch.n_ref_points",
        "arch.ref_refine",
        "arch.load_strict",
    }
    # las claves del arch_hash son un subconjunto del grupo arch y existen en el schema
    for k in ARCH_HASH_KEYS:
        assert k in schema
    assert set(ARCH_HASH_KEYS) <= arch
    assert "arch.ref_refine" not in ARCH_HASH_KEYS  # no entra en el hash


def test_defaults_are_flat(schema):
    d = schema.defaults()
    assert d["optim.lr"] == 0.0003
    assert d["arch.num_queries"] == 12
    assert d["schedule.scheduler"] == "cosine"
    assert len(d) == 32


# -------------------------------------------------------------- validación

def test_validate_choice(schema):
    assert schema.validate_override("arch.num_queries", 12) == 12
    assert schema.validate_override("arch.num_queries", "20") == 20  # coerción str->int
    assert schema.validate_override("arch.ref_refine", "mlp") == "mlp"
    with pytest.raises(ContractError):
        schema.validate_override("arch.num_queries", 8)
    with pytest.raises(ContractError):
        schema.validate_override("arch.ref_refine", "xxl")


def test_validate_int_range(schema):
    assert schema.validate_override("arch.num_decoder_layers", 6) == 6
    assert schema.validate_override("arch.num_decoder_layers", "3") == 3
    with pytest.raises(ContractError):
        schema.validate_override("arch.num_decoder_layers", 0)  # < min 1
    with pytest.raises(ContractError):
        schema.validate_override("arch.num_decoder_layers", 7)  # > max 6
    with pytest.raises(ContractError):
        schema.validate_override("arch.num_decoder_layers", True)  # bool no es int


def test_validate_float_range(schema):
    assert schema.validate_override("optim.lr", 0.0003) == 0.0003
    assert schema.validate_override("optim.lr", "5e-4") == 0.0005
    with pytest.raises(ContractError):
        schema.validate_override("optim.lr", 5e-5)  # < min 1e-4
    with pytest.raises(ContractError):
        schema.validate_override("optim.lr", 2.0)  # > max 1e-3


def test_validate_bool(schema):
    assert schema.validate_override("loss.aux_loss", True) is True
    assert schema.validate_override("loss.aux_loss", "false") is False
    assert schema.validate_override("loss.aux_loss", "on") is True
    with pytest.raises(ContractError):
        schema.validate_override("loss.aux_loss", 1)  # int no es bool


def test_validate_str(schema):
    assert schema.validate_override("data.dataset_manifest", "culane@1") == "culane@1"
    with pytest.raises(ContractError):
        schema.validate_override("data.dataset_manifest", 5)


def test_unknown_override_raises(schema):
    with pytest.raises(ContractError, match="desconocido"):
        schema.validate_override("arch.no_existe", 1)


def test_validate_overrides_batches_errors(schema):
    with pytest.raises(ContractError) as exc:
        schema.validate_overrides({"optim.lr": 9.0, "arch.num_queries": 7})
    msg = str(exc.value)
    assert "optim.lr" in msg and "arch.num_queries" in msg


def test_validate_overrides_returns_coerced(schema):
    out = schema.validate_overrides({"optim.lr": "1e-4", "arch.num_decoder_layers": "4"})
    assert out == {"optim.lr": 0.0001, "arch.num_decoder_layers": 4}


def test_duplicate_path_rejected():
    p = {"path": "a.b", "type": "int", "default": 1, "group": "g"}
    with pytest.raises(ContractError, match="duplicado"):
        ConfigSchema.from_list([p, p])


def test_paramspec_choice_without_choices_rejected():
    with pytest.raises(ContractError, match="choices"):
        ParamSpec.from_dict({"path": "a.b", "type": "choice", "default": 1, "group": "g"})


# ---------------------------------------------------------------- MetricsSpec

def test_metrics_spec_loads_and_orders():
    ms = MetricsSpec.load(FIX / "metrics_spec.json")
    assert len(ms) == 10
    ordered = [m for m, _ in ms.ordered()]
    assert ordered[0] == "f1/global"
    assert ordered[-1] == "gpu/util"
    assert ms.get("f1/global")["higher_is_better"] is True
    assert ms.get("lr")["higher_is_better"] is None


# ------------------------------------------------------------------ ModelInfo

def test_model_info_loads():
    mi = ModelInfo.load(FIX / "model_info.json")
    assert mi.name == "lanetr"
    assert mi.version == "1.0.0"
    assert mi.input == (3, 320, 800)
    assert mi.img_size == (800, 320)
    assert mi.max_lanes == 4
    assert mi.params_m == 25.1
    assert mi.common_format == ".lines.json"


def test_model_info_requires_name():
    with pytest.raises(ContractError):
        ModelInfo.from_dict({"version": "1.0.0"})


# --------------------------------------------------------------- ModelContract

def test_model_contract_load_dir():
    c = ModelContract.load_dir(FIX)
    assert isinstance(c.schema, ConfigSchema) and len(c.schema) == 32
    assert isinstance(c.metrics, MetricsSpec) and len(c.metrics) == 10
    assert isinstance(c.info, ModelInfo) and c.info.name == "lanetr"
