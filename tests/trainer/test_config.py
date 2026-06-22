"""Tests de la config efectiva del trainer (trainer.config). Sin torch ni lanetr.

Usa el `DEFAULT_CONFIG` REAL de lanetr@dd2f8ab (inline) + el CONFIG_SCHEMA real (fixture).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trainer import config as tc
from vroad_mlt import config_resolve
from vroad_mlt.contract import ConfigSchema, ContractError

# Las fixtures del modelo viven en tests/fixtures.
FIX = Path(__file__).parents[1] / "fixtures" / "_model" / "lanetr@dd2f8ab"

# Copia fiel de lanetr.contract.spec.DEFAULT_CONFIG (base congelada).
DEFAULT_CONFIG = {
    "name": "lanetr_dla34_culane",
    "arch": {
        "num_queries": 12, "num_decoder_layers": 6, "n_ref_points": 4,
        "ref_refine": "mlp", "load_strict": True,
        "backbone": "dla34", "pretrained": True, "d_model": 256, "nhead": 8, "dim_ff": 1024,
        "n_points": 4, "num_rows": 144, "img_w": 800, "img_h": 320,
        "ref_y_top": 0.15, "ref_y_bottom": 0.95,
    },
    "optim": {"lr": 3.0e-4, "backbone_lr_mult": 0.1, "weight_decay": 1.0e-4,
              "grad_clip": 0.1, "ema_decay": 0.9999, "slow_mult": 0.1},
    "schedule": {"epochs": 50, "warmup_epochs": 3, "scheduler": "cosine", "min_lr": 1.0e-6},
    "loss": {"w_cls": 1.0, "w_iou": 2.0, "w_xy": 0.5, "w_ext": 0.5, "focal_gamma": 2.0,
             "cost_cls": 1.0, "cost_iou": 2.0, "cost_xy": 0.5, "cost_ext": 0.5,
             "aux_loss": True, "focal_alpha": 0.25, "w_theta": 0.0, "w_smooth": 0.0},
    "data": {"dataset_manifest": "culane@1", "batch_size": 32, "num_workers": 8, "seed": 42,
             "crop_top_ratio": 270.0 / 590.0,
             "aug": {"hflip_prob": 0.5, "rotation_deg": 0.0, "scale_jitter": 0.0,
                     "brightness": 0.0, "contrast": 0.0}},
    "train": {"amp": True, "channels_last": True, "freeze_bn": True, "eval_conf_thresh": 0.5},
}


@pytest.fixture
def schema() -> ConfigSchema:
    return ConfigSchema.load(FIX / "config_schema.json")


def test_no_overrides_returns_base(schema):
    cfg, ah = tc.resolve(DEFAULT_CONFIG, schema, [])
    assert cfg == DEFAULT_CONFIG          # igual a la base
    assert cfg is not DEFAULT_CONFIG      # pero copia (deep_merge no muta)
    assert ah == config_resolve.arch_hash_for(schema, {})  # hash de los defaults


def test_override_merges_into_effective(schema):
    cfg, _ = tc.resolve(DEFAULT_CONFIG, schema, ["optim.lr=1e-4", "data.aug.rotation_deg=5"])
    assert cfg["optim"]["lr"] == 0.0001
    assert cfg["data"]["aug"]["rotation_deg"] == 5.0
    # el resto de la base intacto (claves congeladas incluidas)
    assert cfg["arch"]["backbone"] == "dla34" and cfg["arch"]["d_model"] == 256
    assert cfg["optim"]["weight_decay"] == 1.0e-4
    assert DEFAULT_CONFIG["optim"]["lr"] == 3.0e-4  # base sin mutar


def test_arch_override_changes_arch_hash(schema):
    base_ah = tc.resolve(DEFAULT_CONFIG, schema, [])[1]
    cfg, ah = tc.resolve(DEFAULT_CONFIG, schema, ["arch.num_queries=20"])
    assert cfg["arch"]["num_queries"] == 20
    assert ah != base_ah
    assert ah == config_resolve.arch_hash_for(schema, {"arch.num_queries": 20})


def test_ref_refine_does_not_change_arch_hash(schema):
    base_ah = tc.resolve(DEFAULT_CONFIG, schema, [])[1]
    _, ah = tc.resolve(DEFAULT_CONFIG, schema, ["arch.ref_refine=xs"])
    assert ah == base_ah  # ref_refine no entra en el arch_hash


def test_invalid_override_rejected(schema):
    with pytest.raises(ContractError):
        tc.resolve(DEFAULT_CONFIG, schema, ["optim.lr=9"])  # > max
    with pytest.raises(ContractError, match="desconocido"):
        tc.resolve(DEFAULT_CONFIG, schema, ["arch.no_existe=1"])


def test_arch_values_reads_effective(schema):
    cfg, _ = tc.resolve(DEFAULT_CONFIG, schema, ["arch.num_decoder_layers=3"])
    assert tc.arch_values(cfg) == {
        "arch.num_queries": 12,
        "arch.num_decoder_layers": 3,
        "arch.n_ref_points": 4,
    }
