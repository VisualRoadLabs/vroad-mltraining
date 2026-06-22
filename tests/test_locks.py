"""Tests de la lógica PURA de los locks de GPU (vroad_mlt.locks).

El compare-and-swap real se verifica a mano con `python -m vroad_mlt.locks`.
"""

from __future__ import annotations

import pytest

from vroad_mlt import locks as lk
from vroad_mlt import naming
from vroad_mlt.locks import LocksError, lock_payload, normalize_accelerator, slot_keys


# ----------------------------------------------------- normalize_accelerator

@pytest.mark.parametrize(
    "accel, expected",
    [
        ("a100", "a100"),
        ("A100", "a100"),
        ("NVIDIA_TESLA_A100", "a100"),
        ("nvidia_tesla_a100", "a100"),
        ("l4", "l4"),
        ("L4", "l4"),
        ("NVIDIA_L4", "l4"),
    ],
)
def test_normalize_accelerator(accel, expected):
    assert normalize_accelerator(accel) == expected


def test_normalize_accelerator_unknown():
    with pytest.raises(LocksError):
        normalize_accelerator("V100")


# -------------------------------------------------------------- slot_keys

def test_slot_keys_a100_single():
    assert slot_keys("a100") == ["locks/a100.lock"]


def test_slot_keys_l4_uses_max():
    assert slot_keys("l4", max_l4=3) == [
        "locks/l4/slot-0.lock",
        "locks/l4/slot-1.lock",
        "locks/l4/slot-2.lock",
    ]
    assert slot_keys("l4", max_l4=1) == ["locks/l4/slot-0.lock"]


def test_slot_keys_match_naming_convention():
    # Garantiza que no derivan de las claves de `naming`.
    assert slot_keys("a100") == [naming.a100_lock_key()]
    assert slot_keys("l4", max_l4=3) == naming.l4_lock_keys(3)


def test_slot_keys_custom_root_for_tests():
    assert slot_keys("a100", root="_test_locks/x") == ["_test_locks/x/a100.lock"]
    assert slot_keys("l4", max_l4=2, root="_test_locks/x") == [
        "_test_locks/x/l4/slot-0.lock",
        "_test_locks/x/l4/slot-1.lock",
    ]


# -------------------------------------------------------------- lock_payload

def test_lock_payload():
    p = lock_payload("run-1", "alice", "2026-06-20T10:15:00Z")
    assert p == {"run_id": "run-1", "who": "alice", "claimed_at": "2026-06-20T10:15:00Z"}
