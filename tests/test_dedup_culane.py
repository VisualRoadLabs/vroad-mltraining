"""Tests del dedup de CULane (vroad_mlt.datalake.dedup_culane).

Lógica pura con datos sintéticos.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vroad_mlt.datalake.assets import dedup_culane as dd

LOCAL_LIST = Path("D:/CULane/list")


# --------------------------------------------------------------- pura

def test_normalize_culane_key():
    assert dd.normalize_culane_key("/driver_x/00000.jpg") == "driver_x/00000.jpg"
    assert dd.normalize_culane_key("  driver_x/00000.jpg \n") == "driver_x/00000.jpg"


def test_culane_key_from_gcs_uri():
    uri = "gs://bkt-prod-public-usc1/culane/images/driver_23_30frame/05151649_0422.MP4/00000.jpg"
    assert dd.culane_key_from_gcs_uri(uri) == "driver_23_30frame/05151649_0422.MP4/00000.jpg"


def test_culane_key_from_gcs_uri_requires_images():
    with pytest.raises(ValueError, match="/images/"):
        dd.culane_key_from_gcs_uri("gs://b/culane/x/00000.jpg")


def test_build_keep_set_threshold_inclusive():
    keys = ["/a/0.jpg", "/a/1.jpg", "/a/2.jpg", "/a/3.jpg"]
    diffs = [20.0, 10.0, 15.0, 14.999]  # umbral 15: conserva 0 (20) y 2 (15), descarta 1 y 3
    keep = dd.build_keep_set(keys, diffs, threshold=15.0)
    assert keep == {"a/0.jpg", "a/2.jpg"}


def test_build_keep_set_length_mismatch():
    with pytest.raises(ValueError, match="misma longitud"):
        dd.build_keep_set(["/a/0.jpg"], [1.0, 2.0])


def test_default_threshold_is_15():
    assert dd.DEFAULT_DIFF_THRESHOLD == 15.0


# --------------------------------------------------- REAL (solo en tu máquina)

@pytest.mark.skipif(not (LOCAL_LIST / "train.txt").exists(), reason="ficheros CULane locales no presentes")
def test_real_files_reproduce_train_gt_new():
    keep = dd.keep_set_from_files(LOCAL_LIST / "train.txt", str(LOCAL_LIST / "train_diffs.npz"))
    ref = {
        dd.normalize_culane_key(ln.split()[0])
        for ln in (LOCAL_LIST / "train_gt_new.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }
    assert len(keep) == 55698
    assert keep == ref
