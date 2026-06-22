"""Tests de la evaluación del trainer (trainer.eval).

Sin lanetr: el `predict` (fake, devuelve formato común) y la `transform` se inyectan; usa la MÉTRICA
REAL (vroad_mlt.metric, numpy/scipy/cv2) sobre carriles sintéticos, shards reales (webdataset_io) y
`lines_format` real. Verifica el adaptador dict->LinesFile, el stream de eval (con src_size + GT sin
mutar), la alineación de categorías, predict_dataset y run_eval de punta a punta (F1=1 si pred==GT).
"""

from __future__ import annotations

import io

import pytest
import torch
from PIL import Image

from trainer import eval as te
from vroad_mlt import lines_format, webdataset_io


# ------------------------------------------------------------------ helpers

def _jpg(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 60, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def _vlane(x: int):
    """Carril vertical en `x` (6 puntos enteros, y=10..110)."""
    return [{"x": x, "y": y} for y in range(10, 111, 20)]


def _gt(lanes_x, ts: int = 5):
    return lines_format.parse({"timestamp": ts, "Lines": [_vlane(x) for x in lanes_x]})


def _pred_dict(lanes_x, ts: int = 5, score: float = 0.9):
    return {"timestamp": ts, "Lines": [_vlane(x) for x in lanes_x], "Scores": [score] * len(lanes_x)}


def _fake_transform(sample, rng):
    sample["image"] = torch.zeros(3, 4, 8)  # imita lanetr: image -> tensor
    sample["targets"] = None
    return sample


class FakeGcs:
    def __init__(self, store):
        self.store = store

    def read_bytes(self, uri):
        return self.store[uri]


# ----------------------------------------------------------- to_lines_file

def test_to_lines_file_from_dict_and_passthrough():
    lf = te.to_lines_file(_pred_dict([100]))
    assert isinstance(lf, lines_format.LinesFile)
    assert lf.lines[0][0].x == 100
    assert lf.scores == [0.9]
    same = lines_format.parse(_pred_dict([5]))
    assert te.to_lines_file(same) is same  # ya es LinesFile -> tal cual


# --------------------------------------------------------------- eval_collate

def test_eval_collate_stacks_and_lists():
    gtA, gtB = _gt([1]), _gt([2])
    out = te.eval_collate([
        (torch.zeros(3, 4, 8), gtA, {"key": "a"}),
        (torch.ones(3, 4, 8), gtB, {"key": "b"}),
    ])
    assert out["image"].shape == (2, 3, 4, 8)
    assert out["gt"] == [gtA, gtB]
    assert [m["key"] for m in out["meta"]] == ["a", "b"]


# ----------------------------------------------------- EvalShardDataset stream

def _shard_bytes(samples):
    buf = io.BytesIO()
    items = [
        (key, _jpg(w, h), {"timestamp": 7, "Lines": [_vlane(x) for x in lanes_x]})
        for key, (w, h), lanes_x in samples
    ]
    webdataset_io.write_shard(buf, items)
    return buf.getvalue()


def test_eval_dataset_yields_image_gt_and_src_size():
    data = _shard_bytes([("000000", (200, 120), [100]), ("000001", (64, 32), [10])])
    gcs = FakeGcs({"gs://b/benchmarks/culane@v1/val-00000.tar": data})
    ds = te.EvalShardDataset(
        ["gs://b/benchmarks/culane@v1/val-00000.tar"], lambda: gcs, _fake_transform
    )
    items = list(ds)
    assert len(items) == 2

    img0, gt0, meta0 = items[0]
    assert img0.shape == (3, 4, 8)                     # lo que puso el fake transform
    assert isinstance(gt0, lines_format.LinesFile)
    assert gt0.lines[0][0].x == 100                    # GT original sin mutar
    assert gt0.scores is None                          # GT no lleva Scores
    assert meta0 == {"key": "000000", "timestamp": 7, "src_size": (200, 120)}

    assert items[1][2]["src_size"] == (64, 32)         # cada imagen, su resolución nativa


# --------------------------------------------------------------- categories_for

def test_categories_for_aligns_by_key():
    cats = te.categories_for(["k1", "k0", "k1"], {"k0": "night", "k1": "normal"})
    assert cats == ["normal", "night", "normal"]       # posicional, alineado a las keys


def test_categories_for_strict_raises_on_missing():
    with pytest.raises(ValueError, match="sin categoría"):
        te.categories_for(["k0", "kX"], {"k0": "normal"})
    # no estricto: deja None (no se usa para el desglose de 9 categorías)
    assert te.categories_for(["kX"], {}, strict=False) == [None]


# --------------------------------------------------------------- predict_dataset

def test_predict_dataset_builds_pred_gt_keys():
    gtA, gtB = _gt([100]), _gt([50])
    loader = [{
        "image": torch.zeros(2, 3, 4, 8),
        "gt": [gtA, gtB],
        "meta": [
            {"key": "k0", "timestamp": 5, "src_size": (200, 120)},
            {"key": "k1", "timestamp": 6, "src_size": (200, 120)},
        ],
    }]

    seen = {}

    def fake_predict(images, src_sizes, conf_thresh=0.0, timestamps=None):
        seen["conf_thresh"] = conf_thresh
        seen["src_sizes"] = src_sizes
        seen["timestamps"] = timestamps
        return [_pred_dict([100], ts=timestamps[0]), _pred_dict([50], ts=timestamps[1])]

    preds, gts, keys = te.predict_dataset(fake_predict, loader, "cpu")

    assert seen["conf_thresh"] == 0.0                  # no pre-filtra (para calibrar offline)
    assert seen["src_sizes"] == [(200, 120), (200, 120)]
    assert seen["timestamps"] == [5, 6]
    assert keys == ["k0", "k1"]
    assert gts == [gtA, gtB]
    assert all(isinstance(p, lines_format.LinesFile) for p in preds)
    assert preds[0].lines[0][0].x == 100 and preds[0].scores == [0.9]


# --------------------------------------------------------------- run_eval (e2e)

class _FakeModel(torch.nn.Module):
    """Modelo falso: `predict` devuelve siempre los mismos carriles (formato común)."""

    def __init__(self, lanes_x):
        super().__init__()
        self.lanes_x = lanes_x

    def predict(self, images, src_sizes, conf_thresh=0.0, timestamps=None):
        ts = timestamps or [0] * len(src_sizes)
        return [_pred_dict(self.lanes_x, ts=ts[i]) for i in range(len(src_sizes))]


def _loader_one_lane():
    return [{
        "image": torch.zeros(2, 3, 4, 8),
        "gt": [_gt([100]), _gt([100])],  # GT = un carril en x=100 (igual que la predicción)
        "meta": [
            {"key": "k0", "timestamp": 5, "src_size": (200, 120)},
            {"key": "k1", "timestamp": 5, "src_size": (200, 120)},
        ],
    }]


def test_run_eval_perfect_match_with_categories():
    model = _FakeModel([100])
    results, preds, gts, keys = te.run_eval(
        model, _loader_one_lane(), "cpu",
        categories_map={"k0": "normal", "k1": "night"},
        threshold=0.5, img_shape=(120, 200),  # (H, W)
    )
    assert results["benchmark"] == "culane@v1"
    assert results["f1_global"] == pytest.approx(1.0)  # pred == GT -> TP perfecto
    assert results["metrics_by_category"]["normal"] == pytest.approx(1.0)
    assert results["metrics_by_category"]["night"] == pytest.approx(1.0)
    assert len(preds) == len(gts) == len(keys) == 2


def test_run_eval_restores_train_mode_and_calibrates():
    model = _FakeModel([100])
    model.train()
    results, *_ = te.run_eval(
        model, _loader_one_lane(), "cpu",
        threshold=0.5, calibrate_thresholds=[0.3, 0.6, 0.95], img_shape=(120, 200),
    )
    assert model.training is True                       # run_eval restaura el modo previo
    assert results["f1_global"] == pytest.approx(1.0)
    assert results["threshold"] in (0.3, 0.6, 0.95)     # reporta con el umbral calibrado
