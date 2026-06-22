"""Tests del dataloader del trainer (trainer.data).

Usa torch + Pillow (disponibles), pero NO lanetr: la transformación se inyecta (fake) y el
cliente GCS también. Verifica decode, collate y el streaming de shards (+ varias resoluciones).
"""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image

from trainer import data as td
from vroad_mlt import lines_format, webdataset_io
from vroad_mlt.webdataset_io import Sample


def _jpg(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 60, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def _lines(pts_per_lane, timestamp=7):
    d = {"timestamp": timestamp,
         "Lines": [[{"x": x, "y": y} for x, y in lane] for lane in pts_per_lane]}
    return lines_format.parse(d)


def _fake_transform(sample, rng):
    # imita lo que hace lanetr: image -> tensor, añade targets. No toca meta.
    sample["image"] = torch.zeros(3, 4, 8)
    sample["targets"] = {"n_lanes": len(sample["lanes"])}
    return sample


class FakeGcs:
    def __init__(self, store):
        self.store = store

    def read_bytes(self, uri):
        return self.store[uri]


# --------------------------------------------------------------- decode_sample

def test_decode_sample_image_and_lanes():
    lf = _lines([[(10, 20), (30, 40)], [(1, 2), (3, 4)]])
    s = Sample(key="000000", image=_jpg(64, 32), lines=lf)
    out = td.decode_sample(s)
    assert out["image"].size == (64, 32)        # (W, H) nativos
    assert len(out["lanes"]) == 2
    assert np.array_equal(out["lanes"][0], np.array([[10, 20], [30, 40]], np.float32))
    assert out["meta"] == {"key": "000000", "timestamp": 7}
    assert out["slots"] is None


def test_decode_sample_preserves_any_resolution():
    # Varias resoluciones: cada imagen conserva su tamaño nativo (no se hardcodea).
    for w, h in [(1640, 590), (1280, 720), (800, 320)]:
        s = Sample(key="k", image=_jpg(w, h), lines=_lines([[(0, 0), (1, 1)]]))
        assert td.decode_sample(s)["image"].size == (w, h)


# ------------------------------------------------------------------- collate

def test_collate_stacks_images_and_lists_targets():
    batch = [
        (torch.zeros(3, 4, 8), {"n_lanes": 2}, {"key": "a"}),
        (torch.ones(3, 4, 8), {"n_lanes": 1}, {"key": "b"}),
    ]
    out = td.collate(batch)
    assert out["image"].shape == (2, 3, 4, 8)
    assert out["targets"] == [{"n_lanes": 2}, {"n_lanes": 1}]
    assert [m["key"] for m in out["meta"]] == ["a", "b"]


# ----------------------------------------------------- LaneShardDataset stream

def _make_shard_bytes(n):
    buf = io.BytesIO()
    samples = [
        (f"{i:06d}", _jpg(64, 32),
         {"timestamp": 1000 + i, "Lines": [[{"x": i, "y": 0}, {"x": i, "y": 10}]]})
        for i in range(n)
    ]
    webdataset_io.write_shard(buf, samples)
    return buf.getvalue()


def test_dataset_streams_decodes_and_transforms():
    gcs = FakeGcs({"gs://b/shards/train-00000.tar": _make_shard_bytes(3)})
    ds = td.LaneShardDataset(
        ["gs://b/shards/train-00000.tar"], lambda: gcs, _fake_transform, shuffle=False
    )
    items = list(ds)
    assert len(items) == 3
    img, targets, meta = items[0]
    assert img.shape == (3, 4, 8)              # lo que puso el fake transform
    assert targets == {"n_lanes": 1}            # 1 carril por sample
    assert meta["timestamp"] == 1000


def test_dataset_iterates_multiple_shards():
    gcs = FakeGcs({
        "gs://b/shards/train-00000.tar": _make_shard_bytes(2),
        "gs://b/shards/train-00001.tar": _make_shard_bytes(3),
    })
    ds = td.LaneShardDataset(
        ["gs://b/shards/train-00000.tar", "gs://b/shards/train-00001.tar"],
        lambda: gcs, _fake_transform,
    )
    assert sum(1 for _ in ds) == 5


def test_dataset_shuffle_is_deterministic_per_epoch():
    uris = [f"gs://b/shards/train-{i:05d}.tar" for i in range(8)]
    ds = td.LaneShardDataset(uris, lambda: None, _fake_transform, shuffle=True, seed=1, epoch=0)
    a = ds._shards_for_worker()
    ds.set_epoch(0)
    assert ds._shards_for_worker() == a        # misma época -> mismo orden
    ds.set_epoch(1)
    assert ds._shards_for_worker() != a        # otra época -> otro orden
