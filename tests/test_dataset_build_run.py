"""Tests de la orquestación de dataset-build (jobs/dataset-build/run.py).

Usa fakes en memoria de bq/gcs (inyección de dependencias): prueba la lógica del
runner sin tocar GCP. La verificación real se hace con `python jobs/dataset-build/run.py`.
"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
from google.api_core import exceptions as gax

from vroad_mlt.dataset_spec import Spec
from vroad_mlt.webdataset_io import read_shard

# Cargar run.py por ruta (jobs/dataset-build/ no es un paquete importable).
_RUN_PATH = Path(__file__).parents[1] / "jobs" / "dataset-build" / "run.py"
_spec = importlib.util.spec_from_file_location("dataset_build_run", _RUN_PATH)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)

PUB = "gs://bkt-prod-public-usc1/culane"
OUT = "bkt-dev-datasets-usc1"
GT = b'{"timestamp":1,"Lines":[[{"x":1,"y":2},{"x":3,"y":4}]]}'


class FakeBq:
    def __init__(self, rows_by_key):
        self.rows_by_key = rows_by_key  # (dataset, split) -> [rows]

    def query(self, sql, params=None):
        return list(self.rows_by_key.get((params["dataset"], params["split"]), []))


class FakeGcs:
    def __init__(self):
        self.store: dict[str, bytes] = {}

    def read_bytes(self, uri):
        if uri not in self.store:
            raise gax.NotFound(uri)
        return self.store[uri]

    def read_text(self, uri):
        return self.read_bytes(uri).decode("utf-8")

    def write_bytes(self, uri, data, **kw):
        self.store[uri] = data

    def write_text(self, uri, text, **kw):
        self.store[uri] = text.encode("utf-8")

    def write_json(self, uri, obj, **kw):
        self.store[uri] = json.dumps(obj).encode("utf-8")

    def exists(self, uri):
        return uri in self.store

    def upload_file(self, uri, local_path):
        self.store[uri] = Path(local_path).read_bytes()


def _add_image(gcs, i, *, with_lines=True, clip="clip"):
    img = f"{PUB}/images/{clip}/{i:05d}.jpg"
    gcs.store[img] = f"IMG{i}".encode()
    if with_lines:
        gcs.store[f"{PUB}/label/{clip}/{i:05d}.lines.json"] = GT
    return {"image_id": f"id{i}", "gcs_uri": img, "width": 1640, "height": 590}


def _spec(dedup=False, maxcount=10000):
    return Spec.from_dict({
        "manifest_id": "m", "version": "1", "shard_maxcount": maxcount,
        "sources": [{"source": "public", "dataset": "culane", "version": "1",
                     "splits": ["train"], "dedup": dedup}],
    })


def _shard_uris(gcs):
    return sorted(u for u in gcs.store if "/shards/" in u)


def test_materialize_basic(tmp_path):
    gcs = FakeGcs()
    rows = [_add_image(gcs, i) for i in range(3)]
    bq = FakeBq({("culane", "train"): rows})
    m = run.materialize(_spec(maxcount=2), bq=bq, gcs=gcs, dl_project="P", out_bucket=OUT,
                        tmp_dir=tmp_path, assets_prefix="gs://x/_assets/culane")
    assert m.counts == {"train": 3}
    # maxcount=2 -> 2 shards
    assert [u.split("/")[-1] for u in _shard_uris(gcs)] == ["train-00000.tar", "train-00001.tar"]
    # shards bajo shards/culane@1/
    assert all("/shards/culane@1/" in u for u in _shard_uris(gcs))
    # manifiesto escrito
    assert f"gs://{OUT}/manifests/m@1.json" in gcs.store


def test_materialize_skips_missing_lines(tmp_path):
    gcs = FakeGcs()
    rows = [_add_image(gcs, 0), _add_image(gcs, 1, with_lines=False), _add_image(gcs, 2)]
    bq = FakeBq({("culane", "train"): rows})
    m = run.materialize(_spec(), bq=bq, gcs=gcs, dl_project="P", out_bucket=OUT,
                        tmp_dir=tmp_path, assets_prefix="gs://x/_assets/culane")
    assert m.counts == {"train": 2}  # el que no tiene .lines.json se salta


def test_materialize_shard_content_roundtrips(tmp_path):
    gcs = FakeGcs()
    bq = FakeBq({("culane", "train"): [_add_image(gcs, 0)]})
    run.materialize(_spec(), bq=bq, gcs=gcs, dl_project="P", out_bucket=OUT,
                    tmp_dir=tmp_path, assets_prefix="gs://x/_assets/culane")
    (shard_uri,) = _shard_uris(gcs)
    samples = list(read_shard(io.BytesIO(gcs.store[shard_uri])))
    assert len(samples) == 1
    assert samples[0].key == "00000000"
    assert samples[0].image == b"IMG0"
    assert samples[0].lines.lines[0][0].x == 1


def test_materialize_culane_dedup(tmp_path):
    gcs = FakeGcs()
    # assets: 3 frames con diffs [20, 5, 30] -> umbral 15 conserva clip/00000 y clip/00002
    gcs.store["gs://x/_assets/culane/train.txt"] = (
        "/clip/00000.jpg x\n/clip/00001.jpg x\n/clip/00002.jpg x\n".encode()
    )
    buf = io.BytesIO()
    np.savez(buf, data=np.array([20.0, 5.0, 30.0]))
    gcs.store["gs://x/_assets/culane/train_diffs.npz"] = buf.getvalue()

    rows = [_add_image(gcs, i) for i in range(3)]  # clip/00000..00002
    bq = FakeBq({("culane", "train"): rows})
    m = run.materialize(_spec(dedup=True), bq=bq, gcs=gcs, dl_project="P", out_bucket=OUT,
                        tmp_dir=tmp_path, assets_prefix="gs://x/_assets/culane")
    assert m.counts == {"train": 2}  # clip/00001 (diff 5 < 15) descartado


def test_sample_key():
    assert run.sample_key(0) == "00000000"
    assert run.sample_key(42) == "00000042"
