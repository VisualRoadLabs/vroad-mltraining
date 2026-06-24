"""Tests de lectura/escritura de shards WebDataset (vroad_mlt.webdataset_io)."""

from __future__ import annotations

import io
import tarfile

import pytest

from vroad_mlt import webdataset_io as wds
from vroad_mlt.lines_format import LinesFile
from vroad_mlt.webdataset_io import ShardWriter, WebDatasetError, read_shard, read_shards, write_shard

GT = {"timestamp": 1781646771, "Lines": [[{"x": 12, "y": 590}, {"x": 770, "y": 290}]]}


def _samples(n):
    return [(f"{i:06d}", f"IMG{i}".encode(), dict(GT, timestamp=1000 + i)) for i in range(n)]


# ----------------------------------------------------------------- round-trip

def test_write_read_shard_roundtrip():
    buf = io.BytesIO()
    assert write_shard(buf, _samples(3)) == 3
    buf.seek(0)
    got = list(read_shard(buf))
    assert [s.key for s in got] == ["000000", "000001", "000002"]
    assert got[0].image == b"IMG0"
    assert isinstance(got[0].lines, LinesFile)
    assert got[1].lines.timestamp == 1001


def test_read_handles_compound_lines_extension():
    buf = io.BytesIO()
    write_shard(buf, _samples(1))
    buf.seek(0)
    (s,) = list(read_shard(buf))
    # la clave es la parte ANTES del primer punto (no '000000.lines')
    assert s.key == "000000"
    assert s.lines.lines[0][0].x == 12


def test_lines_are_validated_on_write():
    buf = io.BytesIO()
    with pytest.raises(Exception):  # formato común inválido (timestamp negativo)
        write_shard(buf, [("000000", b"IMG", {"timestamp": -1, "Lines": []})])


def test_write_sample_rejects_bad_key():
    buf = io.BytesIO()
    tar = tarfile.open(fileobj=buf, mode="w")
    with pytest.raises(WebDatasetError, match="key inválida"):
        wds.write_sample(tar, "a.b", b"IMG", GT)
    tar.close()


# ----------------------------------------------------------------- malformado

def test_missing_lines_raises_on_read():
    # tar con solo la imagen (sin .lines.json)
    buf = io.BytesIO()
    tar = tarfile.open(fileobj=buf, mode="w")
    wds._add(tar, "000000.jpg", b"IMG")
    tar.close()
    buf.seek(0)
    with pytest.raises(WebDatasetError, match="lines.json"):
        list(read_shard(buf))


def test_missing_image_raises_on_read():
    buf = io.BytesIO()
    tar = tarfile.open(fileobj=buf, mode="w")
    wds._add(tar, "000000.lines.json", b'{"timestamp":1,"Lines":[]}')
    tar.close()
    buf.seek(0)
    with pytest.raises(WebDatasetError, match="imagen"):
        list(read_shard(buf))


# ----------------------------------------------------------------- ShardWriter

def test_shardwriter_rolls_over(tmp_path):
    with ShardWriter(tmp_path, "train", maxcount=2) as w:
        for key, img, lines in _samples(5):
            w.write(key, img, lines)
    names = [p.name for p in w.shards]
    assert names == ["train-00000.tar", "train-00001.tar", "train-00002.tar"]
    assert w.total == 5
    # releer todo en orden
    got = list(read_shards(w.shards))
    assert [s.key for s in got] == [f"{i:06d}" for i in range(5)]


def test_shardwriter_single_shard_when_under_maxcount(tmp_path):
    with ShardWriter(tmp_path, "val", maxcount=100) as w:
        for key, img, lines in _samples(3):
            w.write(key, img, lines)
    assert [p.name for p in w.shards] == ["val-00000.tar"]
    assert sum(1 for _ in read_shards(w.shards)) == 3


def test_shardwriter_validates_split(tmp_path):
    with pytest.raises(Exception):
        ShardWriter(tmp_path, "holdout")


def test_deterministic_tar_bytes():
    # Mismo contenido -> mismos bytes (mtime=0), reproducible.
    a, b = io.BytesIO(), io.BytesIO()
    write_shard(a, _samples(2))
    write_shard(b, _samples(2))
    assert a.getvalue() == b.getvalue()
