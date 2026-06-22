"""vroad_mlt.webdataset_io — leer/escribir samples (jpg + .lines.json) en shards .tar.

Convención WebDataset: ficheros con el mismo *basename* (la parte antes del primer
punto) = un sample. Aquí cada sample son DOS ficheros:

    000001.jpg            # imagen
    000001.lines.json     # GT en formato común

`dataset-build` ESCRIBE los shards (`train-00000.tar`, ...) y el `trainer` los LEE.
Lógica pura (solo stdlib `tarfile` + `lines_format`): la subida/bajada a GCS la hace
`gcs`. Los `.tar` son deterministas (mtime=0) para que sean reproducibles.
"""

from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Optional, Union

from vroad_mlt import lines_format
from vroad_mlt.lines_format import LinesFile
from vroad_mlt.naming import shard_name, validate_split

__all__ = [
    "WebDatasetError",
    "IMAGE_EXT",
    "LINES_EXT",
    "Sample",
    "write_sample",
    "write_shard",
    "read_shard",
    "read_shards",
    "ShardWriter",
]

IMAGE_EXT = "jpg"
LINES_EXT = "lines.json"
_IMAGE_EXTS = ("jpg", "jpeg", "png")

_Src = Union[str, Path, io.IOBase]
_Lines = Union[LinesFile, Mapping]


class WebDatasetError(ValueError):
    """Shard o sample mal formado."""


@dataclass
class Sample:
    """Un sample de WebDataset: clave + bytes de imagen + GT (formato común parseado)."""

    key: str
    image: bytes
    lines: LinesFile


def _lines_bytes(lines: _Lines) -> bytes:
    """Serializa el GT a bytes en formato común (validándolo)."""
    lf = lines if isinstance(lines, LinesFile) else lines_format.parse(lines)
    return lines_format.dumps(lf).encode("utf-8")


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0  # determinista
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def write_sample(tar: tarfile.TarFile, key: str, image: bytes, lines: _Lines) -> None:
    """Escribe un sample (jpg + lines.json) en un tar abierto. Los dos ficheros, juntos."""
    if "." in key or "/" in key:
        raise WebDatasetError(f"key inválida {key!r}: sin '.' ni '/'")
    _add(tar, f"{key}.{IMAGE_EXT}", image)
    _add(tar, f"{key}.{LINES_EXT}", _lines_bytes(lines))


def write_shard(dest: _Src, samples: Iterable[tuple[str, bytes, _Lines]]) -> int:
    """Escribe varios samples en UN shard. Devuelve cuántos escribió."""
    n = 0
    if isinstance(dest, (str, Path)):
        tar = tarfile.open(dest, "w")
    else:
        tar = tarfile.open(fileobj=dest, mode="w")
    try:
        for key, image, lines in samples:
            write_sample(tar, key, image, lines)
            n += 1
    finally:
        tar.close()
    return n


def _make_sample(key: str, parts: dict[str, bytes]) -> Sample:
    image = next((parts[e] for e in _IMAGE_EXTS if e in parts), None)
    raw = parts.get(LINES_EXT)
    if image is None:
        raise WebDatasetError(f"sample {key!r} sin imagen ({_IMAGE_EXTS})")
    if raw is None:
        raise WebDatasetError(f"sample {key!r} sin {LINES_EXT}")
    return Sample(key=key, image=image, lines=lines_format.loads(raw.decode("utf-8")))


def read_shard(src: _Src) -> Iterator[Sample]:
    """Itera los samples de un shard .tar (agrupa por basename, ficheros consecutivos)."""
    if isinstance(src, (str, Path)):
        tar = tarfile.open(src, "r")
    else:
        tar = tarfile.open(fileobj=src, mode="r")
    try:
        current: Optional[str] = None
        parts: dict[str, bytes] = {}
        for member in tar:
            if not member.isfile():
                continue
            key, _, ext = member.name.partition(".")
            data = tar.extractfile(member).read()  # type: ignore[union-attr]
            if current is None:
                current = key
            elif key != current:
                yield _make_sample(current, parts)
                parts, current = {}, key
            parts[ext] = data
        if current is not None and parts:
            yield _make_sample(current, parts)
    finally:
        tar.close()


def read_shards(srcs: Iterable[_Src]) -> Iterator[Sample]:
    """Itera los samples de varios shards en orden."""
    for s in srcs:
        yield from read_shard(s)


class ShardWriter:
    """Escribe samples en shards `<split>-<NNNNN>.tar` que rotan cada `maxcount`.

        with ShardWriter(out_dir, "train", maxcount=10000) as w:
            w.write(key, image_bytes, lines)
        w.shards  # rutas de los shards escritos
    """

    def __init__(
        self,
        out_dir: Union[str, Path],
        split: str,
        *,
        maxcount: int = 10000,
        digits: int = 5,
        on_shard_done: Optional[Callable[[Path], None]] = None,
    ) -> None:
        validate_split(split)
        if maxcount < 1:
            raise WebDatasetError("maxcount debe ser >= 1")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.maxcount = maxcount
        self.digits = digits
        # Callback que se llama con la ruta de cada shald al CERRARSE (para subirlo y
        # borrarlo, y no acumular todos los shards en disco/RAM).
        self.on_shard_done = on_shard_done
        self.shards: list[Path] = []
        self.total = 0
        self._tar: Optional[tarfile.TarFile] = None
        self._in_shard = 0
        self._index = 0

    def _finalize_current(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None
            if self.on_shard_done is not None:
                self.on_shard_done(self.shards[-1])

    def _roll(self) -> None:
        self._finalize_current()
        path = self.out_dir / shard_name(self.split, self._index, digits=self.digits)
        self._tar = tarfile.open(path, "w")
        self.shards.append(path)
        self._in_shard = 0
        self._index += 1

    def write(self, key: str, image: bytes, lines: _Lines) -> None:
        if self._tar is None or self._in_shard >= self.maxcount:
            self._roll()
        assert self._tar is not None
        write_sample(self._tar, key, image, lines)
        self._in_shard += 1
        self.total += 1

    def close(self) -> None:
        self._finalize_current()

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- CLI


def _main(argv: Optional[list[str]] = None) -> int:
    """Comprobación manual: escribe samples sintéticos en shards y los relee.

        python -m vroad_mlt.webdataset_io selftest
    """
    import sys
    import tempfile

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "selftest":
        print(_main.__doc__, file=sys.stderr)
        return 2

    samples = [
        (f"{i:06d}", f"FAKEJPG{i}".encode(),
         {"timestamp": 1000 + i, "Lines": [[{"x": i, "y": 590}, {"x": i + 10, "y": 290}]]})
        for i in range(5)
    ]
    with tempfile.TemporaryDirectory() as d:
        with ShardWriter(d, "train", maxcount=2) as w:
            for key, img, lines in samples:
                w.write(key, img, lines)
        print(f"shards escritos: {[p.name for p in w.shards]} (total {w.total})")
        read = list(read_shards(w.shards))
        print(f"samples releídos: {len(read)}")
        ok = (
            len(read) == 5
            and [s.key for s in read] == [k for k, _, _ in samples]
            and read[0].image == b"FAKEJPG0"
            and read[0].lines.lines[0][0].y == 590
        )
        print("OK selftest" if ok else "FALLO: round-trip incorrecto")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
