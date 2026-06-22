"""vroad_mlt.manifest — el manifiesto de dataset (linaje + hash de contenido).

Un manifiesto describe CON QUÉ datos se entrena: las fuentes (dataset@version +
split_map + filtros), los `counts` por split, el patrón de shards y la lista de
frames-curva. Su `content_hash` da inmutabilidad lógica (detecta si alguien
re-materializa el mismo id@version con datos distintos).

El `ManifestLock` es el provenance CONGELADO de una corrida (lo escribe el trainer
en `manifest.lock.json`): id@version + hash + fuentes resueltas + counts + cuándo.

Lógica pura (solo stdlib): construir/validar/serializar/hashar. La lectura/escritura
a GCS la hace `gcs` (el llamante pasa el dict/JSON).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence

__all__ = [
    "ManifestError",
    "DEFAULT_SHARD_PATTERN",
    "HASH_PREFIX",
    "Source",
    "Manifest",
    "ManifestLock",
    "compute_content_hash",
]

DEFAULT_SHARD_PATTERN = "shards/{dataset}@{version}/{split}-{shard}.tar"
HASH_PREFIX = "sha256:"


class ManifestError(ValueError):
    """Un manifiesto (o lock) está mal formado."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_content_hash(manifest_dict: Mapping[str, Any]) -> str:
    """Hash del CONTENIDO: todo menos `content_hash` y `created_at` (volátiles)."""
    content = {k: v for k, v in manifest_dict.items() if k not in ("content_hash", "created_at")}
    return HASH_PREFIX + hashlib.sha256(_canonical(content).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Source:
    """Una fuente del manifiesto: dataset@version + mapeo de splits + filtros."""

    dataset: str
    version: str
    split_map: dict[str, str]
    filters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Source":
        for k in ("dataset", "version", "split_map"):
            if k not in d:
                raise ManifestError(f"source sin '{k}': {dict(d)!r}")
        if not isinstance(d["split_map"], Mapping):
            raise ManifestError("source.split_map debe ser un objeto")
        return cls(
            dataset=d["dataset"],
            version=str(d["version"]),
            split_map=dict(d["split_map"]),
            filters=dict(d.get("filters", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "version": self.version,
            "split_map": dict(self.split_map),
            "filters": dict(self.filters),
        }


@dataclass(frozen=True)
class Manifest:
    """Un manifiesto de dataset."""

    manifest_id: str
    version: str
    sources: tuple[Source, ...]
    counts: dict[str, int]
    created_at: Optional[str] = None
    content_hash: Optional[str] = None
    curve_keys_uri: Optional[str] = None
    shard_pattern: str = DEFAULT_SHARD_PATTERN

    # ------------------------------------------------------------- construir
    @classmethod
    def build(
        cls,
        manifest_id: str,
        version: str,
        sources: Sequence[Source],
        counts: Mapping[str, int],
        *,
        created_at: Optional[str] = None,
        curve_keys_uri: Optional[str] = None,
        shard_pattern: str = DEFAULT_SHARD_PATTERN,
    ) -> "Manifest":
        """Crea un manifiesto y calcula su `content_hash`."""
        m = cls(
            manifest_id=manifest_id,
            version=str(version),
            sources=tuple(sources),
            counts=dict(counts),
            created_at=created_at,
            content_hash=None,
            curve_keys_uri=curve_keys_uri,
            shard_pattern=shard_pattern,
        )
        return replace(m, content_hash=m.compute_hash())

    # ------------------------------------------------------------- hash
    def compute_hash(self) -> str:
        """(Re)calcula el hash de contenido (independiente de created_at/content_hash)."""
        return compute_content_hash(self.to_dict())

    def verify_hash(self) -> bool:
        """True si el `content_hash` guardado coincide con el contenido actual."""
        return self.content_hash is not None and self.content_hash == self.compute_hash()

    # ------------------------------------------------------------- serializar
    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"manifest_id": self.manifest_id, "version": self.version}
        if self.created_at is not None:
            out["created_at"] = self.created_at
        if self.content_hash is not None:
            out["content_hash"] = self.content_hash
        out["sources"] = [s.to_dict() for s in self.sources]
        out["counts"] = dict(self.counts)
        if self.curve_keys_uri is not None:
            out["curve_keys_uri"] = self.curve_keys_uri
        out["shard_pattern"] = self.shard_pattern
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Manifest":
        for k in ("manifest_id", "version", "sources", "counts"):
            if k not in d:
                raise ManifestError(f"manifiesto sin '{k}'")
        if not isinstance(d["sources"], list):
            raise ManifestError("'sources' debe ser una lista")
        if not isinstance(d["counts"], Mapping):
            raise ManifestError("'counts' debe ser un objeto split->int")
        return cls(
            manifest_id=d["manifest_id"],
            version=str(d["version"]),
            sources=tuple(Source.from_dict(s) for s in d["sources"]),
            counts={k: int(v) for k, v in d["counts"].items()},
            created_at=d.get("created_at"),
            content_hash=d.get("content_hash"),
            curve_keys_uri=d.get("curve_keys_uri"),
            shard_pattern=d.get("shard_pattern", DEFAULT_SHARD_PATTERN),
        )

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        return cls.from_dict(json.loads(text))

    # ------------------------------------------------------------- integración
    def to_experiment_fields(self) -> dict[str, Any]:
        """Campos de linaje para `tbl_experiments`."""
        return {
            "dataset_manifest_id": self.manifest_id,
            "dataset_manifest_version": self.version,
            "dataset_manifest_hash": self.content_hash,
            "dataset_sources_json": [s.to_dict() for s in self.sources],
            "num_train": self.counts.get("train"),
            "num_val": self.counts.get("val"),
            "num_test": self.counts.get("test"),
        }


@dataclass(frozen=True)
class ManifestLock:
    """Provenance congelado de una corrida (`manifest.lock.json`)."""

    manifest_id: str
    version: str
    content_hash: Optional[str]
    resolved_sources: list[dict]
    counts: dict[str, int]
    resolved_at: Optional[str] = None

    @classmethod
    def from_manifest(cls, manifest: Manifest, *, resolved_at: Optional[str] = None) -> "ManifestLock":
        return cls(
            manifest_id=manifest.manifest_id,
            version=manifest.version,
            content_hash=manifest.content_hash,
            resolved_sources=[s.to_dict() for s in manifest.sources],
            counts=dict(manifest.counts),
            resolved_at=resolved_at,
        )

    def verify(self, manifest: Manifest) -> bool:
        """True si el manifiesto (re-leído) coincide en id@version y hash de contenido."""
        return (
            self.manifest_id == manifest.manifest_id
            and self.version == manifest.version
            and self.content_hash == manifest.content_hash
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "manifest_id": self.manifest_id,
            "version": self.version,
            "content_hash": self.content_hash,
            "resolved_sources": self.resolved_sources,
            "counts": dict(self.counts),
        }
        if self.resolved_at is not None:
            out["resolved_at"] = self.resolved_at
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ManifestLock":
        for k in ("manifest_id", "version", "resolved_sources", "counts"):
            if k not in d:
                raise ManifestError(f"lock sin '{k}'")
        return cls(
            manifest_id=d["manifest_id"],
            version=str(d["version"]),
            content_hash=d.get("content_hash"),
            resolved_sources=list(d["resolved_sources"]),
            counts={k: int(v) for k, v in d["counts"].items()},
            resolved_at=d.get("resolved_at"),
        )

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "ManifestLock":
        return cls.from_dict(json.loads(text))


# --------------------------------------------------------------------------- CLI


def _main(argv: Optional[list[str]] = None) -> int:
    """Comprobación manual: construye el manifiesto de ejemplo y lo resume.

        python -m vroad_mlt.manifest
    """
    sources = [
        Source("culane", "1", {"train": "train", "val": "val", "test": "test"}, {}),
        Source("user-curves", "2", {"train": "train"}, {"road_geometry": "curve"}),
    ]
    m = Manifest.build(
        "culane-mix-curves", "3", sources,
        {"train": 61234, "val": 9675, "test": 34680},
        created_at="20260620T101500Z",
        curve_keys_uri="gs://bkt-dev-datasets-usc1/manifests/culane-mix-curves@3.curve_keys.json",
    )
    print(m.to_json())
    print(f"\nverify_hash: {m.verify_hash()}")
    print(f"round-trip from_json preserva hash: {Manifest.from_json(m.to_json()).content_hash == m.content_hash}")
    lock = ManifestLock.from_manifest(m, resolved_at="20260620T101800Z")
    print(f"\nmanifest.lock.json:\n{lock.to_json()}")
    print(f"\nlock.verify(manifest): {lock.verify(m)}")
    print(f"\nexperiment fields: {json.dumps(m.to_experiment_fields(), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
