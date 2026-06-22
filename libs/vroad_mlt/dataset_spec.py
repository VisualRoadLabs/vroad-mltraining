"""vroad_mlt.dataset_spec — la spec de entrada de `dataset-build`.

Es lo que el dashboard genera y el job consume (ver jobs/dataset-build/SPEC.md).
Describe qué fuentes/splits/filtros materializar; el job produce de ahí los shards
y el manifiesto. Lógica pura (validación + conversión a fuentes de manifiesto).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from vroad_mlt.manifest import Source
from vroad_mlt.naming import validate_split

__all__ = ["SpecError", "DEFAULT_SHARD_MAXCOUNT", "VALID_SOURCES", "SourceSpec", "Spec"]

DEFAULT_SHARD_MAXCOUNT = 10000
VALID_SOURCES = ("public", "user")


class SpecError(ValueError):
    """La spec de dataset-build está mal formada."""


@dataclass(frozen=True)
class SourceSpec:
    """Una fuente a materializar: dataset@version + splits + filtros (+ dedup CULane)."""

    source: str
    dataset: str
    version: str
    splits: tuple[str, ...]
    filters: dict = field(default_factory=dict)
    dedup: bool = False

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SourceSpec":
        for k in ("source", "dataset", "version", "splits"):
            if k not in d:
                raise SpecError(f"source sin '{k}': {dict(d)!r}")
        if d["source"] not in VALID_SOURCES:
            raise SpecError(f"source inválido {d['source']!r} (válidos: {VALID_SOURCES})")
        splits = tuple(d["splits"])
        if not splits:
            raise SpecError("'splits' no puede estar vacío")
        for s in splits:
            validate_split(s)
        return cls(
            source=d["source"],
            dataset=d["dataset"],
            version=str(d["version"]),
            splits=splits,
            filters=dict(d.get("filters", {})),
            dedup=bool(d.get("dedup", False)),
        )

    def to_manifest_source(self) -> Source:
        """Convierte a una `Source` de manifiesto (split_map identidad)."""
        return Source(
            dataset=self.dataset,
            version=self.version,
            split_map={s: s for s in self.splits},
            filters=dict(self.filters),
        )


@dataclass(frozen=True)
class Spec:
    """La spec completa de un dataset a materializar."""

    manifest_id: str
    version: str
    sources: tuple[SourceSpec, ...]
    shard_maxcount: int = DEFAULT_SHARD_MAXCOUNT

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Spec":
        for k in ("manifest_id", "version", "sources"):
            if k not in d:
                raise SpecError(f"spec sin '{k}'")
        if not isinstance(d["sources"], list) or not d["sources"]:
            raise SpecError("'sources' debe ser una lista no vacía")
        return cls(
            manifest_id=d["manifest_id"],
            version=str(d["version"]),
            sources=tuple(SourceSpec.from_dict(s) for s in d["sources"]),
            shard_maxcount=int(d.get("shard_maxcount", DEFAULT_SHARD_MAXCOUNT)),
        )

    @classmethod
    def from_json(cls, text: str) -> "Spec":
        return cls.from_dict(json.loads(text))

    def to_manifest_sources(self) -> list[Source]:
        return [s.to_manifest_source() for s in self.sources]
