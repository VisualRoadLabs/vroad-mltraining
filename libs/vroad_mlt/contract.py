"""vroad_mlt.contract — lector ligero de los artefactos declarativos del modelo.

El modelo (lanetr) exporta TRES ficheros a `_model/<name>@<sha>/` (lo hace su CI):
- `config_schema.json` : lista de parámetros modificables (CONFIG_SCHEMA),
- `metrics_spec.json`  : etiqueta/orden de las métricas escalares (METRICS_SPEC),
- `model_info.json`    : identidad/forma del modelo (MODEL_INFO).

Este módulo SOLO los lee y valida (JSON puro, sin `torch`). Lo usan:
- el dashboard, para generar el formulario y etiquetar las curvas,
- `config_resolve`, para defaults, validar overrides y calcular el `arch_hash`.

La parte PESADA del contrato (importar `lanetr`, `build_model`, `conformance`)
vive aparte y necesita `torch`; llega en un paso futuro.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

__all__ = [
    "ContractError",
    "VALID_TYPES",
    "ARCH_HASH_KEYS",
    "CONFIG_SCHEMA_FILE",
    "METRICS_SPEC_FILE",
    "MODEL_INFO_FILE",
    "ParamSpec",
    "ConfigSchema",
    "MetricsSpec",
    "ModelInfo",
    "ModelContract",
]

VALID_TYPES = ("choice", "int", "float", "bool", "str")

# Los parámetros que determinan la FORMA de los pesos -> entran en el arch_hash
# (subconjunto del grupo 'arch'; NO incluye ref_refine ni load_strict).
ARCH_HASH_KEYS = ("arch.num_queries", "arch.num_decoder_layers", "arch.n_ref_points")

CONFIG_SCHEMA_FILE = "config_schema.json"
METRICS_SPEC_FILE = "metrics_spec.json"
MODEL_INFO_FILE = "model_info.json"

_PathLike = Union[str, Path]


class ContractError(ValueError):
    """Un artefacto del contrato es inválido, o un valor no cumple su parámetro."""


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _read_json(path: _PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------- ParamSpec


@dataclass(frozen=True)
class ParamSpec:
    """Un parámetro del CONFIG_SCHEMA (una entrada del formulario)."""

    path: str
    type: str
    default: Any
    group: str
    label: str = ""
    help: str = ""
    choices: Optional[tuple] = None
    min: Optional[float] = None
    max: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ParamSpec":
        for k in ("path", "type", "default", "group"):
            if k not in d:
                raise ContractError(f"parámetro sin '{k}': {dict(d)!r}")
        t = d["type"]
        if t not in VALID_TYPES:
            raise ContractError(f"{d['path']}: tipo inválido {t!r} (válidos: {VALID_TYPES})")
        choices = d.get("choices")
        if t == "choice" and not choices:
            raise ContractError(f"{d['path']}: tipo 'choice' sin 'choices'")
        return cls(
            path=d["path"],
            type=t,
            default=d["default"],
            group=d["group"],
            label=d.get("label", ""),
            help=d.get("help", ""),
            choices=tuple(choices) if choices is not None else None,
            min=d.get("min"),
            max=d.get("max"),
        )

    def _check_range(self, v: float) -> None:
        if self.min is not None and v < self.min:
            raise ContractError(f"{self.path}: {v} < min {self.min}")
        if self.max is not None and v > self.max:
            raise ContractError(f"{self.path}: {v} > max {self.max}")

    def validate(self, value: Any, *, coerce: bool = True) -> Any:
        """Valida (y opcionalmente coerciona desde string) un valor para este parámetro.

        Devuelve el valor ya tipado, o lanza `ContractError`. `coerce` permite pasar
        strings (lo que llega del `--set` del CLI o de un formulario web).
        """
        t = self.type
        is_str = isinstance(value, str)

        if t == "bool":
            if isinstance(value, bool):
                return value
            if coerce and is_str:
                low = value.strip().lower()
                if low in ("true", "1", "yes", "on"):
                    return True
                if low in ("false", "0", "no", "off"):
                    return False
            raise ContractError(f"{self.path}: se esperaba bool, no {value!r}")

        if t == "int":
            v = value
            if coerce and is_str:
                try:
                    v = int(value)
                except ValueError:
                    raise ContractError(f"{self.path}: '{value}' no es un entero")
            if not _is_int(v):
                raise ContractError(f"{self.path}: se esperaba int, no {value!r}")
            self._check_range(v)
            return v

        if t == "float":
            v = value
            if coerce and is_str:
                try:
                    v = float(value)
                except ValueError:
                    raise ContractError(f"{self.path}: '{value}' no es un float")
            if not _is_number(v):
                raise ContractError(f"{self.path}: se esperaba float, no {value!r}")
            v = float(v)
            self._check_range(v)
            return v

        if t == "str":
            if is_str:
                return value
            raise ContractError(f"{self.path}: se esperaba str, no {value!r}")

        # t == "choice"
        if value in self.choices:  # type: ignore[operator]
            return value
        if coerce and is_str:
            for c in self.choices:  # type: ignore[union-attr]
                if str(c) == value:
                    return c
        raise ContractError(f"{self.path}: {value!r} no está en choices {self.choices}")


# --------------------------------------------------------------- ConfigSchema


@dataclass(frozen=True)
class ConfigSchema:
    """El CONFIG_SCHEMA: la lista de parámetros modificables del modelo."""

    params: tuple[ParamSpec, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for p in self.params:
            if p.path in seen:
                raise ContractError(f"path duplicado en el schema: {p.path}")
            seen.add(p.path)
        object.__setattr__(self, "_by_path", {p.path: p for p in self.params})

    # --- constructores ---
    @classmethod
    def from_list(cls, items: Sequence[Mapping[str, Any]]) -> "ConfigSchema":
        if not isinstance(items, Sequence):
            raise ContractError("config_schema debe ser una lista de parámetros")
        return cls(tuple(ParamSpec.from_dict(d) for d in items))

    @classmethod
    def from_json(cls, text: str) -> "ConfigSchema":
        return cls.from_list(json.loads(text))

    @classmethod
    def load(cls, path: _PathLike) -> "ConfigSchema":
        return cls.from_list(_read_json(path))

    # --- consulta ---
    def __len__(self) -> int:
        return len(self.params)

    def __iter__(self) -> Iterator[ParamSpec]:
        return iter(self.params)

    def __contains__(self, path: object) -> bool:
        return path in self._by_path  # type: ignore[attr-defined]

    def get(self, path: str) -> ParamSpec:
        try:
            return self._by_path[path]  # type: ignore[attr-defined]
        except KeyError:
            raise ContractError(f"parámetro desconocido: {path}")

    def paths(self) -> list[str]:
        return [p.path for p in self.params]

    def groups(self) -> list[str]:
        """Grupos en orden de primera aparición (arch, optim, schedule, loss, data)."""
        out: list[str] = []
        for p in self.params:
            if p.group not in out:
                out.append(p.group)
        return out

    def by_group(self, group: str) -> list[ParamSpec]:
        return [p for p in self.params if p.group == group]

    def arch_params(self) -> list[ParamSpec]:
        return self.by_group("arch")

    def defaults(self) -> dict[str, Any]:
        """Mapa plano path -> default."""
        return {p.path: p.default for p in self.params}

    # --- validación de overrides ---
    def validate_override(self, path: str, value: Any, *, coerce: bool = True) -> Any:
        """Valida un override (path debe estar en el schema)."""
        return self.get(path).validate(value, coerce=coerce)

    def validate_overrides(self, overrides: Mapping[str, Any], *, coerce: bool = True) -> dict[str, Any]:
        """Valida varios overrides a la vez; agrupa TODOS los errores."""
        out: dict[str, Any] = {}
        errors: list[str] = []
        for path, value in overrides.items():
            try:
                out[path] = self.validate_override(path, value, coerce=coerce)
            except ContractError as e:
                errors.append(str(e))
        if errors:
            raise ContractError("overrides inválidos -> " + "; ".join(errors))
        return out


# ---------------------------------------------------------------- MetricsSpec


@dataclass(frozen=True)
class MetricsSpec:
    """El METRICS_SPEC: etiqueta/orden/sentido de las métricas escalares."""

    entries: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MetricsSpec":
        if not isinstance(d, Mapping):
            raise ContractError("metrics_spec debe ser un objeto metric -> spec")
        return cls(dict(d))

    @classmethod
    def from_json(cls, text: str) -> "MetricsSpec":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: _PathLike) -> "MetricsSpec":
        return cls.from_dict(_read_json(path))

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, metric: object) -> bool:
        return metric in self.entries

    def get(self, metric: str) -> Mapping[str, Any]:
        return self.entries[metric]

    def ordered(self) -> list[tuple[str, Mapping[str, Any]]]:
        """(metric, spec) ordenados por `order` (y nombre como desempate)."""
        return sorted(self.entries.items(), key=lambda kv: (kv[1].get("order", 1e9), kv[0]))


# ------------------------------------------------------------------ ModelInfo


@dataclass(frozen=True)
class ModelInfo:
    """El MODEL_INFO: identidad/forma del modelo (acceso por atributo + `raw`)."""

    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModelInfo":
        if not isinstance(d, Mapping) or "name" not in d:
            raise ContractError("model_info debe ser un objeto con al menos 'name'")
        return cls(dict(d))

    @classmethod
    def from_json(cls, text: str) -> "ModelInfo":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: _PathLike) -> "ModelInfo":
        return cls.from_dict(_read_json(path))

    @property
    def name(self) -> str:
        return self.raw["name"]

    @property
    def version(self) -> Optional[str]:
        return self.raw.get("version")

    @property
    def input(self) -> Optional[tuple]:
        v = self.raw.get("input")
        return tuple(v) if v is not None else None

    @property
    def img_size(self) -> Optional[tuple]:
        v = self.raw.get("img_size")
        return tuple(v) if v is not None else None

    @property
    def num_rows(self) -> Optional[int]:
        return self.raw.get("num_rows")

    @property
    def max_lanes(self) -> Optional[int]:
        return self.raw.get("max_lanes")

    @property
    def params_m(self) -> Optional[float]:
        return self.raw.get("params_m")

    @property
    def common_format(self) -> Optional[str]:
        return self.raw.get("common_format")


# --------------------------------------------------------------- ModelContract


@dataclass(frozen=True)
class ModelContract:
    """Los tres artefactos juntos (lo que el dashboard lee de `_model/<name>@<sha>/`)."""

    schema: ConfigSchema
    metrics: MetricsSpec
    info: ModelInfo

    @classmethod
    def load_dir(cls, directory: _PathLike) -> "ModelContract":
        d = Path(directory)
        return cls(
            schema=ConfigSchema.load(d / CONFIG_SCHEMA_FILE),
            metrics=MetricsSpec.load(d / METRICS_SPEC_FILE),
            info=ModelInfo.load(d / MODEL_INFO_FILE),
        )


# --------------------------------------------------------------------------- CLI


def _default_fixture_dir() -> Path:
    return Path(__file__).parents[2] / "tests" / "fixtures" / "_model" / "lanetr@dd2f8ab"


def _main(argv: Optional[list[str]] = None) -> int:
    """Comprobación manual: carga el contrato y lo resume.

        python -m vroad_mlt.contract [directorio_del_modelo]

    Sin argumento usa la fixture de `tests/fixtures/_model/lanetr@dd2f8ab/`.
    """
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    directory = Path(args[0]) if args else _default_fixture_dir()
    try:
        c = ModelContract.load_dir(directory)
    except (OSError, ContractError) as e:
        print(f"ERROR cargando el contrato en {directory}: {e}", file=sys.stderr)
        return 2

    print(f"model: {c.info.name} v{c.info.version}  ({c.info.params_m}M params)")
    print(f"  input={c.info.input}  img_size={c.info.img_size}  max_lanes={c.info.max_lanes}")
    print(f"schema: {len(c.schema)} parámetros en grupos {c.schema.groups()}")
    for g in c.schema.groups():
        print(f"  {g:<9}: {[p.path.split('.', 1)[1] for p in c.schema.by_group(g)]}")
    print(f"arch_hash usa: {list(ARCH_HASH_KEYS)}")
    print(f"métricas (orden): {[m for m, _ in c.metrics.ordered()]}")
    print("--- validación de ejemplo ---")
    for path, val in [("optim.lr", "1e-4"), ("arch.num_queries", "12"), ("optim.lr", "9")]:
        try:
            print(f"  {path} = {val!r:>6} -> {c.schema.validate_override(path, val)}")
        except ContractError as e:
            print(f"  {path} = {val!r:>6} -> RECHAZADO: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
