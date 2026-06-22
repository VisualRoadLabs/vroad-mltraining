"""vroad_mlt.config_resolve — overrides `--set` y `arch_hash` (sin tocar la base).

La config base CONGELADA es de `lanetr` (`lanetr.contract.spec.DEFAULT_CONFIG` y su
`configs/lanetr_culane.yaml`); aquí NO se duplica. Este módulo solo:

- parsea los `--set clave=valor` del lanzamiento -> dict de overrides anidado,
  VALIDÁNDOLOS contra el `CONFIG_SCHEMA` (reusa `contract`, con su coerción de tipos),
- calcula el `arch_hash` ESTABLE a partir de los 3 valores `arch` efectivos
  (default del schema salvo que se pisen): es la clave por la que el dashboard filtra
  los padres compatibles en fine-tuning,
- ofrece un `deep_merge(base, overrides)` genérico (no muta) para previsualizar.

El `config.yaml` efectivo COMPLETO (con las claves congeladas) lo arma el trainer
llamando a `lanetr.contract.build.merge_config(nested)`; eso vive en el trainer.

Lógica pura (solo stdlib): no toca GCP ni importa torch/lanetr.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from vroad_mlt.contract import ARCH_HASH_KEYS, ConfigSchema

__all__ = [
    "ConfigResolveError",
    "ResolvedOverrides",
    "parse_set",
    "to_nested",
    "flatten",
    "deep_merge",
    "resolve_overrides",
    "effective_arch_values",
    "arch_hash",
    "arch_hash_for",
    "ARCH_HASH_LEN",
]

# Longitud (hex) del arch_hash. Las combinaciones de arch son poquísimas
# (num_queries × num_decoder_layers × n_ref_points), así que 16 hex sobran.
ARCH_HASH_LEN = 16


class ConfigResolveError(ValueError):
    """Un `--set` está mal formado o produce un conflicto de estructura."""


@dataclass(frozen=True)
class ResolvedOverrides:
    """Los overrides ya validados y tipados, en sus dos formas."""

    flat: dict[str, Any]    # {"optim.lr": 0.0001, "arch.num_queries": 20}
    nested: dict[str, Any]  # {"optim": {"lr": 0.0001}, "arch": {"num_queries": 20}}


def parse_set(items: Sequence[str]) -> dict[str, str]:
    """`['optim.lr=1e-4', 'arch.num_queries=20']` -> `{'optim.lr':'1e-4', ...}`.

    Parte por el primer `=` (el valor puede contenerlo). Última asignación gana
    si una clave se repite.
    """
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ConfigResolveError(f"--set mal formado (falta '='): {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigResolveError(f"--set sin clave: {item!r}")
        out[key] = value.strip()
    return out


def to_nested(flat: Mapping[str, Any]) -> dict[str, Any]:
    """Convierte claves con puntos en un dict anidado.

    `{'data.aug.hflip_prob': 0.5}` -> `{'data': {'aug': {'hflip_prob': 0.5}}}`.
    Detecta conflictos hoja/sub-bloque (p. ej. `a=1` y `a.b=2`).
    """
    root: dict[str, Any] = {}
    for path, val in flat.items():
        parts = path.split(".")
        node = root
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                if isinstance(node.get(part), dict):
                    raise ConfigResolveError(f"conflicto en {path!r}: {part!r} ya es un sub-bloque")
                node[part] = val
            else:
                nxt = node.get(part)
                if nxt is None:
                    nxt = {}
                    node[part] = nxt
                elif not isinstance(nxt, dict):
                    raise ConfigResolveError(f"conflicto en {path!r}: {part!r} ya tiene un valor")
                node = nxt
    return root


def flatten(nested: Mapping[str, Any], _prefix: str = "") -> dict[str, Any]:
    """Inversa de `to_nested`: dict anidado -> claves con puntos."""
    out: dict[str, Any] = {}
    for key, val in nested.items():
        full = f"{_prefix}{key}"
        if isinstance(val, Mapping):
            out.update(flatten(val, full + "."))
        else:
            out[full] = val
    return out


def deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """`base` fusionado con `overrides` (recursivo), SIN mutar `base`."""
    result = copy.deepcopy(dict(base))

    def _upd(b: dict, u: Mapping) -> None:
        for k, v in u.items():
            if isinstance(v, Mapping) and isinstance(b.get(k), dict):
                _upd(b[k], v)
            else:
                b[k] = copy.deepcopy(v) if isinstance(v, (dict, list)) else v

    _upd(result, overrides)
    return result


def resolve_overrides(
    schema: ConfigSchema, set_items: Sequence[str], *, coerce: bool = True
) -> ResolvedOverrides:
    """Parsea + valida los `--set` contra el schema; devuelve flat y nested tipados.

    Lanza `ConfigResolveError` si un `--set` está mal formado, o `ContractError`
    (de `contract`) si un valor no cumple su parámetro / el path no existe.
    """
    flat_raw = parse_set(set_items)
    flat_typed = schema.validate_overrides(flat_raw, coerce=coerce)
    return ResolvedOverrides(flat=flat_typed, nested=to_nested(flat_typed))


def effective_arch_values(schema: ConfigSchema, overrides_flat: Mapping[str, Any]) -> dict[str, Any]:
    """Los 3 valores `arch` efectivos: default del schema salvo que se pisen."""
    out: dict[str, Any] = {}
    for key in ARCH_HASH_KEYS:
        out[key] = overrides_flat[key] if key in overrides_flat else schema.get(key).default
    return out


def arch_hash(values: Mapping[str, Any]) -> str:
    """Hash estable de un dict de valores `arch` (orden de claves irrelevante)."""
    canonical = json.dumps({k: values[k] for k in sorted(values)}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:ARCH_HASH_LEN]


def arch_hash_for(schema: ConfigSchema, overrides_flat: Mapping[str, Any]) -> str:
    """`arch_hash` de la arch EFECTIVA (defaults incluidos) dados unos overrides."""
    return arch_hash(effective_arch_values(schema, overrides_flat))


# --------------------------------------------------------------------------- CLI


def _default_schema_path():
    from pathlib import Path

    return (
        Path(__file__).parents[2]
        / "tests" / "fixtures" / "_model" / "lanetr@dd2f8ab" / "config_schema.json"
    )


def _main(argv: list[str] | None = None) -> int:
    """Comprobación manual: resuelve unos `--set` y calcula el arch_hash.

        python -m vroad_mlt.config_resolve [--schema PATH] [--set k=v ...]

    Sin `--schema` usa la fixture del modelo real.
    """
    import sys
    from pathlib import Path

    from vroad_mlt.contract import ContractError

    args = list(sys.argv[1:] if argv is None else argv)
    schema_path = _default_schema_path()
    set_items: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--schema":
            schema_path = Path(args[i + 1]); i += 2
        elif args[i] == "--set":
            set_items.append(args[i + 1]); i += 2
        else:
            print(f"argumento no reconocido: {args[i]}", file=sys.stderr); return 2

    schema = ConfigSchema.load(schema_path)
    try:
        resolved = resolve_overrides(schema, set_items)
    except (ConfigResolveError, ContractError) as e:
        print(f"RECHAZADO: {e}", file=sys.stderr)
        return 2

    print(f"--set recibidos : {set_items}")
    print(f"overrides flat  : {resolved.flat}")
    print(f"overrides nested: {json.dumps(resolved.nested, ensure_ascii=False)}")
    print(f"arch efectiva   : {effective_arch_values(schema, resolved.flat)}")
    print(f"arch_hash        : {arch_hash_for(schema, resolved.flat)}")
    print(f"arch_hash (base) : {arch_hash_for(schema, {})}   <- sin overrides (defaults)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
