"""trainer.config — config EFECTIVA del entrenamiento.

La base congelada es de `lanetr` (`lanetr.contract.spec.DEFAULT_CONFIG`, sin torch). El trainer
la fusiona con los overrides `--set` (validados contra el `CONFIG_SCHEMA` del modelo) y calcula
el `arch_hash` de la arquitectura efectiva (para filtrar padres compatibles en fine-tuning).

La lógica de fusión/validación/hash es pura (reusa `vroad_mlt.config_resolve` + `contract`); por
eso `resolve()` recibe la `base` por argumento y se puede testear sin `lanetr`. En runtime,
`default_config()` la importa de `lanetr`.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from vroad_mlt import config_resolve
from vroad_mlt.contract import ARCH_HASH_KEYS, ConfigSchema

__all__ = ["default_config", "resolve", "arch_values"]


def default_config() -> dict:
    """Copia profunda del `DEFAULT_CONFIG` de lanetr (import perezoso: necesita lanetr instalado)."""
    from lanetr.contract.spec import DEFAULT_CONFIG  # noqa: PLC0415 - perezoso a propósito

    return copy.deepcopy(DEFAULT_CONFIG)


def _dig(cfg: Mapping[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def arch_values(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Los 3 valores `arch` que determinan el `arch_hash`, leídos de una config efectiva."""
    return {key: _dig(cfg, key) for key in ARCH_HASH_KEYS}


def resolve(
    base: Mapping[str, Any], schema: ConfigSchema, set_items: Sequence[str]
) -> tuple[dict, str]:
    """Config efectiva = `base` (DEFAULT_CONFIG) + overrides `--set` validados; y su `arch_hash`.

    Devuelve `(cfg_efectiva, arch_hash)`. Lanza `ContractError`/`ConfigResolveError` si un
    `--set` no cumple el schema o está mal formado.
    """
    resolved = config_resolve.resolve_overrides(schema, list(set_items))
    cfg = config_resolve.deep_merge(base, resolved.nested)
    return cfg, config_resolve.arch_hash(arch_values(cfg))
