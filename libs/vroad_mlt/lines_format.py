"""vroad_mlt.lines_format — el formato común `.lines.json` (el contrato de datos).

Todo carril —GT de cualquier dataset (público o de usuario) y toda predicción del
modelo— se representa con la MISMA estructura. Es lo que hace que la métrica y la
visualización sean agnósticas al modelo.

GT (sin confianza):

    {
      "timestamp": 1781646771,
      "Lines": [
        [{"x": -11, "y": 550}, {"x": 19, "y": 540}, {"x": 769, "y": 290}],
        ...
      ]
    }

Predicción (añade `Scores`, una confianza por carril, mismo orden y longitud):

    {
      "timestamp": 1781646771,
      "Lines": [ [...], [...] ],
      "Scores": [0.98, 0.62]
    }

Reglas y matices:
- `timestamp` — INTEGER, época en segundos (UTC).
- `Lines` — lista de carriles; cada carril es una lista de puntos {"x": int, "y": int}
  en píxeles de la imagen de ORIGEN. `x` puede salirse del marco (`< 0` o `>= ancho`);
  `y` desciende (~cada 10 px). La resolución NO va en el fichero (se conoce por la fuente).
- `Scores` — OPCIONAL, lista de FLOAT en [0, 1] alineada índice-a-índice con `Lines`.
  Ausente en el GT público; presente en las predicciones del modelo.
- NO hay metadata en el fichero (ni `source`, ni `frame_id`, ni modelo, etc.).
- El tope de `MAX_LANES` (<= 4) NO es regla de formato: se aplica al consumir/decodificar,
  por eso aquí no se rechaza un fichero con más carriles.

Este módulo es lógica pura (solo stdlib): parsear, validar y serializar el formato.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

__all__ = [
    "MAX_LANES",
    "MIN_POINTS_PER_LANE",
    "LinesFormatError",
    "Point",
    "LinesFile",
    "parse",
    "validate",
    "to_dict",
    "loads",
    "dumps",
    "load",
    "dump",
    "summary",
]

# Tope de carriles que el consumidor aplica al decodificar. No es
# una regla de validación del formato: aquí se documenta, no se impone.
MAX_LANES = 4
# Un carril es una línea: necesita al menos 2 puntos para poder rasterizarse.
MIN_POINTS_PER_LANE = 2

_ALLOWED_TOP_KEYS = {"timestamp", "Lines", "Scores"}
_ALLOWED_POINT_KEYS = {"x", "y"}


class LinesFormatError(ValueError):
    """El contenido no cumple el formato común `.lines.json`."""


def _is_int(v: Any) -> bool:
    # En Python `bool` es subclase de `int`; un coord. no puede ser True/False.
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


@dataclass(frozen=True)
class Point:
    """Un punto en píxeles de la imagen de origen. `x` puede salirse del marco."""

    x: int
    y: int


@dataclass
class LinesFile:
    """Un fichero `.lines.json` parseado.

    - `timestamp`: época en segundos (UTC).
    - `lines`: lista de carriles (cada carril, lista de `Point`).
    - `scores`: confianza por carril (solo predicciones); `None` en el GT.
    """

    timestamp: int
    lines: list[list[Point]] = field(default_factory=list)
    scores: Optional[list[float]] = None

    @property
    def is_prediction(self) -> bool:
        """True si lleva `Scores` (predicción); False si es GT."""
        return self.scores is not None

    @property
    def num_lanes(self) -> int:
        return len(self.lines)


def parse(obj: Mapping[str, Any]) -> LinesFile:
    """Valida un dict ya cargado de JSON y devuelve un `LinesFile`.

    Lanza `LinesFormatError` con un mensaje claro (y la ruta del campo) si algo
    no cumple el contrato.
    """
    if not isinstance(obj, Mapping):
        raise LinesFormatError(
            f"el contenido raíz debe ser un objeto JSON, no {type(obj).__name__}"
        )

    extra = set(obj.keys()) - _ALLOWED_TOP_KEYS
    if extra:
        raise LinesFormatError(
            "claves de primer nivel no permitidas (el formato no lleva metadata): "
            + ", ".join(sorted(extra))
        )

    # --- timestamp ---
    if "timestamp" not in obj:
        raise LinesFormatError("falta el campo obligatorio 'timestamp'")
    ts = obj["timestamp"]
    if not _is_int(ts):
        raise LinesFormatError(
            f"'timestamp' debe ser un entero (época en segundos), no {type(ts).__name__}"
        )
    if ts < 0:
        raise LinesFormatError(f"'timestamp' no puede ser negativo: {ts}")

    # --- Lines ---
    if "Lines" not in obj:
        raise LinesFormatError("falta el campo obligatorio 'Lines'")
    raw_lines = obj["Lines"]
    if not isinstance(raw_lines, list):
        raise LinesFormatError(
            f"'Lines' debe ser una lista de carriles, no {type(raw_lines).__name__}"
        )

    lines: list[list[Point]] = []
    for i, raw_lane in enumerate(raw_lines):
        if not isinstance(raw_lane, list):
            raise LinesFormatError(
                f"'Lines[{i}]' debe ser una lista de puntos, no {type(raw_lane).__name__}"
            )
        if len(raw_lane) < MIN_POINTS_PER_LANE:
            raise LinesFormatError(
                f"'Lines[{i}]' tiene {len(raw_lane)} punto(s); "
                f"un carril necesita al menos {MIN_POINTS_PER_LANE}"
            )
        lane: list[Point] = []
        for j, raw_pt in enumerate(raw_lane):
            where = f"Lines[{i}][{j}]"
            if not isinstance(raw_pt, Mapping):
                raise LinesFormatError(
                    f"'{where}' debe ser un objeto {{'x','y'}}, no {type(raw_pt).__name__}"
                )
            missing = _ALLOWED_POINT_KEYS - set(raw_pt.keys())
            if missing:
                raise LinesFormatError(
                    f"'{where}' le falta(n) la(s) clave(s): " + ", ".join(sorted(missing))
                )
            extra_pt = set(raw_pt.keys()) - _ALLOWED_POINT_KEYS
            if extra_pt:
                raise LinesFormatError(
                    f"'{where}' tiene clave(s) no permitida(s): " + ", ".join(sorted(extra_pt))
                )
            if not _is_int(raw_pt["x"]):
                raise LinesFormatError(
                    f"'{where}.x' debe ser un entero, no {type(raw_pt['x']).__name__}"
                )
            if not _is_int(raw_pt["y"]):
                raise LinesFormatError(
                    f"'{where}.y' debe ser un entero, no {type(raw_pt['y']).__name__}"
                )
            lane.append(Point(int(raw_pt["x"]), int(raw_pt["y"])))
        lines.append(lane)

    # --- Scores (opcional) ---
    scores: Optional[list[float]] = None
    if obj.get("Scores") is not None:
        raw_scores = obj["Scores"]
        if not isinstance(raw_scores, list):
            raise LinesFormatError(
                f"'Scores' debe ser una lista, no {type(raw_scores).__name__}"
            )
        if len(raw_scores) != len(lines):
            raise LinesFormatError(
                f"'Scores' ({len(raw_scores)}) debe tener la misma longitud que "
                f"'Lines' ({len(lines)}): una confianza por carril"
            )
        scores = []
        for k, s in enumerate(raw_scores):
            if not _is_number(s):
                raise LinesFormatError(
                    f"'Scores[{k}]' debe ser un número, no {type(s).__name__}"
                )
            if not (0.0 <= float(s) <= 1.0):
                raise LinesFormatError(f"'Scores[{k}]' debe estar en [0, 1]: {s}")
            scores.append(float(s))

    return LinesFile(timestamp=ts, lines=lines, scores=scores)


def validate(obj: Mapping[str, Any]) -> None:
    """Como `parse`, pero sin devolver nada: solo valida (lanza si es inválido)."""
    parse(obj)


def to_dict(lf: LinesFile) -> dict[str, Any]:
    """Serializa un `LinesFile` al dict exacto del formato (orden estable).

    Incluye `Scores` solo si está presente (es una predicción).
    """
    out: dict[str, Any] = {
        "timestamp": lf.timestamp,
        "Lines": [[{"x": p.x, "y": p.y} for p in lane] for lane in lf.lines],
    }
    if lf.scores is not None:
        out["Scores"] = list(lf.scores)
    return out


def loads(text: str) -> LinesFile:
    """Parsea un `.lines.json` desde una cadena."""
    return parse(json.loads(text))


def dumps(lf: LinesFile, *, indent: Optional[int] = None) -> str:
    """Serializa a una cadena JSON."""
    return json.dumps(to_dict(lf), indent=indent, ensure_ascii=False)


def load(path: Union[str, Path]) -> LinesFile:
    """Carga y valida un `.lines.json` desde disco."""
    return loads(Path(path).read_text(encoding="utf-8"))


def dump(lf: LinesFile, path: Union[str, Path], *, indent: Optional[int] = None) -> None:
    """Escribe un `LinesFile` a disco en formato común."""
    Path(path).write_text(dumps(lf, indent=indent), encoding="utf-8")


def summary(lf: LinesFile) -> dict[str, Any]:
    """Resumen legible para inspección humana (lo usa el CLI)."""
    all_pts = [p for lane in lf.lines for p in lane]
    xs = [p.x for p in all_pts]
    ys = [p.y for p in all_pts]
    return {
        "kind": "prediction" if lf.is_prediction else "gt",
        "timestamp": lf.timestamp,
        "num_lanes": lf.num_lanes,
        "points_per_lane": [len(lane) for lane in lf.lines],
        "total_points": len(all_pts),
        "x_range": [min(xs), max(xs)] if xs else None,
        "y_range": [min(ys), max(ys)] if ys else None,
        "scores": lf.scores,
    }


# --------------------------------------------------------------------------- CLI


def _main(argv: Optional[Sequence[str]] = None) -> int:
    """Comprobación manual: valida un `.lines.json` y muestra un resumen.

        python -m vroad_mlt.lines_format <ruta.lines.json | ->   [--json]

    Usa `-` para leer de stdin. Sale 0 si es válido, 2 si no.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    if len(args) != 1:
        print(_main.__doc__, file=sys.stderr)
        return 2
    src = args[0]

    try:
        text = sys.stdin.read() if src == "-" else Path(src).read_text(encoding="utf-8")
    except OSError as e:
        print(f"ERROR: no se pudo leer {src!r}: {e}", file=sys.stderr)
        return 2

    try:
        lf = loads(text)
    except json.JSONDecodeError as e:
        print(f"INVÁLIDO: JSON mal formado: {e}", file=sys.stderr)
        return 2
    except LinesFormatError as e:
        print(f"INVÁLIDO: {e}", file=sys.stderr)
        return 2

    if as_json:
        print(dumps(lf, indent=2))
        return 0

    s = summary(lf)
    print(f"OK  ({'predicción' if lf.is_prediction else 'GT'})")
    print(f"  timestamp     : {s['timestamp']}")
    print(f"  carriles      : {s['num_lanes']}  (puntos/carril: {s['points_per_lane']})")
    print(f"  puntos totales: {s['total_points']}")
    print(f"  rango x       : {s['x_range']}")
    print(f"  rango y       : {s['y_range']}")
    if lf.is_prediction:
        print(f"  scores        : {s['scores']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
