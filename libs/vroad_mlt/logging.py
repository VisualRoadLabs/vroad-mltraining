"""vroad_mlt.logging — logging estructurado (JSON a stdout) + formato decimal.

Todos los jobs/servicios loguean igual: una línea JSON por evento en stdout, que
Cloud Run y Vertex recogen y mandan a Cloud Logging. Cloud Logging reconoce los
campos especiales `severity`, `message` y `time`; cualquier otro campo (p. ej.
`run_id`, `epoch`, `step`) cae en `jsonPayload` y sirve para filtrar.

Además trae `fmt_decimal`: los `loss_*`/`lr` del `train.log` se quieren en DECIMAL,
sin notación científica (p. ej. `1e-4` -> `0.0001`).

Diseño:
- Lógica pura (solo stdlib `logging`/`json`); no toca GCP.
- `setup_logging()` una vez en el arranque; `get_logger(name, **contexto)` devuelve
  un logger con contexto pegado (p. ej. `run_id`) que se mezcla en cada evento.
- El contexto y los `extra` por llamada se combinan (no se pisan).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional, TextIO

__all__ = [
    "SEVERITY",
    "JsonFormatter",
    "setup_logging",
    "get_logger",
    "fmt_decimal",
]

# Niveles stdlib -> severidades de Cloud Logging.
SEVERITY = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
}

# Atributos estándar de un LogRecord: todo lo demás que aparezca en el record es
# contexto/extra del usuario y se vuelca al JSON.
_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Formatea cada LogRecord como una línea JSON apta para Cloud Logging."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc)
        payload: dict[str, Any] = {
            "severity": SEVERITY.get(record.levelno, record.levelname),
            "message": record.getMessage(),
            "time": ts.isoformat().replace("+00:00", "Z"),
            "logger": record.name,
        }
        for key, val in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = val
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        # default=str: nunca peta por un valor no serializable.
        return json.dumps(payload, ensure_ascii=False, default=str)


class _ContextAdapter(logging.LoggerAdapter):
    """Pega un contexto fijo (p. ej. `run_id`) y lo mezcla con los `extra`."""

    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        extra = dict(self.extra or {})
        extra.update(kwargs.get("extra") or {})
        kwargs["extra"] = extra
        return msg, kwargs

    def bind(self, **more: Any) -> "_ContextAdapter":
        """Devuelve un logger nuevo con contexto adicional (no muta el actual)."""
        return _ContextAdapter(self.logger, {**(self.extra or {}), **more})


def setup_logging(
    level: int = logging.INFO,
    *,
    stream: Optional[TextIO] = None,
    force: bool = False,
) -> logging.Logger:
    """Configura el logging estructurado a stdout. Idempotente.

    Llamar una vez al arrancar el proceso. `force=True` reemplaza handlers
    (útil en tests). `stream` permite redirigir (por defecto stdout).
    """
    root = logging.getLogger()
    if force:
        for h in list(root.handlers):
            root.removeHandler(h)
    if not any(getattr(h, "_vroad_json", False) for h in root.handlers):
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(JsonFormatter())
        handler._vroad_json = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(level)
    return root


def get_logger(name: str = "vroad_mlt", **context: Any) -> _ContextAdapter:
    """Devuelve un logger con contexto pegado.

    El `context` (p. ej. `run_id="..."`) se incluye en cada evento. Las claves
    NO pueden chocar con atributos reservados de LogRecord (`message`, `name`...).
    """
    return _ContextAdapter(logging.getLogger(name), dict(context))


def fmt_decimal(x: float, places: int = 6, *, trim: bool = True) -> str:
    """Formatea un float en DECIMAL, sin notación científica.

    `1e-4` -> `'0.0001'`. Con `trim` (por defecto) quita ceros finales
    sobrantes manteniendo al menos un decimal (`3e-4` -> `'0.0003'`, `1.0` -> `'1.0'`).
    """
    s = f"{float(x):.{places}f}"
    if trim and "." in s:
        s = s.rstrip("0").rstrip(".")
        if "." not in s:
            s += ".0"
    return s


# --------------------------------------------------------------------------- CLI


def _main(argv: Optional[list[str]] = None) -> int:
    """Comprobación manual: emite unos eventos de ejemplo y demuestra fmt_decimal.

        python -m vroad_mlt.logging
    """
    setup_logging(force=True)
    log = get_logger("vroad_mlt.demo", run_id="lr-sweep__lr1e-4__20260620T101500Z")
    log.info("entrenamiento iniciado", extra={"epoch": 0, "lr": 3e-4})
    log.bind(epoch=1).info("época terminada", extra={"loss_total": 0.1234, "f1_global": 0.76})
    log.warning("memoria GPU alta", extra={"gpu_mem_mb": 21500})
    try:
        1 / 0
    except ZeroDivisionError:
        log.error("fallo de ejemplo", exc_info=True)

    print("---- fmt_decimal ----", file=sys.stderr)
    for v in (1e-4, 3e-4, 1e-6, 0.1234567, 1.0, 0.0):
        print(f"  {v!r:>12} -> {fmt_decimal(v)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
