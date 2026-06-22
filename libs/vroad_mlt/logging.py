"""vroad_mlt.logging — logging legible `[LEVEL] mensaje  clave=valor` (o JSON opcional).

Por defecto emite TEXTO `[INFO] mensaje  clave=valor ...` (en local y en Cloud Run, para
que se lea cómodo). Con `LOG_FORMAT=json` emite una línea JSON por evento
(`severity`/`message`/`time` + campos), que Cloud Logging parsea para filtrar por severidad.

En modo texto, INFO/DEBUG van a stdout y WARNING+ a stderr: así en Cloud Run los errores
salen con severidad ERROR y el resto con INFO.

Además trae `fmt_decimal`: los `loss_*`/`lr` del `train.log` se quieren en DECIMAL,
sin notación científica (`1e-4` -> `0.0001`).

Diseño:
- Solo stdlib; no toca GCP.
- `setup_logging()` una vez al arrancar; `get_logger(name, **contexto)` pega contexto
  (p. ej. `run_id`) que se mezcla con los `extra` de cada evento (no se pisan).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional, TextIO

__all__ = [
    "SEVERITY",
    "JsonFormatter",
    "TextFormatter",
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


class TextFormatter(logging.Formatter):
    """Formatea como `[LEVEL] mensaje  clave=valor ...` (legible en consola)."""

    def format(self, record: logging.LogRecord) -> str:
        out = f"[{record.levelname}] {record.getMessage()}"
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")
        }
        if extras:
            out += "  " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            out += "\n" + self.formatException(record.exc_info)
        return out


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


def _resolve_fmt(fmt: Optional[str]) -> str:
    if fmt:
        return fmt.lower()
    env = os.environ.get("LOG_FORMAT")
    if env:
        return env.lower()
    return "text"  # texto por defecto (también en Cloud Run); JSON solo con LOG_FORMAT=json


def _mark(handler: logging.Handler) -> logging.Handler:
    handler._vroad_handler = True  # type: ignore[attr-defined]
    return handler


def setup_logging(
    level: int = logging.INFO,
    *,
    fmt: Optional[str] = None,
    stream: Optional[TextIO] = None,
    force: bool = False,
) -> logging.Logger:
    """Configura el logging. Idempotente. Llamar una vez al arrancar.

    `fmt`: 'text' (`[LEVEL] ...`, por defecto) o 'json'. Si es None usa `LOG_FORMAT`
    (o 'text'). En texto sin `stream`, INFO/DEBUG -> stdout y WARNING+ -> stderr.
    `force=True` reemplaza handlers (tests).
    """
    resolved = _resolve_fmt(fmt)
    root = logging.getLogger()
    if force:
        for h in list(root.handlers):
            root.removeHandler(h)

    if not any(getattr(h, "_vroad_handler", False) for h in root.handlers):
        if resolved == "json":
            h = logging.StreamHandler(stream or sys.stdout)
            h.setFormatter(JsonFormatter())
            root.addHandler(_mark(h))
        elif stream is not None:
            h = logging.StreamHandler(stream)
            h.setFormatter(TextFormatter())
            root.addHandler(_mark(h))
        else:
            formatter = TextFormatter()
            out = logging.StreamHandler(sys.stdout)
            out.setFormatter(formatter)
            out.addFilter(lambda r: r.levelno < logging.WARNING)
            root.addHandler(_mark(out))
            err = logging.StreamHandler(sys.stderr)
            err.setFormatter(formatter)
            err.setLevel(logging.WARNING)
            root.addHandler(_mark(err))

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
    log.info("training started", extra={"epoch": 0, "lr": 3e-4})
    log.bind(epoch=1).info("epoch finished", extra={"loss_total": 0.1234, "f1_global": 0.76})
    log.warning("high GPU memory", extra={"gpu_mem_mb": 21500})
    try:
        1 / 0
    except ZeroDivisionError:
        log.error("example failure", exc_info=True)

    print("---- fmt_decimal ----", file=sys.stderr)
    for v in (1e-4, 3e-4, 1e-6, 0.1234567, 1.0, 0.0):
        print(f"  {v!r:>12} -> {fmt_decimal(v)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
