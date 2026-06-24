"""Tests del logging estructurado (vroad_mlt.logging)."""

from __future__ import annotations

import io
import json
import logging

import pytest

from vroad_mlt import logging as mlog


@pytest.fixture
def captured():
    """Configura logging a un buffer y devuelve un lector de las líneas JSON."""
    buf = io.StringIO()
    mlog.setup_logging(level=logging.DEBUG, stream=buf, force=True, fmt="json")

    def lines():
        return [json.loads(ln) for ln in buf.getvalue().splitlines() if ln.strip()]

    yield lines
    # limpia handlers para no contaminar otros tests
    mlog.setup_logging(force=True, stream=io.StringIO())


# ---------------------------------------------------------------- formato JSON

def test_basic_event_has_core_fields(captured):
    mlog.get_logger("t").info("hola")
    (rec,) = captured()
    assert rec["severity"] == "INFO"
    assert rec["message"] == "hola"
    assert rec["logger"] == "t"
    assert rec["time"].endswith("Z")


def test_severity_mapping(captured):
    log = mlog.get_logger("t")
    log.warning("w")
    log.error("e")
    sev = [r["severity"] for r in captured()]
    assert sev == ["WARNING", "ERROR"]


def test_printf_style_message(captured):
    mlog.get_logger("t").info("epoch %d lr %s", 3, "3e-4")
    (rec,) = captured()
    assert rec["message"] == "epoch 3 lr 3e-4"


# ------------------------------------------------------------------- contexto

def test_bound_context_appears_in_payload(captured):
    mlog.get_logger("t", run_id="R1").info("x")
    (rec,) = captured()
    assert rec["run_id"] == "R1"


def test_context_and_extra_merge(captured):
    log = mlog.get_logger("t", run_id="R1")
    log.info("x", extra={"epoch": 5})
    (rec,) = captured()
    assert rec["run_id"] == "R1" and rec["epoch"] == 5


def test_bind_adds_context_without_mutating(captured):
    log = mlog.get_logger("t", run_id="R1")
    child = log.bind(epoch=2)
    child.info("a")
    log.info("b")  # el padre NO debe tener epoch
    a, b = captured()
    assert a["run_id"] == "R1" and a["epoch"] == 2
    assert "epoch" not in b


# ----------------------------------------------------------------- excepciones

def test_exception_is_captured(captured):
    log = mlog.get_logger("t")
    try:
        raise ValueError("boom")
    except ValueError:
        log.error("falló", exc_info=True)
    (rec,) = captured()
    assert "ValueError: boom" in rec["exception"]


# ------------------------------------------------------------------ idempotencia

def test_setup_logging_is_idempotent():
    buf = io.StringIO()
    mlog.setup_logging(stream=buf, force=True)
    n_after_first = len(logging.getLogger().handlers)
    mlog.setup_logging(stream=buf)  # sin force: no debe añadir otro handler nuestro
    n_after_second = len(logging.getLogger().handlers)
    assert n_after_first == n_after_second
    mlog.setup_logging(force=True, stream=io.StringIO())  # limpia


def test_non_serializable_extra_does_not_crash(captured):
    mlog.get_logger("t").info("x", extra={"obj": object()})
    (rec,) = captured()
    assert "obj" in rec  # serializado vía default=str


# ------------------------------------------------------------------ fmt_decimal

@pytest.mark.parametrize(
    "value, expected",
    [
        (1e-4, "0.0001"),
        (3e-4, "0.0003"),
        (1e-6, "0.000001"),
        (1.0, "1.0"),
        (0.0, "0.0"),
        (0.1234567, "0.123457"),  # redondeo a 6 decimales
    ],
)
def test_fmt_decimal_no_scientific_notation(value, expected):
    out = mlog.fmt_decimal(value)
    assert "e" not in out.lower()
    assert out == expected


def test_fmt_decimal_places_and_no_trim():
    assert mlog.fmt_decimal(3e-4, places=8) == "0.0003"
    assert mlog.fmt_decimal(0.5, places=4, trim=False) == "0.5000"


# ------------------------------------------------------------- formato texto

def test_text_format_basic():
    buf = io.StringIO()
    mlog.setup_logging(stream=buf, force=True, fmt="text")
    mlog.get_logger("t").info("hello world")
    out = buf.getvalue().strip()
    assert out.startswith("[INFO] hello world")
    mlog.setup_logging(force=True, stream=io.StringIO())


def test_text_format_appends_extras():
    buf = io.StringIO()
    mlog.setup_logging(stream=buf, force=True, fmt="text")
    mlog.get_logger("t", run_id="R1").info("split done", extra={"written": 20})
    out = buf.getvalue().strip()
    assert out.startswith("[INFO] split done")
    assert "run_id=R1" in out and "written=20" in out
    mlog.setup_logging(force=True, stream=io.StringIO())


# ---------------------------------------------------------- selección de formato

def test_resolve_fmt_explicit_and_env(monkeypatch):
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    assert mlog._resolve_fmt(None) == "text"           # texto por defecto (también en Cloud Run)
    assert mlog._resolve_fmt("json") == "json"           # argumento explícito gana
    monkeypatch.setenv("LOG_FORMAT", "json")
    assert mlog._resolve_fmt(None) == "json"             # env LOG_FORMAT
    assert mlog._resolve_fmt("text") == "text"           # el argumento sigue ganando al env


def test_text_routes_warning_to_stderr(capsys):
    mlog.setup_logging(force=True)  # sin stream -> stdout (<WARNING) + stderr (WARNING+)
    log = mlog.get_logger("t")
    log.info("an info line")
    log.warning("a warning line")
    out, err = capsys.readouterr()
    assert "[INFO] an info line" in out and "[WARNING]" not in out
    assert "[WARNING] a warning line" in err and "[INFO]" not in err
    mlog.setup_logging(force=True, stream=io.StringIO())  # limpia
