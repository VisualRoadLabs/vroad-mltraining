"""trainer.sinks.workdir — artefactos HUMANOS del workdir (bkt-dev-training-workdirs).

Logs legibles (`train/eval/gpu.log`), `results.json`, `config.yaml`, `manifest.lock.json` y los
checkpoints best/last. Los `loss_*` van en DECIMAL (sin notación científica) vía `fmt_decimal`.
`gcs` se inyecta (testeable con un FakeGcs).
"""

from __future__ import annotations

import io
from datetime import timedelta
from typing import Any, Mapping, Optional

from vroad_mlt.logging import fmt_decimal
from vroad_mlt.naming import Workdir, gs_uri

__all__ = ["format_train_line", "format_eval_line", "format_gpu_line", "WorkdirSink"]


def _fmt_eta(seconds: float) -> str:
    """Segundos -> `H:MM:SS` (legible en el log)."""
    return str(timedelta(seconds=int(seconds)))


def format_train_line(
    epoch: int, step: int, metrics: Mapping[str, Any], *, eta_s: Optional[float] = None
) -> str:
    """Línea de `train.log`: época/iteración + `loss_*`/lr/grad_norm (DECIMAL) + ETA opcional."""
    parts = [f"{name}={fmt_decimal(value)}" for name, value in metrics.items()]
    if eta_s is not None:
        parts.append(f"eta={_fmt_eta(eta_s)}")
    return f"[epoch {int(epoch):03d} step {int(step):06d}] " + " ".join(parts)


def format_eval_line(epoch: int, results: Mapping[str, Any]) -> str:
    """Línea de `eval.log`: F1 global + F1 por categoría (de `results.json`)."""
    parts = [f"f1/global={fmt_decimal(results['f1_global'])}"]
    parts += [f"{cat}={fmt_decimal(v)}" for cat, v in results.get("metrics_by_category", {}).items()]
    return f"[epoch {int(epoch):03d}] " + " ".join(parts)


def format_gpu_line(epoch: int, *, mem_peak_mb: float, util_mean: float, util_max: float) -> str:
    """Línea de `gpu.log`: memoria pico (MB) + utilización media/máx por época."""
    return (f"[epoch {int(epoch):03d}] mem_peak_mb={int(mem_peak_mb)} "
            f"util_mean={fmt_decimal(util_mean)} util_max={fmt_decimal(util_max)}")


class WorkdirSink:
    """Escribe los artefactos del workdir en GCS: logs, JSONs y checkpoints best/last.

    Los logs se ACUMULAN en memoria (texto pequeño) y se vuelcan enteros en cada `flush` (sobrescribe
    el objeto). Los checkpoints se serializan con `torch.save` a un buffer y se suben (sin disco
    local: vale en el tmpfs de Vertex). `save_epoch` guarda `last.pth` siempre y `best.pth` cuando
    mejora la F1.
    """

    def __init__(self, gcs: Any, bucket: str, workdir: Workdir) -> None:
        self._gcs = gcs
        self._bucket = bucket
        self._wd = workdir
        self._buffers: dict[str, list[str]] = {"train": [], "eval": [], "gpu": []}
        self._best_f1: Optional[float] = None

    def _uri(self, key: str) -> str:
        return gs_uri(self._bucket, key)

    # ----------------------------------------------------------------- logs
    def log_train(self, epoch: int, step: int, metrics: Mapping[str, Any], *,
                  eta_s: Optional[float] = None, flush: bool = False) -> None:
        self._buffers["train"].append(format_train_line(epoch, step, metrics, eta_s=eta_s))
        if flush:
            self.flush()

    def log_eval(self, epoch: int, results: Mapping[str, Any], *, flush: bool = True) -> None:
        self._buffers["eval"].append(format_eval_line(epoch, results))
        if flush:
            self.flush()

    def log_gpu(self, epoch: int, *, mem_peak_mb: float, util_mean: float, util_max: float,
                flush: bool = True) -> None:
        self._buffers["gpu"].append(format_gpu_line(epoch, mem_peak_mb=mem_peak_mb,
                                                     util_mean=util_mean, util_max=util_max))
        if flush:
            self.flush()

    def flush(self) -> None:
        """Vuelca los 3 logs acumulados a GCS (sobrescribe el objeto con todo el contenido)."""
        for name, key in (("train", self._wd.train_log), ("eval", self._wd.eval_log),
                          ("gpu", self._wd.gpu_log)):
            lines = self._buffers[name]
            if lines:
                self._gcs.write_text(self._uri(key), "\n".join(lines) + "\n")

    # ------------------------------------------------------------- JSON/config
    def write_results(self, results: Mapping[str, Any]) -> str:
        uri = self._uri(self._wd.results_json)
        self._gcs.write_json(uri, dict(results), indent=2)
        return uri

    def write_config(self, config_yaml: str) -> str:
        uri = self._uri(self._wd.config_yaml)
        self._gcs.write_text(uri, config_yaml)
        return uri

    def write_manifest_lock(self, lock: Mapping[str, Any]) -> str:
        uri = self._uri(self._wd.manifest_lock)
        self._gcs.write_json(uri, dict(lock), indent=2)
        return uri

    # ------------------------------------------------------------- checkpoints
    def save_checkpoint(self, state_dict: Any, which: str = "last") -> str:
        """Serializa `state_dict` (torch.save) a un buffer y lo sube a `checkpoints/<which>.pth`."""
        import torch  # noqa: PLC0415 - perezoso (el sink de BQ no necesita torch)

        buf = io.BytesIO()
        torch.save(state_dict, buf)
        uri = self._uri(self._wd.checkpoint(which))
        self._gcs.write_bytes(uri, buf.getvalue())
        return uri

    def save_epoch(self, state_dict: Any, f1: float) -> tuple[Optional[str], str]:
        """Guarda `last.pth` SIEMPRE; `best.pth` solo si `f1` mejora. Devuelve `(best_uri|None, last_uri)`."""
        last_uri = self.save_checkpoint(state_dict, "last")
        best_uri: Optional[str] = None
        if self._best_f1 is None or f1 > self._best_f1:
            self._best_f1 = f1
            best_uri = self.save_checkpoint(state_dict, "best")
        return best_uri, last_uri

    @property
    def best_f1(self) -> Optional[float]:
        return self._best_f1
