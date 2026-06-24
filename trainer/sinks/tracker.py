"""trainer.sinks.tracker — telemetría a Vertex AI Experiments (+ TensorBoard).

Un experimento por `study`, un run por entrenamiento. Es telemetría SECUNDARIA: la fuente de verdad
de las curvas es `tbl_train_metrics` (BQ). Por eso el tracker NUNCA debe romper el entrenamiento → si
aiplatform no está o falla, cae a no-op. `aiplatform` se inyecta (testeable sin GCP).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

__all__ = ["NoopTracker", "VertexTracker", "make_tracker"]


def _as_param(value: Any) -> Any:
    """Vertex `log_params` admite escalares; lo demás (listas/dicts) se pasa como str."""
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def _as_float(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {str(name): float(value) for name, value in metrics.items()}


class NoopTracker:
    """Tracker que no hace nada (local, tests, o si Vertex no está disponible)."""

    def log_params(self, params: Mapping[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Mapping[str, Any], *, step: Optional[int] = None) -> None:
        pass

    def log_summary(self, metrics: Mapping[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


class VertexTracker:
    """Adaptador fino sobre Vertex AI Experiments (+ TensorBoard).

    `backend` es el módulo `aiplatform` (INYECTABLE → testeable sin GCP). Hace `init` + `start_run`
    al construirse (`resume=True` para que un REINTENTO del job continúe el mismo run). Las métricas
    con `step` van como SERIE TEMPORAL (las pinta TensorBoard); las de resumen, como escalares.
    """

    def __init__(self, *, project: str, location: str, experiment: str, run_name: str,
                 tensorboard: Optional[str] = None, backend: Any = None) -> None:
        self._backend = backend if backend is not None else _import_aiplatform()
        self._backend.init(project=project, location=location, experiment=experiment,
                           experiment_tensorboard=tensorboard)
        self._backend.start_run(run_name, resume=True)

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._backend.log_params({str(k): _as_param(v) for k, v in params.items()})

    def log_metrics(self, metrics: Mapping[str, Any], *, step: Optional[int] = None) -> None:
        m = _as_float(metrics)
        if not m:
            return
        if step is None:
            self._backend.log_metrics(m)
        else:
            self._backend.log_time_series_metrics(m, step=int(step))

    def log_summary(self, metrics: Mapping[str, Any]) -> None:
        m = _as_float(metrics)
        if m:
            self._backend.log_metrics(m)

    def close(self) -> None:
        self._backend.end_run()


def _import_aiplatform() -> Any:
    from google.cloud import aiplatform  # noqa: PLC0415 - perezoso (no romper imports sin GCP)

    return aiplatform


def make_tracker(
    *,
    project: Optional[str] = None,
    location: Optional[str] = None,
    experiment: Optional[str] = None,
    run_name: Optional[str] = None,
    tensorboard: Optional[str] = None,
    backend: Any = None,
    enabled: bool = True,
    log: Any = None,
) -> Any:
    """`VertexTracker` si está habilitado y se puede crear; si no, `NoopTracker` (no rompe el run)."""
    if not enabled or not (project and experiment and run_name):
        return NoopTracker()
    try:
        return VertexTracker(project=project, location=location, experiment=experiment,
                             run_name=run_name, tensorboard=tensorboard, backend=backend)
    except Exception as e:  # noqa: BLE001 - el tracker es secundario; nunca tumba el entrenamiento
        if log:
            log.warning("tracker deshabilitado (Vertex no disponible)", extra={"error": str(e)})
        return NoopTracker()
