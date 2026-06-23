"""trainer.run — orquestador del entrenamiento (el pegamento de T1–T5).

Punto de entrada del Vertex Custom Job: `python -m trainer.run`. Junta config (T1), datos (T2),
loop+receta (T3), evaluación (T4) y los sinks (T5):

  config efectiva ← DEFAULT_CONFIG + --set (validados con el CONFIG_SCHEMA del modelo)
  dataset         ← manifiesto → shards de train; benchmark CULane fijo para evaluar
  por época       → train_epoch (loop) + run_eval (F1 sobre el benchmark, con EMA)
  sinks           → train/eval/gpu.log + tbl_train_metrics + Experiments/TensorBoard
                    + checkpoints best/last (best = mejor F1)
  al final        → results.json + fila final en tbl_experiments

Los ESTADOS (`RUNNING`/`SUCCEEDED`) los pone el workflow, no esto: el trainer solo escribe métricas
y las columnas finales (MERGE parcial). Las piezas pesadas (lanetr/torch) se INYECTAN como costuras
→ la orquestación se testea con fakes (sin lanetr/GPU/GCP). En runtime apuntan a las reales.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from vroad_mlt import naming
from vroad_mlt.contract import ConfigSchema
from vroad_mlt.manifest import Manifest, ManifestLock

from trainer import config, data, eval as ev, loop, sinks

__all__ = ["RunSpec", "run_training", "resolve_shards", "step_event", "f1_event", "experiment_fields", "main"]


# ----------------------------------------------------------------- entrada (spec)


@dataclass
class RunSpec:
    """Todo lo que define una corrida (de los args + env + settings)."""

    run_id: str
    manifest_ref: str          # "culane@1"
    run_kind: str              # scratch | finetune | eval
    set_items: list[str]       # ["optim.lr=1e-4", ...]
    datasets_bucket: str
    workdir_bucket: str
    project: str
    region: str
    lanetr_sha: str
    experiment: Optional[str] = None
    tensorboard: Optional[str] = None
    parent_ckpt: Optional[str] = None
    eval_every: int = 1
    log_every: int = 50
    # Smoke/debug: limitar épocas, pasos de train y batches de eval (None = completo).
    max_epochs: Optional[int] = None
    max_train_steps: Optional[int] = None
    max_eval_batches: Optional[int] = None
    # standalone: el trainer gestiona su propia fila de tbl_experiments (lo que hace el workflow).
    standalone: bool = False
    # Override del nº de workers del dataloader (None = el de la config; 0 = sin multiprocessing).
    num_workers: Optional[int] = None


# ----------------------------------------------------------------- helpers puros


def split_manifest_ref(ref: str) -> tuple[str, str]:
    """`'culane@1'` -> `('culane', '1')`."""
    if "@" not in ref:
        raise ValueError(f"manifest debe ser '<id>@<version>': {ref!r}")
    mid, ver = ref.rsplit("@", 1)
    if not mid or not ver:
        raise ValueError(f"manifest mal formado: {ref!r}")
    return mid, ver


def resolve_shards(gcs: Any, bucket: str, manifest: Manifest, split: str) -> list[str]:
    """URIs de los shards de un split, sumando todas las `sources` del manifiesto (mezclas)."""
    uris: list[str] = []
    for src in manifest.sources:
        src_split = src.split_map.get(split)
        if not src_split:
            continue
        prefix = naming.gs_uri(bucket, naming.shards_prefix(src.dataset, src.version), f"{src_split}-")
        uris += sorted(gcs.list_uris(prefix))
    return uris


def to_yaml(cfg: dict) -> str:
    """Serializa la config efectiva a YAML (para `config.yaml`)."""
    import yaml  # noqa: PLC0415 - dep del trainer

    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)


def step_event(info: dict) -> dict:
    """Métricas de un paso de train: `loss/<term>` + lr + grad_norm (para logs/BQ/tracker)."""
    ev = {f"loss/{name}": value for name, value in info["losses"].items()}
    ev["lr"] = info["lr"]
    ev["grad_norm"] = info["grad_norm"]
    return ev


def f1_event(results: dict) -> dict:
    """Métricas de una evaluación: `f1/global` + `f1/<categoria>`."""
    ev = {"f1/global": results["f1_global"]}
    ev.update({f"f1/{cat}": v for cat, v in results.get("metrics_by_category", {}).items()})
    return ev


def experiment_fields(
    spec: RunSpec, manifest: Manifest, results: dict, *,
    arch_hash: str, params_m: float, best_ckpt_uri: str, workdir_uri: str, config_uri: str,
) -> dict:
    """Columnas finales del trainer para `tbl_experiments` (MERGE parcial; sin None)."""
    fields: dict[str, Any] = {
        **manifest.to_experiment_fields(),               # linaje + num_train/val/test
        "arch_hash": arch_hash,
        "params_m": params_m,
        "model_sha": spec.lanetr_sha,
        "f1_global": results["f1_global"],
        "threshold": results.get("threshold"),
        "metrics_by_category_json": results.get("metrics_by_category"),
        "best_ckpt_uri": best_ckpt_uri,
        "workdir_uri": workdir_uri,
        "config_uri": config_uri,
    }
    if spec.parent_ckpt:
        fields["base_ckpt_uri"] = spec.parent_ckpt
    return {k: v for k, v in fields.items() if v is not None}  # omite None (upsert parcial)


def gpu_stats(device: str) -> dict:
    """Memoria pico (MB) + utilización GPU. La util es BEST-EFFORT (pynvml; 0 si no está)."""
    if device != "cuda":
        return {"mem_peak_mb": 0.0, "util_mean": 0.0, "util_max": 0.0}
    import torch  # noqa: PLC0415

    mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    util = _nvml_util_pct()
    return {"mem_peak_mb": mem, "util_mean": util, "util_max": util}


def _nvml_util_pct() -> float:
    try:
        import pynvml  # noqa: PLC0415

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        pynvml.nvmlShutdown()
        return float(util)
    except Exception:  # noqa: BLE001 - util es opcional; nunca rompe el entrenamiento
        return 0.0


def _cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _reset_peak_mem(device: str) -> None:
    if device == "cuda":
        import torch  # noqa: PLC0415

        torch.cuda.reset_peak_memory_stats()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def start_experiment_row(bq: Any, spec: "RunSpec", study: str, variant: str) -> None:
    """Crea/asegura la fila base de `tbl_experiments` (lo que normalmente hace el workflow).

    Solo en `--standalone` (lanzamientos locales sin workflow): pone las columnas REQUIRED
    (study/variant/run_kind/status/started_at) para que el MERGE parcial final tenga fila que tocar.
    """
    bq.merge_upsert("ds_experiments", "tbl_experiments", [{
        "run_id": spec.run_id, "study": study, "variant": variant,
        "run_kind": spec.run_kind, "status": "RUNNING", "started_at": _utcnow(),
    }], ["run_id"], types={"started_at": "TIMESTAMP"})


def load_parent(model: Any, gcs: Any, uri: str, *, strict: bool, log: Any = None) -> None:
    """Carga los pesos de un checkpoint padre (fine-tuning). `strict` exige forma idéntica."""
    import io  # noqa: PLC0415

    import torch  # noqa: PLC0415

    state = torch.load(io.BytesIO(gcs.read_bytes(uri)), map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=strict)  # strict=True aborta si no casa
    if log and not strict:
        log.info("parent cargado (parcial)", extra={
            "missing": len(getattr(result, "missing_keys", [])),
            "unexpected": len(getattr(result, "unexpected_keys", []))})


# ----------------------------------------------------------------- orquestación


def run_training(
    spec: RunSpec,
    *,
    gcs: Any,
    bq: Any,
    default_config: Callable[[], dict] = config.default_config,
    make_loader: Callable = data.make_loader,
    make_eval_loader: Callable = ev.make_eval_loader,
    build_trainables: Callable = loop.build_trainables,
    train_epoch: Callable = loop.train_epoch,
    run_eval: Callable = ev.run_eval,
    make_tracker: Callable = sinks.make_tracker,
    device: Optional[str] = None,
    log: Any = None,
) -> dict:
    """Entrena de punta a punta y devuelve los `results` de la mejor época."""
    study, variant, ts = naming.parse_run_id(spec.run_id)
    wd = naming.Workdir(study, variant, ts)

    # --- 1) config efectiva (T1) ---
    schema_uri = naming.gs_uri(
        spec.datasets_bucket, naming.model_artifacts_prefix("lanetr", spec.lanetr_sha), "config_schema.json"
    )
    schema = ConfigSchema.from_json(gcs.read_text(schema_uri))
    cfg, arch_hash = config.resolve(default_config(), schema, spec.set_items)
    if spec.num_workers is not None:                     # override (smoke/debug; 0 = sin multiprocessing)
        cfg["data"]["num_workers"] = spec.num_workers

    # --- 2) dataset (manifiesto -> shards) ---
    mid, mver = split_manifest_ref(spec.manifest_ref)
    manifest = Manifest.from_dict(gcs.read_json(naming.gs_uri(spec.datasets_bucket, naming.manifest_key(mid, mver))))
    train_shards = resolve_shards(gcs, spec.datasets_bucket, manifest, "train")
    bench_shards = ev.benchmark_shard_uris(gcs, spec.datasets_bucket, "test")
    categories = ev.load_categories(gcs, spec.datasets_bucket)

    # --- 3) sinks (T5) ---
    wd_sink = sinks.WorkdirSink(gcs, spec.workdir_bucket, wd)
    bq_sink = sinks.BqMetricsSink(bq)
    if spec.standalone:                                   # sin workflow: el trainer crea su fila
        start_experiment_row(bq, spec, study, variant)
    tracker = make_tracker(project=spec.project, location=spec.region, experiment=spec.experiment,
                           run_name=spec.run_id, tensorboard=spec.tensorboard, log=log)
    wd_sink.write_config(to_yaml(cfg))
    wd_sink.write_manifest_lock(ManifestLock.from_manifest(manifest, resolved_at=naming.utc_timestamp()).to_dict())

    # --- 4) modelo + loaders (T3/T2/T4) ---
    device = device or ("cuda" if _cuda_available() else "cpu")
    loop.set_backends(device)
    epochs = int(cfg["schedule"]["epochs"])
    if spec.max_epochs:                                   # smoke: recorta el nº de épocas
        epochs = min(epochs, spec.max_epochs)
    batch_size = int(cfg["data"]["batch_size"])
    iters = max(1, math.ceil(manifest.counts.get("train", batch_size) / batch_size))

    model, criterion, optimizer, scheduler, ema, prep = build_trainables(
        cfg, device, iters_per_epoch=iters, epochs=epochs)
    if spec.parent_ckpt:
        load_parent(model, gcs, spec.parent_ckpt, strict=bool(cfg["arch"]["load_strict"]), log=log)

    gcs_factory = data.GcsClientFactory(spec.project)    # pickleable (Windows spawn / forkserver)
    train_loader = make_loader(cfg, "train", train_shards, gcs_factory)
    bench_loader = make_eval_loader(cfg, bench_shards, gcs_factory)
    eval_model = ema.ema if ema is not None else model
    params_m = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)

    tracker.log_params({"manifest": spec.manifest_ref, "run_kind": spec.run_kind, "arch_hash": arch_hash,
                        **{f"set/{i}": s for i, s in enumerate(spec.set_items)}})

    # --- 5) bucle por época ---
    threshold = float(cfg["train"]["eval_conf_thresh"])
    best_results: Optional[dict] = None
    for epoch in range(epochs):
        if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        _reset_peak_mem(device)

        def on_step(info: dict, _epoch: int = epoch) -> None:
            if spec.log_every <= 0 or info["step"] % spec.log_every != 0:
                return
            gstep = _epoch * iters + info["step"]
            ev_step = step_event(info)
            wd_sink.log_train(_epoch, gstep, ev_step)
            bq_sink.log_metrics(spec.run_id, ev_step, step=gstep, epoch=_epoch)
            tracker.log_metrics(ev_step, step=gstep)

        train_iter = (itertools.islice(train_loader, spec.max_train_steps)
                      if spec.max_train_steps else train_loader)
        train_epoch(model, criterion, optimizer, scheduler, train_iter, device,
                    ema=ema, prepare_targets=prep, amp=bool(cfg["train"]["amp"]),
                    channels_last=bool(cfg["train"]["channels_last"]),
                    grad_clip=float(cfg["optim"]["grad_clip"]), on_step=on_step)

        if epoch % max(1, spec.eval_every) == 0 or epoch == epochs - 1:
            eval_iter = (itertools.islice(bench_loader, spec.max_eval_batches)
                         if spec.max_eval_batches else bench_loader)
            results, *_ = run_eval(eval_model, eval_iter, device,
                                   categories_map=categories, threshold=threshold)
            wd_sink.log_eval(epoch, results)
            bq_sink.log_metrics(spec.run_id, f1_event(results), step=epoch * iters + iters, epoch=epoch)
            tracker.log_metrics(f1_event(results), step=epoch)
            best_uri, _ = wd_sink.save_epoch(eval_model.state_dict(), results["f1_global"])
            if best_uri is not None:                     # mejoró -> best.pth y results de la mejor época
                best_results = results

        wd_sink.log_gpu(epoch, **gpu_stats(device))
        wd_sink.flush()

    if best_results is None:                              # por si nunca se evaluó (epochs=0)
        best_results = {"f1_global": 0.0, "threshold": threshold, "metrics_by_category": {}}

    # --- 6) cierre ---
    wd_sink.write_results(best_results)
    fields = experiment_fields(
        spec, manifest, best_results, arch_hash=arch_hash, params_m=params_m,
        best_ckpt_uri=naming.gs_uri(spec.workdir_bucket, wd.checkpoint("best")),
        workdir_uri=wd.uri(spec.workdir_bucket),
        config_uri=naming.gs_uri(spec.workdir_bucket, wd.config_yaml),
    )
    final_types: Optional[dict] = None
    if spec.standalone:                                   # sin workflow: el trainer cierra su fila
        fields["status"] = "SUCCEEDED"
        fields["ended_at"] = _utcnow()
        final_types = {"ended_at": "TIMESTAMP"}
    bq_sink.finalize_experiment(spec.run_id, fields, types=final_types)
    tracker.log_summary({"f1_global": best_results["f1_global"]})
    tracker.close()
    return best_results


# --------------------------------------------------------------------------- CLI


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Entrena un modelo (Vertex Custom Job).")
    ap.add_argument("--study", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--manifest", required=True, help="dataset/mezcla: '<id>@<version>'")
    ap.add_argument("--run-kind", default="scratch", choices=["scratch", "finetune", "eval"])
    ap.add_argument("--set", action="append", default=[], dest="set", help="override 'a.b=c' (repetible)")
    ap.add_argument("--eval-every", type=int, default=1, help="evalúa cada N épocas")
    ap.add_argument("--log-every", type=int, default=50, help="loguea métricas de train cada N pasos")
    ap.add_argument("--max-epochs", type=int, default=None, help="(smoke) recorta el nº de épocas")
    ap.add_argument("--max-train-steps", type=int, default=None, help="(smoke) corta el train a N pasos/época")
    ap.add_argument("--max-eval-batches", type=int, default=None, help="(smoke) corta la eval a N batches")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="(smoke/debug) override de workers del dataloader; 0 = sin multiprocessing")
    ap.add_argument("--standalone", action="store_true",
                    help="sin workflow: el trainer crea/cierra su propia fila de tbl_experiments")
    return ap.parse_args(sys.argv[1:] if argv is None else argv)


def main(argv: Optional[list[str]] = None) -> int:
    from vroad_mlt.bq import BigQuery
    from vroad_mlt.config import get_settings
    from vroad_mlt.gcs import Gcs
    from vroad_mlt.logging import get_logger, setup_logging

    setup_logging()
    log = get_logger("trainer")
    args = parse_args(argv)
    settings = get_settings()
    env = os.environ

    # run_id: lo da Vertex (env RUN_ID); en local se construye de study/variant + timestamp.
    run_id = env.get("RUN_ID") or naming.run_id(args.study, args.variant, naming.utc_timestamp())
    spec = RunSpec(
        run_id=run_id, manifest_ref=args.manifest, run_kind=args.run_kind, set_items=list(args.set),
        datasets_bucket=settings.bucket_datasets, workdir_bucket=settings.bucket_workdirs,
        project=settings.project_training, region=settings.region, lanetr_sha=settings.lanetr_sha,
        experiment=env.get("VERTEX_EXPERIMENT"), tensorboard=env.get("TENSORBOARD_LOG_DIR"),
        parent_ckpt=env.get("PARENT_CKPT") or None,
        eval_every=args.eval_every, log_every=args.log_every,
        max_epochs=args.max_epochs, max_train_steps=args.max_train_steps,
        max_eval_batches=args.max_eval_batches, standalone=args.standalone,
        num_workers=args.num_workers,
    )

    gcs = Gcs.from_settings(settings)
    bq = BigQuery.from_settings(settings)
    log.info("training start", extra={"run_id": run_id, "manifest": spec.manifest_ref,
                                      "run_kind": spec.run_kind, "standalone": spec.standalone})
    try:
        results = run_training(spec, gcs=gcs, bq=bq, log=log)
    except Exception:
        if spec.standalone:  # marca la fila como FAILED (best-effort) para no dejarla en RUNNING
            try:
                bq.merge_upsert("ds_experiments", "tbl_experiments",
                                [{"run_id": run_id, "status": "FAILED", "ended_at": _utcnow()}],
                                ["run_id"], types={"ended_at": "TIMESTAMP"})
            except Exception:  # noqa: BLE001
                pass
        log.exception("training failed")
        raise
    log.info("training done", extra={"run_id": run_id, "f1_global": results["f1_global"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
