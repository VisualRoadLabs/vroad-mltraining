"""trainer.submit — lanza un entrenamiento en Vertex AI (Custom Job) A MANO.

Es lo que hará el workflow `wf-dev-train-single` (todavía sin el lock de GPU): construye el worker
pool spec (imagen del trainer + args + env), crea el `CustomJob` y lo manda a Vertex. Sirve para
PROBAR el trainer en la nube antes de tener el workflow.

El env del job = el `.env` local (proyecto/buckets/...) + `RUN_ID` + `LANETR_SHA` (en Vertex no hay
`.env` en la imagen). Los args (study/variant/manifest/--set/--standalone/--max-*) se reenvían al
entrypoint `python -m trainer.run`.

    python -m trainer.submit --tag <git-sha> --study smoke --variant t1 --manifest culane@1 \
        --standalone --max-epochs 1 --max-train-steps 20 --max-eval-batches 5 [--tier iter|final] [--dry-run]

`aiplatform` se importa de forma perezosa y es INYECTABLE → la construcción del job se testea sin GCP.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from vroad_mlt import naming
from vroad_mlt.config import parse_env_file

__all__ = ["trainer_args", "worker_pool_spec", "submit", "main"]


def trainer_args(
    *, study: str, variant: str, manifest: str, run_kind: str, set_items: Sequence[str],
    standalone: bool, max_epochs: Optional[int], max_train_steps: Optional[int],
    max_eval_batches: Optional[int],
) -> list[str]:
    """Args del contenedor (se anexan al entrypoint `python -m trainer.run`)."""
    args = ["--study", study, "--variant", variant, "--manifest", manifest, "--run-kind", run_kind]
    for item in set_items:
        args += ["--set", item]
    if standalone:
        args.append("--standalone")
    for flag, value in (("--max-epochs", max_epochs), ("--max-train-steps", max_train_steps),
                        ("--max-eval-batches", max_eval_batches)):
        if value is not None:
            args += [flag, str(value)]
    return args


def worker_pool_spec(
    image: str, args: Sequence[str], env: dict, *, machine: str, accelerator: str, accelerator_count: int = 1
) -> list[dict]:
    """El `worker_pool_specs` del Custom Job: 1 réplica, 1 GPU, imagen + args + env."""
    return [{
        "machine_spec": {"machine_type": machine, "accelerator_type": accelerator,
                         "accelerator_count": accelerator_count},
        "replica_count": 1,
        "container_spec": {"image_uri": image, "args": list(args),
                           "env": [{"name": k, "value": str(v)} for k, v in env.items()]},
    }]


def submit(
    settings: Any,
    *,
    tag: str,
    study: str,
    variant: str,
    manifest: str,
    set_items: Sequence[str] = (),
    run_kind: str = "scratch",
    standalone: bool = False,
    max_epochs: Optional[int] = None,
    max_train_steps: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    tier: str = "iter",
    env_file: Optional[str] = ".env",
    experiment: Optional[str] = None,
    tensorboard: bool = False,
    aiplatform: Any = None,
    run_id: Optional[str] = None,
    dry_run: bool = False,
    log: Any = None,
) -> dict:
    """Crea (o describe, con `dry_run`) el Vertex Custom Job del trainer. Devuelve un resumen."""
    run_id = run_id or naming.run_id(study, variant, naming.utc_timestamp())
    image = settings.image_uri("trainer", tag)
    machine = settings.machine_iter if tier == "iter" else settings.machine_final
    accelerator = settings.accel_iter if tier == "iter" else settings.accel_final

    env = dict(parse_env_file(env_file)) if env_file and Path(env_file).exists() else {}
    env["RUN_ID"] = run_id
    env["LANETR_SHA"] = settings.lanetr_sha
    if tensorboard:
        env["TENSORBOARD_LOG_DIR"] = f"gs://{settings.bucket_vertex_staging}/tensorboard/{run_id}/"
    if experiment:
        env["VERTEX_EXPERIMENT"] = experiment

    args = trainer_args(study=study, variant=variant, manifest=manifest, run_kind=run_kind,
                        set_items=set_items, standalone=standalone, max_epochs=max_epochs,
                        max_train_steps=max_train_steps, max_eval_batches=max_eval_batches)
    specs = worker_pool_spec(image, args, env, machine=machine, accelerator=accelerator)
    base_output = f"gs://{settings.bucket_vertex_staging}/jobs/{run_id}/"
    summary = {"run_id": run_id, "image": image, "machine": machine, "accelerator": accelerator,
               "service_account": settings.sa_trainer, "base_output_dir": base_output, "args": args}

    if dry_run:
        summary["worker_pool_specs"] = specs
        return summary

    mod = aiplatform if aiplatform is not None else _import_aiplatform()
    mod.init(project=settings.project_training, location=settings.region,
             staging_bucket=f"gs://{settings.bucket_vertex_staging}")
    job = mod.CustomJob(display_name=run_id, worker_pool_specs=specs, base_output_dir=base_output)
    job.run(service_account=settings.sa_trainer, sync=False)
    if log:
        log.info("vertex job submitted", extra={"run_id": run_id, "image": image})
    summary["submitted"] = True
    return summary


def _import_aiplatform() -> Any:
    from google.cloud import aiplatform  # noqa: PLC0415

    return aiplatform


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Lanza el trainer en Vertex AI (Custom Job).")
    ap.add_argument("--tag", required=True, help="tag de la imagen trainer en Artifact Registry (git sha)")
    ap.add_argument("--study", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--manifest", required=True, help="'<id>@<version>'")
    ap.add_argument("--run-kind", default="scratch", choices=["scratch", "finetune", "eval"])
    ap.add_argument("--set", action="append", default=[], dest="set", help="override 'a.b=c' (repetible)")
    ap.add_argument("--tier", default="iter", choices=["iter", "final"], help="iter=L4 | final=A100")
    ap.add_argument("--standalone", action="store_true", help="el trainer gestiona su fila de tbl_experiments")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--max-train-steps", type=int, default=None)
    ap.add_argument("--max-eval-batches", type=int, default=None)
    ap.add_argument("--experiment", default=None, help="Vertex Experiment (por defecto: ninguno)")
    ap.add_argument("--tensorboard", action="store_true", help="añade TENSORBOARD_LOG_DIR")
    ap.add_argument("--env-file", default=".env", help="env del job (por defecto el .env local)")
    ap.add_argument("--dry-run", action="store_true", help="solo imprime el spec del job")
    return ap.parse_args(sys.argv[1:] if argv is None else argv)


def main(argv: Optional[list[str]] = None) -> int:
    import json

    from vroad_mlt.config import get_settings
    from vroad_mlt.logging import get_logger, setup_logging

    setup_logging()
    args = parse_args(argv)
    settings = get_settings()
    summary = submit(
        settings, tag=args.tag, study=args.study, variant=args.variant, manifest=args.manifest,
        set_items=args.set, run_kind=args.run_kind, standalone=args.standalone, tier=args.tier,
        max_epochs=args.max_epochs, max_train_steps=args.max_train_steps,
        max_eval_batches=args.max_eval_batches, experiment=args.experiment, tensorboard=args.tensorboard,
        env_file=args.env_file, dry_run=args.dry_run, log=get_logger("trainer.submit"),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
