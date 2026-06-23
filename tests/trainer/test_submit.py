"""Tests del lanzador de Vertex (trainer.submit).

Sin GCP: `aiplatform` y `Settings` se inyectan (fakes). Verifica los args del contenedor, el worker
pool spec (imagen/máquina/GPU/env) y que `submit` crea el CustomJob con la SA y el output correctos.
"""

from __future__ import annotations

from types import SimpleNamespace

from trainer import submit as sub


def _settings():
    return SimpleNamespace(
        image_uri=lambda name, tag: f"us-central1-docker.pkg.dev/cicd/ml-training/{name}:{tag}",
        region="us-central1", project_training="vr-prj-dev-training-v1",
        machine_iter="g2-standard-12", accel_iter="NVIDIA_L4",
        machine_final="a2-highgpu-1g", accel_final="NVIDIA_TESLA_A100",
        bucket_vertex_staging="bkt-dev-vertex-staging-usc1",
        sa_trainer="sa-mlt-trainer@vr-prj-dev-training-v1.iam.gserviceaccount.com",
        lanetr_sha="dd2f8ab",
    )


class FakeJob:
    def __init__(self, **kw):
        self.kw = kw
        self.ran = None

    def run(self, **kw):
        self.ran = kw


class FakeAip:
    def __init__(self):
        self.inited = None
        self.jobs = []

    def init(self, **kw):
        self.inited = kw

    def CustomJob(self, **kw):
        job = FakeJob(**kw)
        self.jobs.append(job)
        return job


# ----------------------------------------------------------------- trainer_args

def test_trainer_args_forwards_flags():
    args = sub.trainer_args(study="smoke", variant="t1", manifest="culane@1", run_kind="scratch",
                            set_items=["optim.lr=1e-4"], standalone=True,
                            max_epochs=1, max_train_steps=20, max_eval_batches=5)
    assert args == ["--study", "smoke", "--variant", "t1", "--manifest", "culane@1",
                    "--run-kind", "scratch", "--set", "optim.lr=1e-4", "--standalone",
                    "--max-epochs", "1", "--max-train-steps", "20", "--max-eval-batches", "5"]


def test_trainer_args_omits_unset():
    args = sub.trainer_args(study="s", variant="v", manifest="m@1", run_kind="scratch",
                            set_items=[], standalone=False, max_epochs=None,
                            max_train_steps=None, max_eval_batches=None)
    assert "--standalone" not in args and "--max-epochs" not in args


# --------------------------------------------------------------- worker_pool_spec

def test_worker_pool_spec_shape():
    spec = sub.worker_pool_spec("img:1", ["--study", "s"], {"RUN_ID": "r", "X": 2},
                                machine="g2-standard-12", accelerator="NVIDIA_L4")
    (pool,) = spec
    assert pool["machine_spec"] == {"machine_type": "g2-standard-12",
                                    "accelerator_type": "NVIDIA_L4", "accelerator_count": 1}
    assert pool["replica_count"] == 1
    assert pool["container_spec"]["image_uri"] == "img:1"
    assert pool["container_spec"]["args"] == ["--study", "s"]
    assert {"name": "RUN_ID", "value": "r"} in pool["container_spec"]["env"]
    assert {"name": "X", "value": "2"} in pool["container_spec"]["env"]  # valores a str


# --------------------------------------------------------------------- submit

def test_submit_dry_run_builds_spec_without_gcp():
    s = sub.submit(_settings(), tag="abc123", study="smoke", variant="t1", manifest="culane@1",
                   standalone=True, max_epochs=1, max_train_steps=20, max_eval_batches=5,
                   env_file=None, dry_run=True, run_id="smoke__t1__20260623T100000Z")
    assert s["image"] == "us-central1-docker.pkg.dev/cicd/ml-training/trainer:abc123"
    assert s["machine"] == "g2-standard-12" and s["accelerator"] == "NVIDIA_L4"
    assert s["base_output_dir"] == "gs://bkt-dev-vertex-staging-usc1/jobs/smoke__t1__20260623T100000Z/"
    env = {e["name"]: e["value"] for e in s["worker_pool_specs"][0]["container_spec"]["env"]}
    assert env["RUN_ID"] == "smoke__t1__20260623T100000Z" and env["LANETR_SHA"] == "dd2f8ab"
    assert "--standalone" in s["args"]


def test_submit_final_tier_uses_a100():
    s = sub.submit(_settings(), tag="x", study="s", variant="v", manifest="m@1",
                   tier="final", env_file=None, dry_run=True)
    assert s["machine"] == "a2-highgpu-1g" and s["accelerator"] == "NVIDIA_TESLA_A100"


def test_submit_creates_custom_job_with_fake_aiplatform():
    aip = FakeAip()
    sub.submit(_settings(), tag="abc", study="smoke", variant="t1", manifest="culane@1",
               env_file=None, aiplatform=aip, run_id="smoke__t1__20260623T100000Z")
    assert aip.inited["project"] == "vr-prj-dev-training-v1"
    assert aip.inited["staging_bucket"] == "gs://bkt-dev-vertex-staging-usc1"
    (job,) = aip.jobs
    assert job.kw["display_name"] == "smoke__t1__20260623T100000Z"
    assert job.kw["base_output_dir"].endswith("/jobs/smoke__t1__20260623T100000Z/")
    assert job.ran["service_account"] == "sa-mlt-trainer@vr-prj-dev-training-v1.iam.gserviceaccount.com"
    assert job.ran["sync"] is False
