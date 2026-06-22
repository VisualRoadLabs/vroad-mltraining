"""Runner del job `job-dev-dataset-build-usc1`: materializa shards + manifiesto.

Lee la spec (ver SPEC.md), consulta el Data Lake (read-only), descarga imágenes +
`.lines.json`, los empaqueta en shards WebDataset y escribe el manifiesto en
`bkt-dev-datasets-usc1`. Es la única frontera hacia el Data Lake.

    python jobs/dataset-build/run.py --spec <spec.json | gs://...> [--dry-run] [--limit N] [--workers N]

Las descargas GCS van CONCURRENTES (cuello de botella = red); la escritura al shard es
secuencial. La función `materialize` recibe `bq`/`gcs` por inyección (testeable con fakes);
`main` crea los clientes reales. Verificación real: con tu terminal contra dev.
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from google.api_core import exceptions as gax  # type: ignore[import-untyped]

from vroad_mlt import lines_format, naming
from vroad_mlt.dataset_spec import Spec
from vroad_mlt.datalake.assets import dedup_culane as dd
from vroad_mlt.datalake.queries import build_images_query, label_uri_for_image
from vroad_mlt.manifest import Manifest
from vroad_mlt.webdataset_io import ShardWriter

# Descargas concurrentes por defecto (el cuello de botella es la red, no la CPU).
DEFAULT_WORKERS = 16


def sample_key(index: int, *, digits: int = 8) -> str:
    """Clave WebDataset secuencial por (dataset@version, split)."""
    return f"{index:0{digits}d}"


def load_culane_keep_set(gcs: Any, assets_prefix: str, threshold: float = dd.DEFAULT_DIFF_THRESHOLD) -> set[str]:
    """Carga el keep-set del dedup de CULane desde GCS (train.txt + train_diffs.npz)."""
    text = gcs.read_text(f"{assets_prefix}/train.txt")
    keys = [dd.normalize_culane_key(ln.split()[0]) for ln in text.splitlines() if ln.strip()]
    diffs = dd.load_diffs(io.BytesIO(gcs.read_bytes(f"{assets_prefix}/train_diffs.npz")))
    return dd.build_keep_set(keys, list(diffs), threshold)


def _download_one(gcs: Any, img_uri: str):
    """Descarga (imagen, líneas) de una imagen; `None` si falta el `.lines.json` o la imagen."""
    try:
        label_uri = label_uri_for_image(img_uri)
        lines = lines_format.loads(gcs.read_text(label_uri))  # valida formato común
        image = gcs.read_bytes(img_uri)
        return image, lines
    except gax.NotFound:
        return None  # p. ej. CurveLanes test sin GT


def materialize(
    spec: Spec,
    *,
    bq: Any,
    gcs: Any,
    dl_project: str,
    out_bucket: str,
    tmp_dir: Any,
    assets_prefix: str,
    dedup_threshold: float = dd.DEFAULT_DIFF_THRESHOLD,
    limit: Optional[int] = None,
    workers: int = DEFAULT_WORKERS,
    log: Any = None,
) -> Manifest:
    """Materializa la spec: shards a `out_bucket` + manifiesto. Devuelve el `Manifest`."""
    counts: dict[str, int] = {}

    for src in spec.sources:
        keep: Optional[set[str]] = None
        if src.dedup:
            if src.dataset == "culane":
                keep = load_culane_keep_set(gcs, assets_prefix, dedup_threshold)
                if log:
                    log.info("CULane dedup active", extra={"keep": len(keep)})
            elif log:
                log.warning("dedup ignored (CULane only)", extra={"dataset": src.dataset})

        for split in src.splits:
            # El dedup de CULane es de TRAIN (train_diffs.npz); NO se aplica a val/test.
            keep_split = keep if (keep is not None and split == "train") else None
            # Con --limit, lo metemos en el SQL (no bajar todo). Si hay dedup, sobre-pedimos
            # para acabar con ~limit muestras tras descartar.
            sql_limit = limit
            if limit is not None and keep_split is not None:
                sql_limit = limit * 4
            sql, params = build_images_query(
                dl_project, src.source, src.dataset, split, src.filters, limit=sql_limit
            )
            rows = bq.query(sql, params)

            # Candidatos tras el dedup (filtrado en memoria, barato).
            candidates: list[str] = []
            filtered = 0
            for row in rows:
                u = row["gcs_uri"]
                if keep_split is not None and dd.culane_key_from_gcs_uri(u) not in keep_split:
                    filtered += 1
                    continue
                candidates.append(u)

            # Cada shard se SUBE a GCS y se BORRA del disco en cuanto se cierra: en Cloud Run el
            # disco local es RAM, así no acumulamos ~10GB y el job no se queda sin memoria (OOM).
            shards_prefix = naming.shards_prefix(src.dataset, src.version)

            def _on_shard_done(path: Path, _prefix: str = shards_prefix) -> None:
                gcs.upload_file(naming.gs_uri(out_bucket, _prefix, path.name), path)
                path.unlink(missing_ok=True)

            # Descarga CONCURRENTE (cuello de botella = red); escritura secuencial al shard.
            out_dir = Path(tmp_dir) / f"{src.dataset}@{src.version}"
            writer = ShardWriter(out_dir, split, maxcount=spec.shard_maxcount, on_shard_done=_on_shard_done)
            written = skipped = 0
            batch_size = max(workers * 8, 1)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                stop = False
                for start in range(0, len(candidates), batch_size):
                    if stop:
                        break
                    batch = candidates[start:start + batch_size]
                    for result in ex.map(lambda u: _download_one(gcs, u), batch):
                        if limit is not None and written >= limit:
                            stop = True
                            break
                        if result is None:
                            skipped += 1
                            continue
                        image, lines = result
                        writer.write(sample_key(written), image, lines)
                        written += 1
                        if log and written % 5000 == 0:
                            log.info("progress", extra={"dataset": src.dataset, "split": split, "written": written})
            writer.close()  # sube y borra el último shard

            counts[split] = counts.get(split, 0) + written
            if log:
                log.info(
                    "split done",
                    extra={"dataset": src.dataset, "split": split, "written": written,
                           "skipped": skipped, "filtered": filtered, "shards": len(writer.shards)},
                )

    manifest = Manifest.build(
        spec.manifest_id, spec.version, spec.to_manifest_sources(), counts,
        created_at=naming.utc_timestamp(),
    )
    manifest_uri = naming.gs_uri(out_bucket, naming.manifest_key(spec.manifest_id, spec.version))
    gcs.write_json(manifest_uri, manifest.to_dict(), indent=2)
    if log:
        log.info("manifest written", extra={"uri": manifest_uri, "counts": counts})
    return manifest


def _read_spec(spec_arg: str, cfg: Any) -> str:
    """Lee la spec desde una ruta local o una URI `gs://` (para Cloud Run)."""
    if spec_arg.startswith("gs://"):
        from vroad_mlt.gcs import Gcs

        return Gcs.from_settings(cfg).read_text(spec_arg)
    return Path(spec_arg).read_text(encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    from vroad_mlt.config import get_settings
    from vroad_mlt.logging import get_logger, setup_logging

    ap = argparse.ArgumentParser(description="Materializa shards + manifiesto desde el Data Lake.")
    ap.add_argument("--spec", required=True, help="ruta al JSON de la spec")
    ap.add_argument("--dry-run", action="store_true", help="solo imprime las consultas")
    ap.add_argument("--limit", type=int, default=None, help="máx. samples por split (pruebas)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="descargas GCS concurrentes")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    setup_logging()
    log = get_logger("dataset-build")
    cfg = get_settings()
    spec = Spec.from_json(_read_spec(args.spec, cfg))

    if args.dry_run:
        for src in spec.sources:
            for split in src.splits:
                sql, params = build_images_query(cfg.project_datalake, src.source, src.dataset, split, src.filters)
                print(f"\n# {src.dataset}@{src.version} [{split}]\n{sql}\nparams={params}")
        return 0

    from vroad_mlt.bq import BigQuery
    from vroad_mlt.gcs import Gcs

    bq = BigQuery.from_settings(cfg)
    # Pool de conexiones >= workers para no descartar conexiones con descargas concurrentes.
    gcs = Gcs.from_settings(cfg, pool_maxsize=args.workers + 4)
    assets_prefix = naming.gs_uri(cfg.bucket_datasets, "_assets", "culane")
    with tempfile.TemporaryDirectory() as tmp:
        materialize(
            spec, bq=bq, gcs=gcs, dl_project=cfg.project_datalake,
            out_bucket=cfg.bucket_datasets, tmp_dir=tmp, assets_prefix=assets_prefix,
            limit=args.limit, workers=args.workers, log=log,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
