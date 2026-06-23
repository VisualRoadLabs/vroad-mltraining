"""trainer.evaluation.evaluate — evaluación por época: predicciones -> formato común -> métrica.

CAPA 1 (primaria, la vara de medir): NUESTRA métrica F1 (port de CLRNet, validada == C++) sobre el
benchmark CULane FIJO (`benchmarks/culane@v1/`, todo a 1640×590), con desglose en las 9 categorías
oficiales. Se aplica a CUALQUIER modelo (scratch, finetune, otras mezclas) → comparación justa. Las
capas 2 (dominio propio) y 3 (C++ oficial) quedan fuera de aquí (esta última es fase 2).

Flujo: el modelo predice en 800×320 y `model.predict(images, src_sizes, timestamps)` mapea a la
resolución NATIVA y emite FORMATO COMÚN (dicts `{timestamp, Lines, Scores}`); los convertimos a
`LinesFile` y se los pasamos a `metric.score`. La calibración del umbral es OFFLINE: barre `Scores`
sobre las predicciones ya hechas, sin re-inferir (por eso predecimos con `conf_thresh` bajo, para
no pre-filtrar y conservar todas las candidatas con su confianza).

CUIDADO CON LOS EJES:
- `image.size` de PIL y `src_sizes` de `predict` son **(W, H)**.
- `img_shape` de la métrica es **(H, W)** (= `IMG_SHAPE_CULANE` = (590, 1640)). No los mezcles.
El `img_shape` único de la métrica solo es correcto si TODAS las imágenes comparten resolución;
por eso la métrica primaria va sobre el benchmark CULane (uniforme), no sobre val de resolución
mixta.

Testeable sin `lanetr`: el callable `predict` y la `transform` se INYECTAN; en runtime se cablean
`model.predict` y la transform `val` de `lanetr`.
"""

from __future__ import annotations

import io
import random
from typing import Any, Callable, Optional, Sequence

import torch
from torch.utils.data import DataLoader, IterableDataset

from trainer.data import decode_sample
from vroad_mlt import lines_format, metric, naming, webdataset_io

__all__ = [
    "DEFAULT_THRESHOLDS",
    "to_lines_file",
    "eval_collate",
    "EvalShardDataset",
    "make_eval_loader",
    "predict_dataset",
    "categories_for",
    "evaluate",
    "calibrate",
    "run_eval",
    "load_categories",
    "benchmark_shard_uris",
]

# Predict callable: `predict(images, src_sizes, conf_thresh=, timestamps=) -> list[dict]` (común).
Predict = Callable[..., list[dict]]
Transform = Callable[[dict, random.Random], dict]
GcsFactory = Callable[[], Any]

# Barrido por defecto para calibrar el umbral offline (0.05 … 0.95).
DEFAULT_THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(1, 20))


def to_lines_file(pred: Any) -> "lines_format.LinesFile":
    """Predicción en formato común -> `LinesFile` validado (lo que consume `metric.score`).

    `model.predict` emite dicts `{timestamp, Lines, Scores}`; `parse` exige `timestamp` (por eso a
    `predict` se le pasan los timestamps del GT). Si ya viniera un `LinesFile`, se devuelve tal cual.
    """
    if isinstance(pred, lines_format.LinesFile):
        return pred
    return lines_format.parse(pred)


def eval_collate(batch: list[tuple]) -> dict:
    """`[(img_tensor, gt_lines_file, meta)]` -> `{image:(B,3,H,W), gt:[LinesFile], meta:[dict]}`."""
    images = torch.stack([b[0] for b in batch])
    return {"image": images, "gt": [b[1] for b in batch], "meta": [b[2] for b in batch]}


class EvalShardDataset(IterableDataset):
    """Stream del benchmark/val: imagen 800×320 (entrada del modelo) + GT NATIVO (formato común).

    A diferencia del loader de entrenamiento, NO usa los targets codificados: lleva el GT crudo
    (`Sample.lines`, formato común a resolución nativa, sin mutar) y el `src_size` nativo para que
    `predict` mapee las predicciones de vuelta. Orden ESTABLE (sin shuffle) para alinear con las
    categorías.
    """

    def __init__(self, shard_uris: Sequence[str], gcs_factory: GcsFactory, transform: Transform) -> None:
        self.shard_uris = list(shard_uris)
        self.gcs_factory = gcs_factory
        self.transform = transform

    def __iter__(self):
        gcs = self.gcs_factory()
        rng = random.Random(0)  # eval: sin augment; el rng queda inerte
        for uri in self.shard_uris:
            data = gcs.read_bytes(uri)
            for sample in webdataset_io.read_shard(io.BytesIO(data)):
                decoded = decode_sample(sample)
                src_w, src_h = decoded["image"].size  # (W, H) nativo, ANTES de transformar
                out = self.transform(decoded, rng)
                meta = {"key": sample.key, "timestamp": sample.lines.timestamp, "src_size": (src_w, src_h)}
                yield out["image"], sample.lines, meta  # GT = LinesFile original (no mutado)


def make_eval_loader(
    cfg: dict,
    shard_uris: Sequence[str],
    gcs_factory: GcsFactory,
    *,
    transform: Optional[Transform] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> DataLoader:
    """DataLoader de evaluación (sin shuffle, sin drop_last). `transform=None` -> la `val` de lanetr."""
    if transform is None:
        from trainer.data import build_transform  # noqa: PLC0415 - perezoso (lanetr)

        transform = build_transform(cfg, "val")  # augment=False
    dataset = EvalShardDataset(shard_uris, gcs_factory, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size if batch_size is not None else cfg["data"]["batch_size"],
        num_workers=num_workers if num_workers is not None else cfg["data"]["num_workers"],
        collate_fn=eval_collate,
        pin_memory=True,
        drop_last=False,
    )


def predict_dataset(
    predict: Predict,
    loader,
    device: str,
    *,
    channels_last: bool = False,
    conf_thresh: float = 0.0,
) -> tuple[list, list, list[str]]:
    """Corre inferencia sobre el loader -> `(pred_files, gt_files, keys)` (todo en formato común).

    `conf_thresh` BAJO (0.0) para NO pre-filtrar: deja las candidatas (top-`max_lanes`) con su
    `Score`, de modo que el umbral se calibra OFFLINE con `metric`. Los `timestamps` del GT se pasan
    a `predict` para que el dict lleve `timestamp` (que `lines_format.parse` exige).

    SIN autocast bf16: `model.predict` decodifica a numpy (`.cpu().numpy()`), que no soporta bf16. La
    evaluación corre en fp32 (los pesos ya son fp32; el bf16 es solo el cómputo de entrenamiento).
    """
    pred_files: list = []
    gt_files: list = []
    keys: list[str] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            if channels_last and device == "cuda":
                images = images.to(memory_format=torch.channels_last)
            metas = batch["meta"]
            src_sizes = [m["src_size"] for m in metas]  # (W, H)
            timestamps = [int(m["timestamp"]) for m in metas]
            preds = predict(images, src_sizes, conf_thresh=conf_thresh, timestamps=timestamps)
            for p, gt, m in zip(preds, batch["gt"], metas):
                pred_files.append(to_lines_file(p))
                gt_files.append(gt)
                keys.append(m["key"])
    return pred_files, gt_files, keys


def categories_for(
    keys: Sequence[str], categories_map: dict, *, strict: bool = True
) -> list[str]:
    """Lista POSICIONAL de categorías alineada a `keys` (lo que pide `evaluate_by_category`).

    Mira cada `key` en `categories_map` (de `categories.json`). En `strict` (por defecto) falla
    ruidosamente si alguna key no tiene categoría: el desglose debe cubrir las 9 exactamente, no
    inventar una décima "unknown".
    """
    cats = [categories_map.get(k) for k in keys]
    if strict:
        missing = [k for k, c in zip(keys, cats) if c is None]
        if missing:
            raise ValueError(
                f"{len(missing)} imágenes sin categoría en categories.json (p.ej. {missing[:3]})"
            )
    return cats


def evaluate(
    pred_files: Sequence,
    gt_files: Sequence,
    *,
    categories: Optional[Sequence[str]] = None,
    threshold: float = 0.5,
    img_shape=metric.IMG_SHAPE_CULANE,
    benchmark: str = "culane@v1",
) -> dict:
    """`results.json`: F1 global + (si hay `categories`) F1 por categoría, a un umbral dado."""
    return metric.score(
        pred_files, gt_files, categories, threshold=threshold, img_shape=img_shape, benchmark=benchmark
    )


def calibrate(
    pred_files: Sequence,
    gt_files: Sequence,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    img_shape=metric.IMG_SHAPE_CULANE,
) -> tuple[float, dict]:
    """Calibra el umbral OFFLINE (barre `Scores` sobre las predicciones ya hechas) -> mejor umbral."""
    return metric.calibrate_threshold(pred_files, gt_files, list(thresholds), img_shape=img_shape)


def run_eval(
    model,
    loader,
    device: str,
    *,
    categories_map: Optional[dict] = None,
    threshold: float = 0.5,
    calibrate_thresholds: Optional[Sequence[float]] = None,
    channels_last: bool = False,
    conf_thresh: float = 0.0,
    img_shape=metric.IMG_SHAPE_CULANE,
    benchmark: str = "culane@v1",
    predict: Optional[Predict] = None,
) -> tuple[dict, list, list, list[str]]:
    """Evalúa el modelo sobre un loader -> `(results, pred_files, gt_files, keys)`.

    Pone el modelo en `eval()` (y restaura el modo previo). Usa `model.predict` salvo que se inyecte
    `predict`. Si `calibrate_thresholds` se pasa (una lista o `DEFAULT_THRESHOLDS`), calibra el umbral
    offline y reporta con el mejor; si no, usa `threshold` (por defecto 0.5, el punto de operación).
    """
    if predict is None:
        predict = model.predict
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    try:
        pred_files, gt_files, keys = predict_dataset(
            predict, loader, device, channels_last=channels_last, conf_thresh=conf_thresh
        )
    finally:
        if was_training and hasattr(model, "train"):
            model.train()

    categories = categories_for(keys, categories_map) if categories_map is not None else None
    threshold_used = threshold
    if calibrate_thresholds is not None:
        threshold_used, _ = calibrate(
            pred_files, gt_files, thresholds=calibrate_thresholds, img_shape=img_shape
        )
    results = evaluate(
        pred_files,
        gt_files,
        categories=categories,
        threshold=threshold_used,
        img_shape=img_shape,
        benchmark=benchmark,
    )
    return results, pred_files, gt_files, keys


# ------------------------------------------------------------- helpers de GCS (runtime)


def load_categories(gcs, datasets_bucket: str, *, benchmark: str = "culane@v1") -> dict:
    """Lee `benchmarks/<benchmark>/categories.json` (mapa key->categoría) del bucket de datasets."""
    uri = f"gs://{datasets_bucket}/{naming.benchmark_categories_key(benchmark)}"
    return gcs.read_json(uri)


def benchmark_shard_uris(gcs, datasets_bucket: str, split: str, *, benchmark: str = "culane@v1") -> list[str]:
    """Lista los shards `<split>-NNNNN.tar` del benchmark, en orden."""
    prefix = f"gs://{datasets_bucket}/{naming.benchmark_prefix(benchmark)}/{split}-"
    return sorted(gcs.list_uris(prefix))
