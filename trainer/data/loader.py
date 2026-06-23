"""trainer.data.loader — dataloader: shards WebDataset -> entrada del modelo (800×320) + targets.

Por cada sample del shard (`.jpg` + `.lines.json` en formato común) se construye un dict estilo
`lanetr` y se le aplica la transformación de `lanetr` (recorta el cielo, redimensiona a 800×320,
augmenta y codifica los targets de filas-ancla). El batch sale como `{image, targets, meta}`.

VARIAS RESOLUCIONES: cada imagen se decodifica a su resolución REAL (CULane 1640×590, usuario
1280×720, otras…) y la transformación recorta+redimensiona usando ese tamaño (`crop_top_ratio`
es una FRACCIÓN de la altura, vale para cualquier resolución). El tamaño nativo queda en
`meta['src_size']` (lo necesita la evaluación para mapear las predicciones de vuelta).

Diseño testeable: la `transform` de lanetr y el cliente `gcs` se INYECTAN (así el streaming y el
collate se prueban sin lanetr ni GCP). En runtime, `make_loader` los cablea de verdad.
"""

from __future__ import annotations

import io
import random
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from vroad_mlt import webdataset_io
from vroad_mlt.webdataset_io import Sample

__all__ = ["decode_sample", "collate", "LaneShardDataset", "build_transform", "make_loader",
           "GcsClientFactory"]


class GcsClientFactory:
    """Crea un cliente GCS por worker del DataLoader. PICKLABLE (a diferencia de un `lambda`).

    En Windows (`spawn`) y en Python 3.14+ (`forkserver`) el dataset se SERIALIZA hacia cada worker,
    así que el `gcs_factory` debe poder picklearse (un lambda local no). Además `google-cloud-storage`
    no es fork/spawn-safe → conviene un cliente NUEVO por worker.
    """

    def __init__(self, project: str) -> None:
        self.project = project

    def __call__(self):
        from vroad_mlt.gcs import Gcs  # noqa: PLC0415 - perezoso (extra `gcp`)

        return Gcs.from_project(self.project)

# Transformación estilo lanetr: muta `sample` (image->tensor, añade targets) y lo devuelve.
Transform = Callable[[dict, random.Random], dict]
GcsFactory = Callable[[], Any]


def decode_sample(sample: Sample) -> dict:
    """`webdataset_io.Sample` -> dict estilo lanetr (imagen PIL + carriles en px NATIVOS).

    La imagen se decodifica a su resolución REAL; los carriles vienen en píxeles de esa misma
    resolución (del `.lines.json`). El recorte+resize a 800×320 lo hace la transformación.
    """
    img = Image.open(io.BytesIO(sample.image)).convert("RGB")
    lanes = [np.array([[p.x, p.y] for p in lane], dtype=np.float32) for lane in sample.lines.lines]
    return {
        "image": img,
        "lanes": lanes,
        "slots": None,
        # `src_size` = (W, H) NATIVO (lo necesita la evaluación para mapear de vuelta con predict()).
        "meta": {"key": sample.key, "timestamp": sample.lines.timestamp, "src_size": img.size},
    }


def collate(batch: list[tuple]) -> dict:
    """`[(img_tensor, targets, meta)]` -> `{image:(B,3,H,W), targets:[...], meta:[...]}`."""
    images = torch.stack([b[0] for b in batch])
    return {"image": images, "targets": [b[1] for b in batch], "meta": [b[2] for b in batch]}


class LaneShardDataset(IterableDataset):
    """Stream de samples desde shards `.tar` en GCS, transformados al espacio del modelo.

    - `shard_uris`: `gs://.../<split>-NNNNN.tar`.
    - `gcs_factory()`: crea un cliente GCS por worker (storage no es fork-safe).
    - `transform(sample, rng)`: la transformación de lanetr (inyectada -> testeable sin lanetr).
    - `shuffle`: baraja el orden de shards por época (semilla + epoch).
    """

    def __init__(
        self,
        shard_uris: Sequence[str],
        gcs_factory: GcsFactory,
        transform: Transform,
        *,
        shuffle: bool = False,
        seed: int = 42,
        epoch: int = 0,
    ) -> None:
        self.shard_uris = list(shard_uris)
        self.gcs_factory = gcs_factory
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        """Cambiar la época reordena los shards (barajado distinto por época)."""
        self.epoch = epoch

    def _shards_for_worker(self) -> list[str]:
        shards = list(self.shard_uris)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(shards)
        info = get_worker_info()
        if info is not None:  # reparte shards entre los workers del DataLoader
            shards = shards[info.id :: info.num_workers]
        return shards

    def __iter__(self):
        gcs = self.gcs_factory()
        info = get_worker_info()
        wid = 0 if info is None else info.id
        rng = random.Random(self.seed + self.epoch * 1000 + wid)
        for uri in self._shards_for_worker():
            data = gcs.read_bytes(uri)
            for sample in webdataset_io.read_shard(io.BytesIO(data)):
                out = self.transform(decode_sample(sample), rng)
                yield out["image"], out["targets"], out["meta"]


def build_transform(cfg: dict, split: str) -> Transform:
    """Transformación de lanetr para un split (import perezoso de lanetr)."""
    from lanetr.data.transforms import build_transforms  # noqa: PLC0415 - perezoso

    aug = cfg["data"]["aug"]
    return build_transforms(
        split,
        img_w=cfg["arch"]["img_w"],
        img_h=cfg["arch"]["img_h"],
        crop_top_ratio=cfg["data"]["crop_top_ratio"],
        augment=(split == "train"),
        encode_targets=True,
        num_rows=cfg["arch"]["num_rows"],
        hflip_prob=aug["hflip_prob"],
        rotation_deg=aug["rotation_deg"],
        scale_jitter=aug["scale_jitter"],
        brightness=aug["brightness"],
        contrast=aug["contrast"],
    )


def make_loader(
    cfg: dict,
    split: str,
    shard_uris: Sequence[str],
    gcs_factory: GcsFactory,
    *,
    shuffle: Optional[bool] = None,
    transform: Optional[Transform] = None,
) -> DataLoader:
    """DataLoader del split desde los shards. `transform=None` -> el de lanetr (runtime)."""
    if shuffle is None:
        shuffle = split == "train"
    if transform is None:
        transform = build_transform(cfg, split)
    dataset = LaneShardDataset(
        shard_uris, gcs_factory, transform, shuffle=shuffle, seed=cfg["data"]["seed"]
    )
    return DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate,
        pin_memory=True,
        drop_last=(split == "train"),
    )
