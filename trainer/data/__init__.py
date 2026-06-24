"""trainer.data — carga de datos: shards WebDataset -> entrada del modelo (800×320) + targets."""

from trainer.data.loader import (
    GcsClientFactory,
    LaneShardDataset,
    build_transform,
    collate,
    decode_sample,
    make_loader,
)

__all__ = ["decode_sample", "collate", "LaneShardDataset", "build_transform", "make_loader",
           "GcsClientFactory"]
