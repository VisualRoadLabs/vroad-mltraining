"""trainer.training — la receta de entrenamiento: loop, EMA y scheduler (estilo LaneTR)."""

from trainer.training.loop import (
    ModelEMA,
    build_scheduler,
    build_trainables,
    set_backends,
    train_epoch,
)

__all__ = ["ModelEMA", "build_scheduler", "set_backends", "train_epoch", "build_trainables"]
