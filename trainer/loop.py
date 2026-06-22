"""trainer.loop — construir el modelo y entrenar una época (la receta de LaneTR).

Receta (portada de tu repo de investigación + CLAUDE §14): AdamW con LR por grupos
(backbone 0.1×, módulos "lentos" 0.1×, resto 1×), scheduler warmup lineal + cosine, EMA con
warmup del decay (estilo CLRerNet), bf16 (sin GradScaler), channels_last, TF32, FrozenBatchNorm,
grad-clip 0.1.

El paquete `lanetr` aporta `build_model`/`build_criterion`/`build_param_groups`/`prepare_targets`;
el **EMA** y el **scheduler** los implementamos aquí (no viven en el paquete). `train_epoch` recibe
todo por argumento (modelo, criterion, optimizer, …) → el mecanismo del loop se testea con un
modelo dummy de torch, sin `lanetr`. `build_trainables` (que sí usa `lanetr`) se verifica con GPU.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

__all__ = [
    "ModelEMA",
    "build_scheduler",
    "set_backends",
    "train_epoch",
    "build_trainables",
]


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


class ModelEMA:
    """Media móvil exponencial de los pesos (estilo CLRerNet): `ema = d·ema + (1-d)·modelo`.

    El decay arranca con warmup `d = decay·(1 - exp(-updates/tau))` para no quedar anclado a la
    inicialización aleatoria en los primeros pasos (clave en DETR). Promedia también los buffers
    de BatchNorm (floats). El checkpoint final se evalúa/entrega con estos pesos EMA.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: float = 2000.0) -> None:
        self.ema = deepcopy(_unwrap(model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / self.tau))
        msd = _unwrap(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach().to(v.device), alpha=1.0 - d)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd) -> None:
        self.ema.load_state_dict(sd)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_iters: int,
    warmup_iters: int,
    min_lr_ratio: float = 0.0,
):
    """LambdaLR: rampa lineal hasta `warmup_iters`, luego cosine hasta `min_lr_ratio`."""

    def fn(it: int) -> float:
        if it < warmup_iters:
            return (it + 1) / max(1, warmup_iters)
        prog = min(1.0, (it - warmup_iters) / max(1, total_iters - warmup_iters))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def set_backends(device: str, *, tf32: bool = True) -> None:
    """Activa TF32 (matmul/cudnn) en CUDA. No-op en CPU."""
    if device == "cuda" and tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def train_epoch(
    model: nn.Module,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loader,
    device: str,
    *,
    ema: Optional[ModelEMA] = None,
    prepare_targets: Optional[Callable] = None,
    amp: bool = False,
    channels_last: bool = False,
    grad_clip: float = 0.1,
    on_step: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Entrena UNA época. Devuelve la media de cada término de pérdida.

    `prepare_targets(targets, device)` mueve los targets a tensores (inyectado: el de `lanetr`).
    `on_step(info)` se llama por iteración con `{step, lr, grad_norm, losses}` (para loguear).
    """
    model.train()
    sums: dict[str, float] = {}
    n = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        if channels_last and device == "cuda":
            images = images.to(memory_format=torch.channels_last)
        targets = batch["targets"]
        if prepare_targets is not None:
            targets = prepare_targets(targets, device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp):
            pred = model(images)
            losses = criterion(pred, targets)

        optimizer.zero_grad()
        losses["total"].backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        n += 1
        step_losses = {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in losses.items()}
        for k, v in step_losses.items():
            sums[k] = sums.get(k, 0.0) + v
        if on_step is not None:
            on_step({"step": n, "lr": optimizer.param_groups[0]["lr"],
                     "grad_norm": grad_norm, "losses": step_losses})

    return {k: v / max(n, 1) for k, v in sums.items()}


def build_trainables(cfg: dict, device: str, *, iters_per_epoch: int, epochs: int):
    """Construye (modelo, criterion, optimizer, scheduler, ema, prepare_targets) con `lanetr`.

    Import perezoso de `lanetr` → este módulo se importa sin `lanetr` (el loop se testea con
    dummies). Se verifica en runtime con GPU.
    """
    from lanetr.contract.build import build_criterion, build_model, build_param_groups  # noqa: PLC0415
    from lanetr.losses.criterion import prepare_targets  # noqa: PLC0415

    model = build_model(cfg).to(device)
    if cfg["train"]["channels_last"] and device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if cfg["train"]["freeze_bn"]:
        model.backbone.eval()  # mantener BN congelado en modo eval

    criterion = build_criterion(cfg)
    optimizer = torch.optim.AdamW(
        build_param_groups(model, cfg), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"]
    )
    total_iters = max(1, iters_per_epoch * epochs)
    warmup_iters = min(cfg["schedule"]["warmup_epochs"] * iters_per_epoch, total_iters)
    min_lr_ratio = cfg["schedule"]["min_lr"] / cfg["optim"]["lr"]
    scheduler = build_scheduler(optimizer, total_iters, warmup_iters, min_lr_ratio)
    ema = ModelEMA(model, cfg["optim"]["ema_decay"], tau=2000.0)
    return model, criterion, optimizer, scheduler, ema, prepare_targets
