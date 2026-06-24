"""Tests del mecanismo del loop de entrenamiento (trainer.training.loop).

Usa torch (CPU) con un modelo/criterion DUMMY — sin lanetr. Verifica EMA, scheduler y que una
época entrena de verdad (params cambian, EMA y scheduler avanzan). El modelo real (lanetr) se
prueba con GPU vía build_trainables.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from trainer.training.loop import ModelEMA, build_scheduler, train_epoch


# ----------------------------------------------------------------- ModelEMA

def test_model_ema_moves_toward_model():
    m = nn.Linear(4, 2)
    ema = ModelEMA(m, decay=0.9, tau=1.0)  # deepcopy de los pesos ORIGINALES
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)  # el modelo se mueve +1
    ema.update(m)
    assert ema.updates == 1
    ema_w = ema.ema.state_dict()["weight"]
    m_w = m.state_dict()["weight"]
    # d = 0.9·(1-e^-1) ≈ 0.569 -> ema = original + (1-d) ≈ original+0.43: entre original y nuevo
    assert torch.all(ema_w < m_w) and torch.all(ema_w > m_w - 1.0)


# --------------------------------------------------------------- scheduler

def test_scheduler_warmup_then_cosine():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    p.grad = torch.zeros_like(p)  # para llamar opt.step() antes que sched.step() (orden correcto)
    sched = build_scheduler(opt, total_iters=10, warmup_iters=4, min_lr_ratio=0.0)

    def step():
        opt.step()
        sched.step()

    assert opt.param_groups[0]["lr"] == pytest.approx(0.25)  # it0: (0+1)/4
    for _ in range(3):
        step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0)    # it3: fin del warmup
    for _ in range(6):
        step()
    assert opt.param_groups[0]["lr"] < 1.0                    # it9: cosine ha decaído


# ---------------------------------------------------------------- train_epoch

class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3 * 4 * 8, 4)

    def forward(self, images):
        return self.lin(images.flatten(1))


def _crit(pred, targets):
    loss = pred.pow(2).mean()
    return {"total": loss, "cls": loss.detach()}


def _loader(n):
    return [{"image": torch.randn(2, 3, 4, 8), "targets": [None, None], "meta": [{}, {}]}
            for _ in range(n)]


def test_train_epoch_updates_model_ema_and_logs():
    model = _TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = build_scheduler(opt, total_iters=3, warmup_iters=1, min_lr_ratio=0.1)
    ema = ModelEMA(model, decay=0.9, tau=1.0)
    before = [p.detach().clone() for p in model.parameters()]
    steps = []

    out = train_epoch(model, _crit, opt, sched, _loader(3), "cpu",
                      ema=ema, on_step=steps.append, grad_clip=0.1)

    assert set(out.keys()) == {"total", "cls"}      # medias de cada término
    assert isinstance(out["total"], float)
    assert ema.updates == 3                          # un update por iteración
    assert len(steps) == 3                           # on_step por iteración
    assert "lr" in steps[0] and "grad_norm" in steps[0] and "losses" in steps[0]
    # el modelo ha cambiado (entrenó de verdad)
    assert any(not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters()))


def test_train_epoch_uses_prepare_targets():
    seen = {}

    def fake_prepare(targets, device):
        seen["called"] = True
        return targets

    model = _TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = build_scheduler(opt, total_iters=1, warmup_iters=1)
    train_epoch(model, _crit, opt, sched, _loader(1), "cpu", prepare_targets=fake_prepare)
    assert seen.get("called") is True
