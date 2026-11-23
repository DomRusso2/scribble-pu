"""Regression tests for the BrierNNPULoss term.

Goes to ink-detection/koine_machines/training/loss/tests/test_brier_nnpu.py
(next to test_label_smoothing.py). CPU only, runs in about a second.
"""
import torch

from koine_machines.training.loss.brier_nnpu import VillaBrierNNPULoss
from koine_machines.training.loss.losses import create_loss_from_config


def _config(**overrides):
    term = {
        "name": "BrierNNPULoss",
        "metric_name": "brier_nnpu",
        "weight": 1.0,
        "alpha": 14.5,
        "rho": 0.96,
        "y_lo": 0.25,
        "y_hi": 0.82,
        "w_blank": 0.1667,
        "batch_neg_risk": True,
    }
    term.update(overrides)
    return {"loss": {"terms": [term]}}


def _scribble_batch(seed=0, batch=2, size=64):
    """Positive 1-px lines on half the strokes, the rest unlabelled, plus an ignore border."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, 1, size, size, generator=generator)
    target = torch.zeros(batch, 2, size, size)
    target[:, 0, 10:54, 20] = 1.0  # one scribbled stroke centreline per tile
    target[:, 0, 32, 8:40] = 1.0
    target[:, 1, :2, :] = 1.0  # ignore channel (last), like use_ignore_label=True
    return logits, target


def test_brier_nnpu_builds_from_config():
    loss = create_loss_from_config(_config())
    module = loss.terms[0].module

    assert isinstance(module, VillaBrierNNPULoss)
    assert module.alpha == 14.5
    assert module.y_lo == 0.25 and module.y_hi == 0.82
    assert module.batch_neg_risk is True


def test_composite_loss_is_finite_and_logs_aux_metrics():
    loss = create_loss_from_config(_config())
    logits, target = _scribble_batch()
    logits.requires_grad_(True)

    total = loss(logits, target)
    total.backward()

    assert torch.isfinite(total)
    assert torch.isfinite(logits.grad).all()
    assert "loss_terms/brier_nnpu_raw" in loss.latest_metrics
    assert "loss_aux/brier_nnpu/clamped_frac" in loss.latest_metrics


def test_clamp_never_pushes_confirmed_ink_toward_background():
    """When the negative risk goes below zero, Kiryo nnPU takes an ascent step that
    lowers the scribbled pixels. The clamp must leave only the positive pull on them."""
    module = VillaBrierNNPULoss(alpha=14.5, rho=0.96, y_lo=0.25, y_hi=0.82, w_blank=0.0)
    size = 64
    target = torch.zeros(1, 2, size, size)
    target[0, 0, 8:56, 32] = 1.0
    positive = target[:, :1] > 0.5

    # Unlabelled pixels sit at y_lo and scribbled ink at p = 0.6, so R_neg < 0.
    logits = torch.full((1, 1, size, size), torch.logit(torch.tensor(0.25)).item())
    logits[positive] = torch.logit(torch.tensor(0.6)).item()
    logits.requires_grad_(True)

    total, aux = module(logits, target)
    total.backward()

    assert aux["clamped_frac"].item() == 1.0
    assert aux["neg_risk"].item() < 0.0
    # A gradient-descent step must raise every scribbled pixel (negative gradient).
    assert (logits.grad[positive] < 0).all()


def test_certain_blank_channel_pulls_toward_y_lo():
    module = VillaBrierNNPULoss(w_blank=0.5)
    size = 64
    target = torch.zeros(1, 3, size, size)
    target[0, 0, 8:56, 32] = 1.0  # positive scribble
    target[0, 1, :, 48:] = 1.0  # certain-blank papyrus
    blank = target[:, 1:2] > 0.5

    logits = torch.zeros(1, 1, size, size, requires_grad=True)  # p = 0.5 > y_lo everywhere
    total, aux = module(logits, target)
    total.backward()

    assert torch.isfinite(total)
    assert aux["blank_mean"].item() > 0.25
    # Descent lowers blank pixels toward y_lo.
    assert (logits.grad[blank] > 0).all()
