"""Unit tests verifying mathematical guarantees of ScribblePU BrierNNPULoss (CPU only)."""
import os
import sys
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO)

from scribble_pu.losses import BrierNNPULoss, certain_blank_brier_loss
from scribble_pu.certain_blank import build_certain_blank_mask
import numpy as np


def test_scar_unbiasedness():
    """Under exact SCAR (rho=1.0, alpha=1/c), expected BrierNNPULoss equals supervised Brier risk."""
    torch.manual_seed(0)
    y_lo, y_hi = 0.25, 0.82
    c = 0.20
    loss_fn = BrierNNPULoss(y_lo=y_lo, y_hi=y_hi, rho=1.0, pi_u_cap=0.95, pi_tot_cap=0.95, mode='no_ascent')

    # Large sample so Monte Carlo variance is tiny
    N = 200_000
    y = (torch.rand(1, 1, 1, N) < 0.25)
    logits = torch.where(y, torch.randn(1, 1, 1, N) * 0.5 + 1.0, torch.randn(1, 1, 1, N) * 0.5 - 1.0)
    p = torch.sigmoid(logits)
    sup_risk = torch.where(y, (p - y_hi) ** 2, (p - y_lo) ** 2).mean().item()

    P = y & (torch.rand_like(logits) < c)
    U = ~P
    pu_loss, stats = loss_fn(logits, P, U, alpha=1.0 / c)
    assert stats['n_clamped'] == 0.0
    diff = abs(float(pu_loss.item()) - sup_risk)
    assert diff < 2e-3, f"PU risk {pu_loss.item():.5f} != supervised risk {sup_risk:.5f} (diff={diff:.5f})"
    print(f"[PASS] test_scar_unbiasedness: pu={pu_loss.item():.5f} sup={sup_risk:.5f} diff={diff:.5f}")


def test_no_ascent_never_pushes_positives_down_on_clamp():
    """When R_neg < 0 (e.g. bright ridge positives + dim U), No-Ascent keeps pushing P UP toward y_hi,
    whereas vanilla Kiryo pushes P DOWN toward y_lo."""
    y_lo, y_hi = 0.25, 0.82
    # Construct a tile where P is at 0.72 (< y_hi=0.82) and U is mostly near 0.26 -> R_neg < 0!
    logits_base = torch.full((1, 1, 64, 64), -1.05)  # sigmoid ~ 0.259
    P = torch.zeros((1, 1, 64, 64), dtype=torch.bool)
    P[:, :, :8, :8] = True  # 64 positive pixels
    logits_base[P] = 0.94   # sigmoid ~ 0.719 (< 0.82 -> should be pushed UP!)
    U = ~P

    # 1. No-Ascent
    lo_na = logits_base.clone().requires_grad_(True)
    loss_na, st_na = BrierNNPULoss(y_lo=y_lo, y_hi=y_hi, rho=0.96, mode='no_ascent')(lo_na, P, U, alpha=7.0)
    loss_na.backward()
    assert st_na['n_clamped'] == 1.0, "Expected clamp in this test"
    grad_p_na = lo_na.grad[P].mean().item()
    # Gradient wrt logit should be NEGATIVE so gradient descent INCREASES logit toward y_hi=0.82!
    assert grad_p_na < 0.0, f"No-Ascent should push P UP (grad < 0), got {grad_p_na:+.6f}"

    # 2. Vanilla Kiryo (contrast!)
    lo_k = logits_base.clone().requires_grad_(True)
    loss_k, st_k = BrierNNPULoss(y_lo=y_lo, y_hi=y_hi, rho=0.96, mode='kiryo')(lo_k, P, U, alpha=7.0)
    loss_k.backward()
    assert st_k['n_clamped'] == 1.0
    grad_p_k = lo_k.grad[P].mean().item()
    # Kiryo ascent flips the sign and pushes confirmed positives P DOWN toward y_lo!
    assert grad_p_k > 0.0, f"Expected Kiryo clamp-ascent to push P DOWN (grad > 0), got {grad_p_k:+.6f}"
    print(f"[PASS] test_no_ascent_vs_kiryo: grad_P(no_ascent)={grad_p_na:+.5f} (UP) vs grad_P(kiryo)={grad_p_k:+.5f} (DOWN!)")


def test_unlabeled_tile_skipped_not_negative():
    """A tile with nP < min_pos_px must be skipped (zero gradient), never treated as all-negative."""
    logits = torch.zeros((1, 1, 32, 32), requires_grad=True)
    P = torch.zeros((1, 1, 32, 32), dtype=torch.bool)
    P[:, :, 0, :5] = True  # only 5 px < 30
    U = ~P
    loss, st = BrierNNPULoss(min_pos_px=30, mode='no_ascent')(logits, P, U, alpha=7.0)
    loss.backward()
    assert st['n_used'] == 0.0
    assert float(loss.item()) == 0.0
    assert float(logits.grad.abs().max().item()) == 0.0
    print("[PASS] test_unlabeled_tile_skipped_not_negative")


def test_certain_blank_masking():
    """certain_blank_brier_loss must only backprop through masked certain-blank pixels."""
    logits = torch.zeros((2, 1, 16, 16), requires_grad=True)
    mask = torch.zeros((2, 16, 16), dtype=torch.bool)
    mask[:, :4, :4] = True
    loss, mean_p = certain_blank_brier_loss(logits, mask, y_lo=0.25)
    loss.backward()
    assert abs(mean_p - 0.5) < 1e-5
    assert float(logits.grad[:, 0][~mask].abs().max().item()) == 0.0
    assert float(logits.grad[:, 0][mask].abs().min().item()) > 0.0
    print("[PASS] test_certain_blank_masking")


def test_build_certain_blank_mask():
    """build_certain_blank_mask must exclude ink on any z-slice, pixels within margin_px, and unsupervised pixels."""
    ink = np.zeros((5, 64, 64), dtype=np.uint8)
    ink[3, 32, 32] = 1  # ink on slice 3 only
    sup = np.zeros((5, 64, 64), dtype=np.uint8)
    sup[:, 4:60, 4:60] = 1
    cb = build_certain_blank_mask(ink, sup, margin_px=16)
    assert cb.shape == (64, 64)
    assert not cb[32, 32], "Ink pixel must be excluded"
    assert not cb[32, 47], "Pixel 15 px from ink must be excluded (< 16)"
    assert cb[32, 48], "Supervised pixel 16 px from ink must be included"
    assert not cb[0, 0], "Unsupervised pixel must be excluded"
    print("[PASS] test_build_certain_blank_mask")


if __name__ == '__main__':
    test_scar_unbiasedness()
    test_no_ascent_never_pushes_positives_down_on_clamp()
    test_unlabeled_tile_skipped_not_negative()
    test_certain_blank_masking()
    test_build_certain_blank_mask()
    print("ALL 5 UNIT TESTS PASSED!")
