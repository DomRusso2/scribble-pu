"""No-Ascent Brier-nnPU Loss for Positive-Unlabeled Scribble Ink Detection.

Drop-in loss term for `koine_machines.training.loss.losses.LOSS_TERM_BUILDERS`:

    "BrierNNPULoss": _build_brier_nnpu_term

Target layout matches `LabelSmoothedDCAndBCELoss(use_ignore_label=True)`:
  - `target[:, :1]`: positive scribble mask P (> 0.5 = confirmed ink scribble;
    0 = unlabeled).
  - `target[:, -1:]` (optional when C >= 2): ignore mask (> 0.5 = excluded from U).
  - `target[:, 1:2]` (optional when C == 3): certain-blank mask (> 0.5 = verified
    non-ink papyrus pixels anchored directly to `y_lo` with weight `w_blank`).

Why No-Ascent Brier-nnPU?
  Standard Kiryo et al. (2017) nnPU applies gradient ascent (-R_neg) whenever
  the per-tile negative risk estimate R_neg = R_u_neg - pi_u * R_p_neg < 0.
  On 2D segmentation scribbles (either bright medial centerlines or spatially
  clustered partial-letter strokes), R_neg < 0 occurs on 16-60% of tiles and
  Kiryo's ascent branch actively pushes confirmed positive scribbles toward y_lo.
  Replacing the ascent branch with `clamp(R_neg, min=0.0)` keeps `pi_tot * R_p_pos`
  active on every tile so confirmed positive ink is never pulled toward background.
"""
from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn


class VillaBrierNNPULoss(nn.Module):
    def __init__(
        self,
        alpha: float = 7.0,
        rho: float = 0.96,
        y_lo: float = 0.25,
        y_hi: float = 0.82,
        w_blank: float = 1.0 / 6.0,
        min_pos_px: int = 30,
        min_unlabeled_px: int = 50,
        pi_u_cap: float = 0.55,
        pi_tot_cap: float = 0.65,
        batch_neg_risk: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.y_lo = float(y_lo)
        self.y_hi = float(y_hi)
        self.w_blank = float(w_blank)
        self.min_pos_px = int(min_pos_px)
        self.min_unlabeled_px = int(min_unlabeled_px)
        self.pi_u_cap = float(pi_u_cap)
        self.pi_tot_cap = float(pi_tot_cap)
        self.batch_neg_risk = bool(batch_neg_risk)

    def forward(
        self, net_output: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits = net_output[:, :1].float()
        if logits.dim() == 5 and logits.shape[2] == 1:
            logits = logits[:, :, 0]
        tgt = target.float()
        if tgt.dim() == 5 and tgt.shape[2] == 1:
            tgt = tgt[:, :, 0]

        pos = tgt[:, :1] > 0.5
        if tgt.shape[1] >= 3:
            certain_blank = tgt[:, 1:2] > 0.5
            valid = tgt[:, 2:3] <= 0.5
        elif tgt.shape[1] == 2:
            certain_blank = torch.zeros_like(pos)
            valid = tgt[:, 1:2] <= 0.5
        else:
            certain_blank = torch.zeros_like(pos)
            valid = torch.ones_like(pos)

        pos = pos & valid
        unlabeled = valid & (~pos) & (~certain_blank)

        p = torch.sigmoid(logits)
        l_pos = (p - self.y_hi) ** 2
        l_neg = (p - self.y_lo) ** 2
        eff_alpha = max(1.05, self.alpha * self.rho)

        losses = []
        n_clamped = 0
        neg_risks = []
        pis = []

        if self.batch_neg_risk:
            nP = float(pos.sum().item())
            nU = float(unlabeled.sum().item())
            if nP >= self.min_pos_px and nU >= self.min_unlabeled_px:
                nV = nP + nU
                frac_U = nU / nV
                pi_u = min(self.pi_u_cap, max(0.0, (eff_alpha - 1.0) * nP / nU))
                pi_tot = min(self.pi_tot_cap, max(0.01, eff_alpha * nP / nV))
                R_neg = l_neg[unlabeled].mean() - pi_u * l_neg[pos].mean()
                if float(R_neg.item()) < 0.0:
                    n_clamped = 1
                neg_risks.append(float(R_neg.item()))
                pis.append(pi_u)
                losses.append(pi_tot * l_pos[pos].mean() + frac_U * torch.clamp(R_neg, min=0.0))
        else:
            for i in range(p.shape[0]):
                Pi, Ui = pos[i], unlabeled[i]
                nP, nU = float(Pi.sum().item()), float(Ui.sum().item())
                if nP < self.min_pos_px or nU < self.min_unlabeled_px:
                    continue
                nV = nP + nU
                frac_U = nU / nV
                pi_u = min(self.pi_u_cap, max(0.0, (eff_alpha - 1.0) * nP / nU))
                pi_tot = min(self.pi_tot_cap, max(0.01, eff_alpha * nP / nV))
                R_neg = l_neg[i][Ui].mean() - pi_u * l_neg[i][Pi].mean()
                if float(R_neg.item()) < 0.0:
                    n_clamped += 1
                neg_risks.append(float(R_neg.item()))
                pis.append(pi_u)
                losses.append(pi_tot * l_pos[i][Pi].mean() + frac_U * torch.clamp(R_neg, min=0.0))

        pu_loss = torch.stack(losses).mean() if losses else logits.sum() * 0.0

        blank_mask = certain_blank & valid
        if blank_mask.any():
            blank_loss = l_neg[blank_mask].mean()
            total = (1.0 - self.w_blank) * pu_loss + self.w_blank * blank_loss
            blank_mean = p[blank_mask].detach().mean()
        else:
            total = pu_loss
            blank_mean = logits.new_tensor(0.0)

        aux = {
            "clamped_frac": logits.new_tensor(n_clamped / max(len(losses), 1)),
            "neg_risk": logits.new_tensor(sum(neg_risks) / len(neg_risks) if neg_risks else 0.0),
            "pi_u": logits.new_tensor(sum(pis) / len(pis) if pis else 0.0),
            "blank_mean": blank_mean,
        }
        return total, aux
