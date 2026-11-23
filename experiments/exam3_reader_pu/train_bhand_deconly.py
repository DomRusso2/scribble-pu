"""ScribblePU Exam 3 — Reader-PU v2 (B-Hand DecOnly: Frozen V3 Encoder + No-Ascent Brier-nnPU).

Hypothesis tested cleanly against B-Hand v1:
  B-Hand v1 (unfrozen encoder, W_BLK=1/12, Y_LO=0.25) achieved dramatic separation on
  0306B (First Letters row), w029 (M1 0.9145), and 0009B (M1 0.8829, +0.023 over peak
  morphological erosion, 0 craters), with two mild tradeoffs:
    1. FA80 rose (+0.021 on soup50) because 65% of hand tiles clamped with W_BLK=1/12.
    2. Never-seen faint scroll 0841_ag1 GATE dipped (-0.028 on soup50) because unfreezing
       the 30.2M-param encoder on 4 high-SNR hand scrolls drifted faint-scroll features.

Fix tested here (ONLY 2 clean changes, 800 steps, ~9 min GPU):
  1. Freeze V3's 13-scroll encoder (`enc_lr = 0.0`, `requires_grad = False`), training
     ONLY the 4.32M-param 2D decoder (`dec_lr = 1.2e-4`) so V3's faint-scroll encoder
     features stay 100% intact.
  2. Set `W_BLK = 2/12 (1/6)` and `Y_LO = 0.24` (~61/255, exact V3 background floor) on
     verified certain-blank 0172 tiles (`FastBlank0172Pool`), keeping `W_INK = 10/12`.
"""
import os
import sys
import copy
import json
import time
import random
import numpy as np
import torch

R = r'C:\Users\nikox\Documents\Vesuvius'
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
EX1 = os.path.join(REPO, 'experiments', 'exam1_from_scratch')
GEM = os.path.join(REPO, 'experiments', 'common')  # shared helper kt_nnpu.py

for p in (REPO, HERE, EX1, GEM):
    if p not in sys.path:
        sys.path.insert(0, p)

import kt_nnpu as BASE
from train_bhand import (
    build_handonly_model_and_loader, logits_of, eval_train_telemetry,
    make_soup, save_ckpt, HAND_QUOTAS, CKPT_DIR, LOG_DIR,
)
from train_exam1 import FastBlank0172Pool
from scribble_pu.losses import BrierNNPULoss, certain_blank_brier_loss

DEVICE = torch.device('cuda')
STEPS = 800
Y_LO, Y_HI, RHO = 0.24, 0.82, 0.96
W_BLK = 2.0 / 12.0
W_INK = 10.0 / 12.0
DEC_LR, WD = 1.2e-4, 1e-4
LOGF = os.path.join(LOG_DIR, 'train_bhand_deconly.log')


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOGF, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    v3p = os.path.join(R, 'release', 'reader-v4-v3xe1', 'checkpoints', 'reader-v3.pth')
    model, loader, cfg, v3_ck = build_handonly_model_and_loader(v3p)
    pad_to = BASE.configured_input_pad_depth(cfg)
    v3_sd_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    ref_model = copy.deepcopy(model).to(DEVICE).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # Freeze V3's 13-scroll shared encoder + depth fusion; train ONLY the 2D decoder!
    dec = []
    enc_n = dec_n = 0
    for n, p in model.named_parameters():
        if n.startswith('network.shared_encoder.') or n.startswith('depth_fusion.'):
            p.requires_grad_(False)
            enc_n += p.numel()
        else:
            p.requires_grad_(True)
            dec.append(p)
            dec_n += p.numel()
    log(f'[DECONLY] Frozen encoder={enc_n/1e6:.2f}M | Trainable decoder={dec_n/1e6:.2f}M | W_BLK={W_BLK:.3f} Y_LO={Y_LO}')

    blanks = FastBlank0172Pool(cfg, pad_to, n_pool=256, seed=42)

    it = iter(loader)
    eval_batches = []
    hn = hd = 0.0
    for st in range(25):
        b = next(it)
        img = BASE.center_pad_input_depth(b['image'].float(), pad_to)
        tgt = BASE.unpack_ink_and_sup(b)
        geom = b['geom']
        P, U, _, full = BASE.extract_centerlines_and_masks(tgt, geom)
        hn += float(full.sum())
        hd += float(P.sum())
        if st < 8:
            eval_batches.append((img, tgt, geom))
    alpha_hand = hn / max(hd, 1.0)
    log(f'[PRIOR] alpha_hand={alpha_hand:.3f} eff={alpha_hand*RHO:.3f}')

    criterion = BrierNNPULoss(y_lo=Y_LO, y_hi=Y_HI, rho=RHO, min_pos_px=30, min_unlabeled_px=50, mode='no_ascent')
    opt = torch.optim.AdamW(dec, lr=DEC_LR, weight_decay=WD)
    lr_fn = lambda s: (s + 1) / 60.0 if s < 60 else 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * (s - 60) / max(1, STEPS - 60)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    blk_hist = []
    cl_tot = tiles_tot = 0
    t0 = time.time()
    model.train()

    for step in range(1, STEPS + 1):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        img_all = BASE.center_pad_input_depth(b['image'].float(), pad_to)
        tgt_all = BASE.unpack_ink_and_sup(b)
        geom_all = b['geom']
        bimg, bmask = blanks.batch(2)

        opt.zero_grad(set_to_none=True)
        lv = 0.0
        for s0 in range(0, img_all.shape[0], 4):
            s1 = min(img_all.shape[0], s0 + 4)
            img = img_all[s0:s1].to(DEVICE, non_blocking=True)
            tgt = tgt_all[s0:s1].to(DEVICE, non_blocking=True)
            geom = geom_all[s0:s1].to(DEVICE, non_blocking=True)
            P, U, _, _ = BASE.extract_centerlines_and_masks(tgt, geom)
            lo = logits_of(model, img)
            l_mb, st = criterion(lo, P, U, alpha=alpha_hand)
            if st['n_used'] > 0:
                w_mb = W_INK * (st['n_used'] / 8.0)
                (l_mb * w_mb).backward()
                lv += float(l_mb.item()) * w_mb
                cl_tot += int(st['n_clamped'])
                tiles_tot += int(st['n_used'])

        lb = logits_of(model, bimg.to(DEVICE, non_blocking=True))
        l_blk, bm = certain_blank_brier_loss(lb, bmask.to(DEVICE), y_lo=Y_LO)
        (l_blk * W_BLK).backward()
        lv += float(l_blk.item()) * W_BLK
        blk_hist.append(bm)

        torch.nn.utils.clip_grad_norm_(dec, 1.0)
        opt.step()
        sched.step()

        if step in (1, 50, 100, 200, 400, 600, 800):
            tel = eval_train_telemetry(model, ref_model, eval_batches)
            log(
                f'step {step:3d}/{STEPS} | loss={lv:.5f} | blk={np.mean(blk_hist[-10:]):.3f} '
                f'| cl={100.0*cl_tot/max(tiles_tot,1):.1f}% | handP={tel["handP"][0]:.3f}({tel["handP"][1]:+.3f}) '
                f'| inkU={tel["inkU"][0]:.3f}({tel["inkU"][1]:+.3f}) | gap={tel["gap"][0]:.3f}({tel["gap"][1]:+.3f}) '
                f'| bg={tel["bg"][0]:.3f}({tel["bg"][1]:+.3f}) | {time.time()-t0:.0f}s'
            )

    raw_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    meta = dict(variant='bhand_deconly_no_ascent', y_lo=Y_LO, y_hi=Y_HI, rho=RHO,
                alpha_hand=alpha_hand, w_blk=W_BLK, enc_lr=0.0, dec_lr=DEC_LR, steps=STEPS, hand_quotas=HAND_QUOTAS)
    save_ckpt(os.path.join(CKPT_DIR, 'reader_pu_deconly_raw.pth'), raw_sd, v3_ck, STEPS, {**meta, 'soup_w': 1.0})
    save_ckpt(os.path.join(CKPT_DIR, 'reader_pu_deconly_soup65.pth'), make_soup(raw_sd, v3_sd_cpu, 0.65), v3_ck, STEPS, {**meta, 'soup_w': 0.65})
    save_ckpt(os.path.join(CKPT_DIR, 'reader_pu_deconly_soup50.pth'), make_soup(raw_sd, v3_sd_cpu, 0.50), v3_ck, STEPS, {**meta, 'soup_w': 0.50})
    log(f'[DONE] Total DecOnly training time: {(time.time()-t0)/60.0:.1f} min')


if __name__ == '__main__':
    main()
