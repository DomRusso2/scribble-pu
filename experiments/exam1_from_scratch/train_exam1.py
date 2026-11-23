"""ScribblePU Exam 1 — From-Scratch 2.5D Ink UNet Benchmark (Positive-Only Scribbles vs Full-Mask Oracle).

Trains the EXACT koine_machines 2.5D UNet (`vesuvius_unet_3d_stem_2d`) FROM RANDOM INITIALIZATION
on 3 hand-labeled scrolls (0139N:2, 1667H:3, Paris4H:3; batch_size=8; holding out ALL of 0814H and
0009B_125754 as completely unseen scrolls!) plus 2 verified certain-blank 0172 tiles per step.

Scribble regime (`--scribble half_centerline`, default):
  Simulates realistic fast human scribbling: 1-px medial centerlines on ~50% of letter regions
  (random 64x64 spatial mask with p=0.50 intersected with the medial ridge -> ~6-7% of ink pixels
  labeled, 50% of strokes completely unlabeled, ZERO background masks, ZERO supervision masks on
  ink tiles!).
  Alternative (`--scribble partial_letter20`): 20% of 48x48 letter regions painted at normal width
  (80% of letters completely unlabeled, ZERO background masks).

Matched Arms (`--arm`):
  1. oracle_fullmask : Standard villa BCE (ls=0.5) + 0.25*Dice on 100% full masks + supervision masks.
  2. naive_bce       : Standard BCE (ls=0.5) treating P as 1 and all U as 0.
  3. collar_bce      : Standard BCE (ls=0.5) treating P as 1, ignoring 12-px collar around P, rest U as 0.
  4. kiryo_nnpu      : Vanilla Kiryo et al. (2017) gradient-ascent Brier-nnPU + 0172 certain-blanks.
  5. scribble_pu     : Our No-Ascent Brier-nnPU (`mode='no_ascent'` or `'no_ascent_batch'`) + 0172 certain-blanks.

Usage:
  python train_exam1.py --arm oracle_fullmask --steps 3000
"""
import os
import sys
import json
import time
import random
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

R = r'C:\Users\nikox\Documents\Vesuvius'
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
EX3 = os.path.join(REPO, 'experiments', 'exam3_reader_pu')
GEM = os.path.join(REPO, 'experiments', 'common')  # shared helper kt_nnpu.py

for p in (REPO, EX3, GEM):
    if p not in sys.path:
        sys.path.insert(0, p)

import kt_nnpu as BASE
from train_bhand import CertainBlank0172Sampler, logits_of
from scribble_pu.losses import BrierNNPULoss, certain_blank_brier_loss
from koine_machines.models.make_model import make_model
from koine_machines.training.samplers import FixedScrollPriorStratifiedBatchSampler


class FastBlank0172Pool:
    """Pre-samples N diverse certain-blank 0172 tiles in RAM (~15s once) and applies fresh random
    rot90/flips each step (0.2ms/step vs 350ms/step live zarr reads -> saves 18 min per 3k-step arm!)."""

    def __init__(self, cfg, pad_to, n_pool=256, seed=42):
        sampler = CertainBlank0172Sampler(cfg, pad_to, patch=384, seed=seed)
        self.pad_to = pad_to
        self.rng = np.random.default_rng(seed + 7)
        t0 = time.time()
        self.vols = []
        self.masks = []
        for _ in range(n_pool):
            v, m = sampler.one()
            self.vols.append(v.astype(np.float16))
            self.masks.append(m.astype(np.bool_))
        print(f'[BLANK-POOL] Cached {n_pool} diverse 0172 certain-blank tiles in {time.time()-t0:.1f}s', flush=True)

    def batch(self, n=2):
        vs, ms = [], []
        for _ in range(n):
            idx = int(self.rng.integers(0, len(self.vols)))
            v = self.vols[idx].astype(np.float32)
            m = self.masks[idx]
            k = int(self.rng.integers(0, 4))
            if k:
                v = np.rot90(v, k, axes=(1, 2))
                m = np.rot90(m, k)
            if self.rng.random() < 0.5:
                v = v[:, :, ::-1]
                m = m[:, ::-1]
            if self.rng.random() < 0.5:
                v = v[:, ::-1, :]
                m = m[::-1, :]
            vs.append(np.ascontiguousarray(v))
            ms.append(np.ascontiguousarray(m))
        bimg = torch.from_numpy(np.stack(vs)[:, None])
        bimg = BASE.center_pad_input_depth(bimg, self.pad_to)
        bmask = torch.from_numpy(np.stack(ms))
        return bimg, bmask

DEVICE = torch.device('cuda')
Y_LO, Y_HI, RHO = 0.25, 0.82, 0.96
W_BLK = 2.0 / 12.0   # exact 2/12 ratio matching 2 blank tiles + 8 ink tiles
W_INK = 10.0 / 12.0
TRAIN_QUOTAS = {'0139N': 2, '1667H': 3, 'Paris4H': 3}  # 0814H & 0009B strictly held out!


def build_exam1_model_and_loader():
    cfg = json.load(open(os.path.join(R, 'data', 'readerv4', 'retrain', '1bn_smoke.json')))
    cfg['out_dir'] = HERE
    cfg['batch_size'] = 8
    cfg['dataloader_workers'] = 4
    cfg['checkpoint'] = None  # Strictly random initialization (ZERO warm start!)

    BASE.LA._D.InkDataset = BASE.NNPUDataset
    shared_ds = BASE.NNPUDataset(cfg, do_augmentations=False)
    allowed = set(TRAIN_QUOTAS.keys())
    train_patches = [
        p for p in shared_ds.training_patches
        if cfg['datasets'][p.segment.dataset_idx].get('sampling_scroll') in allowed
    ]
    ds = BASE.NNPUDataset(cfg, do_augmentations=True, patches=train_patches, segments=shared_ds.segments)
    sampler = FixedScrollPriorStratifiedBatchSampler(
        train_patches, cfg['datasets'],
        batch_quotas=TRAIN_QUOTAS, batch_size=8, seed=42,
    )
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True)

    # Fresh random initialization (ZERO warm start!)
    torch.manual_seed(42)
    model = make_model(cfg).to(DEVICE)
    return model, loader, cfg


def make_spatial_block_keep(shape, gen, p_keep=0.5, cell=64, device=DEVICE):
    B, _, H, W = shape
    gh = (H // cell) + 2
    gw = (W // cell) + 2
    out = torch.zeros(shape, dtype=torch.bool, device=device)
    for i in range(B):
        grid = (torch.rand((1, gh, gw), device=device, generator=gen) < p_keep)
        up = grid.repeat_interleave(cell, dim=1).repeat_interleave(cell, dim=2)
        oy = int(torch.randint(0, cell, (1,), device=device, generator=gen).item())
        ox = int(torch.randint(0, cell, (1,), device=device, generator=gen).item())
        out[i] = up[:, oy:oy + H, ox:ox + W]
    return out


def make_scribble_masks(P_rdg, full, valid, scribble_mode, gen):
    """Returns (P_scribble, U_scribble) where U = valid & ~P (NO background/supervision distinction!)."""
    if scribble_mode == 'half_centerline':
        keep = make_spatial_block_keep(P_rdg.shape, gen, p_keep=0.50, cell=64, device=P_rdg.device)
        P = P_rdg & keep
    elif scribble_mode == 'full_centerline':
        P = P_rdg
    elif scribble_mode == 'partial_letter20':
        keep = make_spatial_block_keep(full.shape, gen, p_keep=0.20, cell=48, device=full.device)
        P = full & valid & keep
    else:
        raise ValueError(scribble_mode)
    U = valid & (~P)
    return P, U


def dilate_mask(mask_b, radius=12):
    k = 2 * radius + 1
    return F.max_pool2d(mask_b.float(), kernel_size=k, stride=1, padding=radius) > 0.5


def bce_ls(logits, targets, valid_mask, ls=0.5):
    if not valid_mask.any():
        return logits.sum() * 0.0
    t = targets.float() * (1.0 - ls) + 0.5 * ls  # ls=0.5 -> [0.25, 0.75] matching villa!
    return F.binary_cross_entropy_with_logits(logits.float()[valid_mask], t[valid_mask])


def soft_dice_loss(logits, targets, valid_mask, eps=1.0):
    if not valid_mask.any():
        return logits.sum() * 0.0
    p = torch.sigmoid(logits.float())[valid_mask]
    t = targets.float()[valid_mask]
    inter = (p * t).sum()
    return 1.0 - (2.0 * inter + eps) / (p.sum() + t.sum() + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['oracle_fullmask', 'naive_bce', 'collar_bce', 'collar_bce_bal', 'kiryo_nnpu', 'scribble_pu', 'scribble_pu_batch'])
    ap.add_argument('--scribble', default='half_centerline', choices=['half_centerline', 'full_centerline', 'partial_letter20'])
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--tag_suffix', default='')
    args = ap.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    tag = f'{args.arm}_{args.scribble}_s{args.steps}{args.tag_suffix}'
    ckpt_dir = os.path.join(HERE, 'ckpt')
    log_dir = os.path.join(HERE, 'logs')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logf = os.path.join(log_dir, f'{tag}.log')

    def log(msg):
        s = f"{time.strftime('%H:%M:%S')} {msg}"
        print(s, flush=True)
        with open(logf, 'a', encoding='utf-8') as f:
            f.write(s + '\n')

    model, loader, cfg = build_exam1_model_and_loader()
    pad_to = BASE.configured_input_pad_depth(cfg)
    blanks = FastBlank0172Pool(cfg, pad_to, n_pool=256, seed=42)

    # Calibrate alpha over 25 batches using the exact scribble generator
    it = iter(loader)
    gen_cal = torch.Generator(device='cpu')
    gen_cal.manual_seed(1234)
    hn = hd = 0.0
    for _ in range(25):
        b = next(it)
        tgt = BASE.unpack_ink_and_sup(b)
        P_rdg, U_rdg, _, full = BASE.extract_centerlines_and_masks(tgt, b['geom'])
        P_s, _ = make_scribble_masks(P_rdg, full, P_rdg | U_rdg, args.scribble, gen_cal)
        hn += float((full & (P_rdg | U_rdg)).sum())
        hd += float(P_s.sum())
    alpha = hn / max(hd, 1.0)
    log(f'[EXAM1] tag={tag} | alpha={alpha:.3f} (eff={alpha*RHO:.3f}) | lr={args.lr}')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = min(300, max(50, args.steps // 10))
    lr_fn = lambda s: (s + 1) / warmup if s < warmup else 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * (s - warmup) / max(1, args.steps - warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(42)

    pu_mode = 'kiryo' if args.arm == 'kiryo_nnpu' else ('no_ascent_batch' if args.arm == 'scribble_pu_batch' else 'no_ascent')
    pu_loss = BrierNNPULoss(y_lo=Y_LO, y_hi=Y_HI, rho=RHO, min_pos_px=30, min_unlabeled_px=50, mode=pu_mode)

    t0 = time.time()
    blk_hist = []
    cl_tot = tiles_tot = 0
    model.train()

    for step in range(1, args.steps + 1):
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
            P_rdg, U_rdg, _, full = BASE.extract_centerlines_and_masks(tgt, geom)
            valid = P_rdg | U_rdg
            lo = logits_of(model, img)

            if args.arm == 'oracle_fullmask':
                l_mb = bce_ls(lo, full, valid, ls=0.5) + 0.25 * soft_dice_loss(lo, full, valid)
                w_mb = W_INK * ((s1 - s0) / 8.0)
            elif args.arm == 'naive_bce':
                P_s, _ = make_scribble_masks(P_rdg, full, valid, args.scribble, gen)
                l_mb = bce_ls(lo, P_s, valid, ls=0.5)
                w_mb = W_INK * ((s1 - s0) / 8.0)
            elif args.arm == 'collar_bce':
                P_s, _ = make_scribble_masks(P_rdg, full, valid, args.scribble, gen)
                collar = dilate_mask(P_s, radius=12) & (~P_s)
                eff_valid = valid & (~collar)
                l_mb = bce_ls(lo, P_s, eff_valid, ls=0.5)
                w_mb = W_INK * ((s1 - s0) / 8.0)
            elif args.arm == 'collar_bce_bal':
                P_s, _ = make_scribble_masks(P_rdg, full, valid, args.scribble, gen)
                collar = dilate_mask(P_s, radius=12) & (~P_s)
                eff_valid = valid & (~collar)
                t_sm = P_s.float() * 0.5 + 0.25
                pw = torch.where(P_s, torch.full_like(t_sm, alpha), torch.ones_like(t_sm))
                l_mb = (F.binary_cross_entropy_with_logits(lo.float(), t_sm, reduction='none') * pw)[eff_valid].mean()
                w_mb = W_INK * ((s1 - s0) / 8.0)
            else:  # kiryo_nnpu / scribble_pu / scribble_pu_batch
                P_s, U_s = make_scribble_masks(P_rdg, full, valid, args.scribble, gen)
                l_mb, st = pu_loss(lo, P_s, U_s, alpha=alpha)
                cl_tot += int(st['n_clamped'])
                tiles_tot += int(st['n_used'])
                w_mb = W_INK * (st['n_used'] / 8.0) if st['n_used'] > 0 else 0.0

            if w_mb > 0.0:
                (l_mb * w_mb).backward()
                lv += float(l_mb.item()) * w_mb

        lb = logits_of(model, bimg.to(DEVICE, non_blocking=True))
        if args.arm in ('oracle_fullmask', 'naive_bce', 'collar_bce', 'collar_bce_bal'):
            l_blk = bce_ls(lb[:, 0], torch.zeros_like(bmask, device=DEVICE), bmask.to(DEVICE), ls=0.5)
            bm = float(torch.sigmoid(lb.float()[:, 0]).detach()[bmask.to(DEVICE)].mean())
        else:
            l_blk, bm = certain_blank_brier_loss(lb, bmask.to(DEVICE), y_lo=Y_LO)
        (l_blk * W_BLK).backward()
        lv += float(l_blk.item()) * W_BLK
        blk_hist.append(bm)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step == 1 or step % 200 == 0 or step == args.steps:
            log(
                f'step {step:4d}/{args.steps} | loss={lv:.5f} | blk={np.mean(blk_hist[-10:]):.3f} '
                f'| cl={100.0*cl_tot/max(tiles_tot,1):.1f}% | lr={sched.get_last_lr()[0]:.2e} '
                f'| {(time.time()-t0)/60.0:.1f}m'
            )

    v3p = os.path.join(R, 'release', 'reader-v4-v3xe1', 'checkpoints', 'reader-v3.pth')
    v3_ck = torch.load(v3p, map_location='cpu', weights_only=False)
    out_p = os.path.join(ckpt_dir, f'{tag}.pth')
    torch.save({
        'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'config': v3_ck['config'],
        'step': args.steps,
        'exam1_meta': vars(args) | {'alpha': alpha},
    }, out_p + '.tmp')
    os.replace(out_p + '.tmp', out_p)
    log(f'[DONE] Saved {out_p} in {(time.time()-t0)/60.0:.1f}m')


if __name__ == '__main__':
    main()
