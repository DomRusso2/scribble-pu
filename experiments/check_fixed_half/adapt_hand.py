"""Check H: adaptation from a FIXED set of lines (PREREG_TWO_CHECKS.md, frozen 2026-11-23 14:13:17).

Imports Exam 2's script (antigravity/scribble-pu/experiments/exam2_adaptation/adapt_0009b.py) and changes ONLY the
two things the prereg allows:
  (1) target tiles come from the traced crop only (tiles start inside the crop; valid = False outside the crop, so
      no pixel outside the crop enters any loss),
  (2) P is FIXED per arm (no per-step redraw):
        H1 = hand-drawn lines (arm dropped before any run; kept in the code for reuse)
        H2 = kt_nnpu 1-px ridge (pre-dilation) of the truth, ONE fixed random half of 64-px blocks in the crop;
             alpha measured from its labels (script control)
        H3 = full masks in the crop (Exam 2's oracle_fullmask arm code path)
Everything else (base model, 600 steps, lr 1e-4, AdamW + schedule, 4 target + 4 replay + 2 blank tiles, loss weights
.45/.45/.10, eval) is Exam 2's code, called unchanged.
Usage: python adapt_hand.py --arm H1|H2|H3 [--alpha A]
"""
import argparse
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch

EX2 = r'C:\Users\nikox\Documents\antigravity\scribble-pu\experiments\exam2_adaptation'
sys.path.insert(0, EX2)
import adapt_0009b as A2  # noqa: E402

HC = r'C:\Users\nikox\Documents\Vesuvius\data\scribblepu\hand_check'
BASE_CK = os.path.join(A2.REPO, 'experiments', 'exam1_from_scratch', 'ckpt', 'oracle_fullmask_half_centerline_s5000_lr1e3.pth')
A2.HERE = HC  # eval_adapted() writes reads under HERE/reads/<tag>


def kt_ridge_1px(m):
    """kt_nnpu.py L208-218 ridge before its 3x3 dilation (1-px continuous medial ridge)."""
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    p = np.pad(dt, 1, mode='constant')
    c = p[1:-1, 1:-1]
    h = (c >= p[1:-1, :-2]) & (c >= p[1:-1, 2:]) & ((c > p[1:-1, :-2]) | (c > p[1:-1, 2:]))
    v = (c >= p[:-2, 1:-1]) & (c >= p[2:, 1:-1]) & ((c > p[:-2, 1:-1]) | (c > p[2:, 1:-1]))
    d1 = (c >= p[:-2, :-2]) & (c >= p[2:, 2:]) & ((c > p[:-2, :-2]) | (c > p[2:, 2:]))
    d2 = (c >= p[:-2, 2:]) & (c >= p[2:, :-2]) & ((c > p[:-2, 2:]) | (c > p[2:, :-2]))
    r = ((c >= 1.5) & (h | v | d1 | d2)) | ((c >= 0.9) & ((h.astype(np.int8) + v.astype(np.int8) + d1.astype(np.int8) + d2.astype(np.int8)) >= 2))
    dt_loc = cv2.dilate(dt, k15)
    return r & (dt >= 0.65 * dt_loc) & (m > 0)


def thin_1px(mask):
    """Zhang-Suen thinning via OpenCV ximgproc if present, else a morphological skeleton."""
    m = (mask > 0).astype(np.uint8) * 255
    try:
        return cv2.ximgproc.thinning(m) > 0
    except AttributeError:
        skel = np.zeros_like(m)
        el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        cur = m.copy()
        while cur.any():
            er = cv2.erode(cur, el)
            op = cv2.dilate(er, el)
            skel |= cv2.subtract(cur, op)
            cur = er
        return skel > 0


class CropFixedSampler(A2.Target0009BSampler):
    def __init__(self, cfg, pad_to, fixedP, crop, seed=42):
        super().__init__(cfg, pad_to, seg='20250919133208', seed=seed)
        cy, cx, C = crop
        inside = np.zeros_like(self.valid_ct)
        inside[cy:cy + C, cx:cx + C] = True
        self.valid_ct = self.valid_ct & inside          # change (1): nothing outside the crop enters a loss
        self.full_ink = self.full_ink & inside
        self.ridge = fixedP & self.valid_ct             # change (2): the 'ridge' channel now carries the FIXED P
        self.coords = [(y, x) for (y, x) in self.coords if cy <= y <= cy + C - 384 and cx <= x <= cx + C - 384]
        print(f'[H] crop tiles: {len(self.coords)} | fixed P px in crop: {int(self.ridge.sum())}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['H1', 'H2', 'H3'])
    ap.add_argument('--alpha', type=float, default=None, help='H1: rule-of-thumb alpha (required for H1)')
    ap.add_argument('--steps', type=int, default=600)
    args = ap.parse_args()
    cy, cx, C = [int(v) for v in np.load(os.path.join(HC, 'crop_yx.npy'))]

    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model, loader, cfg = A2.build_exam1_model_and_loader()
    ck = torch.load(BASE_CK, map_location='cpu', weights_only=False)
    sd = ck['model'] if 'model' in ck else ck
    model.load_state_dict({k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in sd.items()}, strict=True)
    pad_to = A2.BASE.configured_input_pad_depth(cfg)

    probe = A2.Target0009BSampler(cfg, pad_to, seg='20250919133208', seed=42)
    full_ct, valid_ct = probe.full_ink, probe.valid_ct
    fixedP = np.zeros_like(full_ct)
    info = {}
    if args.arm == 'H1':
        lines = np.load(os.path.join(HC, 'user_lines_1px_crop.npy'))
        fixedP[cy:cy + C, cx:cx + C] = lines
        assert args.alpha is not None, 'H1 needs the rule-of-thumb alpha'
        alpha = float(args.alpha)
    elif args.arm == 'H2':
        crop_ink = (full_ct & valid_ct)[cy:cy + C, cx:cx + C].astype(np.uint8)
        rid = kt_ridge_1px(crop_ink)
        rng = np.random.default_rng(42)
        gh, gw = C // 64 + 2, C // 64 + 2
        grid = rng.random((gh, gw)) < 0.5
        up = np.repeat(np.repeat(grid, 64, axis=0), 64, axis=1)
        oy, ox = int(rng.integers(0, 64)), int(rng.integers(0, 64))
        keep = up[oy:oy + C, ox:ox + C]
        fixedP[cy:cy + C, cx:cx + C] = rid & keep
        alpha = float(crop_ink.sum()) / max(float(fixedP.sum()), 1.0)
    else:
        alpha = 1.0
    del probe
    tgt = CropFixedSampler(cfg, pad_to, fixedP, (cy, cx, C), seed=42)
    crop_ink_px = int((full_ct & valid_ct)[cy:cy + C, cx:cx + C].sum())
    info.update(arm=args.arm, alpha=alpha, fixedP_px=int(tgt.ridge.sum()), crop_ink_px=crop_ink_px,
                fixedP_frac_of_ink=float(tgt.ridge.sum()) / max(crop_ink_px, 1), n_tiles=len(tgt.coords))
    print('[H] ' + json.dumps(info), flush=True)
    blanks = A2.FastBlank0172Pool(cfg, pad_to, n_pool=128, seed=42)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_fn = lambda s: (s + 1) / 40.0 if s < 40 else 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * (s - 40) / max(1, args.steps - 40)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    pu_loss = A2.BrierNNPULoss(y_lo=A2.Y_LO, y_hi=A2.Y_HI, rho=A2.RHO, mode='no_ascent_batch')
    it = iter(loader)
    t0 = time.time()
    model.train()
    dev = A2.DEVICE
    for step in range(1, args.steps + 1):
        t_img, t_full, t_P, t_val = tgt.batch(4)
        t_img, t_full, t_P, t_val = t_img.to(dev), t_full.to(dev), t_P.to(dev), t_val.to(dev)
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        s_img = A2.BASE.center_pad_input_depth(b['image'][:4].float(), pad_to).to(dev)
        s_tgt = A2.BASE.unpack_ink_and_sup(b)[:4].to(dev)
        s_geom = b['geom'][:4].to(dev)
        sP, sU, _, sfull = A2.BASE.extract_centerlines_and_masks(s_tgt, s_geom)
        sval = sP | sU
        bimg, bmask = blanks.batch(2)

        opt.zero_grad(set_to_none=True)
        lo_t = A2.logits_of(model, t_img)
        if args.arm == 'H3':
            l_tgt = A2.bce_ls(lo_t, t_full, t_val, ls=0.5) + 0.25 * A2.soft_dice_loss(lo_t, t_full, t_val)
        else:
            Ps = t_P & t_val
            Us = t_val & (~Ps)
            l_tgt, _ = pu_loss(lo_t, Ps, Us, alpha=alpha)
        (l_tgt * 0.45).backward()
        lo_s = A2.logits_of(model, s_img)
        if args.arm == 'H3':
            l_src = A2.bce_ls(lo_s, sfull, sval, ls=0.5) + 0.25 * A2.soft_dice_loss(lo_s, sfull, sval)
        else:
            ps = torch.sigmoid(lo_s.float())
            l_src = torch.where(sfull, (ps - A2.Y_HI) ** 2, (ps - A2.Y_LO) ** 2)[sval].mean()
        (l_src * 0.45).backward()
        lb = A2.logits_of(model, bimg.to(dev))
        if args.arm == 'H3':
            l_blk = A2.bce_ls(lb[:, 0], torch.zeros_like(bmask, device=dev), bmask.to(dev), ls=0.5)
        else:
            l_blk, _ = A2.certain_blank_brier_loss(lb, bmask.to(dev), y_lo=A2.Y_LO)
        (l_blk * 0.10).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step == 1 or step % 150 == 0 or step == args.steps:
            print(f'  step {step:3d}/{args.steps} | l_tgt={float(l_tgt):.4f} l_src={float(l_src):.4f} | {time.time()-t0:.0f}s', flush=True)
    info['train_seconds'] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(HC, 'ckpt'), exist_ok=True)
    tag = f'hand_{args.arm}_s{args.steps}'
    out_ck = os.path.join(HC, 'ckpt', f'{tag}.pth')
    torch.save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'config': ck['config'], 'step': args.steps}, out_ck)
    res = A2.eval_adapted(tag, out_ck)
    res.update(info)
    print(f'=== {tag} ===\n' + json.dumps(res, indent=2), flush=True)
    json.dump(res, open(os.path.join(HC, f'{tag}.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
