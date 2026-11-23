"""ScribblePU Exam 2A — Real-World Unseen-Protocol Scroll Adaptation (0009B 133208 -> 125754 + 140325).

Takes our Exam 1 3-scroll base model (trained ONLY on 0139N + 1667H + Paris4H — has NEVER seen the
116 keV / 8.64 um 0009B scan protocol!) and adapts it using ONLY ~5-6% positive-only scribbles on
ONE winding of 0009B (20250919133208, ZERO background masks!) + 1:1 source replay + 0172 certain-blanks.

Evaluates on:
  - 20250919125754 (unseen 0009B winding #1)
  - 20250919140325 (unseen 0009B winding #2)
  - pherc0814-46527 (unseen Scroll 4 forgetting check!)
  - w047_0_512 + blank_1024_5120 (FA80 / calibration check!)

Matched Arms (`--arm`):
  - oracle_fullmask : 100% full mask on 133208 (upper bound!)
  - naive_bce       : ~6% scribbles on 133208 as positive, rest as negative
  - collar_bce      : ~6% scribbles + 12-px ignored collar, rest as negative
  - kiryo_nnpu      : ~6% scribbles + vanilla Kiryo per-tile ascent Brier-nnPU
  - scribble_pu     : ~6% scribbles + our No-Ascent Batch Brier-nnPU (ZERO background masks!)
"""
import os
import sys
import json
import time
import random
import argparse
import logging
from pathlib import Path
import numpy as np
import cv2
import zarr
import tifffile
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

R = r'C:\Users\nikox\Documents\Vesuvius'
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
EX1 = os.path.join(REPO, 'experiments', 'exam1_from_scratch')
EX3 = os.path.join(REPO, 'experiments', 'exam3_reader_pu')
GEM = os.path.join(REPO, 'experiments', 'common')  # shared helper kt_nnpu.py

for p in (REPO, EX1, EX3, GEM):
    if p not in sys.path:
        sys.path.insert(0, p)

import kt_nnpu as BASE
import r1bn_common as R1BN
import read_1a as RA
import ys2
from v8ft_eval import auc as auc_sub
from train_bhand import logits_of
from train_exam1 import (
    FastBlank0172Pool, build_exam1_model_and_loader,
    make_scribble_masks, dilate_mask, bce_ls, soft_dice_loss,
)
from eval_exam1 import load_0814_labels
from scribble_pu.losses import BrierNNPULoss, certain_blank_brier_loss
from koine_machines.data.ink_dataset import _normalize_image_crop
import koine_machines.inference.infer as INF

DEVICE = torch.device('cuda')
Y_LO, Y_HI, RHO = 0.25, 0.82, 0.96
P009B_DIR = os.path.join(R, 'data', 'p009b')


class Target0009BSampler:
    """Samples 384x384 tiles from 0009B 20250919133208 in the EXACT CT frame with robust_mad normalization."""

    def __init__(self, cfg, pad_to, seg='20250919133208', seed=42):
        self.cfg = cfg
        self.pad_to = pad_to
        self.rng = np.random.default_rng(seed)
        sv = zarr.open(os.path.join(P009B_DIR, seg, 'sv864.zarr'), mode='r')
        self.sv = sv['0'] if hasattr(sv, 'keys') and '0' in sv else sv
        T_raw = np.load(os.path.join(P009B_DIR, seg, 'ink24_on864.npy')).astype(np.float32)
        dy, dx = R1BN.offsets()['0009B']
        self.T_ct = ndi.shift(T_raw, (-dy, -dx), order=1, mode='constant', cval=0.0, prefilter=False)
        self.valid_ct = ndi.shift((T_raw > 0).astype(np.float32), (-dy, -dx), order=1, mode='constant', cval=0.0, prefilter=False) > 0.999
        self.full_ink = self.valid_ct & (self.T_ct >= 140.0)
        # Precompute 1-px EDT medial ridge (exact same cv2 ridge logic as kt_nnpu.py; fast, zero skimage dep!)
        dt = cv2.distanceTransform(self.full_ink.astype(np.uint8), cv2.DIST_L2, 5)
        mx = cv2.dilate(dt, np.ones((5, 5), np.uint8))
        self.ridge = self.full_ink & (dt >= 2.0) & (dt >= mx - 0.5)
        # Precompute valid crop centers with ink
        Hh, W = self.full_ink.shape
        self.coords = []
        for y in range(256, Hh - 384 - 256, 128):
            for x in range(256, W - 384 - 256, 128):
                if self.full_ink[y:y + 384, x:x + 384].sum() >= 600 and self.valid_ct[y:y + 384, x:x + 384].mean() >= 0.85:
                    self.coords.append((y, x))
        print(f'[0009B-133208] {len(self.coords)} ink-bearing tiles ready | shift=({-dy:.1f},{-dx:.1f})', flush=True)

    def batch(self, n=4):
        D_, Hh, W = self.sv.shape
        imgs, fulls, ridges, valids = [], [], [], []
        for _ in range(n):
            y0, x0 = self.coords[int(self.rng.integers(0, len(self.coords)))]
            y0 = int(np.clip(y0 + int(self.rng.integers(-64, 65)), 64, Hh - 384 - 64))
            x0 = int(np.clip(x0 + int(self.rng.integers(-64, 65)), 64, W - 384 - 64))
            d0 = max(0, min(D_ - 17, (D_ - 17) // 2 + int(self.rng.integers(-2, 3))))
            vol = np.asarray(self.sv[d0:d0 + 17, y0:y0 + 384, x0:x0 + 384], dtype=np.float32)
            vn = _normalize_image_crop(vol, self.cfg)
            fi = self.full_ink[y0:y0 + 384, x0:x0 + 384].copy()
            ri = self.ridge[y0:y0 + 384, x0:x0 + 384].copy()
            vi = self.valid_ct[y0:y0 + 384, x0:x0 + 384].copy()
            k = int(self.rng.integers(0, 4))
            if k:
                vn = np.rot90(vn, k, axes=(1, 2)); fi = np.rot90(fi, k); ri = np.rot90(ri, k); vi = np.rot90(vi, k)
            if self.rng.random() < 0.5:
                vn = vn[:, :, ::-1]; fi = fi[:, ::-1]; ri = ri[:, ::-1]; vi = vi[:, ::-1]
            if self.rng.random() < 0.5:
                vn = vn[:, ::-1, :]; fi = fi[::-1, :]; ri = ri[::-1, :]; vi = vi[::-1, :]
            imgs.append(np.ascontiguousarray(vn))
            fulls.append(np.ascontiguousarray(fi))
            ridges.append(np.ascontiguousarray(ri))
            valids.append(np.ascontiguousarray(vi))
        img_t = BASE.center_pad_input_depth(torch.from_numpy(np.stack(imgs)[:, None]), self.pad_to)
        full_t = torch.from_numpy(np.stack(fulls)[:, None])
        ridge_t = torch.from_numpy(np.stack(ridges)[:, None])
        valid_t = torch.from_numpy(np.stack(valids)[:, None])
        return img_t, full_t, ridge_t, valid_t


def eval_adapted(tag, ckpt_path):
    od = os.path.join(HERE, 'reads', tag)
    os.makedirs(od, exist_ok=True)
    EV = os.path.join(R, 'data', 'v8ft', 'eval')
    jobs = [
        ('0009B_125754', os.path.join(EV, 'v2', '0009B_125754_sv.zarr'), 'forward'),
        ('0009B_140325', os.path.join(P009B_DIR, '20250919140325', 'sv864.zarr'), 'forward'),
        ('pherc0814-46527', os.path.join(R, 'data', 'teacher', 'p0814_iso9.zarr'), 'forward'),
        ('w047_0_512', os.path.join(EV, 'v2', 'w047_0_512_sv.zarr'), 'forward'),
        ('blank_1024_5120', os.path.join(EV, 'v2', 'blank_1024_5120_sv.zarr'), 'forward'),
    ]
    need = [j for j in jobs if not os.path.exists(os.path.join(od, f'{j[0]}.tif'))]
    if need:
        argv = [
            need[0][1], str(ckpt_path), os.path.join(od, '_probe.tif'),
            '--overlap', '0.5', '--blend-mode', 'hann', '--batch-size', '8',
            '--no-compile', '--direction', 'forward', '--num-workers', '2',
        ]
        args = INF.normalize_inference_paths(INF.parse_args(argv))
        cm = INF.configure_model(args)
        cm, dev, _ = INF.prepare_model_for_inference(args=args, configured_model=cm)
        logging.getLogger().setLevel(logging.WARNING)
        for name, sv, direc in need:
            INF.infer_single_zarr(args=args, configured_model=cm, device=dev,
                                  input_zarr=str(sv), output_tiff=Path(os.path.join(od, f'{name}.tif')),
                                  layer_direction=direc)
        del cm
        torch.cuda.empty_cache()

    M = {name: tifffile.imread(os.path.join(od, f'{name}.tif')).astype(np.float32) for name, _, _ in jobs}
    res = {}

    # 1. 0009B_125754
    mk9 = dict(np.load(os.path.join(RA.SU, '0009B_125754_masks.npz')))
    lab9, msk9 = mk9['lab'].astype(bool), mk9['msk'].astype(bool)
    m9 = M['0009B_125754']
    hh, ww = min(lab9.shape[0], m9.shape[0]), min(lab9.shape[1], m9.shape[1])
    M9 = ys2.masks(ys2.base(lab9[:hh, :ww]), msk9[:hh, :ww] & (m9[:hh, :ww] > 0), RA.SHIFT_0009B)
    s9 = ys2.score(m9[:hh, :ww], M9, 256, 50)
    res.update(b125_GATE=s9['GATE'], b125_M4=s9['M4'], b125_M1=s9['M1'], b125_rAUC=round(float(auc_sub(m9[:hh, :ww][M9['ridge']], m9[:hh, :ww][M9['far']])), 4))

    # 2. 0009B_140325 (unseen winding #2!)
    T14 = np.load(os.path.join(P009B_DIR, '20250919140325', 'ink24_on864.npy')).astype(np.float32)
    m14 = M['0009B_140325']
    h14, w14 = min(T14.shape[0], m14.shape[0]), min(T14.shape[1], m14.shape[1])
    lab14 = (T14[:h14, :w14] >= 140.0)
    msk14 = (T14[:h14, :w14] > 0) & (m14[:h14, :w14] > 0)
    M14 = ys2.masks(ys2.base(lab14), msk14, RA.SHIFT_0009B)
    s14 = ys2.score(m14[:h14, :w14], M14, 256, 50)
    res.update(b140_GATE=s14['GATE'], b140_M4=s14['M4'], b140_M1=s14['M1'], b140_rAUC=round(float(auc_sub(m14[:h14, :w14][M14['ridge']], m14[:h14, :w14][M14['far']])), 4))

    # 3. 0814H forgetting check
    P814, S814, V814 = load_0814_labels()
    d814 = RA.score_dev_map(P814, S814, V814, M['pherc0814-46527'], do_align=False)
    res.update(s814_GATE=d814['all']['GATE'], s814_M4=d814['all']['M4'])

    # 4. w047 + blank_1024
    mw = dict(np.load(os.path.join(RA.SU, 'w047_0_512_masks.npz')))
    mb = dict(np.load(os.path.join(RA.SU, 'blank_1024_5120_masks.npz')))
    pos, bval = mw['pos'].astype(bool), mb['valid'].astype(bool)
    thr = float(np.percentile(M['w047_0_512'][pos], 20))
    res.update(FA80=round(float((M['blank_1024_5120'][bval] >= thr).mean()), 4), thr=thr)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_ckpt', required=True)
    ap.add_argument('--arm', required=True, choices=['base_eval', 'oracle_fullmask', 'naive_bce', 'collar_bce', 'collar_bce_bal', 'kiryo_nnpu', 'scribble_pu'])
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--alpha_mult', type=float, default=1.0)
    ap.add_argument('--tag_prefix', default='adapt0009b')
    args = ap.parse_args()

    os.makedirs(os.path.join(HERE, 'ckpt'), exist_ok=True)
    if args.arm == 'base_eval':
        btag = f'{args.tag_prefix}_base'
        res = eval_adapted(btag, args.base_ckpt)
        print(f'=== {btag} ===\n' + json.dumps(res, indent=2), flush=True)
        with open(os.path.join(HERE, f'{args.tag_prefix}_base_eval.json'), 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2)
        return

    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model, loader, cfg = build_exam1_model_and_loader()
    ck = torch.load(args.base_ckpt, map_location='cpu', weights_only=False)
    sd = ck['model'] if 'model' in ck else ck
    model.load_state_dict({k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in sd.items()}, strict=True)
    pad_to = BASE.configured_input_pad_depth(cfg)

    tgt_sampler = Target0009BSampler(cfg, pad_to, seg='20250919133208', seed=42)
    blanks = FastBlank0172Pool(cfg, pad_to, n_pool=128, seed=42)

    # Calibrate alpha on 0009B 133208 scribbles (50% of 1-px ridges ~ 5-6% of ink!)
    gen_c = torch.Generator(device='cpu'); gen_c.manual_seed(77)
    hn = hd = 0.0
    for _ in range(12):
        _, ft, rt, vt = tgt_sampler.batch(4)
        Ps, _ = make_scribble_masks(rt, ft, vt, 'half_centerline', gen_c)
        hn += float((ft & vt).sum()); hd += float(Ps.sum())
    alpha_tgt = (hn / max(hd, 1.0)) * args.alpha_mult
    print(f'[EXAM2A] arm={args.arm} | alpha_0009B={alpha_tgt:.2f} (mult={args.alpha_mult:.2f}) | steps={args.steps}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_fn = lambda s: (s + 1) / 40.0 if s < 40 else 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * (s - 40) / max(1, args.steps - 40)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    gen = torch.Generator(device=DEVICE); gen.manual_seed(42)
    pu_loss = BrierNNPULoss(y_lo=Y_LO, y_hi=Y_HI, rho=RHO, mode=('kiryo' if args.arm == 'kiryo_nnpu' else 'no_ascent_batch'))

    it = iter(loader)
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        # 1. Target scroll 0009B (4 tiles, positive-only scribbles!)
        t_img, t_full, t_rdg, t_val = tgt_sampler.batch(4)
        t_img, t_full, t_rdg, t_val = t_img.to(DEVICE), t_full.to(DEVICE), t_rdg.to(DEVICE), t_val.to(DEVICE)

        # 2. Source replay (4 tiles from 3-scroll base!)
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        s_img = BASE.center_pad_input_depth(b['image'][:4].float(), pad_to).to(DEVICE)
        s_tgt = BASE.unpack_ink_and_sup(b)[:4].to(DEVICE)
        s_geom = b['geom'][:4].to(DEVICE)
        sP, sU, _, sfull = BASE.extract_centerlines_and_masks(s_tgt, s_geom)
        sval = sP | sU

        # 3. 0172 certain-blanks (2 tiles)
        bimg, bmask = blanks.batch(2)

        opt.zero_grad(set_to_none=True)
        lo_t = logits_of(model, t_img)
        if args.arm == 'oracle_fullmask':
            l_tgt = bce_ls(lo_t, t_full, t_val, ls=0.5) + 0.25 * soft_dice_loss(lo_t, t_full, t_val)
        elif args.arm == 'naive_bce':
            Ps, _ = make_scribble_masks(t_rdg, t_full, t_val, 'half_centerline', gen)
            l_tgt = bce_ls(lo_t, Ps, t_val, ls=0.5)
        elif args.arm == 'collar_bce':
            Ps, _ = make_scribble_masks(t_rdg, t_full, t_val, 'half_centerline', gen)
            collar = dilate_mask(Ps, radius=12) & (~Ps)
            l_tgt = bce_ls(lo_t, Ps, t_val & (~collar), ls=0.5)
        elif args.arm == 'collar_bce_bal':
            Ps, _ = make_scribble_masks(t_rdg, t_full, t_val, 'half_centerline', gen)
            collar = dilate_mask(Ps, radius=12) & (~Ps)
            vm = t_val & (~collar)
            t_sm = Ps.float() * 0.5 + 0.25
            pw = torch.where(Ps, torch.full_like(t_sm, alpha_tgt), torch.ones_like(t_sm))
            l_tgt = (F.binary_cross_entropy_with_logits(lo_t.float(), t_sm, reduction='none') * pw)[vm].mean()
        else:
            Ps, Us = make_scribble_masks(t_rdg, t_full, t_val, 'half_centerline', gen)
            l_tgt, _ = pu_loss(lo_t, Ps, Us, alpha=alpha_tgt)
        (l_tgt * 0.45).backward()

        lo_s = logits_of(model, s_img)
        if args.arm in ('oracle_fullmask', 'naive_bce', 'collar_bce', 'collar_bce_bal'):
            l_src = bce_ls(lo_s, sfull, sval, ls=0.5) + 0.25 * soft_dice_loss(lo_s, sfull, sval)
        else:
            # Source replay on Brier scale matching PU loss
            ps = torch.sigmoid(lo_s.float())
            l_src = torch.where(sfull, (ps - Y_HI) ** 2, (ps - Y_LO) ** 2)[sval].mean()
        (l_src * 0.45).backward()

        lb = logits_of(model, bimg.to(DEVICE))
        if args.arm in ('oracle_fullmask', 'naive_bce', 'collar_bce', 'collar_bce_bal'):
            l_blk = bce_ls(lb[:, 0], torch.zeros_like(bmask, device=DEVICE), bmask.to(DEVICE), ls=0.5)
        else:
            l_blk, _ = certain_blank_brier_loss(lb, bmask.to(DEVICE), y_lo=Y_LO)
        (l_blk * 0.10).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step == 1 or step % 150 == 0 or step == args.steps:
            print(f'  step {step:3d}/{args.steps} | l_tgt={float(l_tgt):.4f} l_src={float(l_src):.4f} | {time.time()-t0:.0f}s', flush=True)

    mult_s = '' if abs(args.alpha_mult - 1.0) < 1e-6 else f'_am{int(round(args.alpha_mult*100))}'
    tag = f'{args.tag_prefix}_{args.arm}{mult_s}_s{args.steps}'
    out_ck = os.path.join(HERE, 'ckpt', f'{tag}.pth')
    torch.save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'config': ck['config'], 'step': args.steps}, out_ck)
    res = eval_adapted(tag, out_ck)
    print(f'=== {tag} ===\n' + json.dumps(res, indent=2), flush=True)
    with open(os.path.join(HERE, f'{tag}.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
