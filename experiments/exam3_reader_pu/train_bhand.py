"""ScribblePU Exam 3 — Reader-PU (B-Hand: Hand-Only + 0172 Certain-Blanks + No-Ascent Brier-nnPU).

Trains V3's exact drop-in 2.5D UNet architecture (warm-started from reader-v3.pth) for 1,000 steps
using ONLY:
  - 8 hand-labeled ink tiles per batch (0139N:2, 0814H:1, 1667H:2, Paris4H:3; ZERO native teacher tiles,
    ZERO dev/Table-P tiles!) with 1-px medial centerlines P + unlabeled U,
  - 2 verified certain-blank 0172 tiles per batch (max33(ink)==0 & sup & edt>=16px, robust_mad p1/p99, NO Z-flip),
  - No-Ascent Brier-nnPU loss: pi_tot * Rp_pos + frac_U * clamp(Ru_neg - pi_u * Rp_neg, min=0.0),
    Y_LO=0.25, Y_HI=0.82, rho=0.96, w_ink=11/12, w_blk=1/12.

Saves strictly inside scribble-pu/experiments/exam3_reader_pu/ckpt/ (Vesuvius/ is strictly read-only!):
  - reader_pu_bhand_step500.pth
  - reader_pu_bhand_raw.pth (step 1,000)
  - reader_pu_bhand_soup65.pth (0.65 * raw + 0.35 * V3)
  - reader_pu_bhand_soup50.pth (0.50 * raw + 0.50 * V3)
"""
import os
import sys
import copy
import json
import time
import random
import numpy as np
import cv2
import zarr
import torch
from torch.utils.data import DataLoader
from scipy import ndimage as ndi

R = r'C:\Users\nikox\Documents\Vesuvius'
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
GEM = os.path.join(REPO, 'experiments', 'common')  # shared helper kt_nnpu.py

for p in (REPO, GEM):
    if p not in sys.path:
        sys.path.insert(0, p)

import kt_nnpu as BASE
import diag_sgd_vs_adamw_200 as S
from scribble_pu.losses import BrierNNPULoss, certain_blank_brier_loss
from koine_machines.models.make_model import make_model
from koine_machines.training.samplers import FixedScrollPriorStratifiedBatchSampler
from koine_machines.data.ink_dataset import _normalize_image_crop

DEVICE = torch.device('cuda')
STEPS = 1000
Y_LO, Y_HI, RHO = 0.25, 0.82, 0.96
W_BLK = 1.0 / 12.0
W_INK = 11.0 / 12.0
ENC_LR, DEC_LR, WD = 8e-6, 1.2e-4, 1e-4
MARGIN, MIN_CLEAN = 16, 0.30
HAND_QUOTAS = {'0139N': 2, '0814H': 1, '1667H': 2, 'Paris4H': 3}

CKPT_DIR = os.path.join(HERE, 'ckpt')
LOG_DIR = os.path.join(HERE, 'logs')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
LOGF = os.path.join(LOG_DIR, 'train_bhand.log')


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOGF, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def logits_of(model, img):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model(img)
        lo = out['ink'] if isinstance(out, dict) else out
        if isinstance(lo, (list, tuple)):
            lo = lo[0]
        if lo.dim() == 5:
            lo = lo[:, :, 0]
    return lo


class CertainBlank0172Sampler:
    """Verified certain-blank 0172 tiles: max33(ink)==0 & sup & edt>=16px, robust_mad p1/p99, NO Z-flip."""

    def __init__(self, cfg, pad_to, patch=384, seed=42):
        v3 = json.load(open(os.path.join(R, 'data', 'v2ctx_long384', 'train_config.json')))
        d = v3['datasets'][12]
        assert d['sampling_scroll'] == '0172'
        self.segs = []
        for seg in d['segments']:
            sv = zarr.open(d['surface_volume_paths'][seg], mode='r')
            sv = sv['0'] if hasattr(sv, 'keys') and '0' in sv else sv
            lp = os.path.join(d['segments_path'], seg)
            il = zarr.open(os.path.join(lp, seg + '_inklabels.zarr'), mode='r')
            il = il['0'] if hasattr(il, 'keys') and '0' in il else il
            sm = zarr.open(os.path.join(lp, seg + '_supervision_mask.zarr'), mode='r')
            sm = sm['0'] if hasattr(sm, 'keys') and '0' in sm else sm
            assert sv.shape == il.shape == sm.shape
            self.segs.append((seg, sv, il, sm))
        self.cfg = cfg
        self.pad_to = pad_to
        self.patch = patch
        self.rng = np.random.default_rng(seed)
        log(f'[BLANK-0172] {len(self.segs)} segments ready')

    def one(self):
        P, M = self.patch, MARGIN
        for _ in range(300):
            seg, sv, il, sm = self.segs[int(self.rng.integers(0, len(self.segs)))]
            D_, Hh, W = sv.shape
            d0 = max(0, min(D_ - 17, (D_ - 17) // 2 + int(self.rng.integers(-2, 3))))
            y0 = int(self.rng.integers(256, max(257, Hh - P - 256)))
            x0 = int(self.rng.integers(256, max(257, W - P - 256)))
            vol = np.asarray(sv[d0:d0 + 17, y0:y0 + P, x0:x0 + P], dtype=np.float32)
            if float((vol > 0).mean()) < 0.90 or float(vol.std()) < 5.0:
                continue
            ink = np.asarray(il[:, y0 - M:y0 + P + M, x0 - M:x0 + P + M]).max(0) > 0
            sup = np.asarray(sm[:, y0 - M:y0 + P + M, x0 - M:x0 + P + M]).max(0) > 0
            dist = ndi.distance_transform_edt(~ink) if ink.any() else np.full(ink.shape, 1e9)
            clean = (sup & ~ink & (dist >= M))[M:M + P, M:M + P]
            if float(clean.mean()) < MIN_CLEAN:
                continue
            vn = _normalize_image_crop(vol, self.cfg)
            k = int(self.rng.integers(0, 4))
            if k:
                vn = np.rot90(vn, k, axes=(1, 2))
                clean = np.rot90(clean, k)
            if self.rng.random() < 0.5:
                vn = vn[:, :, ::-1]
                clean = clean[:, ::-1]
            if self.rng.random() < 0.5:
                vn = vn[:, ::-1, :]
                clean = clean[::-1, :]
            return np.ascontiguousarray(vn), np.ascontiguousarray(clean)
        raise RuntimeError('Failed to sample certain-blank 0172 tile in 300 tries')

    def batch(self, n=2):
        out = [self.one() for _ in range(n)]
        bimg = torch.from_numpy(np.stack([o[0] for o in out])[:, None])
        bimg = BASE.center_pad_input_depth(bimg, self.pad_to)
        bmask = torch.from_numpy(np.stack([o[1] for o in out]))
        return bimg, bmask


def build_handonly_model_and_loader(v3p):
    cfg = json.load(open(os.path.join(R, 'data', 'readerv4', 'retrain', '1bn_smoke.json')))
    cfg['out_dir'] = HERE
    cfg['batch_size'] = 8
    cfg['dataloader_workers'] = 4
    cfg['checkpoint'] = v3p

    BASE.LA._D.InkDataset = BASE.NNPUDataset
    shared_ds = BASE.NNPUDataset(cfg, do_augmentations=False)
    hand_patches = [
        p for p in shared_ds.training_patches
        if cfg['datasets'][p.segment.dataset_idx].get('detail_kind') != 'native_teacher'
    ]
    ds = BASE.NNPUDataset(cfg, do_augmentations=True, patches=hand_patches, segments=shared_ds.segments)
    sampler = FixedScrollPriorStratifiedBatchSampler(
        hand_patches, cfg['datasets'],
        batch_quotas=HAND_QUOTAS, batch_size=8, seed=42,
    )
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True)

    model = make_model(cfg)
    ck = torch.load(v3p, map_location='cpu', weights_only=False)
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    clean_sd = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    assert len(missing) == 0, f'Missing keys: {missing[:5]}'
    log(f'[INIT] Loaded {os.path.basename(v3p)} | hand_patches={len(hand_patches)} | missing={len(missing)} unexp={len(unexpected)}')
    return model.to(DEVICE), loader, cfg, ck


def stroke_edge_mask(full_b, valid_b, max_dt=2.5):
    B, _, H, W = full_b.shape
    out = np.zeros((B, 1, H, W), dtype=np.bool_)
    f_np = (full_b & valid_b).detach().cpu().numpy()
    for i in range(B):
        m = f_np[i, 0].astype(np.uint8)
        if not m.any():
            continue
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        out[i, 0] = (m > 0) & (dt <= max_dt)
    return torch.from_numpy(out).to(full_b.device)


def eval_train_telemetry(model, ref_model, eval_batches):
    model.eval()
    h_t, h_v, e_t, e_v, u_t, u_v, g_t, g_v, bg_t, bg_v = ([] for _ in range(10))
    with torch.no_grad():
        for img_all, tgt_all, geom_all in eval_batches:
            for s0 in range(0, img_all.shape[0], 4):
                img = img_all[s0:s0 + 4].to(DEVICE)
                tgt = tgt_all[s0:s0 + 4].to(DEVICE)
                geom = geom_all[s0:s0 + 4].to(DEVICE)
                P, U, is_nat, full = BASE.extract_centerlines_and_masks(tgt, geom)
                gap = S.closing_gap_mask(full, P | U, is_nat)
                edge = stroke_edge_mask(full, P | U, max_dt=2.5)
                bg = U & (~full) & (~gap)
                pt = torch.sigmoid(logits_of(model, img).float())
                pv = torch.sigmoid(logits_of(ref_model, img).float())
                for at, av, m in [
                    (h_t, h_v, P),
                    (e_t, e_v, edge),
                    (u_t, u_v, full & ~P),
                    (g_t, g_v, gap),
                    (bg_t, bg_v, bg),
                ]:
                    if m.any():
                        at.append(pt[m].cpu())
                        av.append(pv[m].cpu())
    model.train()

    def st(at, av):
        mt, mv = float(torch.cat(at).median()), float(torch.cat(av).median())
        return round(mt, 4), round(mt - mv, 4)

    return {
        'handP': st(h_t, h_v),
        'edge': st(e_t, e_v),
        'inkU': st(u_t, u_v),
        'gap': st(g_t, g_v),
        'bg': st(bg_t, bg_v),
    }


def make_soup(raw_sd, v3_sd, w_raw):
    out = {}
    for k, v in raw_sd.items():
        if k in v3_sd and torch.is_floating_point(v):
            out[k] = (w_raw * v.float() + (1.0 - w_raw) * v3_sd[k].float()).to(v.dtype)
        else:
            out[k] = v.clone()
    return out


def save_ckpt(path, sd, v3_ck, step, meta):
    payload = {
        'model': sd,
        'config': v3_ck['config'],
        'step': step,
        'scribblepu_meta': meta,
    }
    torch.save(payload, path + '.tmp')
    os.replace(path + '.tmp', path)
    log(f'[SAVE] {path}')


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    v3p = os.path.join(R, 'release', 'reader-v4-v3xe1', 'checkpoints', 'reader-v3.pth')
    model, loader, cfg, v3_ck = build_handonly_model_and_loader(v3p)
    pad_to = BASE.configured_input_pad_depth(cfg)
    v3_sd_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    nbuf = [n for n, _ in model.named_buffers()]
    log(f'[BUFFERS] count={len(nbuf)} {nbuf[:6]}')

    ref_model = copy.deepcopy(model).to(DEVICE).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    blanks = CertainBlank0172Sampler(cfg, pad_to, patch=384, seed=42)

    # Calibrate alpha_hand over 25 batches + hold 8 fixed training-scroll batches for drift telemetry (never trained on in those exact crops)
    it = iter(loader)
    eval_batches = []
    hn = hd = 0.0
    t0 = time.time()
    for st in range(25):
        b = next(it)
        img = BASE.center_pad_input_depth(b['image'].float(), pad_to)
        tgt = BASE.unpack_ink_and_sup(b)
        geom = b['geom']
        P, U, is_nat, full = BASE.extract_centerlines_and_masks(tgt, geom)
        hn += float(full.sum())
        hd += float(P.sum())
        if st < 8:
            eval_batches.append((img, tgt, geom))
    alpha_hand = hn / max(hd, 1.0)
    log(f'[PRIOR] alpha_hand={alpha_hand:.3f} rho={RHO} eff_alpha={alpha_hand*RHO:.3f} ({time.time()-t0:.1f}s)')

    enc, dec = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (enc if (n.startswith('network.shared_encoder.') or n.startswith('depth_fusion.')) else dec).append(p)
    log(f'[PARAMS] enc={sum(p.numel() for p in enc)/1e6:.2f}M dec={sum(p.numel() for p in dec)/1e6:.2f}M')

    criterion = BrierNNPULoss(y_lo=Y_LO, y_hi=Y_HI, rho=RHO, min_pos_px=30, min_unlabeled_px=50, mode='no_ascent')
    opt = torch.optim.AdamW([{'params': enc, 'lr': ENC_LR}, {'params': dec, 'lr': DEC_LR}], weight_decay=WD)
    lr_fn = lambda s: (s + 1) / 80.0 if s < 80 else 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * (s - 80) / max(1, STEPS - 80)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    history = []
    blk_hist = []
    cl_tot = 0
    tiles_tot = 0
    t_train = time.time()
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
        loss_val = 0.0
        cl_step = 0
        neg_rs = []

        for s0 in range(0, img_all.shape[0], 4):
            s1 = min(img_all.shape[0], s0 + 4)
            img = img_all[s0:s1].to(DEVICE, non_blocking=True)
            tgt = tgt_all[s0:s1].to(DEVICE, non_blocking=True)
            geom = geom_all[s0:s1].to(DEVICE, non_blocking=True)
            P, U, _, _ = BASE.extract_centerlines_and_masks(tgt, geom)
            lo = logits_of(model, img)
            loss_mb, st = criterion(lo, P, U, alpha=alpha_hand)
            if st['n_used'] > 0:
                w_mb = W_INK * (st['n_used'] / 8.0)
                (loss_mb * w_mb).backward()
                loss_val += float(loss_mb.item()) * w_mb
                cl_step += int(st['n_clamped'])
                cl_tot += int(st['n_clamped'])
                tiles_tot += int(st['n_used'])
                neg_rs.append(st['neg_risk'])

        lb = logits_of(model, bimg.to(DEVICE, non_blocking=True))
        loss_blk, blk_mean = certain_blank_brier_loss(lb, bmask.to(DEVICE), y_lo=Y_LO)
        (loss_blk * W_BLK).backward()
        loss_val += float(loss_blk.item()) * W_BLK
        blk_hist.append(blk_mean)

        if not np.isfinite(loss_val):
            log(f'[WARN] Non-finite loss at step {step}; skipping step')
            opt.zero_grad(set_to_none=True)
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step == 1 or step % 50 == 0:
            log(
                f'step {step:4d}/{STEPS} | loss={loss_val:.5f} | blk={np.mean(blk_hist[-10:]):.3f} '
                f'| cl={cl_step}/8 (cum {100.0*cl_tot/max(tiles_tot,1):.1f}%) '
                f'| neg_r={np.mean(neg_rs) if neg_rs else 0.0:+.4f} | lr={sched.get_last_lr()[1]:.2e} '
                f'| {time.time()-t_train:.0f}s'
            )

        if step in (50, 100, 200, 400, 500, 600, 800, 1000):
            tel = eval_train_telemetry(model, ref_model, eval_batches)
            tel['step'] = step
            tel['blk_mean10'] = round(float(np.mean(blk_hist[-10:])), 4)
            tel['cl_pct'] = round(100.0 * cl_tot / max(tiles_tot, 1), 1)
            history.append(tel)
            log(
                f'  [TEL@{step:4d}] handP={tel["handP"][0]:.3f}({tel["handP"][1]:+.3f}) '
                f'| inkU={tel["inkU"][0]:.3f}({tel["inkU"][1]:+.3f}) '
                f'| edge={tel["edge"][0]:.3f}({tel["edge"][1]:+.3f}) '
                f'| gap={tel["gap"][0]:.3f}({tel["gap"][1]:+.3f}) '
                f'| bg={tel["bg"][0]:.3f}({tel["bg"][1]:+.3f})'
            )

        if step == 500:
            meta500 = dict(variant='bhand_no_ascent', y_lo=Y_LO, y_hi=Y_HI, rho=RHO, alpha_hand=alpha_hand,
                           w_blk=W_BLK, enc_lr=ENC_LR, dec_lr=DEC_LR, hand_quotas=HAND_QUOTAS, soup_w=1.0)
            sd500 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_ckpt(os.path.join(CKPT_DIR, 'reader_pu_bhand_step500.pth'), sd500, v3_ck, 500, meta500)

    raw_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    meta_base = dict(variant='bhand_no_ascent', y_lo=Y_LO, y_hi=Y_HI, rho=RHO, alpha_hand=alpha_hand,
                     w_blk=W_BLK, enc_lr=ENC_LR, dec_lr=DEC_LR, hand_quotas=HAND_QUOTAS)
    save_ckpt(os.path.join(CKPT_DIR, 'reader_pu_bhand_raw.pth'), raw_sd, v3_ck, STEPS, {**meta_base, 'soup_w': 1.0})
    save_ckpt(os.path.join(CKPT_DIR, 'reader_pu_bhand_soup65.pth'), make_soup(raw_sd, v3_sd_cpu, 0.65), v3_ck, STEPS, {**meta_base, 'soup_w': 0.65})
    save_ckpt(os.path.join(CKPT_DIR, 'reader_pu_bhand_soup50.pth'), make_soup(raw_sd, v3_sd_cpu, 0.50), v3_ck, STEPS, {**meta_base, 'soup_w': 0.50})

    with open(os.path.join(LOG_DIR, 'train_bhand_telemetry.json'), 'w', encoding='utf-8') as f:
        json.dump({'alpha_hand': alpha_hand, 'history': history}, f, indent=2)
    log(f'[DONE] Total training time: {(time.time()-t_train)/60.0:.1f} min')


if __name__ == '__main__':
    main()
