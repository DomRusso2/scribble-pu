"""Kill Test #1: Brier-nnPU Centerline-Trained 9 um Ink Reader (err + sean Nov 19 callout).
Trains V3's exact drop-in architecture (warm-started from long256 ckpt_030000, V3's own parent!)
with pure Kiryo et al. Non-Negative Positive-Unlabeled (nnPU) Brier risk using ONLY thin medial
centerlines P + unlabeled U + centerline-estimated prior pi_U (NO Dice, NO gap weights, NO margins!).
Evaluates via official koine_machines.inference.infer on dev + unseen 0841_ag1 + public 0306B
(ZERO Table P touches, ZERO writes inside Vesuvius/).
"""
import os, sys, json, time, copy, importlib.util
import numpy as np
import cv2
import zarr
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import ndimage as ndi

R = r'C:\Users\nikox\Documents\Vesuvius'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')  # local outputs of this helper
os.makedirs(OUT, exist_ok=True)

# Vendor + retrain paths (strictly read-only imports!)
for p in (
    os.path.join(R, 'vendor', 'villa-ink', 'ink-detection'),
    os.path.join(R, 'vendor', 'villa-ink', 'vesuvius', 'src'),
    os.path.join(R, 'data', 'readerv4', 'retrain'),
    os.path.join(R, 'data', 'readerv4', 'yardstick_v2'),
    os.path.join(R, 'vendor'),
):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512,garbage_collection_threshold:0.8')

import timm  # noqa: F401
import launch_1bn as LBN
import launch_1a as LA
import r1a_common as C
import r1bn_common as N
import read_1a as RA
import ys2
import v2_run as v2r
from v8ft_eval import auc as auc_sub
from koine_machines.models.make_model import make_model
from koine_machines.models.input_padding import center_pad_input_depth, configured_input_pad_depth
from koine_machines.training.samplers import FixedScrollPriorStratifiedBatchSampler

DEVICE = torch.device('cuda')
torch.backends.cudnn.benchmark = True

F_RC = 2
F_NATIVE = 64
Y_LO = 0.24   # ~61/255 (matches V3 blank median 63, zero uint8 tie crush!)
Y_HI = 0.85   # ~217/255 (matches V3 ink p99 ~209, zero over-brightening!)

_NAT_MAP = {
    '0009B_nat': ('0009B', 'p009b'),
    '0139_nat': ('0139', 'v2'),
    '0814_nat': ('0814', 'v2'),
    '0500P2_nat': ('0500P2', 'v2'),
}


class NNPUDataset(LBN.Geom1bNDataset):
    """Self-contained zero-disk-geom dataset:
    - Hand tiles: returns zero geom (is_nat=0); exact EDT medial ridge is extracted on the fly post-aug!
    - Native teacher tiles: crops mmap'd 2.4 um teacher npy with 12-px halo, shifts by (-dy,-dx) into
      the EXACT CT frame (matching prep_1bn.py), masks interpolated borders & flat CT pixels.
    """

    def __init__(self, config, *a, **kw):
        super().__init__(config, *a, **kw)
        self._off = N.offsets()
        self._tmaps = {}

    def __getstate__(self):
        st = super().__getstate__()
        st['_tmaps'] = {}
        return st

    def geom_for(self, patch):
        seg = patch.segment
        d = self._cfg_ds[seg.dataset_idx]
        z0, y0, x0, z1, y1, x1 = (int(v) for v in patch.bbox)
        hp, wp = y1 - y0, x1 - x0
        g = np.zeros((5, 1, hp, wp), np.float32)
        if d.get('detail_kind') != 'native_teacher':
            return torch.from_numpy(g)

        g[2] = float(F_NATIVE)
        grp = d['sampling_scroll']
        scroll, kind = _NAT_MAP[grp]
        name = seg.segment_relpath
        key = (grp, name)
        if key not in self._tmaps:
            self._tmaps[key] = np.load(N.teacher_path(kind, scroll, name), mmap_mode='r')
        T = self._tmaps[key]
        Hh, W = T.shape
        dy, dx = self._off[scroll]
        halo = 12
        # Build padded window around [y0-halo:y1+halo, x0-halo:x1+halo]
        win = np.zeros((hp + 2 * halo, wp + 2 * halo), np.float32)
        wy0, wy1 = y0 - halo, y1 + halo
        wx0, wx1 = x0 - halo, x1 + halo
        sy0, sy1 = max(0, wy0), min(Hh, wy1)
        sx0, sx1 = max(0, wx0), min(W, wx1)
        if sy1 > sy0 and sx1 > sx0:
            win[sy0 - wy0:sy1 - wy0, sx0 - wx0:sx1 - wx0] = np.asarray(T[sy0:sy1, sx0:sx1], dtype=np.float32)
            sh = ndi.shift(win, (-dy, -dx), order=1, mode='constant', cval=0.0, prefilter=False)
            vs = ndi.shift((win > 0).astype(np.float32), (-dy, -dx), order=1, mode='constant', cval=0.0, prefilter=False)
            tc = sh[halo:halo + hp, halo:halo + wp]
            vc = vs[halo:halo + hp, halo:halo + wp] > 0.999
            tc = np.where(vc, np.clip(np.rint(tc), 0.0, 255.0), 0.0)
            g[0, 0] = tc
        return torch.from_numpy(g)


def build_model_and_loader(init_ckpt, batch_size=10):
    cfg = json.load(open(os.path.join(R, 'data', 'readerv4', 'retrain', '1bn_smoke.json')))
    cfg['out_dir'] = OUT
    cfg['batch_size'] = batch_size
    cfg['dataloader_workers'] = 4
    cfg['checkpoint'] = init_ckpt

    # Instantiate self-contained NNPUDataset (strictly excludes w029, w031, 0009B_125754, 0841, 0306B, 1447!)
    LA._D.InkDataset = NNPUDataset
    shared_ds = NNPUDataset(cfg, do_augmentations=False)
    ds = NNPUDataset(
        cfg,
        do_augmentations=True,
        patches=shared_ds.training_patches,
        segments=shared_ds.segments,
    )
    sampler = FixedScrollPriorStratifiedBatchSampler(
        shared_ds.training_patches,
        cfg['datasets'],
        batch_quotas=cfg['fixed_scroll_prior']['target_batch_counts'],
        batch_size=batch_size,
        seed=42,
    )
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True)

    # Build exact V3 model architecture (100% drop-in compatible with koine_machines!)
    model = make_model(cfg)
    ck = torch.load(init_ckpt, map_location='cpu', weights_only=False)
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    # Strip module./_orig_mod. if present
    clean_sd = {}
    for k, v in sd.items():
        nk = k.replace('_orig_mod.', '').replace('module.', '')
        clean_sd[nk] = v
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    print(f'[INIT] Loaded {os.path.basename(init_ckpt)} | missing={len(missing)} unexpected={len(unexpected)}', flush=True)
    assert len(missing) == 0, f'Missing keys: {missing[:5]}'
    return model.to(DEVICE), loader, cfg


def unpack_ink_and_sup(batch, device=None):
    ink = batch['inklabels'] if 'inklabels' in batch else batch['ink']
    if ink.dim() == 5:
        ink = (torch.amax(ink, dim=2) > 0).float()
    else:
        ink = (ink > 0.5).float()
    if 'supervision_mask' in batch:
        sup = batch['supervision_mask']
        if sup.dim() == 5:
            sup = (torch.amax(sup, dim=2) > 0).float()
        else:
            sup = (sup > 0.5).float()
    else:
        sup = torch.ones_like(ink[:, 0:1])
    tgt = torch.cat([ink[:, 0:1], sup[:, 0:1]], dim=1)
    if device is not None:
        tgt = tgt.to(device, non_blocking=True)
    return tgt


def extract_centerlines_and_masks(target, geom):
    """Extract thin sure-positive centerlines P and unlabeled set U (hand + native).
    - Hand tiles: exact EDT medial ridge of binary ink mask (1-2 px spine, dilated 1 px strictly inside ink).
    - Native tiles: top crest T_ct >= 220 of CT-aligned continuous 2.4 um teacher.
    NO hard-negative collars, NO gap masks, NO margin zones!
    Returns: P (B,1,H,W bool), U (B,1,H,W bool), is_nat (B,1,1,1 bool), full_ink_ref (for prior calibration ONLY).
    """
    dev = target.device
    g = geom.float()
    if g.dim() == 5:
        g = g[:, :, 0]
    fl = g[:, 2:3].round().to(torch.int32)
    is_nat = ((fl & F_NATIVE) > 0).flatten(1).any(1).view(-1, 1, 1, 1)

    # Hand tiles: target[:, 0:1] = ink, target[:, 1:2] = supervision_mask
    ink_h = target[:, 0:1] > 0.5
    sup_h = (target[:, 1:2] > 0.5) & (~is_nat)

    # Exact continuous 1D Euclidean-distance medial ridge on hand tiles (rotation-invariant, <1 ms/batch)
    B, _, H, W = ink_h.shape
    ridge_np = np.zeros((B, 1, H, W), dtype=np.bool_)
    ink_np = (ink_h & sup_h).detach().cpu().numpy()
    nat_np = is_nat.view(-1).detach().cpu().numpy()
    k3 = np.ones((3, 3), np.uint8)
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    for i in range(B):
        if nat_np[i]:
            continue
        m = ink_np[i, 0].astype(np.uint8)
        if m.any():
            dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
            p = np.pad(dt, 1, mode='constant')
            c = p[1:-1, 1:-1]
            h = (c >= p[1:-1, :-2]) & (c >= p[1:-1, 2:]) & ((c > p[1:-1, :-2]) | (c > p[1:-1, 2:]))
            v = (c >= p[:-2, 1:-1]) & (c >= p[2:, 1:-1]) & ((c > p[:-2, 1:-1]) | (c > p[2:, 1:-1]))
            d1 = (c >= p[:-2, :-2]) & (c >= p[2:, 2:]) & ((c > p[:-2, :-2]) | (c > p[2:, 2:]))
            d2 = (c >= p[:-2, 2:]) & (c >= p[2:, :-2]) & ((c > p[:-2, 2:]) | (c > p[2:, :-2]))
            r = ((c >= 1.5) & (h | v | d1 | d2)) | ((c >= 0.9) & ((h.astype(np.int8) + v.astype(np.int8) + d1.astype(np.int8) + d2.astype(np.int8)) >= 2))
            dt_loc = cv2.dilate(dt, k15)
            rs = r & (dt >= 0.65 * dt_loc) & (m > 0)
            # Dilate 1 px strictly inside ink (3-px crisp continuous medial spine)
            ridge_np[i, 0] = (cv2.dilate(rs.astype(np.uint8), k3) > 0) & (m > 0)
    spine_h = torch.from_numpy(ridge_np).to(dev)

    # Native tiles: T_ct in g[:, 0:1] (0..255, 0 = invalid)
    T_ct = g[:, 0:1]
    valid_n = is_nat & (T_ct > 0)
    # Only the crispest peaks (T_ct >= 220) as sure-positive P; EVERYTHING else in valid_n is UNLABELED U!
    spine_n = valid_n & (T_ct >= 220.0)
    ink_n_ref = valid_n & (T_ct >= 140.0)

    valid = sup_h | valid_n
    P = spine_h | spine_n
    U = valid & (~P)
    full_ref = (ink_h & sup_h) | ink_n_ref
    return P, U, is_nat, full_ref


def brier_nnpu_loss(logits, P, U, is_nat, alpha_hand, alpha_nat, rho=0.80, beta=0.0):
    """Exact Kiryo et al. (NeurIPS 2017) Non-Negative Positive-Unlabeled Brier Risk.
    Strictly proper scoring rule whose exact Bayes-optimal minimizer is:
      p*(x) = Y_LO + (Y_HI - Y_LO) * P(ink | x).
    """
    p = torch.sigmoid(logits.float())
    l_pos = (p - Y_HI) ** 2    # loss if y = 1
    l_neg = (p - Y_LO) ** 2    # loss if y = 0

    B = logits.shape[0]
    losses = []
    n_clamped = 0
    neg_risks = []
    pis = []

    for i in range(B):
        Pi = P[i]
        Ui = U[i]
        nP = float(Pi.sum().item())
        nU = float(Ui.sum().item())
        if nU < 50:
            continue
        if nP < 30:
            # Pure blank tile -> exact pi_U = 0.0 -> pure negative Brier loss
            losses.append(l_neg[i][Ui].mean())
            pis.append(0.0)
            continue

        alpha = (alpha_nat if bool(is_nat[i].item()) else alpha_hand) * rho
        alpha = max(alpha, 1.05)
        nV = nP + nU
        frac_U = nU / nV
        pi_u = min(0.55, max(0.0, (alpha - 1.0) * nP / nU))
        pi_tot = min(0.65, max(0.01, alpha * nP / nV))
        pis.append(pi_u)

        R_p_pos = l_pos[i][Pi].mean()
        R_p_neg = l_neg[i][Pi].mean()
        R_u_neg = l_neg[i][Ui].mean()
        R_neg = R_u_neg - pi_u * R_p_neg
        neg_risks.append(float(R_neg.item()))

        if float(R_neg.item()) >= beta:
            # Exact unbiased PU risk with frac_U factor -> exact Bayes optimum!
            losses.append(pi_tot * R_p_pos + frac_U * R_neg)
        else:
            # Kiryo et al. Algorithm 1 non-negative gradient ascent step!
            n_clamped += 1
            losses.append(-frac_U * R_neg)

    loss = torch.stack(losses).mean() if losses else logits.sum() * 0.0
    stats = {
        'n_clamped': n_clamped,
        'neg_risk': float(np.mean(neg_risks)) if neg_risks else 0.0,
        'pi_u': float(np.mean(pis)) if pis else 0.0,
        'p_P': float(p[P].mean().item()) if P.any() else 0.5,
        'p_U': float(p[U].mean().item()) if U.any() else 0.5,
    }
    return loss, stats


def train_nnpu(steps=800, rho=0.80, init_name='long256'):
    if init_name == 'long256':
        init_ckpt = os.path.join(R, 'data', 'v2ctx_v2_256_long256', 'train', 'ckpt_030000.pth')
    else:
        init_ckpt = os.path.join(R, 'release', 'reader-v4-v3xe1', 'checkpoints', 'reader-v3.pth')

    model, loader, cfg = build_model_and_loader(init_ckpt, batch_size=10)
    input_pad_depth_to = configured_input_pad_depth(cfg)
    print(f'[PAD] configured_input_pad_depth={input_pad_depth_to}', flush=True)

    # Step 0: Measure physical stroke-to-centerline ratio alpha ONCE from first 25 batches (err's exact recipe!)
    it = iter(loader)
    h_num, h_den, n_num, n_den = 0.0, 0.0, 0.0, 0.0
    t_cal = time.time()
    cached_batches = []
    for bi in range(25):
        batch = next(it)
        if bi == 0:
            print('[BATCH KEYS]', {k: tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__ for k, v in batch.items()}, flush=True)
        cached_batches.append(batch)
        P, U, is_nat, full_ref = extract_centerlines_and_masks(unpack_ink_and_sup(batch), batch['geom'])
        hm = ~is_nat
        nm = is_nat
        h_num += float((full_ref & hm).sum().item()); h_den += float((P & hm).sum().item())
        n_num += float((full_ref & nm).sum().item()); n_den += float((P & nm).sum().item())
    alpha_hand = h_num / max(h_den, 1.0)
    alpha_nat = n_num / max(n_den, 1.0)
    print(f'[PRIOR CALIB] Measured from 25 batches in {time.time()-t_cal:.1f}s: alpha_hand={alpha_hand:.3f}, alpha_nat={alpha_nat:.3f} (rho={rho})', flush=True)

    # Parameter groups: lower LR on encoder + depth_fusion, higher on decoder
    enc_params, dec_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith('network.shared_encoder.') or n.startswith('depth_fusion.'):
            enc_params.append(p)
        else:
            dec_params.append(p)
    opt = torch.optim.AdamW([
        {'params': enc_params, 'lr': 3e-5},
        {'params': dec_params, 'lr': 1.5e-4},
    ], weight_decay=1e-4)
    def lr_fn(s):
        if s < 80:
            return (s + 1) / 80.0
        prog = (s - 80) / max(1, steps - 80)
        return 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ema_loss, ema_pP, ema_pU = None, None, None
    clamp_tot = 0

    for step in range(1, steps + 1):
        if step <= len(cached_batches):
            batch = cached_batches[step - 1]
        else:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

        img_all = center_pad_input_depth(batch['image'].float(), input_pad_depth_to)
        tgt_all = unpack_ink_and_sup(batch)
        geom_all = batch['geom']

        # Micro-batch 2x5 (100% mathematically identical with InstanceNorm2d; keeps VRAM ~8 GB, zero WDDM thrash!)
        opt.zero_grad(set_to_none=True)
        lv = 0.0
        st_acc = {'n_clamped': 0, 'neg_risk': [], 'pi_u': [], 'p_P': [], 'p_U': []}
        for s0 in range(0, img_all.shape[0], 5):
            s1 = min(img_all.shape[0], s0 + 5)
            img = img_all[s0:s1].to(DEVICE, non_blocking=True)
            tgt = tgt_all[s0:s1].to(DEVICE, non_blocking=True)
            geom = geom_all[s0:s1].to(DEVICE, non_blocking=True)
            P, U, is_nat, _ = extract_centerlines_and_masks(tgt, geom)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(img)
                logits = out['ink'] if isinstance(out, dict) else out
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                if logits.dim() == 5:
                    logits = logits[:, :, 0]
            loss_mb, st_mb = brier_nnpu_loss(logits, P, U, is_nat, alpha_hand, alpha_nat, rho=rho, beta=0.0)
            w_mb = (s1 - s0) / float(img_all.shape[0])
            (loss_mb * w_mb).backward()
            lv += float(loss_mb.item()) * w_mb
            st_acc['n_clamped'] += st_mb['n_clamped']
            st_acc['neg_risk'].append(st_mb['neg_risk'])
            st_acc['pi_u'].append(st_mb['pi_u'])
            st_acc['p_P'].append(st_mb['p_P'])
            st_acc['p_U'].append(st_mb['p_U'])
        st = {
            'n_clamped': st_acc['n_clamped'],
            'neg_risk': float(np.mean(st_acc['neg_risk'])),
            'pi_u': float(np.mean(st_acc['pi_u'])),
            'p_P': float(np.mean(st_acc['p_P'])),
            'p_U': float(np.mean(st_acc['p_U'])),
        }
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        ema_loss = lv if ema_loss is None else 0.9 * ema_loss + 0.1 * lv
        ema_pP = st['p_P'] if ema_pP is None else 0.9 * ema_pP + 0.1 * st['p_P']
        ema_pU = st['p_U'] if ema_pU is None else 0.9 * ema_pU + 0.1 * st['p_U']
        clamp_tot += st['n_clamped']

        if step in (1, 20, 50) or step % 100 == 0:
            dt = time.time() - t0
            vram = torch.cuda.max_memory_allocated() / 1e9
            print(f'[nnPU] step {step:4d}/{steps} | loss={ema_loss:.5f} | p_P={ema_pP:.3f} p_U={ema_pU:.3f} | neg_r={st["neg_risk"]:+.4f} clamp={st["n_clamped"]}/10 | pi_u={st["pi_u"]:.3f} | vram={vram:.1f}GB | {step/dt:.2f} it/s | {dt:.0f}s', flush=True)

    # Save exact koine_machines-compatible checkpoint!
    v3_ck = torch.load(os.path.join(R, 'release', 'reader-v4-v3xe1', 'checkpoints', 'reader-v3.pth'), map_location='cpu', weights_only=False)
    save_cfg = copy.deepcopy(v3_ck['config'])
    ckpt_path = os.path.join(OUT, f'nnpu_{init_name}_rho{int(rho*100)}_s{steps}.pth')
    torch.save({'model': model.state_dict(), 'config': save_cfg, 'step': steps,
                'nnpu_meta': {'init': init_name, 'rho': rho, 'alpha_hand': alpha_hand, 'alpha_nat': alpha_nat}}, ckpt_path)
    print(f'[nnPU] Saved {ckpt_path} in {time.time()-t0:.1f}s (total tile clamps={clamp_tot}/{steps*10})', flush=True)
    del model, opt, loader, cached_batches
    torch.cuda.empty_cache()
    return ckpt_path


def run_eval(ckpt_path, tag='nnpu_l256'):
    """Official koine_machines inference + exact score_1bn.py metrics + contrast-curve control + visual PNG!"""
    t0 = time.time()
    rd_dir = os.path.join(OUT, 'reads', tag)
    os.makedirs(rd_dir, exist_ok=True)

    EV = os.path.join(R, 'data', 'v8ft', 'eval')
    AG1 = '20260220214732-auto_grown_20260220144552'
    E1 = json.load(open(C.E1_CFG))
    jobs = [
        ('w047_0_512', os.path.join(EV, 'v2', 'w047_0_512_sv.zarr'), 'forward'),
        ('blank_1024_5120', os.path.join(EV, 'v2', 'blank_1024_5120_sv.zarr'), 'forward'),
        ('0009B_125754', os.path.join(EV, 'v2', '0009B_125754_sv.zarr'), 'forward'),
        ('pherc1667-w029', E1['datasets'][5]['surface_volume_paths']['pherc1667-w029'], 'forward'),
        ('0841_ag1', os.path.join(R, 'data', 'v2', '0841', AG1, 'sv.zarr'), 'forward'),
        ('p306', os.path.join(R, 'data', 'v3l', 'p306', 'sv24.zarr'), 'reverse'),
    ]

    # Use official koine_machines infer_single_zarr (exact same code path as Reader V3!)
    import koine_machines.inference.infer as I
    from pathlib import Path
    import logging
    argv = [jobs[0][1], str(ckpt_path), os.path.join(rd_dir, '_probe.tif'), '--overlap', '0.5', '--blend-mode', 'hann',
            '--batch-size', '8', '--no-compile', '--direction', 'forward', '--num-workers', '2']
    args = I.normalize_inference_paths(I.parse_args(argv))
    cm = I.configure_model(args)
    cm, dev, _ = I.prepare_model_for_inference(args=args, configured_model=cm)
    logging.getLogger().setLevel(logging.WARNING)

    preds = {}
    for name, sv, direc in jobs:
        out_p = Path(os.path.join(rd_dir, f'{name}.tif'))
        ts = time.time()
        if not out_p.exists():
            I.infer_single_zarr(args=args, configured_model=cm, device=dev, input_zarr=str(sv),
                                output_tiff=out_p, layer_direction=direc)
        preds[name] = tifffile.imread(str(out_p)).astype(np.float32)
        print(f'[INFER] {name:16s} ({direc:7s}) shape={preds[name].shape} in {time.time()-ts:.1f}s', flush=True)

    # Load V3 reference maps for side-by-side scoring & V3+gamma2 contrast control!
    v3 = {
        'w047_0_512': tifffile.imread(os.path.join(EV, 'long384', 'w047_0_512_raw.tif')).astype(np.float32),
        'blank_1024_5120': tifffile.imread(os.path.join(EV, 'long384', 'blank_1024_5120_raw.tif')).astype(np.float32),
        '0009B_125754': tifffile.imread(os.path.join(EV, 'long384', '0009B_125754_raw.tif')).astype(np.float32),
        '0841_ag1': tifffile.imread(os.path.join(R, 'data', 'readerv4', 'fl', 'reads', f'0841_{AG1}_v3.tif')).astype(np.float32),
        'p306': tifffile.imread(os.path.join(R, 'data', 'v3l', 'p306', 'v3_reverse.tif')).astype(np.float32),
    }

    # Exact score_1bn.py masks
    mw = dict(np.load(os.path.join(RA.SU, 'w047_0_512_masks.npz')))
    mb = dict(np.load(os.path.join(RA.SU, 'blank_1024_5120_masks.npz')))
    pos, neg, bval = mw['pos'].astype(bool), mw['neg'].astype(bool), mb['valid'].astype(bool)
    core = pos & (ndi.distance_transform_edt(pos) >= 8)
    P29, S29, V29, _ = RA.dev_labels('pherc1667-w029')
    y0_29, y1_29, x0_29, x1_29 = RA.crop_box(S29 | V29, P29.shape)
    B29 = ys2.base(P29[y0_29:y1_29, x0_29:x1_29])
    mk9 = dict(np.load(os.path.join(RA.SU, '0009B_125754_masks.npz')))

    # 0841_ag1 outside Table P buffer
    d841 = os.path.join(R, 'data', 'v2', '0841', AG1); hd = os.path.join(d841, 'human')
    sh841 = zarr.open(os.path.join(d841, 'sv.zarr'), mode='r')['0'].shape[1:]
    ink841 = v2r.mask_on_canvas(zarr.open(os.path.join(hd, 'inklabels.zarr'), mode='r')['0'], sh841)
    rev841 = np.zeros(sh841, bool)
    for nm in ('supervision.zarr', 'validation.zarr'):
        if os.path.exists(os.path.join(hd, nm)):
            rev841 |= v2r.mask_on_canvas(zarr.open(os.path.join(hd, nm), mode='r')['0'], sh841)
    ex841 = np.zeros(sh841, bool)
    ex841[0:1401 + 256, 1720 - 256:3121 + 256] = True
    val841 = rev841 & ~ex841 & (v3['0841_ag1'][:sh841[0], :sh841[1]] > 0) & (preds['0841_ag1'][:sh841[0], :sh841[1]] > 0)
    M841 = ys2.masks(ys2.base(ink841 & rev841), val841)

    # 0306B papyrologist tracing + E2 boundaries
    spec = importlib.util.spec_from_file_location('fe', os.path.join(R, 'data', 'readerv4', 'fl_exam.py'))
    fe = importlib.util.module_from_spec(spec); spec.loader.exec_module(fe)
    p306_npz = np.load(os.path.join(R, 'data', 'v3l', 'p306', 'ink_canvas.npz'))
    p306_ink = p306_npz['ink'].astype(bool) | p306_npz['red'].astype(bool)
    dt306 = ndi.distance_transform_edt(~p306_ink)
    p306_bg20 = (dt306 <= 80) & (dt306 > 20)
    p306_gap = (cv2.morphologyEx(p306_ink.astype(np.uint8), cv2.MORPH_CLOSE, C.disk(57)) > 0) & (~p306_ink) & (dt306 >= 4) & (dt306 <= 20)
    g306 = json.load(open(os.path.join(R, 'data', 'readerv4', 'winners_grid', 'evidence_grid_line1_at_papyrologist_letters.json'), encoding='utf-8'))
    e306 = [x - 320 for x in g306['column_x_edges']]
    gy0, gy1 = 1499 - g306['band_y'][1], 1499 - g306['band_y'][0]

    def e2_gap(m):
        band = m[gy0:gy1, e306[0]:e306[-1]]; v = band > 0
        lo, hi = np.percentile(band[v], 50), np.percentile(band[v], 99.9)
        t = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1); t[~v] = np.nan
        P = np.nanmean(t, 0); e = [x - e306[0] for x in e306]; c = []
        for k in fe.BOUNDS:
            core_f = lambda a, b: np.nanmean(P[a + (b - a) // 4: b - (b - a) // 4])
            seg = P[max(0, e[k] - 17): e[k] + 18]
            den = (core_f(e[k - 1], e[k]) + core_f(e[k], e[k + 1])) / 2
            c.append(round(float(1 - np.nanmin(seg) / max(den, 1e-6)), 3))
        return round(float(np.mean(c)), 3), int(sum(x >= 0.5 for x in c))

    def logit_blend(a, b, wb=0.5):
        pa = np.clip(a.astype(np.float64) / 255.0, 0.5 / 255, 254.5 / 255)
        pb = np.clip(b.astype(np.float64) / 255.0, 0.5 / 255, 254.5 / 255)
        out = np.rint(255.0 / (1.0 + np.exp(-((1 - wb) * np.log(pa / (1 - pa)) + wb * np.log(pb / (1 - pb))))))
        out[(a == 0) | (b == 0)] = 0
        return out.astype(np.float32)

    arms = {
        'V3': v3,
        tag: preds,
        f'V3+{tag}_50': {k: logit_blend(v3[k], preds[k], 0.5) for k in v3},
    }

    res = {}
    for aname, M in arms.items():
        r = {}
        w, b = M['w047_0_512'], M['blank_1024_5120']
        thr = float(np.percentile(w[pos], 20))
        r.update(thr=round(thr, 1), FA80=round(float((b[bval] >= thr).mean()), 4), blank_med=round(float(np.median(b[bval])), 1),
                 cP20=round(float(np.percentile(w[core], 20)), 1), AUCcb=round(float(auc_sub(w[core], b[bval])), 4),
                 AUCpn=round(float(auc_sub(w[pos], w[neg])), 4))
        lab, msk = mk9['lab'].astype(bool), mk9['msk'].astype(bool); m9 = M['0009B_125754']
        hh, ww = min(lab.shape[0], m9.shape[0]), min(lab.shape[1], m9.shape[1])
        M9 = ys2.masks(ys2.base(lab[:hh, :ww]), msk[:hh, :ww] & (m9[:hh, :ww] > 0), RA.SHIFT_0009B)
        s9 = ys2.score(m9[:hh, :ww], M9, 256, 50)
        r.update(b9_M1=s9['M1'], b9_M1n=s9['M1_narrow'], b9_M2=s9['M2'], b9_M4=s9['M4'], b9_GATE=s9['GATE'],
                 b9_rAUC=round(float(auc_sub(m9[:hh, :ww][M9['ridge']], m9[:hh, :ww][M9['far']])), 4),
                 b9_p99=round(float(np.percentile(m9[m9 > 0], 99)), 1))
        g841 = M['0841_ag1'][:sh841[0], :sh841[1]]
        s841 = ys2.score(g841, M841, 256, 50)
        r.update(s841_M1n=s841['M1_narrow'], s841_M2=s841['M2'], s841_M4=s841['M4'], s841_GATE=s841['GATE'],
                 s841_rAUC=round(float(auc_sub(g841[M841['ridge']], g841[M841['far']])), 4))
        mp = M['p306']
        ph, pw = min(mp.shape[0], p306_ink.shape[0]), min(mp.shape[1], p306_ink.shape[1])
        mp_ = mp[:ph, :pw]
        e2m, e2n = e2_gap(mp)
        r.update(p306_AUC=round(float(auc_sub(mp_[p306_ink[:ph, :pw] & (mp_ > 0)], mp_[p306_bg20[:ph, :pw] & (mp_ > 0)])), 4),
                 p306_gapAUC=round(float(auc_sub(mp_[p306_ink[:ph, :pw] & (mp_ > 0)], mp_[p306_gap[:ph, :pw] & (mp_ > 0)])), 4),
                 p306_E2=e2m, p306_E2_ge05=e2n,
                 p306_p01=round(float(np.percentile(mp[mp > 0], 1)), 1))
        if 'pherc1667-w029' in M:
            d29 = RA.score_dev_map(P29, S29, V29, M['pherc1667-w029'], do_align=False)
            m29 = M['pherc1667-w029'][y0_29:y1_29, x0_29:x1_29]; A29 = (S29 | V29)[y0_29:y1_29, x0_29:x1_29] & (m29 > 0)
            MM = ys2.masks(B29, A29); band = A29 & (B29['dt_in'] >= 8) & (B29['dt_in'] < 0.45 * MM['h'])
            r.update(w29_M1=d29['all']['M1'], w29_M1n=d29['all']['M1_narrow'], w29_M2=d29['all']['M2'],
                     w29_GATE=d29['all']['GATE'], w29_vGATE=d29['val_only']['GATE'], w29_vM1n=d29['val_only']['M1_narrow'],
                     w29_band_mean=round(float(m29[band].mean()), 1),
                     w29_p99=round(float(np.percentile(M['pherc1667-w029'][M['pherc1667-w029'] > 0], 99)), 1))
        res[aname] = r
        print(f'[SCORE] {aname:16s} | FA80={r["FA80"]:.4f} cP20={r["cP20"]:5.1f} AUCcb={r["AUCcb"]:.4f} AUCpn={r["AUCpn"]:.4f} | b9: M1={r["b9_M1"]:.4f} M1n={r["b9_M1n"]:.4f} M4={r["b9_M4"]:.4f} GATE={r["b9_GATE"]:.4f} | 0841: M2={r["s841_M2"]:.4f} GATE={r["s841_GATE"]:.4f} | 0306B: AUC={r["p306_AUC"]:.4f} gapAUC={r["p306_gapAUC"]:.4f} E2={r["p306_E2"]} p01={r["p306_p01"]}', flush=True)
        if 'w29_M1' in r:
            print(f'   w029: M1={r["w29_M1"]} M1n={r["w29_M1n"]} M2={r["w29_M2"]} GATE={r["w29_GATE"]} vGATE={r["w29_vGATE"]} vM1n={r["w29_vM1n"]} band={r["w29_band_mean"]} p99={r["w29_p99"]} (V3 ref: .8221 / .7270 / .8815 / .9512 / .9285 / .3122 / 160.8 / 201)', flush=True)

    # Save visual side-by-side comparison PNG on PUBLIC 0306B (incl. V3+gamma2 contrast control!)
    ys_i, xs_i = np.nonzero(p306_ink)
    cy0, cy1 = max(0, int(ys_i.min()) - 60), min(p306_ink.shape[0], int(ys_i.max()) + 60)
    cx0, cx1 = max(0, int(xs_i.min()) - 60), min(p306_ink.shape[1], int(xs_i.max()) + 60)

    def tone_vis(img, gamma=1.0):
        sub = img[cy0:cy1, cx0:cx1]; v = sub > 0
        lo, hi = np.percentile(sub[v], [50, 99.9]) if v.any() else (0, 255)
        u = np.clip((sub - lo) / max(hi - lo, 1.0), 0, 1) ** gamma
        t = (255 * (1 - u)).astype(np.uint8); t[~v] = 225
        return np.flipud(t)

    panels = [
        ('Reader V3 (plain)', tone_vis(v3['p306'], 1.0)),
        ('Reader V3 + gamma=2.2 (contrast-curve control!)', tone_vis(v3['p306'], 2.2)),
        (f'Brier-nnPU Centerline ({tag}, ZERO gap weights / ZERO Dice)', tone_vis(preds['p306'], 1.0)),
        (f'V3 + {tag} (50/50 logit blend)', tone_vis(arms[f'V3+{tag}_50']['p306'], 1.0)),
        ('Papyrologist Tracing (Ground Truth)', np.flipud((255 - p306_ink[cy0:cy1, cx0:cx1].astype(np.uint8) * 230))),
    ]
    rows_im = []
    for title, im in panels:
        rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        bar = np.full((34, rgb.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(bar, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
        rows_im.append(np.vstack([bar, rgb]))
    sheet = np.vstack(rows_im)
    if sheet.shape[1] > 2400:
        s = 2400.0 / sheet.shape[1]
        sheet = cv2.resize(sheet, (2400, int(sheet.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(OUT, f'kt_nnpu_{tag}_0306b.png'), sheet)
    json.dump(res, open(os.path.join(OUT, f'kt_nnpu_{tag}_results.json'), 'w'), indent=2)
    print(f'[DONE] {tag} evaluated in {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    ckpt = train_nnpu(steps=800, rho=0.80, init_name='long256')
    run_eval(ckpt, tag='nnpu_l256')
