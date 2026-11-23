"""ZERO-EXAM 50-step Training-Batch-Only Drift & Gradient Diagnostic (scratch only!).
Compares exact C2 / E3_DRIFT behavior on the IDENTICAL 50 training batches as train_exam3.py:
  0. Baseline (frozen Kiryo ridge nnPU - reproduces the measured -0.18 dimming)
  1. Option A (SCAR: random 15% of full ink mask, edges included, Kiryo nnPU)
  2. Option B (Ridge P + No-Ascent ReLU: pi_tot * Rp_pos + frac_U * clamp(Ru_neg - pi_u * Rp_neg.detach(), min=0))
  3. Option B+ (Ridge P + Calibrated No-Ascent: same as B, with c_scar estimated from step-0 ratio E[l_neg(inkU)]/E[l_neg(P)] OR fixed 0.5)
  4. Option D (SAR-PU Prior-Calibrated Soft Brier on U: P -> Y_HI, U -> soft-EM target y_soft = Y_LO + (Y_HI - Y_LO) * sigmoid(z_u.detach() + tau) matching pi_u!)
  5. Option D2 (SAR-PU Neg-Weight Only on U: P -> Y_HI, U -> (1 - w_pos) * l_neg + 0.5 * w_pos * l_pos)
ZERO exam pixels read! ZERO files touched in Vesuvius/!
"""
import os, sys, copy, time, json, random
import numpy as np
import torch

R = r'C:\Users\nikox\Documents\Vesuvius'
GEM = os.path.dirname(os.path.abspath(__file__))  # helpers live next to this file
EX3 = os.path.join(R, 'data', 'scribblepu', 'exam3')
sys.path.insert(0, GEM)
sys.path.insert(0, EX3)

import kt_nnpu as BASE
from koine_machines.data.ink_dataset import _normalize_image_crop
import zarr
from scipy import ndimage as ndi

DEVICE = torch.device('cuda')
Y_LO, Y_HI, RHO = 0.25, 0.82, 0.96
W_BLK, W_INK_TILE = 1.0 / 12.0, 11.0 / 120.0
ENC_LR, DEC_LR, WD = 8e-6, 1.2e-4, 1e-4
CFG = json.load(open(os.path.join(R, 'data', 'readerv4', 'retrain', '1bn_smoke.json')))


def match_prior_tau(logits_u, target_pi, iters=16):
    lo, hi = -10.0, 10.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if float(torch.sigmoid(logits_u + mid).mean().item()) < target_pi:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def logits_of(model, img):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model(img)
        lo = out['ink'] if isinstance(out, dict) else out
        if isinstance(lo, (list, tuple)): lo = lo[0]
        if lo.dim() == 5: lo = lo[:, :, 0]
    return lo


def cache_50_steps(loader, pad_to, n_steps=50):
    """Cache 50 ink batches + 50 certain-blank pairs ONCE so every arm sees BIT-IDENTICAL batches!"""
    # Build certain-blank sampler matching train_exam3.py exactly
    v3 = json.load(open(os.path.join(R, 'data', 'v2ctx_long384', 'train_config.json')))
    d = v3['datasets'][12]
    segs = []
    for seg in d['segments']:
        sv = zarr.open(d['surface_volume_paths'][seg], mode='r'); sv = sv['0'] if hasattr(sv, 'keys') and '0' in sv else sv
        lp = os.path.join(d['segments_path'], seg)
        il = zarr.open(os.path.join(lp, seg + '_inklabels.zarr'), mode='r'); il = il['0'] if hasattr(il, 'keys') and '0' in il else il
        sm = zarr.open(os.path.join(lp, seg + '_supervision_mask.zarr'), mode='r'); sm = sm['0'] if hasattr(sm, 'keys') and '0' in sm else sm
        segs.append((seg, sv, il, sm))
    rng = np.random.default_rng(42)
    it = iter(loader)
    batches = []
    hn = hd = nn_ = nd = 0.0
    t0 = time.time()
    for st in range(n_steps):
        b = next(it)
        img = BASE.center_pad_input_depth(b['image'].float(), pad_to)
        tgt = BASE.unpack_ink_and_sup(b)
        geom = b['geom']
        if st < 25:
            P, U, is_nat, full = BASE.extract_centerlines_and_masks(tgt, geom)
            hm, nm = ~is_nat, is_nat
            hn += float((full & hm).sum()); hd += float((P & hm).sum())
            nn_ += float((full & nm).sum()); nd += float((P & nm).sum())
        # 2 certain-blank tiles
        bvols, bms = [], []
        for _ in range(2):
            for _t in range(200):
                seg, sv, il, sm = segs[int(rng.integers(0, len(segs)))]
                D, Hh, W = sv.shape
                d0 = max(0, min(D - 17, (D - 17) // 2 + int(rng.integers(-2, 3))))
                y0 = int(rng.integers(256, max(257, Hh - 384 - 256)))
                x0 = int(rng.integers(256, max(257, W - 384 - 256)))
                vol = np.asarray(sv[d0:d0 + 17, y0:y0 + 384, x0:x0 + 384], dtype=np.float32)
                if float((vol > 0).mean()) < 0.90 or float(vol.std()) < 5.0:
                    continue
                ink = np.asarray(il[:, y0 - 16:y0 + 400, x0 - 16:x0 + 400]).max(0) > 0
                sup = np.asarray(sm[:, y0 - 16:y0 + 400, x0 - 16:x0 + 400]).max(0) > 0
                dist = ndi.distance_transform_edt(~ink) if ink.any() else np.full(ink.shape, 1e9)
                clean = (sup & ~ink & (dist >= 16))[16:400, 16:400]
                if float(clean.mean()) < 0.30:
                    continue
                vol = _normalize_image_crop(vol, CFG)
                k = int(rng.integers(0, 4))
                if k:
                    vol = np.rot90(vol, k, axes=(1, 2)); clean = np.rot90(clean, k)
                if rng.random() < 0.5:
                    vol = vol[:, :, ::-1]; clean = clean[:, ::-1]
                if rng.random() < 0.5:
                    vol = vol[:, ::-1, :]; clean = clean[::-1, :]
                bvols.append(np.ascontiguousarray(vol)); bms.append(np.ascontiguousarray(clean))
                break
        bimg = BASE.center_pad_input_depth(torch.from_numpy(np.stack(bvols)[:, None]), pad_to)
        bmask = torch.from_numpy(np.stack(bms))
        batches.append((img, tgt, geom, bimg, bmask))
    a_h, a_n = hn / max(hd, 1.0), nn_ / max(nd, 1.0)
    print(f'[CACHE] 50 batches cached in {time.time()-t0:.1f}s | a_h={a_h:.3f} a_n={a_n:.3f}', flush=True)
    return batches, a_h, a_n


def run_arm(arm_name, base_sd, model_tmpl, ref_model, batches, a_h, a_n):
    model = copy.deepcopy(model_tmpl).to(DEVICE)
    model.load_state_dict(base_sd, strict=True)
    enc, dec = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (enc if (n.startswith('network.shared_encoder.') or n.startswith('depth_fusion.')) else dec).append(p)
    opt = torch.optim.AdamW([{'params': enc, 'lr': ENC_LR}, {'params': dec, 'lr': DEC_LR}], weight_decay=WD)
    lr_fn = lambda s: (s + 1) / 80.0 if s < 80 else 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * (s - 80) / 920.0))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    gen = torch.Generator(device=DEVICE); gen.manual_seed(42)

    clamps_h = clamps_n = 0
    blk_last = []
    t0 = time.time()
    model.train()
    for step, (img_all, tgt_all, geom_all, bimg, bmask) in enumerate(batches, 1):
        opt.zero_grad(set_to_none=True)
        for s0 in range(0, img_all.shape[0], 5):
            s1 = min(img_all.shape[0], s0 + 5)
            img = img_all[s0:s1].to(DEVICE, non_blocking=True)
            tgt = tgt_all[s0:s1].to(DEVICE, non_blocking=True)
            geom = geom_all[s0:s1].to(DEVICE, non_blocking=True)
            P_rdg, U_rdg, is_nat, full = BASE.extract_centerlines_and_masks(tgt, geom)
            valid = P_rdg | U_rdg

            if arm_name == 'OptA_SCAR15':
                # Textbook SCAR: random 15% of ALL labelled ink pixels (edges included!)
                rnd = torch.rand(full.shape, device=DEVICE, generator=gen)
                P = full & valid & (rnd < 0.15)
                U = valid & (~P)
                ah_use, an_use = 1.0 / 0.15, 1.0 / 0.15
            else:
                P, U = P_rdg, U_rdg
                ah_use, an_use = a_h, a_n

            lo = logits_of(model, img).float()
            p = torch.sigmoid(lo)
            lp = (p - Y_HI) ** 2
            ln = (p - Y_LO) ** 2
            losses = []
            for i in range(lo.shape[0]):
                Pi, Ui = P[i], U[i]
                nP, nU = float(Pi.sum().item()), float(Ui.sum().item())
                if nU < 50: continue
                if nP < 30:
                    losses.append(ln[i][Ui].mean()); continue
                is_n = bool(is_nat[i].item())
                alpha = max(1.05, (an_use if is_n else ah_use) * RHO)
                nV = nP + nU; frac_U = nU / nV
                pi_u = min(0.55, max(0.0, (alpha - 1.0) * nP / nU))
                pi_tot = min(0.65, max(0.01, alpha * nP / nV))

                Rp_pos = lp[i][Pi].mean()
                Rp_neg = ln[i][Pi].mean()
                Ru_neg = ln[i][Ui].mean()

                if arm_name in ('Old_Kiryo_Ridge', 'OptA_SCAR15'):
                    R_neg = Ru_neg - pi_u * Rp_neg
                    if float(R_neg.item()) >= 0.0:
                        losses.append(pi_tot * Rp_pos + frac_U * R_neg)
                    else:
                        if is_n: clamps_n += 1
                        else: clamps_h += 1
                        losses.append(-frac_U * R_neg)
                elif arm_name == 'OptB_NoAscent':
                    R_neg = Ru_neg - pi_u * Rp_neg.detach()
                    if float(R_neg.item()) < 0.0:
                        if is_n: clamps_n += 1
                        else: clamps_h += 1
                    losses.append(pi_tot * Rp_pos + frac_U * torch.clamp(R_neg, min=0.0))
                elif arm_name == 'OptD_SAR_EM':
                    with torch.no_grad():
                        zu = lo[i][Ui].detach()
                        tau = match_prior_tau(zu, pi_u)
                        w_pos = torch.sigmoid(zu + tau)
                    y_soft = Y_LO + (Y_HI - Y_LO) * w_pos
                    # Confirmed ridge P -> Y_HI (boosted to pi_tot so P stays strong!), U -> prior-calibrated soft Brier!
                    losses.append(pi_tot * Rp_pos + frac_U * ((p[i][Ui] - y_soft) ** 2).mean())
                elif arm_name == 'OptD2_SAR_NegWt':
                    with torch.no_grad():
                        zu = lo[i][Ui].detach()
                        tau = match_prior_tau(zu, pi_u)
                        w_neg = 1.0 - torch.sigmoid(zu + tau)
                    # Confirmed ridge P -> Y_HI; U -> ONLY background-weighted negative Brier (protects hidden ink, pushes counters/bg to Y_LO!)
                    losses.append(pi_tot * Rp_pos + frac_U * (w_neg * ln[i][Ui]).mean())
            loss_mb = torch.stack(losses).mean() if losses else lo.sum() * 0.0
            (loss_mb * ((s1 - s0) * W_INK_TILE)).backward()

        lb = logits_of(model, bimg.to(DEVICE, non_blocking=True)).float()
        pb = torch.sigmoid(lb)[:, 0]
        bm_dev = bmask.to(DEVICE)
        loss_blk = ((pb - Y_LO) ** 2)[bm_dev].mean()
        (loss_blk * W_BLK).backward()
        if step > 45:
            blk_last.append(float(pb.detach()[bm_dev].mean()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    # Exact E3_DRIFT evaluation on the last 5 training batches (batches[45:50]) vs frozen V3!
    model.eval()
    h_t, h_v, n_t, n_v, u_t, u_v, bg_t, bg_v = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for img_all, tgt_all, geom_all, _, _ in batches[45:50]:
            for s0 in range(0, img_all.shape[0], 5):
                img = img_all[s0:s0 + 5].to(DEVICE); tgt = tgt_all[s0:s0 + 5].to(DEVICE); geom = geom_all[s0:s0 + 5].to(DEVICE)
                P, U, is_nat, full = BASE.extract_centerlines_and_masks(tgt, geom); N_ = is_nat.expand_as(P)
                pt = torch.sigmoid(logits_of(model, img).float())
                pv = torch.sigmoid(logits_of(ref_model, img).float())
                mh, mn, mu, mbg = P & ~N_, P & N_, full & ~P, U & ~full
                if mh.any(): h_t.append(pt[mh].cpu()); h_v.append(pv[mh].cpu())
                if mn.any(): n_t.append(pt[mn].cpu()); n_v.append(pv[mn].cpu())
                if mu.any(): u_t.append(pt[mu].cpu()); u_v.append(pv[mu].cpu())
                if mbg.any(): bg_t.append(pt[mbg].cpu()); bg_v.append(pv[mbg].cpu())
    dh = float(torch.cat(h_t).median() - torch.cat(h_v).median())
    dn = float(torch.cat(n_t).median() - torch.cat(n_v).median())
    du = float(torch.cat(u_t).median() - torch.cat(u_v).median())
    dbg = float(torch.cat(bg_t).median() - torch.cat(bg_v).median())
    mh_t, mh_v = float(torch.cat(h_t).median()), float(torch.cat(h_v).median())
    mu_t, mu_v = float(torch.cat(u_t).median()), float(torch.cat(u_v).median())
    mbg_t, mbg_v = float(torch.cat(bg_t).median()), float(torch.cat(bg_v).median())
    print(f'{arm_name:16s} ({time.time()-t0:4.0f}s) | handP={mh_t:.3f} (d{dh:+.3f}) | inkU={mu_t:.3f} (d{du:+.3f}) | nonInk={mbg_t:.3f} (d{dbg:+.3f}) | natP_d={dn:+.3f} | blk={np.mean(blk_last):.3f} | clamp(H/N)={clamps_h:3d}/{clamps_n:3d}', flush=True)
    del model, opt; torch.cuda.empty_cache()


def main():
    v3p = os.path.join(R, 'release', 'reader-v4-v3xe1', 'checkpoints', 'reader-v3.pth')
    model, loader, cfg = BASE.build_model_and_loader(v3p, batch_size=10)
    pad_to = BASE.configured_input_pad_depth(cfg)
    base_sd = copy.deepcopy(model.state_dict())
    ref_model = copy.deepcopy(model).to(DEVICE).eval()
    for p in ref_model.parameters(): p.requires_grad_(False)
    batches, a_h, a_n = cache_50_steps(loader, pad_to, n_steps=50)
    for arm in ['Old_Kiryo_Ridge', 'OptA_SCAR15', 'OptB_NoAscent', 'OptD_SAR_EM', 'OptD2_SAR_NegWt']:
        run_arm(arm, base_sd, model.cpu(), ref_model, batches, a_h, a_n)


if __name__ == '__main__':
    main()
