"""ZERO-EXAM 100-step Training-Batch-Only Learning & Drift Check (scratch only!).
Answers the review question: does SGD/low-LR AdamW ACTUALLY move weights, converge blanks, and lower training gaps
WITHOUT dimming handP or inkU (unlike AdamW 1.2e-4)?
Measures on cached training batches ONLY (ZERO exam data read, ZERO writes inside Vesuvius!):
  - dec_dW = ||theta_dec - theta_dec_V3||_2 (proves real learning, not a no-op!)
  - blk_first5 -> blk_last5 (proves blank anchor converges!)
  - handP drift, inkU drift, nonInk drift, AND train_gap (closing57 & ~full on hand training tiles!) drift!
"""
import os, sys, copy, time
import numpy as np
import cv2
import torch

R = r'C:\Users\nikox\Documents\Vesuvius'
GEM = os.path.dirname(os.path.abspath(__file__))  # helpers live next to this file
sys.path.insert(0, GEM)
import kt_nnpu as BASE
import diag_50step_pu_options as D

DEVICE = torch.device('cuda')
Y_LO, Y_HI, RHO = 0.25, 0.82, 0.96
W_BLK, W_INK_TILE = 1.0 / 12.0, 11.0 / 120.0
K57 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (57, 57))


def closing_gap_mask(full_b, valid_b, is_nat_b):
    """Inside-counter / narrow-gap pixels on HAND training tiles only (for training-only drift check)."""
    B, _, H, W = full_b.shape
    out = np.zeros((B, 1, H, W), dtype=np.bool_)
    f_np = full_b.detach().cpu().numpy()
    v_np = valid_b.detach().cpu().numpy()
    n_np = is_nat_b.view(-1).detach().cpu().numpy()
    for i in range(B):
        if n_np[i]: continue
        m = f_np[i, 0].astype(np.uint8)
        if not m.any(): continue
        clo = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K57) > 0
        out[i, 0] = clo & (~(m > 0)) & v_np[i, 0]
    return torch.from_numpy(out).to(full_b.device)


def run_arm(name, opt_kind, dec_lr, enc_lr, loss_kind, base_sd, model_tmpl, ref_model, batches, a_h, a_n, steps=100):
    model = copy.deepcopy(model_tmpl).to(DEVICE)
    model.load_state_dict(base_sd, strict=True)
    enc, dec = [], []
    dec_init = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if n.startswith('network.shared_encoder.') or n.startswith('depth_fusion.'):
                enc.append(p)
            else:
                dec.append(p)
                dec_init.append(p.detach().clone())
    if opt_kind == 'sgd':
        opt = torch.optim.SGD([{'params': enc, 'lr': enc_lr}, {'params': dec, 'lr': dec_lr}], momentum=0.9, weight_decay=3e-5)
    else:
        opt = torch.optim.AdamW([{'params': enc, 'lr': enc_lr}, {'params': dec, 'lr': dec_lr}], weight_decay=1e-4)
    lr_fn = lambda s: (s + 1) / 80.0 if s < 80 else 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * (s - 80) / 920.0))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    gen = torch.Generator(device=DEVICE); gen.manual_seed(42)

    t0 = time.time(); blk_f5, blk_l5 = [], []
    clamps_h = clamps_n = 0
    model.train()
    for step in range(1, steps + 1):
        img_all, tgt_all, geom_all, bimg, bmask = batches[(step - 1) % len(batches)]
        opt.zero_grad(set_to_none=True)
        for s0 in range(0, img_all.shape[0], 5):
            s1 = min(img_all.shape[0], s0 + 5)
            img = img_all[s0:s1].to(DEVICE, non_blocking=True)
            tgt = tgt_all[s0:s1].to(DEVICE, non_blocking=True)
            geom = geom_all[s0:s1].to(DEVICE, non_blocking=True)
            P_rdg, U_rdg, is_nat, full = BASE.extract_centerlines_and_masks(tgt, geom)
            valid = P_rdg | U_rdg
            if loss_kind == 'scar15':
                rnd = torch.rand(full.shape, device=DEVICE, generator=gen)
                P = full & valid & (rnd < 0.15)
                U = valid & (~P)
                ah_u, an_u = 1.0 / 0.15, 1.0 / 0.15
            else:
                P, U = P_rdg, U_rdg
                ah_u, an_u = a_h, a_n

            lo = D.logits_of(model, img).float()
            p = torch.sigmoid(lo)
            lp = (p - Y_HI) ** 2; ln = (p - Y_LO) ** 2
            losses = []
            for i in range(lo.shape[0]):
                Pi, Ui = P[i], U[i]
                nP, nU = float(Pi.sum().item()), float(Ui.sum().item())
                if nU < 50: continue
                if nP < 30:
                    losses.append(ln[i][Ui].mean()); continue
                is_n = bool(is_nat[i].item())
                alpha = max(1.05, (an_u if is_n else ah_u) * RHO)
                nV = nP + nU; frac_U = nU / nV
                pi_u = min(0.55, max(0.0, (alpha - 1.0) * nP / nU))
                pi_tot = min(0.65, max(0.01, alpha * nP / nV))
                Rp_pos = lp[i][Pi].mean(); Rp_neg = ln[i][Pi].mean(); Ru_neg = ln[i][Ui].mean()
                if loss_kind == 'scar15':
                    R_neg = Ru_neg - pi_u * Rp_neg
                    if float(R_neg.item()) >= 0.0:
                        losses.append(pi_tot * Rp_pos + frac_U * R_neg)
                    else:
                        if is_n: clamps_n += 1
                        else: clamps_h += 1
                        losses.append(-frac_U * R_neg)
                elif loss_kind == 'ridge_noasc':
                    R_neg = Ru_neg - pi_u * Rp_neg.detach()
                    if float(R_neg.item()) < 0.0:
                        if is_n: clamps_n += 1
                        else: clamps_h += 1
                    losses.append(pi_tot * Rp_pos + frac_U * torch.clamp(R_neg, min=0.0))
                elif loss_kind == 'ridge_kiryo':
                    R_neg = Ru_neg - pi_u * Rp_neg
                    if float(R_neg.item()) >= 0.0:
                        losses.append(pi_tot * Rp_pos + frac_U * R_neg)
                    else:
                        if is_n: clamps_n += 1
                        else: clamps_h += 1
                        losses.append(-frac_U * R_neg)
            if losses:
                (torch.stack(losses).mean() * ((s1 - s0) * W_INK_TILE)).backward()

        lb = D.logits_of(model, bimg.to(DEVICE, non_blocking=True)).float()
        pb = torch.sigmoid(lb)[:, 0]; bm_dev = bmask.to(DEVICE)
        ((pb - Y_LO) ** 2)[bm_dev].mean().mul(W_BLK).backward()
        bm_v = float(pb.detach()[bm_dev].mean())
        if step <= 5: blk_f5.append(bm_v)
        if step > steps - 5: blk_l5.append(bm_v)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    dec_dw = float(np.sqrt(sum(float(((p.detach() - p0) ** 2).sum().item()) for p, p0 in zip(dec, dec_init))))
    model.eval()
    h_t, h_v, u_t, u_v, g_t, g_v, bg_t, bg_v = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for img_all, tgt_all, geom_all, _, _ in batches[45:50]:
            for s0 in range(0, img_all.shape[0], 5):
                img = img_all[s0:s0 + 5].to(DEVICE); tgt = tgt_all[s0:s0 + 5].to(DEVICE); geom = geom_all[s0:s0 + 5].to(DEVICE)
                P, U, is_nat, full = BASE.extract_centerlines_and_masks(tgt, geom); N_ = is_nat.expand_as(P)
                gap = closing_gap_mask(full, P | U, is_nat)
                pt = torch.sigmoid(D.logits_of(model, img).float())
                pv = torch.sigmoid(D.logits_of(ref_model, img).float())
                for at, av, m in [(h_t, h_v, P & ~N_), (u_t, u_v, full & ~P), (g_t, g_v, gap), (bg_t, bg_v, U & ~full & ~N_)]:
                    if m.any(): at.append(pt[m].cpu()); av.append(pv[m].cpu())
    dh = float(torch.cat(h_t).median() - torch.cat(h_v).median())
    du = float(torch.cat(u_t).median() - torch.cat(u_v).median())
    dg = float(torch.cat(g_t).median() - torch.cat(g_v).median())
    dbg = float(torch.cat(bg_t).median() - torch.cat(bg_v).median())
    print(f'{name:24s} ({time.time()-t0:3.0f}s) | dW={dec_dw:5.3f} | blk {np.mean(blk_f5):.3f}->{np.mean(blk_l5):.3f} | handP={float(torch.cat(h_t).median()):.3f} (d{dh:+.3f}) | inkU={float(torch.cat(u_t).median()):.3f} (d{du:+.3f}) | gap_d={dg:+.3f} | bg_d={dbg:+.3f} | cl={clamps_h:3d}/{clamps_n:3d}', flush=True)
    del model, opt; torch.cuda.empty_cache()


def main():
    v3p = os.path.join(R, 'release', 'reader-v4-v3xe1', 'checkpoints', 'reader-v3.pth')
    model, loader, cfg = BASE.build_model_and_loader(v3p, batch_size=10)
    pad_to = BASE.configured_input_pad_depth(cfg)
    base_sd = copy.deepcopy(model.state_dict())
    ref_model = copy.deepcopy(model).to(DEVICE).eval()
    for p in ref_model.parameters(): p.requires_grad_(False)
    batches, a_h, a_n = D.cache_50_steps(loader, pad_to, n_steps=50)
    specs = [
        ('SCAR15_SGD_1e-2',       'sgd',   1e-2, 1e-3, 'scar15'),
        ('SCAR15_SGD_3e-2',       'sgd',   3e-2, 3e-3, 'scar15'),
        ('Ridge_NoAsc_SGD_1e-2',  'sgd',   1e-2, 1e-3, 'ridge_noasc'),
        ('Ridge_NoAsc_SGD_3e-2',  'sgd',   3e-2, 3e-3, 'ridge_noasc'),
        ('Ridge_Kiryo_SGD_1e-2',  'sgd',   1e-2, 1e-3, 'ridge_kiryo'),
        ('SCAR15_AdamW_3e-5',     'adamw', 3e-5, 3e-6, 'scar15'),
        ('Ridge_NoAsc_AdamW_3e-5','adamw', 3e-5, 3e-6, 'ridge_noasc'),
    ]
    for nm, ok, dlr, elr, lk in specs:
        run_arm(nm, ok, dlr, elr, lk, base_sd, model.cpu(), ref_model, batches, a_h, a_n, steps=100)


if __name__ == '__main__':
    main()
