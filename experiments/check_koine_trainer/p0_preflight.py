"""Check K pre-flight P0 (a)+(b) with koine_machines' own dataset, sampler, model and loss (no training).

(a) 3 sampled batches: the positives in the loss target are the label crop's positives; output size == label size
    (no nearest-resize); lines stay thin (mean width 2*area/perimeter of positive blobs ~3 px, not inflated).
(b) the loss target is ONE tensor [lines, ignore] (no deep-supervision list); BrierNNPULoss builds from the config,
    returns finite loss + grads, and logs pi_u > 0 (=> nP >= 30 in the batch) and 0 < pi_u <= .55.
(c) is the 20-step trainer dry run (k_dry.json), run separately.
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

K = r'C:\Users\nikox\Documents\Vesuvius\data\scribblepu\koine_smoke'


def width_2a_over_p(mask2d):
    m = mask2d.astype(np.uint8)
    if m.sum() == 0:
        return float('nan')
    cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cs)
    return 2.0 * float(m.sum()) / max(per, 1.0)


def main():
    from koine_machines.data.ink_dataset import InkDataset
    from koine_machines.training.samplers import FixedScrollPriorStratifiedBatchSampler
    from koine_machines.training.loss.losses import create_loss_from_config
    from koine_machines.models.make_model import make_model

    cfg = json.load(open(os.path.join(K, 'k1.json')))
    print('koine from', sys.modules['koine_machines'].__file__, flush=True)
    shared = InkDataset(dict(cfg), do_augmentations=False)
    print('training patches', len(shared.training_patches), 'validation', len(shared.validation_patches), flush=True)
    train = InkDataset(dict(cfg), do_augmentations=True, patches=shared.training_patches, segments=shared.segments)
    sampler = FixedScrollPriorStratifiedBatchSampler(shared.training_patches, cfg['datasets'],
                                                     batch_quotas=cfg['fixed_scroll_prior']['target_batch_counts'],
                                                     batch_size=cfg['batch_size'], seed=cfg['seed'])
    it = iter(sampler)
    dev = torch.device('cuda')
    model = make_model(cfg).to(dev)
    loss_fn = create_loss_from_config(cfg)
    print('loss terms', [type(t.module).__name__ for t in loss_fn.terms], flush=True)
    report = []
    for bi in range(3):
        idxs = next(it)
        items = [train[i] for i in idxs]
        batch = {k: torch.stack([torch.as_tensor(x[k]) for x in items]) for k in ('image', 'inklabels', 'supervision_mask')}
        raw_pos = int((batch['inklabels'] > 0).amax(dim=2).sum())
        targets = (torch.amax(batch['inklabels'], dim=2) > 0).float()
        sup = torch.amax(batch['supervision_mask'], dim=2)
        ignore = (sup <= 0).float()
        img = batch['image'].float().to(dev)
        with torch.no_grad():
            out = model(img)
        preds = out['ink'] if isinstance(out, dict) else out
        is_list = isinstance(preds, (list, tuple))
        p0 = preds[0] if is_list else preds
        same_size = tuple(p0.shape[-2:]) == tuple(targets.shape[-2:])
        tgt = torch.cat([targets, ignore], dim=1).to(dev)
        logits = p0.detach().float().clone().requires_grad_(True)
        total = loss_fn(logits, tgt)
        total.backward()
        m = loss_fn.latest_metrics
        widths = [width_2a_over_p(targets[i, 0].numpy() > 0) for i in range(targets.shape[0])]
        rec = dict(batch=bi, image=list(batch['image'].shape), inklabels=list(batch['inklabels'].shape),
                   pred=list(p0.shape), pred_is_list=is_list, same_size=same_size, raw_pos=raw_pos,
                   target_pos=int(targets.sum()), sup_px=int((sup > 0).sum()), pos_frac_of_sup=float(targets.sum() / max((sup > 0).sum(), 1)),
                   mean_line_width_px=float(np.nanmean(widths)), loss=float(total), finite_grad=bool(torch.isfinite(logits.grad).all()),
                   pi_u=m.get('loss_aux/brier_nnpu/pi_u'), neg_risk=m.get('loss_aux/brier_nnpu/neg_risk'),
                   clamped=m.get('loss_aux/brier_nnpu/clamped_frac'))
        report.append(rec)
        print(json.dumps(rec), flush=True)
    ok_a = all(r['raw_pos'] == r['target_pos'] and r['same_size'] and r['target_pos'] > 0 and 1.5 <= r['mean_line_width_px'] <= 5.0 for r in report)
    ok_b = all((not r['pred_is_list']) and np.isfinite(r['loss']) and r['finite_grad'] and r['pi_u'] is not None and 0 < r['pi_u'] <= 0.55 for r in report)
    verdict = dict(P0a=ok_a, P0b=ok_b)
    print('P0 VERDICT', verdict, flush=True)
    json.dump(dict(report=report, verdict=verdict), open(os.path.join(K, 'p0_report.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
