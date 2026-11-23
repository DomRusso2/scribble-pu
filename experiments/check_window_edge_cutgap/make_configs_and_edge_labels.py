"""Paul's checks (Discord 2026-11-23): configs for T1 (128 px window) and labels + config for T2 (edge lines).

T1: k1.json / k2.json from Check K, changing ONLY patch_size [17,384,384] -> [17,128,128], the patch cache file and
    out_dir. Same labels (Check K fixed half of the 3-px spine), same alpha (14.64), steps, seed, optimizer, quotas.
T2: Check K1 at 384 px, labels changed ONLY in line placement: the SAME kept 64-px blocks as Check K (same rng seeds),
    and inside each kept block a fixed coin flip (p .5, seed 1000 + segment index, 64-px grid) picks either the Check K
    centre spine (3 px) or an EDGE line = the 3-px inner rim of the stroke (ink pixels at distance 1..3 px from
    background). alpha by the same rule as Check K (quota-weighted ink px / kept px).
"""
import copy
import json
import os
import shutil
import sys

import numpy as np
import zarr

R = r'C:\Users\nikox\Documents\Vesuvius'
K = os.path.join(R, 'data', 'scribblepu', 'koine_smoke')
P = os.path.join(R, 'data', 'scribblepu', 'paul_checks')
sys.path.insert(0, K)
import build_labels as BL  # noqa: E402  (spine_kt_nnpu, fixed_keep, QUOTAS, CFG: the exact Check K code)
import cv2  # noqa: E402


def make_t1():
    for arm in ('k1', 'k2'):
        c = json.load(open(os.path.join(K, f'{arm}.json')))
        assert c['patch_size'] == [17, 384, 384], c['patch_size']
        c['patch_size'] = [17, 128, 128]
        c['patch_cache_filename'] = os.path.join(P, 'patches_k_128.json').replace('\\', '/')
        c['out_dir'] = os.path.join(P, f't1_{arm}_128').replace('\\', '/')
        c['description'] = c['description'] + ' | PAUL T1: identical except a 128 px window'
        json.dump(c, open(os.path.join(P, f't1_{arm}_128.json'), 'w'), indent=1)
        print('T1', arm, c['patch_size'], json.dumps(c['loss'])[:90], c['out_dir'])


def edge_rim(m):
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    return (dt >= 1.0) & (dt <= 3.0) & (m > 0)


def make_t2_labels():
    out = os.path.join(P, 'labels_edge')
    cfg = json.load(open(BL.CFG))
    stats, idx = {}, 0
    for d in cfg['datasets']:
        scroll = d.get('sampling_scroll')
        if scroll not in BL.QUOTAS:
            continue
        for seg in d['segments']:
            src_dir = os.path.join(d['segments_path'], seg)
            dst_dir = os.path.join(out, seg)
            os.makedirs(dst_dir, exist_ok=True)
            ink_src = os.path.join(src_dir, f'{seg}_inklabels.zarr')
            sup_src = os.path.join(src_dir, f'{seg}_supervision_mask.zarr')
            ink_dst = os.path.join(dst_dir, f'{seg}_inklabels.zarr')
            sup_dst = os.path.join(dst_dir, f'{seg}_supervision_mask.zarr')
            for s, t in ((ink_src, ink_dst), (sup_src, sup_dst)):
                if os.path.exists(t):
                    shutil.rmtree(t)
                shutil.copytree(s, t)
            ink3 = zarr.open(ink_src, mode='r')['0'][:]
            sup3 = zarr.open(sup_src, mode='r')['0'][:]
            m = ((ink3.max(axis=0) > 0) & (sup3.max(axis=0) > 0)).astype(np.uint8)
            spine = BL.spine_kt_nnpu(m)
            keep = BL.fixed_keep(m.shape, np.random.default_rng(42 + idx))          # identical to Check K
            choose_edge = BL.fixed_keep(m.shape, np.random.default_rng(1000 + idx))  # per-block coin flip
            rim = edge_rim(m)
            kept = keep & ((choose_edge & rim) | (~choose_edge & spine))
            new3 = (ink3 > 0) & kept[None]
            zout = zarr.open(ink_dst, mode='r+')['0']
            zout[:] = (new3.astype(np.uint8) * int(ink3.max() if ink3.max() > 0 else 255))
            st = dict(scroll=scroll, ink_px=int(m.sum()), kept_px=int(kept.sum()),
                      kept_edge_px=int((kept & choose_edge).sum()), kept_centre_px=int((kept & ~choose_edge).sum()))
            st['kept_frac'] = st['kept_px'] / max(st['ink_px'], 1)
            stats[seg] = st
            print(seg, scroll, st, flush=True)
            idx += 1
    per_scroll = {}
    for s in BL.QUOTAS:
        ink = sum(v['ink_px'] for v in stats.values() if v['scroll'] == s)
        kept = sum(v['kept_px'] for v in stats.values() if v['scroll'] == s)
        per_scroll[s] = dict(ink_px=ink, kept_px=kept, alpha=ink / max(kept, 1), kept_frac=kept / max(ink, 1))
    alpha_w = sum(BL.QUOTAS[s] * per_scroll[s]['alpha'] for s in BL.QUOTAS) / sum(BL.QUOTAS.values())
    json.dump(dict(segments=stats, per_scroll=per_scroll, alpha_quota_weighted=alpha_w),
              open(os.path.join(P, 'label_stats_edge.json'), 'w'), indent=1)
    print('PER SCROLL', per_scroll)
    print(f'ALPHA edge (quota-weighted) = {alpha_w:.2f}')
    return alpha_w


def make_t2_config(alpha_w):
    c = json.load(open(os.path.join(K, 'k1.json')))
    c['datasets'] = copy.deepcopy(c['datasets'])
    for d in c['datasets']:
        d['segments_path'] = os.path.join(P, 'labels_edge').replace('\\', '/')
    c['loss']['terms'][0]['alpha'] = round(float(alpha_w), 2)
    c['out_dir'] = os.path.join(P, 't2_k1_edge').replace('\\', '/')
    c['description'] = 'PAUL T2: Check K1 with centre + edge lines on the same kept blocks (384 px)'
    json.dump(c, open(os.path.join(P, 't2_k1_edge.json'), 'w'), indent=1)
    print('T2', c['patch_size'], c['loss'], c['out_dir'])


if __name__ == '__main__':
    os.makedirs(P, exist_ok=True)
    make_t1()
    a = make_t2_labels()
    make_t2_config(a)
