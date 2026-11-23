"""Check K labels: FIXED half of Exam 1's 3-px spine, written as new koine label zarrs.

PREREG_TWO_CHECKS.md (frozen 2026-11-23 14:13:17), Check K "Labels":
  - spine = kt_nnpu.py L205-220 ridge (exact code), dilated 3x3 inside the ink (3-px spine)
  - ink = hand ink labels inside supervision (amax over Z), as Exam 1 (ink_h & sup_h)
  - ONE fixed random 64-px block keep per segment (p .5, numpy seed 42 + segment index), random grid offset
  - new label zarr = original label & keep-spine (per Z plane), same shape / chunks / dtype; level 0 only rewritten
    (koine reads volume_scale 0); supervision masks copied unchanged. Originals are never opened for writing.
Writes labels/<seg>/<seg>_inklabels.zarr + <seg>_supervision_mask.zarr under data/scribblepu/koine_smoke/,
plus label_stats.json (ink px, spine px, kept px, alpha per scroll and quota-weighted).
"""
import json
import os
import shutil
import sys

import cv2
import numpy as np
import zarr

R = r'C:\Users\nikox\Documents\Vesuvius'
OUT = os.path.join(R, 'data', 'scribblepu', 'koine_smoke', 'labels')
CFG = os.path.join(R, 'data', 'readerv4', 'retrain', '1bn_smoke.json')
QUOTAS = {'0139N': 2, '1667H': 3, 'Paris4H': 3}


def spine_kt_nnpu(m):
    """Exact copy of kt_nnpu.py L208-220 for one 2D uint8 ink mask."""
    k3 = np.ones((3, 3), np.uint8)
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
    rs = r & (dt >= 0.65 * dt_loc) & (m > 0)
    return (cv2.dilate(rs.astype(np.uint8), k3) > 0) & (m > 0)


def fixed_keep(shape, rng, cell=64, p_keep=0.5):
    H, W = shape
    gh, gw = H // cell + 2, W // cell + 2
    grid = rng.random((gh, gw)) < p_keep
    up = np.repeat(np.repeat(grid, cell, axis=0), cell, axis=1)
    oy, ox = int(rng.integers(0, cell)), int(rng.integers(0, cell))
    return up[oy:oy + H, ox:ox + W]


def main():
    cfg = json.load(open(CFG))
    stats = {}
    idx = 0
    for d in cfg['datasets']:
        scroll = d.get('sampling_scroll')
        if scroll not in QUOTAS:
            continue
        for seg in d['segments']:
            src_dir = os.path.join(d['segments_path'], seg)
            dst_dir = os.path.join(OUT, seg)
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
            ink2 = (ink3.max(axis=0) > 0)
            sup2 = (sup3.max(axis=0) > 0)
            m = (ink2 & sup2).astype(np.uint8)
            spine = spine_kt_nnpu(m)
            rng = np.random.default_rng(42 + idx)
            keep = fixed_keep(m.shape, rng)
            kept = spine & keep
            new3 = (ink3 > 0) & kept[None]
            zout = zarr.open(ink_dst, mode='r+')['0']
            zout[:] = (new3.astype(np.uint8) * int(ink3.max() if ink3.max() > 0 else 255))
            st = dict(scroll=scroll, ink_px=int(m.sum()), spine_px=int(spine.sum()), kept_px=int(kept.sum()),
                      sup_px=int(sup2.sum()), shape=list(m.shape), label_max=int(ink3.max()))
            st['spine_frac'] = st['spine_px'] / max(st['ink_px'], 1)
            st['kept_frac'] = st['kept_px'] / max(st['ink_px'], 1)
            stats[seg] = st
            print(seg, scroll, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items() if k != 'shape'}, flush=True)
            idx += 1
    per_scroll = {}
    for s in QUOTAS:
        ink = sum(v['ink_px'] for v in stats.values() if v['scroll'] == s)
        kept = sum(v['kept_px'] for v in stats.values() if v['scroll'] == s)
        per_scroll[s] = dict(ink_px=ink, kept_px=kept, alpha=ink / max(kept, 1), kept_frac=kept / max(ink, 1))
    qsum = sum(QUOTAS.values())
    alpha_w = sum(QUOTAS[s] * per_scroll[s]['alpha'] for s in QUOTAS) / qsum
    out = dict(segments=stats, per_scroll=per_scroll, alpha_quota_weighted=alpha_w)
    json.dump(out, open(os.path.join(OUT, '..', 'label_stats.json'), 'w'), indent=1)
    print('PER SCROLL', {s: {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()} for s, d in per_scroll.items()})
    print(f'ALPHA (quota-weighted) = {alpha_w:.2f}')


if __name__ == '__main__':
    main()
