"""Paul T3: letter-size cut-gap test (rebuild of CUT-2, 02 #160; the 10-20 data + scripts were deleted 11-03).

Question: does ScribblePU retuning (Reader-PU) close cut gaps (unevidenced stroke completion) more than the Reader V3
it came from? Reader v2 (128 px) shown as reference; Check K models report-only.

build : on 4 never-trained regions (0841 w00 / ag1 / ag2, 0009B 125754; eval volumes + hand / 2.4 um labels), pick
        stroke-centre points (Check K ridge code), >= 200 px apart; cut a band ACROSS the stroke (length g along the
        stroke, g = 46 / 69 / 93 / 116 px ~ 0.4 / 0.6 / 0.8 / 1.0 mm, width = local stroke width + 24 px) and fill it
        with blank papyrus copied (all Z) from the same sheet (inside the mask, >= 25 px from any label).
        SHAM = the same shapes pasted with blank onto blank targets (expect no response).
        INK  = the cut shapes filled with CT from a labelled-ink area (expect an ink response).
        Writes cut / ink copies of the level-0 volume (level '0' only; originals never opened for writing).
read  : every model x region x {orig, cut, ink} with the team's inference code (Exam 1's INF path).
score : bg = median read on blank (mask & >= 15 px from labels) of the orig read; centre = mean read in a disk of
        radius g/6 at the cut centre. fill = (cut_c - bg) / (orig_c - bg), only for cuts whose stroke was read in the
        orig (orig_c - bg >= 15). OPEN iff fill <= 0.30 (CUT-2 rule). sham fill uses the paired cut's denominator;
        ink fill = (ink_c - bg) / (orig_c - bg).
Usage: python cutgap.py build | read tag=ckpt [tag=ckpt ...] | score
"""
import glob
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
import tifffile
import zarr

R = r'C:\Users\nikox\Documents\Vesuvius'
P = os.path.join(R, 'data', 'scribblepu', 'paul_checks')
CG = os.path.join(P, 'cutgap')
EV = os.path.join(R, 'data', 'v8ft', 'eval', 'v2')
SU = os.path.join(R, 'data', 'v8ft', 'suite')
REGIONS = ['0841_w00', '0841_ag1', '0841_ag2', '0009B_125754']
WIDTHS = [46, 69, 93, 116]
sys.path.insert(0, os.path.join(R, 'data', 'scribblepu', 'koine_smoke'))
import build_labels as BL  # noqa: E402


def rect_poly(c, d, along, across):
    d = d / (np.linalg.norm(d) + 1e-9)
    n = np.array([-d[1], d[0]])
    a, b = along / 2.0, across / 2.0
    pts = [c + a * d + b * n, c + a * d - b * n, c - a * d - b * n, c - a * d + b * n]
    return np.round(np.array(pts)[:, ::-1]).astype(np.int32)  # (x, y) for cv2


def poly_mask(shape, poly):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [poly], 1)
    return m.astype(bool)


def build():
    os.makedirs(CG, exist_ok=True)
    plan = {}
    for ri, name in enumerate(REGIONS):
        rng = np.random.default_rng(7 + ri)
        mk = np.load(os.path.join(SU, f'{name}_masks.npz'))
        lab, msk = mk['lab'].astype(bool), mk['msk'].astype(bool)
        src = zarr.open(os.path.join(EV, f'{name}_sv.zarr'), mode='r')['0']
        vol = np.asarray(src[:])
        H, W = lab.shape
        assert vol.shape[1:] == (H, W), (vol.shape, lab.shape)
        m8 = (lab & msk).astype(np.uint8)
        dt = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
        spine = BL.spine_kt_nnpu(m8)
        near_lab = cv2.dilate(lab.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))) > 0
        blank_ok = msk & ~near_lab
        margin = 160
        cy, cx = np.nonzero(spine & (dt >= 3))
        keep = (cy > margin) & (cy < H - margin) & (cx > margin) & (cx < W - margin)
        cy, cx = cy[keep], cx[keep]
        order = rng.permutation(len(cy))
        chosen = []
        for i in order:
            p = np.array([cy[i], cx[i]], float)
            if all(np.hypot(*(p - q)) >= 200 for q in chosen):
                chosen.append(p)
        used = np.zeros((H, W), bool)
        cuts, shams = [], []
        vcut, vink = vol.copy(), vol.copy()
        for k, c in enumerate(chosen):
            g = WIDTHS[k % len(WIDTHS)]
            y, x = int(c[0]), int(c[1])
            sy, sx = np.nonzero(spine[max(0, y - 12):y + 13, max(0, x - 12):x + 13])
            if len(sy) < 5:
                continue
            pts = np.stack([sy, sx], 1).astype(float)
            pts -= pts.mean(0)
            d = np.linalg.svd(pts, full_matrices=False)[2][0]
            w_loc = 2.0 * float(dt[max(0, y - 20):y + 21, max(0, x - 20):x + 21].max())
            across = min(w_loc + 24.0, 180.0)
            cm = poly_mask((H, W), rect_poly(c, d, g, across))
            if (cm & used).any():
                continue
            ys, xs = np.nonzero(cm)

            def find_src(ok, need_ink):
                for _ in range(800):
                    oy, ox = int(rng.integers(-500, 501)), int(rng.integers(-500, 501))
                    ty, tx = ys + oy, xs + ox
                    if ty.min() < 0 or tx.min() < 0 or ty.max() >= H or tx.max() >= W:
                        continue
                    if need_ink:
                        if lab[ty, tx].mean() >= 0.5 and msk[ty, tx].all():
                            return oy, ox
                    elif ok[ty, tx].all() and not used[ty, tx].any() and not cm[ty, tx].any():
                        return oy, ox
                return None

            sb = find_src(blank_ok, False)
            si = find_src(None, True)
            if sb is None or si is None:
                continue
            # sham target: a blank place for the same shape, filled from another blank source
            sham = None
            for _ in range(800):
                ty0, tx0 = int(rng.integers(margin, H - margin)), int(rng.integers(margin, W - margin))
                sm = poly_mask((H, W), rect_poly(np.array([ty0, tx0], float), d, g, across))
                if blank_ok[sm].all() and not (sm & used).any() and not (sm & cm).any() \
                        and all(np.hypot(ty0 - q[0], tx0 - q[1]) >= 200 for q in chosen):
                    sys_, sxs = np.nonzero(sm)
                    for _ in range(400):
                        oy, ox = int(rng.integers(-500, 501)), int(rng.integers(-500, 501))
                        uy, ux = sys_ + oy, sxs + ox
                        if uy.min() < 0 or ux.min() < 0 or uy.max() >= H or ux.max() >= W:
                            continue
                        if blank_ok[uy, ux].all() and not sm[uy, ux].any():
                            sham = (ty0, tx0, sm, oy, ox)
                            break
                    if sham:
                        break
            if sham is None:
                continue
            vcut[:, ys, xs] = vol[:, ys + sb[0], xs + sb[1]]
            vink[:, ys, xs] = vol[:, ys + si[0], xs + si[1]]
            ty0, tx0, sm, oy, ox = sham
            sys_, sxs = np.nonzero(sm)
            vcut[:, sys_, sxs] = vol[:, sys_ + oy, sxs + ox]
            used |= cv2.dilate(cm.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
            used |= cv2.dilate(sm.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
            cuts.append(dict(y=y, x=x, g=g, across=round(across, 1), sham_y=ty0, sham_x=tx0))
        for tag, v in (('cut', vcut), ('ink', vink)):
            dst = os.path.join(CG, f'{name}_{tag}_sv.zarr')
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(os.path.join(EV, f'{name}_sv.zarr'), dst)
            zarr.open(dst, mode='r+')['0'][:] = v
        plan[name] = dict(cuts=cuts, reverse=bool(mk['reverse']) if 'reverse' in mk else False, shape=[H, W])
        print(name, 'cuts', len(cuts), {g: sum(c['g'] == g for c in cuts) for g in WIDTHS}, flush=True)
    json.dump(plan, open(os.path.join(CG, 'plan.json'), 'w'), indent=1)


def read(pairs):
    E1_DIR = r'C:\Users\nikox\Documents\antigravity\scribble-pu\experiments\exam1_from_scratch'
    sys.path.insert(0, E1_DIR)
    import eval_exam1 as E1  # noqa: E402
    import logging
    import torch
    from pathlib import Path
    INF = E1.INF
    plan = json.load(open(os.path.join(CG, 'plan.json')))
    for pair in pairs:
        tag, ck = pair.split('=', 1)
        od = os.path.join(CG, 'reads', tag)
        os.makedirs(od, exist_ok=True)
        jobs = []
        for name in REGIONS:
            direc = 'reverse' if plan[name]['reverse'] else 'forward'
            jobs.append((f'{name}_orig', os.path.join(EV, f'{name}_sv.zarr'), direc))
            for v in ('cut', 'ink'):
                jobs.append((f'{name}_{v}', os.path.join(CG, f'{name}_{v}_sv.zarr'), direc))
        need = [j for j in jobs if not os.path.exists(os.path.join(od, f'{j[0]}.tif'))]
        if not need:
            continue
        argv = [need[0][1], str(ck), os.path.join(od, '_probe.tif'), '--overlap', '0.5', '--blend-mode', 'hann',
                '--batch-size', '8', '--no-compile', '--direction', 'forward', '--num-workers', '2']
        args = INF.normalize_inference_paths(INF.parse_args(argv))
        cm = INF.configure_model(args)
        cm, dev, _ = INF.prepare_model_for_inference(args=args, configured_model=cm)
        logging.getLogger().setLevel(logging.WARNING)
        for jn, sv, direc in need:
            t0 = time.time()
            INF.infer_single_zarr(args=args, configured_model=cm, device=dev, input_zarr=str(sv),
                                  output_tiff=Path(os.path.join(od, f'{jn}.tif')), layer_direction=direc)
            print(f'[CUTGAP READ] {tag:14s} {jn:24s} {time.time() - t0:.1f}s', flush=True)
        del cm
        torch.cuda.empty_cache()


def disk_mean(img, y, x, r):
    H, W = img.shape
    yy, xx = np.ogrid[max(0, y - r):min(H, y + r + 1), max(0, x - r):min(W, x + r + 1)]
    sub = img[max(0, y - r):min(H, y + r + 1), max(0, x - r):min(W, x + r + 1)]
    m = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    return float(sub[m].mean())


def score():
    plan = json.load(open(os.path.join(CG, 'plan.json')))
    out = {}
    for od in sorted(glob.glob(os.path.join(CG, 'reads', '*'))):
        tag = os.path.basename(od)
        rows = []
        for name in REGIONS:
            mk = np.load(os.path.join(SU, f'{name}_masks.npz'))
            lab, msk = mk['lab'].astype(bool), mk['msk'].astype(bool)
            rd = {v: tifffile.imread(os.path.join(od, f'{name}_{v}.tif')).astype(np.float32) for v in ('orig', 'cut', 'ink')}
            H = min(lab.shape[0], rd['orig'].shape[0]); W = min(lab.shape[1], rd['orig'].shape[1])
            far = cv2.dilate(lab[:H, :W].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) == 0
            bg = float(np.median(rd['orig'][:H, :W][msk[:H, :W] & far]))
            for c in plan[name]['cuts']:
                r = max(3, c['g'] // 6)
                o = disk_mean(rd['orig'], c['y'], c['x'], r) - bg
                rows.append(dict(region=name, g=c['g'], orig=o,
                                 cut=disk_mean(rd['cut'], c['y'], c['x'], r) - bg,
                                 ink=disk_mean(rd['ink'], c['y'], c['x'], r) - bg,
                                 sham=disk_mean(rd['cut'], c['sham_y'], c['sham_x'], r) - disk_mean(rd['orig'], c['sham_y'], c['sham_x'], r)))
        valid = [w for w in rows if w['orig'] >= 15]
        for w in valid:
            w['fill'] = w['cut'] / w['orig']
            w['ink_fill'] = w['ink'] / w['orig']
            w['sham_fill'] = w['sham'] / w['orig']
        res = dict(n_cuts=len(rows), n_valid=len(valid))
        if valid:
            res['open_rate'] = float(np.mean([w['fill'] <= 0.30 for w in valid]))
            res['median_fill'] = float(np.median([w['fill'] for w in valid]))
            res['median_sham_fill'] = float(np.median([w['sham_fill'] for w in valid]))
            res['median_ink_fill'] = float(np.median([w['ink_fill'] for w in valid]))
            res['by_width'] = {str(g): dict(n=sum(w['g'] == g for w in valid),
                                            open_rate=float(np.mean([w['fill'] <= 0.30 for w in valid if w['g'] == g]))
                                            if any(w['g'] == g for w in valid) else None) for g in WIDTHS}
        out[tag] = res
        print(tag, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != 'by_width'}),
              {g: (d['n'], None if d['open_rate'] is None else round(d['open_rate'], 2)) for g, d in res.get('by_width', {}).items()},
              flush=True)
    json.dump(out, open(os.path.join(CG, 'results_cutgap.json'), 'w'), indent=1)


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build()
    elif sys.argv[1] == 'read':
        read(sys.argv[2:])
    elif sys.argv[1] == 'score':
        score()
