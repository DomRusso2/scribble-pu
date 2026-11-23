"""ScribblePU Exam 1 — Held-Out Scroll Evaluation & Visual Panels for From-Scratch Benchmark.

Evaluates any set of Exam 1 checkpoints (e.g. oracle_fullmask, naive_bce, collar_bce,
kiryo_nnpu, scribble_pu) on strictly HELD-OUT scrolls never seen during Exam 1 training:
  1. 0009B_125754 (unseen fragment — GATE, M4, M1, M1n, ridgeAUC)
  2. pherc0814-46527 (unseen Scroll 4 — GATE, M4, M1, M1n, ridgeAUC)
  3. w047_0_512 + blank_1024_5120 (FA80, AUCcb, AUCpn, cP20)
  4. 0306B p306 public letter-line crop (visual comparison ONLY — zero Table P scoring!)
"""
import os
import sys
import json
import time
import glob
import logging
import argparse
from pathlib import Path
import numpy as np
import cv2
import zarr
import tifffile
import torch
from scipy import ndimage as ndi
from PIL import Image, ImageDraw

R = r'C:\Users\nikox\Documents\Vesuvius'
HERE = os.path.dirname(os.path.abspath(__file__))
READ_DIR = os.path.join(HERE, 'reads')
VIS_DIR = os.path.join(HERE, 'visuals')
os.makedirs(READ_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

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
import timm  # noqa: F401
import read_1a as RA
import ys2
from v8ft_eval import auc as auc_sub
import koine_machines.inference.infer as INF

EV = os.path.join(R, 'data', 'v8ft', 'eval')
S814_LAB = os.path.join(R, 'data', 'ink9um', 'labels', 'aligned-scrollprizeorg-21slices', 'pherc0814-46527')
JOBS = [
    ('w047_0_512', os.path.join(EV, 'v2', 'w047_0_512_sv.zarr'), 'forward'),
    ('blank_1024_5120', os.path.join(EV, 'v2', 'blank_1024_5120_sv.zarr'), 'forward'),
    ('0009B_125754', os.path.join(EV, 'v2', '0009B_125754_sv.zarr'), 'forward'),
    ('pherc0814-46527', os.path.join(R, 'data', 'teacher', 'p0814_iso9.zarr'), 'forward'),
    ('p306', os.path.join(R, 'data', 'v3l', 'p306', 'sv24.zarr'), 'reverse'),
]


def infer_arm(tag, ckpt_path):
    od = os.path.join(READ_DIR, tag)
    os.makedirs(od, exist_ok=True)
    need = [j for j in JOBS if not os.path.exists(os.path.join(od, f'{j[0]}.tif'))]
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
            out_p = Path(os.path.join(od, f'{name}.tif'))
            t0 = time.time()
            INF.infer_single_zarr(args=args, configured_model=cm, device=dev,
                                  input_zarr=str(sv), output_tiff=out_p, layer_direction=direc)
            print(f'[INFER] {tag:30s} | {name:16s} {time.time()-t0:.1f}s', flush=True)
        del cm
        torch.cuda.empty_cache()
    return {name: tifffile.imread(os.path.join(od, f'{name}.tif')).astype(np.float32) for name, _, _ in JOBS}


def load_0814_labels():
    out = {}
    for k in ('inklabels', 'supervision_mask', 'validation_mask'):
        m = glob.glob(os.path.join(S814_LAB, f'*_{k}.zarr'))
        if m:
            arr = np.asarray(zarr.open(m[0], mode='r')['0']) > 0
            out[k] = arr.any(0)
    P = out['inklabels']
    S = out['supervision_mask']
    V = out.get('validation_mask', np.zeros_like(S))
    return P, S, V


def score_arm(M, masks):
    pos, neg, bval, core = masks['w047']
    lab9, msk9 = masks['b9']
    P814, S814, V814 = masks['s814']
    r = {}

    # 1. 0009B_125754 (held out!)
    m9 = M['0009B_125754']
    hh, ww = min(lab9.shape[0], m9.shape[0]), min(lab9.shape[1], m9.shape[1])
    l9, k9, m9c = lab9[:hh, :ww], msk9[:hh, :ww], m9[:hh, :ww]
    val9 = k9 & (m9c > 0)
    B9 = ys2.base(l9)
    M9 = ys2.masks(B9, val9, RA.SHIFT_0009B)
    s9 = ys2.score(m9c, M9, 256, 50)
    r.update(
        b9_GATE=s9['GATE'], b9_M4=s9['M4'], b9_M1=s9['M1'], b9_M1n=s9['M1_narrow'],
        b9_rAUC=round(float(auc_sub(m9c[M9['ridge']], m9c[M9['far']])), 4),
        b9_p50=float(np.percentile(m9c[val9], 50)), b9_p99=float(np.percentile(m9c[val9], 99)),
    )

    # 2. pherc0814-46527 (held out!)
    d814 = RA.score_dev_map(P814, S814, V814, M['pherc0814-46527'], do_align=False)
    r.update(
        s814_GATE=d814['all']['GATE'], s814_M4=d814['all']['M4'],
        s814_M1=d814['all']['M1'], s814_M1n=d814['all']['M1_narrow'],
    )

    # 3. w047 + blank_1024_5120
    w, b = M['w047_0_512'], M['blank_1024_5120']
    thr = float(np.percentile(w[pos], 20))
    r.update(
        thr=thr, FA80=round(float((b[bval] >= thr).mean()), 4),
        cP20=float(np.percentile(w[core], 20)),
        AUCcb=round(float(auc_sub(w[core], b[bval])), 4),
        AUCpn=round(float(auc_sub(w[pos], w[neg])), 4),
    )
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True, help='List of tag=ckpt_path pairs')
    ap.add_argument('--out_prefix', default='exam1')
    args = ap.parse_args()

    mw = dict(np.load(os.path.join(RA.SU, 'w047_0_512_masks.npz')))
    mb = dict(np.load(os.path.join(RA.SU, 'blank_1024_5120_masks.npz')))
    pos, neg, bval = mw['pos'].astype(bool), mw['neg'].astype(bool), mb['valid'].astype(bool)
    core = pos & (ndi.distance_transform_edt(pos) >= 8)
    mk9 = dict(np.load(os.path.join(RA.SU, '0009B_125754_masks.npz')))
    lab9, msk9 = mk9['lab'].astype(bool), mk9['msk'].astype(bool)
    masks = dict(w047=(pos, neg, bval, core), b9=(lab9, msk9), s814=load_0814_labels())

    all_maps, all_res = {}, {}
    for pair in args.ckpts:
        tag, ck = pair.split('=', 1)
        all_maps[tag] = infer_arm(tag, ck)
        all_res[tag] = score_arm(all_maps[tag], masks)
        print(f'\n=== {tag} ===\n{json.dumps(all_res[tag], indent=2)}', flush=True)

    # Save side-by-side 0009B, 0814H, and p306 public letter panels
    def save_square_panel(piece_name, mask_bool, metric_key, out_name):
        ii = cv2.integral(mask_bool.astype(np.float32))
        best = (-1.0, 0, 0)
        H, W = mask_bool.shape
        for y in range(0, max(1, H - 512), 64):
            for x in range(0, max(1, W - 512), 64):
                s = ii[y + 512, x + 512] - ii[y, x + 512] - ii[y + 512, x] + ii[y, x]
                if s > best[0]:
                    best = (s, y, x)
        y0, x0 = best[1], best[2]
        tags = list(all_maps.keys())
        pad, hdr = 10, 34
        canv = np.full((hdr + 512 + pad, len(tags) * (512 + pad) + pad), 28, dtype=np.uint8)
        for i, t in enumerate(tags):
            xx = pad + i * (512 + pad)
            canv[hdr:hdr + 512, xx:xx + 512] = np.clip(np.rint(all_maps[t][piece_name][y0:y0 + 512, x0:x0 + 512]), 0, 255).astype(np.uint8)
        pil = Image.fromarray(canv, mode='L').convert('RGB')
        dr = ImageDraw.Draw(pil)
        for i, t in enumerate(tags):
            xx = pad + i * (512 + pad)
            dr.text((xx + 6, 9), f"{t} ({metric_key}={all_res[t][metric_key]:.3f})", fill=(245, 245, 245))
        out_p = os.path.join(VIS_DIR, out_name)
        pil.save(out_p)
        print(f'[VIS] Saved {out_p}', flush=True)

    save_square_panel('0009B_125754', lab9 & msk9, 'b9_GATE', f'{args.out_prefix}_0009B.png')
    P814, S814, V814 = masks['s814']
    save_square_panel('pherc0814-46527', P814 & (S814 | V814), 's814_GATE', f'{args.out_prefix}_0814H.png')

    # Stacked p306 public line strip
    tags = list(all_maps.keys())
    strips = [(t, np.flipud(all_maps[t]['p306'])[350:850, 420:2320]) for t in tags]
    sh_, sw_ = strips[0][1].shape
    hdr_, pad_ = 28, 8
    canv = np.full((len(strips) * (hdr_ + sh_ + pad_) + pad_, sw_ + 2 * pad_), 28, dtype=np.uint8)
    for i, (_, arr) in enumerate(strips):
        yy = pad_ + i * (hdr_ + sh_ + pad_) + hdr_
        canv[yy:yy + sh_, pad_:pad_ + sw_] = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    pil306 = Image.fromarray(canv, mode='L').convert('RGB')
    dr = ImageDraw.Draw(pil306)
    for i, (t, _) in enumerate(strips):
        yy = pad_ + i * (hdr_ + sh_ + pad_) + 6
        dr.text((pad_ + 8, yy), f"{t} (b9_G={all_res[t]['b9_GATE']:.3f}, 814_G={all_res[t]['s814_GATE']:.3f})", fill=(245, 245, 245))
    out_306 = os.path.join(VIS_DIR, f'{args.out_prefix}_p306.png')
    pil306.save(out_306)
    print(f'[VIS] Saved {out_306}', flush=True)

    out_j = os.path.join(HERE, f'{args.out_prefix}_results.json')
    json.dump(all_res, open(out_j, 'w', encoding='utf-8'), indent=2)
    print(f'[DONE] Saved {out_j}', flush=True)


if __name__ == '__main__':
    main()
