"""ScribblePU Exam 3 — Complete Dev-Tier Evaluation & Public Visual Panels for Reader-PU (B-Hand).

ZERO Table P touches! Strictly read-only inside C:\\Users\\nikox\\Documents\\Vesuvius.
Evaluates:
  - reader_pu_bhand_raw.pth    (step 1,000)
  - reader_pu_bhand_soup65.pth (0.65 * raw + 0.35 * V3)
  - reader_pu_bhand_soup50.pth (0.50 * raw + 0.50 * V3)
against V3, V3-eroded (r=2, 4, 6, 8, 10), and banked soup_65 (from score_1bn.json) on the EXACT
score_1bn.py dev suite, and renders side-by-side public letter panels (0009B, w029, p306).
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
import numpy as np
import cv2
import zarr
import tifffile
import torch
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont

R = r'C:\Users\nikox\Documents\Vesuvius'
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, 'ckpt')
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
import r1a_common as C
import read_1a as RA
import ys2
import v2_run as v2r
from v8ft_eval import auc as auc_sub
import koine_machines.inference.infer as INF

EV = os.path.join(R, 'data', 'v8ft', 'eval')
AG1 = '20260220214732-auto_grown_20260220144552'
AG1_CROP = (0, 1401, 1720, 3121)
BUF = 256
E1 = json.load(open(C.E1_CFG))
SCORE_REF = json.load(open(os.path.join(R, 'data', 'readerv4', 'retrain', 'reads', 'score_1bn.json')))

JOBS = [
    ('w047_0_512', os.path.join(EV, 'v2', 'w047_0_512_sv.zarr'), 'forward'),
    ('blank_1024_5120', os.path.join(EV, 'v2', 'blank_1024_5120_sv.zarr'), 'forward'),
    ('0009B_125754', os.path.join(EV, 'v2', '0009B_125754_sv.zarr'), 'forward'),
    ('pherc1667-w029', E1['datasets'][5]['surface_volume_paths']['pherc1667-w029'], 'forward'),
    ('pherc1667-w031', E1['datasets'][4]['surface_volume_paths']['pherc1667-w031'], 'forward'),
    ('0841_ag1', os.path.join(R, 'data', 'v2', '0841', AG1, 'sv.zarr'), 'forward'),
    ('blank_w053_3200_2880', os.path.join(R, 'data', 'fl', '_rv116', 'blanks', '3200_2880', 'sv.zarr'), 'forward'),
    ('blank_w053_960_5920', os.path.join(R, 'data', 'fl', '_rv116', 'blanks', '960_5920', 'sv.zarr'), 'forward'),
    ('p306', os.path.join(R, 'data', 'v3l', 'p306', 'sv24.zarr'), 'reverse'),
]


def run_infer_arm(tag, ckpt_path, only_names=None):
    od = os.path.join(READ_DIR, tag)
    os.makedirs(od, exist_ok=True)
    jobs = [j for j in JOBS if os.path.exists(j[1]) and (only_names is None or j[0] in only_names)]
    need = [j for j in jobs if not os.path.exists(os.path.join(od, f'{j[0]}.tif'))]
    if need:
        argv = [
            need[0][1], str(ckpt_path), os.path.join(od, '_probe.tif'),
            '--overlap', '0.5', '--blend-mode', 'hann', '--batch-size', '8',
            '--no-compile', '--direction', 'forward', '--num-workers', '2',
        ]
        args = INF.normalize_inference_paths(INF.parse_args(argv))
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
        cm = INF.configure_model(args)
        cm, dev, _ = INF.prepare_model_for_inference(args=args, configured_model=cm)
        logging.getLogger().setLevel(logging.WARNING)
        for name, sv, direc in need:
            out_p = Path(os.path.join(od, f'{name}.tif'))
            t0 = time.time()
            INF.infer_single_zarr(
                args=args, configured_model=cm, device=dev,
                input_zarr=str(sv), output_tiff=out_p, layer_direction=direc,
            )
            print(f'[INFER] {tag:14s} | {name:22s} ({direc:7s}) {time.time()-t0:.1f}s', flush=True)
        del cm
        torch.cuda.empty_cache()
    maps = {}
    for name, _, _ in jobs:
        maps[name] = tifffile.imread(os.path.join(od, f'{name}.tif')).astype(np.float32)
    return maps


def load_v3_maps():
    """Reuse existing V3 read TIFFs on disk where available; only infer w029 if missing."""
    v3 = {
        'w047_0_512': tifffile.imread(os.path.join(EV, 'long384', 'w047_0_512_raw.tif')).astype(np.float32),
        'blank_1024_5120': tifffile.imread(os.path.join(EV, 'long384', 'blank_1024_5120_raw.tif')).astype(np.float32),
        '0009B_125754': tifffile.imread(os.path.join(EV, 'long384', '0009B_125754_raw.tif')).astype(np.float32),
        '0841_ag1': tifffile.imread(os.path.join(R, 'data', 'readerv4', 'fl', 'reads', f'0841_{AG1}_v3.tif')).astype(np.float32),
        'p306': tifffile.imread(os.path.join(R, 'data', 'v3l', 'p306', 'v3_reverse.tif')).astype(np.float32),
    }
    v3p = os.path.join(R, 'release', 'reader-v4-v3xe1', 'checkpoints', 'reader-v3.pth')
    extra = run_infer_arm('v3_ref', v3p, only_names={'pherc1667-w029'})
    v3['pherc1667-w029'] = extra['pherc1667-w029']
    return v3


def erode_map(m, r):
    if r <= 0:
        return m.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    out = cv2.erode(m.astype(np.float32), k)
    out[m == 0] = 0
    return out


def score_arm(name, M, masks, ckpt_path=None):
    pos, neg, bval, core = masks['w047']
    P29, S29, V29, y0_29, y1_29, x0_29, x1_29, B29 = masks['w29']
    P31, S31, V31 = masks['w31']
    lab9, msk9 = masks['b9']
    ink841, rev841, ex841, sh841, B841 = masks['o841']

    r = {}
    # 1. w029
    d29 = RA.score_dev_map(P29, S29, V29, M['pherc1667-w029'], do_align=False)
    r.update(
        w29_M1=d29['all']['M1'], w29_M1n=d29['all']['M1_narrow'], w29_M2=d29['all']['M2'],
        w29_GATE=d29['all']['GATE'], w29_M4=d29['all']['M4'], w29_vGATE=d29['val_only']['GATE'],
    )
    m29c = M['pherc1667-w029'][y0_29:y1_29, x0_29:x1_29]
    A29 = (S29 | V29)[y0_29:y1_29, x0_29:x1_29] & (m29c > 0)
    MM29 = ys2.masks(B29, A29)
    band = A29 & (B29['dt_in'] >= 8) & (B29['dt_in'] < 0.45 * MM29['h'])
    r['w29_band_mean'] = round(float(m29c[band].mean()), 1)
    m29f = M['pherc1667-w029']
    r['w29_p99'] = float(np.percentile(m29f[m29f > 0], 99))

    # 2. w031
    d31 = RA.score_dev_map(P31, S31, V31, M['pherc1667-w031'], do_align=False)
    r.update(w31_GATE=d31['all']['GATE'], w31_M4=d31['all']['M4'])

    # 3. 0009B_125754 (fixed shift (-5,-8) + own-best shift + narrow crater p05)
    m9 = M['0009B_125754']
    hh, ww = min(lab9.shape[0], m9.shape[0]), min(lab9.shape[1], m9.shape[1])
    l9, k9, m9c = lab9[:hh, :ww], msk9[:hh, :ww], m9[:hh, :ww]
    val9 = k9 & (m9c > 0)
    B9 = ys2.base(l9)
    M9 = ys2.masks(B9, val9, RA.SHIFT_0009B)
    s9 = ys2.score(m9c, M9, 256, 50)
    r.update(
        b9_M1=s9['M1'], b9_M1n=s9['M1_narrow'], b9_M2=s9['M2'], b9_M4=s9['M4'], b9_GATE=s9['GATE'],
        b9_rAUC=round(float(auc_sub(m9c[M9['ridge']], m9c[M9['far']])), 4),
        b9_p99=float(np.percentile(m9[m9 > 0], 99)),
        crater_b9_narrow_p05=float(np.percentile(m9c[M9['narrow']], 5)),
    )
    al9 = ys2.align(l9, ys2.rankmap(m9c, val9), val9, rad=12, coarse=3)
    sb9 = ys2.score(m9c, ys2.masks(B9, val9, tuple(al9['best'])), 256, 50)
    r.update(b9_best_shift=al9['best'], b9_M1_best=sb9['M1'], b9_M1n_best=sb9['M1_narrow'])

    # 4. w047 + blank_1024_5120
    w, b = M['w047_0_512'], M['blank_1024_5120']
    thr = float(np.percentile(w[pos], 20))
    r.update(
        thr=thr,
        FA80=round(float((b[bval] >= thr).mean()), 4),
        cP20=float(np.percentile(w[core], 20)),
        AUCcb=round(float(auc_sub(w[core], b[bval])), 4),
        AUCpn=round(float(auc_sub(w[pos], w[neg])), 4),
    )

    # 5. w053 blank crops (compared at V3's thr=98.0, when volumes exist on disk)
    thr_v3 = SCORE_REF['table']['V3']['thr']
    for bp in ('blank_w053_3200_2880', 'blank_w053_960_5920'):
        if bp not in M:
            continue
        bm = M[bp]
        bv = bm > 0
        r[f'{bp}_FA_V3thr'] = round(float((bm[bv] >= thr_v3).mean()), 4)
        r[f'{bp}_p99'] = float(np.percentile(bm[bv], 99))

    # 6. 0841_ag1 strictly outside Table P crop + 256 px
    g841 = M['0841_ag1'][:sh841[0], :sh841[1]]
    val841 = rev841 & (~ex841) & (g841 > 0)
    al8 = ys2.align(ink841 & rev841, ys2.rankmap(g841, val841), val841, rad=12, coarse=3)
    M8b = ys2.masks(B841, val841, tuple(al8['best']))
    s8b = ys2.score(g841, M8b, 256, 50)
    r.update(
        o841_GATE_best=s8b['GATE'], o841_M4_best=s8b['M4'],
        o841_M2_best=s8b['M2'], o841_rAUC_best=round(float(auc_sub(g841[M8b['ridge']], g841[M8b['far']])), 4),
    )

    # 7. p306 crater floor ONLY (ZERO Table P ys2 scoring!)
    q = np.flipud(M['p306'])[150:1150, 320:2480]
    qv = q[q > 0]
    _, n_comp = ndi.label((q < 30) & (q > 0))
    r.update(
        crater_p306_p01=float(np.percentile(qv, 0.1)),
        crater_p306_px_lt30=int(((q < 30) & (q > 0)).sum()),
        crater_p306_comps=int(n_comp),
    )

    if ckpt_path and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        while isinstance(sd, dict) and 'model' in sd and isinstance(sd['model'], dict):
            sd = sd['model']
        if 'network.task_heads.ink.bias' in sd:
            r['ink_bias'] = round(float(sd['network.task_heads.ink.bias'].reshape(-1)[0]), 4)
            r['ink_bias_delta'] = round(r['ink_bias'] - SCORE_REF['table']['V3']['ink_bias'], 4)

    return r


def make_panel(crops, titles, out_path, scale=1):
    """Render clean, identically contrast-stretched (raw uint8 0..255!) side-by-side comparison PNG."""
    h, w = crops[0].shape
    pad = 10
    hdr = 34
    n = len(crops)
    canvas = np.full((hdr + h * scale + pad, n * (w * scale + pad) + pad), 28, dtype=np.uint8)
    for i, c in enumerate(crops):
        img_u8 = np.clip(np.rint(c), 0, 255).astype(np.uint8)
        if scale != 1:
            img_u8 = cv2.resize(img_u8, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        x0 = pad + i * (w * scale + pad)
        canvas[hdr:hdr + h * scale, x0:x0 + w * scale] = img_u8
    pil = Image.fromarray(canvas, mode='L').convert('RGB')
    draw = ImageDraw.Draw(pil)
    for i, t in enumerate(titles):
        x0 = pad + i * (w * scale + pad)
        draw.text((x0 + 6, 9), t, fill=(245, 245, 245))
    pil.save(out_path)
    print(f'[VIS] Saved {out_path} ({pil.size})', flush=True)


def best_ink_window(mask_bool, h=512, w=512):
    ii = cv2.integral(mask_bool.astype(np.float32))
    best = (-1.0, 0, 0)
    H, W = mask_bool.shape
    for y in range(0, max(1, H - h), 64):
        for x in range(0, max(1, W - w), 64):
            s = ii[y + h, x + w] - ii[y, x + w] - ii[y + h, x] + ii[y, x]
            if s > best[0]:
                best = (s, y, x)
    return best[1], best[1] + h, best[2], best[2] + w


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='bhand', choices=['bhand', 'deconly'])
    cli = ap.parse_args()
    pfx = cli.prefix

    t0 = time.time()
    arms_cfg = [
        (f'{pfx}_raw', os.path.join(CKPT_DIR, f'reader_pu_{pfx}_raw.pth')),
        (f'{pfx}_soup65', os.path.join(CKPT_DIR, f'reader_pu_{pfx}_soup65.pth')),
        (f'{pfx}_soup50', os.path.join(CKPT_DIR, f'reader_pu_{pfx}_soup50.pth')),
    ]

    v3_maps = load_v3_maps()
    arm_maps = {}
    for tag, ck in arms_cfg:
        arm_maps[tag] = run_infer_arm(tag, ck)

    # Prepare masks once
    mw = dict(np.load(os.path.join(RA.SU, 'w047_0_512_masks.npz')))
    mb = dict(np.load(os.path.join(RA.SU, 'blank_1024_5120_masks.npz')))
    pos, neg, bval = mw['pos'].astype(bool), mw['neg'].astype(bool), mb['valid'].astype(bool)
    core = pos & (ndi.distance_transform_edt(pos) >= 8)

    P29, S29, V29, _ = RA.dev_labels('pherc1667-w029')
    y0_29, y1_29, x0_29, x1_29 = RA.crop_box(S29 | V29, P29.shape)
    B29 = ys2.base(P29[y0_29:y1_29, x0_29:x1_29])
    P31, S31, V31, _ = RA.dev_labels('pherc1667-w031')
    mk9 = dict(np.load(os.path.join(RA.SU, '0009B_125754_masks.npz')))
    lab9, msk9 = mk9['lab'].astype(bool), mk9['msk'].astype(bool)

    d841 = os.path.join(R, 'data', 'v2', '0841', AG1)
    hd = os.path.join(d841, 'human')
    sh841 = zarr.open(os.path.join(d841, 'sv.zarr'), mode='r')['0'].shape[1:]
    ink841 = v2r.mask_on_canvas(zarr.open(os.path.join(hd, 'inklabels.zarr'), mode='r')['0'], sh841)
    rev841 = np.zeros(sh841, bool)
    for nm in ('supervision.zarr', 'validation.zarr'):
        if os.path.exists(os.path.join(hd, nm)):
            rev841 |= v2r.mask_on_canvas(zarr.open(os.path.join(hd, nm), mode='r')['0'], sh841)
    ex841 = np.zeros(sh841, bool)
    a0, a1, b0, b1 = AG1_CROP
    ex841[max(0, a0 - BUF):a1 + BUF, max(0, b0 - BUF):b1 + BUF] = True
    B841 = ys2.base(ink841 & rev841)

    masks = dict(
        w047=(pos, neg, bval, core),
        w29=(P29, S29, V29, y0_29, y1_29, x0_29, x1_29, B29),
        w31=(P31, S31, V31),
        b9=(lab9, msk9),
        o841=(ink841, rev841, ex841, sh841, B841),
    )

    results = {}
    for tag, ck in arms_cfg:
        r = score_arm(tag, arm_maps[tag], masks, ckpt_path=ck)
        results[tag] = r
        print(f'\n=== {tag} ===\n{json.dumps(r, indent=2)}', flush=True)

    # V3 morphological erosion controls on 0009B (same exact ys2 scorer!)
    erode_ctrl = {}
    m9_v3 = v3_maps['0009B_125754']
    hh, ww = min(lab9.shape[0], m9_v3.shape[0]), min(lab9.shape[1], m9_v3.shape[1])
    l9, k9 = lab9[:hh, :ww], msk9[:hh, :ww]
    B9 = ys2.base(l9)
    for rad in (0, 2, 4, 6, 8, 10):
        me = erode_map(m9_v3[:hh, :ww], rad)
        val9 = k9 & (me > 0)
        M9 = ys2.masks(B9, val9, RA.SHIFT_0009B)
        s9 = ys2.score(me, M9, 256, 50)
        al9 = ys2.align(l9, ys2.rankmap(me, val9), val9, rad=12, coarse=3)
        sb9 = ys2.score(me, ys2.masks(B9, val9, tuple(al9['best'])), 256, 50)
        erode_ctrl[f'V3_erode_r{rad}'] = dict(
            b9_M1=s9['M1'], b9_M1n=s9['M1_narrow'], b9_M1_best=sb9['M1'], b9_M1n_best=sb9['M1_narrow'],
            b9_GATE=s9['GATE'], b9_M4=s9['M4'], b9_rAUC=round(float(auc_sub(me[M9['ridge']], me[M9['far']])), 4),
        )
    print('\n=== V3 EROSION CONTROL (0009B) ===\n' + json.dumps(erode_ctrl, indent=2), flush=True)

    # Render crisp public visual comparison panels (RAW uint8 0..255, NO per-panel contrast cheating!)
    # 1. 0009B densest letter crop (512x512)
    y0, y1, x0, x1 = best_ink_window(lab9 & msk9, h=512, w=512)
    make_panel(
        [
            v3_maps['0009B_125754'][y0:y1, x0:x1],
            erode_map(v3_maps['0009B_125754'], 6)[y0:y1, x0:x1],
            arm_maps[f'{pfx}_raw']['0009B_125754'][y0:y1, x0:x1],
            arm_maps[f'{pfx}_soup65']['0009B_125754'][y0:y1, x0:x1],
            arm_maps[f'{pfx}_soup50']['0009B_125754'][y0:y1, x0:x1],
        ],
        [
            f"V3 (M1 {erode_ctrl['V3_erode_r0']['b9_M1']:.3f})",
            f"V3 erode r=6 ({erode_ctrl['V3_erode_r6']['b9_M1']:.3f})",
            f"{pfx} raw ({results[f'{pfx}_raw']['b9_M1']:.3f})",
            f"{pfx} soup65 ({results[f'{pfx}_soup65']['b9_M1']:.3f})",
            f"{pfx} soup50 ({results[f'{pfx}_soup50']['b9_M1']:.3f})",
        ],
        os.path.join(VIS_DIR, f'panel_{pfx}_0009B_letters.png'),
    )

    # 2. pherc1667-w029 densest letter crop (512x512)
    yw0, yw1, xw0, xw1 = best_ink_window(P29 & (S29 | V29), h=512, w=512)
    make_panel(
        [
            v3_maps['pherc1667-w029'][yw0:yw1, xw0:xw1],
            arm_maps[f'{pfx}_raw']['pherc1667-w029'][yw0:yw1, xw0:xw1],
            arm_maps[f'{pfx}_soup65']['pherc1667-w029'][yw0:yw1, xw0:xw1],
            arm_maps[f'{pfx}_soup50']['pherc1667-w029'][yw0:yw1, xw0:xw1],
        ],
        [
            f"V3 (w29 M1 {SCORE_REF['table']['V3']['w29_M1']:.3f})",
            f"{pfx} raw ({results[f'{pfx}_raw']['w29_M1']:.3f})",
            f"{pfx} soup65 ({results[f'{pfx}_soup65']['w29_M1']:.3f})",
            f"{pfx} soup50 ({results[f'{pfx}_soup50']['w29_M1']:.3f})",
        ],
        os.path.join(VIS_DIR, f'panel_{pfx}_w029_letters.png'),
    )

    # 3. Public 0306B p306 letter-line strip (stacked vertically for wide readability!)
    strips = [
        ('V3 (p306)', np.flipud(v3_maps['p306'])[350:850, 420:2320]),
        (f'{pfx} raw (p306)', np.flipud(arm_maps[f'{pfx}_raw']['p306'])[350:850, 420:2320]),
        (f'{pfx} soup65 (p306)', np.flipud(arm_maps[f'{pfx}_soup65']['p306'])[350:850, 420:2320]),
        (f'{pfx} soup50 (p306)', np.flipud(arm_maps[f'{pfx}_soup50']['p306'])[350:850, 420:2320]),
    ]
    sh_, sw_ = strips[0][1].shape
    hdr_, pad_ = 28, 8
    canv = np.full((len(strips) * (hdr_ + sh_ + pad_) + pad_, sw_ + 2 * pad_), 28, dtype=np.uint8)
    for i, (_, arr) in enumerate(strips):
        yy = pad_ + i * (hdr_ + sh_ + pad_) + hdr_
        canv[yy:yy + sh_, pad_:pad_ + sw_] = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    pil306 = Image.fromarray(canv, mode='L').convert('RGB')
    dr = ImageDraw.Draw(pil306)
    for i, (lbl, _) in enumerate(strips):
        yy = pad_ + i * (hdr_ + sh_ + pad_) + 6
        dr.text((pad_ + 8, yy), lbl, fill=(245, 245, 245))
    p306_vis = os.path.join(VIS_DIR, f'panel_{pfx}_p306_line.png')
    pil306.save(p306_vis)
    print(f'[VIS] Saved {p306_vis} ({pil306.size})', flush=True)

    out_json = os.path.join(HERE, f'eval_dev_{pfx}.json')
    json.dump(
        {
            'seconds': round(time.time() - t0, 1),
            'v3_ref': SCORE_REF['table']['V3'],
            'soup65_ref': SCORE_REF['table']['soup_65'],
            'v3_w053_ref': {
                'blank_w053_3200_2880': SCORE_REF['report']['blank_w053_3200_2880']['V3'],
                'blank_w053_960_5920': SCORE_REF['report']['blank_w053_960_5920']['V3'],
            },
            'v3_o841_ref': SCORE_REF['report']['0841_ag1_outside']['arms']['V3'],
            'soup65_o841_ref': SCORE_REF['report']['0841_ag1_outside']['arms']['soup_65'],
            'erode_ctrl': erode_ctrl,
            'arms': results,
        },
        open(out_json, 'w', encoding='utf-8'),
        indent=2,
    )
    print(f'\n[DONE] Saved {out_json} in {(time.time()-t0)/60.0:.1f} min', flush=True)


if __name__ == '__main__':
    main()
