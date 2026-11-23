"""Generate landscape dark-theme publication figures for ScribblePU release.

Reads saved inference TIFFs, JSON results and local copies of the public scroll labels (no GPU).
The finished PNGs are already in figures/; rerunning needs those local reads and labels
(set VESUVIUS_DATA_DIR to the folder that holds them).
Produces 7 landscape cards in scribble-pu/figures/:
  - fig0_scribble_workflow.png           : one thin line per letter vs full + background masks (PHerc. Paris 4)
  - fig1_hero_first_letters_0306B.png    : Reader V3 vs Reader-PU on the awarded First Letters row (PHerc. 0306B)
  - fig2_counter_opening_and_erosion.png : w029 (B E) & 0009B (Delta/Alpha) vs V3 + Morphological Erosion
  - fig3_exam1_from_scratch_scribbles.png: 5-column from-scratch comparison on unseen 0009B & 0814H
  - fig4_exam2_adaptation_and_alpha.png  : Unseen-protocol adaptation bar chart + 5-point alpha sweep
  - fig5_adaptation_0009b.png            : 0009B adaptation reads on two held-out segments
  - fig6_koine_trainer_check.png         : stock koine_machines trainer, default loss vs BrierNNPULoss
"""
import os
import sys
import json
import argparse
import numpy as np
import cv2
import zarr
import glob
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
R = os.environ.get('VESUVIUS_DATA_DIR') or os.path.abspath(os.path.join(REPO, '..', '..', 'Vesuvius'))
FIG_DIR = os.path.join(REPO, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

EX3 = os.path.join(REPO, 'experiments', 'exam3_reader_pu')
EX1 = os.path.join(REPO, 'experiments', 'exam1_from_scratch')
EX2 = os.path.join(REPO, 'experiments', 'exam2_adaptation')

BG_HEX = '#121318'
PANEL_HEX = '#1b1d24'
ORANGE = '#ff8c00'
TEXT_HI = '#f4f5f7'
TEXT_LO = '#a8aebc'
GREEN = '#38d39f'
RED = '#ff5c5c'
BLUE = '#4ea1ff'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']


def erode_map(m, r):
    if r <= 0:
        return m.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    out = cv2.erode(m.astype(np.float32), k)
    out[m == 0] = 0
    return out


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


def style_img_ax(ax, img, title, subtitle='', highlight=False):
    ax.imshow(np.clip(img, 0, 255), cmap='gray', vmin=0, vmax=255, interpolation='bilinear')
    ax.set_xticks([])
    ax.set_yticks([])
    color = ORANGE if highlight else '#353945'
    lw = 3.5 if highlight else 1.2
    for sp in ax.spines.values():
        sp.set_edgecolor(color)
        sp.set_linewidth(lw)
    ax.set_title(title, color=(ORANGE if highlight else TEXT_HI), fontsize=13, fontweight='bold', pad=8, loc='left')
    if subtitle:
        ax.text(0.0, -0.04, subtitle, transform=ax.transAxes, color=TEXT_LO, fontsize=10, va='top', ha='left')


def extract_exact_ridge(mask_bool):
    """Exact 1D-directional medial spine extractor from kt_nnpu.py (used in Exam 1 & Reader-PU)."""
    m = mask_bool.astype(np.uint8)
    if not m.any():
        return np.zeros_like(mask_bool, dtype=bool)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    p = np.pad(dt, 1, mode='constant')
    c = p[1:-1, 1:-1]
    h = (c >= p[1:-1, :-2]) & (c >= p[1:-1, 2:]) & ((c > p[1:-1, :-2]) | (c > p[1:-1, 2:]))
    v = (c >= p[:-2, 1:-1]) & (c >= p[2:, 1:-1]) & ((c > p[:-2, 1:-1]) | (c > p[2:, 1:-1]))
    d1 = (c >= p[:-2, :-2]) & (c >= p[2:, 2:]) & ((c > p[:-2, :-2]) | (c > p[2:, 2:]))
    d2 = (c >= p[:-2, 2:]) & (c >= p[2:, :-2]) & ((c > p[:-2, 2:]) | (c > p[2:, :-2]))
    r = ((c >= 1.5) & (h | v | d1 | d2)) | (
        (c >= 0.9) & ((h.astype(np.int8) + v.astype(np.int8) + d1.astype(np.int8) + d2.astype(np.int8)) >= 2)
    )
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k3 = np.ones((3, 3), np.uint8)
    rs = r & (dt >= 0.65 * cv2.dilate(dt, k15)) & (m > 0)
    return (cv2.dilate(rs.astype(np.uint8), k3) > 0) & (m > 0)


def zhang_suen_thin(mask):
    """Zhang-Suen thinning (pure numpy): one-pixel-wide skeleton of a binary mask."""
    img = np.pad(mask.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P = img
            p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            p8, p9 = P[1:-1, :-2], P[:-2, :-2]
            c = P[1:-1, 1:-1]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(x.astype(np.int16) for x in nb)
            seq = nb + [p2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (c == 1) & (B >= 2) & (B <= 6) & (A == 1) & cond
            if rm.any():
                img[1:-1, 1:-1][rm] = 0
                changed = True
    return img[1:-1, 1:-1].astype(bool)


def longest_skeleton_path(skel_pts):
    """Longest geodesic path through one skeleton component (two BFS passes over 8-connected pixels)."""
    from collections import deque
    pts = set(map(tuple, skel_pts))

    def bfs(src):
        prev = {src: None}
        dq = deque([src])
        last = src
        while dq:
            p = dq.popleft()
            last = p
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (p[0] + dy, p[1] + dx)
                    if q in pts and q not in prev:
                        prev[q] = p
                        dq.append(q)
        return last, prev

    a, _ = bfs(next(iter(pts)))
    b, prev = bfs(a)
    path = []
    p = b
    while p is not None:
        path.append(p)
        p = prev[p]
    return np.array(path, dtype=np.float32)


def make_fig0():
    """Workflow card (no results): old full + background masks vs ScribblePU one clean line per traced letter."""
    sd = os.path.join(R, 'data', 'ink9um', 'labels', 'aligned-scrollprizeorg-21slices', 'phercparis4-w05')
    pz = zarr.open(glob.glob(os.path.join(sd, '*_inklabels.zarr'))[0], mode='r')['0']
    sz = zarr.open(glob.glob(os.path.join(sd, '*_supervision_mask.zarr'))[0], mode='r')['0']
    y0, y1, x0, x1 = 7820, 8720, 11520, 13520
    f_crop = (np.asarray(pz[:, y0:y1, x0:x1]) > 0).any(0)
    v_crop = (np.asarray(sz[:, y0:y1, x0:x1]) > 0).any(0)
    f_crop = f_crop & v_crop

    ref = cv2.GaussianBlur(f_crop.astype(np.float32), (11, 11), 3.0) * 0.42 + 0.10
    vis_old = np.stack([ref, ref, ref], axis=-1)
    bg_mask = v_crop & (~f_crop)
    vis_old[bg_mask] = vis_old[bg_mask] * 0.30 + np.array([0.05, 0.38, 0.36]) * 0.70
    vis_old[f_crop] = np.array([0.94, 0.97, 1.00])

    # One clean line per traced letter: smoothed silhouette -> skeleton -> longest path (no side branches) -> smoothed
    f_smooth = cv2.GaussianBlur(f_crop.astype(np.float32), (21, 21), 6.0) >= 0.5
    n_cc, cc_map, stats, cents = cv2.connectedComponentsWithStats(f_smooth.astype(np.uint8), 8)
    # Every other letter in each row (left to right), so half the letters are traced and half are skipped
    keep_ids = np.zeros(n_cc, dtype=bool)
    big = [i for i in range(1, n_cc) if stats[i, cv2.CC_STAT_AREA] >= 400]
    for row in (True, False):
        ids = sorted([i for i in big if (cents[i][1] < f_crop.shape[0] / 2) == row], key=lambda i: cents[i][0])
        for j, i in enumerate(ids):
            keep_ids[i] = (j % 2 == 0)
    skel = zhang_suen_thin(f_smooth)
    vis_new = np.stack([ref, ref, ref], axis=-1)
    canvas = (np.clip(vis_new, 0, 1) * 255).astype(np.uint8).copy()
    n_lines = 0
    for i in range(1, n_cc):
        if not keep_ids[i] or stats[i, cv2.CC_STAT_AREA] < 400:
            continue
        pts = np.argwhere(skel & (cc_map == i))
        if len(pts) < 12:
            continue
        path = longest_skeleton_path(pts)
        k = 9
        if len(path) > k:
            ker = np.ones(k) / k
            ys = np.convolve(np.pad(path[:, 0], k // 2, mode='edge'), ker, mode='valid')
            xs = np.convolve(np.pad(path[:, 1], k // 2, mode='edge'), ker, mode='valid')
            path = np.stack([ys, xs], 1)
        poly = np.round(path[:, ::-1] * 4).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], False, (255, 143, 36), thickness=8, lineType=cv2.LINE_AA, shift=2)
        n_lines += 1

    fig = plt.figure(figsize=(16.5, 5.6), facecolor=BG_HEX)
    gs = fig.add_gridspec(1, 2, left=0.025, right=0.975, top=0.80, bottom=0.10, wspace=0.055)
    fig.text(0.025, 0.915, 'ScribblePU: one quick line through the letters you are sure of, instead of painting full masks',
             color=TEXT_HI, fontsize=18, fontweight='bold')
    fig.text(0.025, 0.865, 'Illustration on real hand-labelled Greek letters (PHerc. Paris 4 w05).',
             color=TEXT_LO, fontsize=11.0)
    ax_old = fig.add_subplot(gs[0, 0])
    ax_old.imshow(np.clip(vis_old, 0, 1), interpolation='bilinear')
    ax_old.set_xticks([])
    ax_old.set_yticks([])
    for sp in ax_old.spines.values():
        sp.set_edgecolor('#4a5568')
        sp.set_linewidth(1.4)
    ax_old.set_title('Before: paint every letter in full (white) + mark the background (teal)',
                     color=TEXT_HI, fontsize=12, fontweight='bold', pad=8, loc='left')
    ax_old.text(0.0, -0.045, 'Any ink you leave unpainted is learned as background',
                transform=ax_old.transAxes, color=TEXT_LO, fontsize=10, va='top', ha='left')
    ax_new = fig.add_subplot(gs[0, 1])
    ax_new.imshow(canvas, interpolation='bilinear')
    ax_new.set_xticks([])
    ax_new.set_yticks([])
    for sp in ax_new.spines.values():
        sp.set_edgecolor(ORANGE)
        sp.set_linewidth(3.2)
    ax_new.set_title('ScribblePU: one thin line per letter you are sure of (orange), nothing else',
                     color=ORANGE, fontsize=12, fontweight='bold', pad=8, loc='left')
    ax_new.text(0.0, -0.045, 'Grey letters are simply left unlabelled. No background mask.',
                transform=ax_new.transAxes, color=TEXT_LO, fontsize=10, va='top', ha='left')
    out = os.path.join(FIG_DIR, 'fig0_scribble_workflow.png')
    fig.savefig(out, dpi=150, facecolor=BG_HEX)
    plt.close(fig)
    print(f'[FIG0] Saved {out} ({n_lines} lines)', flush=True)


def make_fig5():
    """0009B adaptation reads on their own: two held-out segments, windows chosen by TRUE ink density (not by result)."""
    summ = json.load(open(os.path.join(EX2, 'exam2a_v2_summary.json')))
    arms = ['adapt0009b_v2_base', 'adapt0009b_v2_naive_bce_s600', 'adapt0009b_v2_scribble_pu_s600', 'adapt0009b_v2_oracle_fullmask_s600']
    mk9 = dict(np.load(os.path.join(R, 'data', 'v8ft', 'suite', '0009B_125754_masks.npz')))
    r125 = [tifffile.imread(os.path.join(EX2, 'reads', a, '0009B_125754.tif')).astype(np.float32) for a in arms]
    r140 = [tifffile.imread(os.path.join(EX2, 'reads', a, '0009B_140325.tif')).astype(np.float32) for a in arms]
    ay0, ay1, ax0, ax1 = best_ink_window(mk9['lab'].astype(bool) & mk9['msk'].astype(bool), 700, 900)
    T14 = np.load(os.path.join(R, 'data', 'p009b', '20250919140325', 'ink24_on864.npy')).astype(np.float32)
    h14, w14 = min(T14.shape[0], r140[0].shape[0]), min(T14.shape[1], r140[0].shape[1])
    by0, by1, bx0, bx1 = best_ink_window((T14[:h14, :w14] >= 140.0), 700, 900)
    fig = plt.figure(figsize=(16.5, 9.0), facecolor=BG_HEX)
    gs = fig.add_gridspec(2, 4, left=0.025, right=0.975, top=0.85, bottom=0.05, wspace=0.048, hspace=0.16)
    fig.text(0.025, 0.945, 'A different fragment and a scan type the model never saw: PHerc. 0009B (116 keV / 8.64 um)',
             color=TEXT_HI, fontsize=17, fontweight='bold')
    fig.text(0.025, 0.905, 'Two held-out segments after about 9 minutes of adapting the 3-scroll model on ANOTHER segment (133208). '
             'Windows chosen by true ink density. Same input, same raw 0-255 scale.', color=TEXT_LO, fontsize=11.0)
    b, n, s, o = summ['base_eval'], summ['naive_bce'], summ['scribble_pu'], summ['oracle_fullmask']
    rows = [
        (r125, (ay0, ay1, ax0, ax1), 'b125_GATE', 'segment 125754'),
        (r140, (by0, by1, bx0, bx1), 'b140_GATE', 'segment 140325'),
    ]
    titles = ['Before adapting', 'Default training, same lines', 'ScribblePU: 1-px lines on <1% of the ink', 'Full masks (upper bound)']
    for ri, (reads, (y0, y1, x0, x1), key, seg) in enumerate(rows):
        subs = [f"{seg}: GATE {b[key]:.4f}", f"{n[key]:.4f} (worse)",
                f"{s[key]:.4f} ({100 * (s[key] - b[key]) / (o[key] - b[key]):.1f}% of the full-mask gain)", f"{o[key]:.4f}"]
        for c in range(4):
            ax = fig.add_subplot(gs[ri, c])
            style_img_ax(ax, reads[c][y0:y1, x0:x1], titles[c] if ri == 0 else '', subtitle=subs[c], highlight=(c == 2))
    out = os.path.join(FIG_DIR, 'fig5_adaptation_0009b.png')
    fig.savefig(out, dpi=150, facecolor=BG_HEX)
    plt.close(fig)
    print(f'[FIG5] Saved {out}', flush=True)


def make_fig6():
    """Stock koine_machines trainer, same FIXED-half line labels: default BCE vs ScribblePU vs the held-out hand labels."""
    kd = os.path.join(R, 'data', 'scribblepu', 'koine_smoke')
    res = json.load(open(os.path.join(kd, 'results_k.json')))
    rd = os.path.join(kd, 'reads')
    k1_8 = tifffile.imread(os.path.join(rd, 'k1', 'pherc0814-46527.tif')).astype(np.float32)
    k2_8 = tifffile.imread(os.path.join(rd, 'k2', 'pherc0814-46527.tif')).astype(np.float32)
    s814 = os.path.join(R, 'data', 'ink9um', 'labels', 'aligned-scrollprizeorg-21slices', 'pherc0814-46527')
    p8 = (np.asarray(zarr.open(glob.glob(os.path.join(s814, '*_inklabels.zarr'))[0], mode='r')['0']) > 0).any(0)
    h8 = min(p8.shape[0], k1_8.shape[0]); w8 = min(p8.shape[1], k1_8.shape[1])
    sy0, sy1, sx0, sx1 = best_ink_window(p8[:h8, :w8], 560, 900)
    lab = p8[sy0:sy1, sx0:sx1].astype(np.float32) * 175.0 + 40.0
    k1, k2 = res['k1'], res['k2']
    fig = plt.figure(figsize=(16.5, 5.0), facecolor=BG_HEX)
    gs = fig.add_gridspec(1, 3, left=0.025, right=0.975, top=0.74, bottom=0.11, wspace=0.04)
    fig.text(0.025, 0.915, "In the team's own trainer (koine_machines), with half the letters never labelled",
             color=TEXT_HI, fontsize=17, fontweight='bold')
    fig.text(0.025, 0.850, 'Trained from scratch on 3 other scrolls with thin lines through a FIXED half of the letters, no background masks. '
             f'Held-out PHerc. 0814, never trained on. Unseen 0009B: GATE {k1["b9_GATE"]:.4f} vs {k2["b9_GATE"]:.4f}.',
             color=TEXT_LO, fontsize=11.0)
    panels = [
        (k2_8[sy0:sy1, sx0:sx1], 'Default training, same labels', f"GATE {k2['s814_GATE']:.4f}: flat grey", False),
        (k1_8[sy0:sy1, sx0:sx1], 'ScribblePU, same labels', f"GATE {k1['s814_GATE']:.4f}: the letters come out", True),
        (lab, 'Hand ink labels for this window (never used)', 'the true letters, for comparison', False),
    ]
    for c, (img, title, sub, hi) in enumerate(panels):
        ax = fig.add_subplot(gs[0, c])
        style_img_ax(ax, img, title, subtitle=sub, highlight=hi)
    out = os.path.join(FIG_DIR, 'fig6_koine_trainer_check.png')
    fig.savefig(out, dpi=150, facecolor=BG_HEX)
    plt.close(fig)
    print(f'[FIG6] Saved {out}', flush=True)


def make_fig1():
    """Hero landscape card: PHerc. 0306B winning First Letters Prize row (Reader V3 vs Reader-PU)."""
    e3 = json.load(open(os.path.join(EX3, 'eval_dev_bhand.json')))
    v3_p306 = np.flipud(tifffile.imread(os.path.join(R, 'data', 'v3l', 'p306', 'v3_reverse.tif')).astype(np.float32))
    pu_p306 = np.flipud(tifffile.imread(os.path.join(EX3, 'reads', 'bhand_raw', 'p306.tif')).astype(np.float32))
    crop_v3 = v3_p306[350:850, 420:2320]
    crop_pu = pu_p306[350:850, 420:2320]

    fig, axes = plt.subplots(2, 1, figsize=(16, 9.4), facecolor=BG_HEX)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.84, bottom=0.055, hspace=0.22)

    fig.text(0.025, 0.945, 'The awarded PHerc. 0306B First Letters row (Warner & Bodill): Reader V3, then V3 retuned with ScribblePU',
             color=TEXT_HI, fontsize=19, fontweight='bold')
    fig.text(0.025, 0.900,
             'Retuned in 15 minutes from thin lines on OTHER scrolls\' letters: it never saw this row. '
             'Same network, same 9 um scan, same inference command, same 0-255 display scale.',
             color=TEXT_LO, fontsize=11)

    v3_p01 = e3['v3_ref']['crater_p306_p01']
    pu_p01 = e3['arms']['bhand_raw']['crater_p306_p01']
    pu_lt30 = e3['arms']['bhand_raw']['crater_p306_px_lt30']

    style_img_ax(
        axes[0], crop_v3,
        'Reader V3',
        highlight=False,
    )
    style_img_ax(
        axes[1], crop_pu,
        'Reader V3 + ScribblePU (Reader-PU): fused strokes thin out, letter gaps open',
        highlight=True,
    )

    out = os.path.join(FIG_DIR, 'fig1_hero_first_letters_0306B.png')
    fig.savefig(out, dpi=150, facecolor=BG_HEX)
    plt.close(fig)
    print(f'[FIG1] Saved {out}', flush=True)


def make_fig2():
    """2x3 landscape grid: w029 (B E) and 0009B (Delta/Alpha) showing Reader-PU beats morphological erosion."""
    e3 = json.load(open(os.path.join(EX3, 'eval_dev_bhand.json')))
    # Load w029
    v3_29 = tifffile.imread(os.path.join(EX3, 'reads', 'v3_ref', 'pherc1667-w029.tif')).astype(np.float32)
    pu_29 = tifffile.imread(os.path.join(EX3, 'reads', 'bhand_raw', 'pherc1667-w029.tif')).astype(np.float32)
    sd29 = os.path.join(R, 'data', 'ink9um', 'labels', 'aligned-scrollprizeorg-21slices', 'pherc1667-w029')
    P29 = (np.asarray(zarr.open(glob.glob(os.path.join(sd29, '*_inklabels.zarr'))[0], mode='r')['0']) > 0).any(0)
    S29 = (np.asarray(zarr.open(glob.glob(os.path.join(sd29, '*_supervision_mask.zarr'))[0], mode='r')['0']) > 0).any(0)
    V29_f = glob.glob(os.path.join(sd29, '*_validation_mask.zarr'))
    V29 = (np.asarray(zarr.open(V29_f[0], mode='r')['0']) > 0).any(0) if V29_f else np.zeros_like(S29)
    yw0, yw1, xw0, xw1 = best_ink_window(P29 & (S29 | V29), 512, 512)

    # Load 0009B
    v3_9 = tifffile.imread(os.path.join(R, 'data', 'v8ft', 'eval', 'long384', '0009B_125754_raw.tif')).astype(np.float32)
    pu_9 = tifffile.imread(os.path.join(EX3, 'reads', 'bhand_raw', '0009B_125754.tif')).astype(np.float32)
    mk9 = dict(np.load(os.path.join(R, 'data', 'v8ft', 'suite', '0009B_125754_masks.npz')))
    y90, y91, x90, x91 = best_ink_window(mk9['lab'].astype(bool) & mk9['msk'].astype(bool), 512, 512)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.6), facecolor=BG_HEX)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.86, bottom=0.05, wspace=0.06, hspace=0.20)

    fig.text(0.025, 0.950, 'Real Counter Opening Beyond Morphological Erosion (Zero Gap Weights, Zero Negative Masks)',
             color=TEXT_HI, fontsize=18.5, fontweight='bold')
    fig.text(0.025, 0.908,
             'Top: PHerc. 1667 w029 (a segment V3 never trained on). Bottom: held-out PHerc. 0009B segment 125754 (116 keV / 8.64 um). '
             'Windows chosen strictly from ground-truth ink density. Raw 0-255 scale.',
             color=TEXT_LO, fontsize=11)

    r_raw = e3['arms']['bhand_raw']
    v3_ref = e3['v3_ref']
    er = e3['erode_ctrl']

    style_img_ax(axes[0, 0], v3_29[yw0:yw1, xw0:xw1],
                 f"Reader V3   (w029 M1 {v3_ref['w29_M1']:.4f} / M1n {v3_ref['w29_M1n']:.4f})",
                 'Loops of B fused; E blurred into solid white blob')
    style_img_ax(axes[0, 1], erode_map(v3_29, 6)[yw0:yw1, xw0:xw1],
                 'V3 + Morphological Erosion (r=6 px)',
                 'Thins outer strokes blindly; E counter still filled')
    style_img_ax(axes[0, 2], pu_29[yw0:yw1, xw0:xw1],
                 f"Reader-PU   (w029 M1 {r_raw['w29_M1']:.4f} / M1n {r_raw['w29_M1n']:.4f})",
                 '+0.092 M1 / +0.172 M1n over V3; both B loops & E cavities open',
                 highlight=True)

    style_img_ax(axes[1, 0], v3_9[y90:y91, x90:x91],
                 f"Reader V3   (0009B M1 {er['V3_erode_r0']['b9_M1']:.4f} / M1n {er['V3_erode_r0']['b9_M1n']:.4f})",
                 'Hollow triangular counter nearly closed')
    style_img_ax(axes[1, 1], erode_map(v3_9, 10)[y90:y91, x90:x91],
                 f"V3 + Erosion Peak r=10   (M1 {er['V3_erode_r10']['b9_M1']:.4f} / M1n {er['V3_erode_r10']['b9_M1n']:.4f})",
                 'Best possible morphological erosion on 0009B')
    style_img_ax(axes[1, 2], pu_9[y90:y91, x90:x91],
                 f"Reader-PU   (0009B M1 {r_raw['b9_M1']:.4f} / M1n {r_raw['b9_M1n']:.4f})",
                 '+0.046 over V3, +0.023 over peak erosion, p99=210 (V3=209)',
                 highlight=True)

    out = os.path.join(FIG_DIR, 'fig2_counter_opening_and_erosion.png')
    fig.savefig(out, dpi=150, facecolor=BG_HEX)
    plt.close(fig)
    print(f'[FIG2] Saved {out}', flush=True)


def make_fig3():
    """Exam 1 landscape card: 5 matched from-scratch models on unseen 0009B and held-out PHerc. 0814 (0814H)."""
    res = json.load(open(os.path.join(EX1, 'exam1_7arm_s5k_results.json')))
    mk9 = dict(np.load(os.path.join(R, 'data', 'v8ft', 'suite', '0009B_125754_masks.npz')))
    y90, y91, x90, x91 = best_ink_window(mk9['lab'].astype(bool) & mk9['msk'].astype(bool), 512, 512)

    s814_dir = os.path.join(R, 'data', 'ink9um', 'labels', 'aligned-scrollprizeorg-21slices', 'pherc0814-46527')
    P814 = (np.asarray(zarr.open(glob.glob(os.path.join(s814_dir, '*_inklabels.zarr'))[0], mode='r')['0']) > 0).any(0)
    S814 = (np.asarray(zarr.open(glob.glob(os.path.join(s814_dir, '*_supervision_mask.zarr'))[0], mode='r')['0']) > 0).any(0)
    y80, y81, x80, x81 = best_ink_window(P814 & S814, 512, 512)

    cols = [
        ('oracle_s5k',          'Full-Mask Oracle\n(100% mask + sup)', False),
        ('scribble_pu_rdg_s5k', 'ScribblePU\n(6.8% scribbles, 0 bg)', True),
        ('collar_bce_bal_s5k',  'Collar BCE Balanced\n(12px ignore, pw=14.8)', False),
        ('kiryo_nnpu_s5k',      'Vanilla Kiryo nnPU\n(gradient ascent)', False),
        ('collar_bce_s5k',      'Collar BCE Default\n(villa ls=0.5, 12px)', False),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(18.5, 8.8), facecolor=BG_HEX)
    fig.subplots_adjust(left=0.02, right=0.985, top=0.81, bottom=0.06, wspace=0.05, hspace=0.26)

    fig.text(0.02, 0.945, 'Exam 1: Training From Scratch With Only 6.8% Positive-Only Scribbles & Zero Background Masks',
             color=TEXT_HI, fontsize=19, fontweight='bold')
    fig.text(0.02, 0.895,
             'Exact same 34.5M-param 2.5D UNet, seed, lr=1e-3, 5,000 steps on 3 scrolls (0139N, 1667H, Paris4H). '
             'Top: unseen Fragment 0009B (unseen 116 keV / 8.64 um protocol). Bottom: held-out PHerc. 0814 (0814H). Raw 0-255 scale.',
             color=TEXT_LO, fontsize=11)

    for j, (tag, lbl, hi) in enumerate(cols):
        r = res[tag]
        m9 = tifffile.imread(os.path.join(EX1, 'reads', tag, '0009B_125754.tif')).astype(np.float32)
        m8 = tifffile.imread(os.path.join(EX1, 'reads', tag, 'pherc0814-46527.tif')).astype(np.float32)
        style_img_ax(
            axes[0, j], m9[y90:y91, x90:x91],
            lbl,
            f"0009B GATE {r['b9_GATE']:.3f} | M4 {r['b9_M4']:.3f} | p99 {r['b9_p99']:.0f}",
            highlight=hi,
        )
        style_img_ax(
            axes[1, j], m8[y80:y81, x80:x81],
            f"0814H GATE {r['s814_GATE']:.3f} | M4 {r['s814_M4']:.3f}",
            f"FA80 {r['FA80']:.3f} | cP20 {r['cP20']:.0f}",
            highlight=hi,
        )

    out = os.path.join(FIG_DIR, 'fig3_exam1_from_scratch_scribbles.png')
    fig.savefig(out, dpi=150, facecolor=BG_HEX)
    plt.close(fig)
    print(f'[FIG3] Saved {out}', flush=True)


def make_fig4():
    """Exam 2A landscape 2-panel chart: unseen-protocol adaptation gain + 5-point alpha robustness."""
    s_path = os.path.join(EX2, 'exam2a_v2_summary.json')
    a_path = os.path.join(EX2, 'exam2a_alpha_sweep.json')
    if not (os.path.exists(s_path) and os.path.exists(a_path)):
        print('[FIG4] Waiting for Exam 2A v2 + alpha sweep JSONs...', flush=True)
        return
    summ = json.load(open(s_path))
    sweep = json.load(open(a_path))
    base140 = summ['base_eval']['b140_GATE']
    base125 = summ['base_eval']['b125_GATE']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2), facecolor=BG_HEX)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.80, bottom=0.18, wspace=0.22)

    fig.text(0.06, 0.935, 'Exam 2: Adapting to an Unseen Scan Protocol (0009B 116 keV) From Lines on Under 1% of the Ink',
             color=TEXT_HI, fontsize=17.5, fontweight='bold')
    fig.text(0.06, 0.880,
             'One scribbled segment (133208, zero background masks) + 1:1 source replay -> evaluated on two held-out segments (140325 & 125754).',
             color=TEXT_LO, fontsize=11)

    # Panel 1: Held-out segment GATE gain
    arms = [
        ('naive_bce',       'Naive BCE\n(same lines)',   RED),
        ('collar_bce',      'Collar BCE\n(12px)',        RED),
        ('collar_bce_bal',  'Collar BCE\nBalanced',      '#7e8799'),
        ('kiryo_nnpu',      'Kiryo nnPU\n(ascent)',      BLUE),
        ('scribble_pu',     'ScribblePU\n(No-Ascent)',   ORANGE),
        ('oracle_fullmask', 'Full-Mask\nOracle (100%)',  GREEN),
    ]
    x = np.arange(len(arms))
    d140 = [summ[k]['b140_GATE'] - base140 for k, _, _ in arms]
    d125 = [summ[k]['b125_GATE'] - base125 for k, _, _ in arms]
    w = 0.36

    ax1.set_facecolor(PANEL_HEX)
    bars1 = ax1.bar(x - w/2, d140, width=w, color=[c for _, _, c in arms], alpha=0.95, label='Held-out segment 140325 (hard)')
    bars2 = ax1.bar(x + w/2, d125, width=w, color=[c for _, _, c in arms], alpha=0.50, edgecolor='white', linewidth=0.8, label='Held-out segment 125754')
    ax1.axhline(0.0, color='#d0d4dc', linewidth=1.0, linestyle='--')
    for b in bars1:
        h = b.get_height()
        va = 'bottom' if h >= 0 else 'top'
        dy = 0.002 if h >= 0 else -0.002
        ax1.text(b.get_x() + b.get_width()/2, h + dy, f'{h:+.3f}', ha='center', va=va, color=TEXT_HI, fontsize=9, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl, _ in arms], color=TEXT_HI, fontsize=9.5)
    ax1.set_ylabel('GATE Gain vs Unadapted 3-Scroll Base', color=TEXT_HI, fontsize=11)
    ax1.set_title('A. Held-Out Segment Adaptation Gain (600 steps)', color=TEXT_HI, fontsize=12.5, fontweight='bold', pad=10)
    ax1.set_ylim(-0.085, 0.096)
    ax1.tick_params(colors=TEXT_LO)
    ax1.legend(facecolor=BG_HEX, edgecolor='#353945', labelcolor=TEXT_HI, fontsize=9, loc='lower right')
    for sp in ax1.spines.values():
        sp.set_edgecolor('#353945')

    # Panel 2: 5-point Alpha robustness curve
    mults = ['0.50x', '0.75x', '1.00x', '1.25x', '1.50x']
    mx = [0.50, 0.75, 1.00, 1.25, 1.50]
    g140 = [sweep[m]['b140_GATE'] for m in mults]
    g125 = [sweep[m]['b125_GATE'] for m in mults]
    g814 = [sweep[m]['s814_GATE'] for m in mults]

    ax2.set_facecolor(PANEL_HEX)
    ax2.plot(mx, g125, '-o', color=ORANGE, linewidth=2.5, markersize=8, label='0009B 125754 GATE (held-out)')
    ax2.plot(mx, g140, '-s', color=GREEN, linewidth=2.5, markersize=8, label='0009B 140325 GATE (held-out)')
    ax2.plot(mx, g814, '-^', color=BLUE, linewidth=2.0, markersize=7, label='0814H GATE (no forgetting)')
    ax2.axhline(base125, color=ORANGE, linestyle=':', alpha=0.5, label=f'Base 125754 ({base125:.3f})')
    ax2.axhline(base140, color=GREEN, linestyle=':', alpha=0.5, label=f'Base 140325 ({base140:.3f})')
    for xi, yi in zip(mx, g140):
        ax2.text(xi, yi + 0.012, f'{yi:.3f}', ha='center', color=GREEN, fontsize=9, fontweight='bold')
    for xi, yi in zip(mx, g125):
        ax2.text(xi, yi + 0.012, f'{yi:.3f}', ha='center', color=ORANGE, fontsize=9, fontweight='bold')
    ax2.set_xticks(mx)
    ax2.set_xticklabels(mults, color=TEXT_HI, fontsize=10)
    ax2.set_xlabel('Multiplier on True Ink-to-Scribble Ratio (alpha)', color=TEXT_HI, fontsize=11)
    ax2.set_ylabel('Held-Out GATE Score', color=TEXT_HI, fontsize=11)
    ax2.set_title('B. Prior Sensitivity: Over-Guessing alpha Is Safe', color=TEXT_HI, fontsize=12.5, fontweight='bold', pad=10)
    ax2.set_ylim(min(min(g140), base140) - 0.05, 1.00)
    ax2.tick_params(colors=TEXT_LO)
    ax2.legend(facecolor=BG_HEX, edgecolor='#353945', labelcolor=TEXT_HI, fontsize=8.5, loc='center right', bbox_to_anchor=(0.98, 0.56))
    for sp in ax2.spines.values():
        sp.set_edgecolor('#353945')

    out = os.path.join(FIG_DIR, 'fig4_exam2_adaptation_and_alpha.png')
    fig.savefig(out, dpi=150, facecolor=BG_HEX)
    plt.close(fig)
    print(f'[FIG4] Saved {out}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=['fig0', 'fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6'])
    args = ap.parse_args()
    if 'fig0' in args.only: make_fig0()
    if 'fig1' in args.only: make_fig1()
    if 'fig2' in args.only: make_fig2()
    if 'fig3' in args.only: make_fig3()
    if 'fig4' in args.only: make_fig4()
    if 'fig5' in args.only: make_fig5()
    if 'fig6' in args.only: make_fig6()
