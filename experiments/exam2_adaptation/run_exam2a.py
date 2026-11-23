"""Sequential runner for Exam 2A: Unseen-Protocol Scroll Adaptation (0009B 133208 -> 125754 + 140325)."""
import os
import json
import subprocess
import numpy as np
import cv2
import tifffile
from PIL import Image, ImageDraw

PY = r'C:\Users\nikox\Documents\Vesuvius\.venv\Scripts\python.exe'
HERE = os.path.dirname(os.path.abspath(__file__))
ADAPT = os.path.join(HERE, 'adapt_0009b.py')
BASE_CK = os.path.abspath(os.path.join(HERE, '..', 'exam1_from_scratch', 'ckpt', 'oracle_fullmask_half_centerline_s3000.pth'))
R = r'C:\Users\nikox\Documents\Vesuvius'

arms = ('base_eval', 'oracle_fullmask', 'naive_bce', 'collar_bce', 'collar_bce_bal', 'kiryo_nnpu', 'scribble_pu')
for arm in arms:
    jpath = os.path.join(HERE, 'adapt0009b_base_eval.json' if arm == 'base_eval' else f'adapt0009b_{arm}_s600.json')
    if os.path.exists(jpath):
        print(f'[SKIP] Already done: {arm}', flush=True)
        continue
    print(f'\n================ EXAM 2A: {arm} ================', flush=True)
    subprocess.run([PY, '-u', ADAPT, '--base_ckpt', BASE_CK, '--arm', arm, '--steps', '600', '--lr', '1e-4'], check=True)

# Aggregate summary + render 7-arm side-by-side visuals on unseen 125754 and 140325!
summary = {}
tag_map = {}
for arm in arms:
    jpath = os.path.join(HERE, 'adapt0009b_base_eval.json' if arm == 'base_eval' else f'adapt0009b_{arm}_s600.json')
    summary[arm] = json.load(open(jpath))
    tag_map[arm] = 'base_3scroll' if arm == 'base_eval' else f'adapt0009b_{arm}_s600'

with open(os.path.join(HERE, 'exam2a_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

vis_dir = os.path.join(HERE, 'visuals')
os.makedirs(vis_dir, exist_ok=True)
mk9 = dict(np.load(os.path.join(R, 'data', 'v8ft', 'suite', '0009B_125754_masks.npz')))
lab125 = mk9['lab'].astype(bool) & mk9['msk'].astype(bool)
T14 = np.load(os.path.join(R, 'data', 'p009b', '20250919140325', 'ink24_on864.npy'))
lab140 = (T14 >= 140.0)

for piece, mask_b, mkey, out_png in [
    ('0009B_125754', lab125, 'b125_GATE', 'exam2a_125754.png'),
    ('0009B_140325', lab140, 'b140_GATE', 'exam2a_140325.png'),
]:
    ii = cv2.integral(mask_b.astype(np.float32))
    best = (-1.0, 0, 0)
    H, W = mask_b.shape
    for y in range(0, max(1, H - 512), 64):
        for x in range(0, max(1, W - 512), 64):
            s = ii[y + 512, x + 512] - ii[y, x + 512] - ii[y + 512, x] + ii[y, x]
            if s > best[0]:
                best = (s, y, x)
    y0, x0 = best[1], best[2]
    pad, hdr = 10, 34
    canv = np.full((hdr + 512 + pad, len(arms) * (512 + pad) + pad), 28, dtype=np.uint8)
    for i, a in enumerate(arms):
        arr = tifffile.imread(os.path.join(HERE, 'reads', tag_map[a], f'{piece}.tif')).astype(np.float32)
        xx = pad + i * (512 + pad)
        canv[hdr:hdr + 512, xx:xx + 512] = np.clip(np.rint(arr[y0:y0 + 512, x0:x0 + 512]), 0, 255).astype(np.uint8)
    pil = Image.fromarray(canv, mode='L').convert('RGB')
    dr = ImageDraw.Draw(pil)
    for i, a in enumerate(arms):
        xx = pad + i * (512 + pad)
        dr.text((xx + 6, 9), f"{a} ({summary[a][mkey]:.3f})", fill=(245, 245, 245))
    out_p = os.path.join(vis_dir, out_png)
    pil.save(out_p)
    print(f'[VIS] Saved {out_p}', flush=True)

print('\n=== EXAM 2A COMPLETE SUMMARY ===')
print(json.dumps(summary, indent=2), flush=True)
