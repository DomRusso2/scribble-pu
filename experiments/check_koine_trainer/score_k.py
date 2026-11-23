"""Check K read + score (PREREG_TWO_CHECKS.md, frozen 2026-11-23 14:13:17).

Uses Exam 1's scorer unchanged (experiments/exam1_from_scratch/eval_exam1.py: infer_arm + score_arm -> b9 GATE / p99,
FA80, 0814H, ...) with its read folder redirected here, plus Exam 2's 0009B 140325 scoring (report-only), copied
verbatim from adapt_0009b.py eval_adapted() step 2. Usage: python score_k.py k1=<ckpt> k2=<ckpt>
"""
import json
import os
import sys

import numpy as np

K = r'C:\Users\nikox\Documents\Vesuvius\data\scribblepu\koine_smoke'
E1_DIR = r'C:\Users\nikox\Documents\antigravity\scribble-pu\experiments\exam1_from_scratch'
sys.path.insert(0, E1_DIR)
import eval_exam1 as E1  # noqa: E402  (sets its own sys.path: vendor koine, read_1a, ys2, v8ft_eval)
from scipy import ndimage as ndi  # noqa: E402

E1.READ_DIR = os.path.join(K, 'reads')
os.makedirs(E1.READ_DIR, exist_ok=True)
P009B = os.path.join(E1.R, 'data', 'p009b')
E1.JOBS = list(E1.JOBS) + [('0009B_140325', os.path.join(P009B, '20250919140325', 'sv864.zarr'), 'forward')]


def score_140325(m14):
    T14 = np.load(os.path.join(P009B, '20250919140325', 'ink24_on864.npy')).astype(np.float32)
    h14, w14 = min(T14.shape[0], m14.shape[0]), min(T14.shape[1], m14.shape[1])
    lab14 = (T14[:h14, :w14] >= 140.0)
    msk14 = (T14[:h14, :w14] > 0) & (m14[:h14, :w14] > 0)
    M14 = E1.ys2.masks(E1.ys2.base(lab14), msk14, E1.RA.SHIFT_0009B)
    s14 = E1.ys2.score(m14[:h14, :w14], M14, 256, 50)
    return dict(b140_GATE=s14['GATE'], b140_M4=s14['M4'], b140_M1=s14['M1'],
                b140_p99=float(np.percentile(m14[:h14, :w14][msk14], 99)))


def main():
    mw = dict(np.load(os.path.join(E1.RA.SU, 'w047_0_512_masks.npz')))
    mb = dict(np.load(os.path.join(E1.RA.SU, 'blank_1024_5120_masks.npz')))
    pos, neg, bval = mw['pos'].astype(bool), mw['neg'].astype(bool), mb['valid'].astype(bool)
    core = pos & (ndi.distance_transform_edt(pos) >= 8)
    mk9 = dict(np.load(os.path.join(E1.RA.SU, '0009B_125754_masks.npz')))
    masks = dict(w047=(pos, neg, bval, core), b9=(mk9['lab'].astype(bool), mk9['msk'].astype(bool)),
                 s814=E1.load_0814_labels())
    out_p = os.path.join(K, 'results_k.json')
    res = json.load(open(out_p)) if os.path.exists(out_p) else {}
    for pair in sys.argv[1:]:
        tag, ck = pair.split('=', 1)
        M = E1.infer_arm(tag, ck)
        r = E1.score_arm(M, masks)
        r.update(score_140325(M['0009B_140325']))
        r['ckpt'] = ck
        res[tag] = r
        print(f'=== {tag} ===\n{json.dumps(r, indent=1)}', flush=True)
        json.dump(res, open(out_p, 'w'), indent=1)


if __name__ == '__main__':
    main()
