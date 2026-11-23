"""Sequential runner for remaining Exam 1 matched baselines + full 5-arm held-out evaluation."""
import os
import sys
import subprocess

PY = r'C:\Users\nikox\Documents\Vesuvius\.venv\Scripts\python.exe'
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, 'train_exam1.py')
EVAL = os.path.join(HERE, 'eval_exam1.py')
CK = os.path.join(HERE, 'ckpt')

for arm in ('collar_bce', 'naive_bce', 'kiryo_nnpu'):
    out_p = os.path.join(CK, f'{arm}_half_centerline_s3000.pth')
    if os.path.exists(out_p):
        print(f'[SKIP] Already exists: {out_p}', flush=True)
        continue
    print(f'\n================ RUNNING {arm} ================', flush=True)
    subprocess.run([PY, '-u', TRAIN, '--arm', arm, '--scribble', 'half_centerline', '--steps', '3000', '--lr', '3e-4'], check=True)

print('\n================ EVALUATING ALL 5 ARMS ================', flush=True)
ckpts = [
    f'oracle_s3k={os.path.join(CK, "oracle_fullmask_half_centerline_s3000.pth")}',
    f'naive_bce_s3k={os.path.join(CK, "naive_bce_half_centerline_s3000.pth")}',
    f'collar_bce_s3k={os.path.join(CK, "collar_bce_half_centerline_s3000.pth")}',
    f'kiryo_nnpu_s3k={os.path.join(CK, "kiryo_nnpu_half_centerline_s3000.pth")}',
    f'scribble_pu_s3k={os.path.join(CK, "scribble_pu_batch_half_centerline_s3000.pth")}',
]
subprocess.run([PY, '-u', EVAL, '--out_prefix', 'exam1_5arm', '--ckpts'] + ckpts, check=True)
