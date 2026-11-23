"""Sequential runner for Converged 7-Arm Exam 1 (5,000 steps, lr=1e-3, W_BLK=2/12) + Full Held-Out Eval."""
import os
import subprocess

PY = r'C:\Users\nikox\Documents\Vesuvius\.venv\Scripts\python.exe'
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, 'train_exam1.py')
EVAL = os.path.join(HERE, 'eval_exam1.py')
CK = os.path.join(HERE, 'ckpt')

runs = [
    ('oracle_fullmask',   'half_centerline',  'oracle_s5k'),
    ('scribble_pu_batch', 'half_centerline',  'scribble_pu_rdg_s5k'),
    ('collar_bce_bal',    'half_centerline',  'collar_bce_bal_s5k'),
    ('scribble_pu_batch', 'partial_letter20', 'scribble_pu_p20_s5k'),
    ('kiryo_nnpu',        'half_centerline',  'kiryo_nnpu_s5k'),
    ('collar_bce',        'half_centerline',  'collar_bce_s5k'),
    ('naive_bce',         'half_centerline',  'naive_bce_s5k'),
]

ckpts = []
for arm, scrib, label in runs:
    out_p = os.path.join(CK, f'{arm}_{scrib}_s5000_lr1e3.pth')
    ckpts.append(f'{label}={out_p}')
    if os.path.exists(out_p):
        print(f'[SKIP] Already trained: {out_p}', flush=True)
        continue
    print(f'\n================ EXAM 1 CONVERGED: {label} ({arm} / {scrib}) ================', flush=True)
    subprocess.run(
        [PY, '-u', TRAIN, '--arm', arm, '--scribble', scrib, '--steps', '5000', '--lr', '1e-3', '--tag_suffix', '_lr1e3'],
        check=True,
    )

print('\n================ EVALUATING ALL 7 CONVERGED EXAM 1 ARMS ================', flush=True)
subprocess.run([PY, '-u', EVAL, '--out_prefix', 'exam1_7arm_s5k', '--ckpts'] + ckpts, check=True)
