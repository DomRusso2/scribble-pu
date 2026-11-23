"""Phase 3 Runner: Converged Exam 2A (0009B 133208 -> 125754 + 140325, 7 arms) + 5-Point Alpha Sweep."""
import os
import json
import subprocess

PY = r'C:\Users\nikox\Documents\Vesuvius\.venv\Scripts\python.exe'
HERE = os.path.dirname(os.path.abspath(__file__))
ADAPT = os.path.join(HERE, 'adapt_0009b.py')
BASE_CK = os.path.abspath(os.path.join(HERE, '..', 'exam1_from_scratch', 'ckpt', 'oracle_fullmask_half_centerline_s5000_lr1e3.pth'))

# 1. Main 7-arm unseen-protocol adaptation on converged 5k-step 3-scroll base
arms = ('base_eval', 'oracle_fullmask', 'naive_bce', 'collar_bce', 'collar_bce_bal', 'kiryo_nnpu', 'scribble_pu')
summary = {}
for arm in arms:
    if arm == 'base_eval':
        jpath = os.path.join(HERE, 'adapt0009b_v2_base_eval.json')
        if not os.path.exists(jpath):
            print(f'\n================ EXAM 2A (v2): {arm} ================', flush=True)
            subprocess.run([PY, '-u', ADAPT, '--base_ckpt', BASE_CK, '--arm', 'base_eval', '--tag_prefix', 'adapt0009b_v2'], check=True)
    else:
        jpath = os.path.join(HERE, f'adapt0009b_v2_{arm}_s600.json')
        if not os.path.exists(jpath):
            print(f'\n================ EXAM 2A (v2): {arm} ================', flush=True)
            subprocess.run([
                PY, '-u', ADAPT, '--base_ckpt', BASE_CK, '--arm', arm,
                '--steps', '600', '--lr', '1e-4', '--tag_prefix', 'adapt0009b_v2',
            ], check=True)
    summary[arm] = json.load(open(jpath))

with open(os.path.join(HERE, 'exam2a_v2_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

# 2. 5-point Alpha Robustness Sweep (0.50x, 0.75x, 1.00x, 1.25x, 1.50x) on ScribblePU!
sweep = {}
for mult in (0.50, 0.75, 1.00, 1.25, 1.50):
    if abs(mult - 1.0) < 1e-6:
        sweep['1.00x'] = summary['scribble_pu']
        continue
    ms = int(round(mult * 100))
    jpath = os.path.join(HERE, f'adapt0009b_v2_scribble_pu_am{ms}_s600.json')
    if not os.path.exists(jpath):
        print(f'\n================ ALPHA SWEEP: {mult:.2f}x ================', flush=True)
        subprocess.run([
            PY, '-u', ADAPT, '--base_ckpt', BASE_CK, '--arm', 'scribble_pu',
            '--steps', '600', '--lr', '1e-4', '--alpha_mult', f'{mult:.2f}', '--tag_prefix', 'adapt0009b_v2',
        ], check=True)
    sweep[f'{mult:.2f}x'] = json.load(open(jpath))

with open(os.path.join(HERE, 'exam2a_alpha_sweep.json'), 'w', encoding='utf-8') as f:
    json.dump(sweep, f, indent=2)

print('\n=== EXAM 2A (v2) COMPLETE ===\n' + json.dumps(summary, indent=2), flush=True)
print('\n=== ALPHA ROBUSTNESS SWEEP ===\n' + json.dumps(sweep, indent=2), flush=True)
