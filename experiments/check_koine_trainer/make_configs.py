"""Check K configs (PREREG_TWO_CHECKS.md, frozen 2026-11-23 14:13:17): k1.json, k2.json, k_dry.json.

From data/readerv4/retrain/1bn_smoke.json, changing ONLY what the prereg lists:
checkpoint null, adamw lr 1e-3, batch 8, quotas 0139N 2 / 1667H 3 / Paris4H 3, the 3 hand datasets only
(segments_path -> the fixed-half label copies), num_iterations 3000, seed 42, a new patch cache, our out_dir.
The 'r1a' block belongs to our own retrain scripts and is dropped (koine never reads it).
K1 loss = BrierNNPULoss (alpha from label_stats.json), K2 loss = the file's default (BCE ls .5 + Dice).
"""
import copy
import json
import os

R = r'C:\Users\nikox\Documents\Vesuvius'
K = os.path.join(R, 'data', 'scribblepu', 'koine_smoke')
src = json.load(open(os.path.join(R, 'data', 'readerv4', 'retrain', '1bn_smoke.json')))
stats = json.load(open(os.path.join(K, 'label_stats.json')))
alpha = round(float(stats['alpha_quota_weighted']), 2)
QUOTAS = {'0139N': 2, '1667H': 3, 'Paris4H': 3}

base = copy.deepcopy(src)
base.pop('r1a', None)
base['checkpoint'] = None
base['weights_only'] = False
base['optimizer'] = 'adamw'
base['learning_rate'] = 1e-3
base['batch_size'] = 8
base['num_iterations'] = 3000
base['seed'] = 42
base['fixed_scroll_prior'] = {'seed': 42, 'target_batch_counts': dict(QUOTAS)}
base['patch_cache_filename'] = os.path.join(K, 'patches_k_384.json').replace('\\', '/')
base['save_every'] = 1000
ds = []
for d in src['datasets']:
    if d.get('sampling_scroll') in QUOTAS and d.get('detail_kind') == 'hand':
        d = copy.deepcopy(d)
        d['segments_path'] = os.path.join(K, 'labels').replace('\\', '/')
        ds.append(d)
assert sorted(x['sampling_scroll'] for x in ds) == sorted(QUOTAS), [x['sampling_scroll'] for x in ds]
base['datasets'] = ds

k1 = copy.deepcopy(base)
k1['description'] = 'Check K1: BrierNNPULoss on a FIXED half of the 3-px spine, stock 2-channel targets, from scratch'
k1['loss'] = {'terms': [{
    'name': 'BrierNNPULoss', 'metric_name': 'brier_nnpu', 'weight': 1.0, 'alpha': alpha, 'rho': 0.96,
    'y_lo': 0.25, 'y_hi': 0.82, 'w_blank': 0.1667, 'batch_neg_risk': True}]}
k1['out_dir'] = os.path.join(K, 'k1').replace('\\', '/')

k2 = copy.deepcopy(base)
k2['description'] = 'Check K2: default LabelSmoothedDCAndBCE (ls .5) on the SAME fixed-half labels, from scratch'
k2['loss'] = copy.deepcopy(src['loss'])
k2['out_dir'] = os.path.join(K, 'k2').replace('\\', '/')

dry = copy.deepcopy(k1)
dry['num_iterations'] = 20
dry['log_every'] = 1
dry['save_every'] = 100000
dry['out_dir'] = os.path.join(K, 'k_dry').replace('\\', '/')

for name, c in (('k1', k1), ('k2', k2), ('k_dry', dry)):
    json.dump(c, open(os.path.join(K, f'{name}.json'), 'w'), indent=1)
    print(name, c['loss'], c['optimizer'], c['learning_rate'], c['checkpoint'], c['batch_size'], c['num_iterations'],
          [d['sampling_scroll'] for d in c['datasets']], c['out_dir'])
