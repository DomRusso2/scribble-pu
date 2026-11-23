"""Judge PREREG_PAUL_CHECKS.md (frozen 2026-11-23 20:40:34, sha e1143b54d92e) exactly as written.
Reads results_paul.json (T1 / T2), cutgap/results_cutgap.json (T3) and ../koine_smoke/results_k.json (K references).
Writes RESULTS_PAUL.md + PAUL_VERDICTS.json (not RESULTS_PAUL.json: Windows names ignore case and that would overwrite results_paul.json). Missing inputs are reported as MISSING, never guessed."""
import json
import os

P = r'C:\Users\nikox\Documents\Vesuvius\data\scribblepu\paul_checks'
K = r'C:\Users\nikox\Documents\Vesuvius\data\scribblepu\koine_smoke'


def load(p):
    return json.load(open(p)) if os.path.exists(p) else {}


tk = load(os.path.join(P, 'results_paul.json'))
cg = load(os.path.join(P, 'cutgap', 'results_cutgap.json'))
rk = load(os.path.join(K, 'results_k.json'))
k1r, k2r = rk.get('k1', {}), rk.get('k2', {})
L, J = [], {}


def f(x, n=4):
    return 'n/a' if x is None else (f'{x:.{n}f}' if isinstance(x, float) else str(x))


L.append('# RESULTS: Paul checks (PREREG_PAUL_CHECKS.md, frozen 2026-11-23 20:40:34)\n')
L.append(f"References (Check K, 384 px): K1 0009B {f(k1r.get('b9_GATE'))} / 0814 {f(k1r.get('s814_GATE'))} / p99 "
         f"{f(k1r.get('b9_p99'), 0)}; K2 {f(k2r.get('b9_GATE'))} / {f(k2r.get('s814_GATE'))} / {f(k2r.get('b9_p99'), 0)}\n")

# T1
a, b = tk.get('t1k1_128'), tk.get('t1k2_128')
if a and b:
    d9, d8 = a['b9_GATE'] - b['b9_GATE'], a['s814_GATE'] - b['s814_GATE']
    if d9 >= 0.10 and d8 >= 0.10 and a['b9_p99'] >= 150:
        v = 'PASS (STRONG)'
    elif d9 >= 0.05 and d8 >= 0.05 and a['b9_p99'] >= 150:
        v = 'PASS'
    elif d9 <= 0.02 or d8 <= 0.02 or a['b9_p99'] < 120:
        v = 'FAIL'
    else:
        v = 'INCONCLUSIVE (between the bars)'
    J['T1'] = dict(verdict=v, k1_128=a, k2_128=b, d9=d9, d8=d8)
    L.append(f"## T1 (128 px window): {v}\n- ScribblePU 128: 0009B {f(a['b9_GATE'])}, 0814 {f(a['s814_GATE'])}, p99 "
             f"{f(a['b9_p99'], 0)}, FA80 {f(a.get('FA80'))}, M1 {f(a.get('b9_M1'))}\n- Default 128: 0009B {f(b['b9_GATE'])}, "
             f"0814 {f(b['s814_GATE'])}, p99 {f(b['b9_p99'], 0)}\n- Gap: +{f(d9)} / +{f(d8)} (bars: PASS >= +0.05 both, "
             f"STRONG >= +0.10)\n- ScribblePU 128 vs 384: {f(a['b9_GATE'] / k1r['b9_GATE'] if k1r else None, 3)} / "
             f"{f(a['s814_GATE'] / k1r['s814_GATE'] if k1r else None, 3)} of the 384 px GATE (report-only)\n")
else:
    L.append('## T1: MISSING (t1k1_128 / t1k2_128 not scored)\n')

# T2
e = tk.get('t2k1_edge')
if e and k1r:
    ok = e['b9_GATE'] >= k1r['b9_GATE'] - 0.03 and e['s814_GATE'] >= k1r['s814_GATE'] - 0.03 and e['b9_p99'] >= 150
    v = 'PASS' if ok else 'FAIL'
    J['T2'] = dict(verdict=v, edge=e)
    L.append(f"## T2 (centre + edge lines, 384 px): {v}\n- Edge mix: 0009B {f(e['b9_GATE'])}, 0814 {f(e['s814_GATE'])}, "
             f"p99 {f(e['b9_p99'], 0)} (bar: >= K1 - 0.03 = {f(k1r['b9_GATE'] - 0.03)} / {f(k1r['s814_GATE'] - 0.03)}, "
             f"p99 >= 150)\n- Report-only vs centre-only K1: M1 {f(e.get('b9_M1'))} vs {f(k1r.get('b9_M1'))}, M4 "
             f"{f(e.get('b9_M4'))} vs {f(k1r.get('b9_M4'))}, FA80 {f(e.get('FA80'))} vs {f(k1r.get('FA80'))}\n")
else:
    L.append('## T2: MISSING\n')

# T3


def valid(r):
    return r and r.get('n_valid', 0) >= 25 and r.get('median_sham_fill', 9) <= 0.15 and r.get('median_ink_fill', -9) >= 0.50


v3 = cg.get('v3')
L.append('## T3 (cut-gap test, letter-size gaps 0.4-1.0 mm)\n')
for tag in ('v3', 'pu_bhand', 'pu_deconly', 'v2', 'k1_384', 't1k1_128'):
    r = cg.get(tag)
    if not r:
        L.append(f'- {tag}: MISSING\n')
        continue
    L.append(f"- {tag}: OPEN {f(r.get('open_rate'), 2)} (n valid {r.get('n_valid')}/{r.get('n_cuts')}), median fill "
             f"{f(r.get('median_fill'), 2)}, sham {f(r.get('median_sham_fill'), 2)}, ink {f(r.get('median_ink_fill'), 2)}, "
             f"{'VALID' if valid(r) else 'INVALID'}; by width " +
             str({g: d['open_rate'] if d['open_rate'] is None else round(d['open_rate'], 2) for g, d in r.get('by_width', {}).items()}) + '\n')
pus = [cg.get('pu_bhand'), cg.get('pu_deconly')]
if v3 and all(pus):
    if not valid(v3) or not all(valid(p) for p in pus):
        v = 'INVALID (a validity control failed; see rows)'
    elif all(p['open_rate'] >= v3['open_rate'] for p in pus):
        v = 'PASS (STRONG)'
    elif all(p['open_rate'] >= v3['open_rate'] - 0.10 for p in pus):
        v = 'PASS'
    else:
        v = 'FAIL'
    J['T3'] = dict(verdict=v)
    L.append(f'\n**T3 verdict: {v}** (bar: both Reader-PU OPEN >= V3 OPEN - 0.10; STRONG >= V3)\n')
else:
    L.append('\n**T3 verdict: MISSING**\n')

J['cutgap'] = cg
J['k'] = tk
open(os.path.join(P, 'RESULTS_PAUL.md'), 'w', encoding='utf-8').write('\n'.join(L))
json.dump(J, open(os.path.join(P, 'PAUL_VERDICTS.json'), 'w'), indent=1)
print('\n'.join(L))
