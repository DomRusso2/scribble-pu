# PREREG: Paul's checks on ScribblePU (Discord 2026-11-23)

Paul (TEAM): (1) "is this still with the very large window of reader-v3? if so, try an identical protocol with a smaller
window, as a check against hallucination / unevidenced character completion"; (2) "scribbles also become ambiguous with
a larger window unless some scribbles are near the edge of a stroke, not always the center".
All previous ScribblePU tests used 384 px tiles. Written BEFORE any T1 / T2 / T3 result exists. One seed per arm.
Bars are judged as written; any change after freezing is an AMENDMENT with its time and reason, never a silent edit.

## Arms (team stock koine_machines trainer, villa merge-ink-pipelines 47a9c37 + villa_full.patch; Check K protocol)
- T1-K1-128: Check K1 (BrierNNPULoss, fixed half of the 3-px spine, alpha 14.64, no blank anchors) with ONLY the window
  changed: patch_size [17,128,128] (Reader v2's window, under one letter). Config t1_k1_128.json.
- T1-K2-128: Check K2 (default LabelSmoothedDCAndBCE ls .5), same labels, same 128 px window. Config t1_k2_128.json.
- T2-K1-EDGE: Check K1 at 384 px with the SAME kept 64-px blocks, but in each kept block a fixed coin flip picks the
  centre spine OR an edge line (3-px inner rim of the stroke); alpha 18.12 by Check K's rule. Config t2_k1_edge.json.
- References (already measured, results_k.json): K1-384 (0009B GATE 0.7998, 0814 0.8861, p99 207) and K2-384
  (0.6098 / 0.6006 / 69).
Reads + scores: Check K's score_k.py path unchanged (Exam 1 scorer: b9 = 0009B 125754, s814 = 0814, FA80, p99), each
model read at its own window.

## T1 bars (Paul point 1, from-scratch side)
- PASS: T1-K1-128 beats T1-K2-128 by >= +0.05 GATE on BOTH 0009B (b9) and 0814 (s814), AND T1-K1-128 0009B p99 >= 150
  (not collapsed). STRONG: >= +0.10 on both.
- FAIL: T1-K1-128 <= T1-K2-128 + 0.02 on either, or T1-K1-128 p99 < 120.
- Report-only: T1-K1-128 as a fraction of K1-384; FA80; M1 / M4.

## T2 bars (Paul point 2)
- PASS: T2-K1-EDGE GATE >= K1-384 - 0.03 on BOTH 0009B and 0814, AND p99 >= 150.
- Report-only: M1 (letter separation), M4 (fine strokes), FA80 vs K1-384; whether edge lines help or hurt.

## T3 bars (Paul point 1, retuned-reader side): cut-gap test, rebuild of CUT-2 (02 #160)
- Volumes: cutgap.py build (done 20:40, 58 cuts on 0841 w00 / ag1 / ag2 + 0009B 125754, g = 46 / 69 / 93 / 116 px,
  blank paste from the same sheet; SHAM blank-on-blank; INK paste from labelled ink), plan.json.
- Models: Reader V3 (release/reader-v3/checkpoints/reader-v3.pth), Reader-PU bhand, Reader-PU deconly-soup50 (the HF
  files), Reader v2 (data/v2/train/ckpt_040000.pth) as reference; report-only: K1-384 (koine_smoke/k1/ckpt_003000.pth)
  and T1-K1-128.
- Metric (CUT-2 rule): fill = (cut_c - bg) / (orig_c - bg) at the cut centre, valid only where the stroke was read
  (orig_c - bg >= 15); OPEN iff fill <= 0.30.
- VALIDITY per model (else that model is INVALID, not FAIL): median sham fill <= 0.15 AND median ink fill >= 0.50 AND
  n_valid >= 25.
- PASS: for BOTH Reader-PU checkpoints, pooled OPEN rate >= V3 pooled OPEN rate - 0.10 (ScribblePU retuning adds no
  extra gap closing beyond its base). STRONG: Reader-PU OPEN rate >= V3's.
- FAIL: either Reader-PU OPEN rate < V3's - 0.10.
- Report-only: per-width OPEN rates, median fill, Reader v2 (known from #160 to keep more gaps open than V3), K1-384 vs
  T1-K1-128.

## Run order (one GPU job at a time; unattended chain run_paul.sh; RUN_STATE.txt logs each step)
T1-K1-128 train -> T1-K2-128 train -> T2-K1-EDGE train -> score T1 + T2 -> T3 reads -> T3 score -> RESULTS.
A 20-step dry run of t1_k1_128 ran before freezing (plumbing only, no evaluation). A K2-only smoke read on one cut
region may run before the chain to test the read code (K2 is not a T3 arm).

## What we will say
PASS / FAIL exactly as above, with the numbers, one seed, script-drawn lines. A FAIL is reported to Paul as a FAIL.

FROZEN 2026-11-23 20:40:34
