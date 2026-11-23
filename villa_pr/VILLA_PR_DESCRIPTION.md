<!--
NOTES FOR THE AUTHOR (delete this block before pasting)
- Base branch: merge-ink-pipelines (koine_machines is not on villa main; main moved the ink code to vesuvius.ink_detection in #1456).
- Apply with: git apply villa_full.patch (adds brier_nnpu.py, the 17-line registration in losses.py, tests/test_brier_nnpu.py).
  Checked 2026-11-23 on a fresh clone of the live branch head 47a9c37: applies cleanly, all loss tests pass (5 passed),
  and a full 3,000-step koine_machines training run with it works (the example below).
- PR title: Add BrierNNPULoss: train ink models from 1-pixel lines on the letters you are sure of (no full or background masks)
- villa's CONTRIBUTING asks for human-written commentary on LLM-assisted PRs: rewrite "Why / where this is useful" in your own words.
- Tick the checkbox only after you have looked at the proof yourself.
- Attach under Proof: figures/fig6_koine_trainer_check.png (first), figures/fig5_adaptation_0009b.png, figures/fig0_scribble_workflow.png.
-->

**In one sentence:** With this loss you can train or fine-tune a `koine_machines` ink model from quick 1-pixel lines on the letters you are sure of, with no full letter masks and no background masks; the letters you skip are left unlabelled instead of being learned as background.

**One real example:** Starting from random weights in this branch's stock `koine_machines` trainer and dataset, I trained 3,000 steps on the hand-labelled `PHerc. 0139`, `1667` and `Paris 4` segments with their ink labels cut down (by script) to thin lines through a fixed half of the letters, and `BrierNNPULoss` produced clean ink maps on held-out `PHerc. 0009B` and `PHerc. 0814` (GATE `0.7998` and `0.8861`, peak brightness `p99 = 207`).

**Before:** On exactly the same labels, the default `LabelSmoothedDCAndBCELoss` treats every unlabelled stroke as background and collapses to flat grey: GATE `0.6098` and `0.6006`, `p99 = 69`. So training or adapting a model meant painting every letter in full first.

**After this PR:** Add one `BrierNNPULoss` entry to `loss.terms`, draw lines on the strokes you are sure of into the ink label image, and leave the rest unlabelled. Nothing else in the training loop, the dataset or the inference CLI changes.

**Proof:** (attach the figures) Fig. 1: held-out reads from the two models above, same input, same 0-255 scale: default training is flat grey, `BrierNNPULoss` shows the letters. Fig. 2: adapting a 3-scroll model to `PHerc. 0009B` (a 116 keV / 8.64 µm scan it never saw) from 1-pixel lines on one segment, read on two held-out segments (`125754` and `140325`). Fig. 3: what the labels look like: full ink and background masks vs one thin line through the letters you are sure of.

| Stock `koine_machines`, 3,000 steps, fixed half of the letters as lines | `BrierNNPULoss` | Default loss, same labels |
|---|---:|---:|
| Held-out `0009B 125754` GATE / `p99` | **0.7998 / 207** | 0.6098 / 69 |
| Held-out `PHerc. 0814` GATE | **0.8861** | 0.6006 |
| False ink on blank papyrus (FA80, lower is better) | **0.6248** | 0.8554 |

**Why / where this is useful:** (rewrite in your own words) It makes step 4 of the pseudo-labelling loop on the [Open Problems page](https://scrollprize.org/2026_open_problems#pseudo-labeling-bootstrapping-weak-signals), "Create conservative pseudo-labels", trainable as it is: label only the ink you are sure of, and the rest is not taught as background. Anyone fitting a reader to a new scroll, a new scan protocol or a First Letters region can label only the strokes they are sure of, in minutes, instead of painting full masks.

- [ ] I personally verified that the example and proof above were produced by this PR on the stated data.

## Details

**Files:** `ink-detection/koine_machines/training/loss/brier_nnpu.py` (new, `VillaBrierNNPULoss`), `losses.py` (+17 lines: builder registered as `LOSS_TERM_BUILDERS["BrierNNPULoss"]`), `tests/test_brier_nnpu.py` (new, 4 CPU tests: builds from config; finite loss and gradients inside `CompositeLoss` with aux metrics; the clamp never pushes labelled pixels toward background; the optional certain-blank channel pulls toward `y_lo`). Tested on `merge-ink-pipelines` @ `47a9c37`:

```bash
pytest ink-detection/koine_machines/training/loss/tests -q   # 5 passed
```

**Method:** Brier-score positive-unlabelled loss (targets `y_hi = 0.82` for labelled ink and `y_lo = 0.25` for background, close to the effective targets of the team's 9 µm recipe: BCE with 0.5 label smoothing plus Dice). Kiryo et al. (2017) nnPU takes a gradient-ascent step whenever the estimated negative risk `R_neg = R_u_neg - pi_u * R_p_neg` drops below zero. With centre lines this happens on 16-60% of tiles (centre lines sit on the brightest ink cores, and labelled ink is clustered), and the ascent step pushes confirmed ink toward background. This term clamps `R_neg` at zero instead, computed over the microbatch (`batch_neg_risk=True`), so the positive term always stays active; when `R_neg >= 0` it is identical to unbiased Brier-nnPU. Targets use the existing two-channel layout (lines, ignore); an optional middle channel can mark verified blank papyrus, and the example above runs without it.

**Config:**

```json
"loss": {
  "terms": [
    {
      "name": "BrierNNPULoss",
      "metric_name": "brier_nnpu",
      "weight": 1.0,
      "alpha": 14.5,
      "rho": 0.96,
      "y_lo": 0.25,
      "y_hi": 0.82,
      "w_blank": 0.1667,
      "batch_neg_risk": true
    }
  ]
}
```

`alpha` = rough true-ink pixels per labelled pixel: `stroke_width_px / (line_width_px x fraction_of_strokes_labelled)` (`14.5` here is for 3-pixel lines on half the strokes; the example run used `14.64`). Guessing a little high is safe: at 1.25x and 1.5x the adaptation gain matches or beats full masks; 0.75x keeps 75% of it.

**More results** (public scrolls; code, data and every number at https://github.com/DomRusso2/scribble-pu):
- **Adaptation to a new scan protocol** (`PHerc. 0009B`, 600 steps, about 9 minutes): 1-pixel lines covering under 1% of one segment's ink pixels give `+0.0793` GATE on held-out segment `140325`, 97.1% of the `+0.0817` from full masks; default BCE gives `-0.0693` and class-balanced BCE `+0.0482`. With a fixed half of the lines on one 2048 x 2048 area, the gain is 107% (`140325`) and 100% (`125754`) of what full masks on that area give.
- **From scratch** on `0139`, `1667` and `Paris 4` with thin centre lines on 6.8% of the ink (no stroke borders, no background masks): `0009B` GATE `0.8701` vs `0.8727` for full masks (99.7%; 92.5% in a shorter 3k-step run), with higher fine-stroke recall (M4 `0.8375` vs `0.7933`) and letter separation (M1 `0.7945` vs `0.7015`).
- **Retuning Reader V3** with this loss on thin centre lines lifts letter separation (M1) on held-out `0009B` from `0.8371` to `0.8829` (+0.0226 beyond V3's best morphological erosion) and on `PHerc. 1667 w029` from `0.8221` to `0.9145` (Reader-PU, a sharper second view next to V3 for crowded rows).

**Notes:** one training run per arm; the lines in these tests were drawn by a script along existing ink labels, so the comparisons with full masks are exact.
