#!/usr/bin/env bash
# Window / edge-scribble / cut-gap checks, unattended chain (PREREG_PAUL_CHECKS.md frozen 2026-11-23 20:40:34). One GPU job at a time.
# Each step logs start / exit to RUN_STATE.txt; a failed step is logged and the chain moves on.
# Results: RESULTS_PAUL.md (verdicts judged by eval_bars.py exactly as frozen). DONE marker at the end.
S=${VILLA_DIR:?set VILLA_DIR to a villa merge-ink-pipelines checkout with villa_full.patch applied}
V=C:/Users/nikox/Documents/Vesuvius
PY=$V/.venv/Scripts/python.exe
P=$V/data/scribblepu/paul_checks
HF=C:/Users/nikox/Documents/antigravity/scribble-pu_hf_card
ST=$P/RUN_STATE.txt
cd $V
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $ST; }
rm -f $P/DONE
log "CHAIN START (pid $$)"
for arm in t1_k1_128 t1_k2_128 t2_k1_edge; do
  log "$arm train start"
  PYTHONPATH="$S/ink-detection;$S/vesuvius/src" PYTHONUNBUFFERED=1 $PY -m koine_machines.training.train $P/$arm.json > $P/$arm.log 2>&1
  log "$arm train exit $?"
done
log "score T1+T2 start"
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 $PY $P/score_checks.py t1k1_128=$P/t1_k1_128/ckpt_003000.pth t1k2_128=$P/t1_k2_128/ckpt_003000.pth t2k1_edge=$P/t2_k1_edge/ckpt_003000.pth > $P/score_paul.log 2>&1
log "score T1+T2 exit $?"
for pair in v3=$V/release/reader-v3/checkpoints/reader-v3.pth pu_bhand=$HF/reader-pu-bhand.pth pu_deconly=$HF/reader-pu-deconly-soup50.pth v2=$V/data/v2/train/ckpt_040000.pth k1_384=$V/data/scribblepu/koine_smoke/k1/ckpt_003000.pth t1k1_128=$P/t1_k1_128/ckpt_003000.pth; do
  log "T3 read ${pair%%=*} start"
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 $PY $P/cutgap.py read $pair >> $P/cutgap_read.log 2>&1
  log "T3 read ${pair%%=*} exit $?"
done
log "T3 score start"
PYTHONIOENCODING=utf-8 $PY $P/cutgap.py score > $P/cutgap_score.log 2>&1
log "T3 score exit $?"
PYTHONIOENCODING=utf-8 $PY $P/eval_bars.py > $P/eval_bars.log 2>&1
log "eval_bars exit $? -> RESULTS_PAUL.md"
log "CHAIN DONE"
touch $P/DONE
