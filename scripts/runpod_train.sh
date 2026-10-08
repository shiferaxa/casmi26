#!/usr/bin/env bash
# Run one training job of the v2 recipe on the rented box and leave its checkpoints in /workspace/runs/<tag>.
#   bash runpod_train.sh <tag> [CASMI_CFG_<key>=<value> ...]
# Example, three seeds queued in one nohup:
#   nohup bash -c 'for s in 11 12 13; do bash runpod_train.sh seed$s CASMI_CFG_SEED=$s CASMI_CFG_STEPS=80000 CASMI_CFG_TIME_BUDGET_H=6; done' > /workspace/runs/queue.log 2>&1 &
# Any CFG key of kaggle/train_fp_v2/notebook.py can be overridden (seed, steps, batch, d, layers, lr, ...).
set -euo pipefail
TAG=$1; shift
W=/workspace
OUT=$W/runs/$TAG; mkdir -p $OUT
cp $W/src/notebook.py $OUT/notebook.py
env CASMI_INPUT_ROOT=$W/input CASMI_OUT_DIR=$OUT CASMI_TAG=$TAG "$@" python $OUT/notebook.py 2>&1 | tee $OUT/train.log
