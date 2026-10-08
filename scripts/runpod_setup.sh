#!/usr/bin/env bash
# One-time setup of a rented GPU box (RunPod PyTorch template, volume mounted at /workspace).
# Run as root on the pod:  bash runpod_setup.sh
# Needs the Kaggle token in /workspace/.kaggle/access_token (scp it there first, never commit it).
# Pulls the training inputs once onto the volume, then scripts/runpod_train.sh runs the recipe.
set -euo pipefail
W=/workspace
mkdir -p $W/input $W/runs $W/src
export KAGGLE_CONFIG_DIR=$W/.kaggle
# the template's system Python is PEP 668 managed and already has torch, so install into it
pip install -q --break-system-packages kaggle numpy
if [ ! -f $W/input/prep/tokens_mz.npy ]; then
  kaggle kernels output amhashiferaw/casmi26-prep -p $W/input/prep
fi
if [ ! -f $W/input/train_fp/train_fp.npz ]; then
  kaggle datasets download amhashiferaw/casmi26-train-fp -p $W/input/train_fp --unzip
fi
if [ ! -f $W/input/coconut/coco_fp.npy ]; then
  kaggle datasets download prvsiyan/coconut-casmi26-candidates -p $W/input/coconut --unzip
fi
du -sh $W/input/*
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "setup done"
