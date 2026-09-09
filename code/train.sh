#!/usr/bin/env bash
# Train RDBM on radiography data. Run from RDBM_radio/code/.
set -euo pipefail

# auto-detect CUDA devices
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES





# --- edit these! (or override the same names as env vars) ---

# NDJSON of input/target pairs
dataset_json="${dataset_json:-../data-generation/fix_test_realistic_v2/dataset.json}"
# Experiment name; checkpoints go under save_folder/<expid>/
expid="${expid:-radio_4}"

# 1 = log to wandb, 0 = off
use_wandb="${use_wandb:-0}"
wandb_project="${wandb_project:-radio}"

# optimizer: muon (default) or adam
optim="${optim:-muon}"
# Learning rate: muon defaults to 1e-3, adam defaults to 8e-5.
lr="${lr:-}"
wd="${wd:-0.1}"
adam_beta2="${adam_beta2:-0.95}"
# Leave empty to use train.py defaults (warmup_stable_decay for muon)
lr_scheduler="${lr_scheduler:-}"
warmup_ratio="${warmup_ratio:-}"
decay_ratio="${decay_ratio:-}"

# UNet attention width
attn_heads="${attn_heads:-4}"
attn_dim_head="${attn_dim_head:-16}"

patch_size=256

train_batch_size=16
train_num_steps=10001
save_and_sample_every="${save_and_sample_every:-1000}"

amp="${amp:-1}"
mixed_precision="${mixed_precision:-bf16}"
compile="${compile:-1}"
resume="${resume:-auto}"
# auto | 8 | 16
bit_depth="${bit_depth:-auto}"
# Leave empty unless you need an explicit peak (e.g. 255 or 65535)
data_range="${data_range:-}"

# Leave empty to use all visible GPUs
nproc="${nproc:-}"







# --- automatic defaults (no need to edit below this point) ---

if [[ -z "${nproc}" ]]; then
  nproc="$(python -c 'import torch; print(max(torch.cuda.device_count(), 1))')"
fi

if [[ -z "${lr}" ]]; then
  if [[ "${optim}" == "adam" ]]; then
    lr=8e-5
  else
    lr=1e-3
  fi
fi

echo "expid=${expid}  dataset_json=${dataset_json}  nproc=${nproc}"

torchrun --standalone --nnodes 1 --nproc_per_node "${nproc}" train.py \
  --exp_id="${expid}" \
  --dataset_json="${dataset_json}" \
  --channels 1 \
  --bit_depth "${bit_depth}" \
  ${data_range:+--data_range "${data_range}"} \
  --attn_heads "${attn_heads}" \
  --attn_dim_head "${attn_dim_head}" \
  --patch_size "${patch_size}" \
  --optim "${optim}" \
  --lr "${lr}" \
  --wd "${wd}" \
  --adam_beta2 "${adam_beta2}" \
  ${lr_scheduler:+--lr_scheduler "${lr_scheduler}"} \
  ${warmup_ratio:+--warmup_ratio "${warmup_ratio}"} \
  ${decay_ratio:+--decay_ratio "${decay_ratio}"} \
  --train_batch_size "${train_batch_size}" \
  --train_num_steps "${train_num_steps}" \
  --save_and_sample_every "${save_and_sample_every}" \
  --amp "${amp}" \
  --mixed_precision "${mixed_precision}" \
  --compile "${compile}" \
  --resume "${resume}" \
  --results_folder ./save_folder \
  --use_wandb "${use_wandb}" \
  --wandb_project "${wandb_project}"
