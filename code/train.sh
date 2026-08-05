#!/usr/bin/env bash
# Train RDBM on the radiography (radio) dataset.
# Run from RDBM_radio/code/
# Optimizers: optim=muon (default) or optim=adam
set -euo pipefail

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# Optional: set CUDA_VISIBLE_DEVICES to restrict GPUs (e.g. CUDA_VISIBLE_DEVICES=0,1).
# If unset, all devices visible to the process are used.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

expid="${expid:-radio_2}"
# radiography/ sits next to RDBM_radio/ under $HOME
dataset_json="${dataset_json:-../../radiography/fix_test_realistic_v2/dataset.json}"
optim="${optim:-muon}"
use_wandb="${use_wandb:-1}"
wandb_project="${wandb_project:-radio}"
attn_heads="${attn_heads:-4}"
attn_dim_head="${attn_dim_head:-16}"
amp="${amp:-1}"
mixed_precision="${mixed_precision:-bf16}"
compile="${compile:-1}"
save_and_sample_every="${save_and_sample_every:-1000}"
resume="${resume:-auto}"

# Auto-detect process count from visible CUDA devices unless nproc is set.
if [[ -z "${nproc:-}" ]]; then
  nproc="$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)"
fi

# Defaults depend on optimizer; override via env if desired
if [[ "${optim}" == "adam" ]]; then
  lr="${lr:-8e-5}"
else
  lr="${lr:-1e-3}"
fi
wd="${wd:-0.1}"

echo "expid=${expid}"
echo "dataset_json=${dataset_json}"
echo "nproc=${nproc} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
# Cosine LR is the default for muon, with 10% linear warmup then cosine decay.
# Override with lr_scheduler=none|cosine|... and/or warmup_ratio=0.05 etc.
lr_scheduler="${lr_scheduler:-}"
warmup_ratio="${warmup_ratio:-}"

echo "optim=${optim} lr=${lr} wd=${wd}"
echo "use_wandb=${use_wandb} wandb_project=${wandb_project}"
echo "lr_scheduler=${lr_scheduler:-cosine(default for muon)} warmup_ratio=${warmup_ratio:-0.1(default for muon)}"
echo "attn_heads=${attn_heads} attn_dim_head=${attn_dim_head}"
echo "amp=${amp} mixed_precision=${mixed_precision} compile=${compile}"
echo "save_and_sample_every=${save_and_sample_every} resume=${resume}"

torchrun --standalone --nnodes 1 --nproc_per_node "${nproc}" train.py \
  --exp_id="${expid}" \
  --dataset_json="${dataset_json}" \
  --channels 1 \
  --attn_heads "${attn_heads}" \
  --attn_dim_head "${attn_dim_head}" \
  --crop_size 256 \
  --patch_size 256 \
  --optim "${optim}" \
  --lr "${lr}" \
  --wd "${wd}" \
  ${lr_scheduler:+--lr_scheduler "${lr_scheduler}"} \
  ${warmup_ratio:+--warmup_ratio "${warmup_ratio}"} \
  --train_batch_size 16 \
  --train_num_steps 10001 \
  --save_and_sample_every "${save_and_sample_every}" \
  --amp "${amp}" \
  --mixed_precision "${mixed_precision}" \
  --compile "${compile}" \
  --resume "${resume}" \
  --results_folder ./save_folder \
  --use_wandb "${use_wandb}" \
  --wandb_project "${wandb_project}"
