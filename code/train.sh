#!/usr/bin/env bash
# Train RDBM on radiography data. Run from RDBM_radio/code/.
set -euo pipefail

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES

expid="${expid:-radio_4}"
dataset_json="${dataset_json:-../data-generation/fix_test_realistic_v2/dataset.json}"
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
bit_depth="${bit_depth:-auto}"
data_range="${data_range:-}"

if [[ -z "${nproc:-}" ]]; then
  nproc="$(python -c 'import torch; print(max(torch.cuda.device_count(), 1))')"
fi

if [[ "${optim}" == "adam" ]]; then
  lr="${lr:-8e-5}"
else
  lr="${lr:-1e-3}"
fi
wd="${wd:-0.1}"
lr_scheduler="${lr_scheduler:-}"
warmup_ratio="${warmup_ratio:-}"
decay_ratio="${decay_ratio:-}"
adam_beta2="${adam_beta2:-0.95}"

echo "expid=${expid}"
echo "dataset_json=${dataset_json}"
echo "nproc=${nproc} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
echo "optim=${optim} lr=${lr} wd=${wd} adam_beta2=${adam_beta2}"
echo "use_wandb=${use_wandb} wandb_project=${wandb_project}"
echo "lr_scheduler=${lr_scheduler:-warmup_stable_decay} warmup_ratio=${warmup_ratio:-0.1} decay_ratio=${decay_ratio:-0.5}"
echo "attn_heads=${attn_heads} attn_dim_head=${attn_dim_head}"
echo "amp=${amp} mixed_precision=${mixed_precision} compile=${compile}"
echo "save_and_sample_every=${save_and_sample_every} resume=${resume}"
echo "bit_depth=${bit_depth} data_range=${data_range:-auto}"

torchrun --standalone --nnodes 1 --nproc_per_node "${nproc}" train.py \
  --exp_id="${expid}" \
  --dataset_json="${dataset_json}" \
  --channels 1 \
  --bit_depth "${bit_depth}" \
  ${data_range:+--data_range "${data_range}"} \
  --attn_heads "${attn_heads}" \
  --attn_dim_head "${attn_dim_head}" \
  --crop_size 256 \
  --patch_size 256 \
  --optim "${optim}" \
  --lr "${lr}" \
  --wd "${wd}" \
  --adam_beta2 "${adam_beta2}" \
  ${lr_scheduler:+--lr_scheduler "${lr_scheduler}"} \
  ${warmup_ratio:+--warmup_ratio "${warmup_ratio}"} \
  ${decay_ratio:+--decay_ratio "${decay_ratio}"} \
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
