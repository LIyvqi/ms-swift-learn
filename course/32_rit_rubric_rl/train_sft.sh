#!/usr/bin/env bash

# 先让 Base 模型学会显式审核格式，为 ORM 与 RiT 提供完全相同的起点。
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TRAIN_DATA="${RIT_DATA}/sft_train.jsonl"
VAL_DATA="${RIT_DATA}/sft_validation.jsonl"
OUTPUT="${SFT_OUTPUT:-${RIT_OUTPUT}/sft_format}"
EXTRA_ARGS=(
  --num_train_epochs "${SFT_EPOCHS:-1}"
  --eval_strategy epoch
  --save_strategy steps
  --save_steps "${SFT_SAVE_STEPS:-50}"
)

if [[ "${SMOKE:-0}" == "1" ]]; then
  TRAIN_DATA="${RIT_DATA}/sft_smoke.jsonl"
  VAL_DATA="${RIT_DATA}/sft_smoke.jsonl"
  OUTPUT="${SFT_OUTPUT:-${RIT_OUTPUT}/sft_smoke}"
  EXTRA_ARGS=(
    --max_steps "${SMOKE_STEPS:-3}"
    --eval_strategy steps
    --eval_steps "${SMOKE_STEPS:-3}"
    --save_strategy steps
    --save_steps "${SMOKE_STEPS:-3}"
  )
fi

rit_require_data

swift sft \
  --model "${RIT_MODEL}" \
  --dataset "${TRAIN_DATA}" \
  --val_dataset "${VAL_DATA}" \
  --tuner_type lora \
  --lora_rank "${LORA_RANK:-32}" \
  --lora_alpha "${LORA_ALPHA:-64}" \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --enable_thinking true \
  --max_length "${MAX_LENGTH:-1536}" \
  --per_device_train_batch_size "${SFT_BATCH:-8}" \
  --per_device_eval_batch_size "${SFT_EVAL_BATCH:-8}" \
  --gradient_accumulation_steps "${SFT_GRAD_ACC:-1}" \
  --learning_rate "${SFT_LEARNING_RATE:-5e-5}" \
  --warmup_ratio 0.05 \
  --logging_steps 5 \
  --save_total_limit "${SFT_SAVE_LIMIT:-2}" \
  --save_only_model true \
  --gradient_checkpointing false \
  --dataset_num_proc 2 \
  --dataloader_num_workers 2 \
  --report_to tensorboard \
  --output_dir "${OUTPUT}" \
  "${EXTRA_ARGS[@]}"
