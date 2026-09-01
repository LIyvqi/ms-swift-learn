#!/usr/bin/env bash

# 关闭自由思维链，让模型只学习五个短结构化审核字段。
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TRAIN_DATA="${RIT_DATA}/structured_sft_train.jsonl"
VAL_DATA="${RIT_DATA}/structured_sft_validation.jsonl"
OUTPUT="${STRUCTURED_SFT_OUTPUT:-${RIT_OUTPUT}/structured_sft}"
EXTRA_ARGS=(
  --num_train_epochs "${SFT_EPOCHS:-1}"
  --eval_strategy epoch
  --save_strategy steps
  --save_steps "${SFT_SAVE_STEPS:-50}"
)

if [[ "${SMOKE:-0}" == "1" ]]; then
  TRAIN_DATA="${RIT_DATA}/structured_sft_smoke.jsonl"
  VAL_DATA="${RIT_DATA}/structured_sft_smoke.jsonl"
  OUTPUT="${STRUCTURED_SFT_OUTPUT:-${RIT_OUTPUT}/structured_sft_smoke}"
  EXTRA_ARGS=(
    --max_steps "${SMOKE_STEPS:-3}"
    --eval_strategy steps
    --eval_steps "${SMOKE_STEPS:-3}"
    --save_strategy steps
    --save_steps "${SMOKE_STEPS:-3}"
  )
fi

rit_require_structured_data

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
  --enable_thinking false \
  --add_non_thinking_prefix false \
  --max_length "${MAX_LENGTH:-1280}" \
  --per_device_train_batch_size "${SFT_BATCH:-12}" \
  --per_device_eval_batch_size "${SFT_EVAL_BATCH:-12}" \
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
