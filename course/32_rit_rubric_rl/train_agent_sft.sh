#!/usr/bin/env bash

# 用无自由 think 的多轮专家轨迹学习检索与完成动作。
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/agent_common.sh"

TRAIN_DATA="${RIT_AGENT_DATA}/sft_train.jsonl"
VAL_DATA="${RIT_AGENT_DATA}/sft_validation.jsonl"
OUTPUT="${AGENT_SFT_OUTPUT:-${RIT_AGENT_OUTPUT}/sft}"
EXTRA_ARGS=(--num_train_epochs "${AGENT_SFT_EPOCHS:-1}" --eval_strategy epoch --save_strategy epoch)
if [[ "${SMOKE:-0}" == "1" ]]; then
  TRAIN_DATA="${RIT_AGENT_DATA}/sft_smoke.jsonl"
  VAL_DATA="${RIT_AGENT_DATA}/sft_smoke.jsonl"
  OUTPUT="${AGENT_SFT_OUTPUT:-${RIT_AGENT_OUTPUT}/sft_smoke}"
  EXTRA_ARGS=(--max_steps "${SMOKE_STEPS:-3}" --eval_strategy steps --eval_steps "${SMOKE_STEPS:-3}" --save_strategy steps --save_steps "${SMOKE_STEPS:-3}")
fi

rit_agent_require_data
swift sft \
  --model "${RIT_AGENT_MODEL}" \
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
  --max_length "${MAX_LENGTH:-3072}" \
  --per_device_train_batch_size "${AGENT_SFT_BATCH:-6}" \
  --per_device_eval_batch_size "${AGENT_SFT_EVAL_BATCH:-6}" \
  --gradient_accumulation_steps "${AGENT_SFT_GRAD_ACC:-1}" \
  --learning_rate "${AGENT_SFT_LEARNING_RATE:-5e-5}" \
  --warmup_ratio 0.05 \
  --logging_steps 5 \
  --save_total_limit 2 \
  --save_only_model true \
  --gradient_checkpointing false \
  --dataset_num_proc 2 \
  --dataloader_num_workers 2 \
  --report_to tensorboard \
  --output_dir "${OUTPUT}" \
  "${EXTRA_ARGS[@]}"
