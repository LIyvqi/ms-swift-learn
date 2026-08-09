#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"

MODEL="${ROOT}/models/Qwen3.5-0.8B-Base"
DATA="${ROOT}/datasets/agent_r1_news"
EPOCHS="${SFT_EPOCHS:-2}"
OUTPUT="${SFT_OUTPUT:-${ROOT}/outputs/25_agent_r1_news/sft_${EPOCHS}epoch}"
TRAIN_DATA="${DATA}/sft_train.jsonl"
VAL_DATA="${DATA}/sft_val.jsonl"

if [[ "${SMOKE:-0}" == "1" ]]; then
  EPOCHS=1
  TRAIN_DATA="${DATA}/sft_smoke.jsonl"
  VAL_DATA="${DATA}/sft_smoke.jsonl"
  OUTPUT="${SFT_OUTPUT:-${ROOT}/outputs/25_agent_r1_news/sft_smoke}"
fi

# SFT 同时学习检索、反思、组合和决策动作；所有 assistant 轮次都参与损失。
swift sft \
  --model "${MODEL}" \
  --dataset "${TRAIN_DATA}" \
  --val_dataset "${VAL_DATA}" \
  --tuner_type lora \
  --lora_rank "${LORA_RANK:-32}" \
  --lora_alpha "${LORA_ALPHA:-64}" \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --enable_thinking true \
  --max_length "${MAX_LENGTH:-4608}" \
  --per_device_train_batch_size "${SFT_BATCH:-8}" \
  --per_device_eval_batch_size "${SFT_EVAL_BATCH:-8}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${SFT_LEARNING_RATE:-5e-5}" \
  --warmup_ratio 0.05 \
  --num_train_epochs "${EPOCHS}" \
  --eval_strategy epoch \
  --save_strategy epoch \
  --logging_steps 5 \
  --save_total_limit "${SFT_SAVE_LIMIT:-3}" \
  --save_only_model true \
  --gradient_checkpointing false \
  --dataloader_num_workers 0 \
  --dataset_num_proc 1 \
  --report_to tensorboard \
  --output_dir "${OUTPUT}"
