#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"

MODEL="${ROOT}/models/Qwen3.5-0.8B-Base"
DATA="${ROOT}/datasets/fudan_news_4class"
OUTPUT="${SFT_OUTPUT:-${ROOT}/outputs/08_rloo_classification/sft}"

swift sft \
  --model "${MODEL}" \
  --dataset "${DATA}/sft_train.jsonl" \
  --val_dataset "${DATA}/val.jsonl" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --max_length 768 \
  --per_device_train_batch_size "${SFT_BATCH:-16}" \
  --per_device_eval_batch_size "${SFT_BATCH:-16}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${SFT_LEARNING_RATE:-1e-4}" \
  --warmup_ratio 0.05 \
  --num_train_epochs "${SFT_EPOCHS:-2}" \
  --eval_strategy epoch \
  --save_strategy epoch \
  --logging_steps 5 \
  --save_total_limit 1 \
  --save_only_model true \
  --gradient_checkpointing false \
  --dataloader_num_workers 0 \
  --dataset_num_proc 1 \
  --report_to tensorboard \
  --output_dir "${OUTPUT}"
