#!/usr/bin/env bash

# 先教会类别+置信度结构；均匀占位数值不携带正确率含义。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/course/confidence_common.sh"
activate_confidence_env

MODEL="${ROOT}/models/Qwen3.5-0.8B-Base"
DATA="${ROOT}/datasets/confidence_news"
OUTPUT="${FORMAT_SFT_OUTPUT:-${ROOT}/outputs/28_rlcr_confidence/format_sft}"

swift sft \
  --model "${MODEL}" \
  --dataset "${DATA}/rlcr_sft.jsonl" \
  --val_dataset "${DATA}/rlcr_sft_val.jsonl" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --max_length 768 \
  --per_device_train_batch_size "${FORMAT_SFT_BATCH:-16}" \
  --per_device_eval_batch_size "${FORMAT_SFT_BATCH:-16}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${FORMAT_SFT_LR:-1e-4}" \
  --warmup_ratio 0.05 \
  --num_train_epochs "${FORMAT_SFT_EPOCHS:-2}" \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 1 \
  --save_only_model true \
  --logging_steps 5 \
  --gradient_checkpointing false \
  --dataset_num_proc 1 \
  --dataloader_num_workers 0 \
  --report_to tensorboard \
  --output_dir "${OUTPUT}"
