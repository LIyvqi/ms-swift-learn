#!/usr/bin/env bash

# 训练一个路由、单体基线或内容审核专家 LoRA。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/course/30_macaron_mol_audit/common.sh"

TARGET="${TARGET:-${1:-}}"
macaron_validate_target "${TARGET}"
macaron_require_data "${TARGET}"

TRAIN_DATA="${MACARON_DATA}/views/${TARGET}_train.jsonl"
VALIDATION_DATA="${MACARON_DATA}/views/${TARGET}_validation.jsonl"
OUTPUT_DIR="${MACARON_OUTPUT}/${TARGET}"

# 本机实测 batch=32 会瞬时占满 191.67 GiB；专家序列稍长，默认降到 30 留出稳定余量。
# 验证需物化大词表 logits，因而单独使用 batch=8。所有 batch 都可用环境变量覆盖。
if [[ "${TARGET}" == l[1-4] ]]; then
  DEFAULT_TRAIN_BATCH=30
else
  DEFAULT_TRAIN_BATCH=32
fi
swift sft \
  --model "${MACARON_MODEL}" \
  --dataset "${TRAIN_DATA}" \
  --val_dataset "${VALIDATION_DATA}" \
  --tuner_type lora \
  --lora_rank "${LORA_RANK:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --max_length "${MAX_LENGTH:-1536}" \
  --per_device_train_batch_size "${TRAIN_BATCH:-${DEFAULT_TRAIN_BATCH}}" \
  --per_device_eval_batch_size "${EVAL_BATCH:-8}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --warmup_ratio 0.05 \
  --num_train_epochs "${EPOCHS:-3}" \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 1 \
  --save_only_model true \
  --logging_steps 2 \
  --gradient_checkpointing false \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --report_to tensorboard \
  --seed 20260826 \
  --output_dir "${OUTPUT_DIR}"
