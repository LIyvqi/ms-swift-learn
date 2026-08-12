#!/usr/bin/env bash

# 按论文核心做全参数 SFT：Memory 只看问题，目标规则原文不出现在推理上下文中。
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"

MODEL="${ROOT}/models/Qwen3.5-0.8B-Base"
DATA="${ROOT}/datasets/memo_rule_memory"
OUTPUT="${MEMORY_OUTPUT:-${ROOT}/outputs/26_memo_rule_memory/full_sft}"

SPAN=(
  --num_train_epochs "${MEMORY_EPOCHS:-3}"
  --eval_strategy epoch
  --save_strategy epoch
)
if [[ -n "${MEMORY_MAX_STEPS:-}" ]]; then
  # 容量测试不跑整轮验证，只确认真实全参前向、反向和保存可以完成。
  SPAN=(--max_steps "${MEMORY_MAX_STEPS}" --eval_strategy no --save_strategy steps --save_steps "${MEMORY_MAX_STEPS}")
fi

swift sft \
  --model "${MODEL}" \
  --dataset "${DATA}/memory_train.jsonl" \
  --val_dataset "${DATA}/memory_val.jsonl" \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --enable_thinking false \
  --add_non_thinking_prefix false \
  --max_length "${MEMORY_MAX_LENGTH:-768}" \
  --per_device_train_batch_size "${MEMORY_BATCH:-64}" \
  --per_device_eval_batch_size "${MEMORY_EVAL_BATCH:-64}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${MEMORY_LR:-2e-5}" \
  --lr_scheduler_type constant_with_warmup \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --logging_steps 1 \
  --save_total_limit 3 \
  --save_only_model true \
  --gradient_checkpointing false \
  --group_by_length true \
  --torch_empty_cache_steps 1 \
  --dataloader_num_workers 2 \
  --dataset_num_proc 2 \
  --report_to tensorboard \
  --output_dir "${OUTPUT}" \
  "${SPAN[@]}"
