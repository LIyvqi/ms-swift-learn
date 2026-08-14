#!/usr/bin/env bash

# 从同一个 Base 独立训练候选正确性 Reward/Verifier，不复用分类策略参数。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/course/confidence_common.sh"
activate_confidence_env
# ROCm 长序列全参数训练容易产生显存碎片，扩展段分配器可减少不可用保留块。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${ROOT}/models/Qwen3.5-0.8B-Base"
DATA="${ROOT}/datasets/confidence_news"
OUTPUT="${VERIFIER_OUTPUT:-${ROOT}/outputs/29_independent_confidence_verifier/verifier}"
TRAIN_DATA="${DATA}/verifier_train.jsonl"
EXTRA_ARGS=(
  --num_train_epochs "${VERIFIER_EPOCHS:-2}"
  --eval_strategy epoch
  --save_strategy steps
  --save_steps "${VERIFIER_SAVE_STEPS:-20}"
)

if [[ "${SMOKE:-0}" == "1" ]]; then
  TRAIN_DATA="${DATA}/verifier_smoke.jsonl"
  OUTPUT="${VERIFIER_OUTPUT:-${ROOT}/outputs/29_independent_confidence_verifier/verifier_smoke}"
  EXTRA_ARGS=(--max_steps 1 --eval_strategy no --save_strategy steps --save_steps 1)
fi
if [[ -n "${VERIFIER_RESUME:-}" ]]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "${VERIFIER_RESUME}")
fi

swift rlhf \
  --rlhf_type rm \
  --model "${MODEL}" \
  --dataset "${TRAIN_DATA}" \
  --val_dataset "${DATA}/verifier_val.jsonl" \
  --tuner_type full \
  --center_rewards_coefficient "${CENTER_COEF:-0.01}" \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --max_length 768 \
  --truncation_strategy left \
  --per_device_train_batch_size "${VERIFIER_BATCH:-32}" \
  --per_device_eval_batch_size "${VERIFIER_EVAL_BATCH:-16}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${VERIFIER_LR:-1e-5}" \
  --warmup_ratio 0.05 \
  --logging_steps 1 \
  --save_total_limit 1 \
  --save_only_model true \
  --gradient_checkpointing false \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --report_to tensorboard \
  --output_dir "${OUTPUT}" \
  "${EXTRA_ARGS[@]}"
