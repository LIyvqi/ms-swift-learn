#!/usr/bin/env bash

# 用逐状态样本修复多轮推理时重复 locate 的策略塌缩。
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BASE_SFT_ROOT="${SFT_OUTPUT:-${HIERARCHICAL_OUTPUT}/sft_${SFT_EPOCHS:-2}epoch}"
TRAIN_DATA="${HIERARCHICAL_DATA}/sft_state_train.jsonl"
VAL_DATA="${HIERARCHICAL_DATA}/sft_state_validation.jsonl"
OUTPUT="${STATE_SFT_OUTPUT:-${HIERARCHICAL_OUTPUT}/sft_state_repair}"
EXTRA_ARGS=(--num_train_epochs "${STATE_SFT_EPOCHS:-1}")

if [[ "${SMOKE:-0}" == "1" ]]; then
  BASE_SFT_ROOT="${SFT_OUTPUT:-${HIERARCHICAL_OUTPUT}/sft_smoke}"
  TRAIN_DATA="${HIERARCHICAL_DATA}/sft_state_smoke.jsonl"
  VAL_DATA="${HIERARCHICAL_DATA}/sft_state_smoke.jsonl"
  OUTPUT="${STATE_SFT_OUTPUT:-${HIERARCHICAL_OUTPUT}/sft_state_smoke}"
  EXTRA_ARGS=(--max_steps "${STATE_MAX_STEPS:-3}")
elif [[ -n "${STATE_MAX_STEPS:-}" ]]; then
  EXTRA_ARGS=(--max_steps "${STATE_MAX_STEPS}")
fi
SFT_ADAPTER="${SFT_ADAPTER:-$(hierarchical_latest_checkpoint "${BASE_SFT_ROOT}")}"

hierarchical_require_data
if [[ ! -s "${TRAIN_DATA}" ]]; then
  echo "缺少逐状态训练数据：${TRAIN_DATA}；请先运行 prepare_data.py" >&2
  exit 1
fi

# last_round 会去掉历史思考并只监督当前动作，与真实逐轮推理的上下文处理保持一致。
swift sft \
  --model "${HIERARCHICAL_MODEL}" \
  --adapters "${SFT_ADAPTER}" \
  --dataset "${TRAIN_DATA}" \
  --val_dataset "${VAL_DATA}" \
  --tuner_type lora \
  --lora_rank "${LORA_RANK:-32}" \
  --lora_alpha "${LORA_ALPHA:-64}" \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --enable_thinking true \
  --loss_scale last_round \
  --max_length "${MAX_LENGTH:-6144}" \
  --per_device_train_batch_size "${SFT_BATCH:-2}" \
  --per_device_eval_batch_size "${SFT_EVAL_BATCH:-2}" \
  --gradient_accumulation_steps "${SFT_GRAD_ACC:-4}" \
  --learning_rate "${STATE_SFT_LEARNING_RATE:-1e-5}" \
  --warmup_ratio 0.05 \
  --eval_strategy epoch \
  --save_strategy steps \
  --save_steps "${STATE_SFT_SAVE_STEPS:-50}" \
  --logging_steps 5 \
  --save_total_limit 3 \
  --save_only_model true \
  --gradient_checkpointing false \
  --dataset_num_proc 1 \
  --dataloader_num_workers 0 \
  --report_to tensorboard \
  --output_dir "${OUTPUT}" \
  "${EXTRA_ARGS[@]}"
