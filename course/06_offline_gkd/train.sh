#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

STYLE="${STYLE:-cot}"
if [[ "${STYLE}" != "cot" && "${STYLE}" != "direct" ]]; then
  echo "STYLE 只能是 cot 或 direct" >&2
  exit 2
fi
STUDENT="${STUDENT:-$(latest_checkpoint "$(output_path 02_full_sft_mixed)")}"
TEACHER_ADAPTER="${TEACHER_ADAPTER:-$(latest_checkpoint "$(output_path "01_lora_${STYLE}")")}"
mapfile -t SPAN < <(training_span_args)

# lmbda=0 表示完全离线、离策略的 GKD，学生只学习数据集中已经固定的回答。
swift rlhf \
  --rlhf_type gkd \
  --model "${STUDENT}" \
  --teacher_model "${MODEL_BASE}" \
  --teacher_adapters "${TEACHER_ADAPTER}" \
  --dataset "$(dataset_path "${STYLE}")" \
  --lmbda "${GKD_LMBDA:-0}" \
  --beta "${GKD_BETA:-0.5}" \
  --temperature 1.0 \
  --sft_alpha "${SFT_ALPHA:-0.1}" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --max_length 512 \
  --per_device_train_batch_size "${RL_BATCH:-2}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --logging_steps 1 \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-1}" \
  --save_only_model true \
  --gradient_checkpointing false \
  --dataset_num_proc 1 \
  --dataloader_num_workers 0 \
  --report_to tensorboard \
  --output_dir "$(output_path "06_offline_gkd_${STYLE}")" \
  "${SPAN[@]}"
