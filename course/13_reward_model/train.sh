#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/course/alignment_common.sh"
mapfile -t SPAN < <(alignment_span_args)
SFT_MODEL="$(alignment_sft_merged_model)"
OUTPUT="${RM_OUTPUT:-${ROOT}/outputs/13_reward_model/rm${ALIGNMENT_SUFFIX}}"

swift rlhf \
  --rlhf_type rm \
  --model "${SFT_MODEL}" \
  --dataset "${ALIGNMENT_DATA}/rm_${ALIGNMENT_SPLIT}.jsonl" \
  --val_dataset "${ALIGNMENT_DATA}/rm_val.jsonl" \
  --tuner_type full \
  --center_rewards_coefficient "${CENTER_COEF:-0.01}" \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --per_device_train_batch_size "${RM_BATCH:-128}" \
  --per_device_eval_batch_size "${EVAL_BATCH:-128}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${RM_LR:-1e-5}" \
  --max_length "${ALIGNMENT_MAX_LENGTH:-384}" \
  --truncation_strategy left \
  --logging_steps 1 \
  --save_total_limit 1 \
  --save_only_model true \
  --dataset_num_proc 8 \
  --dataloader_num_workers 8 \
  --report_to tensorboard \
  --output_dir "${OUTPUT}" \
  "${SPAN[@]}"
