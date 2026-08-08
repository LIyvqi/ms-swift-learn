#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/course/alignment_common.sh"
mapfile -t SPAN < <(alignment_span_args)
SFT_ADAPTER="$(alignment_sft_checkpoint)"
OUTPUT="${ORPO_OUTPUT:-${ROOT}/outputs/18_orpo/orpo${ALIGNMENT_SUFFIX}}"

swift rlhf \
  --rlhf_type orpo \
  --model "${ALIGNMENT_MODEL}" \
  --adapters "${SFT_ADAPTER}" \
  --dataset "${ALIGNMENT_DATA}/pairwise_${ALIGNMENT_SPLIT}.jsonl" \
  --val_dataset "${ALIGNMENT_DATA}/pairwise_val.jsonl" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --beta "${ORPO_LAMBDA:-0.1}" \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --per_device_train_batch_size "${ORPO_BATCH:-32}" \
  --per_device_eval_batch_size "${EVAL_BATCH:-32}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${ORPO_LR:-2e-5}" \
  --max_length "${ALIGNMENT_MAX_LENGTH:-384}" \
  --truncation_strategy left \
  --logging_steps 1 \
  --save_total_limit 1 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 8 \
  --report_to tensorboard \
  --output_dir "${OUTPUT}" \
  "${SPAN[@]}"
