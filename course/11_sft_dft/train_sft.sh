#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/course/alignment_common.sh"
mapfile -t SPAN < <(alignment_span_args)

OUTPUT="${SFT_OUTPUT:-${ROOT}/outputs/11_sft_dft/sft${ALIGNMENT_SUFFIX}}"

swift sft \
  --model "${ALIGNMENT_MODEL}" \
  --dataset "${ALIGNMENT_DATA}/sft_${ALIGNMENT_SPLIT}.jsonl" \
  --val_dataset "${ALIGNMENT_DATA}/sft_val.jsonl" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --per_device_train_batch_size "${SFT_BATCH:-48}" \
  --per_device_eval_batch_size "${EVAL_BATCH:-64}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${SFT_LR:-2e-4}" \
  --max_length 768 \
  --warmup_ratio 0.05 \
  --logging_steps 1 \
  --save_total_limit 1 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 8 \
  --report_to tensorboard \
  --output_dir "${OUTPUT}" \
  "${SPAN[@]}"
