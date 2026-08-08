#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/course/alignment_common.sh"
mapfile -t SPAN < <(alignment_span_args)
SFT_ADAPTER="$(alignment_sft_checkpoint)"
OUTPUT="${KTO_OUTPUT:-${ROOT}/outputs/15_kto/kto${ALIGNMENT_SUFFIX}}"

swift rlhf \
  --rlhf_type kto \
  --model "${ALIGNMENT_MODEL}" \
  --adapters "${SFT_ADAPTER}" \
  --ref_adapters "${SFT_ADAPTER}" \
  --dataset "${ALIGNMENT_DATA}/kto_${ALIGNMENT_SPLIT}.jsonl" \
  --val_dataset "${ALIGNMENT_DATA}/kto_val.jsonl" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --beta "${KTO_BETA:-0.1}" \
  --desirable_weight "${DESIRABLE_WEIGHT:-1.0}" \
  --undesirable_weight "${UNDESIRABLE_WEIGHT:-1.0}" \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --per_device_train_batch_size "${KTO_BATCH:-64}" \
  --per_device_eval_batch_size "${EVAL_BATCH:-32}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${KTO_LR:-2e-5}" \
  --max_length "${ALIGNMENT_MAX_LENGTH:-384}" \
  --truncation_strategy left \
  --logging_steps 1 \
  --save_total_limit 1 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 8 \
  --report_to tensorboard \
  --output_dir "${OUTPUT}" \
  "${SPAN[@]}"
