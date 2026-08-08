#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"

if [[ "${SMOKE:-0}" == "1" ]]; then
  DATA_SPLIT=smoke
  OUTPUT="${REAL_SFT_OUTPUT:-${ROOT}/outputs/22_real_regression/sft_smoke}"
  SPAN=(--max_steps 1 --save_steps 1 --eval_strategy no)
else
  DATA_SPLIT=train
  OUTPUT="${REAL_SFT_OUTPUT:-${ROOT}/outputs/22_real_regression/sft}"
  SPAN=(--num_train_epochs "${EPOCHS:-5}" --save_strategy epoch --eval_strategy epoch)
fi

swift sft \
  --model "${ROOT}/models/Qwen3.5-0.8B-Base" \
  --dataset "${ROOT}/datasets/real_judge_1to5/sft_${DATA_SPLIT}.jsonl" \
  --val_dataset "${ROOT}/datasets/real_judge_1to5/sft_val.jsonl" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --per_device_train_batch_size "${REAL_SFT_BATCH:-96}" \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate "${REAL_SFT_LR:-2e-4}" \
  --max_length 768 \
  --logging_steps 1 \
  --save_total_limit 1 \
  --save_only_model true \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --report_to tensorboard \
  --output_dir "${OUTPUT}" \
  "${SPAN[@]}"
