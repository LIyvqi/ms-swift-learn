#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

STYLE="${STYLE:-cot}"
if [[ "${STYLE}" != "cot" && "${STYLE}" != "direct" ]]; then
  echo "STYLE 只能是 cot 或 direct" >&2
  exit 2
fi

mapfile -t SPAN < <(training_span_args)

swift sft \
  --model "${MODEL_BASE}" \
  --dataset "$(dataset_path "${STYLE}")" \
  --val_dataset "$(dataset_path "${STYLE}" val)" \
  --load_from_cache_file true \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --add_non_thinking_prefix true \
  --loss_scale default+ignore_empty_think \
  --max_length 512 \
  --per_device_train_batch_size "${SFT_BATCH}" \
  --per_device_eval_batch_size "${SFT_BATCH}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --warmup_ratio 0.05 \
  --logging_steps 5 \
  --save_total_limit 1 \
  --save_only_model true \
  --gradient_checkpointing false \
  --group_by_length "${GROUP_BY_LENGTH:-true}" \
  --dataloader_num_workers "${DATALOADER_WORKERS:-4}" \
  --dataset_num_proc "${DATASET_NUM_PROC:-4}" \
  --report_to tensorboard \
  --output_dir "$(output_path "01_lora_${STYLE}")" \
  "${SPAN[@]}"
