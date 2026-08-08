#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
mapfile -t SPAN < <(training_span_args)

swift sft \
  --model "${MODEL_BASE}" \
  --dataset "$(dataset_path mixed)" \
  --val_dataset "$(dataset_path mixed val)" \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --max_length 512 \
  --per_device_train_batch_size "${SFT_BATCH}" \
  --per_device_eval_batch_size "${SFT_BATCH}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --warmup_ratio 0.05 \
  --weight_decay 0.1 \
  --logging_steps 5 \
  --save_total_limit 1 \
  --save_only_model true \
  --gradient_checkpointing false \
  --dataloader_num_workers 0 \
  --dataset_num_proc 1 \
  --report_to tensorboard \
  --output_dir "$(output_path 02_full_sft_mixed)" \
  "${SPAN[@]}"
