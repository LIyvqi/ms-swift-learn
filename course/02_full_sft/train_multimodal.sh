#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common_multimodal.sh"

STYLE="${STYLE:-mixed}"
if [[ "${STYLE}" != "cot" && "${STYLE}" != "direct" && "${STYLE}" != "mixed" ]]; then
  echo "STYLE 只能是 cot、direct 或 mixed" >&2
  exit 2
fi
mapfile -t SPAN < <(training_span_args)

# 课程默认完整更新语言模型，但冻结已经预训练好的视觉编码器和对齐层。
swift sft \
  --model "${MODEL_BASE}" \
  --dataset "$(multimodal_dataset_path "${STYLE}")" \
  --val_dataset "$(multimodal_dataset_path "${STYLE}" val)" \
  --tuner_type full \
  --freeze_vit "${FREEZE_VIT:-true}" \
  --freeze_aligner "${FREEZE_ALIGNER:-true}" \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --max_length "${MAX_LENGTH:-2048}" \
  --max_pixels "${MAX_PIXELS:-1048576}" \
  --per_device_train_batch_size "${MM_SFT_BATCH:-${SFT_BATCH}}" \
  --per_device_eval_batch_size "${MM_EVAL_BATCH:-${MM_SFT_BATCH:-${SFT_BATCH}}}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --warmup_ratio 0.05 \
  --weight_decay 0.1 \
  --logging_steps 1 \
  --save_total_limit 1 \
  --save_only_model true \
  --gradient_checkpointing false \
  --group_by_length "${GROUP_BY_LENGTH:-true}" \
  --dataloader_num_workers 2 \
  --dataset_num_proc 2 \
  --report_to tensorboard \
  --output_dir "$(output_path "02_full_sft_multimodal_${STYLE}")" \
  "${SPAN[@]}"
