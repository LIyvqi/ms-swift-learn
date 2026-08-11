#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common_multimodal.sh"

STYLE="${STYLE:-cot}"
if [[ "${STYLE}" != "cot" && "${STYLE}" != "direct" ]]; then
  echo "STYLE 只能是 cot 或 direct" >&2
  exit 2
fi
mapfile -t SPAN < <(training_span_args)

# 默认冻结视觉编码器与对齐层，只在语言模型上加入 LoRA；小数据上更稳定。
swift sft \
  --model "${MODEL_BASE}" \
  --dataset "$(multimodal_dataset_path "${STYLE}")" \
  --val_dataset "$(multimodal_dataset_path "${STYLE}" val)" \
  --tuner_type lora \
  --lora_rank "${LORA_RANK:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --lora_dropout 0.05 \
  --freeze_vit "${FREEZE_VIT:-true}" \
  --freeze_aligner "${FREEZE_ALIGNER:-true}" \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --max_length "${MAX_LENGTH:-2048}" \
  --max_pixels "${MAX_PIXELS:-1048576}" \
  --per_device_train_batch_size "${MM_SFT_BATCH:-${SFT_BATCH}}" \
  --per_device_eval_batch_size "${MM_EVAL_BATCH:-${MM_SFT_BATCH:-${SFT_BATCH}}}" \
  --eval_accumulation_steps "${EVAL_ACCUMULATION_STEPS:-1}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --warmup_ratio 0.05 \
  --logging_steps 1 \
  --save_total_limit 1 \
  --save_only_model true \
  --gradient_checkpointing false \
  --group_by_length "${GROUP_BY_LENGTH:-true}" \
  --torch_empty_cache_steps "${EMPTY_CACHE_STEPS:-1}" \
  --dataloader_num_workers 2 \
  --dataset_num_proc 2 \
  --report_to tensorboard \
  --output_dir "$(output_path "01_lora_multimodal_${STYLE}")" \
  "${SPAN[@]}"
