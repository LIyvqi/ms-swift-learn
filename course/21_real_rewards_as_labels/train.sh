#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/course/alignment_common.sh"
mapfile -t SPAN < <(alignment_span_args)
SFT_ADAPTER="$(alignment_sft_checkpoint)"
PLUGIN="${ROOT}/course/plugins/classification_rewards.py"
COMPAT_PLUGIN="${ROOT}/course/plugins/real_loss_compat.py"
OUTPUT="${REAL_LABEL_OUTPUT:-${ROOT}/outputs/21_real_rewards_as_labels/real${ALIGNMENT_SUFFIX}}"

swift rlhf \
  --rlhf_type grpo \
  --loss_type real \
  --real_tau "${REAL_TAU:-0.5}" \
  --model "${ALIGNMENT_MODEL}" \
  --adapters "${SFT_ADAPTER}" \
  --ref_adapters "${SFT_ADAPTER}" \
  --dataset "${ALIGNMENT_DATA}/prompts_${ALIGNMENT_SPLIT}.jsonl" \
  --external_plugins "${PLUGIN}" "${COMPAT_PLUGIN}" \
  --reward_funcs course_classification_accuracy course_classification_format \
  --reward_weights 1.0 0.2 \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "${VLLM_MEMORY:-0.50}" \
  --vllm_max_model_len 1024 \
  --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
  --vllm_mm_processor_cache_gb 0 \
  --vllm_enforce_eager true \
  --sleep_level 1 \
  --num_generations "${NUM_GENERATIONS:-8}" \
  --per_device_train_batch_size "${REAL_BATCH:-32}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${REAL_LR:-5e-6}" \
  --beta 0.001 \
  --max_length "${ALIGNMENT_MAX_LENGTH:-384}" \
  --max_completion_length "${REAL_MAX_COMPLETION_LENGTH:-24}" \
  --temperature 1.5 \
  --logging_steps 1 \
  --save_total_limit 1 \
  --save_only_model true \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --log_completions true \
  --report_to tensorboard \
  --output_dir "${OUTPUT}" \
  "${SPAN[@]}"
