#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common_multimodal.sh"

STYLE="${STYLE:-cot}"
if [[ "${STYLE}" != "cot" && "${STYLE}" != "direct" ]]; then
  echo "STYLE 只能是 cot 或 direct" >&2
  exit 2
fi
STUDENT="${STUDENT:-$(latest_multimodal_student)}"
mapfile -t SPAN < <(training_span_args)

if [[ "${STYLE}" == "cot" ]]; then
  REWARD_FUNCS=(
    course_multimodal_accuracy
    course_multimodal_cot_structure
    course_multimodal_visual_grounding
    course_multimodal_consistency
  )
  REWARD_WEIGHTS=(1.0 0.25 0.20 0.15)
  ENABLE_THINKING=true
  COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
else
  REWARD_FUNCS=(course_multimodal_accuracy course_multimodal_direct_format)
  REWARD_WEIGHTS=(1.0 0.25)
  ENABLE_THINKING=false
  COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
fi

swift rlhf \
  --rlhf_type grpo \
  --model "${STUDENT}" \
  --dataset "$(multimodal_dataset_path "prompts_${STYLE}")" \
  --external_plugins "${PLUGIN_MULTIMODAL}" \
  --reward_funcs "${REWARD_FUNCS[@]}" \
  --reward_weights "${REWARD_WEIGHTS[@]}" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --freeze_vit true \
  --freeze_aligner true \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --enable_thinking "${ENABLE_THINKING}" \
  --add_non_thinking_prefix false \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "${VLLM_MEMORY:-0.55}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-4096}" \
  --vllm_limit_mm_per_prompt '{"image":1,"video":0}' \
  --vllm_mm_processor_cache_gb "${MM_PROCESSOR_CACHE_GB:-2}" \
  --vllm_enforce_eager true \
  --sleep_level 1 \
  --num_generations "${NUM_GENERATIONS:-4}" \
  --generation_batch_size "${GENERATION_BATCH:-${RL_BATCH:-4}}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --per_device_train_batch_size "${RL_BATCH:-4}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${LEARNING_RATE:-5e-6}" \
  --beta "${BETA:-0.001}" \
  --max_grad_norm "${MAX_GRAD_NORM:-0.5}" \
  --max_length "${MAX_LENGTH:-1536}" \
  --max_completion_length "${COMPLETION_LENGTH}" \
  --max_pixels "${MAX_PIXELS:-1048576}" \
  --logging_steps 1 \
  --save_total_limit 1 \
  --save_only_model true \
  --dataset_num_proc 2 \
  --dataloader_num_workers 2 \
  --log_completions true \
  --report_to tensorboard \
  --output_dir "$(output_path "03_grpo_multimodal_${STYLE}")" \
  "${SPAN[@]}"
