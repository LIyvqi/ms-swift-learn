#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common_multimodal.sh"

STYLE="${STYLE:-cot}"
if [[ "${STYLE}" != "cot" && "${STYLE}" != "direct" ]]; then
  echo "STYLE 只能是 cot 或 direct" >&2
  exit 2
fi
STUDENT="${STUDENT:-$(latest_multimodal_student)}"
TEACHER_ADAPTER="${TEACHER_ADAPTER:-$(latest_multimodal_teacher "${STYLE}")}"
mapfile -t SPAN < <(training_span_args)

if [[ "${STYLE}" == "cot" ]]; then
  ENABLE_THINKING=true
  COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
else
  ENABLE_THINKING=false
  COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
fi

# OPD 教师提示最长 1678 token，与 2048 token CoT 合计不超过 4096。
# 教师和学生共享基础视觉编码器，蒸馏当前 rollout token 的教师分布。
swift rlhf \
  --rlhf_type grpo \
  --model "${STUDENT}" \
  --teacher_model "${MODEL_BASE}" \
  --teacher_adapters "${TEACHER_ADAPTER}" \
  --teacher_kl_coef "${TEACHER_KL_COEF:-0.3}" \
  --dataset "$(multimodal_dataset_path "prompts_${STYLE}")" \
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
  --vllm_gpu_memory_utilization "${VLLM_MEMORY:-0.50}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-4096}" \
  --vllm_limit_mm_per_prompt '{"image":1,"video":0}' \
  --vllm_mm_processor_cache_gb "${MM_PROCESSOR_CACHE_GB:-2}" \
  --vllm_enforce_eager true \
  --sleep_level 1 \
  --num_generations 1 \
  --generation_batch_size "${GENERATION_BATCH:-${RL_BATCH:-4}}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --per_device_train_batch_size "${RL_BATCH:-4}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${LEARNING_RATE:-5e-6}" \
  --max_grad_norm "${MAX_GRAD_NORM:-0.5}" \
  --max_length "${MAX_LENGTH:-4096}" \
  --max_completion_length "${COMPLETION_LENGTH}" \
  --max_pixels "${MAX_PIXELS:-1048576}" \
  --logging_steps 1 \
  --save_total_limit 1 \
  --save_only_model true \
  --dataset_num_proc 2 \
  --dataloader_num_workers 2 \
  --log_completions true \
  --report_to tensorboard \
  --output_dir "$(output_path "04_opd_multimodal_${STYLE}")" \
  "${SPAN[@]}"
