#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

STYLE="${STYLE:-cot}"
if [[ "${STYLE}" != "cot" && "${STYLE}" != "direct" ]]; then
  echo "STYLE 只能是 cot 或 direct" >&2
  exit 2
fi
GRPO_PROFILE="${GRPO_PROFILE:-legacy}"
if [[ "${GRPO_PROFILE}" != "legacy" && "${GRPO_PROFILE}" != "qwen35" ]]; then
  echo "GRPO_PROFILE 只能是 legacy 或 qwen35" >&2
  exit 2
fi
STUDENT="${STUDENT:-$(latest_checkpoint "$(output_path 02_full_sft_mixed)")}"
mapfile -t SPAN < <(training_span_args)

if [[ "${STYLE}" == "cot" ]]; then
  echo "说明：历史 STYLE=cot 仅保留逐步提示，但本脚本显式关闭 thinking；真正的显式 CoT 请运行 train_cot_rules.sh。" >&2
fi

# legacy 保留历史课程参数；qwen35 补充官方最佳实践中的 8 候选、非对称裁剪和不缩放奖励。
if [[ "${GRPO_PROFILE}" == "qwen35" ]]; then
  NUM_GENERATIONS_VALUE="${NUM_GENERATIONS:-8}"
  RL_BATCH_VALUE="${RL_BATCH:-8}"
  LEARNING_RATE_VALUE="${LEARNING_RATE:-5e-6}"
  TEMPERATURE_VALUE="${TEMPERATURE:-1.0}"
  SCALE_REWARDS_VALUE="${SCALE_REWARDS:-none}"
  EPSILON_HIGH_VALUE="${EPSILON_HIGH:-0.28}"
else
  NUM_GENERATIONS_VALUE="${NUM_GENERATIONS:-2}"
  RL_BATCH_VALUE="${RL_BATCH:-2}"
  LEARNING_RATE_VALUE="${LEARNING_RATE:-2e-5}"
  TEMPERATURE_VALUE="${TEMPERATURE:-1.0}"
  SCALE_REWARDS_VALUE="${SCALE_REWARDS:-group}"
  EPSILON_HIGH_VALUE="${EPSILON_HIGH:-0.2}"
fi
OUTPUT_NAME="03_grpo_${STYLE}"
if [[ "${GRPO_PROFILE}" == "qwen35" ]]; then
  OUTPUT_NAME="${OUTPUT_NAME}_qwen35"
fi

swift rlhf \
  --rlhf_type grpo \
  --model "${STUDENT}" \
  --dataset "$(dataset_path "prompts_${STYLE}")" \
  --external_plugins "${PLUGIN_GSM8K}" \
  --reward_funcs course_gsm8k_accuracy course_gsm8k_format \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --enable_thinking false \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.35 \
  --vllm_max_model_len 1024 \
  --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
  --vllm_mm_processor_cache_gb 0 \
  --vllm_enforce_eager true \
  --sleep_level 1 \
  --num_generations "${NUM_GENERATIONS_VALUE}" \
  --temperature "${TEMPERATURE_VALUE}" \
  --per_device_train_batch_size "${RL_BATCH_VALUE}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${LEARNING_RATE_VALUE}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.0 \
  --epsilon "${EPSILON:-0.2}" \
  --epsilon_high "${EPSILON_HIGH_VALUE}" \
  --scale_rewards "${SCALE_REWARDS_VALUE}" \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --max_length 512 \
  --max_completion_length "${MAX_COMPLETION_LENGTH:-256}" \
  --logging_steps 1 \
  --save_total_limit 1 \
  --save_only_model true \
  --dataset_num_proc 1 \
  --dataloader_num_workers 0 \
  --log_completions true \
  --report_to tensorboard \
  --output_dir "$(output_path "${OUTPUT_NAME}")" \
  "${SPAN[@]}"
