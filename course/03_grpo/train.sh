#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

STYLE="${STYLE:-cot}"
if [[ "${STYLE}" != "cot" && "${STYLE}" != "direct" ]]; then
  echo "STYLE 只能是 cot 或 direct" >&2
  exit 2
fi
STUDENT="${STUDENT:-$(latest_checkpoint "$(output_path 02_full_sft_mixed)")}"
mapfile -t SPAN < <(training_span_args)

if [[ "${STYLE}" == "cot" ]]; then
  echo "说明：历史 STYLE=cot 仅保留逐步提示，但本脚本显式关闭 thinking；真正的显式 CoT 请运行 train_cot_rules.sh。" >&2
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
  --num_generations "${NUM_GENERATIONS:-2}" \
  --per_device_train_batch_size "${RL_BATCH:-2}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
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
  --output_dir "$(output_path "03_grpo_${STYLE}")" \
  "${SPAN[@]}"
