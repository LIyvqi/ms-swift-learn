#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

STUDENT="${STUDENT:-$(latest_checkpoint "$(output_path 02_full_sft_mixed)")}"
TEACHERS='[{"url":"http://127.0.0.1:8001","tags":["cot"]},{"url":"http://127.0.0.1:8002","tags":["direct"]}]'
mapfile -t SPAN < <(training_span_args)

swift rlhf \
  --rlhf_type grpo \
  --model "${STUDENT}" \
  --teacher_model_server "${TEACHERS}" \
  --teacher_tag_key teacher_tag \
  --teacher_kl_coef "${TEACHER_KL_COEF:-0.5}" \
  --dataset "$(dataset_path prompts_multi)" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.35 \
  --vllm_max_model_len 1024 \
  --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
  --vllm_mm_processor_cache_gb 0 \
  --vllm_enforce_eager true \
  --sleep_level 1 \
  --num_generations 1 \
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
  --output_dir "$(output_path 05_mopd)" \
  "${SPAN[@]}"
