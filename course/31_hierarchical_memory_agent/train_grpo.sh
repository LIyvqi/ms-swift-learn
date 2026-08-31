#!/usr/bin/env bash

# 在 GYM 多轮环境中优化库选择、目录定位、检索和最终审核策略。
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SFT_EPOCHS="${SFT_EPOCHS:-2}"
SFT_ROOT="${SFT_OUTPUT:-${HIERARCHICAL_OUTPUT}/sft_${SFT_EPOCHS}epoch}"
GRPO_EPOCHS="${GRPO_EPOCHS:-2}"
TRAIN_DATA="${HIERARCHICAL_DATA}/rl_train.jsonl"
OUTPUT="${GRPO_OUTPUT:-${HIERARCHICAL_OUTPUT}/grpo_${GRPO_EPOCHS}epoch}"
EXTRA_ARGS=(--num_train_epochs "${GRPO_EPOCHS}" --save_strategy steps --save_steps "${GRPO_SAVE_STEPS:-120}")

if [[ "${SMOKE:-0}" == "1" ]]; then
  SFT_ROOT="${SFT_OUTPUT:-${HIERARCHICAL_OUTPUT}/sft_smoke}"
  TRAIN_DATA="${HIERARCHICAL_DATA}/rl_smoke.jsonl"
  OUTPUT="${GRPO_OUTPUT:-${HIERARCHICAL_OUTPUT}/grpo_smoke}"
  EXTRA_ARGS=(--max_steps "${SMOKE_STEPS:-2}" --save_strategy steps --save_steps "${SMOKE_STEPS:-2}")
fi

hierarchical_require_data
SFT_ADAPTER="${SFT_ADAPTER:-$(hierarchical_latest_checkpoint "${SFT_ROOT}")}"

# 四个可解释奖励之后，框架会自动追加环境逐步累计奖励。
swift rlhf \
  --rlhf_type grpo \
  --model "${HIERARCHICAL_MODEL}" \
  --adapters "${SFT_ADAPTER}" \
  --ref_adapters "${SFT_ADAPTER}" \
  --dataset "${TRAIN_DATA}" \
  --external_plugins "${HIERARCHICAL_PLUGIN}" \
  --reward_funcs \
    course_hierarchical_decision \
    course_hierarchical_navigation \
    course_hierarchical_grounding \
    course_hierarchical_efficiency \
  --reward_weights 1.2 0.6 0.35 0.25 1.0 \
  --use_gym_env true \
  --multi_turn_scheduler course_hierarchical_memory_scheduler \
  --gym_env course_hierarchical_memory_audit \
  --max_turns "${MAX_TURNS:-5}" \
  --completion_length_limit_scope per_round \
  --tuner_type lora \
  --lora_rank "${LORA_RANK:-32}" \
  --lora_alpha "${LORA_ALPHA:-64}" \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --enable_thinking true \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "${VLLM_MEMORY:-0.4}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-6144}" \
  --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
  --vllm_mm_processor_cache_gb 0 \
  --vllm_enforce_eager true \
  --sleep_level 1 \
  --num_generations "${NUM_GENERATIONS:-3}" \
  --generation_batch_size "${GENERATION_BATCH:-12}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --per_device_train_batch_size "${RL_BATCH:-6}" \
  --gradient_accumulation_steps 1 \
  --optim "${OPTIMIZER:-adamw_torch}" \
  --learning_rate "${GRPO_LEARNING_RATE:-1e-6}" \
  --beta "${GRPO_BETA:-0.01}" \
  --max_grad_norm 1.0 \
  --max_length "${MAX_LENGTH:-4608}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH:-256}" \
  --logging_steps 1 \
  --save_total_limit "${GRPO_SAVE_LIMIT:-12}" \
  --save_only_model "${SAVE_ONLY_MODEL:-false}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-true}" \
  --dataset_num_proc 1 \
  --dataloader_num_workers 0 \
  --log_completions true \
  --report_to tensorboard \
  --output_dir "${OUTPUT}" \
  "${EXTRA_ARGS[@]}"
