#!/usr/bin/env bash

# 让 ORM 与 RiT Agent 共用完全一致的环境、起点和 GRPO 参数。
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/agent_common.sh"

AGENT_METHOD="${AGENT_METHOD:-}"
case "${AGENT_METHOD}" in
  orm|rit) ;;
  *) echo "AGENT_METHOD 只能是 orm 或 rit" >&2; exit 2 ;;
esac

TRAIN_DATA="${RIT_AGENT_DATA}/rl_train.jsonl"
SFT_ROOT="${AGENT_SFT_OUTPUT:-${RIT_AGENT_OUTPUT}/sft}"
OUTPUT="${AGENT_GRPO_OUTPUT:-${RIT_AGENT_OUTPUT}/${AGENT_METHOD}_grpo}"
STEPS="${AGENT_RL_STEPS:-30}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  TRAIN_DATA="${RIT_AGENT_DATA}/rl_smoke.jsonl"
  SFT_ROOT="${AGENT_SFT_OUTPUT:-${RIT_AGENT_OUTPUT}/sft_smoke}"
  OUTPUT="${AGENT_GRPO_OUTPUT:-${RIT_AGENT_OUTPUT}/${AGENT_METHOD}_grpo_smoke}"
  STEPS="${SMOKE_STEPS:-2}"
fi

rit_agent_require_data
SFT_ADAPTER="${SFT_ADAPTER:-$(rit_agent_latest_checkpoint "${SFT_ROOT}")}"
REWARD_FUNCS=(course_rit_agent_response course_rit_agent_process course_rit_agent_gated)
if [[ "${AGENT_METHOD}" == "rit" ]]; then
  # 最后一路是框架自动追加的环境累计分；这里只优化 gated reward。
  REWARD_WEIGHTS=(0.0 0.0 1.0 0.0)
else
  REWARD_WEIGHTS=(1.0 0.0 0.0 0.0)
fi

swift rlhf \
  --rlhf_type grpo \
  --model "${RIT_AGENT_MODEL}" \
  --adapters "${SFT_ADAPTER}" \
  --ref_adapters "${SFT_ADAPTER}" \
  --dataset "${TRAIN_DATA}" \
  --external_plugins "${RIT_AGENT_PLUGIN}" \
  --reward_funcs "${REWARD_FUNCS[@]}" \
  --reward_weights "${REWARD_WEIGHTS[@]}" \
  --use_gym_env true \
  --multi_turn_scheduler course_rit_audit_agent_scheduler \
  --gym_env course_rit_audit_agent \
  --max_turns 3 \
  --completion_length_limit_scope per_round \
  --tuner_type lora \
  --lora_rank "${LORA_RANK:-32}" \
  --lora_alpha "${LORA_ALPHA:-64}" \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --enable_thinking false \
  --add_non_thinking_prefix false \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "${VLLM_MEMORY:-0.50}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-4096}" \
  --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
  --vllm_mm_processor_cache_gb 0 \
  --vllm_enforce_eager true \
  --sleep_level 1 \
  --num_generations "${NUM_GENERATIONS:-8}" \
  --generation_batch_size "${GENERATION_BATCH:-16}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --per_device_train_batch_size "${AGENT_RL_BATCH:-8}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${AGENT_GRPO_LEARNING_RATE:-2e-6}" \
  --beta "${AGENT_GRPO_BETA:-0.01}" \
  --max_grad_norm 1.0 \
  --max_length "${MAX_LENGTH:-3072}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH:-384}" \
  --max_steps "${STEPS}" \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS:-${STEPS}}" \
  --save_total_limit 2 \
  --save_only_model false \
  --gradient_checkpointing true \
  --dataset_num_proc 1 \
  --dataloader_num_workers 0 \
  --log_completions true \
  --report_to tensorboard \
  --output_dir "${OUTPUT}"
