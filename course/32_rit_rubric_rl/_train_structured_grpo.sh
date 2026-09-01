#!/usr/bin/env bash

# 让短结构化 ORM 与逐字段 RiT 使用相同数据、起点和 GRPO 超参数。
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

STRUCTURED_METHOD="${STRUCTURED_METHOD:-}"
case "${STRUCTURED_METHOD}" in
  orm|rit) ;;
  *) echo "STRUCTURED_METHOD 只能是 orm 或 rit" >&2; exit 2 ;;
esac

TRAIN_DATA="${RIT_DATA}/structured_rl_train.jsonl"
SFT_ROOT="${STRUCTURED_SFT_OUTPUT:-${RIT_OUTPUT}/structured_sft}"
OUTPUT="${STRUCTURED_GRPO_OUTPUT:-${RIT_OUTPUT}/structured_${STRUCTURED_METHOD}_grpo}"
STEPS="${RL_STEPS:-30}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  TRAIN_DATA="${RIT_DATA}/structured_rl_smoke.jsonl"
  SFT_ROOT="${STRUCTURED_SFT_OUTPUT:-${RIT_OUTPUT}/structured_sft_smoke}"
  OUTPUT="${STRUCTURED_GRPO_OUTPUT:-${RIT_OUTPUT}/structured_${STRUCTURED_METHOD}_grpo_smoke}"
  STEPS="${SMOKE_STEPS:-2}"
fi

rit_require_structured_data
SFT_ADAPTER="${SFT_ADAPTER:-$(rit_latest_checkpoint "${SFT_ROOT}")}"
REWARD_FUNCS=(course_rit_structured_outcome)
REWARD_WEIGHTS=(1.0)
if [[ "${STRUCTURED_METHOD}" == "rit" ]]; then
  REWARD_FUNCS=(course_rit_structured_gated course_rit_structured_outcome)
  REWARD_WEIGHTS=(1.0 0.0)
fi

export RIT_ALPHA="${RIT_ALPHA:-1.0}"
export RIT_GATE="${RIT_GATE:-min}"
export RIT_OUTCOME_MODE="${RIT_OUTCOME_MODE:-strict}"

swift rlhf \
  --rlhf_type grpo \
  --model "${RIT_MODEL}" \
  --adapters "${SFT_ADAPTER}" \
  --ref_adapters "${SFT_ADAPTER}" \
  --dataset "${TRAIN_DATA}" \
  --external_plugins "${RIT_PLUGIN}" \
  --reward_funcs "${REWARD_FUNCS[@]}" \
  --reward_weights "${REWARD_WEIGHTS[@]}" \
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
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-2304}" \
  --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
  --vllm_mm_processor_cache_gb 0 \
  --vllm_enforce_eager true \
  --sleep_level 1 \
  --num_generations "${NUM_GENERATIONS:-8}" \
  --generation_batch_size "${GENERATION_BATCH:-16}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --per_device_train_batch_size "${RL_BATCH:-8}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${GRPO_LEARNING_RATE:-2e-6}" \
  --beta "${GRPO_BETA:-0.01}" \
  --max_grad_norm 1.0 \
  --max_length "${MAX_LENGTH:-1280}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH:-384}" \
  --max_steps "${STEPS}" \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS:-${STEPS}}" \
  --save_total_limit "${SAVE_LIMIT:-2}" \
  --gradient_checkpointing true \
  --dataset_num_proc 2 \
  --dataloader_num_workers 0 \
  --log_completions true \
  --report_to tensorboard \
  --output_dir "${OUTPUT}"
