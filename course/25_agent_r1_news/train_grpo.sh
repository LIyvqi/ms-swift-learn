#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"

latest_checkpoint() {
  local directory="$1"
  local checkpoint
  checkpoint="$({ find "${directory}" -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null || true; } \
    | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "${checkpoint}" ]]; then
    echo "找不到前置 Agent-SFT 检查点：${directory}" >&2
    return 1
  fi
  printf '%s\n' "${checkpoint}"
}

MODEL="${ROOT}/models/Qwen3.5-0.8B-Base"
DATA="${ROOT}/datasets/agent_r1_news"
PLUGIN="${ROOT}/course/plugins/agent_r1_news.py"
SFT_EPOCHS="${SFT_EPOCHS:-2}"
SFT_ROOT="${SFT_OUTPUT:-${ROOT}/outputs/25_agent_r1_news/sft_${SFT_EPOCHS}epoch}"
GRPO_EPOCHS="${GRPO_EPOCHS:-2}"
OUTPUT="${GRPO_OUTPUT:-${ROOT}/outputs/25_agent_r1_news/grpo_${GRPO_EPOCHS}epoch}"
TRAIN_DATA="${DATA}/rl_train.jsonl"
EXTRA_ARGS=()

if [[ "${SMOKE:-0}" == "1" ]]; then
  SFT_ROOT="${SFT_OUTPUT:-${ROOT}/outputs/25_agent_r1_news/sft_smoke}"
  TRAIN_DATA="${DATA}/rl_smoke.jsonl"
  OUTPUT="${GRPO_OUTPUT:-${ROOT}/outputs/25_agent_r1_news/grpo_smoke}"
  SAVE_LIMIT="${GRPO_SAVE_LIMIT:-2}"
  SAVE_ONLY="${SAVE_ONLY_MODEL:-true}"
  EXTRA_ARGS+=(--max_steps "${SMOKE_STEPS:-2}" --save_steps "${SMOKE_STEPS:-2}")
else
  SAVE_LIMIT="${GRPO_SAVE_LIMIT:-12}"
  SAVE_ONLY="${SAVE_ONLY_MODEL:-false}"
  EXTRA_ARGS+=(
    --num_train_epochs "${GRPO_EPOCHS}"
    --save_strategy "${GRPO_SAVE_STRATEGY:-steps}"
    --save_steps "${GRPO_SAVE_STEPS:-240}"
  )
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
SFT_ADAPTER="${SFT_ADAPTER:-$(latest_checkpoint "${SFT_ROOT}")}"

# 前五列是自定义阶段奖励，最后一列是环境累计过程奖励。
swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL}" \
  --adapters "${SFT_ADAPTER}" \
  --ref_adapters "${SFT_ADAPTER}" \
  --dataset "${TRAIN_DATA}" \
  --external_plugins "${PLUGIN}" \
  --reward_funcs \
    course_agent_news_retrieval \
    course_agent_news_composition \
    course_agent_news_decision \
    course_agent_news_protocol \
    course_agent_news_reflection \
  --reward_weights 0.8 0.8 1.2 0.25 0.35 1.0 \
  --use_gym_env true \
  --multi_turn_scheduler course_agent_r1_news_scheduler \
  --gym_env course_agent_r1_news \
  --max_turns "${MAX_TURNS:-6}" \
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
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-5120}" \
  --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
  --vllm_mm_processor_cache_gb 0 \
  --vllm_enforce_eager true \
  --sleep_level 1 \
  --num_generations "${NUM_GENERATIONS:-3}" \
  --generation_batch_size "${GENERATION_BATCH:-12}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --per_device_train_batch_size "${RL_BATCH:-6}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${GRPO_LEARNING_RATE:-3e-6}" \
  --beta "${GRPO_BETA:-0.001}" \
  --max_grad_norm 1.0 \
  --max_length "${MAX_LENGTH:-3584}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH:-160}" \
  --logging_steps 1 \
  --save_total_limit "${SAVE_LIMIT}" \
  --save_only_model "${SAVE_ONLY}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-true}" \
  --dataset_num_proc 1 \
  --dataloader_num_workers 0 \
  --log_completions true \
  --report_to tensorboard \
  --output_dir "${OUTPUT}" \
  "${EXTRA_ARGS[@]}"
