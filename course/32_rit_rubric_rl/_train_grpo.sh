#!/usr/bin/env bash

# 本文件由 ORM 与 RiT 两个公开入口调用，确保除奖励外的训练参数完全一致。
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

METHOD="${METHOD:-}"
case "${METHOD}" in
  orm|rit) ;;
  *)
    echo "METHOD 只能是 orm 或 rit" >&2
    exit 2
    ;;
esac

TRAIN_DATA="${RIT_DATA}/rl_train.jsonl"
SFT_ROOT="${SFT_OUTPUT:-${RIT_OUTPUT}/sft_format}"
OUTPUT="${GRPO_OUTPUT:-${RIT_OUTPUT}/${METHOD}_grpo}"
STEPS="${RL_STEPS:-30}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  TRAIN_DATA="${RIT_DATA}/rl_smoke.jsonl"
  SFT_ROOT="${SFT_OUTPUT:-${RIT_OUTPUT}/sft_smoke}"
  OUTPUT="${GRPO_OUTPUT:-${RIT_OUTPUT}/${METHOD}_grpo_smoke}"
  STEPS="${SMOKE_STEPS:-2}"
fi

rit_require_data
SFT_ADAPTER="${SFT_ADAPTER:-$(rit_latest_checkpoint "${SFT_ROOT}")}"

REWARD_FUNCS=(course_rit_outcome)
REWARD_WEIGHTS=(1.0)
if [[ "${METHOD}" == "rit" ]]; then
  if [[ "${RIT_JUDGE_MODE:-local}" == "api" ]]; then
    for variable in RIT_JUDGE_API_BASE RIT_JUDGE_API_KEY RIT_JUDGE_MODEL; do
      if [[ -z "${!variable:-}" ]]; then
        echo "API rubric 模式缺少环境变量：${variable}" >&2
        exit 2
      fi
    done
    REWARD_FUNCS=(course_rit_api_gated course_rit_outcome)
    REWARD_WEIGHTS=(1.0 0.0)
  else
    REWARD_FUNCS=(course_rit_gated course_rit_outcome course_rit_thinking)
    REWARD_WEIGHTS=(1.0 0.0 0.0)
  fi
fi

# reasoning 任务按论文默认使用 alpha=1 和 min gate；环境变量用于显式消融。
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
  --enable_thinking true \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "${VLLM_MEMORY:-0.50}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-3072}" \
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
  --epsilon "${GRPO_EPSILON:-0.2}" \
  --max_grad_norm 1.0 \
  --max_length "${MAX_LENGTH:-1536}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH:-768}" \
  --max_steps "${STEPS}" \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS:-${STEPS}}" \
  --save_total_limit "${SAVE_LIMIT:-2}" \
  --save_only_model false \
  --gradient_checkpointing true \
  --dataset_num_proc 2 \
  --dataloader_num_workers 0 \
  --log_completions true \
  --report_to tensorboard \
  --output_dir "${OUTPUT}"
