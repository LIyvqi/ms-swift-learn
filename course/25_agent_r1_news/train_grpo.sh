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
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-0}"
OUTPUT="${GRPO_OUTPUT:-${ROOT}/outputs/25_agent_r1_news/grpo_${GRPO_EPOCHS}epoch}"
TRAIN_DATA="${DATA}/rl_train.jsonl"
EXTRA_ARGS=()

if [[ "${SMOKE:-0}" == "1" ]]; then
  SFT_ROOT="${SFT_OUTPUT:-${ROOT}/outputs/25_agent_r1_news/sft_smoke}"
  TRAIN_DATA="${DATA}/rl_smoke.jsonl"
  OUTPUT="${GRPO_OUTPUT:-${ROOT}/outputs/25_agent_r1_news/grpo_smoke}"
  SAVE_LIMIT="${GRPO_SAVE_LIMIT:-2}"
  SAVE_ONLY="${SAVE_ONLY_MODEL:-true}"
  SAVE_STEPS="${SMOKE_STEPS:-2}"
  EXTRA_ARGS+=(--max_steps "${SAVE_STEPS}" --save_steps "${SAVE_STEPS}")
else
  SAVE_LIMIT="${GRPO_SAVE_LIMIT:-24}"
  SAVE_ONLY="${SAVE_ONLY_MODEL:-false}"
  SAVE_STEPS="${GRPO_SAVE_STEPS:-120}"
  if [[ "${GRPO_MAX_STEPS}" -gt 0 ]]; then
    EXTRA_ARGS+=(
      --max_steps "${GRPO_MAX_STEPS}"
      --save_strategy steps
      --save_steps "${SAVE_STEPS}"
    )
  else
    EXTRA_ARGS+=(
      --num_train_epochs "${GRPO_EPOCHS}"
      --save_strategy "${GRPO_SAVE_STRATEGY:-steps}"
      --save_steps "${SAVE_STEPS}"
    )
  fi
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  RESUME_PATH="$(realpath "${RESUME_FROM_CHECKPOINT}")"
  # Transformers 会从旧 trainer_state.json 恢复保存间隔；始终复制后同步为本次配置。
  RESUME_COPY="$(realpath -m "${OUTPUT}/resume_${RESUME_PATH##*/}")"
  if [[ "${RESUME_PATH}" != "${RESUME_COPY}" ]]; then
    mkdir -p "${RESUME_COPY}"
    cp -a "${RESUME_PATH}/." "${RESUME_COPY}/"
  fi
  RESUME_PATH="${RESUME_COPY}"
  if [[ -f "${RESUME_PATH}/trainer_state.json" ]]; then
    python - "${RESUME_PATH}/trainer_state.json" "${SAVE_STEPS}" <<'PY'
import json
import sys
from pathlib import Path

状态文件 = Path(sys.argv[1])
保存间隔 = int(sys.argv[2])
状态 = json.loads(状态文件.read_text(encoding="utf-8"))
状态["save_steps"] = 保存间隔
状态["eval_steps"] = float(保存间隔)
状态文件.write_text(
    json.dumps(状态, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
  fi
  if [[ "${RESUME_RESET_OPTIMIZER:-false}" == "true" ]]; then
    # ROCm/BF16 或跨节点恢复可能让 Adam 状态与当前参数 dtype/device 不一致；只重建优化器和调度器。
    [[ ! -f "${RESUME_COPY}/optimizer.pt" ]] || mv -f \
      "${RESUME_COPY}/optimizer.pt" "${RESUME_COPY}/optimizer.pt.disabled"
    [[ ! -f "${RESUME_COPY}/scheduler.pt" ]] || mv -f \
      "${RESUME_COPY}/scheduler.pt" "${RESUME_COPY}/scheduler.pt.disabled"
  fi
  EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_PATH}")
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
  --optim "${OPTIMIZER:-adamw_torch}" \
  --learning_rate "${GRPO_LEARNING_RATE:-1e-6}" \
  --beta "${GRPO_BETA:-0.01}" \
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
