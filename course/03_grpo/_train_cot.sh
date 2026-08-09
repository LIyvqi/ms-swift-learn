#!/usr/bin/env bash

# 本文件由三个公开入口调用；通常不要直接执行。
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
mapfile -t SPAN < <(training_span_args)

REWARD_MODE="${REWARD_MODE:-rules}"
if [[ "${REWARD_MODE}" != "rules" && "${REWARD_MODE}" != "judge" && "${REWARD_MODE}" != "hybrid" ]]; then
  echo "REWARD_MODE 只能是 rules、judge 或 hybrid" >&2
  exit 2
fi

# 每次训练前确定性地刷新显式 CoT 视图，避免误用旧的宽松提示。
python "${PROJECT_ROOT}/course/03_grpo/prepare_cot_data.py"

# 冒烟后缀只应影响本次输出，不能把学生起点退回到仅训练一步的旧冒烟模型。
DEFAULT_STUDENT="$({ find "${PROJECT_ROOT}/outputs" -type d \
  -path '*/02_full_sft_mixed*/v*/checkpoint-*' -printf '%T@ %p\n' 2>/dev/null || true; } \
  | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -z "${STUDENT:-${DEFAULT_STUDENT}}" ]]; then
  echo "找不到 02_full_sft_mixed 的前置检查点，请先完成第 02 课" >&2
  exit 1
fi
STUDENT="${STUDENT:-${DEFAULT_STUDENT}}"
REWARD_FUNCS=(course_gsm8k_accuracy course_gsm8k_cot_structure)
REWARD_WEIGHTS=(1.0 0.2)
NUM_GENERATIONS_VALUE="${NUM_GENERATIONS:-8}"
RL_BATCH_VALUE="${RL_BATCH:-8}"

if [[ "${REWARD_MODE}" == "rules" || "${REWARD_MODE}" == "hybrid" ]]; then
  REWARD_FUNCS+=(
    course_gsm8k_cot_calculation
    course_gsm8k_cot_grounding
    course_gsm8k_cot_consistency
  )
  REWARD_WEIGHTS+=(0.5 0.15 0.15)
fi

if [[ "${REWARD_MODE}" == "judge" || "${REWARD_MODE}" == "hybrid" ]]; then
  for variable in GRPO_JUDGE_API_BASE GRPO_JUDGE_API_KEY GRPO_JUDGE_MODEL; do
    if [[ -z "${!variable:-}" ]]; then
      echo "大模型裁判模式缺少环境变量：${variable}" >&2
      exit 2
    fi
  done
  REWARD_FUNCS+=(course_gsm8k_cot_llm_judge)
  if [[ "${REWARD_MODE}" == "judge" ]]; then
    REWARD_WEIGHTS+=(0.8)
  else
    REWARD_WEIGHTS+=(0.4)
  fi
  # API 裁判的成本随 rollout 数线性增加，默认使用较小批量；调用方仍可覆盖。
  NUM_GENERATIONS_VALUE="${NUM_GENERATIONS:-4}"
  RL_BATCH_VALUE="${RL_BATCH:-16}"
fi

swift rlhf \
  --rlhf_type grpo \
  --model "${STUDENT}" \
  --dataset "$(dataset_path prompts_cot_explicit)" \
  --external_plugins "${PLUGIN_GSM8K}" \
  --reward_funcs "${REWARD_FUNCS[@]}" \
  --reward_weights "${REWARD_WEIGHTS[@]}" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --enable_thinking true \
  --add_non_thinking_prefix false \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "${VLLM_MEMORY:-0.60}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-4096}" \
  --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
  --vllm_mm_processor_cache_gb 0 \
  --vllm_enforce_eager true \
  --sleep_level 1 \
  --num_generations "${NUM_GENERATIONS_VALUE}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --per_device_train_batch_size "${RL_BATCH_VALUE}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${LEARNING_RATE:-5e-6}" \
  --beta "${BETA:-0.001}" \
  --scale_rewards "${SCALE_REWARDS:-group}" \
  --max_grad_norm "${MAX_GRAD_NORM:-0.5}" \
  --max_length 512 \
  --max_completion_length "${MAX_COMPLETION_LENGTH:-2048}" \
  --logging_steps 1 \
  --save_total_limit 1 \
  --save_only_model true \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --log_completions true \
  --report_to tensorboard \
  --output_dir "$(output_path "03_grpo_explicit_cot_${REWARD_MODE}")" \
  "${SPAN[@]}"
