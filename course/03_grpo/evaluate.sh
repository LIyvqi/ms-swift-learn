#!/usr/bin/env bash

# 在同一显式 CoT 验证集上比较 GRPO 前后的生成质量。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"

latest_checkpoint() {
  find "$1" -type d -name 'checkpoint-*' -printf '%T@ %p\n' \
    | sort -nr | head -n 1 | cut -d' ' -f2-
}

latest_full_sft() {
  find "${ROOT}/outputs" -type d -path '*/02_full_sft_mixed*/v*/checkpoint-*' -printf '%T@ %p\n' \
    | sort -nr | head -n 1 | cut -d' ' -f2-
}

run_eval() {
  local name="$1"
  shift
  local result="${ROOT}/outputs/03_grpo_eval/${name}.jsonl"
  mkdir -p "$(dirname -- "${result}")"
  if [[ ! -f "${result}" || "$(wc -l <"${result}")" -ne 100 ]]; then
    swift infer \
      "$@" \
      --val_dataset "${ROOT}/datasets/gsm8k_1k/prompts_cot_explicit_val.jsonl" \
      --infer_backend vllm \
      --stream false \
      --temperature 0 \
      --enable_thinking true \
      --max_new_tokens "${EVAL_MAX_NEW_TOKENS:-2048}" \
      --vllm_max_model_len "${EVAL_VLLM_MAX_MODEL_LEN:-4096}" \
      --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
      --vllm_mm_processor_cache_gb 0 \
      --vllm_enforce_eager true \
      --vllm_gpu_memory_utilization "${EVAL_VLLM_MEMORY:-0.50}" \
      --result_path "${result}"
  fi
  python "${ROOT}/course/03_grpo/score_cot.py" "${result}" \
    --reference "${ROOT}/datasets/gsm8k_1k/prompts_cot_explicit_val.jsonl"
}

TARGET="${TARGET:-all}"
if [[ "${TARGET}" != "sft" && "${TARGET}" != "grpo" && "${TARGET}" != "all" ]]; then
  echo "TARGET 只能是 sft、grpo 或 all" >&2
  exit 2
fi

python "${ROOT}/course/03_grpo/prepare_cot_data.py" >/dev/null

if [[ "${TARGET}" == "sft" || "${TARGET}" == "all" ]]; then
  STUDENT="${STUDENT:-$(latest_full_sft)}"
  run_eval sft_before_grpo --model "${STUDENT}"
fi
if [[ "${TARGET}" == "grpo" || "${TARGET}" == "all" ]]; then
  GRPO_ADAPTER="${GRPO_ADAPTER:-$(latest_checkpoint "${ROOT}/outputs/03_grpo_explicit_cot_rules_100step")}" 
  GRPO_EVAL_NAME="${GRPO_EVAL_NAME:-explicit_cot_rules_100step}"
  run_eval "${GRPO_EVAL_NAME}" --adapters "${GRPO_ADAPTER}"
fi
