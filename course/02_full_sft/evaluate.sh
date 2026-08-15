#!/usr/bin/env bash

# 分别用 Direct 与 thinking 模式评测同一个混合全参 SFT 学生。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"

STYLE="${STYLE:-both}"
if [[ "${STYLE}" != "cot" && "${STYLE}" != "direct" && "${STYLE}" != "both" ]]; then
  echo "STYLE 只能是 cot、direct 或 both" >&2
  exit 2
fi

if [[ -z "${STUDENT:-}" ]]; then
  STUDENT="$({ find "${ROOT}/outputs" -type d \
    -path '*/02_full_sft_mixed*/v*/checkpoint-*' -printf '%T@ %p\n' 2>/dev/null || true; } \
    | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi
if [[ -z "${STUDENT}" ]]; then
  echo "找不到混合全参 SFT 检查点，请先训练或通过 STUDENT 指定" >&2
  exit 1
fi

evaluate_style() {
  local style="$1"
  local enable_thinking=false
  local max_new_tokens="${DIRECT_MAX_NEW_TOKENS:-256}"
  local vllm_max_model_len="${DIRECT_VLLM_MAX_MODEL_LEN:-1024}"
  if [[ "${style}" == "cot" ]]; then
    enable_thinking=true
    max_new_tokens="${COT_MAX_NEW_TOKENS:-2048}"
    vllm_max_model_len="${COT_VLLM_MAX_MODEL_LEN:-4096}"
  fi

  local result_dir="${ROOT}/outputs/02_full_sft_eval"
  local result_path="${result_dir}/${EVAL_TAG:-best_practice}_${style}.jsonl"
  local log_path="${result_dir}/${EVAL_TAG:-best_practice}_${style}.log"
  mkdir -p "${result_dir}"

  swift infer \
    --model "${STUDENT}" \
    --val_dataset "${ROOT}/datasets/gsm8k_1k/${style}_val.jsonl" \
    --infer_backend vllm \
    --stream false \
    --temperature 0 \
    --enable_thinking "${enable_thinking}" \
    --max_new_tokens "${max_new_tokens}" \
    --vllm_max_model_len "${vllm_max_model_len}" \
    --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
    --vllm_mm_processor_cache_gb 0 \
    --vllm_enforce_eager true \
    --vllm_gpu_memory_utilization "${VLLM_MEMORY:-0.50}" \
    --result_path "${result_path}" \
    >"${log_path}" 2>&1

  if [[ "${style}" == "cot" ]]; then
    python "${ROOT}/course/03_grpo/score_cot.py" "${result_path}" \
      --reference "${ROOT}/datasets/gsm8k_1k/cot_val.jsonl"
  else
    python "${ROOT}/course/07_tuning/score_gsm8k.py" "${result_path}"
  fi
}

if [[ "${STYLE}" == "both" ]]; then
  evaluate_style direct
  evaluate_style cot
else
  evaluate_style "${STYLE}"
fi
