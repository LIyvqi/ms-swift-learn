#!/usr/bin/env bash

# 使用与训练风格一致的 Qwen3.5 thinking 模板评测 LoRA 教师。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"

STYLE="${STYLE:-cot}"
if [[ "${STYLE}" != "cot" && "${STYLE}" != "direct" ]]; then
  echo "STYLE 只能是 cot 或 direct" >&2
  exit 2
fi

if [[ -z "${ADAPTER:-}" ]]; then
  ADAPTER="$({ find "${ROOT}/outputs" -type d \
    -path "*/01_lora_${STYLE}*/v*/checkpoint-*" -printf '%T@ %p\n' 2>/dev/null || true; } \
    | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi
if [[ -z "${ADAPTER}" ]]; then
  echo "找不到 ${STYLE} LoRA 检查点，请先训练或通过 ADAPTER 指定" >&2
  exit 1
fi

ENABLE_THINKING=false
if [[ "${STYLE}" == "cot" ]]; then
  ENABLE_THINKING=true
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-${COT_MAX_NEW_TOKENS:-2048}}"
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${COT_VLLM_MAX_MODEL_LEN:-4096}}"
else
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-${DIRECT_MAX_NEW_TOKENS:-256}}"
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${DIRECT_VLLM_MAX_MODEL_LEN:-1024}}"
fi

RESULT_DIR="${ROOT}/outputs/01_lora_eval"
RESULT_PATH="${RESULT_DIR}/${EVAL_TAG:-best_practice}_${STYLE}.jsonl"
LOG_PATH="${RESULT_DIR}/${EVAL_TAG:-best_practice}_${STYLE}.log"
mkdir -p "${RESULT_DIR}"

# Direct 必须关闭 thinking；显式 CoT 必须开启 thinking，且给足生成长度。
swift infer \
  --adapters "${ADAPTER}" \
  --val_dataset "${ROOT}/datasets/gsm8k_1k/${STYLE}_val.jsonl" \
  --infer_backend vllm \
  --stream false \
  --temperature 0 \
  --enable_thinking "${ENABLE_THINKING}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
  --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
  --vllm_mm_processor_cache_gb 0 \
  --vllm_enforce_eager true \
  --vllm_gpu_memory_utilization "${VLLM_MEMORY:-0.50}" \
  --result_path "${RESULT_PATH}" \
  >"${LOG_PATH}" 2>&1

if [[ "${STYLE}" == "cot" ]]; then
  python "${ROOT}/course/03_grpo/score_cot.py" "${RESULT_PATH}" \
    --reference "${ROOT}/datasets/gsm8k_1k/cot_val.jsonl"
else
  python "${ROOT}/course/07_tuning/score_gsm8k.py" "${RESULT_PATH}"
fi
