#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"
cd "${ROOT}"

latest_checkpoint() {
  local directory="$1"
  local checkpoint
  checkpoint="$({ find "${directory}" -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null || true; } \
    | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "${checkpoint}" ]]; then
    echo "找不到检查点：${directory}" >&2
    return 1
  fi
  printf '%s\n' "${checkpoint}"
}

evaluate_adapter() {
  local name="$1"
  local adapter="$2"
  local dataset_name="$3"
  local result="${ROOT}/outputs/09_rloo_cot_classification/eval_${name}_${dataset_name}.jsonl"
  rm -f "${result}"
  swift infer \
    --adapters "${adapter}" \
    --val_dataset "${ROOT}/datasets/fudan_news_cot_50/${dataset_name}.jsonl" \
    --infer_backend vllm \
    --stream false \
    --enable_thinking true \
    --temperature 0 \
    --max_new_tokens 160 \
    --vllm_max_model_len 1024 \
    --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
    --vllm_mm_processor_cache_gb 0 \
    --vllm_enforce_eager true \
    --vllm_gpu_memory_utilization "${EVAL_VLLM_MEMORY:-0.45}" \
    --result_path "${result}"
  python "${ROOT}/course/09_rloo_cot_classification/score.py" "${result}" \
    --reference "${ROOT}/datasets/fudan_news_cot_50/${dataset_name}.jsonl"
}

TARGET="${TARGET:-all}"
if [[ "${TARGET}" != "direct_sft" && "${TARGET}" != "cot_sft" && "${TARGET}" != "cot_rloo" && "${TARGET}" != "all" ]]; then
  echo "TARGET 只能是 direct_sft、cot_sft、cot_rloo 或 all，当前值：${TARGET}" >&2
  exit 1
fi

if [[ "${TARGET}" == "direct_sft" || "${TARGET}" == "all" ]]; then
  DIRECT_SFT_ADAPTER="${DIRECT_SFT_ADAPTER:-$(latest_checkpoint "${ROOT}/outputs/08_rloo_classification/sft")}"
  evaluate_adapter direct_sft "${DIRECT_SFT_ADAPTER}" cot_val_320
fi
if [[ "${TARGET}" == "cot_sft" || "${TARGET}" == "all" ]]; then
  COT_SFT_ADAPTER="${COT_SFT_ADAPTER:-$(latest_checkpoint "${ROOT}/outputs/09_rloo_cot_classification/sft")}"
  evaluate_adapter cot_sft "${COT_SFT_ADAPTER}" cot_val_320
  evaluate_adapter cot_sft "${COT_SFT_ADAPTER}" evidence_val
fi
if [[ "${TARGET}" == "cot_rloo" || "${TARGET}" == "all" ]]; then
  COT_RLOO_ADAPTER="${COT_RLOO_ADAPTER:-$(latest_checkpoint "${ROOT}/outputs/09_rloo_cot_classification/rloo_100step")}"
  evaluate_adapter cot_rloo_100step "${COT_RLOO_ADAPTER}" cot_val_320
  evaluate_adapter cot_rloo_100step "${COT_RLOO_ADAPTER}" evidence_val
fi
