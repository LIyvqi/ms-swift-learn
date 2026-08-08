#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"

latest_checkpoint() {
  find "$1" -type d -name 'checkpoint-*' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

evaluate_adapter() {
  local name="$1"
  local adapter="$2"
  local result="${ROOT}/outputs/08_rloo_classification/eval_${name}.jsonl"
  rm -f "${result}"
  swift infer \
    --adapters "${adapter}" \
    --val_dataset "${ROOT}/datasets/fudan_news_4class/val.jsonl" \
    --infer_backend vllm \
    --stream false \
    --temperature 0 \
    --max_new_tokens 32 \
    --vllm_max_model_len 1024 \
    --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
    --vllm_mm_processor_cache_gb 0 \
    --vllm_enforce_eager true \
    --vllm_gpu_memory_utilization 0.45 \
    --result_path "${result}"
  python "${ROOT}/course/08_rloo_classification/score.py" "${result}"
}

evaluate_model() {
  local name="$1"
  local model="$2"
  local result="${ROOT}/outputs/08_rloo_classification/eval_${name}.jsonl"
  rm -f "${result}"
  swift infer \
    --model "${model}" \
    --val_dataset "${ROOT}/datasets/fudan_news_4class/val.jsonl" \
    --infer_backend vllm \
    --stream false \
    --temperature 0 \
    --max_new_tokens 32 \
    --vllm_max_model_len 1024 \
    --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
    --vllm_mm_processor_cache_gb 0 \
    --vllm_enforce_eager true \
    --vllm_gpu_memory_utilization 0.45 \
    --result_path "${result}"
  python "${ROOT}/course/08_rloo_classification/score.py" "${result}"
}

TARGET="${TARGET:-all}"

if [[ "${TARGET}" != "base" && "${TARGET}" != "sft" && "${TARGET}" != "rloo" && "${TARGET}" != "all" ]]; then
  echo "TARGET 只能是 base、sft、rloo 或 all，当前值：${TARGET}" >&2
  exit 1
fi

if [[ "${TARGET}" == "base" || "${TARGET}" == "all" ]]; then
  evaluate_model base "${ROOT}/models/Qwen3.5-0.8B-Base"
fi
if [[ "${TARGET}" == "sft" || "${TARGET}" == "all" ]]; then
  SFT_ROOT="${ROOT}/outputs/08_rloo_classification/sft"
  SFT_ADAPTER="${SFT_ADAPTER:-$(latest_checkpoint "${SFT_ROOT}")}"
  evaluate_adapter sft "${SFT_ADAPTER}"
fi
if [[ "${TARGET}" == "rloo" || "${TARGET}" == "all" ]]; then
  RLOO_ADAPTER="${RLOO_ADAPTER:-$(latest_checkpoint "${ROOT}/outputs/08_rloo_classification/rloo_100step")}"
  evaluate_adapter rloo_100step "${RLOO_ADAPTER}"
fi
