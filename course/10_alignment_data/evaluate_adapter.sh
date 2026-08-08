#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"

if [[ -z "${ADAPTER:-}" ]]; then
  echo "必须通过 ADAPTER 指定待评测的 LoRA checkpoint" >&2
  exit 2
fi
NAME="${NAME:-alignment_adapter}"
RESULT="${RESULT_PATH:-${ROOT}/results/evaluations/alignment_${NAME}.jsonl}"
mkdir -p "$(dirname -- "${RESULT}")"

# 验证数据带 assistant 标准答案；swift infer 会把它作为标签而不是模型输入。
swift infer \
  --adapters "${ADAPTER}" \
  --val_dataset "${ROOT}/datasets/alignment_news/sft_val.jsonl" \
  --infer_backend vllm \
  --stream false \
  --temperature 0 \
  --max_new_tokens "${EVAL_MAX_NEW_TOKENS:-24}" \
  --vllm_max_model_len 1024 \
  --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
  --vllm_mm_processor_cache_gb 0 \
  --vllm_enforce_eager true \
  --vllm_gpu_memory_utilization "${VLLM_MEMORY:-0.60}" \
  --result_path "${RESULT}"

python "${ROOT}/course/08_rloo_classification/score.py" "${RESULT}"
