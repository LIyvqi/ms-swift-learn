#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

STYLE="${STYLE:?请设置 STYLE=cot 或 STYLE=direct}"
PORT="${PORT:?请设置 PORT}"
if [[ "${STYLE}" == "cot" ]]; then
  ADAPTER="${COT_TEACHER_ADAPTER:-$(latest_checkpoint "$(output_path "01_lora_${STYLE}")")}"
else
  ADAPTER="${DIRECT_TEACHER_ADAPTER:-$(latest_checkpoint "$(output_path "01_lora_${STYLE}")")}"
fi

swift deploy \
  --adapters "${STYLE}=${ADAPTER}" \
  --infer_backend vllm \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --max_logprobs 1 \
  --max_length 1024 \
  --vllm_max_model_len 1024 \
  --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
  --vllm_mm_processor_cache_gb 0 \
  --vllm_enforce_eager true \
  --vllm_gpu_memory_utilization 0.15
