#!/usr/bin/env bash
set -euo pipefail

# 用当前持久化环境部署 OpenAI 兼容的冻结基础模型服务。
COURSE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${COURSE_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source ./activate.sh

PORT="${PORT:-8000}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3.5-0.8B-Base}"
API_MODEL="${API_MODEL:-Qwen3.5-0.8B-Base}"

DEPLOY_ARGS=(
  --model "${MODEL_PATH}"
  --served_model_name "${API_MODEL}"
  --infer_backend vllm
  --host 127.0.0.1
  --port "${PORT}"
  --verbose false
  --logprobs true
  --max_logprobs 20
  --max_length 1024
  --vllm_max_model_len 1024
  --vllm_limit_mm_per_prompt '{"image":0,"video":0}'
  --vllm_mm_processor_cache_gb 0
  --vllm_enforce_eager true
  --vllm_gpu_memory_utilization 0.15
)

# 本地教学默认不校验密钥；跨机器部署时通过环境变量启用 Bearer 密钥。
if [[ -n "${JITRL_API_KEY:-}" ]]; then
  DEPLOY_ARGS+=(--api_key "${JITRL_API_KEY}")
fi

exec swift deploy "${DEPLOY_ARGS[@]}"
