#!/usr/bin/env bash

# 第 31 课共用的持久化路径和检查点工具。
set -euo pipefail

HIERARCHICAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${HIERARCHICAL_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/activate.sh"

HIERARCHICAL_MODEL="${PROJECT_ROOT}/models/Qwen3.5-0.8B-Base"
HIERARCHICAL_DATA="${PROJECT_ROOT}/datasets/hierarchical_memory_audit"
HIERARCHICAL_OUTPUT="${PROJECT_ROOT}/outputs/31_hierarchical_memory_agent"
HIERARCHICAL_PLUGIN="${PROJECT_ROOT}/course/plugins/hierarchical_memory_agent.py"

hierarchical_latest_checkpoint() {
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

hierarchical_require_data() {
  local path
  for path in \
    "${HIERARCHICAL_DATA}/source_registry.json" \
    "${HIERARCHICAL_DATA}/sft_train.jsonl" \
    "${HIERARCHICAL_DATA}/rl_train.jsonl"; do
    if [[ ! -s "${path}" ]]; then
      echo "缺少课程数据：${path}；请先运行 prepare_data.py" >&2
      return 1
    fi
  done
}
