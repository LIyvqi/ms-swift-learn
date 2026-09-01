#!/usr/bin/env bash

# 极简安全审核 Agent 的持久化路径和检查点工具。
set -euo pipefail

AGENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${AGENT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/activate.sh"

RIT_AGENT_MODEL="${PROJECT_ROOT}/models/Qwen3.5-0.8B-Base"
RIT_AGENT_DATA="${PROJECT_ROOT}/datasets/rit_audit_agent"
RIT_AGENT_OUTPUT="${PROJECT_ROOT}/outputs/32_rit_rubric_rl/agent"
RIT_AGENT_PLUGIN="${PROJECT_ROOT}/course/plugins/rit_audit_agent.py"

rit_agent_latest_checkpoint() {
  local directory="$1"
  local checkpoint
  checkpoint="$({ find "${directory}" -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null || true; } \
    | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "${checkpoint}" ]]; then
    echo "找不到 Agent 检查点：${directory}" >&2
    return 1
  fi
  printf '%s\n' "${checkpoint}"
}
rit_agent_require_data() {
  local path
  for path in \
    "${RIT_AGENT_DATA}/rules.jsonl" \
    "${RIT_AGENT_DATA}/cases.jsonl" \
    "${RIT_AGENT_DATA}/sft_train.jsonl" \
    "${RIT_AGENT_DATA}/rl_train.jsonl"; do
    if [[ ! -s "${path}" ]]; then
      echo "缺少 Agent 数据：${path}；请先运行 prepare_agent_data.py" >&2
      return 1
    fi
  done
}
