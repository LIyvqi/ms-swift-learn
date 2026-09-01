#!/usr/bin/env bash

# 第 32 课共用的持久化路径、数据检查和检查点查找函数。
set -euo pipefail

RIT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${RIT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/activate.sh"

RIT_MODEL="${PROJECT_ROOT}/models/Qwen3.5-0.8B-Base"
RIT_DATA="${PROJECT_ROOT}/datasets/rit_audit"
RIT_OUTPUT="${PROJECT_ROOT}/outputs/32_rit_rubric_rl"
RIT_PLUGIN="${PROJECT_ROOT}/course/plugins/rit_audit_rewards.py"

rit_latest_checkpoint() {
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

rit_require_data() {
  local path
  for path in \
    "${RIT_DATA}/manifest.json" \
    "${RIT_DATA}/sft_train.jsonl" \
    "${RIT_DATA}/rl_train.jsonl" \
    "${RIT_DATA}/rl_test.jsonl"; do
    if [[ ! -s "${path}" ]]; then
      echo "缺少 RiT 课程数据：${path}；请先运行 prepare_data.py" >&2
      return 1
    fi
  done
}

rit_require_structured_data() {
  local path
  rit_require_data
  for path in \
    "${RIT_DATA}/structured_sft_train.jsonl" \
    "${RIT_DATA}/structured_rl_train.jsonl" \
    "${RIT_DATA}/structured_rl_test.jsonl"; do
    if [[ ! -s "${path}" ]]; then
      echo "缺少 RiT 短结构化数据：${path}；请先运行 prepare_data.py" >&2
      return 1
    fi
  done
}
