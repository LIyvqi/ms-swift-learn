#!/usr/bin/env bash

# 第 30 课共用的持久化环境、数据和检查点工具。
set -euo pipefail

MACARON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${MACARON_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/course/confidence_common.sh"
activate_confidence_env

MACARON_MODEL="${PROJECT_ROOT}/models/Qwen3.5-0.8B-Base"
MACARON_DATA="${MACARON_DIR}/data"
MACARON_OUTPUT="${PROJECT_ROOT}/outputs/30_macaron_mol_audit"

macaron_latest_checkpoint() {
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

macaron_validate_target() {
  case "$1" in
    router|baseline|l1|l2|l3|l4) ;;
    *)
      echo "TARGET 只能是 router、baseline、l1、l2、l3、l4" >&2
      return 2
      ;;
  esac
}

macaron_require_data() {
  local target="$1"
  for path in \
    "${MACARON_DATA}/beavertails_2000.jsonl" \
    "${MACARON_DATA}/views/${target}_train.jsonl" \
    "${MACARON_DATA}/views/${target}_validation.jsonl"; do
    if [[ ! -s "${path}" ]]; then
      echo "缺少课程数据：${path}；请先运行 prepare_data.py" >&2
      return 1
    fi
  done
}
