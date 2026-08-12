#!/usr/bin/env bash

# 运行正式 Memory 训练并写入可观察状态；已存在的历史状态会被本轮时间戳覆盖。
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
STATUS="${ROOT}/outputs/26_memo_rule_memory/status"
mkdir -p "${STATUS}"
printf '开始时间：%s\n' "$(date --iso-8601=seconds)" >"${STATUS}/running.txt"
rm -f "${STATUS}/done.txt" "${STATUS}/failed.txt"

失败处理() {
  local code=$?
  printf '失败时间：%s\n退出码：%s\n' "$(date --iso-8601=seconds)" "${code}" >"${STATUS}/failed.txt"
  rm -f "${STATUS}/running.txt"
  return "${code}"
}
trap 失败处理 ERR

bash "${ROOT}/course/26_memo_rule_memory/train_memory.sh"
printf '完成时间：%s\n' "$(date --iso-8601=seconds)" >"${STATUS}/done.txt"
rm -f "${STATUS}/running.txt"
printf '训练已完成。\n'
