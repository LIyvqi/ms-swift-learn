#!/usr/bin/env bash

# 顺序训练单体对照、L0 路由器和 L1～L4 四个专业 LoRA。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/course/30_macaron_mol_audit/common.sh"

for target in baseline router l1 l2 l3 l4; do
  if [[ "${SKIP_EXISTING:-1}" == "1" ]] && macaron_latest_checkpoint "${MACARON_OUTPUT}/${target}" >/dev/null 2>&1; then
    echo "已有 ${target} 检查点，跳过训练"
    continue
  fi
  echo "开始训练 ${target}"
  TARGET="${target}" bash "${MACARON_DIR}/train_adapter.sh"
done

echo "全部 LoRA 检查点："
for target in baseline router l1 l2 l3 l4; do
  printf '%-8s %s\n' "${target}" "$(macaron_latest_checkpoint "${MACARON_OUTPUT}/${target}")"
done
