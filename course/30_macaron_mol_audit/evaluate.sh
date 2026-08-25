#!/usr/bin/env bash

# 在相同四种检索上下文上依次生成六个 LoRA，再汇总端到端指标。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/course/30_macaron_mol_audit/common.sh"

GENERATION_DIR="${MACARON_OUTPUT}/evaluation/generations"
mkdir -p "${GENERATION_DIR}"

for target in baseline router l1 l2 l3 l4; do
  adapter="$(macaron_latest_checkpoint "${MACARON_OUTPUT}/${target}")"
  echo "生成 ${target}：${adapter}"
  python "${MACARON_DIR}/infer_adapter.py" \
    --target "${target}" \
    --adapter "${adapter}" \
    --batch-size "${INFER_BATCH:-64}" \
    --maximum-samples "${MAXIMUM_SAMPLES:-0}" \
    --output "${GENERATION_DIR}/${target}.jsonl"
done

python "${MACARON_DIR}/score.py" \
  --generation-dir "${GENERATION_DIR}" \
  --output-dir "${MACARON_OUTPUT}/evaluation"
