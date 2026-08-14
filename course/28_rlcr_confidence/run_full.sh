#!/usr/bin/env bash

# 一键完成数据、格式 SFT、三组 RL 对照与真实校准评测。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/course/confidence_common.sh"
activate_confidence_env

python "${ROOT}/course/28_rlcr_confidence/prepare_data.py"
PYTHONPATH="${ROOT}/course/28_rlcr_confidence:${PYTHONPATH}" \
  python -m pytest -q "${ROOT}/course/28_rlcr_confidence/test_rlcr.py"

bash "${ROOT}/course/28_rlcr_confidence/train_format_sft.sh"
for METHOD in correctness brier log; do
  METHOD="${METHOD}" bash "${ROOT}/course/28_rlcr_confidence/train_rlcr.sh"
done

FORMAT_ADAPTER="$(latest_confidence_checkpoint "${ROOT}/outputs/28_rlcr_confidence/format_sft")"
CORRECTNESS_ADAPTER="$(latest_confidence_checkpoint "${ROOT}/outputs/28_rlcr_confidence/correctness_100step")"
BRIER_ADAPTER="$(latest_confidence_checkpoint "${ROOT}/outputs/28_rlcr_confidence/brier_100step")"
LOG_ADAPTER="$(latest_confidence_checkpoint "${ROOT}/outputs/28_rlcr_confidence/log_100step")"

python "${ROOT}/course/28_rlcr_confidence/evaluate.py" \
  --format-sft "${FORMAT_ADAPTER}" \
  --correctness "${CORRECTNESS_ADAPTER}" \
  --brier "${BRIER_ADAPTER}" \
  --log-score "${LOG_ADAPTER}" \
  --batch-size "${EVAL_BATCH:-128}"
