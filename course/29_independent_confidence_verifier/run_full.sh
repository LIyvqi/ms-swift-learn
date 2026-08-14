#!/usr/bin/env bash

# 一键训练独立 Verifier，并对第 28 课 Brier-RLCR 策略评分。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/course/confidence_common.sh"
activate_confidence_env

python "${ROOT}/course/28_rlcr_confidence/prepare_data.py"
PYTHONPATH="${ROOT}/course/28_rlcr_confidence:${ROOT}/course/29_independent_confidence_verifier:${PYTHONPATH}" \
  python -m pytest -q "${ROOT}/course/29_independent_confidence_verifier/test_verifier.py"

bash "${ROOT}/course/29_independent_confidence_verifier/train_verifier.sh"
VERIFIER="$(latest_confidence_checkpoint "${ROOT}/outputs/29_independent_confidence_verifier/verifier")"

if [[ ! -f "${ROOT}/outputs/28_rlcr_confidence/evaluation/brier_rlcr.jsonl" ]]; then
  echo "缺少第 28 课 Brier-RLCR 真实策略轨迹，请先运行 course/28_rlcr_confidence/run_full.sh" >&2
  exit 1
fi

python "${ROOT}/course/29_independent_confidence_verifier/evaluate_verifier.py" \
  --verifier "${VERIFIER}" \
  --batch-size "${VERIFIER_SCORE_BATCH:-32}"
