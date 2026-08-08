#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"

latest_checkpoint() {
  local directory="$1"
  local checkpoint
  checkpoint="$({ find "${directory}" -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null || true; } \
    | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "${checkpoint}" ]]; then
    echo "找不到 REAL SFT 检查点：${directory}" >&2
    return 1
  fi
  printf '%s\n' "${checkpoint}"
}

if [[ "${SMOKE:-0}" == "1" ]]; then
  SFT_ROOT="${REAL_SFT_OUTPUT:-${ROOT}/outputs/22_real_regression/sft_smoke}"
  TRAIN_DATA="${ROOT}/datasets/real_judge_1to5/prompts_smoke.jsonl"
  STEPS="${REAL_STEPS:-1}"
  OUTPUT="${REAL_OUTPUT:-${ROOT}/outputs/22_real_regression/real_smoke}"
else
  SFT_ROOT="${REAL_SFT_OUTPUT:-${ROOT}/outputs/22_real_regression/sft}"
  TRAIN_DATA="${ROOT}/datasets/real_judge_1to5/prompts_train.jsonl"
  STEPS="${REAL_STEPS:-50}"
  OUTPUT="${REAL_OUTPUT:-${ROOT}/outputs/22_real_regression/real_${STEPS}step}"
fi
ADAPTER="${REAL_SFT_ADAPTER:-$(latest_checkpoint "${SFT_ROOT}")}"

python "${ROOT}/course/22_real_regression/train_real.py" \
  --model "${ROOT}/models/Qwen3.5-0.8B-Base" \
  --adapter "${ADAPTER}" \
  --train-data "${TRAIN_DATA}" \
  --eval-data "${ROOT}/datasets/real_judge_1to5/prompts_val.jsonl" \
  --output-dir "${OUTPUT}" \
  --max-steps "${STEPS}" \
  --batch-prompts "${REAL_BATCH_PROMPTS:-16}" \
  --num-rollouts "${REAL_ROLLOUTS:-4}" \
  --max-new-tokens "${REAL_MAX_NEW_TOKENS:-48}" \
  --temperature "${REAL_TEMPERATURE:-1.2}" \
  --learning-rate "${REAL_LR:-2e-5}" \
  --beta-supp "${REAL_BETA_SUPP:-1.0}" \
  --beta-supp-extra "${REAL_BETA_SUPP_EXTRA:-0.01}" \
  --format-penalty "${REAL_FORMAT_PENALTY:-1.0}"
