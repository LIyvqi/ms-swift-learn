#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"
cd "${ROOT}"

if [[ -z "${GRPO_RUN_DIR:-}" ]]; then
  echo "必须通过 GRPO_RUN_DIR 指定 GRPO 运行目录" >&2
  exit 1
fi
if [[ -z "${SFT_ADAPTER:-}" ]]; then
  echo "必须通过 SFT_ADAPTER 指定训练前的 SFT adapter" >&2
  exit 1
fi

RUN_DIR="$(realpath "${GRPO_RUN_DIR}")"
SFT_ADAPTER_PATH="$(realpath "${SFT_ADAPTER}")"
OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${ROOT}/outputs/25_agent_r1_news}"
PREFIX="${EVAL_PREFIX:-grpo_formal}"
DATASET="${EVAL_DATASET:-${ROOT}/datasets/agent_r1_news/rl_val.jsonl}"
STEPS="${GRPO_EVAL_STEPS:-720 960 1200 1440 2160 2880}"
SELECTION_OFFSET="${SELECTION_OFFSET:-0}"
SELECTION_SAMPLES="${SELECTION_SAMPLES:-120}"
HELDOUT_OFFSET="${HELDOUT_OFFSET:-120}"
HELDOUT_SAMPLES="${HELDOUT_SAMPLES:-840}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-12}"
mkdir -p "${OUTPUT_ROOT}"

SFT_SELECTION="${OUTPUT_ROOT}/${PREFIX}_sft_selection_evaluation.json"
python "${ROOT}/course/25_agent_r1_news/evaluate_agent.py" \
  --adapter "${SFT_ADAPTER_PATH}" \
  --dataset "${DATASET}" \
  --sample-offset "${SELECTION_OFFSET}" \
  --maximum-samples "${SELECTION_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --output "${SFT_SELECTION}"

# 所有候选只看选择集；evaluate_formal_run 同时生成统一对比表和训练曲线。
GRPO_RUN_DIR="${RUN_DIR}" \
SFT_EVAL_RESULT="${SFT_SELECTION}" \
EARLY_CHECKPOINT="${EARLY_CHECKPOINT:-}" \
EARLY_STEP="${EARLY_STEP:-}" \
GRPO_EVAL_STEPS="${STEPS}" \
EVAL_PREFIX="${PREFIX}" \
EVAL_DATASET="${DATASET}" \
EVAL_OFFSET="${SELECTION_OFFSET}" \
EVAL_SAMPLES="${SELECTION_SAMPLES}" \
EVAL_BATCH_SIZE="${BATCH_SIZE}" \
GRPO_SEGMENTS="${GRPO_SEGMENTS:-}" \
bash "${ROOT}/course/25_agent_r1_news/evaluate_formal_run.sh"

CANDIDATES=()
if [[ -n "${EARLY_CHECKPOINT:-}" ]]; then
  effective_early_step="${EARLY_STEP:-${EARLY_CHECKPOINT##*-}}"
  CANDIDATES+=("${OUTPUT_ROOT}/${PREFIX}_checkpoint_${effective_early_step}_evaluation.json")
fi
for step in ${STEPS}; do
  CANDIDATES+=("${OUTPUT_ROOT}/${PREFIX}_checkpoint_${step}_evaluation.json")
done

SELECTION_RESULT="${OUTPUT_ROOT}/${PREFIX}_selection.json"
python "${ROOT}/course/25_agent_r1_news/select_best_evaluation.py" \
  "${CANDIDATES[@]}" \
  --output "${SELECTION_RESULT}"

BEST_ADAPTER="$(python - "${SELECTION_RESULT}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["best"]["adapter"])
PY
)"

# 留出集不再参与任何模型选择，只比较训练前 SFT 和已锁定的最佳 GRPO。
SFT_HELDOUT="${OUTPUT_ROOT}/${PREFIX}_sft_heldout_evaluation.json"
BEST_HELDOUT="${OUTPUT_ROOT}/${PREFIX}_best_heldout_evaluation.json"
python "${ROOT}/course/25_agent_r1_news/evaluate_agent.py" \
  --adapter "${SFT_ADAPTER_PATH}" \
  --dataset "${DATASET}" \
  --sample-offset "${HELDOUT_OFFSET}" \
  --maximum-samples "${HELDOUT_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --output "${SFT_HELDOUT}"
python "${ROOT}/course/25_agent_r1_news/evaluate_agent.py" \
  --adapter "${BEST_ADAPTER}" \
  --dataset "${DATASET}" \
  --sample-offset "${HELDOUT_OFFSET}" \
  --maximum-samples "${HELDOUT_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --output "${BEST_HELDOUT}"

python "${ROOT}/course/25_agent_r1_news/compare_evaluations.py" \
  "${SFT_HELDOUT}" "${BEST_HELDOUT}" \
  --labels SFT 最佳GRPO \
  --output "${OUTPUT_ROOT}/${PREFIX}_heldout_comparison.md"
python "${ROOT}/course/25_agent_r1_news/analyze_failures.py" \
  "${SFT_HELDOUT}" \
  --output "${OUTPUT_ROOT}/${PREFIX}_sft_heldout_failures.json" > /dev/null
python "${ROOT}/course/25_agent_r1_news/analyze_failures.py" \
  "${BEST_HELDOUT}" \
  --output "${OUTPUT_ROOT}/${PREFIX}_best_heldout_failures.json" > /dev/null

echo "选择集比较与独立留出评测已完成：${OUTPUT_ROOT}/${PREFIX}_heldout_comparison.md"
