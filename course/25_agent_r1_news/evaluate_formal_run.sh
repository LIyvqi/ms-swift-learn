#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"

if [[ -z "${GRPO_RUN_DIR:-}" ]]; then
  echo "必须通过 GRPO_RUN_DIR 指定正式 GRPO 运行目录" >&2
  exit 1
fi

RUN_DIR="$(realpath "${GRPO_RUN_DIR}")"
OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${ROOT}/outputs/25_agent_r1_news}"
EVAL_STEPS="${GRPO_EVAL_STEPS:-720 1440 2160 2880}"
EVAL_SAMPLES="${EVAL_SAMPLES:-120}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-12}"
EVAL_PREFIX="${EVAL_PREFIX:-grpo_formal}"
DATASET="${EVAL_DATASET:-${ROOT}/datasets/agent_r1_news/rl_val.jsonl}"
mkdir -p "${OUTPUT_ROOT}"

COMPARE_FILES=()
COMPARE_LABELS=()
if [[ -n "${SFT_EVAL_RESULT:-}" ]]; then
  if [[ ! -f "${SFT_EVAL_RESULT}" ]]; then
    echo "找不到 SFT 基线评测：${SFT_EVAL_RESULT}" >&2
    exit 1
  fi
  COMPARE_FILES+=("${SFT_EVAL_RESULT}")
  COMPARE_LABELS+=("SFT")
fi

# 恢复链外的早期 checkpoint 也可纳入同一验证子集，便于观察训练初段。
if [[ -n "${EARLY_CHECKPOINT:-}" ]]; then
  EARLY_STEP="${EARLY_STEP:-${EARLY_CHECKPOINT##*-}}"
  EARLY_RESULT="${OUTPUT_ROOT}/${EVAL_PREFIX}_checkpoint_${EARLY_STEP}_evaluation.json"
  python "${ROOT}/course/25_agent_r1_news/evaluate_agent.py" \
    --adapter "${EARLY_CHECKPOINT}" \
    --dataset "${DATASET}" \
    --maximum-samples "${EVAL_SAMPLES}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --output "${EARLY_RESULT}"
  COMPARE_FILES+=("${EARLY_RESULT}")
  COMPARE_LABELS+=("GRPO-${EARLY_STEP}")
fi

RUN_DIR="${RUN_DIR}" \
EVAL_STEPS="${EVAL_STEPS}" \
EVAL_PREFIX="${EVAL_PREFIX}" \
EVAL_DATASET="${DATASET}" \
EVAL_SAMPLES="${EVAL_SAMPLES}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
EVAL_OUTPUT_ROOT="${OUTPUT_ROOT}" \
bash "${ROOT}/course/25_agent_r1_news/evaluate_checkpoints.sh"

for step in ${EVAL_STEPS}; do
  result="${OUTPUT_ROOT}/${EVAL_PREFIX}_checkpoint_${step}_evaluation.json"
  if [[ ! -f "${result}" ]]; then
    echo "缺少指定阶段的评测结果：${result}" >&2
    exit 1
  fi
  COMPARE_FILES+=("${result}")
  COMPARE_LABELS+=("GRPO-${step}")
done

python "${ROOT}/course/25_agent_r1_news/compare_evaluations.py" \
  "${COMPARE_FILES[@]}" \
  --labels "${COMPARE_LABELS[@]}" \
  --output "${OUTPUT_ROOT}/${EVAL_PREFIX}_checkpoint_comparison.md"

# 分段格式是空格分隔的“日志路径:起始步:结束步”；路径中不要包含空格。
if [[ -n "${GRPO_SEGMENTS:-}" ]]; then
  read -r -a segments <<< "${GRPO_SEGMENTS}"
  summary_args=()
  for segment in "${segments[@]}"; do
    summary_args+=(--segment "${segment}")
  done
  python "${ROOT}/course/25_agent_r1_news/summarize_resumed_grpo.py" \
    "${summary_args[@]}" \
    --window "${GRPO_SUMMARY_WINDOW:-100}" \
    > "${OUTPUT_ROOT}/${EVAL_PREFIX}_training_summary.json"
fi

echo "正式评测已完成：${OUTPUT_ROOT}/${EVAL_PREFIX}_checkpoint_comparison.md"
