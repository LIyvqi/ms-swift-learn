#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"
OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${ROOT}/outputs/25_agent_r1_news}"
mkdir -p "${OUTPUT_ROOT}"

SFT_ROOT="${SFT_OUTPUT:-${ROOT}/outputs/25_agent_r1_news/sft_2epoch}"
RUN_DIR="${RUN_DIR:-$({ find "${SFT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null || true; } | sort -nr | head -n 1 | cut -d' ' -f2-)}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "找不到 SFT 运行目录：${SFT_ROOT}" >&2
  exit 1
fi

mapfile -t CHECKPOINTS < <(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "运行目录中没有 checkpoint：${RUN_DIR}" >&2
  exit 1
fi

# 对每个 epoch checkpoint 使用同一批动态环境样本，便于选择轮次。
declare -A EVALUATED_STEPS=()
for checkpoint in "${CHECKPOINTS[@]}"; do
  step="${checkpoint##*-}"
  if [[ -n "${EVAL_STEPS:-}" && " ${EVAL_STEPS} " != *" ${step} "* ]]; then
    continue
  fi
  python "${ROOT}/course/25_agent_r1_news/evaluate_agent.py" \
    --adapter "${checkpoint}" \
    --dataset "${EVAL_DATASET:-${ROOT}/datasets/agent_r1_news/rl_smoke.jsonl}" \
    --maximum-samples "${EVAL_SAMPLES:-12}" \
    --sample-offset "${EVAL_OFFSET:-0}" \
    --batch-size "${EVAL_BATCH_SIZE:-12}" \
    --output "${OUTPUT_ROOT}/${EVAL_PREFIX:-sft}_checkpoint_${step}_evaluation.json"
  EVALUATED_STEPS["${step}"]=1
done

# 明确指定的节点必须在本次运行中真实完成，不能误用目录里遗留的同名 JSON。
if [[ -n "${EVAL_STEPS:-}" ]]; then
  for requested_step in ${EVAL_STEPS}; do
    if [[ -z "${EVALUATED_STEPS[${requested_step}]:-}" ]]; then
      echo "当前运行目录缺少指定 checkpoint-${requested_step}：${RUN_DIR}" >&2
      exit 1
    fi
  done
fi
