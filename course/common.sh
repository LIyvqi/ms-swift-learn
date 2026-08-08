#!/usr/bin/env bash

set -euo pipefail

COURSE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${COURSE_DIR}/.." && pwd)"
source "${PROJECT_ROOT}/activate.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_BASE="${PROJECT_ROOT}/models/Qwen3.5-0.8B-Base"
DATA_ROOT="${PROJECT_ROOT}/datasets/gsm8k_1k"
PLUGIN_GSM8K="${COURSE_DIR}/plugins/gsm8k_rewards.py"

if [[ -n "${STEPS:-}" && -n "${EPOCHS:-}" ]]; then
  echo "STEPS 与 EPOCHS 不能同时设置" >&2
  return 2
fi
if [[ "${SMOKE:-0}" == "1" && ( -n "${STEPS:-}" || -n "${EPOCHS:-}" ) ]]; then
  echo "SMOKE 不能与 STEPS 或 EPOCHS 同时设置" >&2
  return 2
fi
if [[ -n "${STEPS:-}" && ! "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEPS 必须是正整数" >&2
  return 2
fi
if [[ -n "${EPOCHS:-}" && ! "${EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCHS 必须是正整数" >&2
  return 2
fi
if [[ -n "${RUN_TAG:-}" && ! "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_TAG 只能包含字母、数字、点、下划线和连字符" >&2
  return 2
fi

if [[ -n "${RUN_TAG:-}" ]]; then
  RUN_SUFFIX="_${RUN_TAG}"
  SFT_BATCH="${SFT_BATCH:-8}"
elif [[ -n "${STEPS:-}" ]]; then
  RUN_SUFFIX="_${STEPS}step"
  SFT_BATCH="${SFT_BATCH:-8}"
elif [[ -n "${EPOCHS:-}" ]]; then
  RUN_SUFFIX="_${EPOCHS}epoch"
  SFT_BATCH="${SFT_BATCH:-8}"
elif [[ "${SMOKE:-0}" == "1" ]]; then
  RUN_SUFFIX="_smoke"
  SFT_BATCH=1
else
  RUN_SUFFIX=""
  SFT_BATCH=8
fi

output_path() {
  printf '%s/outputs/%s%s\n' "${PROJECT_ROOT}" "$1" "${RUN_SUFFIX}"
}

dataset_path() {
  local view="$1"
  local split="${2:-train}"
  if [[ "${SMOKE:-0}" == "1" ]]; then
    split="smoke"
  fi
  printf '%s/%s_%s.jsonl\n' "${DATA_ROOT}" "${view}" "${split}"
}

latest_checkpoint() {
  local root="$1"
  local checkpoint
  checkpoint="$({ find "${root}" -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null || true; } \
    | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "${checkpoint}" ]]; then
    echo "缺少前置 checkpoint：${root}" >&2
    return 1
  fi
  printf '%s\n' "${checkpoint}"
}

training_span_args() {
  if [[ -n "${STEPS:-}" ]]; then
    printf '%s\n' --max_steps "${STEPS}" --save_steps "${STEPS}"
  elif [[ -n "${EPOCHS:-}" ]]; then
    printf '%s\n' --num_train_epochs "${EPOCHS}" --save_strategy epoch --eval_strategy epoch
  elif [[ "${SMOKE:-0}" == "1" ]]; then
    printf '%s\n' --max_steps 1 --save_steps 1
  else
    printf '%s\n' --num_train_epochs 1 --save_strategy epoch
  fi
}
