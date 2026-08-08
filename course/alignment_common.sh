#!/usr/bin/env bash

set -euo pipefail

ALIGNMENT_COURSE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${ALIGNMENT_COURSE_DIR}/.." && pwd)"
source "${PROJECT_ROOT}/activate.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ALIGNMENT_MODEL="${PROJECT_ROOT}/models/Qwen3.5-0.8B-Base"
ALIGNMENT_DATA="${PROJECT_ROOT}/datasets/alignment_news"

if [[ "${SMOKE:-0}" == "1" ]]; then
  ALIGNMENT_SPLIT="smoke"
  ALIGNMENT_SUFFIX="_smoke"
else
  ALIGNMENT_SPLIT="train"
  ALIGNMENT_SUFFIX=""
fi

latest_alignment_checkpoint() {
  local directories=("$@")
  local checkpoint
  checkpoint="$({ find "${directories[@]}" -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null || true; } \
    | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "${checkpoint}" ]]; then
    echo "找不到前置检查点：${directories[*]}" >&2
    return 1
  fi
  printf '%s\n' "${checkpoint}"
}

alignment_span_args() {
  if [[ "${SMOKE:-0}" == "1" ]]; then
    printf '%s\n' --max_steps 1 --save_steps 1 --eval_strategy no
  elif [[ -n "${STEPS:-}" ]]; then
    if [[ ! "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
      echo "STEPS 必须是正整数" >&2
      return 2
    fi
    printf '%s\n' --max_steps "${STEPS}" --save_steps "${STEPS}" --eval_steps "${STEPS}"
  else
    printf '%s\n' --num_train_epochs "${EPOCHS:-3}" --save_strategy epoch --eval_strategy epoch
  fi
}

alignment_sft_checkpoint() {
  local root="${SFT_OUTPUT:-${PROJECT_ROOT}/outputs/11_sft_dft/sft}"
  if [[ "${SMOKE:-0}" == "1" ]]; then
    root="${SFT_OUTPUT:-${PROJECT_ROOT}/outputs/11_sft_dft/sft_smoke}"
  fi
  if [[ -n "${SFT_ADAPTER:-}" ]]; then
    printf '%s\n' "${SFT_ADAPTER}"
  else
    latest_alignment_checkpoint "${root}"
  fi
}

alignment_sft_merged_model() {
  local merged
  if [[ -n "${SFT_MERGED_MODEL:-}" ]]; then
    printf '%s\n' "${SFT_MERGED_MODEL}"
    return
  fi
  if [[ "${SMOKE:-0}" == "1" ]]; then
    merged="${PROJECT_ROOT}/outputs/11_sft_dft/sft_smoke_merged"
  else
    merged="${PROJECT_ROOT}/models/alignment-news-sft-merged"
  fi
  if [[ ! -f "${merged}/config.json" ]]; then
    SMOKE="${SMOKE:-0}" SFT_MERGED_MODEL="${merged}" \
      bash "${PROJECT_ROOT}/course/11_sft_dft/merge_sft.sh" >&2
  fi
  printf '%s\n' "${merged}"
}

alignment_rm_checkpoint() {
  local root="${RM_OUTPUT:-${PROJECT_ROOT}/outputs/13_reward_model/rm}"
  local enhanced_root="${PROJECT_ROOT}/outputs/13_reward_model/rm_hard_negative_left"
  if [[ "${SMOKE:-0}" == "1" ]]; then
    root="${RM_OUTPUT:-${PROJECT_ROOT}/outputs/13_reward_model/rm_smoke}"
  fi
  if [[ -n "${RM_MODEL:-}" ]]; then
    printf '%s\n' "${RM_MODEL}"
  elif [[ -n "${RM_ADAPTER:-}" ]]; then
    printf '%s\n' "${RM_ADAPTER}"
  elif [[ -z "${RM_OUTPUT:-}" && "${SMOKE:-0}" != "1" && -d "${enhanced_root}" ]]; then
    # 本仓库保留了基础 RM 与左截断困难负例增强 RM 两组实验，默认取两处最新的完整检查点。
    latest_alignment_checkpoint "${root}" "${enhanced_root}"
  else
    latest_alignment_checkpoint "${root}"
  fi
}
