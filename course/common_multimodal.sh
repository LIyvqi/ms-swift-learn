#!/usr/bin/env bash

# 01～04 课程共享的多模态路径、数据视图和前置检查点查找函数。
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MM_DATA_ROOT="${PROJECT_ROOT}/datasets/multimodal_200"
PLUGIN_MULTIMODAL="${COURSE_DIR}/plugins/multimodal_rewards.py"

multimodal_dataset_path() {
  local view="$1"
  local split="${2:-train}"
  if [[ "${SMOKE:-0}" == "1" ]]; then
    split="smoke"
  fi
  printf '%s/%s_%s.jsonl\n' "${MM_DATA_ROOT}" "${view}" "${split}"
}

latest_multimodal_student() {
  local checkpoint
  checkpoint="$({ find "${PROJECT_ROOT}/outputs" -type d \
    -path '*/02_full_sft_multimodal*/v*/checkpoint-*' -printf '%T@ %p\n' 2>/dev/null || true; } \
    | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "${checkpoint}" ]]; then
    echo "缺少多模态全参 SFT 学生，请先运行 course/02_full_sft/train_multimodal.sh" >&2
    return 1
  fi
  printf '%s\n' "${checkpoint}"
}

latest_multimodal_teacher() {
  local style="$1"
  local checkpoint
  checkpoint="$({ find "${PROJECT_ROOT}/outputs" -type d \
    -path "*/01_lora_multimodal_${style}*/v*/checkpoint-*" -printf '%T@ %p\n' 2>/dev/null || true; } \
    | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "${checkpoint}" ]]; then
    echo "缺少 ${style} 多模态 LoRA 教师，请先运行第 01 课对应风格" >&2
    return 1
  fi
  printf '%s\n' "${checkpoint}"
}
