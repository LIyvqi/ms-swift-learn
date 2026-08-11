#!/usr/bin/env bash

# 顺序验证第 01～04 课的七条多模态训练链路；任一步失败都会立即停止。
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
source ./activate.sh

STATUS_DIR="${PROJECT_ROOT}/outputs/multimodal_course_status"
mkdir -p "${STATUS_DIR}"
printf '开始时间：%s\n' "$(date --iso-8601=seconds)" >"${STATUS_DIR}/running.txt"
rm -f "${STATUS_DIR}/done.txt" "${STATUS_DIR}/failed.txt"
: >"${STATUS_DIR}/steps.log"

记录步骤() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$1" | tee -a "${STATUS_DIR}/steps.log"
}

失败处理() {
  local code=$?
  printf '失败时间：%s\n退出码：%s\n' "$(date --iso-8601=seconds)" "${code}" >"${STATUS_DIR}/failed.txt"
  记录步骤 "多模态链路失败，退出码 ${code}"
  exit "${code}"
}
trap 失败处理 ERR

记录步骤 "校验数据和自定义奖励"
python tools/validate_multimodal_200.py
python tools/validate_multimodal_template.py
python course/03_grpo/test_multimodal_rewards.py

记录步骤 "01 Direct 多模态 LoRA 冒烟"
SMOKE=1 STYLE=direct MM_SFT_BATCH=2 \
  bash course/01_lora_sft/train_multimodal.sh

记录步骤 "01 CoT 多模态 LoRA 冒烟"
SMOKE=1 STYLE=cot MM_SFT_BATCH=2 \
  bash course/01_lora_sft/train_multimodal.sh

记录步骤 "02 mixed 多模态全参 SFT 冒烟"
SMOKE=1 STYLE=mixed MM_SFT_BATCH=2 \
  bash course/02_full_sft/train_multimodal.sh

记录步骤 "03 Direct 多模态 GRPO 冒烟"
SMOKE=1 STYLE=direct RL_BATCH=2 NUM_GENERATIONS=2 \
  bash course/03_grpo/train_multimodal.sh

记录步骤 "03 显式 CoT 多模态 GRPO 冒烟"
SMOKE=1 STYLE=cot RL_BATCH=2 NUM_GENERATIONS=2 MAX_COMPLETION_LENGTH=1024 \
  bash course/03_grpo/train_multimodal.sh

记录步骤 "04 Direct 多模态 OPD 冒烟"
SMOKE=1 STYLE=direct RL_BATCH=2 \
  bash course/04_opd/train_multimodal.sh

记录步骤 "04 显式 CoT 多模态 OPD 冒烟"
SMOKE=1 STYLE=cot RL_BATCH=2 MAX_COMPLETION_LENGTH=1024 \
  bash course/04_opd/train_multimodal.sh

printf '完成时间：%s\n' "$(date --iso-8601=seconds)" >"${STATUS_DIR}/done.txt"
rm -f "${STATUS_DIR}/running.txt"
记录步骤 "第 01～04 课全部多模态冒烟完成"
