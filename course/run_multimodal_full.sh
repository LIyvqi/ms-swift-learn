#!/usr/bin/env bash

# 顺序执行第 01～04 课的七条正式多模态训练链路，并保存可追踪的阶段状态。
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
source ./activate.sh

STATUS_DIR="${PROJECT_ROOT}/outputs/multimodal_full_status"
mkdir -p "${STATUS_DIR}"
printf '开始时间：%s\n' "$(date --iso-8601=seconds)" >"${STATUS_DIR}/running.txt"
rm -f "${STATUS_DIR}/done.txt" "${STATUS_DIR}/failed.txt"

记录步骤() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$1" | tee -a "${STATUS_DIR}/steps.log"
}

失败处理() {
  local code=$?
  printf '失败时间：%s\n退出码：%s\n' "$(date --iso-8601=seconds)" "${code}" >"${STATUS_DIR}/failed.txt"
  记录步骤 "正式多模态链路失败，退出码 ${code}"
  exit "${code}"
}
trap 失败处理 ERR

SFT_EPOCHS="${MM_FULL_SFT_EPOCHS:-3}"
RL_STEPS="${MM_FULL_RL_STEPS:-100}"
LORA_BATCH="${MM_LORA_BATCH:-12}"
FULL_BATCH="${MM_FULL_BATCH:-8}"
RL_BATCH_SIZE="${MM_RL_BATCH:-6}"
RL_GENERATIONS="${MM_NUM_GENERATIONS:-3}"
RL_GENERATION_BATCH="${MM_GENERATION_BATCH:-12}"
OPD_BATCH_SIZE="${MM_OPD_BATCH:-6}"

记录步骤 "校验 200 条数据、视觉模板与全部自定义奖励"
python tools/validate_multimodal_200.py
python tools/validate_multimodal_template.py
python course/03_grpo/test_multimodal_rewards.py

记录步骤 "01 Direct 多模态 LoRA SFT：${SFT_EPOCHS} epoch，batch=${LORA_BATCH}"
EPOCHS="${SFT_EPOCHS}" STYLE=direct RUN_TAG="full_e${SFT_EPOCHS}" \
MM_SFT_BATCH="${LORA_BATCH}" \
  bash course/01_lora_sft/train_multimodal.sh

记录步骤 "01 显式 CoT 多模态 LoRA SFT：${SFT_EPOCHS} epoch，batch=${LORA_BATCH}"
EPOCHS="${SFT_EPOCHS}" STYLE=cot RUN_TAG="full_e${SFT_EPOCHS}" \
MM_SFT_BATCH="${LORA_BATCH}" \
  bash course/01_lora_sft/train_multimodal.sh

记录步骤 "02 mixed 多模态语言模型全参数 SFT：${SFT_EPOCHS} epoch，batch=${FULL_BATCH}"
EPOCHS="${SFT_EPOCHS}" STYLE=mixed RUN_TAG="full_e${SFT_EPOCHS}" \
MM_SFT_BATCH="${FULL_BATCH}" \
  bash course/02_full_sft/train_multimodal.sh

记录步骤 "03 Direct 多模态 GRPO：${RL_STEPS} step，batch=${RL_BATCH_SIZE}，组大小=${RL_GENERATIONS}"
STEPS="${RL_STEPS}" STYLE=direct RUN_TAG="full_${RL_STEPS}step" \
RL_BATCH="${RL_BATCH_SIZE}" NUM_GENERATIONS="${RL_GENERATIONS}" \
GENERATION_BATCH="${RL_GENERATION_BATCH}" VLLM_MEMORY="${MM_GRPO_VLLM_MEMORY:-0.55}" \
  bash course/03_grpo/train_multimodal.sh

记录步骤 "03 显式 CoT 多模态 GRPO：${RL_STEPS} step，batch=${RL_BATCH_SIZE}，组大小=${RL_GENERATIONS}"
STEPS="${RL_STEPS}" STYLE=cot RUN_TAG="full_${RL_STEPS}step" \
RL_BATCH="${RL_BATCH_SIZE}" NUM_GENERATIONS="${RL_GENERATIONS}" \
GENERATION_BATCH="${RL_GENERATION_BATCH}" VLLM_MEMORY="${MM_GRPO_VLLM_MEMORY:-0.55}" \
MAX_COMPLETION_LENGTH="${MM_COT_COMPLETION_LENGTH:-1024}" \
  bash course/03_grpo/train_multimodal.sh

记录步骤 "04 Direct 多模态 OPD：${RL_STEPS} step，batch=${OPD_BATCH_SIZE}"
STEPS="${RL_STEPS}" STYLE=direct RUN_TAG="full_${RL_STEPS}step" \
RL_BATCH="${OPD_BATCH_SIZE}" GENERATION_BATCH="${OPD_BATCH_SIZE}" \
VLLM_MEMORY="${MM_OPD_VLLM_MEMORY:-0.50}" \
  bash course/04_opd/train_multimodal.sh

记录步骤 "04 显式 CoT 多模态 OPD：${RL_STEPS} step，batch=${OPD_BATCH_SIZE}"
STEPS="${RL_STEPS}" STYLE=cot RUN_TAG="full_${RL_STEPS}step" \
RL_BATCH="${OPD_BATCH_SIZE}" GENERATION_BATCH="${OPD_BATCH_SIZE}" \
VLLM_MEMORY="${MM_OPD_VLLM_MEMORY:-0.50}" \
MAX_COMPLETION_LENGTH="${MM_COT_COMPLETION_LENGTH:-1024}" \
  bash course/04_opd/train_multimodal.sh

记录步骤 "在固定 40 条验证集上执行 Base、SFT、GRPO 与 OPD 真实生成对比"
bash course/evaluate_multimodal_full.sh

printf '完成时间：%s\n' "$(date --iso-8601=seconds)" >"${STATUS_DIR}/done.txt"
rm -f "${STATUS_DIR}/running.txt"
记录步骤 "第 01～04 课七条正式多模态训练全部完成"
