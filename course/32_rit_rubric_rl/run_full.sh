#!/usr/bin/env bash

# 一次运行数据审计、显式 RiT 主实验和可选的短结构化消融。
set -euo pipefail
RIT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${RIT_DIR}/common.sh"

python "${RIT_DIR}/prepare_data.py"
python "${RIT_DIR}/test_rit.py"
python "${RIT_DIR}/audit_lengths.py"
python "${RIT_DIR}/audit_reward_design.py"

bash "${RIT_DIR}/train_sft.sh"
SFT_ADAPTER="$(rit_latest_checkpoint "${RIT_OUTPUT}/sft_format")"
python "${RIT_DIR}/evaluate_model.py" \
  --adapter "${SFT_ADAPTER}" \
  --output "${RIT_OUTPUT}/sft_test_evaluation.json"

SFT_ADAPTER="${SFT_ADAPTER}" RL_STEPS="${RL_STEPS:-30}" \
  bash "${RIT_DIR}/train_orm.sh"
ORM_ADAPTER="$(rit_latest_checkpoint "${RIT_OUTPUT}/orm_grpo")"
python "${RIT_DIR}/evaluate_model.py" \
  --adapter "${ORM_ADAPTER}" \
  --output "${RIT_OUTPUT}/orm_test_evaluation.json"

SFT_ADAPTER="${SFT_ADAPTER}" RL_STEPS="${RL_STEPS:-30}" \
  bash "${RIT_DIR}/train_rit.sh"
RIT_ADAPTER="$(rit_latest_checkpoint "${RIT_OUTPUT}/rit_grpo")"
python "${RIT_DIR}/evaluate_model.py" \
  --adapter "${RIT_ADAPTER}" \
  --output "${RIT_OUTPUT}/rit_test_evaluation.json"

if [[ "${RUN_STRUCTURED:-1}" == "1" ]]; then
  bash "${RIT_DIR}/train_structured_sft.sh"
  STRUCTURED_SFT_ADAPTER="$(rit_latest_checkpoint "${RIT_OUTPUT}/structured_sft")"
  python "${RIT_DIR}/evaluate_structured_model.py" \
    --adapter "${STRUCTURED_SFT_ADAPTER}" \
    --output "${RIT_OUTPUT}/structured_sft_test_evaluation.json"

  SFT_ADAPTER="${STRUCTURED_SFT_ADAPTER}" RL_STEPS="${STRUCTURED_RL_STEPS:-30}" \
    bash "${RIT_DIR}/train_structured_orm.sh"
  STRUCTURED_ORM_ADAPTER="$(rit_latest_checkpoint "${RIT_OUTPUT}/structured_orm_grpo")"
  python "${RIT_DIR}/evaluate_structured_model.py" \
    --adapter "${STRUCTURED_ORM_ADAPTER}" \
    --output "${RIT_OUTPUT}/structured_orm_test_evaluation.json"

  SFT_ADAPTER="${STRUCTURED_SFT_ADAPTER}" RL_STEPS="${STRUCTURED_RL_STEPS:-30}" \
    bash "${RIT_DIR}/train_structured_rit.sh"
  STRUCTURED_RIT_ADAPTER="$(rit_latest_checkpoint "${RIT_OUTPUT}/structured_rit_grpo")"
  python "${RIT_DIR}/evaluate_structured_model.py" \
    --adapter "${STRUCTURED_RIT_ADAPTER}" \
    --output "${RIT_OUTPUT}/structured_rit_test_evaluation.json"
fi

echo "第 32 课完整实验结束，结果位于：${RIT_OUTPUT}"
