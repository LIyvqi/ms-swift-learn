#!/usr/bin/env bash

# 从独立库构建开始，依次执行审计、两阶段 SFT、GRPO 和真实 Agent 评测。
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${PROJECT_ROOT}"

python "${HIERARCHICAL_DIR}/prepare_data.py"
python -m unittest "${HIERARCHICAL_DIR}/test_hierarchical_agent.py" -v
python "${HIERARCHICAL_DIR}/evaluate_pipeline.py"
python "${HIERARCHICAL_DIR}/evaluate_organization_options.py"
python "${HIERARCHICAL_DIR}/analyze_change_impact.py" \
  --rule-id "${IMPACT_RULE_ID:-BT-001}" \
  --output "${HIERARCHICAL_OUTPUT}/change_impact.md"
python "${HIERARCHICAL_DIR}/compile_experience_wiki.py"
python "${HIERARCHICAL_DIR}/audit_lengths.py" \
  "${HIERARCHICAL_DATA}/sft_train.jsonl" \
  --model "${HIERARCHICAL_MODEL}" \
  --limit "${MAX_LENGTH:-6144}"
python "${HIERARCHICAL_DIR}/audit_supervision.py" \
  "${HIERARCHICAL_DATA}/sft_train.jsonl" \
  --model "${HIERARCHICAL_MODEL}" \
  --max-length "${MAX_LENGTH:-6144}"

bash "${HIERARCHICAL_DIR}/train_sft.sh"
BASE_SFT_CHECKPOINT="$(hierarchical_latest_checkpoint "${SFT_OUTPUT:-${HIERARCHICAL_OUTPUT}/sft_${SFT_EPOCHS:-2}epoch}")"
SFT_ADAPTER="${BASE_SFT_CHECKPOINT}" bash "${HIERARCHICAL_DIR}/train_state_sft.sh"
SFT_CHECKPOINT="$(hierarchical_latest_checkpoint "${STATE_SFT_OUTPUT:-${HIERARCHICAL_OUTPUT}/sft_state_repair}")"
python "${HIERARCHICAL_DIR}/evaluate_agent.py" \
  --adapter "${SFT_CHECKPOINT}" \
  --maximum-samples "${EVAL_SAMPLES:-200}" \
  --output "${HIERARCHICAL_OUTPUT}/sft_evaluation.json"

SFT_ADAPTER="${SFT_CHECKPOINT}" bash "${HIERARCHICAL_DIR}/train_grpo.sh"
GRPO_CHECKPOINT="$(hierarchical_latest_checkpoint "${GRPO_OUTPUT:-${HIERARCHICAL_OUTPUT}/grpo_${GRPO_EPOCHS:-2}epoch}")"
python "${HIERARCHICAL_DIR}/evaluate_agent.py" \
  --adapter "${GRPO_CHECKPOINT}" \
  --maximum-samples "${EVAL_SAMPLES:-200}" \
  --output "${HIERARCHICAL_OUTPUT}/grpo_evaluation.json"

python "${HIERARCHICAL_DIR}/compile_experience_wiki.py" \
  --evaluation "${HIERARCHICAL_OUTPUT}/sft_evaluation.json" \
  --evaluation "${HIERARCHICAL_OUTPUT}/grpo_evaluation.json"
