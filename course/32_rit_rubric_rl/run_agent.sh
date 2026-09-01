#!/usr/bin/env bash

# 可断点执行极简安全审核 Agent 的数据、训练和隔离测试全链路。
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/agent_common.sh"

START_STAGE="${AGENT_START_STAGE:-1}"
END_STAGE="${AGENT_END_STAGE:-8}"
EVAL_SAMPLES="${AGENT_EVAL_SAMPLES:-0}"
EVAL_BATCH="${AGENT_EVAL_BATCH:-16}"
EVAL_TOKENS="${AGENT_EVAL_TOKENS:-384}"

run_stage() {
  local stage="$1"
  [[ "${stage}" -ge "${START_STAGE}" && "${stage}" -le "${END_STAGE}" ]]
}

evaluate_checkpoint() {
  local checkpoint="$1"
  local output="$2"
  shift 2
  python "${AGENT_DIR}/evaluate_agent.py" \
    --adapter "${checkpoint}" \
    --maximum-samples "${EVAL_SAMPLES}" \
    --batch-size "${EVAL_BATCH}" \
    --max-new-tokens "${EVAL_TOKENS}" \
    --output "${output}" \
    "$@"
}

if run_stage 1; then
  python "${AGENT_DIR}/prepare_agent_data.py"
  python "${AGENT_DIR}/test_agent.py"
  python "${AGENT_DIR}/audit_agent_lengths.py"
fi

if run_stage 2; then
  bash "${AGENT_DIR}/train_agent_sft.sh"
fi

SFT_CHECKPOINT=""
if run_stage 3 || run_stage 4 || run_stage 5 || run_stage 7; then
  SFT_CHECKPOINT="${SFT_ADAPTER:-$(rit_agent_latest_checkpoint "${RIT_AGENT_OUTPUT}/sft")}"
fi

if run_stage 3; then
  evaluate_checkpoint \
    "${SFT_CHECKPOINT}" \
    "${RIT_AGENT_OUTPUT}/sft_test_with_memory.json"
fi

if run_stage 4; then
  evaluate_checkpoint \
    "${SFT_CHECKPOINT}" \
    "${RIT_AGENT_OUTPUT}/sft_test_without_memory.json" \
    --disable-memory
fi

if run_stage 5; then
  SFT_ADAPTER="${SFT_CHECKPOINT}" bash "${AGENT_DIR}/train_agent_orm.sh"
fi

if run_stage 6; then
  ORM_CHECKPOINT="${ORM_ADAPTER:-$(rit_agent_latest_checkpoint "${RIT_AGENT_OUTPUT}/orm_grpo")}"
  evaluate_checkpoint \
    "${ORM_CHECKPOINT}" \
    "${RIT_AGENT_OUTPUT}/orm_test_with_memory.json"
fi

if run_stage 7; then
  SFT_ADAPTER="${SFT_CHECKPOINT}" bash "${AGENT_DIR}/train_agent_rit.sh"
fi

if run_stage 8; then
  RIT_CHECKPOINT="${RIT_ADAPTER:-$(rit_agent_latest_checkpoint "${RIT_AGENT_OUTPUT}/rit_grpo")}"
  evaluate_checkpoint \
    "${RIT_CHECKPOINT}" \
    "${RIT_AGENT_OUTPUT}/rit_test_with_memory.json"
fi
