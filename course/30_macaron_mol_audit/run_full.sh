#!/usr/bin/env bash

# 从数据审计开始，完成六个 LoRA 训练、四种 RAG 消融和结果导出。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/course/30_macaron_mol_audit/common.sh"
cd "${ROOT}"

START_STAGE="${START_STAGE:-1}"
if [[ ! "${START_STAGE}" =~ ^[1-4]$ ]]; then
  echo "START_STAGE 必须是 1～4" >&2
  exit 2
fi

STATUS_DIR="${MACARON_OUTPUT}/status"
mkdir -p "${STATUS_DIR}"
if [[ "${START_STAGE}" == "1" ]]; then
  : >"${STATUS_DIR}/steps.log"
else
  touch "${STATUS_DIR}/steps.log"
fi
rm -f "${STATUS_DIR}/done.txt" "${STATUS_DIR}/failed.txt"
printf '开始时间：%s\n' "$(date --iso-8601=seconds)" >"${STATUS_DIR}/running.txt"

记录步骤() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$1" | tee -a "${STATUS_DIR}/steps.log"
}

清理监控() {
  if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
  fi
}

失败处理() {
  local code=$?
  printf '失败时间：%s\n退出码：%s\n' "$(date --iso-8601=seconds)" "${code}" >"${STATUS_DIR}/failed.txt"
  rm -f "${STATUS_DIR}/running.txt"
  记录步骤 "流水线失败，退出码 ${code}"
  exit "${code}"
}
trap 失败处理 ERR
trap 清理监控 EXIT

MONITOR_ARGUMENTS=(
  --pid "$$"
  --output "${STATUS_DIR}/gpu_samples.jsonl"
  --interval "${GPU_SAMPLE_INTERVAL:-1}"
)
if [[ "${START_STAGE}" != "1" && -s "${STATUS_DIR}/gpu_samples.jsonl" ]]; then
  MONITOR_ARGUMENTS+=(--append)
fi
python tools/monitor_rocm.py "${MONITOR_ARGUMENTS[@]}" &
GPU_MONITOR_PID=$!

if [[ "${START_STAGE}" -le 1 ]]; then
  记录步骤 "准备并审计 2000 条分层数据、规则库、案例库和检索上下文"
  python "${MACARON_DIR}/prepare_data.py"
  PYTHONPATH="${MACARON_DIR}:${PYTHONPATH:-}" python -m pytest -q "${MACARON_DIR}/test_macaron.py"
  PYTHONPATH="${MACARON_DIR}:${PYTHONPATH:-}" python "${MACARON_DIR}/audit_data.py"
fi

if [[ "${START_STAGE}" -le 2 ]]; then
  记录步骤 "训练单体基线、L0 路由和 L1～L4 四个专业 LoRA"
  bash "${MACARON_DIR}/train_all.sh"
fi

if [[ "${START_STAGE}" -le 3 ]]; then
  记录步骤 "执行 none/rules/cases/full 四种真实生成消融"
  bash "${MACARON_DIR}/evaluate.sh"
fi

记录步骤 "汇总结果与物理显存"
清理监控
GPU_MONITOR_PID=""
python "${MACARON_DIR}/summarize_gpu.py" \
  --samples "${STATUS_DIR}/gpu_samples.jsonl" \
  --steps "${STATUS_DIR}/steps.log" \
  --output-json "${STATUS_DIR}/gpu_summary.json" \
  --output-md "${STATUS_DIR}/gpu_summary.md"
python "${MACARON_DIR}/export_results.py"

printf '完成时间：%s\n' "$(date --iso-8601=seconds)" >"${STATUS_DIR}/done.txt"
rm -f "${STATUS_DIR}/running.txt"
记录步骤 "第 30 课全部完成"
