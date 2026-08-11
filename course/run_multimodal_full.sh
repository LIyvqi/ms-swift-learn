#!/usr/bin/env bash

# 顺序执行第 01～04 课的七条正式多模态训练链路，并保存可追踪的阶段状态。
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
source ./activate.sh

START_STAGE="${MM_START_STAGE:-1}"
if [[ ! "${START_STAGE}" =~ ^[1-8]$ ]]; then
  echo "MM_START_STAGE 必须是 1～8 的整数" >&2
  exit 2
fi

STATUS_DIR="${PROJECT_ROOT}/outputs/multimodal_full_status"
mkdir -p "${STATUS_DIR}"
printf '开始时间：%s\n' "$(date --iso-8601=seconds)" >"${STATUS_DIR}/running.txt"
rm -f "${STATUS_DIR}/done.txt" "${STATUS_DIR}/failed.txt"
if [[ "${START_STAGE}" -eq 1 ]]; then
  : >"${STATUS_DIR}/steps.log"
else
  touch "${STATUS_DIR}/steps.log"
fi

记录步骤() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$1" | tee -a "${STATUS_DIR}/steps.log"
}

失败处理() {
  local code=$?
  printf '失败时间：%s\n退出码：%s\n' "$(date --iso-8601=seconds)" "${code}" >"${STATUS_DIR}/failed.txt"
  rm -f "${STATUS_DIR}/running.txt"
  记录步骤 "正式多模态链路失败，退出码 ${code}"
  exit "${code}"
}
trap 失败处理 ERR

清理显存监控() {
  if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
  fi
}
trap 清理显存监控 EXIT

# 按秒保留整条流水线的物理显存与 GPU 利用率，阶段边界由 steps.log 对齐。
GPU_MONITOR_APPEND=()
if [[ "${START_STAGE}" -gt 1 ]]; then
  GPU_MONITOR_APPEND=(--append)
fi
python tools/monitor_rocm.py \
  --pid "$$" \
  --output "${STATUS_DIR}/gpu_samples.jsonl" \
  --interval "${MM_GPU_SAMPLE_INTERVAL:-1}" \
  "${GPU_MONITOR_APPEND[@]}" &
GPU_MONITOR_PID=$!

SFT_EPOCHS="${MM_FULL_SFT_EPOCHS:-3}"
RL_STEPS="${MM_FULL_RL_STEPS:-100}"
# batch=24 首轮能训练，但验证/保存后进入第二轮时会因显存峰值叠加 OOM。
# batch=16 降低了单步峰值，但多种动态形状在轮次边界叠加时仍会 OOM。
# Direct 正式值使用 12；CoT 的监督序列更长，batch=12 在第二轮最长批次仍会 OOM，
# 因此 CoT LoRA 与含 CoT 的 mixed 全参训练使用 8，并逐批释放验证 logits 和缓存。
LORA_BATCH="${MM_LORA_BATCH:-12}"
COT_LORA_BATCH="${MM_COT_LORA_BATCH:-8}"
FULL_BATCH="${MM_FULL_BATCH:-8}"
# 验证会物化全词表 logits，不能盲目继承训练 batch。
LORA_EVAL_BATCH="${MM_LORA_EVAL_BATCH:-8}"
FULL_EVAL_BATCH="${MM_FULL_EVAL_BATCH:-8}"
RL_BATCH_SIZE="${MM_RL_BATCH:-6}"
# CoT 的长度分布有长尾：batch=6 遇到单条 2048-token rollout 时，
# 外部物理显存实测达到 183.98/191.69 GiB。保留 12 条集中生成，
# 只把反向 batch 降为 3，在不明显牺牲生成吞吐的情况下留出动态余量。
COT_RL_BATCH_SIZE="${MM_COT_RL_BATCH:-3}"
RL_GENERATIONS="${MM_NUM_GENERATIONS:-3}"
RL_GENERATION_BATCH="${MM_GENERATION_BATCH:-12}"
DIRECT_GRPO_VLLM_MEMORY="${MM_DIRECT_GRPO_VLLM_MEMORY:-${MM_GRPO_VLLM_MEMORY:-0.55}}"
COT_GRPO_VLLM_MEMORY="${MM_COT_GRPO_VLLM_MEMORY:-${MM_GRPO_VLLM_MEMORY:-0.50}}"
RL_SAVE_STEPS="${MM_RL_SAVE_STEPS:-25}"
# 正式在线训练保留优化器和调度器状态，发生原生进程异常后才能精确恢复。
ONLINE_SAVE_ONLY_MODEL="${MM_ONLINE_SAVE_ONLY_MODEL:-false}"
# 本机 ROCm vLLM 的 CuMemAllocator 在长时间反复休眠/唤醒后会原生 abort。
# 正式配置用较小 CoT batch 与常驻 vLLM 共存，完全绕开该不稳定路径。
ONLINE_SLEEP_LEVEL="${MM_ONLINE_SLEEP_LEVEL:-0}"
DIRECT_OPD_BATCH_SIZE="${MM_DIRECT_OPD_BATCH:-${MM_OPD_BATCH:-6}}"
# CoT OPD 还会执行教师前向，默认沿用更安全的反向 batch；生成仍可集中处理 6 条。
COT_OPD_BATCH_SIZE="${MM_COT_OPD_BATCH:-3}"
COT_OPD_GENERATION_BATCH="${MM_COT_OPD_GENERATION_BATCH:-6}"

if [[ "${START_STAGE}" -gt 1 ]]; then
  记录步骤 "从阶段 ${START_STAGE} 继续正式流水线，保留先前已完成结果"
fi

记录步骤 "校验 200 条数据、视觉模板与全部自定义奖励"
python tools/validate_multimodal_200.py
python tools/validate_multimodal_template.py
python tools/audit_multimodal_lengths.py
python course/03_grpo/test_multimodal_rewards.py

if [[ "${START_STAGE}" -le 1 ]]; then
  记录步骤 "01 Direct 多模态 LoRA SFT：${SFT_EPOCHS} epoch，batch=${LORA_BATCH}"
  EPOCHS="${SFT_EPOCHS}" STYLE=direct RUN_TAG="full_e${SFT_EPOCHS}" \
  MM_SFT_BATCH="${LORA_BATCH}" MM_EVAL_BATCH="${LORA_EVAL_BATCH}" \
    bash course/01_lora_sft/train_multimodal.sh
fi

if [[ "${START_STAGE}" -le 2 ]]; then
  记录步骤 "01 显式 CoT 多模态 LoRA SFT：${SFT_EPOCHS} epoch，batch=${COT_LORA_BATCH}"
  EPOCHS="${SFT_EPOCHS}" STYLE=cot RUN_TAG="full_e${SFT_EPOCHS}" \
  MM_SFT_BATCH="${COT_LORA_BATCH}" MM_EVAL_BATCH="${LORA_EVAL_BATCH}" \
    bash course/01_lora_sft/train_multimodal.sh
fi

if [[ "${START_STAGE}" -le 3 ]]; then
  记录步骤 "02 mixed 多模态语言模型全参数 SFT：${SFT_EPOCHS} epoch，batch=${FULL_BATCH}"
  EPOCHS="${SFT_EPOCHS}" STYLE=mixed RUN_TAG="full_e${SFT_EPOCHS}" \
  MM_SFT_BATCH="${FULL_BATCH}" MM_EVAL_BATCH="${FULL_EVAL_BATCH}" \
    bash course/02_full_sft/train_multimodal.sh
fi

if [[ "${START_STAGE}" -le 4 ]]; then
  记录步骤 "03 Direct 多模态 GRPO：${RL_STEPS} step，batch=${RL_BATCH_SIZE}，组大小=${RL_GENERATIONS}"
  STEPS="${RL_STEPS}" STYLE=direct RUN_TAG="full_${RL_STEPS}step" \
  RL_BATCH="${RL_BATCH_SIZE}" NUM_GENERATIONS="${RL_GENERATIONS}" \
  GENERATION_BATCH="${RL_GENERATION_BATCH}" VLLM_MEMORY="${DIRECT_GRPO_VLLM_MEMORY}" \
  SAVE_STEPS="${RL_SAVE_STEPS}" SAVE_ONLY_MODEL="${ONLINE_SAVE_ONLY_MODEL}" \
  SLEEP_LEVEL="${ONLINE_SLEEP_LEVEL}" \
    bash course/03_grpo/train_multimodal.sh
fi

if [[ "${START_STAGE}" -le 5 ]]; then
  记录步骤 "03 显式 CoT 多模态 GRPO：${RL_STEPS} step，batch=${COT_RL_BATCH_SIZE}，生成batch=${RL_GENERATION_BATCH}，组大小=${RL_GENERATIONS}"
  STEPS="${RL_STEPS}" STYLE=cot RUN_TAG="full_${RL_STEPS}step" \
  RL_BATCH="${COT_RL_BATCH_SIZE}" NUM_GENERATIONS="${RL_GENERATIONS}" \
  GENERATION_BATCH="${RL_GENERATION_BATCH}" VLLM_MEMORY="${COT_GRPO_VLLM_MEMORY}" \
  SAVE_STEPS="${RL_SAVE_STEPS}" SAVE_ONLY_MODEL="${ONLINE_SAVE_ONLY_MODEL}" \
  SLEEP_LEVEL="${ONLINE_SLEEP_LEVEL}" \
  MAX_COMPLETION_LENGTH="${MM_COT_COMPLETION_LENGTH:-2048}" \
    bash course/03_grpo/train_multimodal.sh
fi

if [[ "${START_STAGE}" -le 6 ]]; then
  记录步骤 "04 Direct 多模态 OPD：${RL_STEPS} step，batch=${DIRECT_OPD_BATCH_SIZE}"
  STEPS="${RL_STEPS}" STYLE=direct RUN_TAG="full_${RL_STEPS}step" \
  RL_BATCH="${DIRECT_OPD_BATCH_SIZE}" GENERATION_BATCH="${DIRECT_OPD_BATCH_SIZE}" \
  VLLM_MEMORY="${MM_OPD_VLLM_MEMORY:-0.50}" \
  SAVE_STEPS="${RL_SAVE_STEPS}" SAVE_ONLY_MODEL="${ONLINE_SAVE_ONLY_MODEL}" \
  SLEEP_LEVEL="${ONLINE_SLEEP_LEVEL}" \
    bash course/04_opd/train_multimodal.sh
fi

if [[ "${START_STAGE}" -le 7 ]]; then
  记录步骤 "04 显式 CoT 多模态 OPD：${RL_STEPS} step，batch=${COT_OPD_BATCH_SIZE}，生成batch=${COT_OPD_GENERATION_BATCH}"
  STEPS="${RL_STEPS}" STYLE=cot RUN_TAG="full_${RL_STEPS}step" \
  RL_BATCH="${COT_OPD_BATCH_SIZE}" GENERATION_BATCH="${COT_OPD_GENERATION_BATCH}" \
  VLLM_MEMORY="${MM_OPD_VLLM_MEMORY:-0.50}" \
  SAVE_STEPS="${RL_SAVE_STEPS}" SAVE_ONLY_MODEL="${ONLINE_SAVE_ONLY_MODEL}" \
  SLEEP_LEVEL="${ONLINE_SLEEP_LEVEL}" \
  MAX_COMPLETION_LENGTH="${MM_COT_COMPLETION_LENGTH:-2048}" \
    bash course/04_opd/train_multimodal.sh
fi

if [[ "${START_STAGE}" -le 8 ]]; then
  记录步骤 "在固定 40 条验证集上执行 Base、SFT、GRPO 与 OPD 真实生成对比"
  bash course/evaluate_multimodal_full.sh
fi

记录步骤 "训练与评测完成，汇总各阶段物理显存和 GPU 利用率"
清理显存监控
GPU_MONITOR_PID=""
python tools/summarize_gpu_samples.py \
  --samples "${STATUS_DIR}/gpu_samples.jsonl" \
  --steps "${STATUS_DIR}/steps.log" \
  --output-json "${STATUS_DIR}/gpu_summary.json" \
  --output-md "${STATUS_DIR}/gpu_summary.md"

printf '完成时间：%s\n' "$(date --iso-8601=seconds)" >"${STATUS_DIR}/done.txt"
rm -f "${STATUS_DIR}/running.txt"
记录步骤 "第 01～04 课七条正式多模态训练全部完成"
