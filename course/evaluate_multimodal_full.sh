#!/usr/bin/env bash

# 在固定 40 条验证集上比较 Base、SFT、GRPO 与 OPD 的真实多模态生成。
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source ./activate.sh

MODEL_BASE="${ROOT}/models/Qwen3.5-0.8B-Base"
DATA_ROOT="${ROOT}/datasets/multimodal_200"
RESULT_ROOT="${ROOT}/outputs/multimodal_full_evaluation"
mkdir -p "${RESULT_ROOT}"

最新检查点() {
  local output_pattern="$1"
  local checkpoint
  checkpoint="$({ find "${ROOT}/outputs" -type d \
    -path "${output_pattern}" -printf '%T@ %p\n' 2>/dev/null || true; } \
    | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "${checkpoint}" ]]; then
    echo "找不到待评测检查点：${output_pattern}" >&2
    return 1
  fi
  printf '%s\n' "${checkpoint}"
}

推理并评分() {
  local name="$1"
  local style="$2"
  shift 2
  local result="${RESULT_ROOT}/${name}.jsonl"
  local summary="${RESULT_ROOT}/${name}.json"
  local dataset="${DATA_ROOT}/${style}_val.jsonl"
  local enable_thinking=false
  local max_tokens=256
  if [[ "${style}" == "cot" ]]; then
    enable_thinking=true
    max_tokens="${MM_EVAL_COT_TOKENS:-2048}"
  fi

  # 只有 40 条固定样本，完整结果存在时跳过重复生成，但仍重新计算评分。
  if [[ ! -f "${result}" || "$(wc -l <"${result}")" -ne 40 ]]; then
    rm -f "${result}"
    swift infer \
      "$@" \
      --val_dataset "${dataset}" \
      --infer_backend vllm \
      --stream false \
      --temperature 0 \
      --enable_thinking "${enable_thinking}" \
      --add_non_thinking_prefix false \
      --max_new_tokens "${max_tokens}" \
      --max_pixels "${MAX_PIXELS:-1048576}" \
      --vllm_max_model_len "${MM_EVAL_MODEL_LEN:-4096}" \
      --vllm_limit_mm_per_prompt '{"image":1,"video":0}' \
      --vllm_mm_processor_cache_gb "${MM_PROCESSOR_CACHE_GB:-2}" \
      --vllm_enforce_eager true \
      --vllm_gpu_memory_utilization "${MM_EVAL_VLLM_MEMORY:-0.55}" \
      --result_path "${result}"
  fi
  python course/score_multimodal.py "${result}" \
    --reference "${dataset}" \
    --output "${summary}" >/dev/null
}

DIRECT_TEACHER="$(最新检查点 '*/01_lora_multimodal_direct_full_e*/v*/checkpoint-*')"
COT_TEACHER="$(最新检查点 '*/01_lora_multimodal_cot_full_e*/v*/checkpoint-*')"
FULL_STUDENT="$(最新检查点 '*/02_full_sft_multimodal_mixed_full_e*/v*/checkpoint-*')"
DIRECT_GRPO="$(最新检查点 '*/03_grpo_multimodal_direct_full_*step/v*/checkpoint-*')"
COT_GRPO="$(最新检查点 '*/03_grpo_multimodal_cot_full_*step/v*/checkpoint-*')"
DIRECT_OPD="$(最新检查点 '*/04_opd_multimodal_direct_full_*step/v*/checkpoint-*')"
COT_OPD="$(最新检查点 '*/04_opd_multimodal_cot_full_*step/v*/checkpoint-*')"

推理并评分 base_direct direct --model "${MODEL_BASE}"
推理并评分 base_cot cot --model "${MODEL_BASE}"
推理并评分 lora_direct direct --adapters "${DIRECT_TEACHER}"
推理并评分 lora_cot cot --adapters "${COT_TEACHER}"
推理并评分 full_student_direct direct --model "${FULL_STUDENT}"
推理并评分 full_student_cot cot --model "${FULL_STUDENT}"
推理并评分 grpo_direct direct --adapters "${DIRECT_GRPO}"
推理并评分 grpo_cot cot --adapters "${COT_GRPO}"
推理并评分 opd_direct direct --adapters "${DIRECT_OPD}"
推理并评分 opd_cot cot --adapters "${COT_OPD}"

python course/compare_multimodal.py \
  "${RESULT_ROOT}/base_direct.json" \
  "${RESULT_ROOT}/base_cot.json" \
  "${RESULT_ROOT}/lora_direct.json" \
  "${RESULT_ROOT}/lora_cot.json" \
  "${RESULT_ROOT}/full_student_direct.json" \
  "${RESULT_ROOT}/full_student_cot.json" \
  "${RESULT_ROOT}/grpo_direct.json" \
  "${RESULT_ROOT}/grpo_cot.json" \
  "${RESULT_ROOT}/opd_direct.json" \
  "${RESULT_ROOT}/opd_cot.json" \
  --labels Base-Direct Base-CoT LoRA-Direct LoRA-CoT FullSFT-Direct FullSFT-CoT GRPO-Direct GRPO-CoT OPD-Direct OPD-CoT \
  --output "${RESULT_ROOT}/COMPARISON.md"

echo "多模态固定验证集真实生成评测全部完成。"
