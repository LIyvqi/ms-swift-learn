#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source ./activate.sh
mkdir -p outputs/tuning_eval outputs/tuning_logs

checkpoint_from() {
  local root="$1"
  find "${root}" -type d -name 'checkpoint-*' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

infer_adapter() {
  local name="$1"
  local adapter="$2"
  local dataset="$3"
  local result="outputs/tuning_eval/${name}.jsonl"
  if [[ -f "${result}" && "$(wc -l <"${result}")" -eq 100 ]]; then
    echo "已有完整评测，跳过：${name}"
    python course/07_tuning/score_gsm8k.py "${result}"
    return
  fi
  rm -f "${result}"
  echo "开始生成评测：${name}"
  swift infer \
    --adapters "${adapter}" \
    --val_dataset "${dataset}" \
    --infer_backend vllm \
    --stream false \
    --temperature 0 \
    --max_new_tokens 256 \
    --vllm_max_model_len 1024 \
    --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
    --vllm_mm_processor_cache_gb 0 \
    --vllm_enforce_eager true \
    --vllm_gpu_memory_utilization 0.5 \
    --result_path "${result}" \
    >"outputs/tuning_logs/eval_${name}.log" 2>&1
  python course/07_tuning/score_gsm8k.py "${result}"
}

infer_model() {
  local name="$1"
  local model="$2"
  local dataset="$3"
  local result="outputs/tuning_eval/${name}.jsonl"
  if [[ -f "${result}" && "$(wc -l <"${result}")" -eq 100 ]]; then
    echo "已有完整评测，跳过：${name}"
    python course/07_tuning/score_gsm8k.py "${result}"
    return
  fi
  rm -f "${result}"
  echo "开始生成评测：${name}"
  swift infer \
    --model "${model}" \
    --val_dataset "${dataset}" \
    --infer_backend vllm \
    --stream false \
    --temperature 0 \
    --max_new_tokens 256 \
    --vllm_max_model_len 1024 \
    --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
    --vllm_mm_processor_cache_gb 0 \
    --vllm_enforce_eager true \
    --vllm_gpu_memory_utilization 0.5 \
    --result_path "${result}" \
    >"outputs/tuning_logs/eval_${name}.log" 2>&1
  python course/07_tuning/score_gsm8k.py "${result}"
}

infer_adapter lora_cot_100step \
  "$(checkpoint_from outputs/01_lora_cot_100step)" datasets/gsm8k_1k/cot_val.jsonl
infer_adapter lora_cot_best \
  "$(checkpoint_from outputs/01_lora_cot_tune_e3_lr1e4)" datasets/gsm8k_1k/cot_val.jsonl
infer_adapter lora_cot_e1_lr1e4 \
  "$(checkpoint_from outputs/01_lora_cot_tune_e1_lr1e4)" datasets/gsm8k_1k/cot_val.jsonl
infer_adapter lora_cot_e2_lr1e4 \
  "$(checkpoint_from outputs/01_lora_cot_tune_e2_lr1e4)" datasets/gsm8k_1k/cot_val.jsonl
infer_adapter lora_cot_e3_lr5e5 \
  "$(checkpoint_from outputs/01_lora_cot_tune_e3_lr5e5)" datasets/gsm8k_1k/cot_val.jsonl
infer_adapter lora_direct_100step \
  "$(checkpoint_from outputs/01_lora_direct_100step)" datasets/gsm8k_1k/direct_val.jsonl
infer_adapter lora_direct_best \
  "$(checkpoint_from outputs/01_lora_direct_tune_e3_lr5e5)" datasets/gsm8k_1k/direct_val.jsonl
infer_adapter lora_direct_e1_lr1e4 \
  "$(checkpoint_from outputs/01_lora_direct_tune_e1_lr1e4)" datasets/gsm8k_1k/direct_val.jsonl
infer_adapter lora_direct_e2_lr1e4 \
  "$(checkpoint_from outputs/01_lora_direct_tune_e2_lr1e4)" datasets/gsm8k_1k/direct_val.jsonl
infer_adapter lora_direct_e3_lr1e4 \
  "$(checkpoint_from outputs/01_lora_direct_tune_e3_lr1e4)" datasets/gsm8k_1k/direct_val.jsonl
infer_model full_sft_100step \
  "$(checkpoint_from outputs/02_full_sft_mixed_100step)" datasets/gsm8k_1k/mixed_val.jsonl
infer_model full_sft_best \
  "$(checkpoint_from outputs/02_full_sft_mixed_tune_e2_lr1e5)" datasets/gsm8k_1k/mixed_val.jsonl
infer_model full_sft_e1_lr1e5 \
  "$(checkpoint_from outputs/02_full_sft_mixed_tune_e1_lr1e5)" datasets/gsm8k_1k/mixed_val.jsonl
infer_model full_sft_e3_lr1e5 \
  "$(checkpoint_from outputs/02_full_sft_mixed_tune_e3_lr1e5)" datasets/gsm8k_1k/mixed_val.jsonl
infer_model full_sft_e3_lr5e6 \
  "$(checkpoint_from outputs/02_full_sft_mixed_tune_e3_lr5e6)" datasets/gsm8k_1k/mixed_val.jsonl

echo "生成评测全部完成"
