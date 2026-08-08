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

checkpoint_named() {
  local root="$1"
  local name="$2"
  find "${root}" -type d -name "${name}" -print -quit
}

infer_adapter() {
  local name="$1"
  local adapter="$2"
  local dataset="$3"
  local result="outputs/tuning_eval/${name}.jsonl"
  if [[ -f "${result}" && "$(wc -l <"${result}")" -eq 100 ]]; then
    echo "已有完整评测，跳过生成：${name}"
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

infer_adapter grpo_cot_tune_200 \
  "$(checkpoint_from outputs/03_grpo_cot_tune_200_lr5e6_g4)" datasets/gsm8k_1k/cot_val.jsonl
infer_adapter grpo_direct_tune_200 \
  "$(checkpoint_from outputs/03_grpo_direct_tune_200_lr5e6_g4)" datasets/gsm8k_1k/direct_val.jsonl
infer_adapter opd_cot_tune_200 \
  "$(checkpoint_from outputs/04_opd_cot_tune_200_lr5e6_kl02)" datasets/gsm8k_1k/cot_val.jsonl
infer_adapter opd_direct_tune_200 \
  "$(checkpoint_from outputs/04_opd_direct_tune_200_lr5e6_kl02)" datasets/gsm8k_1k/direct_val.jsonl
infer_adapter mopd_tune_200 \
  "$(checkpoint_from outputs/05_mopd_tune_200_lr5e6_kl02)" datasets/gsm8k_1k/mixed_val.jsonl
infer_adapter mopd_tune_100 \
  "$(checkpoint_from outputs/05_mopd_tune_100_lr5e6_kl02)" datasets/gsm8k_1k/mixed_val.jsonl
infer_adapter gkd_cot_beta0 \
  "$(checkpoint_from outputs/06_offline_gkd_cot_tune_e1_lr1e5_beta0)" datasets/gsm8k_1k/cot_val.jsonl
infer_adapter gkd_cot_beta05 \
  "$(checkpoint_from outputs/06_offline_gkd_cot_tune_e1_lr1e5_beta0p5)" datasets/gsm8k_1k/cot_val.jsonl
infer_adapter gkd_direct_beta0 \
  "$(checkpoint_from outputs/06_offline_gkd_direct_tune_e1_lr1e5_beta0)" datasets/gsm8k_1k/direct_val.jsonl
infer_adapter gkd_direct_beta05 \
  "$(checkpoint_from outputs/06_offline_gkd_direct_tune_e1_lr1e5_beta0p5)" datasets/gsm8k_1k/direct_val.jsonl

for style in cot direct; do
  for beta_tag in 0 0p5; do
    root="outputs/06_offline_gkd_${style}_tune_e2_lr2e5_b16_beta${beta_tag}"
    infer_adapter "gkd_${style}_b16_beta${beta_tag}_e1" \
      "$(checkpoint_named "${root}" checkpoint-56)" "datasets/gsm8k_1k/${style}_val.jsonl"
    infer_adapter "gkd_${style}_b16_beta${beta_tag}_e2" \
      "$(checkpoint_named "${root}" checkpoint-112)" "datasets/gsm8k_1k/${style}_val.jsonl"
  done
done

echo "强化学习与蒸馏模型的生成评测全部完成"
