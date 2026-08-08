#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
mkdir -p outputs/tuning_logs

checkpoint_from() {
  local root="$1"
  find "${root}" -type d -name 'checkpoint-*' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

run_once() {
  local output_root="$1"
  local log_file="$2"
  shift 2
  if find "${output_root}" -type d -name 'checkpoint-*' -print -quit 2>/dev/null | grep -q .; then
    echo "已有检查点，跳过：${output_root}"
    return
  fi
  echo "开始补充训练：${output_root}"
  "$@" >"${log_file}" 2>&1
  echo "补充训练完成：${output_root}"
}

BEST_STUDENT="$(checkpoint_from outputs/02_full_sft_mixed_tune_e3_lr5e6)"
BEST_COT_TEACHER="$(checkpoint_from outputs/01_lora_cot_tune_e3_lr1e4)"
BEST_DIRECT_TEACHER="$(checkpoint_from outputs/01_lora_direct_tune_e1_lr1e4)"

tag="tune_100_lr5e6_kl02"
run_once "outputs/05_mopd_${tag}" "outputs/tuning_logs/05_mopd_${tag}.log" \
  env STEPS=100 RUN_TAG="${tag}" STUDENT="${BEST_STUDENT}" \
  COT_TEACHER_ADAPTER="${BEST_COT_TEACHER}" DIRECT_TEACHER_ADAPTER="${BEST_DIRECT_TEACHER}" \
  LEARNING_RATE=5e-6 TEACHER_KL_COEF=0.2 MAX_GRAD_NORM=0.5 MAX_COMPLETION_LENGTH=192 \
  bash course/05_mopd/run.sh

for style in cot direct; do
  if [[ "${style}" == "cot" ]]; then
    teacher="${BEST_COT_TEACHER}"
  else
    teacher="${BEST_DIRECT_TEACHER}"
  fi
  for beta in 0 0.5; do
    beta_tag="${beta/./p}"
    tag="tune_e2_lr2e5_b16_beta${beta_tag}"
    run_once "outputs/06_offline_gkd_${style}_${tag}" "outputs/tuning_logs/06_gkd_${style}_${tag}.log" \
      env EPOCHS=2 RUN_TAG="${tag}" STYLE="${style}" STUDENT="${BEST_STUDENT}" \
      TEACHER_ADAPTER="${teacher}" LEARNING_RATE=2e-5 GKD_BETA="${beta}" RL_BATCH=16 \
      MAX_GRAD_NORM=0.5 SAVE_TOTAL_LIMIT=2 bash course/06_offline_gkd/train.sh
  done
done

echo "轮次对照补充实验全部完成"
