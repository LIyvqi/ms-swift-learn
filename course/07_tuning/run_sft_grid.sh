#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
mkdir -p outputs/tuning_logs

run_once() {
  local output_root="$1"
  local log_file="$2"
  shift 2
  if find "${output_root}" -type d -name 'checkpoint-*' -print -quit 2>/dev/null | grep -q .; then
    echo "已有检查点，跳过：${output_root}"
    return
  fi
  echo "开始训练：${output_root}"
  "$@" >"${log_file}" 2>&1
  echo "训练完成：${output_root}"
}

for style in cot direct; do
  for epochs in 1 2 3; do
    tag="tune_e${epochs}_lr1e4"
    run_once "outputs/01_lora_${style}_${tag}" "outputs/tuning_logs/01_lora_${style}_${tag}.log" \
      env EPOCHS="${epochs}" RUN_TAG="${tag}" LEARNING_RATE=1e-4 STYLE="${style}" \
      bash course/01_lora_sft/train.sh
  done
  tag="tune_e3_lr5e5"
  run_once "outputs/01_lora_${style}_${tag}" "outputs/tuning_logs/01_lora_${style}_${tag}.log" \
    env EPOCHS=3 RUN_TAG="${tag}" LEARNING_RATE=5e-5 STYLE="${style}" \
    bash course/01_lora_sft/train.sh
done

for epochs in 1 2 3; do
  tag="tune_e${epochs}_lr1e5"
  run_once "outputs/02_full_sft_mixed_${tag}" "outputs/tuning_logs/02_full_sft_${tag}.log" \
    env EPOCHS="${epochs}" RUN_TAG="${tag}" LEARNING_RATE=1e-5 \
    bash course/02_full_sft/train.sh
done
tag="tune_e3_lr5e6"
run_once "outputs/02_full_sft_mixed_${tag}" "outputs/tuning_logs/02_full_sft_${tag}.log" \
  env EPOCHS=3 RUN_TAG="${tag}" LEARNING_RATE=5e-6 \
  bash course/02_full_sft/train.sh

echo "监督微调参数矩阵全部完成"
