#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/activate.sh"

latest_checkpoint() {
  local directory="$1"
  local checkpoint
  checkpoint="$({ find "${directory}" -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null || true; } \
    | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "${checkpoint}" ]]; then
    echo "找不到前置 Direct-SFT 检查点：${directory}" >&2
    return 1
  fi
  printf '%s\n' "${checkpoint}"
}

MODEL="${ROOT}/models/Qwen3.5-0.8B-Base"
DATA="${ROOT}/datasets/fudan_news_cot_50"
DIRECT_SFT_ROOT="${DIRECT_SFT_OUTPUT:-${ROOT}/outputs/08_rloo_classification/sft}"
DIRECT_SFT_ADAPTER="${DIRECT_SFT_ADAPTER:-$(latest_checkpoint "${DIRECT_SFT_ROOT}")}"
OUTPUT="${COT_SFT_OUTPUT:-${ROOT}/outputs/09_rloo_cot_classification/sft}"

# 在已学会分类的适配器上继续训练，让小模型先掌握稳定的 CoT 输出结构。
swift sft \
  --model "${MODEL}" \
  --adapters "${DIRECT_SFT_ADAPTER}" \
  --dataset "${DATA}/sft_train.jsonl" \
  --val_dataset "${DATA}/evidence_val.jsonl" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --max_length 1024 \
  --per_device_train_batch_size "${COT_SFT_BATCH:-8}" \
  --per_device_eval_batch_size "${COT_SFT_BATCH:-8}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${COT_SFT_LEARNING_RATE:-5e-5}" \
  --warmup_ratio 0.05 \
  --num_train_epochs "${COT_SFT_EPOCHS:-5}" \
  --eval_strategy epoch \
  --save_strategy epoch \
  --logging_steps 1 \
  --save_total_limit 1 \
  --save_only_model true \
  --gradient_checkpointing false \
  --dataloader_num_workers 0 \
  --dataset_num_proc 1 \
  --report_to tensorboard \
  --output_dir "${OUTPUT}"
