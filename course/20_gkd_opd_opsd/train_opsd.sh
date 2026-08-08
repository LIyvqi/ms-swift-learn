#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/course/alignment_common.sh"
mapfile -t SPAN < <(alignment_span_args)
SFT_ADAPTER="$(alignment_sft_checkpoint)"
OUTPUT="${OPSD_OUTPUT:-${ROOT}/outputs/20_gkd_opd_opsd/opsd${ALIGNMENT_SUFFIX}}"

# 当前 ROCm vLLM 0.26 的单进程执行器会直接读取 torchrun 环境变量；
# GKD CLI 不像 GRPO CLI 那样自动补齐，因此单卡运行时在这里提供安全默认值。
export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29521}"

# GKD 在线数据加载器会丢弃不足一个 batch 的尾批；冒烟集只有 16 条。
if [[ "${SMOKE:-0}" == "1" ]]; then
  OPSD_BATCH_VALUE="${OPSD_BATCH:-16}"
else
  OPSD_BATCH_VALUE="${OPSD_BATCH:-64}"
fi

swift rlhf \
  --rlhf_type gkd \
  --model "${ALIGNMENT_MODEL}" \
  --adapters "${SFT_ADAPTER}" \
  --dataset "${ALIGNMENT_DATA}/opsd_${ALIGNMENT_SPLIT}.jsonl" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --lmbda 1.0 \
  --beta "${GKD_BETA:-0.5}" \
  --sft_alpha 0.0 \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "${VLLM_MEMORY:-0.50}" \
  --vllm_max_model_len 1024 \
  --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
  --vllm_mm_processor_cache_gb 0 \
  --vllm_enforce_eager true \
  --sleep_level 1 \
  --temperature 1.2 \
  --per_device_train_batch_size "${OPSD_BATCH_VALUE}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${OPSD_LR:-2e-5}" \
  --max_length "${ALIGNMENT_MAX_LENGTH:-384}" \
  --truncation_strategy left \
  --max_completion_length "${OPSD_MAX_COMPLETION_LENGTH:-24}" \
  --logging_steps 1 \
  --save_total_limit 1 \
  --save_only_model true \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --report_to tensorboard \
  --output_dir "${OUTPUT}" \
  "${SPAN[@]}"
