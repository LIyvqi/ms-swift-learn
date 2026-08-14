#!/usr/bin/env bash

# 用相同起点训练正确性基线、Brier-RLCR 或对数奖励对照。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/course/confidence_common.sh"
activate_confidence_env

METHOD="${METHOD:-brier}"
if [[ "${METHOD}" != "correctness" && "${METHOD}" != "brier" && "${METHOD}" != "log" ]]; then
  echo "METHOD 只能是 correctness、brier 或 log" >&2
  exit 1
fi

MODEL="${ROOT}/models/Qwen3.5-0.8B-Base"
DATA="${ROOT}/datasets/confidence_news"
PLUGIN="${ROOT}/course/28_rlcr_confidence/rlcr_rewards.py"
FORMAT_ROOT="${FORMAT_SFT_OUTPUT:-${ROOT}/outputs/28_rlcr_confidence/format_sft}"
FORMAT_ADAPTER="${FORMAT_ADAPTER:-$(latest_confidence_checkpoint "${FORMAT_ROOT}")}"
RLCR_STEPS="${RLCR_STEPS:-100}"
OUTPUT="${RLCR_OUTPUT:-${ROOT}/outputs/28_rlcr_confidence/${METHOD}_${RLCR_STEPS}step}"
TRAIN_DATA="${DATA}/rlcr_train.jsonl"

if [[ "${SMOKE:-0}" == "1" ]]; then
  RLCR_STEPS=1
  TRAIN_DATA="${DATA}/rlcr_smoke.jsonl"
  OUTPUT="${RLCR_OUTPUT:-${ROOT}/outputs/28_rlcr_confidence/${METHOD}_smoke}"
fi

case "${METHOD}" in
  correctness)
    REWARD_FUNCS=(course_rlcr_accuracy course_rlcr_format)
    REWARD_WEIGHTS=(1.0 0.2)
    ;;
  brier)
    REWARD_FUNCS=(course_rlcr_accuracy course_rlcr_brier course_rlcr_format)
    REWARD_WEIGHTS=(1.0 1.0 0.2)
    ;;
  log)
    REWARD_FUNCS=(course_rlcr_accuracy course_rlcr_log_score course_rlcr_format)
    REWARD_WEIGHTS=(1.0 0.2 0.2)
    ;;
esac

swift rlhf \
  --rlhf_type grpo \
  --advantage_estimator rloo \
  --kl_in_reward true \
  --scale_rewards none \
  --model "${MODEL}" \
  --adapters "${FORMAT_ADAPTER}" \
  --ref_adapters "${FORMAT_ADAPTER}" \
  --dataset "${TRAIN_DATA}" \
  --external_plugins "${PLUGIN}" \
  --reward_funcs "${REWARD_FUNCS[@]}" \
  --reward_weights "${REWARD_WEIGHTS[@]}" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "${VLLM_MEMORY:-0.40}" \
  --vllm_max_model_len 1024 \
  --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
  --vllm_mm_processor_cache_gb 0 \
  --vllm_enforce_eager true \
  --sleep_level "${RLCR_SLEEP_LEVEL:-1}" \
  --num_generations "${NUM_GENERATIONS:-4}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --per_device_train_batch_size "${RLCR_BATCH:-16}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${RLCR_LR:-5e-6}" \
  --beta "${RLCR_BETA:-0.001}" \
  --max_grad_norm 1.0 \
  --max_length 768 \
  --max_completion_length 64 \
  --max_steps "${RLCR_STEPS}" \
  --logging_steps 1 \
  --save_steps "${RLCR_STEPS}" \
  --save_total_limit 1 \
  --save_only_model true \
  --dataset_num_proc 1 \
  --dataloader_num_workers 0 \
  --log_completions true \
  --report_to tensorboard \
  --output_dir "${OUTPUT}"
