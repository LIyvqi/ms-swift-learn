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
    echo "找不到 CoT-SFT 检查点：${directory}" >&2
    return 1
  fi
  printf '%s\n' "${checkpoint}"
}

MODEL="${ROOT}/models/Qwen3.5-0.8B-Base"
DATA="${ROOT}/datasets/fudan_news_cot_50"
PLUGIN="${ROOT}/course/plugins/cot_classification_rewards.py"
COT_SFT_ROOT="${COT_SFT_OUTPUT:-${ROOT}/outputs/09_rloo_cot_classification/sft}"
COT_SFT_ADAPTER="${COT_SFT_ADAPTER:-$(latest_checkpoint "${COT_SFT_ROOT}")}"
RLOO_STEPS="${RLOO_STEPS:-100}"
OUTPUT="${COT_RLOO_OUTPUT:-${ROOT}/outputs/09_rloo_cot_classification/rloo_${RLOO_STEPS}step}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  TRAIN_DATA="${DATA}/rl_smoke.jsonl"
  RLOO_STEPS=1
  OUTPUT="${COT_RLOO_OUTPUT:-${ROOT}/outputs/09_rloo_cot_classification/rloo_smoke}"
else
  TRAIN_DATA="${DATA}/rl_train.jsonl"
fi

swift rlhf \
  --rlhf_type grpo \
  --advantage_estimator rloo \
  --kl_in_reward true \
  --scale_rewards none \
  --model "${MODEL}" \
  --adapters "${COT_SFT_ADAPTER}" \
  --ref_adapters "${COT_SFT_ADAPTER}" \
  --dataset "${TRAIN_DATA}" \
  --external_plugins "${PLUGIN}" \
  --reward_funcs course_cot_label_accuracy course_cot_structure course_cot_evidence course_cot_consistency \
  --reward_weights 1.0 0.3 0.5 0.2 \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --enable_thinking true \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "${VLLM_MEMORY:-0.55}" \
  --vllm_max_model_len 1024 \
  --vllm_limit_mm_per_prompt '{"image":0,"video":0}' \
  --vllm_mm_processor_cache_gb 0 \
  --vllm_enforce_eager true \
  --sleep_level 1 \
  --num_generations "${NUM_GENERATIONS:-4}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --per_device_train_batch_size "${COT_RL_BATCH:-16}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${COT_RLOO_LEARNING_RATE:-5e-6}" \
  --beta "${COT_RLOO_BETA:-0.001}" \
  --max_grad_norm 1.0 \
  --max_length 1024 \
  --max_completion_length 160 \
  --max_steps "${RLOO_STEPS}" \
  --logging_steps 1 \
  --save_steps "${RLOO_STEPS}" \
  --save_total_limit 1 \
  --save_only_model true \
  --dataset_num_proc 1 \
  --dataloader_num_workers 0 \
  --log_completions true \
  --report_to tensorboard \
  --output_dir "${OUTPUT}"
