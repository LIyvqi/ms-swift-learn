#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/course/alignment_common.sh"
SFT_ADAPTER="$(alignment_sft_checkpoint)"
RM_MODEL="$(alignment_rm_checkpoint)"
OUTPUT="${PPO_OUTPUT:-${ROOT}/outputs/14_ppo/ppo${ALIGNMENT_SUFFIX}}"

# TRL 的 PPO 按 total_episodes 计算更新次数，max_steps 不直接控制 PPO 主循环。
# 冒烟只生成一个短批次；正式训练时让 STEPS 仍然准确表示 PPO 更新次数。
if [[ "${SMOKE:-0}" == "1" ]]; then
  PPO_BATCH_VALUE="${PPO_BATCH:-8}"
  PPO_EPOCHS_VALUE="${PPO_EPOCHS:-1}"
  PPO_MINI_BATCHES_VALUE="${PPO_MINI_BATCHES:-1}"
  COMPLETION_LENGTH_VALUE="${PPO_MAX_COMPLETION_LENGTH:-8}"
  PPO_TRAIN_EPOCHS_VALUE="${PPO_TRAIN_EPOCHS:-0.5}"
else
  PPO_BATCH_VALUE="${PPO_BATCH:-64}"
  PPO_EPOCHS_VALUE="${PPO_EPOCHS:-4}"
  PPO_MINI_BATCHES_VALUE="${PPO_MINI_BATCHES:-4}"
  COMPLETION_LENGTH_VALUE="${PPO_MAX_COMPLETION_LENGTH:-24}"
  if [[ -n "${STEPS:-}" ]]; then
    # 正式集有 256 条；换算后一个 PPO 更新仍对应一个 STEPS。
    PPO_TRAIN_EPOCHS_VALUE="$(awk -v batch="${PPO_BATCH_VALUE}" -v steps="${STEPS}" \
      'BEGIN { printf "%.8f", batch * steps / 256 }')"
  else
    PPO_TRAIN_EPOCHS_VALUE="${PPO_TRAIN_EPOCHS:-3}"
  fi
fi

swift rlhf \
  --rlhf_type ppo \
  --model "${ALIGNMENT_MODEL}" \
  --adapters "${SFT_ADAPTER}" \
  --reward_model "${RM_MODEL}" \
  --dataset "${ALIGNMENT_DATA}/prompts_${ALIGNMENT_SPLIT}.jsonl" \
  --val_dataset "${ALIGNMENT_DATA}/prompts_val.jsonl" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --per_device_train_batch_size "${PPO_BATCH_VALUE}" \
  --per_device_eval_batch_size "${EVAL_BATCH:-32}" \
  --local_rollout_forward_batch_size "${ROLLOUT_BATCH:-64}" \
  --num_ppo_epochs "${PPO_EPOCHS_VALUE}" \
  --num_mini_batches "${PPO_MINI_BATCHES_VALUE}" \
  --num_train_epochs "${PPO_TRAIN_EPOCHS_VALUE}" \
  --learning_rate "${PPO_LR:-1e-5}" \
  --kl_coef "${PPO_KL:-0.1}" \
  --cliprange 0.2 \
  --cliprange_value 0.2 \
  --vf_coef 0.1 \
  --gamma 1.0 \
  --lam 0.95 \
  --whiten_rewards true \
  --max_length "${ALIGNMENT_MAX_LENGTH:-384}" \
  --truncation_strategy left \
  --max_completion_length "${COMPLETION_LENGTH_VALUE}" \
  --temperature 1.2 \
  --logging_steps 1 \
  --save_steps "${PPO_SAVE_STEPS:-4}" \
  --save_total_limit 1 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 8 \
  --report_to tensorboard \
  --output_dir "${OUTPUT}"
