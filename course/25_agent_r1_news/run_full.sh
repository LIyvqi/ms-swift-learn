#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# 默认执行两轮全量 SFT 和两轮全量 GRPO；可用同名环境变量或 GRPO_MAX_STEPS 覆盖。
bash "${ROOT}/course/25_agent_r1_news/train_sft.sh"
bash "${ROOT}/course/25_agent_r1_news/train_grpo.sh"
