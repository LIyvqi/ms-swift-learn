#!/usr/bin/env bash

# 同时组合答案、结构、可执行计算代理与大模型裁判奖励。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REWARD_MODE=hybrid exec bash "${ROOT}/course/03_grpo/_train_cot.sh"
