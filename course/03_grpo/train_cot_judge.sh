#!/usr/bin/env bash

# 使用 OpenAI 兼容大模型裁判评价显式思考，密钥只从当前环境读取。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REWARD_MODE=judge exec bash "${ROOT}/course/03_grpo/_train_cot.sh"
