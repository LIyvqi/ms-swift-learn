#!/usr/bin/env bash

# 使用本地可执行规则评价显式思考，不产生外部 API 费用。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REWARD_MODE=rules exec bash "${ROOT}/course/03_grpo/_train_cot.sh"
