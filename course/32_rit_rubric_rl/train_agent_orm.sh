#!/usr/bin/env bash

# 结果奖励对照：Agent 仍可调用两库，但训练只看最终完整答案。
set -euo pipefail
AGENT_METHOD=orm exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_train_agent_grpo.sh"
