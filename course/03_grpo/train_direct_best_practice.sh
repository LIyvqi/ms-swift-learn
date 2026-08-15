#!/usr/bin/env bash

# 在现有 Direct-GRPO 上启用适配单卡 0.8B 模型的 Qwen3.5 最佳实践参数。
set -euo pipefail
STYLE=direct GRPO_PROFILE=qwen35 exec bash "$(dirname -- "$0")/train.sh"
