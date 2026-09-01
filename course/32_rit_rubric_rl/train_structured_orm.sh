#!/usr/bin/env bash

# 短结构化输出的结果奖励对照组。
set -euo pipefail
export STRUCTURED_METHOD=orm
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_train_structured_grpo.sh" "$@"
