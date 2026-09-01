#!/usr/bin/env bash

# 短结构化输出的逐字段 RiT 奖励实验。
set -euo pipefail
export STRUCTURED_METHOD=rit
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_train_structured_grpo.sh" "$@"
