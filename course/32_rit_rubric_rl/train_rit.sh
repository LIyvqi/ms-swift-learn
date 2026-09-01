#!/usr/bin/env bash

# RiT 主实验：thinking rubrics 与结果奖励融合，再由硬门控限制总分。
set -euo pipefail
export METHOD=rit
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_train_grpo.sh" "$@"
