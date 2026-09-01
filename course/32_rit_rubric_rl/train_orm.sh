#!/usr/bin/env bash

# 论文 ORM 对照：GRPO 只看最终审核结果，不直接评价中间思考。
set -euo pipefail
export METHOD=orm
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_train_grpo.sh" "$@"
