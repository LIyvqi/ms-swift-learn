#!/usr/bin/env bash
set -euo pipefail

# 使用工作区中的持久化环境运行单元测试和本地模型消融实验。
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/activate.sh"
python "${ROOT_DIR}/course/24_kcr_jitrl/test_kcr_core.py"
python "${ROOT_DIR}/course/24_kcr_jitrl/run_experiment.py" "$@"
