#!/usr/bin/env bash
set -euo pipefail

# 调用方只需临时导出 KCR_JITRL_API_KEY，本脚本不会读取或保存密钥文本。
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/activate.sh"
python "${ROOT_DIR}/course/24_kcr_jitrl/test_kcr_core.py"
python "${ROOT_DIR}/course/24_kcr_jitrl/run_api_experiment.py" "$@"
