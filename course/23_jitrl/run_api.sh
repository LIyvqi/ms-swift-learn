#!/usr/bin/env bash
set -euo pipefail

# API 服务需先由 serve_api.sh 或其他 OpenAI 兼容部署工具启动。
COURSE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${COURSE_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source ./activate.sh

python course/23_jitrl/test_api_policy.py
python course/23_jitrl/run_api_experiment.py "$@"
